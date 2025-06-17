import pandas as pd
import numpy as np
from typing import Dict, Tuple, Type
from strategies import TradingStrategy
from features import engineer_features
from ml_strategy import MLTradingStrategy
import logging

logger = logging.getLogger(__name__)

class GenericBacktester:
    """
    Generic backtester that works with any trading strategy
    
    Args:
        strategy: TradingStrategy class to use
        strategy_params: Parameters to pass to strategy constructor
        initial_capital: Starting capital in USDT
        risk_per_trade: Percentage of capital to risk per trade
    """
    def __init__(self, 
                 strategy: Type[TradingStrategy],
                 strategy_params: dict = {},
                 initial_capital: float = 10000.0,
                 risk_per_trade: float = 0.01,
                 confidence_threshold: float = 0.0,
                 leverage: float = 1.0,
                 funding_rate_daily: float = 0.0001, # Default to 0 (all trades), 0.01% daily funding
                 selected_markets: list = None): 
        self.strategy = strategy(**strategy_params)
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.selected_markets = selected_markets
        self.confidence_threshold = confidence_threshold
        self.leverage = leverage
        if self.leverage <= 0:
            raise ValueError("Leverage must be positive.")
        self.funding_rate_daily = funding_rate_daily
        self.portfolio = initial_capital  # Represents available cash
        # market -> {'size': float (notional), 'entry_price': float, 'entry_fee': float, 
        #            'margin_used': float, 'borrowed_amount': float, 'entry_timestamp': pd.Timestamp, 'leverage': float}
        self.positions: Dict[str, Dict[str, any]] = {}
        self.trade_log = []
        self.equity_curve = []
        self.data = None

    def calculate_position_size(self, price: float, volatility: float) -> float:
        """
        Calculate NOTIONAL position size based on volatility, risk parameters, and leverage.
        
        Args:
            price: Current price of the asset
            volatility: Current volatility measure (ATR equivalent, absolute price change)
            
        Returns:
            Notional position size in asset units.
        """
        if volatility <= 0 or self.leverage == 0 or price <= 0:
            return 0
            
        # Risk amount based on initial capital to stabilize trade sizing under leverage
        # self.portfolio (cash) can fluctuate wildly due to margin calls.
        risk_amount_dollar = self.initial_capital * self.risk_per_trade
        
        # Notional size such that if price moves by `volatility`, PnL on notional is `risk_amount_dollar`
        notional_size_based_on_risk = risk_amount_dollar / volatility
        
        # Max affordable notional size based on available margin
        # Margin required = (Notional Size * price) / leverage
        # We need Margin required <= self.portfolio * 0.99 (leaving 1% buffer)
        max_notional_affordable = (self.portfolio * 0.99 * self.leverage) / price
        
        base_notional_size = min(notional_size_based_on_risk, max_notional_affordable)
        
        # Adaptive sizing based on market conditions (applies to notional size)
        if volatility > price * 0.001:  # >0.1% price fluctuation
            final_notional_size = base_notional_size * 1.5
        else:
            final_notional_size = base_notional_size
        
        # Ensure margin needed for the final_notional_size does not exceed available portfolio cash
        margin_needed = (final_notional_size * price) / self.leverage if self.leverage > 0 else float('inf')

        if margin_needed > self.portfolio:
            # Adjust notional size to what can be afforded with current cash
            final_notional_size = (self.portfolio * self.leverage) / price if price > 0 else 0
            
        return final_notional_size if self.leverage > 0 else 0

    def execute_trade(self, market: str, price: float, size: float, direction: str, timestamp: pd.Timestamp):
        """
        Execute a trade and update portfolio (cash).
        
        Args:
            market: Trading pair symbol
            price: Execution price
            size: Notional position size in asset units
            direction: 'long' or 'exit'
            timestamp: Time of trade
        """
        if self.leverage == 0 and size > 0: # Should be caught by calculate_position_size
            return

        notional_value = price * size
        
        if direction == 'long':
            if self.leverage == 1.0:
                margin_used = notional_value
                borrowed_amount = 0.0
            else:
                margin_used = notional_value / self.leverage
                borrowed_amount = notional_value - margin_used

            if margin_used > self.portfolio and not np.isclose(margin_used, self.portfolio): # Add tolerance for float precision
                # This check is a safeguard; calculate_position_size should prevent this.
                print(f"Timestamp: {timestamp}, Market: {market} - Insufficient cash for margin. Needed: {margin_used:.2f}, Have: {self.portfolio:.2f}. Skipping trade.")
                return

            entry_fee = notional_value * 0.1 * 0.01 # Regular user fee on notional value, 0.1 percent

            self.portfolio -= (margin_used + entry_fee) # Cash decreases
            position_details = {
                'size': size,
                'entry_price': price,
                'entry_fee_paid': entry_fee,
                'margin_used': margin_used,
                'borrowed_amount': borrowed_amount,
                'entry_timestamp': timestamp,
                'leverage': self.leverage
            }

            # If using ML strategy, add triple barrier info
            if isinstance(self.strategy, MLTradingStrategy):
                position_details['tp_price'] = price * (1 + self.strategy.tp_pct)
                position_details['sl_price'] = price * (1 - self.strategy.sl_pct)
                position_details['time_barrier'] = timestamp + pd.Timedelta(seconds=self.strategy.horizon_seconds)

            self.positions[market] = position_details
            self.trade_log.append({
                'timestamp': timestamp,
                'market': market,
                'action': 'enter',
                'price': price,
                'size': size, # Notional size
                'fee': entry_fee,
                'leverage': self.leverage,
                'margin_used': margin_used
            })
        elif direction == 'exit' and market in self.positions:
            pos_details = self.positions[market]
            position_notional_size = pos_details['size']
            entry_price = pos_details['entry_price']
            entry_fee_paid = pos_details['entry_fee_paid'] # Fee paid at entry
            margin_used = pos_details['margin_used']
            borrowed_amount = pos_details['borrowed_amount']
            entry_timestamp = pos_details['entry_timestamp']
            position_leverage = pos_details['leverage']

            exit_notional_value = price * position_notional_size # `price` here is exit_price
            exit_fee = exit_notional_value * 0.1 * 0.01 # Regular user fee on notional value, 0.1 percent

            funding_cost = 0.0
            if position_leverage > 1.0 and borrowed_amount > 0:
                days_held = (timestamp - entry_timestamp).total_seconds() / (24 * 60 * 60)
                days_held = max(0, days_held) # Ensure non-negative
                funding_cost = borrowed_amount * self.funding_rate_daily * days_held
            
            pnl_on_notional = (price - entry_price) * position_notional_size
            
            # Update cash portfolio
            self.portfolio += (margin_used + pnl_on_notional - exit_fee - funding_cost)

            profit_for_trade = pnl_on_notional - entry_fee_paid - exit_fee - funding_cost
            
            self.trade_log.append({
                'timestamp': timestamp,
                'market': market,
                'action': 'exit',
                'price': price, # This is exit_price
                'size': position_notional_size, # Notional size
                'fee': exit_fee, # Exit fee
                'funding_cost': funding_cost,
                'profit': profit_for_trade,
                'leverage': position_leverage
            })
            del self.positions[market]
        
        # Record portfolio value (cash)
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': self.portfolio, # Tracks cash available
            'cash': self.portfolio,
            'positions': {m: {'size': p['size'], 'entry_price': p['entry_price'], 'leverage': p['leverage']} for m, p in self.positions.items()} # Snapshot of open positions
        })

    def backtest(self, data: pd.DataFrame):
        """
        Run backtest on the provided market data
        
        Args:
            data: DataFrame with features
        """
        # Reset state
        self.portfolio = self.initial_capital
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []

        # --- ML Workflow Enhancement ---
        # 1. Engineer features if using an ML strategy
        logger.info("Running feature engineering...")
        # The raw data needs to be processed to create features for the model.
        # We assume the input `data` is the raw data from the loader.
        data_with_features = engineer_features(data.copy())
        
        # 2. Apply strategy to generate signals
        logger.info("Generating signals from strategy...")
        data_with_signals = self.strategy.generate_signals(data_with_features)
        
        # From here on, use the data with signals
        data = data_with_signals
        # --- End of ML Workflow Enhancement ---

        # Filter data for selected markets if specified
        if self.selected_markets:
            data = data[data['market'].isin(self.selected_markets)]
            if data.empty:
                print(f"Warning: No data found for selected markets: {self.selected_markets}. Backtest will not run.")
                self.data = data # Store empty dataframe
                return # Exit if no data for selected markets

        self.data = data

        # Add initial equity point IF data is not empty and has a DatetimeIndex
        if not data.empty and isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 0:
            # Use the earliest timestamp from the data for the initial equity record
            initial_timestamp = data.index.min() 
            self.equity_curve.append({
                'timestamp': initial_timestamp, # Or data.index[0] if data is sorted
                'portfolio_value': self.initial_capital,
                'cash': self.initial_capital,
                'positions': {} # No positions at the start
            })
        
        # Sort data by timestamp to process events chronologically across all markets
        data = data.sort_index()

        # Dictionary to keep track of whether we are in a trade for each market
        # Initialize based on unique markets in the (potentially filtered) data
        active_markets = data['market'].unique()
        if not active_markets.size: # Check if active_markets is empty
             print("Warning: No active markets to trade after filtering. Backtest will not proceed with trading logic.")
             # Still record initial equity if not already done, and then can return or let it pass if equity curve is desired
             if not self.equity_curve: # Ensure initial equity is recorded if it wasn't
                if not data.empty and isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 0:
                    initial_timestamp = data.index.min()
                    self.equity_curve.append({
                        'timestamp': initial_timestamp,
                        'portfolio_value': self.initial_capital,
                        'cash': self.initial_capital,
                        'positions': {}
                    })
             return


        in_trade_for_market: Dict[str, bool] = {market: False for market in active_markets}
        
        # Iterate through the data chronologically
        for idx, row in data.iterrows():
            market = row['market']
            current_price = row['mid_price']
            # Use 'volatility_20p' from features, with a fallback
            volatility = row.get('volatility_20p', 0.01) 
            confidence = row.get('confidence', 0.0)

            # --- Triple Barrier Exit Logic ---
            if in_trade_for_market.get(market, False) and market in self.positions:
                pos = self.positions[market]
                exit_reason = None
                if 'tp_price' in pos and current_price >= pos['tp_price']:
                    exit_reason = 'TP'
                elif 'sl_price' in pos and current_price <= pos['sl_price']:
                    exit_reason = 'SL'
                elif 'time_barrier' in pos and idx >= pos['time_barrier']:
                    exit_reason = 'Time'
                
                if exit_reason:
                    logger.info(f"Exit for {market} at {current_price:.2f} due to {exit_reason} at {idx}")
                    self.execute_trade(market, current_price, 0, 'exit', idx)
                    in_trade_for_market[market] = False
                    continue # Skip to next row after processing exit

            # Check for entry signal for the current market
            if not in_trade_for_market.get(market, False) and row.get('entry_signal', False) and confidence >= self.confidence_threshold:
                position_size = self.calculate_position_size(current_price, volatility)
                if position_size > 0:
                    self.execute_trade(market, current_price, position_size, 'long', idx)
                    in_trade_for_market[market] = True
            
            # Check for explicit exit signal from strategy (e.g., for non-triple-barrier models)
            elif in_trade_for_market.get(market, False) and row.get('exit_signal', False):
                if market in self.positions:
                    self.execute_trade(market, current_price, 0, 'exit', idx)
                    in_trade_for_market[market] = False

        # Close any open positions at the end of the backtest for each market
        # We need the last known price for each market to close positions
        # Ensure data is not empty before attempting groupby
        if not data.empty:
            last_prices = data.groupby('market')['mid_price'].last()
            last_timestamps = data.groupby('market').apply(lambda x: x.index[-1])

            for market, is_in_trade in in_trade_for_market.items():
                if is_in_trade and market in self.positions: # Check self.positions as well for robustness
                    if market in last_prices and market in last_timestamps:
                        last_price_for_market = last_prices[market]
                        last_timestamp_for_market = last_timestamps[market]
                        self.execute_trade(market, last_price_for_market, 0, 'exit', last_timestamp_for_market)
                        in_trade_for_market[market] = False # Update status
                    else:
                        # This case should ideally not happen if data is consistent
                        print(f"Warning: Could not find last price/timestamp for market {market} to close open position.")
        elif self.equity_curve and not self.positions: # If data was empty from the start but initial equity was logged
            pass # Nothing to close
        else: # Data became empty after filtering, or was empty and no initial equity logged
             print("Warning: Data is empty, cannot close open positions.")
                
    def get_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get backtest results
        
        Returns:
            trade_log: DataFrame of all trades executed
            equity_curve: DataFrame of portfolio value over time
        """
        trade_df = pd.DataFrame(self.trade_log)
        equity_df = pd.DataFrame(self.equity_curve)
        return trade_df, equity_df

    def calculate_metrics(self, trade_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
        """
        Calculate performance metrics
        
        Args:
            trade_df: Trade log DataFrame
            equity_df: Equity curve DataFrame
            
        Returns:
            Dictionary of performance metrics
        """
        # Basic metrics
        metrics = {
            'final_portfolio': self.portfolio,
            'return_pct': ((self.portfolio / self.initial_capital) - 1) * 100,
            'num_trades': len(trade_df) // 2  # Each trade has enter/exit
        }
        
        if len(trade_df) > 0:
            # Win rate
            profits = trade_df[trade_df['action'] == 'exit']['profit']
            win_rate = (profits > 0).mean() * 100
            
            # Risk-adjusted returns (annualized)
            equity_returns = equity_df['portfolio_value'].pct_change().dropna()
            
            if len(equity_returns) > 1 and equity_returns.std() > 0:
                # Calculate annualization factor
                # Assuming crypto markets trade 365 days a year.
                # Try to infer periods per day from the equity curve timestamps
                if len(equity_df['timestamp']) > 1:
                    time_diff_seconds = (equity_df['timestamp'].iloc[1] - equity_df['timestamp'].iloc[0]).total_seconds()
                    if time_diff_seconds > 0:
                        periods_per_day = (24 * 60 * 60) / time_diff_seconds
                        intervals_per_year = periods_per_day * 365 
                    else: # Fallback if time difference is zero or negative (unlikely)
                        intervals_per_year = 365 * 24 * 60 * 12 # Default to 5-second intervals
                else: # Fallback if only one equity point (or initial point only)
                    intervals_per_year = 365 * 24 * 60 * 12 # Default to 5-second intervals

                sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(intervals_per_year)
            else:
                sharpe_ratio = 0.0 # Ensure float
            
            metrics.update({
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': (equity_df['portfolio_value'] / equity_df['portfolio_value'].cummax() - 1).min() * 100
            })
        
        return metrics

if __name__ == "__main__":
    print("Trend Following Backtester ready")
