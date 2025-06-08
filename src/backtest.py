import pandas as pd
import numpy as np
from typing import Dict, Tuple, Type
from strategies import TradingStrategy

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
                 confidence_threshold: float = 0.0):  # Default to 0 (all trades)
        self.strategy = strategy(**strategy_params)
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.confidence_threshold = confidence_threshold
        self.portfolio = initial_capital
        self.positions: Dict[str, Dict[str, float]] = {}  # market -> {'size': float, 'entry_price': float, 'entry_fee': float}
        self.trade_log = []
        self.equity_curve = []
        self.data = None

    def calculate_position_size(self, price: float, volatility: float) -> float:
        """
        Calculate position size based on volatility and risk parameters
        
        Args:
            price: Current price of the asset
            volatility: Current volatility measure (ATR equivalent)
            
        Returns:
            Position size in asset units
        """
        if volatility <= 0:
            return 0
            
        risk_amount = self.portfolio * self.risk_per_trade
        position_size = risk_amount / volatility
        
        # Add constraints to prevent overexposure:
        # 1. Never risk more than 5% of portfolio in a single trade
        max_risk_size = self.portfolio * 0.05 / price
        # 2. Ensure position doesn't exceed available capital
        max_affordable = (self.portfolio * 0.99) / price  # Leave 1% buffer
        
        # Adaptive sizing based on market conditions
        base_size = min(position_size, max_risk_size, max_affordable)
        
        # Scale up during high volatility
        if volatility > price * 0.001:  # >0.1%
            return base_size * 1.5
        return base_size

    def execute_trade(self, market: str, price: float, size: float, direction: str, timestamp: pd.Timestamp):
        """
        Execute a trade and update portfolio
        
        Args:
            market: Trading pair symbol
            price: Execution price
            size: Position size in asset units
            direction: 'long' or 'exit'
            timestamp: Time of trade
        """
        cost = price * size
        
        if direction == 'long':
            fee = cost * 0.0002  # Fee for entry
            self.portfolio -= (cost + fee)
            self.positions[market] = {'size': size, 'entry_price': price, 'entry_fee': fee}
            self.trade_log.append({
                'timestamp': timestamp,
                'market': market,
                'action': 'enter',
                'price': price,
                'size': size,
                'fee': fee 
            })
        elif direction == 'exit' and market in self.positions:
            current_position_details = self.positions[market]
            position_size = current_position_details['size']
            entry_price = current_position_details['entry_price']
            entry_fee = current_position_details['entry_fee']

            exit_value = price * position_size # `price` here is exit_price
            exit_fee = exit_value * 0.0002 # Fee for exit

            self.portfolio += (exit_value - exit_fee)

            # Calculate profit for this round trip
            profit_for_trade = (exit_value - (entry_price * position_size)) - (entry_fee + exit_fee)
            
            self.trade_log.append({
                'timestamp': timestamp,
                'market': market,
                'action': 'exit',
                'price': price, # This is exit_price
                'size': position_size,
                'fee': exit_fee, # This is exit_fee
                'profit': profit_for_trade
            })
            del self.positions[market]
        
        # Record portfolio value
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': self.portfolio,
            'cash': self.portfolio,
            'positions': self.positions.copy()
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
        
        # Apply strategy to generate signals
        data = self.strategy.generate_signals(data)
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
        
        # Group by market and iterate
        grouped = data.groupby('market')
        for market, group in grouped:
            # Sort by time
            group = group.sort_index()
            in_trade = False
            
            for idx, row in group.iterrows():
                # Check for entry signal
                # confidence = row.get('confidence', 0.0)
                # print(f"Confidence level for {market} at {idx}: {confidence}")
                if not in_trade and row.get('entry_signal', False) and row.get('confidence', 0) >= self.confidence_threshold:
                    position_size = self.calculate_position_size(
                        row['mid_price'], 
                        row.get('volatility', 0.01)  # Default volatility if not present
                    )
                    if position_size > 0:
                        self.execute_trade(market, row['mid_price'], position_size, 'long', idx)
                        in_trade = True
                
                # Check for exit signal
                if in_trade and row.get('exit_signal', False):
                    self.execute_trade(market, row['mid_price'], 0, 'exit', idx)
                    in_trade = False
            
            # Close any open positions at end
            if in_trade:
                self.execute_trade(market, group.iloc[-1]['mid_price'], 0, 'exit', group.index[-1])
                
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
