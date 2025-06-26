import polars as pl
import numpy as np
from typing import Dict, Tuple, Type
from strategies import TradingStrategy
from features import engineer_features
from ml_strategy import MLTradingStrategy
from risk_manager import RiskManager, RiskConfig
import logging
import datetime

logger = logging.getLogger(__name__)

class GenericBacktester:
    def __init__(self, 
                 strategy: Type[TradingStrategy],
                 strategy_params: dict = {},
                 initial_capital: float = 10000.0,
                 risk_per_trade: float = 0.01,
                 confidence_threshold: float = 0.0,
                 leverage: float = 1.0,
                 funding_rate_daily: float = 0.0001,
                 selected_markets: list = None,
                 verbose_logging: bool = True,
                 risk_config: RiskConfig = None): 
        self.strategy = strategy(**strategy_params)
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.selected_markets = selected_markets
        self.confidence_threshold = confidence_threshold
        self.leverage = leverage
        if self.leverage <= 0:
            raise ValueError("Leverage must be positive.")
        self.funding_rate_daily = funding_rate_daily
        self.portfolio = initial_capital
        self.positions: Dict[str, Dict[str, any]] = {}
        self.trade_log = []
        self.equity_curve = []
        self.data = None
        self.verbose_logging = verbose_logging
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Initialize risk management
        if risk_config is None:
            risk_config = RiskConfig()
            risk_config.max_risk_per_trade_pct = risk_per_trade  # Use provided risk_per_trade
        self.risk_manager = RiskManager(risk_config)
        
        logger.info(f"Backtester initialized with risk management: "
                   f"max_position={risk_config.max_position_pct:.1%}, "
                   f"max_exposure={risk_config.max_total_exposure_pct:.1%}, "
                   f"risk_per_trade={risk_config.max_risk_per_trade_pct:.1%}")

    def _format_currency(self, amount: float) -> str:
        """Format currency amounts for display"""
        return f"${amount:,.2f}"
    
    def _format_percentage(self, value: float) -> str:
        """Format percentage for display"""
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"
    
    def _format_position_info(self, market: str, pos_details: dict, current_price: float) -> str:
        """Format current position information"""
        if not pos_details:
            return ""
        
        entry_price = pos_details['entry_price']
        size = pos_details['size']
        unrealized_pnl = (current_price - entry_price) * size
        unrealized_pct = ((current_price / entry_price) - 1) * 100
        
        pnl_color = "\033[92m" if unrealized_pnl >= 0 else "\033[91m"  # Green for profit, red for loss
        reset_color = "\033[0m"
        
        return f"{market}: {size:.4f}@{self._format_currency(entry_price)} {pnl_color}({self._format_percentage(unrealized_pct)}){reset_color}"
    
    def _display_portfolio_status(self, timestamp: datetime.datetime, current_prices: dict = None):
        """Display current portfolio status"""
        if not self.verbose_logging:
            return
            
        positions_info = []
        if current_prices:
            for market, pos_details in self.positions.items():
                if market in current_prices:
                    positions_info.append(self._format_position_info(market, pos_details, current_prices[market]))
        
        positions_str = " | ".join(positions_info) if positions_info else "None"
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Portfolio: {self._format_currency(self.portfolio)} | "
              f"Total P&L: {self._format_currency(self.total_pnl)} | Win Rate: {win_rate:.1f}% | "
              f"Open Positions: {positions_str}")

    def calculate_position_size(self, price: float, volatility: float, market: str) -> Tuple[float, str]:
        """
        Calculate position size using the risk manager for safe sizing
        
        Returns:
            Tuple[float, str]: (position_size, reason_for_size)
        """
        if volatility <= 0 or self.leverage == 0 or price <= 0:
            return 0.0, "Invalid inputs (price, volatility, or leverage)"
        
        # Use risk manager for safe position sizing
        position_size, reason = self.risk_manager.calculate_safe_position_size(
            portfolio_value=self.portfolio,
            price=price,
            volatility=volatility,
            current_positions=self.positions,
            market=market,
            leverage=self.leverage
        )
        
        return position_size, reason

    def execute_trade(self, market: str, price: float, size: float, direction: str, timestamp: datetime.datetime, exit_reason: str = None):
        if self.leverage == 0 and size > 0:
            return

        notional_value = price * size
        
        if direction == 'long':
            if self.leverage == 1.0:
                margin_used = notional_value
                borrowed_amount = 0.0
            else:
                margin_used = notional_value / self.leverage
                borrowed_amount = notional_value - margin_used

            if margin_used > self.portfolio and not np.isclose(margin_used, self.portfolio):
                if self.verbose_logging:
                    print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] INSUFFICIENT FUNDS | {market} | "
                          f"Needed: {self._format_currency(margin_used)}, Available: {self._format_currency(self.portfolio)}")
                return

            entry_fee = notional_value * 0.1 * 0.01
            old_portfolio = self.portfolio
            self.portfolio -= (margin_used + entry_fee)
            
            position_details = {
                'size': size, 'entry_price': price, 'entry_fee_paid': entry_fee,
                'margin_used': margin_used, 'borrowed_amount': borrowed_amount,
                'entry_timestamp': timestamp, 'leverage': self.leverage
            }

            # Add ML strategy specific details
            tp_info = sl_info = ""
            if isinstance(self.strategy, MLTradingStrategy):
                position_details['tp_price'] = price * (1 + self.strategy.tp_pct)
                position_details['sl_price'] = price * (1 - self.strategy.sl_pct)
                tp_info = f" | TP: {self._format_currency(position_details['tp_price'])}"
                sl_info = f" | SL: {self._format_currency(position_details['sl_price'])}"

            self.positions[market] = position_details
            
            # Enhanced entry logging
            if self.verbose_logging:
                leverage_info = f" | {self.leverage}x Leverage" if self.leverage > 1 else ""
                print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] \033[94mENTRY\033[0m | {market} | "
                      f"Price: {self._format_currency(price)} | Size: {size:.4f} | "
                      f"Notional: {self._format_currency(notional_value)} | "
                      f"Margin: {self._format_currency(margin_used)} | "
                      f"Fee: {self._format_currency(entry_fee)}{leverage_info}{tp_info}{sl_info} | "
                      f"Portfolio: {self._format_currency(old_portfolio)} → {self._format_currency(self.portfolio)}")
            
            self.trade_log.append({
                'timestamp': timestamp, 'market': market, 'action': 'enter', 'price': price,
                'size': size, 'fee': entry_fee, 'leverage': self.leverage, 'margin_used': margin_used
            })
            
        elif direction == 'exit' and market in self.positions:
            pos_details = self.positions[market]
            exit_notional_value = price * pos_details['size']
            exit_fee = exit_notional_value * 0.1 * 0.01
            
            funding_cost = 0.0
            if pos_details['leverage'] > 1.0 and pos_details['borrowed_amount'] > 0:
                days_held = (timestamp - pos_details['entry_timestamp']).total_seconds() / (24 * 60 * 60)
                funding_cost = pos_details['borrowed_amount'] * self.funding_rate_daily * max(0, days_held)
            
            pnl_on_notional = (price - pos_details['entry_price']) * pos_details['size']
            old_portfolio = self.portfolio
            self.portfolio += (pos_details['margin_used'] + pnl_on_notional - exit_fee - funding_cost)
            profit_for_trade = pnl_on_notional - pos_details['entry_fee_paid'] - exit_fee - funding_cost
            
            # Update trade statistics
            self.total_trades += 1
            self.total_pnl += profit_for_trade
            if profit_for_trade > 0:
                self.winning_trades += 1
            
            # Calculate holding period
            holding_period = timestamp - pos_details['entry_timestamp']
            holding_minutes = holding_period.total_seconds() / 60
            
            # Enhanced exit logging
            if self.verbose_logging:
                pnl_color = "\033[92m" if profit_for_trade >= 0 else "\033[91m"  # Green for profit, red for loss
                pnl_pct = ((price / pos_details['entry_price']) - 1) * 100
                exit_reason_info = f" | Reason: {exit_reason}" if exit_reason else ""
                funding_info = f" | Funding: {self._format_currency(funding_cost)}" if funding_cost > 0 else ""
                
                print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] \033[93mEXIT\033[0m | {market} | "
                      f"Price: {self._format_currency(price)} | Size: {pos_details['size']:.4f} | "
                      f"Entry: {self._format_currency(pos_details['entry_price'])} | "
                      f"P&L: {pnl_color}{self._format_currency(profit_for_trade)} ({self._format_percentage(pnl_pct)})\033[0m | "
                      f"Fee: {self._format_currency(exit_fee)}{funding_info} | "
                      f"Hold: {holding_minutes:.1f}m{exit_reason_info} | "
                      f"Portfolio: {self._format_currency(old_portfolio)} → {self._format_currency(self.portfolio)}")
            
            self.trade_log.append({
                'timestamp': timestamp, 'market': market, 'action': 'exit', 'price': price,
                'size': pos_details['size'], 'fee': exit_fee, 'funding_cost': funding_cost,
                'profit': profit_for_trade, 'leverage': pos_details['leverage']
            })
            del self.positions[market]
        
        self.equity_curve.append({
            'timestamp': timestamp, 'portfolio_value': self.portfolio, 'cash': self.portfolio,
            'positions': {m: {'size': p['size'], 'entry_price': p['entry_price'], 'leverage': p['leverage']} for m, p in self.positions.items()}
        })

    def backtest(self, data: pl.DataFrame):
        self.portfolio = self.initial_capital
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []

        logger.info("Running feature engineering...")
        data_with_features = engineer_features(data.clone())
        
        logger.info("Generating signals from strategy...")
        data_with_signals = self.strategy.generate_signals(data_with_features)
        
        data = data_with_signals

        if self.selected_markets:
            data = data.filter(pl.col('market').is_in(self.selected_markets))
            if data.is_empty():
                print(f"Warning: No data found for selected markets: {self.selected_markets}. Backtest will not run.")
                self.data = data
                return

        self.data = data

        if not data.is_empty():
            initial_timestamp = data['ts_utc'].min()
            self.equity_curve.append({
                'timestamp': initial_timestamp, 'portfolio_value': self.initial_capital,
                'cash': self.initial_capital, 'positions': {}
            })
        
        data = data.sort('ts_utc')
        active_markets = data['market'].unique().to_list()
        if not active_markets:
             print("Warning: No active markets to trade after filtering.")
             return

        in_trade_for_market: Dict[str, bool] = {market: False for market in active_markets}
        
        for row in data.iter_rows(named=True):
            market = row['market']
            current_price = row['mid_price']
            volatility = row.get('volatility_20p')
            if volatility is None:
                volatility = 0.01
            confidence = row.get('confidence', 0.0)
            idx = row['ts_utc']

            if in_trade_for_market.get(market, False) and market in self.positions:
                pos = self.positions[market]
                exit_reason = None
                if 'tp_price' in pos and current_price >= pos['tp_price']: exit_reason = 'TP'
                elif 'sl_price' in pos and current_price <= pos['sl_price']: exit_reason = 'SL'
                # elif 'time_barrier' in pos and idx >= pos['time_barrier']: exit_reason = 'Time'
                
                if exit_reason:
                    logger.info(f"Exit for {market} at {current_price:.2f} due to {exit_reason} at {idx}")
                    self.execute_trade(market, current_price, 0, 'exit', idx, exit_reason)
                    in_trade_for_market[market] = False
                    continue

            if not in_trade_for_market.get(market, False) and row.get('entry_signal', False) and confidence >= self.confidence_threshold:
                position_size, size_reason = self.calculate_position_size(current_price, volatility, market)
                if position_size > 0:
                    self.execute_trade(market, current_price, position_size, 'long', idx)
                    in_trade_for_market[market] = True
                elif self.verbose_logging:
                    # Log why position wasn't taken
                    print(f"[{idx.strftime('%Y-%m-%d %H:%M:%S')}] \033[93mSKIP\033[0m | {market} | "
                          f"Signal detected but position size = 0 | Reason: {size_reason}")
            
            elif in_trade_for_market.get(market, False) and row.get('exit_signal', False):
                if market in self.positions:
                    self.execute_trade(market, current_price, 0, 'exit', idx, 'Signal')
                    in_trade_for_market[market] = False

        if not data.is_empty():
            last_prices = data.group_by('market').agg(pl.last('mid_price').alias('mid_price'), pl.last('ts_utc').alias('ts_utc'))
            for row in last_prices.iter_rows(named=True):
                market = row['market']
                if in_trade_for_market.get(market, False) and market in self.positions:
                    self.execute_trade(market, row['mid_price'], 0, 'exit', row['ts_utc'], 'End')
                    in_trade_for_market[market] = False
        else:
             print("Warning: Data is empty, cannot close open positions.")
                
    def get_results(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        trade_df = pl.DataFrame(self.trade_log)
        equity_df = pl.DataFrame(self.equity_curve)
        return trade_df, equity_df

    def calculate_metrics(self, trade_df: pl.DataFrame, equity_df: pl.DataFrame) -> dict:
        metrics = {
            'final_portfolio': self.portfolio,
            'return_pct': ((self.portfolio / self.initial_capital) - 1) * 100,
            'num_trades': trade_df.shape[0] // 2
        }
        
        # Calculate risk-adjusted metrics
        if not equity_df.is_empty():
            # Get current portfolio risk state for final analysis
            current_prices = {}
            if self.data is not None and not self.data.is_empty():
                latest_data = self.data.group_by('market').agg(pl.last('mid_price').alias('mid_price'))
                for row in latest_data.iter_rows(named=True):
                    current_prices[row['market']] = row['mid_price']
            
            # Portfolio risk metrics
            portfolio_risk = self.risk_manager._calculate_portfolio_risk(
                self.portfolio, self.positions, current_prices
            )
            
            metrics.update({
                'final_exposure_pct': portfolio_risk.total_exposure_pct * 100,
                'final_cash_pct': portfolio_risk.available_cash_pct * 100,
                'final_risk_score': portfolio_risk.risk_score,
                'max_concurrent_positions': len(self.positions)
            })
        
        if not trade_df.is_empty():
            profits = trade_df.filter(pl.col('action') == 'exit')['profit']
            win_rate = (profits > 0).mean() * 100
            
            # Position size analysis
            entry_trades = trade_df.filter(pl.col('action') == 'enter')
            if not entry_trades.is_empty():
                notional_values = entry_trades['size'] * entry_trades['price']
                avg_position_size = notional_values.mean()
                max_position_size = notional_values.max()
                position_size_std = notional_values.std()
                
                metrics.update({
                    'avg_position_size': avg_position_size,
                    'max_position_size': max_position_size,
                    'position_size_std': position_size_std,
                    'avg_position_pct': (avg_position_size / self.initial_capital) * 100,
                    'max_position_pct': (max_position_size / self.initial_capital) * 100
                })
            
            equity_returns = equity_df['portfolio_value'].pct_change().drop_nulls()
            
            if len(equity_returns) > 1 and equity_returns.std() > 0:
                time_diff_seconds = (equity_df['timestamp'][1] - equity_df['timestamp'][0]).total_seconds()
                periods_per_day = (24 * 60 * 60) / time_diff_seconds if time_diff_seconds > 0 else 1
                intervals_per_year = periods_per_day * 365
                sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * np.sqrt(intervals_per_year)
                
                # Additional risk metrics
                downside_returns = equity_returns.filter(equity_returns < 0)
                if len(downside_returns) > 0:
                    sortino_ratio = (equity_returns.mean() / downside_returns.std()) * np.sqrt(intervals_per_year)
                else:
                    sortino_ratio = float('inf') if equity_returns.mean() > 0 else 0.0
                
                # Value at Risk (95% confidence)
                var_95 = np.percentile(equity_returns.to_numpy(), 5) * 100
            else:
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                var_95 = 0.0
            
            metrics.update({
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'var_95_pct': var_95,
                'max_drawdown': (equity_df['portfolio_value'] / equity_df['portfolio_value'].cum_max() - 1).min() * 100,
                'volatility_annualized': equity_returns.std() * np.sqrt(intervals_per_year) * 100 if len(equity_returns) > 1 else 0.0
            })
        
        return metrics
    
    def print_risk_summary(self):
        """Print a comprehensive risk summary at the end of backtest"""
        if not self.verbose_logging:
            return
            
        current_prices = {}
        if self.data is not None and not self.data.is_empty():
            latest_data = self.data.group_by('market').agg(pl.last('mid_price').alias('mid_price'))
            for row in latest_data.iter_rows(named=True):
                current_prices[row['market']] = row['mid_price']
        
        risk_report = self.risk_manager.format_risk_report(
            self.portfolio, self.positions, current_prices
        )
        
        print("\n" + "="*60)
        print(risk_report)
        print("="*60)

if __name__ == "__main__":
    print("Trend Following Backtester ready")
