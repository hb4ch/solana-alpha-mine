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

class BacktestLogger:
    """Handles all logging and display for backtesting"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def format_currency(self, amount: float) -> str:
        """Format currency amounts for display"""
        return f"${amount:,.2f}"
    
    def format_percentage(self, value: float) -> str:
        """Format percentage for display"""
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"
    
    def log_entry(self, timestamp: datetime.datetime, market: str, price: float, 
                  size: float, notional: float, margin: float, fee: float, 
                  leverage: float, portfolio_before: float, portfolio_after: float,
                  tp_price: float = None, sl_price: float = None):
        """Log trade entry"""
        if not self.verbose:
            return
            
        leverage_info = f" | {leverage}x Leverage" if leverage > 1 else ""
        tp_info = f" | TP: {self.format_currency(tp_price)}" if tp_price else ""
        sl_info = f" | SL: {self.format_currency(sl_price)}" if sl_price else ""
        
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] \033[94mENTRY\033[0m | {market} | "
              f"Price: {self.format_currency(price)} | Size: {size:.4f} | "
              f"Notional: {self.format_currency(notional)} | "
              f"Margin: {self.format_currency(margin)} | "
              f"Fee: {self.format_currency(fee)}{leverage_info}{tp_info}{sl_info} | "
              f"Portfolio: {self.format_currency(portfolio_before)} → {self.format_currency(portfolio_after)}")
    
    def log_exit(self, timestamp: datetime.datetime, market: str, price: float, 
                 entry_price: float, size: float, profit: float, fee: float,
                 funding_cost: float, holding_minutes: float, exit_reason: str,
                 portfolio_before: float, portfolio_after: float):
        """Log trade exit"""
        if not self.verbose:
            return
            
        pnl_color = "\033[92m" if profit >= 0 else "\033[91m"  # Green for profit, red for loss
        pnl_pct = ((price / entry_price) - 1) * 100
        exit_reason_info = f" | Reason: {exit_reason}" if exit_reason else ""
        funding_info = f" | Funding: {self.format_currency(funding_cost)}" if funding_cost > 0 else ""
        
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] \033[93mEXIT\033[0m | {market} | "
              f"Price: {self.format_currency(price)} | Size: {size:.4f} | "
              f"Entry: {self.format_currency(entry_price)} | "
              f"P&L: {pnl_color}{self.format_currency(profit)} ({self.format_percentage(pnl_pct)})\033[0m | "
              f"Fee: {self.format_currency(fee)}{funding_info} | "
              f"Hold: {holding_minutes:.1f}m{exit_reason_info} | "
              f"Portfolio: {self.format_currency(portfolio_before)} → {self.format_currency(portfolio_after)}")
    
    def log_skip(self, timestamp: datetime.datetime, market: str, reason: str):
        """Log skipped signal"""
        if not self.verbose:
            return
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] \033[93mSKIP\033[0m | {market} | {reason}")
    
    def log_insufficient_funds(self, timestamp: datetime.datetime, market: str, 
                             needed: float, available: float):
        """Log insufficient funds"""
        if not self.verbose:
            return
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] INSUFFICIENT FUNDS | {market} | "
              f"Needed: {self.format_currency(needed)}, Available: {self.format_currency(available)}")


class TradeExecutor:
    """Handles trade execution logic"""
    
    def __init__(self, logger: BacktestLogger, funding_rate_daily: float = 0.0001):
        self.logger = logger
        self.funding_rate_daily = funding_rate_daily
        self.TRADING_FEE_PCT = 0.001  # 0.1%
    
    def execute_entry(self, market: str, price: float, size: float, timestamp: datetime.datetime,
                     portfolio: float, leverage: float, positions: Dict, trade_log: list,
                     equity_curve: list, strategy) -> Tuple[float, Dict]:
        """Execute trade entry and return updated portfolio and positions"""
        if leverage == 0 and size > 0:
            return portfolio, positions
        
        notional_value = price * size
        
        if leverage == 1.0:
            margin_used = notional_value
            borrowed_amount = 0.0
        else:
            margin_used = notional_value / leverage
            borrowed_amount = notional_value - margin_used
        
        if margin_used > portfolio and not np.isclose(margin_used, portfolio):
            self.logger.log_insufficient_funds(timestamp, market, margin_used, portfolio)
            return portfolio, positions
        
        entry_fee = notional_value * self.TRADING_FEE_PCT
        old_portfolio = portfolio
        portfolio -= (margin_used + entry_fee)
        
        position_details = {
            'size': size, 'entry_price': price, 'entry_fee_paid': entry_fee,
            'margin_used': margin_used, 'borrowed_amount': borrowed_amount,
            'entry_timestamp': timestamp, 'leverage': leverage
        }
        
        # Add strategy-specific details
        tp_price = sl_price = None
        if isinstance(strategy, MLTradingStrategy):
            tp_price = price * (1 + strategy.tp_pct)
            sl_price = price * (1 - strategy.sl_pct)
            position_details['tp_price'] = tp_price
            position_details['sl_price'] = sl_price
        
        positions[market] = position_details
        
        self.logger.log_entry(timestamp, market, price, size, notional_value, 
                            margin_used, entry_fee, leverage, old_portfolio, 
                            portfolio, tp_price, sl_price)
        
        trade_log.append({
            'timestamp': timestamp, 'market': market, 'action': 'enter', 
            'price': price, 'size': size, 'fee': entry_fee, 
            'leverage': leverage, 'margin_used': margin_used
        })
        
        self._update_equity_curve(timestamp, portfolio, positions, equity_curve)
        return portfolio, positions
    
    def execute_exit(self, market: str, price: float, timestamp: datetime.datetime,
                    portfolio: float, positions: Dict, trade_log: list,
                    equity_curve: list, exit_reason: str = None) -> Tuple[float, Dict, float]:
        """Execute trade exit and return updated portfolio, positions, and profit"""
        if market not in positions:
            return portfolio, positions, 0.0
        
        pos_details = positions[market]
        exit_notional_value = price * pos_details['size']
        exit_fee = exit_notional_value * self.TRADING_FEE_PCT
        
        # Calculate funding cost
        funding_cost = 0.0
        if pos_details['leverage'] > 1.0 and pos_details['borrowed_amount'] > 0:
            days_held = (timestamp - pos_details['entry_timestamp']).total_seconds() / (24 * 60 * 60)
            funding_cost = pos_details['borrowed_amount'] * self.funding_rate_daily * max(0, days_held)
        
        pnl_on_notional = (price - pos_details['entry_price']) * pos_details['size']
        old_portfolio = portfolio
        portfolio += (pos_details['margin_used'] + pnl_on_notional - exit_fee - funding_cost)
        profit_for_trade = pnl_on_notional - pos_details['entry_fee_paid'] - exit_fee - funding_cost
        
        # Calculate holding period
        holding_period = timestamp - pos_details['entry_timestamp']
        holding_minutes = holding_period.total_seconds() / 60
        
        self.logger.log_exit(timestamp, market, price, pos_details['entry_price'],
                           pos_details['size'], profit_for_trade, exit_fee, 
                           funding_cost, holding_minutes, exit_reason,
                           old_portfolio, portfolio)
        
        trade_log.append({
            'timestamp': timestamp, 'market': market, 'action': 'exit', 
            'price': price, 'size': pos_details['size'], 'fee': exit_fee, 
            'funding_cost': funding_cost, 'profit': profit_for_trade, 
            'leverage': pos_details['leverage']
        })
        
        del positions[market]
        self._update_equity_curve(timestamp, portfolio, positions, equity_curve)
        return portfolio, positions, profit_for_trade
    
    def _update_equity_curve(self, timestamp: datetime.datetime, portfolio: float, 
                           positions: Dict, equity_curve: list):
        """Update equity curve with current state"""
        equity_curve.append({
            'timestamp': timestamp, 'portfolio_value': portfolio, 'cash': portfolio,
            'positions': {m: {'size': p['size'], 'entry_price': p['entry_price'], 
                             'leverage': p['leverage']} for m, p in positions.items()}
        })


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
        
        # Statistics tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Initialize components
        self.logger = BacktestLogger(verbose_logging)
        self.trade_executor = TradeExecutor(self.logger, funding_rate_daily)
        
        # Initialize risk management
        if risk_config is None:
            risk_config = RiskConfig()
            risk_config.max_risk_per_trade_pct = risk_per_trade
        self.risk_manager = RiskManager(risk_config)
        
        logger.info(f"Backtester initialized with risk management: "
                   f"max_position={risk_config.max_position_pct:.1%}, "
                   f"max_exposure={risk_config.max_total_exposure_pct:.1%}, "
                   f"risk_per_trade={risk_config.max_risk_per_trade_pct:.1%}")

    def calculate_position_size(self, price: float, volatility: float, market: str) -> Tuple[float, str]:
        """Calculate position size using the risk manager for safe sizing"""
        # Handle None values safely
        if volatility is None or volatility <= 0 or self.leverage == 0 or price is None or price <= 0:
            return 0.0, "Invalid inputs (price, volatility, or leverage)"
        
        position_size, reason = self.risk_manager.calculate_safe_position_size(
            portfolio_value=self.portfolio,
            price=price,
            volatility=volatility,
            current_positions=self.positions,
            market=market,
            leverage=self.leverage
        )
        
        return position_size, reason

    def _get_price_from_row(self, row: dict) -> float:
        """Extract price from data row, trying different price columns"""
        for price_col in ['mid_price_calc', 'mid_price', 'wap', 'close']:
            if price_col in row and row[price_col] is not None:
                return row[price_col]
        return None

    def _check_exit_conditions(self, market: str, current_price: float) -> str:
        """Check if position should be exited and return reason"""
        if market not in self.positions:
            return None
        
        pos = self.positions[market]
        if 'tp_price' in pos and current_price >= pos['tp_price']:
            return 'TP'
        elif 'sl_price' in pos and current_price <= pos['sl_price']:
            return 'SL'
        return None

    def _process_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Process data and generate trading signals"""
        logger.info("Running feature engineering...")
        data_with_features = engineer_features(data.clone())
        
        logger.info("Generating signals from strategy...")
        data_with_signals = self.strategy.generate_signals(data_with_features)
        
        if self.selected_markets:
            data_with_signals = data_with_signals.filter(pl.col('market').is_in(self.selected_markets))
            if data_with_signals.is_empty():
                logger.warning(f"No data found for selected markets: {self.selected_markets}")
        
        return data_with_signals.sort('ts_utc')

    def backtest(self, data: pl.DataFrame):
        """Run the backtest on provided data"""
        # Reset state
        self.portfolio = self.initial_capital
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Process data and generate signals
        processed_data = self._process_signals(data)
        if processed_data.is_empty():
            logger.warning("No data to backtest after processing")
            return
        
        self.data = processed_data
        
        # Initialize equity curve
        if not processed_data.is_empty():
            initial_timestamp = processed_data['ts_utc'].min()
            self.equity_curve.append({
                'timestamp': initial_timestamp, 'portfolio_value': self.initial_capital,
                'cash': self.initial_capital, 'positions': {}
            })
        
        # Get active markets
        active_markets = processed_data['market'].unique().to_list()
        if not active_markets:
            logger.warning("No active markets to trade after filtering")
            return
        
        # Track which markets are in trade
        in_trade_for_market: Dict[str, bool] = {market: False for market in active_markets}
        
        # Main trading loop
        for row in processed_data.iter_rows(named=True):
            market = row['market']
            current_price = self._get_price_from_row(row)
            if current_price is None:
                continue
            
            volatility = row.get('volatility_20p', 0.01)
            if volatility is None:
                volatility = 0.01  # Default volatility if None
            confidence = row.get('confidence', 0.0)
            timestamp = row['ts_utc']
            
            # Check exit conditions for existing positions
            if in_trade_for_market.get(market, False):
                exit_reason = self._check_exit_conditions(market, current_price)
                if exit_reason:
                    logger.info(f"Exit for {market} at {current_price:.2f} due to {exit_reason} at {timestamp}")
                    self.portfolio, self.positions, profit = self.trade_executor.execute_exit(
                        market, current_price, timestamp, self.portfolio, 
                        self.positions, self.trade_log, self.equity_curve, exit_reason
                    )
                    self._update_trade_stats(profit)
                    in_trade_for_market[market] = False
                    continue
            
            # Check entry conditions
            if (not in_trade_for_market.get(market, False) and 
                row.get('entry_signal', False) and 
                confidence >= self.confidence_threshold):
                
                position_size, size_reason = self.calculate_position_size(current_price, volatility, market)
                if position_size > 0:
                    self.portfolio, self.positions = self.trade_executor.execute_entry(
                        market, current_price, position_size, timestamp, self.portfolio,
                        self.leverage, self.positions, self.trade_log, self.equity_curve, self.strategy
                    )
                    in_trade_for_market[market] = True
                else:
                    self.logger.log_skip(timestamp, market, f"Position size = 0 | Reason: {size_reason}")
            
            # Check manual exit signals
            elif (in_trade_for_market.get(market, False) and row.get('exit_signal', False)):
                self.portfolio, self.positions, profit = self.trade_executor.execute_exit(
                    market, current_price, timestamp, self.portfolio, 
                    self.positions, self.trade_log, self.equity_curve, 'Signal'
                )
                self._update_trade_stats(profit)
                in_trade_for_market[market] = False
        
        # Close any remaining positions
        self._close_remaining_positions(processed_data, in_trade_for_market)

    def _update_trade_stats(self, profit: float):
        """Update trade statistics"""
        self.total_trades += 1
        self.total_pnl += profit
        if profit > 0:
            self.winning_trades += 1

    def _close_remaining_positions(self, data: pl.DataFrame, in_trade_for_market: Dict[str, bool]):
        """Close any remaining open positions at the end of backtest"""
        if data.is_empty():
            logger.warning("Data is empty, cannot close open positions")
            return
        
        # Try different price columns as fallback
        price_col = None
        for col in ['mid_price_calc', 'mid_price', 'wap', 'close']:
            if col in data.columns:
                price_col = col
                break
        
        if price_col is None:
            logger.error("No price column found for closing positions")
            return
        
        last_prices = data.group_by('market').agg(
            pl.last(price_col).alias('mid_price'), 
            pl.last('ts_utc').alias('ts_utc')
        )
        
        for row in last_prices.iter_rows(named=True):
            market = row['market']
            if in_trade_for_market.get(market, False) and market in self.positions:
                self.portfolio, self.positions, profit = self.trade_executor.execute_exit(
                    market, row['mid_price'], row['ts_utc'], self.portfolio, 
                    self.positions, self.trade_log, self.equity_curve, 'End'
                )
                self._update_trade_stats(profit)
                in_trade_for_market[market] = False

    def get_results(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Get backtest results as DataFrames"""
        trade_df = pl.DataFrame(self.trade_log)
        equity_df = pl.DataFrame(self.equity_curve)
        return trade_df, equity_df

    def calculate_metrics(self, trade_df: pl.DataFrame, equity_df: pl.DataFrame) -> dict:
        """Calculate comprehensive backtest metrics"""
        metrics = {
            'final_portfolio': self.portfolio,
            'return_pct': ((self.portfolio / self.initial_capital) - 1) * 100,
            'num_trades': trade_df.shape[0] // 2 if not trade_df.is_empty() else 0
        }
        
        # Add risk-adjusted metrics
        if not equity_df.is_empty():
            current_prices = self._get_current_prices()
            portfolio_risk = self.risk_manager._calculate_portfolio_risk(
                self.portfolio, self.positions, current_prices
            )
            
            metrics.update({
                'final_exposure_pct': portfolio_risk.total_exposure_pct * 100,
                'final_cash_pct': portfolio_risk.available_cash_pct * 100,
                'final_risk_score': portfolio_risk.risk_score,
                'max_concurrent_positions': len(self.positions)
            })
        
        # Add trading performance metrics
        if not trade_df.is_empty():
            self._add_trading_metrics(metrics, trade_df, equity_df)
        
        return metrics

    def _get_current_prices(self) -> dict:
        """Get current market prices"""
        current_prices = {}
        if self.data is not None and not self.data.is_empty():
            # Try different price columns as fallback
            price_col = None
            for col in ['mid_price_calc', 'mid_price', 'wap', 'close']:
                if col in self.data.columns:
                    price_col = col
                    break
            
            if price_col is not None:
                latest_data = self.data.group_by('market').agg(pl.last(price_col).alias('mid_price'))
                for row in latest_data.iter_rows(named=True):
                    current_prices[row['market']] = row['mid_price']
        return current_prices

    def _add_trading_metrics(self, metrics: dict, trade_df: pl.DataFrame, equity_df: pl.DataFrame):
        """Add detailed trading performance metrics"""
        profits = trade_df.filter(pl.col('action') == 'exit')['profit']
        if not profits.is_empty():
            win_rate = (profits > 0).mean() * 100
            metrics['win_rate'] = win_rate
        
        # Position size analysis
        entry_trades = trade_df.filter(pl.col('action') == 'enter')
        if not entry_trades.is_empty():
            notional_values = entry_trades['size'] * entry_trades['price']
            metrics.update({
                'avg_position_size': notional_values.mean(),
                'max_position_size': notional_values.max(),
                'avg_position_pct': (notional_values.mean() / self.initial_capital) * 100,
                'max_position_pct': (notional_values.max() / self.initial_capital) * 100
            })
        
        # Risk metrics
        if len(equity_df) > 1:
            self._add_risk_metrics(metrics, equity_df)

    def _add_risk_metrics(self, metrics: dict, equity_df: pl.DataFrame):
        """Add risk-related metrics"""
        equity_returns = equity_df['portfolio_value'].pct_change().drop_nulls()
        
        if len(equity_returns) > 1 and equity_returns.std() > 0:
            # Calculate annualized metrics
            time_diff_seconds = (equity_df['timestamp'][1] - equity_df['timestamp'][0]).total_seconds()
            periods_per_day = (24 * 60 * 60) / time_diff_seconds if time_diff_seconds > 0 else 1
            intervals_per_year = periods_per_day * 365
            
            sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * np.sqrt(intervals_per_year)
            
            # Sortino ratio
            downside_returns = equity_returns.filter(equity_returns < 0)
            sortino_ratio = ((equity_returns.mean() / downside_returns.std()) * np.sqrt(intervals_per_year) 
                           if len(downside_returns) > 0 else float('inf') if equity_returns.mean() > 0 else 0.0)
            
            # VaR and other metrics
            var_95 = np.percentile(equity_returns.to_numpy(), 5) * 100
            max_drawdown = (equity_df['portfolio_value'] / equity_df['portfolio_value'].cum_max() - 1).min() * 100
            volatility_annualized = equity_returns.std() * np.sqrt(intervals_per_year) * 100
            
            metrics.update({
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'var_95_pct': var_95,
                'max_drawdown': max_drawdown,
                'volatility_annualized': volatility_annualized
            })

    def print_risk_summary(self):
        """Print a comprehensive risk summary at the end of backtest"""
        if not self.logger.verbose:
            return
        
        current_prices = self._get_current_prices()
        risk_report = self.risk_manager.format_risk_report(
            self.portfolio, self.positions, current_prices
        )
        
        print("\n" + "="*60)
        print(risk_report)
        print("="*60)


if __name__ == "__main__":
    print("Trend Following Backtester ready")
