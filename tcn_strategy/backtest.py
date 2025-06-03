"""
Professional backtesting framework for TCN-based quantitative trading strategy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import torch

from config import Config
from model import create_model
from trainer import TCNTrainer

warnings.filterwarnings('ignore')

@dataclass
class Trade:
    """Individual trade record"""
    timestamp: pd.Timestamp
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0.0
    pnl: Optional[float] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    exit_reason: str = ""  # 'signal', 'stop_loss', 'take_profit', 'time_limit'

class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return {}
        
        # Remove NaN values
        returns = returns.dropna()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 * 17280 / len(returns)) - 1  # Assuming 5-second intervals
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 17280)  # Annualized volatility
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Downside metrics
        negative_returns = returns[returns < 0]
        downside_volatility = negative_returns.std() * np.sqrt(252 * 17280) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Win/Loss analysis
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else float('inf')
        
        # Additional metrics
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR (Value at Risk)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'var_95': var_95,
            'var_99': var_99,
            'num_trades': len(returns),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        # Benchmark comparison
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.dropna()
            if len(benchmark_returns) > 0:
                benchmark_total_return = (1 + benchmark_returns).prod() - 1
                alpha = total_return - benchmark_total_return
                
                # Beta calculation
                covariance = returns.cov(benchmark_returns)
                benchmark_variance = benchmark_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Information ratio
                active_returns = returns - benchmark_returns
                tracking_error = active_returns.std()
                information_ratio = active_returns.mean() / tracking_error if tracking_error > 0 else 0
                
                metrics.update({
                    'alpha': alpha,
                    'beta': beta,
                    'information_ratio': information_ratio,
                    'tracking_error': tracking_error
                })
        
        return metrics

class TCNBacktester:
    """
    Professional backtesting framework with realistic market simulation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.trades: List[Trade] = []
        self.portfolio_value = []
        self.positions = []
        self.cash = config.backtest.initial_capital
        self.current_exposure = 0.0
        
        # Performance tracking
        self.daily_returns = []
        self.transaction_costs = []
        self.timestamps = []
        
    def calculate_position_size(self, signal_strength: float, price: float, volatility: float) -> float:
        """Calculate position size based on signal strength and risk management"""
        # Base position size
        base_size = self.config.backtest.max_position_size * abs(signal_strength)
        
        # Risk-adjusted sizing based on volatility
        if self.config.risk.position_sizing_method == "volatility":
            vol_adjustment = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0
            base_size *= vol_adjustment
        
        # Kelly criterion adjustment
        # elif self.config.risk.position_sizing_method == "kelly":
        #     # Simplified Kelly: assume 55% win rate, 1:1 risk/reward
        #     # win_rate = 0.55
        #     # avg_win = 0.01
        #     # avg_loss = -0.01
        #     # kelly_fraction = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
        #     # kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
        #     # base_size = kelly_fraction
        # Fallback to signal strength based sizing if Kelly is not appropriately configured or disabled
        # Ensure base_size is defined if not using 'volatility' or 'kelly'
        # This was already: base_size = self.config.backtest.max_position_size * abs(signal_strength)

        # Apply maximum exposure limit
        available_exposure = self.config.backtest.max_total_exposure - self.current_exposure
        base_size = min(base_size, available_exposure)
        
        # Convert to dollar amount
        position_value = self.cash * base_size
        shares = position_value / price
        
        return shares
    
    def calculate_transaction_cost(self, value: float) -> float:
        """Calculate realistic transaction costs"""
        cost_bps = self.config.backtest.transaction_cost_bps
        return value * (cost_bps / 10000)
    
    def simulate_slippage(self, price: float, volume: float, side: str) -> float:
        """Simulate market impact and slippage"""
        # Simple slippage model based on volume
        base_slippage = 0.0001  # 1 bps base slippage
        volume_impact = min(0.0005, volume / 1000000)  # Additional impact for larger orders
        
        slippage_factor = base_slippage + volume_impact
        
        if side == 'buy':
            return price * (1 + slippage_factor)
        else:
            return price * (1 - slippage_factor)
    
    def check_stop_loss(self, current_price: float, entry_price: float, side: str) -> bool:
        """Check if stop loss should be triggered"""
        stop_loss_pct = self.config.backtest.stop_loss_pct
        
        if side == 'long':
            return current_price <= entry_price * (1 - stop_loss_pct)
        else:  # short
            return current_price >= entry_price * (1 + stop_loss_pct)
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value based on current positions and prices"""
        total_value = self.cash
        
        for trade in self.trades:
            if trade.exit_price is None:  # Open position
                current_price = current_prices.get('SOL_USDC', trade.entry_price)
                
                if trade.side == 'long':
                    position_value = trade.size * current_price
                else:  # short
                    position_value = trade.size * (2 * trade.entry_price - current_price)
                
                total_value += position_value - (trade.size * trade.entry_price)
        
        return total_value
    
    def run_backtest(self, model: torch.nn.Module, test_data: Tuple, df_processed: pd.DataFrame) -> Dict:
        """
        Run comprehensive backtest with realistic market simulation
        """
        print("Starting backtest simulation...")
        
        X_test, targets = test_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Get model predictions in batches to avoid GPU memory issues
        batch_size = 256  # Process in smaller batches
        all_predictions = {f'return_{h}': [] for h in [1, 5, 10, 30]}
        all_predictions.update({f'direction_{h}': [] for h in [1, 5, 10, 30]})
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_end = min(i + batch_size, len(X_test))
                X_batch = torch.FloatTensor(X_test[i:batch_end]).to(device)
                
                batch_predictions = model(X_batch)
                
                # Store predictions
                for key, value in batch_predictions.items():
                    all_predictions[key].append(value.cpu())
        
        # Concatenate all batch predictions
        predictions = {}
        for key, value_list in all_predictions.items():
            if value_list:  # Only concatenate if we have predictions
                predictions[key] = torch.cat(value_list, dim=0)
        
        # Extract prediction data
        # Use a potentially more stable, longer prediction horizon
        horizon_index = 1 # Index 0 is 1-step, Index 1 is 5-steps, etc.
        if len(self.config.data.prediction_horizons) <= horizon_index:
            print(f"Warning: Horizon index {horizon_index} is out of bounds. Using shortest horizon.")
            horizon_index = 0
        main_horizon = self.config.data.prediction_horizons[horizon_index] 
        
        pred_returns = predictions[f'return_{main_horizon}'].squeeze().cpu().numpy()
        pred_directions = torch.softmax(predictions[f'direction_{main_horizon}'], dim=1)[:, 1].cpu().numpy()  # Probability of up
        
        # Get corresponding market data (need last portion for test set)
        test_start_idx = len(df_processed) - len(X_test) - max(self.config.data.prediction_horizons)
        test_df = df_processed.iloc[test_start_idx:test_start_idx + len(X_test)].copy()
        test_df.reset_index(drop=True, inplace=True)
        
        # Initialize tracking variables
        portfolio_values = [self.config.backtest.initial_capital]
        positions = []
        returns = []
        timestamps = []
        open_trades = []
        max_portfolio_value_so_far = self.config.backtest.initial_capital
        stop_trading_due_to_drawdown = False
        
        for i in range(len(pred_returns)):
            current_time = test_df.iloc[i]['ts_utc'] if 'ts_utc' in test_df.columns else pd.Timestamp.now()
            current_price = test_df.iloc[i]['mid_price']
            current_volatility = test_df.iloc[i].get('return_std_60', 0.01)
            
            timestamps.append(current_time)
            
            # Generate trading signal
            confidence = abs(pred_directions[i] - 0.5) * 2  # Convert to 0-1 scale
            signal_strength = confidence if confidence > self.config.backtest.confidence_threshold else 0
            
            # Determine position direction
            if pred_directions[i] > 0.5:
                signal_side = 'long'
            else:
                signal_side = 'short'
            
            # Close existing positions if signal changes or stop loss triggered
            for trade in open_trades[:]:  # Copy list to avoid modification during iteration
                should_close = False
                exit_reason = ""
                
                # Check stop loss
                if self.check_stop_loss(current_price, trade.entry_price, trade.side):
                    should_close = True
                    exit_reason = "stop_loss"
                
                # Check signal reversal
                elif (trade.side == 'long' and signal_side == 'short' and signal_strength > 0) or \
                     (trade.side == 'short' and signal_side == 'long' and signal_strength > 0):
                    should_close = True
                    exit_reason = "signal_reversal"
                
                if should_close:
                    # Close position
                    exit_price = self.simulate_slippage(current_price, abs(trade.size), 'sell' if trade.side == 'long' else 'buy')
                    
                    # Calculate PnL
                    if trade.side == 'long':
                        pnl = trade.size * (exit_price - trade.entry_price)
                    else:  # short
                        pnl = trade.size * (trade.entry_price - exit_price)
                    
                    # Apply transaction costs
                    transaction_cost = self.calculate_transaction_cost(abs(trade.size * exit_price))
                    pnl -= transaction_cost
                    
                    # Update trade record
                    trade.exit_price = exit_price
                    trade.exit_timestamp = current_time
                    trade.pnl = pnl
                    trade.exit_reason = exit_reason
                    
                    # Update cash and exposure
                    self.cash += trade.size * trade.entry_price + pnl
                    self.current_exposure -= abs(trade.size * trade.entry_price) / portfolio_values[-1]
                    
                    # Remove from open trades
                    open_trades.remove(trade)
            
            # Open new position if signal is strong enough and not stopped by drawdown
            if not stop_trading_due_to_drawdown and signal_strength > 0 and len(open_trades) < 3:  # Limit concurrent positions
                position_size = self.calculate_position_size(signal_strength, current_price, current_volatility)
                
                if position_size > 0:
                    entry_price = self.simulate_slippage(current_price, position_size, 'buy' if signal_side == 'long' else 'sell')
                    position_value = position_size * entry_price
                    
                    # Check if we have enough cash
                    if position_value <= self.cash:
                        # Create new trade
                        trade = Trade(
                            timestamp=current_time,
                            side=signal_side,
                            entry_price=entry_price,
                            size=position_size
                        )
                        
                        # Apply transaction costs
                        transaction_cost = self.calculate_transaction_cost(position_value)
                        
                        # Update cash and exposure
                        self.cash -= position_value + transaction_cost
                        self.current_exposure += position_value / portfolio_values[-1]
                        
                        # Add to tracking
                        self.trades.append(trade)
                        open_trades.append(trade)
            
            # Calculate current portfolio value
            current_portfolio_value = self.cash
            for trade in open_trades:
                # For short positions, PnL is entry_price - current_price.
                # Value of short position = size * entry_price + size * (entry_price - current_price)
                # = size * (2 * entry_price - current_price)
                # This seems correct for calculating the equity value of a short position.
                if trade.side == 'long':
                    position_value = trade.size * current_price
                else:  # short
                    # The value of a short position for portfolio calculation is:
                    # Initial margin (trade.size * trade.entry_price) + PnL
                    # PnL for short = trade.size * (trade.entry_price - current_price)
                    # So, current_portfolio_value should reflect cash + current value of open positions.
                    # Cash was reduced by (trade.size * trade.entry_price) when opening.
                    # The current value of the short position part of equity is (trade.size * trade.entry_price) + trade.size * (trade.entry_price - current_price)
                    # = trade.size * (2 * trade.entry_price - current_price)
                    # This seems to be calculating the asset value, not the equity change.
                    # Let's adjust:
                    # current_portfolio_value = self.cash + sum of (current_value_of_longs) + sum of (cash_equivalent_of_shorts)
                    # For a short position, cash increased by (size * entry_price) (ignoring margin for simplicity here, assuming full collateralization)
                    # and then decreases by size * (current_price - entry_price)
                    # The existing logic: current_portfolio_value += position_value - (trade.size * trade.entry_price)
                    # For long: current_portfolio_value += trade.size * current_price - trade.size * trade.entry_price
                    # For short: current_portfolio_value += trade.size * (2 * trade.entry_price - current_price) - trade.size * trade.entry_price
                    #            = trade.size * entry_price - trade.size * current_price
                    # This is correct: it's the PnL of the short position.
                    position_pnl = 0
                    if trade.side == 'long':
                        position_pnl = trade.size * (current_price - trade.entry_price)
                    else: # short
                        position_pnl = trade.size * (trade.entry_price - current_price)
                    current_portfolio_value += (trade.size * trade.entry_price) + position_pnl # Add back initial value + PnL
                
            portfolio_values.append(current_portfolio_value)
            
            # Update max portfolio value and check drawdown
            max_portfolio_value_so_far = max(max_portfolio_value_so_far, current_portfolio_value)
            current_drawdown = (max_portfolio_value_so_far - current_portfolio_value) / max_portfolio_value_so_far
            
            if not stop_trading_due_to_drawdown and current_drawdown > self.config.backtest.max_drawdown_limit:
                print(f"Max drawdown limit of {self.config.backtest.max_drawdown_limit*100:.2f}% breached at {current_time}. Current drawdown: {current_drawdown*100:.2f}%.")
                print("Closing all positions and stopping new trades.")
                stop_trading_due_to_drawdown = True
                # Close all open positions
                for trade_to_close in open_trades[:]:
                    exit_price_dd = self.simulate_slippage(current_price, abs(trade_to_close.size), 'sell' if trade_to_close.side == 'long' else 'buy')
                    if trade_to_close.side == 'long':
                        pnl_dd = trade_to_close.size * (exit_price_dd - trade_to_close.entry_price)
                    else: # short
                        pnl_dd = trade_to_close.size * (trade_to_close.entry_price - exit_price_dd)
                    
                    transaction_cost_dd = self.calculate_transaction_cost(abs(trade_to_close.size * exit_price_dd))
                    pnl_dd -= transaction_cost_dd
                    
                    trade_to_close.exit_price = exit_price_dd
                    trade_to_close.exit_timestamp = current_time
                    trade_to_close.pnl = pnl_dd
                    trade_to_close.exit_reason = "max_drawdown_limit"
                    
                    self.cash += trade_to_close.size * trade_to_close.entry_price + pnl_dd # Add back initial value + PnL
                    self.current_exposure -= abs(trade_to_close.size * trade_to_close.entry_price) / portfolio_values[-2] # Use previous portfolio value for exposure calc
                    open_trades.remove(trade_to_close)
                
                # Recalculate portfolio value after closing positions due to drawdown
                current_portfolio_value = self.cash # All positions closed
                portfolio_values[-1] = current_portfolio_value # Update the last record

            # Calculate period return
            if len(portfolio_values) > 1:
                period_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                returns.append(period_return)
            
            positions.append(len(open_trades)) # Record number of open trades after potential closures
        
        # Close any remaining open positions at the end of the backtest
        final_price = test_df.iloc[-1]['mid_price']
        for trade in open_trades:
            exit_price = self.simulate_slippage(final_price, abs(trade.size), 'sell' if trade.side == 'long' else 'buy')
            
            if trade.side == 'long':
                pnl = trade.size * (exit_price - trade.entry_price)
            else:
                pnl = trade.size * (trade.entry_price - exit_price)
            
            transaction_cost = self.calculate_transaction_cost(abs(trade.size * exit_price))
            pnl -= transaction_cost
            
            trade.exit_price = exit_price
            trade.exit_timestamp = timestamps[-1]
            trade.pnl = pnl
            trade.exit_reason = "backtest_end"
        
        # Calculate performance metrics
        # Ensure returns and timestamps have matching lengths
        if len(returns) != len(timestamps[1:]):
            # Trim to the shorter length
            min_length = min(len(returns), len(timestamps) - 1)
            returns = returns[:min_length]
            timestamps_for_returns = timestamps[1:min_length+1]
        else:
            timestamps_for_returns = timestamps[1:]
        
        returns_series = pd.Series(returns, index=timestamps_for_returns)
        
        # Create benchmark (buy and hold)
        benchmark_returns = test_df['return_1'].dropna()
        benchmark_returns.index = timestamps[:len(benchmark_returns)]
        
        # Performance analysis
        strategy_metrics = PerformanceMetrics.calculate_metrics(returns_series, benchmark_returns)
        
        # Trade analysis
        completed_trades = [t for t in self.trades if t.exit_price is not None]
        trade_pnls = [t.pnl for t in completed_trades if t.pnl is not None]
        
        backtest_results = {
            'strategy_metrics': strategy_metrics,
            'portfolio_values': portfolio_values,
            'returns': returns,
            'timestamps': timestamps,
            'positions': positions,
            'trades': completed_trades,
            'trade_pnls': trade_pnls,
            'num_trades': len(completed_trades),
            'win_rate': len([p for p in trade_pnls if p > 0]) / len(trade_pnls) if trade_pnls else 0,
            'avg_trade_pnl': np.mean(trade_pnls) if trade_pnls else 0,
            'max_concurrent_positions': max(positions) if positions else 0,
            'total_transaction_costs': sum(self.transaction_costs),
            'final_portfolio_value': portfolio_values[-1],
            'benchmark_metrics': PerformanceMetrics.calculate_metrics(benchmark_returns)
        }
        
        print(f"Backtest completed: {len(completed_trades)} trades, "
              f"Final value: ${portfolio_values[-1]:,.2f}, "
              f"Total return: {(portfolio_values[-1]/self.config.backtest.initial_capital - 1)*100:.2f}%")
        
        return backtest_results
    
    def plot_backtest_results(self, results: Dict, save_path: str = None):
        """Create comprehensive backtest visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Portfolio value over time
        portfolio_values = results['portfolio_values']
        timestamps = results['timestamps']
        
        # Ensure timestamps and portfolio_values have same length
        min_length = min(len(timestamps), len(portfolio_values))
        timestamps = timestamps[:min_length]
        portfolio_values = portfolio_values[:min_length]
        
        axes[0, 0].plot(timestamps, portfolio_values, label='Strategy', linewidth=2)
        
        # Add benchmark if available
        if 'benchmark_metrics' in results:
            initial_value = self.config.backtest.initial_capital
            benchmark_values = [initial_value]
            for ret in results.get('benchmark_returns', []):
                benchmark_values.append(benchmark_values[-1] * (1 + ret))
            
            if len(benchmark_values) == len(timestamps):
                axes[0, 0].plot(timestamps, benchmark_values, label='Buy & Hold', alpha=0.7, linestyle='--')
        
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown analysis
        portfolio_series = pd.Series(portfolio_values)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        
        axes[0, 1].fill_between(timestamps, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(timestamps, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown Analysis')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Trade PnL distribution
        trade_pnls = results['trade_pnls']
        if trade_pnls:
            axes[1, 0].hist(trade_pnls, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(trade_pnls), color='red', linestyle='--', label=f'Mean: ${np.mean(trade_pnls):.2f}')
            axes[1, 0].set_title('Trade PnL Distribution')
            axes[1, 0].set_xlabel('PnL ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling returns
        returns = pd.Series(results['returns'])
        rolling_returns = returns.rolling(window=60).sum()  # 5-minute rolling returns
        
        axes[1, 1].plot(timestamps[1:len(rolling_returns)+1], rolling_returns, alpha=0.7)
        axes[1, 1].set_title('Rolling Returns (5-minute window)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive backtest report"""
        strategy_metrics = results['strategy_metrics']
        benchmark_metrics = results.get('benchmark_metrics', {})
        
        report = f"""
=== TCN Strategy Backtest Report ===

PERFORMANCE SUMMARY:
- Initial Capital: ${self.config.backtest.initial_capital:,.2f}
- Final Portfolio Value: ${results['final_portfolio_value']:,.2f}
- Total Return: {strategy_metrics.get('total_return', 0)*100:.2f}%
- Annualized Return: {strategy_metrics.get('annualized_return', 0)*100:.2f}%
- Maximum Drawdown: {strategy_metrics.get('max_drawdown', 0)*100:.2f}%

RISK METRICS:
- Volatility (Annualized): {strategy_metrics.get('volatility', 0)*100:.2f}%
- Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.2f}
- Sortino Ratio: {strategy_metrics.get('sortino_ratio', 0):.2f}
- Calmar Ratio: {strategy_metrics.get('calmar_ratio', 0):.2f}
- VaR (95%): {strategy_metrics.get('var_95', 0)*100:.2f}%
- VaR (99%): {strategy_metrics.get('var_99', 0)*100:.2f}%

TRADING STATISTICS:
- Total Trades: {results['num_trades']}
- Win Rate: {results['win_rate']*100:.1f}%
- Average Trade PnL: ${results['avg_trade_pnl']:.2f}
- Profit Factor: {strategy_metrics.get('profit_factor', 0):.2f}
- Max Concurrent Positions: {results['max_concurrent_positions']}

BENCHMARK COMPARISON:
- Strategy Return: {strategy_metrics.get('total_return', 0)*100:.2f}%
- Benchmark Return: {benchmark_metrics.get('total_return', 0)*100:.2f}%
- Alpha: {strategy_metrics.get('alpha', 0)*100:.2f}%
- Beta: {strategy_metrics.get('beta', 0):.2f}
- Information Ratio: {strategy_metrics.get('information_ratio', 0):.2f}
"""
        
        return report

if __name__ == "__main__":
    # Example usage
    config = Config()
    backtester = TCNBacktester(config)
    
    # This would typically be called from the main training script
    print("Backtest framework ready")
