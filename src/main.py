import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs
qs.extend_pandas()
import argparse
from data_loader import load_all_market_data
from features import calculate_combined_features
from backtest import GenericBacktester
from strategies import MACDStrategy, RSIStrategy, MovingAverageCrossover

# Configuration
MARKETS = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT']
DATA_PATH = 'crypto_tick'
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01

STRATEGIES = {
    'macd': (MACDStrategy, {'fast': 12, 'slow': 26, 'signal': 9}),
    'rsi': (RSIStrategy, {'period': 14, 'overbought': 70, 'oversold': 30}),
    'mac': (MovingAverageCrossover, {'fast_window': 10, 'slow_window': 30})
}

def parse_args():
    parser = argparse.ArgumentParser(description='Alpha Mining Backtester')
    parser.add_argument('--strategy', choices=STRATEGIES.keys(), default='macd',
                        help='Trading strategy to use (default: macd)')
    parser.add_argument('--confidence', type=float, default=0.0,
                        help='Minimum confidence threshold for trades (0-100, default: 0)')
    return parser.parse_args()

def main():
    args = parse_args()
    strategy_class, strategy_params = STRATEGIES[args.strategy]
    
    print(f"Starting alpha mining pipeline with {args.strategy.upper()} strategy (Confidence: {args.confidence})")
    
    # Step 1: Load and preprocess data
    print(f"Loading all available market data...")
    data = load_all_market_data(DATA_PATH, MARKETS)
    if data.empty:
        print("No data loaded. Exiting.")
        return
    print(f"Loaded {len(data)} records across {len(MARKETS)} markets, covering dates from {data.index.min()} to {data.index.max()}")
    
    # Step 2: Feature engineering (common features)
    print("Calculating common features...")
    data = calculate_combined_features(data)
    
    # Step 3: Backtesting
    print(f"Running backtest with {args.strategy.upper()} strategy...")
    backtester = GenericBacktester(
        strategy=strategy_class,
        strategy_params=strategy_params,
        initial_capital=INITIAL_CAPITAL, 
        risk_per_trade=RISK_PER_TRADE,
        confidence_threshold=args.confidence
    )
    backtester.backtest(data)
    
    # Step 4: Get results
    trades, equity = backtester.get_results()
    metrics = backtester.calculate_metrics(trades, equity)

    # Print Trade Log
    if not trades.empty:
        print("\n=== Detailed Trade Log ===")
        # To ensure all columns are displayed and rows are not truncated
        with pd.option_context('display.max_rows', None, 
                               'display.max_columns', None, 
                               'display.width', 1000,
                               'display.colheader_justify', 'left'):
            print(trades)
    else:
        print("\nNo trades were executed.")
    
    # Step 5: Visualization and reporting
    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title()}: {v:.2f}" if isinstance(v, float) else f"{k.replace('_', ' ').title()}: {v}")

    # Generate QuantStats HTML report
    if not equity.empty and 'timestamp' in equity.columns and 'portfolio_value' in equity.columns:
        equity_qs = equity.copy()
        equity_qs['timestamp'] = pd.to_datetime(equity_qs['timestamp'])
        equity_qs = equity_qs.set_index('timestamp')
        
        # Ensure index is unique by taking the last recorded portfolio value for any given timestamp
        equity_qs = equity_qs.loc[~equity_qs.index.duplicated(keep='last')]
        
        if len(equity_qs) < 2:
            print("\nSkipping QuantStats report: Equity curve has less than 2 unique timestamp points after processing.")
        else:
            returns = equity_qs['portfolio_value'].pct_change().dropna()
            if returns.empty:
                print("\nSkipping QuantStats report: No returns data to analyze.")
            elif returns.index.nunique() < 2:
                # This means all returns occurred at the same timestamp or there's only one return point.
                # CAGR calculation in quantstats will fail with ZeroDivisionError.
                print(f"\nSkipping QuantStats report: Not enough distinct timestamps in returns data ({returns.index.nunique()} unique). Annualized metrics like CAGR are undefined.")
            else:
                report_title = f'{args.strategy.upper()} Strategy Backtest Report (Confidence: {args.confidence})'
                report_filename = f'{args.strategy.lower()}_confidence_{args.confidence}_backtest_report.html'

                # Determine periods_per_year for annualization based on actual data frequency
                periods_per_year_for_stats = 365 * 24 * 12 # Default: approx 5-min frequency for a year (crypto)
                if len(returns.index) > 1:
                    sorted_returns_index = returns.index.to_series().sort_values()
                    time_diffs = sorted_returns_index.diff().dropna()
                    if not time_diffs.empty:
                        positive_time_diffs = time_diffs[time_diffs.dt.total_seconds() > 0]
                        if not positive_time_diffs.empty:
                            median_time_diff_seconds = positive_time_diffs.median().total_seconds()
                            if median_time_diff_seconds > 0:
                                periods_in_day = (24 * 60 * 60) / median_time_diff_seconds
                                periods_per_year_for_stats = int(periods_in_day * 365) # Crypto trades 365 days
                if periods_per_year_for_stats <= 0: # Ensure positive
                    periods_per_year_for_stats = 365 * 24 * 12 

                # Check if data spans less than a day
                idx_min_ret = returns.index.min()
                idx_max_ret = returns.index.max()
                
                if (idx_max_ret - idx_min_ret).days < 1:
                    print("\nNote: Backtest data spans less than 1 day. Full QuantStats HTML report is skipped due to potential inaccuracies with CAGR and annualized metrics.")
                    print("Displaying key intraday performance metrics (annualized where appropriate using calculated frequency):")
                    try:
                        # For sub-daily, ensure 'rf' (risk-free rate) is 0 for most crypto contexts unless specified
                        rf_for_stats = 0.0 
                        print(f"  Sharpe Ratio (annualized): {qs.stats.sharpe(returns, periods=periods_per_year_for_stats, annualize=True, rf=rf_for_stats):.2f}")
                        print(f"  Sortino Ratio (annualized): {qs.stats.sortino(returns, periods=periods_per_year_for_stats, annualize=True, rf=rf_for_stats):.2f}")
                        print(f"  Max Drawdown: {qs.stats.max_drawdown(returns)*100:.2f}%")
                        # Win rate and related stats are not dependent on annualization periods
                        print(f"  Win Rate: {qs.stats.win_rate(returns)*100:.2f}%") 
                        if qs.stats.avg_win(returns) is not None:
                             print(f"  Average Win: {qs.stats.avg_win(returns)*100:.2f}%")
                        if qs.stats.avg_loss(returns) is not None:
                             print(f"  Average Loss: {qs.stats.avg_loss(returns)*100:.2f}%")
                        if qs.stats.profit_factor(returns) is not None:
                            print(f"  Profit Factor: {qs.stats.profit_factor(returns):.2f}")
                        # CAGR is generally not meaningful for sub-daily, so we skip it.
                        # Other non-annualized stats can be added here if desired.
                        print(f"  Total Returns: {qs.stats.comp(returns)*100:.2f}%")
                        # qs.stats.volatility `annualize=False` gives daily volatility if input returns are daily.
                        # If input returns are higher frequency, it gives volatility for that frequency.
                        # To get a "daily average" from higher frequency, one might resample or adjust.
                        # For now, let's get the volatility of the given returns series without explicit annualization.
                        print(f"  Volatility (period avg): {qs.stats.volatility(returns, annualize=False)*100:.2f}%")

                    except Exception as e:
                        print(f"  Error calculating individual QuantStats metrics: {e}")
                else:
                    # Attempt full report if data spans >= 1 day
                    try:
                        qs.reports.html(returns, output=report_filename, title=report_title, periods_per_year=periods_per_year_for_stats)
                        print(f"\nGenerated QuantStats HTML report: {report_filename}")
                    except Exception as e:
                        print(f"\nAn unexpected error occurred while generating QuantStats HTML report: {e}")
                        print(f"DEBUG: periods_per_year_for_stats used: {periods_per_year_for_stats}")
                        print(f"DEBUG: Returns data head for report:\n{returns.head()}")
                        print(f"DEBUG: Returns data tail for report:\n{returns.tail()}")
    else:
        print("\nSkipping QuantStats report: Equity data is empty or missing required columns.")
        
    def plot_results(trades, equity, backtester, args, MARKETS):
    # Plot equity curve
        plt.figure(figsize=(12, 6))
        equity.set_index('timestamp')['portfolio_value'].plot(title=f'Portfolio Value ({args.strategy.upper()} Strategy)')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value (USDT)')
        plt.grid(True)
        plt.savefig('equity_curve.png')
        print("Saved equity curve to equity_curve.png")
        
        # Plot trade outcomes
        if not trades.empty:
            # Extract exit trades (which have profit info)
            exit_trades = trades[trades['action'] == 'exit']
            
            plt.figure(figsize=(10, 6))
            sns.histplot(exit_trades['profit'], bins=30, kde=True)
            plt.title('Profit Distribution per Trade')
            plt.xlabel('Profit (USDT)')
            plt.savefig('profit_distribution.png')
            print("Saved profit distribution to profit_distribution.png")
            
            # Market performance breakdown
            plt.figure(figsize=(10, 6))
            market_perf = exit_trades.groupby('market')['profit'].sum()
            market_perf.plot(kind='bar')
            plt.title('Profit by Market')
            plt.ylabel('Total Profit (USDT)')
            plt.savefig('market_performance.png')
            print("Saved market performance to market_performance.png")
        
        # Strategy-specific visualizations
        for market in MARKETS:
            fig = backtester.strategy.plot_indicators(backtester.data, market)
            fig.savefig(f'{args.strategy}_indicators_{market}.png')
            print(f"Saved {args.strategy.upper()} indicators for {market} to {args.strategy}_indicators_{market}.png")
        
    plot_results(trades, equity, backtester, args, MARKETS)
    print("\nPipeline execution complete")

if __name__ == "__main__":
    main()
