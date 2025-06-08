import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
RISK_PER_TRADE = 0.1

STRATEGIES = {
    'macd': (MACDStrategy, {'fast': 12, 'slow': 26, 'signal': 9}),
    'rsi': (RSIStrategy, {'period': 20, 'overbought': 70, 'oversold': 25}),
    'mac': (MovingAverageCrossover, {'fast_window': 10, 'slow_window': 30})
}

def parse_args():
    parser = argparse.ArgumentParser(description='Alpha Mining Backtester')
    parser.add_argument('--strategy', choices=STRATEGIES.keys(), default='macd',
                        help='Trading strategy to use (default: macd)')
    parser.add_argument('--confidence', type=float, default=0.0,
                        help='Minimum confidence threshold for trades (0.0-1.0, default: 0.0)')
    parser.add_argument('--leverage', type=float, default=1.0,
                        help='Leverage to apply (e.g., 1.0, 2.0, 5.0, 10.0, default: 1.0 for no leverage)')
    parser.add_argument('--funding_rate_daily', type=float, default=0.0001,
                        help='Daily funding rate for borrowed capital (e.g., 0.0001 for 0.01%, default: 0.0001)')
    return parser.parse_args()

def main():
    args = parse_args()
    strategy_class, strategy_params = STRATEGIES[args.strategy]
    
    print(f"Starting alpha mining pipeline with {args.strategy.upper()} strategy (Confidence: {args.confidence}, Leverage: {args.leverage}x, Daily Funding Rate: {args.funding_rate_daily*100:.4f}%)")
    
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
        confidence_threshold=args.confidence,
        leverage=args.leverage,
        funding_rate_daily=args.funding_rate_daily
    )
    backtester.backtest(data)
    
    # Step 4: Get results
    trades, equity = backtester.get_results()
    metrics = backtester.calculate_metrics(trades, equity)

    # Display confidence histogram
    if backtester.data is not None and 'confidence' in backtester.data.columns:
        print("\n=== Confidence Score Distribution ===")
        confidence_scores = backtester.data['confidence'].dropna()
        if not confidence_scores.empty:
            hist, bin_edges = np.histogram(confidence_scores, bins=10, range=(0,100)) # Confidence is 0-100
            max_freq = hist.max()
            # Scale histogram bars for text display (e.g., max 50 chars wide)
            bar_scale = 50 / max_freq if max_freq > 0 else 1 
            
            print(f"{'Range':<12} | {'Frequency':<10} | Histogram")
            print("-" * 50)
            for i in range(len(hist)):
                lower_bound = bin_edges[i]
                upper_bound = bin_edges[i+1]
                frequency = hist[i]
                bar = '#' * int(frequency * bar_scale)
                print(f"{lower_bound:.0f}-{upper_bound:<.0f}    \t | {frequency:<10} | {bar}")
        else:
            print("No confidence scores available to display histogram.")

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
                report_title = f'{args.strategy.upper()} Strategy Backtest Report (Conf: {args.confidence}, Lev: {args.leverage}x)'
                report_filename = f'{args.strategy.lower()}_conf_{args.confidence}_lev_{args.leverage}x_report.html'

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
        if not equity.empty and 'timestamp' in equity.columns and 'portfolio_value' in equity.columns:
            # Ensure timestamp is datetime for Plotly
            equity_plot_df = equity.copy()
            equity_plot_df['timestamp'] = pd.to_datetime(equity_plot_df['timestamp'])
            
            fig_equity = px.line(equity_plot_df, x='timestamp', y='portfolio_value', 
                                 title=f'Portfolio Value ({args.strategy.upper()} Strategy)',
                                 labels={'timestamp': 'Time', 'portfolio_value': 'Portfolio Value (USDT)'})
            fig_equity.update_layout(xaxis_title='Time', yaxis_title='Portfolio Value (USDT)')
            fig_equity.write_html('equity_curve.html')
            print("Saved interactive equity curve to equity_curve.html")
        else:
            print("Skipping equity curve plot: Data is empty or missing required columns.")
        
        # Plot trade outcomes
        if not trades.empty:
            exit_trades = trades[trades['action'] == 'exit']
            if not exit_trades.empty:
                # Profit Distribution
                fig_profit_dist = px.histogram(exit_trades, x='profit', nbins=30, 
                                               title='Profit Distribution per Trade',
                                               labels={'profit': 'Profit (USDT)'})
                fig_profit_dist.update_layout(xaxis_title='Profit (USDT)')
                fig_profit_dist.write_html('profit_distribution.html')
                print("Saved interactive profit distribution to profit_distribution.html")
                
                # Market performance breakdown
                market_perf_data = exit_trades.groupby('market')['profit'].sum().reset_index()
                if not market_perf_data.empty:
                    fig_market_perf = px.bar(market_perf_data, x='market', y='profit',
                                             title='Profit by Market',
                                             labels={'market': 'Market', 'profit': 'Total Profit (USDT)'})
                    fig_market_perf.update_layout(xaxis_title='Market', yaxis_title='Total Profit (USDT)')
                    fig_market_perf.write_html('market_performance.html')
                    print("Saved interactive market performance to market_performance.html")
                else:
                    print("Skipping market performance plot: No data after grouping.")
            else:
                print("Skipping trade outcome plots: No exit trades found.")
        else:
            print("Skipping trade outcome plots: No trades found.")
            
        # Strategy-specific visualizations (now interactive HTML)
        print("Generating strategy-specific interactive indicator plots (HTML)...")
        for market in MARKETS:
            try:
                # plot_indicators now returns a plotly.graph_objects.Figure
                plotly_fig = backtester.strategy.plot_indicators(backtester.data, market)
                if plotly_fig is not None: 
                    plot_filename = f'{args.strategy}_indicators_{market}.html'
                    plotly_fig.write_html(plot_filename)
                    print(f"Saved interactive {args.strategy.upper()} indicators for {market} to {plot_filename}")
                else:
                    print(f"Indicator plot for {market} was not generated by strategy.")
            except Exception as e:
                print(f"Error generating/saving interactive indicator plot for {market}: {e}")
        
    plot_results(trades, equity, backtester, args, MARKETS)
    print("\nPipeline execution complete")

if __name__ == "__main__":
    main()
