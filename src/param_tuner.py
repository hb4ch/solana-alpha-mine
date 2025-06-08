import pandas as pd
import numpy as np
import itertools
import time
import argparse # Added for command-line arguments

from data_loader import load_all_market_data
from features import calculate_combined_features
from backtest import GenericBacktester
from strategies import MACDStrategy, RSIStrategy, MovingAverageCrossover

# Configuration (mirroring main.py where applicable)
MARKETS = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT']
DATA_PATH = 'crypto_tick'  # Relative to the project root
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01

# Define parameter grids for tuning
# Note: confidence_threshold is a GenericBacktester param, but tuned alongside strategy params
PARAM_GRIDS = {
    'MACD': {
        'strategy_class': MACDStrategy,
        'params': {
            'fast': [12, 15, 20],
            'slow': [26, 30, 40],
            'signal': [9, 12],
            'confidence_threshold': [0.0, 20.0, 40.0, 60.0] # Scale 0-100, added 0.0
        }
    },
    'RSI': {
        'strategy_class': RSIStrategy,
        'params': {
            'period': [10, 14, 20],
            'overbought': [70, 75],
            'oversold': [25, 30],
            'confidence_threshold': [20.0, 40.0, 60.0] # Scale 0-100
        }
    },
    'MovingAverageCrossover': {
        'strategy_class': MovingAverageCrossover,
        'params': {
            'fast_window': [10, 15, 20],
            'slow_window': [30, 40, 50],
            'confidence_threshold': [0.0, 20.0, 40.0, 60.0] # Scale 0-100, added 0.0
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Parameter Tuner for Alpha Mining Strategies')
    parser.add_argument('--leverage', type=float, default=1.0,
                        help='Fixed leverage to use for all tuning iterations (default: 1.0)')
    parser.add_argument('--funding_rate_daily', type=float, default=0.0001,
                        help='Fixed daily funding rate for all tuning iterations (default: 0.0001)')
    # Potentially add arguments to select which strategies to tune, etc.
    return parser.parse_args()

def run_backtest_iteration(data, strategy_class, strategy_config, confidence_threshold_val, leverage_val, funding_rate_daily_val):
    """
    Runs a single backtest iteration for a given strategy configuration.
    """
    backtester = GenericBacktester(
        strategy=strategy_class,
        strategy_params=strategy_config,
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        confidence_threshold=confidence_threshold_val,
        leverage=leverage_val,
        funding_rate_daily=funding_rate_daily_val
    )
    backtester.backtest(data.copy()) # Pass a copy of data to avoid modification issues
    trades, equity = backtester.get_results()
    
    if trades.empty or equity.empty:
        # Handle cases with no trades or no equity data
        return {
            'final_portfolio': INITIAL_CAPITAL,
            'return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
    metrics = backtester.calculate_metrics(trades, equity)
    return metrics

def main():
    args = parse_args()
    print("Starting parameter tuning process...")
    print(f"Using fixed Leverage: {args.leverage}x, Daily Funding Rate: {args.funding_rate_daily*100:.4f}% for all iterations.")
    
    # Step 1: Load and preprocess data (once)
    print(f"Loading all available market data from '{DATA_PATH}'...")
    raw_data = load_all_market_data(DATA_PATH, MARKETS)
    if raw_data.empty:
        print("No data loaded. Exiting parameter tuning.")
        return
    print(f"Loaded {len(raw_data)} records. Calculating features...")
    processed_data = calculate_combined_features(raw_data)
    print("Data loading and feature calculation complete.")

    all_results = []
    total_iterations = 0
    for strategy_name, config in PARAM_GRIDS.items():
        param_names = list(config['params'].keys())
        param_values = list(config['params'].values())
        total_iterations += np.prod([len(v) for v in param_values])

    print(f"\nTotal iterations to perform: {total_iterations}")
    current_iteration = 0
    tuning_start_time = time.time()

    for strategy_name, config in PARAM_GRIDS.items():
        print(f"\nTuning Strategy: {strategy_name}")
        strategy_class = config['strategy_class']
        
        # Separate strategy-specific params from the backtester's confidence_threshold
        strategy_param_keys = [k for k in config['params'].keys() if k != 'confidence_threshold']
        confidence_threshold_values = config['params'].get('confidence_threshold', [0.0]) # Default if not in grid

        # Create combinations for strategy-specific parameters
        strategy_param_values = [config['params'][k] for k in strategy_param_keys]
        
        # Iterate through confidence thresholds
        for conf_thresh in confidence_threshold_values:
            # Iterate through strategy parameter combinations
            # Handle case where there are no strategy-specific params (e.g. if only tuning confidence)
            if not strategy_param_keys: 
                param_combinations = [()] # A single empty tuple combination
            else:
                param_combinations = list(itertools.product(*strategy_param_values))

            for param_set in param_combinations:
                current_iteration += 1
                iteration_start_time = time.time()

                current_params_dict = dict(zip(strategy_param_keys, param_set))
                
                # Run backtest
                metrics = run_backtest_iteration(
                    processed_data, 
                    strategy_class, 
                    current_params_dict, 
                    conf_thresh,
                    args.leverage, # Pass fixed leverage
                    args.funding_rate_daily # Pass fixed funding rate
                )
                
                # Store results
                result_entry = {
                    'strategy': strategy_name,
                    **current_params_dict,
                    'confidence_threshold': conf_thresh, # Explicitly add confidence_threshold
                    'leverage_used': args.leverage, # Log the leverage used for this tuning session
                    'funding_rate_used': args.funding_rate_daily, # Log the funding rate
                    **metrics
                }
                all_results.append(result_entry)
                
                iteration_duration = time.time() - iteration_start_time
                print(f"Iter {current_iteration}/{total_iterations} ({strategy_name} | Params: {current_params_dict} | Conf: {conf_thresh:.1f}) -> WinRate: {metrics.get('win_rate', 0.0):.2f}%, Trades: {metrics.get('num_trades', 0)}, Return: {metrics.get('return_pct', 0.0):.2f}%. (Took {iteration_duration:.2f}s)")

    tuning_duration = time.time() - tuning_start_time
    print(f"\nParameter tuning completed in {tuning_duration:.2f} seconds.")

    # Display results
    results_df = pd.DataFrame(all_results)
    
    # Sort results: High win rate, then fewer trades (for tie-breaking or preference), then higher return
    # Ensure columns exist before sorting
    sort_columns = []
    if 'win_rate' in results_df.columns:
        sort_columns.append('win_rate')
    if 'num_trades' in results_df.columns:
        sort_columns.append('num_trades')
    if 'return_pct' in results_df.columns:
        sort_columns.append('return_pct')
        
    ascending_order = [False, True, False] # WinRate (D), NumTrades (A), Return (D)
    
    # Filter to only existing columns for sorting
    valid_sort_columns = [col for col in sort_columns if col in results_df.columns]
    valid_ascending_order = [asc for col, asc in zip(sort_columns, ascending_order) if col in results_df.columns]

    if valid_sort_columns:
        sorted_results = results_df.sort_values(
            by=valid_sort_columns,
            ascending=valid_ascending_order
        ).reset_index(drop=True)
    else:
        print("Warning: Key metrics for sorting (win_rate, num_trades, return_pct) not found in results. Displaying unsorted.")
        sorted_results = results_df
        
    print("\n=== Top Parameter Tuning Results ===")
    # Define columns to display, ensure they exist
    display_cols_order = [
        'strategy', 'win_rate', 'num_trades', 'return_pct', 'sharpe_ratio', 'max_drawdown', 'leverage_used'
    ]
    # Add parameter columns dynamically
    param_cols_present = [p for p in PARAM_GRIDS['MACD']['params'].keys()] # Example, get all possible param names
    if 'RSI' in PARAM_GRIDS: param_cols_present.extend(p for p in PARAM_GRIDS['RSI']['params'].keys() if p not in param_cols_present)
    if 'MovingAverageCrossover' in PARAM_GRIDS: param_cols_present.extend(p for p in PARAM_GRIDS['MovingAverageCrossover']['params'].keys() if p not in param_cols_present)
    # Add leverage and funding rate if they were logged, though they are fixed for the session
    # param_cols_present.append('leverage_used') 
    # param_cols_present.append('funding_rate_used')


    final_display_cols = display_cols_order[:1] + \
                         sorted(list(set(p for p in param_cols_present if p in sorted_results.columns))) + \
                         [c for c in display_cols_order[1:] if c in sorted_results.columns]
    
    # Remove duplicates from final_display_cols while preserving order
    seen = set()
    final_display_cols_unique = [x for x in final_display_cols if not (x in seen or seen.add(x))]


    with pd.option_context('display.max_rows', 20, 'display.max_columns', None, 'display.width', 200):
        if not sorted_results.empty:
            print(sorted_results[final_display_cols_unique].head(20))
        else:
            print("No results to display.")

if __name__ == "__main__":
    main()
