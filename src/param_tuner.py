import polars as pl
import numpy as np
import itertools
import time
import argparse
from data_loader import load_and_preprocess_raw_data
from features import engineer_features
from backtest import GenericBacktester
from strategies import MACDStrategy, RSIStrategy, MovingAverageCrossover

MARKETS = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT']
DATA_PATH = 'crypto_tick'
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01

PARAM_GRIDS = {
    'MACD': {
        'strategy_class': MACDStrategy,
        'params': {
            'fast': [12, 15, 20], 'slow': [26, 30, 40], 'signal': [9, 12],
            'confidence_threshold': [0.0, 20.0, 40.0, 60.0]
        }
    },
    'RSI': {
        'strategy_class': RSIStrategy,
        'params': {
            'period': [10, 14, 20], 'overbought': [70, 75], 'oversold': [25, 30],
            'confidence_threshold': [20.0, 40.0, 60.0]
        }
    },
    'MovingAverageCrossover': {
        'strategy_class': MovingAverageCrossover,
        'params': {
            'fast_window': [10, 15, 20], 'slow_window': [30, 40, 50],
            'confidence_threshold': [0.0, 20.0, 40.0, 60.0]
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Parameter Tuner for Alpha Mining Strategies')
    parser.add_argument('--leverage', type=float, default=1.0)
    parser.add_argument('--funding_rate_daily', type=float, default=0.0001)
    return parser.parse_args()

def run_backtest_iteration(data, strategy_class, strategy_config, confidence_threshold_val, leverage_val, funding_rate_daily_val):
    backtester = GenericBacktester(
        strategy=strategy_class, strategy_params=strategy_config,
        initial_capital=INITIAL_CAPITAL, risk_per_trade=RISK_PER_TRADE,
        confidence_threshold=confidence_threshold_val, leverage=leverage_val,
        funding_rate_daily=funding_rate_daily_val
    )
    backtester.backtest(data.clone())
    trades, equity = backtester.get_results()
    
    if trades.is_empty() or equity.is_empty():
        return {'final_portfolio': INITIAL_CAPITAL, 'return_pct': 0.0, 'num_trades': 0,
                'win_rate': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        
    return backtester.calculate_metrics(trades, equity)

def main():
    args = parse_args()
    print(f"Starting parameter tuning with Leverage: {args.leverage}x, Daily Funding Rate: {args.funding_rate_daily*100:.4f}%")
    
    print(f"Loading data from '{DATA_PATH}'...")
    raw_data = load_and_preprocess_raw_data(DATA_PATH, MARKETS)
    if raw_data.is_empty():
        print("No data loaded. Exiting.")
        return
    print(f"Loaded {len(raw_data)} records. Calculating features...")
    processed_data = engineer_features(raw_data)
    print("Data loading and feature calculation complete.")

    all_results = []
    total_iterations = sum(np.prod([len(v) for v in config['params'].values()]) for config in PARAM_GRIDS.values())
    print(f"\nTotal iterations to perform: {total_iterations}")
    current_iteration = 0
    tuning_start_time = time.time()

    for strategy_name, config in PARAM_GRIDS.items():
        print(f"\nTuning Strategy: {strategy_name}")
        strategy_class = config['strategy_class']
        param_keys = [k for k in config['params'].keys() if k != 'confidence_threshold']
        confidence_thresholds = config['params'].get('confidence_threshold', [0.0])
        param_values = [config['params'][k] for k in param_keys]
        
        for conf_thresh in confidence_thresholds:
            param_combinations = list(itertools.product(*param_values)) if param_keys else [()]
            for param_set in param_combinations:
                current_iteration += 1
                iteration_start_time = time.time()
                current_params = dict(zip(param_keys, param_set))
                
                metrics = run_backtest_iteration(
                    processed_data, strategy_class, current_params, conf_thresh,
                    args.leverage, args.funding_rate_daily
                )
                
                result_entry = {'strategy': strategy_name, **current_params, 'confidence_threshold': conf_thresh,
                                'leverage_used': args.leverage, 'funding_rate_used': args.funding_rate_daily, **metrics}
                all_results.append(result_entry)
                
                duration = time.time() - iteration_start_time
                print(f"Iter {current_iteration}/{total_iterations} ({strategy_name} | {current_params} | Conf: {conf_thresh:.1f}) -> "
                      f"WinRate: {metrics.get('win_rate', 0.0):.2f}%, Trades: {metrics.get('num_trades', 0)}, "
                      f"Return: {metrics.get('return_pct', 0.0):.2f}%. (Took {duration:.2f}s)")

    print(f"\nTuning completed in {time.time() - tuning_start_time:.2f}s.")

    results_df = pl.DataFrame(all_results)
    
    sort_cols = ['win_rate', 'num_trades', 'return_pct']
    if all(c in results_df.columns for c in sort_cols):
        sorted_results = results_df.sort(by=sort_cols, descending=[True, False, True])
    else:
        print("Warning: Key metrics not found. Displaying unsorted.")
        sorted_results = results_df
        
    print("\n=== Top Parameter Tuning Results ===")
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        if not sorted_results.is_empty():
            print(sorted_results)
        else:
            print("No results to display.")

if __name__ == "__main__":
    main()
