import pandas as pd
import numpy as np
import plotly.express as px
import argparse
import os
import logging

# --- Project Modules ---
from data_loader import load_and_preprocess_raw_data
from backtest import GenericBacktester
from strategies import MACDStrategy, RSIStrategy, MovingAverageCrossover
from ml_strategy import MLTradingStrategy
from model_trainer import run_training_pipeline, MODELS_DIR

# --- Configuration ---
MARKETS_TO_LOAD = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT']
TARGET_MARKET = 'SOL_USDT' # The market we are building the model for and primarily trading
DATA_PATH = 'crypto_tick'
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.02 # Risk 2% of initial capital per trade
LEVERAGE = 2.0

# The feature list is now generated and saved by the trainer, and loaded by the strategy.
# This avoids inconsistencies.

STRATEGY_CONFIG = {
    'macd': (MACDStrategy, {'fast': 12, 'slow': 26, 'signal': 9}),
    'rsi': (RSIStrategy, {'period': 14, 'overbought': 70, 'oversold': 30}),
    'mac': (MovingAverageCrossover, {'fast_window': 10, 'slow_window': 30}),
    'ml': (MLTradingStrategy, {
        'model_path': os.path.join(MODELS_DIR, f"{TARGET_MARKET}_lgbm_model.joblib"),
        'scaler_path': os.path.join(MODELS_DIR, f"{TARGET_MARKET}_scaler.joblib"),
        'features_path': os.path.join(MODELS_DIR, f"{TARGET_MARKET}_features.joblib"),
        'prediction_threshold': 0.55, # Only take trades with >55% confidence
        'tp_pct': 0.005, # 0.5% take profit
        'sl_pct': 0.0025, # 0.25% stop loss
        'horizon_seconds': 60 * 30 # 30 minute horizon
    })
}
# The base feature list to be used by the trainer.
# The trainer will dynamically add one-hot encoded columns.
BASE_FEATURES = [
    'wap', 'obi_level5', 'bid_depth_level5', 'ask_depth_level5',
    'rsi_14', 'macd_line', 'macd_histogram',
    'volume_1m_sma20', 'spread_abs_sma20', 'volatility_20p',
    'sol_vs_btc_price_ratio', 'sol_vs_eth_price_ratio', 'sol_vs_bnb_price_ratio',
    'btc_return_lag1', 'eth_return_lag1', 'bnb_return_lag1',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Alpha Mining Backtester')
    parser.add_argument('--strategy', choices=STRATEGY_CONFIG.keys(), default='ml',
                        help='Trading strategy to use')
    parser.add_argument('--confidence', type=float, default=0.55,
                        help='Minimum confidence threshold for ML strategy trades')
    parser.add_argument('--leverage', type=float, default=LEVERAGE,
                        help='Leverage to apply')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip the model training step and use existing model files')
    return parser.parse_args()

def main():
    args = parse_args()
    strategy_class, strategy_params = STRATEGY_CONFIG[args.strategy]

    # --- Step 1: Model Training (for ML strategy) ---
    if args.strategy == 'ml' and not args.skip_training:
        print("=== Starting Model Training Pipeline ===")
        run_training_pipeline(
            markets_to_load=MARKETS_TO_LOAD,
            target_market=TARGET_MARKET,
            features_list=BASE_FEATURES,
            # Pass triple barrier params to training to ensure consistency
            tp_pct=strategy_params['tp_pct'],
            sl_pct=strategy_params['sl_pct'],
            training_horizon_seconds=strategy_params['horizon_seconds']
        )
        print("=== Model Training Pipeline Finished ===")
    elif args.strategy == 'ml':
        print("--- Skipping training, using existing model files. ---")

    # --- Step 2: Load Data for Backtest ---
    print(f"\n=== Starting Backtest for {args.strategy.upper()} Strategy ===")
    print("Loading data for backtesting period...")
    # For a real scenario, you might load a different date range for backtesting
    # than for training. Here we load all data for simplicity.
    raw_data = load_and_preprocess_raw_data(base_path=DATA_PATH, markets=MARKETS_TO_LOAD)
    if raw_data.empty:
        print("No data loaded for backtesting. Exiting.")
        return
    
    print(f"Loaded {len(raw_data)} records across {len(raw_data['market'].unique())} markets.")
    
    # --- Step 3: Backtesting ---
    # Update strategy params from args if needed
    if args.strategy == 'ml':
        strategy_params['prediction_threshold'] = args.confidence

    print("Initializing backtester...")
    backtester = GenericBacktester(
        strategy=strategy_class,
        strategy_params=strategy_params,
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        confidence_threshold=args.confidence if args.strategy == 'ml' else 0.0,
        leverage=args.leverage,
        selected_markets=[TARGET_MARKET] # Backtest only on the target market
    )
    
    print("Running backtest...")
    # The backtester will internally handle feature engineering for the ML strategy
    backtester.backtest(raw_data)
    
    # --- Step 4: Results and Reporting ---
    trades, equity = backtester.get_results()
    
    if trades.empty:
        print("\nNo trades were executed during the backtest.")
        return

    metrics = backtester.calculate_metrics(trades, equity)

    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title()}: {v:.4f}" if isinstance(v, float) else f"{k.replace('_', ' ').title()}: {v}")

    # --- Step 5: Visualization ---
    print("\n--- Generating Visualizations ---")
    plot_results(trades, equity, backtester, args, TARGET_MARKET)
    
    print("\n=== Pipeline Execution Complete ===")

def plot_results(trades, equity, backtester, args, market_to_plot):
    # Plot equity curve
    if not equity.empty:
        fig_equity = px.line(equity, x='timestamp', y='portfolio_value', 
                             title=f'Portfolio Value ({args.strategy.upper()} Strategy, {args.leverage}x Leverage)',
                             labels={'timestamp': 'Time', 'portfolio_value': 'Portfolio Value (USDT)'})
        fig_equity.write_html('equity_curve.html')
        print("Saved interactive equity curve to equity_curve.html")

    # Plot trade outcomes
    exit_trades = trades[trades['action'] == 'exit']
    if not exit_trades.empty:
        fig_profit_dist = px.histogram(exit_trades, x='profit', nbins=50, 
                                       title='Profit & Loss Distribution per Trade',
                                       labels={'profit': 'Profit (USDT)'})
        fig_profit_dist.write_html('profit_distribution.html')
        print("Saved interactive P&L distribution to profit_distribution.html")

    # Strategy-specific plot
    try:
        if isinstance(backtester.strategy, MLTradingStrategy):
            # The ML strategy plot method now takes the trade log
            fig_indicators = backtester.strategy.plot_indicators(backtester.data, trades, market_to_plot)
        else:
            fig_indicators = backtester.strategy.plot_indicators(backtester.data, market_to_plot)
        
        if fig_indicators:
            plot_filename = f'{args.strategy}_indicators_{market_to_plot}.html'
            fig_indicators.write_html(plot_filename)
            print(f"Saved interactive strategy plot for {market_to_plot} to {plot_filename}")
    except Exception as e:
        print(f"Could not generate strategy-specific plot: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
