import polars as pl
import plotly.express as px
import argparse
import os
import logging
from data_loader import load_and_preprocess_raw_data, clear_cache

from backtest import GenericBacktester
from strategies import MACDStrategy, RSIStrategy, MovingAverageCrossover, GridStrategy
from ml_strategy import MLTradingStrategy
from model_trainer import run_training_pipeline, MODELS_DIR
from risk_manager import RiskConfig

MARKETS_TO_LOAD = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT']
TARGET_MARKET = 'SOL_USDT'
DATA_PATH = 'crypto_tick'
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.005
LEVERAGE = 1.0

STRATEGY_CONFIG = {
    'macd': (MACDStrategy, {'fast': 12, 'slow': 26, 'signal': 9}),
    'rsi': (RSIStrategy, {'period': 14, 'overbought': 70, 'oversold': 30}),
    'mac': (MovingAverageCrossover, {'fast_window': 10, 'slow_window': 30}),
    'grid': (GridStrategy, {'grid_size': 0.5, 'sma_fast': 20, 'sma_slow': 50, 'trend_threshold': 0.001}),
    'ml': (MLTradingStrategy, {
        'model_path': os.path.join(MODELS_DIR, f"{TARGET_MARKET}_quantile_model.pth"),
        'scaler_path': os.path.join(MODELS_DIR, f"{TARGET_MARKET}_quantile_scaler.joblib"),
        'features_path': os.path.join(MODELS_DIR, f"{TARGET_MARKET}_quantile_features.joblib"),
        'confidence_threshold': 0.3, 'tp_pct': 0.008, 'sl_pct': 0.005,
        'horizon_seconds': 60 * 60  # 1 hour for neural network
    })
}
# Updated features list to match the refactored features.py output
BASE_FEATURES = [
    # L2 Order Book Features
    'bid_depth_total', 'ask_depth_total', 'spread_abs_calc', 'spread_rel_calc', 
    'wap', 'obi_total',
    
    # Technical Indicators
    'rsi_7', 'rsi_14', 'rsi_21', 'rsi_30',
    'rsi_overbought', 'rsi_oversold', 'rsi_momentum', 'rsi_ema_9',
    'rsi_overbought_flag', 'rsi_oversold_flag',
    'macd_line', 'macd_signal_line', 'macd_histogram',
    'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
    'stoch_k', 'williams_r', 'cci', 'atr',
    
    # Volume & Volatility Features
    'volume_sma_20', 'volume_std_20', 'volume_change', 'volume_ratio',
    'volatility_20p', 'volatility_60p', 'spread_rel_sma_20', 'spread_rel_std_20',
    'returns_1p', 'returns_5p', 'returns_20p', 'sharpe_20p',
    
    # Advanced Order Book Features
    'size_imbalance_l1', 'size_imbalance_l2', 'size_imbalance_l3',
    'value_imbalance_l1', 'value_imbalance_l2', 'value_imbalance_l3',
    'bid_slope', 'ask_slope', 'bid_vwap_3l', 'ask_vwap_3l',
    'bid_price_stability', 'ask_price_stability',
    
    # Microstructure Features
    'price_direction', 'uptick_flag', 'obi_change', 'obi_trend',
    'spread_change', 'spread_trend',
    
    # Cross-Market Features (if available)
    'sol_btc_ratio', 'sol_eth_ratio', 'sol_bnb_ratio',
    'btc_return_1p', 'eth_return_1p', 'bnb_return_1p',
    
    # Time Features
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'minute_sin', 'minute_cos',
    'asian_session', 'european_session', 'us_session'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Alpha Mining Backtester')
    parser.add_argument('--strategy', choices=STRATEGY_CONFIG.keys(), default='ml')
    parser.add_argument('--confidence', type=float, default=0.55)
    parser.add_argument('--leverage', type=float, default=LEVERAGE)
    parser.add_argument('--skip_training', action='store_true')
    
    # Risk management parameters
    parser.add_argument('--max_position_pct', type=float, default=0.15, 
                        help='Maximum position size as percentage of portfolio (default: 15%%)')
    parser.add_argument('--max_exposure_pct', type=float, default=0.80,
                        help='Maximum total exposure as percentage of portfolio (default: 80%%)')
    parser.add_argument('--min_cash_pct', type=float, default=0.20,
                        help='Minimum cash reserve as percentage of portfolio (default: 20%%)')
    parser.add_argument('--max_positions', type=int, default=5,
                        help='Maximum number of concurrent positions (default: 5)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    strategy_class, strategy_params = STRATEGY_CONFIG[args.strategy]

    if args.strategy == 'ml' and not args.skip_training:
        print("=== Starting Model Training Pipeline ===")
        run_training_pipeline(
            markets_to_load=MARKETS_TO_LOAD, target_market=TARGET_MARKET,
            features_list=BASE_FEATURES, tp_pct=strategy_params['tp_pct'],
            sl_pct=strategy_params['sl_pct'], training_horizon_seconds=strategy_params['horizon_seconds']
        )
        print("=== Model Training Pipeline Finished ===")
    elif args.strategy == 'ml':
        print("--- Skipping training, using existing model files. ---")

    print(f"\n=== Starting Backtest for {args.strategy.upper()} Strategy ===")
    clear_cache()
    raw_data = load_and_preprocess_raw_data(base_path=DATA_PATH, markets=MARKETS_TO_LOAD)
    if raw_data.is_empty():
        print("No data loaded for backtesting. Exiting.")
        return
    
    print(f"Loaded {len(raw_data)} records across {raw_data['market'].n_unique()} markets.")
    
    if args.strategy == 'ml':
        strategy_params['confidence_threshold'] = args.confidence

    # Configure risk management
    risk_config = RiskConfig(
        max_position_pct=args.max_position_pct,
        max_total_exposure_pct=args.max_exposure_pct,
        min_cash_reserve_pct=args.min_cash_pct,
        max_risk_per_trade_pct=RISK_PER_TRADE,
        max_concurrent_positions=args.max_positions,
        volatility_scaling_enabled=True,
        volatility_lookback_factor=2.0
    )
    
    print(f"Risk Management Configuration:")
    print(f"  Max Position Size: {args.max_position_pct:.1%}")
    print(f"  Max Total Exposure: {args.max_exposure_pct:.1%}")
    print(f"  Min Cash Reserve: {args.min_cash_pct:.1%}")
    print(f"  Risk Per Trade: {RISK_PER_TRADE:.1%}")
    print(f"  Max Concurrent Positions: {args.max_positions}")

    backtester = GenericBacktester(
        strategy=strategy_class, strategy_params=strategy_params,
        initial_capital=INITIAL_CAPITAL, risk_per_trade=RISK_PER_TRADE,
        confidence_threshold=args.confidence if args.strategy == 'ml' else 0.0,
        leverage=args.leverage, selected_markets=[TARGET_MARKET],
        verbose_logging=True, risk_config=risk_config
    )
    
    print("Running backtest...")
    backtester.backtest(raw_data)
    
    trades, equity = backtester.get_results()
    
    if trades.is_empty():
        print("\nNo trades were executed during the backtest.")
        return

    metrics = backtester.calculate_metrics(trades, equity)
    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title()}: {v:.4f}" if isinstance(v, float) else f"{k.replace('_', ' ').title()}: {v}")

    # Print comprehensive risk summary
    backtester.print_risk_summary()

    print("\n--- Generating Visualizations ---")
    plot_results(trades, equity, backtester, args, TARGET_MARKET)
    
    print("\n=== Pipeline Execution Complete ===")

def plot_results(trades, equity, backtester, args, market_to_plot):
    if not equity.is_empty():
        fig_equity = px.line(equity.to_pandas(), x='timestamp', y='portfolio_value', 
                             title=f'Portfolio Value ({args.strategy.upper()} Strategy, {args.leverage}x Leverage)',
                             labels={'timestamp': 'Time', 'portfolio_value': 'Portfolio Value (USDT)'})
        fig_equity.write_html('equity_curve.html')
        print("Saved interactive equity curve to equity_curve.html")

    exit_trades = trades.filter(pl.col('action') == 'exit')
    if not exit_trades.is_empty():
        fig_profit_dist = px.histogram(exit_trades.to_pandas(), x='profit', nbins=50, 
                                       title='Profit & Loss Distribution per Trade',
                                       labels={'profit': 'Profit (USDT)'})
        fig_profit_dist.write_html('profit_distribution.html')
        print("Saved interactive P&L distribution to profit_distribution.html")

    try:
        if isinstance(backtester.strategy, MLTradingStrategy):
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
