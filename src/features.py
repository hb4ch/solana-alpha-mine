import polars as pl
import numpy as np
import logging

logger = logging.getLogger(__name__)

def engineer_features(df: pl.DataFrame, N_levels: int = 5) -> pl.DataFrame:
    """
    Main function to engineer features for the ML model using Polars.
    """
    logger.info("Starting feature engineering...")
    
    df = df.sort("ts_utc")

    # --- L2 Order Book Features ---
    logger.info("Calculating L2 order book features...")
    if 'bid_levels_parsed' not in df.columns or 'ask_levels_parsed' not in df.columns:
        logger.error("Missing 'bid_levels_parsed' or 'ask_levels_parsed'. L2 features cannot be created.")
        return df

    for i in range(N_levels):
        df = df.with_columns([
            pl.col('bid_levels_parsed').list.get(i).list.get(0).alias(f'bid_price_l{i+1}'),
            pl.col('bid_levels_parsed').list.get(i).list.get(1).alias(f'bid_size_l{i+1}'),
            pl.col('ask_levels_parsed').list.get(i).list.get(0).alias(f'ask_price_l{i+1}'),
            pl.col('ask_levels_parsed').list.get(i).list.get(1).alias(f'ask_size_l{i+1}')
        ])

    best_bid_price = pl.col('bid_price_l1')
    best_ask_price = pl.col('ask_price_l1')
    best_bid_size = pl.col('bid_size_l1')
    best_ask_size = pl.col('ask_size_l1')
    
    denominator = best_bid_size + best_ask_size
    df = df.with_columns(
        pl.when(denominator > 0)
          .then((best_bid_price * best_ask_size + best_ask_price * best_bid_size) / denominator)
          .otherwise(None)
          .alias('wap')
    )

    bid_size_cols = [f'bid_size_l{i+1}' for i in range(N_levels)]
    ask_size_cols = [f'ask_size_l{i+1}' for i in range(N_levels)]
    
    df = df.with_columns([
        pl.sum_horizontal(pl.col(bid_size_cols)).alias('bid_depth_level5'),
        pl.sum_horizontal(pl.col(ask_size_cols)).alias('ask_depth_level5')
    ])
    
    total_volume = pl.col('bid_depth_level5') + pl.col('ask_depth_level5')
    df = df.with_columns(
        pl.when(total_volume > 0)
          .then((pl.col('bid_depth_level5') - pl.col('ask_depth_level5')) / total_volume)
          .otherwise(None)
          .alias('obi_level5')
    )

    # --- Technical Indicators ---
    logger.info("Calculating technical indicators...")
    df = df.with_columns([
        rsi(pl.col('mid_price'), period=7).over('market').alias('rsi_7'),
        rsi(pl.col('mid_price'), period=14).over('market').alias('rsi_14'),
        rsi(pl.col('mid_price'), period=21).over('market').alias('rsi_21')
    ])
    
    df = df.with_columns([
        pl.col('rsi_14').rolling_max(20).over('market').alias('rsi_overbought'),
        pl.col('rsi_14').rolling_min(20).over('market').alias('rsi_oversold'),
        pl.col('rsi_14').diff(1).over('market').alias('rsi_momentum'),
        pl.col('rsi_14').ewm_mean(span=9).over('market').alias('rsi_ema_9')
    ])

    macd_results = macd(pl.col('mid_price')).over('market')
    df = df.with_columns([
        macd_results.struct.field('macd_line').alias('macd_line'),
        macd_results.struct.field('macd_signal_line').alias('macd_signal_line'),
        macd_results.struct.field('macd_histogram').alias('macd_histogram')
    ])

    # --- Volatility & Volume Features ---
    logger.info("Calculating volatility and volume features...")
    if 'volume_1m' in df.columns:
        df = df.with_columns(pl.col('volume_1m').rolling_mean(window_size=20).over('market').alias('volume_1m_sma20'))
    
    if 'spread_abs' in df.columns:
        df = df.with_columns(pl.col('spread_abs').rolling_mean(window_size=20).over('market').alias('spread_abs_sma20'))

    df = df.with_columns(
        pl.col('mid_price').cast(pl.Float64).pct_change().rolling_std(window_size=20).over('market').alias('volatility_20p')
    )

    # --- Cross-Market Features ---
    logger.info("Pivoting data for cross-market features...")
    df_pivot_price = df.pivot(index='ts_utc', columns='market', values='mid_price')
    df_pivot_price = df_pivot_price.fill_null(strategy='forward')

    other_markets = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    for other_market in other_markets:
        if 'SOL_USDT' in df_pivot_price.columns and other_market in df_pivot_price.columns:
            price_ratio = df_pivot_price.select([
                pl.col('ts_utc'),
                (pl.col('SOL_USDT') / pl.col(other_market)).alias(f'sol_vs_{other_market.split("_")[0].lower()}_price_ratio')
            ])
            df = df.join(price_ratio, on='ts_utc', how='left')
            
            returns = df_pivot_price.select([
                pl.col('ts_utc'),
                pl.col(other_market).pct_change().alias(f'{other_market.split("_")[0].lower()}_return_lag1')
            ])
            df = df.join(returns, on='ts_utc', how='left')

    # --- Time-based Features ---
    logger.info("Calculating time-based features...")
    df = df.with_columns([
        pl.col('ts_utc').dt.hour().alias('hour_of_day'),
        pl.col('ts_utc').dt.weekday().alias('day_of_week')
    ])
    df = df.with_columns([
        (2 * np.pi * pl.col('hour_of_day') / 24.0).sin().alias('hour_sin'),
        (2 * np.pi * pl.col('hour_of_day') / 24.0).cos().alias('hour_cos'),
        (2 * np.pi * pl.col('day_of_week') / 7.0).sin().alias('day_sin'),
        (2 * np.pi * pl.col('day_of_week') / 7.0).cos().alias('day_cos')
    ])

    # --- Market Regime Features ---
    logger.info("Calculating market regime features...")
    if 'market_phase' in df.columns:
        df = df.rename({'market_phase': 'phase'})
        df = df.to_dummies(columns=['phase'], separator='_')
    if 'volatility_regime' in df.columns:
        df = df.rename({'volatility_regime': 'volregime'})
        df = df.to_dummies(columns=['volregime'], separator='_')

    # --- Cleanup ---
    df = df.drop(['bid_levels_parsed', 'ask_levels_parsed'])

    logger.info("Finished feature engineering.")
    return df

# Helper functions for indicators
def rsi(series: pl.Expr, period=14) -> pl.Expr:
    delta = series.diff(1)
    gain = delta.clip(lower_bound=0).rolling_mean(window_size=period)
    loss = (-delta).clip(lower_bound=0).rolling_mean(window_size=period)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series: pl.Expr, fast=12, slow=26, signal=9) -> pl.Expr:
    ema_fast = series.ewm_mean(span=fast)
    ema_slow = series.ewm_mean(span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm_mean(span=signal)
    macd_hist = macd_line - signal_line
    return pl.struct([
        macd_line.alias('macd_line'),
        signal_line.alias('macd_signal_line'),
        macd_hist.alias('macd_histogram')
    ])

if __name__ == "__main__":
    print("Feature engineering module ready.")
