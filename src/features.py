import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame, N_levels: int = 5) -> pd.DataFrame:
    """
    Main function to engineer features for the ML model.
    This function includes L2 order book, cross-market, technical,
    and other feature types. Optimized for performance.
    """
    logger.info("Starting feature engineering...")
    
    df = df.sort_index()

    # --- L2 Order Book Features (Optimized) ---
    logger.info("Calculating L2 order book features...")

    if 'bid_levels_parsed' not in df.columns or 'ask_levels_parsed' not in df.columns:
        logger.error("Missing 'bid_levels_parsed' or 'ask_levels_parsed'. L2 features cannot be created.")
        return df

    # Vectorized extraction of L2 data into columns
    for i in range(N_levels):
        df[f'bid_price_l{i+1}'] = df['bid_levels_parsed'].str[i].str[0]
        df[f'bid_size_l{i+1}'] = df['bid_levels_parsed'].str[i].str[1]
        df[f'ask_price_l{i+1}'] = df['ask_levels_parsed'].str[i].str[0]
        df[f'ask_size_l{i+1}'] = df['ask_levels_parsed'].str[i].str[1]

    # Vectorized WAP calculation
    best_bid_price = df['bid_price_l1']
    best_ask_price = df['ask_price_l1']
    best_bid_size = df['bid_size_l1']
    best_ask_size = df['ask_size_l1']
    
    denominator = best_bid_size + best_ask_size
    df['wap'] = np.where(
        denominator > 0,
        (best_bid_price * best_ask_size + best_ask_price * best_bid_size) / denominator,
        np.nan
    )

    # Vectorized OBI and Depth calculation
    bid_size_cols = [f'bid_size_l{i+1}' for i in range(N_levels)]
    ask_size_cols = [f'ask_size_l{i+1}' for i in range(N_levels)]
    
    total_bid_volume = df[bid_size_cols].sum(axis=1)
    total_ask_volume = df[ask_size_cols].sum(axis=1)
    
    df['bid_depth_level5'] = total_bid_volume
    df['ask_depth_level5'] = total_ask_volume
    
    total_volume = total_bid_volume + total_ask_volume
    df['obi_level5'] = np.where(
        total_volume > 0,
        (total_bid_volume - total_ask_volume) / total_volume,
        np.nan
    )

    # --- Technical Indicators ---
    logger.info("Calculating technical indicators...")
    df['rsi_14'] = df.groupby('market')['mid_price'].transform(lambda x: rsi(x, period=14))
    
    # Calculate MACD per group and assign back to the original dataframe
    for market_name, group_df in df.groupby('market'):
        macd_results = macd(group_df['mid_price'])
        df.loc[group_df.index, 'macd_line'] = macd_results['macd_line']
        df.loc[group_df.index, 'macd_signal_line'] = macd_results['macd_signal_line']
        df.loc[group_df.index, 'macd_histogram'] = macd_results['macd_histogram']


    # --- Volatility & Volume Features ---
    logger.info("Calculating volatility and volume features...")
    if 'volume_1m' in df.columns:
        df['volume_1m_sma20'] = df.groupby('market')['volume_1m'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    
    if 'spread_abs' in df.columns:
        df['spread_abs_sma20'] = df.groupby('market')['spread_abs'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())

    df['mid_price_numeric'] = pd.to_numeric(df['mid_price'], errors='coerce')
    df['mid_price_return'] = df.groupby('market')['mid_price_numeric'].pct_change()
    df['volatility_20p'] = df.groupby('market')['mid_price_return'].transform(lambda x: x.rolling(window=20, min_periods=1).std())

    # --- Cross-Market Features ---
    logger.info("Pivoting data for cross-market features...")
    df_pivot_price = df.pivot_table(index=df.index, columns='market', values='mid_price_numeric')
    df_pivot_price = df_pivot_price.ffill()

    other_markets = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    for other_market in other_markets:
        if 'SOL_USDT' in df_pivot_price.columns and other_market in df_pivot_price.columns:
            df[f'sol_vs_{other_market.split("_")[0].lower()}_price_ratio'] = (df_pivot_price['SOL_USDT'] / df_pivot_price[other_market]).reindex(df.index, method='ffill')
            df_pivot_returns = df_pivot_price.pct_change()
            df[f'{other_market.split("_")[0].lower()}_return_lag1'] = df_pivot_returns[other_market].shift(1).reindex(df.index, method='ffill')
    
    logger.info("Finished cross-market feature calculations.")
    df.drop(columns=['mid_price_numeric'], inplace=True, errors='ignore')

    # --- Time-based Features ---
    logger.info("Calculating time-based features...")
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

    # --- Market Regime Features ---
    logger.info("Calculating market regime features...")
    if 'market_phase' in df.columns:
        df = pd.get_dummies(df, columns=['market_phase'], prefix='phase', dummy_na=False)
    if 'volatility_regime' in df.columns:
        df = pd.get_dummies(df, columns=['volatility_regime'], prefix='volregime', dummy_na=False)

    # --- Cleanup ---
    # Drop intermediate parsed columns
    df.drop(columns=['bid_levels_parsed', 'ask_levels_parsed'], inplace=True, errors='ignore')

    logger.info("Finished feature engineering.")
    return df

# Helper functions for indicators
def rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return pd.DataFrame({
        'macd_line': macd_line,
        'macd_signal_line': signal_line,
        'macd_histogram': macd_hist
    })

if __name__ == "__main__":
    print("Feature engineering module ready.")
