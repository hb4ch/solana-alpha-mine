import pandas as pd
import numpy as np

def calculate_trend_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate trend-following features including moving averages and momentum indicators
    
    Args:
        df: DataFrame with market data
        window: Number of periods for moving averages
    
    Returns:
        DataFrame with added trend features
    """
    # Sort by timestamp
    df = df.sort_index()
    
    # Calculate moving averages
    df['sma'] = df.groupby('market')['mid_price'].transform(lambda x: x.rolling(window).mean())
    df['ema'] = df.groupby('market')['mid_price'].transform(lambda x: x.ewm(span=window//2).mean())
    
    # Momentum indicators
    df['momentum'] = df.groupby('market')['mid_price'].transform(lambda x: x.pct_change(periods=window))
    df['roc'] = df.groupby('market')['mid_price'].transform(lambda x: (x / x.shift(window)) - 1)
    
    # Trend direction
    df['trend_direction'] = np.where(
        (df['mid_price'] > df['sma']) & (df['ema'] > df['sma']), 1, 0)
    
    return df

def calculate_volume_features(df: pd.DataFrame, volume_col: str = 'volume_1m') -> pd.DataFrame:
    """
    Calculate volume-based features for trend confirmation
    
    Args:
        df: DataFrame with market data
        volume_col: Name of the volume column to use
    
    Returns:
        DataFrame with added volume features
    """
    # Sort by timestamp
    df = df.sort_index()
    
    # Volume metrics
    df['vol_ratio'] = df.groupby('market')[volume_col].transform(
        lambda x: x / x.rolling(60).mean())
    df['vol_accel'] = df.groupby('market')[volume_col].transform(
        lambda x: x.pct_change())
    
    # Volume-weighted price
    df['vwap'] = df.groupby('market').apply(
        lambda g: (g['mid_price'] * g[volume_col]).cumsum() / g[volume_col].cumsum()
    ).reset_index(level=0, drop=True)
    
    # Volume confirmation signal
    df['volume_ok'] = (df['vol_ratio'] > 1.2) & (df['vol_accel'] > 0)
    
    return df

def calculate_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all features for trend-following system
    
    Args:
        df: DataFrame with market data
    
    Returns:
        DataFrame with all features added
    """
    df = calculate_trend_features(df)
    df = calculate_volume_features(df)
    
    # Combined trend-volume signal
    df['entry_signal'] = df['trend_direction'] & df['volume_ok']
    
    # Volatility measure (using spread)
    df['volatility'] = df.groupby('market')['spread_abs'].transform(
        lambda x: x.rolling(60).std())
    
    return df

if __name__ == "__main__":
    # Test feature calculation
    print("Feature calculation module ready")
