import polars as pl
import numpy as np
import logging
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

def engineer_features(df: pl.DataFrame, N_levels: int = 5, parallel: bool = True) -> pl.DataFrame:
    """
    Main function to engineer features for the ML model using optimized Polars operations.
    
    Args:
        df: Input DataFrame with market data
        N_levels: Number of order book levels to process
        parallel: Whether to use parallel processing for feature groups
    
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting optimized feature engineering...")
    start_time = time.time()
    
    # Ensure we have the 'market' column - handle 'pair' -> 'market' conversion
    if 'pair' in df.columns and 'market' not in df.columns:
        df = df.with_columns(pl.col('pair').alias('market'))
        logger.info("Renamed 'pair' column to 'market'")
    elif 'market' not in df.columns:
        logger.error("Neither 'market' nor 'pair' column found in data")
        return df
    
    # Check required columns
    if 'bid_levels_parsed' not in df.columns or 'ask_levels_parsed' not in df.columns:
        logger.error("Missing 'bid_levels_parsed' or 'ask_levels_parsed'. L2 features cannot be created.")
        return df
    
    # Use eager evaluation to avoid lazy frame column resolution issues
    logger.info("Using eager feature engineering to avoid lazy evaluation issues...")
    df = df.sort("ts_utc")
    
    # Sequential feature engineering with eager evaluation
    result_df = _engineer_features_sequential_eager(df, N_levels)
    
    # Post-processing: feature selection and cleanup
    result_df = _optimize_features(result_df)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Feature engineering completed in {elapsed_time:.2f} seconds")
    logger.info(f"Generated {len(result_df.columns)} features")
    
    return result_df

def _engineer_features_sequential_eager(df: pl.DataFrame, N_levels: int) -> pl.DataFrame:
    """Sequential feature engineering using eager evaluation to avoid lazy frame issues."""
    logger.info("Processing features with eager evaluation...")
    
    # L2 Order Book Features (eager)
    df = _add_l2_features_eager(df, N_levels)
    
    # Technical Indicators (eager)
    df = _add_technical_indicators_eager(df)
    
    # Volume and Volatility Features (eager)
    df = _add_volume_volatility_features_eager(df)
    
    # Advanced Order Book Features (eager)
    df = _add_advanced_orderbook_features_eager(df, N_levels)
    
    # Market Microstructure Features (eager)
    df = _add_microstructure_features_eager(df)
    
    # Cross-Market Features (eager)
    df = _add_cross_market_features_eager(df)
    
    # Time-based Features (eager)
    df = _add_time_features_eager(df)
    
    # Market Regime Features (eager)
    df = _add_regime_features_eager(df)
    
    return df

def _engineer_features_sequential(lazy_df: pl.LazyFrame, N_levels: int) -> pl.LazyFrame:
    """Sequential feature engineering with optimized batching."""
    
    # L2 Order Book Features (vectorized)
    lazy_df = _add_l2_features_optimized(lazy_df, N_levels)
    
    # Technical Indicators (batched)
    lazy_df = _add_technical_indicators_batched(lazy_df)
    
    # Volume and Volatility Features
    lazy_df = _add_volume_volatility_features(lazy_df)
    
    # Advanced Order Book Features
    lazy_df = _add_advanced_orderbook_features(lazy_df, N_levels)
    
    # Market Microstructure Features
    lazy_df = _add_microstructure_features(lazy_df)
    
    # Cross-Market Features (optimized)
    lazy_df = _add_cross_market_features_optimized(lazy_df)
    
    # Time-based Features
    lazy_df = _add_time_features(lazy_df)
    
    # Market Regime Features
    lazy_df = _add_regime_features(lazy_df)
    
    return lazy_df

def _engineer_features_parallel(lazy_df: pl.LazyFrame, N_levels: int) -> pl.LazyFrame:
    """Parallel feature engineering for large datasets."""
    
    # Convert to eager for parallel processing
    df = lazy_df.collect()
    
    feature_groups = [
        ("l2_features", lambda x: _add_l2_features_optimized(x.lazy(), N_levels).collect()),
        ("technical_indicators", lambda x: _add_technical_indicators_batched(x.lazy()).collect()),
        ("volume_volatility", lambda x: _add_volume_volatility_features(x.lazy()).collect()),
        ("advanced_orderbook", lambda x: _add_advanced_orderbook_features(x.lazy(), N_levels).collect()),
        ("microstructure", lambda x: _add_microstructure_features(x.lazy()).collect()),
    ]
    
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(feature_groups), 4)) as executor:
        future_to_group = {
            executor.submit(func, df): group_name 
            for group_name, func in feature_groups
        }
        
        for future in future_to_group:
            group_name = future_to_group[future]
            try:
                results[group_name] = future.result()
            except Exception as e:
                logger.warning(f"Error in {group_name}: {e}")
                results[group_name] = df
    
    # Merge results
    base_df = df.lazy()
    for group_name, result_df in results.items():
        if result_df.shape[1] > df.shape[1]:  # Only merge if new features were added
            new_cols = [col for col in result_df.columns if col not in df.columns]
            if new_cols:
                base_df = base_df.join(
                    result_df.lazy().select(['ts_utc', 'market'] + new_cols),
                    on=['ts_utc', 'market'],
                    how='left'
                )
    
    # Add remaining features that require the full dataset
    base_df = _add_cross_market_features_optimized(base_df)
    base_df = _add_time_features(base_df)
    base_df = _add_regime_features(base_df)
    
    return base_df

def _add_l2_features_eager(df: pl.DataFrame, N_levels: int) -> pl.DataFrame:
    """Eager L2 order book feature extraction to avoid lazy evaluation issues."""
    logger.info("Processing L2 order book features (eager)...")
    
    try:
        # Step 1: Extract all levels in batch operations
        bid_price_exprs = [
            pl.col('bid_levels_parsed').list.get(i).list.get(0).alias(f'bid_price_l{i+1}')
            for i in range(N_levels)
        ]
        bid_size_exprs = [
            pl.col('bid_levels_parsed').list.get(i).list.get(1).alias(f'bid_size_l{i+1}')
            for i in range(N_levels)
        ]
        ask_price_exprs = [
            pl.col('ask_levels_parsed').list.get(i).list.get(0).alias(f'ask_price_l{i+1}')
            for i in range(N_levels)
        ]
        ask_size_exprs = [
            pl.col('ask_levels_parsed').list.get(i).list.get(1).alias(f'ask_size_l{i+1}')
            for i in range(N_levels)
        ]
        
        # Add all L2 extractions in one operation
        df = df.with_columns(
            bid_price_exprs + bid_size_exprs + ask_price_exprs + ask_size_exprs
        )
        
        # Step 2: Calculate depth totals
        bid_size_cols = [f'bid_size_l{i+1}' for i in range(N_levels)]
        ask_size_cols = [f'ask_size_l{i+1}' for i in range(N_levels)]
        
        # Ensure columns exist before calculating, with null handling
        df = df.with_columns([
            # Basic depths and spreads with null handling
            pl.sum_horizontal([pl.col(col).fill_null(0) for col in bid_size_cols]).alias('bid_depth_total'),
            pl.sum_horizontal([pl.col(col).fill_null(0) for col in ask_size_cols]).alias('ask_depth_total'),
            
            # Safe price and spread calculations with null checking
            pl.when(pl.col('ask_price_l1').is_not_null() & pl.col('bid_price_l1').is_not_null())
              .then(pl.col('ask_price_l1') - pl.col('bid_price_l1'))
              .otherwise(None)
              .alias('spread_abs_calc'),
            
            pl.when((pl.col('ask_price_l1').is_not_null()) & (pl.col('bid_price_l1').is_not_null()) & (pl.col('bid_price_l1') > 0))
              .then((pl.col('ask_price_l1') - pl.col('bid_price_l1')) / pl.col('bid_price_l1'))
              .otherwise(None)
              .alias('spread_rel_calc'),
            
            pl.when(pl.col('ask_price_l1').is_not_null() & pl.col('bid_price_l1').is_not_null())
              .then((pl.col('bid_price_l1') + pl.col('ask_price_l1')) / 2)
              .otherwise(None)
              .alias('mid_price_calc'),
        ])
        
        # Step 3: Calculate derived features using the depth totals
        df = df.with_columns([
            # Weighted Average Price (WAP)
            pl.when((pl.col('bid_size_l1') + pl.col('ask_size_l1')) > 0)
              .then((pl.col('bid_price_l1') * pl.col('ask_size_l1') + pl.col('ask_price_l1') * pl.col('bid_size_l1')) / 
                    (pl.col('bid_size_l1') + pl.col('ask_size_l1')))
              .otherwise(None)
              .alias('wap'),
            
            # Order Book Imbalance (OBI)
            pl.when((pl.col('bid_depth_total') + pl.col('ask_depth_total')) > 0)
              .then((pl.col('bid_depth_total') - pl.col('ask_depth_total')) / 
                    (pl.col('bid_depth_total') + pl.col('ask_depth_total')))
              .otherwise(None)
              .alias('obi_total'),
        ])
        
    except Exception as e:
        logger.warning(f"Error in L2 features: {e}")
        # Add minimal features if processing fails
        df = df.with_columns([
            pl.lit(None).alias('bid_depth_total'),
            pl.lit(None).alias('ask_depth_total'),
            pl.lit(None).alias('spread_abs_calc'),
            pl.lit(None).alias('spread_rel_calc'),
            pl.lit(None).alias('mid_price_calc'),
            pl.lit(None).alias('wap'),
            pl.lit(None).alias('obi_total'),
        ])
    
    return df

def _add_technical_indicators_eager(df: pl.DataFrame) -> pl.DataFrame:
    """Eager technical indicators processing."""
    logger.info("Processing technical indicators (eager)...")
    
    try:
        # Use existing mid_price or fallback to mid_price_calc
        price_col = 'mid_price' if 'mid_price' in df.columns else 'mid_price_calc'
        
        # RSI calculations (multiple periods in one go)
        rsi_periods = [7, 14, 21, 30]
        rsi_exprs = [
            rsi(pl.col(price_col), period=period).over('market').alias(f'rsi_{period}')
            for period in rsi_periods
        ]
        
        df = df.with_columns(rsi_exprs)
        
        # RSI-derived features
        df = df.with_columns([
            pl.col('rsi_14').rolling_max(20).over('market').alias('rsi_overbought'),
            pl.col('rsi_14').rolling_min(20).over('market').alias('rsi_oversold'),
            pl.col('rsi_14').diff(1).over('market').alias('rsi_momentum'),
            pl.col('rsi_14').ewm_mean(span=9).over('market').alias('rsi_ema_9'),
            (pl.col('rsi_14') > 70).cast(pl.Int8).alias('rsi_overbought_flag'),
            (pl.col('rsi_14') < 30).cast(pl.Int8).alias('rsi_oversold_flag'),
        ])
        
        # MACD
        macd_results = macd(pl.col(price_col)).over('market')
        df = df.with_columns([
            macd_results.struct.field('macd_line').alias('macd_line'),
            macd_results.struct.field('macd_signal_line').alias('macd_signal_line'),
            macd_results.struct.field('macd_histogram').alias('macd_histogram')
        ])
        
        # Additional technical indicators
        df = df.with_columns([
            # Bollinger Bands
            bollinger_bands(pl.col(price_col), period=20).over('market').struct.field('upper').alias('bb_upper'),
            bollinger_bands(pl.col(price_col), period=20).over('market').struct.field('lower').alias('bb_lower'),
            bollinger_bands(pl.col(price_col), period=20).over('market').struct.field('width').alias('bb_width'),
            bollinger_bands(pl.col(price_col), period=20).over('market').struct.field('position').alias('bb_position'),
            
            # Stochastic Oscillator
            stochastic_oscillator(pl.col(price_col), period=14).over('market').alias('stoch_k'),
            
            # Williams %R
            williams_r(pl.col(price_col), period=14).over('market').alias('williams_r'),
            
            # Commodity Channel Index
            cci(pl.col(price_col), period=20).over('market').alias('cci'),
            
            # Average True Range (using mid_price as proxy)
            atr(pl.col(price_col), period=14).over('market').alias('atr'),
        ])
        
    except Exception as e:
        logger.warning(f"Error in technical indicators: {e}")
        # Add null features if processing fails
        null_features = ['rsi_7', 'rsi_14', 'rsi_21', 'rsi_30', 'rsi_overbought', 'rsi_oversold', 
                        'rsi_momentum', 'rsi_ema_9', 'rsi_overbought_flag', 'rsi_oversold_flag',
                        'macd_line', 'macd_signal_line', 'macd_histogram', 'bb_upper', 'bb_lower',
                        'bb_width', 'bb_position', 'stoch_k', 'williams_r', 'cci', 'atr']
        for feature in null_features:
            if feature not in df.columns:
                df = df.with_columns(pl.lit(None).alias(feature))
    
    return df

def _add_volume_volatility_features_eager(df: pl.DataFrame) -> pl.DataFrame:
    """Eager volume and volatility features processing."""
    logger.info("Processing volume and volatility features (eager)...")
    
    try:
        price_col = 'mid_price' if 'mid_price' in df.columns else 'mid_price_calc'
        
        # Volume features (conditionally add if volume_1m exists)
        if 'volume_1m' in df.columns:
            volume_features = [
                pl.col('volume_1m').rolling_mean(window_size=20).over('market').alias('volume_sma_20'),
                pl.col('volume_1m').rolling_std(window_size=20).over('market').alias('volume_std_20'),
                pl.col('volume_1m').pct_change().over('market').alias('volume_change'),
                pl.col('volume_1m').rolling_max(20).over('market').alias('volume_max_20'),
                (pl.col('volume_1m') / pl.col('volume_1m').rolling_mean(20).over('market')).alias('volume_ratio'),
            ]
            df = df.with_columns(volume_features)
        else:
            # Add null volume features
            volume_features = ['volume_sma_20', 'volume_std_20', 'volume_change', 'volume_max_20', 'volume_ratio']
            for feature in volume_features:
                df = df.with_columns(pl.lit(None).alias(feature))
        
        # Volatility features
        volatility_features = [
            pl.col(price_col).cast(pl.Float64).pct_change().rolling_std(window_size=20).over('market').alias('volatility_20p'),
            pl.col(price_col).cast(pl.Float64).pct_change().rolling_std(window_size=60).over('market').alias('volatility_60p'),
            
            # Price momentum features
            pl.col(price_col).pct_change(1).over('market').alias('returns_1p'),
            pl.col(price_col).pct_change(5).over('market').alias('returns_5p'),
            pl.col(price_col).pct_change(20).over('market').alias('returns_20p'),
            
            # Rolling Sharpe ratio (simplified)
            (pl.col(price_col).pct_change().rolling_mean(20) / 
             pl.col(price_col).pct_change().rolling_std(20)).over('market').alias('sharpe_20p'),
        ]
        
        # Add spread-related features if available
        if 'spread_rel_calc' in df.columns:
            volatility_features.extend([
                pl.col('spread_rel_calc').rolling_mean(window_size=20).over('market').alias('spread_rel_sma_20'),
                pl.col('spread_rel_calc').rolling_std(window_size=20).over('market').alias('spread_rel_std_20'),
            ])
        else:
            volatility_features.extend([
                pl.lit(None).alias('spread_rel_sma_20'),
                pl.lit(None).alias('spread_rel_std_20'),
            ])
        
        df = df.with_columns(volatility_features)
        
    except Exception as e:
        logger.warning(f"Error in volume/volatility features: {e}")
        # Add null features
        null_features = ['volume_sma_20', 'volume_std_20', 'volume_change', 'volume_max_20', 'volume_ratio',
                        'volatility_20p', 'volatility_60p', 'spread_rel_sma_20', 'spread_rel_std_20',
                        'returns_1p', 'returns_5p', 'returns_20p', 'sharpe_20p']
        for feature in null_features:
            if feature not in df.columns:
                df = df.with_columns(pl.lit(None).alias(feature))
    
    return df

def _add_advanced_orderbook_features_eager(df: pl.DataFrame, N_levels: int) -> pl.DataFrame:
    """Eager advanced order book features processing."""
    logger.info("Processing advanced order book features (eager)...")
    
    try:
        # Multi-level imbalances
        imbalance_features = []
        for level in range(1, min(N_levels + 1, 4)):  # Levels 1-3
            if f'bid_size_l{level}' in df.columns and f'ask_size_l{level}' in df.columns:
                imbalance_features.extend([
                    pl.when((pl.col(f'bid_size_l{level}') + pl.col(f'ask_size_l{level}')) > 0)
                      .then(pl.col(f'bid_size_l{level}') / (pl.col(f'bid_size_l{level}') + pl.col(f'ask_size_l{level}')))
                      .otherwise(None)
                      .alias(f'size_imbalance_l{level}'),
                    
                    pl.when((pl.col(f'bid_price_l{level}') * pl.col(f'bid_size_l{level}') + 
                            pl.col(f'ask_price_l{level}') * pl.col(f'ask_size_l{level}')) > 0)
                      .then((pl.col(f'bid_price_l{level}') * pl.col(f'bid_size_l{level}')) / 
                           (pl.col(f'bid_price_l{level}') * pl.col(f'bid_size_l{level}') + 
                            pl.col(f'ask_price_l{level}') * pl.col(f'ask_size_l{level}')))
                      .otherwise(None)
                      .alias(f'value_imbalance_l{level}'),
                ])
            else:
                imbalance_features.extend([
                    pl.lit(None).alias(f'size_imbalance_l{level}'),
                    pl.lit(None).alias(f'value_imbalance_l{level}'),
                ])
        
        # Order book slope (price impact estimation)
        slope_features = []
        if all(f'bid_price_l{i}' in df.columns for i in [1, 2, 3]) and all(f'bid_size_l{i}' in df.columns for i in [1, 2, 3]):
            slope_features.extend([
                pl.when((pl.col('bid_size_l1') + pl.col('bid_size_l2') + pl.col('bid_size_l3')) > 0)
                  .then((pl.col('bid_price_l1') - pl.col('bid_price_l3')) / 
                       (pl.col('bid_size_l1') + pl.col('bid_size_l2') + pl.col('bid_size_l3')))
                  .otherwise(None)
                  .alias('bid_slope'),
                
                pl.when((pl.col('ask_size_l1') + pl.col('ask_size_l2') + pl.col('ask_size_l3')) > 0)
                  .then((pl.col('ask_price_l3') - pl.col('ask_price_l1')) / 
                       (pl.col('ask_size_l1') + pl.col('ask_size_l2') + pl.col('ask_size_l3')))
                  .otherwise(None)
                  .alias('ask_slope'),
            ])
        else:
            slope_features.extend([
                pl.lit(None).alias('bid_slope'),
                pl.lit(None).alias('ask_slope'),
            ])
        
        # Liquidity measures
        liquidity_features = []
        if all(f'bid_price_l{i}' in df.columns for i in [1, 2, 3]) and all(f'bid_size_l{i}' in df.columns for i in [1, 2, 3]):
            liquidity_features.extend([
                pl.when(pl.sum_horizontal([pl.col(f'bid_size_l{i+1}') for i in range(3)]) > 0)
                  .then(pl.sum_horizontal([pl.col(f'bid_price_l{i+1}') * pl.col(f'bid_size_l{i+1}') for i in range(3)]) / 
                       pl.sum_horizontal([pl.col(f'bid_size_l{i+1}') for i in range(3)]))
                  .otherwise(None)
                  .alias('bid_vwap_3l'),
                
                pl.when(pl.sum_horizontal([pl.col(f'ask_size_l{i+1}') for i in range(3)]) > 0)
                  .then(pl.sum_horizontal([pl.col(f'ask_price_l{i+1}') * pl.col(f'ask_size_l{i+1}') for i in range(3)]) / 
                       pl.sum_horizontal([pl.col(f'ask_size_l{i+1}') for i in range(3)]))
                  .otherwise(None)
                  .alias('ask_vwap_3l'),
                
                # Order book stability
                pl.col('bid_price_l1').rolling_std(5).over('market').alias('bid_price_stability'),
                pl.col('ask_price_l1').rolling_std(5).over('market').alias('ask_price_stability'),
            ])
        else:
            liquidity_features.extend([
                pl.lit(None).alias('bid_vwap_3l'),
                pl.lit(None).alias('ask_vwap_3l'),
                pl.lit(None).alias('bid_price_stability'),
                pl.lit(None).alias('ask_price_stability'),
            ])
        
        df = df.with_columns(imbalance_features + slope_features + liquidity_features)
        
    except Exception as e:
        logger.warning(f"Error in advanced orderbook features: {e}")
        # Add null features
        null_features = []
        for level in range(1, 4):
            null_features.extend([f'size_imbalance_l{level}', f'value_imbalance_l{level}'])
        null_features.extend(['bid_slope', 'ask_slope', 'bid_vwap_3l', 'ask_vwap_3l', 
                             'bid_price_stability', 'ask_price_stability'])
        for feature in null_features:
            if feature not in df.columns:
                df = df.with_columns(pl.lit(None).alias(feature))
    
    return df

def _add_microstructure_features_eager(df: pl.DataFrame) -> pl.DataFrame:
    """Eager microstructure features processing."""
    logger.info("Processing microstructure features (eager)...")
    
    try:
        price_col = 'mid_price_calc' if 'mid_price_calc' in df.columns else 'mid_price'
        
        microstructure_features = [
            # Trade direction proxy (mid price changes)
            pl.col(price_col).diff(1).over('market').alias('price_direction'),
            (pl.col(price_col).diff(1) > 0).cast(pl.Int8).over('market').alias('uptick_flag'),
        ]
        
        # Order flow proxy
        if 'obi_total' in df.columns:
            microstructure_features.extend([
                pl.col('obi_total').diff(1).over('market').alias('obi_change'),
                pl.col('obi_total').rolling_mean(10).over('market').alias('obi_trend'),
            ])
        else:
            microstructure_features.extend([
                pl.lit(None).alias('obi_change'),
                pl.lit(None).alias('obi_trend'),
            ])
        
        # Spread dynamics
        if 'spread_abs_calc' in df.columns:
            microstructure_features.extend([
                pl.col('spread_abs_calc').diff(1).over('market').alias('spread_change'),
                pl.col('spread_abs_calc').rolling_mean(10).over('market').alias('spread_trend'),
            ])
        else:
            microstructure_features.extend([
                pl.lit(None).alias('spread_change'),
                pl.lit(None).alias('spread_trend'),
            ])
        
        df = df.with_columns(microstructure_features)
        
    except Exception as e:
        logger.warning(f"Error in microstructure features: {e}")
        # Add null features
        null_features = ['price_direction', 'uptick_flag', 'obi_change', 'obi_trend', 
                        'spread_change', 'spread_trend']
        for feature in null_features:
            if feature not in df.columns:
                df = df.with_columns(pl.lit(None).alias(feature))
    
    return df

def _add_cross_market_features_eager(df: pl.DataFrame) -> pl.DataFrame:
    """Eager cross-market features processing."""
    logger.info("Processing cross-market features (eager)...")
    
    try:
        # Use existing mid_price or fallback to mid_price_calc
        price_col = 'mid_price' if 'mid_price' in df.columns else 'mid_price_calc'
        
        # Create pivot for prices only
        df_pivot_price = df.pivot(
            index='ts_utc', 
            columns='market', 
            values=price_col
        ).fill_null(strategy='forward')
        
        available_markets = [col for col in df_pivot_price.columns if col != 'ts_utc']
        
        # Generate cross-market ratios and correlations
        if 'SOL_USDT' in available_markets:
            for other_market in ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']:
                if other_market in available_markets:
                    market_abbrev = other_market.split('_')[0].lower()
                    
                    # Price ratio
                    price_ratio = df_pivot_price.select([
                        pl.col('ts_utc'),
                        pl.when(pl.col(other_market) > 0)
                          .then(pl.col('SOL_USDT') / pl.col(other_market))
                          .otherwise(None)
                          .alias(f'sol_{market_abbrev}_ratio')
                    ])
                    
                    # Lagged returns
                    returns = df_pivot_price.select([
                        pl.col('ts_utc'),
                        pl.col(other_market).pct_change().alias(f'{market_abbrev}_return_1p'),
                        pl.col(other_market).pct_change(5).alias(f'{market_abbrev}_return_5p'),
                    ])
                    
                    # Join features back to main dataset
                    df = df.join(price_ratio, on='ts_utc', how='left')
                    df = df.join(returns, on='ts_utc', how='left')
        else:
            # Add null cross-market features if SOL_USDT not available
            for other_market in ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']:
                market_abbrev = other_market.split('_')[0].lower()
                df = df.with_columns([
                    pl.lit(None).alias(f'sol_{market_abbrev}_ratio'),
                    pl.lit(None).alias(f'{market_abbrev}_return_1p'),
                    pl.lit(None).alias(f'{market_abbrev}_return_5p'),
                ])
        
    except Exception as e:
        logger.warning(f"Error in cross-market features: {e}")
        # Add null cross-market features
        for other_market in ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']:
            market_abbrev = other_market.split('_')[0].lower()
            null_features = [f'sol_{market_abbrev}_ratio', f'{market_abbrev}_return_1p', f'{market_abbrev}_return_5p']
            for feature in null_features:
                if feature not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(feature))
    
    return df

def _add_time_features_eager(df: pl.DataFrame) -> pl.DataFrame:
    """Eager time features processing."""
    logger.info("Processing time-based features (eager)...")
    
    try:
        time_features = [
            # Basic time components
            pl.col('ts_utc').dt.hour().alias('hour_of_day'),
            pl.col('ts_utc').dt.weekday().alias('day_of_week'),
            pl.col('ts_utc').dt.minute().alias('minute_of_hour'),
            
            # Cyclical encoding
            (2 * np.pi * pl.col('ts_utc').dt.hour() / 24.0).sin().alias('hour_sin'),
            (2 * np.pi * pl.col('ts_utc').dt.hour() / 24.0).cos().alias('hour_cos'),
            (2 * np.pi * pl.col('ts_utc').dt.weekday() / 7.0).sin().alias('day_sin'),
            (2 * np.pi * pl.col('ts_utc').dt.weekday() / 7.0).cos().alias('day_cos'),
            (2 * np.pi * pl.col('ts_utc').dt.minute() / 60.0).sin().alias('minute_sin'),
            (2 * np.pi * pl.col('ts_utc').dt.minute() / 60.0).cos().alias('minute_cos'),
            
            # Trading session indicators
            pl.when(pl.col('ts_utc').dt.hour().is_in([0, 1, 2, 3, 4, 5]))
              .then(1).otherwise(0).alias('asian_session'),
            pl.when(pl.col('ts_utc').dt.hour().is_in([8, 9, 10, 11, 12, 13, 14, 15]))
              .then(1).otherwise(0).alias('european_session'),
            pl.when(pl.col('ts_utc').dt.hour().is_in([14, 15, 16, 17, 18, 19, 20, 21]))
              .then(1).otherwise(0).alias('us_session'),
        ]
        
        df = df.with_columns(time_features)
        
    except Exception as e:
        logger.warning(f"Error in time features: {e}")
        # Add null features
        null_features = ['hour_of_day', 'day_of_week', 'minute_of_hour', 'hour_sin', 'hour_cos',
                        'day_sin', 'day_cos', 'minute_sin', 'minute_cos', 'asian_session',
                        'european_session', 'us_session']
        for feature in null_features:
            if feature not in df.columns:
                df = df.with_columns(pl.lit(None).alias(feature))
    
    return df

def _add_regime_features_eager(df: pl.DataFrame) -> pl.DataFrame:
    """Eager regime features processing."""
    logger.info("Processing market regime features (eager)...")
    
    try:
        # Market phase dummies (if available)
        if 'market_phase' in df.columns:
            df = df.rename({'market_phase': 'phase'})
            df = df.to_dummies(columns=['phase'], separator='_')
        
        # Volatility regime dummies (if available)
        if 'volatility_regime' in df.columns:
            df = df.rename({'volatility_regime': 'volregime'})
            df = df.to_dummies(columns=['volregime'], separator='_')
            
    except Exception as e:
        logger.warning(f"Error in regime features: {e}")
    
    return df

def _add_l2_features_optimized(lazy_df: pl.LazyFrame, N_levels: int) -> pl.LazyFrame:
    """Optimized L2 order book feature extraction."""
    logger.info("Processing L2 order book features...")
    
    # Step 1: Extract all levels in batch operations
    bid_price_exprs = [
        pl.col('bid_levels_parsed').list.get(i).list.get(0).alias(f'bid_price_l{i+1}')
        for i in range(N_levels)
    ]
    bid_size_exprs = [
        pl.col('bid_levels_parsed').list.get(i).list.get(1).alias(f'bid_size_l{i+1}')
        for i in range(N_levels)
    ]
    ask_price_exprs = [
        pl.col('ask_levels_parsed').list.get(i).list.get(0).alias(f'ask_price_l{i+1}')
        for i in range(N_levels)
    ]
    ask_size_exprs = [
        pl.col('ask_levels_parsed').list.get(i).list.get(1).alias(f'ask_size_l{i+1}')
        for i in range(N_levels)
    ]
    
    # Add all L2 extractions in one operation
    lazy_df = lazy_df.with_columns(
        bid_price_exprs + bid_size_exprs + ask_price_exprs + ask_size_exprs
    )
    
    # Step 2: Calculate depth totals
    bid_size_cols = [f'bid_size_l{i+1}' for i in range(N_levels)]
    ask_size_cols = [f'ask_size_l{i+1}' for i in range(N_levels)]
    
    # Ensure columns exist before calculating, with null handling
    lazy_df = lazy_df.with_columns([
        # Basic depths and spreads with null handling
        pl.sum_horizontal([pl.col(col).fill_null(0) for col in bid_size_cols]).alias('bid_depth_total'),
        pl.sum_horizontal([pl.col(col).fill_null(0) for col in ask_size_cols]).alias('ask_depth_total'),
        
        # Safe price and spread calculations with null checking
        pl.when(pl.col('ask_price_l1').is_not_null() & pl.col('bid_price_l1').is_not_null())
          .then(pl.col('ask_price_l1') - pl.col('bid_price_l1'))
          .otherwise(None)
          .alias('spread_abs_calc'),
        
        pl.when((pl.col('ask_price_l1').is_not_null()) & (pl.col('bid_price_l1').is_not_null()) & (pl.col('bid_price_l1') > 0))
          .then((pl.col('ask_price_l1') - pl.col('bid_price_l1')) / pl.col('bid_price_l1'))
          .otherwise(None)
          .alias('spread_rel_calc'),
        
        pl.when(pl.col('ask_price_l1').is_not_null() & pl.col('bid_price_l1').is_not_null())
          .then((pl.col('bid_price_l1') + pl.col('ask_price_l1')) / 2)
          .otherwise(None)
          .alias('mid_price_calc'),
    ])
    
    # Step 3: Calculate derived features using the depth totals
    lazy_df = lazy_df.with_columns([
        # Weighted Average Price (WAP)
        pl.when((pl.col('bid_size_l1') + pl.col('ask_size_l1')) > 0)
          .then((pl.col('bid_price_l1') * pl.col('ask_size_l1') + pl.col('ask_price_l1') * pl.col('bid_size_l1')) / 
                (pl.col('bid_size_l1') + pl.col('ask_size_l1')))
          .otherwise(None)
          .alias('wap'),
        
        # Order Book Imbalance (OBI)
        pl.when((pl.col('bid_depth_total') + pl.col('ask_depth_total')) > 0)
          .then((pl.col('bid_depth_total') - pl.col('ask_depth_total')) / 
                (pl.col('bid_depth_total') + pl.col('ask_depth_total')))
          .otherwise(None)
          .alias('obi_total'),
    ])
    
    return lazy_df

def _add_technical_indicators_batched(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Add technical indicators in batched operations."""
    logger.info("Processing technical indicators...")
    
    # RSI calculations (multiple periods in one go)
    rsi_periods = [7, 14, 21, 30]
    rsi_exprs = [
        rsi(pl.col('mid_price'), period=period).over('market').alias(f'rsi_{period}')
        for period in rsi_periods
    ]
    
    lazy_df = lazy_df.with_columns(rsi_exprs)
    
    # RSI-derived features
    lazy_df = lazy_df.with_columns([
        pl.col('rsi_14').rolling_max(20).over('market').alias('rsi_overbought'),
        pl.col('rsi_14').rolling_min(20).over('market').alias('rsi_oversold'),
        pl.col('rsi_14').diff(1).over('market').alias('rsi_momentum'),
        pl.col('rsi_14').ewm_mean(span=9).over('market').alias('rsi_ema_9'),
        (pl.col('rsi_14') > 70).cast(pl.Int8).alias('rsi_overbought_flag'),
        (pl.col('rsi_14') < 30).cast(pl.Int8).alias('rsi_oversold_flag'),
    ])
    
    # MACD
    macd_results = macd(pl.col('mid_price')).over('market')
    lazy_df = lazy_df.with_columns([
        macd_results.struct.field('macd_line').alias('macd_line'),
        macd_results.struct.field('macd_signal_line').alias('macd_signal_line'),
        macd_results.struct.field('macd_histogram').alias('macd_histogram')
    ])
    
    # Additional technical indicators
    lazy_df = lazy_df.with_columns([
        # Bollinger Bands
        bollinger_bands(pl.col('mid_price'), period=20).over('market').struct.field('upper').alias('bb_upper'),
        bollinger_bands(pl.col('mid_price'), period=20).over('market').struct.field('lower').alias('bb_lower'),
        bollinger_bands(pl.col('mid_price'), period=20).over('market').struct.field('width').alias('bb_width'),
        bollinger_bands(pl.col('mid_price'), period=20).over('market').struct.field('position').alias('bb_position'),
        
        # Stochastic Oscillator
        stochastic_oscillator(pl.col('mid_price'), period=14).over('market').alias('stoch_k'),
        
        # Williams %R
        williams_r(pl.col('mid_price'), period=14).over('market').alias('williams_r'),
        
        # Commodity Channel Index
        cci(pl.col('mid_price'), period=20).over('market').alias('cci'),
        
        # Average True Range (using mid_price as proxy)
        atr(pl.col('mid_price'), period=14).over('market').alias('atr'),
    ])
    
    return lazy_df

def _add_volume_volatility_features(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Add volume and volatility-based features."""
    logger.info("Processing volume and volatility features...")
    
    volume_features = []
    
    # Volume features (conditionally add if volume_1m exists)
    # We'll add them to the final feature list if they work
    volume_features.extend([
        pl.col('volume_1m').rolling_mean(window_size=20).over('market').alias('volume_sma_20'),
        pl.col('volume_1m').rolling_std(window_size=20).over('market').alias('volume_std_20'),
        pl.col('volume_1m').pct_change().over('market').alias('volume_change'),
        pl.col('volume_1m').rolling_max(20).over('market').alias('volume_max_20'),
        (pl.col('volume_1m') / pl.col('volume_1m').rolling_mean(20).over('market')).alias('volume_ratio'),
    ])
    
    # Volatility features - use the correct column names from L2 features
    volatility_features = [
        pl.col('mid_price').cast(pl.Float64).pct_change().rolling_std(window_size=20).over('market').alias('volatility_20p'),
        pl.col('mid_price').cast(pl.Float64).pct_change().rolling_std(window_size=60).over('market').alias('volatility_60p'),
        pl.col('spread_rel_calc').rolling_mean(window_size=20).over('market').alias('spread_rel_sma_20'),
        pl.col('spread_rel_calc').rolling_std(window_size=20).over('market').alias('spread_rel_std_20'),
        
        # Price momentum features
        pl.col('mid_price').pct_change(1).over('market').alias('returns_1p'),
        pl.col('mid_price').pct_change(5).over('market').alias('returns_5p'),
        pl.col('mid_price').pct_change(20).over('market').alias('returns_20p'),
        
        # Rolling Sharpe ratio (simplified)
        (pl.col('mid_price').pct_change().rolling_mean(20) / 
         pl.col('mid_price').pct_change().rolling_std(20)).over('market').alias('sharpe_20p'),
    ]
    
    lazy_df = lazy_df.with_columns(volume_features + volatility_features)
    
    return lazy_df

def _add_advanced_orderbook_features(lazy_df: pl.LazyFrame, N_levels: int) -> pl.LazyFrame:
    """Add advanced order book features."""
    logger.info("Processing advanced order book features...")
    
    # Multi-level imbalances
    imbalance_features = []
    for level in range(1, min(N_levels + 1, 4)):  # Levels 1-3
        imbalance_features.extend([
            (pl.col(f'bid_size_l{level}') / (pl.col(f'bid_size_l{level}') + pl.col(f'ask_size_l{level}')))
            .alias(f'size_imbalance_l{level}'),
            
            ((pl.col(f'bid_price_l{level}') * pl.col(f'bid_size_l{level}')) / 
             (pl.col(f'bid_price_l{level}') * pl.col(f'bid_size_l{level}') + 
              pl.col(f'ask_price_l{level}') * pl.col(f'ask_size_l{level}')))
            .alias(f'value_imbalance_l{level}'),
        ])
    
    # Order book slope (price impact estimation)
    slope_features = [
        # Bid side slope
        ((pl.col('bid_price_l1') - pl.col('bid_price_l3')) / 
         (pl.col('bid_size_l1') + pl.col('bid_size_l2') + pl.col('bid_size_l3')))
        .alias('bid_slope'),
        
        # Ask side slope  
        ((pl.col('ask_price_l3') - pl.col('ask_price_l1')) / 
         (pl.col('ask_size_l1') + pl.col('ask_size_l2') + pl.col('ask_size_l3')))
        .alias('ask_slope'),
    ]
    
    # Liquidity measures
    liquidity_features = [
        # Depth-weighted average prices
        (pl.sum_horizontal([pl.col(f'bid_price_l{i+1}') * pl.col(f'bid_size_l{i+1}') for i in range(3)]) / 
         pl.sum_horizontal([pl.col(f'bid_size_l{i+1}') for i in range(3)]))
        .alias('bid_vwap_3l'),
        
        (pl.sum_horizontal([pl.col(f'ask_price_l{i+1}') * pl.col(f'ask_size_l{i+1}') for i in range(3)]) / 
         pl.sum_horizontal([pl.col(f'ask_size_l{i+1}') for i in range(3)]))
        .alias('ask_vwap_3l'),
        
        # Order book stability
        pl.col('bid_price_l1').rolling_std(5).over('market').alias('bid_price_stability'),
        pl.col('ask_price_l1').rolling_std(5).over('market').alias('ask_price_stability'),
    ]
    
    lazy_df = lazy_df.with_columns(imbalance_features + slope_features + liquidity_features)
    
    return lazy_df

def _add_microstructure_features(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Add market microstructure features."""
    logger.info("Processing microstructure features...")
    
    microstructure_features = [
        # Trade direction proxy (mid price changes)
        pl.col('mid_price_calc').diff(1).over('market').alias('price_direction'),
        (pl.col('mid_price_calc').diff(1) > 0).cast(pl.Int8).over('market').alias('uptick_flag'),
        
        # Order flow proxy
        pl.col('obi_total').diff(1).over('market').alias('obi_change'),
        pl.col('obi_total').rolling_mean(10).over('market').alias('obi_trend'),
        
        # Spread dynamics
        pl.col('spread_abs_calc').diff(1).over('market').alias('spread_change'),
        pl.col('spread_abs_calc').rolling_mean(10).over('market').alias('spread_trend'),
        
        # Volume-weighted features (if volume available)
    ]
    
    lazy_df = lazy_df.with_columns(microstructure_features)
    
    return lazy_df

def _add_cross_market_features_optimized(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Optimized cross-market feature generation."""
    logger.info("Processing cross-market features...")
    
    # More efficient pivot operation
    df = lazy_df.collect()
    
    # Create pivot for prices only
    df_pivot_price = df.pivot(
        index='ts_utc', 
        columns='market', 
        values='mid_price'
    ).fill_null(strategy='forward')
    
    available_markets = [col for col in df_pivot_price.columns if col != 'ts_utc']
    
    cross_market_features = []
    
    # Generate cross-market ratios and correlations
    if 'SOL_USDT' in available_markets:
        for other_market in ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']:
            if other_market in available_markets:
                market_abbrev = other_market.split('_')[0].lower()
                
                # Price ratio
                price_ratio = df_pivot_price.select([
                    pl.col('ts_utc'),
                    (pl.col('SOL_USDT') / pl.col(other_market)).alias(f'sol_{market_abbrev}_ratio')
                ])
                
                # Lagged returns
                returns = df_pivot_price.select([
                    pl.col('ts_utc'),
                    pl.col(other_market).pct_change().alias(f'{market_abbrev}_return_1p'),
                    pl.col(other_market).pct_change(5).alias(f'{market_abbrev}_return_5p'),
                ])
                
                # Rolling correlation (simplified)
                correlation = df_pivot_price.select([
                    pl.col('ts_utc'),
                    pl.corr(pl.col('SOL_USDT'), pl.col(other_market), method='pearson').over(
                        pl.int_range(pl.len()).map_elements(lambda x: max(0, x - 20), return_dtype=pl.Int64)
                    ).alias(f'sol_{market_abbrev}_corr_20p')
                ]) if df_pivot_price.shape[0] > 20 else None
                
                # Join features back to main dataset
                df = df.join(price_ratio, on='ts_utc', how='left')
                df = df.join(returns, on='ts_utc', how='left')
                if correlation is not None:
                    df = df.join(correlation, on='ts_utc', how='left')
    
    return df.lazy()

def _add_time_features(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Add time-based features with cyclical encoding."""
    logger.info("Processing time-based features...")
    
    time_features = [
        # Basic time components
        pl.col('ts_utc').dt.hour().alias('hour_of_day'),
        pl.col('ts_utc').dt.weekday().alias('day_of_week'),
        pl.col('ts_utc').dt.minute().alias('minute_of_hour'),
        
        # Cyclical encoding
        (2 * np.pi * pl.col('ts_utc').dt.hour() / 24.0).sin().alias('hour_sin'),
        (2 * np.pi * pl.col('ts_utc').dt.hour() / 24.0).cos().alias('hour_cos'),
        (2 * np.pi * pl.col('ts_utc').dt.weekday() / 7.0).sin().alias('day_sin'),
        (2 * np.pi * pl.col('ts_utc').dt.weekday() / 7.0).cos().alias('day_cos'),
        (2 * np.pi * pl.col('ts_utc').dt.minute() / 60.0).sin().alias('minute_sin'),
        (2 * np.pi * pl.col('ts_utc').dt.minute() / 60.0).cos().alias('minute_cos'),
        
        # Trading session indicators
        pl.when(pl.col('ts_utc').dt.hour().is_in([0, 1, 2, 3, 4, 5]))
          .then(1).otherwise(0).alias('asian_session'),
        pl.when(pl.col('ts_utc').dt.hour().is_in([8, 9, 10, 11, 12, 13, 14, 15]))
          .then(1).otherwise(0).alias('european_session'),
        pl.when(pl.col('ts_utc').dt.hour().is_in([14, 15, 16, 17, 18, 19, 20, 21]))
          .then(1).otherwise(0).alias('us_session'),
    ]
    
    lazy_df = lazy_df.with_columns(time_features)
    
    return lazy_df

def _add_regime_features(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Add market regime features."""
    logger.info("Processing market regime features...")
    
    # Convert to eager for regime processing
    df = lazy_df.collect()
    
    # Market phase dummies (if available)
    if 'market_phase' in df.columns:
        df = df.rename({'market_phase': 'phase'})
        df = df.to_dummies(columns=['phase'], separator='_')
    
    # Volatility regime dummies (if available)
    if 'volatility_regime' in df.columns:
        df = df.rename({'volatility_regime': 'volregime'})
        df = df.to_dummies(columns=['volregime'], separator='_')
    
    return df.lazy()

def _optimize_features(df: pl.DataFrame) -> pl.DataFrame:
    """Optimize features by removing redundant and highly correlated features."""
    logger.info("Optimizing feature set...")
    
    # Remove original parsed columns
    columns_to_drop = ['bid_levels_parsed', 'ask_levels_parsed']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if columns_to_drop:
        df = df.drop(columns_to_drop)
    
    # Important features that should never be removed
    critical_features = {
        'bid_depth_total', 'ask_depth_total', 'spread_abs_calc', 'spread_rel_calc', 
        'mid_price_calc', 'wap', 'obi_total', 'rsi_14', 'macd_line', 
        'volatility_20p', 'returns_1p', 'hour_sin', 'hour_cos'
    }
    
    # Identify and remove highly correlated features
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    
    if len(numeric_cols) > 1:
        # Calculate correlation matrix for feature reduction
        correlation_threshold = 0.95
        features_to_remove = _find_correlated_features(df, numeric_cols, correlation_threshold, critical_features)
        
        if features_to_remove:
            logger.info(f"Removing {len(features_to_remove)} highly correlated features")
            df = df.drop(features_to_remove)
    
    logger.info(f"Final feature count: {len(df.columns)}")
    return df

def _find_correlated_features(df: pl.DataFrame, numeric_cols: List[str], threshold: float, critical_features: set = None) -> List[str]:
    """Find highly correlated features to remove."""
    if critical_features is None:
        critical_features = set()
    
    features_to_remove = set()
    
    # Simple correlation-based removal (could be enhanced with more sophisticated methods)
    for i, col1 in enumerate(numeric_cols):
        if col1 in features_to_remove:
            continue
        for col2 in numeric_cols[i+1:]:
            if col2 in features_to_remove:
                continue
            try:
                # Calculate correlation for non-null values
                valid_data = df.select([col1, col2]).drop_nulls()
                if valid_data.shape[0] > 10:
                    corr = valid_data.select(pl.corr(col1, col2)).item()
                    if abs(corr) > threshold:
                        # Don't remove critical features
                        if col1 in critical_features and col2 not in critical_features:
                            features_to_remove.add(col2)
                        elif col2 in critical_features and col1 not in critical_features:
                            features_to_remove.add(col1)
                        elif col1 not in critical_features and col2 not in critical_features:
                            # Remove the feature with more nulls or the second one if equal
                            null_count1 = df[col1].null_count()
                            null_count2 = df[col2].null_count()
                            if null_count1 >= null_count2:
                                features_to_remove.add(col1)
                            else:
                                features_to_remove.add(col2)
                        # If both are critical, don't remove either
            except Exception:
                continue
    
    return list(features_to_remove)

# =============================================================================
# Technical Indicator Helper Functions
# =============================================================================

def rsi(series: pl.Expr, period: int = 14) -> pl.Expr:
    """Relative Strength Index."""
    delta = series.diff(1)
    gain = delta.clip(lower_bound=0).rolling_mean(window_size=period)
    loss = (-delta).clip(lower_bound=0).rolling_mean(window_size=period)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series: pl.Expr, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.Expr:
    """Moving Average Convergence Divergence."""
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

def bollinger_bands(series: pl.Expr, period: int = 20, std_dev: int = 2) -> pl.Expr:
    """Bollinger Bands."""
    sma = series.rolling_mean(window_size=period)
    std = series.rolling_std(window_size=period)
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    width = (upper - lower) / sma
    position = (series - lower) / (upper - lower)
    
    return pl.struct([
        upper.alias('upper'),
        lower.alias('lower'),
        width.alias('width'),
        position.alias('position')
    ])

def stochastic_oscillator(series: pl.Expr, period: int = 14) -> pl.Expr:
    """Stochastic Oscillator %K."""
    lowest_low = series.rolling_min(window_size=period)
    highest_high = series.rolling_max(window_size=period)
    return ((series - lowest_low) / (highest_high - lowest_low)) * 100

def williams_r(series: pl.Expr, period: int = 14) -> pl.Expr:
    """Williams %R."""
    highest_high = series.rolling_max(window_size=period)
    lowest_low = series.rolling_min(window_size=period)
    return ((highest_high - series) / (highest_high - lowest_low)) * -100

def cci(series: pl.Expr, period: int = 20) -> pl.Expr:
    """Commodity Channel Index."""
    typical_price = series  # Using mid_price as proxy
    sma = typical_price.rolling_mean(window_size=period)
    mean_deviation = (typical_price - sma).abs().rolling_mean(window_size=period)
    return (typical_price - sma) / (0.015 * mean_deviation)

def atr(series: pl.Expr, period: int = 14) -> pl.Expr:
    """Average True Range (simplified using mid_price)."""
    high_low = series.rolling_max(2) - series.rolling_min(2)
    return high_low.rolling_mean(window_size=period)

if __name__ == "__main__":
    print("Optimized feature engineering module ready.")
