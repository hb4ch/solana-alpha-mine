import os
import pandas as pd
import pyarrow.parquet as pq
from typing import List
import glob
import ast
import logging

logger = logging.getLogger(__name__)

def parse_l2_levels(level_str):
    """Safely parse stringified list of lists for bid/ask levels."""
    if pd.isna(level_str):
        return []
    try:
        return ast.literal_eval(level_str)
    except (ValueError, SyntaxError):
        logger.warning(f"Could not parse L2 level string: {level_str}")
        return [] # Return empty list on error

def load_all_market_data(base_path: str, markets: List[str]) -> pd.DataFrame:
    """
    Load and concatenate all market data from partitioned Parquet files
    for all available dates.
    """
    all_dfs = []
    for market in markets:
        date_paths = glob.glob(os.path.join(base_path, f'market={market}', 'date=*'))
        
        market_dfs = []
        for date_path in date_paths:
            parquet_file_path = os.path.join(date_path, 'enhanced_l2.parquet')
            if os.path.exists(parquet_file_path):
                df = pd.read_parquet(parquet_file_path)
                for col in df.columns:
                    if df[col].dtype == 'category':
                        df[col] = df[col].astype(str)
                df['market'] = market
                market_dfs.append(df)
        
        if market_dfs:
            all_dfs.extend(market_dfs)
    
    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['ts_utc'] = pd.to_datetime(combined_df['ts_utc'], unit='ms')
    combined_df.set_index('ts_utc', inplace=True)
    return combined_df.sort_index()

def load_and_preprocess_raw_data(base_path: str, markets: list):
    """Loads raw data and performs initial parsing for L2 order book."""
    logger.info(f"Loading and preprocessing raw data for markets: {markets} from {base_path}")
    df_combined = load_all_market_data(base_path=base_path, markets=markets)
    
    if df_combined.empty:
        logger.error("No data loaded. Exiting.")
        return pd.DataFrame()

    logger.info(f"Loaded combined data shape: {df_combined.shape}")

    # Parse L2 order book data (bid_levels, ask_levels)
    logger.info("Parsing L2 order book data (bid_levels, ask_levels)...")
    if 'bid_levels' in df_combined.columns:
        df_combined['bid_levels_parsed'] = df_combined['bid_levels'].apply(parse_l2_levels)
    else:
        logger.warning("Column 'bid_levels' not found.")
        df_combined['bid_levels_parsed'] = pd.Series([[] for _ in range(len(df_combined))])

    if 'ask_levels' in df_combined.columns:
        df_combined['ask_levels_parsed'] = df_combined['ask_levels'].apply(parse_l2_levels)
    else:
        logger.warning("Column 'ask_levels' not found.")
        df_combined['ask_levels_parsed'] = pd.Series([[] for _ in range(len(df_combined))])
        
    logger.info("Finished parsing L2 data.")
    return df_combined

if __name__ == "__main__":
    # Example usage
    df = load_and_preprocess_raw_data(
        base_path='crypto_tick',
        markets=['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT']
    )
    if not df.empty:
        print(f"Loaded data with shape: {df.shape}")
        print(df.head())
        print("\nData loaded for markets:", df['market'].unique())
        print("Date range:", df.index.min(), "to", df.index.max())
        print("Parsed columns check:", 'bid_levels_parsed' in df.columns)
    else:
        print("No data loaded.")
