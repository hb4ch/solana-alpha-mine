import os
import pandas as pd
import pyarrow.parquet as pq
from typing import List
import glob

def load_all_market_data(base_path: str, markets: List[str]) -> pd.DataFrame:
    """
    Load and concatenate all market data from partitioned Parquet files
    for all available dates.
    
    Args:
        base_path: Base directory of the dataset
        markets: List of market symbols (e.g., ['BTC_USDT', 'ETH_USDT'])
        
    Returns:
        Combined DataFrame with all market data from all dates
    """
    all_dfs = []
    for market in markets:
        # Find all date directories for the current market
        date_paths = glob.glob(os.path.join(base_path, f'market={market}', 'date=*'))
        
        market_dfs = []
        for date_path in date_paths:
            parquet_file_path = os.path.join(date_path, 'enhanced_l2.parquet')
            if os.path.exists(parquet_file_path):
                # Read Parquet file
                df = pd.read_parquet(parquet_file_path)
                
                # Ensure consistent column types
                for col in df.columns:
                    if df[col].dtype == 'category':
                        df[col] = df[col].astype(str)
                
                # Add market identifier
                df['market'] = market
                market_dfs.append(df)
        
        if market_dfs:
            all_dfs.extend(market_dfs)
    
    if not all_dfs:
        return pd.DataFrame() # Return empty DataFrame if no data found

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['ts_utc'] = pd.to_datetime(combined_df['ts_utc'], unit='ms')
    combined_df.set_index('ts_utc', inplace=True)
    return combined_df.sort_index()

if __name__ == "__main__":
    # Example usage
    df = load_all_market_data(
        base_path='crypto_tick',
        markets=['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT']
    )
    if not df.empty:
        print(f"Loaded data with shape: {df.shape}")
        print(df.head())
        print("\nData loaded for markets:", df['market'].unique())
        print("Date range:", df.index.min(), "to", df.index.max())
    else:
        print("No data loaded.")
