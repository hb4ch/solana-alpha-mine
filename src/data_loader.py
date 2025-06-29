import os
import polars as pl
from typing import List, Optional, Dict, Tuple
import glob
import logging
import time
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import the C++ optimized L2 parser
try:
    from .fast_l2_parser_wrapper import parse_l2_levels_optimized, get_parser_stats, reset_parser_stats
except ImportError:
    from fast_l2_parser_wrapper import parse_l2_levels_optimized, get_parser_stats, reset_parser_stats

logger = logging.getLogger(__name__)

# Cache directory for parsed L2 data
CACHE_DIR = Path("cache/l2_parsed")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_key(file_path: str, modification_time: float) -> str:
    """Generate a cache key based on file path and modification time."""
    key_string = f"{file_path}_{modification_time}"
    return hashlib.md5(key_string.encode()).hexdigest()

def load_single_file_with_cache(file_info: Tuple[str, str]) -> Optional[pl.DataFrame]:
    """
    Load a single parquet file with intelligent caching for parsed L2 data.
    Uses only the C++ optimized L2 parser.
    """
    parquet_file_path, market = file_info
    
    if not os.path.exists(parquet_file_path) or os.path.getsize(parquet_file_path) <= 12:
        return None
    
    # Check cache first
    mod_time = os.path.getmtime(parquet_file_path)
    cache_key = get_cache_key(parquet_file_path, mod_time)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
                logger.debug(f"Loaded from cache: {parquet_file_path}")
                return df
        except Exception as e:
            logger.warning(f"Cache read failed for {parquet_file_path}: {e}")
    
    # Load and process file
    try:
        df = pl.read_parquet(parquet_file_path)
        df = df.with_columns(pl.lit(market).alias('market'))
        
        # Parse L2 data using C++ optimized parser only
        if 'bid_levels' in df.columns:
            df = df.with_columns(
                pl.col('bid_levels').map_elements(
                    parse_l2_levels_optimized, 
                    return_dtype=pl.List(pl.List(pl.Float64))
                ).alias('bid_levels_parsed')
            )
        else:
            df = df.with_columns(
                pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias('bid_levels_parsed')
            )

        if 'ask_levels' in df.columns:
            df = df.with_columns(
                pl.col('ask_levels').map_elements(
                    parse_l2_levels_optimized, 
                    return_dtype=pl.List(pl.List(pl.Float64))
                ).alias('ask_levels_parsed')
            )
        else:
            df = df.with_columns(
                pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias('ask_levels_parsed')
            )
        
        # Cache the result
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            logger.debug(f"Cached processed data: {parquet_file_path}")
        except Exception as e:
            logger.warning(f"Cache write failed for {parquet_file_path}: {e}")
        
        return df
        
    except Exception as e:
        logger.warning(f"Could not read/process Parquet file: {parquet_file_path}. Error: {e}")
        return None

def load_market_data_streaming(base_path: str, markets: List[str], 
                              date_limit: Optional[int] = None,
                              max_workers: int = None) -> pl.DataFrame:
    """
    Streaming data loader that processes files in date order with memory optimization.
    Uses only the C++ optimized L2 parser.
    """
    start_time = time.time()
    
    # Collect and sort file tasks by date
    file_tasks = []
    date_file_map = {}
    
    for market in markets:
        date_paths = sorted(glob.glob(os.path.join(base_path, f'market={market}', 'date=*')))
        for date_path in date_paths:
            date_str = os.path.basename(date_path).replace('date=', '')
            parquet_file_path = os.path.join(date_path, 'enhanced_l2.parquet')
            
            if date_str not in date_file_map:
                date_file_map[date_str] = []
            date_file_map[date_str].append((parquet_file_path, market))
    
    # Process dates in order, optionally limiting number of dates
    sorted_dates = sorted(date_file_map.keys())
    if date_limit:
        sorted_dates = sorted_dates[-date_limit:]  # Take most recent dates
    
    logger.info(f"Processing {len(sorted_dates)} dates across {len(markets)} markets")
    
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)
    
    all_dfs = []
    total_files = sum(len(date_file_map[date]) for date in sorted_dates)
    processed_files = 0
    
    # Process date by date to control memory usage
    for date in sorted_dates:
        date_files = date_file_map[date]
        date_dfs = []
        
        # Process files for this date in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(load_single_file_with_cache, task): task for task in date_files}
            
            for future in as_completed(future_to_task):
                try:
                    df = future.result()
                    if df is not None and not df.is_empty():
                        date_dfs.append(df)
                    processed_files += 1
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"Error loading {task[0]}: {e}")
                    processed_files += 1
        
        # Combine data for this date
        if date_dfs:
            date_combined = pl.concat(date_dfs)
            all_dfs.append(date_combined)
            logger.info(f"Processed date {date}: {len(date_dfs)} files, {date_combined.shape[0]} rows")
        
        # Clean up to free memory
        del date_dfs
    
    load_time = time.time() - start_time
    logger.info(f"Streaming loading completed: {processed_files}/{total_files} files in {load_time:.2f} seconds")
    
    if not all_dfs:
        return pl.DataFrame()

    # Final concatenation and processing
    logger.info("Final concatenation and timestamp processing...")
    concat_start = time.time()
    
    combined_df = pl.concat(all_dfs)
    
    # Convert timestamp and sort
    combined_df = combined_df.with_columns(
        pl.from_epoch(pl.col('ts_utc'), time_unit='ms').alias('ts_utc')
    )
    combined_df = combined_df.sort('ts_utc')
    
    concat_time = time.time() - concat_start
    logger.info(f"Final processing completed in {concat_time:.2f} seconds")
    
    return combined_df

def clear_cache():
    """Clear the L2 parsing cache."""
    try:
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
        logger.info("L2 parsing cache cleared")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    try:
        cache_files = list(CACHE_DIR.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024)
        }
    except Exception:
        return {"files": 0, "total_size_mb": 0}

def load_and_preprocess_raw_data(base_path: str, markets: List[str], 
                                date_limit: Optional[int] = None,
                                max_workers: Optional[int] = None) -> pl.DataFrame:
    """
    Unified data loader with C++ optimized L2 parsing, streaming, and caching.
    
    Args:
        base_path: Base path to the data directory
        markets: List of market symbols to load
        date_limit: Limit number of most recent dates to process (None for all)
        max_workers: Maximum number of worker threads
    
    Returns:
        Combined DataFrame with C++ optimized L2 parsing
    """
    total_start = time.time()
    logger.info(f"Loading data with C++ optimized L2 parsing: markets={markets}")
    
    df_combined = load_market_data_streaming(
        base_path=base_path, 
        markets=markets, 
        date_limit=date_limit,
        max_workers=max_workers
    )
    
    if df_combined.is_empty():
        logger.error("No data loaded.")
        return pl.DataFrame()

    total_time = time.time() - total_start
    logger.info(f"C++ optimized loading completed in {total_time:.2f} seconds")
    logger.info(f"Final dataset shape: {df_combined.shape}")
    
    # Log cache statistics
    cache_stats = get_cache_stats()
    logger.info(f"Cache: {cache_stats['files']} files, {cache_stats['total_size_mb']:.1f} MB")
    
    # Log C++ parser statistics
    parser_stats = get_parser_stats()
    if parser_stats.get('cpp_parser_available'):
        logger.info(f"C++ Parser Stats: {parser_stats.get('success_rate', 0):.1f}% success rate, "
                   f"{parser_stats.get('total_parsed', 0)} total parsed")
    
    return df_combined

if __name__ == "__main__":
    # Test the unified loader
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    print("Testing unified C++ optimized data loader...")
    df = load_and_preprocess_raw_data(
        base_path='crypto_tick',
        markets=['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT'],
        date_limit=3  # Test with just 3 most recent dates
    )
    
    if not df.is_empty():
        print(f"✅ Loaded data with shape: {df.shape}")
        print(f"Markets: {df['market'].unique().to_list()}")
        print(f"Date range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")
        print(f"L2 parsing check: bid_levels_parsed={('bid_levels_parsed' in df.columns)}, ask_levels_parsed={('ask_levels_parsed' in df.columns)}")
        
        # Show sample of parsed data
        if 'bid_levels_parsed' in df.columns:
            sample_bid = df.select('bid_levels_parsed').head(1).item(0, 0)
            print(f"Sample bid levels: {sample_bid}")
    else:
        print("❌ No data loaded")
