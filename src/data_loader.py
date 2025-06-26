import os
import polars as pl
from typing import List
import glob
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time

logger = logging.getLogger(__name__)

def parse_l2_levels_optimized(level_str: str) -> list:
    """Optimized parsing of stringified list of lists for bid/ask levels."""
    if level_str is None or level_str == "":
        return []
    
    # Try JSON parsing first (faster than ast.literal_eval)
    try:
        # Handle cases where the string might already be in JSON format
        if level_str.startswith('['):
            return json.loads(level_str)
        else:
            # Fallback to ast.literal_eval for Python-specific formats
            import ast
            return ast.literal_eval(level_str)
    except (ValueError, SyntaxError, json.JSONDecodeError):
        logger.warning(f"Could not parse L2 level string: {level_str[:100]}...")
        return []

def parse_l2_levels(level_str: str) -> list:
    """Legacy function - kept for backward compatibility."""
    return parse_l2_levels_optimized(level_str)

def load_single_file(file_info: tuple) -> pl.DataFrame:
    """
    Load a single parquet file with error handling.
    
    Args:
        file_info: tuple of (parquet_file_path, market)
    
    Returns:
        DataFrame with market column added, or empty DataFrame on error
    """
    parquet_file_path, market = file_info
    
    if not os.path.exists(parquet_file_path) or os.path.getsize(parquet_file_path) <= 12:
        return pl.DataFrame()
    
    try:
        df = pl.read_parquet(parquet_file_path)
        df = df.with_columns(pl.lit(market).alias('market'))
        return df
    except pl.exceptions.ComputeError as e:
        logger.warning(f"Could not read Parquet file: {parquet_file_path}. Error: {e}")
        return pl.DataFrame()

def load_all_market_data(base_path: str, markets: List[str], max_workers: int = None) -> pl.DataFrame:
    """
    Load and concatenate all market data from partitioned Parquet files
    for all available dates using Polars with parallel processing.
    
    Args:
        base_path: Base path to the data directory
        markets: List of market symbols to load
        max_workers: Maximum number of worker threads (None for auto-detect)
    
    Returns:
        Combined DataFrame with all market data
    """
    start_time = time.time()
    
    # Collect all file paths that need to be loaded
    file_tasks = []
    for market in markets:
        date_paths = glob.glob(os.path.join(base_path, f'market={market}', 'date=*'))
        for date_path in date_paths:
            parquet_file_path = os.path.join(date_path, 'enhanced_l2.parquet')
            file_tasks.append((parquet_file_path, market))
    
    logger.info(f"Found {len(file_tasks)} files to load across {len(markets)} markets")
    
    if not file_tasks:
        return pl.DataFrame()
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(len(file_tasks), os.cpu_count() or 4)
    
    all_dfs = []
    successful_loads = 0
    
    # Use ThreadPoolExecutor for I/O-bound parallel loading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(load_single_file, task): task for task in file_tasks}
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            try:
                df = future.result()
                if not df.is_empty():
                    all_dfs.append(df)
                    successful_loads += 1
            except Exception as e:
                task = future_to_task[future]
                logger.error(f"Unexpected error loading {task[0]}: {e}")
    
    load_time = time.time() - start_time
    logger.info(f"Parallel loading completed: {successful_loads}/{len(file_tasks)} files loaded in {load_time:.2f} seconds")
    
    if not all_dfs:
        return pl.DataFrame()

    # Concatenate all DataFrames
    concat_start = time.time()
    combined_df = pl.concat(all_dfs)
    
    # Convert timestamp and sort
    combined_df = combined_df.with_columns(
        pl.from_epoch(pl.col('ts_utc'), time_unit='ms').alias('ts_utc')
    )
    combined_df = combined_df.sort('ts_utc')
    
    concat_time = time.time() - concat_start
    logger.info(f"Data concatenation and sorting completed in {concat_time:.2f} seconds")
    
    return combined_df

def parse_l2_chunk(chunk: pl.DataFrame) -> pl.DataFrame:
    """
    Parse L2 data for a single chunk.
    """
    if 'bid_levels' in chunk.columns:
        chunk = chunk.with_columns(
            pl.col('bid_levels').map_elements(
                parse_l2_levels_optimized, 
                return_dtype=pl.List(pl.List(pl.Float64))
            ).alias('bid_levels_parsed')
        )
    else:
        chunk = chunk.with_columns(
            pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias('bid_levels_parsed')
        )

    if 'ask_levels' in chunk.columns:
        chunk = chunk.with_columns(
            pl.col('ask_levels').map_elements(
                parse_l2_levels_optimized, 
                return_dtype=pl.List(pl.List(pl.Float64))
            ).alias('ask_levels_parsed')
        )
    else:
        chunk = chunk.with_columns(
            pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias('ask_levels_parsed')
        )
    
    return chunk

def parse_l2_parallel_chunks(df: pl.DataFrame, chunk_size: int = 50000, max_workers: int = None) -> pl.DataFrame:
    """
    Parse L2 data using parallel processing with chunked data.
    """
    parsing_start = time.time()
    
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 workers to avoid memory issues
    
    total_rows = df.shape[0]
    chunks = []
    
    # Split DataFrame into chunks
    for i in range(0, total_rows, chunk_size):
        chunk = df.slice(i, min(chunk_size, total_rows - i))
        chunks.append(chunk)
    
    logger.info(f"Processing L2 parsing in {len(chunks)} chunks using {max_workers} workers")
    
    parsed_chunks = []
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunk parsing tasks
        future_to_chunk = {executor.submit(parse_l2_chunk, chunk): i for i, chunk in enumerate(chunks)}
        
        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                parsed_chunk = future.result()
                parsed_chunks.append((chunk_idx, parsed_chunk))
            except Exception as e:
                logger.error(f"Error parsing chunk {chunk_idx}: {e}")
                # Add the original chunk without parsing as fallback
                parsed_chunks.append((chunk_idx, chunks[chunk_idx]))
    
    # Sort by original chunk order and concatenate
    parsed_chunks.sort(key=lambda x: x[0])
    result_chunks = [chunk for _, chunk in parsed_chunks]
    
    if result_chunks:
        result_df = pl.concat(result_chunks)
    else:
        result_df = df
    
    parsing_time = time.time() - parsing_start
    logger.info(f"Parallel L2 parsing completed in {parsing_time:.2f} seconds")
    
    return result_df

def parse_l2_vectorized(df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized L2 parsing using Polars lazy evaluation for better performance.
    """
    parsing_start = time.time()
    
    # Use lazy DataFrame for deferred execution and optimization
    lazy_df = df.lazy()
    
    # Parse L2 order book data with optimized operations
    if 'bid_levels' in df.columns:
        lazy_df = lazy_df.with_columns(
            pl.col('bid_levels').map_elements(
                parse_l2_levels_optimized, 
                return_dtype=pl.List(pl.List(pl.Float64))
            ).alias('bid_levels_parsed')
        )
    else:
        logger.warning("Column 'bid_levels' not found.")
        lazy_df = lazy_df.with_columns(
            pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias('bid_levels_parsed')
        )

    if 'ask_levels' in df.columns:
        lazy_df = lazy_df.with_columns(
            pl.col('ask_levels').map_elements(
                parse_l2_levels_optimized, 
                return_dtype=pl.List(pl.List(pl.Float64))
            ).alias('ask_levels_parsed')
        )
    else:
        logger.warning("Column 'ask_levels' not found.")
        lazy_df = lazy_df.with_columns(
            pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias('ask_levels_parsed')
        )
    
    # Materialize the lazy DataFrame
    result_df = lazy_df.collect()
    
    parsing_time = time.time() - parsing_start
    logger.info(f"L2 parsing completed in {parsing_time:.2f} seconds")
    
    return result_df

def load_and_preprocess_raw_data(base_path: str, markets: list, use_parallel_chunks: bool = True, 
                                  use_lazy: bool = True, chunk_size: int = 50000) -> pl.DataFrame:
    """
    Loads raw data and performs initial parsing for L2 order book with optimizations.
    
    Args:
        base_path: Base path to the data directory
        markets: List of market symbols to load
        use_parallel_chunks: Whether to use parallel chunked processing for L2 parsing
        use_lazy: Whether to use lazy evaluation for L2 parsing (only used if use_parallel_chunks=False)
        chunk_size: Size of chunks for parallel processing
    
    Returns:
        Combined DataFrame with parsed L2 data
    """
    total_start = time.time()
    logger.info(f"Loading and preprocessing raw data for markets: {markets} from {base_path}")
    
    df_combined = load_all_market_data(base_path=base_path, markets=markets)
    
    if df_combined.is_empty():
        logger.error("No data loaded. Exiting.")
        return pl.DataFrame()

    logger.info(f"Loaded combined data shape: {df_combined.shape}")

    # Parse L2 order book data with optimization
    logger.info("Parsing L2 order book data (bid_levels, ask_levels)...")
    
    if use_parallel_chunks and df_combined.shape[0] > chunk_size:
        # Use parallel chunked processing for large datasets
        df_combined = parse_l2_parallel_chunks(df_combined, chunk_size=chunk_size)
    elif use_lazy:
        # Use lazy evaluation for medium datasets
        df_combined = parse_l2_vectorized(df_combined)
    else:
        # Fallback to original parsing method for small datasets
        if 'bid_levels' in df_combined.columns:
            df_combined = df_combined.with_columns(
                pl.col('bid_levels').map_elements(parse_l2_levels, return_dtype=pl.List(pl.List(pl.Float64))).alias('bid_levels_parsed')
            )
        else:
            logger.warning("Column 'bid_levels' not found.")
            df_combined = df_combined.with_columns(pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias('bid_levels_parsed'))

        if 'ask_levels' in df_combined.columns:
            df_combined = df_combined.with_columns(
                pl.col('ask_levels').map_elements(parse_l2_levels, return_dtype=pl.List(pl.List(pl.Float64))).alias('ask_levels_parsed')
            )
        else:
            logger.warning("Column 'ask_levels' not found.")
            df_combined = df_combined.with_columns(pl.lit(None).alias('ask_levels_parsed'))
    
    total_time = time.time() - total_start
    logger.info(f"Total data loading and preprocessing completed in {total_time:.2f} seconds")
    
    return df_combined

if __name__ == "__main__":
    # Example usage
    df = load_and_preprocess_raw_data(
        base_path='crypto_tick',
        markets=['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT']
    )
    if not df.is_empty():
        print(f"Loaded data with shape: {df.shape}")
        print(df.head())
        print("\nData loaded for markets:", df['market'].unique().to_list())
        print("Date range:", df['ts_utc'].min(), "to", df['ts_utc'].max())
        print("Parsed columns check:", 'bid_levels_parsed' in df.columns)
    else:
        print("No data loaded.")
