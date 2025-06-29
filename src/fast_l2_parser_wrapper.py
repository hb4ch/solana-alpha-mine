"""
Fast L2 Parser Integration Wrapper

This module provides a seamless integration between the C++ fast_l2_parser extension
and the existing Python data loading pipeline.
"""

import sys
import os
import logging
from typing import List, Optional, Union, Tuple
import time

# Add the C++ extension path
cpp_extension_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cpp_extensions', 'l2_parser')
if cpp_extension_path not in sys.path:
    sys.path.insert(0, cpp_extension_path)

logger = logging.getLogger(__name__)

# Try to import the fast C++ parser
try:
    import fast_l2_parser
    CPP_PARSER_AVAILABLE = True
    logger.info("C++ fast_l2_parser successfully loaded")
except ImportError as e:
    CPP_PARSER_AVAILABLE = False
    logger.warning(f"C++ fast_l2_parser not available: {e}")
    logger.warning("Falling back to Python parsing")

# Fallback to original Python parsing
def parse_l2_levels_python_fallback(level_str: str) -> list:
    """Python fallback parsing function."""
    if level_str is None or level_str == "":
        return []
    
    try:
        # First try JSON parsing (fastest)
        if level_str.startswith('['):
            import json
            return json.loads(level_str)
        else:
            # Fallback to ast.literal_eval for Python-specific formats
            import ast
            return ast.literal_eval(level_str)
    except (ValueError, SyntaxError, json.JSONDecodeError):
        # Try ast.literal_eval as fallback
        try:
            import ast
            return ast.literal_eval(level_str)
        except (ValueError, SyntaxError):
            logger.warning(f"Could not parse L2 level string: {level_str[:100]}...")
            return []

def parse_l2_levels_fast(level_str: str) -> list:
    """
    High-performance L2 parsing using C++ extension with Python fallback.
    
    Args:
        level_str: String representation of L2 levels like "[[684.5, 0.123], [684.6, 0.456]]"
    
    Returns:
        List of [price, quantity] pairs
    """
    if level_str is None or level_str == "":
        return []
    
    if CPP_PARSER_AVAILABLE:
        try:
            # Use C++ parser for maximum speed
            prices, quantities = fast_l2_parser.parse_l2_to_lists(level_str)
            # Convert back to list of pairs format expected by existing code
            return [[p, q] for p, q in zip(prices, quantities)]
        except Exception as e:
            logger.warning(f"C++ parser failed, falling back to Python: {e}")
            return parse_l2_levels_python_fallback(level_str)
    else:
        # Use Python fallback
        return parse_l2_levels_python_fallback(level_str)

def parse_l2_batch_fast(level_strings: List[str]) -> List[list]:
    """
    Batch parsing of L2 strings using C++ extension.
    
    Args:
        level_strings: List of L2 string representations
    
    Returns:
        List of parsed L2 levels (each as list of [price, quantity] pairs)
    """
    if not level_strings:
        return []
    
    if CPP_PARSER_AVAILABLE:
        try:
            # Use C++ batch parser
            results = fast_l2_parser.parse_l2_batch(level_strings)
            
            # Convert results to expected format
            parsed_results = []
            for result in results:
                if result is not None:
                    prices, quantities = result
                    parsed_results.append([[p, q] for p, q in zip(prices, quantities)])
                else:
                    parsed_results.append([])
            
            return parsed_results
        except Exception as e:
            logger.warning(f"C++ batch parser failed, falling back to Python: {e}")
            return [parse_l2_levels_python_fallback(s) for s in level_strings]
    else:
        # Use Python fallback
        return [parse_l2_levels_python_fallback(s) for s in level_strings]

def get_parser_stats() -> dict:
    """Get parsing statistics from C++ extension."""
    if CPP_PARSER_AVAILABLE:
        try:
            stats = fast_l2_parser.get_stats()
            return {
                'cpp_parser_available': True,
                'total_parsed': stats.total_parsed,
                'successful_parses': stats.successful_parses,
                'failed_parses': stats.failed_parses,
                'average_levels_per_parse': stats.average_levels_per_parse,
                'success_rate': stats.successful_parses / max(stats.total_parsed, 1) * 100
            }
        except Exception as e:
            logger.error(f"Error getting parser stats: {e}")
            return {'cpp_parser_available': True, 'error': str(e)}
    else:
        return {'cpp_parser_available': False}

def reset_parser_stats():
    """Reset parsing statistics."""
    if CPP_PARSER_AVAILABLE:
        try:
            fast_l2_parser.reset_stats()
        except Exception as e:
            logger.error(f"Error resetting parser stats: {e}")

def benchmark_parsing_methods(test_strings: List[str], num_iterations: int = 1) -> dict:
    """
    Benchmark C++ vs Python parsing performance.
    
    Args:
        test_strings: List of L2 strings to parse
        num_iterations: Number of times to repeat the test
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'test_strings_count': len(test_strings),
        'iterations': num_iterations,
        'cpp_available': CPP_PARSER_AVAILABLE
    }
    
    # Test Python parsing
    python_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        python_results = [parse_l2_levels_python_fallback(s) for s in test_strings]
        python_times.append(time.time() - start_time)
    
    results['python_avg_time'] = sum(python_times) / len(python_times)
    results['python_total_levels'] = sum(len(r) for r in python_results)
    
    # Test C++ parsing if available
    if CPP_PARSER_AVAILABLE:
        cpp_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            cpp_results = [parse_l2_levels_fast(s) for s in test_strings]
            cpp_times.append(time.time() - start_time)
        
        results['cpp_avg_time'] = sum(cpp_times) / len(cpp_times)
        results['cpp_total_levels'] = sum(len(r) for r in cpp_results)
        results['speedup'] = results['python_avg_time'] / results['cpp_avg_time']
        
        # Verify results are the same
        results['results_match'] = python_results == cpp_results
    
    return results

def validate_installation() -> bool:
    """
    Validate that the C++ extension is properly installed and working.
    
    Returns:
        True if C++ extension is working correctly
    """
    if not CPP_PARSER_AVAILABLE:
        logger.error("C++ extension not available")
        return False
    
    # Test basic functionality
    test_cases = [
        "[]",
        "[[684.5, 0.123]]",
        "[[684.5, 0.123], [684.6, 0.456], [684.7, 0.789]]",
        "[[1234.56, 10.5], [1234.57, 20.3]]"
    ]
    
    try:
        for test_case in test_cases:
            result = parse_l2_levels_fast(test_case)
            logger.debug(f"Test case '{test_case}' -> {result}")
        
        # Test batch parsing
        batch_results = parse_l2_batch_fast(test_cases)
        logger.debug(f"Batch parsing results: {batch_results}")
        
        # Test statistics
        stats = get_parser_stats()
        logger.info(f"Parser stats: {stats}")
        
        logger.info("C++ extension validation successful")
        return True
        
    except Exception as e:
        logger.error(f"C++ extension validation failed: {e}")
        return False

# Export the main parsing function
parse_l2_levels_optimized = parse_l2_levels_fast

if __name__ == "__main__":
    # Quick test of the wrapper
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Fast L2 Parser Wrapper")
    print("=" * 40)
    
    # Validate installation
    is_valid = validate_installation()
    print(f"C++ Extension Valid: {is_valid}")
    
    if is_valid:
        # Run a simple benchmark
        test_strings = [
            "[[684.5, 0.123], [684.6, 0.456]]",
            "[[1000.0, 1.0], [1001.0, 2.0], [1002.0, 3.0]]",
            "[[100.5, 0.5], [100.6, 0.6], [100.7, 0.7], [100.8, 0.8]]"
        ] * 100  # Repeat for meaningful timing
        
        benchmark_results = benchmark_parsing_methods(test_strings, num_iterations=3)
        
        print("\nBenchmark Results:")
        print(f"Test strings: {benchmark_results['test_strings_count']}")
        print(f"Python avg time: {benchmark_results['python_avg_time']:.4f}s")
        if benchmark_results['cpp_available']:
            print(f"C++ avg time: {benchmark_results['cpp_avg_time']:.4f}s")
            print(f"Speedup: {benchmark_results['speedup']:.2f}x")
            print(f"Results match: {benchmark_results['results_match']}")
        
        print(f"\nParser Stats:")
        stats = get_parser_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
