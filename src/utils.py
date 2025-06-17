# src/utils.py
import logging
import os
import pandas as pd

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logger(logger_name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(os.path.join(LOGS_DIR, log_file), mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)
    # Avoid duplicate logs if called multiple times
    l.propagate = False 


# Example usage:
# from utils import setup_logger
# logger = logging.getLogger("my_module_logger") # Get a logger instance
# setup_logger("my_module_logger", "my_module.log") # Configure it
# logger.info("This is a test log from my_module.")


def save_dataframe_to_csv(df: pd.DataFrame, path: str, filename: str):
    """Saves a DataFrame to a CSV file in the specified path."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    try:
        df.to_csv(filepath, index=False)
        logging.info(f"DataFrame successfully saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to {filepath}: {e}")

if __name__ == '__main__':
    # Basic test for logger setup
    main_logger = logging.getLogger("utils_main")
    setup_logger("utils_main", "utils_test.log")
    main_logger.info("Utils logger test: This should go to console and utils_test.log")
    main_logger.warning("This is a test warning.")
    
    # Test CSV saving
    # sample_df = pd.DataFrame({'colA': [1, 2], 'colB': ['X', 'Y']})
    # save_dataframe_to_csv(sample_df, "data_output", "sample_data.csv")
    # main_logger.info("Tested save_dataframe_to_csv. Check 'data_output' directory.")
    pass
