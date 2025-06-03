"""
Professional data pipeline for TCN-based quantitative trading strategy
"""

import pandas as pd
import numpy as np
import glob
import warnings
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
# Try to import talib, use fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using fallback implementations for technical indicators.")

from config import Config

warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Professional data processing pipeline with advanced feature engineering
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_names = []
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load and concatenate all parquet files"""
        print("Loading raw tick data...")
        data_files = glob.glob(f"{self.config.data.raw_data_path}/*/l1.parquet")
        data_files = sorted(data_files)
        
        df_list = []
        for file in data_files:
            df = pd.read_parquet(file)
            date_str = file.split('/')[1].replace('date=', '')
            df['date'] = date_str
            df_list.append(df)
        
        df = pd.concat(df_list, ignore_index=True)
        df['ts_utc'] = pd.to_datetime(df['ts_utc'])
        df = df.sort_values('ts_utc').reset_index(drop=True)
        
        print(f"Loaded {len(df):,} ticks across {len(data_files)} days")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate tick data"""
        print("Cleaning data...")
        initial_count = len(df)
        
        # Remove invalid prices and sizes
        df = df[(df['bid_px'] > 0) & (df['ask_px'] > 0)]
        df = df[(df['bid_sz'] > 0) & (df['ask_sz'] > 0)]
        df = df[df['bid_px'] <= df['ask_px']]  # Sanity check
        
        # Remove extreme outliers (more than 5 standard deviations)
        for col in ['bid_px', 'ask_px']:
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[z_scores < 5]
        
        # Remove duplicate timestamps (keep last)
        df = df.drop_duplicates(subset=['ts_utc'], keep='last')
        
        # Ensure chronological order
        df = df.sort_values('ts_utc').reset_index(drop=True)
        
        print(f"Removed {initial_count - len(df):,} invalid/duplicate ticks")
        return df
    
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price and volume features"""
        print("Calculating basic features...")
        
        # Price features
        df['mid_price'] = (df['bid_px'] + df['ask_px']) / 2
        df['spread'] = df['ask_px'] - df['bid_px']
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
        df['log_price'] = np.log(df['mid_price'])
        
        # Returns at multiple horizons
        for horizon in [1, 5, 10, 30]:
            df[f'return_{horizon}'] = df['mid_price'].pct_change(horizon)
            df[f'log_return_{horizon}'] = df['log_price'].diff(horizon)
        
        # Volume features
        df['total_volume'] = df['bid_sz'] + df['ask_sz']
        df['volume_imbalance'] = (df['bid_sz'] - df['ask_sz']) / df['total_volume']
        df['order_flow_imbalance'] = df['bid_sz'] / df['ask_sz']
        
        return df
    
    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced microstructure features"""
        if not self.config.features.microstructure_features:
            return df
        
        print("Calculating microstructure features...")
        
        # Tick intensity (arrival rate)
        df['time_diff'] = df['ts_utc'].diff().dt.total_seconds()
        df['tick_intensity'] = 1 / df['time_diff'].rolling(window=10).mean()
        
        # Price impact measures
        df['price_impact'] = df['mid_price'].diff() / df['total_volume'].shift(1)
        df['effective_spread'] = 2 * np.abs(df['mid_price'].diff())
        
        # Quote slope (how fast quotes are moving)
        df['quote_slope'] = df['mid_price'].diff() / df['time_diff']
        
        # Trade sign classification (Lee-Ready algorithm approximation)
        df['trade_sign'] = np.where(df['mid_price'].diff() > 0, 1, 
                          np.where(df['mid_price'].diff() < 0, -1, 0))
        
        # Order flow toxicity (Kyle's lambda)
        window = 20
        df['price_change'] = df['mid_price'].diff()
        df['signed_volume'] = df['trade_sign'] * df['total_volume']
        df['kyle_lambda'] = (df['price_change'].rolling(window).cov(df['signed_volume']) / 
                           df['signed_volume'].rolling(window).var())
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using TA-Lib or fallback implementations"""
        if not self.config.features.technical_indicators:
            return df
        
        print("Calculating technical indicators...")
        prices = df['mid_price'].values
        high = df['ask_px'].values
        low = df['bid_px'].values
        
        if TALIB_AVAILABLE:
            # Use TA-Lib if available
            # Moving averages
            for window in [10, 20, 50]:
                df[f'sma_{window}'] = talib.SMA(prices, timeperiod=window)
                df[f'ema_{window}'] = talib.EMA(prices, timeperiod=window)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(prices, timeperiod=20)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI
            df['rsi'] = talib.RSI(prices, timeperiod=14)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(prices)
            
            # Stochastic oscillator
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, prices)
            
            # Williams %R
            df['williams_r'] = talib.WILLR(high, low, prices)
            
            # Average True Range (volatility)
            df['atr'] = talib.ATR(high, low, prices)
        else:
            # Use pandas fallback implementations
            # Moving averages
            for window in [10, 20, 50]:
                df[f'sma_{window}'] = df['mid_price'].rolling(window).mean()
                df[f'ema_{window}'] = df['mid_price'].ewm(span=window).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['mid_price'].rolling(20).mean()
            bb_std = df['mid_price'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['mid_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI (simplified)
            delta = df['mid_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['mid_price'].ewm(span=12).mean()
            ema_26 = df['mid_price'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Stochastic oscillator (simplified)
            lowest_low = df['bid_px'].rolling(window=14).min()
            highest_high = df['ask_px'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['mid_price'] - lowest_low) / (highest_high - lowest_low))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R (simplified)
            df['williams_r'] = -100 * ((highest_high - df['mid_price']) / (highest_high - lowest_low))
            
            # Average True Range (simplified)
            high_low = df['ask_px'] - df['bid_px']
            high_close = np.abs(df['ask_px'] - df['mid_price'].shift(1))
            low_close = np.abs(df['bid_px'] - df['mid_price'].shift(1))
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
        
        return df
    
    def calculate_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features across multiple timeframes"""
        print("Calculating multi-timeframe features...")
        
        for window in self.config.features.multi_timeframe_windows:
            # Price statistics
            df[f'price_mean_{window}'] = df['mid_price'].rolling(window).mean()
            df[f'price_std_{window}'] = df['mid_price'].rolling(window).std()
            df[f'price_skew_{window}'] = df['mid_price'].rolling(window).skew()
            df[f'price_kurt_{window}'] = df['mid_price'].rolling(window).kurt()
            
            # Return statistics
            df[f'return_mean_{window}'] = df['return_1'].rolling(window).mean()
            df[f'return_std_{window}'] = df['return_1'].rolling(window).std()
            df[f'return_skew_{window}'] = df['return_1'].rolling(window).skew()
            
            # Volume statistics
            df[f'volume_mean_{window}'] = df['total_volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['total_volume'].rolling(window).std()
            
            # Spread statistics
            df[f'spread_mean_{window}'] = df['spread_bps'].rolling(window).mean()
            df[f'spread_std_{window}'] = df['spread_bps'].rolling(window).std()
            
            # High/Low over window
            df[f'high_{window}'] = df['ask_px'].rolling(window).max()
            df[f'low_{window}'] = df['bid_px'].rolling(window).min()
            df[f'range_{window}'] = df[f'high_{window}'] - df[f'low_{window}']
        
        return df
    
    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime indicators"""
        if not self.config.features.regime_features:
            return df
        
        print("Calculating regime features...")
        
        # Volatility regime
        df['realized_vol'] = df['return_1'].rolling(window=60).std() * np.sqrt(17280)  # Annualized
        df['vol_regime'] = (df['realized_vol'] > df['realized_vol'].rolling(window=288).quantile(0.7)).astype(int)
        
        # Trend regime
        df['trend_strength'] = np.abs(df['return_1'].rolling(window=60).sum())
        df['trend_regime'] = (df['trend_strength'] > df['trend_strength'].rolling(window=288).quantile(0.7)).astype(int)
        
        # Market stress indicator
        df['stress_indicator'] = (df['spread_bps'] > df['spread_bps'].rolling(window=60).quantile(0.9)).astype(int)
        
        # Momentum indicators
        for window in [20, 60, 120]:
            df[f'momentum_{window}'] = df['return_1'].rolling(window).sum()
            df[f'momentum_rank_{window}'] = df[f'momentum_{window}'].rolling(window*2).rank(pct=True)
        
        return df
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for different prediction horizons"""
        print("Creating target variables...")
        
        for horizon in self.config.data.prediction_horizons:
            # Regression targets (future returns)
            df[f'target_return_{horizon}'] = df['return_1'].shift(-horizon)
            
            # Classification targets (direction)
            df[f'target_direction_{horizon}'] = (df[f'target_return_{horizon}'] > 0).astype(int)
            
            # Multi-class targets (strong up/up/flat/down/strong down)
            thresholds = df[f'target_return_{horizon}'].quantile([0.2, 0.4, 0.6, 0.8])
            # Handle duplicate edges by adding small offsets
            unique_thresholds = thresholds.drop_duplicates().tolist()
            if len(unique_thresholds) < 4:
                # If we have too few unique thresholds, create manual bins
                returns_std = df[f'target_return_{horizon}'].std()
                unique_thresholds = [-returns_std, -returns_std/2, returns_std/2, returns_std]
            
            try:
                df[f'target_multiclass_{horizon}'] = pd.cut(
                    df[f'target_return_{horizon}'], 
                    bins=[-np.inf] + unique_thresholds + [np.inf], 
                    labels=list(range(len(unique_thresholds) + 1)),
                    duplicates='drop'
                ).astype(int)
            except ValueError:
                # Fallback: use simple binary classification
                df[f'target_multiclass_{horizon}'] = df[f'target_direction_{horizon}']
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers using IQR method"""
        print("Removing outliers...")
        initial_count = len(df)
        
        # Select numeric columns for outlier detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith('target_') or col in ['ts_utc', 'slot']:
                continue
                
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"Removed {initial_count - len(df):,} outlier records")
        return df
    
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Create sequences for TCN training"""
        print("Preparing sequences...")
        
        # Select feature columns (exclude non-feature columns)
        feature_cols = [col for col in df.columns if not col.startswith(('ts_utc', 'date', 'source', 'raw_hash', 'slot', 'target_'))]
        feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64]]
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + [col for col in df.columns if col.startswith('target_')]].dropna()
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(df_clean[feature_cols])
        
        # Store feature names
        self.feature_names = feature_cols
        
        sequences = []
        targets = {f'target_return_{h}': [] for h in self.config.data.prediction_horizons}
        targets.update({f'target_direction_{h}': [] for h in self.config.data.prediction_horizons})
        
        sequence_length = self.config.data.sequence_length
        
        for i in range(len(X_scaled) - sequence_length - max(self.config.data.prediction_horizons)):
            # Create sequence
            seq = X_scaled[i:i + sequence_length]
            sequences.append(seq)
            
            # Create targets for each horizon
            target_idx = i + sequence_length
            for horizon in self.config.data.prediction_horizons:
                targets[f'target_return_{horizon}'].append(
                    df_clean[f'target_return_{horizon}'].iloc[target_idx]
                )
                targets[f'target_direction_{horizon}'].append(
                    df_clean[f'target_direction_{horizon}'].iloc[target_idx]
                )
        
        X = np.array(sequences)
        targets = {k: np.array(v) for k, v in targets.items()}
        
        print(f"Created {len(X):,} sequences of length {sequence_length}")
        print(f"Feature dimension: {X.shape[2]}")
        
        return X, targets
    
    def train_val_test_split(self, X: np.ndarray, targets: Dict[str, np.ndarray]) -> Tuple:
        """Split data with temporal ordering (no shuffle)"""
        print("Splitting data...")
        
        n_samples = len(X)
        train_end = int(n_samples * self.config.data.train_split)
        val_end = int(n_samples * (self.config.data.train_split + self.config.data.val_split))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        targets_split = {}
        for key, target_array in targets.items():
            targets_split[f'{key}_train'] = target_array[:train_end]
            targets_split[f'{key}_val'] = target_array[train_end:val_end]
            targets_split[f'{key}_test'] = target_array[val_end:]
        
        print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        return X_train, X_val, X_test, targets_split
    
    def process_data(self) -> Tuple:
        """Main data processing pipeline"""
        print("Starting data processing pipeline...")
        
        # Load and clean data
        df = self.load_raw_data()
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.calculate_basic_features(df)
        df = self.calculate_microstructure_features(df)
        df = self.calculate_technical_indicators(df)
        df = self.calculate_multi_timeframe_features(df)
        df = self.calculate_regime_features(df)
        
        # Create targets
        df = self.create_targets(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Save processed data
        processed_file = f"{self.config.data.processed_data_path}/processed_data.parquet"
        df.to_parquet(processed_file)
        print(f"Saved processed data to {processed_file}")
        
        # Create sequences
        X, targets = self.prepare_sequences(df)
        
        # Split data
        X_train, X_val, X_test, targets_split = self.train_val_test_split(X, targets)
        
        # Update model config with input channels
        self.config.model.input_channels = X.shape[2]
        
        print("Data processing completed!")
        
        return X_train, X_val, X_test, targets_split, df

if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config)
    X_train, X_val, X_test, targets, df = processor.process_data()
