"""
Configuration file for TCN-based quantitative trading strategy
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataConfig:
    """Data processing configuration"""
    raw_data_path: str = "market=SOL_USDC" # Corrected path relative to project root
    processed_data_path: str = "tcn_strategy/data/processed"
    sequence_length: int = 120  # 2 minutes at 5-second intervals
    prediction_horizons: List[int] = None  # [1, 5, 10, 30] steps ahead
    train_split: float = 0.6  # 60% for training
    val_split: float = 0.2    # 20% for validation
    test_split: float = 0.2   # 20% for testing
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 5, 10, 30]

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    price_features: bool = True
    volume_features: bool = True
    spread_features: bool = True
    microstructure_features: bool = True
    technical_indicators: bool = True
    regime_features: bool = True
    multi_timeframe_windows: List[int] = None  # [6, 12, 36, 72] ticks
    
    def __post_init__(self):
        if self.multi_timeframe_windows is None:
            self.multi_timeframe_windows = [6, 12, 36, 72]  # 30s, 1m, 3m, 6m

@dataclass
class ModelConfig:
    """TCN model configuration"""
    input_channels: int = None  # Will be set based on features
    hidden_channels: int = 64
    num_layers: int = 5
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.4
    layer_norm: bool = True
    residual_connections: bool = True
    attention: bool = False  # Disable attention to reduce complexity
    ensemble_size: int = 2   # Reduce ensemble size

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4  # Stronger regularization
    num_epochs: int = 50       # Fewer epochs
    early_stopping_patience: int = 10
    lr_scheduler: str = "cosine"  # "cosine" or "plateau"
    gradient_clip: float = 1.0
    device: str = "cuda"

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.05  # 5% per position
    max_total_exposure: float = 0.3   # 30% total exposure
    transaction_cost_bps: float = 10   # 10 bps total cost
    confidence_threshold: float = 0.4
    stop_loss_pct: float = 0.02       # 2% stop loss
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    rebalance_frequency: str = "5min"
    
@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_daily_loss: float = 0.02     # 2% daily loss limit
    max_total_drawdown: float = 0.15 # 15% total drawdown limit
    position_sizing_method: str = "kelly"  # "fixed", "kelly", "volatility"
    lookback_volatility: int = 288   # 24 hours for volatility calc
    var_confidence: float = 0.05     # 5% VaR
    
class Config:
    """Main configuration class"""
    def __init__(self):
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.backtest = BacktestConfig()
        self.risk = RiskConfig()
        
        # Create directories
        os.makedirs("tcn_strategy/data/processed", exist_ok=True)
        os.makedirs("tcn_strategy/models", exist_ok=True)
        os.makedirs("tcn_strategy/results", exist_ok=True)
        os.makedirs("tcn_strategy/logs", exist_ok=True)
