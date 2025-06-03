"""
Separate training and backtesting pipelines for TCN-based quantitative trading strategy
"""

import os
import sys
import warnings
import traceback
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from config import Config
from data_pipeline import DataProcessor
from trainer import TCNTrainer
from backtest import TCNBacktester
from model import create_model

warnings.filterwarnings('ignore')

def check_environment():
    """Check if all required dependencies are available"""
    required_packages = [
        'torch', 'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'seaborn', 'scipy'
    ]
    
    optional_packages = ['talib']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"Missing required packages: {missing_required}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"Optional packages not found (fallback implementations will be used): {missing_optional}")
    
    print("✓ All required packages are available")
    return True

def print_system_info():
    """Print system information"""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

def run_training_only(config: Config, use_ensemble: bool = True) -> dict:
    """Run only the training pipeline"""
    print("🚀 Starting TCN Model Training Pipeline")
    print("=" * 50)
    
    results = {
        'data_processing': None,
        'training': None,
        'config': config
    }
    
    try:
        # Step 1: Data Processing
        print("\n📊 Step 1: Data Processing and Feature Engineering")
        print("-" * 50)
        
        processor = DataProcessor(config)
        X_train, X_val, X_test, targets, df_processed = processor.process_data()
        
        print(f"✓ Data processing completed successfully")
        print(f"  - Training samples: {len(X_train):,}")
        print(f"  - Validation samples: {len(X_val):,}")
        print(f"  - Test samples: {len(X_test):,}")
        print(f"  - Features: {X_train.shape[2]}")
        print(f"  - Sequence length: {X_train.shape[1]}")
        
        results['data_processing'] = {
            'train_samples': len(X_train),
            'val_samples': len(X_val), 
            'test_samples': len(X_test),
            'features': X_train.shape[2],
            'sequence_length': X_train.shape[1]
        }
        
        # Step 2: Model Training
        print("\n🤖 Step 2: TCN Model Training")
        print("-" * 50)
        
        trainer = TCNTrainer(config)
        trainer.config.model.input_channels = X_train.shape[2]  # Update based on actual features
        
        training_results = trainer.train(use_ensemble=use_ensemble)
        
        print(f"✓ Model training completed successfully")
        print("Training Results:")
        for horizon, metrics in training_results.items():
            if isinstance(metrics, dict):
                print(f"  Horizon {horizon.split('_')[1]}:")
                print(f"    - Return RMSE: {metrics.get('return_rmse', 0):.6f}")
                print(f"    - Return R²: {metrics.get('return_r2', 0):.3f}")
                print(f"    - Direction Accuracy: {metrics.get('direction_accuracy', 0):.3f}")
        
        results['training'] = training_results
        
        print("\n✅ Training completed successfully!")
        print("Model saved to: tcn_strategy/models/best_model.pth")
        print("Results saved to: tcn_strategy/results/")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error occurred during training:")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return results

def run_backtest_only(config: Config) -> dict:
    """Run only the backtesting pipeline using pre-trained model"""
    print("📈 Starting TCN Strategy Backtesting")
    print("=" * 40)
    
    results = {
        'backtest': None,
        'config': config
    }
    
    try:
        # Check if trained model exists
        model_path = "tcn_strategy/models/best_model.pth"
        if not os.path.exists(model_path):
            print("❌ No trained model found. Please run training first.")
            print(f"Expected model at: {model_path}")
            return results
        
        # Load processed data (should be cached)
        data_path = "tcn_strategy/data/processed/processed_data.parquet"
        if not os.path.exists(data_path):
            print("❌ No processed data found. Please run training first.")
            print(f"Expected data at: {data_path}")
            return results
        
        print("✓ Found existing model and processed data")
        
        # Load data
        print("\n📊 Loading processed data...")
        processor = DataProcessor(config)
        X_train, X_val, X_test, targets, df_processed = processor.process_data()
        
        # Load and prepare model
        print("\n🤖 Loading trained model...")
        
        # First load checkpoint to get the saved config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract config from checkpoint if available
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # Update current config with saved model parameters
            if 'model' in saved_config:
                saved_model_config = saved_config['model']
                # Copy attributes from saved model config
                for attr in dir(saved_model_config):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(saved_model_config, attr)
                            if hasattr(config.model, attr):
                                setattr(config.model, attr, value)
                        except AttributeError:
                            continue
        
        trainer = TCNTrainer(config)
        trainer.config.model.input_channels = X_train.shape[2]
        trainer.build_model(use_ensemble=False)  # Build architecture first
        
        # Now load the state dict
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully")
        
        # Remove the redundant load_best_model call since we loaded manually
        # if trainer.load_best_model():
    
        
        # Prepare test data for backtesting
        test_data = (X_test, targets)
        
        # Run backtest
        print("\n📈 Running strategy backtesting...")
        backtester = TCNBacktester(config)
        backtest_results = backtester.run_backtest(trainer.model, test_data, df_processed)
        
        print(f"✓ Backtesting completed successfully")
        print("Backtest Results:")
        strategy_metrics = backtest_results['strategy_metrics']
        print(f"  - Total Return: {strategy_metrics.get('total_return', 0)*100:.2f}%")
        print(f"  - Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  - Max Drawdown: {strategy_metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  - Win Rate: {backtest_results['win_rate']*100:.1f}%")
        print(f"  - Total Trades: {backtest_results['num_trades']}")
        
        results['backtest'] = backtest_results
        
        # Generate reports and visualizations
        print("\n📊 Generating reports and visualizations...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate backtest plot
        plot_path = f"tcn_strategy/results/backtest_results_{timestamp}.png"
        backtester.plot_backtest_results(backtest_results, save_path=plot_path)
        print(f"✓ Backtest visualization saved to {plot_path}")
        
        # Generate comprehensive report
        report = backtester.generate_report(backtest_results)
        report_path = f"tcn_strategy/results/backtest_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Backtest report saved to {report_path}")
        
        print("\n✅ Backtesting completed successfully!")
        print("Check the tcn_strategy/results/ directory for detailed outputs.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error occurred during backtesting:")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return results

def main():
    """Main execution function"""
    print("🎯 TCN-Based Quantitative Trading Strategy")
    print("==========================================")
    
    # Check environment
    if not check_environment():
        return
    
    # Print system info
    print_system_info()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main_separate.py train [--quick]     # Train model")
        print("  python main_separate.py backtest            # Run backtest only")
        return
    
    command = sys.argv[1]
    use_quick = '--quick' in sys.argv
    
    config = Config()
    
    if use_quick:
        # Reduce configuration for quick testing
        config.training.num_epochs = 5
        config.training.batch_size = 32
        config.model.num_layers = 4
        config.model.hidden_channels = 64
        config.model.ensemble_size = 1
        config.data.sequence_length = 60
        print("\n🧪 Using Quick Test Configuration")
        print(f"  - Epochs: {config.training.num_epochs}")
        print(f"  - Batch size: {config.training.batch_size}")
        print(f"  - Model layers: {config.model.num_layers}")
        print(f"  - Hidden channels: {config.model.hidden_channels}")
    
    if command == 'train':
        results = run_training_only(config, use_ensemble=not use_quick)
    elif command == 'backtest':
        results = run_backtest_only(config)
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: train, backtest")
        return
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
