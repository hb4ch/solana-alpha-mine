"""
Main execution script for TCN-based quantitative trading strategy
"""

import os
import sys
import warnings
import traceback
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from config import Config, DataConfig, ModelConfig, FeatureConfig, TrainingConfig, BacktestConfig, RiskConfig
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

def run_full_pipeline(config: Config, use_ensemble: bool = True) -> dict:
    """
    Run the complete TCN trading strategy pipeline
    """
    print("🚀 Starting TCN Quantitative Trading Strategy Pipeline")
    print("=" * 60)
    
    results = {
        'data_processing': None,
        'training': None,
        'backtest': None,
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
        
        # Step 3: Backtesting
        print("\n📈 Step 3: Strategy Backtesting")
        print("-" * 50)
        
        # Load the best trained model
        trainer.load_best_model()
        
        # Prepare test data for backtesting
        test_data = (X_test, targets)
        
        # Run backtest
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
        
        # Step 4: Generate Reports and Visualizations
        print("\n📊 Step 4: Generating Reports and Visualizations")
        print("-" * 50)
        
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
        
        # Save combined results
        results_summary = {
            'timestamp': timestamp,
            'data_processing': results['data_processing'],
            'training_best_metrics': {
                horizon: {
                    'return_rmse': metrics.get('return_rmse', 0),
                    'direction_accuracy': metrics.get('direction_accuracy', 0)
                }
                for horizon, metrics in training_results.items()
                if isinstance(metrics, dict)
            },
            'backtest_summary': {
                'total_return': strategy_metrics.get('total_return', 0),
                'sharpe_ratio': strategy_metrics.get('sharpe_ratio', 0),
                'max_drawdown': strategy_metrics.get('max_drawdown', 0),
                'win_rate': backtest_results['win_rate'],
                'num_trades': backtest_results['num_trades'],
                'final_value': backtest_results['final_portfolio_value']
            }
        }
        
        # Print final summary
        print("\n🎯 FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"Initial Capital: ${config.backtest.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${backtest_results['final_portfolio_value']:,.2f}")
        print(f"Total Return: {strategy_metrics.get('total_return', 0)*100:.2f}%")
        print(f"Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Maximum Drawdown: {strategy_metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"Win Rate: {backtest_results['win_rate']*100:.1f}%")
        print(f"Total Trades: {backtest_results['num_trades']}")
        print()
        print("📁 Results saved to:")
        print(f"  - Plots: tcn_strategy/results/")
        print(f"  - Models: tcn_strategy/models/")
        print(f"  - Reports: {report_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error occurred during pipeline execution:")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return results

# The run_training_only function was missing, let's define it properly.
# It seems the duplicated block was an attempt to define run_training_only or was part of it.
# Based on the structure, run_training_only should be defined here.
# The content of the deleted block seems to be a mix of data processing and training steps,
# similar to what run_full_pipeline does but perhaps intended to stop after training.

def run_training_only(config: Config, use_ensemble: bool = True) -> dict:
    """Run only the training part of the pipeline"""
    print("🚀 Starting TCN Training Pipeline")
    print("=" * 60)

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
        print(f"  - Test samples: {len(X_test):,}") # Test samples are prepared but not used in training-only
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
        print("Training Results (Test Set Evaluation):")
        for horizon, metrics in training_results.items():
            if isinstance(metrics, dict):
                print(f"  Horizon {horizon.split('_')[1]}:")
                print(f"    - Return RMSE: {metrics.get('return_rmse', 0):.6f}")
                print(f"    - Return R²: {metrics.get('return_r2', 0):.3f}")
                print(f"    - Direction Accuracy: {metrics.get('direction_accuracy', 0):.3f}")
        
        results['training'] = training_results
        
        print("\n📁 Results saved to:")
        print(f"  - Plots: tcn_strategy/results/")
        print(f"  - Models: tcn_strategy/models/")

        return results

    except Exception as e:
        print(f"\n❌ Error occurred during training pipeline execution:")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return results

def run_backtest_only(config: Config) -> dict:
    """Run only the backtesting part of the pipeline using a pre-trained model"""
    print("🚀 Starting TCN Backtesting Pipeline")
    print("=" * 60)

    results = {
        'data_processing': None,
        'backtest': None,
        'config': config
    }

    try:
        # Step 1: Data Processing (to get test data and df_processed)
        print("\n📊 Step 1: Data Processing for Backtest")
        print("-" * 50)
        
        processor = DataProcessor(config)
        # We only need X_test, targets for test, and df_processed for backtesting
        _, _, X_test, targets, df_processed = processor.process_data()
        
        print(f"✓ Data processing for backtest completed successfully")
        print(f"  - Test samples: {len(X_test):,}")
        if X_test.size > 0:
            print(f"  - Features: {X_test.shape[2]}")
            print(f"  - Sequence length: {X_test.shape[1]}")
        else:
            print("  - No test data available after processing.")
            results['data_processing'] = {'test_samples': 0}
            return results


        results['data_processing'] = {
            'test_samples': len(X_test),
            'features': X_test.shape[2] if X_test.size > 0 else 0,
            'sequence_length': X_test.shape[1] if X_test.size > 0 else 0
        }

        # Step 2: Load Pre-trained Model
        print("\n🤖 Step 2: Loading Pre-trained TCN Model")
        print("-" * 50)
        
        # Initialize trainer to use its model loading and structure
        # The model configuration should match the one used for training the saved model.
        # We need to set input_channels correctly.
        # This assumes config.model.input_channels is correctly set or can be inferred.
        # For robustness, it's better if the saved model checkpoint includes this.
        # For now, we rely on the current config or what DataProcessor sets.
        
        device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        checkpoint_path = "tcn_strategy/models/best_model.pth"
        if not os.path.exists(checkpoint_path):
            print(f"❌ Error: Best model not found at {checkpoint_path}")
            print("Please train a model first using 'python main.py train' or 'python main.py full'.")
            return results
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # Explicitly set weights_only=False
        
        # Reconstruct config from checkpoint for model creation
        temp_config_for_model_load = Config()
        saved_checkpoint_config_dict = checkpoint.get('config', {})

        # It's safer to update attributes individually or ensure full compatibility
        # For ModelConfig and DataConfig, we can try to re-initialize if they exist in saved config
        if 'data' in saved_checkpoint_config_dict and isinstance(saved_checkpoint_config_dict['data'], DataConfig):
            temp_config_for_model_load.data = saved_checkpoint_config_dict['data']
        elif 'data' in saved_checkpoint_config_dict and isinstance(saved_checkpoint_config_dict['data'], dict):
             # Fallback if it's somehow still a dict (e.g. older checkpoint)
            temp_config_for_model_load.data = DataConfig(**saved_checkpoint_config_dict['data'])
        else: # Fallback to current config's data settings if not in checkpoint or not a DataConfig/dict
            print("Warning: 'data' config not found in checkpoint or is of unexpected type. Using current config's prediction_horizons.")
            temp_config_for_model_load.data.prediction_horizons = config.data.prediction_horizons

        if 'model' in saved_checkpoint_config_dict and isinstance(saved_checkpoint_config_dict['model'], ModelConfig):
            temp_config_for_model_load.model = saved_checkpoint_config_dict['model']
        elif 'model' in saved_checkpoint_config_dict and isinstance(saved_checkpoint_config_dict['model'], dict):
            # Fallback if it's somehow still a dict
            temp_config_for_model_load.model = ModelConfig(**saved_checkpoint_config_dict['model'])
        else:
            print("Warning: 'model' config not found in checkpoint or is of unexpected type. Using default ModelConfig.")
            # temp_config_for_model_load.model is already a default ModelConfig instance

        # CRITICAL: Override input_channels with actual from processed test data
        # This ensures the model architecture matches the data it will process,
        # regardless of what was saved in the checkpoint's ModelConfig.
        if temp_config_for_model_load.model is not None:
             temp_config_for_model_load.model.input_channels = X_test.shape[2]
        else:
            # This case should ideally not happen if 'model' was handled above.
            # If it does, we might need to initialize a default ModelConfig here.
            print("Error: temp_config_for_model_load.model is None. Cannot set input_channels.")
            # Initialize a default ModelConfig if it's None, then set input_channels
            temp_config_for_model_load.model = ModelConfig()
            temp_config_for_model_load.model.input_channels = X_test.shape[2]
        
        # Determine if the saved model was an ensemble by inspecting state_dict keys
        was_ensemble_trained = False
        if 'model_state_dict' in checkpoint:
            for key in checkpoint['model_state_dict'].keys():
                if key.startswith("models.0."): # TCNEnsemble prefixes member model keys with "models.X."
                    was_ensemble_trained = True
                    break
        else:
            print("⚠️ Warning: 'model_state_dict' not found in checkpoint. Cannot determine if ensemble.")
            # Defaulting was_ensemble_trained to True, adjust if necessary or handle error
            was_ensemble_trained = True

        print(f"Inferred 'use_ensemble' for loading: {was_ensemble_trained}")

        # Create a model instance first, then load state_dict
        model_to_load = create_model(temp_config_for_model_load, use_ensemble=was_ensemble_trained)
        model_to_load.to(device)
        
        # Check if input_channels in checkpoint's saved config matches current data (already done by override)
        # but good for a warning if there was a mismatch with the *original* saved input_channels
        if 'model' in saved_checkpoint_config_dict:
            original_model_config_from_checkpoint = saved_checkpoint_config_dict['model']
            # Check if it's a ModelConfig instance and has the attribute
            if isinstance(original_model_config_from_checkpoint, ModelConfig) and \
               hasattr(original_model_config_from_checkpoint, 'input_channels'):
                original_saved_input_channels = original_model_config_from_checkpoint.input_channels
                if original_saved_input_channels != X_test.shape[2]:
                    print(f"⚠️ Warning: Input channels mismatch. Originally saved model (in checkpoint's ModelConfig) had: {original_saved_input_channels}, "
                          f"Current data requires: {X_test.shape[2]}. Model structure adapted using current data's feature count.")
            # Fallback for older checkpoints where 'model' might be a dict within saved_checkpoint_config_dict
            elif isinstance(original_model_config_from_checkpoint, dict) and \
                 'input_channels' in original_model_config_from_checkpoint:
                original_saved_input_channels = original_model_config_from_checkpoint['input_channels']
                if original_saved_input_channels != X_test.shape[2]:
                    print(f"⚠️ Warning: Input channels mismatch (from dict in checkpoint). Originally saved model had: {original_saved_input_channels}, "
                          f"Current data requires: {X_test.shape[2]}. Model structure adapted using current data's feature count.")
            # else: No warning if input_channels not found in the original saved model config.
        
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        model_to_load.eval() # Set model to evaluation mode
        print(f"✓ Model loaded successfully from {checkpoint_path}")

        # Step 3: Backtesting
        print("\n📈 Step 3: Strategy Backtesting")
        print("-" * 50)
        
        test_data_for_backtest = (torch.FloatTensor(X_test), targets) # Ensure X_test is a tensor
        
        backtester = TCNBacktester(config)
        # Pass the loaded model directly
        backtest_results = backtester.run_backtest(model_to_load, test_data_for_backtest, df_processed)
        
        print(f"✓ Backtesting completed successfully")
        print("Backtest Results:")
        strategy_metrics = backtest_results['strategy_metrics']
        print(f"  - Total Return: {strategy_metrics.get('total_return', 0)*100:.2f}%")
        print(f"  - Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  - Max Drawdown: {strategy_metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  - Win Rate: {backtest_results['win_rate']*100:.1f}%")
        print(f"  - Total Trades: {backtest_results['num_trades']}")
        
        results['backtest'] = backtest_results
        
        # Step 4: Generate Reports and Visualizations
        print("\n📊 Step 4: Generating Reports and Visualizations")
        print("-" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plot_path = f"tcn_strategy/results/backtest_results_{timestamp}.png"
        backtester.plot_backtest_results(backtest_results, save_path=plot_path)
        print(f"✓ Backtest visualization saved to {plot_path}")
        
        report = backtester.generate_report(backtest_results)
        report_path = f"tcn_strategy/results/backtest_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Backtest report saved to {report_path}")

        # Print final summary for backtest
        print("\n🎯 BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Initial Capital: ${config.backtest.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${backtest_results['final_portfolio_value']:,.2f}")
        print(f"Total Return: {strategy_metrics.get('total_return', 0)*100:.2f}%")
        print(f"Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Maximum Drawdown: {strategy_metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"Win Rate: {backtest_results['win_rate']*100:.1f}%")
        print(f"Total Trades: {backtest_results['num_trades']}")
        print()
        print("📁 Results saved to:")
        print(f"  - Plots: {plot_path}")
        print(f"  - Reports: {report_path}")

        return results

    except Exception as e:
        print(f"\n❌ Error occurred during backtesting pipeline execution:")
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
        print("  python main.py train [--quick]     # Train model")
        print("  python main.py backtest            # Run backtest only")
        print("  python main.py full [--quick]      # Run full pipeline")
        return
    
    command = sys.argv[1]
    use_quick = '--quick' in sys.argv
    
    config = Config()
    
    if use_quick:
        # Reduce configuration for quick testing
        config.training.num_epochs = 5
        config.training.batch_size = 32
        config.model.num_layers = 2  # Reduced for quick mode
        config.model.hidden_channels = 16  # Reduced for quick mode
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
    elif command == 'full':
        # Run full pipeline (legacy)
        results = run_full_pipeline(config, use_ensemble=not use_quick)
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: train, backtest, full")
        return
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
