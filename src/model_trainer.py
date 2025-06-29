import polars as pl
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
try:
    from .data_loader import load_and_preprocess_raw_data
except ImportError:
    from data_loader import load_and_preprocess_raw_data
from features import engineer_features
from neural_network import QuantileTrainer, load_quantile_model
import logging
import torch

LOGS_DIR = "logs"
MODELS_DIR = "models"
DATA_BASE_PATH = "crypto_tick"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "model_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_default_model_config():
    """Get default neural network configuration"""
    return {
        'hidden_dims': [512, 256, 128, 64],
        'num_classes': 10,
        'dropout_rate': 0.4,
        'learning_rate': 0.001,
        'batch_size': 1024,
        'epochs': 100,
        'patience': 15
    }

def train_quantile_model(df: pl.DataFrame, features_list: list,
                        market_to_predict: str = 'SOL_USDT',
                        horizon_seconds: int = 3600,
                        model_config: dict = None):
    """
    Train a neural network for quantile prediction
    
    Args:
        df: DataFrame with market data and features
        features_list: List of feature column names to use
        market_to_predict: Market symbol to predict
        horizon_seconds: Prediction horizon in seconds
        model_config: Neural network configuration dict
        
    Returns:
        Trained model, scaler, and training results
    """
    logger.info("Starting neural network quantile training...")
    
    if model_config is None:
        model_config = get_default_model_config()
    
    # Initialize trainer
    trainer = QuantileTrainer(model_config)
    
    # Prepare data
    try:
        X, y_returns, valid_mask = trainer.prepare_data(
            df, features_list, market_to_predict, horizon_seconds
        )
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None, None
    
    # Train model
    try:
        training_results = trainer.train_model(X, y_returns, valid_mask)
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None, None
    
    # Save model files
    model_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_quantile_model.pth")
    scaler_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_quantile_scaler.joblib")
    config_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_quantile_config.joblib")
    features_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_quantile_features.joblib")
    
    try:
        trainer.save_model(model_filename, scaler_filename, config_filename)
        joblib.dump(features_list, features_filename)
        logger.info(f"Model files saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None, None, None
    
    return trainer.model, trainer.scaler, training_results

def evaluate_quantile_model(model_path: str, scaler_path: str, features_path: str,
                           df: pl.DataFrame, market_to_predict: str = 'SOL_USDT',
                           horizon_seconds: int = 3600):
    """
    Evaluate a trained quantile model
    
    Args:
        model_path: Path to saved model
        scaler_path: Path to saved scaler
        features_path: Path to saved features list
        df: DataFrame with market data
        market_to_predict: Market to evaluate on
        horizon_seconds: Prediction horizon
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("Evaluating quantile model...")
    
    try:
        # Load model and components
        model, scaler, model_info = load_quantile_model(model_path, scaler_path)
        features_list = joblib.load(features_path)
        
        # Prepare evaluation data
        trainer = QuantileTrainer(model_info['config'])
        X, y_returns, valid_mask = trainer.prepare_data(
            df, features_list, market_to_predict, horizon_seconds
        )
        
        # Filter valid samples
        X_valid = X[valid_mask]
        y_valid = y_returns[valid_mask]
        
        if len(X_valid) < 100:
            logger.warning(f"Insufficient evaluation samples: {len(X_valid)}")
            return None
        
        # Use last 20% as test set
        test_idx = int(len(X_valid) * 0.8)
        X_test = X_valid[test_idx:]
        y_test = y_valid[test_idx:]
        
        # Create quantile targets for evaluation
        y_quantiles_test, _ = trainer.create_quantile_targets(y_test, n_quantiles=10)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        device = next(model.parameters()).device
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        with torch.no_grad():
            probs, predicted_quantiles, max_probs = model.predict_quantiles(X_test_tensor)
            
        # Convert to numpy
        probs_np = probs.cpu().numpy()
        predicted_quantiles_np = predicted_quantiles.cpu().numpy()
        max_probs_np = max_probs.cpu().numpy()
        
        # Calculate metrics
        accuracy = np.mean(predicted_quantiles_np == y_quantiles_test)
        
        # Calculate per-quantile accuracies
        quantile_accuracies = {}
        for i in range(10):
            mask = y_quantiles_test == i
            if np.sum(mask) > 0:
                quantile_accuracies[f'quantile_{i}'] = np.mean(
                    predicted_quantiles_np[mask] == y_quantiles_test[mask]
                )
        
        # Generate trading signals for evaluation
        buy_signals = predicted_quantiles_np == 9  # Highest quantile
        sell_signals = predicted_quantiles_np == 0  # Lowest quantile
        
        # Calculate signal statistics
        n_buy_signals = np.sum(buy_signals)
        n_sell_signals = np.sum(sell_signals)
        avg_buy_confidence = np.mean(max_probs_np[buy_signals]) if n_buy_signals > 0 else 0
        avg_sell_confidence = np.mean(max_probs_np[sell_signals]) if n_sell_signals > 0 else 0
        
        results = {
            'overall_accuracy': accuracy * 100,
            'quantile_accuracies': quantile_accuracies,
            'n_test_samples': len(X_test),
            'n_buy_signals': n_buy_signals,
            'n_sell_signals': n_sell_signals,
            'avg_buy_confidence': avg_buy_confidence,
            'avg_sell_confidence': avg_sell_confidence,
            'signal_rate': (n_buy_signals + n_sell_signals) / len(X_test) * 100
        }
        
        logger.info("Model evaluation completed:")
        logger.info(f"  Overall accuracy: {accuracy*100:.2f}%")
        logger.info(f"  Buy signals: {n_buy_signals} ({n_buy_signals/len(X_test)*100:.1f}%)")
        logger.info(f"  Sell signals: {n_sell_signals} ({n_sell_signals/len(X_test)*100:.1f}%)")
        logger.info(f"  Avg buy confidence: {avg_buy_confidence:.3f}")
        logger.info(f"  Avg sell confidence: {avg_sell_confidence:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None

def run_training_pipeline(markets_to_load: list, target_market: str,
                          features_list: list,
                          tp_pct: float = 0.004, sl_pct: float = 0.002, 
                          training_horizon_seconds: int = 3600,
                          model_config: dict = None):
    """
    Run the complete neural network training pipeline
    
    Args:
        markets_to_load: List of markets to load data for
        target_market: Market to predict
        features_list: List of feature names
        tp_pct: Take profit percentage (not used in quantile model but kept for compatibility)
        sl_pct: Stop loss percentage (not used in quantile model but kept for compatibility)
        training_horizon_seconds: Prediction horizon in seconds
        model_config: Neural network configuration
    """
    logger.info("Starting neural network training pipeline...")
    
    # Load and preprocess data
    df_raw = load_and_preprocess_raw_data(base_path=DATA_BASE_PATH, markets=markets_to_load)
    if df_raw.is_empty():
        logger.error("No data loaded. Exiting.")
        return
    
    # Engineer features
    df_featured = engineer_features(df_raw)
    
    # Include regime and phase features
    final_features_to_use = features_list + [
        col for col in df_featured.columns 
        if col.startswith(('phase_', 'volregime_')) and col not in features_list
    ]
    final_features_to_use = [f for f in final_features_to_use if f in df_featured.columns]
    logger.info(f"Final features for training: {len(final_features_to_use)} features")
    
    # Drop rows with missing feature values
    df_processed = df_featured.drop_nulls(subset=final_features_to_use)
    if df_processed.is_empty():
        logger.error("DataFrame empty after feature engineering and NaN drop.")
        return
    
    # Check if target market has sufficient data
    target_data = df_processed.filter(pl.col('market') == target_market)
    if target_data.is_empty() or len(target_data) < 10000:
        logger.error(f"Insufficient data for {target_market}. Need at least 10000 samples.")
        return
    
    logger.info(f"Training data size for {target_market}: {len(target_data)} samples")
    
    # Train quantile model
    model, scaler, training_results = train_quantile_model(
        df_processed, 
        features_list=final_features_to_use,
        market_to_predict=target_market,
        horizon_seconds=training_horizon_seconds,
        model_config=model_config
    )
    
    if model is None:
        logger.error("Model training failed")
        return
    
    # Evaluate model
    model_path = os.path.join(MODELS_DIR, f"{target_market}_quantile_model.pth")
    scaler_path = os.path.join(MODELS_DIR, f"{target_market}_quantile_scaler.joblib")
    features_path = os.path.join(MODELS_DIR, f"{target_market}_quantile_features.joblib")
    
    eval_results = evaluate_quantile_model(
        model_path, scaler_path, features_path,
        df_processed, target_market, training_horizon_seconds
    )
    
    if eval_results:
        logger.info("Training pipeline completed successfully!")
        return {
            'training_results': training_results,
            'evaluation_results': eval_results,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'features_path': features_path
        }
    else:
        logger.warning("Training completed but evaluation failed")
        return {
            'training_results': training_results,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'features_path': features_path
        }

def generate_feature_importance_analysis(model_path: str, scaler_path: str, 
                                       features_path: str, df: pl.DataFrame,
                                       market_to_predict: str = 'SOL_USDT',
                                       n_samples: int = 1000):
    """
    Generate feature importance analysis using permutation importance
    
    Args:
        model_path: Path to trained model
        scaler_path: Path to scaler
        features_path: Path to features list
        df: DataFrame with data
        market_to_predict: Market to analyze
        n_samples: Number of samples to use for analysis
        
    Returns:
        Dictionary with feature importance scores
    """
    logger.info("Generating feature importance analysis...")
    
    try:
        # Load model and components
        model, scaler, model_info = load_quantile_model(model_path, scaler_path)
        features_list = joblib.load(features_path)
        
        # Prepare data
        trainer = QuantileTrainer(model_info['config'])
        X, y_returns, valid_mask = trainer.prepare_data(
            df, features_list, market_to_predict, 3600
        )
        
        # Use subset for efficiency
        X_valid = X[valid_mask]
        y_valid = y_returns[valid_mask]
        
        if len(X_valid) > n_samples:
            indices = np.random.choice(len(X_valid), n_samples, replace=False)
            X_subset = X_valid[indices]
            y_subset = y_valid[indices]
        else:
            X_subset = X_valid
            y_subset = y_valid
        
        # Create quantile targets
        y_quantiles, _ = trainer.create_quantile_targets(y_subset, n_quantiles=10)
        
        # Scale features
        X_scaled = scaler.transform(X_subset)
        
        # Get baseline accuracy
        device = next(model.parameters()).device
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        with torch.no_grad():
            _, baseline_preds, _ = model.predict_quantiles(X_tensor)
            baseline_preds = baseline_preds.cpu().numpy()
        
        baseline_accuracy = np.mean(baseline_preds == y_quantiles)
        
        # Calculate permutation importance
        feature_importance = {}
        
        for i, feature_name in enumerate(features_list):
            # Permute feature
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get predictions with permuted feature
            X_perm_tensor = torch.FloatTensor(X_permuted).to(device)
            with torch.no_grad():
                _, perm_preds, _ = model.predict_quantiles(X_perm_tensor)
                perm_preds = perm_preds.cpu().numpy()
            
            perm_accuracy = np.mean(perm_preds == y_quantiles)
            importance = baseline_accuracy - perm_accuracy
            feature_importance[feature_name] = importance
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        logger.info("Top 10 most important features:")
        for i, (feature, importance) in enumerate(list(sorted_importance.items())[:10]):
            logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        return sorted_importance
        
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {e}")
        return None

if __name__ == "__main__":
    logger.info("model_trainer.py executed directly. To run the full pipeline, run main.py.")
