import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Optional, Dict, Any
import joblib
import os

logger = logging.getLogger(__name__)

class QuantileNet(nn.Module):
    """
    Feed-forward neural network for quantile prediction.
    Predicts which quantile a future normalized return will fall into.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256, 128, 64], 
                 num_classes: int = 10, dropout_rate: float = 0.4):
        super(QuantileNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Create layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=1)
    
    def predict_quantiles(self, x):
        """Return both probabilities and predicted quantile classes"""
        with torch.no_grad():
            probs = self.forward(x)
            predicted_quantiles = torch.argmax(probs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            return probs, predicted_quantiles, max_probs

class QuantileDataset(Dataset):
    """PyTorch Dataset for quantile prediction"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class QuantileTrainer:
    """
    Trainer class for the quantile prediction neural network
    """
    
    def __init__(self, model_config: Dict[str, Any], device: str = None):
        self.config = model_config
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        self.quantile_thresholds = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def create_quantile_targets(self, returns: np.ndarray, n_quantiles: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create quantile targets from normalized returns
        
        Args:
            returns: Array of normalized returns
            n_quantiles: Number of quantiles to create
            
        Returns:
            Tuple of (quantile_labels, quantile_thresholds)
        """
        # Remove NaN values for quantile calculation
        valid_returns = returns[~np.isnan(returns)]
        
        # Calculate quantile thresholds (equal frequency)
        quantile_thresholds = np.quantile(valid_returns, np.linspace(0, 1, n_quantiles + 1))
        
        # Assign quantile labels
        quantile_labels = np.digitize(returns, quantile_thresholds[1:-1])
        quantile_labels = np.clip(quantile_labels, 0, n_quantiles - 1)
        
        self.quantile_thresholds = quantile_thresholds
        
        logger.info(f"Created {n_quantiles} quantiles:")
        for i, (lower, upper) in enumerate(zip(quantile_thresholds[:-1], quantile_thresholds[1:])):
            count = np.sum(quantile_labels == i)
            logger.info(f"  Quantile {i}: [{lower:.6f}, {upper:.6f}] - {count} samples")
        
        return quantile_labels, quantile_thresholds
    
    def prepare_data(self, df: pl.DataFrame, features_list: list, 
                     market_to_predict: str = 'SOL_USDT',
                     horizon_seconds: int = 3600) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by calculating normalized returns and quantile targets
        
        Args:
            df: DataFrame with market data and features
            features_list: List of feature column names
            market_to_predict: Market symbol to predict
            horizon_seconds: Prediction horizon in seconds
            
        Returns:
            Tuple of (features, targets, valid_mask)
        """
        logger.info(f"Preparing data for {market_to_predict} with {horizon_seconds}s horizon...")
        
        # Filter for target market
        df_market = df.filter(pl.col('market') == market_to_predict)
        if df_market.is_empty():
            raise ValueError(f"No data for {market_to_predict}")
        
        # Calculate time steps for horizon
        time_diffs = df_market['ts_utc'].diff().median()
        ticks_per_second = 1 / (time_diffs.total_seconds() if time_diffs else 5)
        horizon_steps = int(horizon_seconds * ticks_per_second) or 1
        
        logger.info(f"Using {horizon_steps} steps for {horizon_seconds}s horizon")
        
        # Calculate forward returns - handle different price column names
        price_col = None
        for col in ['mid_price', 'mid_price_calc', 'wap', 'close', 'price']:
            if col in df_market.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError("No price column found. Available columns: " + str(df_market.columns))
        
        logger.info(f"Using price column: {price_col}")
        future_prices = df_market[price_col].shift(-horizon_steps)
        returns = (future_prices - df_market[price_col]) / df_market[price_col]
        
        # Calculate rolling statistics for normalization (20-period window)
        rolling_mean = returns.rolling_mean(window_size=20)
        rolling_std = returns.rolling_std(window_size=20)
        
        # Normalize returns (demean and scale by std)
        normalized_returns = (returns - rolling_mean) / rolling_std
        
        # Add normalized returns to dataframe
        df_market = df_market.with_columns([
            returns.alias('raw_returns'),
            normalized_returns.alias('normalized_returns')
        ])
        
        # Extract features and ensure no missing required features
        missing_features = [f for f in features_list if f not in df_market.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            features_list = [f for f in features_list if f in df_market.columns]
        
        # Get feature matrix and handle invalid values
        X = df_market.select(features_list).fill_null(0).to_numpy()
        
        # Replace any remaining NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Get normalized returns
        norm_returns = df_market['normalized_returns'].to_numpy()
        
        # Create valid mask (no NaN in normalized returns and finite features)
        feature_valid = np.all(np.isfinite(X), axis=1)
        returns_valid = np.isfinite(norm_returns)
        valid_mask = feature_valid & returns_valid
        
        logger.info(f"Data preparation complete:")
        logger.info(f"  Total samples: {len(X)}")
        logger.info(f"  Valid samples: {np.sum(valid_mask)}")
        logger.info(f"  Features: {len(features_list)}")
        
        return X, norm_returns, valid_mask
    
    def train_model(self, X: np.ndarray, y_returns: np.ndarray, valid_mask: np.ndarray,
                    test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the quantile prediction model
        
        Args:
            X: Feature matrix
            y_returns: Normalized returns
            valid_mask: Boolean mask for valid samples
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining data)
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        # Filter valid samples
        X_valid = X[valid_mask]
        y_valid = y_returns[valid_mask]
        
        if len(X_valid) < 1000:
            raise ValueError(f"Insufficient valid samples: {len(X_valid)}")
        
        # Create quantile targets
        y_quantiles, quantile_thresholds = self.create_quantile_targets(
            y_valid, n_quantiles=self.config['num_classes']
        )
        
        # Time series split (no shuffling to avoid data leakage)
        n_samples = len(X_valid)
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Split data
        X_train = X_valid[:val_idx]
        X_val = X_valid[val_idx:test_idx]
        X_test = X_valid[test_idx:]
        
        y_train = y_quantiles[:val_idx]
        y_val = y_quantiles[val_idx:test_idx]
        y_test = y_quantiles[test_idx:]
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = QuantileDataset(X_train_scaled, y_train)
        val_dataset = QuantileDataset(X_val_scaled, y_val)
        test_dataset = QuantileDataset(X_test_scaled, y_test)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        self.config['input_dim'] = X_train_scaled.shape[1]
        
        # Extract only parameters needed for QuantileNet initialization
        model_params = {
            'input_dim': self.config['input_dim'],
            'hidden_dims': self.config['hidden_dims'],
            'num_classes': self.config['num_classes'],
            'dropout_rate': self.config['dropout_rate']
        }
        self.model = QuantileNet(**model_params).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_targets.size(0)
                    correct += (predicted == batch_targets).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total
            
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'scaler': self.scaler,
                    'quantile_thresholds': quantile_thresholds
                }, 'models/best_quantile_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= self.config['patience']:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        checkpoint = torch.load('models/best_quantile_model.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test evaluation
        test_results = self._evaluate_model(test_loader)
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Test accuracy: {test_results['accuracy']:.2f}%")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss,
            'test_results': test_results,
            'quantile_thresholds': quantile_thresholds
        }
    
    def _evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = np.zeros(self.config['num_classes'])
        class_total = np.zeros(self.config['num_classes'])
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_features)
                _, predicted = torch.max(outputs, 1)
                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()
                
                # Per-class accuracy
                c = (predicted == batch_targets).squeeze()
                for i in range(batch_targets.size(0)):
                    label = batch_targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        accuracy = 100 * correct / total
        class_accuracies = {}
        for i in range(self.config['num_classes']):
            if class_total[i] > 0:
                class_accuracies[f'quantile_{i}'] = 100 * class_correct[i] / class_total[i]
        
        return {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies
        }
    
    def save_model(self, model_path: str, scaler_path: str, config_path: str):
        """Save trained model, scaler, and configuration"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'quantile_thresholds': self.quantile_thresholds
        }, model_path)
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        
        # Save config
        joblib.dump(self.config, config_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Config saved to {config_path}")

def load_quantile_model(model_path: str, scaler_path: str, device: str = None) -> Tuple[QuantileNet, StandardScaler, Dict]:
    """Load a trained quantile model"""
    device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    quantile_thresholds = checkpoint['quantile_thresholds']
    
    # Extract only parameters needed for QuantileNet initialization
    model_params = {
        'input_dim': config['input_dim'],
        'hidden_dims': config['hidden_dims'],
        'num_classes': config['num_classes'],
        'dropout_rate': config['dropout_rate']
    }
    
    # Initialize and load model
    model = QuantileNet(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    return model, scaler, {'config': config, 'quantile_thresholds': quantile_thresholds}

if __name__ == "__main__":
    print("Neural Network module ready.")
