"""
Professional training pipeline for TCN-based quantitative trading strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
import os
import json
from datetime import datetime
import warnings
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

from config import Config
from model import create_model, MultiTaskLoss, count_parameters
from data_pipeline import DataProcessor

warnings.filterwarnings('ignore')

class EarlyStopping:
    """Early stopping implementation"""
    def __init__(self, patience: int = 15, min_delta: float = 0.0001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class MetricsTracker:
    """Track and log training metrics"""
    def __init__(self, config: Config):
        self.config = config
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {f'horizon_{h}': {'return_mse': [], 'direction_acc': []} 
                             for h in config.data.prediction_horizons}
        self.val_metrics = {f'horizon_{h}': {'return_mse': [], 'direction_acc': []} 
                           for h in config.data.prediction_horizons}
        self.learning_rates = []
        
    def update(self, train_loss: float, val_loss: float, train_metrics: Dict, val_metrics: Dict, lr: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        
        for horizon in self.config.data.prediction_horizons:
            self.train_metrics[f'horizon_{horizon}']['return_mse'].append(
                train_metrics.get(f'return_mse_{horizon}', 0)
            )
            self.train_metrics[f'horizon_{horizon}']['direction_acc'].append(
                train_metrics.get(f'direction_acc_{horizon}', 0)
            )
            self.val_metrics[f'horizon_{horizon}']['return_mse'].append(
                val_metrics.get(f'return_mse_{horizon}', 0)
            )
            self.val_metrics[f'horizon_{horizon}']['direction_acc'].append(
                val_metrics.get(f'direction_acc_{horizon}', 0)
            )
    
    def plot_training_history(self, save_path: str):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', alpha=0.7)
        axes[0, 0].plot(self.val_losses, label='Val Loss', alpha=0.7)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 1].plot(self.learning_rates, alpha=0.7)
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Return MSE for different horizons
        for horizon in self.config.data.prediction_horizons:
            axes[1, 0].plot(self.val_metrics[f'horizon_{horizon}']['return_mse'], 
                           label=f'Horizon {horizon}', alpha=0.7)
        axes[1, 0].set_title('Return Prediction MSE (Validation)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Direction accuracy for different horizons
        for horizon in self.config.data.prediction_horizons:
            axes[1, 1].plot(self.val_metrics[f'horizon_{horizon}']['direction_acc'], 
                           label=f'Horizon {horizon}', alpha=0.7)
        axes[1, 1].set_title('Direction Prediction Accuracy (Validation)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class TCNTrainer:
    """
    Professional trainer for TCN models with comprehensive validation and monitoring
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        self.metrics_tracker = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def prepare_data(self):
        """Prepare data loaders"""
        print("Preparing data...")
        processor = DataProcessor(self.config)
        X_train, X_val, X_test, targets, df = processor.process_data()
        
        # Update config with actual input channels
        self.config.model.input_channels = X_train.shape[2]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Create data loaders for each horizon
        self.train_loaders = {}
        self.val_loaders = {}
        self.test_loaders = {}
        
        # We'll use the first horizon for training, but evaluate on all
        main_horizon = self.config.data.prediction_horizons[0]
        
        # Training data
        train_dataset = TensorDataset(
            X_train_tensor,
            torch.FloatTensor(targets[f'target_return_{main_horizon}_train']),
            torch.LongTensor(targets[f'target_direction_{main_horizon}_train'])
        )
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Validation data  
        val_dataset = TensorDataset(
            X_val_tensor,
            torch.FloatTensor(targets[f'target_return_{main_horizon}_val']),
            torch.LongTensor(targets[f'target_direction_{main_horizon}_val'])
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Test data
        test_dataset = TensorDataset(
            X_test_tensor,
            torch.FloatTensor(targets[f'target_return_{main_horizon}_test']),
            torch.LongTensor(targets[f'target_direction_{main_horizon}_test'])
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Store all targets for comprehensive evaluation
        self.all_targets = targets
        self.test_data = (X_test_tensor, targets)
        
        print(f"Data prepared - Train: {len(self.train_loader.dataset)}, "
              f"Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
        
        return df
    
    def build_model(self, use_ensemble: bool = True):
        """Build and initialize model"""
        print("Building model...")
        self.model = create_model(self.config, use_ensemble=use_ensemble)
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = MultiTaskLoss(self.config)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Initialize scheduler
        if self.config.training.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=10
            )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(self.config)
        
        print(f"Model built with {count_parameters(self.model):,} parameters")
    
    def train_epoch(self) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        metrics = {}
        
        for batch_idx, (X_batch, y_return, y_direction) in enumerate(self.train_loader):
            X_batch = X_batch.to(self.device)
            y_return = y_return.to(self.device)
            y_direction = y_direction.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_batch)
            
            # Prepare targets for loss calculation
            targets = {}
            main_horizon = self.config.data.prediction_horizons[0]
            targets[f'target_return_{main_horizon}'] = y_return
            targets[f'target_direction_{main_horizon}'] = y_direction
            
            # Calculate loss
            loss, loss_components = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                pred_return = predictions[f'return_{main_horizon}'].squeeze()
                pred_direction = torch.argmax(predictions[f'direction_{main_horizon}'], dim=1)
                
                return_mse = mean_squared_error(y_return.cpu().numpy(), pred_return.cpu().numpy())
                direction_acc = accuracy_score(y_direction.cpu().numpy(), pred_direction.cpu().numpy())
                
                metrics[f'return_mse_{main_horizon}'] = return_mse
                metrics[f'direction_acc_{main_horizon}'] = direction_acc
        
        return total_loss / len(self.train_loader), metrics
    
    def validate_epoch(self) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        metrics = {}
        all_predictions = {f'return_{h}': [] for h in self.config.data.prediction_horizons}
        all_predictions.update({f'direction_{h}': [] for h in self.config.data.prediction_horizons})
        all_targets = {f'return_{h}': [] for h in self.config.data.prediction_horizons}
        all_targets.update({f'direction_{h}': [] for h in self.config.data.prediction_horizons})
        
        with torch.no_grad():
            for X_batch, y_return, y_direction in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_return = y_return.to(self.device)
                y_direction = y_direction.to(self.device)
                
                # Forward pass
                predictions = self.model(X_batch)
                
                # Prepare targets for loss calculation
                targets = {}
                main_horizon = self.config.data.prediction_horizons[0]
                targets[f'target_return_{main_horizon}'] = y_return
                targets[f'target_direction_{main_horizon}'] = y_direction
                
                # Calculate loss
                loss, _ = self.criterion(predictions, targets)
                total_loss += loss.item()
                
                # Store predictions and targets
                for horizon in self.config.data.prediction_horizons:
                    if f'return_{horizon}' in predictions:
                        all_predictions[f'return_{horizon}'].append(
                            np.atleast_1d(predictions[f'return_{horizon}'].squeeze().cpu().numpy())
                        )
                        all_predictions[f'direction_{horizon}'].append(
                            torch.argmax(predictions[f'direction_{horizon}'], dim=1).cpu().numpy()
                        )
                        
                        if horizon == main_horizon:
                            all_targets[f'return_{horizon}'].append(y_return.cpu().numpy())
                            all_targets[f'direction_{horizon}'].append(y_direction.cpu().numpy())
        
        # Calculate metrics
        for horizon in self.config.data.prediction_horizons:
            if all_predictions[f'return_{horizon}'] and all_targets[f'return_{horizon}']:
                pred_return = np.concatenate(all_predictions[f'return_{horizon}'])
                true_return = np.concatenate(all_targets[f'return_{horizon}'])
                pred_direction = np.concatenate(all_predictions[f'direction_{horizon}'])
                true_direction = np.concatenate(all_targets[f'direction_{horizon}'])
                
                return_mse = mean_squared_error(true_return, pred_return)
                direction_acc = accuracy_score(true_direction, pred_direction)
                
                metrics[f'return_mse_{horizon}'] = return_mse
                metrics[f'direction_acc_{horizon}'] = direction_acc
        
        return total_loss / len(self.val_loader), metrics
    
    def train(self, use_ensemble: bool = True) -> Dict:
        """Main training loop"""
        print("Starting training...")
        
        # Prepare data and model
        df = self.prepare_data()
        self.build_model(use_ensemble=use_ensemble)
        
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.num_epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_loss, train_metrics = self.train_epoch()
            
            # Validate epoch
            val_loss, val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.config.training.lr_scheduler == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics tracker
            self.metrics_tracker.update(train_loss, val_loss, train_metrics, val_metrics, current_lr)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs} "
                  f"| Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} "
                  f"| LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Print detailed metrics for main horizon
            main_horizon = self.config.data.prediction_horizons[0]
            if f'return_mse_{main_horizon}' in val_metrics:
                print(f"  Val Return MSE: {val_metrics[f'return_mse_{main_horizon}']:.6f} "
                      f"| Val Direction Acc: {val_metrics[f'direction_acc_{main_horizon}']:.3f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.1f} minutes")
        
        # Save final results
        results = self.evaluate_model()
        self.save_training_results(results, df)
        
        return results
    
    def evaluate_model(self) -> Dict:
        """Comprehensive model evaluation"""
        print("Evaluating model...")
        self.model.eval()
        
        results = {}
        _, full_targets_dict = self.test_data # Unpack the full targets dictionary

        # Initialize accumulators for predictions and true targets for each horizon
        all_preds_for_horizon = {
            f'{type}_{h}': [] for h in self.config.data.prediction_horizons for type in ['return', 'direction']
        }
        all_true_for_horizon = {
            f'{type}_{h}': [] for h in self.config.data.prediction_horizons for type in ['return', 'direction']
        }
        
        processed_samples = 0
        with torch.no_grad():
            for X_batch, _, _ in self.test_loader: # y_batch from loader is for main horizon only, ignore for now
                X_batch = X_batch.to(self.device)
                batch_size = X_batch.shape[0]
                
                batch_model_predictions = self.model(X_batch) # Get predictions for this batch

                for horizon in self.config.data.prediction_horizons:
                    # Store model predictions for this batch and horizon
                    pred_return_b = batch_model_predictions[f'return_{horizon}'].squeeze().cpu().numpy()
                    pred_direction_b = torch.argmax(batch_model_predictions[f'direction_{horizon}'], dim=1).cpu().numpy()
                    
                    all_preds_for_horizon[f'return_{horizon}'].append(np.atleast_1d(pred_return_b))
                    all_preds_for_horizon[f'direction_{horizon}'].append(pred_direction_b)

                    # Get and store corresponding true targets for this batch and horizon
                    true_return_b = full_targets_dict[f'target_return_{horizon}_test'][processed_samples : processed_samples + batch_size]
                    true_direction_b = full_targets_dict[f'target_direction_{horizon}_test'][processed_samples : processed_samples + batch_size]
                    
                    all_true_for_horizon[f'return_{horizon}'].append(np.atleast_1d(true_return_b))
                    all_true_for_horizon[f'direction_{horizon}'].append(true_direction_b)
                
                processed_samples += batch_size

            # After processing all batches, concatenate and calculate metrics
            for horizon in self.config.data.prediction_horizons:
                y_pred_return = np.concatenate(all_preds_for_horizon[f'return_{horizon}'])
                y_pred_direction = np.concatenate(all_preds_for_horizon[f'direction_{horizon}'])
                
                y_true_return = np.concatenate(all_true_for_horizon[f'return_{horizon}'])
                y_true_direction = np.concatenate(all_true_for_horizon[f'direction_{horizon}'])
                
                # Calculate metrics
                return_mse = mean_squared_error(y_true_return, y_pred_return)
                return_r2 = r2_score(y_true_return, y_pred_return)
                direction_acc = accuracy_score(y_true_direction, y_pred_direction)
                
                # Direction distribution
                direction_report = classification_report(y_true_direction, y_pred_direction, output_dict=True)
                
                results[f'horizon_{horizon}'] = {
                    'return_mse': return_mse,
                    'return_rmse': np.sqrt(return_mse),
                    'return_r2': return_r2,
                    'direction_accuracy': direction_acc,
                    'direction_precision': direction_report['weighted avg']['precision'],
                    'direction_recall': direction_report['weighted avg']['recall'],
                    'direction_f1': direction_report['weighted avg']['f1-score']
                }
                
                print(f"Horizon {horizon} - Return RMSE: {np.sqrt(return_mse):.6f}, "
                      f"R²: {return_r2:.3f}, Direction Acc: {direction_acc:.3f}")
        
        return results
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }
        
        # Save latest checkpoint
        checkpoint_path = "tcn_strategy/models/latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = "tcn_strategy/models/best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch+1}")
    
    def save_training_results(self, results: Dict, df: pd.DataFrame):
        """Save comprehensive training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics plot
        plot_path = f"tcn_strategy/results/training_history_{timestamp}.png"
        self.metrics_tracker.plot_training_history(plot_path)
        
        # Save results JSON
        results_path = f"tcn_strategy/results/results_{timestamp}.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: float(v) if isinstance(v, np.floating) else v 
                                       for k, v in value.items()}
                else:
                    json_results[key] = float(value) if isinstance(value, np.floating) else value
            json.dump(json_results, f, indent=4)
        
        # Save config
        config_path = f"tcn_strategy/results/config_{timestamp}.json"
        with open(config_path, 'w') as f:
            config_dict = {
                'data': self.config.data.__dict__,
                'features': self.config.features.__dict__,
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
                'backtest': self.config.backtest.__dict__,
                'risk': self.config.risk.__dict__
            }
            json.dump(config_dict, f, indent=4)
        
        print(f"Results saved to tcn_strategy/results/")
    
    def load_best_model(self):
        """Load the best saved model"""
        checkpoint_path = "tcn_strategy/models/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Best model loaded successfully")
            return True
        else:
            print("No saved model found")
            return False

if __name__ == "__main__":
    # Train the model
    config = Config()
    trainer = TCNTrainer(config)
    results = trainer.train(use_ensemble=True)
    print("Training completed!")
