"""
Advanced TCN (Temporal Convolutional Network) implementation for quantitative trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from config import Config
import copy # Moved import copy to the top

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that ensures no future information leakage
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]  # Remove future information
        return x

class TemporalBlock(nn.Module):
    """
    Temporal block with residual connections, normalization, and dropout
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.2, layer_norm: bool = True):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.activation = nn.GELU()  # GELU often works better than ReLU for time series
        self.dropout = nn.Dropout(dropout)
        
        if layer_norm:
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        # x shape: [batch, channels, sequence_length]
        residual = self.residual(x)
        
        # First convolution block
        out = self.conv1(x)
        out = out.transpose(1, 2)  # [batch, seq_len, channels] for LayerNorm
        out = self.norm1(out)
        out = out.transpose(1, 2)  # Back to [batch, channels, seq_len]
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        return out + residual

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for temporal importance weighting
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create causal mask to prevent looking into the future
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)

class AdvancedTCN(nn.Module):
    """
    Advanced TCN with dilated convolutions, residual connections, and attention
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        input_channels = config.model.input_channels
        hidden_channels = config.model.hidden_channels
        num_layers = config.model.num_layers
        kernel_size = config.model.kernel_size
        dilation_base = config.model.dilation_base
        dropout = config.model.dropout
        layer_norm = config.model.layer_norm
        
        # Input projection
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, 1)
        
        # TCN layers with exponentially increasing dilation
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_base ** i
            in_channels = hidden_channels
            out_channels = hidden_channels
            
            self.tcn_layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout, layer_norm)
            )
        
        # Optional attention mechanism
        if config.model.attention:
            self.attention = MultiHeadAttention(hidden_channels, num_heads=8, dropout=dropout)
        else:
            self.attention = None
        
        # Global pooling and output layers for different prediction horizons
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Separate heads for different prediction horizons and tasks
        self.prediction_heads = nn.ModuleDict()
        
        for horizon in config.data.prediction_horizons:
            # Regression head (return prediction)
            self.prediction_heads[f'return_{horizon}'] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1)
            )
            
            # Classification head (direction prediction)
            self.prediction_heads[f'direction_{horizon}'] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 2)  # Binary classification
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: [batch, sequence_length, features]
        batch_size, seq_len, features = x.shape
        
        # Transpose for Conv1d: [batch, features, sequence_length]
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Pass through TCN layers
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)
        
        # Apply attention if enabled
        if self.attention is not None:
            # Transpose for attention: [batch, sequence_length, features]
            x_attn = x.transpose(1, 2)
            x_attn = self.attention(x_attn)
            # Transpose back: [batch, features, sequence_length]
            x = x_attn.transpose(1, 2) + x  # Residual connection
        
        # Global pooling to get final representation
        x_pooled = self.global_pool(x).squeeze(-1)  # [batch, hidden_channels]
        
        # Generate predictions for each horizon
        predictions = {}
        for horizon in self.config.data.prediction_horizons:
            predictions[f'return_{horizon}'] = self.prediction_heads[f'return_{horizon}'](x_pooled)
            predictions[f'direction_{horizon}'] = self.prediction_heads[f'direction_{horizon}'](x_pooled)
        
        return predictions

class TCNEnsemble(nn.Module):
    """
    Ensemble of TCN models for improved robustness and performance
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ensemble_size = config.model.ensemble_size
        
        # Create ensemble of models with slightly different architectures
        self.models = nn.ModuleList()
        # import copy # Ensure copy is imported if not already at the top of the file # Removed from here

        for i in range(self.ensemble_size):
            # Create a new Config object for this specific ensemble member
            # to ensure its model parameters are independent.
            member_config = Config() 
            member_config.data = copy.deepcopy(config.data) # Deepcopy data config
            member_config.model = copy.deepcopy(config.model) # Deepcopy the model config part

            # Add some randomness/variation to ensemble members' model configs
            if i == 1: # Second model
                # Modify the hidden_channels for the second model
                # Ensure it's divisible by num_heads (8 for MultiHeadAttention)
                original_hidden_channels = config.model.hidden_channels # Use original for calculation
                new_hidden_channels = int(original_hidden_channels * 1.2)
                # Ensure divisible by 8 (assuming num_heads=8 in MultiHeadAttention)
                member_config.model.hidden_channels = (new_hidden_channels // 8) * 8 if new_hidden_channels >= 8 else original_hidden_channels 
                if member_config.model.hidden_channels == 0 and new_hidden_channels > 0 : # handle case where it rounds to 0
                    member_config.model.hidden_channels = 8


            elif i == 2: # Third model
                member_config.model.num_layers = max(6, config.model.num_layers - 1)
                member_config.model.dropout = min(0.3, config.model.dropout + 0.05)
            
            # All other models (i=0, and i > 2 if ensemble_size > 3) will use the base config.model settings
            # as per the deepcopy.
            
            self.models.append(AdvancedTCN(member_config)) # Pass the member-specific config
    
    def forward(self, x):
        # Get predictions from all ensemble members
        ensemble_predictions = []
        for model in self.models:
            predictions = model(x)
            ensemble_predictions.append(predictions)
        
        # Average the predictions
        averaged_predictions = {}
        for horizon in self.config.data.prediction_horizons:
            # Average return predictions
            return_preds = torch.stack([pred[f'return_{horizon}'] for pred in ensemble_predictions])
            averaged_predictions[f'return_{horizon}'] = return_preds.mean(dim=0)
            
            # Average direction predictions (logits)
            direction_preds = torch.stack([pred[f'direction_{horizon}'] for pred in ensemble_predictions])
            averaged_predictions[f'direction_{horizon}'] = direction_preds.mean(dim=0)
        
        return averaged_predictions

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for simultaneous regression and classification
    """
    def __init__(self, config: Config, regression_weight: float = 1.0, classification_weight: float = 1.0):
        super().__init__()
        self.config = config
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        total_loss = 0
        loss_components = {}
        
        for horizon in self.config.data.prediction_horizons:
            # Only calculate loss for horizons that have targets
            if f'target_return_{horizon}' in targets and f'target_direction_{horizon}' in targets:
                # Regression loss (return prediction)
                return_loss = self.mse_loss(
                    predictions[f'return_{horizon}'].squeeze(),
                    targets[f'target_return_{horizon}']
                )
                loss_components[f'return_loss_{horizon}'] = return_loss
                total_loss += self.regression_weight * return_loss
                
                # Classification loss (direction prediction)
                direction_loss = self.ce_loss(
                    predictions[f'direction_{horizon}'],
                    targets[f'target_direction_{horizon}'].long()
                )
                loss_components[f'direction_loss_{horizon}'] = direction_loss
                total_loss += self.classification_weight * direction_loss
        
        loss_components['total_loss'] = total_loss
        return total_loss, loss_components

def create_model(config: Config, use_ensemble: bool = True) -> nn.Module:
    """
    Factory function to create TCN model
    """
    if use_ensemble:
        return TCNEnsemble(config)
    else:
        return AdvancedTCN(config)

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    config = Config()
    config.model.input_channels = 50  # Example
    
    model = create_model(config, use_ensemble=False)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 32
    seq_len = config.data.sequence_length
    features = config.model.input_channels
    
    x = torch.randn(batch_size, seq_len, features)
    predictions = model(x)
    
    print("Model output shapes:")
    for key, value in predictions.items():
        print(f"{key}: {value.shape}")
