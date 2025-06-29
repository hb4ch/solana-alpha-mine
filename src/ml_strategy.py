import polars as pl
import numpy as np
import torch
import joblib
import logging
from strategies import TradingStrategy
from neural_network import load_quantile_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class MLTradingStrategy(TradingStrategy):
    def __init__(self, model_path: str, scaler_path: str, features_path: str,
                 confidence_threshold: float = 0.3,
                 tp_pct: float = 0.004, sl_pct: float = 0.002, horizon_seconds: int = 3600):
        """
        Neural Network Quantile-based Trading Strategy
        
        Args:
            model_path: Path to trained PyTorch quantile model
            scaler_path: Path to feature scaler
            features_path: Path to features list
            confidence_threshold: Minimum confidence for signal generation
            tp_pct: Take profit percentage (for reference)
            sl_pct: Stop loss percentage (for reference)
            horizon_seconds: Prediction horizon in seconds
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        self.confidence_threshold = confidence_threshold
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.horizon_seconds = horizon_seconds
        
        # Device for PyTorch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Load neural network components
            self.model, self.scaler, self.model_info = load_quantile_model(
                self.model_path, self.scaler_path, self.device
            )
            self.features = joblib.load(self.features_path)
            
            logger.info(f"Neural network quantile model loaded from {self.model_path}")
            logger.info(f"Scaler loaded from {self.scaler_path}")
            logger.info(f"Features loaded from {self.features_path}")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Model predicts {self.model_info['config']['num_classes']} quantiles")
            
        except FileNotFoundError as e:
            logger.error(f"Error loading model/scaler/features: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing ML strategy: {e}")
            raise

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate trading signals using quantile predictions
        
        Signal logic:
        - BUY: If predicted quantile with highest probability is quantile 9 (highest return quantile)
        - SELL: If predicted quantile with highest probability is quantile 0 (lowest return quantile)
        - NO SIGNAL: For all other quantiles
        """
        df = data.clone()

        # Check for missing features and add them with default values
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features will be filled with defaults: {missing_features}")
            
            # Add missing features with appropriate default values
            for feature in missing_features:
                if 'phase_' in feature:
                    # Market phase features - default to phase_UNKNOWN = 1, others = 0
                    default_val = 1.0 if feature == 'phase_UNKNOWN' else 0.0
                elif 'volregime_' in feature:
                    # Volatility regime features - default to volregime_medium = 1, others = 0  
                    default_val = 1.0 if feature == 'volregime_medium' else 0.0
                elif 'confidence' in feature:
                    # Phase/regime confidence - default to low confidence
                    default_val = 0.3
                else:
                    # All other missing features default to 0
                    default_val = 0.0
                
                df = df.with_columns(pl.lit(default_val).alias(feature))

        # Extract and prepare features
        X = df.select(self.features).fill_null(0)
        
        if X.shape[0] == 0:
            logger.warning("No data available for prediction")
            return df.with_columns([
                pl.lit(False).alias('entry_signal'),
                pl.lit(False).alias('exit_signal'),
                pl.lit(0.0).alias('confidence'),
                pl.lit(-1).alias('predicted_quantile'),
                pl.lit(0).alias('signal_direction')
            ])
        
        try:
            # Convert to numpy and handle infinite values
            X_numpy = X.to_numpy()
            
            # Replace infinite values and NaN
            X_numpy = np.nan_to_num(X_numpy, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Scale features
            X_scaled = self.scaler.transform(X_numpy)
            
            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                probabilities, predicted_quantiles, max_probabilities = self.model.predict_quantiles(X_tensor)
            
            # Convert to numpy
            probs_np = probabilities.cpu().numpy()
            pred_quantiles_np = predicted_quantiles.cpu().numpy()
            max_probs_np = max_probabilities.cpu().numpy()
            
            # Generate trading signals based on predicted quantiles
            # BUY: predicted quantile is 9 (highest returns)
            # SELL: predicted quantile is 0 (lowest returns)
            buy_signals = (pred_quantiles_np == 9) & (max_probs_np >= self.confidence_threshold)
            sell_signals = (pred_quantiles_np == 0) & (max_probs_np >= self.confidence_threshold)
            
            # Create signal direction: 1 for buy, -1 for sell, 0 for no signal
            signal_direction = np.where(buy_signals, 1, 
                                      np.where(sell_signals, -1, 0))
            
            # Entry signals (both buy and sell are entry signals)
            entry_signals = buy_signals | sell_signals
            
            # For this strategy, we don't use exit signals - let risk management handle exits
            exit_signals = np.zeros_like(entry_signals, dtype=bool)
            
            # Add prediction results to dataframe
            df = df.with_columns([
                pl.Series("confidence", max_probs_np),
                pl.Series("predicted_quantile", pred_quantiles_np),
                pl.Series("signal_direction", signal_direction),
                pl.Series("entry_signal", entry_signals),
                pl.Series("exit_signal", exit_signals)
            ])
            
            # Add individual quantile probabilities for analysis
            for i in range(self.model_info['config']['num_classes']):
                df = df.with_columns(
                    pl.Series(f"quantile_{i}_prob", probs_np[:, i])
                )
            
            # Log signal statistics
            n_buy_signals = np.sum(buy_signals)
            n_sell_signals = np.sum(sell_signals)
            total_signals = n_buy_signals + n_sell_signals
            
            if total_signals > 0:
                avg_confidence = np.mean(max_probs_np[entry_signals])
                logger.info(f"Signal generation complete:")
                logger.info(f"  Buy signals: {n_buy_signals}")
                logger.info(f"  Sell signals: {n_sell_signals}")
                logger.info(f"  Total signals: {total_signals}")
                logger.info(f"  Average confidence: {avg_confidence:.3f}")
            else:
                logger.info("No signals generated (confidence threshold not met)")

        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            return df.with_columns([
                pl.lit(False).alias('entry_signal'),
                pl.lit(False).alias('exit_signal'),
                pl.lit(0.0).alias('confidence'),
                pl.lit(-1).alias('predicted_quantile'),
                pl.lit(0).alias('signal_direction')
            ])

        return df

    def plot_indicators(self, data: pl.DataFrame, trade_log: pl.DataFrame, market: str) -> 'go.Figure':
        """
        Plot quantile predictions and trading signals
        """
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            subplot_titles=(
                f'{market} Price and Signals', 
                'Quantile Predictions',
                'Prediction Confidence'
            ),
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Filter data for the specific market
        market_data = data.filter(pl.col('market') == market).to_pandas()
        if market_data.empty:
            logger.warning(f"No data for market {market}")
            return fig
            
        market_data['ts_utc'] = market_data['ts_utc'].dt.to_pydatetime()
        market_data = market_data.set_index('ts_utc')
        
        # Plot 1: Price and signals
        fig.add_trace(
            go.Scatter(
                x=market_data.index, 
                y=market_data['mid_price'], 
                name='Mid Price', 
                line=dict(color='blue')
            ), 
            row=1, col=1
        )

        # Add trading signals from trade log
        if not trade_log.is_empty():
            market_trades = trade_log.filter(pl.col('market') == market).to_pandas()
            entry_trades = market_trades[market_trades['action'] == 'enter']
            exit_trades = market_trades[market_trades['action'] == 'exit']

            if not entry_trades.empty:
                # Separate buy and sell entries
                buy_entries = entry_trades[entry_trades.get('side', 'long') == 'long']
                sell_entries = entry_trades[entry_trades.get('side', 'short') == 'short']
                
                if not buy_entries.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_entries['timestamp'], 
                            y=buy_entries['price'],
                            name='Buy Entry', 
                            mode='markers',
                            marker=dict(color='green', size=10, symbol='triangle-up')
                        ), 
                        row=1, col=1
                    )
                
                if not sell_entries.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_entries['timestamp'], 
                            y=sell_entries['price'],
                            name='Sell Entry', 
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='triangle-down')
                        ), 
                        row=1, col=1
                    )

            if not exit_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exit_trades['timestamp'], 
                        y=exit_trades['price'],
                        name='Exit', 
                        mode='markers',
                        marker=dict(color='orange', size=8, symbol='x')
                    ), 
                    row=1, col=1
                )
        
        # Plot 2: Quantile predictions
        if 'predicted_quantile' in market_data.columns:
            # Create color map for quantiles
            colors = ['red' if q == 0 else 'green' if q == 9 else 'gray' 
                     for q in market_data['predicted_quantile']]
            
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data['predicted_quantile'],
                    name='Predicted Quantile',
                    mode='markers',
                    marker=dict(color=colors, size=4),
                    text=market_data['predicted_quantile'],
                    hovertemplate='Quantile: %{text}<br>Time: %{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add horizontal lines for key quantiles
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="Sell Quantile (0)", row=2, col=1)
            fig.add_hline(y=9, line_dash="dash", line_color="green", 
                         annotation_text="Buy Quantile (9)", row=2, col=1)
        
        # Plot 3: Prediction confidence
        if 'confidence' in market_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data['confidence'],
                    name='Confidence',
                    line=dict(color='purple'),
                    fill='tonexty'
                ),
                row=3, col=1
            )
            
            # Add confidence threshold line
            fig.add_hline(y=self.confidence_threshold, line_dash="dash", 
                         line_color="orange", 
                         annotation_text=f"Threshold ({self.confidence_threshold})", 
                         row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{market} Neural Network Quantile Strategy Analysis',
            height=800,
            showlegend=True
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Quantile", row=2, col=1, range=[-0.5, 9.5])
        fig.update_yaxes(title_text="Confidence", row=3, col=1, range=[0, 1])
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        return fig

    def get_strategy_info(self) -> dict:
        """
        Get information about the strategy configuration
        """
        return {
            'strategy_type': 'Neural Network Quantile Prediction',
            'model_path': self.model_path,
            'num_quantiles': self.model_info['config']['num_classes'],
            'confidence_threshold': self.confidence_threshold,
            'prediction_horizon_seconds': self.horizon_seconds,
            'device': self.device,
            'num_features': len(self.features),
            'model_config': self.model_info['config']
        }

    def analyze_predictions(self, data: pl.DataFrame, market: str = None) -> dict:
        """
        Analyze prediction patterns and signal distribution
        
        Args:
            data: DataFrame with prediction results
            market: Optional market filter
            
        Returns:
            Dictionary with analysis results
        """
        if market:
            analysis_data = data.filter(pl.col('market') == market)
        else:
            analysis_data = data
        
        if analysis_data.is_empty():
            return {}
        
        # Convert to pandas for easier analysis
        df_analysis = analysis_data.to_pandas()
        
        # Signal distribution
        signal_counts = df_analysis['signal_direction'].value_counts()
        
        # Quantile prediction distribution
        quantile_counts = df_analysis['predicted_quantile'].value_counts().sort_index()
        
        # Confidence statistics
        confidence_stats = {
            'mean': df_analysis['confidence'].mean(),
            'std': df_analysis['confidence'].std(),
            'min': df_analysis['confidence'].min(),
            'max': df_analysis['confidence'].max(),
            'q25': df_analysis['confidence'].quantile(0.25),
            'q75': df_analysis['confidence'].quantile(0.75)
        }
        
        # Signal confidence
        buy_mask = df_analysis['signal_direction'] == 1
        sell_mask = df_analysis['signal_direction'] == -1
        
        signal_confidence = {}
        if buy_mask.any():
            signal_confidence['buy_avg_confidence'] = df_analysis.loc[buy_mask, 'confidence'].mean()
        if sell_mask.any():
            signal_confidence['sell_avg_confidence'] = df_analysis.loc[sell_mask, 'confidence'].mean()
        
        return {
            'total_predictions': len(df_analysis),
            'signal_distribution': signal_counts.to_dict(),
            'quantile_distribution': quantile_counts.to_dict(),
            'confidence_stats': confidence_stats,
            'signal_confidence': signal_confidence,
            'signal_rate': (signal_counts.get(1, 0) + signal_counts.get(-1, 0)) / len(df_analysis) * 100
        }

if __name__ == '__main__':
    print("Neural Network ML Trading Strategy module ready.")
