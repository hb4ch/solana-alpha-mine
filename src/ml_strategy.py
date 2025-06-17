import pandas as pd
import numpy as np
import joblib
import logging
from strategies import TradingStrategy
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class MLTradingStrategy(TradingStrategy):
    def __init__(self, model_path: str, scaler_path: str, features_path: str,
                 prediction_threshold: float = 0.55,
                 tp_pct: float = 0.004, sl_pct: float = 0.002, horizon_seconds: int = 3600):
        """
        Initializes the MLTradingStrategy.

        Args:
            model_path (str): Path to the trained model file.
            scaler_path (str): Path to the trained preprocessor (scaler).
            features_path (str): Path to the saved list of feature names.
            prediction_threshold (float): Minimum probability for a positive prediction to be considered an entry signal.
            tp_pct (float): Take-profit percentage for the triple barrier.
            sl_pct (float): Stop-loss percentage for the triple barrier.
            horizon_seconds (int): Time horizon for the triple barrier.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        self.prediction_threshold = prediction_threshold
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.horizon_seconds = horizon_seconds
        
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.features = joblib.load(self.features_path)
            logger.info(f"ML model loaded from {self.model_path}")
            logger.info(f"Scaler loaded from {self.scaler_path}")
            logger.info(f"Features loaded from {self.features_path}")
        except FileNotFoundError as e:
            logger.error(f"Error loading model/scaler/features: {e}")
            raise

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals using the loaded ML model.

        Args:
            data (pd.DataFrame): DataFrame with market data and pre-calculated features.

        Returns:
            pd.DataFrame: DataFrame with 'entry_signal', 'exit_signal', and 'confidence' columns.
        """
        df = data.copy()

        # 1. Ensure all required features are present.
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing required features for prediction: {missing_features}")
            df['entry_signal'] = False
            df['exit_signal'] = False
            df['confidence'] = 0.0
            return df

        # Create a features DataFrame
        X = df.copy()

        # Align columns with the feature list from training
        # This adds missing columns (e.g., for rare dummy variables) and fills them with 0
        # It also removes any columns present in the backtest data but not in the training data
        X = X.reindex(columns=self.features, fill_value=0)
        
        # The order of columns must also match
        X = X[self.features]

        # Handle any remaining NaNs that might exist in the original data
        X.fillna(0, inplace=True)

        # 2. Preprocess the features using the loaded scaler.
        X_scaled = self.scaler.transform(X)

        # 3. Make predictions.
        try:
            pred_probas = self.model.predict_proba(X_scaled)[:, 1]
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            df['entry_signal'] = False
            df['exit_signal'] = False
            df['confidence'] = 0.0
            return df

        # 4. Convert predictions to signals.
        df['confidence'] = pred_probas
        df['entry_signal'] = (df['confidence'] >= self.prediction_threshold)
        df['exit_signal'] = False  # Exits are handled by the backtester's triple barrier logic

        logger.info(f"Signal generation complete. Found {df['entry_signal'].sum()} potential entry signals.")
        return df

    def plot_indicators(self, data: pd.DataFrame, trade_log: pd.DataFrame, market: str) -> 'go.Figure':
        """
        Plots the price, and overlays entry and exit points from the trade log.

        Args:
            data (pd.DataFrame): The original data with prices.
            trade_log (pd.DataFrame): The log of trades from the backtest.
            market (str): The market to plot.
        
        Returns:
            go.Figure: A Plotly figure object.
        """
        fig = go.Figure()
        market_data = data[data['market'] == market]
        
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['mid_price'], name='Mid Price', line=dict(color='blue')))

        # Filter trade log for the specific market
        market_trades = trade_log[trade_log['market'] == market]
        entry_trades = market_trades[market_trades['action'] == 'enter']
        exit_trades = market_trades[market_trades['action'] == 'exit']

        fig.add_trace(go.Scatter(
            x=entry_trades['timestamp'], y=entry_trades['price'],
            name='Entry', mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))

        fig.add_trace(go.Scatter(
            x=exit_trades['timestamp'], y=exit_trades['price'],
            name='Exit', mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))

        fig.update_layout(
            title=f'{market} Backtest with ML Strategy Signals',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        return fig

if __name__ == '__main__':
    # This section is for conceptual testing and will not run without actual model files.
    print("MLTradingStrategy module ready.")
    # Example usage:
    # features_list = [...] # Load from a config file or define here
    # strategy = MLTradingStrategy(
    #     model_path='models/SOL_USDT_lgbm_model.joblib',
    #     scaler_path='models/SOL_USDT_scaler.joblib',
    #     features=features_list,
    #     prediction_threshold=0.6
    # )
    # sample_data_with_features = pd.DataFrame(...) # Load and engineer features
    # signals_df = strategy.generate_signals(sample_data_with_features)
    # print(signals_df.head())
    pass
