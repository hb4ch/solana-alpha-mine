import polars as pl
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

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data.clone()

        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing required features for prediction: {missing_features}")
            return df.with_columns([
                pl.lit(False).alias('entry_signal'),
                pl.lit(False).alias('exit_signal'),
                pl.lit(0.0).alias('confidence')
            ])

        X = df.select(self.features).fill_null(0)
        
        X_scaled = self.scaler.transform(X.to_numpy())

        try:
            pred_probas = self.model.predict_proba(X_scaled)[:, 1]
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            return df.with_columns([
                pl.lit(False).alias('entry_signal'),
                pl.lit(False).alias('exit_signal'),
                pl.lit(0.0).alias('confidence')
            ])

        df = df.with_columns(pl.Series("confidence", pred_probas))
        df = df.with_columns(
            (pl.col('confidence') >= self.prediction_threshold).alias('entry_signal')
        )
        df = df.with_columns(pl.lit(False).alias('exit_signal'))

        logger.info(f"Signal generation complete. Found {df['entry_signal'].sum()} potential entry signals.")
        return df

    def plot_indicators(self, data: pl.DataFrame, trade_log: pl.DataFrame, market: str) -> 'go.Figure':
        fig = go.Figure()
        market_data = data.filter(pl.col('market') == market).to_pandas()
        market_data['ts_utc'] = market_data['ts_utc'].dt.to_pydatetime()
        market_data = market_data.set_index('ts_utc')
        
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['mid_price'], name='Mid Price', line=dict(color='blue')))

        market_trades = trade_log.filter(pl.col('market') == market).to_pandas()
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
    print("MLTradingStrategy module ready.")
    pass
