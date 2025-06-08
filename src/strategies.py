import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # No longer directly used for plotting here
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Optional

class TradingStrategy:
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals (entry/exit) from market data"""
        raise NotImplementedError
        
    def plot_indicators(self, data: pd.DataFrame, market: str) -> go.Figure:
        """Plot technical indicators for visual analysis"""
        raise NotImplementedError

class MACDStrategy(TradingStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().reset_index()
        
        # Calculate MACD components
        df['ema_fast'] = df.groupby('market')['mid_price'].transform(
            lambda x: x.ewm(span=self.fast).mean()
        )
        df['ema_slow'] = df.groupby('market')['mid_price'].transform(
            lambda x: x.ewm(span=self.slow).mean()
        )
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal_line'] = df.groupby('market')['macd'].transform(
            lambda x: x.ewm(span=self.signal).mean()
        )
        df['histogram'] = df['macd'] - df['signal_line']
        
        # Generate signals
        df['prev_hist'] = df.groupby('market')['histogram'].shift(1)
        df['entry_signal'] = (df['histogram'] > 0) & (df['prev_hist'] <= 0)
        df['exit_signal'] = (df['histogram'] < 0) & (df['prev_hist'] >= 0)
        
        # Confidence score (0-100)
        # Increased scaling factor for more sensitivity
        df['confidence'] = (abs(df['histogram']) / df['mid_price'] * 200000).clip(0, 100)
        
        return df.set_index('ts_utc')

    def plot_indicators(self, data: pd.DataFrame, market: str, min_confidence: float = 30) -> go.Figure:
        market_data = data[data['market'] == market].copy() # Use .copy() to avoid SettingWithCopyWarning
        # Ensure index is datetime for Plotly
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f'{market} Price', f'{market} MACD'))
        
        # Price
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['mid_price'], name='Price', line=dict(color='blue')), 
                      row=1, col=1)

        # Entry and Exit signals based on confidence
        entry_points = market_data[(market_data['entry_signal']) & (market_data['confidence'] >= min_confidence)]
        exit_points = market_data[(market_data['exit_signal']) & (market_data['confidence'] >= min_confidence)]
        
        fig.add_trace(go.Scatter(x=entry_points.index, y=entry_points['mid_price'], mode='markers', name='Entry Signal (Confident)', 
                                 marker=dict(color='green', size=10, symbol='triangle-up'), legendgroup='signals', legendgrouptitle_text='Signals'), 
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=exit_points.index, y=exit_points['mid_price'], mode='markers', name='Exit Signal (Confident)', 
                                 marker=dict(color='red', size=10, symbol='triangle-down'), legendgroup='signals'), 
                      row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['macd'], name='MACD', line=dict(color='purple')), 
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['signal_line'], name='Signal Line', line=dict(color='orange')), 
                      row=2, col=1)
        
        # Histogram
        colors = ['green' if val >= 0 else 'red' for val in market_data['histogram']]
        fig.add_trace(go.Bar(x=market_data.index, y=market_data['histogram'], name='Histogram', marker_color=colors), 
                      row=2, col=1)
        
        fig.update_layout(height=700, title_text=f'{market} MACD Analysis', legend_title_text='Indicators')
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="MACD Value", row=2, col=1)
        
        return fig

class MovingAverageCrossover(TradingStrategy):
    def __init__(self, fast_window: int = 10, slow_window: int = 30):
        self.fast_window = fast_window
        self.slow_window = slow_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().reset_index()
        
        # Calculate moving averages
        df['fast_ma'] = df.groupby('market')['mid_price'].transform(
            lambda x: x.rolling(self.fast_window).mean()
        )
        df['slow_ma'] = df.groupby('market')['mid_price'].transform(
            lambda x: x.rolling(self.slow_window).mean()
        )
        
        # Generate signals
        df['prev_fast'] = df.groupby('market')['fast_ma'].shift(1)
        df['prev_slow'] = df.groupby('market')['slow_ma'].shift(1)
        df['entry_signal'] = (df['fast_ma'] > df['slow_ma']) & (df['prev_fast'] <= df['prev_slow'])
        df['exit_signal'] = (df['fast_ma'] < df['slow_ma']) & (df['prev_fast'] >= df['prev_slow'])
        
        # Confidence score (0-100)
        gap = abs(df['fast_ma'] - df['slow_ma']) / df['mid_price']
        # Adjusted scaling: e.g., a 0.2% gap (0.002) -> confidence 10; 1% gap (0.01) -> confidence 50
        df['confidence'] = (gap * 5000).clip(0, 100)
        
        return df.set_index('ts_utc')

    def plot_indicators(self, data: pd.DataFrame, market: str, min_confidence: float = 30) -> go.Figure:
        market_data = data[data['market'] == market].copy()
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)

        fig = go.Figure()
        
        # Price and MAs
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['mid_price'], name='Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['fast_ma'], name=f'Fast MA ({self.fast_window})', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['slow_ma'], name=f'Slow MA ({self.slow_window})', line=dict(color='green', dash='dash')))
        
        # Signal markers (filtered by confidence)
        entry_points = market_data[(market_data['entry_signal']) & (market_data['confidence'] >= min_confidence)]
        exit_points = market_data[(market_data['exit_signal']) & (market_data['confidence'] >= min_confidence)]
        
        fig.add_trace(go.Scatter(x=entry_points.index, y=entry_points['mid_price'], mode='markers', name='Entry Signal (Confident)', 
                                 marker=dict(color='green', size=10, symbol='triangle-up'), legendgroup='signals', legendgrouptitle_text='Signals'))
        fig.add_trace(go.Scatter(x=exit_points.index, y=exit_points['mid_price'], mode='markers', name='Exit Signal (Confident)', 
                                 marker=dict(color='red', size=10, symbol='triangle-down'), legendgroup='signals'))
        
        fig.update_layout(title=f'{market} Moving Average Crossover Analysis',
                          xaxis_title='Time', yaxis_title='Price (USDT)',
                          height=600, legend_title_text='Indicators')
        return fig

class RSIStrategy(TradingStrategy):
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().reset_index() # Ensure 'market' is a column and we have a default index
        
        # Calculate RSI
        delta = df.groupby('market')['mid_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0) # Ensure loss is positive
        
        # Calculate Wilder's EMA for average gain and loss
        # adjust=False is crucial for Wilder's RSI.
        # min_periods ensures that EMA starts after enough data.
        avg_gain = gain.groupby(df['market']).transform(
            lambda x: x.ewm(alpha=1/self.period, adjust=False, min_periods=self.period).mean()
        )
        avg_loss = loss.groupby(df['market']).transform(
            lambda x: x.ewm(alpha=1/self.period, adjust=False, min_periods=self.period).mean()
        )
        
        # Calculate Relative Strength (RS)
        rs = np.zeros_like(avg_gain, dtype=float) # Initialize rs array

        # Case 1: avg_loss > 0
        mask_loss_positive = avg_loss > 0
        rs[mask_loss_positive] = avg_gain[mask_loss_positive] / avg_loss[mask_loss_positive]
        
        # Case 2: avg_loss == 0 and avg_gain > 0 (RSI should be 100)
        # For RSI formula 100 - (100 / (1 + RS)), RS needs to be np.inf
        mask_loss_zero_gain_pos = (avg_loss == 0) & (avg_gain > 0)
        rs[mask_loss_zero_gain_pos] = np.inf
        
        # Case 3: avg_loss == 0 and avg_gain == 0 (RSI should be 50)
        # For RSI formula, if RS = 1, then RSI = 100 - (100 / 2) = 50.
        mask_both_zero = (avg_loss == 0) & (avg_gain == 0)
        rs[mask_both_zero] = 1.0 # Results in RSI of 50

        # Handle initial NaN periods from ewm if not covered (e.g. if avg_gain is NaN but avg_loss is not)
        # This will make rs NaN for these periods.
        rs[avg_gain.isna() | avg_loss.isna()] = np.nan

        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['prev_rsi'] = df.groupby('market')['rsi'].shift(1)
        df['entry_signal'] = (df['rsi'] > self.oversold) & (df['prev_rsi'] <= self.oversold)
        df['exit_signal'] = (df['rsi'] < self.overbought) & (df['prev_rsi'] >= self.overbought)
        
        # Confidence score (0-100)
        # abs(df['rsi'] - 50) gives distance from midline (0 to 50). Multiply by 2 to scale to 0-100.
        df['confidence'] = abs(df['rsi'] - 50) * 2 
        df['confidence'] = df['confidence'].clip(0, 100) # Ensure it's within [0,100]
                                                        # NaNs in rsi will propagate to confidence.
        
        return df.set_index('ts_utc')

    def plot_indicators(self, data: pd.DataFrame, market: str, min_confidence: float = 30) -> go.Figure:
        market_data = data[data['market'] == market].copy()
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f'{market} Price', f'{market} RSI'))
        
        # Price
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['mid_price'], name='Price', line=dict(color='blue')), 
                      row=1, col=1)

        # Entry and Exit signals based on confidence
        entry_points = market_data[(market_data['entry_signal']) & (market_data['confidence'] >= min_confidence)]
        exit_points = market_data[(market_data['exit_signal']) & (market_data['confidence'] >= min_confidence)]
        
        fig.add_trace(go.Scatter(x=entry_points.index, y=entry_points['mid_price'], mode='markers', name='Entry Signal (Confident)', 
                                 marker=dict(color='green', size=10, symbol='triangle-up'), legendgroup='signals', legendgrouptitle_text='Signals'), 
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=exit_points.index, y=exit_points['mid_price'], mode='markers', name='Exit Signal (Confident)', 
                                 marker=dict(color='red', size=10, symbol='triangle-down'), legendgroup='signals'), 
                      row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['rsi'], name='RSI', line=dict(color='purple')), 
                      row=2, col=1)
        
        # Overbought/Oversold lines
        fig.add_hline(y=self.overbought, line_dash="dash", line_color="red", annotation_text="Overbought", 
                      annotation_position="bottom right", row=2, col=1)
        fig.add_hline(y=self.oversold, line_dash="dash", line_color="green", annotation_text="Oversold", 
                      annotation_position="bottom right", row=2, col=1)
        
        fig.update_layout(height=700, title_text=f'{market} RSI Analysis', legend_title_text='Indicators')
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="RSI Value", row=2, col=1, range=[0, 100]) # RSI is 0-100
        
        return fig
