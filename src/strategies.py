import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class TradingStrategy:
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals (entry/exit) from market data"""
        raise NotImplementedError
        
    def plot_indicators(self, data: pd.DataFrame, market: str) -> plt.Figure:
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
        df['confidence'] = abs(df['histogram']) / df['mid_price'] * 10000
        df['confidence'] = df['confidence'].clip(0, 100)
        
        return df.set_index('ts_utc')

    def plot_indicators(self, data: pd.DataFrame, market: str) -> plt.Figure:
        market_data = data[data['market'] == market]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Price and MACD
        ax1.plot(market_data.index, market_data['mid_price'], label='Price')
        ax1.set_title(f'{market} Price and MACD')
        ax1.legend()
        
        ax2.plot(market_data.index, market_data['macd'], label='MACD', color='blue')
        ax2.plot(market_data.index, market_data['signal_line'], label='Signal', color='orange')
        ax2.bar(market_data.index, market_data['histogram'], 
               label='Histogram', color=np.where(market_data['histogram'] > 0, 'green', 'red'))
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.legend()
        
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
        df['confidence'] = (gap * 100).clip(0, 100)
        
        return df.set_index('ts_utc')

    def plot_indicators(self, data: pd.DataFrame, market: str) -> plt.Figure:
        market_data = data[data['market'] == market]
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Price and MAs
        ax.plot(market_data.index, market_data['mid_price'], label='Price')
        ax.plot(market_data.index, market_data['fast_ma'], label=f'Fast MA ({self.fast_window})', linestyle='--')
        ax.plot(market_data.index, market_data['slow_ma'], label=f'Slow MA ({self.slow_window})', linestyle='--')
        
        # Signal markers
        entry_points = market_data[market_data['entry_signal']]
        exit_points = market_data[market_data['exit_signal']]
        ax.scatter(entry_points.index, entry_points['mid_price'], marker='^', color='green', label='Entry')
        ax.scatter(exit_points.index, exit_points['mid_price'], marker='v', color='red', label='Exit')
        
        ax.set_title(f'{market} Moving Average Crossover')
        ax.legend()
        return fig

class RSIStrategy(TradingStrategy):
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().reset_index()
        
        # Calculate RSI
        delta = df.groupby('market')['mid_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.groupby('market').transform(lambda x: x.ewm(alpha=1/self.period).mean())
        avg_loss = loss.groupby('market').transform(lambda x: x.ewm(alpha=1/self.period).mean())
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['prev_rsi'] = df.groupby('market')['rsi'].shift(1)
        df['entry_signal'] = (df['rsi'] > self.oversold) & (df['prev_rsi'] <= self.oversold)
        df['exit_signal'] = (df['rsi'] < self.overbought) & (df['prev_rsi'] >= self.overbought)
        
        # Confidence score (0-100)
        df['confidence'] = abs(df['rsi'] - 50) * 2  # Distance from 50 scaled to 0-100
        df['confidence'] = df['confidence'].clip(0, 100)
        
        return df.set_index('ts_utc')

    def plot_indicators(self, data: pd.DataFrame, market: str) -> plt.Figure:
        market_data = data[data['market'] == market]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Price
        ax1.plot(market_data.index, market_data['mid_price'], label='Price')
        ax1.set_title(f'{market} Price and RSI')
        ax1.legend()
        
        # RSI
        ax2.plot(market_data.index, market_data['rsi'], label='RSI', color='purple')
        ax2.axhline(self.overbought, color='red', linestyle='--', label='Overbought')
        ax2.axhline(self.oversold, color='green', linestyle='--', label='Oversold')
        ax2.legend()
        
        return fig
