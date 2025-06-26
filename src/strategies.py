import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Optional

class TradingStrategy:
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals (entry/exit) from market data"""
        raise NotImplementedError
        
    def plot_indicators(self, data: pl.DataFrame, market: str) -> go.Figure:
        """Plot technical indicators for visual analysis"""
        raise NotImplementedError

class MACDStrategy(TradingStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data.with_columns([
            pl.col('mid_price').ewm_mean(span=self.fast).over('market').alias('ema_fast'),
            pl.col('mid_price').ewm_mean(span=self.slow).over('market').alias('ema_slow')
        ])
        df = df.with_columns(
            (pl.col('ema_fast') - pl.col('ema_slow')).alias('macd')
        )
        df = df.with_columns(
            pl.col('macd').ewm_mean(span=self.signal).over('market').alias('signal_line')
        )
        df = df.with_columns(
            (pl.col('macd') - pl.col('signal_line')).alias('histogram')
        )
        
        df = df.with_columns(
            pl.col('histogram').shift(1).over('market').alias('prev_hist')
        )
        df = df.with_columns([
            ((pl.col('histogram') > 0) & (pl.col('prev_hist') <= 0)).alias('entry_signal'),
            ((pl.col('histogram') < 0) & (pl.col('prev_hist') >= 0)).alias('exit_signal')
        ])
        
        df = df.with_columns(
            (abs(pl.col('histogram')) / pl.col('mid_price') * 200000).clip(0, 100).alias('confidence')
        )
        
        return df

    def plot_indicators(self, data: pl.DataFrame, market: str, min_confidence: float = 30) -> go.Figure:
        market_data = data.filter(pl.col('market') == market).to_pandas()
        market_data['ts_utc'] = market_data['ts_utc'].dt.to_pydatetime()
        market_data = market_data.set_index('ts_utc')

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f'{market} Price', f'{market} MACD'))
        
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['mid_price'], name='Price', line=dict(color='blue')), 
                      row=1, col=1)

        entry_points = market_data[(market_data['entry_signal']) & (market_data['confidence'] >= min_confidence)]
        exit_points = market_data[(market_data['exit_signal']) & (market_data['confidence'] >= min_confidence)]
        
        fig.add_trace(go.Scatter(x=entry_points.index, y=entry_points['mid_price'], mode='markers', name='Entry Signal (Confident)', 
                                 marker=dict(color='green', size=10, symbol='triangle-up'), legendgroup='signals', legendgrouptitle_text='Signals'), 
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=exit_points.index, y=exit_points['mid_price'], mode='markers', name='Exit Signal (Confident)', 
                                 marker=dict(color='red', size=10, symbol='triangle-down'), legendgroup='signals'), 
                      row=1, col=1)
        
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['macd'], name='MACD', line=dict(color='purple')), 
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['signal_line'], name='Signal Line', line=dict(color='orange')), 
                      row=2, col=1)
        
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
    
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data.with_columns([
            pl.col('mid_price').rolling_mean(self.fast_window).over('market').alias('fast_ma'),
            pl.col('mid_price').rolling_mean(self.slow_window).over('market').alias('slow_ma')
        ])
        
        df = df.with_columns([
            pl.col('fast_ma').shift(1).over('market').alias('prev_fast'),
            pl.col('slow_ma').shift(1).over('market').alias('prev_slow')
        ])
        
        df = df.with_columns([
            ((pl.col('fast_ma') > pl.col('slow_ma')) & (pl.col('prev_fast') <= pl.col('prev_slow'))).alias('entry_signal'),
            ((pl.col('fast_ma') < pl.col('slow_ma')) & (pl.col('prev_fast') >= pl.col('prev_slow'))).alias('exit_signal')
        ])
        
        gap = abs(pl.col('fast_ma') - pl.col('slow_ma')) / pl.col('mid_price')
        df = df.with_columns(
            (gap * 5000).clip(0, 100).alias('confidence')
        )
        
        return df

    def plot_indicators(self, data: pl.DataFrame, market: str, min_confidence: float = 30) -> go.Figure:
        market_data = data.filter(pl.col('market') == market).to_pandas()
        market_data['ts_utc'] = market_data['ts_utc'].dt.to_pydatetime()
        market_data = market_data.set_index('ts_utc')

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['mid_price'], name='Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['fast_ma'], name=f'Fast MA ({self.fast_window})', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['slow_ma'], name=f'Slow MA ({self.slow_window})', line=dict(color='green', dash='dash')))
        
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
    
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        # Calculate RSI using a more straightforward approach
        df = data.with_columns([
            pl.col('mid_price').diff(1).over('market').alias('price_delta')
        ])
        
        df = df.with_columns([
            pl.col('price_delta').clip(lower_bound=0).alias('gain'),
            (-pl.col('price_delta')).clip(lower_bound=0).alias('loss')
        ])
        
        df = df.with_columns([
            pl.col('gain').ewm_mean(alpha=1/self.period, adjust=False).over('market').alias('avg_gain'),
            pl.col('loss').ewm_mean(alpha=1/self.period, adjust=False).over('market').alias('avg_loss')
        ])
        
        df = df.with_columns([
            (pl.col('avg_gain') / pl.col('avg_loss')).alias('rs')
        ])
        
        df = df.with_columns([
            (100 - (100 / (1 + pl.col('rs')))).alias('rsi')
        ])
        
        df = df.with_columns(
            pl.col('rsi').shift(1).over('market').alias('prev_rsi')
        )
        
        df = df.with_columns([
            ((pl.col('rsi') > self.oversold) & (pl.col('prev_rsi') <= self.oversold)).alias('entry_signal'),
            ((pl.col('rsi') < self.overbought) & (pl.col('prev_rsi') >= self.overbought)).alias('exit_signal')
        ])
        
        df = df.with_columns(
            (abs(pl.col('rsi') - 50) * 2).clip(0, 100).alias('confidence')
        )
        
        return df

    def plot_indicators(self, data: pl.DataFrame, market: str, min_confidence: float = 30) -> go.Figure:
        market_data = data.filter(pl.col('market') == market).to_pandas()
        market_data['ts_utc'] = market_data['ts_utc'].dt.to_pydatetime()
        market_data = market_data.set_index('ts_utc')

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f'{market} Price', f'{market} RSI'))
        
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['mid_price'], name='Price', line=dict(color='blue')), 
                      row=1, col=1)

        entry_points = market_data[(market_data['entry_signal']) & (market_data['confidence'] >= min_confidence)]
        exit_points = market_data[(market_data['exit_signal']) & (market_data['confidence'] >= min_confidence)]
        
        fig.add_trace(go.Scatter(x=entry_points.index, y=entry_points['mid_price'], mode='markers', name='Entry Signal (Confident)', 
                                 marker=dict(color='green', size=10, symbol='triangle-up'), legendgroup='signals', legendgrouptitle_text='Signals'), 
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=exit_points.index, y=exit_points['mid_price'], mode='markers', name='Exit Signal (Confident)', 
                                 marker=dict(color='red', size=10, symbol='triangle-down'), legendgroup='signals'), 
                      row=1, col=1)
        
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['rsi'], name='RSI', line=dict(color='purple')), 
                      row=2, col=1)
        
        fig.add_hline(y=self.overbought, line_dash="dash", line_color="red", annotation_text="Overbought", 
                      annotation_position="bottom right", row=2, col=1)
        fig.add_hline(y=self.oversold, line_dash="dash", line_color="green", annotation_text="Oversold", 
                      annotation_position="bottom right", row=2, col=1)
        
        fig.update_layout(height=700, title_text=f'{market} RSI Analysis', legend_title_text='Indicators')
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="RSI Value", row=2, col=1, range=[0, 100])
        
        return fig

class GridStrategy(TradingStrategy):
    def __init__(self, grid_size: float = 0.1, sma_fast: int = 20, sma_slow: int = 50, trend_threshold: float = 0.005):
        self.grid_size = grid_size
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.trend_threshold = trend_threshold

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        # 1. Trend Filter
        df = data.with_columns([
            pl.col('mid_price').rolling_mean(self.sma_fast).over('market').alias('sma_fast'),
            pl.col('mid_price').rolling_mean(self.sma_slow).over('market').alias('sma_slow')
        ])
        df = df.with_columns(
            (abs(pl.col('sma_fast') - pl.col('sma_slow')) / pl.col('mid_price')).alias('trend_strength')
        )
        df = df.with_columns(
            (pl.col('trend_strength') < self.trend_threshold).alias('is_ranging')
        )

        # 2. Grid Logic (centered on the slow SMA)
        df = df.with_columns(
            ((pl.col('mid_price') - pl.col('sma_slow')) / self.grid_size).floor().alias('grid_index')
        )
        df = df.with_columns(
            pl.col('grid_index').shift(1).over('market').alias('prev_grid_index')
        )

        # 3. Generate Signals (Buy low, Sell high, only when ranging)
        df = df.with_columns([
            ((pl.col('grid_index') < pl.col('prev_grid_index')) & pl.col('is_ranging')).alias('entry_signal'), # Buy on drop
            ((pl.col('grid_index') > pl.col('prev_grid_index')) & pl.col('is_ranging')).alias('exit_signal')   # Sell on rise
        ])
        
        df = df.with_columns(
            pl.lit(100).alias('confidence')
        )
        
        return df

    def plot_indicators(self, data: pl.DataFrame, market: str, min_confidence: float = 0) -> go.Figure:
        market_data = data.filter(pl.col('market') == market).to_pandas()
        market_data['ts_utc'] = market_data['ts_utc'].dt.to_pydatetime()
        market_data = market_data.set_index('ts_utc')

        market_data['entry_signal'] = market_data['entry_signal'].fillna(False)
        market_data['exit_signal'] = market_data['exit_signal'].fillna(False)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f'{market} Price and Signals', 'Trend Strength'))

        # Plot price and SMAs
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['mid_price'], name='Price', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['sma_fast'], name=f'SMA ({self.sma_fast})', line=dict(color='orange', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['sma_slow'], name=f'SMA ({self.sma_slow})', line=dict(color='purple', dash='dash')), row=1, col=1)

        # Plot signals
        entry_points = market_data[market_data['entry_signal']]
        exit_points = market_data[market_data['exit_signal']]
        
        fig.add_trace(go.Scatter(x=entry_points.index, y=entry_points['mid_price'], mode='markers', name='Buy Signal', 
                                 marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
        fig.add_trace(go.Scatter(x=exit_points.index, y=exit_points['mid_price'], mode='markers', name='Sell Signal', 
                                 marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

        # Plot Trend Strength
        fig.add_trace(go.Scatter(x=market_data.index, y=market_data['trend_strength'], name='Trend Strength', line=dict(color='grey')), row=2, col=1)
        fig.add_hline(y=self.trend_threshold, line_dash="dash", line_color="red", annotation_text="Ranging Threshold", row=2, col=1)
        
        fig.update_layout(title=f'{market} Grid Strategy Analysis', height=700)
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Normalized Strength", row=2, col=1)
        return fig
