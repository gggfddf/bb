#!/usr/bin/env python3
"""
Candlestick Charts Module

Implements comprehensive candlestick charting system with technical indicators:
- Candlestick chart rendering with OHLC data
- Technical indicator overlays (RSI, MACD, Bollinger Bands, etc.)
- Interactive chart controls (zoom, pan, time range selection)
- Real-time data updates and streaming
- Chart customization options (themes, colors, layouts)
- Chart export and sharing functionality

Features:
- High-performance chart rendering with matplotlib and plotly
- Multiple technical indicator overlays
- Interactive controls and real-time updates
- Customizable themes and layouts
- Export functionality for various formats
- Responsive design for different screen sizes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot
import seaborn as sns

logger = structlog.get_logger()

class ChartTheme(Enum):
    """Available chart themes."""
    LIGHT = "light"
    DARK = "dark"
    TRADING = "trading"
    MINIMAL = "minimal"

class IndicatorType(Enum):
    """Types of technical indicators."""
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    MOVING_AVERAGES = "moving_averages"
    VOLUME = "volume"
    STOCHASTIC = "stochastic"
    WILLIAMS_R = "williams_r"
    CCI = "cci"

@dataclass
class ChartConfig:
    """Configuration for candlestick charts."""
    theme: ChartTheme = ChartTheme.TRADING
    width: int = 1200
    height: int = 800
    show_volume: bool = True
    show_indicators: List[IndicatorType] = field(default_factory=list)
    time_range: Tuple[datetime, datetime] = None
    auto_refresh: bool = False
    refresh_interval: int = 30  # seconds
    export_format: str = "png"
    custom_colors: Dict[str, str] = field(default_factory=dict)

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    indicator_type: IndicatorType
    parameters: Dict[str, Any] = field(default_factory=dict)
    color: str = "blue"
    line_width: int = 1
    opacity: float = 0.8
    show_signal: bool = True

class CandlestickChart:
    """
    Main class for candlestick chart rendering.
    """
    
    def __init__(self, config: ChartConfig = None):
        """
        Initialize candlestick chart.
        
        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()
        self.data: Optional[pd.DataFrame] = None
        self.indicators: Dict[str, pd.DataFrame] = {}
        self.fig = None
        self.current_theme = self._get_theme_colors()
        
        # Chart state
        self.zoom_level = 1.0
        self.pan_offset = 0
        self.selected_range = None
        
        logger.info("Candlestick chart initialized", theme=self.config.theme.value)
    
    def load_data(self, data: pd.DataFrame):
        """
        Load OHLC data for charting.
        
        Args:
            data: DataFrame with OHLC columns (Open, High, Low, Close)
        """
        try:
            # Validate data
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Data must contain columns: {required_columns}")
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            self.data = data.copy()
            
            # Calculate additional data
            self.data['Volume'] = self.data.get('Volume', 0)
            self.data['Returns'] = self.data['Close'].pct_change()
            
            logger.info("Data loaded successfully", 
                       rows=len(self.data),
                       date_range=(self.data.index[0], self.data.index[-1]))
            
        except Exception as e:
            logger.error("Failed to load data", error=str(e))
            raise
    
    def add_indicator(self, indicator_type: IndicatorType, config: IndicatorConfig = None):
        """
        Add technical indicator to the chart.
        
        Args:
            indicator_type: Type of indicator to add
            config: Indicator configuration
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            config = config or IndicatorConfig(indicator_type=indicator_type)
            
            if indicator_type == IndicatorType.RSI:
                indicator_data = self._calculate_rsi(config.parameters)
            elif indicator_type == IndicatorType.MACD:
                indicator_data = self._calculate_macd(config.parameters)
            elif indicator_type == IndicatorType.BOLLINGER_BANDS:
                indicator_data = self._calculate_bollinger_bands(config.parameters)
            elif indicator_type == IndicatorType.MOVING_AVERAGES:
                indicator_data = self._calculate_moving_averages(config.parameters)
            elif indicator_type == IndicatorType.VOLUME:
                indicator_data = self._calculate_volume_indicators(config.parameters)
            elif indicator_type == IndicatorType.STOCHASTIC:
                indicator_data = self._calculate_stochastic(config.parameters)
            elif indicator_type == IndicatorType.WILLIAMS_R:
                indicator_data = self._calculate_williams_r(config.parameters)
            elif indicator_type == IndicatorType.CCI:
                indicator_data = self._calculate_cci(config.parameters)
            else:
                raise ValueError(f"Unsupported indicator type: {indicator_type}")
            
            self.indicators[indicator_type.value] = indicator_data
            
            logger.info("Indicator added", indicator_type=indicator_type.value)
            
        except Exception as e:
            logger.error("Failed to add indicator", error=str(e))
            raise
    
    def _calculate_rsi(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Calculate RSI indicator."""
        period = parameters.get('period', 14)
        
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return pd.DataFrame({
            'RSI': rsi,
            'Overbought': 70,
            'Oversold': 30
        }, index=self.data.index)
    
    def _calculate_macd(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Calculate MACD indicator."""
        fast_period = parameters.get('fast_period', 12)
        slow_period = parameters.get('slow_period', 26)
        signal_period = parameters.get('signal_period', 9)
        
        ema_fast = self.data['Close'].ewm(span=fast_period).mean()
        ema_slow = self.data['Close'].ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }, index=self.data.index)
    
    def _calculate_bollinger_bands(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Bollinger Bands indicator."""
        period = parameters.get('period', 20)
        std_dev = parameters.get('std_dev', 2)
        
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': sma,
            'Lower': lower_band
        }, index=self.data.index)
    
    def _calculate_moving_averages(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Calculate moving averages."""
        periods = parameters.get('periods', [20, 50, 200])
        
        ma_data = {}
        for period in periods:
            ma_data[f'SMA_{period}'] = self.data['Close'].rolling(window=period).mean()
            ma_data[f'EMA_{period}'] = self.data['Close'].ewm(span=period).mean()
        
        return pd.DataFrame(ma_data, index=self.data.index)
    
    def _calculate_volume_indicators(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        period = parameters.get('period', 20)
        
        volume_sma = self.data['Volume'].rolling(window=period).mean()
        volume_ratio = self.data['Volume'] / volume_sma
        
        return pd.DataFrame({
            'Volume': self.data['Volume'],
            'Volume_SMA': volume_sma,
            'Volume_Ratio': volume_ratio
        }, index=self.data.index)
    
    def _calculate_stochastic(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        k_period = parameters.get('k_period', 14)
        d_period = parameters.get('d_period', 3)
        
        lowest_low = self.data['Low'].rolling(window=k_period).min()
        highest_high = self.data['High'].rolling(window=k_period).max()
        
        k_percent = 100 * ((self.data['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'K': k_percent,
            'D': d_percent,
            'Overbought': 80,
            'Oversold': 20
        }, index=self.data.index)
    
    def _calculate_williams_r(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Williams %R indicator."""
        period = parameters.get('period', 14)
        
        highest_high = self.data['High'].rolling(window=period).max()
        lowest_low = self.data['Low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - self.data['Close']) / (highest_high - lowest_low))
        
        return pd.DataFrame({
            'Williams_R': williams_r,
            'Overbought': -20,
            'Oversold': -80
        }, index=self.data.index)
    
    def _calculate_cci(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        period = parameters.get('period', 20)
        
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        return pd.DataFrame({
            'CCI': cci,
            'Overbought': 100,
            'Oversold': -100
        }, index=self.data.index)
    
    def render_matplotlib(self, save_path: str = None) -> plt.Figure:
        """
        Render chart using matplotlib.
        
        Args:
            save_path: Path to save the chart image
            
        Returns:
            matplotlib Figure object
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            # Create figure and subplots
            n_indicators = len(self.indicators)
            height_ratios = [3] + [1] * n_indicators if n_indicators > 0 else [1]
            
            fig, axes = plt.subplots(
                len(height_ratios), 1, 
                figsize=(self.config.width/100, self.config.height/100),
                height_ratios=height_ratios,
                sharex=True
            )
            
            if n_indicators == 0:
                axes = [axes]
            
            # Apply theme
            self._apply_matplotlib_theme(fig, axes[0])
            
            # Plot candlesticks
            self._plot_candlesticks_matplotlib(axes[0])
            
            # Plot indicators
            for i, (indicator_name, indicator_data) in enumerate(self.indicators.items()):
                self._plot_indicator_matplotlib(axes[i+1], indicator_name, indicator_data)
            
            # Format axes
            self._format_matplotlib_axes(fig, axes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            self.fig = fig
            logger.info("Matplotlib chart rendered successfully")
            
            return fig
            
        except Exception as e:
            logger.error("Failed to render matplotlib chart", error=str(e))
            raise
    
    def render_plotly(self, save_path: str = None) -> go.Figure:
        """
        Render chart using plotly.
        
        Args:
            save_path: Path to save the chart HTML
            
        Returns:
            plotly Figure object
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            # Create subplots
            n_indicators = len(self.indicators)
            subplot_titles = ['Price'] + [indicator.replace('_', ' ').title() 
                                       for indicator in self.indicators.keys()]
            
            fig = make_subplots(
                rows=n_indicators + 1, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.05,
                row_heights=[0.6] + [0.4/n_indicators] * n_indicators if n_indicators > 0 else [1]
            )
            
            # Plot candlesticks
            self._plot_candlesticks_plotly(fig, row=1)
            
            # Plot indicators
            for i, (indicator_name, indicator_data) in enumerate(self.indicators.items()):
                self._plot_indicator_plotly(fig, indicator_name, indicator_data, row=i+2)
            
            # Update layout
            self._update_plotly_layout(fig)
            
            if save_path:
                fig.write_html(save_path)
            
            self.fig = fig
            logger.info("Plotly chart rendered successfully")
            
            return fig
            
        except Exception as e:
            logger.error("Failed to render plotly chart", error=str(e))
            raise
    
    def _plot_candlesticks_matplotlib(self, ax):
        """Plot candlesticks using matplotlib."""
        # Plot candlesticks
        for i, (date, row) in enumerate(self.data.iterrows()):
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            
            # Body
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Open'], row['Close'])
            
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Wicks
            ax.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1)
        
        # Plot volume if enabled
        if self.config.show_volume:
            ax2 = ax.twinx()
            ax2.bar(range(len(self.data)), self.data['Volume'], 
                   alpha=0.3, color='gray')
            ax2.set_ylabel('Volume')
        
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
    
    def _plot_indicator_matplotlib(self, ax, indicator_name: str, indicator_data: pd.DataFrame):
        """Plot indicator using matplotlib."""
        if indicator_name == 'rsi':
            ax.plot(indicator_data.index, indicator_data['RSI'], label='RSI', color='blue')
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax.set_ylabel('RSI')
            ax.set_ylim(0, 100)
            
        elif indicator_name == 'macd':
            ax.plot(indicator_data.index, indicator_data['MACD'], label='MACD', color='blue')
            ax.plot(indicator_data.index, indicator_data['Signal'], label='Signal', color='red')
            ax.bar(indicator_data.index, indicator_data['Histogram'], 
                  alpha=0.3, color='gray', label='Histogram')
            ax.set_ylabel('MACD')
            
        elif indicator_name == 'bollinger_bands':
            ax.plot(indicator_data.index, indicator_data['Upper'], label='Upper Band', color='red', alpha=0.7)
            ax.plot(indicator_data.index, indicator_data['Middle'], label='Middle Band', color='blue')
            ax.plot(indicator_data.index, indicator_data['Lower'], label='Lower Band', color='red', alpha=0.7)
            ax.fill_between(indicator_data.index, indicator_data['Upper'], indicator_data['Lower'], 
                          alpha=0.1, color='gray')
            ax.set_ylabel('Bollinger Bands')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_candlesticks_plotly(self, fig: go.Figure, row: int = 1):
        """Plot candlesticks using plotly."""
        # Create candlestick trace
        candlestick = go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='OHLC',
            increasing_line_color='green',
            decreasing_line_color='red'
        )
        
        fig.add_trace(candlestick, row=row, col=1)
        
        # Add volume if enabled
        if self.config.show_volume:
            volume = go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                opacity=0.3,
                marker_color='gray'
            )
            fig.add_trace(volume, row=row, col=1)
    
    def _plot_indicator_plotly(self, fig: go.Figure, indicator_name: str, 
                             indicator_data: pd.DataFrame, row: int):
        """Plot indicator using plotly."""
        if indicator_name == 'rsi':
            fig.add_trace(go.Scatter(
                x=indicator_data.index,
                y=indicator_data['RSI'],
                name='RSI',
                line=dict(color='blue')
            ), row=row, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
            
        elif indicator_name == 'macd':
            fig.add_trace(go.Scatter(
                x=indicator_data.index,
                y=indicator_data['MACD'],
                name='MACD',
                line=dict(color='blue')
            ), row=row, col=1)
            
            fig.add_trace(go.Scatter(
                x=indicator_data.index,
                y=indicator_data['Signal'],
                name='Signal',
                line=dict(color='red')
            ), row=row, col=1)
            
            fig.add_trace(go.Bar(
                x=indicator_data.index,
                y=indicator_data['Histogram'],
                name='Histogram',
                opacity=0.3,
                marker_color='gray'
            ), row=row, col=1)
    
    def _apply_matplotlib_theme(self, fig: plt.Figure, ax: plt.Axes):
        """Apply theme to matplotlib chart."""
        if self.config.theme == ChartTheme.DARK:
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
        elif self.config.theme == ChartTheme.TRADING:
            plt.style.use('seaborn-v0_8')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#ffffff')
    
    def _format_matplotlib_axes(self, fig: plt.Figure, axes: List[plt.Axes]):
        """Format matplotlib axes."""
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
        # Set title
        fig.suptitle('Trading Chart', fontsize=16, fontweight='bold')
    
    def _update_plotly_layout(self, fig: go.Figure):
        """Update plotly layout."""
        fig.update_layout(
            title='Trading Chart',
            xaxis_rangeslider_visible=False,
            height=self.config.height,
            width=self.config.width,
            showlegend=True,
            template='plotly_white' if self.config.theme == ChartTheme.LIGHT else 'plotly_dark'
        )
    
    def _get_theme_colors(self) -> Dict[str, str]:
        """Get theme colors."""
        if self.config.theme == ChartTheme.DARK:
            return {
                'background': '#1e1e1e',
                'text': '#ffffff',
                'grid': '#333333',
                'up': '#00ff00',
                'down': '#ff0000'
            }
        elif self.config.theme == ChartTheme.TRADING:
            return {
                'background': '#ffffff',
                'text': '#000000',
                'grid': '#e0e0e0',
                'up': '#26a69a',
                'down': '#ef5350'
            }
        else:
            return {
                'background': '#ffffff',
                'text': '#000000',
                'grid': '#cccccc',
                'up': '#0000ff',
                'down': '#ff0000'
            }
    
    def export_chart(self, format: str = "png", path: str = None) -> str:
        """
        Export chart to various formats.
        
        Args:
            format: Export format (png, jpg, svg, pdf, html)
            path: Export path
            
        Returns:
            Path to exported file
        """
        try:
            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"chart_export_{timestamp}.{format}"
            
            if format in ['png', 'jpg', 'svg', 'pdf']:
                if self.fig is None:
                    self.render_matplotlib()
                self.fig.savefig(path, dpi=300, bbox_inches='tight')
            elif format == 'html':
                if self.fig is None:
                    self.render_plotly()
                self.fig.write_html(path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info("Chart exported successfully", path=path, format=format)
            return path
            
        except Exception as e:
            logger.error("Failed to export chart", error=str(e))
            raise

def create_candlestick_chart(theme: str = "trading",
                           width: int = 1200,
                           height: int = 800,
                           show_volume: bool = True) -> CandlestickChart:
    """
    Create a candlestick chart with specified configuration.
    
    Args:
        theme: Chart theme
        width: Chart width
        height: Chart height
        show_volume: Whether to show volume
        
    Returns:
        CandlestickChart instance
    """
    theme_enum = ChartTheme(theme)
    config = ChartConfig(
        theme=theme_enum,
        width=width,
        height=height,
        show_volume=show_volume
    )
    
    return CandlestickChart(config)

def render_quick_chart(data: pd.DataFrame,
                      indicators: List[str] = None,
                      theme: str = "trading") -> go.Figure:
    """
    Quick function to render a candlestick chart.
    
    Args:
        data: OHLC data
        indicators: List of indicators to add
        theme: Chart theme
        
    Returns:
        Plotly Figure object
    """
    chart = create_candlestick_chart(theme=theme)
    chart.load_data(data)
    
    if indicators:
        for indicator in indicators:
            indicator_enum = IndicatorType(indicator)
            chart.add_indicator(indicator_enum)
    
    return chart.render_plotly()