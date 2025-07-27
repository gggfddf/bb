#!/usr/bin/env python3
"""
Performance Visualization Components Module

Implements comprehensive performance visualization components for trading strategies:
- Equity curve visualization
- Drawdown analysis charts
- Performance metrics dashboards
- Risk-return scatter plots
- Returns heatmaps
- Strategy comparison visualizations

Features:
- Interactive equity curves with zoom and pan
- Comprehensive drawdown analysis
- Performance metrics widgets and dashboards
- Risk-return visualization
- Returns distribution and heatmaps
- Multi-strategy comparison tools
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

logger = structlog.get_logger()

class ChartType(Enum):
    """Types of performance charts."""
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    RETURNS_DISTRIBUTION = "returns_distribution"
    RISK_RETURN = "risk_return"
    HEATMAP = "heatmap"
    COMPARISON = "comparison"
    METRICS_DASHBOARD = "metrics_dashboard"

class VisualizationEngine(Enum):
    """Visualization engines."""
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    SEABORN = "seaborn"

@dataclass
class ChartConfig:
    """Configuration for performance charts."""
    chart_type: ChartType
    engine: VisualizationEngine = VisualizationEngine.PLOTLY
    figsize: Tuple[int, int] = (12, 8)
    theme: str = "default"
    interactive: bool = True
    save_path: Optional[str] = None
    dpi: int = 300

@dataclass
class PerformanceData:
    """Performance data for visualization."""
    strategy_name: str
    returns: np.ndarray
    dates: Optional[np.ndarray] = None
    equity_curve: Optional[np.ndarray] = None
    drawdown: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, float]] = None

class EquityCurveVisualizer:
    """Visualize equity curves."""
    
    def __init__(self, config: ChartConfig):
        """
        Initialize equity curve visualizer.
        
        Args:
            config: Chart configuration
        """
        self.config = config
    
    def plot_equity_curve(self, performance_data: List[PerformanceData], 
                         show_drawdown: bool = True) -> Any:
        """
        Plot equity curve(s).
        
        Args:
            performance_data: List of performance data
            show_drawdown: Whether to show drawdown subplot
            
        Returns:
            Plot object
        """
        if self.config.engine == VisualizationEngine.PLOTLY:
            return self._plot_equity_curve_plotly(performance_data, show_drawdown)
        else:
            return self._plot_equity_curve_matplotlib(performance_data, show_drawdown)
    
    def _plot_equity_curve_plotly(self, performance_data: List[PerformanceData], 
                                 show_drawdown: bool) -> go.Figure:
        """Plot equity curve using Plotly."""
        if show_drawdown:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Equity Curve', 'Drawdown'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, data in enumerate(performance_data):
            # Calculate equity curve if not provided
            if data.equity_curve is None:
                equity_curve = np.cumprod(1 + data.returns)
            else:
                equity_curve = data.equity_curve
            
            # Create dates if not provided
            if data.dates is None:
                dates = pd.date_range(start='2020-01-01', periods=len(equity_curve), freq='D')
            else:
                dates = data.dates
            
            # Plot equity curve
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=equity_curve,
                    mode='lines',
                    name=f'{data.strategy_name} - Equity',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            if show_drawdown:
                # Calculate drawdown if not provided
                if data.drawdown is None:
                    drawdown = self._calculate_drawdown(equity_curve)
                else:
                    drawdown = data.drawdown
                
                # Plot drawdown
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=drawdown,
                        mode='lines',
                        name=f'{data.strategy_name} - Drawdown',
                        line=dict(color=colors[i % len(colors)], dash='dash'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title='Strategy Performance - Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if show_drawdown:
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def _plot_equity_curve_matplotlib(self, performance_data: List[PerformanceData], 
                                     show_drawdown: bool) -> plt.Figure:
        """Plot equity curve using Matplotlib."""
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.config.figsize)
            ax2 = None
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(performance_data)))
        
        for i, data in enumerate(performance_data):
            # Calculate equity curve if not provided
            if data.equity_curve is None:
                equity_curve = np.cumprod(1 + data.returns)
            else:
                equity_curve = data.equity_curve
            
            # Create dates if not provided
            if data.dates is None:
                dates = pd.date_range(start='2020-01-01', periods=len(equity_curve), freq='D')
            else:
                dates = data.dates
            
            # Plot equity curve
            ax1.plot(dates, equity_curve, label=data.strategy_name, 
                    color=colors[i], linewidth=2)
            
            if show_drawdown and ax2 is not None:
                # Calculate drawdown if not provided
                if data.drawdown is None:
                    drawdown = self._calculate_drawdown(equity_curve)
                else:
                    drawdown = data.drawdown
                
                # Plot drawdown
                ax2.fill_between(dates, drawdown, 0, alpha=0.3, color=colors[i])
                ax2.plot(dates, drawdown, color=colors[i], linewidth=1)
        
        # Format equity curve subplot
        ax1.set_title('Strategy Performance - Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format drawdown subplot
        if ax2 is not None:
            ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def _calculate_drawdown(self, equity_curve: np.ndarray) -> np.ndarray:
        """Calculate drawdown series."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        return drawdown

class DrawdownAnalyzer:
    """Analyze and visualize drawdowns."""
    
    def __init__(self, config: ChartConfig):
        """
        Initialize drawdown analyzer.
        
        Args:
            config: Chart configuration
        """
        self.config = config
    
    def plot_drawdown_analysis(self, performance_data: List[PerformanceData]) -> Any:
        """
        Plot comprehensive drawdown analysis.
        
        Args:
            performance_data: List of performance data
            
        Returns:
            Plot object
        """
        if self.config.engine == VisualizationEngine.PLOTLY:
            return self._plot_drawdown_analysis_plotly(performance_data)
        else:
            return self._plot_drawdown_analysis_matplotlib(performance_data)
    
    def _plot_drawdown_analysis_plotly(self, performance_data: List[PerformanceData]) -> go.Figure:
        """Plot drawdown analysis using Plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Drawdown Over Time', 'Drawdown Distribution', 
                          'Underwater Periods', 'Recovery Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, data in enumerate(performance_data):
            # Calculate equity curve and drawdown
            if data.equity_curve is None:
                equity_curve = np.cumprod(1 + data.returns)
            else:
                equity_curve = data.equity_curve
            
            if data.dates is None:
                dates = pd.date_range(start='2020-01-01', periods=len(equity_curve), freq='D')
            else:
                dates = data.dates
            
            drawdown = self._calculate_drawdown(equity_curve)
            
            # Plot drawdown over time
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=drawdown,
                    mode='lines',
                    name=f'{data.strategy_name}',
                    line=dict(color=colors[i % len(colors)]),
                    fill='tonexty',
                    fillcolor=colors[i % len(colors)],
                    opacity=0.3
                ),
                row=1, col=1
            )
            
            # Plot drawdown distribution
            fig.add_trace(
                go.Histogram(
                    x=drawdown,
                    name=f'{data.strategy_name}',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=2
            )
            
            # Analyze underwater periods
            underwater_periods = self._analyze_underwater_periods(drawdown)
            
            # Plot underwater periods
            for period in underwater_periods:
                fig.add_trace(
                    go.Scatter(
                        x=[period['start'], period['end']],
                        y=[period['max_dd'], period['max_dd']],
                        mode='markers+lines',
                        name=f"{data.strategy_name} - Underwater",
                        line=dict(color=colors[i % len(colors)], width=3),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title='Drawdown Analysis',
            template='plotly_white',
            height=800
        )
        
        # Update subplot titles
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
        fig.update_xaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Max Drawdown (%)", row=2, col=1)
        
        return fig
    
    def _plot_drawdown_analysis_matplotlib(self, performance_data: List[PerformanceData]) -> plt.Figure:
        """Plot drawdown analysis using Matplotlib."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(performance_data)))
        
        for i, data in enumerate(performance_data):
            # Calculate equity curve and drawdown
            if data.equity_curve is None:
                equity_curve = np.cumprod(1 + data.returns)
            else:
                equity_curve = data.equity_curve
            
            if data.dates is None:
                dates = pd.date_range(start='2020-01-01', periods=len(equity_curve), freq='D')
            else:
                dates = data.dates
            
            drawdown = self._calculate_drawdown(equity_curve)
            
            # Plot drawdown over time
            ax1.fill_between(dates, drawdown, 0, alpha=0.3, color=colors[i])
            ax1.plot(dates, drawdown, color=colors[i], linewidth=2, label=data.strategy_name)
            
            # Plot drawdown distribution
            ax2.hist(drawdown, bins=30, alpha=0.7, color=colors[i], label=data.strategy_name)
        
        # Format subplots
        ax1.set_title('Drawdown Over Time', fontweight='bold')
        ax1.set_ylabel('Drawdown (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        ax2.set_title('Drawdown Distribution', fontweight='bold')
        ax2.set_xlabel('Drawdown (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _calculate_drawdown(self, equity_curve: np.ndarray) -> np.ndarray:
        """Calculate drawdown series."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        return drawdown
    
    def _analyze_underwater_periods(self, drawdown: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze underwater periods."""
        underwater_periods = []
        in_underwater = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_underwater:
                # Start of underwater period
                in_underwater = True
                start_idx = i
            elif dd >= 0 and in_underwater:
                # End of underwater period
                in_underwater = False
                period_drawdown = drawdown[start_idx:i]
                underwater_periods.append({
                    'start': start_idx,
                    'end': i,
                    'max_dd': np.min(period_drawdown),
                    'duration': i - start_idx
                })
        
        return underwater_periods

class RiskReturnVisualizer:
    """Visualize risk-return relationships."""
    
    def __init__(self, config: ChartConfig):
        """
        Initialize risk-return visualizer.
        
        Args:
            config: Chart configuration
        """
        self.config = config
    
    def plot_risk_return_scatter(self, performance_data: List[PerformanceData]) -> Any:
        """
        Plot risk-return scatter plot.
        
        Args:
            performance_data: List of performance data
            
        Returns:
            Plot object
        """
        if self.config.engine == VisualizationEngine.PLOTLY:
            return self._plot_risk_return_plotly(performance_data)
        else:
            return self._plot_risk_return_matplotlib(performance_data)
    
    def _plot_risk_return_plotly(self, performance_data: List[PerformanceData]) -> go.Figure:
        """Plot risk-return scatter using Plotly."""
        fig = go.Figure()
        
        for data in performance_data:
            # Calculate metrics
            total_return = np.prod(1 + data.returns) - 1
            volatility = np.std(data.returns) * np.sqrt(252)
            sharpe_ratio = np.mean(data.returns) / np.std(data.returns) * np.sqrt(252) if np.std(data.returns) > 0 else 0
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=[volatility * 100],  # Convert to percentage
                    y=[total_return * 100],  # Convert to percentage
                    mode='markers+text',
                    name=data.strategy_name,
                    text=[data.strategy_name],
                    textposition="top center",
                    marker=dict(
                        size=15,
                        color=sharpe_ratio,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio")
                    ),
                    hovertemplate=f"Strategy: {data.strategy_name}<br>" +
                                f"Return: %{{y:.2f}}%<br>" +
                                f"Volatility: %{{x:.2f}}%<br>" +
                                f"Sharpe: {sharpe_ratio:.2f}<extra></extra>"
                )
            )
        
        # Add efficient frontier line (simplified)
        volatilities = np.linspace(0, 30, 100)
        efficient_frontier = volatilities * 0.5  # Simplified efficient frontier
        
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=efficient_frontier,
                mode='lines',
                name='Efficient Frontier',
                line=dict(dash='dash', color='gray'),
                showlegend=True
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Volatility (%)',
            yaxis_title='Total Return (%)',
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    
    def _plot_risk_return_matplotlib(self, performance_data: List[PerformanceData]) -> plt.Figure:
        """Plot risk-return scatter using Matplotlib."""
        fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(performance_data)))
        
        for i, data in enumerate(performance_data):
            # Calculate metrics
            total_return = np.prod(1 + data.returns) - 1
            volatility = np.std(data.returns) * np.sqrt(252)
            sharpe_ratio = np.mean(data.returns) / np.std(data.returns) * np.sqrt(252) if np.std(data.returns) > 0 else 0
            
            # Create scatter plot
            scatter = ax.scatter(
                volatility * 100,  # Convert to percentage
                total_return * 100,  # Convert to percentage
                c=[sharpe_ratio],
                s=100,
                cmap='RdYlGn',
                alpha=0.7,
                label=data.strategy_name
            )
            
            # Add strategy name
            ax.annotate(data.strategy_name, 
                       (volatility * 100, total_return * 100),
                       xytext=(5, 5), textcoords='offset points')
        
        # Add efficient frontier line (simplified)
        volatilities = np.linspace(0, 30, 100)
        efficient_frontier = volatilities * 0.5  # Simplified efficient frontier
        ax.plot(volatilities, efficient_frontier, '--', color='gray', alpha=0.7, label='Efficient Frontier')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio')
        
        # Format plot
        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk-Return Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig

class ReturnsHeatmapVisualizer:
    """Visualize returns heatmaps."""
    
    def __init__(self, config: ChartConfig):
        """
        Initialize returns heatmap visualizer.
        
        Args:
            config: Chart configuration
        """
        self.config = config
    
    def plot_returns_heatmap(self, performance_data: List[PerformanceData]) -> Any:
        """
        Plot returns heatmap.
        
        Args:
            performance_data: List of performance data
            
        Returns:
            Plot object
        """
        if self.config.engine == VisualizationEngine.PLOTLY:
            return self._plot_returns_heatmap_plotly(performance_data)
        else:
            return self._plot_returns_heatmap_matplotlib(performance_data)
    
    def _plot_returns_heatmap_plotly(self, performance_data: List[PerformanceData]) -> go.Figure:
        """Plot returns heatmap using Plotly."""
        # Create monthly returns matrix
        monthly_returns = {}
        
        for data in performance_data:
            if data.dates is None:
                dates = pd.date_range(start='2020-01-01', periods=len(data.returns), freq='D')
            else:
                dates = data.dates
            
            # Convert to pandas series and resample to monthly
            returns_series = pd.Series(data.returns, index=dates)
            monthly_ret = returns_series.resample('M').apply(lambda x: np.prod(1 + x) - 1)
            monthly_returns[data.strategy_name] = monthly_ret
        
        # Create heatmap data
        df_monthly = pd.DataFrame(monthly_returns)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=df_monthly.values,
            x=df_monthly.columns,
            y=df_monthly.index.strftime('%Y-%m'),
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(df_monthly.values * 100, 2),
            texttemplate="%{text}%",
            textfont={"size": 10},
            colorbar=dict(title="Monthly Return (%)")
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Strategy',
            yaxis_title='Month',
            template='plotly_white'
        )
        
        return fig
    
    def _plot_returns_heatmap_matplotlib(self, performance_data: List[PerformanceData]) -> plt.Figure:
        """Plot returns heatmap using Matplotlib."""
        # Create monthly returns matrix
        monthly_returns = {}
        
        for data in performance_data:
            if data.dates is None:
                dates = pd.date_range(start='2020-01-01', periods=len(data.returns), freq='D')
            else:
                dates = data.dates
            
            # Convert to pandas series and resample to monthly
            returns_series = pd.Series(data.returns, index=dates)
            monthly_ret = returns_series.resample('M').apply(lambda x: np.prod(1 + x) - 1)
            monthly_returns[data.strategy_name] = monthly_ret
        
        # Create heatmap data
        df_monthly = pd.DataFrame(monthly_returns)
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)
        
        im = ax.imshow(df_monthly.values * 100, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        # Set labels
        ax.set_xticks(range(len(df_monthly.columns)))
        ax.set_xticklabels(df_monthly.columns, rotation=45)
        ax.set_yticks(range(len(df_monthly.index)))
        ax.set_yticklabels(df_monthly.index.strftime('%Y-%m'))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Monthly Return (%)')
        
        # Add text annotations
        for i in range(len(df_monthly.index)):
            for j in range(len(df_monthly.columns)):
                text = ax.text(j, i, f'{df_monthly.iloc[i, j]*100:.1f}%',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Monthly Returns Heatmap')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Month')
        
        plt.tight_layout()
        return fig

class PerformanceDashboard:
    """Comprehensive performance dashboard."""
    
    def __init__(self, config: ChartConfig):
        """
        Initialize performance dashboard.
        
        Args:
            config: Chart configuration
        """
        self.config = config
        self.equity_visualizer = EquityCurveVisualizer(config)
        self.drawdown_analyzer = DrawdownAnalyzer(config)
        self.risk_return_visualizer = RiskReturnVisualizer(config)
        self.heatmap_visualizer = ReturnsHeatmapVisualizer(config)
    
    def create_dashboard(self, performance_data: List[PerformanceData]) -> Any:
        """
        Create comprehensive performance dashboard.
        
        Args:
            performance_data: List of performance data
            
        Returns:
            Dashboard object
        """
        if self.config.engine == VisualizationEngine.PLOTLY:
            return self._create_dashboard_plotly(performance_data)
        else:
            return self._create_dashboard_matplotlib(performance_data)
    
    def _create_dashboard_plotly(self, performance_data: List[PerformanceData]) -> go.Figure:
        """Create dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curves', 'Risk-Return Analysis', 
                          'Drawdown Analysis', 'Monthly Returns Heatmap',
                          'Performance Metrics', 'Returns Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add equity curves
        equity_fig = self.equity_visualizer.plot_equity_curve(performance_data, show_drawdown=False)
        for trace in equity_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Add risk-return scatter
        risk_return_fig = self.risk_return_visualizer.plot_risk_return_scatter(performance_data)
        for trace in risk_return_fig.data:
            fig.add_trace(trace, row=1, col=2)
        
        # Add drawdown analysis
        drawdown_fig = self.drawdown_analyzer.plot_drawdown_analysis(performance_data)
        for trace in drawdown_fig.data[:len(performance_data)]:  # Only main drawdown traces
            fig.add_trace(trace, row=2, col=1)
        
        # Add returns distribution
        for data in performance_data:
            fig.add_trace(
                go.Histogram(
                    x=data.returns * 100,
                    name=f'{data.strategy_name} - Returns',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Performance Dashboard',
            template='plotly_white',
            height=1200,
            showlegend=True
        )
        
        return fig
    
    def _create_dashboard_matplotlib(self, performance_data: List[PerformanceData]) -> plt.Figure:
        """Create dashboard using Matplotlib."""
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Equity curves
        ax1 = fig.add_subplot(gs[0, 0])
        equity_fig = self.equity_visualizer.plot_equity_curve(performance_data, show_drawdown=False)
        # Copy content from equity_fig to ax1
        for line in equity_fig.axes[0].lines:
            ax1.plot(line.get_xdata(), line.get_ydata(), 
                    color=line.get_color(), label=line.get_label())
        ax1.set_title('Equity Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Risk-return scatter
        ax2 = fig.add_subplot(gs[0, 1])
        risk_return_fig = self.risk_return_visualizer.plot_risk_return_scatter(performance_data)
        # Copy content from risk_return_fig to ax2
        for collection in risk_return_fig.axes[0].collections:
            ax2.scatter(collection.get_offsets()[:, 0], collection.get_offsets()[:, 1],
                       c=collection.get_array(), cmap=collection.get_cmap())
        ax2.set_title('Risk-Return Analysis')
        ax2.grid(True, alpha=0.3)
        
        # Drawdown analysis
        ax3 = fig.add_subplot(gs[1, 0])
        drawdown_fig = self.drawdown_analyzer.plot_drawdown_analysis(performance_data)
        # Copy content from drawdown_fig to ax3
        for line in drawdown_fig.axes[0, 0].lines:
            ax3.plot(line.get_xdata(), line.get_ydata(), 
                    color=line.get_color(), label=line.get_label())
        ax3.set_title('Drawdown Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Returns distribution
        ax4 = fig.add_subplot(gs[1, 1])
        for data in performance_data:
            ax4.hist(data.returns * 100, bins=30, alpha=0.7, label=data.strategy_name)
        ax4.set_title('Returns Distribution')
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Performance metrics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Calculate metrics
        metrics_data = []
        for data in performance_data:
            total_return = np.prod(1 + data.returns) - 1
            volatility = np.std(data.returns) * np.sqrt(252)
            sharpe_ratio = np.mean(data.returns) / np.std(data.returns) * np.sqrt(252) if np.std(data.returns) > 0 else 0
            max_dd = self._calculate_max_drawdown(np.cumprod(1 + data.returns))
            
            metrics_data.append([
                data.strategy_name,
                f'{total_return*100:.2f}%',
                f'{volatility*100:.2f}%',
                f'{sharpe_ratio:.2f}',
                f'{max_dd*100:.2f}%'
            ])
        
        table = ax5.table(cellText=metrics_data,
                         colLabels=['Strategy', 'Total Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        return fig
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return -np.min(drawdown)

def create_performance_visualizer(chart_type: str = "equity_curve",
                                engine: str = "plotly") -> Any:
    """
    Create a performance visualizer.
    
    Args:
        chart_type: Type of chart to create
        engine: Visualization engine
        
    Returns:
        Visualizer instance
    """
    config = ChartConfig(
        chart_type=ChartType(chart_type),
        engine=VisualizationEngine(engine)
    )
    
    if chart_type == "equity_curve":
        return EquityCurveVisualizer(config)
    elif chart_type == "drawdown":
        return DrawdownAnalyzer(config)
    elif chart_type == "risk_return":
        return RiskReturnVisualizer(config)
    elif chart_type == "heatmap":
        return ReturnsHeatmapVisualizer(config)
    elif chart_type == "dashboard":
        return PerformanceDashboard(config)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

if __name__ == "__main__":
    # Demo of performance visualization
    np.random.seed(42)
    
    # Generate sample performance data
    n_days = 252
    performance_data = []
    
    strategy_names = ['Strategy_A', 'Strategy_B', 'Strategy_C']
    
    for i, name in enumerate(strategy_names):
        # Generate returns with different characteristics
        base_return = 0.0001 * (i + 1)
        volatility = 0.02 + 0.005 * i
        returns = np.random.normal(base_return, volatility, n_days)
        
        data = PerformanceData(
            strategy_name=name,
            returns=returns
        )
        performance_data.append(data)
    
    # Create visualizer
    visualizer = create_performance_visualizer("dashboard", "plotly")
    
    # Create dashboard
    dashboard = visualizer.create_dashboard(performance_data)
    
    # Show dashboard
    dashboard.show()