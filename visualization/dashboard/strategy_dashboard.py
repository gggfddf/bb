#!/usr/bin/env python3
"""
Strategy Performance Dashboard Module

Implements comprehensive strategy performance dashboard:
- Strategy overview panels
- Performance metrics widgets
- Risk analysis components
- Strategy comparison views
- Portfolio allocation charts
- Alert and notification panels

Features:
- Comprehensive strategy overview and monitoring
- Real-time performance metrics and KPIs
- Risk analysis and visualization
- Strategy comparison and benchmarking
- Portfolio allocation and asset distribution
- Alert and notification management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class DashboardComponent(Enum):
    """Dashboard component types."""
    OVERVIEW = "overview"
    PERFORMANCE = "performance"
    RISK = "risk"
    COMPARISON = "comparison"
    PORTFOLIO = "portfolio"
    ALERTS = "alerts"

class MetricType(Enum):
    """Metric types for dashboard."""
    RETURNS = "returns"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"

class AlertLevel(Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class DashboardConfig:
    """Configuration for dashboard."""
    components: List[DashboardComponent]
    refresh_interval: int = 30  # seconds
    max_strategies: int = 10
    enable_alerts: bool = True
    theme: str = "light"
    layout: str = "grid"

@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Alert:
    """Alert/notification for dashboard."""
    id: str
    level: AlertLevel
    message: str
    strategy_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class StrategyOverviewPanel:
    """Strategy overview panel component."""
    
    def __init__(self, strategies: List[str]):
        """
        Initialize strategy overview panel.
        
        Args:
            strategies: List of strategy names
        """
        self.strategies = strategies
        self.strategy_data = {}
        self.active_strategies = set()
    
    def update_strategy_data(self, strategy_name: str, metrics: StrategyMetrics):
        """Update strategy data."""
        self.strategy_data[strategy_name] = metrics
        self.active_strategies.add(strategy_name)
    
    def create_overview_chart(self) -> go.Figure:
        """Create overview chart."""
        if not self.strategy_data:
            return go.Figure()
        
        # Create summary metrics
        strategy_names = list(self.strategy_data.keys())
        total_returns = [self.strategy_data[name].total_return for name in strategy_names]
        sharpe_ratios = [self.strategy_data[name].sharpe_ratio for name in strategy_names]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Returns', 'Sharpe Ratios', 'Max Drawdown', 'Win Rate'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Total Returns
        fig.add_trace(
            go.Bar(x=strategy_names, y=total_returns, name='Total Return', 
                   marker_color='green'),
            row=1, col=1
        )
        
        # Sharpe Ratios
        fig.add_trace(
            go.Bar(x=strategy_names, y=sharpe_ratios, name='Sharpe Ratio',
                   marker_color='blue'),
            row=1, col=2
        )
        
        # Max Drawdown
        max_drawdowns = [self.strategy_data[name].max_drawdown for name in strategy_names]
        fig.add_trace(
            go.Bar(x=strategy_names, y=max_drawdowns, name='Max Drawdown',
                   marker_color='red'),
            row=2, col=1
        )
        
        # Win Rate
        win_rates = [self.strategy_data[name].win_rate for name in strategy_names]
        fig.add_trace(
            go.Bar(x=strategy_names, y=win_rates, name='Win Rate',
                   marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Strategy Overview",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.strategy_data:
            return {}
        
        returns = [metrics.total_return for metrics in self.strategy_data.values()]
        sharpe_ratios = [metrics.sharpe_ratio for metrics in self.strategy_data.values()]
        
        return {
            'total_strategies': len(self.strategy_data),
            'active_strategies': len(self.active_strategies),
            'avg_return': np.mean(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'best_sharpe': max(sharpe_ratios)
        }

class PerformanceMetricsWidget:
    """Performance metrics widget component."""
    
    def __init__(self):
        """Initialize performance metrics widget."""
        self.metrics_history = {}
        self.current_metrics = {}
    
    def add_metrics(self, strategy_name: str, metrics: StrategyMetrics):
        """Add metrics for strategy."""
        if strategy_name not in self.metrics_history:
            self.metrics_history[strategy_name] = []
        
        self.metrics_history[strategy_name].append(metrics)
        self.current_metrics[strategy_name] = metrics
        
        # Keep only recent history
        if len(self.metrics_history[strategy_name]) > 100:
            self.metrics_history[strategy_name] = self.metrics_history[strategy_name][-100:]
    
    def create_performance_chart(self, strategy_name: str) -> go.Figure:
        """Create performance chart for strategy."""
        if strategy_name not in self.metrics_history:
            return go.Figure()
        
        history = self.metrics_history[strategy_name]
        timestamps = [m.timestamp for m in history]
        returns = [m.total_return for m in history]
        sharpe_ratios = [m.sharpe_ratio for m in history]
        drawdowns = [m.max_drawdown for m in history]
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Cumulative Returns', 'Sharpe Ratio', 'Max Drawdown'),
            vertical_spacing=0.1
        )
        
        # Cumulative Returns
        fig.add_trace(
            go.Scatter(x=timestamps, y=returns, mode='lines', name='Returns',
                      line=dict(color='green')),
            row=1, col=1
        )
        
        # Sharpe Ratio
        fig.add_trace(
            go.Scatter(x=timestamps, y=sharpe_ratios, mode='lines', name='Sharpe',
                      line=dict(color='blue')),
            row=2, col=1
        )
        
        # Max Drawdown
        fig.add_trace(
            go.Scatter(x=timestamps, y=drawdowns, mode='lines', name='Drawdown',
                      line=dict(color='red'), fill='tonexty'),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f"Performance Metrics - {strategy_name}",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def get_metrics_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Get metrics summary for strategy."""
        if strategy_name not in self.current_metrics:
            return {}
        
        metrics = self.current_metrics[strategy_name]
        return {
            'total_return': f"{metrics.total_return:.2%}",
            'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
            'max_drawdown': f"{metrics.max_drawdown:.2%}",
            'volatility': f"{metrics.volatility:.2%}",
            'win_rate': f"{metrics.win_rate:.2%}",
            'profit_factor': f"{metrics.profit_factor:.2f}",
            'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
            'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
            'beta': f"{metrics.beta:.2f}",
            'alpha': f"{metrics.alpha:.2%}"
        }

class RiskAnalysisComponent:
    """Risk analysis component."""
    
    def __init__(self):
        """Initialize risk analysis component."""
        self.risk_metrics = {}
        self.var_data = {}
        self.correlation_data = {}
    
    def add_risk_data(self, strategy_name: str, returns: np.ndarray, 
                     var_confidence: float = 0.95):
        """Add risk data for strategy."""
        # Calculate risk metrics
        volatility = np.std(returns) * np.sqrt(252)
        var = np.percentile(returns, (1 - var_confidence) * 100)
        cvar = np.mean(returns[returns <= var])
        
        self.risk_metrics[strategy_name] = {
            'volatility': volatility,
            'var': var,
            'cvar': cvar,
            'skewness': self._calculate_skewness(returns),
            'kurtosis': self._calculate_kurtosis(returns),
            'max_drawdown': self._calculate_max_drawdown(returns)
        }
        
        self.var_data[strategy_name] = returns
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        return np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis."""
        return np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4) - 3
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def create_risk_chart(self) -> go.Figure:
        """Create risk analysis chart."""
        if not self.risk_metrics:
            return go.Figure()
        
        strategy_names = list(self.risk_metrics.keys())
        volatilities = [self.risk_metrics[name]['volatility'] for name in strategy_names]
        vars = [self.risk_metrics[name]['var'] for name in strategy_names]
        max_drawdowns = [self.risk_metrics[name]['max_drawdown'] for name in strategy_names]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volatility', 'Value at Risk', 'Max Drawdown', 'Risk-Return'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Volatility
        fig.add_trace(
            go.Bar(x=strategy_names, y=volatilities, name='Volatility',
                   marker_color='purple'),
            row=1, col=1
        )
        
        # Value at Risk
        fig.add_trace(
            go.Bar(x=strategy_names, y=vars, name='VaR',
                   marker_color='red'),
            row=1, col=2
        )
        
        # Max Drawdown
        fig.add_trace(
            go.Bar(x=strategy_names, y=max_drawdowns, name='Max Drawdown',
                   marker_color='orange'),
            row=2, col=1
        )
        
        # Risk-Return scatter
        returns = [0.1] * len(strategy_names)  # Placeholder returns
        fig.add_trace(
            go.Scatter(x=volatilities, y=returns, mode='markers+text',
                      text=strategy_names, textposition="top center",
                      marker=dict(size=10, color='blue')),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Risk Analysis",
            height=600,
            showlegend=False
        )
        
        return fig

class StrategyComparisonView:
    """Strategy comparison view component."""
    
    def __init__(self):
        """Initialize strategy comparison view."""
        self.comparison_data = {}
        self.benchmark_data = None
    
    def add_strategy_data(self, strategy_name: str, returns: np.ndarray, 
                         metrics: StrategyMetrics):
        """Add strategy data for comparison."""
        self.comparison_data[strategy_name] = {
            'returns': returns,
            'metrics': metrics
        }
    
    def set_benchmark(self, benchmark_returns: np.ndarray, benchmark_name: str = "Benchmark"):
        """Set benchmark data."""
        self.benchmark_data = {
            'returns': benchmark_returns,
            'name': benchmark_name
        }
    
    def create_comparison_chart(self) -> go.Figure:
        """Create strategy comparison chart."""
        if not self.comparison_data:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add strategy lines
        for strategy_name, data in self.comparison_data.items():
            cumulative_returns = np.cumprod(1 + data['returns'])
            fig.add_trace(go.Scatter(
                y=cumulative_returns,
                mode='lines',
                name=strategy_name
            ))
        
        # Add benchmark if available
        if self.benchmark_data:
            benchmark_cumulative = np.cumprod(1 + self.benchmark_data['returns'])
            fig.add_trace(go.Scatter(
                y=benchmark_cumulative,
                mode='lines',
                name=self.benchmark_data['name'],
                line=dict(dash='dash', color='black')
            ))
        
        fig.update_layout(
            title="Strategy Comparison",
            xaxis_title="Time",
            yaxis_title="Cumulative Returns",
            height=500
        )
        
        return fig
    
    def create_metrics_comparison_table(self) -> pd.DataFrame:
        """Create metrics comparison table."""
        if not self.comparison_data:
            return pd.DataFrame()
        
        comparison_data = []
        for strategy_name, data in self.comparison_data.items():
            metrics = data['metrics']
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{metrics.total_return:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Profit Factor': f"{metrics.profit_factor:.2f}"
            })
        
        return pd.DataFrame(comparison_data)

class PortfolioAllocationChart:
    """Portfolio allocation chart component."""
    
    def __init__(self):
        """Initialize portfolio allocation chart."""
        self.allocation_data = {}
        self.asset_weights = {}
    
    def update_allocation(self, strategy_name: str, asset_weights: Dict[str, float]):
        """Update asset allocation for strategy."""
        self.asset_weights[strategy_name] = asset_weights
        
        # Convert to allocation data format
        assets = list(asset_weights.keys())
        weights = list(asset_weights.values())
        
        self.allocation_data[strategy_name] = {
            'assets': assets,
            'weights': weights
        }
    
    def create_allocation_chart(self, strategy_name: str) -> go.Figure:
        """Create allocation chart for strategy."""
        if strategy_name not in self.allocation_data:
            return go.Figure()
        
        data = self.allocation_data[strategy_name]
        
        fig = go.Figure(data=[go.Pie(
            labels=data['assets'],
            values=data['weights'],
            hole=0.3
        )])
        
        fig.update_layout(
            title=f"Portfolio Allocation - {strategy_name}",
            height=400
        )
        
        return fig
    
    def create_allocation_comparison(self) -> go.Figure:
        """Create allocation comparison chart."""
        if not self.allocation_data:
            return go.Figure()
        
        # Create stacked bar chart
        strategies = list(self.allocation_data.keys())
        all_assets = set()
        
        for data in self.allocation_data.values():
            all_assets.update(data['assets'])
        
        all_assets = sorted(list(all_assets))
        
        fig = go.Figure()
        
        for asset in all_assets:
            weights = []
            for strategy in strategies:
                if strategy in self.allocation_data:
                    asset_idx = self.allocation_data[strategy]['assets'].index(asset) if asset in self.allocation_data[strategy]['assets'] else -1
                    weight = self.allocation_data[strategy]['weights'][asset_idx] if asset_idx >= 0 else 0
                    weights.append(weight)
                else:
                    weights.append(0)
            
            fig.add_trace(go.Bar(
                name=asset,
                x=strategies,
                y=weights
            ))
        
        fig.update_layout(
            title="Portfolio Allocation Comparison",
            barmode='stack',
            height=500
        )
        
        return fig

class AlertNotificationPanel:
    """Alert and notification panel component."""
    
    def __init__(self, max_alerts: int = 50):
        """
        Initialize alert notification panel.
        
        Args:
            max_alerts: Maximum number of alerts to keep
        """
        self.alerts = []
        self.max_alerts = max_alerts
        self.alert_callbacks = []
    
    def add_alert(self, alert: Alert):
        """Add new alert."""
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Alert callback error", error=str(e))
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for new alerts."""
        self.alert_callbacks.append(callback)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unacknowledged) alerts."""
        return [alert for alert in self.alerts if not alert.acknowledged]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by level."""
        return [alert for alert in self.alerts if alert.level == level]
    
    def create_alert_summary(self) -> Dict[str, Any]:
        """Create alert summary."""
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = len(self.get_alerts_by_level(level))
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'acknowledged_alerts': total_alerts - active_alerts,
            'level_counts': level_counts
        }

class StrategyDashboard:
    """Main strategy performance dashboard."""
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize strategy dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        self.overview_panel = StrategyOverviewPanel([])
        self.performance_widget = PerformanceMetricsWidget()
        self.risk_component = RiskAnalysisComponent()
        self.comparison_view = StrategyComparisonView()
        self.allocation_chart = PortfolioAllocationChart()
        self.alert_panel = AlertNotificationPanel()
        
        self.strategies = []
        self.is_running = False
    
    def add_strategy(self, strategy_name: str):
        """Add strategy to dashboard."""
        if strategy_name not in self.strategies:
            self.strategies.append(strategy_name)
            self.overview_panel.strategies = self.strategies
    
    def update_strategy_metrics(self, strategy_name: str, metrics: StrategyMetrics):
        """Update strategy metrics."""
        self.add_strategy(strategy_name)
        
        # Update all components
        self.overview_panel.update_strategy_data(strategy_name, metrics)
        self.performance_widget.add_metrics(strategy_name, metrics)
    
    def update_strategy_returns(self, strategy_name: str, returns: np.ndarray):
        """Update strategy returns data."""
        self.risk_component.add_risk_data(strategy_name, returns)
        self.comparison_view.add_strategy_data(strategy_name, returns, 
                                              StrategyMetrics(strategy_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    
    def update_allocation(self, strategy_name: str, asset_weights: Dict[str, float]):
        """Update portfolio allocation."""
        self.allocation_chart.update_allocation(strategy_name, asset_weights)
    
    def add_alert(self, alert: Alert):
        """Add alert to dashboard."""
        self.alert_panel.add_alert(alert)
    
    def create_dashboard_layout(self) -> Dict[str, Any]:
        """Create complete dashboard layout."""
        layout = {
            'overview': self.overview_panel.create_overview_chart(),
            'performance': {},
            'risk': self.risk_component.create_risk_chart(),
            'comparison': self.comparison_view.create_comparison_chart(),
            'allocation': self.allocation_chart.create_allocation_comparison(),
            'alerts': self.alert_panel.create_alert_summary()
        }
        
        # Add performance charts for each strategy
        for strategy in self.strategies:
            layout['performance'][strategy] = self.performance_widget.create_performance_chart(strategy)
        
        return layout
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary."""
        return {
            'total_strategies': len(self.strategies),
            'overview_stats': self.overview_panel.get_summary_stats(),
            'alert_summary': self.alert_panel.create_alert_summary(),
            'last_updated': datetime.now()
        }

def create_strategy_dashboard(config: DashboardConfig = None) -> StrategyDashboard:
    """
    Create a strategy performance dashboard.
    
    Args:
        config: Dashboard configuration
        
    Returns:
        StrategyDashboard instance
    """
    if config is None:
        config = DashboardConfig(components=list(DashboardComponent))
    
    return StrategyDashboard(config)

if __name__ == "__main__":
    # Demo of strategy dashboard
    config = DashboardConfig(
        components=[DashboardComponent.OVERVIEW, DashboardComponent.PERFORMANCE, 
                   DashboardComponent.RISK, DashboardComponent.COMPARISON,
                   DashboardComponent.PORTFOLIO, DashboardComponent.ALERTS]
    )
    
    dashboard = create_strategy_dashboard(config)
    
    # Add sample strategies
    strategies = ["Strategy A", "Strategy B", "Strategy C"]
    for strategy in strategies:
        dashboard.add_strategy(strategy)
        
        # Add sample metrics
        metrics = StrategyMetrics(
            strategy_name=strategy,
            total_return=np.random.uniform(0.05, 0.25),
            sharpe_ratio=np.random.uniform(0.5, 2.0),
            max_drawdown=np.random.uniform(-0.15, -0.05),
            volatility=np.random.uniform(0.1, 0.3),
            win_rate=np.random.uniform(0.4, 0.7),
            profit_factor=np.random.uniform(1.0, 3.0),
            calmar_ratio=np.random.uniform(0.5, 2.0),
            sortino_ratio=np.random.uniform(0.5, 2.0),
            beta=np.random.uniform(0.8, 1.2),
            alpha=np.random.uniform(-0.05, 0.05)
        )
        dashboard.update_strategy_metrics(strategy, metrics)
        
        # Add sample allocation
        allocation = {
            "Stocks": np.random.uniform(0.3, 0.6),
            "Bonds": np.random.uniform(0.2, 0.4),
            "Cash": np.random.uniform(0.1, 0.3)
        }
        dashboard.update_allocation(strategy, allocation)
    
    # Add sample alert
    alert = Alert(
        id="alert_001",
        level=AlertLevel.WARNING,
        message="Strategy A drawdown exceeded 10%",
        strategy_name="Strategy A"
    )
    dashboard.add_alert(alert)
    
    print("Strategy dashboard created successfully!")
    print(f"Strategies: {dashboard.strategies}")
    print(f"Components: {[comp.value for comp in config.components]}")
    
    # Get dashboard summary
    summary = dashboard.get_dashboard_summary()
    print(f"Dashboard summary: {summary}")