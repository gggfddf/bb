#!/usr/bin/env python3
"""
Risk Metrics Visualization Module

Implements comprehensive risk metrics visualization:
- VaR (Value at Risk) charts
- Drawdown visualization
- Volatility analysis charts
- Correlation matrices
- Stress testing visualizations
- Risk attribution charts

Features:
- Comprehensive VaR analysis and visualization
- Drawdown analysis and monitoring
- Volatility clustering and regime detection
- Correlation analysis and heatmaps
- Stress testing scenarios and impact analysis
- Risk attribution and factor decomposition
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
from scipy import stats
from scipy.stats import norm, t
import warnings

logger = structlog.get_logger()

class RiskMetricType(Enum):
    """Risk metric types."""
    VAR = "var"
    CVAR = "cvar"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    STRESS_TEST = "stress_test"
    RISK_ATTRIBUTION = "risk_attribution"

class ConfidenceLevel(Enum):
    """Confidence levels for risk metrics."""
    P90 = 0.90
    P95 = 0.95
    P99 = 0.99
    P99_5 = 0.995

@dataclass
class RiskMetrics:
    """Risk metrics data structure."""
    strategy_name: str
    var: Dict[float, float]  # confidence_level -> var_value
    cvar: Dict[float, float]  # confidence_level -> cvar_value
    max_drawdown: float
    current_drawdown: float
    volatility: float
    skewness: float
    kurtosis: float
    beta: float
    alpha: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StressTestScenario:
    """Stress test scenario."""
    name: str
    description: str
    market_shock: float
    volatility_shock: float
    correlation_shock: float
    expected_loss: float
    confidence_interval: Tuple[float, float]

class VaRVisualization:
    """VaR (Value at Risk) visualization component."""
    
    def __init__(self):
        """Initialize VaR visualization."""
        self.var_data = {}
        self.var_history = {}
    
    def calculate_var(self, returns: np.ndarray, confidence_levels: List[float] = None) -> Dict[float, float]:
        """
        Calculate VaR for different confidence levels.
        
        Args:
            returns: Array of returns
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary mapping confidence levels to VaR values
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99, 0.995]
        
        var_values = {}
        for confidence in confidence_levels:
            var = np.percentile(returns, (1 - confidence) * 100)
            var_values[confidence] = var
        
        return var_values
    
    def calculate_cvar(self, returns: np.ndarray, confidence_levels: List[float] = None) -> Dict[float, float]:
        """
        Calculate CVaR (Conditional VaR) for different confidence levels.
        
        Args:
            returns: Array of returns
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary mapping confidence levels to CVaR values
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99, 0.995]
        
        cvar_values = {}
        for confidence in confidence_levels:
            var = np.percentile(returns, (1 - confidence) * 100)
            cvar = np.mean(returns[returns <= var])
            cvar_values[confidence] = cvar
        
        return cvar_values
    
    def add_strategy_data(self, strategy_name: str, returns: np.ndarray):
        """Add strategy data for VaR analysis."""
        var_values = self.calculate_var(returns)
        cvar_values = self.calculate_cvar(returns)
        
        self.var_data[strategy_name] = {
            'returns': returns,
            'var': var_values,
            'cvar': cvar_values
        }
        
        # Store historical VaR
        if strategy_name not in self.var_history:
            self.var_history[strategy_name] = []
        
        self.var_history[strategy_name].append({
            'timestamp': datetime.now(),
            'var': var_values,
            'cvar': cvar_values
        })
    
    def create_var_chart(self, strategy_name: str = None) -> go.Figure:
        """Create VaR visualization chart."""
        if strategy_name and strategy_name not in self.var_data:
            return go.Figure()
        
        if strategy_name:
            strategies = [strategy_name]
        else:
            strategies = list(self.var_data.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('VaR by Confidence Level', 'CVaR by Confidence Level', 
                          'VaR vs CVaR', 'VaR Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, strategy in enumerate(strategies):
            if strategy not in self.var_data:
                continue
            
            data = self.var_data[strategy]
            confidence_levels = list(data['var'].keys())
            var_values = list(data['var'].values())
            cvar_values = list(data['cvar'].values())
            
            # VaR by confidence level
            fig.add_trace(
                go.Bar(x=[f"{c*100:.0f}%" for c in confidence_levels], 
                       y=var_values, name=f'{strategy} VaR',
                       marker_color=colors[i % len(colors)]),
                row=1, col=1
            )
            
            # CVaR by confidence level
            fig.add_trace(
                go.Bar(x=[f"{c*100:.0f}%" for c in confidence_levels], 
                       y=cvar_values, name=f'{strategy} CVaR',
                       marker_color=colors[i % len(colors)], opacity=0.7),
                row=1, col=2
            )
            
            # VaR vs CVaR scatter
            fig.add_trace(
                go.Scatter(x=var_values, y=cvar_values, mode='markers+text',
                          text=[f"{c*100:.0f}%" for c in confidence_levels],
                          textposition="top center", name=f'{strategy}',
                          marker=dict(size=10, color=colors[i % len(colors)])),
                row=2, col=1
            )
            
            # VaR distribution
            fig.add_trace(
                go.Histogram(x=data['returns'], nbinsx=50, name=f'{strategy} Returns',
                            marker_color=colors[i % len(colors)], opacity=0.7),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Value at Risk (VaR) Analysis",
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_var_timeseries(self, strategy_name: str, confidence_level: float = 0.95) -> go.Figure:
        """Create VaR time series chart."""
        if strategy_name not in self.var_history:
            return go.Figure()
        
        history = self.var_history[strategy_name]
        timestamps = [h['timestamp'] for h in history]
        var_values = [h['var'].get(confidence_level, 0) for h in history]
        cvar_values = [h['cvar'].get(confidence_level, 0) for h in history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=var_values, mode='lines', name='VaR',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=cvar_values, mode='lines', name='CVaR',
            line=dict(color='darkred')
        ))
        
        fig.update_layout(
            title=f"VaR Time Series - {strategy_name} ({confidence_level*100:.0f}% confidence)",
            xaxis_title="Time",
            yaxis_title="VaR/CVaR",
            height=400
        )
        
        return fig

class DrawdownVisualization:
    """Drawdown visualization component."""
    
    def __init__(self):
        """Initialize drawdown visualization."""
        self.drawdown_data = {}
    
    def calculate_drawdown(self, returns: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Calculate drawdown series and statistics.
        
        Args:
            returns: Array of returns
            
        Returns:
            Tuple of (drawdown_series, max_drawdown, current_drawdown)
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = np.min(drawdown)
        current_drawdown = drawdown[-1]
        
        return drawdown, max_drawdown, current_drawdown
    
    def add_strategy_data(self, strategy_name: str, returns: np.ndarray):
        """Add strategy data for drawdown analysis."""
        drawdown_series, max_drawdown, current_drawdown = self.calculate_drawdown(returns)
        
        self.drawdown_data[strategy_name] = {
            'returns': returns,
            'drawdown_series': drawdown_series,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'cumulative_returns': np.cumprod(1 + returns)
        }
    
    def create_drawdown_chart(self, strategy_name: str = None) -> go.Figure:
        """Create drawdown visualization chart."""
        if strategy_name and strategy_name not in self.drawdown_data:
            return go.Figure()
        
        if strategy_name:
            strategies = [strategy_name]
        else:
            strategies = list(self.drawdown_data.keys())
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Returns', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, strategy in enumerate(strategies):
            if strategy not in self.drawdown_data:
                continue
            
            data = self.drawdown_data[strategy]
            time_index = np.arange(len(data['cumulative_returns']))
            
            # Cumulative returns
            fig.add_trace(
                go.Scatter(x=time_index, y=data['cumulative_returns'], 
                          mode='lines', name=f'{strategy} Returns',
                          line=dict(color=colors[i % len(colors)])),
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(x=time_index, y=data['drawdown_series'], 
                          mode='lines', name=f'{strategy} Drawdown',
                          line=dict(color=colors[i % len(colors)]),
                          fill='tonexty'),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Drawdown Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_drawdown_comparison(self) -> go.Figure:
        """Create drawdown comparison chart."""
        if not self.drawdown_data:
            return go.Figure()
        
        strategies = list(self.drawdown_data.keys())
        max_drawdowns = [self.drawdown_data[s]['max_drawdown'] for s in strategies]
        current_drawdowns = [self.drawdown_data[s]['current_drawdown'] for s in strategies]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=strategies,
            y=max_drawdowns,
            name='Maximum Drawdown',
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            x=strategies,
            y=current_drawdowns,
            name='Current Drawdown',
            marker_color='orange'
        ))
        
        fig.update_layout(
            title="Drawdown Comparison",
            barmode='group',
            height=400
        )
        
        return fig

class VolatilityVisualization:
    """Volatility analysis visualization component."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize volatility visualization.
        
        Args:
            window_size: Rolling window size for volatility calculation
        """
        self.window_size = window_size
        self.volatility_data = {}
    
    def calculate_rolling_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Calculate rolling volatility."""
        return pd.Series(returns).rolling(window=self.window_size).std().values * np.sqrt(252)
    
    def add_strategy_data(self, strategy_name: str, returns: np.ndarray):
        """Add strategy data for volatility analysis."""
        rolling_vol = self.calculate_rolling_volatility(returns)
        overall_vol = np.std(returns) * np.sqrt(252)
        
        self.volatility_data[strategy_name] = {
            'returns': returns,
            'rolling_volatility': rolling_vol,
            'overall_volatility': overall_vol,
            'volatility_regime': self._detect_volatility_regime(rolling_vol)
        }
    
    def _detect_volatility_regime(self, rolling_vol: np.ndarray) -> str:
        """Detect volatility regime."""
        if len(rolling_vol) < 10:
            return "insufficient_data"
        
        recent_vol = np.mean(rolling_vol[-10:])
        historical_vol = np.mean(rolling_vol[:-10]) if len(rolling_vol) > 10 else recent_vol
        
        if recent_vol > historical_vol * 1.5:
            return "high_volatility"
        elif recent_vol < historical_vol * 0.7:
            return "low_volatility"
        else:
            return "normal_volatility"
    
    def create_volatility_chart(self, strategy_name: str = None) -> go.Figure:
        """Create volatility analysis chart."""
        if strategy_name and strategy_name not in self.volatility_data:
            return go.Figure()
        
        if strategy_name:
            strategies = [strategy_name]
        else:
            strategies = list(self.volatility_data.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rolling Volatility', 'Volatility Distribution', 
                          'Returns vs Volatility', 'Volatility Regime'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, strategy in enumerate(strategies):
            if strategy not in self.volatility_data:
                continue
            
            data = self.volatility_data[strategy]
            time_index = np.arange(len(data['rolling_volatility']))
            
            # Rolling volatility
            fig.add_trace(
                go.Scatter(x=time_index, y=data['rolling_volatility'], 
                          mode='lines', name=f'{strategy} Volatility',
                          line=dict(color=colors[i % len(colors)])),
                row=1, col=1
            )
            
            # Volatility distribution
            fig.add_trace(
                go.Histogram(x=data['rolling_volatility'][~np.isnan(data['rolling_volatility'])], 
                            nbinsx=30, name=f'{strategy}',
                            marker_color=colors[i % len(colors)], opacity=0.7),
                row=1, col=2
            )
            
            # Returns vs Volatility scatter
            valid_mask = ~np.isnan(data['rolling_volatility'])
            fig.add_trace(
                go.Scatter(x=data['returns'][valid_mask], 
                          y=data['rolling_volatility'][valid_mask], 
                          mode='markers', name=f'{strategy}',
                          marker=dict(size=5, color=colors[i % len(colors)])),
                row=2, col=1
            )
        
        # Volatility regime comparison
        regimes = []
        regime_counts = {}
        for strategy in strategies:
            if strategy in self.volatility_data:
                regime = self.volatility_data[strategy]['volatility_regime']
                regimes.append(regime)
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if regime_counts:
            fig.add_trace(
                go.Bar(x=list(regime_counts.keys()), y=list(regime_counts.values()),
                       name='Regime Count', marker_color='purple'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Volatility Analysis",
            height=700,
            showlegend=True
        )
        
        return fig

class CorrelationVisualization:
    """Correlation matrix visualization component."""
    
    def __init__(self):
        """Initialize correlation visualization."""
        self.correlation_data = {}
    
    def calculate_correlation_matrix(self, returns_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Calculate correlation matrix from returns dictionary."""
        # Align returns to same length
        min_length = min(len(returns) for returns in returns_dict.values())
        aligned_returns = {}
        
        for strategy, returns in returns_dict.items():
            aligned_returns[strategy] = returns[-min_length:]
        
        # Create DataFrame and calculate correlation
        df = pd.DataFrame(aligned_returns)
        return df.corr()
    
    def add_strategy_data(self, strategy_name: str, returns: np.ndarray):
        """Add strategy data for correlation analysis."""
        self.correlation_data[strategy_name] = returns
    
    def create_correlation_heatmap(self) -> go.Figure:
        """Create correlation matrix heatmap."""
        if len(self.correlation_data) < 2:
            return go.Figure()
        
        corr_matrix = self.calculate_correlation_matrix(self.correlation_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Strategy Correlation Matrix",
            height=500,
            width=500
        )
        
        return fig
    
    def create_correlation_network(self) -> go.Figure:
        """Create correlation network visualization."""
        if len(self.correlation_data) < 2:
            return go.Figure()
        
        corr_matrix = self.calculate_correlation_matrix(self.correlation_data)
        
        # Create network edges
        edges = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) > 0.3:  # Only show significant correlations
                    edges.append({
                        'source': corr_matrix.columns[i],
                        'target': corr_matrix.columns[j],
                        'correlation': correlation
                    })
        
        if not edges:
            return go.Figure()
        
        # Create network visualization
        nodes = list(corr_matrix.columns)
        node_x = []
        node_y = []
        
        # Simple circular layout
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            node_x.append(np.cos(angle))
            node_y.append(np.sin(angle))
        
        # Create edge traces
        edge_traces = []
        for edge in edges:
            source_idx = nodes.index(edge['source'])
            target_idx = nodes.index(edge['target'])
            
            edge_traces.append(go.Scatter(
                x=[node_x[source_idx], node_x[target_idx]],
                y=[node_y[source_idx], node_y[target_idx]],
                mode='lines',
                line=dict(width=abs(edge['correlation']) * 5, color='gray'),
                showlegend=False,
                hoverinfo='text',
                text=f"{edge['source']} - {edge['target']}: {edge['correlation']:.2f}"
            ))
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=nodes,
            textposition="middle center",
            marker=dict(size=20, color='lightblue'),
            showlegend=False
        )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Strategy Correlation Network",
            height=500,
            showlegend=False
        )
        
        return fig

class StressTestVisualization:
    """Stress testing visualization component."""
    
    def __init__(self):
        """Initialize stress testing visualization."""
        self.scenarios = []
        self.stress_test_results = {}
    
    def add_stress_scenario(self, scenario: StressTestScenario):
        """Add stress test scenario."""
        self.scenarios.append(scenario)
    
    def run_stress_test(self, strategy_name: str, base_returns: np.ndarray, 
                       scenarios: List[StressTestScenario] = None) -> Dict[str, float]:
        """Run stress test for strategy."""
        if scenarios is None:
            scenarios = self.scenarios
        
        results = {}
        base_var = np.percentile(base_returns, 5)  # 95% VaR
        
        for scenario in scenarios:
            # Apply stress factors
            stressed_returns = base_returns * (1 + scenario.market_shock)
            stressed_vol = np.std(stressed_returns) * (1 + scenario.volatility_shock)
            
            # Calculate stressed VaR
            stressed_var = np.percentile(stressed_returns, 5)
            var_change = (stressed_var - base_var) / abs(base_var) if base_var != 0 else 0
            
            results[scenario.name] = {
                'var_change': var_change,
                'expected_loss': scenario.expected_loss,
                'market_shock': scenario.market_shock,
                'volatility_shock': scenario.volatility_shock
            }
        
        self.stress_test_results[strategy_name] = results
        return results
    
    def create_stress_test_chart(self, strategy_name: str = None) -> go.Figure:
        """Create stress testing visualization chart."""
        if strategy_name and strategy_name not in self.stress_test_results:
            return go.Figure()
        
        if strategy_name:
            strategies = [strategy_name]
        else:
            strategies = list(self.stress_test_results.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('VaR Change by Scenario', 'Expected Loss by Scenario',
                          'Market Shock Impact', 'Volatility Shock Impact'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, strategy in enumerate(strategies):
            if strategy not in self.stress_test_results:
                continue
            
            results = self.stress_test_results[strategy]
            scenarios = list(results.keys())
            var_changes = [results[s]['var_change'] for s in scenarios]
            expected_losses = [results[s]['expected_loss'] for s in scenarios]
            market_shocks = [results[s]['market_shock'] for s in scenarios]
            vol_shocks = [results[s]['volatility_shock'] for s in scenarios]
            
            # VaR Change
            fig.add_trace(
                go.Bar(x=scenarios, y=var_changes, name=f'{strategy} VaR Change',
                       marker_color=colors[i % len(colors)]),
                row=1, col=1
            )
            
            # Expected Loss
            fig.add_trace(
                go.Bar(x=scenarios, y=expected_losses, name=f'{strategy} Expected Loss',
                       marker_color=colors[i % len(colors)], opacity=0.7),
                row=1, col=2
            )
            
            # Market Shock Impact
            fig.add_trace(
                go.Scatter(x=market_shocks, y=var_changes, mode='markers+text',
                          text=scenarios, textposition="top center",
                          name=f'{strategy} Market Impact',
                          marker=dict(size=10, color=colors[i % len(colors)])),
                row=2, col=1
            )
            
            # Volatility Shock Impact
            fig.add_trace(
                go.Scatter(x=vol_shocks, y=var_changes, mode='markers+text',
                          text=scenarios, textposition="top center",
                          name=f'{strategy} Vol Impact',
                          marker=dict(size=10, color=colors[i % len(colors)])),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Stress Testing Analysis",
            height=700,
            showlegend=True
        )
        
        return fig

class RiskAttributionVisualization:
    """Risk attribution visualization component."""
    
    def __init__(self):
        """Initialize risk attribution visualization."""
        self.attribution_data = {}
    
    def calculate_risk_attribution(self, strategy_name: str, returns: np.ndarray,
                                 factor_returns: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate risk attribution to factors."""
        # Simple factor attribution using linear regression
        if not factor_returns:
            return {}
        
        # Align data
        min_length = min(len(returns), *[len(factor_returns[f]) for f in factor_returns])
        aligned_returns = returns[-min_length:]
        aligned_factors = {f: factor_returns[f][-min_length:] for f in factor_returns}
        
        # Create factor matrix
        factor_matrix = np.column_stack([aligned_factors[f] for f in aligned_factors])
        
        # Linear regression
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(factor_matrix, aligned_returns)
            
            # Calculate factor contributions
            factor_contributions = {}
            for i, factor in enumerate(aligned_factors.keys()):
                factor_contributions[factor] = model.coef_[i]
            
            return factor_contributions
        except ImportError:
            # Fallback to correlation-based attribution
            factor_contributions = {}
            for factor, factor_returns in aligned_factors.items():
                correlation = np.corrcoef(aligned_returns, factor_returns)[0, 1]
                factor_contributions[factor] = correlation if not np.isnan(correlation) else 0
            
            return factor_contributions
    
    def add_strategy_data(self, strategy_name: str, returns: np.ndarray,
                         factor_returns: Dict[str, np.ndarray]):
        """Add strategy data for risk attribution."""
        attribution = self.calculate_risk_attribution(strategy_name, returns, factor_returns)
        
        self.attribution_data[strategy_name] = {
            'returns': returns,
            'factor_returns': factor_returns,
            'attribution': attribution
        }
    
    def create_attribution_chart(self, strategy_name: str = None) -> go.Figure:
        """Create risk attribution visualization chart."""
        if strategy_name and strategy_name not in self.attribution_data:
            return go.Figure()
        
        if strategy_name:
            strategies = [strategy_name]
        else:
            strategies = list(self.attribution_data.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Factor Contributions', 'Risk Attribution Pie',
                          'Factor Correlation', 'Attribution Over Time'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, strategy in enumerate(strategies):
            if strategy not in self.attribution_data:
                continue
            
            data = self.attribution_data[strategy]
            factors = list(data['attribution'].keys())
            contributions = list(data['attribution'].values())
            
            # Factor contributions bar chart
            fig.add_trace(
                go.Bar(x=factors, y=contributions, name=f'{strategy}',
                       marker_color=colors[i % len(colors)]),
                row=1, col=1
            )
            
            # Risk attribution pie chart
            fig.add_trace(
                go.Pie(labels=factors, values=np.abs(contributions), name=f'{strategy}'),
                row=1, col=2
            )
        
        # Factor correlation heatmap (for first strategy)
        if strategies and strategies[0] in self.attribution_data:
            data = self.attribution_data[strategies[0]]
            if data['factor_returns']:
                factor_names = list(data['factor_returns'].keys())
                corr_matrix = np.corrcoef([data['factor_returns'][f] for f in factor_names])
                
                fig.add_trace(
                    go.Heatmap(z=corr_matrix, x=factor_names, y=factor_names,
                              colorscale='RdBu', zmid=0),
                    row=2, col=1
                )
        
        fig.update_layout(
            title="Risk Attribution Analysis",
            height=700,
            showlegend=True
        )
        
        return fig

class RiskVisualizationManager:
    """Main risk visualization manager."""
    
    def __init__(self):
        """Initialize risk visualization manager."""
        self.var_viz = VaRVisualization()
        self.drawdown_viz = DrawdownVisualization()
        self.volatility_viz = VolatilityVisualization()
        self.correlation_viz = CorrelationVisualization()
        self.stress_test_viz = StressTestVisualization()
        self.attribution_viz = RiskAttributionVisualization()
        
        self.strategies = []
    
    def add_strategy_data(self, strategy_name: str, returns: np.ndarray,
                         factor_returns: Dict[str, np.ndarray] = None):
        """Add strategy data to all visualization components."""
        if strategy_name not in self.strategies:
            self.strategies.append(strategy_name)
        
        # Add to all components
        self.var_viz.add_strategy_data(strategy_name, returns)
        self.drawdown_viz.add_strategy_data(strategy_name, returns)
        self.volatility_viz.add_strategy_data(strategy_name, returns)
        self.correlation_viz.add_strategy_data(strategy_name, returns)
        
        if factor_returns:
            self.attribution_viz.add_strategy_data(strategy_name, returns, factor_returns)
    
    def add_stress_scenario(self, scenario: StressTestScenario):
        """Add stress test scenario."""
        self.stress_test_viz.add_stress_scenario(scenario)
    
    def run_stress_tests(self, scenarios: List[StressTestScenario] = None):
        """Run stress tests for all strategies."""
        for strategy in self.strategies:
            if strategy in self.var_viz.var_data:
                returns = self.var_viz.var_data[strategy]['returns']
                self.stress_test_viz.run_stress_test(strategy, returns, scenarios)
    
    def create_comprehensive_risk_report(self) -> Dict[str, go.Figure]:
        """Create comprehensive risk visualization report."""
        report = {
            'var_analysis': self.var_viz.create_var_chart(),
            'drawdown_analysis': self.drawdown_viz.create_drawdown_chart(),
            'volatility_analysis': self.volatility_viz.create_volatility_chart(),
            'correlation_matrix': self.correlation_viz.create_correlation_heatmap(),
            'correlation_network': self.correlation_viz.create_correlation_network(),
            'stress_testing': self.stress_test_viz.create_stress_test_chart(),
            'risk_attribution': self.attribution_viz.create_attribution_chart()
        }
        
        return report

def create_risk_visualization_manager() -> RiskVisualizationManager:
    """
    Create a risk visualization manager.
    
    Returns:
        RiskVisualizationManager instance
    """
    return RiskVisualizationManager()

if __name__ == "__main__":
    # Demo of risk visualization
    manager = create_risk_visualization_manager()
    
    # Add sample strategies
    strategies = ["Strategy A", "Strategy B", "Strategy C"]
    for strategy in strategies:
        # Generate sample returns
        returns = np.random.randn(252) * 0.02  # Daily returns
        manager.add_strategy_data(strategy, returns)
    
    # Add stress test scenarios
    scenarios = [
        StressTestScenario("Market Crash", "Severe market downturn", -0.20, 0.50, 0.30, -0.15, (-0.20, -0.10)),
        StressTestScenario("Volatility Spike", "High volatility period", -0.05, 0.80, 0.20, -0.08, (-0.12, -0.04)),
        StressTestScenario("Correlation Breakdown", "Diversification failure", -0.10, 0.30, 0.60, -0.12, (-0.18, -0.06))
    ]
    
    for scenario in scenarios:
        manager.add_stress_scenario(scenario)
    
    # Run stress tests
    manager.run_stress_tests()
    
    print("Risk visualization manager created successfully!")
    print(f"Strategies: {manager.strategies}")
    print(f"Stress scenarios: {len(scenarios)}")
    
    # Create comprehensive report
    report = manager.create_comprehensive_risk_report()
    print(f"Generated {len(report)} risk visualization charts")