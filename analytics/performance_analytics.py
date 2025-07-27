#!/usr/bin/env python3
"""
Performance Analytics System

Implements comprehensive performance analytics for trading:
- Performance metrics calculation
- Risk-adjusted returns analysis
- Attribution analysis
- Performance reporting
- Benchmark comparison
- Portfolio analytics

Features:
- Advanced performance metrics and ratios
- Risk-adjusted return analysis
- Attribution and factor analysis
- Performance reporting and visualization
- Benchmark comparison and analysis
- Portfolio analytics and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class MetricType(Enum):
    """Performance metric type enumeration."""
    RETURN = "return"
    RISK = "risk"
    RATIO = "ratio"
    DRAWDOWN = "drawdown"
    ATTRIBUTION = "attribution"

class AttributionType(Enum):
    """Attribution type enumeration."""
    ASSET_ALLOCATION = "asset_allocation"
    STOCK_SELECTION = "stock_selection"
    INTERACTION = "interaction"
    TIMING = "timing"

@dataclass
class PerformanceMetrics:
    """Performance metrics structure."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0

@dataclass
class AttributionResult:
    """Attribution result structure."""
    attribution_type: AttributionType
    contribution: float
    weight: float
    return_contribution: float
    excess_return: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Performance report structure."""
    report_id: str
    start_date: datetime
    end_date: datetime
    metrics: PerformanceMetrics
    attribution: List[AttributionResult]
    benchmark_comparison: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalyticsConfig:
    """Performance analytics configuration."""
    risk_free_rate: float = 0.02
    benchmark_return: float = 0.08
    benchmark_volatility: float = 0.15
    var_confidence_level: float = 0.95
    enable_attribution: bool = True
    enable_benchmark_comparison: bool = True
    enable_visualization: bool = True

class PerformanceCalculator:
    """Performance metrics calculator."""
    
    def __init__(self, config: AnalyticsConfig):
        """
        Initialize performance calculator.
        
        Args:
            config: Analytics configuration
        """
        self.config = config
        
    def calculate_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            metrics = PerformanceMetrics()
            
            # Basic return metrics
            metrics.total_return = self._calculate_total_return(returns)
            metrics.annualized_return = self._calculate_annualized_return(returns)
            metrics.volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted ratios
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
            metrics.calmar_ratio = self._calculate_calmar_ratio(returns)
            
            # Drawdown metrics
            metrics.max_drawdown = self._calculate_max_drawdown(returns)
            
            # Win/loss metrics
            win_rate, profit_factor, avg_win, avg_loss = self._calculate_win_loss_metrics(returns)
            metrics.win_rate = win_rate
            metrics.profit_factor = profit_factor
            metrics.average_win = avg_win
            metrics.average_loss = avg_loss
            
            # Risk metrics
            metrics.var_95 = self._calculate_var(returns, 0.95)
            metrics.cvar_95 = self._calculate_cvar(returns, 0.95)
            
            # Benchmark comparison
            if benchmark_returns is not None:
                metrics.beta = self._calculate_beta(returns, benchmark_returns)
                metrics.alpha = self._calculate_alpha(returns, benchmark_returns)
                metrics.information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
                metrics.treynor_ratio = self._calculate_treynor_ratio(returns, benchmark_returns)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics()
            
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total return."""
        try:
            return (1 + returns).prod() - 1
        except:
            return 0.0
            
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        try:
            total_return = self._calculate_total_return(returns)
            years = len(returns) / 252
            return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        except:
            return 0.0
            
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            excess_returns = returns - self.config.risk_free_rate / 252
            return excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        except:
            return 0.0
            
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        try:
            excess_returns = returns - self.config.risk_free_rate / 252
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            return excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
        except:
            return 0.0
            
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        try:
            annual_return = self._calculate_annualized_return(returns)
            max_dd = self._calculate_max_drawdown(returns)
            return (annual_return - self.config.risk_free_rate) / max_dd if max_dd > 0 else 0
        except:
            return 0.0
            
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except:
            return 0.0
            
    def _calculate_win_loss_metrics(self, returns: pd.Series) -> Tuple[float, float, float, float]:
        """Calculate win/loss metrics."""
        try:
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
            avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            return win_rate, profit_factor, avg_win, avg_loss
        except:
            return 0.0, 0.0, 0.0, 0.0
            
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        try:
            return abs(np.percentile(returns, (1 - confidence_level) * 100))
        except:
            return 0.0
            
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk."""
        try:
            var = self._calculate_var(returns, confidence_level)
            tail_returns = returns[returns <= -var]
            return abs(tail_returns.mean()) if len(tail_returns) > 0 else var
        except:
            return 0.0
            
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta."""
        try:
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            return covariance / benchmark_variance if benchmark_variance > 0 else 0
        except:
            return 0.0
            
    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate alpha."""
        try:
            beta = self._calculate_beta(returns, benchmark_returns)
            portfolio_return = returns.mean() * 252
            benchmark_return = benchmark_returns.mean() * 252
            return portfolio_return - (self.config.risk_free_rate + beta * (benchmark_return - self.config.risk_free_rate))
        except:
            return 0.0
            
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio."""
        try:
            excess_returns = returns - benchmark_returns
            return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        except:
            return 0.0
            
    def _calculate_treynor_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Treynor ratio."""
        try:
            beta = self._calculate_beta(returns, benchmark_returns)
            excess_return = returns.mean() * 252 - self.config.risk_free_rate
            return excess_return / beta if beta > 0 else 0
        except:
            return 0.0

class AttributionAnalyzer:
    """Attribution analyzer."""
    
    def __init__(self, config: AnalyticsConfig):
        """
        Initialize attribution analyzer.
        
        Args:
            config: Analytics configuration
        """
        self.config = config
        
    def calculate_attribution(self, portfolio_returns: pd.Series, 
                            benchmark_returns: pd.Series,
                            portfolio_weights: Dict[str, float],
                            benchmark_weights: Dict[str, float]) -> List[AttributionResult]:
        """Calculate attribution analysis."""
        try:
            attribution_results = []
            
            # Asset allocation attribution
            allocation_attribution = self._calculate_asset_allocation_attribution(
                portfolio_returns, benchmark_returns, portfolio_weights, benchmark_weights
            )
            attribution_results.append(allocation_attribution)
            
            # Stock selection attribution
            selection_attribution = self._calculate_stock_selection_attribution(
                portfolio_returns, benchmark_returns, portfolio_weights, benchmark_weights
            )
            attribution_results.append(selection_attribution)
            
            # Interaction attribution
            interaction_attribution = self._calculate_interaction_attribution(
                portfolio_returns, benchmark_returns, portfolio_weights, benchmark_weights
            )
            attribution_results.append(interaction_attribution)
            
            return attribution_results
            
        except Exception as e:
            logger.error(f"Error calculating attribution: {e}")
            return []
            
    def _calculate_asset_allocation_attribution(self, portfolio_returns: pd.Series,
                                              benchmark_returns: pd.Series,
                                              portfolio_weights: Dict[str, float],
                                              benchmark_weights: Dict[str, float]) -> AttributionResult:
        """Calculate asset allocation attribution."""
        try:
            # Simplified calculation
            portfolio_return = portfolio_returns.mean() * 252
            benchmark_return = benchmark_returns.mean() * 252
            
            # Calculate allocation effect
            allocation_effect = 0.0
            for asset in portfolio_weights:
                if asset in benchmark_weights:
                    weight_diff = portfolio_weights[asset] - benchmark_weights[asset]
                    asset_return = portfolio_return  # Simplified
                    allocation_effect += weight_diff * asset_return
                    
            return AttributionResult(
                attribution_type=AttributionType.ASSET_ALLOCATION,
                contribution=allocation_effect,
                weight=sum(portfolio_weights.values()),
                return_contribution=allocation_effect,
                excess_return=portfolio_return - benchmark_return
            )
            
        except Exception as e:
            logger.error(f"Error calculating asset allocation attribution: {e}")
            return AttributionResult(
                attribution_type=AttributionType.ASSET_ALLOCATION,
                contribution=0.0,
                weight=0.0,
                return_contribution=0.0,
                excess_return=0.0
            )
            
    def _calculate_stock_selection_attribution(self, portfolio_returns: pd.Series,
                                             benchmark_returns: pd.Series,
                                             portfolio_weights: Dict[str, float],
                                             benchmark_weights: Dict[str, float]) -> AttributionResult:
        """Calculate stock selection attribution."""
        try:
            # Simplified calculation
            portfolio_return = portfolio_returns.mean() * 252
            benchmark_return = benchmark_returns.mean() * 252
            
            # Calculate selection effect
            selection_effect = 0.0
            for asset in portfolio_weights:
                if asset in benchmark_weights:
                    benchmark_weight = benchmark_weights[asset]
                    asset_return_diff = portfolio_return - benchmark_return  # Simplified
                    selection_effect += benchmark_weight * asset_return_diff
                    
            return AttributionResult(
                attribution_type=AttributionType.STOCK_SELECTION,
                contribution=selection_effect,
                weight=sum(benchmark_weights.values()),
                return_contribution=selection_effect,
                excess_return=portfolio_return - benchmark_return
            )
            
        except Exception as e:
            logger.error(f"Error calculating stock selection attribution: {e}")
            return AttributionResult(
                attribution_type=AttributionType.STOCK_SELECTION,
                contribution=0.0,
                weight=0.0,
                return_contribution=0.0,
                excess_return=0.0
            )
            
    def _calculate_interaction_attribution(self, portfolio_returns: pd.Series,
                                         benchmark_returns: pd.Series,
                                         portfolio_weights: Dict[str, float],
                                         benchmark_weights: Dict[str, float]) -> AttributionResult:
        """Calculate interaction attribution."""
        try:
            # Simplified calculation
            portfolio_return = portfolio_returns.mean() * 252
            benchmark_return = benchmark_returns.mean() * 252
            
            # Calculate interaction effect
            interaction_effect = 0.0
            for asset in portfolio_weights:
                if asset in benchmark_weights:
                    weight_diff = portfolio_weights[asset] - benchmark_weights[asset]
                    return_diff = portfolio_return - benchmark_return  # Simplified
                    interaction_effect += weight_diff * return_diff
                    
            return AttributionResult(
                attribution_type=AttributionType.INTERACTION,
                contribution=interaction_effect,
                weight=sum(portfolio_weights.values()),
                return_contribution=interaction_effect,
                excess_return=portfolio_return - benchmark_return
            )
            
        except Exception as e:
            logger.error(f"Error calculating interaction attribution: {e}")
            return AttributionResult(
                attribution_type=AttributionType.INTERACTION,
                contribution=0.0,
                weight=0.0,
                return_contribution=0.0,
                excess_return=0.0
            )

class PerformanceAnalytics:
    """Main performance analytics system."""
    
    def __init__(self, config: AnalyticsConfig):
        """
        Initialize performance analytics.
        
        Args:
            config: Analytics configuration
        """
        self.config = config
        self.calculator = PerformanceCalculator(config)
        self.attribution_analyzer = AttributionAnalyzer(config)
        self.reports: List[PerformanceReport] = []
        
    def generate_report(self, returns: pd.Series, benchmark_returns: pd.Series = None,
                       portfolio_weights: Dict[str, float] = None,
                       benchmark_weights: Dict[str, float] = None,
                       start_date: datetime = None, end_date: datetime = None) -> PerformanceReport:
        """Generate comprehensive performance report."""
        try:
            # Calculate performance metrics
            metrics = self.calculator.calculate_metrics(returns, benchmark_returns)
            
            # Calculate attribution if enabled
            attribution = []
            if self.config.enable_attribution and portfolio_weights and benchmark_weights:
                attribution = self.attribution_analyzer.calculate_attribution(
                    returns, benchmark_returns, portfolio_weights, benchmark_weights
                )
                
            # Calculate benchmark comparison
            benchmark_comparison = {}
            if self.config.enable_benchmark_comparison and benchmark_returns is not None:
                benchmark_comparison = self._calculate_benchmark_comparison(returns, benchmark_returns)
                
            # Create report
            report = PerformanceReport(
                report_id=str(uuid.uuid4()),
                start_date=start_date or returns.index[0],
                end_date=end_date or returns.index[-1],
                metrics=metrics,
                attribution=attribution,
                benchmark_comparison=benchmark_comparison
            )
            
            # Store report
            self.reports.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return self._create_default_report()
            
    def _calculate_benchmark_comparison(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        try:
            comparison = {}
            
            # Return comparison
            portfolio_return = returns.mean() * 252
            benchmark_return = benchmark_returns.mean() * 252
            comparison['excess_return'] = portfolio_return - benchmark_return
            
            # Risk comparison
            portfolio_vol = returns.std() * np.sqrt(252)
            benchmark_vol = benchmark_returns.std() * np.sqrt(252)
            comparison['excess_volatility'] = portfolio_vol - benchmark_vol
            
            # Sharpe ratio comparison
            portfolio_sharpe = self.calculator._calculate_sharpe_ratio(returns)
            benchmark_sharpe = self.calculator._calculate_sharpe_ratio(benchmark_returns)
            comparison['excess_sharpe'] = portfolio_sharpe - benchmark_sharpe
            
            # Information ratio
            comparison['information_ratio'] = self.calculator._calculate_information_ratio(returns, benchmark_returns)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error calculating benchmark comparison: {e}")
            return {}
            
    def _create_default_report(self) -> PerformanceReport:
        """Create default performance report."""
        return PerformanceReport(
            report_id=str(uuid.uuid4()),
            start_date=datetime.now(),
            end_date=datetime.now(),
            metrics=PerformanceMetrics(),
            attribution=[],
            benchmark_comparison={}
        )
        
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary of all reports."""
        if not self.reports:
            return {}
            
        summary = {
            'total_reports': len(self.reports),
            'average_sharpe_ratio': np.mean([r.metrics.sharpe_ratio for r in self.reports]),
            'average_max_drawdown': np.mean([r.metrics.max_drawdown for r in self.reports]),
            'average_total_return': np.mean([r.metrics.total_return for r in self.reports]),
            'recent_reports': len(self.reports[-5:])  # Last 5 reports
        }
        
        return summary

def create_performance_analytics(config: AnalyticsConfig = None) -> PerformanceAnalytics:
    """Create a performance analytics system."""
    return PerformanceAnalytics(config or AnalyticsConfig())

# Demo of performance analytics system
if __name__ == "__main__":
    # Create performance analytics
    config = AnalyticsConfig(
        risk_free_rate=0.02,
        benchmark_return=0.08,
        benchmark_volatility=0.15,
        enable_attribution=True,
        enable_benchmark_comparison=True
    )
    
    analytics = create_performance_analytics(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    portfolio_returns = pd.Series(np.random.randn(252) * 0.02 + 0.001, index=dates)
    benchmark_returns = pd.Series(np.random.randn(252) * 0.015 + 0.0005, index=dates)
    
    # Sample weights
    portfolio_weights = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4}
    benchmark_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.5}
    
    # Generate performance report
    report = analytics.generate_report(
        returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        portfolio_weights=portfolio_weights,
        benchmark_weights=benchmark_weights
    )
    
    print(f"Performance Report:")
    print(f"Total Return: {report.metrics.total_return:.4f}")
    print(f"Annualized Return: {report.metrics.annualized_return:.4f}")
    print(f"Sharpe Ratio: {report.metrics.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {report.metrics.max_drawdown:.4f}")
    print(f"Win Rate: {report.metrics.win_rate:.4f}")
    
    print(f"\nAttribution Analysis:")
    for attribution in report.attribution:
        print(f"{attribution.attribution_type.value}: {attribution.contribution:.4f}")
    
    print(f"\nBenchmark Comparison:")
    for metric, value in report.benchmark_comparison.items():
        print(f"{metric}: {value:.4f}")
    
    # Get summary
    summary = analytics.get_report_summary()
    print(f"\nAnalytics Summary:")
    print(f"Total reports: {summary.get('total_reports', 0)}")
    print(f"Average Sharpe ratio: {summary.get('average_sharpe_ratio', 0):.4f}")
    
    print("Performance analytics system completed successfully!")