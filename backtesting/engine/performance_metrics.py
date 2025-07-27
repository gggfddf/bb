"""
Performance Metrics Calculation

A comprehensive performance metrics calculation module for trading strategies.
Implements various return and risk metrics used in quantitative finance.

Features:
- Return calculations (total, annualized, geometric)
- Risk metrics (volatility, VaR, CVaR)
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis (maximum drawdown, drawdown duration)
- Information ratio and other advanced metrics
- Rolling metrics calculation
- Benchmark comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize

logger = structlog.get_logger()

class MetricType(Enum):
    """Types of performance metrics."""
    RETURN = "return"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN = "drawdown"
    ADVANCED = "advanced"

@dataclass
class PerformanceMetrics:
    """Container for performance metrics results."""
    returns: pd.Series
    metrics: Dict[str, float]
    rolling_metrics: Dict[str, pd.Series] = field(default_factory=dict)
    drawdown_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_comparison: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ReturnCalculator:
    """Calculate various types of returns."""
    
    @staticmethod
    def calculate_total_return(prices: pd.Series) -> float:
        """Calculate total return from price series."""
        if len(prices) < 2:
            return 0.0
        
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    
    @staticmethod
    def calculate_annualized_return(prices: pd.Series, 
                                  periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        if len(prices) < 2:
            return 0.0
        
        total_return = ReturnCalculator.calculate_total_return(prices)
        num_periods = len(prices) - 1
        
        if num_periods <= 0:
            return 0.0
        
        # Annualized return = (1 + total_return)^(periods_per_year/num_periods) - 1
        annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
        
        return annualized_return
    
    @staticmethod
    def calculate_geometric_mean_return(returns: pd.Series, 
                                      periods_per_year: int = 252) -> float:
        """Calculate geometric mean return."""
        if len(returns) == 0:
            return 0.0
        
        # Remove any zero or negative returns that would cause issues
        positive_returns = returns[returns > -1]
        
        if len(positive_returns) == 0:
            return 0.0
        
        # Geometric mean = (product of (1 + r))^(1/n) - 1
        geometric_mean = np.prod(1 + positive_returns) ** (1 / len(positive_returns)) - 1
        
        # Annualize
        annualized_geometric = (1 + geometric_mean) ** periods_per_year - 1
        
        return annualized_geometric
    
    @staticmethod
    def calculate_rolling_returns(prices: pd.Series, 
                                window: int = 252) -> pd.Series:
        """Calculate rolling returns over a specified window."""
        if len(prices) < window + 1:
            return pd.Series(dtype=float)
        
        rolling_returns = prices.pct_change(window)
        return rolling_returns

class RiskCalculator:
    """Calculate various risk metrics."""
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, 
                           periods_per_year: int = 252) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 2:
            return 0.0
        
        # Calculate standard deviation and annualize
        volatility = clean_returns.std() * np.sqrt(periods_per_year)
        
        return volatility
    
    @staticmethod
    def calculate_var(returns: pd.Series, 
                     confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return 0.0
        
        # Calculate VaR using historical simulation
        var = np.percentile(clean_returns, confidence_level * 100)
        
        return var
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, 
                      confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% CVaR)
        
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return 0.0
        
        # Calculate VaR first
        var = RiskCalculator.calculate_var(clean_returns, confidence_level)
        
        # Calculate CVaR as the mean of returns below VaR
        tail_returns = clean_returns[clean_returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        cvar = tail_returns.mean()
        
        return cvar
    
    @staticmethod
    def calculate_downside_deviation(returns: pd.Series, 
                                   target_return: float = 0.0,
                                   periods_per_year: int = 252) -> float:
        """Calculate downside deviation (for Sortino ratio)."""
        if len(returns) == 0:
            return 0.0
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return 0.0
        
        # Calculate downside returns
        downside_returns = clean_returns[clean_returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        # Calculate downside deviation and annualize
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        
        return downside_deviation
    
    @staticmethod
    def calculate_rolling_volatility(returns: pd.Series, 
                                   window: int = 252) -> pd.Series:
        """Calculate rolling volatility."""
        if len(returns) < window:
            return pd.Series(dtype=float)
        
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        return rolling_vol

class DrawdownCalculator:
    """Calculate drawdown metrics."""
    
    @staticmethod
    def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        if len(equity_curve) < 2:
            return pd.Series(dtype=float)
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        drawdown_series = DrawdownCalculator.calculate_drawdown_series(equity_curve)
        
        if len(drawdown_series) == 0:
            return 0.0
        
        max_drawdown = drawdown_series.min()
        
        return max_drawdown
    
    @staticmethod
    def calculate_drawdown_duration(equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown duration statistics."""
        drawdown_series = DrawdownCalculator.calculate_drawdown_series(equity_curve)
        
        if len(drawdown_series) == 0:
            return {
                'max_duration': 0,
                'avg_duration': 0,
                'current_duration': 0
            }
        
        # Find drawdown periods (consecutive negative values)
        drawdown_periods = []
        current_period_start = None
        
        for i, drawdown in enumerate(drawdown_series):
            if drawdown < 0 and current_period_start is None:
                current_period_start = i
            elif drawdown >= 0 and current_period_start is not None:
                drawdown_periods.append(i - current_period_start)
                current_period_start = None
        
        # Handle case where drawdown period extends to the end
        if current_period_start is not None:
            drawdown_periods.append(len(drawdown_series) - current_period_start)
        
        if not drawdown_periods:
            return {
                'max_duration': 0,
                'avg_duration': 0,
                'current_duration': 0
            }
        
        return {
            'max_duration': max(drawdown_periods),
            'avg_duration': np.mean(drawdown_periods),
            'current_duration': drawdown_periods[-1] if drawdown_periods else 0
        }

class RiskAdjustedReturnCalculator:
    """Calculate risk-adjusted return metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                              risk_free_rate: float = 0.02,
                              periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = clean_returns - risk_free_rate / periods_per_year
        
        # Calculate Sharpe ratio
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
        
        return sharpe_ratio
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, 
                               risk_free_rate: float = 0.02,
                               target_return: float = 0.0,
                               periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = clean_returns - risk_free_rate / periods_per_year
        
        # Calculate downside deviation
        downside_deviation = RiskCalculator.calculate_downside_deviation(
            clean_returns, target_return, periods_per_year
        )
        
        if downside_deviation == 0:
            return 0.0
        
        # Calculate Sortino ratio
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year)
        
        return sortino_ratio
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, 
                              equity_curve: pd.Series,
                              periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio."""
        if len(returns) < 2 or len(equity_curve) < 2:
            return 0.0
        
        # Calculate annualized return
        annualized_return = ReturnCalculator.calculate_annualized_return(equity_curve, periods_per_year)
        
        # Calculate maximum drawdown
        max_drawdown = abs(DrawdownCalculator.calculate_max_drawdown(equity_curve))
        
        if max_drawdown == 0:
            return 0.0
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / max_drawdown
        
        return calmar_ratio
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series, 
                                  benchmark_returns: pd.Series,
                                  periods_per_year: int = 252) -> float:
        """Calculate information ratio."""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        # Align the series
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        strategy_returns = aligned_data.iloc[:, 0]
        benchmark_returns_aligned = aligned_data.iloc[:, 1]
        
        # Calculate active returns
        active_returns = strategy_returns - benchmark_returns_aligned
        
        # Calculate information ratio
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(periods_per_year)
        
        return information_ratio
    
    @staticmethod
    def calculate_treynor_ratio(returns: pd.Series, 
                               benchmark_returns: pd.Series,
                               risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
        """Calculate Treynor ratio."""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        # Align the series
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        strategy_returns = aligned_data.iloc[:, 0]
        benchmark_returns_aligned = aligned_data.iloc[:, 1]
        
        # Calculate beta
        covariance = np.cov(strategy_returns, benchmark_returns_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_returns_aligned)
        
        if benchmark_variance == 0:
            return 0.0
        
        beta = covariance / benchmark_variance
        
        if beta == 0:
            return 0.0
        
        # Calculate excess return
        excess_return = strategy_returns.mean() - risk_free_rate / periods_per_year
        
        # Calculate Treynor ratio
        treynor_ratio = excess_return / beta * periods_per_year
        
        return treynor_ratio

class AdvancedMetricsCalculator:
    """Calculate advanced performance metrics."""
    
    @staticmethod
    def calculate_ulcer_index(returns: pd.Series) -> float:
        """Calculate Ulcer Index."""
        if len(returns) < 2:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate drawdown series
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Calculate Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        
        return ulcer_index
    
    @staticmethod
    def calculate_gain_to_pain_ratio(returns: pd.Series) -> float:
        """Calculate Gain-to-Pain ratio."""
        if len(returns) == 0:
            return 0.0
        
        # Separate gains and losses
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_gains = gains.sum() if len(gains) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        
        if total_losses == 0:
            return float('inf') if total_gains > 0 else 0
        
        gain_to_pain_ratio = total_gains / total_losses
        
        return gain_to_pain_ratio
    
    @staticmethod
    def calculate_profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor."""
        if len(returns) == 0:
            return 0.0
        
        # Separate gains and losses
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_gains = gains.sum() if len(gains) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        
        if total_losses == 0:
            return float('inf') if total_gains > 0 else 0
        
        profit_factor = total_gains / total_losses
        
        return profit_factor
    
    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """Calculate win rate."""
        if len(returns) == 0:
            return 0.0
        
        winning_trades = len(returns[returns > 0])
        total_trades = len(returns)
        
        win_rate = winning_trades / total_trades
        
        return win_rate
    
    @staticmethod
    def calculate_avg_win_loss_ratio(returns: pd.Series) -> float:
        """Calculate average win/loss ratio."""
        if len(returns) == 0:
            return 0.0
        
        # Separate gains and losses
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(gains) == 0 or len(losses) == 0:
            return 0.0
        
        avg_gain = gains.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0:
            return float('inf') if avg_gain > 0 else 0
        
        avg_win_loss_ratio = avg_gain / avg_loss
        
        return avg_win_loss_ratio

class PerformanceMetricsCalculator:
    """Main performance metrics calculator."""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252,
                 confidence_level: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.confidence_level = confidence_level
    
    def calculate_all_metrics(self, 
                            equity_curve: pd.Series,
                            benchmark_equity: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            equity_curve: Series of equity values
            benchmark_equity: Optional benchmark equity curve for comparison
        
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        logger.info("Calculating performance metrics", 
                   data_length=len(equity_curve))
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Return metrics
        metrics.update(self._calculate_return_metrics(equity_curve, returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted return metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns, equity_curve))
        
        # Drawdown metrics
        drawdown_series = DrawdownCalculator.calculate_drawdown_series(equity_curve)
        metrics.update(self._calculate_drawdown_metrics(equity_curve, drawdown_series))
        
        # Advanced metrics
        metrics.update(self._calculate_advanced_metrics(returns))
        
        # Benchmark comparison
        benchmark_comparison = {}
        if benchmark_equity is not None:
            benchmark_comparison = self._calculate_benchmark_comparison(
                equity_curve, returns, benchmark_equity
            )
        
        # Rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(equity_curve, returns)
        
        # Create result object
        result = PerformanceMetrics(
            returns=returns,
            metrics=metrics,
            rolling_metrics=rolling_metrics,
            drawdown_series=drawdown_series,
            benchmark_comparison=benchmark_comparison,
            metadata={
                'risk_free_rate': self.risk_free_rate,
                'periods_per_year': self.periods_per_year,
                'confidence_level': self.confidence_level
            }
        )
        
        logger.info("Performance metrics calculation completed",
                   num_metrics=len(metrics))
        
        return result
    
    def _calculate_return_metrics(self, equity_curve: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate return-related metrics."""
        metrics = {}
        
        # Total return
        metrics['total_return'] = ReturnCalculator.calculate_total_return(equity_curve)
        
        # Annualized return
        metrics['annualized_return'] = ReturnCalculator.calculate_annualized_return(
            equity_curve, self.periods_per_year
        )
        
        # Geometric mean return
        metrics['geometric_mean_return'] = ReturnCalculator.calculate_geometric_mean_return(
            returns, self.periods_per_year
        )
        
        # Average return
        metrics['average_return'] = returns.mean() * self.periods_per_year
        
        return metrics
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        metrics = {}
        
        # Volatility
        metrics['volatility'] = RiskCalculator.calculate_volatility(
            returns, self.periods_per_year
        )
        
        # VaR
        metrics['var'] = RiskCalculator.calculate_var(returns, self.confidence_level)
        
        # CVaR
        metrics['cvar'] = RiskCalculator.calculate_cvar(returns, self.confidence_level)
        
        # Downside deviation
        metrics['downside_deviation'] = RiskCalculator.calculate_downside_deviation(
            returns, 0.0, self.periods_per_year
        )
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        metrics = {}
        
        # Sharpe ratio
        metrics['sharpe_ratio'] = RiskAdjustedReturnCalculator.calculate_sharpe_ratio(
            returns, self.risk_free_rate, self.periods_per_year
        )
        
        # Sortino ratio
        metrics['sortino_ratio'] = RiskAdjustedReturnCalculator.calculate_sortino_ratio(
            returns, self.risk_free_rate, 0.0, self.periods_per_year
        )
        
        # Calmar ratio
        metrics['calmar_ratio'] = RiskAdjustedReturnCalculator.calculate_calmar_ratio(
            returns, equity_curve, self.periods_per_year
        )
        
        return metrics
    
    def _calculate_drawdown_metrics(self, equity_curve: pd.Series, drawdown_series: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-related metrics."""
        metrics = {}
        
        # Maximum drawdown
        metrics['max_drawdown'] = DrawdownCalculator.calculate_max_drawdown(equity_curve)
        
        # Drawdown duration
        duration_stats = DrawdownCalculator.calculate_drawdown_duration(equity_curve)
        metrics.update({
            'max_drawdown_duration': duration_stats['max_duration'],
            'avg_drawdown_duration': duration_stats['avg_duration'],
            'current_drawdown_duration': duration_stats['current_duration']
        })
        
        return metrics
    
    def _calculate_advanced_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate advanced performance metrics."""
        metrics = {}
        
        # Ulcer Index
        metrics['ulcer_index'] = AdvancedMetricsCalculator.calculate_ulcer_index(returns)
        
        # Gain-to-Pain ratio
        metrics['gain_to_pain_ratio'] = AdvancedMetricsCalculator.calculate_gain_to_pain_ratio(returns)
        
        # Profit factor
        metrics['profit_factor'] = AdvancedMetricsCalculator.calculate_profit_factor(returns)
        
        # Win rate
        metrics['win_rate'] = AdvancedMetricsCalculator.calculate_win_rate(returns)
        
        # Average win/loss ratio
        metrics['avg_win_loss_ratio'] = AdvancedMetricsCalculator.calculate_avg_win_loss_ratio(returns)
        
        return metrics
    
    def _calculate_benchmark_comparison(self, 
                                      equity_curve: pd.Series,
                                      returns: pd.Series,
                                      benchmark_equity: pd.Series) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        comparison = {}
        
        # Calculate benchmark returns
        benchmark_returns = benchmark_equity.pct_change().dropna()
        
        # Information ratio
        comparison['information_ratio'] = RiskAdjustedReturnCalculator.calculate_information_ratio(
            returns, benchmark_returns, self.periods_per_year
        )
        
        # Treynor ratio
        comparison['treynor_ratio'] = RiskAdjustedReturnCalculator.calculate_treynor_ratio(
            returns, benchmark_returns, self.risk_free_rate, self.periods_per_year
        )
        
        # Beta
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) >= 2:
            strategy_returns = aligned_data.iloc[:, 0]
            benchmark_returns_aligned = aligned_data.iloc[:, 1]
            
            covariance = np.cov(strategy_returns, benchmark_returns_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_returns_aligned)
            
            if benchmark_variance > 0:
                comparison['beta'] = covariance / benchmark_variance
            else:
                comparison['beta'] = 0.0
        else:
            comparison['beta'] = 0.0
        
        return comparison
    
    def _calculate_rolling_metrics(self, equity_curve: pd.Series, returns: pd.Series) -> Dict[str, pd.Series]:
        """Calculate rolling metrics."""
        rolling_metrics = {}
        
        # Rolling returns
        rolling_metrics['rolling_returns'] = ReturnCalculator.calculate_rolling_returns(equity_curve, 252)
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = RiskCalculator.calculate_rolling_volatility(returns, 252)
        
        # Rolling Sharpe ratio
        if len(returns) >= 252:
            rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
            rolling_metrics['rolling_sharpe'] = rolling_sharpe
        
        return rolling_metrics

# Convenience functions
def calculate_performance_metrics(equity_curve: pd.Series,
                                risk_free_rate: float = 0.02,
                                benchmark_equity: Optional[pd.Series] = None) -> PerformanceMetrics:
    """Calculate performance metrics with default parameters."""
    calculator = PerformanceMetricsCalculator(risk_free_rate=risk_free_rate)
    return calculator.calculate_all_metrics(equity_curve, benchmark_equity)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    return RiskAdjustedReturnCalculator.calculate_sharpe_ratio(returns, risk_free_rate)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown."""
    return DrawdownCalculator.calculate_max_drawdown(equity_curve)

def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk."""
    return RiskCalculator.calculate_var(returns, confidence_level)