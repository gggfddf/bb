#!/usr/bin/env python3
"""
Strategy Ranking System Module

Implements comprehensive strategy ranking systems for trading strategies:
- Multi-criteria ranking algorithms
- Risk-adjusted performance metrics
- Strategy stability assessment
- Ranking aggregation methods
- Dynamic ranking updates
- Ranking visualization tools

Features:
- Multiple ranking algorithms (TOPSIS, AHP, ELECTRE)
- Risk-adjusted performance evaluation
- Strategy stability and consistency analysis
- Dynamic ranking updates based on market conditions
- Ranking aggregation and consensus methods
- Comprehensive ranking visualization
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
import seaborn as sns
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
import joblib
import pickle

logger = structlog.get_logger()

class RankingMethod(Enum):
    """Ranking methods for strategy evaluation."""
    TOPSIS = "topsis"
    AHP = "ahp"
    ELECTRE = "electre"
    SIMPLE_WEIGHTED = "simple_weighted"
    CONCORDANCE = "concordance"

class MetricType(Enum):
    """Types of performance metrics."""
    RETURN = "return"
    RISK = "risk"
    SHARPE = "sharpe"
    DRAWDOWN = "drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    STABILITY = "stability"

@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    stability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RankingConfig:
    """Configuration for strategy ranking."""
    ranking_method: RankingMethod
    metrics: List[MetricType]
    weights: List[float]
    normalization_method: str = "min_max"
    stability_window: int = 30
    update_frequency: str = "daily"
    enable_dynamic_weights: bool = True

@dataclass
class RankingResult:
    """Result from strategy ranking."""
    rankings: Dict[str, float]
    scores: Dict[str, float]
    stability_analysis: Dict[str, Any]
    ranking_history: List[Dict[str, Any]]
    consensus_ranking: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceCalculator:
    """Calculate performance metrics for strategies."""
    
    def __init__(self):
        """Initialize performance calculator."""
        pass
    
    def calculate_performance(self, returns: np.ndarray, strategy_name: str) -> StrategyPerformance:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Strategy returns
            strategy_name: Name of the strategy
            
        Returns:
            Strategy performance metrics
        """
        if len(returns) == 0:
            return StrategyPerformance(
                strategy_name=strategy_name,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                win_rate=0.0,
                profit_factor=1.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                stability_score=0.0
            )
        
        # Basic metrics
        total_return = self._calculate_total_return(returns)
        volatility = self._calculate_volatility(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        win_rate = self._calculate_win_rate(returns)
        profit_factor = self._calculate_profit_factor(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, total_return, max_drawdown)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        stability_score = self._calculate_stability_score(returns)
        
        return StrategyPerformance(
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            stability_score=stability_score
        )
    
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return."""
        return np.prod(1 + returns) - 1
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        return np.std(returns) * np.sqrt(252)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return -np.min(drawdown)
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate."""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf if len(positive_returns) > 0 else 1.0
        
        gross_profit = np.sum(positive_returns)
        gross_loss = abs(np.sum(negative_returns))
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, total_return: float, max_dd: float) -> float:
        """Calculate Calmar ratio."""
        if max_dd == 0:
            return 0.0
        return total_return / max_dd
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_stability_score(self, returns: np.ndarray, window: int = 30) -> float:
        """Calculate stability score based on rolling performance."""
        if len(returns) < window:
            return 0.0
        
        # Calculate rolling Sharpe ratios
        rolling_sharpes = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            sharpe = self._calculate_sharpe_ratio(window_returns)
            rolling_sharpes.append(sharpe)
        
        if len(rolling_sharpes) == 0:
            return 0.0
        
        # Stability is inverse of coefficient of variation
        mean_sharpe = np.mean(rolling_sharpes)
        std_sharpe = np.std(rolling_sharpes)
        
        if mean_sharpe == 0:
            return 0.0
        
        return mean_sharpe / (std_sharpe + 1e-8)

class TOPSISRanker:
    """TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) ranking."""
    
    def __init__(self, weights: List[float], normalization_method: str = "min_max"):
        """
        Initialize TOPSIS ranker.
        
        Args:
            weights: Weights for each criterion
            normalization_method: Method for normalization
        """
        self.weights = np.array(weights)
        self.normalization_method = normalization_method
    
    def rank(self, performance_data: List[StrategyPerformance], 
             metrics: List[MetricType]) -> Dict[str, float]:
        """
        Rank strategies using TOPSIS.
        
        Args:
            performance_data: List of strategy performances
            metrics: List of metrics to consider
            
        Returns:
            Dictionary mapping strategy names to scores
        """
        # Create decision matrix
        decision_matrix = self._create_decision_matrix(performance_data, metrics)
        
        # Normalize decision matrix
        normalized_matrix = self._normalize_matrix(decision_matrix)
        
        # Calculate weighted normalized matrix
        weighted_matrix = normalized_matrix * self.weights.reshape(1, -1)
        
        # Find ideal and anti-ideal solutions
        ideal_solution = np.max(weighted_matrix, axis=0)
        anti_ideal_solution = np.min(weighted_matrix, axis=0)
        
        # Calculate distances to ideal and anti-ideal solutions
        distances_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution)**2, axis=1))
        distances_to_anti_ideal = np.sqrt(np.sum((weighted_matrix - anti_ideal_solution)**2, axis=1))
        
        # Calculate relative closeness
        relative_closeness = distances_to_anti_ideal / (distances_to_ideal + distances_to_anti_ideal)
        
        # Create ranking dictionary
        strategy_names = [perf.strategy_name for perf in performance_data]
        rankings = dict(zip(strategy_names, relative_closeness))
        
        return rankings
    
    def _create_decision_matrix(self, performance_data: List[StrategyPerformance], 
                              metrics: List[MetricType]) -> np.ndarray:
        """Create decision matrix from performance data."""
        n_strategies = len(performance_data)
        n_metrics = len(metrics)
        
        matrix = np.zeros((n_strategies, n_metrics))
        
        for i, perf in enumerate(performance_data):
            for j, metric in enumerate(metrics):
                if metric == MetricType.RETURN:
                    matrix[i, j] = perf.total_return
                elif metric == MetricType.SHARPE:
                    matrix[i, j] = perf.sharpe_ratio
                elif metric == MetricType.DRAWDOWN:
                    matrix[i, j] = -perf.max_drawdown  # Negative because lower is better
                elif metric == MetricType.WIN_RATE:
                    matrix[i, j] = perf.win_rate
                elif metric == MetricType.PROFIT_FACTOR:
                    matrix[i, j] = perf.profit_factor
                elif metric == MetricType.STABILITY:
                    matrix[i, j] = perf.stability_score
                else:
                    matrix[i, j] = 0.0
        
        return matrix
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize decision matrix."""
        if self.normalization_method == "min_max":
            min_vals = np.min(matrix, axis=0)
            max_vals = np.max(matrix, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            return (matrix - min_vals) / range_vals
        elif self.normalization_method == "z_score":
            mean_vals = np.mean(matrix, axis=0)
            std_vals = np.std(matrix, axis=0)
            std_vals[std_vals == 0] = 1  # Avoid division by zero
            return (matrix - mean_vals) / std_vals
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

class AHPRanker:
    """AHP (Analytic Hierarchy Process) ranking."""
    
    def __init__(self, weights: List[float]):
        """
        Initialize AHP ranker.
        
        Args:
            weights: Weights for each criterion
        """
        self.weights = np.array(weights)
    
    def rank(self, performance_data: List[StrategyPerformance], 
             metrics: List[MetricType]) -> Dict[str, float]:
        """
        Rank strategies using AHP.
        
        Args:
            performance_data: List of strategy performances
            metrics: List of metrics to consider
            
        Returns:
            Dictionary mapping strategy names to scores
        """
        # Create decision matrix
        decision_matrix = self._create_decision_matrix(performance_data, metrics)
        
        # Normalize matrix
        normalized_matrix = self._normalize_matrix(decision_matrix)
        
        # Calculate weighted scores
        weighted_scores = normalized_matrix @ self.weights
        
        # Create ranking dictionary
        strategy_names = [perf.strategy_name for perf in performance_data]
        rankings = dict(zip(strategy_names, weighted_scores))
        
        return rankings
    
    def _create_decision_matrix(self, performance_data: List[StrategyPerformance], 
                              metrics: List[MetricType]) -> np.ndarray:
        """Create decision matrix from performance data."""
        n_strategies = len(performance_data)
        n_metrics = len(metrics)
        
        matrix = np.zeros((n_strategies, n_metrics))
        
        for i, perf in enumerate(performance_data):
            for j, metric in enumerate(metrics):
                if metric == MetricType.RETURN:
                    matrix[i, j] = perf.total_return
                elif metric == MetricType.SHARPE:
                    matrix[i, j] = perf.sharpe_ratio
                elif metric == MetricType.DRAWDOWN:
                    matrix[i, j] = -perf.max_drawdown
                elif metric == MetricType.WIN_RATE:
                    matrix[i, j] = perf.win_rate
                elif metric == MetricType.PROFIT_FACTOR:
                    matrix[i, j] = perf.profit_factor
                elif metric == MetricType.STABILITY:
                    matrix[i, j] = perf.stability_score
                else:
                    matrix[i, j] = 0.0
        
        return matrix
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize decision matrix using sum normalization."""
        col_sums = np.sum(matrix, axis=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        return matrix / col_sums

class StabilityAnalyzer:
    """Analyze strategy stability over time."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize stability analyzer.
        
        Args:
            window_size: Size of rolling window
        """
        self.window_size = window_size
    
    def analyze_stability(self, strategy_returns: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze stability of strategies.
        
        Args:
            strategy_returns: Dictionary mapping strategy names to returns
            
        Returns:
            Stability analysis results
        """
        stability_results = {}
        
        for strategy_name, returns in strategy_returns.items():
            if len(returns) < self.window_size:
                stability_results[strategy_name] = {
                    'stability_score': 0.0,
                    'consistency': 0.0,
                    'trend_stability': 0.0,
                    'volatility_stability': 0.0
                }
                continue
            
            # Calculate rolling metrics
            rolling_sharpes = self._calculate_rolling_sharpes(returns)
            rolling_volatilities = self._calculate_rolling_volatilities(returns)
            
            # Calculate stability metrics
            stability_score = self._calculate_stability_score(rolling_sharpes)
            consistency = self._calculate_consistency(rolling_sharpes)
            trend_stability = self._calculate_trend_stability(rolling_sharpes)
            volatility_stability = self._calculate_volatility_stability(rolling_volatilities)
            
            stability_results[strategy_name] = {
                'stability_score': stability_score,
                'consistency': consistency,
                'trend_stability': trend_stability,
                'volatility_stability': volatility_stability,
                'rolling_sharpes': rolling_sharpes,
                'rolling_volatilities': rolling_volatilities
            }
        
        return stability_results
    
    def _calculate_rolling_sharpes(self, returns: np.ndarray) -> np.ndarray:
        """Calculate rolling Sharpe ratios."""
        rolling_sharpes = []
        
        for i in range(self.window_size, len(returns)):
            window_returns = returns[i-self.window_size:i]
            if np.std(window_returns) > 0:
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
            else:
                sharpe = 0.0
            rolling_sharpes.append(sharpe)
        
        return np.array(rolling_sharpes)
    
    def _calculate_rolling_volatilities(self, returns: np.ndarray) -> np.ndarray:
        """Calculate rolling volatilities."""
        rolling_vols = []
        
        for i in range(self.window_size, len(returns)):
            window_returns = returns[i-self.window_size:i]
            vol = np.std(window_returns) * np.sqrt(252)
            rolling_vols.append(vol)
        
        return np.array(rolling_vols)
    
    def _calculate_stability_score(self, rolling_metrics: np.ndarray) -> float:
        """Calculate overall stability score."""
        if len(rolling_metrics) == 0:
            return 0.0
        
        # Stability is inverse of coefficient of variation
        mean_metric = np.mean(rolling_metrics)
        std_metric = np.std(rolling_metrics)
        
        if mean_metric == 0:
            return 0.0
        
        return mean_metric / (std_metric + 1e-8)
    
    def _calculate_consistency(self, rolling_metrics: np.ndarray) -> float:
        """Calculate consistency of performance."""
        if len(rolling_metrics) < 2:
            return 0.0
        
        # Calculate autocorrelation
        autocorr = np.corrcoef(rolling_metrics[:-1], rolling_metrics[1:])[0, 1]
        return max(0, autocorr) if not np.isnan(autocorr) else 0.0
    
    def _calculate_trend_stability(self, rolling_metrics: np.ndarray) -> float:
        """Calculate trend stability."""
        if len(rolling_metrics) < 2:
            return 0.0
        
        # Calculate trend and its stability
        x = np.arange(len(rolling_metrics))
        slope, _ = np.polyfit(x, rolling_metrics, 1)
        
        # Trend stability is inverse of trend magnitude
        return 1.0 / (1.0 + abs(slope))
    
    def _calculate_volatility_stability(self, rolling_volatilities: np.ndarray) -> float:
        """Calculate volatility stability."""
        return self._calculate_stability_score(rolling_volatilities)

class StrategyRanker:
    """Main strategy ranking system."""
    
    def __init__(self, config: RankingConfig):
        """
        Initialize strategy ranker.
        
        Args:
            config: Ranking configuration
        """
        self.config = config
        self.performance_calculator = PerformanceCalculator()
        self.stability_analyzer = StabilityAnalyzer(config.stability_window)
        
        # Initialize rankers
        if config.ranking_method == RankingMethod.TOPSIS:
            self.ranker = TOPSISRanker(config.weights, config.normalization_method)
        elif config.ranking_method == RankingMethod.AHP:
            self.ranker = AHPRanker(config.weights)
        else:
            raise ValueError(f"Unsupported ranking method: {config.ranking_method}")
        
        # Ranking history
        self.ranking_history = []
        
        logger.info("Strategy ranker initialized", 
                   method=config.ranking_method.value,
                   n_metrics=len(config.metrics))
    
    def rank_strategies(self, strategy_returns: Dict[str, np.ndarray]) -> RankingResult:
        """
        Rank strategies based on performance.
        
        Args:
            strategy_returns: Dictionary mapping strategy names to returns
            
        Returns:
            Ranking results
        """
        logger.info("Starting strategy ranking", n_strategies=len(strategy_returns))
        
        # Calculate performance metrics
        performance_data = []
        for strategy_name, returns in strategy_returns.items():
            performance = self.performance_calculator.calculate_performance(returns, strategy_name)
            performance_data.append(performance)
        
        # Analyze stability
        stability_analysis = self.stability_analyzer.analyze_stability(strategy_returns)
        
        # Update performance data with stability scores
        for perf in performance_data:
            if perf.strategy_name in stability_analysis:
                perf.stability_score = stability_analysis[perf.strategy_name]['stability_score']
        
        # Rank strategies
        rankings = self.ranker.rank(performance_data, self.config.metrics)
        
        # Calculate consensus ranking
        consensus_ranking = self._calculate_consensus_ranking(rankings)
        
        # Record ranking
        ranking_record = {
            'timestamp': datetime.now(),
            'rankings': rankings.copy(),
            'consensus_ranking': consensus_ranking.copy(),
            'n_strategies': len(strategy_returns)
        }
        self.ranking_history.append(ranking_record)
        
        # Create result
        result = RankingResult(
            rankings=rankings,
            scores=rankings,  # For now, scores are the same as rankings
            stability_analysis=stability_analysis,
            ranking_history=self.ranking_history,
            consensus_ranking=consensus_ranking
        )
        
        logger.info("Strategy ranking completed", 
                   top_strategy=max(rankings, key=rankings.get) if rankings else None)
        
        return result
    
    def _calculate_consensus_ranking(self, rankings: Dict[str, float]) -> Dict[str, int]:
        """Calculate consensus ranking from scores."""
        if not rankings:
            return {}
        
        # Sort strategies by score (descending)
        sorted_strategies = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        # Assign ranks (1-based)
        consensus_ranking = {}
        for rank, (strategy_name, _) in enumerate(sorted_strategies, 1):
            consensus_ranking[strategy_name] = rank
        
        return consensus_ranking
    
    def get_top_strategies(self, rankings: Dict[str, float], n: int = 5) -> List[str]:
        """Get top N strategies."""
        if not rankings:
            return []
        
        sorted_strategies = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        return [strategy for strategy, _ in sorted_strategies[:n]]
    
    def plot_rankings(self, result: RankingResult, save_path: str = None):
        """Plot ranking results."""
        if not result.rankings:
            logger.warning("No rankings to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot current rankings
        strategies = list(result.rankings.keys())
        scores = list(result.rankings.values())
        
        ax1.barh(strategies, scores)
        ax1.set_xlabel('Ranking Score')
        ax1.set_title('Strategy Rankings')
        ax1.grid(True, alpha=0.3)
        
        # Plot consensus rankings
        consensus_items = sorted(result.consensus_ranking.items(), key=lambda x: x[1])
        consensus_strategies = [item[0] for item in consensus_items]
        consensus_ranks = [item[1] for item in consensus_items]
        
        ax2.barh(consensus_strategies, consensus_ranks)
        ax2.set_xlabel('Rank')
        ax2.set_title('Consensus Rankings')
        ax2.grid(True, alpha=0.3)
        
        # Plot stability analysis
        if result.stability_analysis:
            stability_scores = [result.stability_analysis.get(name, {}).get('stability_score', 0) 
                              for name in strategies]
            ax3.barh(strategies, stability_scores)
            ax3.set_xlabel('Stability Score')
            ax3.set_title('Strategy Stability')
            ax3.grid(True, alpha=0.3)
        
        # Plot ranking history (if available)
        if len(self.ranking_history) > 1:
            timestamps = [record['timestamp'] for record in self.ranking_history]
            top_strategy_scores = [record['rankings'].get(max(record['rankings'], key=record['rankings'].get), 0) 
                                 for record in self.ranking_history]
            
            ax4.plot(timestamps, top_strategy_scores, 'bo-')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Top Strategy Score')
            ax4.set_title('Top Strategy Score Over Time')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def create_strategy_ranker(ranking_method: str = "topsis",
                          metrics: List[str] = None,
                          weights: List[float] = None) -> StrategyRanker:
    """
    Create a strategy ranker.
    
    Args:
        ranking_method: Method for ranking
        metrics: List of metrics to consider
        weights: Weights for each metric
        
    Returns:
        StrategyRanker instance
    """
    if metrics is None:
        metrics = ['sharpe', 'return', 'drawdown', 'stability']
    
    if weights is None:
        weights = [1.0 / len(metrics)] * len(metrics)
    
    # Convert string metrics to MetricType
    metric_types = []
    for metric in metrics:
        if metric == 'return':
            metric_types.append(MetricType.RETURN)
        elif metric == 'sharpe':
            metric_types.append(MetricType.SHARPE)
        elif metric == 'drawdown':
            metric_types.append(MetricType.DRAWDOWN)
        elif metric == 'win_rate':
            metric_types.append(MetricType.WIN_RATE)
        elif metric == 'profit_factor':
            metric_types.append(MetricType.PROFIT_FACTOR)
        elif metric == 'stability':
            metric_types.append(MetricType.STABILITY)
        else:
            logger.warning(f"Unknown metric: {metric}")
    
    config = RankingConfig(
        ranking_method=RankingMethod(ranking_method),
        metrics=metric_types,
        weights=weights
    )
    
    return StrategyRanker(config)

if __name__ == "__main__":
    # Demo of strategy ranking
    np.random.seed(42)
    
    # Generate sample strategy returns
    n_days = 252
    n_strategies = 5
    
    strategy_returns = {}
    strategy_names = ['Strategy_A', 'Strategy_B', 'Strategy_C', 'Strategy_D', 'Strategy_E']
    
    for i, name in enumerate(strategy_names):
        # Generate returns with different characteristics
        base_return = 0.0001 * (i + 1)
        volatility = 0.02 + 0.005 * i
        returns = np.random.normal(base_return, volatility, n_days)
        strategy_returns[name] = returns
    
    # Create strategy ranker
    ranker = create_strategy_ranker(
        ranking_method="topsis",
        metrics=['sharpe', 'return', 'drawdown', 'stability'],
        weights=[0.4, 0.3, 0.2, 0.1]
    )
    
    # Rank strategies
    result = ranker.rank_strategies(strategy_returns)
    
    print("Strategy Rankings:")
    for strategy, score in sorted(result.rankings.items(), key=lambda x: x[1], reverse=True):
        print(f"{strategy}: {score:.4f}")
    
    print(f"\nTop 3 strategies: {ranker.get_top_strategies(result.rankings, 3)}")
    
    # Plot results
    ranker.plot_rankings(result)