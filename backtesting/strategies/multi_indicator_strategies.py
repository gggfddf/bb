"""
Multi-Indicator Combination Strategy Testing Framework

A comprehensive framework for testing trading strategies that combine multiple indicators
using ensemble methods, voting systems, and optimization techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime
import warnings

try:
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = structlog.get_logger()

class CombinationMethod(Enum):
    """Methods for combining multiple indicators."""
    VOTING = "voting"
    WEIGHTED_AVERAGE = "weighted_average"
    ENSEMBLE = "ensemble"
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"

class VotingType(Enum):
    """Types of voting systems."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    THRESHOLD = "threshold"

@dataclass
class MultiIndicatorConfig:
    """Configuration for multi-indicator strategy."""
    indicators: List[str]
    combination_method: CombinationMethod
    voting_type: VotingType = VotingType.MAJORITY
    weights: Optional[List[float]] = None
    thresholds: Dict[str, float] = field(default_factory=dict)
    min_agreement: float = 0.5
    ensemble_size: int = 10

@dataclass
class CombinationResult:
    """Results from multi-indicator strategy."""
    combined_signals: np.ndarray
    individual_signals: Dict[str, np.ndarray]
    agreement_scores: np.ndarray
    confidence_scores: np.ndarray
    performance_metrics: Dict[str, float]

class MultiIndicatorStrategy:
    """Base class for multi-indicator strategies."""
    
    def __init__(self, config: MultiIndicatorConfig):
        self.config = config
        self.individual_strategies = {}
        self.combined_signals = None
    
    def add_indicator_strategy(self, name: str, strategy: Any):
        """Add an individual indicator strategy."""
        self.individual_strategies[name] = strategy
        logger.info(f"Added indicator strategy: {name}")
    
    def generate_individual_signals(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate signals from all individual indicators."""
        signals = {}
        
        for name, strategy in self.individual_strategies.items():
            if hasattr(strategy, 'generate_signals'):
                signals[name] = strategy.generate_signals(data)
            else:
                logger.warning(f"Strategy {name} does not have generate_signals method")
        
        return signals
    
    def combine_signals_voting(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine signals using voting system."""
        if not signals:
            return np.zeros(len(next(iter(signals.values()))) if signals else 0)
        
        signal_matrix = np.array(list(signals.values()))
        n_indicators = len(signals)
        
        if self.config.voting_type == VotingType.MAJORITY:
            # Simple majority voting
            combined = np.sum(signal_matrix, axis=0)
            return np.where(combined > n_indicators / 2, 1, 
                           np.where(combined < -n_indicators / 2, -1, 0))
        
        elif self.config.voting_type == VotingType.WEIGHTED:
            # Weighted voting
            if self.config.weights is None:
                weights = np.ones(n_indicators) / n_indicators
            else:
                weights = np.array(self.config.weights)
                weights = weights / np.sum(weights)
            
            weighted_sum = np.sum(signal_matrix * weights[:, np.newaxis], axis=0)
            return np.where(weighted_sum > 0.1, 1, 
                           np.where(weighted_sum < -0.1, -1, 0))
        
        elif self.config.voting_type == VotingType.UNANIMOUS:
            # Unanimous voting (all indicators must agree)
            combined = np.sum(signal_matrix, axis=0)
            return np.where(combined == n_indicators, 1,
                           np.where(combined == -n_indicators, -1, 0))
        
        elif self.config.voting_type == VotingType.THRESHOLD:
            # Threshold-based voting
            agreement_ratio = np.sum(signal_matrix != 0, axis=0) / n_indicators
            combined = np.sum(signal_matrix, axis=0)
            
            return np.where((agreement_ratio >= self.config.min_agreement) & (combined > 0), 1,
                           np.where((agreement_ratio >= self.config.min_agreement) & (combined < 0), -1, 0))
        
        return np.zeros(len(next(iter(signals.values()))))
    
    def combine_signals_weighted_average(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine signals using weighted average."""
        if not signals:
            return np.zeros(len(next(iter(signals.values()))) if signals else 0)
        
        signal_matrix = np.array(list(signals.values()))
        
        if self.config.weights is None:
            weights = np.ones(len(signals)) / len(signals)
        else:
            weights = np.array(self.config.weights)
            weights = weights / np.sum(weights)
        
        weighted_avg = np.sum(signal_matrix * weights[:, np.newaxis], axis=0)
        
        # Apply thresholds
        buy_threshold = self.config.thresholds.get('buy', 0.3)
        sell_threshold = self.config.thresholds.get('sell', -0.3)
        
        return np.where(weighted_avg > buy_threshold, 1,
                       np.where(weighted_avg < sell_threshold, -1, 0))
    
    def combine_signals_ensemble(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine signals using ensemble methods."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for ensemble methods")
            return self.combine_signals_voting(signals)
        
        # Convert signals to classification problem
        signal_matrix = np.array(list(signals.values())).T
        
        # Create ensemble classifier
        estimators = []
        for i in range(self.config.ensemble_size):
            clf = RandomForestClassifier(n_estimators=10, random_state=42+i)
            estimators.append((f'rf_{i}', clf))
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Prepare training data
        X = signal_matrix[:-1]  # Use all but last for training
        y = np.where(signal_matrix[1:, 0] > 0, 1, 
                    np.where(signal_matrix[1:, 0] < 0, -1, 0))  # Use first indicator as target
        
        # Train ensemble
        ensemble.fit(X, y)
        
        # Predict
        predictions = ensemble.predict(signal_matrix)
        
        return predictions
    
    def combine_signals_sequential(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine signals using sequential logic."""
        if not signals:
            return np.zeros(len(next(iter(signals.values()))) if signals else 0)
        
        signal_names = list(signals.keys())
        combined = np.zeros(len(signals[signal_names[0]]))
        
        for i, name in enumerate(signal_names):
            signal = signals[name]
            
            if i == 0:
                # First indicator sets the base signal
                combined = signal
            else:
                # Subsequent indicators confirm or filter the signal
                # Only keep signal if both indicators agree
                combined = np.where((combined == signal) & (signal != 0), signal, 0)
        
        return combined
    
    def combine_signals_conditional(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine signals using conditional logic."""
        if not signals:
            return np.zeros(len(next(iter(signals.values()))) if signals else 0)
        
        signal_names = list(signals.keys())
        combined = np.zeros(len(signals[signal_names[0]]))
        
        # Primary indicator (first in list)
        primary_signal = signals[signal_names[0]]
        
        # Secondary indicators (rest of the list)
        secondary_signals = {name: signals[name] for name in signal_names[1:]}
        
        for i in range(len(combined)):
            if primary_signal[i] != 0:
                # Check if secondary indicators confirm
                confirmations = 0
                total_secondary = len(secondary_signals)
                
                for secondary_signal in secondary_signals.values():
                    if secondary_signal[i] == primary_signal[i]:
                        confirmations += 1
                
                # Only execute if enough confirmations
                if confirmations / total_secondary >= self.config.min_agreement:
                    combined[i] = primary_signal[i]
        
        return combined
    
    def generate_combined_signals(self, data: pd.DataFrame) -> CombinationResult:
        """Generate combined signals from all indicators."""
        # Generate individual signals
        individual_signals = self.generate_individual_signals(data)
        
        # Combine signals based on method
        if self.config.combination_method == CombinationMethod.VOTING:
            combined_signals = self.combine_signals_voting(individual_signals)
        elif self.config.combination_method == CombinationMethod.WEIGHTED_AVERAGE:
            combined_signals = self.combine_signals_weighted_average(individual_signals)
        elif self.config.combination_method == CombinationMethod.ENSEMBLE:
            combined_signals = self.combine_signals_ensemble(individual_signals)
        elif self.config.combination_method == CombinationMethod.SEQUENTIAL:
            combined_signals = self.combine_signals_sequential(individual_signals)
        elif self.config.combination_method == CombinationMethod.CONDITIONAL:
            combined_signals = self.combine_signals_conditional(individual_signals)
        else:
            combined_signals = self.combine_signals_voting(individual_signals)
        
        # Calculate agreement scores
        agreement_scores = self.calculate_agreement_scores(individual_signals)
        
        # Calculate confidence scores
        confidence_scores = self.calculate_confidence_scores(individual_signals, combined_signals)
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(combined_signals, data)
        
        result = CombinationResult(
            combined_signals=combined_signals,
            individual_signals=individual_signals,
            agreement_scores=agreement_scores,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics
        )
        
        return result
    
    def calculate_agreement_scores(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate agreement scores between indicators."""
        if not signals:
            return np.array([])
        
        signal_matrix = np.array(list(signals.values()))
        n_indicators = len(signals)
        
        # Calculate percentage of indicators that agree on direction
        agreement_scores = np.sum(signal_matrix != 0, axis=0) / n_indicators
        
        return agreement_scores
    
    def calculate_confidence_scores(self, individual_signals: Dict[str, np.ndarray], 
                                  combined_signals: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for combined signals."""
        if not individual_signals:
            return np.zeros(len(combined_signals))
        
        signal_matrix = np.array(list(individual_signals.values()))
        confidence_scores = np.zeros(len(combined_signals))
        
        for i in range(len(combined_signals)):
            if combined_signals[i] != 0:
                # Count how many indicators agree with the combined signal
                agreements = np.sum(signal_matrix[:, i] == combined_signals[i])
                total = len(individual_signals)
                confidence_scores[i] = agreements / total
        
        return confidence_scores
    
    def calculate_performance_metrics(self, signals: np.ndarray, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for the combined strategy."""
        if len(signals) == 0:
            return {}
        
        # Calculate returns
        price_returns = data['close'].pct_change().values
        price_returns[0] = 0
        
        strategy_returns = signals * price_returns
        
        # Calculate metrics
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        peak = np.maximum.accumulate(1 + cumulative_returns)
        drawdown = (1 + cumulative_returns) / peak - 1
        max_drawdown = np.min(drawdown)
        
        # Calculate win rate
        winning_trades = strategy_returns > 0
        win_rate = np.sum(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': np.std(strategy_returns),
            'avg_return': np.mean(strategy_returns)
        }

class MultiIndicatorOptimizer:
    """Optimizer for multi-indicator strategies."""
    
    def __init__(self):
        self.best_config = None
        self.optimization_results = []
    
    def optimize_weights(self, strategy: MultiIndicatorStrategy, data: pd.DataFrame,
                        metric: str = 'sharpe_ratio') -> List[float]:
        """Optimize indicator weights using grid search."""
        n_indicators = len(strategy.individual_strategies)
        
        # Generate weight combinations
        weight_step = 0.1
        weights_list = []
        
        for i in range(int(1/weight_step) + 1):
            for j in range(int(1/weight_step) + 1):
                if i + j <= int(1/weight_step):
                    w1 = i * weight_step
                    w2 = j * weight_step
                    w3 = 1 - w1 - w2
                    if w3 >= 0:
                        weights_list.append([w1, w2, w3])
        
        best_weights = None
        best_score = float('-inf')
        
        for weights in weights_list:
            if len(weights) == n_indicators:
                strategy.config.weights = weights
                result = strategy.generate_combined_signals(data)
                score = result.performance_metrics.get(metric, 0)
                
                if score > best_score:
                    best_score = score
                    best_weights = weights
        
        logger.info(f"Best weights: {best_weights}, Best {metric}: {best_score}")
        return best_weights
    
    def optimize_thresholds(self, strategy: MultiIndicatorStrategy, data: pd.DataFrame,
                           metric: str = 'sharpe_ratio') -> Dict[str, float]:
        """Optimize signal thresholds."""
        threshold_range = np.arange(0.1, 0.9, 0.1)
        
        best_thresholds = {}
        best_score = float('-inf')
        
        for buy_threshold in threshold_range:
            for sell_threshold in threshold_range:
                if buy_threshold > sell_threshold:
                    thresholds = {
                        'buy': buy_threshold,
                        'sell': -sell_threshold
                    }
                    
                    strategy.config.thresholds = thresholds
                    result = strategy.generate_combined_signals(data)
                    score = result.performance_metrics.get(metric, 0)
                    
                    if score > best_score:
                        best_score = score
                        best_thresholds = thresholds
        
        logger.info(f"Best thresholds: {best_thresholds}, Best {metric}: {best_score}")
        return best_thresholds

# Convenience functions
def create_multi_indicator_strategy(indicators: List[str], 
                                  combination_method: CombinationMethod = CombinationMethod.VOTING,
                                  voting_type: VotingType = VotingType.MAJORITY) -> MultiIndicatorStrategy:
    """Create a multi-indicator strategy."""
    config = MultiIndicatorConfig(
        indicators=indicators,
        combination_method=combination_method,
        voting_type=voting_type
    )
    return MultiIndicatorStrategy(config)

def create_voting_strategy(indicators: List[str], voting_type: VotingType = VotingType.MAJORITY) -> MultiIndicatorStrategy:
    """Create a voting-based multi-indicator strategy."""
    return create_multi_indicator_strategy(indicators, CombinationMethod.VOTING, voting_type)

def create_weighted_strategy(indicators: List[str], weights: Optional[List[float]] = None) -> MultiIndicatorStrategy:
    """Create a weighted average multi-indicator strategy."""
    config = MultiIndicatorConfig(
        indicators=indicators,
        combination_method=CombinationMethod.WEIGHTED_AVERAGE,
        weights=weights
    )
    return MultiIndicatorStrategy(config)

def create_optimizer() -> MultiIndicatorOptimizer:
    """Create a multi-indicator optimizer."""
    return MultiIndicatorOptimizer()