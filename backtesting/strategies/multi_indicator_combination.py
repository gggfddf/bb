#!/usr/bin/env python3
"""
Multi-Indicator Combination Testing Framework

Implements comprehensive multi-indicator combination testing for trading strategies:
- Multiple indicator combinations
- Weighted voting systems
- Consensus-based strategies
- Performance optimization
- Risk management integration
- Strategy validation and backtesting

Features:
- Advanced multi-indicator combination algorithms
- Weighted voting and consensus mechanisms
- Performance optimization and validation
- Risk management integration
- Comprehensive backtesting framework
- Strategy ranking and selection
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
import itertools
from scipy import stats
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = structlog.get_logger()

class CombinationMethod(Enum):
    """Combination method enumeration."""
    VOTING = "voting"
    WEIGHTED_VOTING = "weighted_voting"
    CONSENSUS = "consensus"
    ENSEMBLE = "ensemble"
    MACHINE_LEARNING = "machine_learning"
    STATISTICAL = "statistical"

class VotingType(Enum):
    """Voting type enumeration."""
    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    THRESHOLD = "threshold"
    WEIGHTED = "weighted"

@dataclass
class IndicatorSignal:
    """Indicator signal structure."""
    indicator_name: str
    signal: int  # -1, 0, 1
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CombinationConfig:
    """Combination configuration."""
    method: CombinationMethod = CombinationMethod.WEIGHTED_VOTING
    voting_type: VotingType = VotingType.WEIGHTED
    threshold: float = 0.6
    min_indicators: int = 2
    max_indicators: int = 10
    enable_optimization: bool = True
    optimization_metric: str = "sharpe_ratio"
    risk_management: bool = True
    stop_loss: float = 0.02
    take_profit: float = 0.04

@dataclass
class CombinationResult:
    """Combination result structure."""
    combination_id: str
    indicators: List[str]
    method: CombinationMethod
    final_signal: int
    confidence: float
    agreement_ratio: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class IndicatorCombiner:
    """Multi-indicator combination system."""
    
    def __init__(self, config: CombinationConfig):
        """
        Initialize indicator combiner.
        
        Args:
            config: Combination configuration
        """
        self.config = config
        self.indicator_signals: Dict[str, List[IndicatorSignal]] = defaultdict(list)
        self.combination_history: List[CombinationResult] = []
        
    def add_indicator_signal(self, indicator_name: str, signal: int, confidence: float,
                           timestamp: datetime = None, metadata: Dict[str, Any] = None):
        """Add indicator signal."""
        if timestamp is None:
            timestamp = datetime.now()
            
        indicator_signal = IndicatorSignal(
            indicator_name=indicator_name,
            signal=signal,
            confidence=confidence,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        self.indicator_signals[indicator_name].append(indicator_signal)
        logger.debug(f"Added signal for {indicator_name}: {signal} (confidence: {confidence})")
        
    def combine_signals(self, indicator_names: List[str] = None) -> CombinationResult:
        """
        Combine signals from multiple indicators.
        
        Args:
            indicator_names: List of indicator names to combine (None for all)
            
        Returns:
            Combination result
        """
        if indicator_names is None:
            indicator_names = list(self.indicator_signals.keys())
            
        if len(indicator_names) < self.config.min_indicators:
            raise ValueError(f"Need at least {self.config.min_indicators} indicators")
            
        if len(indicator_names) > self.config.max_indicators:
            indicator_names = indicator_names[:self.config.max_indicators]
            
        # Get latest signals for each indicator
        latest_signals = []
        for indicator_name in indicator_names:
            if indicator_name in self.indicator_signals and self.indicator_signals[indicator_name]:
                latest_signals.append(self.indicator_signals[indicator_name][-1])
                
        if len(latest_signals) < self.config.min_indicators:
            raise ValueError(f"Not enough valid signals: {len(latest_signals)}")
            
        # Combine signals based on method
        if self.config.method == CombinationMethod.VOTING:
            final_signal, confidence, agreement = self._voting_combination(latest_signals)
        elif self.config.method == CombinationMethod.WEIGHTED_VOTING:
            final_signal, confidence, agreement = self._weighted_voting_combination(latest_signals)
        elif self.config.method == CombinationMethod.CONSENSUS:
            final_signal, confidence, agreement = self._consensus_combination(latest_signals)
        elif self.config.method == CombinationMethod.ENSEMBLE:
            final_signal, confidence, agreement = self._ensemble_combination(latest_signals)
        elif self.config.method == CombinationMethod.STATISTICAL:
            final_signal, confidence, agreement = self._statistical_combination(latest_signals)
        else:
            raise ValueError(f"Unsupported combination method: {self.config.method}")
            
        # Create combination result
        result = CombinationResult(
            combination_id=str(uuid.uuid4()),
            indicators=indicator_names,
            method=self.config.method,
            final_signal=final_signal,
            confidence=confidence,
            agreement_ratio=agreement,
            timestamp=datetime.now()
        )
        
        self.combination_history.append(result)
        return result
        
    def _voting_combination(self, signals: List[IndicatorSignal]) -> Tuple[int, float, float]:
        """Simple voting combination."""
        votes = [signal.signal for signal in signals]
        confidences = [signal.confidence for signal in signals]
        
        if self.config.voting_type == VotingType.MAJORITY:
            final_signal = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
        elif self.config.voting_type == VotingType.UNANIMOUS:
            if all(vote == 1 for vote in votes):
                final_signal = 1
            elif all(vote == -1 for vote in votes):
                final_signal = -1
            else:
                final_signal = 0
        elif self.config.voting_type == VotingType.THRESHOLD:
            positive_votes = sum(1 for vote in votes if vote == 1)
            negative_votes = sum(1 for vote in votes if vote == -1)
            total_votes = len(votes)
            
            if positive_votes / total_votes >= self.config.threshold:
                final_signal = 1
            elif negative_votes / total_votes >= self.config.threshold:
                final_signal = -1
            else:
                final_signal = 0
        else:
            final_signal = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
            
        confidence = np.mean(confidences)
        agreement = sum(1 for vote in votes if vote == final_signal) / len(votes)
        
        return final_signal, confidence, agreement
        
    def _weighted_voting_combination(self, signals: List[IndicatorSignal]) -> Tuple[int, float, float]:
        """Weighted voting combination."""
        votes = [signal.signal for signal in signals]
        confidences = [signal.confidence for signal in signals]
        
        # Calculate weighted votes
        weighted_votes = [vote * confidence for vote, confidence in zip(votes, confidences)]
        total_weight = sum(confidences)
        
        if total_weight == 0:
            return 0, 0.0, 0.0
            
        final_signal = 1 if sum(weighted_votes) > 0 else (-1 if sum(weighted_votes) < 0 else 0)
        confidence = np.mean(confidences)
        agreement = sum(1 for vote in votes if vote == final_signal) / len(votes)
        
        return final_signal, confidence, agreement
        
    def _consensus_combination(self, signals: List[IndicatorSignal]) -> Tuple[int, float, float]:
        """Consensus-based combination."""
        votes = [signal.signal for signal in signals]
        confidences = [signal.confidence for signal in signals]
        
        # Check for consensus
        unique_votes = set(votes)
        if len(unique_votes) == 1:
            final_signal = list(unique_votes)[0]
            confidence = np.mean(confidences)
            agreement = 1.0
        else:
            # No consensus, use weighted average
            weighted_votes = [vote * confidence for vote, confidence in zip(votes, confidences)]
            final_signal = 1 if sum(weighted_votes) > 0 else (-1 if sum(weighted_votes) < 0 else 0)
            confidence = np.mean(confidences)
            agreement = sum(1 for vote in votes if vote == final_signal) / len(votes)
            
        return final_signal, confidence, agreement
        
    def _ensemble_combination(self, signals: List[IndicatorSignal]) -> Tuple[int, float, float]:
        """Ensemble-based combination."""
        votes = [signal.signal for signal in signals]
        confidences = [signal.confidence for signal in signals]
        
        # Use ensemble methods (simplified)
        # In practice, you might use sklearn ensemble methods
        weighted_votes = [vote * confidence for vote, confidence in zip(votes, confidences)]
        
        # Calculate ensemble signal
        if sum(weighted_votes) > 0.5:
            final_signal = 1
        elif sum(weighted_votes) < -0.5:
            final_signal = -1
        else:
            final_signal = 0
            
        confidence = np.mean(confidences)
        agreement = sum(1 for vote in votes if vote == final_signal) / len(votes)
        
        return final_signal, confidence, agreement
        
    def _statistical_combination(self, signals: List[IndicatorSignal]) -> Tuple[int, float, float]:
        """Statistical combination."""
        votes = [signal.signal for signal in signals]
        confidences = [signal.confidence for signal in signals]
        
        # Use statistical methods (mean, median, mode)
        mean_vote = np.mean(votes)
        median_vote = np.median(votes)
        
        # Calculate statistical measures
        vote_std = np.std(votes)
        confidence_std = np.std(confidences)
        
        # Determine final signal based on statistical measures
        if abs(mean_vote) > 0.5 and vote_std < 1.0:
            final_signal = 1 if mean_vote > 0 else -1
        else:
            final_signal = 0
            
        confidence = np.mean(confidences) * (1 - confidence_std)  # Penalize high variance
        agreement = sum(1 for vote in votes if vote == final_signal) / len(votes)
        
        return final_signal, confidence, agreement

class MultiIndicatorStrategy:
    """Multi-indicator strategy implementation."""
    
    def __init__(self, config: CombinationConfig):
        """
        Initialize multi-indicator strategy.
        
        Args:
            config: Combination configuration
        """
        self.config = config
        self.combiner = IndicatorCombiner(config)
        self.performance_history: List[Dict[str, Any]] = []
        self.risk_manager = None  # Would integrate with risk management system
        
    def add_indicator_signal(self, indicator_name: str, signal: int, confidence: float,
                           timestamp: datetime = None, metadata: Dict[str, Any] = None):
        """Add indicator signal to strategy."""
        self.combiner.add_indicator_signal(indicator_name, signal, confidence, timestamp, metadata)
        
    def generate_signal(self, indicator_names: List[str] = None) -> CombinationResult:
        """Generate trading signal from multiple indicators."""
        return self.combiner.combine_signals(indicator_names)
        
    def backtest_strategy(self, data: pd.DataFrame, indicator_signals: Dict[str, List[IndicatorSignal]]) -> Dict[str, Any]:
        """
        Backtest multi-indicator strategy.
        
        Args:
            data: Market data
            indicator_signals: Dictionary of indicator signals by timestamp
            
        Returns:
            Backtest results
        """
        results = {
            'signals': [],
            'returns': [],
            'positions': [],
            'performance_metrics': {}
        }
        
        for timestamp in data.index:
            # Get signals for this timestamp
            if timestamp in indicator_signals:
                for indicator_name, signal in indicator_signals[timestamp].items():
                    self.combiner.add_indicator_signal(
                        indicator_name, signal.signal, signal.confidence, timestamp
                    )
                    
            # Generate combined signal
            try:
                combined_signal = self.generate_signal()
                results['signals'].append(combined_signal)
                
                # Calculate position and returns
                position = self._calculate_position(combined_signal, data.loc[timestamp])
                returns = self._calculate_returns(position, data.loc[timestamp])
                
                results['positions'].append(position)
                results['returns'].append(returns)
                
            except Exception as e:
                logger.warning(f"Error generating signal at {timestamp}: {e}")
                results['signals'].append(None)
                results['positions'].append(0)
                results['returns'].append(0)
                
        # Calculate performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(results['returns'])
        
        return results
        
    def _calculate_position(self, signal: CombinationResult, data: pd.Series) -> float:
        """Calculate position based on signal."""
        if signal.final_signal == 1:
            return 1.0  # Long position
        elif signal.final_signal == -1:
            return -1.0  # Short position
        else:
            return 0.0  # No position
            
    def _calculate_returns(self, position: float, data: pd.Series) -> float:
        """Calculate returns based on position."""
        # Simplified return calculation
        # In practice, you'd use actual price data
        if 'returns' in data:
            return position * data['returns']
        else:
            return 0.0
            
    def _calculate_performance_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate performance metrics."""
        returns_array = np.array(returns)
        
        if len(returns_array) == 0:
            return {}
            
        metrics = {
            'total_return': np.sum(returns_array),
            'mean_return': np.mean(returns_array),
            'volatility': np.std(returns_array),
            'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns_array),
            'win_rate': np.sum(returns_array > 0) / len(returns_array),
            'profit_factor': self._calculate_profit_factor(returns_array)
        }
        
        return metrics
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
        
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        elif len(positive_returns) == 0:
            return 0.0
        else:
            return np.sum(positive_returns) / abs(np.sum(negative_returns))

class MultiIndicatorTester:
    """Multi-indicator combination tester."""
    
    def __init__(self, config: CombinationConfig = None):
        """
        Initialize multi-indicator tester.
        
        Args:
            config: Combination configuration
        """
        self.config = config or CombinationConfig()
        self.strategies: Dict[str, MultiIndicatorStrategy] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
        
    def create_strategy(self, strategy_name: str, indicators: List[str]) -> MultiIndicatorStrategy:
        """Create a multi-indicator strategy."""
        strategy = MultiIndicatorStrategy(self.config)
        self.strategies[strategy_name] = strategy
        logger.info(f"Created strategy: {strategy_name} with indicators: {indicators}")
        return strategy
        
    def test_combination(self, strategy_name: str, data: pd.DataFrame, 
                        indicator_signals: Dict[str, List[IndicatorSignal]]) -> Dict[str, Any]:
        """Test a specific indicator combination."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
            
        strategy = self.strategies[strategy_name]
        results = strategy.backtest_strategy(data, indicator_signals)
        
        self.test_results[strategy_name] = results
        logger.info(f"Tested combination: {strategy_name}")
        
        return results
        
    def test_all_combinations(self, data: pd.DataFrame, indicator_signals: Dict[str, List[IndicatorSignal]],
                             max_indicators: int = 5) -> Dict[str, Any]:
        """Test all possible indicator combinations."""
        all_results = {}
        
        # Generate all combinations
        indicator_names = list(indicator_signals.keys())
        
        for n in range(2, min(max_indicators + 1, len(indicator_names) + 1)):
            for combination in itertools.combinations(indicator_names, n):
                strategy_name = f"combination_{'_'.join(combination)}"
                strategy = self.create_strategy(strategy_name, list(combination))
                
                try:
                    results = self.test_combination(strategy_name, data, indicator_signals)
                    all_results[strategy_name] = results
                except Exception as e:
                    logger.error(f"Error testing combination {strategy_name}: {e}")
                    
        return all_results
        
    def rank_strategies(self, metric: str = "sharpe_ratio") -> pd.DataFrame:
        """Rank strategies by performance metric."""
        rankings = []
        
        for strategy_name, results in self.test_results.items():
            if 'performance_metrics' in results and metric in results['performance_metrics']:
                rankings.append({
                    'strategy': strategy_name,
                    'metric': results['performance_metrics'][metric],
                    'indicators': len(results.get('signals', [])),
                    'total_return': results['performance_metrics'].get('total_return', 0),
                    'volatility': results['performance_metrics'].get('volatility', 0)
                })
                
        if rankings:
            df = pd.DataFrame(rankings)
            return df.sort_values('metric', ascending=False)
        else:
            return pd.DataFrame()

def create_multi_indicator_tester(config: CombinationConfig = None) -> MultiIndicatorTester:
    """Create a multi-indicator tester."""
    return MultiIndicatorTester(config)

# Demo of multi-indicator combination testing
if __name__ == "__main__":
    # Create tester
    config = CombinationConfig(
        method=CombinationMethod.WEIGHTED_VOTING,
        voting_type=VotingType.WEIGHTED,
        threshold=0.6,
        min_indicators=2,
        max_indicators=5,
        enable_optimization=True,
        optimization_metric="sharpe_ratio"
    )
    
    tester = create_multi_indicator_tester(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Create sample indicator signals
    indicator_signals = {}
    for date in dates:
        indicator_signals[date] = {
            'RSI': IndicatorSignal('RSI', np.random.choice([-1, 0, 1]), np.random.uniform(0.5, 0.9), date),
            'MACD': IndicatorSignal('MACD', np.random.choice([-1, 0, 1]), np.random.uniform(0.5, 0.9), date),
            'BB': IndicatorSignal('BB', np.random.choice([-1, 0, 1]), np.random.uniform(0.5, 0.9), date),
            'SMA': IndicatorSignal('SMA', np.random.choice([-1, 0, 1]), np.random.uniform(0.5, 0.9), date),
            'EMA': IndicatorSignal('EMA', np.random.choice([-1, 0, 1]), np.random.uniform(0.5, 0.9), date)
        }
    
    # Test combinations
    results = tester.test_all_combinations(data, indicator_signals, max_indicators=3)
    
    # Rank strategies
    rankings = tester.rank_strategies("sharpe_ratio")
    print("Strategy Rankings:")
    print(rankings.head())
    
    print("Multi-indicator combination testing completed successfully!")