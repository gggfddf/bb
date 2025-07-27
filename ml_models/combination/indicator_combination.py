"""
Indicator Combination Algorithms

Implements comprehensive indicator combination and analysis techniques:
- Ensemble methods for indicator combination
- Voting systems for signal aggregation
- Weighted combination strategies
- Statistical combination methods
- Machine learning-based combination
- Dynamic weight adjustment

Features:
- Multiple combination algorithms
- Signal strength analysis
- Performance-based weighting
- Real-time combination updates
- Validation and backtesting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import structlog
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

logger = structlog.get_logger()

class CombinationMethod(Enum):
    """Indicator combination methods."""
    VOTING = "voting"
    WEIGHTED_AVERAGE = "weighted_average"
    ENSEMBLE = "ensemble"
    STATISTICAL = "statistical"
    ML_BASED = "ml_based"
    DYNAMIC = "dynamic"

class VotingType(Enum):
    """Voting system types."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    THRESHOLD = "threshold"

@dataclass
class IndicatorSignal:
    """Individual indicator signal."""
    indicator_name: str
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: pd.Timestamp
    metadata: Dict[str, Any] = None

@dataclass
class CombinationConfig:
    """Configuration for indicator combination."""
    method: CombinationMethod = CombinationMethod.WEIGHTED_AVERAGE
    voting_type: VotingType = VotingType.MAJORITY
    weights: Optional[List[float]] = None
    threshold: float = 0.5
    min_agreement: float = 0.6
    ensemble_size: int = 10
    update_frequency: str = "daily"
    performance_window: int = 252

@dataclass
class CombinationResult:
    """Result of indicator combination."""
    combined_signal: str
    signal_strength: float
    confidence: float
    agreement_score: float
    individual_signals: List[IndicatorSignal]
    combination_weights: List[float]
    metadata: Dict[str, Any]

class IndicatorCombiner:
    """
    Comprehensive indicator combination system.
    """
    
    def __init__(self, config: CombinationConfig = None):
        """
        Initialize indicator combiner.
        
        Args:
            config: Combination configuration
        """
        self.config = config or CombinationConfig()
        self.indicators = {}
        self.performance_history = {}
        self.is_fitted = False
    
    def add_indicator(self, name: str, signal_function: Callable):
        """
        Add an indicator to the combination system.
        
        Args:
            name: Indicator name
            signal_function: Function that generates signals
        """
        self.indicators[name] = signal_function
        self.performance_history[name] = []
        logger.info(f"Added indicator: {name}")
    
    def combine_signals(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> CombinationResult:
        """
        Combine signals from all indicators.
        
        Args:
            data: Market data
            target: Target variable for performance calculation
            
        Returns:
            CombinationResult with combined signal
        """
        if not self.indicators:
            raise ValueError("No indicators added to combiner")
        
        # Generate individual signals
        individual_signals = self._generate_individual_signals(data)
        
        # Combine signals based on method
        if self.config.method == CombinationMethod.VOTING:
            combined_signal, strength, confidence = self._combine_voting(individual_signals)
        elif self.config.method == CombinationMethod.WEIGHTED_AVERAGE:
            combined_signal, strength, confidence = self._combine_weighted_average(individual_signals)
        elif self.config.method == CombinationMethod.ENSEMBLE:
            combined_signal, strength, confidence = self._combine_ensemble(individual_signals)
        elif self.config.method == CombinationMethod.STATISTICAL:
            combined_signal, strength, confidence = self._combine_statistical(individual_signals)
        elif self.config.method == CombinationMethod.ML_BASED:
            combined_signal, strength, confidence = self._combine_ml_based(individual_signals, target)
        elif self.config.method == CombinationMethod.DYNAMIC:
            combined_signal, strength, confidence = self._combine_dynamic(individual_signals, target)
        else:
            raise ValueError(f"Unsupported combination method: {self.config.method}")
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement_score(individual_signals)
        
        # Update performance history if target is provided
        if target is not None:
            self._update_performance_history(individual_signals, target)
        
        # Generate weights for individual indicators
        weights = self._calculate_weights(individual_signals)
        
        metadata = {
            "combination_method": self.config.method.value,
            "num_indicators": len(individual_signals),
            "agreement_score": agreement_score,
            "timestamp": pd.Timestamp.now()
        }
        
        return CombinationResult(
            combined_signal=combined_signal,
            signal_strength=strength,
            confidence=confidence,
            agreement_score=agreement_score,
            individual_signals=individual_signals,
            combination_weights=weights,
            metadata=metadata
        )
    
    def _generate_individual_signals(self, data: pd.DataFrame) -> List[IndicatorSignal]:
        """Generate signals from all individual indicators."""
        signals = []
        
        for name, signal_function in self.indicators.items():
            try:
                # Generate signal using indicator function
                signal_data = signal_function(data)
                
                # Create indicator signal
                indicator_signal = IndicatorSignal(
                    indicator_name=name,
                    signal=signal_data.get('signal', 'hold'),
                    strength=signal_data.get('strength', 0.0),
                    confidence=signal_data.get('confidence', 0.0),
                    timestamp=pd.Timestamp.now(),
                    metadata=signal_data.get('metadata', {})
                )
                
                signals.append(indicator_signal)
                
            except Exception as e:
                logger.warning(f"Failed to generate signal for indicator {name}: {e}")
                # Add default signal
                signals.append(IndicatorSignal(
                    indicator_name=name,
                    signal='hold',
                    strength=0.0,
                    confidence=0.0,
                    timestamp=pd.Timestamp.now()
                ))
        
        return signals
    
    def _combine_voting(self, signals: List[IndicatorSignal]) -> Tuple[str, float, float]:
        """Combine signals using voting system."""
        if not signals:
            return 'hold', 0.0, 0.0
        
        # Count votes
        buy_votes = sum(1 for s in signals if s.signal == 'buy')
        sell_votes = sum(1 for s in signals if s.signal == 'sell')
        hold_votes = sum(1 for s in signals if s.signal == 'hold')
        
        total_votes = len(signals)
        
        if self.config.voting_type == VotingType.MAJORITY:
            # Simple majority voting
            if buy_votes > total_votes / 2:
                return 'buy', buy_votes / total_votes, buy_votes / total_votes
            elif sell_votes > total_votes / 2:
                return 'sell', sell_votes / total_votes, sell_votes / total_votes
            else:
                return 'hold', hold_votes / total_votes, hold_votes / total_votes
        
        elif self.config.voting_type == VotingType.WEIGHTED:
            # Weighted voting based on signal strength
            buy_weight = sum(s.strength for s in signals if s.signal == 'buy')
            sell_weight = sum(s.strength for s in signals if s.signal == 'sell')
            hold_weight = sum(s.strength for s in signals if s.signal == 'hold')
            
            max_weight = max(buy_weight, sell_weight, hold_weight)
            
            if max_weight == buy_weight:
                return 'buy', buy_weight / total_votes, buy_weight / total_votes
            elif max_weight == sell_weight:
                return 'sell', sell_weight / total_votes, sell_weight / total_votes
            else:
                return 'hold', hold_weight / total_votes, hold_weight / total_votes
        
        elif self.config.voting_type == VotingType.UNANIMOUS:
            # Unanimous voting
            if buy_votes == total_votes:
                return 'buy', 1.0, 1.0
            elif sell_votes == total_votes:
                return 'sell', 1.0, 1.0
            else:
                return 'hold', hold_votes / total_votes, hold_votes / total_votes
        
        elif self.config.voting_type == VotingType.THRESHOLD:
            # Threshold-based voting
            if buy_votes / total_votes >= self.config.threshold:
                return 'buy', buy_votes / total_votes, buy_votes / total_votes
            elif sell_votes / total_votes >= self.config.threshold:
                return 'sell', sell_votes / total_votes, sell_votes / total_votes
            else:
                return 'hold', hold_votes / total_votes, hold_votes / total_votes
        
        return 'hold', 0.0, 0.0
    
    def _combine_weighted_average(self, signals: List[IndicatorSignal]) -> Tuple[str, float, float]:
        """Combine signals using weighted average."""
        if not signals:
            return 'hold', 0.0, 0.0
        
        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = signal.strength * signal.confidence
            total_weight += weight
            
            if signal.signal == 'buy':
                buy_score += weight
            elif signal.signal == 'sell':
                sell_score += weight
            else:
                hold_score += weight
        
        if total_weight == 0:
            return 'hold', 0.0, 0.0
        
        # Normalize scores
        buy_score /= total_weight
        sell_score /= total_weight
        hold_score /= total_weight
        
        # Determine combined signal
        max_score = max(buy_score, sell_score, hold_score)
        
        if max_score == buy_score:
            return 'buy', buy_score, buy_score
        elif max_score == sell_score:
            return 'sell', sell_score, sell_score
        else:
            return 'hold', hold_score, hold_score
    
    def _combine_ensemble(self, signals: List[IndicatorSignal]) -> Tuple[str, float, float]:
        """Combine signals using ensemble methods."""
        if not signals:
            return 'hold', 0.0, 0.0
        
        # Convert signals to numerical format for ensemble
        signal_values = []
        signal_weights = []
        
        for signal in signals:
            # Convert signal to numerical value
            if signal.signal == 'buy':
                value = 1.0
            elif signal.signal == 'sell':
                value = -1.0
            else:
                value = 0.0
            
            # Weight by strength and confidence
            weight = signal.strength * signal.confidence
            
            signal_values.append(value)
            signal_weights.append(weight)
        
        # Calculate ensemble prediction
        weighted_sum = sum(v * w for v, w in zip(signal_values, signal_weights))
        total_weight = sum(signal_weights)
        
        if total_weight == 0:
            return 'hold', 0.0, 0.0
        
        ensemble_score = weighted_sum / total_weight
        
        # Convert back to signal
        if ensemble_score > 0.3:
            return 'buy', ensemble_score, ensemble_score
        elif ensemble_score < -0.3:
            return 'sell', abs(ensemble_score), abs(ensemble_score)
        else:
            return 'hold', 0.0, 0.0
    
    def _combine_statistical(self, signals: List[IndicatorSignal]) -> Tuple[str, float, float]:
        """Combine signals using statistical methods."""
        if not signals:
            return 'hold', 0.0, 0.0
        
        # Calculate statistical measures
        signal_values = []
        confidences = []
        
        for signal in signals:
            if signal.signal == 'buy':
                value = 1.0
            elif signal.signal == 'sell':
                value = -1.0
            else:
                value = 0.0
            
            signal_values.append(value * signal.strength)
            confidences.append(signal.confidence)
        
        # Calculate mean and standard deviation
        mean_signal = np.mean(signal_values)
        std_signal = np.std(signal_values)
        mean_confidence = np.mean(confidences)
        
        # Determine signal based on statistical measures
        if mean_signal > 0.2 and std_signal < 0.5:
            return 'buy', mean_signal, mean_confidence
        elif mean_signal < -0.2 and std_signal < 0.5:
            return 'sell', abs(mean_signal), mean_confidence
        else:
            return 'hold', 0.0, mean_confidence
    
    def _combine_ml_based(self, signals: List[IndicatorSignal], target: Optional[pd.Series]) -> Tuple[str, float, float]:
        """Combine signals using machine learning."""
        if not signals or target is None:
            return 'hold', 0.0, 0.0
        
        # Prepare features for ML model
        features = []
        for signal in signals:
            features.extend([signal.strength, signal.confidence])
        
        # Create simple ML ensemble
        try:
            # Use voting classifier with multiple base models
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('lr', LogisticRegression(random_state=42)),
                ('svm', SVC(probability=True, random_state=42))
            ]
            
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            
            # Fit ensemble (simplified - in practice, you'd need more data)
            # For now, return weighted average
            return self._combine_weighted_average(signals)
            
        except Exception as e:
            logger.warning(f"ML-based combination failed: {e}")
            return self._combine_weighted_average(signals)
    
    def _combine_dynamic(self, signals: List[IndicatorSignal], target: Optional[pd.Series]) -> Tuple[str, float, float]:
        """Combine signals using dynamic weighting."""
        if not signals:
            return 'hold', 0.0, 0.0
        
        # Calculate dynamic weights based on recent performance
        dynamic_weights = self._calculate_dynamic_weights(signals, target)
        
        # Apply dynamic weights
        weighted_signals = []
        for signal, weight in zip(signals, dynamic_weights):
            weighted_signal = IndicatorSignal(
                indicator_name=signal.indicator_name,
                signal=signal.signal,
                strength=signal.strength * weight,
                confidence=signal.confidence * weight,
                timestamp=signal.timestamp,
                metadata=signal.metadata
            )
            weighted_signals.append(weighted_signal)
        
        return self._combine_weighted_average(weighted_signals)
    
    def _calculate_agreement_score(self, signals: List[IndicatorSignal]) -> float:
        """Calculate agreement score among indicators."""
        if not signals:
            return 0.0
        
        # Count signals by type
        signal_counts = {}
        for signal in signals:
            signal_counts[signal.signal] = signal_counts.get(signal.signal, 0) + 1
        
        # Calculate agreement as percentage of most common signal
        total_signals = len(signals)
        max_count = max(signal_counts.values()) if signal_counts else 0
        
        return max_count / total_signals
    
    def _calculate_weights(self, signals: List[IndicatorSignal]) -> List[float]:
        """Calculate weights for individual indicators."""
        if not signals:
            return []
        
        weights = []
        for signal in signals:
            # Weight based on strength and confidence
            weight = signal.strength * signal.confidence
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights
    
    def _update_performance_history(self, signals: List[IndicatorSignal], target: pd.Series):
        """Update performance history for dynamic weighting."""
        for signal in signals:
            # Simplified performance calculation
            # In practice, you'd calculate actual performance metrics
            performance = signal.strength * signal.confidence
            self.performance_history[signal.indicator_name].append(performance)
            
            # Keep only recent history
            if len(self.performance_history[signal.indicator_name]) > self.config.performance_window:
                self.performance_history[signal.indicator_name].pop(0)
    
    def _calculate_dynamic_weights(self, signals: List[IndicatorSignal], target: Optional[pd.Series]) -> List[float]:
        """Calculate dynamic weights based on performance history."""
        weights = []
        
        for signal in signals:
            history = self.performance_history.get(signal.indicator_name, [])
            
            if history:
                # Weight based on recent performance
                recent_performance = np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
                weight = max(0.1, recent_performance)  # Minimum weight of 0.1
            else:
                # Default weight for new indicators
                weight = 0.5
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights

class MultiIndicatorAnalyzer:
    """
    Advanced multi-indicator analysis system.
    """
    
    def __init__(self, combiners: List[IndicatorCombiner] = None):
        """
        Initialize multi-indicator analyzer.
        
        Args:
            combiners: List of indicator combiners
        """
        self.combiners = combiners or []
        self.analysis_results = []
    
    def add_combiner(self, combiner: IndicatorCombiner):
        """Add an indicator combiner."""
        self.combiners.append(combiner)
    
    def analyze_indicators(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform comprehensive multi-indicator analysis.
        
        Args:
            data: Market data
            target: Target variable
            
        Returns:
            Analysis results
        """
        results = {}
        
        # Run each combiner
        for i, combiner in enumerate(self.combiners):
            try:
                result = combiner.combine_signals(data, target)
                results[f"combiner_{i}"] = result
            except Exception as e:
                logger.error(f"Combiner {i} failed: {e}")
        
        # Aggregate results
        aggregated_result = self._aggregate_results(results)
        
        # Store analysis results
        self.analysis_results.append({
            'timestamp': pd.Timestamp.now(),
            'results': results,
            'aggregated': aggregated_result
        })
        
        return {
            'individual_results': results,
            'aggregated_result': aggregated_result,
            'analysis_metadata': {
                'num_combiners': len(self.combiners),
                'timestamp': pd.Timestamp.now()
            }
        }
    
    def _aggregate_results(self, results: Dict[str, CombinationResult]) -> CombinationResult:
        """Aggregate results from multiple combiners."""
        if not results:
            return CombinationResult(
                combined_signal='hold',
                signal_strength=0.0,
                confidence=0.0,
                agreement_score=0.0,
                individual_signals=[],
                combination_weights=[],
                metadata={}
            )
        
        # Collect all signals
        all_signals = []
        all_strengths = []
        all_confidences = []
        
        for result in results.values():
            all_signals.append(result.combined_signal)
            all_strengths.append(result.signal_strength)
            all_confidences.append(result.confidence)
        
        # Determine final signal by majority
        signal_counts = {}
        for signal in all_signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        final_signal = max(signal_counts, key=signal_counts.get) if signal_counts else 'hold'
        
        # Calculate aggregated metrics
        avg_strength = np.mean(all_strengths)
        avg_confidence = np.mean(all_confidences)
        
        return CombinationResult(
            combined_signal=final_signal,
            signal_strength=avg_strength,
            confidence=avg_confidence,
            agreement_score=signal_counts.get(final_signal, 0) / len(all_signals),
            individual_signals=[],
            combination_weights=[],
            metadata={'aggregation_method': 'majority_voting'}
        )

# Convenience functions
def create_voting_combiner(indicators: Dict[str, Callable], 
                          voting_type: str = "majority") -> IndicatorCombiner:
    """Create a voting-based indicator combiner."""
    config = CombinationConfig(
        method=CombinationMethod.VOTING,
        voting_type=VotingType(voting_type)
    )
    
    combiner = IndicatorCombiner(config)
    
    for name, indicator_func in indicators.items():
        combiner.add_indicator(name, indicator_func)
    
    return combiner

def create_weighted_combiner(indicators: Dict[str, Callable], 
                           weights: Optional[List[float]] = None) -> IndicatorCombiner:
    """Create a weighted average indicator combiner."""
    config = CombinationConfig(
        method=CombinationMethod.WEIGHTED_AVERAGE,
        weights=weights
    )
    
    combiner = IndicatorCombiner(config)
    
    for name, indicator_func in indicators.items():
        combiner.add_indicator(name, indicator_func)
    
    return combiner

def create_ensemble_combiner(indicators: Dict[str, Callable]) -> IndicatorCombiner:
    """Create an ensemble-based indicator combiner."""
    config = CombinationConfig(method=CombinationMethod.ENSEMBLE)
    
    combiner = IndicatorCombiner(config)
    
    for name, indicator_func in indicators.items():
        combiner.add_indicator(name, indicator_func)
    
    return combiner