#!/usr/bin/env python3
"""
Cross-Timeframe Analysis Module

Implements comprehensive cross-timeframe analysis for trading strategies:
- Multi-timeframe signal generation
- Timeframe consistency analysis
- Timeframe optimization
- Performance comparison across timeframes
- Robustness testing

Features:
- Support for multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d, 1w)
- Signal alignment and confirmation across timeframes
- Timeframe-specific performance metrics
- Optimization of timeframe combinations
- Consistency and reliability analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

logger = structlog.get_logger()

class TimeframeType(Enum):
    """Supported timeframes."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe."""
    timeframe: TimeframeType
    timestamp: datetime
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossTimeframeResult:
    """Result of cross-timeframe analysis."""
    primary_timeframe: TimeframeType
    supporting_timeframes: List[TimeframeType]
    combined_signal: str
    signal_strength: float
    consistency_score: float
    timeframe_alignment: Dict[TimeframeType, str]
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TimeframeOptimizationResult:
    """Result of timeframe optimization."""
    best_combination: List[TimeframeType]
    performance_score: float
    consistency_score: float
    robustness_score: float
    optimization_metrics: Dict[str, float]
    recommendations: List[str]

class CrossTimeframeAnalyzer:
    """
    Main class for cross-timeframe analysis.
    """
    
    def __init__(self, 
                 primary_timeframe: TimeframeType = TimeframeType.DAY_1,
                 supporting_timeframes: List[TimeframeType] = None):
        """
        Initialize cross-timeframe analyzer.
        
        Args:
            primary_timeframe: Main timeframe for analysis
            supporting_timeframes: Supporting timeframes for confirmation
        """
        self.primary_timeframe = primary_timeframe
        self.supporting_timeframes = supporting_timeframes or [
            TimeframeType.HOUR_4,
            TimeframeType.HOUR_1,
            TimeframeType.MINUTE_15
        ]
        
        # Analysis parameters
        self.min_consistency_threshold = 0.7
        self.signal_strength_threshold = 0.6
        self.confidence_threshold = 0.5
        
        # Performance tracking
        self.analysis_history: List[CrossTimeframeResult] = []
        self.optimization_results: List[TimeframeOptimizationResult] = []
    
    def analyze_timeframe_signals(self, 
                                data_dict: Dict[TimeframeType, pd.DataFrame],
                                strategy_function: callable,
                                strategy_params: Dict[str, Any] = None) -> CrossTimeframeResult:
        """
        Analyze signals across multiple timeframes.
        
        Args:
            data_dict: Dictionary of data for each timeframe
            strategy_function: Function to generate signals
            strategy_params: Parameters for strategy function
            
        Returns:
            CrossTimeframeResult with combined analysis
        """
        try:
            logger.info("Starting cross-timeframe analysis", 
                       primary_timeframe=self.primary_timeframe.value,
                       supporting_timeframes=[tf.value for tf in self.supporting_timeframes])
            
            # Generate signals for each timeframe
            timeframe_signals = {}
            for timeframe in [self.primary_timeframe] + self.supporting_timeframes:
                if timeframe in data_dict:
                    signals = self._generate_timeframe_signals(
                        data_dict[timeframe], strategy_function, strategy_params
                    )
                    timeframe_signals[timeframe] = signals
            
            # Analyze signal alignment
            alignment_analysis = self._analyze_signal_alignment(timeframe_signals)
            
            # Generate combined signal
            combined_signal = self._generate_combined_signal(timeframe_signals, alignment_analysis)
            
            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(timeframe_signals, alignment_analysis)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(timeframe_signals, data_dict)
            
            # Create result
            result = CrossTimeframeResult(
                primary_timeframe=self.primary_timeframe,
                supporting_timeframes=self.supporting_timeframes,
                combined_signal=combined_signal['signal'],
                signal_strength=combined_signal['strength'],
                consistency_score=consistency_score,
                timeframe_alignment=alignment_analysis,
                performance_metrics=performance_metrics
            )
            
            # Store in history
            self.analysis_history.append(result)
            
            logger.info("Cross-timeframe analysis completed",
                       combined_signal=result.combined_signal,
                       consistency_score=result.consistency_score,
                       signal_strength=result.signal_strength)
            
            return result
            
        except Exception as e:
            logger.error("Cross-timeframe analysis failed", error=str(e))
            raise
    
    def _generate_timeframe_signals(self, 
                                  data: pd.DataFrame,
                                  strategy_function: callable,
                                  strategy_params: Dict[str, Any]) -> List[TimeframeSignal]:
        """Generate signals for a specific timeframe."""
        try:
            signals = []
            
            # Apply strategy function
            if strategy_params:
                strategy_result = strategy_function(data, **strategy_params)
            else:
                strategy_result = strategy_function(data)
            
            # Extract signals from result
            if hasattr(strategy_result, 'signals'):
                signal_series = strategy_result.signals
            elif isinstance(strategy_result, pd.Series):
                signal_series = strategy_result
            else:
                raise ValueError("Strategy function must return signals")
            
            # Convert to TimeframeSignal objects
            for timestamp, signal_value in signal_series.items():
                if pd.notna(signal_value) and signal_value != 'hold':
                    signal = TimeframeSignal(
                        timeframe=self.primary_timeframe,
                        timestamp=timestamp,
                        signal=str(signal_value),
                        strength=1.0,  # Default strength
                        confidence=0.8  # Default confidence
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error("Failed to generate timeframe signals", error=str(e))
            return []
    
    def _analyze_signal_alignment(self, 
                                timeframe_signals: Dict[TimeframeType, List[TimeframeSignal]]) -> Dict[TimeframeType, str]:
        """Analyze alignment of signals across timeframes."""
        alignment = {}
        
        for timeframe, signals in timeframe_signals.items():
            if signals:
                # Get most recent signal
                latest_signal = max(signals, key=lambda x: x.timestamp)
                alignment[timeframe] = latest_signal.signal
            else:
                alignment[timeframe] = 'hold'
        
        return alignment
    
    def _generate_combined_signal(self, 
                                timeframe_signals: Dict[TimeframeType, List[TimeframeSignal]],
                                alignment: Dict[TimeframeType, str]) -> Dict[str, Any]:
        """Generate combined signal from multiple timeframes."""
        # Count signals by type
        signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for timeframe, signal in alignment.items():
            signal_counts[signal] += 1
        
        # Determine primary signal
        if signal_counts['buy'] > signal_counts['sell']:
            primary_signal = 'buy'
        elif signal_counts['sell'] > signal_counts['buy']:
            primary_signal = 'sell'
        else:
            primary_signal = 'hold'
        
        # Calculate signal strength based on agreement
        total_timeframes = len(alignment)
        agreement_ratio = max(signal_counts['buy'], signal_counts['sell']) / total_timeframes
        signal_strength = min(agreement_ratio * 1.5, 1.0)  # Boost strength for agreement
        
        return {
            'signal': primary_signal,
            'strength': signal_strength,
            'agreement_ratio': agreement_ratio
        }
    
    def _calculate_consistency_score(self, 
                                   timeframe_signals: Dict[TimeframeType, List[TimeframeSignal]],
                                   alignment: Dict[TimeframeType, str]) -> float:
        """Calculate consistency score across timeframes."""
        if not alignment:
            return 0.0
        
        # Calculate agreement ratio
        signal_values = list(alignment.values())
        unique_signals = set(signal_values)
        
        if len(unique_signals) == 1:
            return 1.0  # Perfect consistency
        elif 'hold' in unique_signals and len(unique_signals) == 2:
            # One signal type + hold
            non_hold_signals = [s for s in signal_values if s != 'hold']
            if len(set(non_hold_signals)) == 1:
                return 0.8  # Good consistency
            else:
                return 0.4  # Mixed signals
        else:
            # Multiple conflicting signals
            return 0.2
    
    def _calculate_performance_metrics(self, 
                                     timeframe_signals: Dict[TimeframeType, List[TimeframeSignal]],
                                     data_dict: Dict[TimeframeType, pd.DataFrame]) -> Dict[str, float]:
        """Calculate performance metrics for each timeframe."""
        metrics = {}
        
        for timeframe, signals in timeframe_signals.items():
            if timeframe in data_dict and signals:
                # Calculate basic metrics
                total_signals = len(signals)
                buy_signals = len([s for s in signals if s.signal == 'buy'])
                sell_signals = len([s for s in signals if s.signal == 'sell'])
                
                metrics[f"{timeframe.value}_total_signals"] = total_signals
                metrics[f"{timeframe.value}_buy_signals"] = buy_signals
                metrics[f"{timeframe.value}_sell_signals"] = sell_signals
                metrics[f"{timeframe.value}_signal_ratio"] = buy_signals / max(sell_signals, 1)
        
        return metrics
    
    def optimize_timeframe_combination(self, 
                                     data_dict: Dict[TimeframeType, pd.DataFrame],
                                     strategy_function: callable,
                                     strategy_params: Dict[str, Any] = None,
                                     optimization_metric: str = 'consistency') -> TimeframeOptimizationResult:
        """
        Optimize timeframe combination for best performance.
        
        Args:
            data_dict: Dictionary of data for each timeframe
            strategy_function: Function to generate signals
            strategy_params: Parameters for strategy function
            optimization_metric: Metric to optimize ('consistency', 'performance', 'robustness')
            
        Returns:
            TimeframeOptimizationResult with optimization results
        """
        try:
            logger.info("Starting timeframe optimization", optimization_metric=optimization_metric)
            
            # Test different timeframe combinations
            all_timeframes = list(data_dict.keys())
            best_combination = None
            best_score = 0.0
            optimization_results = []
            
            # Test combinations of 2-4 timeframes
            for n_timeframes in range(2, min(5, len(all_timeframes) + 1)):
                from itertools import combinations
                
                for combination in combinations(all_timeframes, n_timeframes):
                    # Set this combination
                    self.primary_timeframe = combination[0]
                    self.supporting_timeframes = list(combination[1:])
                    
                    # Run analysis
                    result = self.analyze_timeframe_signals(data_dict, strategy_function, strategy_params)
                    
                    # Calculate score based on optimization metric
                    if optimization_metric == 'consistency':
                        score = result.consistency_score
                    elif optimization_metric == 'performance':
                        score = result.signal_strength
                    elif optimization_metric == 'robustness':
                        score = (result.consistency_score + result.signal_strength) / 2
                    else:
                        score = result.consistency_score
                    
                    optimization_results.append({
                        'combination': list(combination),
                        'score': score,
                        'consistency': result.consistency_score,
                        'strength': result.signal_strength
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_combination = list(combination)
            
            # Create optimization result
            opt_result = TimeframeOptimizationResult(
                best_combination=best_combination,
                performance_score=best_score,
                consistency_score=best_score,
                robustness_score=best_score,
                optimization_metrics={'best_score': best_score},
                recommendations=[
                    f"Best combination: {[tf.value for tf in best_combination]}",
                    f"Optimization metric: {optimization_metric}",
                    f"Score: {best_score:.3f}"
                ]
            )
            
            # Store result
            self.optimization_results.append(opt_result)
            
            logger.info("Timeframe optimization completed",
                       best_combination=[tf.value for tf in best_combination],
                       best_score=best_score)
            
            return opt_result
            
        except Exception as e:
            logger.error("Timeframe optimization failed", error=str(e))
            raise
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed."""
        if not self.analysis_history:
            return {}
        
        # Calculate summary statistics
        consistency_scores = [r.consistency_score for r in self.analysis_history]
        signal_strengths = [r.signal_strength for r in self.analysis_history]
        
        summary = {
            'total_analyses': len(self.analysis_history),
            'avg_consistency_score': np.mean(consistency_scores),
            'avg_signal_strength': np.mean(signal_strengths),
            'best_consistency_score': max(consistency_scores),
            'best_signal_strength': max(signal_strengths),
            'optimization_results': len(self.optimization_results)
        }
        
        return summary

def create_cross_timeframe_analyzer(primary_timeframe: str = "1d",
                                  supporting_timeframes: List[str] = None) -> CrossTimeframeAnalyzer:
    """
    Create a cross-timeframe analyzer with specified timeframes.
    
    Args:
        primary_timeframe: Primary timeframe string
        supporting_timeframes: List of supporting timeframe strings
        
    Returns:
        CrossTimeframeAnalyzer instance
    """
    # Convert string timeframes to TimeframeType
    primary_tf = TimeframeType(primary_timeframe)
    
    if supporting_timeframes:
        supporting_tfs = [TimeframeType(tf) for tf in supporting_timeframes]
    else:
        supporting_tfs = None
    
    return CrossTimeframeAnalyzer(primary_tf, supporting_tfs)

def analyze_timeframe_consistency(data_dict: Dict[str, pd.DataFrame],
                                strategy_function: callable,
                                strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Quick function to analyze timeframe consistency.
    
    Args:
        data_dict: Dictionary with timeframe strings as keys
        strategy_function: Function to generate signals
        strategy_params: Parameters for strategy function
        
    Returns:
        Dictionary with consistency analysis results
    """
    # Convert string keys to TimeframeType
    tf_data_dict = {TimeframeType(k): v for k, v in data_dict.items()}
    
    analyzer = CrossTimeframeAnalyzer()
    result = analyzer.analyze_timeframe_signals(tf_data_dict, strategy_function, strategy_params)
    
    return {
        'consistency_score': result.consistency_score,
        'signal_strength': result.signal_strength,
        'combined_signal': result.combined_signal,
        'timeframe_alignment': {tf.value: signal for tf, signal in result.timeframe_alignment.items()}
    }