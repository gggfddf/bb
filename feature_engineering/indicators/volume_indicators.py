"""
Volume Indicators Module

Implements volume-based technical indicators:
- Elder's Force Index
- Coppock Curve

Each indicator includes:
- Multiple calculation methods
- Parameter optimization capabilities
- Signal generation
- Validation and error handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import structlog

logger = structlog.get_logger()

class SignalType(Enum):
    """Signal types for indicator outputs."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

@dataclass
class IndicatorResult:
    """Container for indicator calculation results."""
    values: np.ndarray
    signals: np.ndarray
    signal_strength: np.ndarray
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

class EldersForceIndex:
    """
    Elder's Force Index
    
    A volume-based indicator that measures the power behind price movements.
    Combines price change and volume to identify potential trend reversals.
    """
    
    def __init__(self, period: int = 13, signal_period: int = 9, threshold: float = 0.5):
        """
        Initialize Elder's Force Index indicator.
        
        Args:
            period: Period for smoothing the Force Index
            signal_period: Period for signal line calculation
            threshold: Threshold for signal generation
        """
        self.period = period
        self.signal_period = signal_period
        self.threshold = threshold
        
        if period < 2 or signal_period < 2:
            raise ValueError("All periods must be at least 2")
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
    
    def calculate(self, close: np.ndarray, volume: np.ndarray) -> IndicatorResult:
        """
        Calculate Elder's Force Index.
        
        Args:
            close: Close prices array
            volume: Volume array
            
        Returns:
            IndicatorResult with Force Index values and signals
        """
        if len(close) != len(volume):
            raise ValueError("Close and volume arrays must have the same length")
        
        if len(close) < max(self.period, self.signal_period):
            raise ValueError(f"Not enough data points. Need at least {max(self.period, self.signal_period)}")
        
        # Calculate price change
        price_change = np.diff(close)
        price_change = np.insert(price_change, 0, 0)
        
        # Calculate Force Index
        force_index = price_change * volume
        
        # Smooth the Force Index
        smoothed_fi = pd.Series(force_index).ewm(span=self.period).mean().values
        
        # Calculate signal line
        signal_line = pd.Series(smoothed_fi).ewm(span=self.signal_period).mean().values
        
        # Calculate histogram (difference between Force Index and signal line)
        histogram = smoothed_fi - signal_line
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Divergence signals
        for i in range(1, len(close)):
            if not (np.isnan(smoothed_fi[i]) or np.isnan(signal_line[i]) or 
                   np.isnan(smoothed_fi[i-1]) or np.isnan(signal_line[i-1])):
                
                # Bullish divergence: price down but Force Index up
                if (price_change[i] < 0 and smoothed_fi[i] > smoothed_fi[i-1] and 
                    smoothed_fi[i] > signal_line[i]):
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                
                # Bearish divergence: price up but Force Index down
                elif (price_change[i] > 0 and smoothed_fi[i] < smoothed_fi[i-1] and 
                      smoothed_fi[i] < signal_line[i]):
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
        
        # Threshold-based signals
        threshold_multiplier = self.threshold * np.std(histogram[~np.isnan(histogram)])
        
        # Strong positive Force Index
        strong_positive = histogram > threshold_multiplier
        signals[strong_positive] = SignalType.BUY.value
        signal_strength[strong_positive] = np.minimum(
            histogram[strong_positive] / threshold_multiplier, 2.0
        )
        
        # Strong negative Force Index
        strong_negative = histogram < -threshold_multiplier
        signals[strong_negative] = SignalType.SELL.value
        signal_strength[strong_negative] = np.minimum(
            -histogram[strong_negative] / threshold_multiplier, 2.0
        )
        
        # Zero-line crossover signals
        for i in range(1, len(close)):
            if not (np.isnan(histogram[i]) or np.isnan(histogram[i-1])):
                # Bullish crossover
                if histogram[i] > 0 and histogram[i-1] <= 0:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
                
                # Bearish crossover
                elif histogram[i] < 0 and histogram[i-1] >= 0:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
        
        # Strong signals for extreme values
        extreme_positive = histogram > threshold_multiplier * 2
        extreme_negative = histogram < -threshold_multiplier * 2
        
        signals[extreme_positive] = SignalType.STRONG_BUY.value
        signals[extreme_negative] = SignalType.STRONG_SELL.value
        
        return IndicatorResult(
            values=np.column_stack([smoothed_fi, signal_line, histogram]),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'signal_period': self.signal_period,
                'threshold': self.threshold
            },
            metadata={
                'indicator_type': "Elder's Force Index",
                'calculation_method': 'standard',
                'components': ['force_index', 'signal_line', 'histogram']
            }
        )
    
    def optimize_parameters(self, close: np.ndarray, volume: np.ndarray,
                          target_returns: np.ndarray, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize Elder's Force Index parameters using grid search.
        
        Args:
            close: Close prices array
            volume: Volume array
            target_returns: Target returns for optimization
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = -np.inf
        
        periods = param_ranges.get('period', [10, 13, 20])
        signal_periods = param_ranges.get('signal_period', [7, 9, 14])
        thresholds = param_ranges.get('threshold', [0.3, 0.5, 0.7, 1.0])
        
        for period in periods:
            for signal_period in signal_periods:
                for threshold in thresholds:
                    try:
                        self.period = period
                        self.signal_period = signal_period
                        self.threshold = threshold
                        
                        result = self.calculate(close, volume)
                        
                        # Calculate performance score
                        score = self._calculate_performance_score(result, target_returns)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'period': period,
                                'signal_period': signal_period,
                                'threshold': threshold,
                                'score': score
                            }
                    
                    except Exception as e:
                        logger.warning(f"Parameter combination failed: {e}")
                        continue
        
        return best_params
    
    def _calculate_performance_score(self, result: IndicatorResult, target_returns: np.ndarray) -> float:
        """Calculate performance score for parameter optimization."""
        if len(result.signals) != len(target_returns):
            return -np.inf
        
        # Calculate signal accuracy
        signal_numeric = np.where(result.signals == SignalType.BUY.value, 1,
                                np.where(result.signals == SignalType.SELL.value, -1, 0))
        
        # Forward returns (next period)
        forward_returns = np.roll(target_returns, -1)
        forward_returns[-1] = 0  # Last value
        
        # Calculate directional accuracy
        correct_signals = np.sum((signal_numeric[:-1] > 0) & (forward_returns[:-1] > 0)) + \
                         np.sum((signal_numeric[:-1] < 0) & (forward_returns[:-1] < 0))
        
        total_signals = np.sum(signal_numeric[:-1] != 0)
        
        if total_signals == 0:
            return -np.inf
        
        accuracy = correct_signals / total_signals
        
        # Add penalty for too many signals (overfitting)
        signal_frequency = total_signals / len(signal_numeric[:-1])
        if signal_frequency > 0.3:  # More than 30% signals
            accuracy *= 0.8
        
        return accuracy

class CoppockCurve:
    """
    Coppock Curve
    
    A long-term momentum indicator designed to identify major bottoms in the market.
    Combines rate of change calculations with smoothing.
    """
    
    def __init__(self, wma_period: int = 10, roc1_period: int = 14, roc2_period: int = 11):
        """
        Initialize Coppock Curve indicator.
        
        Args:
            wma_period: Period for weighted moving average smoothing
            roc1_period: First rate of change period
            roc2_period: Second rate of change period
        """
        self.wma_period = wma_period
        self.roc1_period = roc1_period
        self.roc2_period = roc2_period
        
        if wma_period < 2 or roc1_period < 2 or roc2_period < 2:
            raise ValueError("All periods must be at least 2")
    
    def calculate(self, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Coppock Curve.
        
        Args:
            close: Close prices array
            
        Returns:
            IndicatorResult with Coppock Curve values and signals
        """
        if len(close) < max(self.roc1_period, self.roc2_period) + self.wma_period:
            raise ValueError(f"Not enough data points. Need at least {max(self.roc1_period, self.roc2_period) + self.wma_period}")
        
        # Calculate rate of change for both periods
        roc1 = np.full(len(close), np.nan)
        roc2 = np.full(len(close), np.nan)
        
        # ROC1 calculation
        for i in range(self.roc1_period, len(close)):
            roc1[i] = ((close[i] - close[i - self.roc1_period]) / close[i - self.roc1_period]) * 100
        
        # ROC2 calculation
        for i in range(self.roc2_period, len(close)):
            roc2[i] = ((close[i] - close[i - self.roc2_period]) / close[i - self.roc2_period]) * 100
        
        # Sum the ROC values
        roc_sum = roc1 + roc2
        
        # Apply weighted moving average smoothing
        coppock = np.full(len(close), np.nan)
        
        for i in range(self.wma_period - 1, len(close)):
            if not np.isnan(roc_sum[i]):
                # Calculate weighted moving average
                weights = np.arange(1, self.wma_period + 1)
                values = roc_sum[i - self.wma_period + 1:i + 1]
                valid_mask = ~np.isnan(values)
                
                if np.sum(valid_mask) >= self.wma_period // 2:  # At least half valid values
                    coppock[i] = np.average(values[valid_mask], weights=weights[valid_mask])
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Zero-line crossover signals
        for i in range(1, len(close)):
            if not (np.isnan(coppock[i]) or np.isnan(coppock[i-1])):
                # Bullish crossover (Coppock turns positive)
                if coppock[i] > 0 and coppock[i-1] <= 0:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 1.0
                
                # Bearish crossover (Coppock turns negative)
                elif coppock[i] < 0 and coppock[i-1] >= 0:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 1.0
        
        # Momentum signals
        for i in range(1, len(close)):
            if not (np.isnan(coppock[i]) or np.isnan(coppock[i-1])):
                # Strong upward momentum
                if coppock[i] > coppock[i-1] * 1.1:  # 10% increase
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.7)
                
                # Strong downward momentum
                elif coppock[i] < coppock[i-1] * 0.9:  # 10% decrease
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.7)
        
        # Extreme value signals
        valid_coppock = coppock[~np.isnan(coppock)]
        if len(valid_coppock) > 0:
            coppock_std = np.std(valid_coppock)
            coppock_mean = np.mean(valid_coppock)
            
            # Very positive values (strong buy)
            very_positive = coppock > coppock_mean + 2 * coppock_std
            signals[very_positive] = SignalType.STRONG_BUY.value
            
            # Very negative values (strong sell)
            very_negative = coppock < coppock_mean - 2 * coppock_std
            signals[very_negative] = SignalType.STRONG_SELL.value
        
        # Divergence signals (price vs Coppock)
        for i in range(20, len(close)):  # Need enough history for divergence
            if not np.isnan(coppock[i]):
                # Price making new highs but Coppock not confirming
                recent_highs = close[i-20:i+1]
                recent_coppock = coppock[i-20:i+1]
                
                if (close[i] >= np.max(recent_highs[:-1]) and 
                    coppock[i] < np.max(recent_coppock[:-1])):
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
                
                # Price making new lows but Coppock not confirming
                elif (close[i] <= np.min(recent_highs[:-1]) and 
                      coppock[i] > np.min(recent_coppock[:-1])):
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
        
        return IndicatorResult(
            values=np.column_stack([coppock, roc1, roc2]),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'wma_period': self.wma_period,
                'roc1_period': self.roc1_period,
                'roc2_period': self.roc2_period
            },
            metadata={
                'indicator_type': 'Coppock Curve',
                'calculation_method': 'standard',
                'components': ['coppock', 'roc1', 'roc2']
            }
        )
    
    def optimize_parameters(self, close: np.ndarray, target_returns: np.ndarray,
                          param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize Coppock Curve parameters using grid search.
        
        Args:
            close: Close prices array
            target_returns: Target returns for optimization
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = -np.inf
        
        wma_periods = param_ranges.get('wma_period', [8, 10, 12, 14])
        roc1_periods = param_ranges.get('roc1_period', [11, 14, 20])
        roc2_periods = param_ranges.get('roc2_period', [10, 11, 14])
        
        for wma_period in wma_periods:
            for roc1_period in roc1_periods:
                for roc2_period in roc2_periods:
                    try:
                        self.wma_period = wma_period
                        self.roc1_period = roc1_period
                        self.roc2_period = roc2_period
                        
                        result = self.calculate(close)
                        
                        # Calculate performance score
                        score = self._calculate_performance_score(result, target_returns)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'wma_period': wma_period,
                                'roc1_period': roc1_period,
                                'roc2_period': roc2_period,
                                'score': score
                            }
                    
                    except Exception as e:
                        logger.warning(f"Parameter combination failed: {e}")
                        continue
        
        return best_params
    
    def _calculate_performance_score(self, result: IndicatorResult, target_returns: np.ndarray) -> float:
        """Calculate performance score for parameter optimization."""
        if len(result.signals) != len(target_returns):
            return -np.inf
        
        # Calculate signal accuracy
        signal_numeric = np.where(result.signals == SignalType.BUY.value, 1,
                                np.where(result.signals == SignalType.SELL.value, -1, 0))
        
        # Forward returns (next period)
        forward_returns = np.roll(target_returns, -1)
        forward_returns[-1] = 0  # Last value
        
        # Calculate directional accuracy
        correct_signals = np.sum((signal_numeric[:-1] > 0) & (forward_returns[:-1] > 0)) + \
                         np.sum((signal_numeric[:-1] < 0) & (forward_returns[:-1] < 0))
        
        total_signals = np.sum(signal_numeric[:-1] != 0)
        
        if total_signals == 0:
            return -np.inf
        
        accuracy = correct_signals / total_signals
        
        # Add penalty for too many signals (overfitting)
        signal_frequency = total_signals / len(signal_numeric[:-1])
        if signal_frequency > 0.25:  # More than 25% signals
            accuracy *= 0.8
        
        return accuracy

# Convenience functions for easy usage
def calculate_elders_force_index(close: np.ndarray, volume: np.ndarray,
                                period: int = 13, signal_period: int = 9, threshold: float = 0.5) -> IndicatorResult:
    """Calculate Elder's Force Index with default parameters."""
    force_index = EldersForceIndex(period=period, signal_period=signal_period, threshold=threshold)
    return force_index.calculate(close, volume)

def calculate_coppock_curve(close: np.ndarray, wma_period: int = 10, 
                           roc1_period: int = 14, roc2_period: int = 11) -> IndicatorResult:
    """Calculate Coppock Curve with default parameters."""
    coppock = CoppockCurve(wma_period=wma_period, roc1_period=roc1_period, roc2_period=roc2_period)
    return coppock.calculate(close)