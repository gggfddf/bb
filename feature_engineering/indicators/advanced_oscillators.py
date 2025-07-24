"""
Advanced Oscillators Module

Implements advanced oscillator-based technical indicators:
- SuperTrend
- TSI (True Strength Index)
- Ulcer Index

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

class SuperTrend:
    """
    SuperTrend Indicator
    
    A trend-following indicator that combines ATR with price action to identify trend direction.
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        Initialize SuperTrend indicator.
        
        Args:
            period: Period for ATR calculation
            multiplier: ATR multiplier for band calculation
        """
        self.period = period
        self.multiplier = multiplier
        
        if period < 2:
            raise ValueError("Period must be at least 2")
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate SuperTrend.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            
        Returns:
            IndicatorResult with SuperTrend values and signals
        """
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("All price arrays must have the same length")
        
        if len(high) < self.period:
            raise ValueError(f"Not enough data points. Need at least {self.period}")
        
        # Calculate ATR
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]  # First value
        
        atr = pd.Series(tr).rolling(window=self.period).mean().values
        
        # Calculate basic upper and lower bands
        basic_upper = (high + low) / 2 + (self.multiplier * atr)
        basic_lower = (high + low) / 2 - (self.multiplier * atr)
        
        # Initialize SuperTrend arrays
        supertrend = np.full(len(close), np.nan)
        direction = np.full(len(close), np.nan)
        
        # Initialize first values
        supertrend[0] = basic_lower[0]
        direction[0] = 1  # 1 for uptrend, -1 for downtrend
        
        # Calculate SuperTrend
        for i in range(1, len(close)):
            if np.isnan(atr[i]):
                continue
            
            # Update SuperTrend based on direction
            if direction[i-1] == 1:  # Uptrend
                if close[i] > basic_upper[i]:
                    supertrend[i] = basic_lower[i]
                    direction[i] = 1
                else:
                    if basic_lower[i] < supertrend[i-1]:
                        supertrend[i] = basic_lower[i]
                    else:
                        supertrend[i] = supertrend[i-1]
                    direction[i] = 1
            else:  # Downtrend
                if close[i] < basic_lower[i]:
                    supertrend[i] = basic_upper[i]
                    direction[i] = -1
                else:
                    if basic_upper[i] > supertrend[i-1]:
                        supertrend[i] = basic_upper[i]
                    else:
                        supertrend[i] = supertrend[i-1]
                    direction[i] = -1
            
            # Check for trend reversal
            if direction[i-1] == 1 and close[i] < supertrend[i-1]:
                direction[i] = -1
                supertrend[i] = basic_upper[i]
            elif direction[i-1] == -1 and close[i] > supertrend[i-1]:
                direction[i] = 1
                supertrend[i] = basic_lower[i]
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Trend change signals
        for i in range(1, len(close)):
            if direction[i] != direction[i-1]:
                if direction[i] == 1:  # Trend changed to uptrend
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 1.0
                else:  # Trend changed to downtrend
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 1.0
        
        # Strong trend signals (price far from SuperTrend)
        trend_strength_threshold = 0.02  # 2% from SuperTrend
        
        for i in range(len(close)):
            if not np.isnan(supertrend[i]):
                price_distance = abs(close[i] - supertrend[i]) / supertrend[i]
                
                if price_distance > trend_strength_threshold:
                    if direction[i] == 1:  # Strong uptrend
                        signals[i] = SignalType.STRONG_BUY.value
                        signal_strength[i] = min(price_distance / trend_strength_threshold, 2.0)
                    else:  # Strong downtrend
                        signals[i] = SignalType.STRONG_SELL.value
                        signal_strength[i] = min(price_distance / trend_strength_threshold, 2.0)
        
        return IndicatorResult(
            values=np.column_stack([supertrend, direction, atr]),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'multiplier': self.multiplier
            },
            metadata={
                'indicator_type': 'SuperTrend',
                'calculation_method': 'standard',
                'components': ['supertrend', 'direction', 'atr']
            }
        )
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          target_returns: np.ndarray, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize SuperTrend parameters using grid search.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            target_returns: Target returns for optimization
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = -np.inf
        
        periods = param_ranges.get('period', [7, 10, 14, 21])
        multipliers = param_ranges.get('multiplier', [2.0, 3.0, 4.0, 5.0])
        
        for period in periods:
            for multiplier in multipliers:
                try:
                    self.period = period
                    self.multiplier = multiplier
                    
                    result = self.calculate(high, low, close)
                    
                    # Calculate performance score
                    score = self._calculate_performance_score(result, target_returns)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'period': period,
                            'multiplier': multiplier,
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
        
        # Calculate signal accuracy with forward returns
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
        if signal_frequency > 0.2:  # More than 20% signals
            accuracy *= 0.8
        
        return accuracy

class TSI:
    """
    True Strength Index (TSI)
    
    A momentum oscillator that shows both trend direction and overbought/oversold conditions.
    """
    
    def __init__(self, first_period: int = 25, second_period: int = 13, signal_period: int = 9,
                 overbought: float = 25, oversold: float = -25):
        """
        Initialize TSI indicator.
        
        Args:
            first_period: First smoothing period
            second_period: Second smoothing period
            signal_period: Signal line period
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        self.first_period = first_period
        self.second_period = second_period
        self.signal_period = signal_period
        self.overbought = overbought
        self.oversold = oversold
        
        if first_period < 2 or second_period < 2 or signal_period < 2:
            raise ValueError("All periods must be at least 2")
        if overbought <= oversold:
            raise ValueError("Overbought level must be greater than oversold level")
    
    def calculate(self, close: np.ndarray) -> IndicatorResult:
        """
        Calculate TSI.
        
        Args:
            close: Close prices array
            
        Returns:
            IndicatorResult with TSI values and signals
        """
        if len(close) < max(self.first_period, self.second_period, self.signal_period):
            raise ValueError(f"Not enough data points. Need at least {max(self.first_period, self.second_period, self.signal_period)}")
        
        # Calculate price change
        price_change = np.diff(close)
        price_change = np.insert(price_change, 0, 0)
        
        # Calculate absolute price change
        abs_price_change = np.abs(price_change)
        
        # First smoothing of price change
        first_smooth = pd.Series(price_change).ewm(span=self.first_period).mean().values
        
        # Second smoothing of price change
        second_smooth = pd.Series(first_smooth).ewm(span=self.second_period).mean().values
        
        # First smoothing of absolute price change
        first_abs_smooth = pd.Series(abs_price_change).ewm(span=self.first_period).mean().values
        
        # Second smoothing of absolute price change
        second_abs_smooth = pd.Series(first_abs_smooth).ewm(span=self.second_period).mean().values
        
        # Calculate TSI
        tsi = np.full(len(close), np.nan)
        valid_mask = (second_abs_smooth != 0) & ~np.isnan(second_abs_smooth)
        tsi[valid_mask] = (second_smooth[valid_mask] / second_abs_smooth[valid_mask]) * 100
        
        # Calculate signal line
        signal_line = pd.Series(tsi).ewm(span=self.signal_period).mean().values
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Overbought/oversold signals
        oversold_mask = tsi <= self.oversold
        overbought_mask = tsi >= self.overbought
        
        signals[oversold_mask] = SignalType.BUY.value
        signal_strength[oversold_mask] = (self.oversold - tsi[oversold_mask]) / abs(self.oversold)
        
        signals[overbought_mask] = SignalType.SELL.value
        signal_strength[overbought_mask] = (tsi[overbought_mask] - self.overbought) / abs(self.overbought)
        
        # Signal line crossover signals
        for i in range(1, len(close)):
            if not (np.isnan(tsi[i]) or np.isnan(signal_line[i]) or 
                   np.isnan(tsi[i-1]) or np.isnan(signal_line[i-1])):
                
                # Bullish crossover
                if tsi[i] > signal_line[i] and tsi[i-1] <= signal_line[i-1]:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.8)
                
                # Bearish crossover
                elif tsi[i] < signal_line[i] and tsi[i-1] >= signal_line[i-1]:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.8)
        
        # Strong signals for extreme values
        strong_oversold = tsi <= self.oversold * 1.5  # 50% beyond oversold
        strong_overbought = tsi >= self.overbought * 1.5  # 50% beyond overbought
        
        signals[strong_oversold] = SignalType.STRONG_BUY.value
        signals[strong_overbought] = SignalType.STRONG_SELL.value
        
        return IndicatorResult(
            values=np.column_stack([tsi, signal_line]),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'first_period': self.first_period,
                'second_period': self.second_period,
                'signal_period': self.signal_period,
                'overbought': self.overbought,
                'oversold': self.oversold
            },
            metadata={
                'indicator_type': 'TSI',
                'calculation_method': 'standard',
                'components': ['tsi', 'signal_line']
            }
        )
    
    def optimize_parameters(self, close: np.ndarray, target_returns: np.ndarray,
                          param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize TSI parameters using grid search.
        
        Args:
            close: Close prices array
            target_returns: Target returns for optimization
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = -np.inf
        
        first_periods = param_ranges.get('first_period', [20, 25, 30])
        second_periods = param_ranges.get('second_period', [10, 13, 20])
        signal_periods = param_ranges.get('signal_period', [7, 9, 14])
        overbought_levels = param_ranges.get('overbought', [20, 25, 30])
        oversold_levels = param_ranges.get('oversold', [-30, -25, -20])
        
        for first_period in first_periods:
            for second_period in second_periods:
                for signal_period in signal_periods:
                    for overbought in overbought_levels:
                        for oversold in oversold_levels:
                            if overbought <= oversold:
                                continue
                            
                            try:
                                self.first_period = first_period
                                self.second_period = second_period
                                self.signal_period = signal_period
                                self.overbought = overbought
                                self.oversold = oversold
                                
                                result = self.calculate(close)
                                
                                # Calculate performance score
                                score = self._calculate_performance_score(result, target_returns)
                                
                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'first_period': first_period,
                                        'second_period': second_period,
                                        'signal_period': signal_period,
                                        'overbought': overbought,
                                        'oversold': oversold,
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
        return accuracy

class UlcerIndex:
    """
    Ulcer Index
    
    A volatility indicator that measures downside risk and drawdowns.
    Lower values indicate less downside risk.
    """
    
    def __init__(self, period: int = 14, threshold: float = 5.0):
        """
        Initialize Ulcer Index indicator.
        
        Args:
            period: Lookback period for calculation
            threshold: Threshold for signal generation
        """
        self.period = period
        self.threshold = threshold
        
        if period < 2:
            raise ValueError("Period must be at least 2")
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
    
    def calculate(self, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Ulcer Index.
        
        Args:
            close: Close prices array
            
        Returns:
            IndicatorResult with Ulcer Index values and signals
        """
        if len(close) < self.period:
            raise ValueError(f"Not enough data points. Need at least {self.period}")
        
        # Calculate highest high over the period
        highest_high = pd.Series(close).rolling(window=self.period).max().values
        
        # Calculate drawdown percentage
        drawdown_pct = np.full(len(close), np.nan)
        valid_mask = ~np.isnan(highest_high)
        drawdown_pct[valid_mask] = ((close[valid_mask] - highest_high[valid_mask]) / 
                                   highest_high[valid_mask]) * 100
        
        # Calculate squared drawdown
        squared_drawdown = np.square(drawdown_pct)
        squared_drawdown[drawdown_pct > 0] = 0  # Only consider negative drawdowns
        
        # Calculate Ulcer Index
        ulcer_index = pd.Series(squared_drawdown).rolling(window=self.period).mean().values
        ulcer_index = np.sqrt(ulcer_index)
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Low volatility signals (buy opportunities)
        low_volatility = ulcer_index <= self.threshold
        signals[low_volatility] = SignalType.BUY.value
        signal_strength[low_volatility] = (self.threshold - ulcer_index[low_volatility]) / self.threshold
        
        # High volatility signals (sell opportunities)
        high_volatility = ulcer_index >= self.threshold * 2  # 2x threshold
        signals[high_volatility] = SignalType.SELL.value
        signal_strength[high_volatility] = (ulcer_index[high_volatility] - self.threshold * 2) / (self.threshold * 2)
        
        # Trend change signals (decreasing volatility)
        for i in range(1, len(close)):
            if not (np.isnan(ulcer_index[i]) or np.isnan(ulcer_index[i-1])):
                if ulcer_index[i] < ulcer_index[i-1] * 0.8:  # 20% decrease
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
                elif ulcer_index[i] > ulcer_index[i-1] * 1.2:  # 20% increase
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
        
        # Strong signals for extreme values
        very_low_volatility = ulcer_index <= self.threshold * 0.5  # 50% of threshold
        very_high_volatility = ulcer_index >= self.threshold * 3  # 3x threshold
        
        signals[very_low_volatility] = SignalType.STRONG_BUY.value
        signals[very_high_volatility] = SignalType.STRONG_SELL.value
        
        return IndicatorResult(
            values=np.column_stack([ulcer_index, drawdown_pct, squared_drawdown]),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'threshold': self.threshold
            },
            metadata={
                'indicator_type': 'Ulcer Index',
                'calculation_method': 'standard',
                'components': ['ulcer_index', 'drawdown_pct', 'squared_drawdown']
            }
        )
    
    def optimize_parameters(self, close: np.ndarray, target_returns: np.ndarray,
                          param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize Ulcer Index parameters using grid search.
        
        Args:
            close: Close prices array
            target_returns: Target returns for optimization
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = -np.inf
        
        periods = param_ranges.get('period', [10, 14, 20, 30])
        thresholds = param_ranges.get('threshold', [3.0, 5.0, 7.0, 10.0])
        
        for period in periods:
            for threshold in thresholds:
                try:
                    self.period = period
                    self.threshold = threshold
                    
                    result = self.calculate(close)
                    
                    # Calculate performance score
                    score = self._calculate_performance_score(result, target_returns)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'period': period,
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
        if signal_frequency > 0.4:  # More than 40% signals
            accuracy *= 0.8
        
        return accuracy

# Convenience functions for easy usage
def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        period: int = 10, multiplier: float = 3.0) -> IndicatorResult:
    """Calculate SuperTrend with default parameters."""
    supertrend = SuperTrend(period=period, multiplier=multiplier)
    return supertrend.calculate(high, low, close)

def calculate_tsi(close: np.ndarray, first_period: int = 25, second_period: int = 13,
                 signal_period: int = 9, overbought: float = 25, oversold: float = -25) -> IndicatorResult:
    """Calculate TSI with default parameters."""
    tsi = TSI(first_period=first_period, second_period=second_period, signal_period=signal_period,
              overbought=overbought, oversold=oversold)
    return tsi.calculate(close)

def calculate_ulcer_index(close: np.ndarray, period: int = 14, threshold: float = 5.0) -> IndicatorResult:
    """Calculate Ulcer Index with default parameters."""
    ulcer = UlcerIndex(period=period, threshold=threshold)
    return ulcer.calculate(close)