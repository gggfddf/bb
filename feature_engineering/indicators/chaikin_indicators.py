"""
Chaikin Indicators Module

Implements Chaikin-based technical indicators:
- Chaikin Oscillator
- Chaikin Money Flow

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

class ChaikinOscillator:
    """
    Chaikin Oscillator
    
    A volume-based indicator that combines price and volume to measure buying and selling pressure.
    Uses the Accumulation/Distribution Line with moving averages.
    """
    
    def __init__(self, fast_period: int = 3, slow_period: int = 10, signal_period: int = 9):
        """
        Initialize Chaikin Oscillator indicator.
        
        Args:
            fast_period: Period for fast EMA of A/D Line
            slow_period: Period for slow EMA of A/D Line
            signal_period: Period for signal line calculation
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        if fast_period < 2 or slow_period < 2 or signal_period < 2:
            raise ValueError("All periods must be at least 2")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 volume: np.ndarray) -> IndicatorResult:
        """
        Calculate Chaikin Oscillator.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array
            
        Returns:
            IndicatorResult with Chaikin Oscillator values and signals
        """
        if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
            raise ValueError("All arrays must have the same length")
        
        if len(high) < max(self.fast_period, self.slow_period, self.signal_period):
            raise ValueError(f"Not enough data points. Need at least {max(self.fast_period, self.slow_period, self.signal_period)}")
        
        # Calculate Money Flow Multiplier
        mfm = np.full(len(close), np.nan)
        valid_mask = (high != low) & (volume > 0)
        
        mfm[valid_mask] = ((close[valid_mask] - low[valid_mask]) - (high[valid_mask] - close[valid_mask])) / \
                         (high[valid_mask] - low[valid_mask])
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate Accumulation/Distribution Line
        ad_line = np.cumsum(mfv)
        
        # Calculate EMAs of A/D Line
        fast_ema = pd.Series(ad_line).ewm(span=self.fast_period).mean().values
        slow_ema = pd.Series(ad_line).ewm(span=self.slow_period).mean().values
        
        # Calculate Chaikin Oscillator
        chaikin_osc = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = pd.Series(chaikin_osc).ewm(span=self.signal_period).mean().values
        
        # Calculate histogram
        histogram = chaikin_osc - signal_line
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Zero-line crossover signals
        for i in range(1, len(close)):
            if not (np.isnan(chaikin_osc[i]) or np.isnan(chaikin_osc[i-1])):
                # Bullish crossover (oscillator turns positive)
                if chaikin_osc[i] > 0 and chaikin_osc[i-1] <= 0:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 1.0
                
                # Bearish crossover (oscillator turns negative)
                elif chaikin_osc[i] < 0 and chaikin_osc[i-1] >= 0:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 1.0
        
        # Signal line crossover signals
        for i in range(1, len(close)):
            if not (np.isnan(chaikin_osc[i]) or np.isnan(signal_line[i]) or 
                   np.isnan(chaikin_osc[i-1]) or np.isnan(signal_line[i-1])):
                
                # Bullish crossover
                if chaikin_osc[i] > signal_line[i] and chaikin_osc[i-1] <= signal_line[i-1]:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.8)
                
                # Bearish crossover
                elif chaikin_osc[i] < signal_line[i] and chaikin_osc[i-1] >= signal_line[i-1]:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.8)
        
        # Divergence signals
        for i in range(20, len(close)):  # Need enough history for divergence
            if not np.isnan(chaikin_osc[i]):
                # Price making new highs but oscillator not confirming
                recent_highs = close[i-20:i+1]
                recent_osc = chaikin_osc[i-20:i+1]
                
                if (close[i] >= np.max(recent_highs[:-1]) and 
                    chaikin_osc[i] < np.max(recent_osc[:-1])):
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
                
                # Price making new lows but oscillator not confirming
                elif (close[i] <= np.min(recent_highs[:-1]) and 
                      chaikin_osc[i] > np.min(recent_osc[:-1])):
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
        
        # Extreme value signals
        valid_osc = chaikin_osc[~np.isnan(chaikin_osc)]
        if len(valid_osc) > 0:
            osc_std = np.std(valid_osc)
            osc_mean = np.mean(valid_osc)
            
            # Very positive values (strong buy)
            very_positive = chaikin_osc > osc_mean + 2 * osc_std
            signals[very_positive] = SignalType.STRONG_BUY.value
            
            # Very negative values (strong sell)
            very_negative = chaikin_osc < osc_mean - 2 * osc_std
            signals[very_negative] = SignalType.STRONG_SELL.value
        
        return IndicatorResult(
            values=np.column_stack([chaikin_osc, signal_line, histogram, ad_line]),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period
            },
            metadata={
                'indicator_type': 'Chaikin Oscillator',
                'calculation_method': 'standard',
                'components': ['oscillator', 'signal_line', 'histogram', 'ad_line']
            }
        )
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          volume: np.ndarray, target_returns: np.ndarray, 
                          param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize Chaikin Oscillator parameters using grid search.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array
            target_returns: Target returns for optimization
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = -np.inf
        
        fast_periods = param_ranges.get('fast_period', [3, 5, 7])
        slow_periods = param_ranges.get('slow_period', [10, 14, 20])
        signal_periods = param_ranges.get('signal_period', [7, 9, 14])
        
        for fast_period in fast_periods:
            for slow_period in slow_periods:
                if fast_period >= slow_period:
                    continue
                
                for signal_period in signal_periods:
                    try:
                        self.fast_period = fast_period
                        self.slow_period = slow_period
                        self.signal_period = signal_period
                        
                        result = self.calculate(high, low, close, volume)
                        
                        # Calculate performance score
                        score = self._calculate_performance_score(result, target_returns)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'fast_period': fast_period,
                                'slow_period': slow_period,
                                'signal_period': signal_period,
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

class ChaikinMoneyFlow:
    """
    Chaikin Money Flow
    
    A volume-weighted average of price that measures buying and selling pressure.
    Ranges from -1 to +1, where positive values indicate buying pressure.
    """
    
    def __init__(self, period: int = 20, overbought: float = 0.25, oversold: float = -0.25):
        """
        Initialize Chaikin Money Flow indicator.
        
        Args:
            period: Period for calculation
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
        if period < 2:
            raise ValueError("Period must be at least 2")
        if overbought <= oversold:
            raise ValueError("Overbought level must be greater than oversold level")
        if overbought > 1 or oversold < -1:
            raise ValueError("Overbought and oversold levels must be between -1 and 1")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 volume: np.ndarray) -> IndicatorResult:
        """
        Calculate Chaikin Money Flow.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array
            
        Returns:
            IndicatorResult with Chaikin Money Flow values and signals
        """
        if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
            raise ValueError("All arrays must have the same length")
        
        if len(high) < self.period:
            raise ValueError(f"Not enough data points. Need at least {self.period}")
        
        # Calculate Money Flow Multiplier
        mfm = np.full(len(close), np.nan)
        valid_mask = (high != low) & (volume > 0)
        
        mfm[valid_mask] = ((close[valid_mask] - low[valid_mask]) - (high[valid_mask] - close[valid_mask])) / \
                         (high[valid_mask] - low[valid_mask])
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate Chaikin Money Flow
        cmf = np.full(len(close), np.nan)
        
        for i in range(self.period - 1, len(close)):
            period_mfv = mfv[i - self.period + 1:i + 1]
            period_volume = volume[i - self.period + 1:i + 1]
            
            valid_period_mask = ~np.isnan(period_mfv) & (period_volume > 0)
            
            if np.sum(valid_period_mask) >= self.period // 2:  # At least half valid values
                cmf[i] = np.sum(period_mfv[valid_period_mask]) / np.sum(period_volume[valid_period_mask])
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Overbought/oversold signals
        oversold_mask = cmf <= self.oversold
        overbought_mask = cmf >= self.overbought
        
        signals[oversold_mask] = SignalType.BUY.value
        signal_strength[oversold_mask] = (self.oversold - cmf[oversold_mask]) / abs(self.oversold)
        
        signals[overbought_mask] = SignalType.SELL.value
        signal_strength[overbought_mask] = (cmf[overbought_mask] - self.overbought) / abs(self.overbought)
        
        # Zero-line crossover signals
        for i in range(1, len(close)):
            if not (np.isnan(cmf[i]) or np.isnan(cmf[i-1])):
                # Bullish crossover (CMF turns positive)
                if cmf[i] > 0 and cmf[i-1] <= 0:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.8)
                
                # Bearish crossover (CMF turns negative)
                elif cmf[i] < 0 and cmf[i-1] >= 0:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.8)
        
        # Momentum signals
        for i in range(1, len(close)):
            if not (np.isnan(cmf[i]) or np.isnan(cmf[i-1])):
                # Strong upward momentum
                if cmf[i] > cmf[i-1] * 1.2:  # 20% increase
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
                
                # Strong downward momentum
                elif cmf[i] < cmf[i-1] * 0.8:  # 20% decrease
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
        
        # Strong signals for extreme values
        strong_oversold = cmf <= self.oversold * 1.5  # 50% beyond oversold
        strong_overbought = cmf >= self.overbought * 1.5  # 50% beyond overbought
        
        signals[strong_oversold] = SignalType.STRONG_BUY.value
        signals[strong_overbought] = SignalType.STRONG_SELL.value
        
        # Divergence signals (price vs CMF)
        for i in range(20, len(close)):  # Need enough history for divergence
            if not np.isnan(cmf[i]):
                # Price making new highs but CMF not confirming
                recent_highs = close[i-20:i+1]
                recent_cmf = cmf[i-20:i+1]
                
                if (close[i] >= np.max(recent_highs[:-1]) and 
                    cmf[i] < np.max(recent_cmf[:-1])):
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
                
                # Price making new lows but CMF not confirming
                elif (close[i] <= np.min(recent_highs[:-1]) and 
                      cmf[i] > np.min(recent_cmf[:-1])):
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = max(signal_strength[i], 0.6)
        
        return IndicatorResult(
            values=np.column_stack([cmf, mfm, mfv]),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'overbought': self.overbought,
                'oversold': self.oversold
            },
            metadata={
                'indicator_type': 'Chaikin Money Flow',
                'calculation_method': 'standard',
                'components': ['cmf', 'mfm', 'mfv']
            }
        )
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          volume: np.ndarray, target_returns: np.ndarray, 
                          param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize Chaikin Money Flow parameters using grid search.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array
            target_returns: Target returns for optimization
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = -np.inf
        
        periods = param_ranges.get('period', [14, 20, 30])
        overbought_levels = param_ranges.get('overbought', [0.2, 0.25, 0.3])
        oversold_levels = param_ranges.get('oversold', [-0.3, -0.25, -0.2])
        
        for period in periods:
            for overbought in overbought_levels:
                for oversold in oversold_levels:
                    if overbought <= oversold:
                        continue
                    
                    try:
                        self.period = period
                        self.overbought = overbought
                        self.oversold = oversold
                        
                        result = self.calculate(high, low, close, volume)
                        
                        # Calculate performance score
                        score = self._calculate_performance_score(result, target_returns)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'period': period,
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
        
        # Add penalty for too many signals (overfitting)
        signal_frequency = total_signals / len(signal_numeric[:-1])
        if signal_frequency > 0.35:  # More than 35% signals
            accuracy *= 0.8
        
        return accuracy

# Convenience functions for easy usage
def calculate_chaikin_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                volume: np.ndarray, fast_period: int = 3, slow_period: int = 10,
                                signal_period: int = 9) -> IndicatorResult:
    """Calculate Chaikin Oscillator with default parameters."""
    chaikin_osc = ChaikinOscillator(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
    return chaikin_osc.calculate(high, low, close, volume)

def calculate_chaikin_money_flow(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                volume: np.ndarray, period: int = 20, overbought: float = 0.25,
                                oversold: float = -0.25) -> IndicatorResult:
    """Calculate Chaikin Money Flow with default parameters."""
    chaikin_mf = ChaikinMoneyFlow(period=period, overbought=overbought, oversold=oversold)
    return chaikin_mf.calculate(high, low, close, volume)