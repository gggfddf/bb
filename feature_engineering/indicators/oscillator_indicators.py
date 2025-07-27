"""
Oscillator Indicators Module

Implements oscillator-based technical indicators:
- Williams %R
- Keltner Channel
- Donchian Channel

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

class WilliamsPercentR:
    """
    Williams %R Oscillator
    
    A momentum indicator that measures overbought/oversold levels.
    Ranges from 0 to -100, where 0 is overbought and -100 is oversold.
    """
    
    def __init__(self, period: int = 14, overbought: float = -20, oversold: float = -80):
        """
        Initialize Williams %R indicator.
        
        Args:
            period: Lookback period for high/low calculation
            overbought: Overbought threshold (typically -20)
            oversold: Oversold threshold (typically -80)
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
        if period < 2:
            raise ValueError("Period must be at least 2")
        if overbought >= oversold:
            raise ValueError("Overbought level must be less than oversold level")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Williams %R.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            
        Returns:
            IndicatorResult with Williams %R values and signals
        """
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("All price arrays must have the same length")
        
        if len(high) < self.period:
            raise ValueError(f"Not enough data points. Need at least {self.period}")
        
        # Calculate highest high and lowest low over the period
        highest_high = pd.Series(high).rolling(window=self.period).max().values
        lowest_low = pd.Series(low).rolling(window=self.period).min().values
        
        # Calculate Williams %R
        williams_r = np.full(len(close), np.nan)
        valid_mask = ~(np.isnan(highest_high) | np.isnan(lowest_low))
        
        williams_r[valid_mask] = (
            (highest_high[valid_mask] - close[valid_mask]) / 
            (highest_high[valid_mask] - lowest_low[valid_mask]) * -100
        )
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Oversold conditions (buy signals)
        oversold_mask = williams_r <= self.oversold
        signals[oversold_mask] = SignalType.BUY.value
        signal_strength[oversold_mask] = (self.oversold - williams_r[oversold_mask]) / abs(self.oversold)
        
        # Overbought conditions (sell signals)
        overbought_mask = williams_r >= self.overbought
        signals[overbought_mask] = SignalType.SELL.value
        signal_strength[overbought_mask] = (williams_r[overbought_mask] - self.overbought) / abs(self.overbought)
        
        # Strong signals for extreme values
        strong_oversold = williams_r <= self.oversold * 1.2  # 20% beyond oversold
        strong_overbought = williams_r >= self.overbought * 0.8  # 20% beyond overbought
        
        signals[strong_oversold] = SignalType.STRONG_BUY.value
        signals[strong_overbought] = SignalType.STRONG_SELL.value
        
        return IndicatorResult(
            values=williams_r,
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'overbought': self.overbought,
                'oversold': self.oversold
            },
            metadata={
                'indicator_type': 'Williams %R',
                'calculation_method': 'standard',
                'valid_data_points': np.sum(valid_mask)
            }
        )
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          target_returns: np.ndarray, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize Williams %R parameters using grid search.
        
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
        
        periods = param_ranges.get('period', [10, 14, 20])
        overbought_levels = param_ranges.get('overbought', [-10, -20, -30])
        oversold_levels = param_ranges.get('oversold', [-70, -80, -90])
        
        for period in periods:
            for overbought in overbought_levels:
                for oversold in oversold_levels:
                    if overbought >= oversold:
                        continue
                    
                    try:
                        self.period = period
                        self.overbought = overbought
                        self.oversold = oversold
                        
                        result = self.calculate(high, low, close)
                        
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
        
        # Simple scoring: correlation between signals and returns
        signal_numeric = np.where(result.signals == SignalType.BUY.value, 1,
                                np.where(result.signals == SignalType.SELL.value, -1, 0))
        
        correlation = np.corrcoef(signal_numeric, target_returns)[0, 1]
        return correlation if not np.isnan(correlation) else -np.inf

class KeltnerChannel:
    """
    Keltner Channel
    
    A volatility-based indicator that uses ATR to create dynamic support/resistance levels.
    """
    
    def __init__(self, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        """
        Initialize Keltner Channel indicator.
        
        Args:
            ema_period: Period for EMA calculation
            atr_period: Period for ATR calculation
            multiplier: ATR multiplier for channel width
        """
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
        
        if ema_period < 2 or atr_period < 2:
            raise ValueError("Periods must be at least 2")
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Keltner Channel.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            
        Returns:
            IndicatorResult with Keltner Channel values and signals
        """
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("All price arrays must have the same length")
        
        min_period = max(self.ema_period, self.atr_period)
        if len(close) < min_period:
            raise ValueError(f"Not enough data points. Need at least {min_period}")
        
        # Calculate EMA
        ema = pd.Series(close).ewm(span=self.ema_period).mean().values
        
        # Calculate ATR
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]  # First value
        
        atr = pd.Series(tr).ewm(span=self.atr_period).mean().values
        
        # Calculate channel bands
        upper_band = ema + (self.multiplier * atr)
        lower_band = ema - (self.multiplier * atr)
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Price relative to bands
        price_position = (close - lower_band) / (upper_band - lower_band)
        
        # Breakout signals
        breakout_threshold = 0.05  # 5% beyond bands
        
        # Upper breakout (potential sell)
        upper_breakout = close > upper_band * (1 + breakout_threshold)
        signals[upper_breakout] = SignalType.SELL.value
        signal_strength[upper_breakout] = (close[upper_breakout] - upper_band[upper_breakout]) / upper_band[upper_breakout]
        
        # Lower breakout (potential buy)
        lower_breakout = close < lower_band * (1 - breakout_threshold)
        signals[lower_breakout] = SignalType.BUY.value
        signal_strength[lower_breakout] = (lower_band[lower_breakout] - close[lower_breakout]) / lower_band[lower_breakout]
        
        # Mean reversion signals (price near bands)
        mean_reversion_threshold = 0.1  # 10% from bands
        
        # Near upper band (potential sell)
        near_upper = (close >= upper_band * (1 - mean_reversion_threshold)) & (close <= upper_band)
        signals[near_upper] = SignalType.SELL.value
        signal_strength[near_upper] = 0.5
        
        # Near lower band (potential buy)
        near_lower = (close >= lower_band) & (close <= lower_band * (1 + mean_reversion_threshold))
        signals[near_lower] = SignalType.BUY.value
        signal_strength[near_lower] = 0.5
        
        return IndicatorResult(
            values=np.column_stack([upper_band, ema, lower_band]),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'ema_period': self.ema_period,
                'atr_period': self.atr_period,
                'multiplier': self.multiplier
            },
            metadata={
                'indicator_type': 'Keltner Channel',
                'calculation_method': 'standard',
                'bands': ['upper', 'middle', 'lower']
            }
        )
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          target_returns: np.ndarray, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize Keltner Channel parameters using grid search.
        
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
        
        ema_periods = param_ranges.get('ema_period', [10, 20, 30])
        atr_periods = param_ranges.get('atr_period', [10, 14, 20])
        multipliers = param_ranges.get('multiplier', [1.5, 2.0, 2.5, 3.0])
        
        for ema_period in ema_periods:
            for atr_period in atr_periods:
                for multiplier in multipliers:
                    try:
                        self.ema_period = ema_period
                        self.atr_period = atr_period
                        self.multiplier = multiplier
                        
                        result = self.calculate(high, low, close)
                        
                        # Calculate performance score
                        score = self._calculate_performance_score(result, target_returns)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'ema_period': ema_period,
                                'atr_period': atr_period,
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

class DonchianChannel:
    """
    Donchian Channel
    
    A volatility indicator that shows the highest high and lowest low over a specified period.
    """
    
    def __init__(self, period: int = 20, breakout_threshold: float = 0.02):
        """
        Initialize Donchian Channel indicator.
        
        Args:
            period: Lookback period for high/low calculation
            breakout_threshold: Threshold for breakout signals (as percentage)
        """
        self.period = period
        self.breakout_threshold = breakout_threshold
        
        if period < 2:
            raise ValueError("Period must be at least 2")
        if breakout_threshold <= 0:
            raise ValueError("Breakout threshold must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Donchian Channel.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            
        Returns:
            IndicatorResult with Donchian Channel values and signals
        """
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("All price arrays must have the same length")
        
        if len(high) < self.period:
            raise ValueError(f"Not enough data points. Need at least {self.period}")
        
        # Calculate highest high and lowest low over the period
        highest_high = pd.Series(high).rolling(window=self.period).max().values
        lowest_low = pd.Series(low).rolling(window=self.period).min().values
        
        # Calculate middle line (average of highest high and lowest low)
        middle_line = (highest_high + lowest_low) / 2
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        # Breakout signals
        upper_breakout = close > highest_high * (1 + self.breakout_threshold)
        lower_breakout = close < lowest_low * (1 - self.breakout_threshold)
        
        # Upper breakout (potential buy - continuation)
        signals[upper_breakout] = SignalType.BUY.value
        signal_strength[upper_breakout] = (close[upper_breakout] - highest_high[upper_breakout]) / highest_high[upper_breakout]
        
        # Lower breakout (potential sell - continuation)
        signals[lower_breakout] = SignalType.SELL.value
        signal_strength[lower_breakout] = (lowest_low[lower_breakout] - close[lower_breakout]) / lowest_low[lower_breakout]
        
        # Mean reversion signals (price near channel boundaries)
        mean_reversion_threshold = self.breakout_threshold * 0.5
        
        # Near upper channel (potential sell)
        near_upper = (close >= highest_high * (1 - mean_reversion_threshold)) & (close <= highest_high)
        signals[near_upper] = SignalType.SELL.value
        signal_strength[near_upper] = 0.5
        
        # Near lower channel (potential buy)
        near_lower = (close >= lowest_low) & (close <= lowest_low * (1 + mean_reversion_threshold))
        signals[near_lower] = SignalType.BUY.value
        signal_strength[near_lower] = 0.5
        
        # Channel width for volatility analysis
        channel_width = (highest_high - lowest_low) / middle_line
        
        return IndicatorResult(
            values=np.column_stack([highest_high, middle_line, lowest_low, channel_width]),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'breakout_threshold': self.breakout_threshold
            },
            metadata={
                'indicator_type': 'Donchian Channel',
                'calculation_method': 'standard',
                'bands': ['upper', 'middle', 'lower', 'width']
            }
        )
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          target_returns: np.ndarray, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize Donchian Channel parameters using grid search.
        
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
        
        periods = param_ranges.get('period', [10, 20, 30, 50])
        breakout_thresholds = param_ranges.get('breakout_threshold', [0.01, 0.02, 0.03, 0.05])
        
        for period in periods:
            for breakout_threshold in breakout_thresholds:
                try:
                    self.period = period
                    self.breakout_threshold = breakout_threshold
                    
                    result = self.calculate(high, low, close)
                    
                    # Calculate performance score
                    score = self._calculate_performance_score(result, target_returns)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'period': period,
                            'breakout_threshold': breakout_threshold,
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
        if signal_frequency > 0.3:  # More than 30% signals
            accuracy *= 0.8
        
        return accuracy

# Convenience functions for easy usage
def calculate_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        period: int = 14, overbought: float = -20, oversold: float = -80) -> IndicatorResult:
    """Calculate Williams %R with default parameters."""
    williams_r = WilliamsPercentR(period=period, overbought=overbought, oversold=oversold)
    return williams_r.calculate(high, low, close)

def calculate_keltner_channel(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                             ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> IndicatorResult:
    """Calculate Keltner Channel with default parameters."""
    keltner = KeltnerChannel(ema_period=ema_period, atr_period=atr_period, multiplier=multiplier)
    return keltner.calculate(high, low, close)

def calculate_donchian_channel(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                              period: int = 20, breakout_threshold: float = 0.02) -> IndicatorResult:
    """Calculate Donchian Channel with default parameters."""
    donchian = DonchianChannel(period=period, breakout_threshold=breakout_threshold)
    return donchian.calculate(high, low, close)