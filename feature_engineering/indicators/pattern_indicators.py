"""
Pattern Indicators for Trading System

This module implements pattern-based indicators including:
- Fractal Indicator: Support and resistance level detection
- Gann HiLo Activator: Trend following system based on Gann principles
- Pattern recognition and signal generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

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

@dataclass
class FractalPoint:
    """Represents a fractal point (support or resistance)."""
    index: int
    price: float
    fractal_type: str  # 'support' or 'resistance'
    strength: float
    confirmed: bool

class FractalIndicator:
    """
    Fractal Indicator implementation.
    
    Identifies support and resistance levels using fractal patterns.
    Fractals are reversal points where price action changes direction.
    """
    
    def __init__(self, lookback_period: int = 5, confirmation_bars: int = 2, 
                 min_strength: float = 0.1):
        """
        Initialize Fractal Indicator.
        
        Args:
            lookback_period: Number of bars to look back for fractal formation
            confirmation_bars: Number of bars required for confirmation
            min_strength: Minimum strength threshold for fractal points
        """
        self.lookback_period = lookback_period
        self.confirmation_bars = confirmation_bars
        self.min_strength = min_strength
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Fractal Indicator parameters."""
        if self.lookback_period < 3:
            raise ValueError("Lookback period must be at least 3")
        if self.confirmation_bars < 1:
            raise ValueError("Confirmation bars must be at least 1")
        if self.min_strength <= 0 or self.min_strength > 1:
            raise ValueError("Min strength must be between 0 and 1")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate fractal support and resistance levels.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            
        Returns:
            IndicatorResult with fractal levels and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close):
                raise ValueError("All price arrays must have the same length")
            
            if len(high) < self.lookback_period * 2 + 1:
                raise ValueError(f"Insufficient data for fractal calculation. Need at least {self.lookback_period * 2 + 1} data points")
            
            # Find fractal points
            support_fractals = self._find_support_fractals(high, low, close)
            resistance_fractals = self._find_resistance_fractals(high, low, close)
            
            # Create fractal levels array
            fractal_levels = self._create_fractal_levels(high, low, close, support_fractals, resistance_fractals)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(fractal_levels, support_fractals, resistance_fractals, close)
            
            # Create metadata
            metadata = {
                "indicator_type": "Fractal",
                "calculation_method": "pattern_recognition",
                "data_points_used": len(close),
                "support_fractals_count": len(support_fractals),
                "resistance_fractals_count": len(resistance_fractals),
                "total_fractals": len(support_fractals) + len(resistance_fractals),
                "support_fractals": [{"index": f.index, "price": f.price, "strength": f.strength} for f in support_fractals],
                "resistance_fractals": [{"index": f.index, "price": f.price, "strength": f.strength} for f in resistance_fractals]
            }
            
            return IndicatorResult(
                values=fractal_levels,
                signals=signals,
                signal_strength=signal_strength,
                parameters={
                    "lookback_period": self.lookback_period,
                    "confirmation_bars": self.confirmation_bars,
                    "min_strength": self.min_strength
                },
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Fractal calculation failed: {e}")
            raise
    
    def _find_support_fractals(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[FractalPoint]:
        """Find support fractal points."""
        support_fractals = []
        
        for i in range(self.lookback_period, len(low) - self.lookback_period):
            # Check if current low is lower than surrounding lows
            current_low = low[i]
            left_lows = low[i - self.lookback_period:i]
            right_lows = low[i + 1:i + self.lookback_period + 1]
            
            if (np.all(current_low <= left_lows) and np.all(current_low <= right_lows)):
                # Calculate strength based on how much lower it is
                left_min = np.min(left_lows)
                right_min = np.min(right_lows)
                strength = min((left_min - current_low) / left_min, (right_min - current_low) / right_min)
                
                if strength >= self.min_strength:
                    # Check confirmation
                    confirmed = self._confirm_support_fractal(high, low, close, i)
                    
                    support_fractals.append(FractalPoint(
                        index=i,
                        price=current_low,
                        fractal_type="support",
                        strength=strength,
                        confirmed=confirmed
                    ))
        
        return support_fractals
    
    def _find_resistance_fractals(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[FractalPoint]:
        """Find resistance fractal points."""
        resistance_fractals = []
        
        for i in range(self.lookback_period, len(high) - self.lookback_period):
            # Check if current high is higher than surrounding highs
            current_high = high[i]
            left_highs = high[i - self.lookback_period:i]
            right_highs = high[i + 1:i + self.lookback_period + 1]
            
            if (np.all(current_high >= left_highs) and np.all(current_high >= right_highs)):
                # Calculate strength based on how much higher it is
                left_max = np.max(left_highs)
                right_max = np.max(right_highs)
                strength = min((current_high - left_max) / left_max, (current_high - right_max) / right_max)
                
                if strength >= self.min_strength:
                    # Check confirmation
                    confirmed = self._confirm_resistance_fractal(high, low, close, i)
                    
                    resistance_fractals.append(FractalPoint(
                        index=i,
                        price=current_high,
                        fractal_type="resistance",
                        strength=strength,
                        confirmed=confirmed
                    ))
        
        return resistance_fractals
    
    def _confirm_support_fractal(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, index: int) -> bool:
        """Confirm support fractal with price action."""
        if index + self.confirmation_bars >= len(close):
            return False
        
        # Check if price bounces up from support level
        support_level = low[index]
        confirmation_prices = close[index + 1:index + self.confirmation_bars + 1]
        
        # Price should stay above or near support level
        return np.all(confirmation_prices >= support_level * 0.98)
    
    def _confirm_resistance_fractal(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, index: int) -> bool:
        """Confirm resistance fractal with price action."""
        if index + self.confirmation_bars >= len(close):
            return False
        
        # Check if price bounces down from resistance level
        resistance_level = high[index]
        confirmation_prices = close[index + 1:index + self.confirmation_bars + 1]
        
        # Price should stay below or near resistance level
        return np.all(confirmation_prices <= resistance_level * 1.02)
    
    def _create_fractal_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                              support_fractals: List[FractalPoint], 
                              resistance_fractals: List[FractalPoint]) -> np.ndarray:
        """Create fractal levels array."""
        fractal_levels = np.full(len(close), np.nan)
        
        # Add support levels
        for fractal in support_fractals:
            if fractal.confirmed:
                fractal_levels[fractal.index] = fractal.price
        
        # Add resistance levels
        for fractal in resistance_fractals:
            if fractal.confirmed:
                fractal_levels[fractal.index] = fractal.price
        
        return fractal_levels
    
    def _generate_signals(self, fractal_levels: np.ndarray, support_fractals: List[FractalPoint],
                         resistance_fractals: List[FractalPoint], close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on fractal levels."""
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        for i in range(len(close)):
            current_price = close[i]
            
            # Find nearest support and resistance
            nearest_support = None
            nearest_resistance = None
            
            # Check support fractals
            for fractal in support_fractals:
                if fractal.index < i and fractal.confirmed:
                    if nearest_support is None or fractal.index > nearest_support.index:
                        nearest_support = fractal
            
            # Check resistance fractals
            for fractal in resistance_fractals:
                if fractal.index < i and fractal.confirmed:
                    if nearest_resistance is None or fractal.index > nearest_resistance.index:
                        nearest_resistance = fractal
            
            # Generate signals
            if nearest_support and nearest_resistance:
                support_level = nearest_support.price
                resistance_level = nearest_resistance.price
                
                # Calculate distance to levels
                support_distance = (current_price - support_level) / support_level
                resistance_distance = (resistance_level - current_price) / resistance_level
                
                # Generate signals based on proximity to levels
                if support_distance <= 0.02:  # Within 2% of support
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = min(1.0, nearest_support.strength)
                elif resistance_distance <= 0.02:  # Within 2% of resistance
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = min(1.0, nearest_resistance.strength)
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize fractal indicator parameters."""
        best_params = {}
        best_metric = float('-inf')
        
        # Parameter ranges to test
        lookback_periods = [3, 5, 7, 10]
        confirmation_bars_list = [1, 2, 3]
        min_strengths = [0.05, 0.1, 0.15, 0.2]
        
        for lookback in lookback_periods:
            for confirmation in confirmation_bars_list:
                for strength in min_strengths:
                    try:
                        self.lookback_period = lookback
                        self.confirmation_bars = confirmation
                        self.min_strength = strength
                        
                        result = self.calculate(high, low, close)
                        metric = self._calculate_performance_metric(result, close, target_metric)
                        
                        if metric > best_metric:
                            best_metric = metric
                            best_params = {
                                "lookback_period": lookback,
                                "confirmation_bars": confirmation,
                                "min_strength": strength
                            }
                    except Exception as e:
                        logger.warning(f"Parameter optimization failed for {lookback}, {confirmation}, {strength}: {e}")
                        continue
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for optimization."""
        if metric == "sharpe_ratio":
            # Calculate returns based on signals
            returns = np.diff(prices) / prices[:-1]
            signal_returns = returns * np.where(result.signals[:-1] == SignalType.BUY.value, 1, 
                                              np.where(result.signals[:-1] == SignalType.SELL.value, -1, 0))
            
            if len(signal_returns) > 0 and np.std(signal_returns) > 0:
                return np.mean(signal_returns) / np.std(signal_returns)
            return 0.0
        else:
            return 0.0


class GannHiLoActivator:
    """
    Gann HiLo Activator implementation.
    
    A trend following system based on Gann principles that identifies
    trend changes and generates buy/sell signals.
    """
    
    def __init__(self, period: int = 14, multiplier: float = 1.0, 
                 activation_threshold: float = 0.02):
        """
        Initialize Gann HiLo Activator.
        
        Args:
            period: Lookback period for high/low calculation
            multiplier: Multiplier for activation levels
            activation_threshold: Threshold for trend activation
        """
        self.period = period
        self.multiplier = multiplier
        self.activation_threshold = activation_threshold
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Gann HiLo Activator parameters."""
        if self.period <= 0:
            raise ValueError("Period must be positive")
        if self.multiplier <= 0:
            raise ValueError("Multiplier must be positive")
        if self.activation_threshold <= 0:
            raise ValueError("Activation threshold must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Gann HiLo Activator values.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            
        Returns:
            IndicatorResult with Gann HiLo values and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close):
                raise ValueError("All price arrays must have the same length")
            
            if len(high) < self.period:
                raise ValueError(f"Insufficient data for Gann HiLo calculation. Need at least {self.period} data points")
            
            # Calculate Gann HiLo levels
            gann_levels = self._calculate_gann_levels(high, low, close)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(gann_levels, close)
            
            # Create metadata
            metadata = {
                "indicator_type": "GannHiLoActivator",
                "calculation_method": "trend_following",
                "data_points_used": len(close),
                "gann_levels_count": len(gann_levels),
                "trend_changes": self._count_trend_changes(signals),
                "activation_events": self._count_activation_events(signals)
            }
            
            return IndicatorResult(
                values=gann_levels,
                signals=signals,
                signal_strength=signal_strength,
                parameters={
                    "period": self.period,
                    "multiplier": self.multiplier,
                    "activation_threshold": self.activation_threshold
                },
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Gann HiLo calculation failed: {e}")
            raise
    
    def _calculate_gann_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Gann HiLo levels."""
        gann_levels = np.full(len(close), np.nan)
        
        for i in range(self.period, len(close)):
            # Calculate highest high and lowest low in period
            period_high = np.max(high[i - self.period:i])
            period_low = np.min(low[i - self.period:i])
            
            # Calculate Gann level
            gann_level = (period_high + period_low) / 2
            
            # Apply multiplier
            gann_levels[i] = gann_level * self.multiplier
        
        return gann_levels
    
    def _generate_signals(self, gann_levels: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on Gann HiLo levels."""
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if np.isnan(gann_levels[i]):
                continue
            
            current_price = close[i]
            previous_price = close[i - 1]
            gann_level = gann_levels[i]
            
            # Calculate price change percentage
            price_change = abs(current_price - gann_level) / gann_level
            
            # Generate signals based on price position relative to Gann level
            if current_price > gann_level and price_change >= self.activation_threshold:
                if previous_price <= gann_level:
                    # Bullish activation
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = min(1.0, price_change / self.activation_threshold)
                else:
                    # Continue bullish trend
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = min(1.0, price_change / self.activation_threshold)
            
            elif current_price < gann_level and price_change >= self.activation_threshold:
                if previous_price >= gann_level:
                    # Bearish activation
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = min(1.0, price_change / self.activation_threshold)
                else:
                    # Continue bearish trend
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = min(1.0, price_change / self.activation_threshold)
        
        return signals, signal_strength
    
    def _count_trend_changes(self, signals: np.ndarray) -> int:
        """Count trend change events."""
        trend_changes = 0
        for i in range(1, len(signals)):
            if (signals[i] in [SignalType.BUY.value, SignalType.STRONG_BUY.value] and
                signals[i-1] in [SignalType.SELL.value, SignalType.STRONG_SELL.value]):
                trend_changes += 1
            elif (signals[i] in [SignalType.SELL.value, SignalType.STRONG_SELL.value] and
                  signals[i-1] in [SignalType.BUY.value, SignalType.STRONG_BUY.value]):
                trend_changes += 1
        return trend_changes
    
    def _count_activation_events(self, signals: np.ndarray) -> int:
        """Count activation events."""
        activations = 0
        for i in range(1, len(signals)):
            if (signals[i] in [SignalType.BUY.value, SignalType.SELL.value] and
                signals[i-1] == SignalType.HOLD.value):
                activations += 1
        return activations
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize Gann HiLo Activator parameters."""
        best_params = {}
        best_metric = float('-inf')
        
        # Parameter ranges to test
        periods = [10, 14, 20, 30]
        multipliers = [0.8, 1.0, 1.2, 1.5]
        thresholds = [0.01, 0.02, 0.03, 0.05]
        
        for period in periods:
            for multiplier in multipliers:
                for threshold in thresholds:
                    try:
                        self.period = period
                        self.multiplier = multiplier
                        self.activation_threshold = threshold
                        
                        result = self.calculate(high, low, close)
                        metric = self._calculate_performance_metric(result, close, target_metric)
                        
                        if metric > best_metric:
                            best_metric = metric
                            best_params = {
                                "period": period,
                                "multiplier": multiplier,
                                "activation_threshold": threshold
                            }
                    except Exception as e:
                        logger.warning(f"Parameter optimization failed for {period}, {multiplier}, {threshold}: {e}")
                        continue
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for optimization."""
        if metric == "sharpe_ratio":
            # Calculate returns based on signals
            returns = np.diff(prices) / prices[:-1]
            signal_returns = returns * np.where(result.signals[:-1] == SignalType.BUY.value, 1, 
                                              np.where(result.signals[:-1] == SignalType.SELL.value, -1, 0))
            
            if len(signal_returns) > 0 and np.std(signal_returns) > 0:
                return np.mean(signal_returns) / np.std(signal_returns)
            return 0.0
        else:
            return 0.0


# Convenience functions
def calculate_fractal(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     lookback_period: int = 5, confirmation_bars: int = 2, 
                     min_strength: float = 0.1) -> IndicatorResult:
    """Calculate Fractal Indicator."""
    indicator = FractalIndicator(lookback_period, confirmation_bars, min_strength)
    return indicator.calculate(high, low, close)

def calculate_gann_hilo(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       period: int = 14, multiplier: float = 1.0, 
                       activation_threshold: float = 0.02) -> IndicatorResult:
    """Calculate Gann HiLo Activator."""
    indicator = GannHiLoActivator(period, multiplier, activation_threshold)
    return indicator.calculate(high, low, close)