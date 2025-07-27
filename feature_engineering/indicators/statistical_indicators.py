"""
Statistical Indicators for Trading System

This module implements statistical and mathematical indicators including:
- Z-Score: Statistical measure of price deviation from mean
- Fibonacci Retracements: Support and resistance levels based on Fibonacci ratios
- Additional statistical analysis tools
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
class FibonacciLevel:
    """Represents a Fibonacci retracement level."""
    ratio: float
    price: float
    level_type: str  # 'retracement' or 'extension'
    strength: float

class ZScoreIndicator:
    """
    Z-Score indicator implementation.
    
    Z-Score measures how many standard deviations a price is from its mean.
    Useful for identifying overbought/oversold conditions and mean reversion opportunities.
    """
    
    def __init__(self, period: int = 20, threshold: float = 2.0, 
                 mean_reversion_threshold: float = 1.5):
        """
        Initialize Z-Score indicator.
        
        Args:
            period: Lookback period for mean and standard deviation calculation
            threshold: Z-Score threshold for extreme values
            mean_reversion_threshold: Threshold for mean reversion signals
        """
        self.period = period
        self.threshold = threshold
        self.mean_reversion_threshold = mean_reversion_threshold
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Z-Score parameters."""
        if self.period <= 0:
            raise ValueError("Z-Score period must be positive")
        if self.threshold <= 0:
            raise ValueError("Z-Score threshold must be positive")
        if self.mean_reversion_threshold <= 0:
            raise ValueError("Mean reversion threshold must be positive")
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate Z-Score values.
        
        Args:
            prices: Array of prices (close prices typically)
            
        Returns:
            IndicatorResult with Z-Score values and signals
        """
        try:
            if len(prices) < self.period:
                raise ValueError(f"Insufficient data for Z-Score calculation. Need at least {self.period} data points")
            
            # Calculate Z-Score
            zscore_values = self._calculate_zscore(prices)
            
            # Calculate rolling statistics
            mean_values = self._calculate_rolling_mean(prices)
            std_values = self._calculate_rolling_std(prices)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(zscore_values, prices, mean_values)
            
            # Create metadata
            metadata = {
                "indicator_type": "ZScore",
                "calculation_method": "statistical_deviation",
                "data_points_used": len(prices),
                "zscore_min": np.nanmin(zscore_values),
                "zscore_max": np.nanmax(zscore_values),
                "zscore_mean": np.nanmean(zscore_values),
                "extreme_values_count": np.sum(np.abs(zscore_values) > self.threshold),
                "mean_reversion_signals": np.sum(np.abs(zscore_values) > self.mean_reversion_threshold)
            }
            
            return IndicatorResult(
                values=zscore_values,
                signals=signals,
                signal_strength=signal_strength,
                parameters={
                    "period": self.period,
                    "threshold": self.threshold,
                    "mean_reversion_threshold": self.mean_reversion_threshold
                },
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Z-Score calculation failed: {e}")
            raise
    
    def _calculate_zscore(self, prices: np.ndarray) -> np.ndarray:
        """Calculate Z-Score values."""
        zscore = np.full(len(prices), np.nan)
        
        for i in range(self.period - 1, len(prices)):
            window_data = prices[i - self.period + 1:i + 1]
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            
            if std_val > 0:
                zscore[i] = (prices[i] - mean_val) / std_val
            else:
                zscore[i] = 0.0
        
        return zscore
    
    def _calculate_rolling_mean(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling mean values."""
        mean_values = np.full(len(prices), np.nan)
        
        for i in range(self.period - 1, len(prices)):
            window_data = prices[i - self.period + 1:i + 1]
            mean_values[i] = np.mean(window_data)
        
        return mean_values
    
    def _calculate_rolling_std(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling standard deviation values."""
        std_values = np.full(len(prices), np.nan)
        
        for i in range(self.period - 1, len(prices)):
            window_data = prices[i - self.period + 1:i + 1]
            std_values[i] = np.std(window_data)
        
        return std_values
    
    def _generate_signals(self, zscore_values: np.ndarray, prices: np.ndarray, 
                         mean_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on Z-Score values."""
        signals = np.full(len(prices), SignalType.HOLD.value)
        signal_strength = np.zeros(len(prices))
        
        for i in range(1, len(prices)):
            if np.isnan(zscore_values[i]):
                continue
            
            current_zscore = zscore_values[i]
            previous_zscore = zscore_values[i - 1]
            current_price = prices[i]
            current_mean = mean_values[i]
            
            # Calculate price deviation from mean
            price_deviation = (current_price - current_mean) / current_mean if current_mean != 0 else 0
            
            # Generate signals based on Z-Score thresholds
            if current_zscore < -self.threshold:
                # Oversold condition - potential buy signal
                if previous_zscore >= -self.threshold:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = min(1.0, abs(current_zscore) / self.threshold)
                elif current_zscore < -self.mean_reversion_threshold:
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = min(1.0, abs(current_zscore) / self.threshold)
            
            elif current_zscore > self.threshold:
                # Overbought condition - potential sell signal
                if previous_zscore <= self.threshold:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = min(1.0, abs(current_zscore) / self.threshold)
                elif current_zscore > self.mean_reversion_threshold:
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = min(1.0, abs(current_zscore) / self.threshold)
            
            # Mean reversion signals
            elif abs(current_zscore) > self.mean_reversion_threshold:
                if current_zscore > 0 and previous_zscore <= 0:
                    # Moving from below mean to above mean
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = min(1.0, abs(current_zscore) / self.mean_reversion_threshold)
                elif current_zscore < 0 and previous_zscore >= 0:
                    # Moving from above mean to below mean
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = min(1.0, abs(current_zscore) / self.mean_reversion_threshold)
        
        return signals, signal_strength
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize Z-Score parameters."""
        best_params = {}
        best_metric = float('-inf')
        
        # Parameter ranges to test
        periods = [10, 20, 30, 50]
        thresholds = [1.5, 2.0, 2.5, 3.0]
        mean_reversion_thresholds = [1.0, 1.5, 2.0, 2.5]
        
        for period in periods:
            for threshold in thresholds:
                for mean_reversion_threshold in mean_reversion_thresholds:
                    if mean_reversion_threshold >= threshold:
                        continue
                    
                    try:
                        self.period = period
                        self.threshold = threshold
                        self.mean_reversion_threshold = mean_reversion_threshold
                        
                        result = self.calculate(prices)
                        metric = self._calculate_performance_metric(result, prices, target_metric)
                        
                        if metric > best_metric:
                            best_metric = metric
                            best_params = {
                                "period": period,
                                "threshold": threshold,
                                "mean_reversion_threshold": mean_reversion_threshold
                            }
                    except Exception as e:
                        logger.warning(f"Parameter optimization failed for {period}, {threshold}, {mean_reversion_threshold}: {e}")
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


class FibonacciRetracements:
    """
    Fibonacci Retracements implementation.
    
    Fibonacci retracements are horizontal lines that indicate where support and resistance
    are likely to occur based on Fibonacci ratios.
    """
    
    def __init__(self, swing_high_threshold: float = 0.05, swing_low_threshold: float = 0.05,
                 retracement_levels: List[float] = None):
        """
        Initialize Fibonacci Retracements.
        
        Args:
            swing_high_threshold: Threshold for identifying swing highs
            swing_low_threshold: Threshold for identifying swing lows
            retracement_levels: List of Fibonacci retracement levels (default: standard levels)
        """
        self.swing_high_threshold = swing_high_threshold
        self.swing_low_threshold = swing_low_threshold
        
        # Standard Fibonacci retracement levels
        if retracement_levels is None:
            self.retracement_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        else:
            self.retracement_levels = retracement_levels
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Fibonacci Retracements parameters."""
        if self.swing_high_threshold <= 0:
            raise ValueError("Swing high threshold must be positive")
        if self.swing_low_threshold <= 0:
            raise ValueError("Swing low threshold must be positive")
        if not all(0 <= level <= 1 for level in self.retracement_levels):
            raise ValueError("All retracement levels must be between 0 and 1")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            
        Returns:
            IndicatorResult with Fibonacci levels and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close):
                raise ValueError("All price arrays must have the same length")
            
            if len(high) < 20:
                raise ValueError("Insufficient data for Fibonacci calculation. Need at least 20 data points")
            
            # Find swing points
            swing_highs = self._find_swing_highs(high, low, close)
            swing_lows = self._find_swing_lows(high, low, close)
            
            # Calculate Fibonacci levels
            fib_levels = self._calculate_fibonacci_levels(high, low, close, swing_highs, swing_lows)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(fib_levels, close, swing_highs, swing_lows)
            
            # Create metadata
            metadata = {
                "indicator_type": "FibonacciRetracements",
                "calculation_method": "fibonacci_ratios",
                "data_points_used": len(close),
                "swing_highs_count": len(swing_highs),
                "swing_lows_count": len(swing_lows),
                "retracement_levels": self.retracement_levels,
                "fib_levels_count": len(fib_levels)
            }
            
            return IndicatorResult(
                values=fib_levels,
                signals=signals,
                signal_strength=signal_strength,
                parameters={
                    "swing_high_threshold": self.swing_high_threshold,
                    "swing_low_threshold": self.swing_low_threshold,
                    "retracement_levels": self.retracement_levels
                },
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Fibonacci calculation failed: {e}")
            raise
    
    def _find_swing_highs(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[Dict[str, Any]]:
        """Find swing high points."""
        swing_highs = []
        
        for i in range(5, len(high) - 5):
            # Check if current high is a swing high
            current_high = high[i]
            left_highs = high[i - 5:i]
            right_highs = high[i + 1:i + 6]
            
            if (np.all(current_high >= left_highs) and np.all(current_high >= right_highs)):
                # Calculate strength
                left_max = np.max(left_highs)
                right_max = np.max(right_highs)
                strength = min((current_high - left_max) / left_max, (current_high - right_max) / right_max)
                
                if strength >= self.swing_high_threshold:
                    swing_highs.append({
                        "index": i,
                        "price": current_high,
                        "strength": strength,
                        "type": "high"
                    })
        
        return swing_highs
    
    def _find_swing_lows(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[Dict[str, Any]]:
        """Find swing low points."""
        swing_lows = []
        
        for i in range(5, len(low) - 5):
            # Check if current low is a swing low
            current_low = low[i]
            left_lows = low[i - 5:i]
            right_lows = low[i + 1:i + 6]
            
            if (np.all(current_low <= left_lows) and np.all(current_low <= right_lows)):
                # Calculate strength
                left_min = np.min(left_lows)
                right_min = np.min(right_lows)
                strength = min((left_min - current_low) / left_min, (right_min - current_low) / right_min)
                
                if strength >= self.swing_low_threshold:
                    swing_lows.append({
                        "index": i,
                        "price": current_low,
                        "strength": strength,
                        "type": "low"
                    })
        
        return swing_lows
    
    def _calculate_fibonacci_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                   swing_highs: List[Dict[str, Any]], 
                                   swing_lows: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate Fibonacci retracement levels."""
        fib_levels = np.full(len(close), np.nan)
        
        # Combine swing points and sort by index
        swing_points = swing_highs + swing_lows
        swing_points.sort(key=lambda x: x["index"])
        
        # Calculate Fibonacci levels for each swing
        for i in range(len(swing_points) - 1):
            current_swing = swing_points[i]
            next_swing = swing_points[i + 1]
            
            start_price = current_swing["price"]
            end_price = next_swing["price"]
            price_range = end_price - start_price
            
            # Calculate Fibonacci levels
            for level in self.retracement_levels:
                fib_price = start_price + (price_range * level)
                
                # Find the closest price point to this Fibonacci level
                closest_index = self._find_closest_price_index(close, fib_price, 
                                                             current_swing["index"], 
                                                             next_swing["index"])
                
                if closest_index is not None:
                    fib_levels[closest_index] = fib_price
        
        return fib_levels
    
    def _find_closest_price_index(self, prices: np.ndarray, target_price: float, 
                                 start_index: int, end_index: int) -> Optional[int]:
        """Find the index of the price closest to the target Fibonacci level."""
        if start_index >= end_index:
            return None
        
        min_distance = float('inf')
        closest_index = None
        
        for i in range(start_index, min(end_index, len(prices))):
            distance = abs(prices[i] - target_price)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index
    
    def _generate_signals(self, fib_levels: np.ndarray, close: np.ndarray,
                         swing_highs: List[Dict[str, Any]], 
                         swing_lows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on Fibonacci levels."""
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        for i in range(len(close)):
            current_price = close[i]
            
            # Find nearest Fibonacci level
            nearest_level = None
            min_distance = float('inf')
            
            for j in range(len(close)):
                if not np.isnan(fib_levels[j]):
                    distance = abs(current_price - fib_levels[j])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_level = {"index": j, "price": fib_levels[j]}
            
            if nearest_level is not None:
                # Calculate distance to Fibonacci level
                distance_pct = abs(current_price - nearest_level["price"]) / nearest_level["price"]
                
                # Generate signals based on proximity to Fibonacci levels
                if distance_pct <= 0.01:  # Within 1% of Fibonacci level
                    # Determine if it's support or resistance
                    if current_price > nearest_level["price"]:
                        # Price above level - potential resistance
                        signals[i] = SignalType.SELL.value
                        signal_strength[i] = min(1.0, 1.0 - distance_pct)
                    else:
                        # Price below level - potential support
                        signals[i] = SignalType.BUY.value
                        signal_strength[i] = min(1.0, 1.0 - distance_pct)
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize Fibonacci Retracements parameters."""
        best_params = {}
        best_metric = float('-inf')
        
        # Parameter ranges to test
        swing_high_thresholds = [0.03, 0.05, 0.07, 0.10]
        swing_low_thresholds = [0.03, 0.05, 0.07, 0.10]
        
        for swing_high_threshold in swing_high_thresholds:
            for swing_low_threshold in swing_low_thresholds:
                try:
                    self.swing_high_threshold = swing_high_threshold
                    self.swing_low_threshold = swing_low_threshold
                    
                    result = self.calculate(high, low, close)
                    metric = self._calculate_performance_metric(result, close, target_metric)
                    
                    if metric > best_metric:
                        best_metric = metric
                        best_params = {
                            "swing_high_threshold": swing_high_threshold,
                            "swing_low_threshold": swing_low_threshold
                        }
                except Exception as e:
                    logger.warning(f"Parameter optimization failed for {swing_high_threshold}, {swing_low_threshold}: {e}")
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
def calculate_zscore(prices: np.ndarray, period: int = 20, threshold: float = 2.0, 
                    mean_reversion_threshold: float = 1.5) -> IndicatorResult:
    """Calculate Z-Score indicator."""
    indicator = ZScoreIndicator(period, threshold, mean_reversion_threshold)
    return indicator.calculate(prices)

def calculate_fibonacci_retracements(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                   swing_high_threshold: float = 0.05, 
                                   swing_low_threshold: float = 0.05,
                                   retracement_levels: List[float] = None) -> IndicatorResult:
    """Calculate Fibonacci Retracements."""
    indicator = FibonacciRetracements(swing_high_threshold, swing_low_threshold, retracement_levels)
    return indicator.calculate(high, low, close)