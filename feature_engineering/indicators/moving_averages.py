"""
Moving Averages for Trading System

This module implements various moving average indicators including:
- Hull Moving Average (HMA): Smoothed moving average with reduced lag
- Weighted Moving Average (WMA): Moving average with weighted importance
- Additional moving average variants and optimizations
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

class HullMovingAverage:
    """
    Hull Moving Average (HMA) implementation.
    
    HMA is a type of moving average that reduces lag while maintaining smoothness.
    It uses weighted moving averages and square root calculations to achieve this.
    """
    
    def __init__(self, period: int = 20, signal_period: int = 9):
        """
        Initialize Hull Moving Average.
        
        Args:
            period: Lookback period for HMA calculation
            signal_period: Period for signal line calculation
        """
        self.period = period
        self.signal_period = signal_period
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate HMA parameters."""
        if self.period <= 0:
            raise ValueError("HMA period must be positive")
        if self.signal_period <= 0:
            raise ValueError("Signal period must be positive")
        if self.signal_period >= self.period:
            raise ValueError("Signal period must be less than HMA period")
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate Hull Moving Average values.
        
        Args:
            prices: Array of prices (close prices typically)
            
        Returns:
            IndicatorResult with HMA values and signals
        """
        try:
            if len(prices) < self.period:
                raise ValueError(f"Insufficient data for HMA calculation. Need at least {self.period} data points")
            
            # Calculate HMA
            hma_values = self._calculate_hma(prices)
            
            # Calculate signal line
            signal_values = self._calculate_signal_line(hma_values)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(hma_values, signal_values, prices)
            
            # Create metadata
            metadata = {
                "indicator_type": "HullMovingAverage",
                "calculation_method": "weighted_smoothing",
                "data_points_used": len(prices),
                "hma_min": np.nanmin(hma_values),
                "hma_max": np.nanmax(hma_values),
                "hma_mean": np.nanmean(hma_values),
                "signal_min": np.nanmin(signal_values),
                "signal_max": np.nanmax(signal_values),
                "signal_mean": np.nanmean(signal_values)
            }
            
            return IndicatorResult(
                values=hma_values,
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period, "signal_period": self.signal_period},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"HMA calculation failed: {e}")
            raise
    
    def _calculate_hma(self, prices: np.ndarray) -> np.ndarray:
        """Calculate Hull Moving Average values."""
        hma = np.full(len(prices), np.nan)
        
        # Calculate WMA for half period
        half_period = self.period // 2
        wma_half = self._weighted_moving_average(prices, half_period)
        
        # Calculate WMA for full period
        wma_full = self._weighted_moving_average(prices, self.period)
        
        # Calculate raw HMA
        raw_hma = 2 * wma_half - wma_full
        
        # Calculate final HMA using WMA of raw HMA
        sqrt_period = int(np.sqrt(self.period))
        hma = self._weighted_moving_average(raw_hma, sqrt_period)
        
        return hma
    
    def _weighted_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Weighted Moving Average."""
        wma = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            # Create weights (linear weights)
            weights = np.arange(1, period + 1)
            
            # Calculate weighted average
            window_data = data[i - period + 1:i + 1]
            wma[i] = np.sum(window_data * weights) / np.sum(weights)
        
        return wma
    
    def _calculate_signal_line(self, hma_values: np.ndarray) -> np.ndarray:
        """Calculate signal line using WMA of HMA."""
        return self._weighted_moving_average(hma_values, self.signal_period)
    
    def _generate_signals(self, hma_values: np.ndarray, signal_values: np.ndarray, 
                         prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on HMA and signal line."""
        signals = np.full(len(prices), SignalType.HOLD.value)
        signal_strength = np.zeros(len(prices))
        
        for i in range(1, len(prices)):
            if np.isnan(hma_values[i]) or np.isnan(signal_values[i]):
                continue
            
            current_price = prices[i]
            current_hma = hma_values[i]
            current_signal = signal_values[i]
            previous_hma = hma_values[i - 1]
            previous_signal = signal_values[i - 1]
            
            # Calculate price position relative to HMA
            price_position = (current_price - current_hma) / current_hma
            
            # Calculate HMA slope
            hma_slope = (current_hma - previous_hma) / previous_hma if previous_hma != 0 else 0
            
            # Calculate signal line slope
            signal_slope = (current_signal - previous_signal) / previous_signal if previous_signal != 0 else 0
            
            # Generate signals
            if (current_hma > current_signal and hma_slope > 0 and 
                price_position > 0.01):  # Price above HMA and trending up
                signals[i] = SignalType.BUY.value
                signal_strength[i] = min(1.0, abs(price_position) + abs(hma_slope))
            elif (current_hma < current_signal and hma_slope < 0 and 
                  price_position < -0.01):  # Price below HMA and trending down
                signals[i] = SignalType.SELL.value
                signal_strength[i] = min(1.0, abs(price_position) + abs(hma_slope))
            elif (current_hma > current_signal and hma_slope > 0.02):  # Strong uptrend
                signals[i] = SignalType.STRONG_BUY.value
                signal_strength[i] = min(1.0, abs(hma_slope))
            elif (current_hma < current_signal and hma_slope < -0.02):  # Strong downtrend
                signals[i] = SignalType.STRONG_SELL.value
                signal_strength[i] = min(1.0, abs(hma_slope))
        
        return signals, signal_strength
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize HMA parameters."""
        best_params = {}
        best_metric = float('-inf')
        
        # Parameter ranges to test
        periods = [10, 20, 30, 50]
        signal_periods = [5, 9, 14, 20]
        
        for period in periods:
            for signal_period in signal_periods:
                if signal_period >= period:
                    continue
                
                try:
                    self.period = period
                    self.signal_period = signal_period
                    
                    result = self.calculate(prices)
                    metric = self._calculate_performance_metric(result, prices, target_metric)
                    
                    if metric > best_metric:
                        best_metric = metric
                        best_params = {
                            "period": period,
                            "signal_period": signal_period
                        }
                except Exception as e:
                    logger.warning(f"Parameter optimization failed for {period}, {signal_period}: {e}")
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


class WeightedMovingAverage:
    """
    Weighted Moving Average (WMA) implementation.
    
    WMA gives more weight to recent data points, making it more responsive
    to recent price changes compared to Simple Moving Average.
    """
    
    def __init__(self, period: int = 20, weight_type: str = "linear"):
        """
        Initialize Weighted Moving Average.
        
        Args:
            period: Lookback period for WMA calculation
            weight_type: Type of weighting ('linear', 'exponential', 'custom')
        """
        self.period = period
        self.weight_type = weight_type
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate WMA parameters."""
        if self.period <= 0:
            raise ValueError("WMA period must be positive")
        if self.weight_type not in ["linear", "exponential", "custom"]:
            raise ValueError("Weight type must be 'linear', 'exponential', or 'custom'")
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate Weighted Moving Average values.
        
        Args:
            prices: Array of prices (close prices typically)
            
        Returns:
            IndicatorResult with WMA values and signals
        """
        try:
            if len(prices) < self.period:
                raise ValueError(f"Insufficient data for WMA calculation. Need at least {self.period} data points")
            
            # Calculate WMA
            wma_values = self._calculate_wma(prices)
            
            # Calculate signal line (SMA for comparison)
            signal_values = self._calculate_signal_line(prices)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(wma_values, signal_values, prices)
            
            # Create metadata
            metadata = {
                "indicator_type": "WeightedMovingAverage",
                "calculation_method": f"weighted_{self.weight_type}",
                "data_points_used": len(prices),
                "wma_min": np.nanmin(wma_values),
                "wma_max": np.nanmax(wma_values),
                "wma_mean": np.nanmean(wma_values),
                "signal_min": np.nanmin(signal_values),
                "signal_max": np.nanmax(signal_values),
                "signal_mean": np.nanmean(signal_values)
            }
            
            return IndicatorResult(
                values=wma_values,
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period, "weight_type": self.weight_type},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"WMA calculation failed: {e}")
            raise
    
    def _calculate_wma(self, prices: np.ndarray) -> np.ndarray:
        """Calculate Weighted Moving Average values."""
        wma = np.full(len(prices), np.nan)
        
        for i in range(self.period - 1, len(prices)):
            window_data = prices[i - self.period + 1:i + 1]
            
            if self.weight_type == "linear":
                weights = np.arange(1, self.period + 1)
            elif self.weight_type == "exponential":
                weights = np.exp(np.arange(1, self.period + 1) / self.period)
            else:  # custom
                weights = self._custom_weights()
            
            # Calculate weighted average
            wma[i] = np.sum(window_data * weights) / np.sum(weights)
        
        return wma
    
    def _custom_weights(self) -> np.ndarray:
        """Generate custom weights (quadratic weighting)."""
        weights = np.arange(1, self.period + 1) ** 2
        return weights
    
    def _calculate_signal_line(self, prices: np.ndarray) -> np.ndarray:
        """Calculate signal line using Simple Moving Average."""
        signal = np.full(len(prices), np.nan)
        
        for i in range(self.period - 1, len(prices)):
            window_data = prices[i - self.period + 1:i + 1]
            signal[i] = np.mean(window_data)
        
        return signal
    
    def _generate_signals(self, wma_values: np.ndarray, signal_values: np.ndarray, 
                         prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on WMA and signal line."""
        signals = np.full(len(prices), SignalType.HOLD.value)
        signal_strength = np.zeros(len(prices))
        
        for i in range(1, len(prices)):
            if np.isnan(wma_values[i]) or np.isnan(signal_values[i]):
                continue
            
            current_price = prices[i]
            current_wma = wma_values[i]
            current_signal = signal_values[i]
            previous_wma = wma_values[i - 1]
            previous_signal = signal_values[i - 1]
            
            # Calculate price position relative to WMA
            price_position = (current_price - current_wma) / current_wma
            
            # Calculate WMA slope
            wma_slope = (current_wma - previous_wma) / previous_wma if previous_wma != 0 else 0
            
            # Calculate signal line slope
            signal_slope = (current_signal - previous_signal) / previous_signal if previous_signal != 0 else 0
            
            # Generate signals
            if (current_wma > current_signal and wma_slope > 0 and 
                price_position > 0.005):  # Price above WMA and trending up
                signals[i] = SignalType.BUY.value
                signal_strength[i] = min(1.0, abs(price_position) + abs(wma_slope))
            elif (current_wma < current_signal and wma_slope < 0 and 
                  price_position < -0.005):  # Price below WMA and trending down
                signals[i] = SignalType.SELL.value
                signal_strength[i] = min(1.0, abs(price_position) + abs(wma_slope))
            elif (current_wma > current_signal and wma_slope > 0.01):  # Strong uptrend
                signals[i] = SignalType.STRONG_BUY.value
                signal_strength[i] = min(1.0, abs(wma_slope))
            elif (current_wma < current_signal and wma_slope < -0.01):  # Strong downtrend
                signals[i] = SignalType.STRONG_SELL.value
                signal_strength[i] = min(1.0, abs(wma_slope))
        
        return signals, signal_strength
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize WMA parameters."""
        best_params = {}
        best_metric = float('-inf')
        
        # Parameter ranges to test
        periods = [10, 20, 30, 50]
        weight_types = ["linear", "exponential", "custom"]
        
        for period in periods:
            for weight_type in weight_types:
                try:
                    self.period = period
                    self.weight_type = weight_type
                    
                    result = self.calculate(prices)
                    metric = self._calculate_performance_metric(result, prices, target_metric)
                    
                    if metric > best_metric:
                        best_metric = metric
                        best_params = {
                            "period": period,
                            "weight_type": weight_type
                        }
                except Exception as e:
                    logger.warning(f"Parameter optimization failed for {period}, {weight_type}: {e}")
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
def calculate_hma(prices: np.ndarray, period: int = 20, signal_period: int = 9) -> IndicatorResult:
    """Calculate Hull Moving Average."""
    indicator = HullMovingAverage(period, signal_period)
    return indicator.calculate(prices)

def calculate_wma(prices: np.ndarray, period: int = 20, weight_type: str = "linear") -> IndicatorResult:
    """Calculate Weighted Moving Average."""
    indicator = WeightedMovingAverage(period, weight_type)
    return indicator.calculate(prices)