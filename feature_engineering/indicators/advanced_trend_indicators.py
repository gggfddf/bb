"""
Advanced Trend Indicators Module

Implements advanced trend-based technical indicators:
- Ichimoku Cloud
- Parabolic SAR
- ADX (Average Directional Index)

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

class IchimokuCloud:
    """
    Ichimoku Cloud implementation.
    
    Ichimoku Cloud is a comprehensive indicator that shows support/resistance,
    momentum, and trend direction using multiple components.
    """
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, 
                 senkou_span_b_period: int = 52, displacement: int = 26):
        """
        Initialize Ichimoku Cloud indicator.
        
        Args:
            tenkan_period: Tenkan-sen period (default: 9)
            kijun_period: Kijun-sen period (default: 26)
            senkou_span_b_period: Senkou Span B period (default: 52)
            displacement: Displacement period (default: 26)
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.displacement = displacement
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Ichimoku Cloud parameters."""
        if any(p <= 0 for p in [self.tenkan_period, self.kijun_period, self.senkou_span_b_period, self.displacement]):
            raise ValueError("All Ichimoku periods must be positive")
        if self.tenkan_period >= self.kijun_period:
            raise ValueError("Tenkan period must be less than Kijun period")
        if self.kijun_period >= self.senkou_span_b_period:
            raise ValueError("Kijun period must be less than Senkou Span B period")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Ichimoku Cloud values.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            
        Returns:
            IndicatorResult with Ichimoku Cloud values and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close):
                raise ValueError("All price arrays must have the same length")
            
            min_period = max(self.tenkan_period, self.kijun_period, self.senkou_span_b_period)
            if len(high) < min_period + self.displacement:
                raise ValueError(f"Insufficient data for Ichimoku calculation. Need at least {min_period + self.displacement} data points")
            
            # Calculate Tenkan-sen (Conversion Line)
            tenkan_sen = self._calculate_midpoint(high, low, self.tenkan_period)
            
            # Calculate Kijun-sen (Base Line)
            kijun_sen = self._calculate_midpoint(high, low, self.kijun_period)
            
            # Calculate Senkou Span A (Leading Span A)
            senkou_span_a = self._shift_forward((tenkan_sen + kijun_sen) / 2, self.displacement)
            
            # Calculate Senkou Span B (Leading Span B)
            senkou_span_b = self._shift_forward(self._calculate_midpoint(high, low, self.senkou_span_b_period), self.displacement)
            
            # Calculate Chikou Span (Lagging Span)
            chikou_span = self._shift_backward(close, self.displacement)
            
            # Calculate cloud thickness and position
            cloud_thickness = senkou_span_a - senkou_span_b
            price_position = np.where(close > senkou_span_a, 1, np.where(close < senkou_span_b, -1, 0))
            
            # Generate signals
            signals, signal_strength = self._generate_signals(close, tenkan_sen, kijun_sen, 
                                                             senkou_span_a, senkou_span_b, 
                                                             chikou_span, price_position)
            
            # Create metadata
            metadata = {
                "indicator_type": "Ichimoku_Cloud",
                "calculation_method": "midpoint_averages",
                "data_points_used": len(close),
                "tenkan_min": np.nanmin(tenkan_sen),
                "tenkan_max": np.nanmax(tenkan_sen),
                "kijun_min": np.nanmin(kijun_sen),
                "kijun_max": np.nanmax(kijun_sen),
                "cloud_thickness_min": np.nanmin(cloud_thickness),
                "cloud_thickness_max": np.nanmax(cloud_thickness)
            }
            
            return IndicatorResult(
                values=np.column_stack([tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, 
                                      chikou_span, cloud_thickness, price_position]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={
                    "tenkan_period": self.tenkan_period,
                    "kijun_period": self.kijun_period,
                    "senkou_span_b_period": self.senkou_span_b_period,
                    "displacement": self.displacement
                },
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Ichimoku Cloud calculation failed: {e}")
            raise
    
    def _calculate_midpoint(self, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """Calculate midpoint (highest high + lowest low) / 2 for given period."""
        midpoint = np.full(len(high), np.nan)
        
        for i in range(period - 1, len(high)):
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            midpoint[i] = (highest_high + lowest_low) / 2
        
        return midpoint
    
    def _shift_forward(self, data: np.ndarray, periods: int) -> np.ndarray:
        """Shift data forward by specified periods."""
        shifted = np.full(len(data), np.nan)
        shifted[periods:] = data[:-periods]
        return shifted
    
    def _shift_backward(self, data: np.ndarray, periods: int) -> np.ndarray:
        """Shift data backward by specified periods."""
        shifted = np.full(len(data), np.nan)
        shifted[:-periods] = data[periods:]
        return shifted
    
    def _generate_signals(self, close: np.ndarray, tenkan_sen: np.ndarray, kijun_sen: np.ndarray,
                         senkou_span_a: np.ndarray, senkou_span_b: np.ndarray,
                         chikou_span: np.ndarray, price_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on Ichimoku Cloud values."""
        signals = np.full(len(close), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]):
                continue
            
            # Tenkan/Kijun crossover signals
            if (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1] and
                price_position[i] == 1):  # Bullish crossover above cloud
                signals[i] = SignalType.STRONG_BUY.value
                signal_strength[i] = 1.0
            elif (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1] and
                  price_position[i] == -1):  # Bearish crossover below cloud
                signals[i] = SignalType.STRONG_SELL.value
                signal_strength[i] = 1.0
            elif (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1] and
                  price_position[i] == 0):  # Bullish crossover in cloud
                signals[i] = SignalType.BUY.value
                signal_strength[i] = 0.7
            elif (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1] and
                  price_position[i] == 0):  # Bearish crossover in cloud
                signals[i] = SignalType.SELL.value
                signal_strength[i] = 0.7
            
            # Cloud breakout signals
            if i >= 1:
                if (price_position[i] == 1 and price_position[i-1] == 0):  # Breakout above cloud
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                elif (price_position[i] == -1 and price_position[i-1] == 0):  # Breakdown below cloud
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
            
            # Chikou Span signals
            if not np.isnan(chikou_span[i]):
                if (chikou_span[i] > close[i] and chikou_span[i-1] <= close[i-1] and
                    price_position[i] == 1):  # Chikou breakout above price
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.6
                elif (chikou_span[i] < close[i] and chikou_span[i-1] >= close[i-1] and
                      price_position[i] == -1):  # Chikou breakdown below price
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.6
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize Ichimoku Cloud parameters using grid search.
        
        Args:
            high: High price data for optimization
            low: Low price data for optimization
            close: Close price data for optimization
            target_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = float('-inf')
        
        # Parameter ranges for optimization
        tenkan_periods = [7, 9, 11, 13]
        kijun_periods = [21, 26, 30, 34]
        senkou_periods = [45, 52, 60]
        displacements = [20, 26, 30]
        
        for tenkan in tenkan_periods:
            for kijun in kijun_periods:
                if tenkan < kijun:
                    for senkou in senkou_periods:
                        if kijun < senkou:
                            for displacement in displacements:
                                try:
                                    # Create Ichimoku instance with test parameters
                                    test_ichimoku = IchimokuCloud(
                                        tenkan_period=tenkan,
                                        kijun_period=kijun,
                                        senkou_span_b_period=senkou,
                                        displacement=displacement
                                    )
                                    result = test_ichimoku.calculate(high, low, close)
                                    
                                    # Calculate performance metric
                                    score = self._calculate_performance_metric(result, close, target_metric)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_params = {
                                            "tenkan_period": tenkan,
                                            "kijun_period": kijun,
                                            "senkou_span_b_period": senkou,
                                            "displacement": displacement,
                                            "score": score
                                        }
                                        
                                except Exception as e:
                                    logger.warning(f"Parameter combination failed: tenkan={tenkan}, kijun={kijun}, senkou={senkou}, displacement={displacement}, error={e}")
                                    continue
        
        if best_params is None:
            logger.warning("No valid parameter combination found, using defaults")
            best_params = {
                "tenkan_period": 9, "kijun_period": 26, "senkou_span_b_period": 52,
                "displacement": 26, "score": 0
            }
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for parameter optimization."""
        try:
            # Simple backtest simulation
            signals = result.signals
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate strategy returns
            strategy_returns = np.zeros(len(returns))
            position = 0
            
            for i, signal in enumerate(signals[1:]):  # Skip first signal
                if signal == SignalType.BUY.value or signal == SignalType.STRONG_BUY.value:
                    position = 1
                elif signal == SignalType.SELL.value or signal == SignalType.STRONG_SELL.value:
                    position = 0
                
                strategy_returns[i] = position * returns[i]
            
            # Calculate metrics
            if metric == "sharpe_ratio":
                if np.std(strategy_returns) == 0:
                    return 0
                return np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            elif metric == "total_return":
                return np.sum(strategy_returns)
            elif metric == "max_drawdown":
                cumulative = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                return np.min(drawdown)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Performance metric calculation failed: {e}")
            return 0

class ParabolicSAR:
    """
    Parabolic SAR (Stop and Reverse) implementation.
    
    Parabolic SAR is a trend-following indicator that provides
    stop-loss levels and trend reversal signals.
    """
    
    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2):
        """
        Initialize Parabolic SAR indicator.
        
        Args:
            acceleration: Acceleration factor (default: 0.02)
            maximum: Maximum acceleration (default: 0.2)
        """
        self.acceleration = acceleration
        self.maximum = maximum
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Parabolic SAR parameters."""
        if self.acceleration <= 0 or self.maximum <= 0:
            raise ValueError("Acceleration and maximum must be positive")
        if self.acceleration >= self.maximum:
            raise ValueError("Acceleration must be less than maximum")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Parabolic SAR values.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            
        Returns:
            IndicatorResult with Parabolic SAR values and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close):
                raise ValueError("All price arrays must have the same length")
            
            if len(high) < 2:
                raise ValueError("Insufficient data for Parabolic SAR calculation")
            
            # Calculate Parabolic SAR
            sar_values = self._calculate_parabolic_sar(high, low, close)
            
            # Determine trend direction
            trend = np.where(close > sar_values, 1, -1)
            
            # Calculate SAR distance from price
            sar_distance = (close - sar_values) / close
            
            # Generate signals
            signals, signal_strength = self._generate_signals(close, sar_values, trend, sar_distance)
            
            # Create metadata
            metadata = {
                "indicator_type": "Parabolic_SAR",
                "calculation_method": "parabolic_stop_and_reverse",
                "data_points_used": len(close),
                "sar_min": np.nanmin(sar_values),
                "sar_max": np.nanmax(sar_values),
                "distance_min": np.nanmin(sar_distance),
                "distance_max": np.nanmax(sar_distance)
            }
            
            return IndicatorResult(
                values=np.column_stack([sar_values, trend, sar_distance]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={"acceleration": self.acceleration, "maximum": self.maximum},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Parabolic SAR calculation failed: {e}")
            raise
    
    def _calculate_parabolic_sar(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Parabolic SAR values."""
        sar = np.full(len(close), np.nan)
        trend = np.full(len(close), 0)
        af = np.full(len(close), self.acceleration)
        ep = np.full(len(close), np.nan)
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1  # Assume uptrend initially
        ep[0] = high[0]
        
        for i in range(1, len(close)):
            # Update extreme point
            if trend[i-1] == 1:  # Uptrend
                ep[i] = max(high[i], ep[i-1])
            else:  # Downtrend
                ep[i] = min(low[i], ep[i-1])
            
            # Calculate SAR
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i] - sar[i-1])
                # Check for trend reversal
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i]
                    ep[i] = low[i]
                    af[i] = self.acceleration
                else:
                    trend[i] = 1
                    # Update acceleration factor
                    if high[i] > ep[i-1]:
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        af[i] = af[i-1]
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i] - sar[i-1])
                # Check for trend reversal
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i]
                    ep[i] = high[i]
                    af[i] = self.acceleration
                else:
                    trend[i] = -1
                    # Update acceleration factor
                    if low[i] < ep[i-1]:
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        af[i] = af[i-1]
            
            # Ensure SAR doesn't penetrate previous bars
            if trend[i] == 1:  # Uptrend
                sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            else:  # Downtrend
                sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
        
        return sar
    
    def _generate_signals(self, close: np.ndarray, sar: np.ndarray, 
                         trend: np.ndarray, sar_distance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on Parabolic SAR values."""
        signals = np.full(len(close), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if np.isnan(sar[i]):
                continue
            
            # Trend reversal signals
            if trend[i] == 1 and trend[i-1] == -1:  # Bullish reversal
                signals[i] = SignalType.STRONG_BUY.value
                signal_strength[i] = 1.0
            elif trend[i] == -1 and trend[i-1] == 1:  # Bearish reversal
                signals[i] = SignalType.STRONG_SELL.value
                signal_strength[i] = 1.0
            
            # Trend continuation signals
            elif trend[i] == 1 and trend[i-1] == 1:  # Uptrend continuation
                if sar_distance[i] > 0.02:  # Strong uptrend
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.7
            elif trend[i] == -1 and trend[i-1] == -1:  # Downtrend continuation
                if sar_distance[i] < -0.02:  # Strong downtrend
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.7
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize Parabolic SAR parameters using grid search.
        
        Args:
            high: High price data for optimization
            low: Low price data for optimization
            close: Close price data for optimization
            target_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = float('-inf')
        
        # Parameter ranges for optimization
        accelerations = [0.01, 0.02, 0.03, 0.04]
        maximums = [0.1, 0.2, 0.3, 0.4]
        
        for acceleration in accelerations:
            for maximum in maximums:
                if acceleration < maximum:
                    try:
                        # Create Parabolic SAR instance with test parameters
                        test_sar = ParabolicSAR(acceleration=acceleration, maximum=maximum)
                        result = test_sar.calculate(high, low, close)
                        
                        # Calculate performance metric
                        score = self._calculate_performance_metric(result, close, target_metric)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "acceleration": acceleration,
                                "maximum": maximum,
                                "score": score
                            }
                            
                    except Exception as e:
                        logger.warning(f"Parameter combination failed: acceleration={acceleration}, maximum={maximum}, error={e}")
                        continue
        
        if best_params is None:
            logger.warning("No valid parameter combination found, using defaults")
            best_params = {"acceleration": 0.02, "maximum": 0.2, "score": 0}
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for parameter optimization."""
        try:
            # Simple backtest simulation
            signals = result.signals
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate strategy returns
            strategy_returns = np.zeros(len(returns))
            position = 0
            
            for i, signal in enumerate(signals[1:]):  # Skip first signal
                if signal == SignalType.BUY.value or signal == SignalType.STRONG_BUY.value:
                    position = 1
                elif signal == SignalType.SELL.value or signal == SignalType.STRONG_SELL.value:
                    position = 0
                
                strategy_returns[i] = position * returns[i]
            
            # Calculate metrics
            if metric == "sharpe_ratio":
                if np.std(strategy_returns) == 0:
                    return 0
                return np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            elif metric == "total_return":
                return np.sum(strategy_returns)
            elif metric == "max_drawdown":
                cumulative = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                return np.min(drawdown)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Performance metric calculation failed: {e}")
            return 0

class ADX:
    """
    Average Directional Index (ADX) implementation.
    
    ADX measures the strength of a trend regardless of its direction.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ADX indicator.
        
        Args:
            period: Lookback period for ADX calculation (default: 14)
        """
        self.period = period
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate ADX parameters."""
        if self.period <= 0:
            raise ValueError("ADX period must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate ADX values.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            
        Returns:
            IndicatorResult with ADX values and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close):
                raise ValueError("All price arrays must have the same length")
            
            if len(high) < self.period + 1:
                raise ValueError(f"Insufficient data for ADX calculation. Need at least {self.period + 1} data points")
            
            # Calculate True Range
            tr = self._calculate_true_range(high, low, close)
            
            # Calculate Directional Movement
            dm_plus, dm_minus = self._calculate_directional_movement(high, low)
            
            # Calculate smoothed values
            tr_smoothed = self._exponential_moving_average(tr, self.period)
            dm_plus_smoothed = self._exponential_moving_average(dm_plus, self.period)
            dm_minus_smoothed = self._exponential_moving_average(dm_minus, self.period)
            
            # Calculate Directional Indicators
            di_plus = 100 * dm_plus_smoothed / tr_smoothed
            di_minus = 100 * dm_minus_smoothed / tr_smoothed
            
            # Calculate Directional Index
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
            
            # Calculate ADX
            adx = self._exponential_moving_average(dx, self.period)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(adx, di_plus, di_minus)
            
            # Create metadata
            metadata = {
                "indicator_type": "ADX",
                "calculation_method": "exponential_moving_average",
                "data_points_used": len(close),
                "adx_min": np.nanmin(adx),
                "adx_max": np.nanmax(adx),
                "di_plus_min": np.nanmin(di_plus),
                "di_plus_max": np.nanmax(di_plus),
                "di_minus_min": np.nanmin(di_minus),
                "di_minus_max": np.nanmax(di_minus)
            }
            
            return IndicatorResult(
                values=np.column_stack([adx, di_plus, di_minus, dx]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"ADX calculation failed: {e}")
            raise
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range."""
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],  # Current high-low
                abs(high[i] - close[i-1]),  # Current high - previous close
                abs(low[i] - close[i-1])  # Current low - previous close
            )
        
        return tr
    
    def _calculate_directional_movement(self, high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Directional Movement."""
        dm_plus = np.zeros(len(high))
        dm_minus = np.zeros(len(high))
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0
            
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0
        
        return dm_plus, dm_minus
    
    def _exponential_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _generate_signals(self, adx: np.ndarray, di_plus: np.ndarray, di_minus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on ADX values."""
        signals = np.full(len(adx), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(adx))
        
        for i in range(1, len(adx)):
            if np.isnan(adx[i]):
                continue
            
            # Trend strength signals
            if adx[i] > 25:  # Strong trend
                if di_plus[i] > di_minus[i]:  # Bullish trend
                    if adx[i] > 40:  # Very strong bullish trend
                        signals[i] = SignalType.STRONG_BUY.value
                        signal_strength[i] = 1.0
                    else:
                        signals[i] = SignalType.BUY.value
                        signal_strength[i] = 0.8
                else:  # Bearish trend
                    if adx[i] > 40:  # Very strong bearish trend
                        signals[i] = SignalType.STRONG_SELL.value
                        signal_strength[i] = 1.0
                    else:
                        signals[i] = SignalType.SELL.value
                        signal_strength[i] = 0.8
            elif adx[i] < 20:  # Weak trend
                if di_plus[i] > di_minus[i]:  # Slight bullish bias
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.3
                else:  # Slight bearish bias
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.3
            
            # DI crossover signals
            if i >= 1:
                if (di_plus[i] > di_minus[i] and di_plus[i-1] <= di_minus[i-1] and
                    adx[i] > 20):  # Bullish DI crossover
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.7
                elif (di_minus[i] > di_plus[i] and di_minus[i-1] <= di_plus[i-1] and
                      adx[i] > 20):  # Bearish DI crossover
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.7
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize ADX parameters using grid search.
        
        Args:
            high: High price data for optimization
            low: Low price data for optimization
            close: Close price data for optimization
            target_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = float('-inf')
        
        # Parameter ranges for optimization
        periods = [10, 14, 20, 30]
        
        for period in periods:
            try:
                # Create ADX instance with test parameters
                test_adx = ADX(period=period)
                result = test_adx.calculate(high, low, close)
                
                # Calculate performance metric
                score = self._calculate_performance_metric(result, close, target_metric)
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        "period": period,
                        "score": score
                    }
                    
            except Exception as e:
                logger.warning(f"Parameter combination failed: period={period}, error={e}")
                continue
        
        if best_params is None:
            logger.warning("No valid parameter combination found, using defaults")
            best_params = {"period": 14, "score": 0}
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for parameter optimization."""
        try:
            # Simple backtest simulation
            signals = result.signals
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate strategy returns
            strategy_returns = np.zeros(len(returns))
            position = 0
            
            for i, signal in enumerate(signals[1:]):  # Skip first signal
                if signal == SignalType.BUY.value or signal == SignalType.STRONG_BUY.value:
                    position = 1
                elif signal == SignalType.SELL.value or signal == SignalType.STRONG_SELL.value:
                    position = 0
                
                strategy_returns[i] = position * returns[i]
            
            # Calculate metrics
            if metric == "sharpe_ratio":
                if np.std(strategy_returns) == 0:
                    return 0
                return np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            elif metric == "total_return":
                return np.sum(strategy_returns)
            elif metric == "max_drawdown":
                cumulative = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                return np.min(drawdown)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Performance metric calculation failed: {e}")
            return 0

# Convenience functions for easy usage
def calculate_ichimoku_cloud(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           tenkan_period: int = 9, kijun_period: int = 26,
                           senkou_span_b_period: int = 52, displacement: int = 26) -> IndicatorResult:
    """Calculate Ichimoku Cloud with default parameters."""
    ichimoku = IchimokuCloud(tenkan_period=tenkan_period, kijun_period=kijun_period,
                            senkou_span_b_period=senkou_span_b_period, displacement=displacement)
    return ichimoku.calculate(high, low, close)

def calculate_parabolic_sar(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          acceleration: float = 0.02, maximum: float = 0.2) -> IndicatorResult:
    """Calculate Parabolic SAR with default parameters."""
    sar = ParabolicSAR(acceleration=acceleration, maximum=maximum)
    return sar.calculate(high, low, close)

def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> IndicatorResult:
    """Calculate ADX with default parameters."""
    adx = ADX(period=period)
    return adx.calculate(high, low, close)