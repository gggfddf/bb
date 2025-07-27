"""
Trend Indicators Module

Implements key trend-based technical indicators:
- Bollinger Bands
- VWAP (Volume Weighted Average Price)
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)

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

class BollingerBands:
    """
    Bollinger Bands implementation.
    
    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    that are standard deviations away from the middle band.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            period: Lookback period for SMA calculation (default: 20)
            std_dev: Number of standard deviations for bands (default: 2.0)
        """
        self.period = period
        self.std_dev = std_dev
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Bollinger Bands parameters."""
        if self.period <= 0:
            raise ValueError("Bollinger Bands period must be positive")
        if self.std_dev <= 0:
            raise ValueError("Standard deviation must be positive")
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate Bollinger Bands values.
        
        Args:
            prices: Array of price data (OHLC or close prices)
            
        Returns:
            IndicatorResult with Bollinger Bands values and signals
        """
        try:
            if len(prices) < self.period:
                raise ValueError(f"Insufficient data for Bollinger Bands calculation. Need at least {self.period} data points")
            
            # Calculate middle band (SMA)
            middle_band = self._simple_moving_average(prices, self.period)
            
            # Calculate standard deviation
            std_dev_values = self._rolling_standard_deviation(prices, self.period)
            
            # Calculate upper and lower bands
            upper_band = middle_band + (self.std_dev * std_dev_values)
            lower_band = middle_band - (self.std_dev * std_dev_values)
            
            # Calculate bandwidth and %B
            bandwidth = (upper_band - lower_band) / middle_band
            percent_b = (prices - lower_band) / (upper_band - lower_band)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(prices, upper_band, middle_band, lower_band, percent_b)
            
            # Create metadata
            metadata = {
                "indicator_type": "Bollinger_Bands",
                "calculation_method": "simple_moving_average",
                "data_points_used": len(prices),
                "upper_band_min": np.nanmin(upper_band),
                "upper_band_max": np.nanmax(upper_band),
                "lower_band_min": np.nanmin(lower_band),
                "lower_band_max": np.nanmax(lower_band),
                "bandwidth_min": np.nanmin(bandwidth),
                "bandwidth_max": np.nanmax(bandwidth)
            }
            
            return IndicatorResult(
                values=np.column_stack([upper_band, middle_band, lower_band, bandwidth, percent_b]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period, "std_dev": self.std_dev},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            raise
    
    def _simple_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        sma = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        return sma
    
    def _rolling_standard_deviation(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        std_dev = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            std_dev[i] = np.std(data[i - period + 1:i + 1])
        
        return std_dev
    
    def _generate_signals(self, prices: np.ndarray, upper_band: np.ndarray, 
                         middle_band: np.ndarray, lower_band: np.ndarray, 
                         percent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on Bollinger Bands values."""
        signals = np.full(len(prices), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(prices))
        
        for i in range(1, len(prices)):
            if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
                continue
            
            # Price touching bands
            if prices[i] <= lower_band[i]:
                if percent_b[i] <= 0.1:  # Strong oversold
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.7
            elif prices[i] >= upper_band[i]:
                if percent_b[i] >= 0.9:  # Strong overbought
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.7
            
            # Squeeze signals (bands narrowing)
            if i >= 5:
                recent_bandwidth = np.nanmean(upper_band[i-5:i] - lower_band[i-5:i])
                current_bandwidth = upper_band[i] - lower_band[i]
                
                if current_bandwidth < recent_bandwidth * 0.8:  # Bands narrowing
                    if prices[i] > middle_band[i]:
                        signals[i] = SignalType.BUY.value
                        signal_strength[i] = 0.6
                    else:
                        signals[i] = SignalType.SELL.value
                        signal_strength[i] = 0.6
            
            # Mean reversion signals
            if i >= 2:
                if (prices[i] < lower_band[i] and 
                    prices[i-1] >= lower_band[i-1] and
                    prices[i] > prices[i-1]):  # Bounce from lower band
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                elif (prices[i] > upper_band[i] and 
                      prices[i-1] <= upper_band[i-1] and
                      prices[i] < prices[i-1]):  # Rejection from upper band
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
        
        return signals, signal_strength
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize Bollinger Bands parameters using grid search.
        
        Args:
            prices: Price data for optimization
            target_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = float('-inf')
        
        # Parameter ranges for optimization
        periods = [10, 15, 20, 25, 30]
        std_devs = [1.5, 2.0, 2.5, 3.0]
        
        for period in periods:
            for std_dev in std_devs:
                try:
                    # Create Bollinger Bands instance with test parameters
                    test_bb = BollingerBands(period=period, std_dev=std_dev)
                    result = test_bb.calculate(prices)
                    
                    # Calculate performance metric
                    score = self._calculate_performance_metric(result, prices, target_metric)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "period": period,
                            "std_dev": std_dev,
                            "score": score
                        }
                        
                except Exception as e:
                    logger.warning(f"Parameter combination failed: period={period}, std_dev={std_dev}, error={e}")
                    continue
        
        if best_params is None:
            logger.warning("No valid parameter combination found, using defaults")
            best_params = {"period": 20, "std_dev": 2.0, "score": 0}
        
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

class VWAP:
    """
    Volume Weighted Average Price (VWAP) implementation.
    
    VWAP is the average price weighted by volume, commonly used
    to determine if a security is trading above or below fair value.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize VWAP indicator.
        
        Args:
            period: Lookback period for VWAP calculation (default: 14)
        """
        self.period = period
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate VWAP parameters."""
        if self.period <= 0:
            raise ValueError("VWAP period must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 volume: np.ndarray) -> IndicatorResult:
        """
        Calculate VWAP values.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            volume: Array of volume data
            
        Returns:
            IndicatorResult with VWAP values and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
                raise ValueError("All input arrays must have the same length")
            
            if len(high) < self.period:
                raise ValueError(f"Insufficient data for VWAP calculation. Need at least {self.period} data points")
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate VWAP
            vwap_values = self._calculate_vwap(typical_price, volume)
            
            # Calculate VWAP bands (standard deviation bands)
            vwap_std = self._rolling_standard_deviation(typical_price, volume, self.period)
            upper_band = vwap_values + (2 * vwap_std)
            lower_band = vwap_values - (2 * vwap_std)
            
            # Calculate VWAP ratio
            vwap_ratio = (close - vwap_values) / vwap_values
            
            # Generate signals
            signals, signal_strength = self._generate_signals(close, vwap_values, upper_band, lower_band, vwap_ratio)
            
            # Create metadata
            metadata = {
                "indicator_type": "VWAP",
                "calculation_method": "volume_weighted_average",
                "data_points_used": len(close),
                "vwap_min": np.nanmin(vwap_values),
                "vwap_max": np.nanmax(vwap_values),
                "upper_band_min": np.nanmin(upper_band),
                "upper_band_max": np.nanmax(upper_band),
                "lower_band_min": np.nanmin(lower_band),
                "lower_band_max": np.nanmax(lower_band)
            }
            
            return IndicatorResult(
                values=np.column_stack([vwap_values, upper_band, lower_band, vwap_ratio]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"VWAP calculation failed: {e}")
            raise
    
    def _calculate_vwap(self, typical_price: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate VWAP values."""
        vwap = np.full(len(typical_price), np.nan)
        
        for i in range(self.period - 1, len(typical_price)):
            # Calculate cumulative price-volume product
            pv_sum = np.sum(typical_price[i - self.period + 1:i + 1] * volume[i - self.period + 1:i + 1])
            # Calculate cumulative volume
            vol_sum = np.sum(volume[i - self.period + 1:i + 1])
            # Calculate VWAP
            vwap[i] = pv_sum / vol_sum if vol_sum > 0 else typical_price[i]
        
        return vwap
    
    def _rolling_standard_deviation(self, typical_price: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation weighted by volume."""
        std_dev = np.full(len(typical_price), np.nan)
        
        for i in range(period - 1, len(typical_price)):
            prices = typical_price[i - period + 1:i + 1]
            volumes = volume[i - period + 1:i + 1]
            
            # Calculate volume-weighted mean
            weighted_mean = np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else np.mean(prices)
            
            # Calculate volume-weighted variance
            weighted_variance = np.sum(volumes * (prices - weighted_mean) ** 2) / np.sum(volumes) if np.sum(volumes) > 0 else np.var(prices)
            
            std_dev[i] = np.sqrt(weighted_variance)
        
        return std_dev
    
    def _generate_signals(self, close: np.ndarray, vwap: np.ndarray, 
                         upper_band: np.ndarray, lower_band: np.ndarray,
                         vwap_ratio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on VWAP values."""
        signals = np.full(len(close), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if np.isnan(vwap[i]):
                continue
            
            # Price relative to VWAP
            if close[i] < vwap[i] * 0.98:  # Significantly below VWAP
                if close[i] <= lower_band[i]:  # Below lower band
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.7
            elif close[i] > vwap[i] * 1.02:  # Significantly above VWAP
                if close[i] >= upper_band[i]:  # Above upper band
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.7
            
            # VWAP crossover signals
            if i >= 1:
                if (close[i] > vwap[i] and close[i-1] <= vwap[i-1] and
                    vwap_ratio[i] > 0.01):  # Price crosses above VWAP
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                elif (close[i] < vwap[i] and close[i-1] >= vwap[i-1] and
                      vwap_ratio[i] < -0.01):  # Price crosses below VWAP
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          volume: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize VWAP parameters using grid search.
        
        Args:
            high: High price data for optimization
            low: Low price data for optimization
            close: Close price data for optimization
            volume: Volume data for optimization
            target_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = float('-inf')
        
        # Parameter ranges for optimization
        periods = [5, 10, 14, 20, 30]
        
        for period in periods:
            try:
                # Create VWAP instance with test parameters
                test_vwap = VWAP(period=period)
                result = test_vwap.calculate(high, low, close, volume)
                
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

class SMA:
    """
    Simple Moving Average (SMA) implementation.
    
    SMA is the arithmetic mean of a set of prices over a specified period.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize SMA indicator.
        
        Args:
            period: Lookback period for SMA calculation (default: 20)
        """
        self.period = period
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate SMA parameters."""
        if self.period <= 0:
            raise ValueError("SMA period must be positive")
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate SMA values.
        
        Args:
            prices: Array of price data (OHLC or close prices)
            
        Returns:
            IndicatorResult with SMA values and signals
        """
        try:
            if len(prices) < self.period:
                raise ValueError(f"Insufficient data for SMA calculation. Need at least {self.period} data points")
            
            # Calculate SMA
            sma_values = self._simple_moving_average(prices, self.period)
            
            # Calculate price deviation from SMA
            price_deviation = (prices - sma_values) / sma_values
            
            # Generate signals
            signals, signal_strength = self._generate_signals(prices, sma_values, price_deviation)
            
            # Create metadata
            metadata = {
                "indicator_type": "SMA",
                "calculation_method": "simple_moving_average",
                "data_points_used": len(prices),
                "sma_min": np.nanmin(sma_values),
                "sma_max": np.nanmax(sma_values),
                "deviation_min": np.nanmin(price_deviation),
                "deviation_max": np.nanmax(price_deviation)
            }
            
            return IndicatorResult(
                values=np.column_stack([sma_values, price_deviation]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"SMA calculation failed: {e}")
            raise
    
    def _simple_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        sma = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        return sma
    
    def _generate_signals(self, prices: np.ndarray, sma: np.ndarray, 
                         price_deviation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on SMA values."""
        signals = np.full(len(prices), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(prices))
        
        for i in range(1, len(prices)):
            if np.isnan(sma[i]):
                continue
            
            # Price relative to SMA
            if price_deviation[i] < -0.05:  # Price significantly below SMA
                signals[i] = SignalType.BUY.value
                signal_strength[i] = 0.7
            elif price_deviation[i] > 0.05:  # Price significantly above SMA
                signals[i] = SignalType.SELL.value
                signal_strength[i] = 0.7
            
            # SMA crossover signals
            if i >= 1:
                if (prices[i] > sma[i] and prices[i-1] <= sma[i-1] and
                    price_deviation[i] > 0.01):  # Price crosses above SMA
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                elif (prices[i] < sma[i] and prices[i-1] >= sma[i-1] and
                      price_deviation[i] < -0.01):  # Price crosses below SMA
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
        
        return signals, signal_strength
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize SMA parameters using grid search.
        
        Args:
            prices: Price data for optimization
            target_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = float('-inf')
        
        # Parameter ranges for optimization
        periods = [5, 10, 15, 20, 25, 30, 50, 100]
        
        for period in periods:
            try:
                # Create SMA instance with test parameters
                test_sma = SMA(period=period)
                result = test_sma.calculate(prices)
                
                # Calculate performance metric
                score = self._calculate_performance_metric(result, prices, target_metric)
                
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
            best_params = {"period": 20, "score": 0}
        
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

class EMA:
    """
    Exponential Moving Average (EMA) implementation.
    
    EMA gives more weight to recent prices compared to SMA,
    making it more responsive to recent price changes.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize EMA indicator.
        
        Args:
            period: Lookback period for EMA calculation (default: 20)
        """
        self.period = period
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate EMA parameters."""
        if self.period <= 0:
            raise ValueError("EMA period must be positive")
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate EMA values.
        
        Args:
            prices: Array of price data (OHLC or close prices)
            
        Returns:
            IndicatorResult with EMA values and signals
        """
        try:
            if len(prices) < self.period:
                raise ValueError(f"Insufficient data for EMA calculation. Need at least {self.period} data points")
            
            # Calculate EMA
            ema_values = self._exponential_moving_average(prices, self.period)
            
            # Calculate price deviation from EMA
            price_deviation = (prices - ema_values) / ema_values
            
            # Generate signals
            signals, signal_strength = self._generate_signals(prices, ema_values, price_deviation)
            
            # Create metadata
            metadata = {
                "indicator_type": "EMA",
                "calculation_method": "exponential_moving_average",
                "data_points_used": len(prices),
                "ema_min": np.nanmin(ema_values),
                "ema_max": np.nanmax(ema_values),
                "deviation_min": np.nanmin(price_deviation),
                "deviation_max": np.nanmax(price_deviation)
            }
            
            return IndicatorResult(
                values=np.column_stack([ema_values, price_deviation]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"EMA calculation failed: {e}")
            raise
    
    def _exponential_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _generate_signals(self, prices: np.ndarray, ema: np.ndarray, 
                         price_deviation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on EMA values."""
        signals = np.full(len(prices), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(prices))
        
        for i in range(1, len(prices)):
            if np.isnan(ema[i]):
                continue
            
            # Price relative to EMA
            if price_deviation[i] < -0.03:  # Price significantly below EMA
                signals[i] = SignalType.BUY.value
                signal_strength[i] = 0.7
            elif price_deviation[i] > 0.03:  # Price significantly above EMA
                signals[i] = SignalType.SELL.value
                signal_strength[i] = 0.7
            
            # EMA crossover signals
            if i >= 1:
                if (prices[i] > ema[i] and prices[i-1] <= ema[i-1] and
                    price_deviation[i] > 0.005):  # Price crosses above EMA
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                elif (prices[i] < ema[i] and prices[i-1] >= ema[i-1] and
                      price_deviation[i] < -0.005):  # Price crosses below EMA
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
            
            # EMA slope signals
            if i >= 5:
                ema_slope = (ema[i] - ema[i-5]) / ema[i-5]
                if ema_slope > 0.02:  # Strong upward slope
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.6
                elif ema_slope < -0.02:  # Strong downward slope
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.6
        
        return signals, signal_strength
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize EMA parameters using grid search.
        
        Args:
            prices: Price data for optimization
            target_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = float('-inf')
        
        # Parameter ranges for optimization
        periods = [5, 10, 15, 20, 25, 30, 50, 100]
        
        for period in periods:
            try:
                # Create EMA instance with test parameters
                test_ema = EMA(period=period)
                result = test_ema.calculate(prices)
                
                # Calculate performance metric
                score = self._calculate_performance_metric(result, prices, target_metric)
                
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
            best_params = {"period": 20, "score": 0}
        
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
def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> IndicatorResult:
    """Calculate Bollinger Bands with default parameters."""
    bb = BollingerBands(period=period, std_dev=std_dev)
    return bb.calculate(prices)

def calculate_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  volume: np.ndarray, period: int = 14) -> IndicatorResult:
    """Calculate VWAP with default parameters."""
    vwap = VWAP(period=period)
    return vwap.calculate(high, low, close, volume)

def calculate_sma(prices: np.ndarray, period: int = 20) -> IndicatorResult:
    """Calculate SMA with default parameters."""
    sma = SMA(period=period)
    return sma.calculate(prices)

def calculate_ema(prices: np.ndarray, period: int = 20) -> IndicatorResult:
    """Calculate EMA with default parameters."""
    ema = EMA(period=period)
    return ema.calculate(prices)