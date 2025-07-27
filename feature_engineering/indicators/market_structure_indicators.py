"""
Market Structure Indicators

Implements market structure analysis indicators:
- Pivot Points (Standard, Fibonacci, Camarilla)
- Volume Profile (Volume Weighted Average Price, Volume Zones)
- Beta and Correlation Analysis
- Support and Resistance Levels

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
from scipy import stats

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

class PivotPoints:
    """
    Pivot Points Indicator
    
    Calculates various types of pivot points including:
    - Standard Pivot Points
    - Fibonacci Pivot Points
    - Camarilla Pivot Points
    """
    
    def __init__(self, pivot_type: str = "standard", lookback_period: int = 1):
        """
        Initialize Pivot Points indicator.
        
        Args:
            pivot_type: Type of pivot points ('standard', 'fibonacci', 'camarilla')
            lookback_period: Period for pivot calculation
        """
        self.pivot_type = pivot_type
        self.lookback_period = lookback_period
        
        if pivot_type not in ["standard", "fibonacci", "camarilla"]:
            raise ValueError("Pivot type must be 'standard', 'fibonacci', or 'camarilla'")
        if lookback_period < 1:
            raise ValueError("Lookback period must be at least 1")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate pivot points.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            
        Returns:
            IndicatorResult with pivot points and signals
        """
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("All price arrays must have the same length")
        
        if len(high) < self.lookback_period:
            raise ValueError(f"Not enough data points. Need at least {self.lookback_period}")
        
        # Calculate pivot points based on type
        if self.pivot_type == "standard":
            pivot_points = self._calculate_standard_pivots(high, low, close)
        elif self.pivot_type == "fibonacci":
            pivot_points = self._calculate_fibonacci_pivots(high, low, close)
        else:  # camarilla
            pivot_points = self._calculate_camarilla_pivots(high, low, close)
        
        # Generate signals
        signals, signal_strength = self._generate_signals(pivot_points, close)
        
        # Create metadata
        metadata = {
            "indicator_type": "PivotPoints",
            "pivot_type": self.pivot_type,
            "calculation_method": f"{self.pivot_type}_pivots",
            "data_points_used": len(close),
            "pivot_levels": pivot_points
        }
        
        return IndicatorResult(
            values=pivot_points,
            signals=signals,
            signal_strength=signal_strength,
            parameters={"pivot_type": self.pivot_type, "lookback_period": self.lookback_period},
            metadata=metadata
        )
    
    def _calculate_standard_pivots(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate standard pivot points."""
        pivots = np.full(len(close), np.nan)
        
        for i in range(self.lookback_period, len(close)):
            # Previous period's high, low, close
            prev_high = high[i - self.lookback_period]
            prev_low = low[i - self.lookback_period]
            prev_close = close[i - self.lookback_period]
            
            # Standard pivot point calculation
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Support and resistance levels
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            
            # Store pivot point
            pivots[i] = pivot
        
        return pivots
    
    def _calculate_fibonacci_pivots(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Fibonacci pivot points."""
        pivots = np.full(len(close), np.nan)
        
        # Fibonacci ratios
        fib_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
        
        for i in range(self.lookback_period, len(close)):
            prev_high = high[i - self.lookback_period]
            prev_low = low[i - self.lookback_period]
            prev_close = close[i - self.lookback_period]
            
            # Fibonacci pivot point
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Fibonacci support and resistance levels
            range_high_low = prev_high - prev_low
            
            r1 = pivot + (fib_ratios[0] * range_high_low)
            r2 = pivot + (fib_ratios[1] * range_high_low)
            r3 = pivot + (fib_ratios[2] * range_high_low)
            r4 = pivot + (fib_ratios[3] * range_high_low)
            r5 = pivot + (fib_ratios[4] * range_high_low)
            
            s1 = pivot - (fib_ratios[0] * range_high_low)
            s2 = pivot - (fib_ratios[1] * range_high_low)
            s3 = pivot - (fib_ratios[2] * range_high_low)
            s4 = pivot - (fib_ratios[3] * range_high_low)
            s5 = pivot - (fib_ratios[4] * range_high_low)
            
            pivots[i] = pivot
        
        return pivots
    
    def _calculate_camarilla_pivots(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Camarilla pivot points."""
        pivots = np.full(len(close), np.nan)
        
        for i in range(self.lookback_period, len(close)):
            prev_high = high[i - self.lookback_period]
            prev_low = low[i - self.lookback_period]
            prev_close = close[i - self.lookback_period]
            
            # Camarilla pivot point
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Camarilla levels
            range_high_low = prev_high - prev_low
            
            r1 = prev_close + (range_high_low * 1.1/12)
            r2 = prev_close + (range_high_low * 1.1/6)
            r3 = prev_close + (range_high_low * 1.1/4)
            r4 = prev_close + (range_high_low * 1.1/2)
            r5 = prev_close + (range_high_low * 1.1)
            
            s1 = prev_close - (range_high_low * 1.1/12)
            s2 = prev_close - (range_high_low * 1.1/6)
            s3 = prev_close - (range_high_low * 1.1/4)
            s4 = prev_close - (range_high_low * 1.1/2)
            s5 = prev_close - (range_high_low * 1.1)
            
            pivots[i] = pivot
        
        return pivots
    
    def _generate_signals(self, pivot_points: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on pivot points."""
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if np.isnan(pivot_points[i]):
                continue
            
            current_price = close[i]
            pivot = pivot_points[i]
            
            # Calculate distance from pivot
            distance_from_pivot = (current_price - pivot) / pivot
            
            # Generate signals based on price position relative to pivot
            if distance_from_pivot > 0.02:  # Price above pivot
                signals[i] = SignalType.BUY.value
                signal_strength[i] = min(1.0, abs(distance_from_pivot))
            elif distance_from_pivot < -0.02:  # Price below pivot
                signals[i] = SignalType.SELL.value
                signal_strength[i] = min(1.0, abs(distance_from_pivot))
            elif abs(distance_from_pivot) < 0.005:  # Price near pivot
                signals[i] = SignalType.HOLD.value
                signal_strength[i] = 0.0
        
        return signals, signal_strength

class VolumeProfile:
    """
    Volume Profile Indicator
    
    Calculates volume-weighted average price (VWAP) and volume zones.
    """
    
    def __init__(self, period: int = 20, volume_threshold: float = 0.1):
        """
        Initialize Volume Profile indicator.
        
        Args:
            period: Lookback period for VWAP calculation
            volume_threshold: Threshold for volume zone identification
        """
        self.period = period
        self.volume_threshold = volume_threshold
        
        if period < 1:
            raise ValueError("Period must be at least 1")
        if volume_threshold <= 0 or volume_threshold > 1:
            raise ValueError("Volume threshold must be between 0 and 1")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> IndicatorResult:
        """
        Calculate volume profile indicators.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array
            
        Returns:
            IndicatorResult with volume profile values and signals
        """
        if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
            raise ValueError("All arrays must have the same length")
        
        if len(high) < self.period:
            raise ValueError(f"Not enough data points. Need at least {self.period}")
        
        # Calculate VWAP
        vwap = self._calculate_vwap(high, low, close, volume)
        
        # Calculate volume zones
        volume_zones = self._calculate_volume_zones(high, low, close, volume)
        
        # Generate signals
        signals, signal_strength = self._generate_signals(vwap, volume_zones, close, volume)
        
        # Create metadata
        metadata = {
            "indicator_type": "VolumeProfile",
            "calculation_method": "volume_weighted",
            "data_points_used": len(close),
            "vwap_values": vwap,
            "volume_zones": volume_zones
        }
        
        return IndicatorResult(
            values=vwap,
            signals=signals,
            signal_strength=signal_strength,
            parameters={"period": self.period, "volume_threshold": self.volume_threshold},
            metadata=metadata
        )
    
    def _calculate_vwap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Volume Weighted Average Price."""
        vwap = np.full(len(close), np.nan)
        
        for i in range(self.period - 1, len(close)):
            # Calculate typical price
            typical_price = (high[i - self.period + 1:i + 1] + 
                           low[i - self.period + 1:i + 1] + 
                           close[i - self.period + 1:i + 1]) / 3
            
            # Calculate VWAP
            volume_slice = volume[i - self.period + 1:i + 1]
            vwap[i] = np.sum(typical_price * volume_slice) / np.sum(volume_slice)
        
        return vwap
    
    def _calculate_volume_zones(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate volume zones."""
        zones = np.full(len(close), np.nan)
        
        for i in range(self.period - 1, len(close)):
            # Calculate volume-weighted price levels
            typical_price = (high[i - self.period + 1:i + 1] + 
                           low[i - self.period + 1:i + 1] + 
                           close[i - self.period + 1:i + 1]) / 3
            
            volume_slice = volume[i - self.period + 1:i + 1]
            
            # Identify high volume zones
            avg_volume = np.mean(volume_slice)
            high_volume_mask = volume_slice > (avg_volume * (1 + self.volume_threshold))
            
            if np.any(high_volume_mask):
                high_volume_prices = typical_price[high_volume_mask]
                zones[i] = np.mean(high_volume_prices)
            else:
                zones[i] = np.mean(typical_price)
        
        return zones
    
    def _generate_signals(self, vwap: np.ndarray, volume_zones: np.ndarray, 
                         close: np.ndarray, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on volume profile."""
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if np.isnan(vwap[i]) or np.isnan(volume_zones[i]):
                continue
            
            current_price = close[i]
            current_vwap = vwap[i]
            current_zone = volume_zones[i]
            
            # Calculate volume ratio
            avg_volume = np.mean(volume[max(0, i - self.period):i + 1])
            volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate price position relative to VWAP
            vwap_distance = (current_price - current_vwap) / current_vwap
            
            # Generate signals
            if vwap_distance > 0.01 and volume_ratio > 1.2:  # Price above VWAP with high volume
                signals[i] = SignalType.BUY.value
                signal_strength[i] = min(1.0, abs(vwap_distance) * volume_ratio)
            elif vwap_distance < -0.01 and volume_ratio > 1.2:  # Price below VWAP with high volume
                signals[i] = SignalType.SELL.value
                signal_strength[i] = min(1.0, abs(vwap_distance) * volume_ratio)
            elif abs(vwap_distance) < 0.005:  # Price near VWAP
                signals[i] = SignalType.HOLD.value
                signal_strength[i] = 0.0
        
        return signals, signal_strength

class BetaCorrelationAnalyzer:
    """
    Beta and Correlation Analysis
    
    Calculates beta coefficient and correlation with market indices.
    """
    
    def __init__(self, lookback_period: int = 252, risk_free_rate: float = 0.02):
        """
        Initialize Beta and Correlation Analyzer.
        
        Args:
            lookback_period: Period for beta calculation (default: 1 year)
            risk_free_rate: Risk-free rate for beta calculation
        """
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        
        if lookback_period < 30:
            raise ValueError("Lookback period must be at least 30")
        if risk_free_rate < 0:
            raise ValueError("Risk-free rate must be non-negative")
    
    def calculate(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> IndicatorResult:
        """
        Calculate beta and correlation.
        
        Args:
            asset_returns: Asset returns array
            market_returns: Market returns array
            
        Returns:
            IndicatorResult with beta and correlation values
        """
        if len(asset_returns) != len(market_returns):
            raise ValueError("Asset and market returns arrays must have the same length")
        
        if len(asset_returns) < self.lookback_period:
            raise ValueError(f"Not enough data points. Need at least {self.lookback_period}")
        
        # Calculate rolling beta and correlation
        beta_values = self._calculate_rolling_beta(asset_returns, market_returns)
        correlation_values = self._calculate_rolling_correlation(asset_returns, market_returns)
        
        # Combine beta and correlation into a single metric
        combined_metric = (beta_values + correlation_values) / 2
        
        # Generate signals
        signals, signal_strength = self._generate_signals(beta_values, correlation_values, asset_returns)
        
        # Create metadata
        metadata = {
            "indicator_type": "BetaCorrelation",
            "calculation_method": "rolling_analysis",
            "data_points_used": len(asset_returns),
            "beta_values": beta_values,
            "correlation_values": correlation_values,
            "risk_free_rate": self.risk_free_rate
        }
        
        return IndicatorResult(
            values=combined_metric,
            signals=signals,
            signal_strength=signal_strength,
            parameters={"lookback_period": self.lookback_period, "risk_free_rate": self.risk_free_rate},
            metadata=metadata
        )
    
    def _calculate_rolling_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
        """Calculate rolling beta coefficient."""
        beta_values = np.full(len(asset_returns), np.nan)
        
        for i in range(self.lookback_period - 1, len(asset_returns)):
            # Get rolling window
            asset_window = asset_returns[i - self.lookback_period + 1:i + 1]
            market_window = market_returns[i - self.lookback_period + 1:i + 1]
            
            # Calculate beta
            covariance = np.cov(asset_window, market_window)[0, 1]
            market_variance = np.var(market_window)
            
            if market_variance > 0:
                beta = covariance / market_variance
            else:
                beta = 1.0
            
            beta_values[i] = beta
        
        return beta_values
    
    def _calculate_rolling_correlation(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
        """Calculate rolling correlation coefficient."""
        correlation_values = np.full(len(asset_returns), np.nan)
        
        for i in range(self.lookback_period - 1, len(asset_returns)):
            # Get rolling window
            asset_window = asset_returns[i - self.lookback_period + 1:i + 1]
            market_window = market_returns[i - self.lookback_period + 1:i + 1]
            
            # Calculate correlation
            correlation = np.corrcoef(asset_window, market_window)[0, 1]
            
            if np.isnan(correlation):
                correlation = 0.0
            
            correlation_values[i] = correlation
        
        return correlation_values
    
    def _generate_signals(self, beta_values: np.ndarray, correlation_values: np.ndarray, 
                         asset_returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on beta and correlation."""
        signals = np.full(len(asset_returns), SignalType.HOLD.value)
        signal_strength = np.zeros(len(asset_returns))
        
        for i in range(1, len(asset_returns)):
            if np.isnan(beta_values[i]) or np.isnan(correlation_values[i]):
                continue
            
            current_beta = beta_values[i]
            current_correlation = correlation_values[i]
            current_return = asset_returns[i]
            
            # Generate signals based on beta and correlation
            if current_beta > 1.2 and current_correlation > 0.7 and current_return > 0:
                # High beta, high correlation, positive return
                signals[i] = SignalType.BUY.value
                signal_strength[i] = min(1.0, (current_beta - 1) * current_correlation)
            elif current_beta < 0.8 and current_correlation < 0.3 and current_return < 0:
                # Low beta, low correlation, negative return
                signals[i] = SignalType.SELL.value
                signal_strength[i] = min(1.0, (1 - current_beta) * (1 - current_correlation))
            else:
                signals[i] = SignalType.HOLD.value
                signal_strength[i] = 0.0
        
        return signals, signal_strength

# Convenience functions
def calculate_pivot_points(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          pivot_type: str = "standard", lookback_period: int = 1) -> IndicatorResult:
    """Calculate pivot points."""
    indicator = PivotPoints(pivot_type, lookback_period)
    return indicator.calculate(high, low, close)

def calculate_volume_profile(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                           period: int = 20, volume_threshold: float = 0.1) -> IndicatorResult:
    """Calculate volume profile."""
    indicator = VolumeProfile(period, volume_threshold)
    return indicator.calculate(high, low, close, volume)

def calculate_beta_correlation(asset_returns: np.ndarray, market_returns: np.ndarray,
                             lookback_period: int = 252, risk_free_rate: float = 0.02) -> IndicatorResult:
    """Calculate beta and correlation."""
    analyzer = BetaCorrelationAnalyzer(lookback_period, risk_free_rate)
    return analyzer.calculate(asset_returns, market_returns)