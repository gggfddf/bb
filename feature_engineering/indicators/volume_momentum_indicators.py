"""
Volume Momentum Indicators Module

Implements volume-based momentum technical indicators:
- Ease of Movement (EMV)
- Accumulation/Distribution Line (A/D)

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

class EaseOfMovement:
    """
    Ease of Movement (EMV) Indicator
    
    EMV measures the relationship between volume and price change, indicating how easily
    a stock's price moves. It helps identify potential trend reversals and overbought/oversold
    conditions.
    
    Formula:
    - Distance Moved = (High + Low) / 2 - (Prior High + Prior Low) / 2
    - Box Ratio = Volume / (High - Low)
    - EMV = Distance Moved / Box Ratio
    - Smoothed EMV = SMA(EMV, period)
    """
    
    def __init__(self, period: int = 14, volume_factor: float = 1000000,
                 overbought: float = 0.3, oversold: float = -0.3):
        """
        Initialize EMV indicator.
        
        Args:
            period: Period for smoothing (default: 14)
            volume_factor: Factor to scale volume (default: 1000000)
            overbought: Overbought threshold (default: 0.3)
            oversold: Oversold threshold (default: -0.3)
        """
        self.period = period
        self.volume_factor = volume_factor
        self.overbought = overbought
        self.oversold = oversold
        
        if period < 1:
            raise ValueError("Period must be at least 1")
        if volume_factor <= 0:
            raise ValueError("Volume factor must be positive")
        if overbought <= oversold:
            raise ValueError("Overbought must be greater than oversold")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, 
                  close: np.ndarray, volume: np.ndarray) -> IndicatorResult:
        """
        Calculate EMV indicator.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            volume: Array of volume data
            
        Returns:
            IndicatorResult with EMV values, signals, and metadata
        """
        if len(high) != len(low) != len(close) != len(volume):
            raise ValueError("All input arrays must have the same length")
        
        if len(high) < self.period + 1:
            raise ValueError(f"Insufficient data. Need at least {self.period + 1} points")
        
        # Calculate midpoint move
        midpoint = (high + low) / 2
        distance_moved = np.zeros_like(high)
        distance_moved[1:] = midpoint[1:] - midpoint[:-1]
        
        # Calculate box ratio (volume / range)
        price_range = high - low
        # Avoid division by zero
        price_range = np.where(price_range == 0, 1e-8, price_range)
        box_ratio = volume / price_range
        
        # Calculate raw EMV
        emv_raw = distance_moved / (box_ratio / self.volume_factor)
        
        # Handle infinite values
        emv_raw = np.where(np.isinf(emv_raw), 0, emv_raw)
        emv_raw = np.where(np.isnan(emv_raw), 0, emv_raw)
        
        # Calculate smoothed EMV
        emv_smoothed = self._calculate_sma(emv_raw, self.period)
        
        # Generate signals
        signals = np.full(len(high), SignalType.HOLD.value)
        signal_strength = np.zeros(len(high))
        
        for i in range(len(high)):
            if np.isnan(emv_smoothed[i]):
                continue
                
            if emv_smoothed[i] > self.overbought:
                signals[i] = SignalType.STRONG_SELL.value
                signal_strength[i] = min((emv_smoothed[i] - self.overbought) / self.overbought, 1.0)
            elif emv_smoothed[i] < self.oversold:
                signals[i] = SignalType.STRONG_BUY.value
                signal_strength[i] = min((self.oversold - emv_smoothed[i]) / abs(self.oversold), 1.0)
            elif emv_smoothed[i] > 0 and i > 0 and emv_smoothed[i] > emv_smoothed[i-1]:
                signals[i] = SignalType.BUY.value
                signal_strength[i] = min(emv_smoothed[i] / self.overbought, 1.0)
            elif emv_smoothed[i] < 0 and i > 0 and emv_smoothed[i] < emv_smoothed[i-1]:
                signals[i] = SignalType.SELL.value
                signal_strength[i] = min(abs(emv_smoothed[i]) / abs(self.oversold), 1.0)
        
        metadata = {
            'indicator_type': 'EMV',
            'period': self.period,
            'volume_factor': self.volume_factor,
            'overbought': self.overbought,
            'oversold': self.oversold,
            'calculation_method': 'ease_of_movement'
        }
        
        return IndicatorResult(
            values=np.column_stack((emv_raw, emv_smoothed)),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'volume_factor': self.volume_factor,
                'overbought': self.overbought,
                'oversold': self.oversold
            },
            metadata=metadata
        )
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.zeros_like(data)
        for i in range(len(data)):
            if i < period - 1:
                sma[i] = np.nan
            else:
                sma[i] = np.mean(data[i-period+1:i+1])
        return sma
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, volume: np.ndarray,
                           target_metric: str = 'sharpe_ratio',
                           param_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Optimize EMV parameters using grid search.
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            volume: Volume data
            target_metric: Metric to optimize ('sharpe_ratio', 'max_drawdown', 'total_return')
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        if param_ranges is None:
            param_ranges = {
                'period': [10, 14, 20, 25],
                'volume_factor': [100000, 500000, 1000000, 2000000],
                'overbought': [0.2, 0.3, 0.4, 0.5],
                'oversold': [-0.5, -0.4, -0.3, -0.2]
            }
        
        best_params = None
        best_score = float('-inf') if target_metric == 'sharpe_ratio' else float('inf')
        
        for period in param_ranges['period']:
            for volume_factor in param_ranges['volume_factor']:
                for overbought in param_ranges['overbought']:
                    for oversold in param_ranges['oversold']:
                        if overbought <= oversold:
                            continue
                        
                        try:
                            self.period = period
                            self.volume_factor = volume_factor
                            self.overbought = overbought
                            self.oversold = oversold
                            
                            result = self.calculate(high, low, close, volume)
                            score = self._calculate_performance_metric(result, target_metric)
                            
                            if target_metric == 'sharpe_ratio' and score > best_score:
                                best_score = score
                                best_params = {
                                    'period': period,
                                    'volume_factor': volume_factor,
                                    'overbought': overbought,
                                    'oversold': oversold
                                }
                            elif target_metric in ['max_drawdown', 'total_return'] and score < best_score:
                                best_score = score
                                best_params = {
                                    'period': period,
                                    'volume_factor': volume_factor,
                                    'overbought': overbought,
                                    'oversold': oversold
                                }
                        except Exception as e:
                            logger.warning(f"Parameter combination failed: {e}")
                            continue
        
        if best_params:
            self.period = best_params['period']
            self.volume_factor = best_params['volume_factor']
            self.overbought = best_params['overbought']
            self.oversold = best_params['oversold']
        
        return best_params or {
            'period': 14,
            'volume_factor': 1000000,
            'overbought': 0.3,
            'oversold': -0.3
        }
    
    def _calculate_performance_metric(self, result: IndicatorResult, metric: str) -> float:
        """Calculate performance metric for optimization."""
        if metric == 'sharpe_ratio':
            returns = np.diff(result.values[:, 1])  # Smoothed EMV changes
            returns = returns[~np.isnan(returns)]
            if len(returns) == 0:
                return 0.0
            return np.mean(returns) / (np.std(returns) + 1e-8)
        elif metric == 'max_drawdown':
            cumulative = np.cumsum(result.values[:, 1])
            cumulative = cumulative[~np.isnan(cumulative)]
            if len(cumulative) == 0:
                return 0.0
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown)
        elif metric == 'total_return':
            values = result.values[:, 1]
            values = values[~np.isnan(values)]
            return np.sum(values)
        else:
            return 0.0

class AccumulationDistribution:
    """
    Accumulation/Distribution Line (A/D) Indicator
    
    A/D measures the cumulative flow of money into and out of a security by combining
    price and volume data. It helps identify potential trend reversals and divergence
    between price and volume.
    
    Formula:
    - Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    - Money Flow Volume = Money Flow Multiplier × Volume
    - A/D = Previous A/D + Money Flow Volume
    """
    
    def __init__(self, smoothing_period: int = 14, divergence_threshold: float = 0.1):
        """
        Initialize A/D indicator.
        
        Args:
            smoothing_period: Period for smoothing (default: 14)
            divergence_threshold: Threshold for divergence detection (default: 0.1)
        """
        self.smoothing_period = smoothing_period
        self.divergence_threshold = divergence_threshold
        
        if smoothing_period < 1:
            raise ValueError("Smoothing period must be at least 1")
        if divergence_threshold <= 0:
            raise ValueError("Divergence threshold must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, 
                  close: np.ndarray, volume: np.ndarray) -> IndicatorResult:
        """
        Calculate A/D indicator.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            volume: Array of volume data
            
        Returns:
            IndicatorResult with A/D values, signals, and metadata
        """
        if len(high) != len(low) != len(close) != len(volume):
            raise ValueError("All input arrays must have the same length")
        
        if len(high) < 2:
            raise ValueError("Need at least 2 data points")
        
        # Calculate money flow multiplier
        price_range = high - low
        # Avoid division by zero
        price_range = np.where(price_range == 0, 1e-8, price_range)
        
        money_flow_multiplier = ((close - low) - (high - close)) / price_range
        
        # Calculate money flow volume
        money_flow_volume = money_flow_multiplier * volume
        
        # Calculate A/D line
        ad_line = np.zeros_like(close)
        ad_line[0] = money_flow_volume[0]
        
        for i in range(1, len(close)):
            ad_line[i] = ad_line[i-1] + money_flow_volume[i]
        
        # Calculate smoothed A/D
        ad_smoothed = self._calculate_sma(ad_line, self.smoothing_period)
        
        # Calculate rate of change
        ad_roc = np.zeros_like(ad_line)
        ad_roc[1:] = (ad_line[1:] - ad_line[:-1]) / (np.abs(ad_line[:-1]) + 1e-8)
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        for i in range(len(close)):
            if np.isnan(ad_smoothed[i]):
                continue
                
            # Detect divergence
            if i > 0:
                price_change = (close[i] - close[i-1]) / close[i-1]
                ad_change = ad_roc[i]
                
                # Bullish divergence: price down, A/D up
                if price_change < -self.divergence_threshold and ad_change > self.divergence_threshold:
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = min(abs(ad_change) / self.divergence_threshold, 1.0)
                # Bearish divergence: price up, A/D down
                elif price_change > self.divergence_threshold and ad_change < -self.divergence_threshold:
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = min(abs(ad_change) / self.divergence_threshold, 1.0)
                # A/D trending up
                elif ad_change > 0 and ad_change > self.divergence_threshold:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = min(ad_change / self.divergence_threshold, 1.0)
                # A/D trending down
                elif ad_change < 0 and ad_change < -self.divergence_threshold:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = min(abs(ad_change) / self.divergence_threshold, 1.0)
        
        metadata = {
            'indicator_type': 'A/D',
            'smoothing_period': self.smoothing_period,
            'divergence_threshold': self.divergence_threshold,
            'calculation_method': 'accumulation_distribution'
        }
        
        return IndicatorResult(
            values=np.column_stack((ad_line, ad_smoothed, ad_roc)),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'smoothing_period': self.smoothing_period,
                'divergence_threshold': self.divergence_threshold
            },
            metadata=metadata
        )
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.zeros_like(data)
        for i in range(len(data)):
            if i < period - 1:
                sma[i] = np.nan
            else:
                sma[i] = np.mean(data[i-period+1:i+1])
        return sma
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, volume: np.ndarray,
                           target_metric: str = 'sharpe_ratio',
                           param_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Optimize A/D parameters using grid search.
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            volume: Volume data
            target_metric: Metric to optimize ('sharpe_ratio', 'max_drawdown', 'total_return')
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        if param_ranges is None:
            param_ranges = {
                'smoothing_period': [10, 14, 20, 25],
                'divergence_threshold': [0.05, 0.1, 0.15, 0.2]
            }
        
        best_params = None
        best_score = float('-inf') if target_metric == 'sharpe_ratio' else float('inf')
        
        for smoothing_period in param_ranges['smoothing_period']:
            for divergence_threshold in param_ranges['divergence_threshold']:
                try:
                    self.smoothing_period = smoothing_period
                    self.divergence_threshold = divergence_threshold
                    
                    result = self.calculate(high, low, close, volume)
                    score = self._calculate_performance_metric(result, target_metric)
                    
                    if target_metric == 'sharpe_ratio' and score > best_score:
                        best_score = score
                        best_params = {
                            'smoothing_period': smoothing_period,
                            'divergence_threshold': divergence_threshold
                        }
                    elif target_metric in ['max_drawdown', 'total_return'] and score < best_score:
                        best_score = score
                        best_params = {
                            'smoothing_period': smoothing_period,
                            'divergence_threshold': divergence_threshold
                        }
                except Exception as e:
                    logger.warning(f"Parameter combination failed: {e}")
                    continue
        
        if best_params:
            self.smoothing_period = best_params['smoothing_period']
            self.divergence_threshold = best_params['divergence_threshold']
        
        return best_params or {
            'smoothing_period': 14,
            'divergence_threshold': 0.1
        }
    
    def _calculate_performance_metric(self, result: IndicatorResult, metric: str) -> float:
        """Calculate performance metric for optimization."""
        if metric == 'sharpe_ratio':
            returns = np.diff(result.values[:, 1])  # Smoothed A/D changes
            returns = returns[~np.isnan(returns)]
            if len(returns) == 0:
                return 0.0
            return np.mean(returns) / (np.std(returns) + 1e-8)
        elif metric == 'max_drawdown':
            cumulative = np.cumsum(result.values[:, 1])
            cumulative = cumulative[~np.isnan(cumulative)]
            if len(cumulative) == 0:
                return 0.0
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown)
        elif metric == 'total_return':
            values = result.values[:, 1]
            values = values[~np.isnan(values)]
            return np.sum(values)
        else:
            return 0.0

# Convenience functions for easy usage
def calculate_emv(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                  period: int = 14, volume_factor: float = 1000000,
                  overbought: float = 0.3, oversold: float = -0.3) -> IndicatorResult:
    """Calculate EMV with default parameters."""
    emv = EaseOfMovement(period=period, volume_factor=volume_factor,
                         overbought=overbought, oversold=oversold)
    return emv.calculate(high, low, close, volume)

def calculate_ad_line(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                      smoothing_period: int = 14, divergence_threshold: float = 0.1) -> IndicatorResult:
    """Calculate A/D Line with default parameters."""
    ad = AccumulationDistribution(smoothing_period=smoothing_period, divergence_threshold=divergence_threshold)
    return ad.calculate(high, low, close, volume)