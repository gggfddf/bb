"""
Advanced Volume Indicators Module

Implements advanced volume-based technical indicators:
- Balance of Power (BOP)
- Vortex Indicator (VI)

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

class BalanceOfPower:
    """
    Balance of Power (BOP) Indicator
    
    BOP measures the strength of buyers vs. sellers by comparing the closing price to the
    trading range. It helps identify potential trend reversals and overbought/oversold conditions.
    
    Formula:
    - BOP = (Close - Open) / (High - Low)
    - Smoothed BOP = SMA(BOP, period)
    """
    
    def __init__(self, period: int = 14, overbought: float = 0.7, oversold: float = -0.7):
        """
        Initialize BOP indicator.
        
        Args:
            period: Period for smoothing (default: 14)
            overbought: Overbought threshold (default: 0.7)
            oversold: Oversold threshold (default: -0.7)
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
        if period < 1:
            raise ValueError("Period must be at least 1")
        if overbought <= oversold:
            raise ValueError("Overbought must be greater than oversold")
    
    def calculate(self, open_prices: np.ndarray, high: np.ndarray, 
                  low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate BOP indicator.
        
        Args:
            open_prices: Array of opening prices
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            
        Returns:
            IndicatorResult with BOP values, signals, and metadata
        """
        if len(open_prices) != len(high) != len(low) != len(close):
            raise ValueError("All input arrays must have the same length")
        
        if len(open_prices) < self.period:
            raise ValueError(f"Insufficient data. Need at least {self.period} points")
        
        # Calculate price range
        price_range = high - low
        # Avoid division by zero
        price_range = np.where(price_range == 0, 1e-8, price_range)
        
        # Calculate BOP
        bop_raw = (close - open_prices) / price_range
        
        # Handle infinite values
        bop_raw = np.where(np.isinf(bop_raw), 0, bop_raw)
        bop_raw = np.where(np.isnan(bop_raw), 0, bop_raw)
        
        # Calculate smoothed BOP
        bop_smoothed = self._calculate_sma(bop_raw, self.period)
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        for i in range(len(close)):
            if np.isnan(bop_smoothed[i]):
                continue
                
            if bop_smoothed[i] > self.overbought:
                signals[i] = SignalType.STRONG_SELL.value
                signal_strength[i] = min((bop_smoothed[i] - self.overbought) / (1 - self.overbought), 1.0)
            elif bop_smoothed[i] < self.oversold:
                signals[i] = SignalType.STRONG_BUY.value
                signal_strength[i] = min((self.oversold - bop_smoothed[i]) / (self.oversold + 1), 1.0)
            elif bop_smoothed[i] > 0 and i > 0 and bop_smoothed[i] > bop_smoothed[i-1]:
                signals[i] = SignalType.BUY.value
                signal_strength[i] = min(bop_smoothed[i] / self.overbought, 1.0)
            elif bop_smoothed[i] < 0 and i > 0 and bop_smoothed[i] < bop_smoothed[i-1]:
                signals[i] = SignalType.SELL.value
                signal_strength[i] = min(abs(bop_smoothed[i]) / abs(self.oversold), 1.0)
        
        metadata = {
            'indicator_type': 'BOP',
            'period': self.period,
            'overbought': self.overbought,
            'oversold': self.oversold,
            'calculation_method': 'balance_of_power'
        }
        
        return IndicatorResult(
            values=np.column_stack((bop_raw, bop_smoothed)),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
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
    
    def optimize_parameters(self, open_prices: np.ndarray, high: np.ndarray, 
                           low: np.ndarray, close: np.ndarray,
                           target_metric: str = 'sharpe_ratio',
                           param_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Optimize BOP parameters using grid search.
        
        Args:
            open_prices: Opening prices
            high: High prices
            low: Low prices
            close: Closing prices
            target_metric: Metric to optimize ('sharpe_ratio', 'max_drawdown', 'total_return')
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        if param_ranges is None:
            param_ranges = {
                'period': [10, 14, 20, 25],
                'overbought': [0.5, 0.6, 0.7, 0.8],
                'oversold': [-0.8, -0.7, -0.6, -0.5]
            }
        
        best_params = None
        best_score = float('-inf') if target_metric == 'sharpe_ratio' else float('inf')
        
        for period in param_ranges['period']:
            for overbought in param_ranges['overbought']:
                for oversold in param_ranges['oversold']:
                    if overbought <= oversold:
                        continue
                    
                    try:
                        self.period = period
                        self.overbought = overbought
                        self.oversold = oversold
                        
                        result = self.calculate(open_prices, high, low, close)
                        score = self._calculate_performance_metric(result, target_metric)
                        
                        if target_metric == 'sharpe_ratio' and score > best_score:
                            best_score = score
                            best_params = {
                                'period': period,
                                'overbought': overbought,
                                'oversold': oversold
                            }
                        elif target_metric in ['max_drawdown', 'total_return'] and score < best_score:
                            best_score = score
                            best_params = {
                                'period': period,
                                'overbought': overbought,
                                'oversold': oversold
                            }
                    except Exception as e:
                        logger.warning(f"Parameter combination failed: {e}")
                        continue
        
        if best_params:
            self.period = best_params['period']
            self.overbought = best_params['overbought']
            self.oversold = best_params['oversold']
        
        return best_params or {
            'period': 14,
            'overbought': 0.7,
            'oversold': -0.7
        }
    
    def _calculate_performance_metric(self, result: IndicatorResult, metric: str) -> float:
        """Calculate performance metric for optimization."""
        if metric == 'sharpe_ratio':
            returns = np.diff(result.values[:, 1])  # Smoothed BOP changes
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

class VortexIndicator:
    """
    Vortex Indicator (VI)
    
    VI identifies the start of a trend and its direction by comparing two oscillating lines.
    It helps identify potential trend reversals and continuation patterns.
    
    Formula:
    - +VM = |Current High - Prior Low|
    - -VM = |Current Low - Prior High|
    - +DM = Current High - Prior High (if positive, else 0)
    - -DM = Prior Low - Current Low (if positive, else 0)
    - +VI = SMA(+VM, period) / SMA(+DM, period)
    - -VI = SMA(-VM, period) / SMA(-DM, period)
    """
    
    def __init__(self, period: int = 14, crossover_threshold: float = 0.1):
        """
        Initialize Vortex Indicator.
        
        Args:
            period: Period for smoothing (default: 14)
            crossover_threshold: Threshold for crossover detection (default: 0.1)
        """
        self.period = period
        self.crossover_threshold = crossover_threshold
        
        if period < 1:
            raise ValueError("Period must be at least 1")
        if crossover_threshold <= 0:
            raise ValueError("Crossover threshold must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Vortex Indicator.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            
        Returns:
            IndicatorResult with VI values, signals, and metadata
        """
        if len(high) != len(low) != len(close):
            raise ValueError("All input arrays must have the same length")
        
        if len(high) < self.period + 1:
            raise ValueError(f"Insufficient data. Need at least {self.period + 1} points")
        
        # Calculate +VM and -VM
        plus_vm = np.zeros_like(high)
        minus_vm = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_vm[i] = abs(high[i] - low[i-1])
            minus_vm[i] = abs(low[i] - high[i-1])
        
        # Calculate +DM and -DM
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > 0 else 0
        
        # Calculate smoothed values
        plus_vm_sma = self._calculate_sma(plus_vm, self.period)
        minus_vm_sma = self._calculate_sma(minus_vm, self.period)
        plus_dm_sma = self._calculate_sma(plus_dm, self.period)
        minus_dm_sma = self._calculate_sma(minus_dm, self.period)
        
        # Calculate VI lines
        plus_vi = np.zeros_like(high)
        minus_vi = np.zeros_like(high)
        
        for i in range(len(high)):
            if plus_dm_sma[i] != 0:
                plus_vi[i] = plus_vm_sma[i] / plus_dm_sma[i]
            else:
                plus_vi[i] = 0
                
            if minus_dm_sma[i] != 0:
                minus_vi[i] = minus_vm_sma[i] / minus_dm_sma[i]
            else:
                minus_vi[i] = 0
        
        # Handle infinite values
        plus_vi = np.where(np.isinf(plus_vi), 0, plus_vi)
        minus_vi = np.where(np.isinf(minus_vi), 0, minus_vi)
        
        # Generate signals
        signals = np.full(len(close), SignalType.HOLD.value)
        signal_strength = np.zeros(len(close))
        
        for i in range(len(close)):
            if np.isnan(plus_vi[i]) or np.isnan(minus_vi[i]):
                continue
                
            # Detect crossovers
            if i > 0:
                # Bullish crossover: +VI crosses above -VI
                if (plus_vi[i] > minus_vi[i] and 
                    plus_vi[i-1] <= minus_vi[i-1] and
                    abs(plus_vi[i] - minus_vi[i]) > self.crossover_threshold):
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = min(abs(plus_vi[i] - minus_vi[i]) / self.crossover_threshold, 1.0)
                # Bearish crossover: -VI crosses above +VI
                elif (minus_vi[i] > plus_vi[i] and 
                      minus_vi[i-1] <= plus_vi[i-1] and
                      abs(minus_vi[i] - plus_vi[i]) > self.crossover_threshold):
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = min(abs(minus_vi[i] - plus_vi[i]) / self.crossover_threshold, 1.0)
                # +VI trending up
                elif plus_vi[i] > plus_vi[i-1] and plus_vi[i] > minus_vi[i]:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = min((plus_vi[i] - minus_vi[i]) / self.crossover_threshold, 1.0)
                # -VI trending up
                elif minus_vi[i] > minus_vi[i-1] and minus_vi[i] > plus_vi[i]:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = min((minus_vi[i] - plus_vi[i]) / self.crossover_threshold, 1.0)
        
        metadata = {
            'indicator_type': 'VI',
            'period': self.period,
            'crossover_threshold': self.crossover_threshold,
            'calculation_method': 'vortex_indicator'
        }
        
        return IndicatorResult(
            values=np.column_stack((plus_vi, minus_vi)),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'crossover_threshold': self.crossover_threshold
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
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           target_metric: str = 'sharpe_ratio',
                           param_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Optimize VI parameters using grid search.
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            target_metric: Metric to optimize ('sharpe_ratio', 'max_drawdown', 'total_return')
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        if param_ranges is None:
            param_ranges = {
                'period': [10, 14, 20, 25],
                'crossover_threshold': [0.05, 0.1, 0.15, 0.2]
            }
        
        best_params = None
        best_score = float('-inf') if target_metric == 'sharpe_ratio' else float('inf')
        
        for period in param_ranges['period']:
            for crossover_threshold in param_ranges['crossover_threshold']:
                try:
                    self.period = period
                    self.crossover_threshold = crossover_threshold
                    
                    result = self.calculate(high, low, close)
                    score = self._calculate_performance_metric(result, target_metric)
                    
                    if target_metric == 'sharpe_ratio' and score > best_score:
                        best_score = score
                        best_params = {
                            'period': period,
                            'crossover_threshold': crossover_threshold
                        }
                    elif target_metric in ['max_drawdown', 'total_return'] and score < best_score:
                        best_score = score
                        best_params = {
                            'period': period,
                            'crossover_threshold': crossover_threshold
                        }
                except Exception as e:
                    logger.warning(f"Parameter combination failed: {e}")
                    continue
        
        if best_params:
            self.period = best_params['period']
            self.crossover_threshold = best_params['crossover_threshold']
        
        return best_params or {
            'period': 14,
            'crossover_threshold': 0.1
        }
    
    def _calculate_performance_metric(self, result: IndicatorResult, metric: str) -> float:
        """Calculate performance metric for optimization."""
        if metric == 'sharpe_ratio':
            # Use the difference between +VI and -VI as returns
            vi_diff = result.values[:, 0] - result.values[:, 1]
            returns = np.diff(vi_diff)
            returns = returns[~np.isnan(returns)]
            if len(returns) == 0:
                return 0.0
            return np.mean(returns) / (np.std(returns) + 1e-8)
        elif metric == 'max_drawdown':
            vi_diff = result.values[:, 0] - result.values[:, 1]
            cumulative = np.cumsum(vi_diff)
            cumulative = cumulative[~np.isnan(cumulative)]
            if len(cumulative) == 0:
                return 0.0
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown)
        elif metric == 'total_return':
            vi_diff = result.values[:, 0] - result.values[:, 1]
            vi_diff = vi_diff[~np.isnan(vi_diff)]
            return np.sum(vi_diff)
        else:
            return 0.0

# Convenience functions for easy usage
def calculate_bop(open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int = 14, overbought: float = 0.7, oversold: float = -0.7) -> IndicatorResult:
    """Calculate BOP with default parameters."""
    bop = BalanceOfPower(period=period, overbought=overbought, oversold=oversold)
    return bop.calculate(open_prices, high, low, close)

def calculate_vortex_indicator(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                              period: int = 14, crossover_threshold: float = 0.1) -> IndicatorResult:
    """Calculate Vortex Indicator with default parameters."""
    vi = VortexIndicator(period=period, crossover_threshold=crossover_threshold)
    return vi.calculate(high, low, close)