"""
Momentum Oscillators Module

Implements momentum-based oscillator technical indicators:
- TRIX
- Detrended Price Oscillator (DPO)

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

class TRIX:
    """
    TRIX (Triple Exponential Average) Indicator
    
    TRIX is a momentum oscillator that shows the percentage rate of change of a triple
    exponentially smoothed moving average of the closing price. It helps identify
    overbought and oversold conditions and potential trend reversals.
    
    Formula:
    - EMA1 = EMA(close, period)
    - EMA2 = EMA(EMA1, period)
    - EMA3 = EMA(EMA2, period)
    - TRIX = 100 * (EMA3 - EMA3[1]) / EMA3[1]
    - Signal = EMA(TRIX, signal_period)
    """
    
    def __init__(self, period: int = 15, signal_period: int = 9, 
                 overbought: float = 0.5, oversold: float = -0.5):
        """
        Initialize TRIX indicator.
        
        Args:
            period: Period for EMA calculation (default: 15)
            signal_period: Period for signal line EMA (default: 9)
            overbought: Overbought threshold (default: 0.5)
            oversold: Oversold threshold (default: -0.5)
        """
        self.period = period
        self.signal_period = signal_period
        self.overbought = overbought
        self.oversold = oversold
        
        if period < 1:
            raise ValueError("Period must be at least 1")
        if signal_period < 1:
            raise ValueError("Signal period must be at least 1")
        if overbought <= oversold:
            raise ValueError("Overbought must be greater than oversold")
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate TRIX indicator.
        
        Args:
            prices: Array of closing prices
            
        Returns:
            IndicatorResult with TRIX values, signals, and metadata
        """
        if len(prices) < self.period * 3 + self.signal_period:
            raise ValueError(f"Insufficient data. Need at least {self.period * 3 + self.signal_period} points")
        
        # Triple exponential smoothing
        ema1 = self._calculate_ema(prices, self.period)
        ema2 = self._calculate_ema(ema1, self.period)
        ema3 = self._calculate_ema(ema2, self.period)
        
        # Calculate TRIX
        trix = np.zeros_like(prices)
        trix[0] = 0  # First value is 0
        
        for i in range(1, len(prices)):
            if ema3[i-1] != 0:
                trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
            else:
                trix[i] = 0
        
        # Calculate signal line
        signal = self._calculate_ema(trix, self.signal_period)
        
        # Generate signals
        signals = np.full(len(prices), SignalType.HOLD.value)
        signal_strength = np.zeros(len(prices))
        
        for i in range(len(prices)):
            if trix[i] > self.overbought and signal[i] > self.overbought:
                signals[i] = SignalType.STRONG_SELL.value
                signal_strength[i] = min(abs(trix[i] - self.overbought) / self.overbought, 1.0)
            elif trix[i] < self.oversold and signal[i] < self.oversold:
                signals[i] = SignalType.STRONG_BUY.value
                signal_strength[i] = min(abs(trix[i] - self.oversold) / abs(self.oversold), 1.0)
            elif trix[i] > signal[i] and trix[i] > 0:
                signals[i] = SignalType.BUY.value
                signal_strength[i] = min(trix[i] / self.overbought, 1.0)
            elif trix[i] < signal[i] and trix[i] < 0:
                signals[i] = SignalType.SELL.value
                signal_strength[i] = min(abs(trix[i]) / abs(self.oversold), 1.0)
        
        metadata = {
            'indicator_type': 'TRIX',
            'period': self.period,
            'signal_period': self.signal_period,
            'overbought': self.overbought,
            'oversold': self.oversold,
            'calculation_method': 'triple_exponential_smoothing'
        }
        
        return IndicatorResult(
            values=np.column_stack((trix, signal)),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'signal_period': self.signal_period,
                'overbought': self.overbought,
                'oversold': self.oversold
            },
            metadata=metadata
        )
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = 'sharpe_ratio',
                           param_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Optimize TRIX parameters using grid search.
        
        Args:
            prices: Historical price data
            target_metric: Metric to optimize ('sharpe_ratio', 'max_drawdown', 'total_return')
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        if param_ranges is None:
            param_ranges = {
                'period': [10, 15, 20, 25],
                'signal_period': [5, 9, 13, 17],
                'overbought': [0.3, 0.5, 0.7, 1.0],
                'oversold': [-1.0, -0.7, -0.5, -0.3]
            }
        
        best_params = None
        best_score = float('-inf') if target_metric == 'sharpe_ratio' else float('inf')
        
        for period in param_ranges['period']:
            for signal_period in param_ranges['signal_period']:
                for overbought in param_ranges['overbought']:
                    for oversold in param_ranges['oversold']:
                        if overbought <= oversold:
                            continue
                        
                        try:
                            self.period = period
                            self.signal_period = signal_period
                            self.overbought = overbought
                            self.oversold = oversold
                            
                            result = self.calculate(prices)
                            score = self._calculate_performance_metric(result, target_metric)
                            
                            if target_metric == 'sharpe_ratio' and score > best_score:
                                best_score = score
                                best_params = {
                                    'period': period,
                                    'signal_period': signal_period,
                                    'overbought': overbought,
                                    'oversold': oversold
                                }
                            elif target_metric in ['max_drawdown', 'total_return'] and score < best_score:
                                best_score = score
                                best_params = {
                                    'period': period,
                                    'signal_period': signal_period,
                                    'overbought': overbought,
                                    'oversold': oversold
                                }
                        except Exception as e:
                            logger.warning(f"Parameter combination failed: {e}")
                            continue
        
        if best_params:
            self.period = best_params['period']
            self.signal_period = best_params['signal_period']
            self.overbought = best_params['overbought']
            self.oversold = best_params['oversold']
        
        return best_params or {
            'period': 15,
            'signal_period': 9,
            'overbought': 0.5,
            'oversold': -0.5
        }
    
    def _calculate_performance_metric(self, result: IndicatorResult, metric: str) -> float:
        """Calculate performance metric for optimization."""
        if metric == 'sharpe_ratio':
            returns = np.diff(result.values[:, 0])  # TRIX line changes
            if len(returns) == 0:
                return 0.0
            return np.mean(returns) / (np.std(returns) + 1e-8)
        elif metric == 'max_drawdown':
            cumulative = np.cumsum(result.values[:, 0])
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown)
        elif metric == 'total_return':
            return np.sum(result.values[:, 0])
        else:
            return 0.0

class DetrendedPriceOscillator:
    """
    Detrended Price Oscillator (DPO)
    
    DPO is a momentum indicator that removes the trend component from price data
    by subtracting a moving average from the price. It helps identify cycles and
    overbought/oversold conditions independent of the trend.
    
    Formula:
    - DPO = Close - SMA(Close, period/2 + 1)
    - Signal = SMA(DPO, signal_period)
    """
    
    def __init__(self, period: int = 20, signal_period: int = 10,
                 overbought_threshold: float = 2.0, oversold_threshold: float = -2.0):
        """
        Initialize DPO indicator.
        
        Args:
            period: Period for trend calculation (default: 20)
            signal_period: Period for signal line (default: 10)
            overbought_threshold: Overbought threshold in standard deviations (default: 2.0)
            oversold_threshold: Oversold threshold in standard deviations (default: -2.0)
        """
        self.period = period
        self.signal_period = signal_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        
        if period < 2:
            raise ValueError("Period must be at least 2")
        if signal_period < 1:
            raise ValueError("Signal period must be at least 1")
        if overbought_threshold <= oversold_threshold:
            raise ValueError("Overbought threshold must be greater than oversold threshold")
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.zeros_like(data)
        for i in range(len(data)):
            if i < period - 1:
                sma[i] = np.nan
            else:
                sma[i] = np.mean(data[i-period+1:i+1])
        return sma
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate DPO indicator.
        
        Args:
            prices: Array of closing prices
            
        Returns:
            IndicatorResult with DPO values, signals, and metadata
        """
        if len(prices) < self.period + self.signal_period:
            raise ValueError(f"Insufficient data. Need at least {self.period + self.signal_period} points")
        
        # Calculate the shifted moving average period
        shifted_period = self.period // 2 + 1
        
        # Calculate the moving average
        sma = self._calculate_sma(prices, shifted_period)
        
        # Calculate DPO
        dpo = prices - sma
        
        # Calculate signal line
        signal = self._calculate_sma(dpo, self.signal_period)
        
        # Calculate standard deviation for threshold calculation
        dpo_std = np.nanstd(dpo)
        if dpo_std == 0:
            dpo_std = 1.0
        
        # Generate signals
        signals = np.full(len(prices), SignalType.HOLD.value)
        signal_strength = np.zeros(len(prices))
        
        for i in range(len(prices)):
            if np.isnan(dpo[i]) or np.isnan(signal[i]):
                continue
                
            # Normalize by standard deviation
            normalized_dpo = dpo[i] / dpo_std
            normalized_signal = signal[i] / dpo_std
            
            if normalized_dpo > self.overbought_threshold and normalized_signal > self.overbought_threshold:
                signals[i] = SignalType.STRONG_SELL.value
                signal_strength[i] = min((normalized_dpo - self.overbought_threshold) / self.overbought_threshold, 1.0)
            elif normalized_dpo < self.oversold_threshold and normalized_signal < self.oversold_threshold:
                signals[i] = SignalType.STRONG_BUY.value
                signal_strength[i] = min((self.oversold_threshold - normalized_dpo) / abs(self.oversold_threshold), 1.0)
            elif dpo[i] > signal[i] and dpo[i] > 0:
                signals[i] = SignalType.BUY.value
                signal_strength[i] = min(normalized_dpo / self.overbought_threshold, 1.0)
            elif dpo[i] < signal[i] and dpo[i] < 0:
                signals[i] = SignalType.SELL.value
                signal_strength[i] = min(abs(normalized_dpo) / abs(self.oversold_threshold), 1.0)
        
        metadata = {
            'indicator_type': 'DPO',
            'period': self.period,
            'signal_period': self.signal_period,
            'overbought_threshold': self.overbought_threshold,
            'oversold_threshold': self.oversold_threshold,
            'calculation_method': 'detrended_price_oscillator'
        }
        
        return IndicatorResult(
            values=np.column_stack((dpo, signal)),
            signals=signals,
            signal_strength=signal_strength,
            parameters={
                'period': self.period,
                'signal_period': self.signal_period,
                'overbought_threshold': self.overbought_threshold,
                'oversold_threshold': self.oversold_threshold
            },
            metadata=metadata
        )
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = 'sharpe_ratio',
                           param_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Optimize DPO parameters using grid search.
        
        Args:
            prices: Historical price data
            target_metric: Metric to optimize ('sharpe_ratio', 'max_drawdown', 'total_return')
            param_ranges: Parameter ranges to test
            
        Returns:
            Dictionary with optimal parameters
        """
        if param_ranges is None:
            param_ranges = {
                'period': [10, 15, 20, 25, 30],
                'signal_period': [5, 10, 15, 20],
                'overbought_threshold': [1.5, 2.0, 2.5, 3.0],
                'oversold_threshold': [-3.0, -2.5, -2.0, -1.5]
            }
        
        best_params = None
        best_score = float('-inf') if target_metric == 'sharpe_ratio' else float('inf')
        
        for period in param_ranges['period']:
            for signal_period in param_ranges['signal_period']:
                for overbought in param_ranges['overbought_threshold']:
                    for oversold in param_ranges['oversold_threshold']:
                        if overbought <= oversold:
                            continue
                        
                        try:
                            self.period = period
                            self.signal_period = signal_period
                            self.overbought_threshold = overbought
                            self.oversold_threshold = oversold
                            
                            result = self.calculate(prices)
                            score = self._calculate_performance_metric(result, target_metric)
                            
                            if target_metric == 'sharpe_ratio' and score > best_score:
                                best_score = score
                                best_params = {
                                    'period': period,
                                    'signal_period': signal_period,
                                    'overbought_threshold': overbought,
                                    'oversold_threshold': oversold
                                }
                            elif target_metric in ['max_drawdown', 'total_return'] and score < best_score:
                                best_score = score
                                best_params = {
                                    'period': period,
                                    'signal_period': signal_period,
                                    'overbought_threshold': overbought,
                                    'oversold_threshold': oversold
                                }
                        except Exception as e:
                            logger.warning(f"Parameter combination failed: {e}")
                            continue
        
        if best_params:
            self.period = best_params['period']
            self.signal_period = best_params['signal_period']
            self.overbought_threshold = best_params['overbought_threshold']
            self.oversold_threshold = best_params['oversold_threshold']
        
        return best_params or {
            'period': 20,
            'signal_period': 10,
            'overbought_threshold': 2.0,
            'oversold_threshold': -2.0
        }
    
    def _calculate_performance_metric(self, result: IndicatorResult, metric: str) -> float:
        """Calculate performance metric for optimization."""
        if metric == 'sharpe_ratio':
            returns = np.diff(result.values[:, 0])  # DPO line changes
            returns = returns[~np.isnan(returns)]
            if len(returns) == 0:
                return 0.0
            return np.mean(returns) / (np.std(returns) + 1e-8)
        elif metric == 'max_drawdown':
            cumulative = np.cumsum(result.values[:, 0])
            cumulative = cumulative[~np.isnan(cumulative)]
            if len(cumulative) == 0:
                return 0.0
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown)
        elif metric == 'total_return':
            values = result.values[:, 0]
            values = values[~np.isnan(values)]
            return np.sum(values)
        else:
            return 0.0

# Convenience functions for easy usage
def calculate_trix(prices: np.ndarray, period: int = 15, signal_period: int = 9,
                   overbought: float = 0.5, oversold: float = -0.5) -> IndicatorResult:
    """Calculate TRIX with default parameters."""
    trix = TRIX(period=period, signal_period=signal_period, overbought=overbought, oversold=oversold)
    return trix.calculate(prices)

def calculate_dpo(prices: np.ndarray, period: int = 20, signal_period: int = 10,
                  overbought_threshold: float = 2.0, oversold_threshold: float = -2.0) -> IndicatorResult:
    """Calculate DPO with default parameters."""
    dpo = DetrendedPriceOscillator(period=period, signal_period=signal_period,
                                   overbought_threshold=overbought_threshold, oversold_threshold=oversold_threshold)
    return dpo.calculate(prices)