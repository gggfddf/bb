"""
Volume Oscillators Module

Implements volume-based and oscillator technical indicators:
- CCI (Commodity Channel Index)
- ATR (Average True Range)
- MFI (Money Flow Index)
- OBV (On-Balance Volume)
- ROC (Rate of Change)

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

class CCI:
    """
    Commodity Channel Index (CCI) implementation.
    
    CCI measures the current price level relative to an average price level
    over a given period of time.
    """
    
    def __init__(self, period: int = 20, constant: float = 0.015):
        """
        Initialize CCI indicator.
        
        Args:
            period: Lookback period (default: 20)
            constant: CCI constant (default: 0.015)
        """
        self.period = period
        self.constant = constant
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate CCI parameters."""
        if self.period <= 0:
            raise ValueError("CCI period must be positive")
        if self.constant <= 0:
            raise ValueError("CCI constant must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate CCI values.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            
        Returns:
            IndicatorResult with CCI values and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close):
                raise ValueError("All price arrays must have the same length")
            
            if len(high) < self.period:
                raise ValueError(f"Insufficient data for CCI calculation. Need at least {self.period} data points")
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate CCI
            cci_values = self._calculate_cci(typical_price)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(cci_values)
            
            # Create metadata
            metadata = {
                "indicator_type": "CCI",
                "calculation_method": "typical_price_deviation",
                "data_points_used": len(close),
                "cci_min": np.nanmin(cci_values),
                "cci_max": np.nanmax(cci_values),
                "cci_mean": np.nanmean(cci_values)
            }
            
            return IndicatorResult(
                values=cci_values,
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period, "constant": self.constant},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"CCI calculation failed: {e}")
            raise
    
    def _calculate_cci(self, typical_price: np.ndarray) -> np.ndarray:
        """Calculate CCI values."""
        cci = np.full(len(typical_price), np.nan)
        
        for i in range(self.period - 1, len(typical_price)):
            # Calculate SMA of typical price
            sma = np.mean(typical_price[i - self.period + 1:i + 1])
            
            # Calculate mean deviation
            mean_deviation = np.mean(np.abs(typical_price[i - self.period + 1:i + 1] - sma))
            
            # Calculate CCI
            if mean_deviation == 0:
                cci[i] = 0
            else:
                cci[i] = (typical_price[i] - sma) / (self.constant * mean_deviation)
        
        return cci
    
    def _generate_signals(self, cci_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on CCI values."""
        signals = np.full(len(cci_values), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(cci_values))
        
        for i, cci in enumerate(cci_values):
            if np.isnan(cci):
                continue
            
            # Overbought/Oversold signals
            if cci <= -100:
                if cci <= -200:  # Strong oversold
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.7
            elif cci >= 100:
                if cci >= 200:  # Strong overbought
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.7
            
            # Zero line crossover signals
            if i >= 1:
                if cci > 0 and cci_values[i-1] <= 0:  # Bullish crossover
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.6
                elif cci < 0 and cci_values[i-1] >= 0:  # Bearish crossover
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.6
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize CCI parameters using grid search."""
        best_params = None
        best_score = float('-inf')
        
        periods = [10, 14, 20, 30]
        constants = [0.01, 0.015, 0.02, 0.025]
        
        for period in periods:
            for constant in constants:
                try:
                    test_cci = CCI(period=period, constant=constant)
                    result = test_cci.calculate(high, low, close)
                    score = self._calculate_performance_metric(result, close, target_metric)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {"period": period, "constant": constant, "score": score}
                        
                except Exception as e:
                    logger.warning(f"Parameter combination failed: period={period}, constant={constant}, error={e}")
                    continue
        
        if best_params is None:
            best_params = {"period": 20, "constant": 0.015, "score": 0}
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for parameter optimization."""
        try:
            signals = result.signals
            returns = np.diff(prices) / prices[:-1]
            strategy_returns = np.zeros(len(returns))
            position = 0
            
            for i, signal in enumerate(signals[1:]):
                if signal == SignalType.BUY.value or signal == SignalType.STRONG_BUY.value:
                    position = 1
                elif signal == SignalType.SELL.value or signal == SignalType.STRONG_SELL.value:
                    position = 0
                strategy_returns[i] = position * returns[i]
            
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

class ATR:
    """
    Average True Range (ATR) implementation.
    
    ATR measures market volatility by decomposing the entire range
    of an asset price for that period.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ATR indicator.
        
        Args:
            period: Lookback period (default: 14)
        """
        self.period = period
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate ATR parameters."""
        if self.period <= 0:
            raise ValueError("ATR period must be positive")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate ATR values.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            
        Returns:
            IndicatorResult with ATR values and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close):
                raise ValueError("All price arrays must have the same length")
            
            if len(high) < self.period + 1:
                raise ValueError(f"Insufficient data for ATR calculation. Need at least {self.period + 1} data points")
            
            # Calculate True Range
            tr = self._calculate_true_range(high, low, close)
            
            # Calculate ATR using exponential moving average
            atr_values = self._exponential_moving_average(tr, self.period)
            
            # Calculate ATR percentage
            atr_percentage = atr_values / close * 100
            
            # Generate signals
            signals, signal_strength = self._generate_signals(atr_values, atr_percentage)
            
            # Create metadata
            metadata = {
                "indicator_type": "ATR",
                "calculation_method": "exponential_moving_average",
                "data_points_used": len(close),
                "atr_min": np.nanmin(atr_values),
                "atr_max": np.nanmax(atr_values),
                "atr_percentage_min": np.nanmin(atr_percentage),
                "atr_percentage_max": np.nanmax(atr_percentage)
            }
            
            return IndicatorResult(
                values=np.column_stack([atr_values, atr_percentage]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
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
    
    def _exponential_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _generate_signals(self, atr_values: np.ndarray, atr_percentage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on ATR values."""
        signals = np.full(len(atr_values), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(atr_values))
        
        for i, (atr, atr_pct) in enumerate(zip(atr_values, atr_percentage)):
            if np.isnan(atr):
                continue
            
            # Volatility-based signals
            if atr_pct > 5:  # High volatility
                signals[i] = SignalType.HOLD.value  # Avoid trading in high volatility
                signal_strength[i] = 0.0
            elif atr_pct < 1:  # Low volatility
                signals[i] = SignalType.HOLD.value  # Low volatility, no clear signals
                signal_strength[i] = 0.0
            else:  # Normal volatility
                # ATR doesn't provide directional signals, just volatility context
                signals[i] = SignalType.HOLD.value
                signal_strength[i] = 0.5
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize ATR parameters using grid search."""
        best_params = None
        best_score = float('-inf')
        
        periods = [10, 14, 20, 30]
        
        for period in periods:
            try:
                test_atr = ATR(period=period)
                result = test_atr.calculate(high, low, close)
                score = self._calculate_performance_metric(result, close, target_metric)
                
                if score > best_score:
                    best_score = score
                    best_params = {"period": period, "score": score}
                    
            except Exception as e:
                logger.warning(f"Parameter combination failed: period={period}, error={e}")
                continue
        
        if best_params is None:
            best_params = {"period": 14, "score": 0}
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for parameter optimization."""
        try:
            signals = result.signals
            returns = np.diff(prices) / prices[:-1]
            strategy_returns = np.zeros(len(returns))
            position = 0
            
            for i, signal in enumerate(signals[1:]):
                if signal == SignalType.BUY.value or signal == SignalType.STRONG_BUY.value:
                    position = 1
                elif signal == SignalType.SELL.value or signal == SignalType.STRONG_SELL.value:
                    position = 0
                strategy_returns[i] = position * returns[i]
            
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

class MFI:
    """
    Money Flow Index (MFI) implementation.
    
    MFI is a momentum indicator that measures the inflow and outflow
    of money into a security over a specific period of time.
    """
    
    def __init__(self, period: int = 14, overbought: float = 80, oversold: float = 20):
        """
        Initialize MFI indicator.
        
        Args:
            period: Lookback period (default: 14)
            overbought: Overbought threshold (default: 80)
            oversold: Oversold threshold (default: 20)
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate MFI parameters."""
        if self.period <= 0:
            raise ValueError("MFI period must be positive")
        if self.overbought <= self.oversold:
            raise ValueError("Overbought threshold must be greater than oversold threshold")
        if not (0 <= self.oversold <= 100 and 0 <= self.overbought <= 100):
            raise ValueError("MFI thresholds must be between 0 and 100")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 volume: np.ndarray) -> IndicatorResult:
        """
        Calculate MFI values.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            volume: Array of volume data
            
        Returns:
            IndicatorResult with MFI values and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
                raise ValueError("All input arrays must have the same length")
            
            if len(high) < self.period + 1:
                raise ValueError(f"Insufficient data for MFI calculation. Need at least {self.period + 1} data points")
            
            # Calculate MFI
            mfi_values = self._calculate_mfi(high, low, close, volume)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(mfi_values)
            
            # Create metadata
            metadata = {
                "indicator_type": "MFI",
                "calculation_method": "money_flow_ratio",
                "data_points_used": len(close),
                "mfi_min": np.nanmin(mfi_values),
                "mfi_max": np.nanmax(mfi_values),
                "mfi_mean": np.nanmean(mfi_values)
            }
            
            return IndicatorResult(
                values=mfi_values,
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period, "overbought": self.overbought, "oversold": self.oversold},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"MFI calculation failed: {e}")
            raise
    
    def _calculate_mfi(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                      volume: np.ndarray) -> np.ndarray:
        """Calculate MFI values."""
        mfi = np.full(len(close), np.nan)
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * volume
        
        # Calculate positive and negative money flow
        positive_money_flow = np.zeros(len(close))
        negative_money_flow = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if typical_price[i] > typical_price[i-1]:
                positive_money_flow[i] = raw_money_flow[i]
            elif typical_price[i] < typical_price[i-1]:
                negative_money_flow[i] = raw_money_flow[i]
        
        # Calculate MFI
        for i in range(self.period, len(close)):
            positive_sum = np.sum(positive_money_flow[i - self.period + 1:i + 1])
            negative_sum = np.sum(negative_money_flow[i - self.period + 1:i + 1])
            
            if negative_sum == 0:
                mfi[i] = 100
            else:
                money_ratio = positive_sum / negative_sum
                mfi[i] = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def _generate_signals(self, mfi_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on MFI values."""
        signals = np.full(len(mfi_values), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(mfi_values))
        
        for i, mfi in enumerate(mfi_values):
            if np.isnan(mfi):
                continue
            
            # Overbought/Oversold signals
            if mfi <= self.oversold:
                if mfi <= self.oversold - 10:  # Strong oversold
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.7
            elif mfi >= self.overbought:
                if mfi >= self.overbought + 10:  # Strong overbought
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.7
            
            # Divergence signals (simplified)
            if i >= 5:
                recent_mfi = mfi_values[i-5:i+1]
                if np.all(recent_mfi < 30):  # Extended oversold
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                elif np.all(recent_mfi > 70):  # Extended overbought
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          volume: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize MFI parameters using grid search."""
        best_params = None
        best_score = float('-inf')
        
        periods = [10, 14, 20, 30]
        overbought_levels = [70, 80, 90]
        oversold_levels = [10, 20, 30]
        
        for period in periods:
            for overbought in overbought_levels:
                for oversold in oversold_levels:
                    if overbought > oversold:
                        try:
                            test_mfi = MFI(period=period, overbought=overbought, oversold=oversold)
                            result = test_mfi.calculate(high, low, close, volume)
                            score = self._calculate_performance_metric(result, close, target_metric)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    "period": period,
                                    "overbought": overbought,
                                    "oversold": oversold,
                                    "score": score
                                }
                                
                        except Exception as e:
                            logger.warning(f"Parameter combination failed: period={period}, overbought={overbought}, oversold={oversold}, error={e}")
                            continue
        
        if best_params is None:
            best_params = {"period": 14, "overbought": 80, "oversold": 20, "score": 0}
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for parameter optimization."""
        try:
            signals = result.signals
            returns = np.diff(prices) / prices[:-1]
            strategy_returns = np.zeros(len(returns))
            position = 0
            
            for i, signal in enumerate(signals[1:]):
                if signal == SignalType.BUY.value or signal == SignalType.STRONG_BUY.value:
                    position = 1
                elif signal == SignalType.SELL.value or signal == SignalType.STRONG_SELL.value:
                    position = 0
                strategy_returns[i] = position * returns[i]
            
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

class OBV:
    """
    On-Balance Volume (OBV) implementation.
    
    OBV measures buying and selling pressure as a cumulative indicator
    that adds volume on up days and subtracts volume on down days.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize OBV indicator.
        
        Args:
            period: Lookback period for OBV analysis (default: 20)
        """
        self.period = period
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate OBV parameters."""
        if self.period <= 0:
            raise ValueError("OBV period must be positive")
    
    def calculate(self, close: np.ndarray, volume: np.ndarray) -> IndicatorResult:
        """
        Calculate OBV values.
        
        Args:
            close: Array of close prices
            volume: Array of volume data
            
        Returns:
            IndicatorResult with OBV values and signals
        """
        try:
            if len(close) != len(volume):
                raise ValueError("Close and volume arrays must have the same length")
            
            if len(close) < 2:
                raise ValueError("Insufficient data for OBV calculation")
            
            # Calculate OBV
            obv_values = self._calculate_obv(close, volume)
            
            # Calculate OBV rate of change
            obv_roc = self._calculate_rate_of_change(obv_values)
            
            # Calculate OBV moving average
            obv_sma = self._simple_moving_average(obv_values, self.period)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(obv_values, obv_roc, obv_sma)
            
            # Create metadata
            metadata = {
                "indicator_type": "OBV",
                "calculation_method": "cumulative_volume",
                "data_points_used": len(close),
                "obv_min": np.nanmin(obv_values),
                "obv_max": np.nanmax(obv_values),
                "obv_roc_min": np.nanmin(obv_roc),
                "obv_roc_max": np.nanmax(obv_roc)
            }
            
            return IndicatorResult(
                values=np.column_stack([obv_values, obv_roc, obv_sma]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"OBV calculation failed: {e}")
            raise
    
    def _calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate OBV values."""
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def _calculate_rate_of_change(self, obv_values: np.ndarray) -> np.ndarray:
        """Calculate rate of change of OBV."""
        roc = np.full(len(obv_values), np.nan)
        
        for i in range(1, len(obv_values)):
            if obv_values[i-1] != 0:
                roc[i] = ((obv_values[i] - obv_values[i-1]) / obv_values[i-1]) * 100
            else:
                roc[i] = 0
        
        return roc
    
    def _simple_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        sma = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        return sma
    
    def _generate_signals(self, obv_values: np.ndarray, obv_roc: np.ndarray, 
                         obv_sma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on OBV values."""
        signals = np.full(len(obv_values), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(obv_values))
        
        for i in range(1, len(obv_values)):
            if np.isnan(obv_sma[i]):
                continue
            
            # OBV trend signals
            if obv_values[i] > obv_sma[i] and obv_values[i-1] <= obv_sma[i-1]:
                # OBV crosses above its moving average
                signals[i] = SignalType.BUY.value
                signal_strength[i] = 0.7
            elif obv_values[i] < obv_sma[i] and obv_values[i-1] >= obv_sma[i-1]:
                # OBV crosses below its moving average
                signals[i] = SignalType.SELL.value
                signal_strength[i] = 0.7
            
            # OBV rate of change signals
            if not np.isnan(obv_roc[i]):
                if obv_roc[i] > 5:  # Strong positive momentum
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.6
                elif obv_roc[i] < -5:  # Strong negative momentum
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.6
            
            # Divergence signals (simplified)
            if i >= 10:
                recent_obv = obv_values[i-10:i+1]
                if np.all(np.diff(recent_obv) > 0):  # Consistent upward trend
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.5
                elif np.all(np.diff(recent_obv) < 0):  # Consistent downward trend
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.5
        
        return signals, signal_strength
    
    def optimize_parameters(self, close: np.ndarray, volume: np.ndarray,
                          target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize OBV parameters using grid search."""
        best_params = None
        best_score = float('-inf')
        
        periods = [10, 15, 20, 25, 30]
        
        for period in periods:
            try:
                test_obv = OBV(period=period)
                result = test_obv.calculate(close, volume)
                score = self._calculate_performance_metric(result, close, target_metric)
                
                if score > best_score:
                    best_score = score
                    best_params = {"period": period, "score": score}
                    
            except Exception as e:
                logger.warning(f"Parameter combination failed: period={period}, error={e}")
                continue
        
        if best_params is None:
            best_params = {"period": 20, "score": 0}
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for parameter optimization."""
        try:
            signals = result.signals
            returns = np.diff(prices) / prices[:-1]
            strategy_returns = np.zeros(len(returns))
            position = 0
            
            for i, signal in enumerate(signals[1:]):
                if signal == SignalType.BUY.value or signal == SignalType.STRONG_BUY.value:
                    position = 1
                elif signal == SignalType.SELL.value or signal == SignalType.STRONG_SELL.value:
                    position = 0
                strategy_returns[i] = position * returns[i]
            
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

class ROC:
    """
    Rate of Change (ROC) implementation.
    
    ROC measures the percentage change in price over a specified period,
    showing momentum by comparing the current price to a price n periods ago.
    """
    
    def __init__(self, period: int = 10):
        """
        Initialize ROC indicator.
        
        Args:
            period: Lookback period (default: 10)
        """
        self.period = period
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate ROC parameters."""
        if self.period <= 0:
            raise ValueError("ROC period must be positive")
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate ROC values.
        
        Args:
            prices: Array of price data (OHLC or close prices)
            
        Returns:
            IndicatorResult with ROC values and signals
        """
        try:
            if len(prices) < self.period + 1:
                raise ValueError(f"Insufficient data for ROC calculation. Need at least {self.period + 1} data points")
            
            # Calculate ROC
            roc_values = self._calculate_roc(prices)
            
            # Calculate ROC moving average
            roc_sma = self._simple_moving_average(roc_values, self.period)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(roc_values, roc_sma)
            
            # Create metadata
            metadata = {
                "indicator_type": "ROC",
                "calculation_method": "percentage_change",
                "data_points_used": len(prices),
                "roc_min": np.nanmin(roc_values),
                "roc_max": np.nanmax(roc_values),
                "roc_mean": np.nanmean(roc_values)
            }
            
            return IndicatorResult(
                values=np.column_stack([roc_values, roc_sma]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"ROC calculation failed: {e}")
            raise
    
    def _calculate_roc(self, prices: np.ndarray) -> np.ndarray:
        """Calculate ROC values."""
        roc = np.full(len(prices), np.nan)
        
        for i in range(self.period, len(prices)):
            if prices[i - self.period] != 0:
                roc[i] = ((prices[i] - prices[i - self.period]) / prices[i - self.period]) * 100
            else:
                roc[i] = 0
        
        return roc
    
    def _simple_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        sma = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        return sma
    
    def _generate_signals(self, roc_values: np.ndarray, roc_sma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on ROC values."""
        signals = np.full(len(roc_values), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(roc_values))
        
        for i, (roc, roc_ma) in enumerate(zip(roc_values, roc_sma)):
            if np.isnan(roc) or np.isnan(roc_ma):
                continue
            
            # ROC level signals
            if roc > 10:  # Strong positive momentum
                signals[i] = SignalType.STRONG_BUY.value
                signal_strength[i] = 1.0
            elif roc > 5:  # Positive momentum
                signals[i] = SignalType.BUY.value
                signal_strength[i] = 0.7
            elif roc < -10:  # Strong negative momentum
                signals[i] = SignalType.STRONG_SELL.value
                signal_strength[i] = 1.0
            elif roc < -5:  # Negative momentum
                signals[i] = SignalType.SELL.value
                signal_strength[i] = 0.7
            
            # ROC crossover signals
            if i >= 1:
                if roc > roc_ma and roc_values[i-1] <= roc_sma[i-1]:
                    # ROC crosses above its moving average
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                elif roc < roc_ma and roc_values[i-1] >= roc_sma[i-1]:
                    # ROC crosses below its moving average
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
            
            # Zero line crossover signals
            if i >= 1:
                if roc > 0 and roc_values[i-1] <= 0:
                    # ROC crosses above zero
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.6
                elif roc < 0 and roc_values[i-1] >= 0:
                    # ROC crosses below zero
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.6
        
        return signals, signal_strength
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Optimize ROC parameters using grid search."""
        best_params = None
        best_score = float('-inf')
        
        periods = [5, 10, 15, 20, 30]
        
        for period in periods:
            try:
                test_roc = ROC(period=period)
                result = test_roc.calculate(prices)
                score = self._calculate_performance_metric(result, prices, target_metric)
                
                if score > best_score:
                    best_score = score
                    best_params = {"period": period, "score": score}
                    
            except Exception as e:
                logger.warning(f"Parameter combination failed: period={period}, error={e}")
                continue
        
        if best_params is None:
            best_params = {"period": 10, "score": 0}
        
        return best_params
    
    def _calculate_performance_metric(self, result: IndicatorResult, prices: np.ndarray, metric: str) -> float:
        """Calculate performance metric for parameter optimization."""
        try:
            signals = result.signals
            returns = np.diff(prices) / prices[:-1]
            strategy_returns = np.zeros(len(returns))
            position = 0
            
            for i, signal in enumerate(signals[1:]):
                if signal == SignalType.BUY.value or signal == SignalType.STRONG_BUY.value:
                    position = 1
                elif signal == SignalType.SELL.value or signal == SignalType.STRONG_SELL.value:
                    position = 0
                strategy_returns[i] = position * returns[i]
            
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
def calculate_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 period: int = 20, constant: float = 0.015) -> IndicatorResult:
    """Calculate CCI with default parameters."""
    cci = CCI(period=period, constant=constant)
    return cci.calculate(high, low, close)

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> IndicatorResult:
    """Calculate ATR with default parameters."""
    atr = ATR(period=period)
    return atr.calculate(high, low, close)

def calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 volume: np.ndarray, period: int = 14, overbought: float = 80, oversold: float = 20) -> IndicatorResult:
    """Calculate MFI with default parameters."""
    mfi = MFI(period=period, overbought=overbought, oversold=oversold)
    return mfi.calculate(high, low, close, volume)

def calculate_obv(close: np.ndarray, volume: np.ndarray, period: int = 20) -> IndicatorResult:
    """Calculate OBV with default parameters."""
    obv = OBV(period=period)
    return obv.calculate(close, volume)

def calculate_roc(prices: np.ndarray, period: int = 10) -> IndicatorResult:
    """Calculate ROC with default parameters."""
    roc = ROC(period=period)
    return roc.calculate(prices)