"""
Momentum Indicators Module

Implements key momentum-based technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator

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

class RSI:
    """
    Relative Strength Index (RSI) implementation.
    
    RSI measures the speed and magnitude of price changes to identify
    overbought or oversold conditions.
    """
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Initialize RSI indicator.
        
        Args:
            period: Lookback period for RSI calculation (default: 14)
            overbought: Overbought threshold (default: 70)
            oversold: Oversold threshold (default: 30)
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate RSI parameters."""
        if self.period <= 0:
            raise ValueError("RSI period must be positive")
        if self.overbought <= self.oversold:
            raise ValueError("Overbought threshold must be greater than oversold threshold")
        if not (0 <= self.oversold <= 100 and 0 <= self.overbought <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate RSI values.
        
        Args:
            prices: Array of price data (OHLC or close prices)
            
        Returns:
            IndicatorResult with RSI values and signals
        """
        try:
            if len(prices) < self.period + 1:
                raise ValueError(f"Insufficient data for RSI calculation. Need at least {self.period + 1} data points")
            
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            
            # Calculate average gains and losses using exponential moving average
            avg_gains = self._exponential_moving_average(gains, self.period)
            avg_losses = self._exponential_moving_average(losses, self.period)
            
            # Calculate RS and RSI
            rs = avg_gains / np.where(avg_losses == 0, 1e-10, avg_losses)  # Avoid division by zero
            rsi_values = 100 - (100 / (1 + rs))
            
            # Generate signals
            signals, signal_strength = self._generate_signals(rsi_values)
            
            # Create metadata
            metadata = {
                "indicator_type": "RSI",
                "calculation_method": "exponential_moving_average",
                "data_points_used": len(prices),
                "min_value": np.nanmin(rsi_values),
                "max_value": np.nanmax(rsi_values),
                "mean_value": np.nanmean(rsi_values)
            }
            
            return IndicatorResult(
                values=rsi_values,
                signals=signals,
                signal_strength=signal_strength,
                parameters={"period": self.period, "overbought": self.overbought, "oversold": self.oversold},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            raise
    
    def _exponential_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _generate_signals(self, rsi_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on RSI values."""
        signals = np.full(len(rsi_values), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(rsi_values))
        
        for i, rsi in enumerate(rsi_values):
            if np.isnan(rsi):
                continue
                
            if rsi <= self.oversold:
                if rsi <= self.oversold - 10:  # Strong oversold
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.7
            elif rsi >= self.overbought:
                if rsi >= self.overbought + 10:  # Strong overbought
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.7
        
        return signals, signal_strength
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize RSI parameters using grid search.
        
        Args:
            prices: Price data for optimization
            target_metric: Metric to optimize for ("sharpe_ratio", "max_drawdown", "total_return")
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = float('-inf')
        
        # Parameter ranges for optimization
        periods = [7, 10, 14, 21, 30]
        overbought_levels = [65, 70, 75, 80]
        oversold_levels = [20, 25, 30, 35]
        
        for period in periods:
            for overbought in overbought_levels:
                for oversold in oversold_levels:
                    if overbought > oversold:
                        try:
                            # Create RSI instance with test parameters
                            test_rsi = RSI(period=period, overbought=overbought, oversold=oversold)
                            result = test_rsi.calculate(prices)
                            
                            # Calculate performance metric
                            score = self._calculate_performance_metric(result, prices, target_metric)
                            
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
            logger.warning("No valid parameter combination found, using defaults")
            best_params = {"period": 14, "overbought": 70, "oversold": 30, "score": 0}
        
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

class MACD:
    """
    Moving Average Convergence Divergence (MACD) implementation.
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate MACD parameters."""
        if self.fast_period <= 0 or self.slow_period <= 0 or self.signal_period <= 0:
            raise ValueError("All MACD periods must be positive")
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
    
    def calculate(self, prices: np.ndarray) -> IndicatorResult:
        """
        Calculate MACD values.
        
        Args:
            prices: Array of price data (OHLC or close prices)
            
        Returns:
            IndicatorResult with MACD values and signals
        """
        try:
            if len(prices) < self.slow_period + self.signal_period:
                raise ValueError(f"Insufficient data for MACD calculation. Need at least {self.slow_period + self.signal_period} data points")
            
            # Calculate EMAs
            fast_ema = self._exponential_moving_average(prices, self.fast_period)
            slow_ema = self._exponential_moving_average(prices, self.slow_period)
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = self._exponential_moving_average(macd_line, self.signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Generate signals
            signals, signal_strength = self._generate_signals(macd_line, signal_line, histogram)
            
            # Create metadata
            metadata = {
                "indicator_type": "MACD",
                "calculation_method": "exponential_moving_average",
                "data_points_used": len(prices),
                "macd_min": np.nanmin(macd_line),
                "macd_max": np.nanmax(macd_line),
                "signal_min": np.nanmin(signal_line),
                "signal_max": np.nanmax(signal_line)
            }
            
            return IndicatorResult(
                values=np.column_stack([macd_line, signal_line, histogram]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={
                    "fast_period": self.fast_period,
                    "slow_period": self.slow_period,
                    "signal_period": self.signal_period
                },
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            raise
    
    def _exponential_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _generate_signals(self, macd_line: np.ndarray, signal_line: np.ndarray, histogram: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on MACD values."""
        signals = np.full(len(macd_line), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(macd_line))
        
        for i in range(1, len(macd_line)):
            if np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
                continue
            
            # MACD crossover signals
            if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                # Bullish crossover
                if histogram[i] > 0:
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.7
            elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                # Bearish crossover
                if histogram[i] < 0:
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.7
            
            # Divergence signals
            if i >= 2:
                # Bullish divergence
                if (macd_line[i] > macd_line[i-1] and 
                    macd_line[i-1] < macd_line[i-2] and 
                    histogram[i] > histogram[i-1]):
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                
                # Bearish divergence
                elif (macd_line[i] < macd_line[i-1] and 
                      macd_line[i-1] > macd_line[i-2] and 
                      histogram[i] < histogram[i-1]):
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
        
        return signals, signal_strength
    
    def optimize_parameters(self, prices: np.ndarray, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize MACD parameters using grid search.
        
        Args:
            prices: Price data for optimization
            target_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = None
        best_score = float('-inf')
        
        # Parameter ranges for optimization
        fast_periods = [8, 10, 12, 14]
        slow_periods = [21, 26, 30, 34]
        signal_periods = [7, 9, 11, 13]
        
        for fast_period in fast_periods:
            for slow_period in slow_periods:
                if fast_period < slow_period:
                    for signal_period in signal_periods:
                        try:
                            # Create MACD instance with test parameters
                            test_macd = MACD(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
                            result = test_macd.calculate(prices)
                            
                            # Calculate performance metric
                            score = self._calculate_performance_metric(result, prices, target_metric)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    "fast_period": fast_period,
                                    "slow_period": slow_period,
                                    "signal_period": signal_period,
                                    "score": score
                                }
                                
                        except Exception as e:
                            logger.warning(f"Parameter combination failed: fast={fast_period}, slow={slow_period}, signal={signal_period}, error={e}")
                            continue
        
        if best_params is None:
            logger.warning("No valid parameter combination found, using defaults")
            best_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9, "score": 0}
        
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

class StochasticOscillator:
    """
    Stochastic Oscillator implementation.
    
    The Stochastic Oscillator is a momentum indicator that compares a closing price
    to its price range over a specific period of time.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, slowing: int = 3, 
                 overbought: float = 80, oversold: float = 20):
        """
        Initialize Stochastic Oscillator.
        
        Args:
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            slowing: Smoothing period for %K (default: 3)
            overbought: Overbought threshold (default: 80)
            oversold: Oversold threshold (default: 20)
        """
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing
        self.overbought = overbought
        self.oversold = oversold
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Stochastic Oscillator parameters."""
        if any(p <= 0 for p in [self.k_period, self.d_period, self.slowing]):
            raise ValueError("All periods must be positive")
        if self.overbought <= self.oversold:
            raise ValueError("Overbought threshold must be greater than oversold threshold")
        if not (0 <= self.oversold <= 100 and 0 <= self.overbought <= 100):
            raise ValueError("Thresholds must be between 0 and 100")
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Stochastic Oscillator values.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            
        Returns:
            IndicatorResult with Stochastic values and signals
        """
        try:
            if len(high) != len(low) or len(high) != len(close):
                raise ValueError("All price arrays must have the same length")
            
            if len(high) < self.k_period + self.slowing:
                raise ValueError(f"Insufficient data for Stochastic calculation. Need at least {self.k_period + self.slowing} data points")
            
            # Calculate %K
            k_percent = self._calculate_k_percent(high, low, close)
            
            # Smooth %K
            k_smoothed = self._simple_moving_average(k_percent, self.slowing)
            
            # Calculate %D
            d_percent = self._simple_moving_average(k_smoothed, self.d_period)
            
            # Generate signals
            signals, signal_strength = self._generate_signals(k_smoothed, d_percent)
            
            # Create metadata
            metadata = {
                "indicator_type": "Stochastic_Oscillator",
                "calculation_method": "simple_moving_average",
                "data_points_used": len(close),
                "k_min": np.nanmin(k_smoothed),
                "k_max": np.nanmax(k_smoothed),
                "d_min": np.nanmin(d_percent),
                "d_max": np.nanmax(d_percent)
            }
            
            return IndicatorResult(
                values=np.column_stack([k_smoothed, d_percent]),
                signals=signals,
                signal_strength=signal_strength,
                parameters={
                    "k_period": self.k_period,
                    "d_period": self.d_period,
                    "slowing": self.slowing,
                    "overbought": self.overbought,
                    "oversold": self.oversold
                },
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Stochastic Oscillator calculation failed: {e}")
            raise
    
    def _calculate_k_percent(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate raw %K values."""
        k_percent = np.zeros(len(close))
        
        for i in range(self.k_period - 1, len(close)):
            highest_high = np.max(high[i - self.k_period + 1:i + 1])
            lowest_low = np.min(low[i - self.k_period + 1:i + 1])
            
            if highest_high == lowest_low:
                k_percent[i] = 50  # Neutral when range is zero
            else:
                k_percent[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
        
        return k_percent
    
    def _simple_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        sma = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        return sma
    
    def _generate_signals(self, k_percent: np.ndarray, d_percent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading signals based on Stochastic values."""
        signals = np.full(len(k_percent), SignalType.HOLD.value, dtype=object)
        signal_strength = np.zeros(len(k_percent))
        
        for i in range(1, len(k_percent)):
            if np.isnan(k_percent[i]) or np.isnan(d_percent[i]):
                continue
            
            # Overbought/Oversold signals
            if k_percent[i] <= self.oversold and d_percent[i] <= self.oversold:
                if k_percent[i] <= self.oversold - 10:  # Strong oversold
                    signals[i] = SignalType.STRONG_BUY.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.7
            elif k_percent[i] >= self.overbought and d_percent[i] >= self.overbought:
                if k_percent[i] >= self.overbought + 10:  # Strong overbought
                    signals[i] = SignalType.STRONG_SELL.value
                    signal_strength[i] = 1.0
                else:
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.7
            
            # Crossover signals
            if i >= 1:
                # Bullish crossover
                if (k_percent[i] > d_percent[i] and 
                    k_percent[i-1] <= d_percent[i-1] and
                    k_percent[i] < 50):  # Below midpoint
                    signals[i] = SignalType.BUY.value
                    signal_strength[i] = 0.8
                
                # Bearish crossover
                elif (k_percent[i] < d_percent[i] and 
                      k_percent[i-1] >= d_percent[i-1] and
                      k_percent[i] > 50):  # Above midpoint
                    signals[i] = SignalType.SELL.value
                    signal_strength[i] = 0.8
        
        return signals, signal_strength
    
    def optimize_parameters(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                          target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize Stochastic Oscillator parameters using grid search.
        
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
        k_periods = [10, 14, 20, 30]
        d_periods = [3, 5, 7]
        slowing_periods = [1, 3, 5]
        overbought_levels = [70, 80, 90]
        oversold_levels = [10, 20, 30]
        
        for k_period in k_periods:
            for d_period in d_periods:
                for slowing in slowing_periods:
                    for overbought in overbought_levels:
                        for oversold in oversold_levels:
                            if overbought > oversold:
                                try:
                                    # Create Stochastic instance with test parameters
                                    test_stoch = StochasticOscillator(
                                        k_period=k_period,
                                        d_period=d_period,
                                        slowing=slowing,
                                        overbought=overbought,
                                        oversold=oversold
                                    )
                                    result = test_stoch.calculate(high, low, close)
                                    
                                    # Calculate performance metric
                                    score = self._calculate_performance_metric(result, close, target_metric)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_params = {
                                            "k_period": k_period,
                                            "d_period": d_period,
                                            "slowing": slowing,
                                            "overbought": overbought,
                                            "oversold": oversold,
                                            "score": score
                                        }
                                        
                                except Exception as e:
                                    logger.warning(f"Parameter combination failed: k={k_period}, d={d_period}, slowing={slowing}, overbought={overbought}, oversold={oversold}, error={e}")
                                    continue
        
        if best_params is None:
            logger.warning("No valid parameter combination found, using defaults")
            best_params = {
                "k_period": 14, "d_period": 3, "slowing": 3,
                "overbought": 80, "oversold": 20, "score": 0
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

# Convenience functions for easy usage
def calculate_rsi(prices: np.ndarray, period: int = 14, overbought: float = 70, oversold: float = 30) -> IndicatorResult:
    """Calculate RSI with default parameters."""
    rsi = RSI(period=period, overbought=overbought, oversold=oversold)
    return rsi.calculate(prices)

def calculate_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> IndicatorResult:
    """Calculate MACD with default parameters."""
    macd = MACD(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
    return macd.calculate(prices)

def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                        k_period: int = 14, d_period: int = 3, slowing: int = 3,
                        overbought: float = 80, oversold: float = 20) -> IndicatorResult:
    """Calculate Stochastic Oscillator with default parameters."""
    stoch = StochasticOscillator(k_period=k_period, d_period=d_period, slowing=slowing,
                                overbought=overbought, oversold=oversold)
    return stoch.calculate(high, low, close)