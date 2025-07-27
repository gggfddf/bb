"""
Single Indicator Strategy Testing Framework

A comprehensive framework for testing trading strategies based on single indicators
including RSI, MACD, Bollinger Bands, Moving Averages, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime
import warnings

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = structlog.get_logger()

class IndicatorType(Enum):
    """Supported indicator types."""
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    SMA = "sma"
    EMA = "ema"
    STOCHASTIC = "stochastic"
    CCI = "cci"
    ATR = "atr"

class SignalType(Enum):
    """Signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class StrategyConfig:
    """Configuration for single indicator strategy."""
    indicator_type: IndicatorType
    parameters: Dict[str, Any]
    signal_thresholds: Dict[str, float]
    position_sizing: str = "fixed"  # "fixed", "kelly", "volatility"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_position_size: float = 1.0

@dataclass
class StrategyResult:
    """Results from strategy backtesting."""
    returns: np.ndarray
    positions: np.ndarray
    signals: np.ndarray
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    config: StrategyConfig
    performance_metrics: Dict[str, float]

class SingleIndicatorStrategy:
    """Base class for single indicator strategies."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.positions = []
        self.signals = []
        self.returns = []
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Generate trading signals based on indicator."""
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def calculate_position_size(self, signal: float, price: float, 
                              volatility: Optional[float] = None) -> float:
        """Calculate position size based on signal and configuration."""
        if self.config.position_sizing == "fixed":
            return self.config.max_position_size * signal
        elif self.config.position_sizing == "kelly" and volatility:
            # Simplified Kelly criterion
            kelly_fraction = signal / volatility if volatility > 0 else 0
            return np.clip(kelly_fraction, -self.config.max_position_size, self.config.max_position_size)
        elif self.config.position_sizing == "volatility" and volatility:
            # Volatility-adjusted position sizing
            vol_adjusted = self.config.max_position_size / (1 + volatility)
            return vol_adjusted * signal
        else:
            return self.config.max_position_size * signal

class RSIStrategy(SingleIndicatorStrategy):
    """RSI-based trading strategy."""
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Generate signals based on RSI."""
        rsi = data['rsi'].values
        signals = np.zeros(len(rsi))
        
        oversold = self.config.parameters.get('oversold', 30)
        overbought = self.config.parameters.get('overbought', 70)
        
        # Buy signal: RSI crosses above oversold threshold
        buy_signal = (rsi > oversold) & (np.roll(rsi, 1) <= oversold)
        signals[buy_signal] = 1
        
        # Sell signal: RSI crosses below overbought threshold
        sell_signal = (rsi < overbought) & (np.roll(rsi, 1) >= overbought)
        signals[sell_signal] = -1
        
        return signals

class MACDStrategy(SingleIndicatorStrategy):
    """MACD-based trading strategy."""
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Generate signals based on MACD."""
        macd = data['macd'].values
        macd_signal = data['macd_signal'].values
        macd_histogram = data['macd_histogram'].values
        
        signals = np.zeros(len(macd))
        
        # Buy signal: MACD crosses above signal line
        buy_signal = (macd > macd_signal) & (np.roll(macd, 1) <= np.roll(macd_signal, 1))
        signals[buy_signal] = 1
        
        # Sell signal: MACD crosses below signal line
        sell_signal = (macd < macd_signal) & (np.roll(macd, 1) >= np.roll(macd_signal, 1))
        signals[sell_signal] = -1
        
        return signals

class BollingerBandsStrategy(SingleIndicatorStrategy):
    """Bollinger Bands-based trading strategy."""
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Generate signals based on Bollinger Bands."""
        price = data['close'].values
        bb_upper = data['bb_upper'].values
        bb_lower = data['bb_lower'].values
        
        signals = np.zeros(len(price))
        
        # Buy signal: price touches lower band
        buy_signal = price <= bb_lower
        signals[buy_signal] = 1
        
        # Sell signal: price touches upper band
        sell_signal = price >= bb_upper
        signals[sell_signal] = -1
        
        return signals

class MovingAverageStrategy(SingleIndicatorStrategy):
    """Moving Average-based trading strategy."""
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Generate signals based on Moving Average crossover."""
        price = data['close'].values
        ma_short = data['ma_short'].values
        ma_long = data['ma_long'].values
        
        signals = np.zeros(len(price))
        
        # Buy signal: short MA crosses above long MA
        buy_signal = (ma_short > ma_long) & (np.roll(ma_short, 1) <= np.roll(ma_long, 1))
        signals[buy_signal] = 1
        
        # Sell signal: short MA crosses below long MA
        sell_signal = (ma_short < ma_long) & (np.roll(ma_short, 1) >= np.roll(ma_long, 1))
        signals[sell_signal] = -1
        
        return signals

class StrategyTester:
    """Framework for testing single indicator strategies."""
    
    def __init__(self):
        self.strategies = {}
        self.results = {}
    
    def add_strategy(self, name: str, strategy: SingleIndicatorStrategy):
        """Add a strategy to the tester."""
        self.strategies[name] = strategy
        logger.info(f"Added strategy: {name}")
    
    def backtest_strategy(self, strategy: SingleIndicatorStrategy, 
                         data: pd.DataFrame, initial_capital: float = 10000) -> StrategyResult:
        """Backtest a single strategy."""
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Calculate returns
        price_returns = data['close'].pct_change().values
        price_returns[0] = 0  # First return is 0
        
        # Calculate strategy returns
        strategy_returns = signals * price_returns
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        
        # Calculate performance metrics
        total_return = cumulative_returns[-1]
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(1 + cumulative_returns)
        drawdown = (1 + cumulative_returns) / peak - 1
        max_drawdown = np.min(drawdown)
        
        # Calculate win rate
        winning_trades = strategy_returns > 0
        win_rate = np.sum(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        # Calculate profit factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        performance_metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'volatility': np.std(strategy_returns),
            'avg_return': np.mean(strategy_returns)
        }
        
        result = StrategyResult(
            returns=strategy_returns,
            positions=signals,
            signals=signals,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            config=strategy.config,
            performance_metrics=performance_metrics
        )
        
        return result
    
    def test_all_strategies(self, data: pd.DataFrame) -> Dict[str, StrategyResult]:
        """Test all strategies on the same dataset."""
        results = {}
        
        for name, strategy in self.strategies.items():
            logger.info(f"Testing strategy: {name}")
            result = self.backtest_strategy(strategy, data)
            results[name] = result
        
        self.results = results
        return results
    
    def optimize_parameters(self, strategy_class: type, data: pd.DataFrame, 
                          param_grid: Dict[str, List], metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Optimize strategy parameters using grid search."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for parameter optimization")
            return {}
        
        best_params = {}
        best_score = float('-inf')
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        from itertools import product
        param_combinations = list(product(*param_values))
        
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            # Create strategy with these parameters
            config = StrategyConfig(
                indicator_type=strategy_class.__name__.replace('Strategy', '').lower(),
                parameters=param_dict,
                signal_thresholds={}
            )
            
            strategy = strategy_class(config)
            result = self.backtest_strategy(strategy, data)
            
            score = result.performance_metrics.get(metric, 0)
            
            if score > best_score:
                best_score = score
                best_params = param_dict
        
        logger.info(f"Best parameters: {best_params}, Best {metric}: {best_score}")
        return best_params
    
    def cross_validate_strategy(self, strategy: SingleIndicatorStrategy, 
                              data: pd.DataFrame, n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform time series cross-validation on a strategy."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for cross-validation")
            return {}
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {
            'sharpe_ratio': [],
            'total_return': [],
            'max_drawdown': [],
            'win_rate': []
        }
        
        for train_idx, test_idx in tscv.split(data):
            test_data = data.iloc[test_idx]
            result = self.backtest_strategy(strategy, test_data)
            
            cv_results['sharpe_ratio'].append(result.sharpe_ratio)
            cv_results['total_return'].append(result.total_return)
            cv_results['max_drawdown'].append(result.max_drawdown)
            cv_results['win_rate'].append(result.win_rate)
        
        return cv_results
    
    def compare_strategies(self) -> pd.DataFrame:
        """Compare performance of all tested strategies."""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, result in self.results.items():
            comparison_data.append({
                'strategy': name,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'volatility': result.performance_metrics['volatility']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
        
        return comparison_df
    
    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> Tuple[str, StrategyResult]:
        """Get the best performing strategy based on a metric."""
        if not self.results:
            return None, None
        
        best_strategy = max(self.results.items(), 
                          key=lambda x: x[1].performance_metrics.get(metric, 0))
        
        return best_strategy

# Convenience functions
def create_rsi_strategy(oversold: int = 30, overbought: int = 70) -> RSIStrategy:
    """Create an RSI strategy."""
    config = StrategyConfig(
        indicator_type=IndicatorType.RSI,
        parameters={'oversold': oversold, 'overbought': overbought},
        signal_thresholds={}
    )
    return RSIStrategy(config)

def create_macd_strategy() -> MACDStrategy:
    """Create a MACD strategy."""
    config = StrategyConfig(
        indicator_type=IndicatorType.MACD,
        parameters={},
        signal_thresholds={}
    )
    return MACDStrategy(config)

def create_bollinger_bands_strategy() -> BollingerBandsStrategy:
    """Create a Bollinger Bands strategy."""
    config = StrategyConfig(
        indicator_type=IndicatorType.BOLLINGER_BANDS,
        parameters={},
        signal_thresholds={}
    )
    return BollingerBandsStrategy(config)

def create_moving_average_strategy() -> MovingAverageStrategy:
    """Create a Moving Average strategy."""
    config = StrategyConfig(
        indicator_type=IndicatorType.SMA,
        parameters={},
        signal_thresholds={}
    )
    return MovingAverageStrategy(config)

def create_strategy_tester() -> StrategyTester:
    """Create a strategy tester."""
    return StrategyTester()