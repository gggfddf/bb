"""
Vectorized Backtesting Engine

A high-performance, vectorized backtesting engine for stock trading strategies.
Uses pandas for fast operations and supports multiple trading strategies.

Features:
- Vectorized signal generation and position management
- Comprehensive performance metrics calculation
- Risk management with position sizing and stop-loss
- Support for multiple timeframes and symbols
- Fast execution with pandas operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()

class SignalType(Enum):
    """Signal types for trading decisions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class PositionType(Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    position_type: PositionType = PositionType.LONG
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """Container for backtest results."""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    signals: pd.Series
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

class RiskManager:
    """Risk management for position sizing and stop-loss."""
    
    def __init__(self, 
                 max_position_size: float = 0.1,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04,
                 max_drawdown: float = 0.2,
                 volatility_lookback: int = 20):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        self.volatility_lookback = volatility_lookback
    
    def calculate_position_size(self, 
                              capital: float, 
                              price: float, 
                              volatility: float,
                              signal_strength: float = 1.0) -> float:
        """Calculate position size based on risk parameters."""
        # Base position size
        base_size = capital * self.max_position_size
        
        # Adjust for volatility (higher volatility = smaller position)
        vol_adjustment = 1.0 / (1.0 + volatility)
        
        # Adjust for signal strength
        strength_adjustment = min(signal_strength, 1.0)
        
        # Calculate final position size
        position_size = base_size * vol_adjustment * strength_adjustment
        
        # Convert to quantity
        quantity = position_size / price
        
        return quantity
    
    def calculate_stop_loss(self, entry_price: float, position_type: PositionType) -> float:
        """Calculate stop loss price."""
        if position_type == PositionType.LONG:
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, position_type: PositionType) -> float:
        """Calculate take profit price."""
        if position_type == PositionType.LONG:
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)

class VectorizedBacktester:
    """High-performance vectorized backtesting engine."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 risk_manager: Optional[RiskManager] = None):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_manager = risk_manager or RiskManager()
        
        # State variables
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_time = None
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    signal_function: Callable,
                    signal_params: Dict[str, Any] = None) -> BacktestResult:
        """
        Run vectorized backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            signal_function: Function that generates signals
            signal_params: Parameters for signal function
        
        Returns:
            BacktestResult with all backtest information
        """
        logger.info("Starting vectorized backtest", 
                   data_length=len(data),
                   initial_capital=self.initial_capital)
        
        # Validate data
        self._validate_data(data)
        
        # Generate signals
        signals = self._generate_signals(data, signal_function, signal_params or {})
        
        # Calculate volatility for risk management
        volatility = self._calculate_volatility(data)
        
        # Initialize arrays for vectorized operations
        self._initialize_arrays(len(data))
        
        # Run vectorized backtest
        self._run_vectorized_backtest(data, signals, volatility)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        
        # Create result object
        result = BacktestResult(
            trades=self.trades,
            equity_curve=pd.Series(self.equity_curve, index=data.index),
            returns=pd.Series(self.returns, index=data.index),
            positions=pd.Series(self.position_array, index=data.index),
            signals=signals,
            metrics=metrics,
            metadata={
                'initial_capital': self.initial_capital,
                'final_capital': self.capital,
                'total_return': (self.capital - self.initial_capital) / self.initial_capital,
                'num_trades': len(self.trades)
            }
        )
        
        logger.info("Backtest completed", 
                   final_capital=self.capital,
                   total_return=result.metadata['total_return'],
                   num_trades=len(self.trades))
        
        return result
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            raise ValueError("Data is empty")
        
        # Check for NaN values
        if data[required_columns].isnull().any().any():
            logger.warning("Data contains NaN values, will be handled during processing")
    
    def _generate_signals(self, 
                         data: pd.DataFrame, 
                         signal_function: Callable,
                         signal_params: Dict[str, Any]) -> pd.Series:
        """Generate trading signals using the provided function."""
        try:
            signals = signal_function(data, **signal_params)
            
            if isinstance(signals, pd.Series):
                return signals
            elif isinstance(signals, np.ndarray):
                return pd.Series(signals, index=data.index)
            else:
                raise ValueError("Signal function must return pd.Series or np.ndarray")
                
        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            raise
    
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling volatility for risk management."""
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        return volatility.fillna(0)
    
    def _initialize_arrays(self, length: int) -> None:
        """Initialize arrays for vectorized operations."""
        self.capital_array = np.zeros(length)
        self.position_array = np.zeros(length)
        self.returns = np.zeros(length)
        self.equity_curve = np.zeros(length)
        
        # Initialize with initial capital
        self.capital_array[0] = self.initial_capital
        self.equity_curve[0] = self.initial_capital
    
    def _run_vectorized_backtest(self, 
                                data: pd.DataFrame, 
                                signals: pd.Series,
                                volatility: pd.Series) -> None:
        """Run the main vectorized backtest loop."""
        current_position = 0.0
        current_price = 0.0
        
        for i in range(1, len(data)):
            self.current_time = data.index[i]
            current_price = data.iloc[i]['close']
            current_vol = volatility.iloc[i]
            
            # Update capital and position arrays
            self.capital_array[i] = self.capital_array[i-1]
            self.position_array[i] = current_position
            
            # Process signals
            signal = signals.iloc[i]
            
            if signal == SignalType.BUY and current_position <= 0:
                # Open long position
                position_size = self.risk_manager.calculate_position_size(
                    self.capital_array[i], current_price, current_vol
                )
                
                # Apply slippage and commission
                entry_price = current_price * (1 + self.slippage_rate)
                commission = entry_price * position_size * self.commission_rate
                
                # Update position and capital
                current_position = position_size
                self.capital_array[i] -= (entry_price * position_size + commission)
                
                # Record trade
                self._record_trade_entry(
                    entry_time=self.current_time,
                    entry_price=entry_price,
                    quantity=position_size,
                    position_type=PositionType.LONG,
                    commission=commission
                )
                
            elif signal == SignalType.SELL and current_position >= 0:
                # Open short position
                position_size = self.risk_manager.calculate_position_size(
                    self.capital_array[i], current_price, current_vol
                )
                
                # Apply slippage and commission
                entry_price = current_price * (1 - self.slippage_rate)
                commission = entry_price * position_size * self.commission_rate
                
                # Update position and capital
                current_position = -position_size
                self.capital_array[i] += (entry_price * position_size - commission)
                
                # Record trade
                self._record_trade_entry(
                    entry_time=self.current_time,
                    entry_price=entry_price,
                    quantity=position_size,
                    position_type=PositionType.SHORT,
                    commission=commission
                )
                
            elif signal == SignalType.CLOSE and current_position != 0:
                # Close position
                exit_price = current_price
                if current_position > 0:  # Long position
                    exit_price *= (1 - self.slippage_rate)
                else:  # Short position
                    exit_price *= (1 + self.slippage_rate)
                
                commission = exit_price * abs(current_position) * self.commission_rate
                
                # Calculate PnL
                if current_position > 0:  # Long position
                    pnl = (exit_price - self.trades[-1].entry_price) * current_position
                    self.capital_array[i] += (exit_price * current_position - commission)
                else:  # Short position
                    pnl = (self.trades[-1].entry_price - exit_price) * abs(current_position)
                    self.capital_array[i] -= (exit_price * abs(current_position) + commission)
                
                # Record trade exit
                self._record_trade_exit(
                    exit_time=self.current_time,
                    exit_price=exit_price,
                    pnl=pnl,
                    commission=commission
                )
                
                current_position = 0.0
            
            # Update equity curve and returns
            self.equity_curve[i] = self.capital_array[i] + (current_position * current_price)
            self.returns[i] = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            
            # Check stop-loss and take-profit
            if current_position != 0:
                self._check_risk_management(current_price, current_position)
    
    def _record_trade_entry(self, 
                           entry_time: datetime,
                           entry_price: float,
                           quantity: float,
                           position_type: PositionType,
                           commission: float) -> None:
        """Record a new trade entry."""
        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            quantity=quantity,
            position_type=position_type,
            commission=commission,
            stop_loss=self.risk_manager.calculate_stop_loss(entry_price, position_type),
            take_profit=self.risk_manager.calculate_take_profit(entry_price, position_type)
        )
        self.trades.append(trade)
    
    def _record_trade_exit(self, 
                          exit_time: datetime,
                          exit_price: float,
                          pnl: float,
                          commission: float) -> None:
        """Record trade exit."""
        if self.trades:
            trade = self.trades[-1]
            trade.exit_time = exit_time
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.commission += commission
            trade.pnl_pct = pnl / (trade.entry_price * trade.quantity)
            trade.exit_reason = "signal"
    
    def _check_risk_management(self, current_price: float, current_position: float) -> None:
        """Check and execute stop-loss and take-profit orders."""
        if not self.trades:
            return
        
        trade = self.trades[-1]
        
        if trade.exit_time is not None:  # Trade already closed
            return
        
        # Check stop-loss
        if trade.stop_loss is not None:
            if (current_position > 0 and current_price <= trade.stop_loss) or \
               (current_position < 0 and current_price >= trade.stop_loss):
                self._execute_risk_exit(trade, current_price, "stop_loss")
                return
        
        # Check take-profit
        if trade.take_profit is not None:
            if (current_position > 0 and current_price >= trade.take_profit) or \
               (current_position < 0 and current_price <= trade.take_profit):
                self._execute_risk_exit(trade, current_price, "take_profit")
                return
    
    def _execute_risk_exit(self, trade: Trade, current_price: float, reason: str) -> None:
        """Execute risk management exit."""
        # Apply slippage
        if trade.position_type == PositionType.LONG:
            exit_price = current_price * (1 - self.slippage_rate)
        else:
            exit_price = current_price * (1 + self.slippage_rate)
        
        commission = exit_price * trade.quantity * self.commission_rate
        
        # Calculate PnL
        if trade.position_type == PositionType.LONG:
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Record exit
        trade.exit_time = self.current_time
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.commission += commission
        trade.pnl_pct = pnl / (trade.entry_price * trade.quantity)
        trade.exit_reason = reason
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {}
        
        # Basic metrics
        total_return = (self.capital_array[-1] - self.initial_capital) / self.initial_capital
        num_trades = len([t for t in self.trades if t.exit_time is not None])
        
        if num_trades == 0:
            return {'total_return': total_return, 'num_trades': 0}
        
        # Trade metrics
        winning_trades = [t for t in self.trades if t.pnl > 0 and t.exit_time is not None]
        losing_trades = [t for t in self.trades if t.pnl < 0 and t.exit_time is not None]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
        
        # Risk metrics
        returns_series = pd.Series(self.returns)
        returns_series = returns_series[returns_series != 0]  # Remove zero returns
        
        if len(returns_series) > 0:
            volatility = returns_series.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (returns_series.mean() * 252) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns_series).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': self.capital_array[-1]
        }

# Convenience functions
def run_simple_backtest(data: pd.DataFrame, 
                       signals: pd.Series,
                       initial_capital: float = 100000.0) -> BacktestResult:
    """Run a simple backtest with basic parameters."""
    backtester = VectorizedBacktester(initial_capital=initial_capital)
    
    def simple_signal_function(data, signals):
        return signals
    
    return backtester.run_backtest(data, simple_signal_function, {'signals': signals})

def calculate_benchmark_metrics(data: pd.DataFrame, 
                               initial_capital: float = 100000.0) -> Dict[str, float]:
    """Calculate buy-and-hold benchmark metrics."""
    returns = data['close'].pct_change().dropna()
    
    if len(returns) == 0:
        return {}
    
    total_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'benchmark_return': total_return,
        'benchmark_volatility': volatility,
        'benchmark_sharpe': sharpe_ratio,
        'benchmark_max_drawdown': max_drawdown
    }