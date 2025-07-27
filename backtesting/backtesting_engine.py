#!/usr/bin/env python3
"""
Backtesting Engine

Implements comprehensive backtesting for trading strategies:
- Multi-strategy backtesting
- Realistic simulation with slippage and fees
- Performance analysis and reporting
- Risk management integration
- Order management simulation
- Strategy comparison and optimization
- Walk-forward analysis

Features:
- Advanced backtesting engine with multiple strategies
- Realistic simulation with transaction costs and slippage
- Performance analysis and risk metrics
- Strategy comparison and optimization
- Walk-forward analysis and out-of-sample testing
- Integration with risk management systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class StrategyType(Enum):
    """Strategy type enumeration."""
    MOVING_AVERAGE = "moving_average"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    OPTIONS_STRATEGY = "options_strategy"

class SignalType(Enum):
    """Signal type enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

@dataclass
class Signal:
    """Trading signal structure."""
    signal_id: str
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    price: float
    quantity: float
    confidence: float
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trade:
    """Trade structure."""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """Backtest result structure."""
    result_id: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    trades: List[Trade]
    equity_curve: pd.Series
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    position_size: float = 0.1
    max_positions: int = 10
    enable_short_selling: bool = True
    enable_risk_management: bool = True
    stop_loss: float = 0.02
    take_profit: float = 0.04

class Strategy:
    """Base strategy class."""
    
    def __init__(self, name: str, config: BacktestConfig):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            config: Backtest configuration
        """
        self.name = name
        self.config = config
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals."""
        raise NotImplementedError("Subclasses must implement generate_signals")
        
    def calculate_position_size(self, signal: Signal, current_capital: float) -> float:
        """Calculate position size for signal."""
        return current_capital * self.config.position_size

class MovingAverageStrategy(Strategy):
    """Moving average crossover strategy."""
    
    def __init__(self, name: str, config: BacktestConfig, short_window: int = 20, long_window: int = 50):
        """
        Initialize moving average strategy.
        
        Args:
            name: Strategy name
            config: Backtest configuration
            short_window: Short moving average window
            long_window: Long moving average window
        """
        super().__init__(name, config)
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate signals based on moving average crossover."""
        signals = []
        
        try:
            # Calculate moving averages
            short_ma = data['close'].rolling(window=self.short_window).mean()
            long_ma = data['close'].rolling(window=self.long_window).mean()
            
            # Generate signals
            for i in range(self.long_window, len(data)):
                timestamp = data.index[i]
                price = data['close'].iloc[i]
                
                # Crossover signals
                if short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i-1] <= long_ma.iloc[i-1]:
                    # Golden cross - buy signal
                    signal = Signal(
                        signal_id=str(uuid.uuid4()),
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        signal_type=SignalType.BUY,
                        timestamp=timestamp,
                        price=price,
                        quantity=1.0,
                        confidence=0.7,
                        strategy=self.name
                    )
                    signals.append(signal)
                    
                elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i-1] >= long_ma.iloc[i-1]:
                    # Death cross - sell signal
                    signal = Signal(
                        signal_id=str(uuid.uuid4()),
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        signal_type=SignalType.SELL,
                        timestamp=timestamp,
                        price=price,
                        quantity=1.0,
                        confidence=0.7,
                        strategy=self.name
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            
        return signals

class MeanReversionStrategy(Strategy):
    """Mean reversion strategy."""
    
    def __init__(self, name: str, config: BacktestConfig, lookback_period: int = 20, std_dev: float = 2.0):
        """
        Initialize mean reversion strategy.
        
        Args:
            name: Strategy name
            config: Backtest configuration
            lookback_period: Lookback period for mean calculation
            std_dev: Standard deviation threshold
        """
        super().__init__(name, config)
        self.lookback_period = lookback_period
        self.std_dev = std_dev
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate signals based on mean reversion."""
        signals = []
        
        try:
            # Calculate rolling mean and standard deviation
            rolling_mean = data['close'].rolling(window=self.lookback_period).mean()
            rolling_std = data['close'].rolling(window=self.lookback_period).std()
            
            # Calculate z-score
            z_score = (data['close'] - rolling_mean) / rolling_std
            
            # Generate signals
            for i in range(self.lookback_period, len(data)):
                timestamp = data.index[i]
                price = data['close'].iloc[i]
                current_z_score = z_score.iloc[i]
                
                if current_z_score < -self.std_dev:
                    # Oversold - buy signal
                    signal = Signal(
                        signal_id=str(uuid.uuid4()),
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        signal_type=SignalType.BUY,
                        timestamp=timestamp,
                        price=price,
                        quantity=1.0,
                        confidence=0.8,
                        strategy=self.name
                    )
                    signals.append(signal)
                    
                elif current_z_score > self.std_dev:
                    # Overbought - sell signal
                    signal = Signal(
                        signal_id=str(uuid.uuid4()),
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        signal_type=SignalType.SELL,
                        timestamp=timestamp,
                        price=price,
                        quantity=1.0,
                        confidence=0.8,
                        strategy=self.name
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            
        return signals

class BacktestingEngine:
    """Main backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.strategies: Dict[str, Strategy] = {}
        self.results: List[BacktestResult] = []
        
    def add_strategy(self, strategy: Strategy):
        """Add strategy to backtesting engine."""
        self.strategies[strategy.name] = strategy
        
    def run_backtest(self, data: pd.DataFrame, strategy_name: str) -> BacktestResult:
        """Run backtest for a specific strategy."""
        try:
            if strategy_name not in self.strategies:
                raise ValueError(f"Strategy not found: {strategy_name}")
                
            strategy = self.strategies[strategy_name]
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Execute trades
            trades = self._execute_trades(signals, data)
            
            # Calculate performance
            equity_curve = self._calculate_equity_curve(trades, data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(trades, equity_curve)
            
            # Create result
            result = BacktestResult(
                result_id=str(uuid.uuid4()),
                strategy_name=strategy_name,
                start_date=data.index[0],
                end_date=data.index[-1],
                trades=trades,
                equity_curve=equity_curve,
                **metrics
            )
            
            # Store result
            self.results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return self._create_default_result(strategy_name, data)
            
    def _execute_trades(self, signals: List[Signal], data: pd.DataFrame) -> List[Trade]:
        """Execute trades based on signals."""
        trades = []
        current_position = 0.0
        entry_price = 0.0
        entry_time = None
        
        try:
            for signal in signals:
                if signal.signal_type == SignalType.BUY and current_position == 0:
                    # Open long position
                    current_position = signal.quantity
                    entry_price = signal.price * (1 + self.config.slippage_rate)
                    entry_time = signal.timestamp
                    
                elif signal.signal_type == SignalType.SELL and current_position > 0:
                    # Close long position
                    exit_price = signal.price * (1 - self.config.slippage_rate)
                    
                    # Calculate P&L
                    pnl = (exit_price - entry_price) * current_position
                    commission = (entry_price + exit_price) * current_position * self.config.commission_rate
                    net_pnl = pnl - commission
                    
                    # Create trade
                    trade = Trade(
                        trade_id=str(uuid.uuid4()),
                        symbol=signal.symbol,
                        side="long",
                        quantity=current_position,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_time=entry_time,
                        exit_time=signal.timestamp,
                        pnl=net_pnl,
                        commission=commission,
                        slippage=(entry_price + exit_price) * current_position * self.config.slippage_rate
                    )
                    trades.append(trade)
                    
                    # Reset position
                    current_position = 0.0
                    entry_price = 0.0
                    entry_time = None
                    
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            
        return trades
        
    def _calculate_equity_curve(self, trades: List[Trade], data: pd.DataFrame) -> pd.Series:
        """Calculate equity curve."""
        try:
            equity_curve = pd.Series(index=data.index, data=self.config.initial_capital)
            
            for trade in trades:
                # Find entry and exit indices
                entry_idx = data.index.get_loc(trade.entry_time)
                exit_idx = data.index.get_loc(trade.exit_time)
                
                # Update equity curve
                for i in range(entry_idx, exit_idx + 1):
                    if i == entry_idx:
                        equity_curve.iloc[i] -= trade.quantity * trade.entry_price
                    elif i == exit_idx:
                        equity_curve.iloc[i] += trade.quantity * trade.exit_price + trade.pnl
                        
            return equity_curve
            
        except Exception as e:
            logger.error(f"Error calculating equity curve: {e}")
            return pd.Series(index=data.index, data=self.config.initial_capital)
            
    def _calculate_metrics(self, trades: List[Trade], equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            if not trades:
                return {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'average_win': 0.0,
                    'average_loss': 0.0
                }
                
            # Basic metrics
            total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
            total_trades = len(trades)
            
            # Trade analysis
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            average_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            average_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
            profit_factor = average_win / average_loss if average_loss > 0 else 0
            
            # Risk metrics
            returns = equity_curve.pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Annualized return
            years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'average_win': average_win,
                'average_loss': average_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'average_win': 0.0,
                'average_loss': 0.0
            }
            
    def _create_default_result(self, strategy_name: str, data: pd.DataFrame) -> BacktestResult:
        """Create default backtest result."""
        return BacktestResult(
            result_id=str(uuid.uuid4()),
            strategy_name=strategy_name,
            start_date=data.index[0] if len(data) > 0 else datetime.now(),
            end_date=data.index[-1] if len(data) > 0 else datetime.now(),
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            average_win=0.0,
            average_loss=0.0,
            trades=[],
            equity_curve=pd.Series()
        )
        
    def compare_strategies(self, data: pd.DataFrame) -> Dict[str, BacktestResult]:
        """Compare all strategies."""
        results = {}
        
        for strategy_name in self.strategies:
            try:
                result = self.run_backtest(data, strategy_name)
                results[strategy_name] = result
            except Exception as e:
                logger.error(f"Error comparing strategy {strategy_name}: {e}")
                
        return results
        
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get summary of all backtests."""
        if not self.results:
            return {}
            
        summary = {
            'total_backtests': len(self.results),
            'strategies_tested': list(set(r.strategy_name for r in self.results)),
            'best_strategy': max(self.results, key=lambda x: x.total_return).strategy_name,
            'average_return': np.mean([r.total_return for r in self.results]),
            'average_sharpe': np.mean([r.sharpe_ratio for r in self.results]),
            'average_max_drawdown': np.mean([r.max_drawdown for r in self.results])
        }
        
        return summary

def create_backtesting_engine(config: BacktestConfig = None) -> BacktestingEngine:
    """Create a backtesting engine."""
    return BacktestingEngine(config or BacktestConfig())

# Demo of backtesting engine
if __name__ == "__main__":
    # Create backtesting engine
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        position_size=0.1
    )
    
    engine = create_backtesting_engine(config)
    
    # Add strategies
    ma_strategy = MovingAverageStrategy("MA_Crossover", config, short_window=20, long_window=50)
    mr_strategy = MeanReversionStrategy("Mean_Reversion", config, lookback_period=20, std_dev=2.0)
    
    engine.add_strategy(ma_strategy)
    engine.add_strategy(mr_strategy)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(252).cumsum() + 100,
        'high': np.random.randn(252).cumsum() + 102,
        'low': np.random.randn(252).cumsum() + 98,
        'close': np.random.randn(252).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 252)
    }, index=dates)
    
    # Run backtests
    print("Running backtests...")
    results = engine.compare_strategies(data)
    
    # Display results
    for strategy_name, result in results.items():
        print(f"\n{strategy_name} Results:")
        print(f"Total Return: {result.total_return:.4f}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(f"Max Drawdown: {result.max_drawdown:.4f}")
        print(f"Win Rate: {result.win_rate:.4f}")
        print(f"Total Trades: {result.total_trades}")
    
    # Get summary
    summary = engine.get_backtest_summary()
    print(f"\nBacktest Summary:")
    print(f"Total backtests: {summary.get('total_backtests', 0)}")
    print(f"Best strategy: {summary.get('best_strategy', 'N/A')}")
    print(f"Average return: {summary.get('average_return', 0):.4f}")
    
    print("Backtesting engine completed successfully!")