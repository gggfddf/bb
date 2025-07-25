#!/usr/bin/env python3
"""
Live Strategy Execution Module

Implements live strategy execution system for real-time trading:
- Strategy execution engine
- Order management system
- Position tracking and management
- Risk controls and limits
- Execution monitoring and reporting
- Strategy performance tracking

Features:
- Real-time strategy execution and monitoring
- Comprehensive order management and tracking
- Advanced position management and risk controls
- Performance tracking and reporting
- Execution monitoring and alerting
- Multi-strategy support and coordination
"""

import asyncio
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import threading
import queue
import uuid
import numpy as np
import pandas as pd

logger = structlog.get_logger()

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    CANCEL = "cancel"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class StrategyStatus(Enum):
    """Strategy status."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class RiskLevel(Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CUSTOM = "custom"

@dataclass
class Order:
    """Order structure."""
    order_id: str
    strategy_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Position structure."""
    strategy_id: str
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyConfig:
    """Strategy configuration."""
    strategy_id: str
    name: str
    description: str
    symbols: List[str]
    risk_level: RiskLevel
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ExecutionMetrics:
    """Execution metrics."""
    strategy_id: str
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    total_volume: float = 0.0
    total_commission: float = 0.0
    average_fill_time: float = 0.0
    success_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class OrderManager:
    """Order management system."""
    
    def __init__(self):
        """Initialize order manager."""
        self.orders = {}
        self.order_queue = queue.Queue()
        self.execution_callbacks = []
        self.running = False
        self.worker_thread = None
    
    def add_order(self, order: Order) -> bool:
        """Add order to queue."""
        try:
            self.orders[order.order_id] = order
            self.order_queue.put(order)
            logger.info("Order added to queue", order_id=order.order_id, 
                       symbol=order.symbol, side=order.side.value)
            return True
        except Exception as e:
            logger.error("Failed to add order", order_id=order.order_id, error=str(e))
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def update_order_status(self, order_id: str, status: OrderStatus, 
                           filled_quantity: float = None, average_price: float = None):
        """Update order status."""
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = status
            
            if filled_quantity is not None:
                order.filled_quantity = filled_quantity
            
            if average_price is not None:
                order.average_price = average_price
            
            # Call execution callbacks
            for callback in self.execution_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error("Order callback error", error=str(e))
            
            logger.info("Order status updated", order_id=order_id, status=status.value)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        if order_id in self.orders:
            self.update_order_status(order_id, OrderStatus.CANCELLED)
            return True
        return False
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get orders for strategy."""
        return [order for order in self.orders.values() if order.strategy_id == strategy_id]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get orders for symbol."""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def add_execution_callback(self, callback: Callable[[Order], None]):
        """Add execution callback."""
        self.execution_callbacks.append(callback)
    
    def start(self):
        """Start order manager."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._order_worker)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            logger.info("Order manager started")
    
    def stop(self):
        """Stop order manager."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Order manager stopped")
    
    def _order_worker(self):
        """Order processing worker."""
        while self.running:
            try:
                order = self.order_queue.get(timeout=1)
                self._process_order(order)
                self.order_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Order worker error", error=str(e))
    
    def _process_order(self, order: Order):
        """Process order."""
        try:
            # Simulate order execution (in production, this would connect to broker)
            if order.order_type == OrderType.MARKET:
                # Market orders are filled immediately
                filled_qty = order.quantity
                avg_price = order.price or self._get_market_price(order.symbol)
                
                self.update_order_status(
                    order.order_id, 
                    OrderStatus.FILLED, 
                    filled_qty, 
                    avg_price
                )
            
            elif order.order_type == OrderType.LIMIT:
                # Limit orders need price check
                current_price = self._get_market_price(order.symbol)
                if order.side == OrderSide.BUY and current_price <= order.price:
                    self.update_order_status(
                        order.order_id, 
                        OrderStatus.FILLED, 
                        order.quantity, 
                        order.price
                    )
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    self.update_order_status(
                        order.order_id, 
                        OrderStatus.FILLED, 
                        order.quantity, 
                        order.price
                    )
                else:
                    # Keep as pending
                    self.update_order_status(order.order_id, OrderStatus.SUBMITTED)
            
            # Add small delay to simulate processing time
            time.sleep(0.1)
        
        except Exception as e:
            logger.error("Order processing error", order_id=order.order_id, error=str(e))
            self.update_order_status(order.order_id, OrderStatus.REJECTED)
    
    def _get_market_price(self, symbol: str) -> float:
        """Get current market price (simulated)."""
        # In production, this would get real market data
        base_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'TSLA': 250.0,
            'BTCUSDT': 45000.0,
            'ETHUSDT': 3000.0
        }
        return base_prices.get(symbol, 100.0) + np.random.normal(0, 1)

class PositionManager:
    """Position tracking and management."""
    
    def __init__(self):
        """Initialize position manager."""
        self.positions = {}  # strategy_id -> symbol -> Position
        self.position_history = []
        self.running = False
        self.update_thread = None
    
    def update_position(self, strategy_id: str, symbol: str, order: Order):
        """Update position based on order execution."""
        position_key = f"{strategy_id}:{symbol}"
        
        if position_key not in self.positions:
            self.positions[position_key] = Position(
                strategy_id=strategy_id,
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                current_price=0.0
            )
        
        position = self.positions[position_key]
        
        if order.status == OrderStatus.FILLED:
            # Calculate new position
            if order.side == OrderSide.BUY:
                new_quantity = position.quantity + order.filled_quantity
                if new_quantity != 0:
                    new_avg_price = ((position.quantity * position.average_price) + 
                                   (order.filled_quantity * order.average_price)) / new_quantity
                else:
                    new_avg_price = order.average_price
                
                position.quantity = new_quantity
                position.average_price = new_avg_price
            
            elif order.side == OrderSide.SELL:
                # Calculate realized P&L
                if position.quantity > 0:
                    realized_pnl = (order.average_price - position.average_price) * order.filled_quantity
                    position.realized_pnl += realized_pnl
                
                position.quantity -= order.filled_quantity
                
                if position.quantity == 0:
                    position.average_price = 0.0
            
            position.timestamp = datetime.now()
            
            # Add to history
            self.position_history.append(position.copy())
            
            logger.info("Position updated", strategy_id=strategy_id, symbol=symbol, 
                       quantity=position.quantity, avg_price=position.average_price)
    
    def get_position(self, strategy_id: str, symbol: str) -> Optional[Position]:
        """Get position for strategy and symbol."""
        position_key = f"{strategy_id}:{symbol}"
        return self.positions.get(position_key)
    
    def get_all_positions(self, strategy_id: str = None) -> List[Position]:
        """Get all positions, optionally filtered by strategy."""
        positions = list(self.positions.values())
        if strategy_id:
            positions = [p for p in positions if p.strategy_id == strategy_id]
        return positions
    
    def calculate_unrealized_pnl(self, strategy_id: str = None):
        """Calculate unrealized P&L for positions."""
        positions = self.get_all_positions(strategy_id)
        
        for position in positions:
            current_price = self._get_current_price(position.symbol)
            position.current_price = current_price
            
            if position.quantity != 0:
                position.unrealized_pnl = (current_price - position.average_price) * position.quantity
                position.total_pnl = position.realized_pnl + position.unrealized_pnl
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        # In production, this would get real market data
        base_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'TSLA': 250.0,
            'BTCUSDT': 45000.0,
            'ETHUSDT': 3000.0
        }
        return base_prices.get(symbol, 100.0) + np.random.normal(0, 1)
    
    def start(self):
        """Start position manager."""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_worker)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("Position manager started")
    
    def stop(self):
        """Stop position manager."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("Position manager stopped")
    
    def _update_worker(self):
        """Position update worker."""
        while self.running:
            try:
                self.calculate_unrealized_pnl()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error("Position update error", error=str(e))

class RiskManager:
    """Risk controls and limits."""
    
    def __init__(self):
        """Initialize risk manager."""
        self.risk_limits = {}
        self.violations = []
        self.risk_callbacks = []
    
    def add_risk_limits(self, strategy_id: str, limits: Dict[str, float]):
        """Add risk limits for strategy."""
        self.risk_limits[strategy_id] = limits
        logger.info("Risk limits added", strategy_id=strategy_id, limits=limits)
    
    def check_order_risk(self, order: Order, strategy_config: StrategyConfig) -> Tuple[bool, str]:
        """Check if order meets risk requirements."""
        try:
            # Check position size limit
            if order.quantity > strategy_config.max_position_size:
                return False, f"Order quantity {order.quantity} exceeds max position size {strategy_config.max_position_size}"
            
            # Check daily loss limit
            daily_pnl = self._get_daily_pnl(order.strategy_id)
            if daily_pnl < -strategy_config.max_daily_loss:
                return False, f"Daily loss {daily_pnl} exceeds limit {strategy_config.max_daily_loss}"
            
            # Check drawdown limit
            drawdown = self._get_current_drawdown(order.strategy_id)
            if drawdown > strategy_config.max_drawdown:
                return False, f"Drawdown {drawdown} exceeds limit {strategy_config.max_drawdown}"
            
            return True, "Risk check passed"
        
        except Exception as e:
            logger.error("Risk check error", error=str(e))
            return False, f"Risk check error: {str(e)}"
    
    def check_position_risk(self, position: Position, strategy_config: StrategyConfig) -> Tuple[bool, str]:
        """Check if position meets risk requirements."""
        try:
            # Check position size
            if abs(position.quantity) > strategy_config.max_position_size:
                return False, f"Position size {position.quantity} exceeds limit {strategy_config.max_position_size}"
            
            # Check unrealized loss
            if position.unrealized_pnl < -strategy_config.max_daily_loss:
                return False, f"Unrealized loss {position.unrealized_pnl} exceeds daily limit {strategy_config.max_daily_loss}"
            
            return True, "Position risk check passed"
        
        except Exception as e:
            logger.error("Position risk check error", error=str(e))
            return False, f"Position risk check error: {str(e)}"
    
    def _get_daily_pnl(self, strategy_id: str) -> float:
        """Get daily P&L for strategy."""
        # In production, this would calculate from actual data
        return np.random.normal(0, 1000)  # Simulated daily P&L
    
    def _get_current_drawdown(self, strategy_id: str) -> float:
        """Get current drawdown for strategy."""
        # In production, this would calculate from actual data
        return abs(np.random.normal(0, 0.05))  # Simulated drawdown
    
    def add_risk_callback(self, callback: Callable[[str, str], None]):
        """Add risk violation callback."""
        self.risk_callbacks.append(callback)
    
    def record_violation(self, strategy_id: str, violation_type: str, message: str):
        """Record risk violation."""
        violation = {
            'strategy_id': strategy_id,
            'violation_type': violation_type,
            'message': message,
            'timestamp': datetime.now()
        }
        self.violations.append(violation)
        
        # Call risk callbacks
        for callback in self.risk_callbacks:
            try:
                callback(strategy_id, message)
            except Exception as e:
                logger.error("Risk callback error", error=str(e))
        
        logger.warning("Risk violation recorded", strategy_id=strategy_id, 
                      violation_type=violation_type, message=message)

class StrategyExecutor:
    """Main strategy execution engine."""
    
    def __init__(self):
        """Initialize strategy executor."""
        self.strategies = {}
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager()
        self.execution_metrics = {}
        
        self.running = False
        self.execution_thread = None
        
        # Setup callbacks
        self.order_manager.add_execution_callback(self._on_order_execution)
        self.risk_manager.add_risk_callback(self._on_risk_violation)
    
    def add_strategy(self, strategy_config: StrategyConfig):
        """Add strategy to executor."""
        self.strategies[strategy_config.strategy_id] = strategy_config
        
        # Initialize metrics
        self.execution_metrics[strategy_config.strategy_id] = ExecutionMetrics(
            strategy_id=strategy_config.strategy_id
        )
        
        # Add risk limits
        risk_limits = {
            'max_position_size': strategy_config.max_position_size,
            'max_daily_loss': strategy_config.max_daily_loss,
            'max_drawdown': strategy_config.max_drawdown
        }
        self.risk_manager.add_risk_limits(strategy_config.strategy_id, risk_limits)
        
        logger.info("Strategy added", strategy_id=strategy_config.strategy_id, 
                   name=strategy_config.name)
    
    def submit_order(self, strategy_id: str, symbol: str, side: OrderSide, 
                    order_type: OrderType, quantity: float, price: float = None) -> Optional[str]:
        """Submit order for strategy."""
        if strategy_id not in self.strategies:
            logger.error("Strategy not found", strategy_id=strategy_id)
            return None
        
        strategy_config = self.strategies[strategy_id]
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )
        
        # Check risk limits
        risk_ok, risk_message = self.risk_manager.check_order_risk(order, strategy_config)
        if not risk_ok:
            self.risk_manager.record_violation(strategy_id, "order_risk", risk_message)
            logger.warning("Order rejected due to risk limits", order_id=order.order_id, 
                          message=risk_message)
            return None
        
        # Submit order
        if self.order_manager.add_order(order):
            # Update metrics
            metrics = self.execution_metrics[strategy_id]
            metrics.total_orders += 1
            metrics.timestamp = datetime.now()
            
            return order.order_id
        
        return None
    
    def start(self):
        """Start strategy executor."""
        if not self.running:
            self.running = True
            
            # Start managers
            self.order_manager.start()
            self.position_manager.start()
            
            # Start execution thread
            self.execution_thread = threading.Thread(target=self._execution_worker)
            self.execution_thread.daemon = True
            self.execution_thread.start()
            
            logger.info("Strategy executor started")
    
    def stop(self):
        """Stop strategy executor."""
        self.running = False
        
        # Stop managers
        self.order_manager.stop()
        self.position_manager.stop()
        
        if self.execution_thread:
            self.execution_thread.join()
        
        logger.info("Strategy executor stopped")
    
    def _execution_worker(self):
        """Main execution worker."""
        while self.running:
            try:
                # Monitor active strategies
                for strategy_id, config in self.strategies.items():
                    if config.enabled:
                        self._monitor_strategy(strategy_id, config)
                
                time.sleep(1)  # Check every second
            
            except Exception as e:
                logger.error("Execution worker error", error=str(e))
    
    def _monitor_strategy(self, strategy_id: str, config: StrategyConfig):
        """Monitor individual strategy."""
        try:
            # Check positions for risk violations
            positions = self.position_manager.get_all_positions(strategy_id)
            for position in positions:
                risk_ok, risk_message = self.risk_manager.check_position_risk(position, config)
                if not risk_ok:
                    self.risk_manager.record_violation(strategy_id, "position_risk", risk_message)
            
            # Update metrics
            self._update_metrics(strategy_id)
        
        except Exception as e:
            logger.error("Strategy monitoring error", strategy_id=strategy_id, error=str(e))
    
    def _on_order_execution(self, order: Order):
        """Handle order execution."""
        try:
            # Update position
            self.position_manager.update_position(
                order.strategy_id, 
                order.symbol, 
                order
            )
            
            # Update metrics
            if order.status == OrderStatus.FILLED:
                metrics = self.execution_metrics[order.strategy_id]
                metrics.filled_orders += 1
                metrics.total_volume += order.filled_quantity
                metrics.total_commission += order.commission
                
                # Calculate average fill time
                fill_time = (datetime.now() - order.timestamp).total_seconds()
                metrics.average_fill_time = (
                    (metrics.average_fill_time * (metrics.filled_orders - 1) + fill_time) / 
                    metrics.filled_orders
                )
                
                metrics.success_rate = metrics.filled_orders / metrics.total_orders
                metrics.timestamp = datetime.now()
        
        except Exception as e:
            logger.error("Order execution callback error", error=str(e))
    
    def _on_risk_violation(self, strategy_id: str, message: str):
        """Handle risk violation."""
        logger.warning("Risk violation detected", strategy_id=strategy_id, message=message)
        # In production, this might trigger strategy pause or other actions
    
    def _update_metrics(self, strategy_id: str):
        """Update execution metrics."""
        try:
            metrics = self.execution_metrics[strategy_id]
            metrics.timestamp = datetime.now()
        except Exception as e:
            logger.error("Metrics update error", strategy_id=strategy_id, error=str(e))
    
    def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """Get strategy status."""
        if strategy_id not in self.strategies:
            return {}
        
        config = self.strategies[strategy_id]
        metrics = self.execution_metrics.get(strategy_id, ExecutionMetrics(strategy_id))
        positions = self.position_manager.get_all_positions(strategy_id)
        
        return {
            'strategy_id': strategy_id,
            'name': config.name,
            'status': StrategyStatus.ACTIVE if config.enabled else StrategyStatus.INACTIVE,
            'positions': len(positions),
            'total_orders': metrics.total_orders,
            'filled_orders': metrics.filled_orders,
            'success_rate': metrics.success_rate,
            'total_volume': metrics.total_volume,
            'total_commission': metrics.total_commission,
            'average_fill_time': metrics.average_fill_time,
            'last_update': metrics.timestamp.isoformat()
        }
    
    def get_all_strategies_status(self) -> List[Dict[str, Any]]:
        """Get status of all strategies."""
        return [self.get_strategy_status(strategy_id) for strategy_id in self.strategies.keys()]

def create_strategy_executor() -> StrategyExecutor:
    """
    Create a strategy execution engine.
    
    Returns:
        StrategyExecutor instance
    """
    return StrategyExecutor()

if __name__ == "__main__":
    # Demo of strategy execution
    executor = create_strategy_executor()
    
    # Add sample strategy
    strategy_config = StrategyConfig(
        strategy_id="strategy_001",
        name="Sample Trading Strategy",
        description="A sample trading strategy for demonstration",
        symbols=["AAPL", "GOOGL", "MSFT"],
        risk_level=RiskLevel.MEDIUM,
        max_position_size=1000.0,
        max_daily_loss=5000.0,
        max_drawdown=0.1
    )
    
    executor.add_strategy(strategy_config)
    
    # Start executor
    executor.start()
    
    print("Strategy executor created successfully!")
    print(f"Strategy: {strategy_config.name}")
    print(f"Symbols: {strategy_config.symbols}")
    print(f"Risk level: {strategy_config.risk_level.value}")
    
    # Submit sample order
    order_id = executor.submit_order(
        strategy_id="strategy_001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0
    )
    
    if order_id:
        print(f"Order submitted: {order_id}")
    
    # Get strategy status
    status = executor.get_strategy_status("strategy_001")
    print(f"Strategy status: {status}")