#!/usr/bin/env python3
"""
Order Management System

Implements comprehensive order management for trading:
- Order creation and validation
- Order routing and execution
- Position tracking and management
- Risk controls and limits
- Order status monitoring
- Execution reporting

Features:
- Advanced order management with multiple order types
- Real-time order routing and execution tracking
- Position management and risk controls
- Order status monitoring and reporting
- Integration with multiple brokers/exchanges
- Compliance and audit trail
"""

import asyncio
import json
import time
import threading
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import queue

logger = structlog.get_logger()

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderTimeInForce(Enum):
    """Order time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill

@dataclass
class Order:
    """Order structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    filled_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Execution:
    """Execution structure."""
    execution_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Position structure."""
    symbol: str
    quantity: float
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    market_value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderConfig:
    """Order management configuration."""
    max_order_size: float = 1000000.0
    max_position_size: float = 5000000.0
    enable_risk_checks: bool = True
    enable_position_tracking: bool = True
    enable_execution_reporting: bool = True
    default_commission: float = 0.001
    order_timeout: int = 300  # 5 minutes
    max_orders_per_symbol: int = 10

class OrderValidator:
    """Order validator."""
    
    def __init__(self, config: OrderConfig):
        """
        Initialize order validator.
        
        Args:
            config: Order configuration
        """
        self.config = config
        
    def validate_order(self, order: Order) -> Tuple[bool, List[str]]:
        """Validate order."""
        issues = []
        
        # Basic validation
        if not order.symbol:
            issues.append("Symbol is required")
            
        if order.quantity <= 0:
            issues.append("Quantity must be positive")
            
        if order.quantity > self.config.max_order_size:
            issues.append(f"Order size exceeds maximum ({self.config.max_order_size})")
            
        # Price validation
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                issues.append("Valid price required for limit orders")
                
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
            if order.stop_price is None or order.stop_price <= 0:
                issues.append("Valid stop price required for stop orders")
                
        # Time in force validation
        if order.time_in_force not in OrderTimeInForce:
            issues.append("Invalid time in force")
            
        return len(issues) == 0, issues

class OrderRouter:
    """Order router."""
    
    def __init__(self, config: OrderConfig):
        """
        Initialize order router.
        
        Args:
            config: Order configuration
        """
        self.config = config
        self.routes: Dict[str, Callable] = {}
        self.order_queue = queue.Queue()
        self.is_running = False
        self.routing_thread = None
        
    def add_route(self, symbol: str, route_handler: Callable):
        """Add order route."""
        self.routes[symbol] = route_handler
        
    def route_order(self, order: Order) -> bool:
        """Route order to appropriate handler."""
        try:
            # Check if route exists
            if order.symbol not in self.routes:
                logger.error(f"No route found for symbol: {order.symbol}")
                return False
                
            # Get route handler
            handler = self.routes[order.symbol]
            
            # Route order
            result = handler(order)
            return result
            
        except Exception as e:
            logger.error(f"Error routing order: {e}")
            return False
            
    def start_routing(self):
        """Start order routing."""
        if not self.is_running:
            self.is_running = True
            self.routing_thread = threading.Thread(target=self._routing_loop)
            self.routing_thread.start()
            logger.info("Order router started")
            
    def stop_routing(self):
        """Stop order routing."""
        self.is_running = False
        if self.routing_thread:
            self.routing_thread.join()
        logger.info("Order router stopped")
        
    def _routing_loop(self):
        """Main routing loop."""
        while self.is_running:
            try:
                # Process queued orders
                while not self.order_queue.empty():
                    order = self.order_queue.get_nowait()
                    self.route_order(order)
                    
                # Sleep briefly
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in routing loop: {e}")
                
    def queue_order(self, order: Order):
        """Queue order for routing."""
        try:
            self.order_queue.put_nowait(order)
        except queue.Full:
            logger.warning("Order queue full, dropping order")

class PositionManager:
    """Position manager."""
    
    def __init__(self, config: OrderConfig):
        """
        Initialize position manager.
        
        Args:
            config: Order configuration
        """
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        
    def update_position(self, execution: Execution):
        """Update position based on execution."""
        try:
            symbol = execution.symbol
            side = execution.side
            quantity = execution.quantity
            price = execution.price
            
            # Get or create position
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=0.0,
                    average_price=0.0
                )
                
            position = self.positions[symbol]
            
            # Update position
            if side == OrderSide.BUY:
                # Buying - increase position
                total_cost = position.quantity * position.average_price + quantity * price
                position.quantity += quantity
                position.average_price = total_cost / position.quantity if position.quantity > 0 else 0
            else:
                # Selling - decrease position
                if position.quantity >= quantity:
                    # Calculate realized P&L
                    realized_pnl = (price - position.average_price) * quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity
                    
                    if position.quantity == 0:
                        position.average_price = 0
                else:
                    logger.warning(f"Insufficient position for sell: {symbol}")
                    
            # Update timestamp
            position.timestamp = execution.timestamp
            
            # Store in history
            self.position_history.append(position)
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return self.positions.get(symbol)
        
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self.positions.copy()
        
    def calculate_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized P&L for position."""
        position = self.get_position(symbol)
        if not position:
            return 0.0
            
        if position.quantity > 0:
            # Long position
            unrealized_pnl = (current_price - position.average_price) * position.quantity
        else:
            # Short position
            unrealized_pnl = (position.average_price - current_price) * abs(position.quantity)
            
        position.unrealized_pnl = unrealized_pnl
        position.total_pnl = position.realized_pnl + unrealized_pnl
        position.market_value = abs(position.quantity) * current_price
        
        return position.total_pnl

class OrderManager:
    """Main order manager."""
    
    def __init__(self, config: OrderConfig):
        """
        Initialize order manager.
        
        Args:
            config: Order configuration
        """
        self.config = config
        self.validator = OrderValidator(config)
        self.router = OrderRouter(config)
        self.position_manager = PositionManager(config)
        self.orders: Dict[str, Order] = {}
        self.executions: List[Execution] = []
        self.order_history: List[Order] = []
        
    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    quantity: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: OrderTimeInForce = OrderTimeInForce.DAY) -> Order:
        """Create a new order."""
        try:
            # Create order
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            # Validate order
            is_valid, issues = self.validator.validate_order(order)
            if not is_valid:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order validation failed: {issues}")
                return order
                
            # Store order
            self.orders[order.order_id] = order
            
            # Route order
            if self.router.route_order(order):
                order.status = OrderStatus.SUBMITTED
            else:
                order.status = OrderStatus.REJECTED
                
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return Order(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                status=OrderStatus.REJECTED
            )
            
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            if order_id not in self.orders:
                logger.warning(f"Order not found: {order_id}")
                return False
                
            order = self.orders[order_id]
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                logger.warning(f"Cannot cancel order in status: {order.status}")
                return False
                
            order.status = OrderStatus.CANCELLED
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
            
    def record_execution(self, execution: Execution):
        """Record order execution."""
        try:
            # Store execution
            self.executions.append(execution)
            
            # Update order
            if execution.order_id in self.orders:
                order = self.orders[execution.order_id]
                order.filled_quantity += execution.quantity
                order.average_price = ((order.average_price * (order.filled_quantity - execution.quantity)) +
                                     (execution.price * execution.quantity)) / order.filled_quantity
                order.commission += execution.commission
                
                if order.filled_quantity >= order.quantity:
                    order.status = OrderStatus.FILLED
                    order.filled_timestamp = execution.timestamp
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
                    
            # Update position
            self.position_manager.update_position(execution)
            
        except Exception as e:
            logger.error(f"Error recording execution: {e}")
            
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
        
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        return [order for order in self.orders.values() if order.symbol == symbol]
        
    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return [order for order in self.orders.values() 
                if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]]
        
    def get_order_summary(self) -> Dict[str, Any]:
        """Get order summary."""
        total_orders = len(self.orders)
        open_orders = len(self.get_open_orders())
        filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        cancelled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
        
        total_executions = len(self.executions)
        total_commission = sum(execution.commission for execution in self.executions)
        
        positions = self.position_manager.get_all_positions()
        total_positions = len(positions)
        
        return {
            'total_orders': total_orders,
            'open_orders': open_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'total_executions': total_executions,
            'total_commission': total_commission,
            'total_positions': total_positions
        }

def create_order_manager(config: OrderConfig = None) -> OrderManager:
    """Create an order manager."""
    return OrderManager(config or OrderConfig())

# Demo of order management system
if __name__ == "__main__":
    # Create order manager
    config = OrderConfig(
        max_order_size=1000000.0,
        max_position_size=5000000.0,
        enable_risk_checks=True,
        enable_position_tracking=True
    )
    
    order_manager = create_order_manager(config)
    
    # Add route handler
    def route_handler(order: Order) -> bool:
        print(f"Routing order: {order.order_id} for {order.symbol}")
        return True
    
    order_manager.router.add_route("AAPL", route_handler)
    order_manager.router.add_route("GOOGL", route_handler)
    
    # Start routing
    order_manager.router.start_routing()
    
    # Create orders
    buy_order = order_manager.create_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )
    
    sell_order = order_manager.create_order(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=50,
        price=150.0
    )
    
    print(f"Created orders:")
    print(f"Buy order: {buy_order.order_id} - {buy_order.status.value}")
    print(f"Sell order: {sell_order.order_id} - {sell_order.status.value}")
    
    # Record executions
    execution1 = Execution(
        execution_id=str(uuid.uuid4()),
        order_id=buy_order.order_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=151.0,
        commission=15.1
    )
    
    order_manager.record_execution(execution1)
    
    # Get summary
    summary = order_manager.get_order_summary()
    print(f"\nOrder Summary:")
    print(f"Total orders: {summary['total_orders']}")
    print(f"Open orders: {summary['open_orders']}")
    print(f"Filled orders: {summary['filled_orders']}")
    print(f"Total executions: {summary['total_executions']}")
    print(f"Total commission: {summary['total_commission']:.2f}")
    
    # Get position
    position = order_manager.position_manager.get_position("AAPL")
    if position:
        print(f"\nAAPL Position:")
        print(f"Quantity: {position.quantity}")
        print(f"Average price: {position.average_price:.2f}")
        print(f"Realized P&L: {position.realized_pnl:.2f}")
    
    # Stop routing
    order_manager.router.stop_routing()
    
    print("Order management system completed successfully!")