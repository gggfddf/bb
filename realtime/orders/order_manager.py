#!/usr/bin/env python3
"""
Live Order Management Module

Implements comprehensive live order management system:
- Order lifecycle management
- Order routing and execution
- Order tracking and monitoring
- Order modification and cancellation
- Order validation and compliance
- Order reporting and analytics

Features:
- Complete order lifecycle management and tracking
- Intelligent order routing and execution optimization
- Real-time order monitoring and status updates
- Advanced order modification and cancellation capabilities
- Comprehensive order validation and compliance checking
- Detailed order reporting and analytics
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
from collections import defaultdict, deque

logger = structlog.get_logger()

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

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
    SUSPENDED = "suspended"

class OrderPriority(Enum):
    """Order priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class ExecutionVenue(Enum):
    """Execution venues."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    ATS = "ats"

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
    time_in_force: str = "DAY"
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    priority: OrderPriority = OrderPriority.NORMAL
    venue: ExecutionVenue = ExecutionVenue.PRIMARY
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderFill:
    """Order fill structure."""
    fill_id: str
    order_id: str
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    venue: ExecutionVenue = ExecutionVenue.PRIMARY
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderConfig:
    """Order management configuration."""
    max_order_size: float = 1000000.0
    max_daily_orders: int = 1000
    max_order_value: float = 10000000.0
    enable_iceberg_orders: bool = True
    enable_twap_orders: bool = True
    enable_vwap_orders: bool = True
    default_venue: ExecutionVenue = ExecutionVenue.PRIMARY
    order_timeout: int = 300  # seconds
    retry_attempts: int = 3

class OrderValidator:
    """Order validation and compliance system."""
    
    def __init__(self, config: OrderConfig):
        """
        Initialize order validator.
        
        Args:
            config: Order management configuration
        """
        self.config = config
        self.validation_rules = {
            'size_limit': self._validate_size_limit,
            'value_limit': self._validate_value_limit,
            'daily_limit': self._validate_daily_limit,
            'price_validation': self._validate_price,
            'time_validation': self._validate_time,
            'symbol_validation': self._validate_symbol
        }
    
    def validate_order(self, order: Order) -> Tuple[bool, List[str]]:
        """
        Validate order against all rules.
        
        Args:
            order: Order to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for rule_name, rule_func in self.validation_rules.items():
            try:
                is_valid, error = rule_func(order)
                if not is_valid:
                    errors.append(error)
            except Exception as e:
                errors.append(f"Validation error in {rule_name}: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _validate_size_limit(self, order: Order) -> Tuple[bool, str]:
        """Validate order size limit."""
        if order.quantity > self.config.max_order_size:
            return False, f"Order quantity {order.quantity} exceeds max size {self.config.max_order_size}"
        return True, ""
    
    def _validate_value_limit(self, order: Order) -> Tuple[bool, str]:
        """Validate order value limit."""
        if order.price and order.quantity * order.price > self.config.max_order_value:
            return False, f"Order value {order.quantity * order.price} exceeds max value {self.config.max_order_value}"
        return True, ""
    
    def _validate_daily_limit(self, order: Order) -> Tuple[bool, str]:
        """Validate daily order limit."""
        # This would typically check against a daily counter
        # For now, we'll assume it's within limits
        return True, ""
    
    def _validate_price(self, order: Order) -> Tuple[bool, str]:
        """Validate order price."""
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not order.price:
            return False, f"Price required for {order.order_type.value} order"
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not order.stop_price:
            return False, f"Stop price required for {order.order_type.value} order"
        
        if order.price and order.price <= 0:
            return False, "Price must be positive"
        
        return True, ""
    
    def _validate_time(self, order: Order) -> Tuple[bool, str]:
        """Validate order time."""
        # Check if order is within market hours
        # This is a simplified check
        current_hour = order.timestamp.hour
        if current_hour < 9 or current_hour > 16:
            return False, "Order outside market hours"
        
        return True, ""
    
    def _validate_symbol(self, order: Order) -> Tuple[bool, str]:
        """Validate order symbol."""
        # Check if symbol is valid
        valid_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
        if order.symbol not in valid_symbols:
            return False, f"Invalid symbol: {order.symbol}"
        
        return True, ""

class OrderRouter:
    """Order routing and execution system."""
    
    def __init__(self, config: OrderConfig):
        """
        Initialize order router.
        
        Args:
            config: Order management configuration
        """
        self.config = config
        self.venues = {
            ExecutionVenue.PRIMARY: self._execute_primary,
            ExecutionVenue.SECONDARY: self._execute_secondary,
            ExecutionVenue.DARK_POOL: self._execute_dark_pool,
            ExecutionVenue.ECN: self._execute_ecn,
            ExecutionVenue.ATS: self._execute_ats
        }
        self.execution_queue = queue.Queue()
        self.running = False
        self.execution_thread = None
    
    def route_order(self, order: Order) -> bool:
        """Route order to appropriate venue."""
        try:
            self.execution_queue.put(order)
            logger.info("Order queued for execution", order_id=order.order_id, 
                       venue=order.venue.value)
            return True
        except Exception as e:
            logger.error("Failed to queue order", order_id=order.order_id, error=str(e))
            return False
    
    def start(self):
        """Start order router."""
        if not self.running:
            self.running = True
            self.execution_thread = threading.Thread(target=self._execution_worker)
            self.execution_thread.daemon = True
            self.execution_thread.start()
            logger.info("Order router started")
    
    def stop(self):
        """Stop order router."""
        self.running = False
        if self.execution_thread:
            self.execution_thread.join()
        logger.info("Order router stopped")
    
    def _execution_worker(self):
        """Order execution worker."""
        while self.running:
            try:
                order = self.execution_queue.get(timeout=1)
                self._execute_order(order)
                self.execution_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Order execution error", error=str(e))
    
    def _execute_order(self, order: Order):
        """Execute order on specified venue."""
        try:
            venue_handler = self.venues.get(order.venue)
            if venue_handler:
                venue_handler(order)
            else:
                logger.error("No handler for venue", venue=order.venue.value)
                order.status = OrderStatus.REJECTED
        except Exception as e:
            logger.error("Order execution error", order_id=order.order_id, error=str(e))
            order.status = OrderStatus.REJECTED
    
    def _execute_primary(self, order: Order):
        """Execute order on primary venue."""
        # Simulate primary venue execution
        if order.order_type == OrderType.MARKET:
            # Market orders are filled immediately
            fill_price = self._get_market_price(order.symbol)
            self._create_fill(order, order.quantity, fill_price)
        elif order.order_type == OrderType.LIMIT:
            # Check if limit price is met
            current_price = self._get_market_price(order.symbol)
            if (order.side == OrderSide.BUY and current_price <= order.price) or \
               (order.side == OrderSide.SELL and current_price >= order.price):
                self._create_fill(order, order.quantity, order.price)
            else:
                order.status = OrderStatus.SUBMITTED
        
        logger.info("Order executed on primary venue", order_id=order.order_id)
    
    def _execute_secondary(self, order: Order):
        """Execute order on secondary venue."""
        # Simulate secondary venue execution
        fill_price = self._get_market_price(order.symbol) * 1.001  # Slightly higher price
        self._create_fill(order, order.quantity, fill_price)
        logger.info("Order executed on secondary venue", order_id=order.order_id)
    
    def _execute_dark_pool(self, order: Order):
        """Execute order on dark pool."""
        # Simulate dark pool execution
        fill_price = self._get_market_price(order.symbol) * 0.999  # Slightly lower price
        self._create_fill(order, order.quantity, fill_price)
        logger.info("Order executed on dark pool", order_id=order.order_id)
    
    def _execute_ecn(self, order: Order):
        """Execute order on ECN."""
        # Simulate ECN execution
        fill_price = self._get_market_price(order.symbol)
        self._create_fill(order, order.quantity, fill_price)
        logger.info("Order executed on ECN", order_id=order.order_id)
    
    def _execute_ats(self, order: Order):
        """Execute order on ATS."""
        # Simulate ATS execution
        fill_price = self._get_market_price(order.symbol) * 1.0005  # Very slight premium
        self._create_fill(order, order.quantity, fill_price)
        logger.info("Order executed on ATS", order_id=order.order_id)
    
    def _get_market_price(self, symbol: str) -> float:
        """Get current market price."""
        # Simulated market prices
        base_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'TSLA': 250.0,
            'AMZN': 3300.0,
            'META': 350.0,
            'NVDA': 450.0,
            'NFLX': 500.0
        }
        return base_prices.get(symbol, 100.0) + (time.time() % 10) * 0.1
    
    def _create_fill(self, order: Order, quantity: float, price: float):
        """Create order fill."""
        fill = OrderFill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            quantity=quantity,
            price=price,
            venue=order.venue
        )
        
        # Update order
        order.filled_quantity = quantity
        order.average_price = price
        order.status = OrderStatus.FILLED
        
        # Store fill
        if 'fills' not in order.metadata:
            order.metadata['fills'] = []
        order.metadata['fills'].append(fill)
        
        logger.info("Order fill created", fill_id=fill.fill_id, order_id=order.order_id, 
                   quantity=quantity, price=price)

class OrderTracker:
    """Order tracking and monitoring system."""
    
    def __init__(self):
        """Initialize order tracker."""
        self.orders = {}
        self.order_history = deque(maxlen=10000)
        self.tracking_callbacks = []
    
    def track_order(self, order: Order):
        """Track order."""
        self.orders[order.order_id] = order
        self.order_history.append(order)
        
        # Call tracking callbacks
        for callback in self.tracking_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error("Order tracking callback error", error=str(e))
        
        logger.info("Order tracked", order_id=order.order_id, status=order.status.value)
    
    def update_order(self, order_id: str, updates: Dict[str, Any]):
        """Update order with new information."""
        if order_id not in self.orders:
            logger.warning("Order not found for update", order_id=order_id)
            return
        
        order = self.orders[order_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        # Track update
        self.track_order(order)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_orders(self, strategy_id: str = None, status: OrderStatus = None) -> List[Order]:
        """Get orders with optional filtering."""
        orders = list(self.orders.values())
        
        if strategy_id:
            orders = [o for o in orders if o.strategy_id == strategy_id]
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        return sorted(orders, key=lambda x: x.timestamp, reverse=True)
    
    def add_tracking_callback(self, callback: Callable[[Order], None]):
        """Add order tracking callback."""
        self.tracking_callbacks.append(callback)

class OrderModifier:
    """Order modification and cancellation system."""
    
    def __init__(self, tracker: OrderTracker):
        """
        Initialize order modifier.
        
        Args:
            tracker: Order tracker
        """
        self.tracker = tracker
        self.modification_history = defaultdict(list)
    
    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> bool:
        """
        Modify existing order.
        
        Args:
            order_id: Order ID to modify
            modifications: Dictionary of modifications
            
        Returns:
            True if modification successful
        """
        order = self.tracker.get_order(order_id)
        if not order:
            logger.error("Order not found for modification", order_id=order_id)
            return False
        
        # Check if order can be modified
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            logger.warning("Order cannot be modified", order_id=order_id, status=order.status.value)
            return False
        
        # Apply modifications
        original_state = {
            'price': order.price,
            'quantity': order.quantity,
            'stop_price': order.stop_price,
            'limit_price': order.limit_price
        }
        
        for key, value in modifications.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        # Record modification
        modification_record = {
            'timestamp': datetime.now(),
            'modifications': modifications,
            'original_state': original_state
        }
        self.modification_history[order_id].append(modification_record)
        
        # Update tracker
        self.tracker.track_order(order)
        
        logger.info("Order modified", order_id=order_id, modifications=modifications)
        return True
    
    def cancel_order(self, order_id: str, reason: str = "User request") -> bool:
        """
        Cancel order.
        
        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason
            
        Returns:
            True if cancellation successful
        """
        order = self.tracker.get_order(order_id)
        if not order:
            logger.error("Order not found for cancellation", order_id=order_id)
            return False
        
        # Check if order can be cancelled
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            logger.warning("Order cannot be cancelled", order_id=order_id, status=order.status.value)
            return False
        
        # Cancel order
        order.status = OrderStatus.CANCELLED
        order.metadata['cancellation_reason'] = reason
        order.metadata['cancelled_at'] = datetime.now()
        
        # Update tracker
        self.tracker.track_order(order)
        
        logger.info("Order cancelled", order_id=order_id, reason=reason)
        return True
    
    def get_modification_history(self, order_id: str) -> List[Dict[str, Any]]:
        """Get modification history for order."""
        return self.modification_history.get(order_id, [])

class OrderAnalytics:
    """Order analytics and reporting system."""
    
    def __init__(self):
        """Initialize order analytics."""
        self.analytics_data = defaultdict(lambda: {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': 0.0,
            'total_value': 0.0,
            'average_fill_time': 0.0,
            'fill_rate': 0.0,
            'by_venue': defaultdict(int),
            'by_type': defaultdict(int),
            'by_status': defaultdict(int)
        })
    
    def record_order(self, order: Order):
        """Record order for analytics."""
        analytics = self.analytics_data['overall']
        analytics['total_orders'] += 1
        analytics['by_venue'][order.venue.value] += 1
        analytics['by_type'][order.order_type.value] += 1
        analytics['by_status'][order.status.value] += 1
        
        if order.status == OrderStatus.FILLED:
            analytics['filled_orders'] += 1
            analytics['total_volume'] += order.filled_quantity
            if order.average_price:
                analytics['total_value'] += order.filled_quantity * order.average_price
        elif order.status == OrderStatus.CANCELLED:
            analytics['cancelled_orders'] += 1
        elif order.status == OrderStatus.REJECTED:
            analytics['rejected_orders'] += 1
        
        # Calculate fill rate
        if analytics['total_orders'] > 0:
            analytics['fill_rate'] = analytics['filled_orders'] / analytics['total_orders']
    
    def get_analytics_summary(self, strategy_id: str = None) -> Dict[str, Any]:
        """Get analytics summary."""
        if strategy_id:
            # Filter orders by strategy
            # This would require access to order data
            analytics = self.analytics_data['overall']
        else:
            analytics = self.analytics_data['overall']
        
        summary = {
            'total_orders': analytics['total_orders'],
            'filled_orders': analytics['filled_orders'],
            'cancelled_orders': analytics['cancelled_orders'],
            'rejected_orders': analytics['rejected_orders'],
            'total_volume': analytics['total_volume'],
            'total_value': analytics['total_value'],
            'fill_rate': analytics['fill_rate'],
            'by_venue': dict(analytics['by_venue']),
            'by_type': dict(analytics['by_type']),
            'by_status': dict(analytics['by_status']),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary

class OrderManager:
    """Main live order management system."""
    
    def __init__(self, config: OrderConfig):
        """
        Initialize order manager.
        
        Args:
            config: Order management configuration
        """
        self.config = config
        self.validator = OrderValidator(config)
        self.router = OrderRouter(config)
        self.tracker = OrderTracker()
        self.modifier = OrderModifier(self.tracker)
        self.analytics = OrderAnalytics()
        
        self.running = False
        
        # Setup tracking callbacks
        self.tracker.add_tracking_callback(self._on_order_update)
    
    def submit_order(self, strategy_id: str, symbol: str, side: OrderSide, 
                    order_type: OrderType, quantity: float, price: float = None,
                    stop_price: float = None, venue: ExecutionVenue = None) -> Optional[str]:
        """
        Submit new order.
        
        Args:
            strategy_id: Strategy ID
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            venue: Execution venue
            
        Returns:
            Order ID if successful, None otherwise
        """
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            venue=venue or self.config.default_venue
        )
        
        # Validate order
        is_valid, errors = self.validator.validate_order(order)
        if not is_valid:
            logger.error("Order validation failed", order_id=order.order_id, errors=errors)
            return None
        
        # Track order
        self.tracker.track_order(order)
        
        # Route for execution
        if self.router.route_order(order):
            logger.info("Order submitted successfully", order_id=order.order_id)
            return order.order_id
        else:
            logger.error("Order submission failed", order_id=order.order_id)
            return None
    
    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> bool:
        """Modify existing order."""
        return self.modifier.modify_order(order_id, modifications)
    
    def cancel_order(self, order_id: str, reason: str = "User request") -> bool:
        """Cancel order."""
        return self.modifier.cancel_order(order_id, reason)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.tracker.get_order(order_id)
    
    def get_orders(self, strategy_id: str = None, status: OrderStatus = None) -> List[Order]:
        """Get orders with optional filtering."""
        return self.tracker.get_orders(strategy_id, status)
    
    def get_analytics(self, strategy_id: str = None) -> Dict[str, Any]:
        """Get order analytics."""
        return self.analytics.get_analytics_summary(strategy_id)
    
    def start(self):
        """Start order manager."""
        if not self.running:
            self.running = True
            self.router.start()
            logger.info("Order manager started")
    
    def stop(self):
        """Stop order manager."""
        self.running = False
        self.router.stop()
        logger.info("Order manager stopped")
    
    def _on_order_update(self, order: Order):
        """Handle order update."""
        # Record for analytics
        self.analytics.record_order(order)
        
        logger.debug("Order updated", order_id=order.order_id, status=order.status.value)

def create_order_manager(config: OrderConfig = None) -> OrderManager:
    """
    Create a live order management system.
    
    Args:
        config: Order management configuration
        
    Returns:
        OrderManager instance
    """
    if config is None:
        config = OrderConfig()
    
    return OrderManager(config)

if __name__ == "__main__":
    # Demo of order management system
    config = OrderConfig(
        max_order_size=1000000.0,
        max_daily_orders=1000,
        max_order_value=10000000.0,
        enable_iceberg_orders=True,
        enable_twap_orders=True,
        enable_vwap_orders=True,
        default_venue=ExecutionVenue.PRIMARY,
        order_timeout=300,
        retry_attempts=3
    )
    
    order_manager = create_order_manager(config)
    
    # Start order manager
    order_manager.start()
    
    print("Order management system created successfully!")
    print(f"Max order size: {config.max_order_size}")
    print(f"Default venue: {config.default_venue.value}")
    
    # Submit sample order
    order_id = order_manager.submit_order(
        strategy_id="strategy_001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0
    )
    
    if order_id:
        print(f"Order submitted: {order_id}")
        
        # Get order details
        order = order_manager.get_order(order_id)
        if order:
            print(f"Order status: {order.status.value}")
            print(f"Order venue: {order.venue.value}")
    
    # Get analytics
    analytics = order_manager.get_analytics()
    print(f"Order analytics: {analytics}")