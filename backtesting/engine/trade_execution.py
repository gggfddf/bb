"""
Trade Execution Simulation

A comprehensive trade execution simulation module that models realistic
market conditions including slippage, commissions, order types, and price impact.

Features:
- Realistic slippage simulation based on market conditions
- Commission calculation for different broker types
- Market and limit order types
- Execution delays and latency simulation
- Price impact modeling for large orders
- Order book simulation for limit orders
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import random

logger = structlog.get_logger()

class OrderType(Enum):
    """Order types for trade execution."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
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
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    execution_delay: timedelta = field(default_factory=lambda: timedelta(0))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Result of order execution."""
    order: Order
    execution_price: float
    execution_quantity: float
    commission: float
    slippage: float
    execution_time: datetime
    success: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class SlippageModel:
    """Models slippage based on market conditions."""
    
    def __init__(self, 
                 base_slippage: float = 0.0001,
                 volatility_multiplier: float = 1.0,
                 volume_multiplier: float = 1.0,
                 order_size_multiplier: float = 1.0):
        self.base_slippage = base_slippage
        self.volatility_multiplier = volatility_multiplier
        self.volume_multiplier = volume_multiplier
        self.order_size_multiplier = order_size_multiplier
    
    def calculate_slippage(self, 
                          order_side: OrderSide,
                          order_size: float,
                          market_price: float,
                          volatility: float,
                          volume: float,
                          spread: float = 0.0002) -> float:
        """
        Calculate slippage based on market conditions.
        
        Args:
            order_side: Buy or sell order
            order_size: Size of the order relative to average volume
            market_price: Current market price
            volatility: Current market volatility
            volume: Current trading volume
            spread: Current bid-ask spread
        
        Returns:
            Slippage as a percentage of the market price
        """
        # Base slippage from spread
        base_slippage = spread / 2
        
        # Volatility adjustment
        vol_adjustment = volatility * self.volatility_multiplier
        
        # Volume adjustment (lower volume = higher slippage)
        avg_volume = volume if volume > 0 else 1
        volume_ratio = min(order_size / avg_volume, 1.0)
        volume_adjustment = (1 - volume_ratio) * self.volume_multiplier
        
        # Order size adjustment
        size_adjustment = (order_size / avg_volume) * self.order_size_multiplier
        
        # Combine all factors
        total_slippage = base_slippage + vol_adjustment + volume_adjustment + size_adjustment
        
        # Add some randomness
        random_factor = np.random.normal(1.0, 0.1)
        total_slippage *= random_factor
        
        # Ensure slippage is positive
        total_slippage = max(total_slippage, 0.0)
        
        # Adjust for order side (buys typically have higher slippage)
        if order_side == OrderSide.BUY:
            total_slippage *= 1.1  # Slightly higher for buys
        
        return total_slippage

class CommissionModel:
    """Models commission costs for different broker types."""
    
    def __init__(self, 
                 commission_type: str = "percentage",
                 commission_rate: float = 0.001,
                 min_commission: float = 1.0,
                 max_commission: float = 29.95):
        self.commission_type = commission_type
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.max_commission = max_commission
    
    def calculate_commission(self, 
                           order_value: float,
                           order_quantity: float,
                           broker_type: str = "standard") -> float:
        """
        Calculate commission based on broker type and order characteristics.
        
        Args:
            order_value: Total value of the order
            order_quantity: Number of shares
            broker_type: Type of broker (standard, discount, premium)
        
        Returns:
            Commission amount
        """
        if broker_type == "free":
            return 0.0
        
        elif broker_type == "discount":
            # Fixed commission per trade
            return min(self.min_commission, order_value * 0.0005)
        
        elif broker_type == "standard":
            # Percentage-based commission
            commission = order_value * self.commission_rate
            return max(self.min_commission, min(commission, self.max_commission))
        
        elif broker_type == "premium":
            # Lower percentage for premium accounts
            commission = order_value * (self.commission_rate * 0.5)
            return max(self.min_commission, min(commission, self.max_commission))
        
        else:
            # Default to standard
            commission = order_value * self.commission_rate
            return max(self.min_commission, min(commission, self.max_commission))

class PriceImpactModel:
    """Models price impact of large orders."""
    
    def __init__(self, 
                 impact_multiplier: float = 0.1,
                 volume_threshold: float = 0.01):
        self.impact_multiplier = impact_multiplier
        self.volume_threshold = volume_threshold
    
    def calculate_price_impact(self, 
                             order_size: float,
                             average_volume: float,
                             volatility: float) -> float:
        """
        Calculate price impact of a large order.
        
        Args:
            order_size: Size of the order
            average_volume: Average daily volume
            volatility: Current market volatility
        
        Returns:
            Price impact as a percentage
        """
        if average_volume <= 0:
            return 0.0
        
        # Calculate order size relative to average volume
        relative_size = order_size / average_volume
        
        if relative_size < self.volume_threshold:
            return 0.0  # No impact for small orders
        
        # Calculate impact based on relative size and volatility
        impact = relative_size * self.impact_multiplier * volatility
        
        # Add some randomness
        random_factor = np.random.normal(1.0, 0.2)
        impact *= random_factor
        
        return max(impact, 0.0)

class OrderBookSimulator:
    """Simulates order book for limit order execution."""
    
    def __init__(self, 
                 spread_multiplier: float = 1.0,
                 depth_levels: int = 5):
        self.spread_multiplier = spread_multiplier
        self.depth_levels = depth_levels
    
    def simulate_order_book(self, 
                           market_price: float,
                           volatility: float,
                           volume: float) -> Dict[str, List[Tuple[float, float]]]:
        """
        Simulate order book with bid and ask levels.
        
        Args:
            market_price: Current market price
            volatility: Market volatility
            volume: Trading volume
        
        Returns:
            Dictionary with 'bids' and 'asks' lists of (price, quantity) tuples
        """
        # Calculate spread based on volatility and volume
        base_spread = 0.0002  # 2 basis points
        vol_adjustment = volatility * 0.1
        volume_adjustment = 1.0 / (1.0 + volume / 1000000)  # Higher volume = tighter spread
        
        spread = base_spread * (1 + vol_adjustment) * volume_adjustment * self.spread_multiplier
        
        # Generate bid and ask levels
        bids = []
        asks = []
        
        for level in range(self.depth_levels):
            # Bid levels (below market price)
            bid_price = market_price * (1 - spread/2 - level * spread/4)
            bid_quantity = np.random.uniform(100, 1000) * (1 + level * 0.5)
            bids.append((bid_price, bid_quantity))
            
            # Ask levels (above market price)
            ask_price = market_price * (1 + spread/2 + level * spread/4)
            ask_quantity = np.random.uniform(100, 1000) * (1 + level * 0.5)
            asks.append((ask_price, ask_quantity))
        
        return {
            'bids': sorted(bids, key=lambda x: x[0], reverse=True),  # Highest bid first
            'asks': sorted(asks, key=lambda x: x[0])  # Lowest ask first
        }
    
    def can_fill_limit_order(self, 
                            order: Order,
                            order_book: Dict[str, List[Tuple[float, float]]]) -> Tuple[bool, float, float]:
        """
        Check if a limit order can be filled.
        
        Args:
            order: The limit order
            order_book: Current order book
        
        Returns:
            Tuple of (can_fill, fill_price, fill_quantity)
        """
        if order.order_type != OrderType.LIMIT or order.limit_price is None:
            return False, 0.0, 0.0
        
        if order.side == OrderSide.BUY:
            # Check if we can buy at or below limit price
            for ask_price, ask_quantity in order_book['asks']:
                if ask_price <= order.limit_price:
                    fill_quantity = min(order.quantity, ask_quantity)
                    return True, ask_price, fill_quantity
        else:
            # Check if we can sell at or above limit price
            for bid_price, bid_quantity in order_book['bids']:
                if bid_price >= order.limit_price:
                    fill_quantity = min(order.quantity, bid_quantity)
                    return True, bid_price, fill_quantity
        
        return False, 0.0, 0.0

class TradeExecutionSimulator:
    """Main trade execution simulator."""
    
    def __init__(self,
                 slippage_model: Optional[SlippageModel] = None,
                 commission_model: Optional[CommissionModel] = None,
                 price_impact_model: Optional[PriceImpactModel] = None,
                 order_book_simulator: Optional[OrderBookSimulator] = None,
                 execution_delay_range: Tuple[float, float] = (0.1, 2.0),
                 fill_probability: float = 0.95):
        self.slippage_model = slippage_model or SlippageModel()
        self.commission_model = commission_model or CommissionModel()
        self.price_impact_model = price_impact_model or PriceImpactModel()
        self.order_book_simulator = order_book_simulator or OrderBookSimulator()
        self.execution_delay_range = execution_delay_range
        self.fill_probability = fill_probability
        
        # Order tracking
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.order_counter = 0
    
    def submit_order(self, 
                    symbol: str,
                    side: OrderSide,
                    order_type: OrderType,
                    quantity: float,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    limit_price: Optional[float] = None) -> Order:
        """
        Submit a new order for execution.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            order_type: Type of order
            quantity: Number of shares
            price: Market price (for market orders)
            stop_price: Stop price (for stop orders)
            limit_price: Limit price (for limit orders)
        
        Returns:
            Order object
        """
        self.order_counter += 1
        order_id = f"ORDER_{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            limit_price=limit_price
        )
        
        self.pending_orders.append(order)
        logger.info("Order submitted", 
                   order_id=order_id,
                   symbol=symbol,
                   side=side.value,
                   order_type=order_type.value,
                   quantity=quantity)
        
        return order
    
    def execute_orders(self, 
                      market_data: Dict[str, Any],
                      current_time: datetime) -> List[ExecutionResult]:
        """
        Execute pending orders based on current market conditions.
        
        Args:
            market_data: Current market data (price, volume, volatility, etc.)
            current_time: Current timestamp
        
        Returns:
            List of execution results
        """
        execution_results = []
        
        # Process pending orders
        for order in self.pending_orders[:]:  # Copy list to avoid modification during iteration
            result = self._execute_single_order(order, market_data, current_time)
            if result:
                execution_results.append(result)
                
                # Update order status
                if result.success:
                    if result.execution_quantity >= order.quantity:
                        order.status = OrderStatus.FILLED
                        self.pending_orders.remove(order)
                        self.filled_orders.append(order)
                    else:
                        order.status = OrderStatus.PARTIALLY_FILLED
                        order.quantity -= result.execution_quantity
                
                # Update order with execution details
                order.filled_quantity += result.execution_quantity
                order.filled_price = result.execution_price
                order.commission += result.commission
                order.slippage += result.slippage
                order.execution_delay = result.execution_time - order.timestamp
        
        return execution_results
    
    def _execute_single_order(self, 
                             order: Order,
                             market_data: Dict[str, Any],
                             current_time: datetime) -> Optional[ExecutionResult]:
        """Execute a single order."""
        try:
            # Check if order should be executed based on probability
            if np.random.random() > self.fill_probability:
                return None
            
            # Calculate execution delay
            delay_seconds = np.random.uniform(*self.execution_delay_range)
            execution_time = current_time + timedelta(seconds=delay_seconds)
            
            # Get market conditions
            market_price = market_data.get('price', 100.0)
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 1000000)
            spread = market_data.get('spread', 0.0002)
            
            # Execute based on order type
            if order.order_type == OrderType.MARKET:
                return self._execute_market_order(order, market_data, execution_time)
            
            elif order.order_type == OrderType.LIMIT:
                return self._execute_limit_order(order, market_data, execution_time)
            
            elif order.order_type == OrderType.STOP:
                return self._execute_stop_order(order, market_data, execution_time)
            
            elif order.order_type == OrderType.STOP_LIMIT:
                return self._execute_stop_limit_order(order, market_data, execution_time)
            
            else:
                logger.warning("Unknown order type", order_type=order.order_type.value)
                return None
                
        except Exception as e:
            logger.error("Error executing order", order_id=order.order_id, error=str(e))
            return ExecutionResult(
                order=order,
                execution_price=0.0,
                execution_quantity=0.0,
                commission=0.0,
                slippage=0.0,
                execution_time=current_time,
                success=False,
                reason=f"Execution error: {str(e)}"
            )
    
    def _execute_market_order(self, 
                             order: Order,
                             market_data: Dict[str, Any],
                             execution_time: datetime) -> ExecutionResult:
        """Execute a market order."""
        market_price = market_data.get('price', 100.0)
        volatility = market_data.get('volatility', 0.02)
        volume = market_data.get('volume', 1000000)
        spread = market_data.get('spread', 0.0002)
        
        # Calculate slippage
        slippage_pct = self.slippage_model.calculate_slippage(
            order.side, order.quantity, market_price, volatility, volume, spread
        )
        
        # Calculate price impact
        price_impact_pct = self.price_impact_model.calculate_price_impact(
            order.quantity, volume, volatility
        )
        
        # Calculate execution price
        if order.side == OrderSide.BUY:
            execution_price = market_price * (1 + slippage_pct + price_impact_pct)
        else:
            execution_price = market_price * (1 - slippage_pct - price_impact_pct)
        
        # Calculate commission
        order_value = execution_price * order.quantity
        commission = self.commission_model.calculate_commission(order_value, order.quantity)
        
        return ExecutionResult(
            order=order,
            execution_price=execution_price,
            execution_quantity=order.quantity,
            commission=commission,
            slippage=slippage_pct * market_price * order.quantity,
            execution_time=execution_time,
            success=True
        )
    
    def _execute_limit_order(self, 
                            order: Order,
                            market_data: Dict[str, Any],
                            execution_time: datetime) -> Optional[ExecutionResult]:
        """Execute a limit order."""
        if order.limit_price is None:
            return None
        
        market_price = market_data.get('price', 100.0)
        volatility = market_data.get('volatility', 0.02)
        volume = market_data.get('volume', 1000000)
        
        # Simulate order book
        order_book = self.order_book_simulator.simulate_order_book(market_price, volatility, volume)
        
        # Check if order can be filled
        can_fill, fill_price, fill_quantity = self.order_book_simulator.can_fill_limit_order(order, order_book)
        
        if not can_fill:
            return None
        
        # Calculate commission
        order_value = fill_price * fill_quantity
        commission = self.commission_model.calculate_commission(order_value, fill_quantity)
        
        return ExecutionResult(
            order=order,
            execution_price=fill_price,
            execution_quantity=fill_quantity,
            commission=commission,
            slippage=0.0,  # No slippage for limit orders
            execution_time=execution_time,
            success=True
        )
    
    def _execute_stop_order(self, 
                           order: Order,
                           market_data: Dict[str, Any],
                           execution_time: datetime) -> Optional[ExecutionResult]:
        """Execute a stop order."""
        if order.stop_price is None:
            return None
        
        market_price = market_data.get('price', 100.0)
        
        # Check if stop condition is met
        if order.side == OrderSide.BUY and market_price >= order.stop_price:
            # Stop buy triggered
            return self._execute_market_order(order, market_data, execution_time)
        elif order.side == OrderSide.SELL and market_price <= order.stop_price:
            # Stop sell triggered
            return self._execute_market_order(order, market_data, execution_time)
        
        return None
    
    def _execute_stop_limit_order(self, 
                                 order: Order,
                                 market_data: Dict[str, Any],
                                 execution_time: datetime) -> Optional[ExecutionResult]:
        """Execute a stop-limit order."""
        if order.stop_price is None or order.limit_price is None:
            return None
        
        market_price = market_data.get('price', 100.0)
        
        # Check if stop condition is met
        if order.side == OrderSide.BUY and market_price >= order.stop_price:
            # Stop buy triggered, now execute as limit order
            return self._execute_limit_order(order, market_data, execution_time)
        elif order.side == OrderSide.SELL and market_price <= order.stop_price:
            # Stop sell triggered, now execute as limit order
            return self._execute_limit_order(order, market_data, execution_time)
        
        return None
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status by ID."""
        for order in self.pending_orders + self.filled_orders:
            if order.order_id == order_id:
                return order
        return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for order in self.pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                logger.info("Order cancelled", order_id=order_id)
                return True
        return False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""
        total_orders = len(self.filled_orders)
        total_commission = sum(order.commission for order in self.filled_orders)
        total_slippage = sum(order.slippage for order in self.filled_orders)
        
        return {
            'total_orders': total_orders,
            'pending_orders': len(self.pending_orders),
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'avg_commission': total_commission / total_orders if total_orders > 0 else 0,
            'avg_slippage': total_slippage / total_orders if total_orders > 0 else 0
        }

# Convenience functions
def create_market_order(symbol: str, side: OrderSide, quantity: float) -> Dict[str, Any]:
    """Create a market order."""
    return {
        'symbol': symbol,
        'side': side,
        'order_type': OrderType.MARKET,
        'quantity': quantity
    }

def create_limit_order(symbol: str, side: OrderSide, quantity: float, limit_price: float) -> Dict[str, Any]:
    """Create a limit order."""
    return {
        'symbol': symbol,
        'side': side,
        'order_type': OrderType.LIMIT,
        'quantity': quantity,
        'limit_price': limit_price
    }

def create_stop_order(symbol: str, side: OrderSide, quantity: float, stop_price: float) -> Dict[str, Any]:
    """Create a stop order."""
    return {
        'symbol': symbol,
        'side': side,
        'order_type': OrderType.STOP,
        'quantity': quantity,
        'stop_price': stop_price
    }