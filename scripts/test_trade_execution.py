#!/usr/bin/env python3
"""
Test script for the Trade Execution Simulation Module.
Tests slippage models, commission models, order types, and execution simulation.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.engine.trade_execution import (
    SlippageModel, CommissionModel, PriceImpactModel, OrderBookSimulator,
    TradeExecutionSimulator, OrderType, OrderSide, OrderStatus,
    create_market_order, create_limit_order, create_stop_order
)
import structlog

logger = structlog.get_logger()

class TradeExecutionTestSuite:
    """Comprehensive test suite for the trade execution simulation."""
    
    def __init__(self):
        self.test_results = {}
    
    async def test_slippage_model(self):
        """Test the SlippageModel class."""
        logger.info("Testing SlippageModel")
        
        try:
            slippage_model = SlippageModel(
                base_slippage=0.0001,
                volatility_multiplier=1.0,
                volume_multiplier=1.0,
                order_size_multiplier=1.0
            )
            
            # Test basic slippage calculation
            slippage_buy = slippage_model.calculate_slippage(
                order_side=OrderSide.BUY,
                order_size=1000,
                market_price=100.0,
                volatility=0.02,
                volume=1000000,
                spread=0.0002
            )
            
            slippage_sell = slippage_model.calculate_slippage(
                order_side=OrderSide.SELL,
                order_size=1000,
                market_price=100.0,
                volatility=0.02,
                volume=1000000,
                spread=0.0002
            )
            
            # Basic assertions
            assert slippage_buy > 0, "Buy slippage should be positive"
            assert slippage_sell > 0, "Sell slippage should be positive"
            assert slippage_buy >= slippage_sell, "Buy slippage should be >= sell slippage"
            
            # Test with different market conditions
            high_vol_slippage = slippage_model.calculate_slippage(
                order_side=OrderSide.BUY,
                order_size=1000,
                market_price=100.0,
                volatility=0.05,  # Higher volatility
                volume=1000000,
                spread=0.0002
            )
            
            low_vol_slippage = slippage_model.calculate_slippage(
                order_side=OrderSide.BUY,
                order_size=1000,
                market_price=100.0,
                volatility=0.01,  # Lower volatility
                volume=1000000,
                spread=0.0002
            )
            
            assert high_vol_slippage > low_vol_slippage, "Higher volatility should result in higher slippage"
            
            self.test_results['slippage_model'] = True
            logger.info("SlippageModel tests passed",
                       buy_slippage=slippage_buy,
                       sell_slippage=slippage_sell,
                       high_vol_slippage=high_vol_slippage,
                       low_vol_slippage=low_vol_slippage)
            
        except Exception as e:
            self.test_results['slippage_model'] = False
            logger.error("SlippageModel tests failed", error=str(e))
            raise
    
    async def test_commission_model(self):
        """Test the CommissionModel class."""
        logger.info("Testing CommissionModel")
        
        try:
            commission_model = CommissionModel(
                commission_type="percentage",
                commission_rate=0.001,
                min_commission=1.0,
                max_commission=29.95
            )
            
            # Test different broker types
            order_value = 10000.0
            order_quantity = 100
            
            # Free broker
            free_commission = commission_model.calculate_commission(
                order_value, order_quantity, "free"
            )
            assert free_commission == 0.0, "Free broker should have zero commission"
            
            # Standard broker
            standard_commission = commission_model.calculate_commission(
                order_value, order_quantity, "standard"
            )
            assert standard_commission > 0, "Standard broker should have positive commission"
            assert standard_commission <= 29.95, "Commission should not exceed max"
            
            # Discount broker
            discount_commission = commission_model.calculate_commission(
                order_value, order_quantity, "discount"
            )
            assert discount_commission > 0, "Discount broker should have positive commission"
            
            # Premium broker
            premium_commission = commission_model.calculate_commission(
                order_value, order_quantity, "premium"
            )
            assert premium_commission > 0, "Premium broker should have positive commission"
            assert premium_commission <= standard_commission, "Premium commission should be <= standard"
            
            # Test minimum commission
            small_order_value = 100.0
            small_commission = commission_model.calculate_commission(
                small_order_value, 1, "standard"
            )
            assert small_commission >= 1.0, "Commission should respect minimum"
            
            self.test_results['commission_model'] = True
            logger.info("CommissionModel tests passed",
                       free_commission=free_commission,
                       standard_commission=standard_commission,
                       discount_commission=discount_commission,
                       premium_commission=premium_commission,
                       small_commission=small_commission)
            
        except Exception as e:
            self.test_results['commission_model'] = False
            logger.error("CommissionModel tests failed", error=str(e))
            raise
    
    async def test_price_impact_model(self):
        """Test the PriceImpactModel class."""
        logger.info("Testing PriceImpactModel")
        
        try:
            price_impact_model = PriceImpactModel(
                impact_multiplier=0.1,
                volume_threshold=0.01
            )
            
            # Test small order (no impact)
            small_impact = price_impact_model.calculate_price_impact(
                order_size=1000,
                average_volume=1000000,
                volatility=0.02
            )
            assert small_impact == 0.0, "Small orders should have no price impact"
            
            # Test large order (should have impact)
            large_impact = price_impact_model.calculate_price_impact(
                order_size=50000,
                average_volume=1000000,
                volatility=0.02
            )
            assert large_impact > 0, "Large orders should have positive price impact"
            
            # Test with zero volume
            zero_vol_impact = price_impact_model.calculate_price_impact(
                order_size=1000,
                average_volume=0,
                volatility=0.02
            )
            assert zero_vol_impact == 0.0, "Zero volume should result in zero impact"
            
            # Test with different volatility
            high_vol_impact = price_impact_model.calculate_price_impact(
                order_size=50000,
                average_volume=1000000,
                volatility=0.05
            )
            
            low_vol_impact = price_impact_model.calculate_price_impact(
                order_size=50000,
                average_volume=1000000,
                volatility=0.01
            )
            
            assert high_vol_impact > low_vol_impact, "Higher volatility should result in higher impact"
            
            self.test_results['price_impact_model'] = True
            logger.info("PriceImpactModel tests passed",
                       small_impact=small_impact,
                       large_impact=large_impact,
                       high_vol_impact=high_vol_impact,
                       low_vol_impact=low_vol_impact)
            
        except Exception as e:
            self.test_results['price_impact_model'] = False
            logger.error("PriceImpactModel tests failed", error=str(e))
            raise
    
    async def test_order_book_simulator(self):
        """Test the OrderBookSimulator class."""
        logger.info("Testing OrderBookSimulator")
        
        try:
            order_book_sim = OrderBookSimulator(
                spread_multiplier=1.0,
                depth_levels=5
            )
            
            # Test order book simulation
            order_book = order_book_sim.simulate_order_book(
                market_price=100.0,
                volatility=0.02,
                volume=1000000
            )
            
            # Check structure
            assert 'bids' in order_book, "Order book should have bids"
            assert 'asks' in order_book, "Order book should have asks"
            assert len(order_book['bids']) == 5, "Should have 5 bid levels"
            assert len(order_book['asks']) == 5, "Should have 5 ask levels"
            
            # Check bid/ask ordering
            bid_prices = [price for price, _ in order_book['bids']]
            ask_prices = [price for price, _ in order_book['asks']]
            
            assert bid_prices == sorted(bid_prices, reverse=True), "Bids should be sorted descending"
            assert ask_prices == sorted(ask_prices), "Asks should be sorted ascending"
            
            # Check that bids are below asks
            assert max(bid_prices) < min(ask_prices), "Bids should be below asks"
            
            # Test limit order fillability
            from backtesting.engine.trade_execution import Order
            
            # Test buy limit order
            buy_order = Order(
                order_id="TEST_001",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100,
                limit_price=min(ask_prices) - 0.01  # Below best ask
            )
            
            can_fill, fill_price, fill_quantity = order_book_sim.can_fill_limit_order(buy_order, order_book)
            assert can_fill, "Buy limit order should be fillable"
            assert fill_price <= buy_order.limit_price, "Fill price should be <= limit price"
            
            # Test sell limit order
            sell_order = Order(
                order_id="TEST_002",
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=100,
                limit_price=max(bid_prices) + 0.01  # Above best bid
            )
            
            can_fill, fill_price, fill_quantity = order_book_sim.can_fill_limit_order(sell_order, order_book)
            assert can_fill, "Sell limit order should be fillable"
            assert fill_price >= sell_order.limit_price, "Fill price should be >= limit price"
            
            self.test_results['order_book_simulator'] = True
            logger.info("OrderBookSimulator tests passed",
                       best_bid=max(bid_prices),
                       best_ask=min(ask_prices),
                       spread=min(ask_prices) - max(bid_prices))
            
        except Exception as e:
            self.test_results['order_book_simulator'] = False
            logger.error("OrderBookSimulator tests failed", error=str(e))
            raise
    
    async def test_market_order_execution(self):
        """Test market order execution."""
        logger.info("Testing market order execution")
        
        try:
            simulator = TradeExecutionSimulator()
            
            # Submit market order
            order = simulator.submit_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100
            )
            
            assert order.order_id is not None, "Order should have ID"
            assert order.status == OrderStatus.PENDING, "Order should be pending"
            assert len(simulator.pending_orders) == 1, "Should have one pending order"
            
            # Execute order
            market_data = {
                'price': 100.0,
                'volatility': 0.02,
                'volume': 1000000,
                'spread': 0.0002
            }
            
            current_time = datetime.now()
            results = simulator.execute_orders(market_data, current_time)
            
            assert len(results) > 0, "Should have execution results"
            
            result = results[0]
            assert result.success, "Order should execute successfully"
            assert result.execution_quantity == 100, "Should execute full quantity"
            assert result.execution_price > 0, "Execution price should be positive"
            assert result.commission > 0, "Commission should be positive"
            
            # Check order status
            assert order.status == OrderStatus.FILLED, "Order should be filled"
            assert len(simulator.pending_orders) == 0, "No pending orders"
            assert len(simulator.filled_orders) == 1, "One filled order"
            
            self.test_results['market_order_execution'] = True
            logger.info("Market order execution test passed",
                       execution_price=result.execution_price,
                       commission=result.commission,
                       slippage=result.slippage)
            
        except Exception as e:
            self.test_results['market_order_execution'] = False
            logger.error("Market order execution test failed", error=str(e))
            raise
    
    async def test_limit_order_execution(self):
        """Test limit order execution."""
        logger.info("Testing limit order execution")
        
        try:
            simulator = TradeExecutionSimulator()
            
            # Submit limit order
            order = simulator.submit_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100,
                limit_price=99.0  # Below market
            )
            
            # Execute with market price above limit (should not fill)
            market_data = {
                'price': 100.0,
                'volatility': 0.02,
                'volume': 1000000,
                'spread': 0.0002
            }
            
            current_time = datetime.now()
            results = simulator.execute_orders(market_data, current_time)
            
            # Order should not fill
            assert len(results) == 0, "Order should not fill when price is above limit"
            assert order.status == OrderStatus.PENDING, "Order should still be pending"
            
            # Execute with market price below limit (should fill)
            market_data['price'] = 98.0
            results = simulator.execute_orders(market_data, current_time)
            
            # Order should fill
            assert len(results) > 0, "Order should fill when price is below limit"
            result = results[0]
            assert result.success, "Order should execute successfully"
            assert result.execution_price <= order.limit_price, "Execution price should be <= limit"
            
            self.test_results['limit_order_execution'] = True
            logger.info("Limit order execution test passed",
                       execution_price=result.execution_price,
                       limit_price=order.limit_price)
            
        except Exception as e:
            self.test_results['limit_order_execution'] = False
            logger.error("Limit order execution test failed", error=str(e))
            raise
    
    async def test_stop_order_execution(self):
        """Test stop order execution."""
        logger.info("Testing stop order execution")
        
        try:
            simulator = TradeExecutionSimulator()
            
            # Submit stop buy order
            order = simulator.submit_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
                quantity=100,
                stop_price=102.0  # Above current market
            )
            
            # Execute with price below stop (should not trigger)
            market_data = {
                'price': 100.0,
                'volatility': 0.02,
                'volume': 1000000,
                'spread': 0.0002
            }
            
            current_time = datetime.now()
            results = simulator.execute_orders(market_data, current_time)
            
            # Order should not trigger
            assert len(results) == 0, "Stop order should not trigger when price is below stop"
            assert order.status == OrderStatus.PENDING, "Order should still be pending"
            
            # Execute with price above stop (should trigger)
            market_data['price'] = 103.0
            results = simulator.execute_orders(market_data, current_time)
            
            # Order should trigger and execute as market order
            assert len(results) > 0, "Stop order should trigger when price is above stop"
            result = results[0]
            assert result.success, "Order should execute successfully"
            assert result.execution_price > 0, "Execution price should be positive"
            
            self.test_results['stop_order_execution'] = True
            logger.info("Stop order execution test passed",
                       execution_price=result.execution_price,
                       stop_price=order.stop_price)
            
        except Exception as e:
            self.test_results['stop_order_execution'] = False
            logger.error("Stop order execution test failed", error=str(e))
            raise
    
    async def test_order_cancellation(self):
        """Test order cancellation."""
        logger.info("Testing order cancellation")
        
        try:
            simulator = TradeExecutionSimulator()
            
            # Submit order
            order = simulator.submit_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100,
                limit_price=99.0
            )
            
            assert len(simulator.pending_orders) == 1, "Should have one pending order"
            
            # Cancel order
            success = simulator.cancel_order(order.order_id)
            assert success, "Order cancellation should succeed"
            
            assert len(simulator.pending_orders) == 0, "No pending orders after cancellation"
            assert order.status == OrderStatus.CANCELLED, "Order status should be cancelled"
            
            # Try to cancel non-existent order
            success = simulator.cancel_order("NON_EXISTENT")
            assert not success, "Cancelling non-existent order should fail"
            
            self.test_results['order_cancellation'] = True
            logger.info("Order cancellation test passed")
            
        except Exception as e:
            self.test_results['order_cancellation'] = False
            logger.error("Order cancellation test failed", error=str(e))
            raise
    
    async def test_execution_summary(self):
        """Test execution summary functionality."""
        logger.info("Testing execution summary")
        
        try:
            simulator = TradeExecutionSimulator()
            
            # Submit and execute multiple orders
            for i in range(5):
                order = simulator.submit_order(
                    symbol="AAPL",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=100
                )
                
                market_data = {
                    'price': 100.0 + i,
                    'volatility': 0.02,
                    'volume': 1000000,
                    'spread': 0.0002
                }
                
                current_time = datetime.now()
                simulator.execute_orders(market_data, current_time)
            
            # Get summary
            summary = simulator.get_execution_summary()
            
            assert summary['total_orders'] == 5, "Should have 5 total orders"
            assert summary['pending_orders'] == 0, "No pending orders"
            assert summary['total_commission'] > 0, "Should have positive total commission"
            assert summary['total_slippage'] > 0, "Should have positive total slippage"
            assert summary['avg_commission'] > 0, "Should have positive average commission"
            assert summary['avg_slippage'] > 0, "Should have positive average slippage"
            
            self.test_results['execution_summary'] = True
            logger.info("Execution summary test passed",
                       total_orders=summary['total_orders'],
                       total_commission=summary['total_commission'],
                       total_slippage=summary['total_slippage'])
            
        except Exception as e:
            self.test_results['execution_summary'] = False
            logger.error("Execution summary test failed", error=str(e))
            raise
    
    async def test_convenience_functions(self):
        """Test convenience functions."""
        logger.info("Testing convenience functions")
        
        try:
            # Test market order creation
            market_order = create_market_order("AAPL", OrderSide.BUY, 100)
            assert market_order['symbol'] == "AAPL"
            assert market_order['side'] == OrderSide.BUY
            assert market_order['order_type'] == OrderType.MARKET
            assert market_order['quantity'] == 100
            
            # Test limit order creation
            limit_order = create_limit_order("AAPL", OrderSide.SELL, 100, 99.0)
            assert limit_order['symbol'] == "AAPL"
            assert limit_order['side'] == OrderSide.SELL
            assert limit_order['order_type'] == OrderType.LIMIT
            assert limit_order['quantity'] == 100
            assert limit_order['limit_price'] == 99.0
            
            # Test stop order creation
            stop_order = create_stop_order("AAPL", OrderSide.BUY, 100, 102.0)
            assert stop_order['symbol'] == "AAPL"
            assert stop_order['side'] == OrderSide.BUY
            assert stop_order['order_type'] == OrderType.STOP
            assert stop_order['quantity'] == 100
            assert stop_order['stop_price'] == 102.0
            
            self.test_results['convenience_functions'] = True
            logger.info("Convenience functions test passed")
            
        except Exception as e:
            self.test_results['convenience_functions'] = False
            logger.error("Convenience functions test failed", error=str(e))
            raise
    
    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("Starting Trade Execution Test Suite")
        
        test_methods = [
            self.test_slippage_model,
            self.test_commission_model,
            self.test_price_impact_model,
            self.test_order_book_simulator,
            self.test_market_order_execution,
            self.test_limit_order_execution,
            self.test_stop_order_execution,
            self.test_order_cancellation,
            self.test_execution_summary,
            self.test_convenience_functions
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed", error=str(e))
                self.test_results[test_method.__name__] = False
        
        # Summary
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        
        logger.info("Trade Execution Test Suite completed",
                   passed_tests=passed_tests,
                   total_tests=total_tests,
                   success_rate=passed_tests/total_tests)
        
        return passed_tests == total_tests

async def main():
    """Main test execution function."""
    logger.info("Starting Trade Execution Test Suite")
    
    test_suite = TradeExecutionTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Trade Execution Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Trade Execution Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())