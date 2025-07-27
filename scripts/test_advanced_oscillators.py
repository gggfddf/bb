#!/usr/bin/env python3
"""
Test script for the Advanced Oscillators Module.
Tests SuperTrend, TSI, and Ulcer Index calculations with various scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering.indicators.advanced_oscillators import (
    SuperTrend, TSI, UlcerIndex,
    calculate_supertrend, calculate_tsi, calculate_ulcer_index,
    SignalType, IndicatorResult
)
import structlog

logger = structlog.get_logger()

class AdvancedOscillatorsTestSuite:
    """Comprehensive test suite for advanced oscillators."""
    
    def __init__(self):
        """Initialize test suite with sample data."""
        self.setup_test_data()
    
    def setup_test_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        n_points = 1000
        
        # Generate realistic price data with trends
        base_price = 100.0
        prices = [base_price]
        
        # Create trending data with some volatility
        for i in range(1, n_points):
            # Add trend component
            trend = 0.001 * (i % 200 - 100)  # Oscillating trend
            
            # Add random component
            random_component = np.random.normal(0, 0.02)
            
            # Add momentum
            momentum = 0.1 * (prices[-1] - base_price) / base_price
            
            new_price = prices[-1] * (1 + trend + random_component + momentum)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Generate OHLC data
        self.high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
        self.low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
        self.close = prices
        self.open = np.roll(prices, 1)
        self.open[0] = prices[0]
        
        # Generate volume data
        self.volume = np.random.randint(1000000, 10000000, n_points)
        
        # Calculate returns for testing
        self.returns = np.diff(self.close) / self.close[:-1]
        self.returns = np.insert(self.returns, 0, 0)
        
        logger.info(f"Generated test data with {n_points} points")
    
    async def test_supertrend_basic(self):
        """Test basic SuperTrend calculation."""
        logger.info("Testing SuperTrend basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_supertrend(self.high, self.low, self.close)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert result.values.shape[1] == 3  # supertrend, direction, atr
            assert len(result.signals) == len(self.close)
            
            # Check component relationships
            supertrend = result.values[:, 0]
            direction = result.values[:, 1]
            atr = result.values[:, 2]
            
            valid_mask = ~(np.isnan(supertrend) | np.isnan(direction) | np.isnan(atr))
            assert np.all(atr[valid_mask] >= 0)  # ATR should be positive
            assert np.all(np.isin(direction[valid_mask], [1, -1]))  # Direction should be 1 or -1
            
            logger.info("✅ SuperTrend basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ SuperTrend basic test failed: {e}")
            return False
    
    async def test_supertrend_parameters(self):
        """Test SuperTrend with different parameters."""
        logger.info("Testing SuperTrend parameter variations")
        
        try:
            # Test different periods
            periods = [7, 10, 14, 21]
            for period in periods:
                result = calculate_supertrend(self.high, self.low, self.close, period=period)
                assert result.parameters['period'] == period
            
            # Test different multipliers
            multipliers = [2.0, 3.0, 4.0, 5.0]
            for multiplier in multipliers:
                result = calculate_supertrend(self.high, self.low, self.close, multiplier=multiplier)
                assert result.parameters['multiplier'] == multiplier
            
            logger.info("✅ SuperTrend parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ SuperTrend parameter test failed: {e}")
            return False
    
    async def test_supertrend_optimization(self):
        """Test SuperTrend parameter optimization."""
        logger.info("Testing SuperTrend parameter optimization")
        
        try:
            supertrend = SuperTrend()
            
            param_ranges = {
                'period': [7, 10, 14],
                'multiplier': [2.0, 3.0, 4.0]
            }
            
            optimal_params = supertrend.optimize_parameters(
                self.high, self.low, self.close, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'period' in optimal_params
            assert 'multiplier' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ SuperTrend optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ SuperTrend optimization test failed: {e}")
            return False
    
    async def test_tsi_basic(self):
        """Test basic TSI calculation."""
        logger.info("Testing TSI basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_tsi(self.close)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert result.values.shape[1] == 2  # tsi, signal_line
            assert len(result.signals) == len(self.close)
            
            # Check TSI values
            tsi = result.values[:, 0]
            signal_line = result.values[:, 1]
            
            valid_mask = ~(np.isnan(tsi) | np.isnan(signal_line))
            
            # TSI should be finite
            assert np.all(np.isfinite(tsi[valid_mask]))
            
            # Check signal types
            valid_signals = set(result.signals)
            expected_signals = {SignalType.BUY.value, SignalType.SELL.value, 
                              SignalType.HOLD.value, SignalType.STRONG_BUY.value, 
                              SignalType.STRONG_SELL.value}
            assert valid_signals.issubset(expected_signals)
            
            logger.info("✅ TSI basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ TSI basic test failed: {e}")
            return False
    
    async def test_tsi_parameters(self):
        """Test TSI with different parameters."""
        logger.info("Testing TSI parameter variations")
        
        try:
            # Test different first periods
            first_periods = [20, 25, 30]
            for first_period in first_periods:
                result = calculate_tsi(self.close, first_period=first_period)
                assert result.parameters['first_period'] == first_period
            
            # Test different overbought/oversold levels
            result = calculate_tsi(self.close, overbought=30, oversold=-30)
            assert result.parameters['overbought'] == 30
            assert result.parameters['oversold'] == -30
            
            logger.info("✅ TSI parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ TSI parameter test failed: {e}")
            return False
    
    async def test_tsi_optimization(self):
        """Test TSI parameter optimization."""
        logger.info("Testing TSI parameter optimization")
        
        try:
            tsi = TSI()
            
            param_ranges = {
                'first_period': [20, 25],
                'second_period': [10, 13],
                'signal_period': [7, 9],
                'overbought': [20, 25],
                'oversold': [-25, -20]
            }
            
            optimal_params = tsi.optimize_parameters(
                self.close, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'first_period' in optimal_params
            assert 'second_period' in optimal_params
            assert 'signal_period' in optimal_params
            assert 'overbought' in optimal_params
            assert 'oversold' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ TSI optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ TSI optimization test failed: {e}")
            return False
    
    async def test_ulcer_index_basic(self):
        """Test basic Ulcer Index calculation."""
        logger.info("Testing Ulcer Index basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_ulcer_index(self.close)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert result.values.shape[1] == 3  # ulcer_index, drawdown_pct, squared_drawdown
            assert len(result.signals) == len(self.close)
            
            # Check component values
            ulcer_index = result.values[:, 0]
            drawdown_pct = result.values[:, 1]
            squared_drawdown = result.values[:, 2]
            
            valid_mask = ~(np.isnan(ulcer_index) | np.isnan(drawdown_pct) | np.isnan(squared_drawdown))
            
            # Ulcer Index should be non-negative
            assert np.all(ulcer_index[valid_mask] >= 0)
            
            # Squared drawdown should be non-negative
            assert np.all(squared_drawdown[valid_mask] >= 0)
            
            logger.info("✅ Ulcer Index basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ulcer Index basic test failed: {e}")
            return False
    
    async def test_ulcer_index_parameters(self):
        """Test Ulcer Index with different parameters."""
        logger.info("Testing Ulcer Index parameter variations")
        
        try:
            # Test different periods
            periods = [10, 14, 20, 30]
            for period in periods:
                result = calculate_ulcer_index(self.close, period=period)
                assert result.parameters['period'] == period
            
            # Test different thresholds
            thresholds = [3.0, 5.0, 7.0, 10.0]
            for threshold in thresholds:
                result = calculate_ulcer_index(self.close, threshold=threshold)
                assert result.parameters['threshold'] == threshold
            
            logger.info("✅ Ulcer Index parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ulcer Index parameter test failed: {e}")
            return False
    
    async def test_ulcer_index_optimization(self):
        """Test Ulcer Index parameter optimization."""
        logger.info("Testing Ulcer Index parameter optimization")
        
        try:
            ulcer = UlcerIndex()
            
            param_ranges = {
                'period': [10, 14, 20],
                'threshold': [3.0, 5.0, 7.0]
            }
            
            optimal_params = ulcer.optimize_parameters(
                self.close, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'period' in optimal_params
            assert 'threshold' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ Ulcer Index optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ulcer Index optimization test failed: {e}")
            return False
    
    async def test_error_handling(self):
        """Test error handling for invalid inputs."""
        logger.info("Testing error handling")
        
        try:
            # Test with insufficient data
            short_high = self.high[:5]
            short_low = self.low[:5]
            short_close = self.close[:5]
            
            with self.assertRaises(ValueError):
                calculate_supertrend(short_high, short_low, short_close, period=10)
            
            with self.assertRaises(ValueError):
                calculate_tsi(short_close, first_period=25)
            
            with self.assertRaises(ValueError):
                calculate_ulcer_index(short_close, period=14)
            
            # Test with mismatched array lengths
            with self.assertRaises(ValueError):
                calculate_supertrend(self.high, self.low[:100], self.close)
            
            logger.info("✅ Error handling test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            return False
    
    def assertRaises(self, exception_type):
        """Context manager for testing exceptions."""
        class AssertRaisesContext:
            def __init__(self, exception_type):
                self.exception_type = exception_type
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self.exception_type} to be raised")
                if not issubclass(exc_type, self.exception_type):
                    return False
                return True
        
        return AssertRaisesContext(exception_type)
    
    async def test_signal_generation(self):
        """Test signal generation logic."""
        logger.info("Testing signal generation")
        
        try:
            # Test SuperTrend signals
            result = calculate_supertrend(self.high, self.low, self.close)
            direction = result.values[:, 1]
            
            # Check that trend changes generate signals
            trend_changes = np.diff(direction)
            trend_change_indices = np.where(trend_changes != 0)[0]
            
            if len(trend_change_indices) > 0:
                # Should have signals at trend change points
                signal_at_changes = result.signals[trend_change_indices + 1]  # +1 because of diff
                assert np.any(np.isin(signal_at_changes, [SignalType.BUY.value, SignalType.SELL.value]))
            
            # Test TSI signals
            result = calculate_tsi(self.close)
            tsi = result.values[:, 0]
            
            # Check that extreme TSI values generate signals
            oversold_indices = np.where(tsi <= -25)[0]
            overbought_indices = np.where(tsi >= 25)[0]
            
            if len(oversold_indices) > 0:
                assert np.any(np.isin(result.signals[oversold_indices], 
                                    [SignalType.BUY.value, SignalType.STRONG_BUY.value]))
            
            if len(overbought_indices) > 0:
                assert np.any(np.isin(result.signals[overbought_indices], 
                                    [SignalType.SELL.value, SignalType.STRONG_SELL.value]))
            
            logger.info("✅ Signal generation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal generation test failed: {e}")
            return False
    
    async def test_performance_benchmark(self):
        """Test performance with larger datasets."""
        logger.info("Testing performance benchmark")
        
        try:
            # Generate larger dataset
            large_n = 10000
            large_high = np.random.uniform(90, 110, large_n)
            large_low = large_high * 0.95
            large_close = (large_high + large_low) / 2
            
            import time
            
            # Benchmark SuperTrend
            start_time = time.time()
            result = calculate_supertrend(large_high, large_low, large_close)
            supertrend_time = time.time() - start_time
            
            # Benchmark TSI
            start_time = time.time()
            result = calculate_tsi(large_close)
            tsi_time = time.time() - start_time
            
            # Benchmark Ulcer Index
            start_time = time.time()
            result = calculate_ulcer_index(large_close)
            ulcer_time = time.time() - start_time
            
            logger.info(f"Performance benchmark (10k points):")
            logger.info(f"  SuperTrend: {supertrend_time:.4f}s")
            logger.info(f"  TSI: {tsi_time:.4f}s")
            logger.info(f"  Ulcer Index: {ulcer_time:.4f}s")
            
            # Ensure reasonable performance (less than 2 seconds for 10k points)
            assert supertrend_time < 2.0
            assert tsi_time < 2.0
            assert ulcer_time < 2.0
            
            logger.info("✅ Performance benchmark test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark test failed: {e}")
            return False
    
    async def test_trend_detection(self):
        """Test trend detection capabilities."""
        logger.info("Testing trend detection")
        
        try:
            # Create trending data
            trend_data = np.linspace(100, 150, 500)  # Upward trend
            trend_data = np.concatenate([trend_data, np.linspace(150, 100, 500)])  # Downward trend
            
            # Test SuperTrend on trending data
            high = trend_data * 1.01
            low = trend_data * 0.99
            close = trend_data
            
            result = calculate_supertrend(high, low, close)
            direction = result.values[:, 1]
            
            # Should detect trend changes
            valid_direction = direction[~np.isnan(direction)]
            assert len(valid_direction) > 0
            
            # Test TSI on trending data
            result = calculate_tsi(close)
            tsi = result.values[:, 0]
            
            # TSI should show momentum
            valid_tsi = tsi[~np.isnan(tsi)]
            assert len(valid_tsi) > 0
            
            logger.info("✅ Trend detection test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trend detection test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("Starting Advanced Oscillators Test Suite")
        
        tests = [
            self.test_supertrend_basic,
            self.test_supertrend_parameters,
            self.test_supertrend_optimization,
            self.test_tsi_basic,
            self.test_tsi_parameters,
            self.test_tsi_optimization,
            self.test_ulcer_index_basic,
            self.test_ulcer_index_parameters,
            self.test_ulcer_index_optimization,
            self.test_error_handling,
            self.test_signal_generation,
            self.test_performance_benchmark,
            self.test_trend_detection
        ]
        
        results = []
        for test in tests:
            try:
                result = await test()
                results.append(result)
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with exception: {e}")
                results.append(False)
        
        passed = sum(results)
        total = len(results)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Test Results: {passed}/{total} tests passed")
        logger.info(f"Success Rate: {passed/total*100:.1f}%")
        logger.info(f"{'='*50}")
        
        if passed == total:
            logger.info("🎉 All tests passed! Advanced oscillators are working correctly.")
        else:
            logger.error("❌ Some tests failed. Please check the implementation.")
        
        return passed == total

async def main():
    """Main test execution function."""
    logger.info("Starting Advanced Oscillators Test Suite")
    
    test_suite = AdvancedOscillatorsTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Advanced Oscillators Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Advanced Oscillators Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())