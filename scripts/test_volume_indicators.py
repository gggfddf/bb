#!/usr/bin/env python3
"""
Test script for the Volume Indicators Module.
Tests Elder's Force Index and Coppock Curve calculations with various scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering.indicators.volume_indicators import (
    EldersForceIndex, CoppockCurve,
    calculate_elders_force_index, calculate_coppock_curve,
    SignalType, IndicatorResult
)
import structlog

logger = structlog.get_logger()

class VolumeIndicatorsTestSuite:
    """Comprehensive test suite for volume indicators."""
    
    def __init__(self):
        """Initialize test suite with sample data."""
        self.setup_test_data()
    
    def setup_test_data(self):
        """Create sample price and volume data for testing."""
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
        
        # Generate volume data with some correlation to price movement
        base_volume = 5000000
        price_changes = np.abs(np.diff(prices))
        price_changes = np.insert(price_changes, 0, 0)
        
        # Volume increases with price volatility
        volume_multiplier = 1 + price_changes / np.mean(price_changes)
        self.volume = np.random.randint(base_volume, base_volume * 2, n_points) * volume_multiplier
        self.volume = self.volume.astype(int)
        
        # Calculate returns for testing
        self.returns = np.diff(self.close) / self.close[:-1]
        self.returns = np.insert(self.returns, 0, 0)
        
        logger.info(f"Generated test data with {n_points} points")
    
    async def test_elders_force_index_basic(self):
        """Test basic Elder's Force Index calculation."""
        logger.info("Testing Elder's Force Index basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_elders_force_index(self.close, self.volume)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert result.values.shape[1] == 3  # force_index, signal_line, histogram
            assert len(result.signals) == len(self.close)
            
            # Check component relationships
            force_index = result.values[:, 0]
            signal_line = result.values[:, 1]
            histogram = result.values[:, 2]
            
            valid_mask = ~(np.isnan(force_index) | np.isnan(signal_line) | np.isnan(histogram))
            
            # Histogram should equal force_index - signal_line
            if np.sum(valid_mask) > 0:
                calculated_histogram = force_index[valid_mask] - signal_line[valid_mask]
                assert np.allclose(histogram[valid_mask], calculated_histogram, rtol=1e-10)
            
            # Check signal types
            valid_signals = set(result.signals)
            expected_signals = {SignalType.BUY.value, SignalType.SELL.value, 
                              SignalType.HOLD.value, SignalType.STRONG_BUY.value, 
                              SignalType.STRONG_SELL.value}
            assert valid_signals.issubset(expected_signals)
            
            logger.info("✅ Elder's Force Index basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Elder's Force Index basic test failed: {e}")
            return False
    
    async def test_elders_force_index_parameters(self):
        """Test Elder's Force Index with different parameters."""
        logger.info("Testing Elder's Force Index parameter variations")
        
        try:
            # Test different periods
            periods = [10, 13, 20]
            for period in periods:
                result = calculate_elders_force_index(self.close, self.volume, period=period)
                assert result.parameters['period'] == period
            
            # Test different signal periods
            signal_periods = [7, 9, 14]
            for signal_period in signal_periods:
                result = calculate_elders_force_index(self.close, self.volume, signal_period=signal_period)
                assert result.parameters['signal_period'] == signal_period
            
            # Test different thresholds
            thresholds = [0.3, 0.5, 0.7, 1.0]
            for threshold in thresholds:
                result = calculate_elders_force_index(self.close, self.volume, threshold=threshold)
                assert result.parameters['threshold'] == threshold
            
            logger.info("✅ Elder's Force Index parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Elder's Force Index parameter test failed: {e}")
            return False
    
    async def test_elders_force_index_optimization(self):
        """Test Elder's Force Index parameter optimization."""
        logger.info("Testing Elder's Force Index parameter optimization")
        
        try:
            force_index = EldersForceIndex()
            
            param_ranges = {
                'period': [10, 13],
                'signal_period': [7, 9],
                'threshold': [0.3, 0.5, 0.7]
            }
            
            optimal_params = force_index.optimize_parameters(
                self.close, self.volume, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'period' in optimal_params
            assert 'signal_period' in optimal_params
            assert 'threshold' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ Elder's Force Index optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Elder's Force Index optimization test failed: {e}")
            return False
    
    async def test_coppock_curve_basic(self):
        """Test basic Coppock Curve calculation."""
        logger.info("Testing Coppock Curve basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_coppock_curve(self.close)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert result.values.shape[1] == 3  # coppock, roc1, roc2
            assert len(result.signals) == len(self.close)
            
            # Check component values
            coppock = result.values[:, 0]
            roc1 = result.values[:, 1]
            roc2 = result.values[:, 2]
            
            valid_mask = ~(np.isnan(coppock) | np.isnan(roc1) | np.isnan(roc2))
            
            # Coppock should be finite
            assert np.all(np.isfinite(coppock[valid_mask]))
            
            # ROC values should be finite
            assert np.all(np.isfinite(roc1[valid_mask]))
            assert np.all(np.isfinite(roc2[valid_mask]))
            
            logger.info("✅ Coppock Curve basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Coppock Curve basic test failed: {e}")
            return False
    
    async def test_coppock_curve_parameters(self):
        """Test Coppock Curve with different parameters."""
        logger.info("Testing Coppock Curve parameter variations")
        
        try:
            # Test different WMA periods
            wma_periods = [8, 10, 12, 14]
            for wma_period in wma_periods:
                result = calculate_coppock_curve(self.close, wma_period=wma_period)
                assert result.parameters['wma_period'] == wma_period
            
            # Test different ROC periods
            roc1_periods = [11, 14, 20]
            for roc1_period in roc1_periods:
                result = calculate_coppock_curve(self.close, roc1_period=roc1_period)
                assert result.parameters['roc1_period'] == roc1_period
            
            roc2_periods = [10, 11, 14]
            for roc2_period in roc2_periods:
                result = calculate_coppock_curve(self.close, roc2_period=roc2_period)
                assert result.parameters['roc2_period'] == roc2_period
            
            logger.info("✅ Coppock Curve parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Coppock Curve parameter test failed: {e}")
            return False
    
    async def test_coppock_curve_optimization(self):
        """Test Coppock Curve parameter optimization."""
        logger.info("Testing Coppock Curve parameter optimization")
        
        try:
            coppock = CoppockCurve()
            
            param_ranges = {
                'wma_period': [8, 10, 12],
                'roc1_period': [11, 14],
                'roc2_period': [10, 11]
            }
            
            optimal_params = coppock.optimize_parameters(
                self.close, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'wma_period' in optimal_params
            assert 'roc1_period' in optimal_params
            assert 'roc2_period' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ Coppock Curve optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Coppock Curve optimization test failed: {e}")
            return False
    
    async def test_error_handling(self):
        """Test error handling for invalid inputs."""
        logger.info("Testing error handling")
        
        try:
            # Test with insufficient data
            short_close = self.close[:5]
            short_volume = self.volume[:5]
            
            with self.assertRaises(ValueError):
                calculate_elders_force_index(short_close, short_volume, period=10)
            
            with self.assertRaises(ValueError):
                calculate_coppock_curve(short_close, wma_period=10)
            
            # Test with mismatched array lengths
            with self.assertRaises(ValueError):
                calculate_elders_force_index(self.close, self.volume[:100])
            
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
            # Test Elder's Force Index signals
            result = calculate_elders_force_index(self.close, self.volume)
            histogram = result.values[:, 2]
            
            # Check that zero-line crossovers generate signals
            zero_crossings = np.diff(np.sign(histogram))
            zero_crossing_indices = np.where(zero_crossings != 0)[0]
            
            if len(zero_crossing_indices) > 0:
                # Should have signals at zero crossing points
                signal_at_crossings = result.signals[zero_crossing_indices + 1]  # +1 because of diff
                assert np.any(np.isin(signal_at_crossings, [SignalType.BUY.value, SignalType.SELL.value]))
            
            # Test Coppock Curve signals
            result = calculate_coppock_curve(self.close)
            coppock = result.values[:, 0]
            
            # Check that zero-line crossovers generate signals
            zero_crossings = np.diff(np.sign(coppock))
            zero_crossing_indices = np.where(zero_crossings != 0)[0]
            
            if len(zero_crossing_indices) > 0:
                # Should have signals at zero crossing points
                signal_at_crossings = result.signals[zero_crossing_indices + 1]  # +1 because of diff
                assert np.any(np.isin(signal_at_crossings, [SignalType.BUY.value, SignalType.SELL.value]))
            
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
            large_close = np.random.uniform(90, 110, large_n)
            large_volume = np.random.randint(1000000, 10000000, large_n)
            
            import time
            
            # Benchmark Elder's Force Index
            start_time = time.time()
            result = calculate_elders_force_index(large_close, large_volume)
            force_index_time = time.time() - start_time
            
            # Benchmark Coppock Curve
            start_time = time.time()
            result = calculate_coppock_curve(large_close)
            coppock_time = time.time() - start_time
            
            logger.info(f"Performance benchmark (10k points):")
            logger.info(f"  Elder's Force Index: {force_index_time:.4f}s")
            logger.info(f"  Coppock Curve: {coppock_time:.4f}s")
            
            # Ensure reasonable performance (less than 2 seconds for 10k points)
            assert force_index_time < 2.0
            assert coppock_time < 2.0
            
            logger.info("✅ Performance benchmark test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark test failed: {e}")
            return False
    
    async def test_volume_price_relationship(self):
        """Test volume-price relationship detection."""
        logger.info("Testing volume-price relationship")
        
        try:
            # Create data with strong volume-price relationship
            n_points = 500
            base_price = 100.0
            prices = [base_price]
            volumes = [5000000]
            
            for i in range(1, n_points):
                # Price movement
                price_change = np.random.normal(0, 0.02)
                new_price = prices[-1] * (1 + price_change)
                prices.append(new_price)
                
                # Volume increases with price volatility
                volume_change = 1 + abs(price_change) * 10
                new_volume = int(volumes[-1] * volume_change)
                volumes.append(new_volume)
            
            prices = np.array(prices)
            volumes = np.array(volumes)
            
            # Test Elder's Force Index on this data
            result = calculate_elders_force_index(prices, volumes)
            force_index = result.values[:, 0]
            
            # Should detect volume-price relationships
            valid_force_index = force_index[~np.isnan(force_index)]
            assert len(valid_force_index) > 0
            
            # Test Coppock Curve on trending data
            trending_prices = np.linspace(100, 150, 500)  # Upward trend
            result = calculate_coppock_curve(trending_prices)
            coppock = result.values[:, 0]
            
            # Should detect trend
            valid_coppock = coppock[~np.isnan(coppock)]
            assert len(valid_coppock) > 0
            
            logger.info("✅ Volume-price relationship test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Volume-price relationship test failed: {e}")
            return False
    
    async def test_divergence_detection(self):
        """Test divergence detection capabilities."""
        logger.info("Testing divergence detection")
        
        try:
            # Create data with price-volume divergence
            n_points = 300
            base_price = 100.0
            prices = [base_price]
            volumes = [5000000]
            
            for i in range(1, n_points):
                if i < 150:
                    # Price up, volume down (bearish divergence)
                    price_change = 0.01
                    volume_change = 0.8
                else:
                    # Price down, volume up (bullish divergence)
                    price_change = -0.01
                    volume_change = 1.2
                
                new_price = prices[-1] * (1 + price_change)
                new_volume = int(volumes[-1] * volume_change)
                
                prices.append(new_price)
                volumes.append(new_volume)
            
            prices = np.array(prices)
            volumes = np.array(volumes)
            
            # Test Elder's Force Index divergence detection
            result = calculate_elders_force_index(prices, volumes)
            
            # Should generate signals for divergence
            signals = result.signals
            assert np.any(np.isin(signals, [SignalType.BUY.value, SignalType.SELL.value]))
            
            logger.info("✅ Divergence detection test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Divergence detection test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("Starting Volume Indicators Test Suite")
        
        tests = [
            self.test_elders_force_index_basic,
            self.test_elders_force_index_parameters,
            self.test_elders_force_index_optimization,
            self.test_coppock_curve_basic,
            self.test_coppock_curve_parameters,
            self.test_coppock_curve_optimization,
            self.test_error_handling,
            self.test_signal_generation,
            self.test_performance_benchmark,
            self.test_volume_price_relationship,
            self.test_divergence_detection
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
            logger.info("🎉 All tests passed! Volume indicators are working correctly.")
        else:
            logger.error("❌ Some tests failed. Please check the implementation.")
        
        return passed == total

async def main():
    """Main test execution function."""
    logger.info("Starting Volume Indicators Test Suite")
    
    test_suite = VolumeIndicatorsTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Volume Indicators Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Volume Indicators Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())