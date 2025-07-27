#!/usr/bin/env python3
"""
Test script for the Chaikin Indicators Module.
Tests Chaikin Oscillator and Chaikin Money Flow calculations with various scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering.indicators.chaikin_indicators import (
    ChaikinOscillator, ChaikinMoneyFlow,
    calculate_chaikin_oscillator, calculate_chaikin_money_flow,
    SignalType, IndicatorResult
)
import structlog

logger = structlog.get_logger()

class ChaikinIndicatorsTestSuite:
    """Comprehensive test suite for Chaikin indicators."""
    
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
    
    async def test_chaikin_oscillator_basic(self):
        """Test basic Chaikin Oscillator calculation."""
        logger.info("Testing Chaikin Oscillator basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_chaikin_oscillator(self.high, self.low, self.close, self.volume)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert result.values.shape[1] == 4  # oscillator, signal_line, histogram, ad_line
            assert len(result.signals) == len(self.close)
            
            # Check component relationships
            oscillator = result.values[:, 0]
            signal_line = result.values[:, 1]
            histogram = result.values[:, 2]
            ad_line = result.values[:, 3]
            
            valid_mask = ~(np.isnan(oscillator) | np.isnan(signal_line) | np.isnan(histogram))
            
            # Histogram should equal oscillator - signal_line
            if np.sum(valid_mask) > 0:
                calculated_histogram = oscillator[valid_mask] - signal_line[valid_mask]
                assert np.allclose(histogram[valid_mask], calculated_histogram, rtol=1e-10)
            
            # Check signal types
            valid_signals = set(result.signals)
            expected_signals = {SignalType.BUY.value, SignalType.SELL.value, 
                              SignalType.HOLD.value, SignalType.STRONG_BUY.value, 
                              SignalType.STRONG_SELL.value}
            assert valid_signals.issubset(expected_signals)
            
            logger.info("✅ Chaikin Oscillator basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Chaikin Oscillator basic test failed: {e}")
            return False
    
    async def test_chaikin_oscillator_parameters(self):
        """Test Chaikin Oscillator with different parameters."""
        logger.info("Testing Chaikin Oscillator parameter variations")
        
        try:
            # Test different fast periods
            fast_periods = [3, 5, 7]
            for fast_period in fast_periods:
                result = calculate_chaikin_oscillator(self.high, self.low, self.close, self.volume, 
                                                    fast_period=fast_period)
                assert result.parameters['fast_period'] == fast_period
            
            # Test different slow periods
            slow_periods = [10, 14, 20]
            for slow_period in slow_periods:
                result = calculate_chaikin_oscillator(self.high, self.low, self.close, self.volume, 
                                                    slow_period=slow_period)
                assert result.parameters['slow_period'] == slow_period
            
            # Test different signal periods
            signal_periods = [7, 9, 14]
            for signal_period in signal_periods:
                result = calculate_chaikin_oscillator(self.high, self.low, self.close, self.volume, 
                                                    signal_period=signal_period)
                assert result.parameters['signal_period'] == signal_period
            
            logger.info("✅ Chaikin Oscillator parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Chaikin Oscillator parameter test failed: {e}")
            return False
    
    async def test_chaikin_oscillator_optimization(self):
        """Test Chaikin Oscillator parameter optimization."""
        logger.info("Testing Chaikin Oscillator parameter optimization")
        
        try:
            chaikin_osc = ChaikinOscillator()
            
            param_ranges = {
                'fast_period': [3, 5],
                'slow_period': [10, 14],
                'signal_period': [7, 9]
            }
            
            optimal_params = chaikin_osc.optimize_parameters(
                self.high, self.low, self.close, self.volume, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'fast_period' in optimal_params
            assert 'slow_period' in optimal_params
            assert 'signal_period' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ Chaikin Oscillator optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Chaikin Oscillator optimization test failed: {e}")
            return False
    
    async def test_chaikin_money_flow_basic(self):
        """Test basic Chaikin Money Flow calculation."""
        logger.info("Testing Chaikin Money Flow basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_chaikin_money_flow(self.high, self.low, self.close, self.volume)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert result.values.shape[1] == 3  # cmf, mfm, mfv
            assert len(result.signals) == len(self.close)
            
            # Check component values
            cmf = result.values[:, 0]
            mfm = result.values[:, 1]
            mfv = result.values[:, 2]
            
            valid_mask = ~(np.isnan(cmf) | np.isnan(mfm) | np.isnan(mfv))
            
            # CMF should be between -1 and 1
            if np.sum(valid_mask) > 0:
                assert np.all(cmf[valid_mask] >= -1) and np.all(cmf[valid_mask] <= 1)
            
            # MFM should be between -1 and 1
            valid_mfm = mfm[~np.isnan(mfm)]
            if len(valid_mfm) > 0:
                assert np.all(valid_mfm >= -1) and np.all(valid_mfm <= 1)
            
            logger.info("✅ Chaikin Money Flow basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Chaikin Money Flow basic test failed: {e}")
            return False
    
    async def test_chaikin_money_flow_parameters(self):
        """Test Chaikin Money Flow with different parameters."""
        logger.info("Testing Chaikin Money Flow parameter variations")
        
        try:
            # Test different periods
            periods = [14, 20, 30]
            for period in periods:
                result = calculate_chaikin_money_flow(self.high, self.low, self.close, self.volume, 
                                                    period=period)
                assert result.parameters['period'] == period
            
            # Test different overbought/oversold levels
            result = calculate_chaikin_money_flow(self.high, self.low, self.close, self.volume, 
                                                overbought=0.3, oversold=-0.3)
            assert result.parameters['overbought'] == 0.3
            assert result.parameters['oversold'] == -0.3
            
            logger.info("✅ Chaikin Money Flow parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Chaikin Money Flow parameter test failed: {e}")
            return False
    
    async def test_chaikin_money_flow_optimization(self):
        """Test Chaikin Money Flow parameter optimization."""
        logger.info("Testing Chaikin Money Flow parameter optimization")
        
        try:
            chaikin_mf = ChaikinMoneyFlow()
            
            param_ranges = {
                'period': [14, 20],
                'overbought': [0.2, 0.25],
                'oversold': [-0.25, -0.2]
            }
            
            optimal_params = chaikin_mf.optimize_parameters(
                self.high, self.low, self.close, self.volume, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'period' in optimal_params
            assert 'overbought' in optimal_params
            assert 'oversold' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ Chaikin Money Flow optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Chaikin Money Flow optimization test failed: {e}")
            return False
    
    async def test_error_handling(self):
        """Test error handling for invalid inputs."""
        logger.info("Testing error handling")
        
        try:
            # Test with insufficient data
            short_high = self.high[:5]
            short_low = self.low[:5]
            short_close = self.close[:5]
            short_volume = self.volume[:5]
            
            with self.assertRaises(ValueError):
                calculate_chaikin_oscillator(short_high, short_low, short_close, short_volume, 
                                           fast_period=3, slow_period=10)
            
            with self.assertRaises(ValueError):
                calculate_chaikin_money_flow(short_high, short_low, short_close, short_volume, 
                                           period=20)
            
            # Test with mismatched array lengths
            with self.assertRaises(ValueError):
                calculate_chaikin_oscillator(self.high, self.low[:100], self.close, self.volume)
            
            # Test with invalid parameters
            with self.assertRaises(ValueError):
                calculate_chaikin_oscillator(self.high, self.low, self.close, self.volume, 
                                           fast_period=10, slow_period=5)  # fast >= slow
            
            with self.assertRaises(ValueError):
                calculate_chaikin_money_flow(self.high, self.low, self.close, self.volume, 
                                           overbought=-0.25, oversold=0.25)  # overbought <= oversold
            
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
            # Test Chaikin Oscillator signals
            result = calculate_chaikin_oscillator(self.high, self.low, self.close, self.volume)
            oscillator = result.values[:, 0]
            
            # Check that zero-line crossovers generate signals
            zero_crossings = np.diff(np.sign(oscillator))
            zero_crossing_indices = np.where(zero_crossings != 0)[0]
            
            if len(zero_crossing_indices) > 0:
                # Should have signals at zero crossing points
                signal_at_crossings = result.signals[zero_crossing_indices + 1]  # +1 because of diff
                assert np.any(np.isin(signal_at_crossings, [SignalType.BUY.value, SignalType.SELL.value]))
            
            # Test Chaikin Money Flow signals
            result = calculate_chaikin_money_flow(self.high, self.low, self.close, self.volume)
            cmf = result.values[:, 0]
            
            # Check that zero-line crossovers generate signals
            zero_crossings = np.diff(np.sign(cmf))
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
            large_high = np.random.uniform(90, 110, large_n)
            large_low = large_high * 0.95
            large_close = (large_high + large_low) / 2
            large_volume = np.random.randint(1000000, 10000000, large_n)
            
            import time
            
            # Benchmark Chaikin Oscillator
            start_time = time.time()
            result = calculate_chaikin_oscillator(large_high, large_low, large_close, large_volume)
            oscillator_time = time.time() - start_time
            
            # Benchmark Chaikin Money Flow
            start_time = time.time()
            result = calculate_chaikin_money_flow(large_high, large_low, large_close, large_volume)
            money_flow_time = time.time() - start_time
            
            logger.info(f"Performance benchmark (10k points):")
            logger.info(f"  Chaikin Oscillator: {oscillator_time:.4f}s")
            logger.info(f"  Chaikin Money Flow: {money_flow_time:.4f}s")
            
            # Ensure reasonable performance (less than 2 seconds for 10k points)
            assert oscillator_time < 2.0
            assert money_flow_time < 2.0
            
            logger.info("✅ Performance benchmark test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark test failed: {e}")
            return False
    
    async def test_money_flow_analysis(self):
        """Test money flow analysis capabilities."""
        logger.info("Testing money flow analysis")
        
        try:
            # Create data with strong buying pressure
            n_points = 500
            base_price = 100.0
            prices = [base_price]
            volumes = [5000000]
            
            for i in range(1, n_points):
                # Price movement with buying pressure
                price_change = np.random.normal(0.01, 0.02)  # Slight upward bias
                new_price = prices[-1] * (1 + price_change)
                prices.append(new_price)
                
                # Volume increases with positive price movement
                volume_change = 1 + abs(price_change) * 5
                new_volume = int(volumes[-1] * volume_change)
                volumes.append(new_volume)
            
            prices = np.array(prices)
            volumes = np.array(volumes)
            
            # Generate OHLC data
            high = prices * 1.01
            low = prices * 0.99
            close = prices
            
            # Test Chaikin Money Flow on this data
            result = calculate_chaikin_money_flow(high, low, close, volumes)
            cmf = result.values[:, 0]
            
            # Should detect buying pressure
            valid_cmf = cmf[~np.isnan(cmf)]
            assert len(valid_cmf) > 0
            
            # Test Chaikin Oscillator on trending data
            trending_prices = np.linspace(100, 150, 500)  # Upward trend
            trending_high = trending_prices * 1.01
            trending_low = trending_prices * 0.99
            trending_volume = np.random.randint(1000000, 10000000, 500)
            
            result = calculate_chaikin_oscillator(trending_high, trending_low, trending_prices, trending_volume)
            oscillator = result.values[:, 0]
            
            # Should detect trend
            valid_oscillator = oscillator[~np.isnan(oscillator)]
            assert len(valid_oscillator) > 0
            
            logger.info("✅ Money flow analysis test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Money flow analysis test failed: {e}")
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
            
            # Generate OHLC data
            high = prices * 1.01
            low = prices * 0.99
            close = prices
            
            # Test Chaikin Money Flow divergence detection
            result = calculate_chaikin_money_flow(high, low, close, volumes)
            
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
        logger.info("Starting Chaikin Indicators Test Suite")
        
        tests = [
            self.test_chaikin_oscillator_basic,
            self.test_chaikin_oscillator_parameters,
            self.test_chaikin_oscillator_optimization,
            self.test_chaikin_money_flow_basic,
            self.test_chaikin_money_flow_parameters,
            self.test_chaikin_money_flow_optimization,
            self.test_error_handling,
            self.test_signal_generation,
            self.test_performance_benchmark,
            self.test_money_flow_analysis,
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
            logger.info("🎉 All tests passed! Chaikin indicators are working correctly.")
        else:
            logger.error("❌ Some tests failed. Please check the implementation.")
        
        return passed == total

async def main():
    """Main test execution function."""
    logger.info("Starting Chaikin Indicators Test Suite")
    
    test_suite = ChaikinIndicatorsTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Chaikin Indicators Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Chaikin Indicators Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())