#!/usr/bin/env python3
"""
Test script for the Oscillator Indicators Module.
Tests Williams %R, Keltner Channel, and Donchian Channel calculations with various scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering.indicators.oscillator_indicators import (
    WilliamsPercentR, KeltnerChannel, DonchianChannel,
    calculate_williams_r, calculate_keltner_channel, calculate_donchian_channel,
    SignalType, IndicatorResult
)
import structlog

logger = structlog.get_logger()

class OscillatorIndicatorsTestSuite:
    """Comprehensive test suite for oscillator indicators."""
    
    def __init__(self):
        """Initialize test suite with sample data."""
        self.setup_test_data()
    
    def setup_test_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        n_points = 1000
        
        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
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
    
    async def test_williams_r_basic(self):
        """Test basic Williams %R calculation."""
        logger.info("Testing Williams %R basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_williams_r(self.high, self.low, self.close)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert len(result.values) == len(self.close)
            assert len(result.signals) == len(self.close)
            assert len(result.signal_strength) == len(self.close)
            
            # Check value ranges
            valid_values = result.values[~np.isnan(result.values)]
            assert np.all(valid_values >= -100) and np.all(valid_values <= 0)
            
            # Check signal types
            valid_signals = set(result.signals)
            expected_signals = {SignalType.BUY.value, SignalType.SELL.value, 
                              SignalType.HOLD.value, SignalType.STRONG_BUY.value, 
                              SignalType.STRONG_SELL.value}
            assert valid_signals.issubset(expected_signals)
            
            logger.info("✅ Williams %R basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Williams %R basic test failed: {e}")
            return False
    
    async def test_williams_r_parameters(self):
        """Test Williams %R with different parameters."""
        logger.info("Testing Williams %R parameter variations")
        
        try:
            # Test different periods
            periods = [10, 14, 20]
            for period in periods:
                result = calculate_williams_r(self.high, self.low, self.close, period=period)
                assert result.parameters['period'] == period
            
            # Test different overbought/oversold levels
            result = calculate_williams_r(self.high, self.low, self.close, 
                                        overbought=-10, oversold=-90)
            assert result.parameters['overbought'] == -10
            assert result.parameters['oversold'] == -90
            
            logger.info("✅ Williams %R parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Williams %R parameter test failed: {e}")
            return False
    
    async def test_williams_r_optimization(self):
        """Test Williams %R parameter optimization."""
        logger.info("Testing Williams %R parameter optimization")
        
        try:
            williams_r = WilliamsPercentR()
            
            param_ranges = {
                'period': [10, 14, 20],
                'overbought': [-10, -20, -30],
                'oversold': [-70, -80, -90]
            }
            
            optimal_params = williams_r.optimize_parameters(
                self.high, self.low, self.close, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'period' in optimal_params
            assert 'overbought' in optimal_params
            assert 'oversold' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ Williams %R optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Williams %R optimization test failed: {e}")
            return False
    
    async def test_keltner_channel_basic(self):
        """Test basic Keltner Channel calculation."""
        logger.info("Testing Keltner Channel basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_keltner_channel(self.high, self.low, self.close)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert result.values.shape[1] == 3  # upper, middle, lower bands
            assert len(result.signals) == len(self.close)
            
            # Check band relationships
            upper_band = result.values[:, 0]
            middle_band = result.values[:, 1]
            lower_band = result.values[:, 2]
            
            valid_mask = ~(np.isnan(upper_band) | np.isnan(middle_band) | np.isnan(lower_band))
            assert np.all(upper_band[valid_mask] >= middle_band[valid_mask])
            assert np.all(middle_band[valid_mask] >= lower_band[valid_mask])
            
            logger.info("✅ Keltner Channel basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Keltner Channel basic test failed: {e}")
            return False
    
    async def test_keltner_channel_parameters(self):
        """Test Keltner Channel with different parameters."""
        logger.info("Testing Keltner Channel parameter variations")
        
        try:
            # Test different EMA periods
            ema_periods = [10, 20, 30]
            for ema_period in ema_periods:
                result = calculate_keltner_channel(self.high, self.low, self.close, 
                                                 ema_period=ema_period)
                assert result.parameters['ema_period'] == ema_period
            
            # Test different multipliers
            multipliers = [1.5, 2.0, 2.5]
            for multiplier in multipliers:
                result = calculate_keltner_channel(self.high, self.low, self.close, 
                                                 multiplier=multiplier)
                assert result.parameters['multiplier'] == multiplier
            
            logger.info("✅ Keltner Channel parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Keltner Channel parameter test failed: {e}")
            return False
    
    async def test_keltner_channel_optimization(self):
        """Test Keltner Channel parameter optimization."""
        logger.info("Testing Keltner Channel parameter optimization")
        
        try:
            keltner = KeltnerChannel()
            
            param_ranges = {
                'ema_period': [10, 20, 30],
                'atr_period': [10, 14, 20],
                'multiplier': [1.5, 2.0, 2.5]
            }
            
            optimal_params = keltner.optimize_parameters(
                self.high, self.low, self.close, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'ema_period' in optimal_params
            assert 'atr_period' in optimal_params
            assert 'multiplier' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ Keltner Channel optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Keltner Channel optimization test failed: {e}")
            return False
    
    async def test_donchian_channel_basic(self):
        """Test basic Donchian Channel calculation."""
        logger.info("Testing Donchian Channel basic calculation")
        
        try:
            # Test with default parameters
            result = calculate_donchian_channel(self.high, self.low, self.close)
            
            # Validate results
            assert isinstance(result, IndicatorResult)
            assert result.values.shape[1] == 4  # upper, middle, lower, width
            assert len(result.signals) == len(self.close)
            
            # Check band relationships
            upper_band = result.values[:, 0]
            middle_band = result.values[:, 1]
            lower_band = result.values[:, 2]
            width = result.values[:, 3]
            
            valid_mask = ~(np.isnan(upper_band) | np.isnan(middle_band) | np.isnan(lower_band))
            assert np.all(upper_band[valid_mask] >= middle_band[valid_mask])
            assert np.all(middle_band[valid_mask] >= lower_band[valid_mask])
            assert np.all(width[valid_mask] >= 0)
            
            logger.info("✅ Donchian Channel basic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Donchian Channel basic test failed: {e}")
            return False
    
    async def test_donchian_channel_parameters(self):
        """Test Donchian Channel with different parameters."""
        logger.info("Testing Donchian Channel parameter variations")
        
        try:
            # Test different periods
            periods = [10, 20, 30, 50]
            for period in periods:
                result = calculate_donchian_channel(self.high, self.low, self.close, period=period)
                assert result.parameters['period'] == period
            
            # Test different breakout thresholds
            thresholds = [0.01, 0.02, 0.03, 0.05]
            for threshold in thresholds:
                result = calculate_donchian_channel(self.high, self.low, self.close, 
                                                 breakout_threshold=threshold)
                assert result.parameters['breakout_threshold'] == threshold
            
            logger.info("✅ Donchian Channel parameter test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Donchian Channel parameter test failed: {e}")
            return False
    
    async def test_donchian_channel_optimization(self):
        """Test Donchian Channel parameter optimization."""
        logger.info("Testing Donchian Channel parameter optimization")
        
        try:
            donchian = DonchianChannel()
            
            param_ranges = {
                'period': [10, 20, 30, 50],
                'breakout_threshold': [0.01, 0.02, 0.03, 0.05]
            }
            
            optimal_params = donchian.optimize_parameters(
                self.high, self.low, self.close, self.returns, param_ranges
            )
            
            assert optimal_params is not None
            assert 'period' in optimal_params
            assert 'breakout_threshold' in optimal_params
            assert 'score' in optimal_params
            
            logger.info(f"✅ Donchian Channel optimization test passed. Optimal params: {optimal_params}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Donchian Channel optimization test failed: {e}")
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
                calculate_williams_r(short_high, short_low, short_close, period=10)
            
            with self.assertRaises(ValueError):
                calculate_keltner_channel(short_high, short_low, short_close, ema_period=10)
            
            with self.assertRaises(ValueError):
                calculate_donchian_channel(short_high, short_low, short_close, period=10)
            
            # Test with mismatched array lengths
            with self.assertRaises(ValueError):
                calculate_williams_r(self.high, self.low[:100], self.close)
            
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
            # Test Williams %R signals
            result = calculate_williams_r(self.high, self.low, self.close)
            
            # Check that oversold values generate buy signals
            oversold_indices = np.where(result.values <= -80)[0]
            if len(oversold_indices) > 0:
                assert np.all(np.isin(result.signals[oversold_indices], 
                                    [SignalType.BUY.value, SignalType.STRONG_BUY.value]))
            
            # Check that overbought values generate sell signals
            overbought_indices = np.where(result.values >= -20)[0]
            if len(overbought_indices) > 0:
                assert np.all(np.isin(result.signals[overbought_indices], 
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
            
            # Benchmark Williams %R
            start_time = time.time()
            result = calculate_williams_r(large_high, large_low, large_close)
            williams_time = time.time() - start_time
            
            # Benchmark Keltner Channel
            start_time = time.time()
            result = calculate_keltner_channel(large_high, large_low, large_close)
            keltner_time = time.time() - start_time
            
            # Benchmark Donchian Channel
            start_time = time.time()
            result = calculate_donchian_channel(large_high, large_low, large_close)
            donchian_time = time.time() - start_time
            
            logger.info(f"Performance benchmark (10k points):")
            logger.info(f"  Williams %R: {williams_time:.4f}s")
            logger.info(f"  Keltner Channel: {keltner_time:.4f}s")
            logger.info(f"  Donchian Channel: {donchian_time:.4f}s")
            
            # Ensure reasonable performance (less than 1 second for 10k points)
            assert williams_time < 1.0
            assert keltner_time < 1.0
            assert donchian_time < 1.0
            
            logger.info("✅ Performance benchmark test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("Starting Oscillator Indicators Test Suite")
        
        tests = [
            self.test_williams_r_basic,
            self.test_williams_r_parameters,
            self.test_williams_r_optimization,
            self.test_keltner_channel_basic,
            self.test_keltner_channel_parameters,
            self.test_keltner_channel_optimization,
            self.test_donchian_channel_basic,
            self.test_donchian_channel_parameters,
            self.test_donchian_channel_optimization,
            self.test_error_handling,
            self.test_signal_generation,
            self.test_performance_benchmark
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
            logger.info("🎉 All tests passed! Oscillator indicators are working correctly.")
        else:
            logger.error("❌ Some tests failed. Please check the implementation.")
        
        return passed == total

async def main():
    """Main test execution function."""
    logger.info("Starting Oscillator Indicators Test Suite")
    
    test_suite = OscillatorIndicatorsTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Oscillator Indicators Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Oscillator Indicators Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())