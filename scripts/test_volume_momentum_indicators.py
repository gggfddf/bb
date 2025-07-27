#!/usr/bin/env python3
"""
Test script for the Volume Momentum Indicators Module.
Tests EMV and A/D Line calculations with various scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering.indicators.volume_momentum_indicators import (
    EaseOfMovement, AccumulationDistribution,
    calculate_emv, calculate_ad_line,
    SignalType, IndicatorResult
)
import structlog

logger = structlog.get_logger()

class VolumeMomentumIndicatorsTestSuite:
    """Comprehensive test suite for volume momentum indicators."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        
    def _generate_test_data(self, length: int = 100, trend: str = 'random') -> tuple:
        """Generate test OHLCV data."""
        np.random.seed(42)
        
        if trend == 'random':
            # Random walk
            returns = np.random.normal(0, 0.02, length)
            close = 100 * np.exp(np.cumsum(returns))
        elif trend == 'uptrend':
            # Upward trend
            returns = np.random.normal(0.001, 0.02, length)
            close = 100 * np.exp(np.cumsum(returns))
        elif trend == 'downtrend':
            # Downward trend
            returns = np.random.normal(-0.001, 0.02, length)
            close = 100 * np.exp(np.cumsum(returns))
        elif trend == 'volatile':
            # High volatility
            returns = np.random.normal(0, 0.05, length)
            close = 100 * np.exp(np.cumsum(returns))
        else:
            # Sine wave pattern
            t = np.linspace(0, 4*np.pi, length)
            close = 100 + 10 * np.sin(t) + np.random.normal(0, 0.5, length)
        
        # Generate OHLC data
        high = close + np.random.uniform(0, 2, length)
        low = close - np.random.uniform(0, 2, length)
        
        # Ensure high >= close >= low
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        
        # Generate volume data
        volume = np.random.uniform(1000000, 10000000, length)
        
        return high, low, close, volume
    
    def _log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        if success:
            logger.info(f"✅ {test_name} - PASSED", message=message)
            self.passed_tests += 1
        else:
            logger.error(f"❌ {test_name} - FAILED", message=message)
            self.failed_tests += 1
        
        self.test_results.append({
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now()
        })
    
    async def test_emv_basic_calculation(self):
        """Test basic EMV calculation."""
        logger.info("Testing EMV basic calculation")
        
        try:
            high, low, close, volume = self._generate_test_data(100)
            emv = EaseOfMovement(period=14, volume_factor=1000000)
            result = emv.calculate(high, low, close, volume)
            
            # Basic validation
            assert result.values.shape[1] == 2, "Should return raw and smoothed EMV"
            assert len(result.signals) == len(high), "Signals should match data length"
            assert len(result.signal_strength) == len(high), "Signal strength should match data length"
            assert result.metadata['indicator_type'] == 'EMV'
            
            # Check for reasonable values
            emv_raw = result.values[:, 0]
            emv_smoothed = result.values[:, 1]
            
            assert not np.all(np.isnan(emv_raw)), "Raw EMV should not be all NaN"
            assert not np.all(np.isnan(emv_smoothed)), "Smoothed EMV should not be all NaN"
            
            self._log_test_result("EMV Basic Calculation", True)
            
        except Exception as e:
            self._log_test_result("EMV Basic Calculation", False, str(e))
    
    async def test_emv_parameter_validation(self):
        """Test EMV parameter validation."""
        logger.info("Testing EMV parameter validation")
        
        try:
            # Test invalid period
            try:
                emv = EaseOfMovement(period=0)
                self._log_test_result("EMV Invalid Period", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            # Test invalid volume factor
            try:
                emv = EaseOfMovement(volume_factor=0)
                self._log_test_result("EMV Invalid Volume Factor", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            # Test invalid overbought/oversold
            try:
                emv = EaseOfMovement(overbought=0.3, oversold=0.4)
                self._log_test_result("EMV Invalid Thresholds", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            self._log_test_result("EMV Parameter Validation", True)
            
        except Exception as e:
            self._log_test_result("EMV Parameter Validation", False, str(e))
    
    async def test_emv_insufficient_data(self):
        """Test EMV with insufficient data."""
        logger.info("Testing EMV insufficient data handling")
        
        try:
            high, low, close, volume = self._generate_test_data(10)  # Too few data points
            emv = EaseOfMovement(period=14)
            
            try:
                result = emv.calculate(high, low, close, volume)
                self._log_test_result("EMV Insufficient Data", False, "Should raise ValueError")
            except ValueError:
                self._log_test_result("EMV Insufficient Data", True)
                
        except Exception as e:
            self._log_test_result("EMV Insufficient Data", False, str(e))
    
    async def test_emv_signal_generation(self):
        """Test EMV signal generation."""
        logger.info("Testing EMV signal generation")
        
        try:
            high, low, close, volume = self._generate_test_data(200)
            emv = EaseOfMovement(period=14, overbought=0.3, oversold=-0.3)
            result = emv.calculate(high, low, close, volume)
            
            # Check signal types
            valid_signals = [signal.value for signal in SignalType]
            for signal in result.signals:
                assert signal in valid_signals, f"Invalid signal type: {signal}"
            
            # Check signal strength range
            assert np.all(result.signal_strength >= 0), "Signal strength should be non-negative"
            assert np.all(result.signal_strength <= 1), "Signal strength should be <= 1"
            
            self._log_test_result("EMV Signal Generation", True)
            
        except Exception as e:
            self._log_test_result("EMV Signal Generation", False, str(e))
    
    async def test_emv_parameter_optimization(self):
        """Test EMV parameter optimization."""
        logger.info("Testing EMV parameter optimization")
        
        try:
            high, low, close, volume = self._generate_test_data(200)
            emv = EaseOfMovement()
            
            # Test optimization with different metrics
            for metric in ['sharpe_ratio', 'max_drawdown', 'total_return']:
                optimal_params = emv.optimize_parameters(high, low, close, volume, target_metric=metric)
                
                assert 'period' in optimal_params
                assert 'volume_factor' in optimal_params
                assert 'overbought' in optimal_params
                assert 'oversold' in optimal_params
                
                # Verify parameters are reasonable
                assert optimal_params['period'] > 0
                assert optimal_params['volume_factor'] > 0
                assert optimal_params['overbought'] > optimal_params['oversold']
            
            self._log_test_result("EMV Parameter Optimization", True)
            
        except Exception as e:
            self._log_test_result("EMV Parameter Optimization", False, str(e))
    
    async def test_ad_line_basic_calculation(self):
        """Test basic A/D Line calculation."""
        logger.info("Testing A/D Line basic calculation")
        
        try:
            high, low, close, volume = self._generate_test_data(100)
            ad = AccumulationDistribution(smoothing_period=14)
            result = ad.calculate(high, low, close, volume)
            
            # Basic validation
            assert result.values.shape[1] == 3, "Should return A/D line, smoothed, and ROC"
            assert len(result.signals) == len(high), "Signals should match data length"
            assert len(result.signal_strength) == len(high), "Signal strength should match data length"
            assert result.metadata['indicator_type'] == 'A/D'
            
            # Check for reasonable values
            ad_line = result.values[:, 0]
            ad_smoothed = result.values[:, 1]
            ad_roc = result.values[:, 2]
            
            assert not np.all(np.isnan(ad_line)), "A/D line should not be all NaN"
            assert not np.all(np.isnan(ad_smoothed)), "Smoothed A/D should not be all NaN"
            
            self._log_test_result("A/D Line Basic Calculation", True)
            
        except Exception as e:
            self._log_test_result("A/D Line Basic Calculation", False, str(e))
    
    async def test_ad_line_parameter_validation(self):
        """Test A/D Line parameter validation."""
        logger.info("Testing A/D Line parameter validation")
        
        try:
            # Test invalid smoothing period
            try:
                ad = AccumulationDistribution(smoothing_period=0)
                self._log_test_result("A/D Invalid Smoothing Period", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            # Test invalid divergence threshold
            try:
                ad = AccumulationDistribution(divergence_threshold=0)
                self._log_test_result("A/D Invalid Divergence Threshold", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            self._log_test_result("A/D Line Parameter Validation", True)
            
        except Exception as e:
            self._log_test_result("A/D Line Parameter Validation", False, str(e))
    
    async def test_ad_line_insufficient_data(self):
        """Test A/D Line with insufficient data."""
        logger.info("Testing A/D Line insufficient data handling")
        
        try:
            high, low, close, volume = self._generate_test_data(1)  # Too few data points
            ad = AccumulationDistribution()
            
            try:
                result = ad.calculate(high, low, close, volume)
                self._log_test_result("A/D Line Insufficient Data", False, "Should raise ValueError")
            except ValueError:
                self._log_test_result("A/D Line Insufficient Data", True)
                
        except Exception as e:
            self._log_test_result("A/D Line Insufficient Data", False, str(e))
    
    async def test_ad_line_signal_generation(self):
        """Test A/D Line signal generation."""
        logger.info("Testing A/D Line signal generation")
        
        try:
            high, low, close, volume = self._generate_test_data(200)
            ad = AccumulationDistribution(divergence_threshold=0.1)
            result = ad.calculate(high, low, close, volume)
            
            # Check signal types
            valid_signals = [signal.value for signal in SignalType]
            for signal in result.signals:
                assert signal in valid_signals, f"Invalid signal type: {signal}"
            
            # Check signal strength range
            assert np.all(result.signal_strength >= 0), "Signal strength should be non-negative"
            assert np.all(result.signal_strength <= 1), "Signal strength should be <= 1"
            
            self._log_test_result("A/D Line Signal Generation", True)
            
        except Exception as e:
            self._log_test_result("A/D Line Signal Generation", False, str(e))
    
    async def test_ad_line_parameter_optimization(self):
        """Test A/D Line parameter optimization."""
        logger.info("Testing A/D Line parameter optimization")
        
        try:
            high, low, close, volume = self._generate_test_data(200)
            ad = AccumulationDistribution()
            
            # Test optimization with different metrics
            for metric in ['sharpe_ratio', 'max_drawdown', 'total_return']:
                optimal_params = ad.optimize_parameters(high, low, close, volume, target_metric=metric)
                
                assert 'smoothing_period' in optimal_params
                assert 'divergence_threshold' in optimal_params
                
                # Verify parameters are reasonable
                assert optimal_params['smoothing_period'] > 0
                assert optimal_params['divergence_threshold'] > 0
            
            self._log_test_result("A/D Line Parameter Optimization", True)
            
        except Exception as e:
            self._log_test_result("A/D Line Parameter Optimization", False, str(e))
    
    async def test_convenience_functions(self):
        """Test convenience functions."""
        logger.info("Testing convenience functions")
        
        try:
            high, low, close, volume = self._generate_test_data(100)
            
            # Test EMV convenience function
            emv_result = calculate_emv(high, low, close, volume)
            assert isinstance(emv_result, IndicatorResult)
            assert emv_result.metadata['indicator_type'] == 'EMV'
            
            # Test A/D Line convenience function
            ad_result = calculate_ad_line(high, low, close, volume)
            assert isinstance(ad_result, IndicatorResult)
            assert ad_result.metadata['indicator_type'] == 'A/D'
            
            self._log_test_result("Convenience Functions", True)
            
        except Exception as e:
            self._log_test_result("Convenience Functions", False, str(e))
    
    async def test_different_market_conditions(self):
        """Test indicators with different market conditions."""
        logger.info("Testing different market conditions")
        
        try:
            market_conditions = ['random', 'uptrend', 'downtrend', 'volatile']
            
            for condition in market_conditions:
                high, low, close, volume = self._generate_test_data(200, trend=condition)
                
                # Test EMV
                emv = EaseOfMovement()
                emv_result = emv.calculate(high, low, close, volume)
                assert len(emv_result.signals) == len(high)
                
                # Test A/D Line
                ad = AccumulationDistribution()
                ad_result = ad.calculate(high, low, close, volume)
                assert len(ad_result.signals) == len(high)
            
            self._log_test_result("Different Market Conditions", True)
            
        except Exception as e:
            self._log_test_result("Different Market Conditions", False, str(e))
    
    async def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        logger.info("Testing edge cases")
        
        try:
            # Test with constant prices
            constant_prices = np.full(100, 100.0)
            constant_volume = np.full(100, 1000000)
            
            emv = EaseOfMovement()
            emv_result = emv.calculate(constant_prices, constant_prices, constant_prices, constant_volume)
            assert len(emv_result.signals) == len(constant_prices)
            
            ad = AccumulationDistribution()
            ad_result = ad.calculate(constant_prices, constant_prices, constant_prices, constant_volume)
            assert len(ad_result.signals) == len(constant_prices)
            
            # Test with very small price changes
            small_changes = 100 + np.random.normal(0, 0.001, 100)
            small_volume = np.random.uniform(100000, 1000000, 100)
            
            emv_result = emv.calculate(small_changes, small_changes, small_changes, small_volume)
            assert len(emv_result.signals) == len(small_changes)
            
            ad_result = ad.calculate(small_changes, small_changes, small_changes, small_volume)
            assert len(ad_result.signals) == len(small_changes)
            
            self._log_test_result("Edge Cases", True)
            
        except Exception as e:
            self._log_test_result("Edge Cases", False, str(e))
    
    async def test_zero_volume_handling(self):
        """Test handling of zero volume scenarios."""
        logger.info("Testing zero volume handling")
        
        try:
            high, low, close, volume = self._generate_test_data(100)
            
            # Set some volumes to zero
            volume[10:20] = 0
            
            emv = EaseOfMovement()
            emv_result = emv.calculate(high, low, close, volume)
            assert len(emv_result.signals) == len(high)
            
            ad = AccumulationDistribution()
            ad_result = ad.calculate(high, low, close, volume)
            assert len(ad_result.signals) == len(high)
            
            self._log_test_result("Zero Volume Handling", True)
            
        except Exception as e:
            self._log_test_result("Zero Volume Handling", False, str(e))
    
    async def test_divergence_detection(self):
        """Test divergence detection in A/D Line."""
        logger.info("Testing divergence detection")
        
        try:
            # Create data with price-volume divergence
            length = 100
            close = np.linspace(100, 120, length)  # Upward price trend
            high = close + 1
            low = close - 1
            volume = np.linspace(1000000, 500000, length)  # Decreasing volume (bearish divergence)
            
            ad = AccumulationDistribution(divergence_threshold=0.05)
            result = ad.calculate(high, low, close, volume)
            
            # Should detect some divergence signals
            signal_counts = np.unique(result.signals, return_counts=True)
            assert len(signal_counts[0]) > 1, "Should generate multiple signal types"
            
            self._log_test_result("Divergence Detection", True)
            
        except Exception as e:
            self._log_test_result("Divergence Detection", False, str(e))
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return success status."""
        logger.info("Starting Volume Momentum Indicators Test Suite")
        
        test_methods = [
            self.test_emv_basic_calculation,
            self.test_emv_parameter_validation,
            self.test_emv_insufficient_data,
            self.test_emv_signal_generation,
            self.test_emv_parameter_optimization,
            self.test_ad_line_basic_calculation,
            self.test_ad_line_parameter_validation,
            self.test_ad_line_insufficient_data,
            self.test_ad_line_signal_generation,
            self.test_ad_line_parameter_optimization,
            self.test_convenience_functions,
            self.test_different_market_conditions,
            self.test_edge_cases,
            self.test_zero_volume_handling,
            self.test_divergence_detection
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Test method {test_method.__name__} failed with exception", error=str(e))
                self.failed_tests += 1
        
        # Print summary
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("Volume Momentum Indicators Test Suite Summary",
                   total_tests=total_tests,
                   passed_tests=self.passed_tests,
                   failed_tests=self.failed_tests,
                   success_rate=f"{success_rate:.1f}%")
        
        return self.failed_tests == 0

async def main():
    """Main test execution function."""
    logger.info("Starting Volume Momentum Indicators Test Suite")
    
    test_suite = VolumeMomentumIndicatorsTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Volume Momentum Indicators Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Volume Momentum Indicators Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())