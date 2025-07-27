#!/usr/bin/env python3
"""
Test script for the Advanced Volume Indicators Module.
Tests BOP and Vortex Indicator calculations with various scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering.indicators.advanced_volume_indicators import (
    BalanceOfPower, VortexIndicator,
    calculate_bop, calculate_vortex_indicator,
    SignalType, IndicatorResult
)
import structlog

logger = structlog.get_logger()

class AdvancedVolumeIndicatorsTestSuite:
    """Comprehensive test suite for advanced volume indicators."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        
    def _generate_test_data(self, length: int = 100, trend: str = 'random') -> tuple:
        """Generate test OHLC data."""
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
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0]  # First open = first close
        
        high = close + np.random.uniform(0, 2, length)
        low = close - np.random.uniform(0, 2, length)
        
        # Ensure high >= close >= low and high >= open >= low
        high = np.maximum(high, np.maximum(close, open_prices))
        low = np.minimum(low, np.minimum(close, open_prices))
        
        return open_prices, high, low, close
    
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
    
    async def test_bop_basic_calculation(self):
        """Test basic BOP calculation."""
        logger.info("Testing BOP basic calculation")
        
        try:
            open_prices, high, low, close = self._generate_test_data(100)
            bop = BalanceOfPower(period=14)
            result = bop.calculate(open_prices, high, low, close)
            
            # Basic validation
            assert result.values.shape[1] == 2, "Should return raw and smoothed BOP"
            assert len(result.signals) == len(close), "Signals should match data length"
            assert len(result.signal_strength) == len(close), "Signal strength should match data length"
            assert result.metadata['indicator_type'] == 'BOP'
            
            # Check for reasonable values
            bop_raw = result.values[:, 0]
            bop_smoothed = result.values[:, 1]
            
            assert not np.all(np.isnan(bop_raw)), "Raw BOP should not be all NaN"
            assert not np.all(np.isnan(bop_smoothed)), "Smoothed BOP should not be all NaN"
            
            # BOP should be between -1 and 1
            assert np.all(bop_raw >= -1) and np.all(bop_raw <= 1), "BOP should be between -1 and 1"
            
            self._log_test_result("BOP Basic Calculation", True)
            
        except Exception as e:
            self._log_test_result("BOP Basic Calculation", False, str(e))
    
    async def test_bop_parameter_validation(self):
        """Test BOP parameter validation."""
        logger.info("Testing BOP parameter validation")
        
        try:
            # Test invalid period
            try:
                bop = BalanceOfPower(period=0)
                self._log_test_result("BOP Invalid Period", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            # Test invalid overbought/oversold
            try:
                bop = BalanceOfPower(overbought=0.5, oversold=0.6)
                self._log_test_result("BOP Invalid Thresholds", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            self._log_test_result("BOP Parameter Validation", True)
            
        except Exception as e:
            self._log_test_result("BOP Parameter Validation", False, str(e))
    
    async def test_bop_insufficient_data(self):
        """Test BOP with insufficient data."""
        logger.info("Testing BOP insufficient data handling")
        
        try:
            open_prices, high, low, close = self._generate_test_data(10)  # Too few data points
            bop = BalanceOfPower(period=14)
            
            try:
                result = bop.calculate(open_prices, high, low, close)
                self._log_test_result("BOP Insufficient Data", False, "Should raise ValueError")
            except ValueError:
                self._log_test_result("BOP Insufficient Data", True)
                
        except Exception as e:
            self._log_test_result("BOP Insufficient Data", False, str(e))
    
    async def test_bop_signal_generation(self):
        """Test BOP signal generation."""
        logger.info("Testing BOP signal generation")
        
        try:
            open_prices, high, low, close = self._generate_test_data(200)
            bop = BalanceOfPower(period=14, overbought=0.7, oversold=-0.7)
            result = bop.calculate(open_prices, high, low, close)
            
            # Check signal types
            valid_signals = [signal.value for signal in SignalType]
            for signal in result.signals:
                assert signal in valid_signals, f"Invalid signal type: {signal}"
            
            # Check signal strength range
            assert np.all(result.signal_strength >= 0), "Signal strength should be non-negative"
            assert np.all(result.signal_strength <= 1), "Signal strength should be <= 1"
            
            self._log_test_result("BOP Signal Generation", True)
            
        except Exception as e:
            self._log_test_result("BOP Signal Generation", False, str(e))
    
    async def test_bop_parameter_optimization(self):
        """Test BOP parameter optimization."""
        logger.info("Testing BOP parameter optimization")
        
        try:
            open_prices, high, low, close = self._generate_test_data(200)
            bop = BalanceOfPower()
            
            # Test optimization with different metrics
            for metric in ['sharpe_ratio', 'max_drawdown', 'total_return']:
                optimal_params = bop.optimize_parameters(open_prices, high, low, close, target_metric=metric)
                
                assert 'period' in optimal_params
                assert 'overbought' in optimal_params
                assert 'oversold' in optimal_params
                
                # Verify parameters are reasonable
                assert optimal_params['period'] > 0
                assert optimal_params['overbought'] > optimal_params['oversold']
            
            self._log_test_result("BOP Parameter Optimization", True)
            
        except Exception as e:
            self._log_test_result("BOP Parameter Optimization", False, str(e))
    
    async def test_vortex_indicator_basic_calculation(self):
        """Test basic Vortex Indicator calculation."""
        logger.info("Testing Vortex Indicator basic calculation")
        
        try:
            open_prices, high, low, close = self._generate_test_data(100)
            vi = VortexIndicator(period=14)
            result = vi.calculate(high, low, close)
            
            # Basic validation
            assert result.values.shape[1] == 2, "Should return +VI and -VI lines"
            assert len(result.signals) == len(close), "Signals should match data length"
            assert len(result.signal_strength) == len(close), "Signal strength should match data length"
            assert result.metadata['indicator_type'] == 'VI'
            
            # Check for reasonable values
            plus_vi = result.values[:, 0]
            minus_vi = result.values[:, 1]
            
            assert not np.all(np.isnan(plus_vi)), "+VI should not be all NaN"
            assert not np.all(np.isnan(minus_vi)), "-VI should not be all NaN"
            
            # VI values should be non-negative
            assert np.all(plus_vi >= 0), "+VI should be non-negative"
            assert np.all(minus_vi >= 0), "-VI should be non-negative"
            
            self._log_test_result("Vortex Indicator Basic Calculation", True)
            
        except Exception as e:
            self._log_test_result("Vortex Indicator Basic Calculation", False, str(e))
    
    async def test_vortex_indicator_parameter_validation(self):
        """Test Vortex Indicator parameter validation."""
        logger.info("Testing Vortex Indicator parameter validation")
        
        try:
            # Test invalid period
            try:
                vi = VortexIndicator(period=0)
                self._log_test_result("VI Invalid Period", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            # Test invalid crossover threshold
            try:
                vi = VortexIndicator(crossover_threshold=0)
                self._log_test_result("VI Invalid Crossover Threshold", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            self._log_test_result("Vortex Indicator Parameter Validation", True)
            
        except Exception as e:
            self._log_test_result("Vortex Indicator Parameter Validation", False, str(e))
    
    async def test_vortex_indicator_insufficient_data(self):
        """Test Vortex Indicator with insufficient data."""
        logger.info("Testing Vortex Indicator insufficient data handling")
        
        try:
            open_prices, high, low, close = self._generate_test_data(10)  # Too few data points
            vi = VortexIndicator(period=14)
            
            try:
                result = vi.calculate(high, low, close)
                self._log_test_result("VI Insufficient Data", False, "Should raise ValueError")
            except ValueError:
                self._log_test_result("VI Insufficient Data", True)
                
        except Exception as e:
            self._log_test_result("VI Insufficient Data", False, str(e))
    
    async def test_vortex_indicator_signal_generation(self):
        """Test Vortex Indicator signal generation."""
        logger.info("Testing Vortex Indicator signal generation")
        
        try:
            open_prices, high, low, close = self._generate_test_data(200)
            vi = VortexIndicator(period=14, crossover_threshold=0.1)
            result = vi.calculate(high, low, close)
            
            # Check signal types
            valid_signals = [signal.value for signal in SignalType]
            for signal in result.signals:
                assert signal in valid_signals, f"Invalid signal type: {signal}"
            
            # Check signal strength range
            assert np.all(result.signal_strength >= 0), "Signal strength should be non-negative"
            assert np.all(result.signal_strength <= 1), "Signal strength should be <= 1"
            
            self._log_test_result("Vortex Indicator Signal Generation", True)
            
        except Exception as e:
            self._log_test_result("Vortex Indicator Signal Generation", False, str(e))
    
    async def test_vortex_indicator_parameter_optimization(self):
        """Test Vortex Indicator parameter optimization."""
        logger.info("Testing Vortex Indicator parameter optimization")
        
        try:
            open_prices, high, low, close = self._generate_test_data(200)
            vi = VortexIndicator()
            
            # Test optimization with different metrics
            for metric in ['sharpe_ratio', 'max_drawdown', 'total_return']:
                optimal_params = vi.optimize_parameters(high, low, close, target_metric=metric)
                
                assert 'period' in optimal_params
                assert 'crossover_threshold' in optimal_params
                
                # Verify parameters are reasonable
                assert optimal_params['period'] > 0
                assert optimal_params['crossover_threshold'] > 0
            
            self._log_test_result("Vortex Indicator Parameter Optimization", True)
            
        except Exception as e:
            self._log_test_result("Vortex Indicator Parameter Optimization", False, str(e))
    
    async def test_convenience_functions(self):
        """Test convenience functions."""
        logger.info("Testing convenience functions")
        
        try:
            open_prices, high, low, close = self._generate_test_data(100)
            
            # Test BOP convenience function
            bop_result = calculate_bop(open_prices, high, low, close)
            assert isinstance(bop_result, IndicatorResult)
            assert bop_result.metadata['indicator_type'] == 'BOP'
            
            # Test Vortex Indicator convenience function
            vi_result = calculate_vortex_indicator(high, low, close)
            assert isinstance(vi_result, IndicatorResult)
            assert vi_result.metadata['indicator_type'] == 'VI'
            
            self._log_test_result("Convenience Functions", True)
            
        except Exception as e:
            self._log_test_result("Convenience Functions", False, str(e))
    
    async def test_different_market_conditions(self):
        """Test indicators with different market conditions."""
        logger.info("Testing different market conditions")
        
        try:
            market_conditions = ['random', 'uptrend', 'downtrend', 'volatile']
            
            for condition in market_conditions:
                open_prices, high, low, close = self._generate_test_data(200, trend=condition)
                
                # Test BOP
                bop = BalanceOfPower()
                bop_result = bop.calculate(open_prices, high, low, close)
                assert len(bop_result.signals) == len(close)
                
                # Test Vortex Indicator
                vi = VortexIndicator()
                vi_result = vi.calculate(high, low, close)
                assert len(vi_result.signals) == len(close)
            
            self._log_test_result("Different Market Conditions", True)
            
        except Exception as e:
            self._log_test_result("Different Market Conditions", False, str(e))
    
    async def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        logger.info("Testing edge cases")
        
        try:
            # Test with constant prices
            constant_prices = np.full(100, 100.0)
            
            bop = BalanceOfPower()
            bop_result = bop.calculate(constant_prices, constant_prices, constant_prices, constant_prices)
            assert len(bop_result.signals) == len(constant_prices)
            
            vi = VortexIndicator()
            vi_result = vi.calculate(constant_prices, constant_prices, constant_prices)
            assert len(vi_result.signals) == len(constant_prices)
            
            # Test with very small price changes
            small_changes = 100 + np.random.normal(0, 0.001, 100)
            
            bop_result = bop.calculate(small_changes, small_changes, small_changes, small_changes)
            assert len(bop_result.signals) == len(small_changes)
            
            vi_result = vi.calculate(small_changes, small_changes, small_changes)
            assert len(vi_result.signals) == len(small_changes)
            
            self._log_test_result("Edge Cases", True)
            
        except Exception as e:
            self._log_test_result("Edge Cases", False, str(e))
    
    async def test_crossover_detection(self):
        """Test crossover detection in Vortex Indicator."""
        logger.info("Testing crossover detection")
        
        try:
            # Create data with clear trend changes
            length = 100
            close = np.concatenate([
                np.linspace(100, 120, length//2),  # Upward trend
                np.linspace(120, 100, length//2)   # Downward trend
            ])
            high = close + 1
            low = close - 1
            
            vi = VortexIndicator(period=10, crossover_threshold=0.05)
            result = vi.calculate(high, low, close)
            
            # Should detect some crossover signals
            signal_counts = np.unique(result.signals, return_counts=True)
            assert len(signal_counts[0]) > 1, "Should generate multiple signal types"
            
            self._log_test_result("Crossover Detection", True)
            
        except Exception as e:
            self._log_test_result("Crossover Detection", False, str(e))
    
    async def test_zero_range_handling(self):
        """Test handling of zero price range scenarios."""
        logger.info("Testing zero range handling")
        
        try:
            open_prices, high, low, close = self._generate_test_data(100)
            
            # Set some ranges to zero
            high[10:20] = close[10:20]
            low[10:20] = close[10:20]
            
            bop = BalanceOfPower()
            bop_result = bop.calculate(open_prices, high, low, close)
            assert len(bop_result.signals) == len(close)
            
            vi = VortexIndicator()
            vi_result = vi.calculate(high, low, close)
            assert len(vi_result.signals) == len(close)
            
            self._log_test_result("Zero Range Handling", True)
            
        except Exception as e:
            self._log_test_result("Zero Range Handling", False, str(e))
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return success status."""
        logger.info("Starting Advanced Volume Indicators Test Suite")
        
        test_methods = [
            self.test_bop_basic_calculation,
            self.test_bop_parameter_validation,
            self.test_bop_insufficient_data,
            self.test_bop_signal_generation,
            self.test_bop_parameter_optimization,
            self.test_vortex_indicator_basic_calculation,
            self.test_vortex_indicator_parameter_validation,
            self.test_vortex_indicator_insufficient_data,
            self.test_vortex_indicator_signal_generation,
            self.test_vortex_indicator_parameter_optimization,
            self.test_convenience_functions,
            self.test_different_market_conditions,
            self.test_edge_cases,
            self.test_crossover_detection,
            self.test_zero_range_handling
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
        
        logger.info("Advanced Volume Indicators Test Suite Summary",
                   total_tests=total_tests,
                   passed_tests=self.passed_tests,
                   failed_tests=self.failed_tests,
                   success_rate=f"{success_rate:.1f}%")
        
        return self.failed_tests == 0

async def main():
    """Main test execution function."""
    logger.info("Starting Advanced Volume Indicators Test Suite")
    
    test_suite = AdvancedVolumeIndicatorsTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Advanced Volume Indicators Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Advanced Volume Indicators Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())