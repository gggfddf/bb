#!/usr/bin/env python3
"""
Test script for the Momentum Oscillators Module.
Tests TRIX and DPO calculations with various scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering.indicators.momentum_oscillators import (
    TRIX, DetrendedPriceOscillator,
    calculate_trix, calculate_dpo,
    SignalType, IndicatorResult
)
import structlog

logger = structlog.get_logger()

class MomentumOscillatorsTestSuite:
    """Comprehensive test suite for momentum oscillators."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        
    def _generate_test_data(self, length: int = 100, trend: str = 'random') -> np.ndarray:
        """Generate test price data."""
        np.random.seed(42)
        
        if trend == 'random':
            # Random walk
            returns = np.random.normal(0, 0.02, length)
            prices = 100 * np.exp(np.cumsum(returns))
        elif trend == 'uptrend':
            # Upward trend
            returns = np.random.normal(0.001, 0.02, length)
            prices = 100 * np.exp(np.cumsum(returns))
        elif trend == 'downtrend':
            # Downward trend
            returns = np.random.normal(-0.001, 0.02, length)
            prices = 100 * np.exp(np.cumsum(returns))
        elif trend == 'volatile':
            # High volatility
            returns = np.random.normal(0, 0.05, length)
            prices = 100 * np.exp(np.cumsum(returns))
        else:
            # Sine wave pattern
            t = np.linspace(0, 4*np.pi, length)
            prices = 100 + 10 * np.sin(t) + np.random.normal(0, 0.5, length)
        
        return prices
    
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
    
    async def test_trix_basic_calculation(self):
        """Test basic TRIX calculation."""
        logger.info("Testing TRIX basic calculation")
        
        try:
            prices = self._generate_test_data(100)
            trix = TRIX(period=15, signal_period=9)
            result = trix.calculate(prices)
            
            # Basic validation
            assert result.values.shape[1] == 2, "Should return TRIX and signal lines"
            assert len(result.signals) == len(prices), "Signals should match price length"
            assert len(result.signal_strength) == len(prices), "Signal strength should match price length"
            assert result.metadata['indicator_type'] == 'TRIX'
            
            # Check for reasonable values
            trix_values = result.values[:, 0]
            signal_values = result.values[:, 1]
            
            assert not np.all(np.isnan(trix_values)), "TRIX values should not be all NaN"
            assert not np.all(np.isnan(signal_values)), "Signal values should not be all NaN"
            
            self._log_test_result("TRIX Basic Calculation", True)
            
        except Exception as e:
            self._log_test_result("TRIX Basic Calculation", False, str(e))
    
    async def test_trix_parameter_validation(self):
        """Test TRIX parameter validation."""
        logger.info("Testing TRIX parameter validation")
        
        try:
            # Test invalid period
            try:
                trix = TRIX(period=0)
                self._log_test_result("TRIX Invalid Period", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            # Test invalid signal period
            try:
                trix = TRIX(signal_period=0)
                self._log_test_result("TRIX Invalid Signal Period", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            # Test invalid overbought/oversold
            try:
                trix = TRIX(overbought=0.5, oversold=0.6)
                self._log_test_result("TRIX Invalid Thresholds", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            self._log_test_result("TRIX Parameter Validation", True)
            
        except Exception as e:
            self._log_test_result("TRIX Parameter Validation", False, str(e))
    
    async def test_trix_insufficient_data(self):
        """Test TRIX with insufficient data."""
        logger.info("Testing TRIX insufficient data handling")
        
        try:
            prices = self._generate_test_data(10)  # Too few data points
            trix = TRIX(period=15, signal_period=9)
            
            try:
                result = trix.calculate(prices)
                self._log_test_result("TRIX Insufficient Data", False, "Should raise ValueError")
            except ValueError:
                self._log_test_result("TRIX Insufficient Data", True)
                
        except Exception as e:
            self._log_test_result("TRIX Insufficient Data", False, str(e))
    
    async def test_trix_signal_generation(self):
        """Test TRIX signal generation."""
        logger.info("Testing TRIX signal generation")
        
        try:
            prices = self._generate_test_data(200)
            trix = TRIX(period=15, signal_period=9, overbought=0.5, oversold=-0.5)
            result = trix.calculate(prices)
            
            # Check signal types
            valid_signals = [signal.value for signal in SignalType]
            for signal in result.signals:
                assert signal in valid_signals, f"Invalid signal type: {signal}"
            
            # Check signal strength range
            assert np.all(result.signal_strength >= 0), "Signal strength should be non-negative"
            assert np.all(result.signal_strength <= 1), "Signal strength should be <= 1"
            
            self._log_test_result("TRIX Signal Generation", True)
            
        except Exception as e:
            self._log_test_result("TRIX Signal Generation", False, str(e))
    
    async def test_trix_parameter_optimization(self):
        """Test TRIX parameter optimization."""
        logger.info("Testing TRIX parameter optimization")
        
        try:
            prices = self._generate_test_data(200)
            trix = TRIX()
            
            # Test optimization with different metrics
            for metric in ['sharpe_ratio', 'max_drawdown', 'total_return']:
                optimal_params = trix.optimize_parameters(prices, target_metric=metric)
                
                assert 'period' in optimal_params
                assert 'signal_period' in optimal_params
                assert 'overbought' in optimal_params
                assert 'oversold' in optimal_params
                
                # Verify parameters are reasonable
                assert optimal_params['period'] > 0
                assert optimal_params['signal_period'] > 0
                assert optimal_params['overbought'] > optimal_params['oversold']
            
            self._log_test_result("TRIX Parameter Optimization", True)
            
        except Exception as e:
            self._log_test_result("TRIX Parameter Optimization", False, str(e))
    
    async def test_dpo_basic_calculation(self):
        """Test basic DPO calculation."""
        logger.info("Testing DPO basic calculation")
        
        try:
            prices = self._generate_test_data(100)
            dpo = DetrendedPriceOscillator(period=20, signal_period=10)
            result = dpo.calculate(prices)
            
            # Basic validation
            assert result.values.shape[1] == 2, "Should return DPO and signal lines"
            assert len(result.signals) == len(prices), "Signals should match price length"
            assert len(result.signal_strength) == len(prices), "Signal strength should match price length"
            assert result.metadata['indicator_type'] == 'DPO'
            
            # Check for reasonable values
            dpo_values = result.values[:, 0]
            signal_values = result.values[:, 1]
            
            # DPO should have some NaN values at the beginning
            assert np.any(np.isnan(dpo_values)), "DPO should have NaN values at start"
            assert not np.all(np.isnan(dpo_values)), "DPO should not be all NaN"
            
            self._log_test_result("DPO Basic Calculation", True)
            
        except Exception as e:
            self._log_test_result("DPO Basic Calculation", False, str(e))
    
    async def test_dpo_parameter_validation(self):
        """Test DPO parameter validation."""
        logger.info("Testing DPO parameter validation")
        
        try:
            # Test invalid period
            try:
                dpo = DetrendedPriceOscillator(period=1)
                self._log_test_result("DPO Invalid Period", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            # Test invalid signal period
            try:
                dpo = DetrendedPriceOscillator(signal_period=0)
                self._log_test_result("DPO Invalid Signal Period", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            # Test invalid thresholds
            try:
                dpo = DetrendedPriceOscillator(overbought_threshold=1.0, oversold_threshold=2.0)
                self._log_test_result("DPO Invalid Thresholds", False, "Should raise ValueError")
                return
            except ValueError:
                pass
            
            self._log_test_result("DPO Parameter Validation", True)
            
        except Exception as e:
            self._log_test_result("DPO Parameter Validation", False, str(e))
    
    async def test_dpo_insufficient_data(self):
        """Test DPO with insufficient data."""
        logger.info("Testing DPO insufficient data handling")
        
        try:
            prices = self._generate_test_data(15)  # Too few data points
            dpo = DetrendedPriceOscillator(period=20, signal_period=10)
            
            try:
                result = dpo.calculate(prices)
                self._log_test_result("DPO Insufficient Data", False, "Should raise ValueError")
            except ValueError:
                self._log_test_result("DPO Insufficient Data", True)
                
        except Exception as e:
            self._log_test_result("DPO Insufficient Data", False, str(e))
    
    async def test_dpo_signal_generation(self):
        """Test DPO signal generation."""
        logger.info("Testing DPO signal generation")
        
        try:
            prices = self._generate_test_data(200)
            dpo = DetrendedPriceOscillator(period=20, signal_period=10)
            result = dpo.calculate(prices)
            
            # Check signal types
            valid_signals = [signal.value for signal in SignalType]
            for signal in result.signals:
                assert signal in valid_signals, f"Invalid signal type: {signal}"
            
            # Check signal strength range
            assert np.all(result.signal_strength >= 0), "Signal strength should be non-negative"
            assert np.all(result.signal_strength <= 1), "Signal strength should be <= 1"
            
            self._log_test_result("DPO Signal Generation", True)
            
        except Exception as e:
            self._log_test_result("DPO Signal Generation", False, str(e))
    
    async def test_dpo_parameter_optimization(self):
        """Test DPO parameter optimization."""
        logger.info("Testing DPO parameter optimization")
        
        try:
            prices = self._generate_test_data(200)
            dpo = DetrendedPriceOscillator()
            
            # Test optimization with different metrics
            for metric in ['sharpe_ratio', 'max_drawdown', 'total_return']:
                optimal_params = dpo.optimize_parameters(prices, target_metric=metric)
                
                assert 'period' in optimal_params
                assert 'signal_period' in optimal_params
                assert 'overbought_threshold' in optimal_params
                assert 'oversold_threshold' in optimal_params
                
                # Verify parameters are reasonable
                assert optimal_params['period'] > 1
                assert optimal_params['signal_period'] > 0
                assert optimal_params['overbought_threshold'] > optimal_params['oversold_threshold']
            
            self._log_test_result("DPO Parameter Optimization", True)
            
        except Exception as e:
            self._log_test_result("DPO Parameter Optimization", False, str(e))
    
    async def test_convenience_functions(self):
        """Test convenience functions."""
        logger.info("Testing convenience functions")
        
        try:
            prices = self._generate_test_data(100)
            
            # Test TRIX convenience function
            trix_result = calculate_trix(prices)
            assert isinstance(trix_result, IndicatorResult)
            assert trix_result.metadata['indicator_type'] == 'TRIX'
            
            # Test DPO convenience function
            dpo_result = calculate_dpo(prices)
            assert isinstance(dpo_result, IndicatorResult)
            assert dpo_result.metadata['indicator_type'] == 'DPO'
            
            self._log_test_result("Convenience Functions", True)
            
        except Exception as e:
            self._log_test_result("Convenience Functions", False, str(e))
    
    async def test_different_market_conditions(self):
        """Test indicators with different market conditions."""
        logger.info("Testing different market conditions")
        
        try:
            market_conditions = ['random', 'uptrend', 'downtrend', 'volatile']
            
            for condition in market_conditions:
                prices = self._generate_test_data(200, trend=condition)
                
                # Test TRIX
                trix = TRIX()
                trix_result = trix.calculate(prices)
                assert len(trix_result.signals) == len(prices)
                
                # Test DPO
                dpo = DetrendedPriceOscillator()
                dpo_result = dpo.calculate(prices)
                assert len(dpo_result.signals) == len(prices)
            
            self._log_test_result("Different Market Conditions", True)
            
        except Exception as e:
            self._log_test_result("Different Market Conditions", False, str(e))
    
    async def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        logger.info("Testing edge cases")
        
        try:
            # Test with constant prices
            constant_prices = np.full(100, 100.0)
            
            trix = TRIX()
            trix_result = trix.calculate(constant_prices)
            assert len(trix_result.signals) == len(constant_prices)
            
            dpo = DetrendedPriceOscillator()
            dpo_result = dpo.calculate(constant_prices)
            assert len(dpo_result.signals) == len(constant_prices)
            
            # Test with very small price changes
            small_changes = 100 + np.random.normal(0, 0.001, 100)
            
            trix_result = trix.calculate(small_changes)
            assert len(trix_result.signals) == len(small_changes)
            
            dpo_result = dpo.calculate(small_changes)
            assert len(dpo_result.signals) == len(small_changes)
            
            self._log_test_result("Edge Cases", True)
            
        except Exception as e:
            self._log_test_result("Edge Cases", False, str(e))
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return success status."""
        logger.info("Starting Momentum Oscillators Test Suite")
        
        test_methods = [
            self.test_trix_basic_calculation,
            self.test_trix_parameter_validation,
            self.test_trix_insufficient_data,
            self.test_trix_signal_generation,
            self.test_trix_parameter_optimization,
            self.test_dpo_basic_calculation,
            self.test_dpo_parameter_validation,
            self.test_dpo_insufficient_data,
            self.test_dpo_signal_generation,
            self.test_dpo_parameter_optimization,
            self.test_convenience_functions,
            self.test_different_market_conditions,
            self.test_edge_cases
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
        
        logger.info("Momentum Oscillators Test Suite Summary",
                   total_tests=total_tests,
                   passed_tests=self.passed_tests,
                   failed_tests=self.failed_tests,
                   success_rate=f"{success_rate:.1f}%")
        
        return self.failed_tests == 0

async def main():
    """Main test execution function."""
    logger.info("Starting Momentum Oscillators Test Suite")
    
    test_suite = MomentumOscillatorsTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Momentum Oscillators Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Momentum Oscillators Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())