#!/usr/bin/env python3
"""
Test script for the Momentum Indicators Module.
Tests RSI, MACD, and Stochastic Oscillator calculations with various scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering.indicators.momentum_indicators import (
    RSI, MACD, StochasticOscillator,
    calculate_rsi, calculate_macd, calculate_stochastic,
    SignalType, IndicatorResult
)
import structlog

logger = structlog.get_logger()

class MomentumIndicatorsTestSuite:
    def __init__(self):
        self.test_results = []
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self) -> Dict[str, np.ndarray]:
        """Generate sample price data for testing."""
        np.random.seed(42)  # For reproducible results
        
        # Generate 100 days of sample data
        n_days = 100
        base_price = 100.0
        
        # Generate price movements
        returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
        prices = [base_price]
        
        for i in range(1, n_days):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Generate OHLC data
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        close = prices
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0]
        
        return {
            "prices": close,
            "high": high,
            "low": low,
            "open": open_prices,
            "returns": returns
        }
    
    async def run_all_tests(self):
        """Run all momentum indicator tests."""
        logger.info("Starting Momentum Indicators Test Suite")
        
        tests = [
            ("RSI Basic Functionality", self.test_rsi_basic),
            ("RSI Parameter Validation", self.test_rsi_parameter_validation),
            ("RSI Signal Generation", self.test_rsi_signal_generation),
            ("RSI Parameter Optimization", self.test_rsi_parameter_optimization),
            ("MACD Basic Functionality", self.test_macd_basic),
            ("MACD Parameter Validation", self.test_macd_parameter_validation),
            ("MACD Signal Generation", self.test_macd_signal_generation),
            ("MACD Parameter Optimization", self.test_macd_parameter_optimization),
            ("Stochastic Basic Functionality", self.test_stochastic_basic),
            ("Stochastic Parameter Validation", self.test_stochastic_parameter_validation),
            ("Stochastic Signal Generation", self.test_stochastic_signal_generation),
            ("Stochastic Parameter Optimization", self.test_stochastic_parameter_optimization),
            ("Convenience Functions", self.test_convenience_functions),
            ("Error Handling", self.test_error_handling),
            ("Integration Test", self.test_integration)
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running test: {test_name}")
                result = await test_func()
                self.test_results.append({
                    "test": test_name,
                    "status": "PASSED" if result else "FAILED",
                    "timestamp": datetime.now()
                })
                logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.test_results.append({
                    "test": test_name,
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now()
                })
        
        self.print_test_summary()
    
    async def test_rsi_basic(self) -> bool:
        """Test basic RSI functionality."""
        try:
            prices = self.sample_data["prices"]
            
            # Test default RSI
            rsi = RSI()
            result = rsi.calculate(prices)
            
            # Validate result structure
            assert isinstance(result, IndicatorResult)
            assert result.values is not None
            assert result.signals is not None
            assert result.signal_strength is not None
            assert result.parameters is not None
            assert result.metadata is not None
            
            # Validate RSI values
            assert len(result.values) == len(prices)
            assert np.all((result.values >= 0) & (result.values <= 100) | np.isnan(result.values))
            
            # Validate signals
            assert len(result.signals) == len(prices)
            valid_signals = [SignalType.BUY.value, SignalType.SELL.value, SignalType.HOLD.value,
                           SignalType.STRONG_BUY.value, SignalType.STRONG_SELL.value]
            assert all(signal in valid_signals for signal in result.signals if signal is not None)
            
            # Validate signal strength
            assert len(result.signal_strength) == len(prices)
            assert np.all((result.signal_strength >= 0) & (result.signal_strength <= 1) | np.isnan(result.signal_strength))
            
            # Test custom parameters
            custom_rsi = RSI(period=21, overbought=75, oversold=25)
            custom_result = custom_rsi.calculate(prices)
            assert custom_result.parameters["period"] == 21
            assert custom_result.parameters["overbought"] == 75
            assert custom_result.parameters["oversold"] == 25
            
            logger.info("RSI basic functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"RSI basic functionality test failed: {e}")
            return False
    
    async def test_rsi_parameter_validation(self) -> bool:
        """Test RSI parameter validation."""
        try:
            # Test invalid period
            try:
                invalid_rsi = RSI(period=0)
                assert False, "Should have raised ValueError for period=0"
            except ValueError:
                pass
            
            # Test invalid thresholds
            try:
                invalid_rsi = RSI(overbought=30, oversold=70)
                assert False, "Should have raised ValueError for overbought < oversold"
            except ValueError:
                pass
            
            # Test out of range thresholds
            try:
                invalid_rsi = RSI(overbought=110, oversold=20)
                assert False, "Should have raised ValueError for overbought > 100"
            except ValueError:
                pass
            
            logger.info("RSI parameter validation test passed")
            return True
            
        except Exception as e:
            logger.error(f"RSI parameter validation test failed: {e}")
            return False
    
    async def test_rsi_signal_generation(self) -> bool:
        """Test RSI signal generation."""
        try:
            prices = self.sample_data["prices"]
            rsi = RSI(period=14, overbought=70, oversold=30)
            result = rsi.calculate(prices)
            
            # Check for signal generation
            buy_signals = sum(1 for signal in result.signals if signal in [SignalType.BUY.value, SignalType.STRONG_BUY.value])
            sell_signals = sum(1 for signal in result.signals if signal in [SignalType.SELL.value, SignalType.STRONG_SELL.value])
            
            # Should have some signals (not all HOLD)
            total_signals = buy_signals + sell_signals
            assert total_signals > 0, "No trading signals generated"
            
            # Check signal strength correlation with signal type
            for i, signal in enumerate(result.signals):
                if signal in [SignalType.STRONG_BUY.value, SignalType.STRONG_SELL.value]:
                    assert result.signal_strength[i] >= 0.8, "Strong signals should have high strength"
                elif signal in [SignalType.BUY.value, SignalType.SELL.value]:
                    assert result.signal_strength[i] >= 0.5, "Regular signals should have moderate strength"
            
            logger.info("RSI signal generation test passed")
            return True
            
        except Exception as e:
            logger.error(f"RSI signal generation test failed: {e}")
            return False
    
    async def test_rsi_parameter_optimization(self) -> bool:
        """Test RSI parameter optimization."""
        try:
            prices = self.sample_data["prices"]
            rsi = RSI()
            
            # Test optimization
            optimal_params = rsi.optimize_parameters(prices, target_metric="sharpe_ratio")
            
            # Validate optimization result
            assert "period" in optimal_params
            assert "overbought" in optimal_params
            assert "oversold" in optimal_params
            assert "score" in optimal_params
            assert optimal_params["score"] is not None
            
            # Test with optimized parameters
            optimized_rsi = RSI(
                period=optimal_params["period"],
                overbought=optimal_params["overbought"],
                oversold=optimal_params["oversold"]
            )
            optimized_result = optimized_rsi.calculate(prices)
            
            assert optimized_result.parameters["period"] == optimal_params["period"]
            assert optimized_result.parameters["overbought"] == optimal_params["overbought"]
            assert optimized_result.parameters["oversold"] == optimal_params["oversold"]
            
            logger.info("RSI parameter optimization test passed")
            return True
            
        except Exception as e:
            logger.error(f"RSI parameter optimization test failed: {e}")
            return False
    
    async def test_macd_basic(self) -> bool:
        """Test basic MACD functionality."""
        try:
            prices = self.sample_data["prices"]
            
            # Test default MACD
            macd = MACD()
            result = macd.calculate(prices)
            
            # Validate result structure
            assert isinstance(result, IndicatorResult)
            assert result.values is not None
            assert result.signals is not None
            assert result.signal_strength is not None
            assert result.parameters is not None
            assert result.metadata is not None
            
            # Validate MACD values (should be 3 columns: MACD line, signal line, histogram)
            assert result.values.shape[1] == 3
            assert len(result.values) == len(prices)
            
            # Validate signals
            assert len(result.signals) == len(prices)
            valid_signals = [SignalType.BUY.value, SignalType.SELL.value, SignalType.HOLD.value,
                           SignalType.STRONG_BUY.value, SignalType.STRONG_SELL.value]
            assert all(signal in valid_signals for signal in result.signals if signal is not None)
            
            # Test custom parameters
            custom_macd = MACD(fast_period=8, slow_period=21, signal_period=5)
            custom_result = custom_macd.calculate(prices)
            assert custom_result.parameters["fast_period"] == 8
            assert custom_result.parameters["slow_period"] == 21
            assert custom_result.parameters["signal_period"] == 5
            
            logger.info("MACD basic functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"MACD basic functionality test failed: {e}")
            return False
    
    async def test_macd_parameter_validation(self) -> bool:
        """Test MACD parameter validation."""
        try:
            # Test invalid periods
            try:
                invalid_macd = MACD(fast_period=0)
                assert False, "Should have raised ValueError for fast_period=0"
            except ValueError:
                pass
            
            # Test fast period >= slow period
            try:
                invalid_macd = MACD(fast_period=30, slow_period=20)
                assert False, "Should have raised ValueError for fast_period >= slow_period"
            except ValueError:
                pass
            
            logger.info("MACD parameter validation test passed")
            return True
            
        except Exception as e:
            logger.error(f"MACD parameter validation test failed: {e}")
            return False
    
    async def test_macd_signal_generation(self) -> bool:
        """Test MACD signal generation."""
        try:
            prices = self.sample_data["prices"]
            macd = MACD()
            result = macd.calculate(prices)
            
            # Check for signal generation
            buy_signals = sum(1 for signal in result.signals if signal in [SignalType.BUY.value, SignalType.STRONG_BUY.value])
            sell_signals = sum(1 for signal in result.signals if signal in [SignalType.SELL.value, SignalType.STRONG_SELL.value])
            
            # Should have some signals (not all HOLD)
            total_signals = buy_signals + sell_signals
            assert total_signals > 0, "No trading signals generated"
            
            # Check MACD line and signal line relationship
            macd_line = result.values[:, 0]
            signal_line = result.values[:, 1]
            histogram = result.values[:, 2]
            
            # Histogram should equal MACD line - signal line
            np.testing.assert_array_almost_equal(histogram, macd_line - signal_line, decimal=10)
            
            logger.info("MACD signal generation test passed")
            return True
            
        except Exception as e:
            logger.error(f"MACD signal generation test failed: {e}")
            return False
    
    async def test_macd_parameter_optimization(self) -> bool:
        """Test MACD parameter optimization."""
        try:
            prices = self.sample_data["prices"]
            macd = MACD()
            
            # Test optimization
            optimal_params = macd.optimize_parameters(prices, target_metric="sharpe_ratio")
            
            # Validate optimization result
            assert "fast_period" in optimal_params
            assert "slow_period" in optimal_params
            assert "signal_period" in optimal_params
            assert "score" in optimal_params
            assert optimal_params["score"] is not None
            
            # Test with optimized parameters
            optimized_macd = MACD(
                fast_period=optimal_params["fast_period"],
                slow_period=optimal_params["slow_period"],
                signal_period=optimal_params["signal_period"]
            )
            optimized_result = optimized_macd.calculate(prices)
            
            assert optimized_result.parameters["fast_period"] == optimal_params["fast_period"]
            assert optimized_result.parameters["slow_period"] == optimal_params["slow_period"]
            assert optimized_result.parameters["signal_period"] == optimal_params["signal_period"]
            
            logger.info("MACD parameter optimization test passed")
            return True
            
        except Exception as e:
            logger.error(f"MACD parameter optimization test failed: {e}")
            return False
    
    async def test_stochastic_basic(self) -> bool:
        """Test basic Stochastic Oscillator functionality."""
        try:
            high = self.sample_data["high"]
            low = self.sample_data["low"]
            close = self.sample_data["prices"]
            
            # Test default Stochastic
            stoch = StochasticOscillator()
            result = stoch.calculate(high, low, close)
            
            # Validate result structure
            assert isinstance(result, IndicatorResult)
            assert result.values is not None
            assert result.signals is not None
            assert result.signal_strength is not None
            assert result.parameters is not None
            assert result.metadata is not None
            
            # Validate Stochastic values (should be 2 columns: %K, %D)
            assert result.values.shape[1] == 2
            assert len(result.values) == len(close)
            
            # Validate %K and %D values (should be between 0 and 100)
            k_values = result.values[:, 0]
            d_values = result.values[:, 1]
            assert np.all((k_values >= 0) & (k_values <= 100) | np.isnan(k_values))
            assert np.all((d_values >= 0) & (d_values <= 100) | np.isnan(d_values))
            
            # Test custom parameters
            custom_stoch = StochasticOscillator(k_period=20, d_period=5, slowing=5, overbought=75, oversold=25)
            custom_result = custom_stoch.calculate(high, low, close)
            assert custom_result.parameters["k_period"] == 20
            assert custom_result.parameters["d_period"] == 5
            assert custom_result.parameters["slowing"] == 5
            assert custom_result.parameters["overbought"] == 75
            assert custom_result.parameters["oversold"] == 25
            
            logger.info("Stochastic basic functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"Stochastic basic functionality test failed: {e}")
            return False
    
    async def test_stochastic_parameter_validation(self) -> bool:
        """Test Stochastic Oscillator parameter validation."""
        try:
            # Test invalid periods
            try:
                invalid_stoch = StochasticOscillator(k_period=0)
                assert False, "Should have raised ValueError for k_period=0"
            except ValueError:
                pass
            
            # Test invalid thresholds
            try:
                invalid_stoch = StochasticOscillator(overbought=20, oversold=80)
                assert False, "Should have raised ValueError for overbought < oversold"
            except ValueError:
                pass
            
            # Test out of range thresholds
            try:
                invalid_stoch = StochasticOscillator(overbought=110, oversold=20)
                assert False, "Should have raised ValueError for overbought > 100"
            except ValueError:
                pass
            
            logger.info("Stochastic parameter validation test passed")
            return True
            
        except Exception as e:
            logger.error(f"Stochastic parameter validation test failed: {e}")
            return False
    
    async def test_stochastic_signal_generation(self) -> bool:
        """Test Stochastic Oscillator signal generation."""
        try:
            high = self.sample_data["high"]
            low = self.sample_data["low"]
            close = self.sample_data["prices"]
            stoch = StochasticOscillator()
            result = stoch.calculate(high, low, close)
            
            # Check for signal generation
            buy_signals = sum(1 for signal in result.signals if signal in [SignalType.BUY.value, SignalType.STRONG_BUY.value])
            sell_signals = sum(1 for signal in result.signals if signal in [SignalType.SELL.value, SignalType.STRONG_SELL.value])
            
            # Should have some signals (not all HOLD)
            total_signals = buy_signals + sell_signals
            assert total_signals > 0, "No trading signals generated"
            
            # Check %K and %D relationship
            k_values = result.values[:, 0]
            d_values = result.values[:, 1]
            
            # %D should be smoother than %K (less variation)
            k_std = np.nanstd(k_values)
            d_std = np.nanstd(d_values)
            assert d_std <= k_std, "%D should be smoother than %K"
            
            logger.info("Stochastic signal generation test passed")
            return True
            
        except Exception as e:
            logger.error(f"Stochastic signal generation test failed: {e}")
            return False
    
    async def test_stochastic_parameter_optimization(self) -> bool:
        """Test Stochastic Oscillator parameter optimization."""
        try:
            high = self.sample_data["high"]
            low = self.sample_data["low"]
            close = self.sample_data["prices"]
            stoch = StochasticOscillator()
            
            # Test optimization
            optimal_params = stoch.optimize_parameters(high, low, close, target_metric="sharpe_ratio")
            
            # Validate optimization result
            assert "k_period" in optimal_params
            assert "d_period" in optimal_params
            assert "slowing" in optimal_params
            assert "overbought" in optimal_params
            assert "oversold" in optimal_params
            assert "score" in optimal_params
            assert optimal_params["score"] is not None
            
            # Test with optimized parameters
            optimized_stoch = StochasticOscillator(
                k_period=optimal_params["k_period"],
                d_period=optimal_params["d_period"],
                slowing=optimal_params["slowing"],
                overbought=optimal_params["overbought"],
                oversold=optimal_params["oversold"]
            )
            optimized_result = optimized_stoch.calculate(high, low, close)
            
            assert optimized_result.parameters["k_period"] == optimal_params["k_period"]
            assert optimized_result.parameters["d_period"] == optimal_params["d_period"]
            assert optimized_result.parameters["slowing"] == optimal_params["slowing"]
            assert optimized_result.parameters["overbought"] == optimal_params["overbought"]
            assert optimized_result.parameters["oversold"] == optimal_params["oversold"]
            
            logger.info("Stochastic parameter optimization test passed")
            return True
            
        except Exception as e:
            logger.error(f"Stochastic parameter optimization test failed: {e}")
            return False
    
    async def test_convenience_functions(self) -> bool:
        """Test convenience functions."""
        try:
            prices = self.sample_data["prices"]
            high = self.sample_data["high"]
            low = self.sample_data["low"]
            close = self.sample_data["prices"]
            
            # Test RSI convenience function
            rsi_result = calculate_rsi(prices)
            assert isinstance(rsi_result, IndicatorResult)
            assert rsi_result.metadata["indicator_type"] == "RSI"
            
            # Test MACD convenience function
            macd_result = calculate_macd(prices)
            assert isinstance(macd_result, IndicatorResult)
            assert macd_result.metadata["indicator_type"] == "MACD"
            
            # Test Stochastic convenience function
            stoch_result = calculate_stochastic(high, low, close)
            assert isinstance(stoch_result, IndicatorResult)
            assert stoch_result.metadata["indicator_type"] == "Stochastic_Oscillator"
            
            logger.info("Convenience functions test passed")
            return True
            
        except Exception as e:
            logger.error(f"Convenience functions test failed: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling scenarios."""
        try:
            # Test insufficient data
            short_prices = self.sample_data["prices"][:5]  # Too short for RSI
            
            try:
                rsi = RSI()
                rsi.calculate(short_prices)
                assert False, "Should have raised ValueError for insufficient data"
            except ValueError:
                pass
            
            # Test mismatched array lengths for Stochastic
            high = self.sample_data["high"]
            low = self.sample_data["low"]
            close_short = self.sample_data["prices"][:50]  # Different length
            
            try:
                stoch = StochasticOscillator()
                stoch.calculate(high, low, close_short)
                assert False, "Should have raised ValueError for mismatched array lengths"
            except ValueError:
                pass
            
            # Test empty arrays
            try:
                rsi = RSI()
                rsi.calculate(np.array([]))
                assert False, "Should have raised ValueError for empty array"
            except ValueError:
                pass
            
            logger.info("Error handling test passed")
            return True
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test integration between indicators."""
        try:
            prices = self.sample_data["prices"]
            high = self.sample_data["high"]
            low = self.sample_data["low"]
            close = self.sample_data["prices"]
            
            # Calculate all indicators
            rsi_result = calculate_rsi(prices)
            macd_result = calculate_macd(prices)
            stoch_result = calculate_stochastic(high, low, close)
            
            # Check that all results have the same length
            assert len(rsi_result.values) == len(macd_result.values) == len(stoch_result.values)
            assert len(rsi_result.signals) == len(macd_result.signals) == len(stoch_result.signals)
            
            # Check that all results have valid metadata
            assert rsi_result.metadata["indicator_type"] == "RSI"
            assert macd_result.metadata["indicator_type"] == "MACD"
            assert stoch_result.metadata["indicator_type"] == "Stochastic_Oscillator"
            
            # Check that all results have valid parameters
            assert "period" in rsi_result.parameters
            assert "fast_period" in macd_result.parameters
            assert "k_period" in stoch_result.parameters
            
            logger.info("Integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print test results summary."""
        logger.info("=" * 60)
        logger.info("MOMENTUM INDICATORS TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "PASSED")
        failed = sum(1 for result in self.test_results if result["status"] == "FAILED")
        errors = sum(1 for result in self.test_results if result["status"] == "ERROR")
        total = len(self.test_results)
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0 or errors > 0:
            logger.info("\nFailed/Error Tests:")
            for result in self.test_results:
                if result["status"] in ["FAILED", "ERROR"]:
                    logger.info(f"  - {result['test']}: {result.get('error', 'Unknown error')}")
        
        logger.info("=" * 60)

async def main():
    """Main test execution function."""
    logger.info("Starting Momentum Indicators Test Suite")
    
    test_suite = MomentumIndicatorsTestSuite()
    await test_suite.run_all_tests()
    
    logger.info("Momentum Indicators Test Suite completed")

if __name__ == "__main__":
    asyncio.run(main())