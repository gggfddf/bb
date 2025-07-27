#!/usr/bin/env python3
"""
Test script for the Performance Metrics Calculation Module.
Tests return calculations, risk metrics, risk-adjusted returns, drawdown analysis, and advanced metrics.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.engine.performance_metrics import (
    ReturnCalculator, RiskCalculator, DrawdownCalculator,
    RiskAdjustedReturnCalculator, AdvancedMetricsCalculator,
    PerformanceMetricsCalculator, calculate_performance_metrics,
    calculate_sharpe_ratio, calculate_max_drawdown, calculate_var
)
import structlog

logger = structlog.get_logger()

class PerformanceMetricsTestSuite:
    """Comprehensive test suite for the performance metrics calculation."""
    
    def __init__(self):
        self.test_results = {}
        self.sample_data = None
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Create sample equity curve data for testing."""
        np.random.seed(42)
        
        # Generate 1000 days of sample data
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        
        # Generate realistic equity curve with some drawdowns
        returns = np.random.normal(0.0005, 0.02, 1000)  # Daily returns
        
        # Add some drawdown periods
        returns[200:250] = np.random.normal(-0.01, 0.03, 50)  # Drawdown period 1
        returns[500:550] = np.random.normal(-0.008, 0.025, 50)  # Drawdown period 2
        
        # Calculate equity curve
        equity_curve = 100000 * np.exp(np.cumsum(returns))
        
        self.sample_data = pd.Series(equity_curve, index=dates)
        logger.info("Sample data created", data_length=len(self.sample_data))
    
    async def test_return_calculations(self):
        """Test return calculation methods."""
        logger.info("Testing return calculations")
        
        try:
            # Test total return
            total_return = ReturnCalculator.calculate_total_return(self.sample_data)
            assert total_return != 0, "Total return should not be zero"
            assert isinstance(total_return, float), "Total return should be float"
            
            # Test annualized return
            annualized_return = ReturnCalculator.calculate_annualized_return(self.sample_data)
            assert annualized_return != 0, "Annualized return should not be zero"
            assert isinstance(annualized_return, float), "Annualized return should be float"
            
            # Test geometric mean return
            returns = self.sample_data.pct_change().dropna()
            geometric_return = ReturnCalculator.calculate_geometric_mean_return(returns)
            assert isinstance(geometric_return, float), "Geometric return should be float"
            
            # Test rolling returns
            rolling_returns = ReturnCalculator.calculate_rolling_returns(self.sample_data, 252)
            assert len(rolling_returns) > 0, "Rolling returns should not be empty"
            assert isinstance(rolling_returns, pd.Series), "Rolling returns should be Series"
            
            # Test edge cases
            empty_series = pd.Series(dtype=float)
            assert ReturnCalculator.calculate_total_return(empty_series) == 0.0, "Empty series should return 0"
            assert ReturnCalculator.calculate_annualized_return(empty_series) == 0.0, "Empty series should return 0"
            
            single_value = pd.Series([100])
            assert ReturnCalculator.calculate_total_return(single_value) == 0.0, "Single value should return 0"
            
            self.test_results['return_calculations'] = True
            logger.info("Return calculations test passed",
                       total_return=total_return,
                       annualized_return=annualized_return,
                       geometric_return=geometric_return)
            
        except Exception as e:
            self.test_results['return_calculations'] = False
            logger.error("Return calculations test failed", error=str(e))
            raise
    
    async def test_risk_calculations(self):
        """Test risk calculation methods."""
        logger.info("Testing risk calculations")
        
        try:
            returns = self.sample_data.pct_change().dropna()
            
            # Test volatility
            volatility = RiskCalculator.calculate_volatility(returns)
            assert volatility > 0, "Volatility should be positive"
            assert isinstance(volatility, float), "Volatility should be float"
            
            # Test VaR
            var_95 = RiskCalculator.calculate_var(returns, 0.05)
            var_99 = RiskCalculator.calculate_var(returns, 0.01)
            assert var_95 > var_99, "95% VaR should be greater than 99% VaR"
            assert isinstance(var_95, float), "VaR should be float"
            
            # Test CVaR
            cvar_95 = RiskCalculator.calculate_cvar(returns, 0.05)
            cvar_99 = RiskCalculator.calculate_cvar(returns, 0.01)
            assert cvar_95 >= var_95, "CVaR should be >= VaR"
            assert cvar_99 >= var_99, "CVaR should be >= VaR"
            assert isinstance(cvar_95, float), "CVaR should be float"
            
            # Test downside deviation
            downside_dev = RiskCalculator.calculate_downside_deviation(returns)
            assert downside_dev >= 0, "Downside deviation should be non-negative"
            assert isinstance(downside_dev, float), "Downside deviation should be float"
            
            # Test rolling volatility
            rolling_vol = RiskCalculator.calculate_rolling_volatility(returns, 252)
            assert len(rolling_vol) > 0, "Rolling volatility should not be empty"
            assert isinstance(rolling_vol, pd.Series), "Rolling volatility should be Series"
            
            # Test edge cases
            empty_returns = pd.Series(dtype=float)
            assert RiskCalculator.calculate_volatility(empty_returns) == 0.0, "Empty returns should return 0"
            assert RiskCalculator.calculate_var(empty_returns) == 0.0, "Empty returns should return 0"
            assert RiskCalculator.calculate_cvar(empty_returns) == 0.0, "Empty returns should return 0"
            
            self.test_results['risk_calculations'] = True
            logger.info("Risk calculations test passed",
                       volatility=volatility,
                       var_95=var_95,
                       cvar_95=cvar_95,
                       downside_deviation=downside_dev)
            
        except Exception as e:
            self.test_results['risk_calculations'] = False
            logger.error("Risk calculations test failed", error=str(e))
            raise
    
    async def test_drawdown_calculations(self):
        """Test drawdown calculation methods."""
        logger.info("Testing drawdown calculations")
        
        try:
            # Test drawdown series
            drawdown_series = DrawdownCalculator.calculate_drawdown_series(self.sample_data)
            assert len(drawdown_series) == len(self.sample_data), "Drawdown series should have same length"
            assert drawdown_series.max() <= 0, "Drawdown should be <= 0"
            assert isinstance(drawdown_series, pd.Series), "Drawdown series should be Series"
            
            # Test maximum drawdown
            max_drawdown = DrawdownCalculator.calculate_max_drawdown(self.sample_data)
            assert max_drawdown <= 0, "Maximum drawdown should be <= 0"
            assert max_drawdown == drawdown_series.min(), "Max drawdown should match series min"
            assert isinstance(max_drawdown, float), "Max drawdown should be float"
            
            # Test drawdown duration
            duration_stats = DrawdownCalculator.calculate_drawdown_duration(self.sample_data)
            assert 'max_duration' in duration_stats, "Should have max_duration"
            assert 'avg_duration' in duration_stats, "Should have avg_duration"
            assert 'current_duration' in duration_stats, "Should have current_duration"
            assert duration_stats['max_duration'] >= 0, "Max duration should be >= 0"
            assert duration_stats['avg_duration'] >= 0, "Avg duration should be >= 0"
            
            # Test edge cases
            empty_series = pd.Series(dtype=float)
            assert DrawdownCalculator.calculate_max_drawdown(empty_series) == 0.0, "Empty series should return 0"
            
            single_value = pd.Series([100])
            assert DrawdownCalculator.calculate_max_drawdown(single_value) == 0.0, "Single value should return 0"
            
            self.test_results['drawdown_calculations'] = True
            logger.info("Drawdown calculations test passed",
                       max_drawdown=max_drawdown,
                       max_duration=duration_stats['max_duration'],
                       avg_duration=duration_stats['avg_duration'])
            
        except Exception as e:
            self.test_results['drawdown_calculations'] = False
            logger.error("Drawdown calculations test failed", error=str(e))
            raise
    
    async def test_risk_adjusted_returns(self):
        """Test risk-adjusted return calculations."""
        logger.info("Testing risk-adjusted returns")
        
        try:
            returns = self.sample_data.pct_change().dropna()
            
            # Test Sharpe ratio
            sharpe_ratio = RiskAdjustedReturnCalculator.calculate_sharpe_ratio(returns)
            assert isinstance(sharpe_ratio, float), "Sharpe ratio should be float"
            
            # Test Sortino ratio
            sortino_ratio = RiskAdjustedReturnCalculator.calculate_sortino_ratio(returns)
            assert isinstance(sortino_ratio, float), "Sortino ratio should be float"
            
            # Test Calmar ratio
            calmar_ratio = RiskAdjustedReturnCalculator.calculate_calmar_ratio(returns, self.sample_data)
            assert isinstance(calmar_ratio, float), "Calmar ratio should be float"
            
            # Test information ratio (with benchmark)
            benchmark_returns = pd.Series(np.random.normal(0.0003, 0.015, len(returns)), index=returns.index)
            info_ratio = RiskAdjustedReturnCalculator.calculate_information_ratio(returns, benchmark_returns)
            assert isinstance(info_ratio, float), "Information ratio should be float"
            
            # Test Treynor ratio
            treynor_ratio = RiskAdjustedReturnCalculator.calculate_treynor_ratio(returns, benchmark_returns)
            assert isinstance(treynor_ratio, float), "Treynor ratio should be float"
            
            # Test edge cases
            empty_returns = pd.Series(dtype=float)
            assert RiskAdjustedReturnCalculator.calculate_sharpe_ratio(empty_returns) == 0.0, "Empty returns should return 0"
            assert RiskAdjustedReturnCalculator.calculate_sortino_ratio(empty_returns) == 0.0, "Empty returns should return 0"
            
            self.test_results['risk_adjusted_returns'] = True
            logger.info("Risk-adjusted returns test passed",
                       sharpe_ratio=sharpe_ratio,
                       sortino_ratio=sortino_ratio,
                       calmar_ratio=calmar_ratio,
                       info_ratio=info_ratio,
                       treynor_ratio=treynor_ratio)
            
        except Exception as e:
            self.test_results['risk_adjusted_returns'] = False
            logger.error("Risk-adjusted returns test failed", error=str(e))
            raise
    
    async def test_advanced_metrics(self):
        """Test advanced performance metrics."""
        logger.info("Testing advanced metrics")
        
        try:
            returns = self.sample_data.pct_change().dropna()
            
            # Test Ulcer Index
            ulcer_index = AdvancedMetricsCalculator.calculate_ulcer_index(returns)
            assert ulcer_index >= 0, "Ulcer Index should be >= 0"
            assert isinstance(ulcer_index, float), "Ulcer Index should be float"
            
            # Test Gain-to-Pain ratio
            gain_to_pain = AdvancedMetricsCalculator.calculate_gain_to_pain_ratio(returns)
            assert gain_to_pain >= 0, "Gain-to-Pain ratio should be >= 0"
            assert isinstance(gain_to_pain, float), "Gain-to-Pain ratio should be float"
            
            # Test Profit Factor
            profit_factor = AdvancedMetricsCalculator.calculate_profit_factor(returns)
            assert profit_factor >= 0, "Profit factor should be >= 0"
            assert isinstance(profit_factor, float), "Profit factor should be float"
            
            # Test Win Rate
            win_rate = AdvancedMetricsCalculator.calculate_win_rate(returns)
            assert 0 <= win_rate <= 1, "Win rate should be between 0 and 1"
            assert isinstance(win_rate, float), "Win rate should be float"
            
            # Test Average Win/Loss Ratio
            avg_win_loss = AdvancedMetricsCalculator.calculate_avg_win_loss_ratio(returns)
            assert avg_win_loss >= 0, "Average win/loss ratio should be >= 0"
            assert isinstance(avg_win_loss, float), "Average win/loss ratio should be float"
            
            # Test edge cases
            empty_returns = pd.Series(dtype=float)
            assert AdvancedMetricsCalculator.calculate_ulcer_index(empty_returns) == 0.0, "Empty returns should return 0"
            assert AdvancedMetricsCalculator.calculate_win_rate(empty_returns) == 0.0, "Empty returns should return 0"
            
            self.test_results['advanced_metrics'] = True
            logger.info("Advanced metrics test passed",
                       ulcer_index=ulcer_index,
                       gain_to_pain=gain_to_pain,
                       profit_factor=profit_factor,
                       win_rate=win_rate,
                       avg_win_loss=avg_win_loss)
            
        except Exception as e:
            self.test_results['advanced_metrics'] = False
            logger.error("Advanced metrics test failed", error=str(e))
            raise
    
    async def test_performance_metrics_calculator(self):
        """Test the main PerformanceMetricsCalculator class."""
        logger.info("Testing PerformanceMetricsCalculator")
        
        try:
            calculator = PerformanceMetricsCalculator(
                risk_free_rate=0.02,
                periods_per_year=252,
                confidence_level=0.05
            )
            
            # Test with sample data
            result = calculator.calculate_all_metrics(self.sample_data)
            
            # Check result structure
            assert isinstance(result, type(calculator.calculate_all_metrics.__annotations__['return'])), "Should return PerformanceMetrics"
            assert len(result.metrics) > 0, "Should have calculated metrics"
            assert len(result.returns) > 0, "Should have returns"
            assert len(result.drawdown_series) > 0, "Should have drawdown series"
            
            # Check that all metric categories are present
            expected_categories = [
                'total_return', 'annualized_return', 'volatility', 'max_drawdown',
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'var', 'cvar',
                'ulcer_index', 'gain_to_pain_ratio', 'profit_factor', 'win_rate'
            ]
            
            for metric in expected_categories:
                assert metric in result.metrics, f"Should have {metric} metric"
                assert isinstance(result.metrics[metric], (int, float)), f"{metric} should be numeric"
            
            # Test with benchmark
            benchmark_data = pd.Series(
                np.random.normal(0.0003, 0.015, len(self.sample_data)),
                index=self.sample_data.index
            ).cumsum() * 100000 + 100000
            
            result_with_benchmark = calculator.calculate_all_metrics(self.sample_data, benchmark_data)
            
            assert len(result_with_benchmark.benchmark_comparison) > 0, "Should have benchmark comparison"
            assert 'information_ratio' in result_with_benchmark.benchmark_comparison, "Should have information ratio"
            assert 'beta' in result_with_benchmark.benchmark_comparison, "Should have beta"
            
            # Test rolling metrics
            assert len(result.rolling_metrics) > 0, "Should have rolling metrics"
            assert 'rolling_returns' in result.rolling_metrics, "Should have rolling returns"
            assert 'rolling_volatility' in result.rolling_metrics, "Should have rolling volatility"
            
            self.test_results['performance_metrics_calculator'] = True
            logger.info("PerformanceMetricsCalculator test passed",
                       num_metrics=len(result.metrics),
                       num_rolling_metrics=len(result.rolling_metrics))
            
        except Exception as e:
            self.test_results['performance_metrics_calculator'] = False
            logger.error("PerformanceMetricsCalculator test failed", error=str(e))
            raise
    
    async def test_convenience_functions(self):
        """Test convenience functions."""
        logger.info("Testing convenience functions")
        
        try:
            returns = self.sample_data.pct_change().dropna()
            
            # Test calculate_performance_metrics
            result = calculate_performance_metrics(self.sample_data)
            assert isinstance(result, type(calculate_performance_metrics.__annotations__['return'])), "Should return PerformanceMetrics"
            assert len(result.metrics) > 0, "Should have calculated metrics"
            
            # Test calculate_sharpe_ratio
            sharpe = calculate_sharpe_ratio(returns)
            assert isinstance(sharpe, float), "Sharpe ratio should be float"
            
            # Test calculate_max_drawdown
            max_dd = calculate_max_drawdown(self.sample_data)
            assert isinstance(max_dd, float), "Max drawdown should be float"
            assert max_dd <= 0, "Max drawdown should be <= 0"
            
            # Test calculate_var
            var = calculate_var(returns)
            assert isinstance(var, float), "VaR should be float"
            
            self.test_results['convenience_functions'] = True
            logger.info("Convenience functions test passed",
                       sharpe=sharpe,
                       max_drawdown=max_dd,
                       var=var)
            
        except Exception as e:
            self.test_results['convenience_functions'] = False
            logger.error("Convenience functions test failed", error=str(e))
            raise
    
    async def test_edge_cases_and_validation(self):
        """Test edge cases and input validation."""
        logger.info("Testing edge cases and validation")
        
        try:
            # Test with very small dataset
            small_data = self.sample_data.iloc[:10]
            result = calculate_performance_metrics(small_data)
            assert len(result.metrics) > 0, "Should handle small datasets"
            
            # Test with constant data (no variation)
            constant_data = pd.Series([100] * 100)
            result = calculate_performance_metrics(constant_data)
            assert result.metrics['volatility'] == 0.0, "Constant data should have zero volatility"
            assert result.metrics['max_drawdown'] == 0.0, "Constant data should have zero drawdown"
            
            # Test with all negative returns
            negative_returns = pd.Series([-0.01] * 100)
            negative_equity = 100000 * (1 + negative_returns).cumprod()
            result = calculate_performance_metrics(negative_equity)
            assert result.metrics['total_return'] < 0, "Negative returns should result in negative total return"
            
            # Test with NaN values
            data_with_nan = self.sample_data.copy()
            data_with_nan.iloc[0] = np.nan
            result = calculate_performance_metrics(data_with_nan)
            assert len(result.metrics) > 0, "Should handle NaN values"
            
            # Test with infinite values
            data_with_inf = self.sample_data.copy()
            data_with_inf.iloc[0] = np.inf
            try:
                result = calculate_performance_metrics(data_with_inf)
                # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected behavior
            
            self.test_results['edge_cases_and_validation'] = True
            logger.info("Edge cases and validation test passed")
            
        except Exception as e:
            self.test_results['edge_cases_and_validation'] = False
            logger.error("Edge cases and validation test failed", error=str(e))
            raise
    
    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("Starting Performance Metrics Test Suite")
        
        test_methods = [
            self.test_return_calculations,
            self.test_risk_calculations,
            self.test_drawdown_calculations,
            self.test_risk_adjusted_returns,
            self.test_advanced_metrics,
            self.test_performance_metrics_calculator,
            self.test_convenience_functions,
            self.test_edge_cases_and_validation
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
        
        logger.info("Performance Metrics Test Suite completed",
                   passed_tests=passed_tests,
                   total_tests=total_tests,
                   success_rate=passed_tests/total_tests)
        
        return passed_tests == total_tests

async def main():
    """Main test execution function."""
    logger.info("Starting Performance Metrics Test Suite")
    
    test_suite = PerformanceMetricsTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Performance Metrics Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Performance Metrics Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())