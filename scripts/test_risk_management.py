#!/usr/bin/env python3
"""
Test script for the Risk Management Framework.
Tests position sizing, stop-loss mechanisms, portfolio risk management, and risk monitoring.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.engine.risk_management import (
    RiskManagementFramework, PositionSizer, StopLossManager, PortfolioRiskManager, RiskMonitor,
    RiskLimit, PositionSizingMethod, StopLossType, RiskMetrics,
    create_risk_framework, calculate_position_size_fixed, calculate_stop_loss_fixed
)
import structlog

logger = structlog.get_logger()

class RiskManagementTestSuite:
    """Comprehensive test suite for the risk management framework."""
    
    def __init__(self):
        self.test_results = {}
        self.sample_data = None
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        # Generate sample returns for multiple assets
        n_days = 252
        n_assets = 5
        
        # Generate correlated returns
        returns_data = {}
        for i in range(n_assets):
            asset_returns = np.random.normal(0.001, 0.02, n_days)
            returns_data[f'ASSET_{i+1}'] = asset_returns
        
        # Create returns dataframe
        returns_df = pd.DataFrame(returns_data)
        
        # Generate sample prices
        prices_data = {}
        for asset in returns_data.keys():
            prices = 100 * np.cumprod(1 + returns_data[asset])
            prices_data[asset] = prices
        
        prices_df = pd.DataFrame(prices_data)
        
        self.sample_data = {
            'returns': returns_df,
            'prices': prices_df,
            'portfolio_value': 1000000.0,
            'risk_limits': RiskLimit(
                max_position_size=0.1,
                max_portfolio_risk=0.02,
                max_drawdown=0.15,
                max_var=0.03,
                max_concentration=0.25,
                max_leverage=2.0
            )
        }
        
        logger.info("Sample data created",
                   n_days=n_days,
                   n_assets=n_assets,
                   portfolio_value=self.sample_data['portfolio_value'])
    
    async def test_position_sizing(self):
        """Test position sizing algorithms."""
        logger.info("Testing position sizing algorithms")
        
        try:
            portfolio_value = self.sample_data['portfolio_value']
            risk_limits = self.sample_data['risk_limits']
            current_price = 100.0
            
            # Test fixed position sizing
            position_sizer = PositionSizer(portfolio_value, risk_limits)
            
            # Fixed size
            fixed_result = position_sizer.calculate_fixed_size(0.05, current_price)
            assert fixed_result.position_size > 0, "Fixed position size should be positive"
            assert fixed_result.allocation_pct == 0.05, "Allocation percentage should match"
            assert fixed_result.method == PositionSizingMethod.FIXED, "Method should be FIXED"
            
            # Kelly criterion
            kelly_result = position_sizer.calculate_kelly_size(0.6, 0.03, -0.015, current_price)
            assert kelly_result.position_size >= 0, "Kelly position size should be non-negative"
            assert kelly_result.method == PositionSizingMethod.KELLY, "Method should be KELLY"
            assert 'kelly_fraction' in kelly_result.parameters, "Should have Kelly fraction"
            
            # Volatility-based sizing
            vol_result = position_sizer.calculate_volatility_size(0.2, 0.15, current_price)
            assert vol_result.position_size >= 0, "Volatility position size should be non-negative"
            assert vol_result.method == PositionSizingMethod.VOLATILITY, "Method should be VOLATILITY"
            
            # Optimal f sizing
            returns = np.random.normal(0.001, 0.02, 100)
            optimal_f_result = position_sizer.calculate_optimal_f_size(returns, current_price)
            assert optimal_f_result.position_size >= 0, "Optimal f position size should be non-negative"
            assert optimal_f_result.method == PositionSizingMethod.OPTIMAL_F, "Method should be OPTIMAL_F"
            
            # Test limits enforcement
            large_allocation_result = position_sizer.calculate_fixed_size(0.5, current_price)
            assert large_allocation_result.allocation_pct <= risk_limits.max_position_size, \
                "Should respect max position size limit"
            
            self.test_results['position_sizing'] = True
            logger.info("Position sizing test passed",
                       fixed_size=fixed_result.position_size,
                       kelly_size=kelly_result.position_size,
                       vol_size=vol_result.position_size,
                       optimal_f_size=optimal_f_result.position_size)
            
        except Exception as e:
            self.test_results['position_sizing'] = False
            logger.error("Position sizing test failed", error=str(e))
            raise
    
    async def test_stop_loss_mechanisms(self):
        """Test stop-loss mechanisms."""
        logger.info("Testing stop-loss mechanisms")
        
        try:
            risk_limits = self.sample_data['risk_limits']
            stop_loss_manager = StopLossManager(risk_limits)
            
            entry_price = 100.0
            current_price = 105.0
            
            # Fixed stop-loss
            fixed_stop = stop_loss_manager.calculate_fixed_stop(entry_price, 'long', 0.05)
            assert fixed_stop.stop_price == entry_price * 0.95, "Fixed stop should be 5% below entry"
            assert fixed_stop.stop_type == StopLossType.FIXED, "Should be FIXED type"
            
            # Short position fixed stop
            short_stop = stop_loss_manager.calculate_fixed_stop(entry_price, 'short', 0.05)
            assert short_stop.stop_price == entry_price * 1.05, "Short stop should be 5% above entry"
            
            # Trailing stop-loss
            trailing_stop = stop_loss_manager.calculate_trailing_stop(
                entry_price, current_price, 'long', 0.05, highest_price=110.0
            )
            assert trailing_stop.stop_price == 110.0 * 0.95, "Trailing stop should be 5% below highest"
            assert trailing_stop.stop_type == StopLossType.TRAILING, "Should be TRAILING type"
            
            # ATR-based stop-loss
            atr_stop = stop_loss_manager.calculate_atr_stop(entry_price, 'long', 2.0, 2.0)
            assert atr_stop.stop_price == entry_price - 4.0, "ATR stop should be 2x ATR below entry"
            assert atr_stop.stop_type == StopLossType.ATR_BASED, "Should be ATR_BASED type"
            
            # Time-based stop-loss
            entry_time = datetime.now() - timedelta(days=15)
            time_stop = stop_loss_manager.calculate_time_stop(entry_time, 30)
            assert time_stop.stop_type == StopLossType.TIME_BASED, "Should be TIME_BASED type"
            assert time_stop.parameters['time_elapsed'] == 15, "Should calculate elapsed time correctly"
            
            self.test_results['stop_loss_mechanisms'] = True
            logger.info("Stop-loss mechanisms test passed",
                       fixed_stop=fixed_stop.stop_price,
                       trailing_stop=trailing_stop.stop_price,
                       atr_stop=atr_stop.stop_price)
            
        except Exception as e:
            self.test_results['stop_loss_mechanisms'] = False
            logger.error("Stop-loss mechanisms test failed", error=str(e))
            raise
    
    async def test_portfolio_risk_management(self):
        """Test portfolio risk management."""
        logger.info("Testing portfolio risk management")
        
        try:
            risk_limits = self.sample_data['risk_limits']
            returns_df = self.sample_data['returns']
            portfolio_risk_manager = PortfolioRiskManager(risk_limits)
            
            # Test with sample weights
            weights = {
                'ASSET_1': 0.3,
                'ASSET_2': 0.3,
                'ASSET_3': 0.2,
                'ASSET_4': 0.1,
                'ASSET_5': 0.1
            }
            
            # Calculate portfolio risk
            risk_metrics = portfolio_risk_manager.calculate_portfolio_risk(returns_df, weights)
            
            # Check metrics structure
            assert isinstance(risk_metrics, RiskMetrics), "Should return RiskMetrics object"
            assert risk_metrics.portfolio_value > 0, "Portfolio value should be positive"
            assert risk_metrics.volatility >= 0, "Volatility should be non-negative"
            assert risk_metrics.max_drawdown <= 0, "Max drawdown should be <= 0"
            assert risk_metrics.concentration_risk >= 0, "Concentration risk should be non-negative"
            assert risk_metrics.leverage >= 0, "Leverage should be non-negative"
            
            # Check risk limits
            violations = portfolio_risk_manager.check_risk_limits(risk_metrics)
            assert isinstance(violations, dict), "Should return violations dictionary"
            assert all(isinstance(v, bool) for v in violations.values()), "Violations should be boolean"
            
            # Test with empty data
            empty_metrics = portfolio_risk_manager.calculate_portfolio_risk(
                pd.DataFrame(), {}
            )
            assert empty_metrics.portfolio_value == 0, "Empty data should return zero portfolio value"
            
            # Test position management
            portfolio_risk_manager.add_position('TEST', {'size': 100, 'price': 50.0})
            assert 'TEST' in portfolio_risk_manager.positions, "Should add position"
            
            portfolio_risk_manager.remove_position('TEST')
            assert 'TEST' not in portfolio_risk_manager.positions, "Should remove position"
            
            # Generate risk report
            report = portfolio_risk_manager.generate_risk_report()
            assert isinstance(report, str), "Should generate string report"
            assert len(report) > 0, "Report should not be empty"
            assert "Portfolio Risk Report" in report, "Report should have title"
            
            self.test_results['portfolio_risk_management'] = True
            logger.info("Portfolio risk management test passed",
                       volatility=risk_metrics.volatility,
                       max_drawdown=risk_metrics.max_drawdown,
                       sharpe_ratio=risk_metrics.sharpe_ratio,
                       n_violations=sum(violations.values()))
            
        except Exception as e:
            self.test_results['portfolio_risk_management'] = False
            logger.error("Portfolio risk management test failed", error=str(e))
            raise
    
    async def test_risk_monitoring(self):
        """Test risk monitoring system."""
        logger.info("Testing risk monitoring system")
        
        try:
            risk_limits = self.sample_data['risk_limits']
            risk_monitor = RiskMonitor(risk_limits, alert_threshold=0.8)
            
            # Test monitoring start/stop
            risk_monitor.start_monitoring()
            assert risk_monitor.monitoring_active, "Monitoring should be active"
            
            risk_monitor.stop_monitoring()
            assert not risk_monitor.monitoring_active, "Monitoring should be inactive"
            
            # Create test metrics
            test_metrics = RiskMetrics(
                portfolio_value=1000000.0,
                total_risk=0.025,
                var_95=-0.025,
                cvar_95=-0.035,
                volatility=0.20,
                current_drawdown=-0.12,
                max_drawdown=-0.12,
                beta=1.0,
                sharpe_ratio=0.8,
                concentration_risk=0.30,
                leverage=1.5
            )
            
            # Test risk alerts
            alerts = risk_monitor.check_risk_alerts(test_metrics)
            assert isinstance(alerts, list), "Should return list of alerts"
            
            # Check for expected alerts (concentration risk should trigger)
            concentration_alerts = [a for a in alerts if 'concentration' in a.lower()]
            assert len(concentration_alerts) > 0, "Should trigger concentration alert"
            
            # Test recent alerts
            recent_alerts = risk_monitor.get_recent_alerts(hours=24)
            assert isinstance(recent_alerts, list), "Should return list of recent alerts"
            assert len(recent_alerts) > 0, "Should have recent alerts"
            
            # Test alert structure
            if recent_alerts:
                alert = recent_alerts[0]
                assert 'timestamp' in alert, "Alert should have timestamp"
                assert 'message' in alert, "Alert should have message"
                assert 'metrics' in alert, "Alert should have metrics"
            
            self.test_results['risk_monitoring'] = True
            logger.info("Risk monitoring test passed",
                       n_alerts=len(alerts),
                       n_recent_alerts=len(recent_alerts))
            
        except Exception as e:
            self.test_results['risk_monitoring'] = False
            logger.error("Risk monitoring test failed", error=str(e))
            raise
    
    async def test_risk_management_framework(self):
        """Test the main risk management framework."""
        logger.info("Testing risk management framework")
        
        try:
            portfolio_value = self.sample_data['portfolio_value']
            risk_limits = self.sample_data['risk_limits']
            returns_df = self.sample_data['returns']
            
            # Create framework
            framework = RiskManagementFramework(portfolio_value, risk_limits)
            
            # Test position sizing
            position_result = framework.calculate_position_size(
                PositionSizingMethod.FIXED,
                allocation_pct=0.05,
                current_price=100.0
            )
            assert position_result.position_size > 0, "Should calculate position size"
            
            # Test stop-loss calculation
            stop_result = framework.calculate_stop_loss(
                StopLossType.FIXED,
                entry_price=100.0,
                position_type='long',
                stop_pct=0.05
            )
            assert stop_result.stop_price > 0, "Should calculate stop price"
            
            # Test portfolio risk update
            weights = {'ASSET_1': 0.5, 'ASSET_2': 0.5}
            risk_metrics = framework.update_portfolio_risk(returns_df, weights)
            assert isinstance(risk_metrics, RiskMetrics), "Should return risk metrics"
            
            # Test risk limit checking
            violations = framework.check_risk_limits(risk_metrics)
            assert isinstance(violations, dict), "Should return violations dictionary"
            
            # Test risk report generation
            report = framework.generate_risk_report()
            assert isinstance(report, str), "Should generate risk report"
            assert len(report) > 0, "Report should not be empty"
            
            # Test recent alerts
            alerts = framework.get_recent_alerts(hours=24)
            assert isinstance(alerts, list), "Should return recent alerts"
            
            self.test_results['risk_management_framework'] = True
            logger.info("Risk management framework test passed",
                       position_size=position_result.position_size,
                       stop_price=stop_result.stop_price,
                       volatility=risk_metrics.volatility,
                       n_violations=sum(violations.values()))
            
        except Exception as e:
            self.test_results['risk_management_framework'] = False
            logger.error("Risk management framework test failed", error=str(e))
            raise
    
    async def test_convenience_functions(self):
        """Test convenience functions."""
        logger.info("Testing convenience functions")
        
        try:
            # Test create_risk_framework
            framework = create_risk_framework(portfolio_value=500000.0, max_drawdown=0.10)
            assert isinstance(framework, RiskManagementFramework), "Should create framework"
            assert framework.portfolio_value == 500000.0, "Should set portfolio value"
            assert framework.risk_limits.max_drawdown == 0.10, "Should set max drawdown"
            
            # Test calculate_position_size_fixed
            position_size = calculate_position_size_fixed(1000000.0, 0.05, 100.0)
            assert position_size > 0, "Should calculate position size"
            assert position_size == 500.0, "Should calculate correct size (1M * 0.05 / 100)"
            
            # Test calculate_stop_loss_fixed
            stop_price = calculate_stop_loss_fixed(100.0, 'long', 0.05)
            assert stop_price == 95.0, "Should calculate correct stop price"
            
            stop_price_short = calculate_stop_loss_fixed(100.0, 'short', 0.05)
            assert stop_price_short == 105.0, "Should calculate correct short stop price"
            
            self.test_results['convenience_functions'] = True
            logger.info("Convenience functions test passed",
                       position_size=position_size,
                       stop_price=stop_price,
                       stop_price_short=stop_price_short)
            
        except Exception as e:
            self.test_results['convenience_functions'] = False
            logger.error("Convenience functions test failed", error=str(e))
            raise
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        logger.info("Testing edge cases")
        
        try:
            # Test with zero portfolio value
            zero_framework = create_risk_framework(portfolio_value=0.0)
            zero_result = zero_framework.calculate_position_size(
                PositionSizingMethod.FIXED,
                allocation_pct=0.05,
                current_price=100.0
            )
            assert zero_result.position_size == 0.0, "Zero portfolio should give zero position"
            
            # Test with zero price
            try:
                zero_price_result = zero_framework.calculate_position_size(
                    PositionSizingMethod.FIXED,
                    allocation_pct=0.05,
                    current_price=0.0
                )
                # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected behavior
            
            # Test with negative allocation
            negative_result = zero_framework.calculate_position_size(
                PositionSizingMethod.FIXED,
                allocation_pct=-0.05,
                current_price=100.0
            )
            assert negative_result.position_size == 0.0, "Negative allocation should give zero position"
            
            # Test with empty returns
            empty_returns = pd.DataFrame()
            empty_weights = {}
            empty_metrics = zero_framework.update_portfolio_risk(empty_returns, empty_weights)
            assert empty_metrics.portfolio_value == 0.0, "Empty data should give zero portfolio value"
            
            # Test invalid position sizing method
            try:
                invalid_result = zero_framework.calculate_position_size("INVALID_METHOD")
                # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected behavior
            
            # Test invalid stop-loss type
            try:
                invalid_stop = zero_framework.calculate_stop_loss("INVALID_TYPE")
                # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected behavior
            
            self.test_results['edge_cases'] = True
            logger.info("Edge cases test passed")
            
        except Exception as e:
            self.test_results['edge_cases'] = False
            logger.error("Edge cases test failed", error=str(e))
            raise
    
    async def test_integration_scenarios(self):
        """Test integration scenarios."""
        logger.info("Testing integration scenarios")
        
        try:
            # Scenario 1: Complete trading workflow
            framework = create_risk_framework(portfolio_value=1000000.0)
            returns_df = self.sample_data['returns']
            
            # Calculate position size
            position_result = framework.calculate_position_size(
                PositionSizingMethod.KELLY,
                win_rate=0.6,
                avg_win=0.03,
                avg_loss=-0.015,
                current_price=100.0
            )
            
            # Calculate stop-loss
            stop_result = framework.calculate_stop_loss(
                StopLossType.TRAILING,
                entry_price=100.0,
                current_price=105.0,
                position_type='long',
                trail_pct=0.05,
                highest_price=110.0
            )
            
            # Update portfolio risk
            weights = {'ASSET_1': 0.4, 'ASSET_2': 0.3, 'ASSET_3': 0.3}
            risk_metrics = framework.update_portfolio_risk(returns_df, weights)
            
            # Check for violations
            violations = framework.check_risk_limits(risk_metrics)
            
            # Generate report
            report = framework.generate_risk_report()
            
            # Verify all components work together
            assert position_result.position_size > 0, "Should calculate position size"
            assert stop_result.stop_price > 0, "Should calculate stop price"
            assert isinstance(risk_metrics, RiskMetrics), "Should calculate risk metrics"
            assert isinstance(violations, dict), "Should check risk limits"
            assert len(report) > 0, "Should generate report"
            
            # Scenario 2: Risk monitoring workflow
            framework.risk_monitor.start_monitoring()
            
            # Simulate multiple risk updates
            for i in range(5):
                # Update with different weights
                weights = {
                    'ASSET_1': 0.2 + i * 0.1,
                    'ASSET_2': 0.3 - i * 0.05,
                    'ASSET_3': 0.5 - i * 0.05
                }
                risk_metrics = framework.update_portfolio_risk(returns_df, weights)
                alerts = framework.get_recent_alerts(hours=1)
            
            framework.risk_monitor.stop_monitoring()
            
            # Verify monitoring worked
            assert len(framework.risk_monitor.alerts) >= 0, "Should have alerts history"
            
            self.test_results['integration_scenarios'] = True
            logger.info("Integration scenarios test passed",
                       position_size=position_result.position_size,
                       stop_price=stop_result.stop_price,
                       volatility=risk_metrics.volatility,
                       n_alerts=len(framework.risk_monitor.alerts))
            
        except Exception as e:
            self.test_results['integration_scenarios'] = False
            logger.error("Integration scenarios test failed", error=str(e))
            raise
    
    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("Starting Risk Management Test Suite")
        
        test_methods = [
            self.test_position_sizing,
            self.test_stop_loss_mechanisms,
            self.test_portfolio_risk_management,
            self.test_risk_monitoring,
            self.test_risk_management_framework,
            self.test_convenience_functions,
            self.test_edge_cases,
            self.test_integration_scenarios
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
        
        logger.info("Risk Management Test Suite completed",
                   passed_tests=passed_tests,
                   total_tests=total_tests,
                   success_rate=passed_tests/total_tests)
        
        return passed_tests == total_tests

async def main():
    """Main test execution function."""
    logger.info("Starting Risk Management Test Suite")
    
    test_suite = RiskManagementTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Risk Management Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Risk Management Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())