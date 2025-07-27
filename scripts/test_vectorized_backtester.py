#!/usr/bin/env python3
"""
Test script for the Vectorized Backtesting Engine.
Tests signal generation, position management, risk management, and performance metrics.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.engine.vectorized_backtester import (
    VectorizedBacktester, RiskManager, SignalType, PositionType,
    run_simple_backtest, calculate_benchmark_metrics
)
import structlog

logger = structlog.get_logger()

class VectorizedBacktesterTestSuite:
    """Comprehensive test suite for the vectorized backtesting engine."""
    
    def __init__(self):
        self.test_results = {}
        self.sample_data = None
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        
        # Generate 1000 days of sample data
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        
        # Generate realistic price movements
        returns = np.random.normal(0.0005, 0.02, 1000)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Add some volatility to create realistic OHLC
            volatility = np.random.uniform(0.005, 0.03)
            
            open_price = price * (1 + np.random.uniform(-volatility/2, volatility/2))
            high_price = max(open_price, price) * (1 + np.random.uniform(0, volatility/2))
            low_price = min(open_price, price) * (1 - np.random.uniform(0, volatility/2))
            close_price = price
            
            volume = np.random.uniform(1000000, 10000000)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        self.sample_data = pd.DataFrame(data, index=dates)
        logger.info("Sample data created", data_length=len(self.sample_data))
    
    async def test_risk_manager(self):
        """Test the RiskManager class."""
        logger.info("Testing RiskManager")
        
        try:
            # Test basic initialization
            risk_manager = RiskManager(
                max_position_size=0.1,
                stop_loss_pct=0.02,
                take_profit_pct=0.04
            )
            
            # Test position size calculation
            position_size = risk_manager.calculate_position_size(
                capital=100000,
                price=100,
                volatility=0.02,
                signal_strength=0.8
            )
            
            assert position_size > 0, "Position size should be positive"
            assert position_size <= 1000, "Position size should be reasonable"
            
            # Test stop loss calculation
            stop_loss_long = risk_manager.calculate_stop_loss(100, PositionType.LONG)
            stop_loss_short = risk_manager.calculate_stop_loss(100, PositionType.SHORT)
            
            assert stop_loss_long == 98.0, f"Long stop loss should be 98.0, got {stop_loss_long}"
            assert stop_loss_short == 102.0, f"Short stop loss should be 102.0, got {stop_loss_short}"
            
            # Test take profit calculation
            take_profit_long = risk_manager.calculate_take_profit(100, PositionType.LONG)
            take_profit_short = risk_manager.calculate_take_profit(100, PositionType.SHORT)
            
            assert take_profit_long == 104.0, f"Long take profit should be 104.0, got {take_profit_long}"
            assert take_profit_short == 96.0, f"Short take profit should be 96.0, got {take_profit_short}"
            
            self.test_results['risk_manager'] = True
            logger.info("RiskManager tests passed")
            
        except Exception as e:
            self.test_results['risk_manager'] = False
            logger.error("RiskManager tests failed", error=str(e))
            raise
    
    async def test_simple_buy_and_hold(self):
        """Test simple buy and hold strategy."""
        logger.info("Testing simple buy and hold strategy")
        
        try:
            # Create simple buy and hold signals
            signals = pd.Series(SignalType.HOLD, index=self.sample_data.index)
            signals.iloc[0] = SignalType.BUY  # Buy at the beginning
            signals.iloc[-1] = SignalType.CLOSE  # Close at the end
            
            # Run backtest
            result = run_simple_backtest(
                data=self.sample_data,
                signals=signals,
                initial_capital=100000
            )
            
            # Basic assertions
            assert len(result.trades) >= 1, "Should have at least one trade"
            assert result.metrics['num_trades'] >= 1, "Should have at least one completed trade"
            assert result.metrics['total_return'] != 0, "Should have some return"
            
            # Check that final trade is closed
            final_trade = result.trades[-1]
            assert final_trade.exit_time is not None, "Final trade should be closed"
            
            self.test_results['simple_buy_hold'] = True
            logger.info("Simple buy and hold test passed", 
                       total_return=result.metrics['total_return'],
                       num_trades=result.metrics['num_trades'])
            
        except Exception as e:
            self.test_results['simple_buy_hold'] = False
            logger.error("Simple buy and hold test failed", error=str(e))
            raise
    
    async def test_moving_average_crossover(self):
        """Test moving average crossover strategy."""
        logger.info("Testing moving average crossover strategy")
        
        try:
            # Calculate moving averages
            short_ma = self.sample_data['close'].rolling(window=10).mean()
            long_ma = self.sample_data['close'].rolling(window=30).mean()
            
            # Generate signals
            signals = pd.Series(SignalType.HOLD, index=self.sample_data.index)
            
            # Buy when short MA crosses above long MA
            buy_signals = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
            signals[buy_signals] = SignalType.BUY
            
            # Sell when short MA crosses below long MA
            sell_signals = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
            signals[sell_signals] = SignalType.SELL
            
            # Run backtest
            backtester = VectorizedBacktester(
                initial_capital=100000,
                commission_rate=0.001,
                slippage_rate=0.0005
            )
            
            def ma_signal_function(data, signals):
                return signals
            
            result = backtester.run_backtest(
                data=self.sample_data,
                signal_function=ma_signal_function,
                signal_params={'signals': signals}
            )
            
            # Basic assertions
            assert len(result.trades) > 0, "Should have trades"
            assert result.metrics['num_trades'] > 0, "Should have completed trades"
            
            # Check that signals are properly processed
            assert len(result.signals) == len(self.sample_data), "Signals length should match data"
            
            self.test_results['ma_crossover'] = True
            logger.info("Moving average crossover test passed",
                       total_return=result.metrics['total_return'],
                       num_trades=result.metrics['num_trades'],
                       win_rate=result.metrics['win_rate'])
            
        except Exception as e:
            self.test_results['ma_crossover'] = False
            logger.error("Moving average crossover test failed", error=str(e))
            raise
    
    async def test_risk_management(self):
        """Test risk management features."""
        logger.info("Testing risk management features")
        
        try:
            # Create a strategy that generates many signals
            signals = pd.Series(SignalType.HOLD, index=self.sample_data.index)
            
            # Generate random buy signals
            buy_indices = np.random.choice(len(signals), size=20, replace=False)
            for idx in buy_indices:
                signals.iloc[idx] = SignalType.BUY
            
            # Generate random sell signals
            sell_indices = np.random.choice(len(signals), size=20, replace=False)
            for idx in sell_indices:
                signals.iloc[idx] = SignalType.SELL
            
            # Run backtest with risk management
            risk_manager = RiskManager(
                max_position_size=0.05,  # Smaller position size
                stop_loss_pct=0.01,      # Tighter stop loss
                take_profit_pct=0.02     # Tighter take profit
            )
            
            backtester = VectorizedBacktester(
                initial_capital=100000,
                commission_rate=0.001,
                slippage_rate=0.0005,
                risk_manager=risk_manager
            )
            
            def test_signal_function(data, signals):
                return signals
            
            result = backtester.run_backtest(
                data=self.sample_data,
                signal_function=test_signal_function,
                signal_params={'signals': signals}
            )
            
            # Check that trades have stop loss and take profit
            for trade in result.trades:
                if trade.exit_time is not None:  # Completed trades
                    assert trade.stop_loss is not None, "Trade should have stop loss"
                    assert trade.take_profit is not None, "Trade should have take profit"
                    assert trade.exit_reason in ["signal", "stop_loss", "take_profit"], \
                        f"Invalid exit reason: {trade.exit_reason}"
            
            self.test_results['risk_management'] = True
            logger.info("Risk management test passed",
                       num_trades=len(result.trades),
                       stop_loss_trades=len([t for t in result.trades if t.exit_reason == "stop_loss"]),
                       take_profit_trades=len([t for t in result.trades if t.exit_reason == "take_profit"]))
            
        except Exception as e:
            self.test_results['risk_management'] = False
            logger.error("Risk management test failed", error=str(e))
            raise
    
    async def test_performance_metrics(self):
        """Test performance metrics calculation."""
        logger.info("Testing performance metrics calculation")
        
        try:
            # Create a simple strategy
            signals = pd.Series(SignalType.HOLD, index=self.sample_data.index)
            signals.iloc[0] = SignalType.BUY
            signals.iloc[-1] = SignalType.CLOSE
            
            result = run_simple_backtest(
                data=self.sample_data,
                signals=signals,
                initial_capital=100000
            )
            
            # Check that all required metrics are present
            required_metrics = [
                'total_return', 'num_trades', 'win_rate', 'avg_win', 'avg_loss',
                'profit_factor', 'volatility', 'sharpe_ratio', 'max_drawdown'
            ]
            
            for metric in required_metrics:
                assert metric in result.metrics, f"Missing metric: {metric}"
                assert isinstance(result.metrics[metric], (int, float)), f"Metric {metric} should be numeric"
            
            # Check metric ranges
            assert 0 <= result.metrics['win_rate'] <= 1, "Win rate should be between 0 and 1"
            assert result.metrics['num_trades'] >= 0, "Number of trades should be non-negative"
            assert result.metrics['max_drawdown'] <= 0, "Max drawdown should be non-positive"
            
            # Test benchmark metrics
            benchmark_metrics = calculate_benchmark_metrics(self.sample_data)
            assert 'benchmark_return' in benchmark_metrics, "Benchmark metrics should include return"
            
            self.test_results['performance_metrics'] = True
            logger.info("Performance metrics test passed",
                       total_return=result.metrics['total_return'],
                       sharpe_ratio=result.metrics['sharpe_ratio'],
                       max_drawdown=result.metrics['max_drawdown'])
            
        except Exception as e:
            self.test_results['performance_metrics'] = False
            logger.error("Performance metrics test failed", error=str(e))
            raise
    
    async def test_data_validation(self):
        """Test data validation features."""
        logger.info("Testing data validation")
        
        try:
            # Test with missing columns
            invalid_data = self.sample_data.drop(columns=['volume'])
            
            try:
                run_simple_backtest(invalid_data, pd.Series(SignalType.HOLD, index=invalid_data.index))
                assert False, "Should raise error for missing columns"
            except ValueError:
                pass  # Expected error
            
            # Test with empty data
            empty_data = pd.DataFrame()
            
            try:
                run_simple_backtest(empty_data, pd.Series())
                assert False, "Should raise error for empty data"
            except ValueError:
                pass  # Expected error
            
            # Test with NaN values (should work with warning)
            data_with_nan = self.sample_data.copy()
            data_with_nan.iloc[0, 0] = np.nan  # Add NaN to first open price
            
            signals = pd.Series(SignalType.HOLD, index=data_with_nan.index)
            signals.iloc[1] = SignalType.BUY
            signals.iloc[-1] = SignalType.CLOSE
            
            result = run_simple_backtest(data_with_nan, signals)
            assert result is not None, "Should handle NaN values gracefully"
            
            self.test_results['data_validation'] = True
            logger.info("Data validation test passed")
            
        except Exception as e:
            self.test_results['data_validation'] = False
            logger.error("Data validation test failed", error=str(e))
            raise
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        logger.info("Testing edge cases")
        
        try:
            # Test with no signals
            no_signals = pd.Series(SignalType.HOLD, index=self.sample_data.index)
            result = run_simple_backtest(self.sample_data, no_signals)
            
            assert result.metrics['num_trades'] == 0, "No trades should be executed"
            assert result.metrics['total_return'] == 0, "No return should be generated"
            
            # Test with all buy signals (no sells)
            all_buy = pd.Series(SignalType.BUY, index=self.sample_data.index)
            result = run_simple_backtest(self.sample_data, all_buy)
            
            # Should have one trade that's not closed
            assert len(result.trades) == 1, "Should have one trade"
            assert result.trades[0].exit_time is None, "Trade should not be closed"
            
            # Test with very small data
            small_data = self.sample_data.iloc[:10]
            small_signals = pd.Series(SignalType.HOLD, index=small_data.index)
            small_signals.iloc[0] = SignalType.BUY
            small_signals.iloc[-1] = SignalType.CLOSE
            
            result = run_simple_backtest(small_data, small_signals)
            assert result is not None, "Should handle small datasets"
            
            self.test_results['edge_cases'] = True
            logger.info("Edge cases test passed")
            
        except Exception as e:
            self.test_results['edge_cases'] = False
            logger.error("Edge cases test failed", error=str(e))
            raise
    
    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("Starting Vectorized Backtester Test Suite")
        
        test_methods = [
            self.test_risk_manager,
            self.test_simple_buy_and_hold,
            self.test_moving_average_crossover,
            self.test_risk_management,
            self.test_performance_metrics,
            self.test_data_validation,
            self.test_edge_cases
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
        
        logger.info("Vectorized Backtester Test Suite completed",
                   passed_tests=passed_tests,
                   total_tests=total_tests,
                   success_rate=passed_tests/total_tests)
        
        return passed_tests == total_tests

async def main():
    """Main test execution function."""
    logger.info("Starting Vectorized Backtester Test Suite")
    
    test_suite = VectorizedBacktesterTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Vectorized Backtester Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Vectorized Backtester Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())