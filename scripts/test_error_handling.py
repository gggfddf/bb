#!/usr/bin/env python3
"""
Error Handling System Test Script

This script tests the comprehensive error handling system including:
- Error handler with retry logic and circuit breaker
- Error monitoring and alerting
- Integration with data ingestion pipeline
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.utils.error_handler import (
    ErrorHandler, 
    global_error_handler, 
    handle_data_collection_error,
    handle_validation_error
)
from data_ingestion.utils.error_monitoring import (
    ErrorMonitoringSystem,
    start_error_monitoring,
    get_monitoring_summary
)
from data_ingestion.utils.data_validator import DataValidator
from data_ingestion.utils.rate_limiter import RateLimiter
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class ErrorHandlingTestSuite:
    """Comprehensive test suite for error handling system."""
    
    def __init__(self):
        self.error_handler = ErrorHandler(
            max_retries=3,
            base_delay=0.1,  # Short delay for testing
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=10.0,
            enable_alerting=True
        )
        self.data_validator = DataValidator()
        self.rate_limiter = RateLimiter(requests_per_minute=60, requests_per_second=10)
        self.test_results = []

    async def run_all_tests(self):
        """Run all error handling tests."""
        logger.info("Starting comprehensive error handling test suite")
        
        tests = [
            ("Basic Error Handler", self.test_basic_error_handler),
            ("Retry Logic", self.test_retry_logic),
            ("Circuit Breaker", self.test_circuit_breaker),
            ("Error Categorization", self.test_error_categorization),
            ("Data Validation Errors", self.test_data_validation_errors),
            ("Rate Limiting Integration", self.test_rate_limiting_integration),
            ("Error Monitoring", self.test_error_monitoring),
            ("Global Error Handler", self.test_global_error_handler),
            ("Integration Test", self.test_integration),
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running test: {test_name}")
                await test_func()
                self.test_results.append((test_name, "PASSED"))
                logger.info(f"Test {test_name} PASSED")
            except Exception as e:
                self.test_results.append((test_name, f"FAILED: {e}"))
                logger.error(f"Test {test_name} FAILED: {e}")
        
        self.print_test_summary()

    async def test_basic_error_handler(self):
        """Test basic error handler functionality."""
        # Test error recording
        test_error = Exception("Test error")
        await self.error_handler._handle_final_failure(test_error, "test_function", (), {})
        
        summary = self.error_handler.get_error_summary()
        assert summary["total_errors"] > 0, "Error should be recorded"
        assert "Exception" in summary["error_stats"], "Error type should be recorded"
        
        logger.info("Basic error handler test passed")

    async def test_retry_logic(self):
        """Test retry logic with exponential backoff."""
        call_count = 0
        
        @self.error_handler.retry_with_backoff(retry_exceptions=(Exception,))
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Simulated failure {call_count}")
            return "Success!"
        
        # This should succeed after 2 retries
        result = await failing_function()
        assert result == "Success!", "Function should succeed after retries"
        assert call_count == 3, f"Function should be called 3 times, got {call_count}"
        
        logger.info("Retry logic test passed")

    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        # Reset circuit breaker
        self.error_handler.circuit_breaker_state = "closed"
        self.error_handler.failure_count = 0
        
        # Test circuit breaker opening
        for i in range(4):  # Exceed threshold
            await self.error_handler._handle_final_failure(
                Exception(f"Failure {i}"), "test_function", (), {}
            )
        
        assert self.error_handler.circuit_breaker_state.value == "open", "Circuit breaker should be open"
        assert not self.error_handler.can_execute(), "Should not allow execution when circuit breaker is open"
        
        # Test circuit breaker half-opening after timeout
        self.error_handler.last_failure_time = datetime.now() - timedelta(seconds=15)
        assert self.error_handler.can_execute(), "Should allow execution when circuit breaker half-opens"
        
        # Test circuit breaker closing after success
        self.error_handler.record_success()
        assert self.error_handler.circuit_breaker_state.value == "closed", "Circuit breaker should close after success"
        
        logger.info("Circuit breaker test passed")

    async def test_error_categorization(self):
        """Test error severity categorization."""
        test_cases = [
            ("Database connection failed", "CRITICAL"),
            ("API rate limit exceeded", "HIGH"),
            ("Data parsing error", "MEDIUM"),
            ("Unknown error", "LOW"),
        ]
        
        for error_message, expected_severity in test_cases:
            severity = self.error_handler._determine_severity(Exception(error_message))
            assert severity.value == expected_severity, f"Expected {expected_severity}, got {severity.value}"
        
        logger.info("Error categorization test passed")

    async def test_data_validation_errors(self):
        """Test data validation error handling."""
        invalid_data = {
            "open": -10,  # Invalid negative price
            "high": 100,
            "low": 50,
            "close": 75,
            "volume": -1000  # Invalid negative volume
        }
        
        try:
            result = handle_validation_error(invalid_data, self.data_validator.validate_stock_data)
            assert False, "Should have raised an exception"
        except Exception as e:
            # Error should be recorded in global error handler
            summary = global_error_handler.get_error_summary()
            assert summary["total_errors"] > 0, "Validation error should be recorded"
        
        logger.info("Data validation error handling test passed")

    async def test_rate_limiting_integration(self):
        """Test rate limiting integration with error handling."""
        # Test rate limiter with error handling
        async def rate_limited_function():
            await self.rate_limiter.wait()
            return "Success"
        
        # This should work without errors
        result = await rate_limited_function()
        assert result == "Success", "Rate limited function should succeed"
        
        logger.info("Rate limiting integration test passed")

    async def test_error_monitoring(self):
        """Test error monitoring system."""
        # Create monitoring system
        monitoring_config = {
            "error_rate_threshold": 0.1,
            "consecutive_failures_threshold": 3,
            "email_alerts_enabled": False,
            "slack_alerts_enabled": False,
            "webhook_alerts_enabled": False
        }
        
        monitoring_system = ErrorMonitoringSystem(monitoring_config)
        
        # Test monitoring rules setup
        assert "error_rate_threshold" in monitoring_system.monitoring_rules
        assert "consecutive_failures_threshold" in monitoring_system.monitoring_rules
        
        # Test alert configuration
        assert monitoring_system.alert_configs["log"].enabled, "Log alerts should be enabled"
        
        logger.info("Error monitoring test passed")

    async def test_global_error_handler(self):
        """Test global error handler functionality."""
        # Test global error handler
        test_error = Exception("Global test error")
        await global_error_handler._handle_final_failure(test_error, "global_test", (), {})
        
        summary = global_error_handler.get_error_summary()
        assert summary["total_errors"] > 0, "Global error handler should record errors"
        
        logger.info("Global error handler test passed")

    async def test_integration(self):
        """Test integration of all error handling components."""
        # Simulate a data collection scenario with errors
        async def simulated_data_collection():
            # Simulate rate limiting
            await self.rate_limiter.wait()
            
            # Simulate data validation
            test_data = {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}
            validation_result = self.data_validator.validate_stock_data(test_data)
            assert validation_result["is_valid"], "Valid data should pass validation"
            
            # Simulate an error
            if True:  # Always trigger for testing
                raise Exception("Simulated data collection error")
        
        # Test with error handling
        try:
            await handle_data_collection_error(simulated_data_collection)
            assert False, "Should have raised an exception"
        except Exception as e:
            # Error should be handled by global error handler
            summary = global_error_handler.get_error_summary()
            assert summary["total_errors"] > 0, "Integration error should be recorded"
        
        logger.info("Integration test passed")

    def print_test_summary(self):
        """Print test results summary."""
        logger.info("=" * 60)
        logger.info("ERROR HANDLING TEST SUITE RESULTS")
        logger.info("=" * 60)
        
        passed = sum(1 for _, result in self.test_results if result == "PASSED")
        total = len(self.test_results)
        
        for test_name, result in self.test_results:
            status = "✅ PASSED" if result == "PASSED" else f"❌ {result}"
            logger.info(f"{test_name:<30} {status}")
        
        logger.info("=" * 60)
        logger.info(f"TOTAL: {passed}/{total} tests passed")
        logger.info("=" * 60)
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Error handling system is working correctly.")
        else:
            logger.error("❌ SOME TESTS FAILED! Please review the error handling system.")

async def main():
    """Main test execution function."""
    logger.info("Starting Error Handling System Test Suite")
    
    # Create and run test suite
    test_suite = ErrorHandlingTestSuite()
    await test_suite.run_all_tests()
    
    # Test global monitoring system
    logger.info("Testing global monitoring system...")
    try:
        await start_error_monitoring()
        monitoring_summary = get_monitoring_summary()
        logger.info(f"Monitoring system started: {monitoring_summary['monitoring_active']}")
        
        # Wait a bit for monitoring to initialize
        await asyncio.sleep(2)
        
        # Get final summary
        final_summary = get_monitoring_summary()
        logger.info("Monitoring system summary:", **final_summary)
        
    except Exception as e:
        logger.error(f"Monitoring system test failed: {e}")
    
    logger.info("Error handling test suite completed")

if __name__ == "__main__":
    asyncio.run(main())