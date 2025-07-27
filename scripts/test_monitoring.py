#!/usr/bin/env python3
"""
Test script for the System Monitoring functionality.
Tests metrics collection, health checks, alerts, and monitoring status.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.monitoring.system_monitor import (
    SystemMonitor,
    Metric,
    Alert,
    MetricType,
    AlertLevel,
    get_system_monitor,
    start_system_monitoring,
    stop_system_monitoring,
    get_monitoring_status
)
import structlog

logger = structlog.get_logger()

class MonitoringTestSuite:
    def __init__(self):
        self.monitor = SystemMonitor()
        self.test_results = []

    async def run_all_tests(self):
        """Run all monitoring tests."""
        logger.info("Starting System Monitoring Test Suite")
        
        tests = [
            ("Monitor Initialization", self.test_monitor_initialization),
            ("Metrics Collection", self.test_metrics_collection),
            ("Health Checks", self.test_health_checks),
            ("Alert System", self.test_alert_system),
            ("Monitoring Status", self.test_monitoring_status),
            ("Metric History", self.test_metric_history),
            ("Alert Resolution", self.test_alert_resolution),
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

    async def test_monitor_initialization(self) -> bool:
        """Test monitor initialization."""
        try:
            # Test monitor creation
            assert self.monitor is not None
            assert self.monitor.orchestrator is not None
            assert self.monitor.scheduler is not None
            
            # Test health checks initialization
            assert len(self.monitor.health_checks) > 0
            expected_checks = [
                'system_resources', 'data_ingestion', 'database_connection',
                'scheduler_status', 'error_rate', 'data_quality',
                'task_queue', 'storage_usage'
            ]
            
            for check_name in expected_checks:
                assert check_name in self.monitor.health_checks
                assert callable(self.monitor.health_checks[check_name])
            
            # Test initial state
            assert self.monitor.is_monitoring is False
            assert len(self.monitor.metrics) == 0
            assert len(self.monitor.alerts) == 0
            
            logger.info("Monitor initialization test passed")
            return True
        except Exception as e:
            logger.error(f"Monitor initialization test failed: {e}")
            return False

    async def test_metrics_collection(self) -> bool:
        """Test metrics collection functionality."""
        try:
            # Test metric creation
            test_metric = Metric(
                name='test_metric',
                value=42.5,
                metric_type=MetricType.GAUGE,
                timestamp=datetime.now(),
                description="Test metric"
            )
            
            assert test_metric.name == 'test_metric'
            assert test_metric.value == 42.5
            assert test_metric.metric_type == MetricType.GAUGE
            assert test_metric.description == "Test metric"
            
            # Test adding metrics to monitor
            self.monitor._add_metric(
                name='test_gauge',
                value=75.0,
                metric_type=MetricType.GAUGE,
                description="Test gauge metric"
            )
            
            self.monitor._add_metric(
                name='test_counter',
                value=100,
                metric_type=MetricType.COUNTER,
                labels={'source': 'test'},
                description="Test counter metric"
            )
            
            assert len(self.monitor.metrics) == 2
            
            # Test metric types
            gauge_metrics = [m for m in self.monitor.metrics if m.metric_type == MetricType.GAUGE]
            counter_metrics = [m for m in self.monitor.metrics if m.metric_type == MetricType.COUNTER]
            
            assert len(gauge_metrics) == 1
            assert len(counter_metrics) == 1
            assert gauge_metrics[0].name == 'test_gauge'
            assert counter_metrics[0].name == 'test_counter'
            
            logger.info("Metrics collection test passed")
            return True
        except Exception as e:
            logger.error(f"Metrics collection test failed: {e}")
            return False

    async def test_health_checks(self) -> bool:
        """Test health check functionality."""
        try:
            # Test system resources health check
            system_result = await self.monitor._check_system_resources()
            assert 'healthy' in system_result
            assert 'message' in system_result
            assert 'metadata' in system_result
            assert isinstance(system_result['healthy'], bool)
            
            # Test data ingestion health check
            ingestion_result = await self.monitor._check_data_ingestion()
            assert 'healthy' in ingestion_result
            assert 'message' in ingestion_result
            assert 'metadata' in ingestion_result
            
            # Test scheduler status health check
            scheduler_result = await self.monitor._check_scheduler_status()
            assert 'healthy' in scheduler_result
            assert 'message' in scheduler_result
            assert 'metadata' in scheduler_result
            
            # Test error rate health check
            error_result = await self.monitor._check_error_rate()
            assert 'healthy' in error_result
            assert 'message' in error_result
            assert 'metadata' in error_result
            
            # Test storage usage health check
            storage_result = await self.monitor._check_storage_usage()
            assert 'healthy' in storage_result
            assert 'message' in storage_result
            assert 'metadata' in storage_result
            
            # Verify metadata contains expected fields
            if system_result['metadata']:
                assert 'cpu_percent' in system_result['metadata']
                assert 'memory_percent' in system_result['metadata']
                assert 'disk_percent' in system_result['metadata']
            
            if storage_result['metadata']:
                assert 'total_gb' in storage_result['metadata']
                assert 'used_gb' in storage_result['metadata']
                assert 'free_gb' in storage_result['metadata']
                assert 'percent_used' in storage_result['metadata']
            
            logger.info("Health checks test passed")
            return True
        except Exception as e:
            logger.error(f"Health checks test failed: {e}")
            return False

    async def test_alert_system(self) -> bool:
        """Test alert system functionality."""
        try:
            # Test alert creation
            test_alert = Alert(
                id='test_alert_1',
                title='Test Alert',
                message='This is a test alert',
                level=AlertLevel.WARNING,
                source='test_source',
                timestamp=datetime.now()
            )
            
            assert test_alert.id == 'test_alert_1'
            assert test_alert.title == 'Test Alert'
            assert test_alert.level == AlertLevel.WARNING
            assert test_alert.resolved is False
            
            # Test creating alerts through monitor
            self.monitor._create_alert(
                title='Test Warning Alert',
                message='This is a warning alert',
                level=AlertLevel.WARNING,
                source='test.warning'
            )
            
            self.monitor._create_alert(
                title='Test Error Alert',
                message='This is an error alert',
                level=AlertLevel.ERROR,
                source='test.error',
                metadata={'error_code': 500}
            )
            
            assert len(self.monitor.alerts) == 2
            
            # Test alert levels
            warning_alerts = [a for a in self.monitor.alerts if a.level == AlertLevel.WARNING]
            error_alerts = [a for a in self.monitor.alerts if a.level == AlertLevel.ERROR]
            
            assert len(warning_alerts) == 1
            assert len(error_alerts) == 1
            assert warning_alerts[0].title == 'Test Warning Alert'
            assert error_alerts[0].title == 'Test Error Alert'
            assert error_alerts[0].metadata.get('error_code') == 500
            
            logger.info("Alert system test passed")
            return True
        except Exception as e:
            logger.error(f"Alert system test failed: {e}")
            return False

    async def test_monitoring_status(self) -> bool:
        """Test monitoring status functionality."""
        try:
            # Get monitoring status
            status = self.monitor.get_monitoring_status()
            
            # Test status structure
            assert 'monitoring_active' in status
            assert 'uptime_seconds' in status
            assert 'total_metrics_collected' in status
            assert 'recent_metrics' in status
            assert 'alert_statistics' in status
            assert 'active_alerts' in status
            assert 'last_cleanup' in status
            assert 'health_checks' in status
            
            # Test status values
            assert isinstance(status['monitoring_active'], bool)
            assert isinstance(status['uptime_seconds'], float)
            assert isinstance(status['total_metrics_collected'], int)
            assert isinstance(status['recent_metrics'], list)
            assert isinstance(status['active_alerts'], list)
            assert isinstance(status['health_checks'], list)
            
            # Test alert statistics
            alert_stats = status['alert_statistics']
            assert 'total_alerts' in alert_stats
            assert 'active_alerts' in alert_stats
            assert 'critical_alerts' in alert_stats
            assert 'error_alerts' in alert_stats
            assert 'warning_alerts' in alert_stats
            
            # Test recent metrics structure
            if status['recent_metrics']:
                metric = status['recent_metrics'][0]
                assert 'name' in metric
                assert 'value' in metric
                assert 'type' in metric
                assert 'timestamp' in metric
                assert 'labels' in metric
                assert 'description' in metric
            
            # Test active alerts structure
            if status['active_alerts']:
                alert = status['active_alerts'][0]
                assert 'id' in alert
                assert 'title' in alert
                assert 'message' in alert
                assert 'level' in alert
                assert 'source' in alert
                assert 'timestamp' in alert
                assert 'metadata' in alert
            
            logger.info("Monitoring status test passed")
            return True
        except Exception as e:
            logger.error(f"Monitoring status test failed: {e}")
            return False

    async def test_metric_history(self) -> bool:
        """Test metric history functionality."""
        try:
            # Add some test metrics with different timestamps
            now = datetime.now()
            
            # Add metrics for the last hour
            for i in range(5):
                timestamp = now - timedelta(minutes=i * 15)
                self.monitor._add_metric(
                    name='test_history_metric',
                    value=10.0 + i,
                    metric_type=MetricType.GAUGE,
                    timestamp=timestamp
                )
            
            # Test metric history retrieval
            history = self.monitor.get_metric_history('test_history_metric', hours=1)
            
            assert len(history) == 5
            assert all('value' in metric for metric in history)
            assert all('timestamp' in metric for metric in history)
            assert all('labels' in metric for metric in history)
            
            # Test history ordering (should be sorted by timestamp)
            timestamps = [metric['timestamp'] for metric in history]
            assert timestamps == sorted(timestamps)
            
            # Test history filtering
            history_30min = self.monitor.get_metric_history('test_history_metric', hours=0.5)
            assert len(history_30min) <= 3  # Should only include metrics from last 30 minutes
            
            # Test non-existent metric
            empty_history = self.monitor.get_metric_history('non_existent_metric', hours=24)
            assert len(empty_history) == 0
            
            logger.info("Metric history test passed")
            return True
        except Exception as e:
            logger.error(f"Metric history test failed: {e}")
            return False

    async def test_alert_resolution(self) -> bool:
        """Test alert resolution functionality."""
        try:
            # Create a test alert
            test_alert_id = 'test_resolution_alert'
            self.monitor._create_alert(
                title='Test Resolution Alert',
                message='This alert should be resolved',
                level=AlertLevel.INFO,
                source='test.resolution'
            )
            
            # Find the alert
            test_alert = None
            for alert in self.monitor.alerts:
                if alert.title == 'Test Resolution Alert':
                    test_alert = alert
                    break
            
            assert test_alert is not None
            assert test_alert.resolved is False
            assert test_alert.resolved_at is None
            
            # Resolve the alert
            resolution_result = self.monitor.resolve_alert(test_alert.id)
            assert resolution_result is True
            
            # Verify alert is resolved
            assert test_alert.resolved is True
            assert test_alert.resolved_at is not None
            assert isinstance(test_alert.resolved_at, datetime)
            
            # Test resolving non-existent alert
            non_existent_result = self.monitor.resolve_alert('non_existent_alert_id')
            assert non_existent_result is False
            
            # Test resolving already resolved alert
            already_resolved_result = self.monitor.resolve_alert(test_alert.id)
            assert already_resolved_result is False
            
            logger.info("Alert resolution test passed")
            return True
        except Exception as e:
            logger.error(f"Alert resolution test failed: {e}")
            return False

    async def test_integration(self) -> bool:
        """Test integration with other components."""
        try:
            # Test global monitor functions
            global_monitor = get_system_monitor()
            assert global_monitor is not None
            assert global_monitor == self.monitor  # Should be the same instance
            
            # Test monitoring status via global function
            global_status = get_monitoring_status()
            assert global_status is not None
            assert 'monitoring_active' in global_status
            
            # Test that monitor can access orchestrator and scheduler
            assert global_monitor.orchestrator is not None
            assert global_monitor.scheduler is not None
            
            # Test health check functions are callable
            for check_name, check_func in global_monitor.health_checks.items():
                assert callable(check_func)
                # Test that health check returns expected structure
                result = await check_func()
                assert isinstance(result, dict)
                assert 'healthy' in result
                assert 'message' in result
                assert 'metadata' in result
            
            # Test metric types enum
            assert MetricType.GAUGE.value == "gauge"
            assert MetricType.COUNTER.value == "counter"
            assert MetricType.HISTOGRAM.value == "histogram"
            assert MetricType.SUMMARY.value == "summary"
            
            # Test alert levels enum
            assert AlertLevel.INFO.value == "info"
            assert AlertLevel.WARNING.value == "warning"
            assert AlertLevel.ERROR.value == "error"
            assert AlertLevel.CRITICAL.value == "critical"
            
            logger.info("Integration test passed")
            return True
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False

    def print_test_summary(self):
        """Print test results summary."""
        logger.info("=" * 60)
        logger.info("SYSTEM MONITORING TEST SUMMARY")
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
    logger.info("Starting System Monitoring Test Suite")
    
    test_suite = MonitoringTestSuite()
    await test_suite.run_all_tests()
    
    logger.info("System Monitoring Test Suite completed")

if __name__ == "__main__":
    asyncio.run(main())