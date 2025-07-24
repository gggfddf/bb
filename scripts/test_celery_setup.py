#!/usr/bin/env python3
"""
Test script for the Celery setup and task functionality.
Tests task creation, execution, monitoring, and management.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.celery_app import (
    get_celery_app,
    get_task_manager,
    CeleryTaskManager
)
from data_ingestion.tasks.data_collection_tasks import (
    collect_stock_data,
    collect_historical_data,
    validate_data_quality
)
from data_ingestion.tasks.data_processing_tasks import (
    process_raw_data,
    aggregate_data,
    calculate_technical_indicators,
    optimize_indicator_parameters
)
from data_ingestion.tasks.maintenance_tasks import (
    cleanup_old_data,
    backup_database,
    health_check,
    optimize_database,
    monitor_data_quality
)
import structlog

logger = structlog.get_logger()

class CelerySetupTestSuite:
    def __init__(self):
        self.celery_app = get_celery_app()
        self.task_manager = get_task_manager()
        self.test_results = []

    async def run_all_tests(self):
        """Run all Celery setup tests."""
        logger.info("Starting Celery Setup Test Suite")
        
        tests = [
            ("Celery App Configuration", self.test_celery_app_configuration),
            ("Task Manager Functionality", self.test_task_manager_functionality),
            ("Data Collection Tasks", self.test_data_collection_tasks),
            ("Data Processing Tasks", self.test_data_processing_tasks),
            ("Maintenance Tasks", self.test_maintenance_tasks),
            ("Task Scheduling", self.test_task_scheduling),
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

    async def test_celery_app_configuration(self) -> bool:
        """Test Celery application configuration."""
        try:
            # Test app creation
            assert self.celery_app is not None
            assert self.celery_app.conf is not None
            
            # Test configuration settings
            config = self.celery_app.conf
            assert config.task_serializer == 'json'
            assert config.accept_content == ['json']
            assert config.result_serializer == 'json'
            assert config.timezone == 'UTC'
            assert config.enable_utc is True
            
            # Test task routing
            assert 'task_routes' in config
            assert 'data_ingestion.tasks.data_collection_tasks.*' in config.task_routes
            
            # Test beat schedule
            assert 'beat_schedule' in config
            assert 'collect-stock-data' in config.beat_schedule
            
            logger.info("Celery app configuration test passed")
            return True
        except Exception as e:
            logger.error(f"Celery app configuration test failed: {e}")
            return False

    async def test_task_manager_functionality(self) -> bool:
        """Test task manager functionality."""
        try:
            # Test task manager creation
            assert self.task_manager is not None
            assert isinstance(self.task_manager, CeleryTaskManager)
            
            # Test scheduled tasks retrieval
            scheduled_tasks = self.task_manager.get_scheduled_tasks()
            assert 'scheduled_tasks' in scheduled_tasks
            assert 'total_scheduled' in scheduled_tasks
            assert scheduled_tasks['total_scheduled'] > 0
            
            # Test worker status (may be empty if no workers running)
            worker_status = self.task_manager.get_worker_status()
            assert 'workers' in worker_status
            assert 'total_workers' in worker_status
            
            # Test queue status (may be empty if no tasks in queue)
            queue_status = self.task_manager.get_queue_status()
            assert 'queues' in queue_status
            assert 'total_reserved' in queue_status
            
            logger.info("Task manager functionality test passed")
            return True
        except Exception as e:
            logger.error(f"Task manager functionality test failed: {e}")
            return False

    async def test_data_collection_tasks(self) -> bool:
        """Test data collection tasks."""
        try:
            # Test task function availability
            assert collect_stock_data is not None
            assert collect_historical_data is not None
            assert validate_data_quality is not None
            
            # Test task function signatures
            import inspect
            collect_sig = inspect.signature(collect_stock_data)
            assert 'symbols' in collect_sig.parameters
            assert 'timeframes' in collect_sig.parameters
            assert 'sources' in collect_sig.parameters
            
            historical_sig = inspect.signature(collect_historical_data)
            assert 'symbol' in historical_sig.parameters
            assert 'start_date' in historical_sig.parameters
            assert 'end_date' in historical_sig.parameters
            
            validate_sig = inspect.signature(validate_data_quality)
            assert 'symbol' in validate_sig.parameters
            assert 'timeframe' in validate_sig.parameters
            
            logger.info("Data collection tasks test passed")
            return True
        except Exception as e:
            logger.error(f"Data collection tasks test failed: {e}")
            return False

    async def test_data_processing_tasks(self) -> bool:
        """Test data processing tasks."""
        try:
            # Test task function availability
            assert process_raw_data is not None
            assert aggregate_data is not None
            assert calculate_technical_indicators is not None
            assert optimize_indicator_parameters is not None
            
            # Test task function signatures
            import inspect
            process_sig = inspect.signature(process_raw_data)
            assert 'symbols' in process_sig.parameters
            assert 'timeframes' in process_sig.parameters
            
            aggregate_sig = inspect.signature(aggregate_data)
            assert 'symbol' in aggregate_sig.parameters
            assert 'source_timeframe' in aggregate_sig.parameters
            assert 'target_timeframe' in aggregate_sig.parameters
            
            indicators_sig = inspect.signature(calculate_technical_indicators)
            assert 'symbol' in indicators_sig.parameters
            assert 'timeframe' in indicators_sig.parameters
            assert 'indicators' in indicators_sig.parameters
            
            optimize_sig = inspect.signature(optimize_indicator_parameters)
            assert 'symbol' in optimize_sig.parameters
            assert 'indicator' in optimize_sig.parameters
            assert 'timeframe' in optimize_sig.parameters
            assert 'parameter_ranges' in optimize_sig.parameters
            
            logger.info("Data processing tasks test passed")
            return True
        except Exception as e:
            logger.error(f"Data processing tasks test failed: {e}")
            return False

    async def test_maintenance_tasks(self) -> bool:
        """Test maintenance tasks."""
        try:
            # Test task function availability
            assert cleanup_old_data is not None
            assert backup_database is not None
            assert health_check is not None
            assert optimize_database is not None
            assert monitor_data_quality is not None
            
            # Test task function signatures
            import inspect
            cleanup_sig = inspect.signature(cleanup_old_data)
            assert 'retention_days' in cleanup_sig.parameters
            assert 'symbols' in cleanup_sig.parameters
            
            backup_sig = inspect.signature(backup_database)
            assert 'backup_type' in backup_sig.parameters
            assert 'compression' in backup_sig.parameters
            assert 'verification' in backup_sig.parameters
            
            health_sig = inspect.signature(health_check)
            assert 'check_database' in health_sig.parameters
            assert 'check_data_sources' in health_sig.parameters
            assert 'check_system_resources' in health_sig.parameters
            
            optimize_sig = inspect.signature(optimize_database)
            assert 'optimize_tables' in optimize_sig.parameters
            assert 'update_statistics' in optimize_sig.parameters
            assert 'vacuum_analyze' in optimize_sig.parameters
            
            monitor_sig = inspect.signature(monitor_data_quality)
            assert 'symbols' in monitor_sig.parameters
            assert 'timeframes' in monitor_sig.parameters
            assert 'quality_threshold' in monitor_sig.parameters
            
            logger.info("Maintenance tasks test passed")
            return True
        except Exception as e:
            logger.error(f"Maintenance tasks test failed: {e}")
            return False

    async def test_task_scheduling(self) -> bool:
        """Test task scheduling configuration."""
        try:
            # Test beat schedule configuration
            beat_schedule = self.celery_app.conf.beat_schedule
            
            # Test collect-stock-data schedule
            collect_task = beat_schedule.get('collect-stock-data')
            assert collect_task is not None
            assert collect_task['task'] == 'data_ingestion.tasks.data_collection_tasks.collect_stock_data'
            assert 'schedule' in collect_task
            
            # Test process-raw-data schedule
            process_task = beat_schedule.get('process-raw-data')
            assert process_task is not None
            assert process_task['task'] == 'data_ingestion.tasks.data_processing_tasks.process_raw_data'
            
            # Test maintenance schedules
            cleanup_task = beat_schedule.get('cleanup-old-data')
            assert cleanup_task is not None
            assert cleanup_task['task'] == 'data_ingestion.tasks.maintenance_tasks.cleanup_old_data'
            
            backup_task = beat_schedule.get('backup-database')
            assert backup_task is not None
            assert backup_task['task'] == 'data_ingestion.tasks.maintenance_tasks.backup_database'
            
            health_task = beat_schedule.get('health-check')
            assert health_task is not None
            assert health_task['task'] == 'data_ingestion.tasks.maintenance_tasks.health_check'
            
            logger.info("Task scheduling test passed")
            return True
        except Exception as e:
            logger.error(f"Task scheduling test failed: {e}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling in tasks."""
        try:
            # Test task annotations for error handling
            task_annotations = self.celery_app.conf.task_annotations
            
            # Test global error handling settings
            global_settings = task_annotations.get('*', {})
            assert 'retry' in global_settings
            assert global_settings['retry'] is True
            assert 'retry_policy' in global_settings
            
            retry_policy = global_settings['retry_policy']
            assert 'max_retries' in retry_policy
            assert 'interval_start' in retry_policy
            assert 'interval_step' in retry_policy
            assert 'interval_max' in retry_policy
            
            # Test time limits
            assert 'time_limit' in global_settings
            assert 'soft_time_limit' in global_settings
            
            # Test rate limiting
            assert 'rate_limit' in global_settings
            
            logger.info("Error handling test passed")
            return True
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False

    async def test_integration(self) -> bool:
        """Test integration between components."""
        try:
            # Test that all task modules can be imported
            from data_ingestion.tasks import (
                collect_stock_data,
                process_raw_data,
                cleanup_old_data,
                backup_database,
                health_check
            )
            
            # Test that task manager can access app
            assert self.task_manager.app == self.celery_app
            
            # Test that scheduled tasks are properly configured
            scheduled_tasks = self.task_manager.get_scheduled_tasks()
            assert scheduled_tasks['total_scheduled'] >= 5  # At least 5 scheduled tasks
            
            # Test task routing configuration
            task_routes = self.celery_app.conf.task_routes
            assert 'data_collection' in [route.get('queue') for route in task_routes.values()]
            assert 'data_processing' in [route.get('queue') for route in task_routes.values()]
            assert 'maintenance' in [route.get('queue') for route in task_routes.values()]
            
            logger.info("Integration test passed")
            return True
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False

    def print_test_summary(self):
        """Print test results summary."""
        logger.info("=" * 60)
        logger.info("CELERY SETUP TEST SUMMARY")
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
    logger.info("Starting Celery Setup Test Suite")
    
    test_suite = CelerySetupTestSuite()
    await test_suite.run_all_tests()
    
    logger.info("Celery Setup Test Suite completed")

if __name__ == "__main__":
    asyncio.run(main())