#!/usr/bin/env python3
"""
Test script for the Data Ingestion Scheduler functionality.
Tests task scheduling, execution, and management.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.scheduler import (
    DataIngestionScheduler,
    ScheduledTask,
    ScheduleType,
    TaskPriority,
    get_scheduler,
    start_scheduler,
    stop_scheduler,
    get_scheduler_status
)
import structlog

logger = structlog.get_logger()

class SchedulerTestSuite:
    def __init__(self):
        self.scheduler = DataIngestionScheduler()
        self.test_results = []

    async def run_all_tests(self):
        """Run all scheduler tests."""
        logger.info("Starting Scheduler Test Suite")
        
        tests = [
            ("Scheduler Initialization", self.test_scheduler_initialization),
            ("Task Management", self.test_task_management),
            ("Schedule Calculation", self.test_schedule_calculation),
            ("Task Execution", self.test_task_execution),
            ("Scheduler Status", self.test_scheduler_status),
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

    async def test_scheduler_initialization(self) -> bool:
        """Test scheduler initialization."""
        try:
            # Test scheduler creation
            assert self.scheduler is not None
            assert self.scheduler.celery_app is not None
            assert self.scheduler.task_manager is not None
            assert self.scheduler.orchestrator is not None
            
            # Test default schedules
            assert len(self.scheduler.scheduled_tasks) > 0
            
            # Check for specific default tasks
            default_task_ids = [
                'collect_stock_data_5min',
                'process_raw_data_10min',
                'calculate_indicators_hourly',
                'cleanup_old_data_daily',
                'backup_database_daily',
                'health_check_15min'
            ]
            
            for task_id in default_task_ids:
                assert task_id in self.scheduler.scheduled_tasks
                task = self.scheduler.scheduled_tasks[task_id]
                assert task.enabled is True
                assert task.next_run is not None
            
            logger.info("Scheduler initialization test passed")
            return True
        except Exception as e:
            logger.error(f"Scheduler initialization test failed: {e}")
            return False

    async def test_task_management(self) -> bool:
        """Test task management operations."""
        try:
            # Test adding a new task
            test_task = self.scheduler.add_scheduled_task(
                id='test_task_1',
                name='Test Task 1',
                task_function='data_ingestion.tasks.data_collection_tasks.collect_stock_data',
                schedule_type=ScheduleType.INTERVAL,
                schedule_config={'minutes': 30},
                priority=TaskPriority.NORMAL
            )
            
            assert test_task.id == 'test_task_1'
            assert test_task.name == 'Test Task 1'
            assert test_task.enabled is True
            assert test_task.next_run is not None
            
            # Test updating a task
            updated_task = self.scheduler.update_scheduled_task(
                'test_task_1',
                priority=TaskPriority.HIGH,
                max_retries=5
            )
            
            assert updated_task is not None
            assert updated_task.priority == TaskPriority.HIGH
            assert updated_task.max_retries == 5
            
            # Test enabling/disabling tasks
            assert self.scheduler.disable_task('test_task_1') is True
            assert self.scheduler.scheduled_tasks['test_task_1'].enabled is False
            
            assert self.scheduler.enable_task('test_task_1') is True
            assert self.scheduler.scheduled_tasks['test_task_1'].enabled is True
            
            # Test removing a task
            assert self.scheduler.remove_scheduled_task('test_task_1') is True
            assert 'test_task_1' not in self.scheduler.scheduled_tasks
            
            # Test removing non-existent task
            assert self.scheduler.remove_scheduled_task('non_existent_task') is False
            
            logger.info("Task management test passed")
            return True
        except Exception as e:
            logger.error(f"Task management test failed: {e}")
            return False

    async def test_schedule_calculation(self) -> bool:
        """Test schedule calculation logic."""
        try:
            # Test interval scheduling
            interval_task = ScheduledTask(
                id='interval_test',
                name='Interval Test',
                task_function='test.task',
                schedule_type=ScheduleType.INTERVAL,
                schedule_config={'minutes': 15}
            )
            
            next_run = self.scheduler._calculate_next_run(interval_task)
            assert next_run is not None
            assert next_run > datetime.now()
            assert next_run <= datetime.now() + timedelta(minutes=16)
            
            # Test cron scheduling
            cron_task = ScheduledTask(
                id='cron_test',
                name='Cron Test',
                task_function='test.task',
                schedule_type=ScheduleType.CRON,
                schedule_config={'minute': 30, 'hour': 14}
            )
            
            next_run = self.scheduler._calculate_next_run(cron_task)
            assert next_run is not None
            
            # Test disabled task
            disabled_task = ScheduledTask(
                id='disabled_test',
                name='Disabled Test',
                task_function='test.task',
                schedule_type=ScheduleType.INTERVAL,
                schedule_config={'minutes': 5},
                enabled=False
            )
            
            next_run = self.scheduler._calculate_next_run(disabled_task)
            assert next_run is None
            
            # Test once scheduling
            once_task = ScheduledTask(
                id='once_test',
                name='Once Test',
                task_function='test.task',
                schedule_type=ScheduleType.ONCE,
                schedule_config={},
                next_run=datetime.now() + timedelta(hours=1)
            )
            
            next_run = self.scheduler._calculate_next_run(once_task)
            assert next_run is not None
            assert next_run == once_task.next_run
            
            logger.info("Schedule calculation test passed")
            return True
        except Exception as e:
            logger.error(f"Schedule calculation test failed: {e}")
            return False

    async def test_task_execution(self) -> bool:
        """Test task execution functionality."""
        try:
            # Test retry delay calculation
            retry_delay_1 = self.scheduler._calculate_retry_delay(1)
            retry_delay_2 = self.scheduler._calculate_retry_delay(2)
            retry_delay_3 = self.scheduler._calculate_retry_delay(3)
            
            assert retry_delay_1 == 60  # 1 minute
            assert retry_delay_2 == 120  # 2 minutes
            assert retry_delay_3 == 240  # 4 minutes
            
            # Test priority conversion
            priority_low = self.scheduler._get_celery_priority(TaskPriority.LOW)
            priority_normal = self.scheduler._get_celery_priority(TaskPriority.NORMAL)
            priority_high = self.scheduler._get_celery_priority(TaskPriority.HIGH)
            priority_critical = self.scheduler._get_celery_priority(TaskPriority.CRITICAL)
            
            assert priority_low == 0
            assert priority_normal == 5
            assert priority_high == 8
            assert priority_critical == 10
            
            # Test task execution (simulated)
            test_task = ScheduledTask(
                id='execution_test',
                name='Execution Test',
                task_function='data_ingestion.tasks.data_collection_tasks.collect_stock_data',
                schedule_type=ScheduleType.INTERVAL,
                schedule_config={'minutes': 60}
            )
            
            # Simulate task execution without actually running it
            original_last_run = test_task.last_run
            original_retry_count = test_task.retry_count
            
            # Update task state as if it was executed
            test_task.last_run = datetime.now()
            test_task.retry_count = 0
            test_task.next_run = self.scheduler._calculate_next_run(test_task)
            
            assert test_task.last_run != original_last_run
            assert test_task.retry_count == 0
            assert test_task.next_run is not None
            
            logger.info("Task execution test passed")
            return True
        except Exception as e:
            logger.error(f"Task execution test failed: {e}")
            return False

    async def test_scheduler_status(self) -> bool:
        """Test scheduler status functionality."""
        try:
            # Get scheduler status
            status = self.scheduler.get_scheduler_status()
            
            # Test status structure
            assert 'scheduler_running' in status
            assert 'total_tasks' in status
            assert 'enabled_tasks' in status
            assert 'running_tasks' in status
            assert 'success_rate' in status
            assert 'recent_executions' in status
            assert 'next_scheduled_tasks' in status
            assert 'running_task_details' in status
            
            # Test status values
            assert isinstance(status['total_tasks'], int)
            assert status['total_tasks'] > 0
            assert isinstance(status['enabled_tasks'], int)
            assert status['enabled_tasks'] >= 0
            assert isinstance(status['running_tasks'], int)
            assert status['running_tasks'] >= 0
            assert isinstance(status['success_rate'], float)
            assert 0 <= status['success_rate'] <= 100
            
            # Test next scheduled tasks
            next_tasks = status['next_scheduled_tasks']
            assert isinstance(next_tasks, list)
            
            for task in next_tasks:
                assert 'task_id' in task
                assert 'task_name' in task
                assert 'next_run' in task
                assert 'priority' in task
                assert task['next_run'] > datetime.now()
            
            # Test task details retrieval
            test_task_id = 'collect_stock_data_5min'
            task_details = self.scheduler.get_task_details(test_task_id)
            
            assert task_details is not None
            assert 'task' in task_details
            assert 'execution_history' in task_details
            assert 'celery_status' in task_details
            
            task_info = task_details['task']
            assert task_info['id'] == test_task_id
            assert task_info['enabled'] is True
            
            # Test non-existent task details
            non_existent_details = self.scheduler.get_task_details('non_existent_task')
            assert non_existent_details is None
            
            logger.info("Scheduler status test passed")
            return True
        except Exception as e:
            logger.error(f"Scheduler status test failed: {e}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling scenarios."""
        try:
            # Test adding duplicate task
            try:
                self.scheduler.add_scheduled_task(
                    id='collect_stock_data_5min',  # Already exists
                    name='Duplicate Task',
                    task_function='test.task',
                    schedule_type=ScheduleType.INTERVAL,
                    schedule_config={'minutes': 5}
                )
                assert False, "Should have raised ValueError"
            except ValueError:
                pass  # Expected
            
            # Test updating non-existent task
            result = self.scheduler.update_scheduled_task('non_existent_task', priority=TaskPriority.HIGH)
            assert result is None
            
            # Test enabling/disabling non-existent task
            assert self.scheduler.enable_task('non_existent_task') is False
            assert self.scheduler.disable_task('non_existent_task') is False
            
            # Test invalid schedule configuration
            invalid_task = ScheduledTask(
                id='invalid_test',
                name='Invalid Test',
                task_function='test.task',
                schedule_type=ScheduleType.INTERVAL,
                schedule_config={}  # Empty config
            )
            
            next_run = self.scheduler._calculate_next_run(invalid_task)
            assert next_run is None  # Should handle gracefully
            
            logger.info("Error handling test passed")
            return True
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False

    async def test_integration(self) -> bool:
        """Test integration with other components."""
        try:
            # Test global scheduler functions
            global_scheduler = get_scheduler()
            assert global_scheduler is not None
            assert global_scheduler == self.scheduler  # Should be the same instance
            
            # Test scheduler status via global function
            global_status = get_scheduler_status()
            assert global_status is not None
            assert 'scheduler_running' in global_status
            
            # Test that scheduler can access Celery app
            assert global_scheduler.celery_app is not None
            assert global_scheduler.task_manager is not None
            
            # Test that default tasks are properly configured
            default_tasks = [
                'collect_stock_data_5min',
                'process_raw_data_10min',
                'calculate_indicators_hourly',
                'cleanup_old_data_daily',
                'backup_database_daily',
                'health_check_15min'
            ]
            
            for task_id in default_tasks:
                task = global_scheduler.scheduled_tasks[task_id]
                assert task.task_function.startswith('data_ingestion.tasks.')
                assert task.schedule_type in [ScheduleType.INTERVAL, ScheduleType.CRON]
                assert task.priority in [TaskPriority.LOW, TaskPriority.NORMAL, TaskPriority.HIGH]
            
            logger.info("Integration test passed")
            return True
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False

    def print_test_summary(self):
        """Print test results summary."""
        logger.info("=" * 60)
        logger.info("SCHEDULER TEST SUMMARY")
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
    logger.info("Starting Scheduler Test Suite")
    
    test_suite = SchedulerTestSuite()
    await test_suite.run_all_tests()
    
    logger.info("Scheduler Test Suite completed")

if __name__ == "__main__":
    asyncio.run(main())