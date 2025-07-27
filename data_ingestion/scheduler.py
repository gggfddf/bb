"""
Data Ingestion Scheduler for managing scheduled tasks.
Handles data collection, processing, and maintenance scheduling.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog

from .celery_app import get_celery_app, get_task_manager
from .orchestrator import DataIngestionOrchestrator
from .utils.error_handler import global_error_handler
from config.settings import DataIngestionSettings, SchedulerSettings

logger = structlog.get_logger()

class ScheduleType(Enum):
    """Types of schedules supported."""
    INTERVAL = "interval"
    CRON = "cron"
    ONCE = "once"
    EVENT_DRIVEN = "event_driven"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ScheduledTask:
    """Represents a scheduled task."""
    id: str
    name: str
    task_function: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class DataIngestionScheduler:
    """
    Main scheduler for data ingestion tasks.
    Manages task scheduling, execution, and monitoring.
    """
    
    def __init__(self, settings: SchedulerSettings = None):
        self.settings = settings or SchedulerSettings()
        self.celery_app = get_celery_app()
        self.task_manager = get_task_manager()
        self.orchestrator = DataIngestionOrchestrator(DataIngestionSettings())
        
        # Task storage
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.running_tasks: Dict[str, Any] = {}
        
        # Scheduler state
        self.is_running = False
        self.scheduler_task = None
        
        # Initialize default schedules
        self._initialize_default_schedules()
    
    def _initialize_default_schedules(self):
        """Initialize default scheduled tasks."""
        default_schedules = [
            {
                'id': 'collect_stock_data_5min',
                'name': 'Collect Stock Data (5min)',
                'task_function': 'data_ingestion.tasks.data_collection_tasks.collect_stock_data',
                'schedule_type': ScheduleType.INTERVAL,
                'schedule_config': {'minutes': 5},
                'priority': TaskPriority.HIGH
            },
            {
                'id': 'process_raw_data_10min',
                'name': 'Process Raw Data (10min)',
                'task_function': 'data_ingestion.tasks.data_processing_tasks.process_raw_data',
                'schedule_type': ScheduleType.INTERVAL,
                'schedule_config': {'minutes': 10},
                'priority': TaskPriority.NORMAL
            },
            {
                'id': 'calculate_indicators_hourly',
                'name': 'Calculate Technical Indicators (Hourly)',
                'task_function': 'data_ingestion.tasks.data_processing_tasks.calculate_technical_indicators',
                'schedule_type': ScheduleType.CRON,
                'schedule_config': {'minute': 0},
                'priority': TaskPriority.NORMAL
            },
            {
                'id': 'cleanup_old_data_daily',
                'name': 'Cleanup Old Data (Daily)',
                'task_function': 'data_ingestion.tasks.maintenance_tasks.cleanup_old_data',
                'schedule_type': ScheduleType.CRON,
                'schedule_config': {'hour': 2, 'minute': 0},
                'priority': TaskPriority.LOW
            },
            {
                'id': 'backup_database_daily',
                'name': 'Backup Database (Daily)',
                'task_function': 'data_ingestion.tasks.maintenance_tasks.backup_database',
                'schedule_type': ScheduleType.CRON,
                'schedule_config': {'hour': 3, 'minute': 0},
                'priority': TaskPriority.HIGH
            },
            {
                'id': 'health_check_15min',
                'name': 'Health Check (15min)',
                'task_function': 'data_ingestion.tasks.maintenance_tasks.health_check',
                'schedule_type': ScheduleType.INTERVAL,
                'schedule_config': {'minutes': 15},
                'priority': TaskPriority.NORMAL
            }
        ]
        
        for schedule_config in default_schedules:
            self.add_scheduled_task(**schedule_config)
    
    def add_scheduled_task(self, id: str, name: str, task_function: str,
                          schedule_type: ScheduleType, schedule_config: Dict[str, Any],
                          priority: TaskPriority = TaskPriority.NORMAL,
                          enabled: bool = True, **kwargs) -> ScheduledTask:
        """
        Add a new scheduled task.
        
        Args:
            id: Unique task identifier
            name: Human-readable task name
            task_function: Celery task function name
            schedule_type: Type of schedule
            schedule_config: Schedule configuration
            priority: Task priority
            enabled: Whether task is enabled
            **kwargs: Additional task parameters
            
        Returns:
            ScheduledTask: Created scheduled task
        """
        if id in self.scheduled_tasks:
            raise ValueError(f"Task with id '{id}' already exists")
        
        task = ScheduledTask(
            id=id,
            name=name,
            task_function=task_function,
            schedule_type=schedule_type,
            schedule_config=schedule_config,
            priority=priority,
            enabled=enabled,
            **kwargs
        )
        
        # Calculate next run time
        task.next_run = self._calculate_next_run(task)
        
        self.scheduled_tasks[id] = task
        logger.info("Added scheduled task", task_id=id, task_name=name, next_run=task.next_run)
        
        return task
    
    def remove_scheduled_task(self, task_id: str) -> bool:
        """
        Remove a scheduled task.
        
        Args:
            task_id: Task identifier to remove
            
        Returns:
            bool: True if task was removed, False if not found
        """
        if task_id in self.scheduled_tasks:
            task = self.scheduled_tasks.pop(task_id)
            logger.info("Removed scheduled task", task_id=task_id, task_name=task.name)
            return True
        return False
    
    def update_scheduled_task(self, task_id: str, **kwargs) -> Optional[ScheduledTask]:
        """
        Update a scheduled task.
        
        Args:
            task_id: Task identifier to update
            **kwargs: Fields to update
            
        Returns:
            ScheduledTask: Updated task, or None if not found
        """
        if task_id not in self.scheduled_tasks:
            return None
        
        task = self.scheduled_tasks[task_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        # Recalculate next run if schedule changed
        if 'schedule_type' in kwargs or 'schedule_config' in kwargs:
            task.next_run = self._calculate_next_run(task)
        
        task.updated_at = datetime.now()
        
        logger.info("Updated scheduled task", task_id=task_id, updates=kwargs)
        return task
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a scheduled task."""
        task = self.update_scheduled_task(task_id, enabled=True)
        return task is not None
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a scheduled task."""
        task = self.update_scheduled_task(task_id, enabled=False)
        return task is not None
    
    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """
        Calculate the next run time for a task.
        
        Args:
            task: Scheduled task
            
        Returns:
            datetime: Next run time, or None if task is disabled
        """
        if not task.enabled:
            return None
        
        now = datetime.now()
        
        if task.schedule_type == ScheduleType.INTERVAL:
            interval = task.schedule_config
            if 'minutes' in interval:
                return now + timedelta(minutes=interval['minutes'])
            elif 'hours' in interval:
                return now + timedelta(hours=interval['hours'])
            elif 'days' in interval:
                return now + timedelta(days=interval['days'])
        
        elif task.schedule_type == ScheduleType.CRON:
            # Simple cron-like scheduling
            cron_config = task.schedule_config
            next_run = now.replace(second=0, microsecond=0)
            
            if 'minute' in cron_config:
                next_run = next_run.replace(minute=cron_config['minute'])
            if 'hour' in cron_config:
                next_run = next_run.replace(hour=cron_config['hour'])
            if 'day' in cron_config:
                next_run = next_run.replace(day=cron_config['day'])
            
            # If next run is in the past, add appropriate interval
            if next_run <= now:
                if 'minute' in cron_config:
                    next_run += timedelta(hours=1)
                elif 'hour' in cron_config:
                    next_run += timedelta(days=1)
                elif 'day' in cron_config:
                    next_run += timedelta(days=30)  # Approximate month
        
        elif task.schedule_type == ScheduleType.ONCE:
            # One-time execution
            if task.next_run and task.next_run > now:
                return task.next_run
            return None
        
        return next_run
    
    async def start_scheduler(self):
        """Start the scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        logger.info("Starting data ingestion scheduler")
        
        # Start the scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # Start orchestrator
        await self.orchestrator.start_collection()
    
    async def stop_scheduler(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        self.is_running = False
        logger.info("Stopping data ingestion scheduler")
        
        # Cancel scheduler task
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Stop orchestrator
        await self.orchestrator.stop_collection()
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                now = datetime.now()
                
                # Check for tasks that need to run
                tasks_to_run = []
                for task in self.scheduled_tasks.values():
                    if (task.enabled and task.next_run and 
                        task.next_run <= now):
                        tasks_to_run.append(task)
                
                # Execute tasks
                for task in tasks_to_run:
                    await self._execute_task(task)
                
                # Sleep for a short interval
                await asyncio.sleep(self.settings.scheduler_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in scheduler loop", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _execute_task(self, task: ScheduledTask):
        """
        Execute a scheduled task.
        
        Args:
            task: Task to execute
        """
        try:
            logger.info("Executing scheduled task", 
                       task_id=task.id, task_name=task.name)
            
            # Update task state
            task.last_run = datetime.now()
            task.retry_count = 0
            
            # Execute task via Celery
            celery_task = self.celery_app.send_task(
                task.task_function,
                kwargs=task.metadata.get('task_args', {}),
                countdown=task.metadata.get('countdown', 0),
                expires=task.metadata.get('expires', None),
                priority=self._get_celery_priority(task.priority)
            )
            
            # Store running task
            self.running_tasks[task.id] = {
                'celery_task_id': celery_task.id,
                'started_at': datetime.now(),
                'task': task
            }
            
            # Update next run time
            task.next_run = self._calculate_next_run(task)
            
            # Record task execution
            self.task_history.append({
                'task_id': task.id,
                'task_name': task.name,
                'executed_at': task.last_run,
                'celery_task_id': celery_task.id,
                'status': 'started'
            })
            
            logger.info("Scheduled task execution started", 
                       task_id=task.id, celery_task_id=celery_task.id)
            
        except Exception as e:
            logger.error("Failed to execute scheduled task", 
                        task_id=task.id, error=str(e))
            
            # Record failure
            self.task_history.append({
                'task_id': task.id,
                'task_name': task.name,
                'executed_at': datetime.now(),
                'status': 'failed',
                'error': str(e)
            })
            
            # Handle retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                retry_delay = self._calculate_retry_delay(task.retry_count)
                task.next_run = datetime.now() + timedelta(seconds=retry_delay)
                logger.info("Scheduled retry for task", 
                           task_id=task.id, retry_count=task.retry_count, 
                           next_retry=task.next_run)
    
    def _get_celery_priority(self, priority: TaskPriority) -> int:
        """Convert task priority to Celery priority."""
        priority_map = {
            TaskPriority.LOW: 0,
            TaskPriority.NORMAL: 5,
            TaskPriority.HIGH: 8,
            TaskPriority.CRITICAL: 10
        }
        return priority_map.get(priority, 5)
    
    def _calculate_retry_delay(self, retry_count: int) -> int:
        """Calculate retry delay with exponential backoff."""
        base_delay = 60  # 1 minute
        return base_delay * (2 ** (retry_count - 1))
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get scheduler status and statistics.
        
        Returns:
            Dict containing scheduler status
        """
        now = datetime.now()
        
        # Calculate statistics
        total_tasks = len(self.scheduled_tasks)
        enabled_tasks = sum(1 for task in self.scheduled_tasks.values() if task.enabled)
        running_tasks = len(self.running_tasks)
        
        # Get recent task history
        recent_history = [
            record for record in self.task_history[-50:]  # Last 50 executions
        ]
        
        # Calculate success rate
        if recent_history:
            successful = sum(1 for record in recent_history if record.get('status') == 'completed')
            success_rate = (successful / len(recent_history)) * 100
        else:
            success_rate = 0.0
        
        return {
            'scheduler_running': self.is_running,
            'total_tasks': total_tasks,
            'enabled_tasks': enabled_tasks,
            'running_tasks': running_tasks,
            'success_rate': success_rate,
            'recent_executions': len(recent_history),
            'next_scheduled_tasks': [
                {
                    'task_id': task.id,
                    'task_name': task.name,
                    'next_run': task.next_run,
                    'priority': task.priority.value
                }
                for task in self.scheduled_tasks.values()
                if task.enabled and task.next_run and task.next_run > now
            ],
            'running_task_details': [
                {
                    'task_id': task_id,
                    'celery_task_id': details['celery_task_id'],
                    'started_at': details['started_at'],
                    'task_name': details['task'].name
                }
                for task_id, details in self.running_tasks.items()
            ]
        }
    
    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dict containing task details, or None if not found
        """
        if task_id not in self.scheduled_tasks:
            return None
        
        task = self.scheduled_tasks[task_id]
        
        # Get task execution history
        task_history = [
            record for record in self.task_history
            if record['task_id'] == task_id
        ]
        
        # Get Celery task status if running
        celery_status = None
        if task_id in self.running_tasks:
            celery_task_id = self.running_tasks[task_id]['celery_task_id']
            celery_status = self.task_manager.get_task_status(celery_task_id)
        
        return {
            'task': {
                'id': task.id,
                'name': task.name,
                'task_function': task.task_function,
                'schedule_type': task.schedule_type.value,
                'schedule_config': task.schedule_config,
                'priority': task.priority.value,
                'enabled': task.enabled,
                'last_run': task.last_run,
                'next_run': task.next_run,
                'retry_count': task.retry_count,
                'max_retries': task.max_retries,
                'timeout': task.timeout,
                'metadata': task.metadata,
                'created_at': task.created_at,
                'updated_at': task.updated_at
            },
            'execution_history': task_history[-20:],  # Last 20 executions
            'celery_status': celery_status
        }

# Global scheduler instance
_scheduler_instance: Optional[DataIngestionScheduler] = None

def get_scheduler() -> DataIngestionScheduler:
    """Get the global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = DataIngestionScheduler()
    return _scheduler_instance

async def start_scheduler():
    """Start the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.start_scheduler()

async def stop_scheduler():
    """Stop the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.stop_scheduler()

def get_scheduler_status() -> Dict[str, Any]:
    """Get the global scheduler status."""
    scheduler = get_scheduler()
    return scheduler.get_scheduler_status()