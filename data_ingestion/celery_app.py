"""
Celery application configuration for distributed data ingestion tasks.
Handles task scheduling, worker management, and distributed processing.
"""

import os
import logging
from typing import Dict, Any, Optional
from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger
import structlog

from config.settings import CelerySettings

logger = structlog.get_logger()

def create_celery_app() -> Celery:
    """
    Create and configure Celery application for data ingestion tasks.
    
    Returns:
        Celery: Configured Celery application instance
    """
    # Get Celery settings
    celery_settings = CelerySettings()
    
    # Create Celery app
    app = Celery(
        'stock_predictor_data_ingestion',
        broker=celery_settings.broker_url,
        backend=celery_settings.result_backend,
        include=[
            'data_ingestion.tasks.data_collection_tasks',
            'data_ingestion.tasks.data_processing_tasks',
            'data_ingestion.tasks.maintenance_tasks'
        ]
    )
    
    # Configure Celery settings
    app.conf.update(
        # Task routing
        task_routes={
            'data_ingestion.tasks.data_collection_tasks.*': {'queue': 'data_collection'},
            'data_ingestion.tasks.data_processing_tasks.*': {'queue': 'data_processing'},
            'data_ingestion.tasks.maintenance_tasks.*': {'queue': 'maintenance'},
        },
        
        # Task serialization
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        
        # Task execution settings
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_always_eager=False,
        task_eager_propagates=True,
        
        # Worker settings
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
        worker_disable_rate_limits=False,
        
        # Result backend settings
        result_expires=3600,  # 1 hour
        result_persistent=True,
        
        # Beat scheduler settings
        beat_schedule={
            'collect-stock-data': {
                'task': 'data_ingestion.tasks.data_collection_tasks.collect_stock_data',
                'schedule': crontab(minute='*/5'),  # Every 5 minutes
                'args': (),
                'options': {'queue': 'data_collection'}
            },
            'process-raw-data': {
                'task': 'data_ingestion.tasks.data_processing_tasks.process_raw_data',
                'schedule': crontab(minute='*/10'),  # Every 10 minutes
                'args': (),
                'options': {'queue': 'data_processing'}
            },
            'cleanup-old-data': {
                'task': 'data_ingestion.tasks.maintenance_tasks.cleanup_old_data',
                'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
                'args': (),
                'options': {'queue': 'maintenance'}
            },
            'backup-database': {
                'task': 'data_ingestion.tasks.maintenance_tasks.backup_database',
                'schedule': crontab(hour=3, minute=0),  # Daily at 3 AM
                'args': (),
                'options': {'queue': 'maintenance'}
            },
            'health-check': {
                'task': 'data_ingestion.tasks.maintenance_tasks.health_check',
                'schedule': crontab(minute='*/15'),  # Every 15 minutes
                'args': (),
                'options': {'queue': 'maintenance'}
            }
        },
        
        # Monitoring and logging
        worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
        worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s',
        
        # Error handling
        task_annotations={
            '*': {
                'rate_limit': '100/m',
                'time_limit': 300,
                'soft_time_limit': 240,
                'retry': True,
                'retry_policy': {
                    'max_retries': 3,
                    'interval_start': 0,
                    'interval_step': 0.2,
                    'interval_max': 0.2,
                }
            }
        }
    )
    
    # Configure logging
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
    
    logger.info("Celery application configured successfully", 
                broker_url=celery_settings.broker_url,
                result_backend=celery_settings.result_backend)
    
    return app

# Create the Celery app instance
celery_app = create_celery_app()

class CeleryTaskManager:
    """
    Manager class for Celery task operations and monitoring.
    """
    
    def __init__(self, app: Celery):
        self.app = app
        self.logger = structlog.get_logger()
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific task.
        
        Args:
            task_id: The task ID to check
            
        Returns:
            Dict containing task status information
        """
        try:
            result = self.app.AsyncResult(task_id)
            return {
                'task_id': task_id,
                'status': result.status,
                'successful': result.successful(),
                'failed': result.failed(),
                'ready': result.ready(),
                'info': result.info if result.ready() else None,
                'traceback': result.traceback if result.failed() else None
            }
        except Exception as e:
            self.logger.error("Failed to get task status", task_id=task_id, error=str(e))
            return {
                'task_id': task_id,
                'status': 'ERROR',
                'error': str(e)
            }
    
    def get_worker_status(self) -> Dict[str, Any]:
        """
        Get the status of all workers.
        
        Returns:
            Dict containing worker status information
        """
        try:
            inspect = self.app.control.inspect()
            
            stats = inspect.stats()
            active = inspect.active()
            registered = inspect.registered()
            revoked = inspect.revoked()
            
            return {
                'workers': {
                    'stats': stats or {},
                    'active': active or {},
                    'registered': registered or {},
                    'revoked': revoked or {}
                },
                'total_workers': len(stats) if stats else 0,
                'active_tasks': sum(len(tasks) for tasks in (active or {}).values()),
                'registered_tasks': sum(len(tasks) for tasks in (registered or {}).values())
            }
        except Exception as e:
            self.logger.error("Failed to get worker status", error=str(e))
            return {
                'error': str(e),
                'workers': {},
                'total_workers': 0,
                'active_tasks': 0,
                'registered_tasks': 0
            }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the status of all queues.
        
        Returns:
            Dict containing queue status information
        """
        try:
            inspect = self.app.control.inspect()
            reserved = inspect.reserved()
            
            queue_stats = {}
            total_reserved = 0
            
            if reserved:
                for worker, tasks in reserved.items():
                    for task in tasks:
                        queue = task.get('delivery_info', {}).get('routing_key', 'default')
                        if queue not in queue_stats:
                            queue_stats[queue] = 0
                        queue_stats[queue] += 1
                        total_reserved += 1
            
            return {
                'queues': queue_stats,
                'total_reserved': total_reserved,
                'queue_count': len(queue_stats)
            }
        except Exception as e:
            self.logger.error("Failed to get queue status", error=str(e))
            return {
                'error': str(e),
                'queues': {},
                'total_reserved': 0,
                'queue_count': 0
            }
    
    def purge_queues(self, queues: Optional[list] = None) -> Dict[str, Any]:
        """
        Purge all tasks from specified queues or all queues.
        
        Args:
            queues: List of queue names to purge, or None for all queues
            
        Returns:
            Dict containing purge results
        """
        try:
            if queues is None:
                # Purge all queues
                result = self.app.control.purge()
                return {
                    'success': True,
                    'message': f'Purged all queues: {result} messages removed',
                    'queues_purged': 'all'
                }
            else:
                # Purge specific queues
                results = {}
                for queue in queues:
                    result = self.app.control.purge(destination=[queue])
                    results[queue] = result
                
                return {
                    'success': True,
                    'message': 'Purged specified queues',
                    'queues_purged': results
                }
        except Exception as e:
            self.logger.error("Failed to purge queues", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def revoke_task(self, task_id: str, terminate: bool = False) -> Dict[str, Any]:
        """
        Revoke a specific task.
        
        Args:
            task_id: The task ID to revoke
            terminate: Whether to terminate the task if it's running
            
        Returns:
            Dict containing revocation result
        """
        try:
            self.app.control.revoke(task_id, terminate=terminate)
            return {
                'success': True,
                'task_id': task_id,
                'terminated': terminate,
                'message': f'Task {task_id} revoked successfully'
            }
        except Exception as e:
            self.logger.error("Failed to revoke task", task_id=task_id, error=str(e))
            return {
                'success': False,
                'task_id': task_id,
                'error': str(e)
            }
    
    def get_scheduled_tasks(self) -> Dict[str, Any]:
        """
        Get all scheduled tasks from the beat schedule.
        
        Returns:
            Dict containing scheduled tasks information
        """
        try:
            schedule = self.app.conf.beat_schedule
            return {
                'scheduled_tasks': schedule,
                'total_scheduled': len(schedule),
                'tasks': [
                    {
                        'name': name,
                        'task': config['task'],
                        'schedule': str(config['schedule']),
                        'args': config.get('args', ()),
                        'options': config.get('options', {})
                    }
                    for name, config in schedule.items()
                ]
            }
        except Exception as e:
            self.logger.error("Failed to get scheduled tasks", error=str(e))
            return {
                'error': str(e),
                'scheduled_tasks': {},
                'total_scheduled': 0,
                'tasks': []
            }

# Create task manager instance
task_manager = CeleryTaskManager(celery_app)

def get_celery_app() -> Celery:
    """Get the configured Celery application instance."""
    return celery_app

def get_task_manager() -> CeleryTaskManager:
    """Get the task manager instance."""
    return task_manager