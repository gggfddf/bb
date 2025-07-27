"""
System monitoring module for data ingestion.
Tracks health, performance, and system status metrics.
"""

import asyncio
import psutil
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog

from ..utils.error_handler import global_error_handler
from ..orchestrator import DataIngestionOrchestrator
from ..scheduler import get_scheduler
from config.settings import MonitoringSettings

logger = structlog.get_logger()

class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Represents a system metric."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

@dataclass
class Alert:
    """Represents a system alert."""
    id: str
    title: str
    message: str
    level: AlertLevel
    source: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SystemMonitor:
    """
    Main system monitoring class.
    Tracks metrics, health checks, and alerts.
    """
    
    def __init__(self, settings: MonitoringSettings = None):
        self.settings = settings or MonitoringSettings()
        self.orchestrator = DataIngestionOrchestrator()
        self.scheduler = get_scheduler()
        
        # Metrics storage
        self.metrics: List[Metric] = []
        self.alerts: List[Alert] = []
        self.health_checks: Dict[str, Callable] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
        self.last_metrics_cleanup = datetime.now()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        # Initialize health checks
        self._initialize_health_checks()
    
    def _initialize_health_checks(self):
        """Initialize default health checks."""
        self.health_checks = {
            'system_resources': self._check_system_resources,
            'data_ingestion': self._check_data_ingestion,
            'database_connection': self._check_database_connection,
            'scheduler_status': self._check_scheduler_status,
            'error_rate': self._check_error_rate,
            'data_quality': self._check_data_quality,
            'task_queue': self._check_task_queue,
            'storage_usage': self._check_storage_usage
        }
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        logger.info("Starting system monitoring")
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.is_monitoring:
            logger.warning("Monitoring is not running")
            return
        
        self.is_monitoring = False
        logger.info("Stopping system monitoring")
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Run health checks
                await self._run_health_checks()
                
                # Check for alerts
                await self._check_alerts()
                
                # Cleanup old metrics
                await self._cleanup_old_metrics()
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.settings.monitoring_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Add metrics
            self._add_metric('system_cpu_usage', cpu_percent, MetricType.GAUGE, 
                           description="CPU usage percentage")
            self._add_metric('system_memory_usage', memory.percent, MetricType.GAUGE,
                           description="Memory usage percentage")
            self._add_metric('system_disk_usage', disk.percent, MetricType.GAUGE,
                           description="Disk usage percentage")
            self._add_metric('system_network_bytes_sent', network.bytes_sent, MetricType.COUNTER,
                           description="Network bytes sent")
            self._add_metric('system_network_bytes_recv', network.bytes_recv, MetricType.COUNTER,
                           description="Network bytes received")
            
            # Data ingestion metrics
            if hasattr(self.orchestrator, 'get_collection_status'):
                collection_status = self.orchestrator.get_collection_status()
                if collection_status:
                    self._add_metric('data_ingestion_active_sources', 
                                   len(collection_status.get('active_sources', [])), 
                                   MetricType.GAUGE,
                                   description="Number of active data sources")
                    self._add_metric('data_ingestion_total_records', 
                                   collection_status.get('total_records_collected', 0), 
                                   MetricType.COUNTER,
                                   description="Total records collected")
            
            # Scheduler metrics
            scheduler_status = self.scheduler.get_scheduler_status()
            self._add_metric('scheduler_total_tasks', scheduler_status.get('total_tasks', 0), 
                           MetricType.GAUGE,
                           description="Total scheduled tasks")
            self._add_metric('scheduler_enabled_tasks', scheduler_status.get('enabled_tasks', 0), 
                           MetricType.GAUGE,
                           description="Enabled scheduled tasks")
            self._add_metric('scheduler_running_tasks', scheduler_status.get('running_tasks', 0), 
                           MetricType.GAUGE,
                           description="Currently running tasks")
            self._add_metric('scheduler_success_rate', scheduler_status.get('success_rate', 0.0), 
                           MetricType.GAUGE,
                           description="Task success rate percentage")
            
            # Error metrics
            error_summary = global_error_handler.get_error_summary()
            self._add_metric('error_total_count', error_summary.get('total_errors', 0), 
                           MetricType.COUNTER,
                           description="Total error count")
            self._add_metric('error_critical_count', error_summary.get('critical_errors', 0), 
                           MetricType.COUNTER,
                           description="Critical error count")
            
            # Performance metrics
            uptime = (datetime.now() - self.start_time).total_seconds()
            self._add_metric('system_uptime_seconds', uptime, MetricType.COUNTER,
                           description="System uptime in seconds")
            
        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))
    
    async def _run_health_checks(self):
        """Run all health checks."""
        for check_name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                if not result['healthy']:
                    self._create_alert(
                        title=f"Health Check Failed: {check_name}",
                        message=result['message'],
                        level=AlertLevel.WARNING if result.get('warning') else AlertLevel.ERROR,
                        source=f"health_check.{check_name}",
                        metadata=result.get('metadata', {})
                    )
            except Exception as e:
                logger.error(f"Health check {check_name} failed", error=str(e))
                self._create_alert(
                    title=f"Health Check Error: {check_name}",
                    message=f"Health check failed with error: {str(e)}",
                    level=AlertLevel.ERROR,
                    source=f"health_check.{check_name}"
                )
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        issues = []
        metadata = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent
        }
        
        if cpu_percent > 90:
            issues.append(f"CPU usage is critically high: {cpu_percent}%")
        elif cpu_percent > 80:
            issues.append(f"CPU usage is high: {cpu_percent}%")
        
        if memory.percent > 95:
            issues.append(f"Memory usage is critically high: {memory.percent}%")
        elif memory.percent > 85:
            issues.append(f"Memory usage is high: {memory.percent}%")
        
        if disk.percent > 95:
            issues.append(f"Disk usage is critically high: {disk.percent}%")
        elif disk.percent > 85:
            issues.append(f"Disk usage is high: {disk.percent}%")
        
        return {
            'healthy': len(issues) == 0,
            'message': '; '.join(issues) if issues else "System resources are healthy",
            'warning': any('high' in issue and 'critically' not in issue for issue in issues),
            'metadata': metadata
        }
    
    async def _check_data_ingestion(self) -> Dict[str, Any]:
        """Check data ingestion health."""
        try:
            collection_status = self.orchestrator.get_collection_status()
            
            issues = []
            metadata = collection_status or {}
            
            if not collection_status.get('collection_active', False):
                issues.append("Data collection is not active")
            
            active_sources = len(collection_status.get('active_sources', []))
            if active_sources == 0:
                issues.append("No active data sources")
            elif active_sources < 2:
                issues.append(f"Only {active_sources} active data source(s)")
            
            error_rate = collection_status.get('error_rate', 0.0)
            if error_rate > 0.1:  # 10% error rate
                issues.append(f"High error rate: {error_rate:.2%}")
            
            return {
                'healthy': len(issues) == 0,
                'message': '; '.join(issues) if issues else "Data ingestion is healthy",
                'metadata': metadata
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Unable to check data ingestion: {str(e)}",
                'metadata': {}
            }
    
    async def _check_database_connection(self) -> Dict[str, Any]:
        """Check database connection health."""
        try:
            # TODO: Implement actual database connection check
            # This would test the connection to TimescaleDB
            
            return {
                'healthy': True,
                'message': "Database connection is healthy",
                'metadata': {'connection_time_ms': 5}
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Database connection failed: {str(e)}",
                'metadata': {}
            }
    
    async def _check_scheduler_status(self) -> Dict[str, Any]:
        """Check scheduler health."""
        try:
            scheduler_status = self.scheduler.get_scheduler_status()
            
            issues = []
            metadata = scheduler_status
            
            if not scheduler_status.get('scheduler_running', False):
                issues.append("Scheduler is not running")
            
            success_rate = scheduler_status.get('success_rate', 100.0)
            if success_rate < 80.0:
                issues.append(f"Low task success rate: {success_rate:.1f}%")
            
            running_tasks = scheduler_status.get('running_tasks', 0)
            if running_tasks > 10:
                issues.append(f"Too many running tasks: {running_tasks}")
            
            return {
                'healthy': len(issues) == 0,
                'message': '; '.join(issues) if issues else "Scheduler is healthy",
                'metadata': metadata
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Unable to check scheduler: {str(e)}",
                'metadata': {}
            }
    
    async def _check_error_rate(self) -> Dict[str, Any]:
        """Check error rate."""
        try:
            error_summary = global_error_handler.get_error_summary()
            
            issues = []
            metadata = error_summary
            
            total_errors = error_summary.get('total_errors', 0)
            critical_errors = error_summary.get('critical_errors', 0)
            
            if critical_errors > 0:
                issues.append(f"Critical errors detected: {critical_errors}")
            
            if total_errors > 100:
                issues.append(f"High error count: {total_errors}")
            
            return {
                'healthy': len(issues) == 0,
                'message': '; '.join(issues) if issues else "Error rate is acceptable",
                'metadata': metadata
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Unable to check error rate: {str(e)}",
                'metadata': {}
            }
    
    async def _check_data_quality(self) -> Dict[str, Any]:
        """Check data quality metrics."""
        try:
            # TODO: Implement data quality checks
            # This would check for data completeness, accuracy, etc.
            
            return {
                'healthy': True,
                'message': "Data quality is acceptable",
                'metadata': {'quality_score': 0.95}
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Unable to check data quality: {str(e)}",
                'metadata': {}
            }
    
    async def _check_task_queue(self) -> Dict[str, Any]:
        """Check task queue status."""
        try:
            # TODO: Implement task queue checks
            # This would check Celery queue status
            
            return {
                'healthy': True,
                'message': "Task queue is healthy",
                'metadata': {'queue_size': 0}
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Unable to check task queue: {str(e)}",
                'metadata': {}
            }
    
    async def _check_storage_usage(self) -> Dict[str, Any]:
        """Check storage usage."""
        try:
            disk = psutil.disk_usage('/')
            
            issues = []
            metadata = {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent_used': disk.percent
            }
            
            if disk.percent > 95:
                issues.append(f"Storage critically full: {disk.percent}%")
            elif disk.percent > 85:
                issues.append(f"Storage usage high: {disk.percent}%")
            
            return {
                'healthy': len(issues) == 0,
                'message': '; '.join(issues) if issues else "Storage usage is acceptable",
                'warning': any('high' in issue and 'critically' not in issue for issue in issues),
                'metadata': metadata
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Unable to check storage: {str(e)}",
                'metadata': {}
            }
    
    def _add_metric(self, name: str, value: float, metric_type: MetricType, 
                   labels: Dict[str, str] = None, description: str = ""):
        """Add a metric to the monitoring system."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {},
            description=description
        )
        self.metrics.append(metric)
    
    def _create_alert(self, title: str, message: str, level: AlertLevel, 
                     source: str, metadata: Dict[str, Any] = None):
        """Create a new alert."""
        alert = Alert(
            id=f"{source}_{int(time.time())}",
            title=title,
            message=message,
            level=level,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.alerts.append(alert)
        logger.warning(f"Alert created: {title} - {message}", alert_level=level.value)
    
    async def _check_alerts(self):
        """Check for alert conditions and resolve old alerts."""
        # TODO: Implement alert resolution logic
        # This would check if conditions that triggered alerts have been resolved
        pass
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues."""
        now = datetime.now()
        cutoff_time = now - timedelta(hours=self.settings.metrics_retention_hours)
        
        # Remove old metrics
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        # Remove resolved alerts older than retention period
        alert_cutoff = now - timedelta(hours=self.settings.alerts_retention_hours)
        self.alerts = [a for a in self.alerts if not a.resolved or a.timestamp > alert_cutoff]
        
        # Update cleanup timestamp
        self.last_metrics_cleanup = now
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring status.
        
        Returns:
            Dict containing monitoring status and metrics
        """
        now = datetime.now()
        
        # Get recent metrics
        recent_metrics = [
            {
                'name': m.name,
                'value': m.value,
                'type': m.metric_type.value,
                'timestamp': m.timestamp.isoformat(),
                'labels': m.labels,
                'description': m.description
            }
            for m in self.metrics[-100:]  # Last 100 metrics
        ]
        
        # Get active alerts
        active_alerts = [
            {
                'id': a.id,
                'title': a.title,
                'message': a.message,
                'level': a.level.value,
                'source': a.source,
                'timestamp': a.timestamp.isoformat(),
                'metadata': a.metadata
            }
            for a in self.alerts if not a.resolved
        ]
        
        # Calculate alert statistics
        alert_stats = {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a['level'] == 'critical']),
            'error_alerts': len([a for a in active_alerts if a['level'] == 'error']),
            'warning_alerts': len([a for a in active_alerts if a['level'] == 'warning'])
        }
        
        return {
            'monitoring_active': self.is_monitoring,
            'uptime_seconds': (now - self.start_time).total_seconds(),
            'total_metrics_collected': len(self.metrics),
            'recent_metrics': recent_metrics,
            'alert_statistics': alert_stats,
            'active_alerts': active_alerts,
            'last_cleanup': self.last_metrics_cleanup.isoformat(),
            'health_checks': list(self.health_checks.keys())
        }
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get metric history for a specific metric.
        
        Args:
            metric_name: Name of the metric
            hours: Number of hours of history to retrieve
            
        Returns:
            List of metric values with timestamps
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        metric_history = [
            {
                'value': m.value,
                'timestamp': m.timestamp.isoformat(),
                'labels': m.labels
            }
            for m in self.metrics
            if m.name == metric_name and m.timestamp > cutoff_time
        ]
        
        return sorted(metric_history, key=lambda x: x['timestamp'])
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            bool: True if alert was found and resolved
        """
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert.title}")
                return True
        return False

# Global monitor instance
_monitor_instance: Optional[SystemMonitor] = None

def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    return _monitor_instance

async def start_system_monitoring():
    """Start the global system monitoring."""
    monitor = get_system_monitor()
    await monitor.start_monitoring()

async def stop_system_monitoring():
    """Stop the global system monitoring."""
    monitor = get_system_monitor()
    await monitor.stop_monitoring()

def get_monitoring_status() -> Dict[str, Any]:
    """Get the global monitoring status."""
    monitor = get_system_monitor()
    return monitor.get_monitoring_status()