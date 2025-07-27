"""
Monitoring System for Trading System

This module provides comprehensive monitoring functionality for the trading system,
tracking system performance, health, and various metrics.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import sqlite3
from pathlib import Path
import queue
import hashlib

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class Metric:
    """Metric structure"""
    metric_id: str
    name: str
    metric_type: MetricType
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str]
    description: Optional[str]

@dataclass
class HealthCheck:
    """Health check structure"""
    check_id: str
    component: str
    status: HealthStatus
    timestamp: datetime
    message: str
    details: Dict[str, Any]
    response_time: Optional[float]

@dataclass
class Alert:
    """Alert structure"""
    alert_id: str
    alert_type: str
    severity: str
    component: str
    message: str
    timestamp: datetime
    acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    resolved: bool
    resolved_at: Optional[datetime]

class SystemMonitor:
    """
    System monitoring for tracking performance, health, and metrics.
    """
    
    def __init__(self, db_path: str = "monitoring/monitor.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Monitoring components
        self.metrics: Dict[str, Metric] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alerts: Dict[str, Alert] = {}
        
        # Monitoring threads
        self.monitoring_thread = None
        self.health_check_thread = None
        self.alert_thread = None
        self.running = False
        
        # Configuration
        self.monitoring_interval = 30  # seconds
        self.health_check_interval = 60  # seconds
        self.alert_check_interval = 30  # seconds
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.metric_callbacks: List[Callable] = []
        
        # Initialize database
        self._init_database()
        
        # Start monitoring
        self.start()
        
        logger.info("System monitor initialized")
    
    def start(self):
        """Start the monitoring system"""
        if not self.running:
            self.running = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
            self.monitoring_thread.start()
            
            # Start health check thread
            self.health_check_thread = threading.Thread(target=self._health_check_worker, daemon=True)
            self.health_check_thread.start()
            
            # Start alert thread
            self.alert_thread = threading.Thread(target=self._alert_worker, daemon=True)
            self.alert_thread.start()
            
            logger.info("System monitor started")
    
    def stop(self):
        """Stop the monitoring system"""
        if self.running:
            self.running = False
            
            # Wait for threads to finish
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            if self.health_check_thread:
                self.health_check_thread.join(timeout=5)
            if self.alert_thread:
                self.alert_thread.join(timeout=5)
            
            logger.info("System monitor stopped")
    
    def record_metric(self, name: str, value: Union[int, float], metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None, description: Optional[str] = None):
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Metric labels
            description: Metric description
        """
        try:
            metric_id = self._generate_metric_id(name, labels)
            
            metric = Metric(
                metric_id=metric_id,
                name=name,
                metric_type=metric_type,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {},
                description=description
            )
            
            self.metrics[metric_id] = metric
            self._store_metric(metric)
            
            # Trigger callbacks
            for callback in self.metric_callbacks:
                try:
                    callback(metric)
                except Exception as e:
                    logger.error(f"Error in metric callback: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """
        Get a metric by name and labels.
        
        Args:
            name: Metric name
            labels: Metric labels
            
        Returns:
            Metric if found
        """
        metric_id = self._generate_metric_id(name, labels)
        return self.metrics.get(metric_id)
    
    def get_metrics(self, name: Optional[str] = None, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None, limit: int = 1000) -> List[Metric]:
        """
        Get metrics with filters.
        
        Args:
            name: Metric name filter
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of metrics to return
            
        Returns:
            List of metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM metrics WHERE 1=1"
                params = []
                
                if name:
                    query += " AND name = ?"
                    params.append(name)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_metric(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []
    
    def record_health_check(self, component: str, status: HealthStatus, message: str,
                          details: Optional[Dict[str, Any]] = None, response_time: Optional[float] = None):
        """
        Record a health check.
        
        Args:
            component: Component name
            status: Health status
            message: Health check message
            details: Additional details
            response_time: Response time in seconds
        """
        try:
            check_id = self._generate_health_check_id(component)
            
            health_check = HealthCheck(
                check_id=check_id,
                component=component,
                status=status,
                timestamp=datetime.utcnow(),
                message=message,
                details=details or {},
                response_time=response_time
            )
            
            self.health_checks[check_id] = health_check
            self._store_health_check(health_check)
            
        except Exception as e:
            logger.error(f"Failed to record health check for {component}: {e}")
    
    def get_health_status(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health status.
        
        Args:
            component: Component name (optional)
            
        Returns:
            Health status information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if component:
                    query = "SELECT * FROM health_checks WHERE component = ? ORDER BY timestamp DESC LIMIT 1"
                    row = conn.execute(query, (component,)).fetchone()
                    if row:
                        return asdict(self._row_to_health_check(row))
                    return {}
                else:
                    # Get latest health check for each component
                    query = """
                        SELECT * FROM health_checks h1
                        WHERE timestamp = (
                            SELECT MAX(timestamp) FROM health_checks h2
                            WHERE h2.component = h1.component
                        )
                    """
                    rows = conn.execute(query).fetchall()
                    
                    return {
                        row['component']: asdict(self._row_to_health_check(row))
                        for row in rows
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {}
    
    def create_alert(self, alert_type: str, severity: str, component: str, message: str):
        """
        Create an alert.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            component: Component name
            message: Alert message
        """
        try:
            alert_id = self._generate_alert_id()
            
            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                component=component,
                message=message,
                timestamp=datetime.utcnow(),
                acknowledged=False,
                acknowledged_by=None,
                acknowledged_at=None,
                resolved=False,
                resolved_at=None
            )
            
            self.alerts[alert_id] = alert
            self._store_alert(alert)
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if alert was acknowledged
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    UPDATE alerts 
                    SET acknowledged = ?, acknowledged_by = ?, acknowledged_at = ?
                    WHERE alert_id = ?
                """, (True, acknowledged_by, datetime.utcnow().isoformat(), alert_id))
                
                conn.commit()
                
                if result.rowcount > 0:
                    # Update local cache
                    if alert_id in self.alerts:
                        self.alerts[alert_id].acknowledged = True
                        self.alerts[alert_id].acknowledged_by = acknowledged_by
                        self.alerts[alert_id].acknowledged_at = datetime.utcnow()
                    
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if alert was resolved
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    UPDATE alerts 
                    SET resolved = ?, resolved_at = ?
                    WHERE alert_id = ?
                """, (True, datetime.utcnow().isoformat(), alert_id))
                
                conn.commit()
                
                if result.rowcount > 0:
                    # Update local cache
                    if alert_id in self.alerts:
                        self.alerts[alert_id].resolved = True
                        self.alerts[alert_id].resolved_at = datetime.utcnow()
                    
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    def get_alerts(self, severity: Optional[str] = None, component: Optional[str] = None,
                  resolved: Optional[bool] = None, limit: int = 100) -> List[Alert]:
        """
        Get alerts with filters.
        
        Args:
            severity: Severity filter
            component: Component filter
            resolved: Resolved filter
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM alerts WHERE 1=1"
                params = []
                
                if severity:
                    query += " AND severity = ?"
                    params.append(severity)
                
                if component:
                    query += " AND component = ?"
                    params.append(component)
                
                if resolved is not None:
                    query += " AND resolved = ?"
                    params.append(resolved)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_alert(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    def add_alert_callback(self, callback: Callable):
        """
        Add an alert callback.
        
        Args:
            callback: Callback function
        """
        self.alert_callbacks.append(callback)
    
    def add_metric_callback(self, callback: Callable):
        """
        Add a metric callback.
        
        Args:
            callback: Callback function
        """
        self.metric_callbacks.append(callback)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            System statistics
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'memory_total': memory.total,
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free': disk.free,
                'disk_total': disk.total,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_count': process_count,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def _init_database(self):
        """Initialize the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        metric_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        labels TEXT,
                        description TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Health checks table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS health_checks (
                        check_id TEXT PRIMARY KEY,
                        component TEXT NOT NULL,
                        status TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        response_time REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Alerts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        acknowledged_by TEXT,
                        acknowledged_at TEXT,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_health_checks_timestamp ON health_checks(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_health_checks_component ON health_checks(component)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_component ON alerts(component)")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _monitoring_worker(self):
        """Monitoring worker thread"""
        while self.running:
            try:
                # Record system metrics
                self._record_system_metrics()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")
                time.sleep(5)
    
    def _health_check_worker(self):
        """Health check worker thread"""
        while self.running:
            try:
                # Perform health checks
                self._perform_health_checks()
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
                time.sleep(5)
    
    def _alert_worker(self):
        """Alert worker thread"""
        while self.running:
            try:
                # Check for alert conditions
                self._check_alert_conditions()
                
                time.sleep(self.alert_check_interval)
                
            except Exception as e:
                logger.error(f"Error in alert worker: {e}")
                time.sleep(5)
    
    def _record_system_metrics(self):
        """Record system metrics"""
        try:
            stats = self.get_system_stats()
            
            # Record CPU usage
            self.record_metric("cpu_usage", stats.get('cpu_percent', 0), MetricType.GAUGE, 
                             {"component": "system"}, "CPU usage percentage")
            
            # Record memory usage
            self.record_metric("memory_usage", stats.get('memory_percent', 0), MetricType.GAUGE,
                             {"component": "system"}, "Memory usage percentage")
            
            # Record disk usage
            self.record_metric("disk_usage", stats.get('disk_percent', 0), MetricType.GAUGE,
                             {"component": "system"}, "Disk usage percentage")
            
            # Record process count
            self.record_metric("process_count", stats.get('process_count', 0), MetricType.GAUGE,
                             {"component": "system"}, "Number of running processes")
            
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
    
    def _perform_health_checks(self):
        """Perform health checks"""
        try:
            # Check system health
            stats = self.get_system_stats()
            
            # CPU health check
            cpu_percent = stats.get('cpu_percent', 0)
            if cpu_percent > 90:
                self.record_health_check("system", HealthStatus.CRITICAL, 
                                       f"High CPU usage: {cpu_percent}%", {"cpu_percent": cpu_percent})
            elif cpu_percent > 80:
                self.record_health_check("system", HealthStatus.WARNING, 
                                       f"Elevated CPU usage: {cpu_percent}%", {"cpu_percent": cpu_percent})
            else:
                self.record_health_check("system", HealthStatus.HEALTHY, 
                                       f"Normal CPU usage: {cpu_percent}%", {"cpu_percent": cpu_percent})
            
            # Memory health check
            memory_percent = stats.get('memory_percent', 0)
            if memory_percent > 90:
                self.record_health_check("memory", HealthStatus.CRITICAL, 
                                       f"High memory usage: {memory_percent}%", {"memory_percent": memory_percent})
            elif memory_percent > 80:
                self.record_health_check("memory", HealthStatus.WARNING, 
                                       f"Elevated memory usage: {memory_percent}%", {"memory_percent": memory_percent})
            else:
                self.record_health_check("memory", HealthStatus.HEALTHY, 
                                       f"Normal memory usage: {memory_percent}%", {"memory_percent": memory_percent})
            
            # Disk health check
            disk_percent = stats.get('disk_percent', 0)
            if disk_percent > 90:
                self.record_health_check("disk", HealthStatus.CRITICAL, 
                                       f"High disk usage: {disk_percent}%", {"disk_percent": disk_percent})
            elif disk_percent > 80:
                self.record_health_check("disk", HealthStatus.WARNING, 
                                       f"Elevated disk usage: {disk_percent}%", {"disk_percent": disk_percent})
            else:
                self.record_health_check("disk", HealthStatus.HEALTHY, 
                                       f"Normal disk usage: {disk_percent}%", {"disk_percent": disk_percent})
            
        except Exception as e:
            logger.error(f"Failed to perform health checks: {e}")
    
    def _check_alert_conditions(self):
        """Check for alert conditions"""
        try:
            stats = self.get_system_stats()
            
            # CPU alert
            cpu_percent = stats.get('cpu_percent', 0)
            if cpu_percent > 95:
                self.create_alert("high_cpu", "critical", "system", f"Critical CPU usage: {cpu_percent}%")
            elif cpu_percent > 85:
                self.create_alert("high_cpu", "warning", "system", f"High CPU usage: {cpu_percent}%")
            
            # Memory alert
            memory_percent = stats.get('memory_percent', 0)
            if memory_percent > 95:
                self.create_alert("high_memory", "critical", "system", f"Critical memory usage: {memory_percent}%")
            elif memory_percent > 85:
                self.create_alert("high_memory", "warning", "system", f"High memory usage: {memory_percent}%")
            
            # Disk alert
            disk_percent = stats.get('disk_percent', 0)
            if disk_percent > 95:
                self.create_alert("high_disk", "critical", "system", f"Critical disk usage: {disk_percent}%")
            elif disk_percent > 85:
                self.create_alert("high_disk", "warning", "system", f"High disk usage: {disk_percent}%")
            
        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")
    
    def _store_metric(self, metric: Metric):
        """Store metric in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics (
                        metric_id, name, metric_type, value, timestamp, labels, description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_id,
                    metric.name,
                    metric.metric_type.value,
                    metric.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.labels),
                    metric.description
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
    
    def _store_health_check(self, health_check: HealthCheck):
        """Store health check in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO health_checks (
                        check_id, component, status, timestamp, message, details, response_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    health_check.check_id,
                    health_check.component,
                    health_check.status.value,
                    health_check.timestamp.isoformat(),
                    health_check.message,
                    json.dumps(health_check.details),
                    health_check.response_time
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store health check: {e}")
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts (
                        alert_id, alert_type, severity, component, message, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.alert_type,
                    alert.severity,
                    alert.component,
                    alert.message,
                    alert.timestamp.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def _row_to_metric(self, row) -> Metric:
        """Convert database row to metric"""
        return Metric(
            metric_id=row['metric_id'],
            name=row['name'],
            metric_type=MetricType(row['metric_type']),
            value=row['value'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            labels=json.loads(row['labels']) if row['labels'] else {},
            description=row['description']
        )
    
    def _row_to_health_check(self, row) -> HealthCheck:
        """Convert database row to health check"""
        return HealthCheck(
            check_id=row['check_id'],
            component=row['component'],
            status=HealthStatus(row['status']),
            timestamp=datetime.fromisoformat(row['timestamp']),
            message=row['message'],
            details=json.loads(row['details']) if row['details'] else {},
            response_time=row['response_time']
        )
    
    def _row_to_alert(self, row) -> Alert:
        """Convert database row to alert"""
        return Alert(
            alert_id=row['alert_id'],
            alert_type=row['alert_type'],
            severity=row['severity'],
            component=row['component'],
            message=row['message'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            acknowledged=bool(row['acknowledged']),
            acknowledged_by=row['acknowledged_by'],
            acknowledged_at=datetime.fromisoformat(row['acknowledged_at']) if row['acknowledged_at'] else None,
            resolved=bool(row['resolved']),
            resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None
        )
    
    def _generate_metric_id(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate metric ID"""
        label_str = json.dumps(labels, sort_keys=True) if labels else ""
        return hashlib.md5(f"{name}_{label_str}".encode()).hexdigest()
    
    def _generate_health_check_id(self, component: str) -> str:
        """Generate health check ID"""
        return hashlib.md5(f"health_{component}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
    
    def _generate_alert_id(self) -> str:
        """Generate alert ID"""
        return hashlib.md5(f"alert_{datetime.utcnow().isoformat()}".encode()).hexdigest()


# Global instance management
_system_monitor_instance = None

def get_system_monitor(db_path: str = "monitoring/monitor.db") -> SystemMonitor:
    """Get or create system monitor instance"""
    global _system_monitor_instance
    
    if _system_monitor_instance is None:
        _system_monitor_instance = SystemMonitor(db_path)
    
    return _system_monitor_instance


def init_system_monitor(db_path: str = "monitoring/monitor.db") -> SystemMonitor:
    """Initialize system monitor with custom database path"""
    global _system_monitor_instance
    
    if _system_monitor_instance:
        _system_monitor_instance.stop()
    
    _system_monitor_instance = SystemMonitor(db_path)
    
    return _system_monitor_instance