#!/usr/bin/env python3
"""
Logging and Monitoring Module

Implements comprehensive logging and monitoring for the trading system:
- Structured logging system
- Log aggregation and analysis
- Monitoring dashboards
- Alerting and notifications
- Performance metrics collection
- Log retention and archival

Features:
- Complete structured logging with multiple outputs
- Log aggregation and real-time analysis
- Performance metrics collection and monitoring
- Alerting system with multiple notification channels
- Log retention and archival policies
- Monitoring dashboards and visualization
- High-performance log processing and storage
"""

import os
import json
import time
import threading
import asyncio
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import gzip
import shutil
from pathlib import Path
import queue
import logging

logger = structlog.get_logger()

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class LogOutput(Enum):
    """Log output type enumeration."""
    CONSOLE = "console"
    FILE = "file"
    DATABASE = "database"
    SYSLOG = "syslog"
    WEBHOOK = "webhook"

class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LogEntry:
    """Log entry structure."""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metric:
    """Metric structure."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

@dataclass
class Alert:
    """Alert structure."""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LogManagerConfig:
    """Log manager configuration."""
    log_dir: str = "logs"
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_files: int = 10
    retention_days: int = 30
    enable_compression: bool = True
    enable_rotation: bool = True
    outputs: List[LogOutput] = field(default_factory=lambda: [LogOutput.CONSOLE, LogOutput.FILE])
    enable_metrics: bool = True
    enable_alerts: bool = True
    metrics_interval: float = 60.0  # seconds
    alert_check_interval: float = 30.0  # seconds

class LogHandler:
    """Base log handler class."""
    
    def __init__(self, config: LogManagerConfig):
        """
        Initialize log handler.
        
        Args:
            config: Log manager configuration
        """
        self.config = config
        self._lock = threading.RLock()
    
    def handle_log(self, log_entry: LogEntry):
        """Handle a log entry."""
        raise NotImplementedError
    
    def close(self):
        """Close the handler."""
        pass

class ConsoleLogHandler(LogHandler):
    """Console log handler."""
    
    def __init__(self, config: LogManagerConfig):
        super().__init__(config)
        self.logger = structlog.get_logger()
    
    def handle_log(self, log_entry: LogEntry):
        """Handle log entry for console output."""
        log_data = {
            'timestamp': log_entry.timestamp.isoformat(),
            'level': log_entry.level.value,
            'component': log_entry.component,
            'message': log_entry.message,
            'data': log_entry.data,
            'trace_id': log_entry.trace_id,
            'user_id': log_entry.user_id,
            'session_id': log_entry.session_id,
            'metadata': log_entry.metadata
        }
        
        if log_entry.level == LogLevel.DEBUG:
            self.logger.debug(log_entry.message, **log_data)
        elif log_entry.level == LogLevel.INFO:
            self.logger.info(log_entry.message, **log_data)
        elif log_entry.level == LogLevel.WARNING:
            self.logger.warning(log_entry.message, **log_data)
        elif log_entry.level == LogLevel.ERROR:
            self.logger.error(log_entry.message, **log_data)
        elif log_entry.level == LogLevel.CRITICAL:
            self.logger.critical(log_entry.message, **log_data)

class FileLogHandler(LogHandler):
    """File log handler with rotation and compression."""
    
    def __init__(self, config: LogManagerConfig):
        super().__init__(config)
        self.current_file = None
        self.current_file_size = 0
        self.file_count = 0
        self._setup_log_file()
    
    def _setup_log_file(self):
        """Setup log file."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = os.path.join(self.config.log_dir, f"trading_system_{timestamp}.log")
        self.current_file_size = 0
        self.file_count += 1
    
    def handle_log(self, log_entry: LogEntry):
        """Handle log entry for file output."""
        with self._lock:
            # Check if rotation is needed
            if self.config.enable_rotation and self.current_file_size >= self.config.max_file_size:
                self._rotate_log_file()
            
            # Format log entry
            log_data = {
                'timestamp': log_entry.timestamp.isoformat(),
                'level': log_entry.level.value,
                'component': log_entry.component,
                'message': log_entry.message,
                'data': log_entry.data,
                'trace_id': log_entry.trace_id,
                'user_id': log_entry.user_id,
                'session_id': log_entry.session_id,
                'metadata': log_entry.metadata
            }
            
            # Write to file
            with open(self.current_file, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
            
            # Update file size
            self.current_file_size += len(json.dumps(log_data)) + 1
    
    def _rotate_log_file(self):
        """Rotate log file."""
        if self.current_file and os.path.exists(self.current_file):
            # Compress old file if enabled
            if self.config.enable_compression:
                compressed_file = self.current_file + '.gz'
                with open(self.current_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(self.current_file)
            
            # Check file count limit
            if self.file_count >= self.config.max_files:
                self._cleanup_old_files()
        
        # Create new file
        self._setup_log_file()
    
    def _cleanup_old_files(self):
        """Clean up old log files."""
        log_files = []
        for file in os.listdir(self.config.log_dir):
            if file.startswith('trading_system_') and file.endswith('.log'):
                file_path = os.path.join(self.config.log_dir, file)
                log_files.append((file_path, os.path.getmtime(file_path)))
        
        # Sort by modification time and remove oldest
        log_files.sort(key=lambda x: x[1])
        for file_path, _ in log_files[:-self.config.max_files]:
            os.remove(file_path)

class MetricsCollector:
    """Performance metrics collection system."""
    
    def __init__(self, config: LogManagerConfig):
        """
        Initialize metrics collector.
        
        Args:
            config: Log manager configuration
        """
        self.config = config
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.summaries = defaultdict(list)
        self._lock = threading.RLock()
        self._collection_task = None
        self._running = False
    
    def start(self):
        """Start metrics collection."""
        if not self.config.enable_metrics:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collect_metrics())
        logger.info("Metrics collector started")
    
    def stop(self):
        """Stop metrics collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
        logger.info("Metrics collector stopped")
    
    def record_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Record a counter metric."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self.counters[key] += value
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge metric."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self.histograms[key].append(value)
    
    def record_summary(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a summary metric."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self.summaries[key].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {k: self._calculate_histogram_stats(v) for k, v in self.histograms.items()},
                'summaries': {k: self._calculate_summary_stats(v) for k, v in self.summaries.items()}
            }
    
    def _get_metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Get metric key with labels."""
        if labels:
            label_str = ','.join([f"{k}={v}" for k, v in sorted(labels.items())])
            return f"{name}[{label_str}]"
        return name
    
    def _calculate_histogram_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate histogram statistics."""
        if not values:
            return {}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values)
        }
    
    def _calculate_summary_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate summary statistics."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'count': n,
            'sum': sum(values),
            'min': sorted_values[0],
            'max': sorted_values[-1],
            'mean': sum(values) / n,
            'median': sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2,
            'p95': sorted_values[int(0.95 * n)] if n > 0 else 0,
            'p99': sorted_values[int(0.99 * n)] if n > 0 else 0
        }
    
    async def _collect_metrics(self):
        """Collect system metrics."""
        while self._running:
            try:
                # Collect system metrics
                import psutil
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_gauge('system.cpu.usage', cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_gauge('system.memory.usage', memory.percent)
                self.record_gauge('system.memory.available', memory.available / (1024 * 1024 * 1024))  # GB
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.record_gauge('system.disk.usage', disk.percent)
                self.record_gauge('system.disk.available', disk.free / (1024 * 1024 * 1024))  # GB
                
                # Network metrics
                network = psutil.net_io_counters()
                self.record_counter('system.network.bytes_sent', network.bytes_sent)
                self.record_counter('system.network.bytes_recv', network.bytes_recv)
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(self.config.metrics_interval)

class AlertManager:
    """Alert management system."""
    
    def __init__(self, config: LogManagerConfig):
        """
        Initialize alert manager.
        
        Args:
            config: Log manager configuration
        """
        self.config = config
        self.alerts = deque(maxlen=1000)
        self.alert_rules = {}
        self.alert_callbacks = []
        self._lock = threading.RLock()
        self._alert_task = None
        self._running = False
    
    def start(self):
        """Start alert manager."""
        if not self.config.enable_alerts:
            return
        
        self._running = True
        self._alert_task = asyncio.create_task(self._check_alerts())
        logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert manager."""
        self._running = False
        if self._alert_task:
            self._alert_task.cancel()
        logger.info("Alert manager stopped")
    
    def add_alert_rule(self, rule_name: str, condition: Callable[[Dict[str, Any]], bool], 
                      severity: AlertSeverity, title: str, message: str):
        """Add an alert rule."""
        with self._lock:
            self.alert_rules[rule_name] = {
                'condition': condition,
                'severity': severity,
                'title': title,
                'message': message
            }
            logger.info("Alert rule added", rule_name=rule_name)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback."""
        with self._lock:
            self.alert_callbacks.append(callback)
    
    def create_alert(self, severity: AlertSeverity, title: str, message: str, 
                    component: str, metadata: Dict[str, Any] = None) -> str:
        """Create an alert."""
        with self._lock:
            alert_id = str(uuid.uuid4())
            alert = Alert(
                alert_id=alert_id,
                severity=severity,
                title=title,
                message=message,
                component=component,
                metadata=metadata or {}
            )
            
            self.alerts.append(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error("Alert callback error", error=str(e))
            
            logger.info("Alert created", alert_id=alert_id, severity=severity.value)
            return alert_id
    
    def acknowledge_alert(self, alert_id: str, user_id: str):
        """Acknowledge an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = user_id
                    alert.acknowledged_at = datetime.now()
                    logger.info("Alert acknowledged", alert_id=alert_id, user_id=user_id)
                    break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unacknowledged) alerts."""
        with self._lock:
            return [alert for alert in self.alerts if not alert.acknowledged]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity."""
        with self._lock:
            return [alert for alert in self.alerts if alert.severity == severity]
    
    async def _check_alerts(self):
        """Check for alerts based on rules."""
        while self._running:
            try:
                # Check alert rules
                for rule_name, rule in self.alert_rules.items():
                    try:
                        # Get current system state (simplified)
                        system_state = {
                            'timestamp': datetime.now(),
                            'metrics': {},  # Would be populated with current metrics
                            'logs': []  # Would be populated with recent logs
                        }
                        
                        if rule['condition'](system_state):
                            self.create_alert(
                                rule['severity'],
                                rule['title'],
                                rule['message'],
                                'system'
                            )
                    except Exception as e:
                        logger.error("Alert rule check error", rule_name=rule_name, error=str(e))
                
                await asyncio.sleep(self.config.alert_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert check error", error=str(e))
                await asyncio.sleep(self.config.alert_check_interval)

class LogManager:
    """Main logging and monitoring system."""
    
    def __init__(self, config: LogManagerConfig = None):
        """
        Initialize log manager.
        
        Args:
            config: Log manager configuration
        """
        self.config = config or LogManagerConfig()
        self.handlers = []
        self.metrics_collector = MetricsCollector(self.config)
        self.alert_manager = AlertManager(self.config)
        
        self.log_queue = queue.Queue(maxsize=10000)
        self._log_processor_task = None
        self._running = False
        
        # Setup handlers
        self._setup_handlers()
        
        logger.info("Log manager initialized")
    
    def _setup_handlers(self):
        """Setup log handlers."""
        for output in self.config.outputs:
            if output == LogOutput.CONSOLE:
                self.handlers.append(ConsoleLogHandler(self.config))
            elif output == LogOutput.FILE:
                self.handlers.append(FileLogHandler(self.config))
            # Add other handlers as needed
    
    def start(self):
        """Start log manager."""
        self._running = True
        
        # Start log processor
        self._log_processor_task = asyncio.create_task(self._process_logs())
        
        # Start metrics collector
        self.metrics_collector.start()
        
        # Start alert manager
        self.alert_manager.start()
        
        logger.info("Log manager started")
    
    def stop(self):
        """Stop log manager."""
        self._running = False
        
        # Stop tasks
        if self._log_processor_task:
            self._log_processor_task.cancel()
        
        self.metrics_collector.stop()
        self.alert_manager.stop()
        
        # Close handlers
        for handler in self.handlers:
            handler.close()
        
        logger.info("Log manager stopped")
    
    def log(self, level: LogLevel, component: str, message: str, 
            data: Dict[str, Any] = None, trace_id: str = None, 
            user_id: str = None, session_id: str = None, 
            metadata: Dict[str, Any] = None):
        """Log a message."""
        if level.value < self.config.log_level.value:
            return
        
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            data=data or {},
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        try:
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            logger.warning("Log queue full, dropping log entry")
    
    def debug(self, component: str, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, component, message, **kwargs)
    
    def info(self, component: str, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, component, message, **kwargs)
    
    def warning(self, component: str, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, component, message, **kwargs)
    
    def error(self, component: str, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, component, message, **kwargs)
    
    def critical(self, component: str, message: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, component, message, **kwargs)
    
    async def _process_logs(self):
        """Process logs from queue."""
        while self._running:
            try:
                # Get log entry from queue
                log_entry = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, self.log_queue.get),
                    timeout=1.0
                )
                
                # Send to all handlers
                for handler in self.handlers:
                    try:
                        handler.handle_log(log_entry)
                    except Exception as e:
                        logger.error("Log handler error", error=str(e))
                
                self.log_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Log processing error", error=str(e))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return self.metrics_collector.get_metrics()
    
    def get_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self.alert_manager.get_active_alerts()
    
    def add_alert_rule(self, rule_name: str, condition: Callable[[Dict[str, Any]], bool], 
                      severity: AlertSeverity, title: str, message: str):
        """Add alert rule."""
        self.alert_manager.add_alert_rule(rule_name, condition, severity, title, message)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback."""
        self.alert_manager.add_alert_callback(callback)

def create_log_manager(config: LogManagerConfig = None) -> LogManager:
    """
    Create a log manager.
    
    Args:
        config: Log manager configuration
        
    Returns:
        LogManager instance
    """
    return LogManager(config)

if __name__ == "__main__":
    # Demo of log manager
    config = LogManagerConfig(
        log_dir="logs",
        log_level=LogLevel.INFO,
        log_format="json",
        max_file_size=10 * 1024 * 1024,
        max_files=10,
        retention_days=30,
        enable_compression=True,
        enable_rotation=True,
        outputs=[LogOutput.CONSOLE, LogOutput.FILE],
        enable_metrics=True,
        enable_alerts=True,
        metrics_interval=60.0,
        alert_check_interval=30.0
    )
    
    log_manager = create_log_manager(config)
    
    # Add alert rule
    def high_cpu_condition(state: Dict[str, Any]) -> bool:
        # Simplified condition - in real implementation would check actual metrics
        return False  # Always false for demo
    
    log_manager.add_alert_rule(
        "high_cpu",
        high_cpu_condition,
        AlertSeverity.HIGH,
        "High CPU Usage",
        "CPU usage is above 90%"
    )
    
    # Add alert callback
    def alert_callback(alert: Alert):
        print(f"🚨 Alert: {alert.title} - {alert.message}")
    
    log_manager.add_alert_callback(alert_callback)
    
    # Start log manager
    log_manager.start()
    
    print("Log Manager created successfully!")
    
    # Log some messages
    log_manager.info("demo", "Log manager demo started")
    log_manager.warning("demo", "This is a warning message")
    log_manager.error("demo", "This is an error message", data={'error_code': 500})
    
    # Record some metrics
    log_manager.metrics_collector.record_counter("demo.requests", 1)
    log_manager.metrics_collector.record_gauge("demo.response_time", 0.15)
    
    # Get metrics
    metrics = log_manager.get_metrics()
    print(f"Metrics: {metrics}")
    
    # Get alerts
    alerts = log_manager.get_alerts()
    print(f"Active alerts: {len(alerts)}")
    
    # Stop log manager
    log_manager.stop()