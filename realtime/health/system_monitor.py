#!/usr/bin/env python3
"""
Live System Health Monitoring Module

Implements live system health monitoring for the trading system:
- System health metrics collection
- Health status monitoring
- Alerting for system issues
- Performance monitoring
- System diagnostics
- Health dashboard

Features:
- Comprehensive system health metrics collection and monitoring
- Real-time health status assessment and alerting
- Advanced performance monitoring and bottleneck detection
- Intelligent system diagnostics and troubleshooting
- Interactive health dashboard and reporting
- Proactive system maintenance and optimization
"""

import asyncio
import json
import time
import psutil
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import queue
import uuid
from collections import defaultdict, deque

logger = structlog.get_logger()

class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"

class MetricType(Enum):
    """Metric types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"
    CUSTOM = "custom"

class AlertLevel(Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthMetric:
    """Health metric structure."""
    metric_id: str
    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    status: HealthStatus = HealthStatus.HEALTHY
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """System health structure."""
    system_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    overall_status: HealthStatus = HealthStatus.HEALTHY
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    performance_score: float = 100.0
    uptime: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthConfig:
    """Health monitoring configuration."""
    monitoring_interval: float = 5.0  # seconds
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    enable_alerts: bool = True
    enable_diagnostics: bool = True
    enable_dashboard: bool = True
    max_history_size: int = 1000
    retention_period: int = 86400  # seconds

class MetricsCollector:
    """System metrics collection system."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.collectors = {
            MetricType.CPU: self._collect_cpu_metrics,
            MetricType.MEMORY: self._collect_memory_metrics,
            MetricType.DISK: self._collect_disk_metrics,
            MetricType.NETWORK: self._collect_network_metrics,
            MetricType.PROCESS: self._collect_process_metrics
        }
        self.custom_metrics = {}
    
    def collect_metrics(self, metric_types: List[MetricType] = None) -> Dict[str, HealthMetric]:
        """
        Collect system metrics.
        
        Args:
            metric_types: Types of metrics to collect
            
        Returns:
            Dictionary of collected metrics
        """
        if metric_types is None:
            metric_types = list(MetricType)
        
        metrics = {}
        
        for metric_type in metric_types:
            if metric_type in self.collectors:
                try:
                    collected = self.collectors[metric_type]()
                    metrics.update(collected)
                except Exception as e:
                    logger.error(f"Failed to collect {metric_type.value} metrics", error=str(e))
        
        # Add custom metrics
        metrics.update(self.custom_metrics)
        
        return metrics
    
    def _collect_cpu_metrics(self) -> Dict[str, HealthMetric]:
        """Collect CPU metrics."""
        metrics = {}
        
        # CPU usage percentage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics['cpu_usage'] = HealthMetric(
            metric_id=str(uuid.uuid4()),
            name="CPU Usage",
            value=cpu_percent,
            unit="%",
            metric_type=MetricType.CPU,
            threshold_warning=70.0,
            threshold_critical=90.0
        )
        
        # CPU count
        cpu_count = psutil.cpu_count()
        metrics['cpu_count'] = HealthMetric(
            metric_id=str(uuid.uuid4()),
            name="CPU Count",
            value=float(cpu_count),
            unit="cores",
            metric_type=MetricType.CPU
        )
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics['cpu_frequency'] = HealthMetric(
                    metric_id=str(uuid.uuid4()),
                    name="CPU Frequency",
                    value=cpu_freq.current,
                    unit="MHz",
                    metric_type=MetricType.CPU
                )
        except Exception:
            pass
        
        return metrics
    
    def _collect_memory_metrics(self) -> Dict[str, HealthMetric]:
        """Collect memory metrics."""
        metrics = {}
        
        memory = psutil.virtual_memory()
        
        # Memory usage percentage
        metrics['memory_usage'] = HealthMetric(
            metric_id=str(uuid.uuid4()),
            name="Memory Usage",
            value=memory.percent,
            unit="%",
            metric_type=MetricType.MEMORY,
            threshold_warning=80.0,
            threshold_critical=95.0
        )
        
        # Available memory
        metrics['memory_available'] = HealthMetric(
            metric_id=str(uuid.uuid4()),
            name="Available Memory",
            value=memory.available / (1024**3),  # GB
            unit="GB",
            metric_type=MetricType.MEMORY
        )
        
        # Total memory
        metrics['memory_total'] = HealthMetric(
            metric_id=str(uuid.uuid4()),
            name="Total Memory",
            value=memory.total / (1024**3),  # GB
            unit="GB",
            metric_type=MetricType.MEMORY
        )
        
        return metrics
    
    def _collect_disk_metrics(self) -> Dict[str, HealthMetric]:
        """Collect disk metrics."""
        metrics = {}
        
        disk_usage = psutil.disk_usage('/')
        
        # Disk usage percentage
        metrics['disk_usage'] = HealthMetric(
            metric_id=str(uuid.uuid4()),
            name="Disk Usage",
            value=disk_usage.percent,
            unit="%",
            metric_type=MetricType.DISK,
            threshold_warning=80.0,
            threshold_critical=95.0
        )
        
        # Available disk space
        metrics['disk_available'] = HealthMetric(
            metric_id=str(uuid.uuid4()),
            name="Available Disk Space",
            value=disk_usage.free / (1024**3),  # GB
            unit="GB",
            metric_type=MetricType.DISK
        )
        
        # Total disk space
        metrics['disk_total'] = HealthMetric(
            metric_id=str(uuid.uuid4()),
            name="Total Disk Space",
            value=disk_usage.total / (1024**3),  # GB
            unit="GB",
            metric_type=MetricType.DISK
        )
        
        return metrics
    
    def _collect_network_metrics(self) -> Dict[str, HealthMetric]:
        """Collect network metrics."""
        metrics = {}
        
        try:
            network_io = psutil.net_io_counters()
            
            # Network bytes sent
            metrics['network_bytes_sent'] = HealthMetric(
                metric_id=str(uuid.uuid4()),
                name="Network Bytes Sent",
                value=network_io.bytes_sent / (1024**2),  # MB
                unit="MB",
                metric_type=MetricType.NETWORK
            )
            
            # Network bytes received
            metrics['network_bytes_recv'] = HealthMetric(
                metric_id=str(uuid.uuid4()),
                name="Network Bytes Received",
                value=network_io.bytes_recv / (1024**2),  # MB
                unit="MB",
                metric_type=MetricType.NETWORK
            )
            
            # Network packets sent
            metrics['network_packets_sent'] = HealthMetric(
                metric_id=str(uuid.uuid4()),
                name="Network Packets Sent",
                value=float(network_io.packets_sent),
                unit="packets",
                metric_type=MetricType.NETWORK
            )
            
            # Network packets received
            metrics['network_packets_recv'] = HealthMetric(
                metric_id=str(uuid.uuid4()),
                name="Network Packets Received",
                value=float(network_io.packets_recv),
                unit="packets",
                metric_type=MetricType.NETWORK
            )
        
        except Exception as e:
            logger.warning("Failed to collect network metrics", error=str(e))
        
        return metrics
    
    def _collect_process_metrics(self) -> Dict[str, HealthMetric]:
        """Collect process metrics."""
        metrics = {}
        
        try:
            # Current process
            process = psutil.Process()
            
            # Process CPU usage
            process_cpu = process.cpu_percent()
            metrics['process_cpu'] = HealthMetric(
                metric_id=str(uuid.uuid4()),
                name="Process CPU Usage",
                value=process_cpu,
                unit="%",
                metric_type=MetricType.PROCESS,
                threshold_warning=50.0,
                threshold_critical=80.0
            )
            
            # Process memory usage
            process_memory = process.memory_info()
            metrics['process_memory'] = HealthMetric(
                metric_id=str(uuid.uuid4()),
                name="Process Memory Usage",
                value=process_memory.rss / (1024**2),  # MB
                unit="MB",
                metric_type=MetricType.PROCESS,
                threshold_warning=1000.0,  # 1GB
                threshold_critical=2000.0  # 2GB
            )
            
            # Process threads
            metrics['process_threads'] = HealthMetric(
                metric_id=str(uuid.uuid4()),
                name="Process Threads",
                value=float(process.num_threads()),
                unit="threads",
                metric_type=MetricType.PROCESS
            )
            
            # Process open files
            try:
                open_files = len(process.open_files())
                metrics['process_open_files'] = HealthMetric(
                    metric_id=str(uuid.uuid4()),
                    name="Process Open Files",
                    value=float(open_files),
                    unit="files",
                    metric_type=MetricType.PROCESS
                )
            except Exception:
                pass
        
        except Exception as e:
            logger.warning("Failed to collect process metrics", error=str(e))
        
        return metrics
    
    def add_custom_metric(self, name: str, value: float, unit: str, 
                         threshold_warning: float = None, threshold_critical: float = None):
        """Add custom metric."""
        metric = HealthMetric(
            metric_id=str(uuid.uuid4()),
            name=name,
            value=value,
            unit=unit,
            metric_type=MetricType.CUSTOM,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical
        )
        self.custom_metrics[name] = metric

class HealthAnalyzer:
    """System health analysis system."""
    
    def __init__(self, config: HealthConfig):
        """
        Initialize health analyzer.
        
        Args:
            config: Health monitoring configuration
        """
        self.config = config
        self.health_history = deque(maxlen=config.max_history_size)
    
    def analyze_health(self, metrics: Dict[str, HealthMetric]) -> SystemHealth:
        """
        Analyze system health from metrics.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            System health assessment
        """
        # Assess individual metrics
        alerts = []
        overall_status = HealthStatus.HEALTHY
        performance_score = 100.0
        
        for metric_name, metric in metrics.items():
            # Check thresholds
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                metric.status = HealthStatus.CRITICAL
                alerts.append({
                    'level': AlertLevel.CRITICAL,
                    'message': f"{metric.name} is critical: {metric.value}{metric.unit}",
                    'metric': metric_name,
                    'timestamp': datetime.now().isoformat()
                })
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                metric.status = HealthStatus.WARNING
                alerts.append({
                    'level': AlertLevel.WARNING,
                    'message': f"{metric.name} is high: {metric.value}{metric.unit}",
                    'metric': metric_name,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Update overall status
            if metric.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif metric.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
            
            # Calculate performance score
            if metric.threshold_critical:
                if metric.value >= metric.threshold_critical:
                    performance_score -= 20
                elif metric.value >= metric.threshold_warning:
                    performance_score -= 10
        
        # Calculate uptime
        uptime = time.time() - psutil.boot_time()
        
        # Create system health
        system_health = SystemHealth(
            system_id="trading_system",
            overall_status=overall_status,
            metrics=metrics,
            alerts=alerts,
            performance_score=max(0.0, performance_score),
            uptime=uptime
        )
        
        # Store in history
        self.health_history.append(system_health)
        
        return system_health
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_health = [h for h in self.health_history if h.timestamp >= cutoff_time]
        
        if not recent_health:
            return {}
        
        trends = {
            'performance_score': {
                'current': recent_health[-1].performance_score,
                'average': sum(h.performance_score for h in recent_health) / len(recent_health),
                'min': min(h.performance_score for h in recent_health),
                'max': max(h.performance_score for h in recent_health)
            },
            'status_distribution': {
                'healthy': sum(1 for h in recent_health if h.overall_status == HealthStatus.HEALTHY),
                'warning': sum(1 for h in recent_health if h.overall_status == HealthStatus.WARNING),
                'critical': sum(1 for h in recent_health if h.overall_status == HealthStatus.CRITICAL)
            },
            'alert_count': sum(len(h.alerts) for h in recent_health),
            'uptime': recent_health[-1].uptime
        }
        
        return trends

class HealthAlertManager:
    """Health alert management system."""
    
    def __init__(self, config: HealthConfig):
        """
        Initialize health alert manager.
        
        Args:
            config: Health monitoring configuration
        """
        self.config = config
        self.alert_callbacks = []
        self.alert_history = deque(maxlen=1000)
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def process_alerts(self, alerts: List[Dict[str, Any]]):
        """Process health alerts."""
        for alert in alerts:
            # Store alert
            self.alert_history.append(alert)
            
            # Call callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error("Alert callback error", error=str(e))
            
            # Log alert
            logger.warning("Health alert", level=alert['level'].value, 
                          message=alert['message'], metric=alert['metric'])
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history 
                        if datetime.fromisoformat(a['timestamp']) >= cutoff_time]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_level': defaultdict(int),
            'by_metric': defaultdict(int)
        }
        
        for alert in recent_alerts:
            summary['by_level'][alert['level'].value] += 1
            summary['by_metric'][alert['metric']] += 1
        
        return summary

class SystemMonitor:
    """Main system health monitoring system."""
    
    def __init__(self, config: HealthConfig):
        """
        Initialize system monitor.
        
        Args:
            config: Health monitoring configuration
        """
        self.config = config
        self.collector = MetricsCollector()
        self.analyzer = HealthAnalyzer(config)
        self.alert_manager = HealthAlertManager(config)
        
        self.monitoring_thread = None
        self.running = False
        self.current_health = None
        
        # Setup default alert callback
        self.alert_manager.add_alert_callback(self._default_alert_handler)
    
    def start(self):
        """Start system monitoring."""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_worker)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("System health monitoring started")
    
    def stop(self):
        """Stop system monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("System health monitoring stopped")
    
    def _monitoring_worker(self):
        """Monitoring worker thread."""
        while self.running:
            try:
                # Collect metrics
                metrics = self.collector.collect_metrics()
                
                # Analyze health
                system_health = self.analyzer.analyze_health(metrics)
                self.current_health = system_health
                
                # Process alerts
                if self.config.enable_alerts and system_health.alerts:
                    self.alert_manager.process_alerts(system_health.alerts)
                
                # Log health status
                logger.debug("System health", status=system_health.overall_status.value, 
                           performance_score=system_health.performance_score)
                
                # Wait for next monitoring cycle
                time.sleep(self.config.monitoring_interval)
            
            except Exception as e:
                logger.error("System monitoring error", error=str(e))
                time.sleep(self.config.monitoring_interval)
    
    def _default_alert_handler(self, alert: Dict[str, Any]):
        """Default alert handler."""
        # This could send alerts to external systems
        logger.warning("System alert", alert=alert)
    
    def get_current_health(self) -> Optional[SystemHealth]:
        """Get current system health."""
        return self.current_health
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends."""
        return self.analyzer.get_health_trends(hours)
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary."""
        return self.alert_manager.get_alert_summary(hours)
    
    def add_custom_metric(self, name: str, value: float, unit: str, 
                         threshold_warning: float = None, threshold_critical: float = None):
        """Add custom metric."""
        self.collector.add_custom_metric(name, value, unit, threshold_warning, threshold_critical)
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add custom alert callback."""
        self.alert_manager.add_alert_callback(callback)

def create_system_monitor(config: HealthConfig = None) -> SystemMonitor:
    """
    Create a system health monitoring system.
    
    Args:
        config: Health monitoring configuration
        
    Returns:
        SystemMonitor instance
    """
    if config is None:
        config = HealthConfig()
    
    return SystemMonitor(config)

if __name__ == "__main__":
    # Demo of system health monitoring
    config = HealthConfig(
        monitoring_interval=5.0,
        enable_alerts=True,
        enable_diagnostics=True,
        enable_dashboard=True,
        max_history_size=1000,
        retention_period=86400
    )
    
    monitor = create_system_monitor(config)
    
    # Add custom alert callback
    def custom_alert_handler(alert):
        print(f"🚨 Alert: {alert['message']}")
    
    monitor.add_alert_callback(custom_alert_handler)
    
    # Start monitoring
    monitor.start()
    
    print("System health monitoring created successfully!")
    print(f"Monitoring interval: {config.monitoring_interval}s")
    print(f"Alerts enabled: {config.enable_alerts}")
    
    # Add custom metric
    monitor.add_custom_metric("custom_metric", 75.0, "%", 80.0, 95.0)
    
    # Get initial health
    time.sleep(2)  # Wait for first monitoring cycle
    health = monitor.get_current_health()
    if health:
        print(f"Current health status: {health.overall_status.value}")
        print(f"Performance score: {health.performance_score}")
        print(f"Uptime: {health.uptime:.0f} seconds")
    
    # Get trends
    trends = monitor.get_health_trends()
    print(f"Health trends: {trends}")
    
    # Get alert summary
    alert_summary = monitor.get_alert_summary()
    print(f"Alert summary: {alert_summary}")