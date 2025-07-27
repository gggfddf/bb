"""
Monitoring Dashboard for Trading System

This module provides monitoring dashboard functionality for visualizing
system metrics, health status, and alerts.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path

from .monitor import SystemMonitor, MetricType, HealthStatus
from ..logging.log_aggregator import LogAggregator, LogLevel, LogSource

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Dashboard type enumeration"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    SECURITY = "security"
    TRADING = "trading"
    CUSTOM = "custom"

@dataclass
class DashboardWidget:
    """Dashboard widget structure"""
    widget_id: str
    widget_type: str
    title: str
    description: Optional[str]
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any]
    data_source: str
    refresh_interval: int  # seconds

@dataclass
class Dashboard:
    """Dashboard structure"""
    dashboard_id: str
    name: str
    dashboard_type: DashboardType
    description: Optional[str]
    widgets: List[DashboardWidget]
    created_at: datetime
    updated_at: datetime
    is_public: bool
    owner: Optional[str]

class MonitoringDashboard:
    """
    Monitoring dashboard for visualizing system metrics and health.
    """
    
    def __init__(self, db_path: str = "monitoring/dashboard.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dashboard components
        self.dashboards: Dict[str, Dashboard] = {}
        self.system_monitor = SystemMonitor()
        self.log_aggregator = LogAggregator()
        
        # Initialize database
        self._init_database()
        
        # Create default dashboards
        self._create_default_dashboards()
        
        logger.info("Monitoring dashboard initialized")
    
    def create_dashboard(self, name: str, dashboard_type: DashboardType, description: Optional[str] = None,
                        owner: Optional[str] = None, is_public: bool = False) -> Dashboard:
        """
        Create a new dashboard.
        
        Args:
            name: Dashboard name
            dashboard_type: Type of dashboard
            description: Dashboard description
            owner: Dashboard owner
            is_public: Whether dashboard is public
            
        Returns:
            Created dashboard
        """
        try:
            dashboard_id = self._generate_dashboard_id()
            
            dashboard = Dashboard(
                dashboard_id=dashboard_id,
                name=name,
                dashboard_type=dashboard_type,
                description=description,
                widgets=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_public=is_public,
                owner=owner
            )
            
            self.dashboards[dashboard_id] = dashboard
            self._store_dashboard(dashboard)
            
            logger.info(f"Created dashboard: {name}")
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            raise
    
    def add_widget(self, dashboard_id: str, widget_type: str, title: str, position: Dict[str, int],
                   config: Dict[str, Any], data_source: str, description: Optional[str] = None,
                   refresh_interval: int = 30) -> DashboardWidget:
        """
        Add a widget to a dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            widget_type: Type of widget
            title: Widget title
            position: Widget position (x, y, width, height)
            config: Widget configuration
            data_source: Data source for widget
            description: Widget description
            refresh_interval: Refresh interval in seconds
            
        Returns:
            Created widget
        """
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard {dashboard_id} not found")
            
            widget_id = self._generate_widget_id()
            
            widget = DashboardWidget(
                widget_id=widget_id,
                widget_type=widget_type,
                title=title,
                description=description,
                position=position,
                config=config,
                data_source=data_source,
                refresh_interval=refresh_interval
            )
            
            dashboard = self.dashboards[dashboard_id]
            dashboard.widgets.append(widget)
            dashboard.updated_at = datetime.utcnow()
            
            self._store_widget(dashboard_id, widget)
            self._update_dashboard(dashboard)
            
            logger.info(f"Added widget {title} to dashboard {dashboard.name}")
            return widget
            
        except Exception as e:
            logger.error(f"Failed to add widget: {e}")
            raise
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """
        Get a dashboard by ID.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            Dashboard if found
        """
        return self.dashboards.get(dashboard_id)
    
    def get_dashboards(self, dashboard_type: Optional[DashboardType] = None, 
                      owner: Optional[str] = None, public_only: bool = False) -> List[Dashboard]:
        """
        Get dashboards with filters.
        
        Args:
            dashboard_type: Dashboard type filter
            owner: Owner filter
            public_only: Only return public dashboards
            
        Returns:
            List of dashboards
        """
        dashboards = []
        
        for dashboard in self.dashboards.values():
            if dashboard_type and dashboard.dashboard_type != dashboard_type:
                continue
            if owner and dashboard.owner != owner:
                continue
            if public_only and not dashboard.is_public:
                continue
            dashboards.append(dashboard)
        
        return sorted(dashboards, key=lambda d: d.updated_at, reverse=True)
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """
        Get dashboard data for rendering.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            Dashboard data
        """
        try:
            dashboard = self.get_dashboard(dashboard_id)
            if not dashboard:
                return {}
            
            dashboard_data = {
                'dashboard': asdict(dashboard),
                'widgets_data': {}
            }
            
            # Get data for each widget
            for widget in dashboard.widgets:
                widget_data = self._get_widget_data(widget)
                dashboard_data['widgets_data'][widget.widget_id] = widget_data
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    def get_system_overview(self) -> Dict[str, Any]:
        """
        Get system overview data.
        
        Returns:
            System overview data
        """
        try:
            # Get system stats
            system_stats = self.system_monitor.get_system_stats()
            
            # Get health status
            health_status = self.system_monitor.get_health_status()
            
            # Get recent alerts
            recent_alerts = self.system_monitor.get_alerts(resolved=False, limit=10)
            
            # Get recent logs
            recent_logs = self.log_aggregator.get_logs(limit=20)
            
            # Get log statistics
            log_stats = self.log_aggregator.get_statistics()
            
            return {
                'system_stats': system_stats,
                'health_status': health_status,
                'recent_alerts': [asdict(alert) for alert in recent_alerts],
                'recent_logs': [asdict(log) for log in recent_logs],
                'log_stats': log_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return {}
    
    def get_performance_metrics(self, time_range: str = "1h") -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            time_range: Time range (1h, 6h, 24h, 7d)
            
        Returns:
            Performance metrics
        """
        try:
            end_time = datetime.utcnow()
            
            if time_range == "1h":
                start_time = end_time - timedelta(hours=1)
            elif time_range == "6h":
                start_time = end_time - timedelta(hours=6)
            elif time_range == "24h":
                start_time = end_time - timedelta(days=1)
            elif time_range == "7d":
                start_time = end_time - timedelta(days=7)
            else:
                start_time = end_time - timedelta(hours=1)
            
            # Get CPU metrics
            cpu_metrics = self.system_monitor.get_metrics("cpu_usage", start_time, end_time)
            
            # Get memory metrics
            memory_metrics = self.system_monitor.get_metrics("memory_usage", start_time, end_time)
            
            # Get disk metrics
            disk_metrics = self.system_monitor.get_metrics("disk_usage", start_time, end_time)
            
            # Get process metrics
            process_metrics = self.system_monitor.get_metrics("process_count", start_time, end_time)
            
            return {
                'cpu_metrics': [asdict(metric) for metric in cpu_metrics],
                'memory_metrics': [asdict(metric) for metric in memory_metrics],
                'disk_metrics': [asdict(metric) for metric in disk_metrics],
                'process_metrics': [asdict(metric) for metric in process_metrics],
                'time_range': time_range,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def get_security_overview(self) -> Dict[str, Any]:
        """
        Get security overview data.
        
        Returns:
            Security overview data
        """
        try:
            # Get security alerts
            security_alerts = self.system_monitor.get_alerts(component="security", resolved=False)
            
            # Get security logs
            security_logs = self.log_aggregator.get_logs(source=LogSource.SECURITY, limit=50)
            
            # Get failed login attempts
            failed_logins = self.log_aggregator.search_logs("failed login", limit=20)
            
            # Get suspicious activities
            suspicious_activities = self.log_aggregator.search_logs("suspicious", limit=20)
            
            return {
                'security_alerts': [asdict(alert) for alert in security_alerts],
                'security_logs': [asdict(log) for log in security_logs],
                'failed_logins': [asdict(log) for log in failed_logins],
                'suspicious_activities': [asdict(log) for log in suspicious_activities],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get security overview: {e}")
            return {}
    
    def get_trading_overview(self) -> Dict[str, Any]:
        """
        Get trading overview data.
        
        Returns:
            Trading overview data
        """
        try:
            # Get trading logs
            trading_logs = self.log_aggregator.get_logs(source=LogSource.TRADING, limit=50)
            
            # Get trading errors
            trading_errors = self.log_aggregator.get_logs(source=LogSource.TRADING, level=LogLevel.ERROR, limit=20)
            
            # Get API usage
            api_logs = self.log_aggregator.get_logs(source=LogSource.API, limit=30)
            
            return {
                'trading_logs': [asdict(log) for log in trading_logs],
                'trading_errors': [asdict(log) for log in trading_errors],
                'api_logs': [asdict(log) for log in api_logs],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get trading overview: {e}")
            return {}
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """
        Delete a dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            True if dashboard was deleted
        """
        try:
            if dashboard_id in self.dashboards:
                del self.dashboards[dashboard_id]
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM dashboards WHERE dashboard_id = ?", (dashboard_id,))
                    conn.execute("DELETE FROM widgets WHERE dashboard_id = ?", (dashboard_id,))
                    conn.commit()
                
                logger.info(f"Deleted dashboard: {dashboard_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete dashboard: {e}")
            return False
    
    def _init_database(self):
        """Initialize the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Dashboards table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dashboards (
                        dashboard_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        dashboard_type TEXT NOT NULL,
                        description TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        is_public BOOLEAN DEFAULT FALSE,
                        owner TEXT
                    )
                """)
                
                # Widgets table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS widgets (
                        widget_id TEXT PRIMARY KEY,
                        dashboard_id TEXT NOT NULL,
                        widget_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        position TEXT NOT NULL,
                        config TEXT NOT NULL,
                        data_source TEXT NOT NULL,
                        refresh_interval INTEGER DEFAULT 30,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (dashboard_id) REFERENCES dashboards (dashboard_id)
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_dashboards_type ON dashboards(dashboard_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_dashboards_owner ON dashboards(owner)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_widgets_dashboard ON widgets(dashboard_id)")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_default_dashboards(self):
        """Create default dashboards"""
        try:
            # System Dashboard
            system_dashboard = self.create_dashboard(
                name="System Overview",
                dashboard_type=DashboardType.SYSTEM,
                description="System performance and health monitoring",
                is_public=True
            )
            
            # Add system widgets
            self.add_widget(
                system_dashboard.dashboard_id,
                "metric_chart",
                "CPU Usage",
                {"x": 0, "y": 0, "width": 6, "height": 4},
                {"chart_type": "line", "color": "#ff6b6b"},
                "cpu_usage",
                "CPU usage over time"
            )
            
            self.add_widget(
                system_dashboard.dashboard_id,
                "metric_chart",
                "Memory Usage",
                {"x": 6, "y": 0, "width": 6, "height": 4},
                {"chart_type": "line", "color": "#4ecdc4"},
                "memory_usage",
                "Memory usage over time"
            )
            
            self.add_widget(
                system_dashboard.dashboard_id,
                "health_status",
                "System Health",
                {"x": 0, "y": 4, "width": 4, "height": 3},
                {"show_details": True},
                "health_status",
                "System health status"
            )
            
            self.add_widget(
                system_dashboard.dashboard_id,
                "alert_list",
                "Recent Alerts",
                {"x": 4, "y": 4, "width": 8, "height": 3},
                {"max_alerts": 10},
                "recent_alerts",
                "Recent system alerts"
            )
            
            # Performance Dashboard
            performance_dashboard = self.create_dashboard(
                name="Performance Metrics",
                dashboard_type=DashboardType.PERFORMANCE,
                description="Detailed performance metrics and analysis",
                is_public=True
            )
            
            # Add performance widgets
            self.add_widget(
                performance_dashboard.dashboard_id,
                "metric_chart",
                "System Performance",
                {"x": 0, "y": 0, "width": 12, "height": 6},
                {"chart_type": "multi_line", "metrics": ["cpu_usage", "memory_usage", "disk_usage"]},
                "performance_metrics",
                "System performance metrics"
            )
            
            self.add_widget(
                performance_dashboard.dashboard_id,
                "metric_gauge",
                "Current CPU",
                {"x": 0, "y": 6, "width": 3, "height": 3},
                {"min": 0, "max": 100, "thresholds": [80, 90]},
                "current_cpu",
                "Current CPU usage"
            )
            
            self.add_widget(
                performance_dashboard.dashboard_id,
                "metric_gauge",
                "Current Memory",
                {"x": 3, "y": 6, "width": 3, "height": 3},
                {"min": 0, "max": 100, "thresholds": [80, 90]},
                "current_memory",
                "Current memory usage"
            )
            
            self.add_widget(
                performance_dashboard.dashboard_id,
                "metric_gauge",
                "Current Disk",
                {"x": 6, "y": 6, "width": 3, "height": 3},
                {"min": 0, "max": 100, "thresholds": [80, 90]},
                "current_disk",
                "Current disk usage"
            )
            
            # Security Dashboard
            security_dashboard = self.create_dashboard(
                name="Security Monitoring",
                dashboard_type=DashboardType.SECURITY,
                description="Security events and alerts monitoring",
                is_public=False
            )
            
            # Add security widgets
            self.add_widget(
                security_dashboard.dashboard_id,
                "log_table",
                "Security Events",
                {"x": 0, "y": 0, "width": 12, "height": 6},
                {"columns": ["timestamp", "level", "message", "ip_address"], "max_rows": 50},
                "security_logs",
                "Recent security events"
            )
            
            self.add_widget(
                security_dashboard.dashboard_id,
                "alert_list",
                "Security Alerts",
                {"x": 0, "y": 6, "width": 6, "height": 4},
                {"max_alerts": 20, "severity_filter": ["critical", "warning"]},
                "security_alerts",
                "Security alerts"
            )
            
            self.add_widget(
                security_dashboard.dashboard_id,
                "log_table",
                "Failed Logins",
                {"x": 6, "y": 6, "width": 6, "height": 4},
                {"columns": ["timestamp", "ip_address", "message"], "max_rows": 20},
                "failed_logins",
                "Failed login attempts"
            )
            
            logger.info("Default dashboards created")
            
        except Exception as e:
            logger.error(f"Failed to create default dashboards: {e}")
    
    def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget"""
        try:
            if widget.data_source == "cpu_usage":
                return self._get_cpu_usage_data(widget)
            elif widget.data_source == "memory_usage":
                return self._get_memory_usage_data(widget)
            elif widget.data_source == "health_status":
                return self._get_health_status_data(widget)
            elif widget.data_source == "recent_alerts":
                return self._get_recent_alerts_data(widget)
            elif widget.data_source == "performance_metrics":
                return self._get_performance_metrics_data(widget)
            elif widget.data_source == "security_logs":
                return self._get_security_logs_data(widget)
            elif widget.data_source == "security_alerts":
                return self._get_security_alerts_data(widget)
            elif widget.data_source == "failed_logins":
                return self._get_failed_logins_data(widget)
            else:
                return {"error": f"Unknown data source: {widget.data_source}"}
                
        except Exception as e:
            logger.error(f"Failed to get widget data: {e}")
            return {"error": str(e)}
    
    def _get_cpu_usage_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get CPU usage data"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        metrics = self.system_monitor.get_metrics("cpu_usage", start_time, end_time)
        
        return {
            "data": [{"timestamp": metric.timestamp.isoformat(), "value": metric.value} for metric in metrics],
            "config": widget.config
        }
    
    def _get_memory_usage_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get memory usage data"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        metrics = self.system_monitor.get_metrics("memory_usage", start_time, end_time)
        
        return {
            "data": [{"timestamp": metric.timestamp.isoformat(), "value": metric.value} for metric in metrics],
            "config": widget.config
        }
    
    def _get_health_status_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get health status data"""
        health_status = self.system_monitor.get_health_status()
        
        return {
            "data": health_status,
            "config": widget.config
        }
    
    def _get_recent_alerts_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get recent alerts data"""
        max_alerts = widget.config.get("max_alerts", 10)
        alerts = self.system_monitor.get_alerts(resolved=False, limit=max_alerts)
        
        return {
            "data": [asdict(alert) for alert in alerts],
            "config": widget.config
        }
    
    def _get_performance_metrics_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get performance metrics data"""
        return self.get_performance_metrics("1h")
    
    def _get_security_logs_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get security logs data"""
        max_rows = widget.config.get("max_rows", 50)
        logs = self.log_aggregator.get_logs(source=LogSource.SECURITY, limit=max_rows)
        
        return {
            "data": [asdict(log) for log in logs],
            "config": widget.config
        }
    
    def _get_security_alerts_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get security alerts data"""
        max_alerts = widget.config.get("max_alerts", 20)
        severity_filter = widget.config.get("severity_filter", [])
        
        alerts = self.system_monitor.get_alerts(component="security", resolved=False, limit=max_alerts)
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity in severity_filter]
        
        return {
            "data": [asdict(alert) for alert in alerts],
            "config": widget.config
        }
    
    def _get_failed_logins_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get failed logins data"""
        max_rows = widget.config.get("max_rows", 20)
        logs = self.log_aggregator.search_logs("failed login", limit=max_rows)
        
        return {
            "data": [asdict(log) for log in logs],
            "config": widget.config
        }
    
    def _store_dashboard(self, dashboard: Dashboard):
        """Store dashboard in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO dashboards (
                        dashboard_id, name, dashboard_type, description, created_at, updated_at, is_public, owner
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dashboard.dashboard_id,
                    dashboard.name,
                    dashboard.dashboard_type.value,
                    dashboard.description,
                    dashboard.created_at.isoformat(),
                    dashboard.updated_at.isoformat(),
                    dashboard.is_public,
                    dashboard.owner
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store dashboard: {e}")
    
    def _store_widget(self, dashboard_id: str, widget: DashboardWidget):
        """Store widget in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO widgets (
                        widget_id, dashboard_id, widget_type, title, description, position, config, data_source, refresh_interval
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    widget.widget_id,
                    dashboard_id,
                    widget.widget_type,
                    widget.title,
                    widget.description,
                    json.dumps(widget.position),
                    json.dumps(widget.config),
                    widget.data_source,
                    widget.refresh_interval
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store widget: {e}")
    
    def _update_dashboard(self, dashboard: Dashboard):
        """Update dashboard in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE dashboards 
                    SET updated_at = ?
                    WHERE dashboard_id = ?
                """, (dashboard.updated_at.isoformat(), dashboard.dashboard_id))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")
    
    def _generate_dashboard_id(self) -> str:
        """Generate dashboard ID"""
        import hashlib
        return hashlib.md5(f"dashboard_{datetime.utcnow().isoformat()}".encode()).hexdigest()
    
    def _generate_widget_id(self) -> str:
        """Generate widget ID"""
        import hashlib
        return hashlib.md5(f"widget_{datetime.utcnow().isoformat()}".encode()).hexdigest()


# Global instance management
_monitoring_dashboard_instance = None

def get_monitoring_dashboard(db_path: str = "monitoring/dashboard.db") -> MonitoringDashboard:
    """Get or create monitoring dashboard instance"""
    global _monitoring_dashboard_instance
    
    if _monitoring_dashboard_instance is None:
        _monitoring_dashboard_instance = MonitoringDashboard(db_path)
    
    return _monitoring_dashboard_instance


def init_monitoring_dashboard(db_path: str = "monitoring/dashboard.db") -> MonitoringDashboard:
    """Initialize monitoring dashboard with custom database path"""
    global _monitoring_dashboard_instance
    
    _monitoring_dashboard_instance = MonitoringDashboard(db_path)
    
    return _monitoring_dashboard_instance