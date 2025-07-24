import asyncio
import json
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import asdict
from enum import Enum
import structlog
from .error_handler import ErrorEvent, ErrorSeverity, global_error_handler

logger = structlog.get_logger()

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AlertConfig:
    channel: AlertChannel
    enabled: bool = True
    recipients: List[str] = None
    webhook_url: Optional[str] = None
    threshold: int = 1
    cooldown_minutes: int = 15

@dataclass
class Alert:
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    source: str
    error_events: List[ErrorEvent]
    metadata: Dict[str, Any]

class ErrorMonitoringSystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alert_configs: Dict[AlertChannel, AlertConfig] = {}
        self.alert_history: List[Alert] = []
        self.monitoring_active = False
        self.metrics = {
            "total_alerts": 0,
            "alerts_by_level": {},
            "alerts_by_source": {},
            "last_alert_time": None
        }
        
        self._setup_alert_configs()
        self._setup_monitoring_rules()

    def _setup_alert_configs(self):
        """Setup alert configurations for different channels"""
        # Email alerts
        self.alert_configs[AlertChannel.EMAIL] = AlertConfig(
            channel=AlertChannel.EMAIL,
            enabled=self.config.get("email_alerts_enabled", False),
            recipients=self.config.get("email_recipients", []),
            threshold=3,
            cooldown_minutes=30
        )
        
        # Slack alerts
        self.alert_configs[AlertChannel.SLACK] = AlertConfig(
            channel=AlertChannel.SLACK,
            enabled=self.config.get("slack_alerts_enabled", False),
            webhook_url=self.config.get("slack_webhook_url"),
            threshold=2,
            cooldown_minutes=15
        )
        
        # Webhook alerts
        self.alert_configs[AlertChannel.WEBHOOK] = AlertConfig(
            channel=AlertChannel.WEBHOOK,
            enabled=self.config.get("webhook_alerts_enabled", False),
            webhook_url=self.config.get("webhook_url"),
            threshold=1,
            cooldown_minutes=10
        )
        
        # Log alerts (always enabled)
        self.alert_configs[AlertChannel.LOG] = AlertConfig(
            channel=AlertChannel.LOG,
            enabled=True,
            threshold=1,
            cooldown_minutes=5
        )

    def _setup_monitoring_rules(self):
        """Setup monitoring rules and thresholds"""
        self.monitoring_rules = {
            "error_rate_threshold": self.config.get("error_rate_threshold", 0.1),  # 10% error rate
            "consecutive_failures_threshold": self.config.get("consecutive_failures_threshold", 5),
            "circuit_breaker_open_threshold": self.config.get("circuit_breaker_open_threshold", 1),
            "data_quality_threshold": self.config.get("data_quality_threshold", 0.8),  # 80% quality
            "response_time_threshold": self.config.get("response_time_threshold", 30.0),  # 30 seconds
        }

    async def start_monitoring(self):
        """Start the error monitoring system"""
        self.monitoring_active = True
        logger.info("Error monitoring system started")
        
        # Start background monitoring tasks
        asyncio.create_task(self._monitor_error_rates())
        asyncio.create_task(self._monitor_circuit_breakers())
        asyncio.create_task(self._cleanup_old_alerts())

    async def stop_monitoring(self):
        """Stop the error monitoring system"""
        self.monitoring_active = False
        logger.info("Error monitoring system stopped")

    async def _monitor_error_rates(self):
        """Monitor error rates and trigger alerts if thresholds are exceeded"""
        while self.monitoring_active:
            try:
                error_summary = global_error_handler.get_error_summary()
                recent_errors = error_summary.get("recent_errors", [])
                
                # Calculate error rate for the last hour
                if recent_errors:
                    error_rate = len(recent_errors) / 60  # Assuming 1 error per minute baseline
                    
                    if error_rate > self.monitoring_rules["error_rate_threshold"]:
                        await self._trigger_error_rate_alert(error_rate, recent_errors)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in error rate monitoring: {e}")
                await asyncio.sleep(60)

    async def _monitor_circuit_breakers(self):
        """Monitor circuit breaker states and trigger alerts"""
        while self.monitoring_active:
            try:
                error_summary = global_error_handler.get_error_summary()
                circuit_state = error_summary.get("circuit_breaker_state")
                failure_count = error_summary.get("failure_count", 0)
                
                if circuit_state == "open" and failure_count >= self.monitoring_rules["circuit_breaker_open_threshold"]:
                    await self._trigger_circuit_breaker_alert(failure_count)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in circuit breaker monitoring: {e}")
                await asyncio.sleep(30)

    async def _trigger_error_rate_alert(self, error_rate: float, recent_errors: List[Dict[str, Any]]):
        """Trigger alert for high error rate"""
        alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.ERROR,
            title="High Error Rate Detected",
            message=f"Error rate of {error_rate:.2%} exceeds threshold of {self.monitoring_rules['error_rate_threshold']:.2%}",
            source="error_monitoring",
            error_events=[ErrorEvent(**error) for error in recent_errors[-5:]],  # Last 5 errors
            metadata={
                "error_rate": error_rate,
                "threshold": self.monitoring_rules["error_rate_threshold"],
                "error_count": len(recent_errors)
            }
        )
        
        await self._send_alert(alert)

    async def _trigger_circuit_breaker_alert(self, failure_count: int):
        """Trigger alert for circuit breaker opening"""
        alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.CRITICAL,
            title="Circuit Breaker Opened",
            message=f"Circuit breaker opened after {failure_count} consecutive failures",
            source="error_monitoring",
            error_events=[],
            metadata={
                "failure_count": failure_count,
                "threshold": self.monitoring_rules["consecutive_failures_threshold"]
            }
        )
        
        await self._send_alert(alert)

    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        # Check cooldown periods
        if self._is_in_cooldown(alert):
            return
        
        # Send through each configured channel
        for channel, config in self.alert_configs.items():
            if not config.enabled:
                continue
            
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert, config)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(alert, config)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert, config)
                elif channel == AlertChannel.LOG:
                    await self._send_log_alert(alert, config)
                
                # Update metrics
                self._update_alert_metrics(alert)
                
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
        
        # Store alert in history
        self.alert_history.append(alert)

    def _is_in_cooldown(self, alert: Alert) -> bool:
        """Check if alert is in cooldown period"""
        if not self.metrics["last_alert_time"]:
            return False
        
        cooldown_minutes = self.alert_configs[AlertChannel.LOG].cooldown_minutes
        cooldown_delta = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - self.metrics["last_alert_time"] < cooldown_delta

    async def _send_email_alert(self, alert: Alert, config: AlertConfig):
        """Send alert via email"""
        if not config.recipients:
            return
        
        subject = f"[{alert.level.value.upper()}] {alert.title}"
        body = self._format_alert_message(alert)
        
        # TODO: Implement actual email sending
        logger.info(f"Email alert sent to {config.recipients}: {subject}")

    async def _send_slack_alert(self, alert: Alert, config: AlertConfig):
        """Send alert via Slack webhook"""
        if not config.webhook_url:
            return
        
        payload = {
            "text": f"*[{alert.level.value.upper()}] {alert.title}*",
            "attachments": [{
                "text": alert.message,
                "color": self._get_slack_color(alert.level),
                "fields": [
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True}
                ]
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(config.webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Slack webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _send_webhook_alert(self, alert: Alert, config: AlertConfig):
        """Send alert via webhook"""
        if not config.webhook_url:
            return
        
        payload = asdict(alert)
        payload["timestamp"] = alert.timestamp.isoformat()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(config.webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    async def _send_log_alert(self, alert: Alert, config: AlertConfig):
        """Send alert via logging"""
        log_level = getattr(logger, alert.level.value)
        log_level(
            f"ALERT: {alert.title}",
            message=alert.message,
            source=alert.source,
            level=alert.level.value,
            metadata=alert.metadata
        )

    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert message for different channels"""
        message = f"""
Alert: {alert.title}
Level: {alert.level.value.upper()}
Time: {alert.timestamp.isoformat()}
Source: {alert.source}

Message: {alert.message}

Metadata: {json.dumps(alert.metadata, indent=2)}
"""
        
        if alert.error_events:
            message += "\nRecent Errors:\n"
            for error in alert.error_events[-3:]:  # Last 3 errors
                message += f"- {error.error_type}: {error.error_message}\n"
        
        return message

    def _get_slack_color(self, level: AlertLevel) -> str:
        """Get Slack color for alert level"""
        colors = {
            AlertLevel.INFO: "good",
            AlertLevel.WARNING: "warning",
            AlertLevel.ERROR: "danger",
            AlertLevel.CRITICAL: "#ff0000"
        }
        return colors.get(level, "good")

    def _update_alert_metrics(self, alert: Alert):
        """Update alert metrics"""
        self.metrics["total_alerts"] += 1
        self.metrics["last_alert_time"] = datetime.now()
        
        # Update level metrics
        level = alert.level.value
        self.metrics["alerts_by_level"][level] = self.metrics["alerts_by_level"].get(level, 0) + 1
        
        # Update source metrics
        source = alert.source
        self.metrics["alerts_by_source"][source] = self.metrics["alerts_by_source"].get(source, 0) + 1

    async def _cleanup_old_alerts(self):
        """Clean up old alerts to prevent memory bloat"""
        while self.monitoring_active:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.alert_history = [
                    alert for alert in self.alert_history
                    if alert.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error in alert cleanup: {e}")
                await asyncio.sleep(3600)

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring system summary"""
        return {
            "monitoring_active": self.monitoring_active,
            "metrics": self.metrics,
            "alert_configs": {
                channel.value: {
                    "enabled": config.enabled,
                    "threshold": config.threshold,
                    "cooldown_minutes": config.cooldown_minutes
                }
                for channel, config in self.alert_configs.items()
            },
            "monitoring_rules": self.monitoring_rules,
            "recent_alerts": [
                asdict(alert) for alert in self.alert_history[-10:]  # Last 10 alerts
            ]
        }

    def add_custom_monitoring_rule(self, name: str, condition: Callable, alert_level: AlertLevel):
        """Add custom monitoring rule"""
        self.monitoring_rules[name] = {
            "condition": condition,
            "alert_level": alert_level
        }
        logger.info(f"Added custom monitoring rule: {name}")

# Global monitoring system instance
global_monitoring_system = ErrorMonitoringSystem()

# Convenience functions for monitoring
async def start_error_monitoring(config: Dict[str, Any] = None):
    """Start the global error monitoring system"""
    if config:
        global global_monitoring_system
        global_monitoring_system = ErrorMonitoringSystem(config)
    
    await global_monitoring_system.start_monitoring()

async def stop_error_monitoring():
    """Stop the global error monitoring system"""
    await global_monitoring_system.stop_monitoring()

def get_monitoring_summary() -> Dict[str, Any]:
    """Get monitoring system summary"""
    return global_monitoring_system.get_monitoring_summary()