#!/usr/bin/env python3
"""
Alert and Notification System Module

Implements comprehensive alert and notification system:
- Real-time alert generation
- Notification delivery system
- Alert configuration interface
- Alert history tracking
- Alert severity levels
- Alert acknowledgment system

Features:
- Real-time alert generation and monitoring
- Multiple notification delivery channels
- Configurable alert rules and thresholds
- Alert history and tracking
- Severity-based alert management
- Alert acknowledgment and escalation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import asyncio
import threading
import queue
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

logger = structlog.get_logger()

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert types."""
    PRICE_ALERT = "price_alert"
    VOLUME_ALERT = "volume_alert"
    TECHNICAL_ALERT = "technical_alert"
    PERFORMANCE_ALERT = "performance_alert"
    SYSTEM_ALERT = "system_alert"
    RISK_ALERT = "risk_alert"

class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    PUSH = "push"

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    conditions: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    notification_sent: bool = False

@dataclass
class NotificationConfig:
    """Notification configuration."""
    channel: NotificationChannel
    enabled: bool = True
    recipients: List[str] = field(default_factory=list)
    template: Optional[str] = None
    webhook_url: Optional[str] = None
    email_config: Optional[Dict[str, str]] = None

class AlertRuleEngine:
    """Alert rule engine for evaluating conditions."""
    
    def __init__(self):
        """Initialize alert rule engine."""
        self.rules = {}
        self.rule_callbacks = {}
    
    def add_rule(self, rule: AlertRule, callback: Callable[[Alert], None] = None):
        """Add alert rule."""
        self.rules[rule.rule_id] = rule
        if callback:
            self.rule_callbacks[rule.rule_id] = callback
    
    def evaluate_rule(self, rule_id: str, data: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate a specific rule against data."""
        if rule_id not in self.rules:
            return None
        
        rule = self.rules[rule_id]
        if not rule.enabled:
            return None
        
        # Evaluate conditions
        if self._evaluate_conditions(rule.conditions, data):
            alert = Alert(
                alert_id=f"{rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule_id,
                alert_type=rule.alert_type,
                severity=rule.severity,
                message=self._generate_message(rule, data),
                data=data
            )
            
            # Call rule callback if exists
            if rule_id in self.rule_callbacks:
                try:
                    self.rule_callbacks[rule_id](alert)
                except Exception as e:
                    logger.error("Rule callback error", rule_id=rule_id, error=str(e))
            
            return alert
        
        return None
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate alert conditions."""
        for condition_type, condition_data in conditions.items():
            if condition_type == "price_threshold":
                if not self._evaluate_price_threshold(condition_data, data):
                    return False
            elif condition_type == "volume_threshold":
                if not self._evaluate_volume_threshold(condition_data, data):
                    return False
            elif condition_type == "technical_indicator":
                if not self._evaluate_technical_indicator(condition_data, data):
                    return False
            elif condition_type == "performance_metric":
                if not self._evaluate_performance_metric(condition_data, data):
                    return False
            elif condition_type == "custom_condition":
                if not self._evaluate_custom_condition(condition_data, data):
                    return False
        
        return True
    
    def _evaluate_price_threshold(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate price threshold condition."""
        symbol = condition.get('symbol')
        threshold = condition.get('threshold')
        operator = condition.get('operator', '>')
        
        if symbol not in data or 'price' not in data[symbol]:
            return False
        
        price = data[symbol]['price']
        
        if operator == '>':
            return price > threshold
        elif operator == '<':
            return price < threshold
        elif operator == '>=':
            return price >= threshold
        elif operator == '<=':
            return price <= threshold
        elif operator == '==':
            return price == threshold
        
        return False
    
    def _evaluate_volume_threshold(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate volume threshold condition."""
        symbol = condition.get('symbol')
        threshold = condition.get('threshold')
        operator = condition.get('operator', '>')
        
        if symbol not in data or 'volume' not in data[symbol]:
            return False
        
        volume = data[symbol]['volume']
        
        if operator == '>':
            return volume > threshold
        elif operator == '<':
            return volume < threshold
        elif operator == '>=':
            return volume >= threshold
        elif operator == '<=':
            return volume <= threshold
        elif operator == '==':
            return volume == threshold
        
        return False
    
    def _evaluate_technical_indicator(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate technical indicator condition."""
        symbol = condition.get('symbol')
        indicator = condition.get('indicator')
        threshold = condition.get('threshold')
        operator = condition.get('operator', '>')
        
        if symbol not in data or 'indicators' not in data[symbol]:
            return False
        
        indicators = data[symbol]['indicators']
        if indicator not in indicators:
            return False
        
        value = indicators[indicator]
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        
        return False
    
    def _evaluate_performance_metric(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate performance metric condition."""
        metric = condition.get('metric')
        threshold = condition.get('threshold')
        operator = condition.get('operator', '>')
        
        if 'performance' not in data or metric not in data['performance']:
            return False
        
        value = data['performance'][metric]
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        
        return False
    
    def _evaluate_custom_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate custom condition."""
        # This can be extended with custom logic
        expression = condition.get('expression')
        if expression:
            try:
                # Simple expression evaluation (be careful with security)
                return eval(expression, {"data": data, "np": np, "pd": pd})
            except Exception as e:
                logger.error("Custom condition evaluation error", error=str(e))
                return False
        
        return True
    
    def _generate_message(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """Generate alert message."""
        template = rule.description
        
        # Replace placeholders with actual values
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    placeholder = f"{{{key}.{sub_key}}}"
                    template = template.replace(placeholder, str(sub_value))
            else:
                placeholder = f"{{{key}}}"
                template = template.replace(placeholder, str(value))
        
        return template

class NotificationManager:
    """Notification delivery manager."""
    
    def __init__(self):
        """Initialize notification manager."""
        self.channels = {}
        self.notification_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
    
    def add_channel(self, channel: NotificationChannel, config: NotificationConfig):
        """Add notification channel."""
        self.channels[channel] = config
    
    def start(self):
        """Start notification manager."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._notification_worker)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            logger.info("Notification manager started")
    
    def stop(self):
        """Stop notification manager."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Notification manager stopped")
    
    def send_notification(self, alert: Alert, channels: List[NotificationChannel] = None):
        """Send notification for alert."""
        if channels is None:
            channels = list(self.channels.keys())
        
        for channel in channels:
            if channel in self.channels and self.channels[channel].enabled:
                self.notification_queue.put((channel, alert))
    
    def _notification_worker(self):
        """Notification worker thread."""
        while self.running:
            try:
                channel, alert = self.notification_queue.get(timeout=1)
                self._send_notification(channel, alert)
                self.notification_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Notification worker error", error=str(e))
    
    def _send_notification(self, channel: NotificationChannel, alert: Alert):
        """Send notification through specific channel."""
        config = self.channels[channel]
        
        try:
            if channel == NotificationChannel.EMAIL:
                self._send_email_notification(config, alert)
            elif channel == NotificationChannel.SMS:
                self._send_sms_notification(config, alert)
            elif channel == NotificationChannel.WEBHOOK:
                self._send_webhook_notification(config, alert)
            elif channel == NotificationChannel.DASHBOARD:
                self._send_dashboard_notification(config, alert)
            elif channel == NotificationChannel.PUSH:
                self._send_push_notification(config, alert)
            
            alert.notification_sent = True
            logger.info("Notification sent", channel=channel.value, alert_id=alert.alert_id)
        
        except Exception as e:
            logger.error("Notification sending failed", channel=channel.value, 
                        alert_id=alert.alert_id, error=str(e))
    
    def _send_email_notification(self, config: NotificationConfig, alert: Alert):
        """Send email notification."""
        if not config.email_config or not config.recipients:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = config.email_config.get('from_email')
            msg['To'] = ', '.join(config.recipients)
            msg['Subject'] = f"Alert: {alert.severity.value.upper()} - {alert.alert_type.value}"
            
            body = self._format_email_body(alert, config.template)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email (configure SMTP settings)
            # This is a placeholder - implement actual SMTP sending
            logger.info("Email notification prepared", recipients=config.recipients)
        
        except Exception as e:
            logger.error("Email notification error", error=str(e))
    
    def _send_sms_notification(self, config: NotificationConfig, alert: Alert):
        """Send SMS notification."""
        # Implement SMS sending logic
        logger.info("SMS notification prepared", recipients=config.recipients)
    
    def _send_webhook_notification(self, config: NotificationConfig, alert: Alert):
        """Send webhook notification."""
        if not config.webhook_url:
            return
        
        try:
            payload = {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            
            response = requests.post(config.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error("Webhook notification error", error=str(e))
    
    def _send_dashboard_notification(self, config: NotificationConfig, alert: Alert):
        """Send dashboard notification."""
        # This would typically update a dashboard or send to a WebSocket
        logger.info("Dashboard notification prepared")
    
    def _send_push_notification(self, config: NotificationConfig, alert: Alert):
        """Send push notification."""
        # Implement push notification logic
        logger.info("Push notification prepared", recipients=config.recipients)
    
    def _format_email_body(self, alert: Alert, template: Optional[str] = None) -> str:
        """Format email body."""
        if template:
            # Use custom template
            return template.format(
                alert_id=alert.alert_id,
                severity=alert.severity.value,
                alert_type=alert.alert_type.value,
                message=alert.message,
                timestamp=alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                data=json.dumps(alert.data, indent=2)
            )
        else:
            # Default template
            return f"""
            <html>
            <body>
                <h2>Alert: {alert.severity.value.upper()}</h2>
                <p><strong>Type:</strong> {alert.alert_type.value}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Data:</strong></p>
                <pre>{json.dumps(alert.data, indent=2)}</pre>
            </body>
            </html>
            """

class AlertHistoryManager:
    """Alert history and tracking manager."""
    
    def __init__(self, max_alerts: int = 10000):
        """
        Initialize alert history manager.
        
        Args:
            max_alerts: Maximum number of alerts to keep in memory
        """
        self.alerts = []
        self.max_alerts = max_alerts
        self.alert_stats = {
            'total_alerts': 0,
            'acknowledged_alerts': 0,
            'severity_counts': {severity.value: 0 for severity in AlertSeverity},
            'type_counts': {alert_type.value: 0 for alert_type in AlertType}
        }
    
    def add_alert(self, alert: Alert):
        """Add alert to history."""
        self.alerts.append(alert)
        self._update_stats(alert, add=True)
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            removed_alert = self.alerts.pop(0)
            self._update_stats(removed_alert, add=False)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                if not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.now()
                    self.alert_stats['acknowledged_alerts'] += 1
                break
    
    def get_alerts(self, filters: Dict[str, Any] = None) -> List[Alert]:
        """Get alerts with optional filtering."""
        if not filters:
            return self.alerts.copy()
        
        filtered_alerts = []
        for alert in self.alerts:
            include = True
            
            for key, value in filters.items():
                if key == 'severity' and alert.severity != value:
                    include = False
                elif key == 'alert_type' and alert.alert_type != value:
                    include = False
                elif key == 'acknowledged' and alert.acknowledged != value:
                    include = False
                elif key == 'date_from' and alert.timestamp < value:
                    include = False
                elif key == 'date_to' and alert.timestamp > value:
                    include = False
            
            if include:
                filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return self.alert_stats.copy()
    
    def _update_stats(self, alert: Alert, add: bool):
        """Update alert statistics."""
        factor = 1 if add else -1
        self.alert_stats['total_alerts'] += factor
        self.alert_stats['severity_counts'][alert.severity.value] += factor
        self.alert_stats['type_counts'][alert.alert_type.value] += factor

class AlertSystem:
    """Main alert and notification system."""
    
    def __init__(self):
        """Initialize alert system."""
        self.rule_engine = AlertRuleEngine()
        self.notification_manager = NotificationManager()
        self.history_manager = AlertHistoryManager()
        
        self.running = False
        self.monitoring_thread = None
        self.data_queue = queue.Queue()
    
    def add_alert_rule(self, rule: AlertRule, callback: Callable[[Alert], None] = None):
        """Add alert rule."""
        self.rule_engine.add_rule(rule, callback)
    
    def add_notification_channel(self, channel: NotificationChannel, config: NotificationConfig):
        """Add notification channel."""
        self.notification_manager.add_channel(channel, config)
    
    def start(self):
        """Start alert system."""
        if not self.running:
            self.running = True
            self.notification_manager.start()
            self.monitoring_thread = threading.Thread(target=self._monitoring_worker)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Alert system started")
    
    def stop(self):
        """Stop alert system."""
        self.running = False
        self.notification_manager.stop()
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Alert system stopped")
    
    def process_data(self, data: Dict[str, Any]):
        """Process data for alert evaluation."""
        self.data_queue.put(data)
    
    def _monitoring_worker(self):
        """Monitoring worker thread."""
        while self.running:
            try:
                data = self.data_queue.get(timeout=1)
                self._evaluate_alerts(data)
                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Monitoring worker error", error=str(e))
    
    def _evaluate_alerts(self, data: Dict[str, Any]):
        """Evaluate all rules against data."""
        for rule_id in self.rule_engine.rules:
            alert = self.rule_engine.evaluate_rule(rule_id, data)
            if alert:
                self._handle_alert(alert)
    
    def _handle_alert(self, alert: Alert):
        """Handle generated alert."""
        # Add to history
        self.history_manager.add_alert(alert)
        
        # Send notifications
        self.notification_manager.send_notification(alert)
        
        logger.info("Alert generated", alert_id=alert.alert_id, 
                   severity=alert.severity.value, message=alert.message)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge alert."""
        self.history_manager.acknowledge_alert(alert_id, acknowledged_by)
    
    def get_alerts(self, filters: Dict[str, Any] = None) -> List[Alert]:
        """Get alerts with optional filtering."""
        return self.history_manager.get_alerts(filters)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return self.history_manager.get_alert_stats()

def create_alert_system() -> AlertSystem:
    """
    Create an alert and notification system.
    
    Returns:
        AlertSystem instance
    """
    return AlertSystem()

if __name__ == "__main__":
    # Demo of alert system
    alert_system = create_alert_system()
    
    # Add notification channels
    email_config = NotificationConfig(
        channel=NotificationChannel.EMAIL,
        enabled=True,
        recipients=["admin@example.com"],
        email_config={
            'from_email': 'alerts@example.com',
            'smtp_server': 'smtp.example.com',
            'smtp_port': 587
        }
    )
    alert_system.add_notification_channel(NotificationChannel.EMAIL, email_config)
    
    # Add alert rules
    price_rule = AlertRule(
        rule_id="price_drop",
        name="Price Drop Alert",
        description="Stock {symbol} price dropped below {threshold}",
        alert_type=AlertType.PRICE_ALERT,
        severity=AlertSeverity.WARNING,
        conditions={
            "price_threshold": {
                "symbol": "AAPL",
                "threshold": 150.0,
                "operator": "<"
            }
        }
    )
    alert_system.add_alert_rule(price_rule)
    
    volume_rule = AlertRule(
        rule_id="high_volume",
        name="High Volume Alert",
        description="Unusual volume detected for {symbol}: {volume}",
        alert_type=AlertType.VOLUME_ALERT,
        severity=AlertSeverity.INFO,
        conditions={
            "volume_threshold": {
                "symbol": "AAPL",
                "threshold": 1000000,
                "operator": ">"
            }
        }
    )
    alert_system.add_alert_rule(volume_rule)
    
    # Start alert system
    alert_system.start()
    
    # Simulate data processing
    sample_data = {
        "AAPL": {
            "price": 145.0,
            "volume": 1200000,
            "indicators": {
                "rsi": 30,
                "macd": -0.5
            }
        },
        "performance": {
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.05
        }
    }
    
    alert_system.process_data(sample_data)
    
    print("Alert system created successfully!")
    print("Added notification channels and alert rules")
    print("Processing sample data...")
    
    # Get alert statistics
    stats = alert_system.get_alert_stats()
    print(f"Alert statistics: {stats}")