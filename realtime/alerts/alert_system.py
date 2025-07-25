#!/usr/bin/env python3
"""
Real-time Alert System Module

Implements comprehensive real-time alert system for trading operations:
- Multi-channel alert delivery
- Alert prioritization system
- Alert aggregation and deduplication
- Alert escalation rules
- Alert acknowledgment system
- Alert history and analytics

Features:
- Multi-channel alert delivery (email, SMS, webhook, dashboard, push)
- Intelligent alert prioritization and routing
- Advanced alert aggregation and deduplication
- Configurable escalation rules and workflows
- Comprehensive acknowledgment and tracking system
- Real-time alert analytics and reporting
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import threading
import queue
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from collections import defaultdict, deque

logger = structlog.get_logger()

class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertCategory(Enum):
    """Alert categories."""
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    MARKET = "market"
    TECHNICAL = "technical"
    BUSINESS = "business"

class AlertStatus(Enum):
    """Alert status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"

class DeliveryChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    PUSH = "push"
    SLACK = "slack"
    TELEGRAM = "telegram"

@dataclass
class Alert:
    """Alert structure."""
    alert_id: str
    title: str
    message: str
    category: AlertCategory
    priority: AlertPriority
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.PENDING
    channels: List[DeliveryChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    escalation_level: int = 0
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    category: AlertCategory
    priority: AlertPriority
    conditions: Dict[str, Any]
    channels: List[DeliveryChannel]
    recipients: List[str]
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class AlertConfig:
    """Alert system configuration."""
    default_channels: List[DeliveryChannel] = field(default_factory=list)
    default_recipients: List[str] = field(default_factory=list)
    escalation_timeout: int = 300  # seconds
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    aggregation_window: int = 300  # seconds
    deduplication_enabled: bool = True
    analytics_enabled: bool = True

class AlertAggregator:
    """Alert aggregation and deduplication system."""
    
    def __init__(self, config: AlertConfig):
        """
        Initialize alert aggregator.
        
        Args:
            config: Alert system configuration
        """
        self.config = config
        self.alert_cache = {}
        self.aggregation_groups = defaultdict(list)
        self.deduplication_hashes = set()
    
    def should_aggregate(self, alert: Alert, existing_alerts: List[Alert]) -> bool:
        """Check if alert should be aggregated with existing alerts."""
        if not self.config.deduplication_enabled:
            return False
        
        # Check for exact duplicates
        alert_hash = self._generate_alert_hash(alert)
        if alert_hash in self.deduplication_hashes:
            return True
        
        # Check for similar alerts within aggregation window
        recent_alerts = [a for a in existing_alerts 
                        if (datetime.now() - a.timestamp).total_seconds() < self.config.aggregation_window]
        
        for existing_alert in recent_alerts:
            if self._are_alerts_similar(alert, existing_alert):
                return True
        
        return False
    
    def aggregate_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Aggregate similar alerts."""
        if not alerts:
            return []
        
        aggregated = []
        processed_hashes = set()
        
        for alert in alerts:
            alert_hash = self._generate_alert_hash(alert)
            
            if alert_hash in processed_hashes:
                continue
            
            # Find similar alerts
            similar_alerts = [a for a in alerts if self._are_alerts_similar(alert, a)]
            
            if len(similar_alerts) > 1:
                # Create aggregated alert
                aggregated_alert = self._create_aggregated_alert(similar_alerts)
                aggregated.append(aggregated_alert)
                processed_hashes.add(alert_hash)
                
                # Mark original alerts as aggregated
                for similar_alert in similar_alerts:
                    similar_alert.metadata['aggregated'] = True
                    similar_alert.metadata['aggregated_into'] = aggregated_alert.alert_id
            else:
                aggregated.append(alert)
                processed_hashes.add(alert_hash)
        
        return aggregated
    
    def _generate_alert_hash(self, alert: Alert) -> str:
        """Generate hash for alert deduplication."""
        hash_string = f"{alert.category.value}:{alert.priority.value}:{alert.source}:{alert.title}"
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _are_alerts_similar(self, alert1: Alert, alert2: Alert) -> bool:
        """Check if two alerts are similar for aggregation."""
        # Same category and priority
        if alert1.category != alert2.category or alert1.priority != alert2.priority:
            return False
        
        # Same source
        if alert1.source != alert2.source:
            return False
        
        # Similar titles (simple similarity check)
        title_similarity = self._calculate_similarity(alert1.title, alert2.title)
        if title_similarity < 0.8:  # 80% similarity threshold
            return False
        
        # Within aggregation window
        time_diff = abs((alert1.timestamp - alert2.timestamp).total_seconds())
        if time_diff > self.config.aggregation_window:
            return False
        
        return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple implementation)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_aggregated_alert(self, alerts: List[Alert]) -> Alert:
        """Create aggregated alert from similar alerts."""
        if not alerts:
            return None
        
        # Use the first alert as base
        base_alert = alerts[0]
        
        # Create aggregated alert
        aggregated_alert = Alert(
            alert_id=str(uuid.uuid4()),
            title=f"[AGGREGATED] {base_alert.title} ({len(alerts)} occurrences)",
            message=f"Multiple similar alerts detected:\n\n" + "\n".join([f"- {a.message}" for a in alerts[:3]]),
            category=base_alert.category,
            priority=base_alert.priority,
            source=base_alert.source,
            channels=base_alert.channels,
            recipients=base_alert.recipients,
            metadata={
                'aggregated_from': [a.alert_id for a in alerts],
                'occurrence_count': len(alerts),
                'first_occurrence': min(a.timestamp for a in alerts),
                'last_occurrence': max(a.timestamp for a in alerts)
            }
        )
        
        return aggregated_alert

class AlertRouter:
    """Alert routing and delivery system."""
    
    def __init__(self, config: AlertConfig):
        """
        Initialize alert router.
        
        Args:
            config: Alert system configuration
        """
        self.config = config
        self.delivery_handlers = {
            DeliveryChannel.EMAIL: self._deliver_email,
            DeliveryChannel.SMS: self._deliver_sms,
            DeliveryChannel.WEBHOOK: self._deliver_webhook,
            DeliveryChannel.DASHBOARD: self._deliver_dashboard,
            DeliveryChannel.PUSH: self._deliver_push,
            DeliveryChannel.SLACK: self._deliver_slack,
            DeliveryChannel.TELEGRAM: self._deliver_telegram
        }
        self.delivery_queue = queue.Queue()
        self.running = False
        self.delivery_thread = None
    
    def route_alert(self, alert: Alert) -> bool:
        """Route alert to appropriate delivery channels."""
        try:
            # Add to delivery queue
            self.delivery_queue.put(alert)
            logger.info("Alert queued for delivery", alert_id=alert.alert_id, 
                       channels=len(alert.channels))
            return True
        except Exception as e:
            logger.error("Failed to queue alert", alert_id=alert.alert_id, error=str(e))
            return False
    
    def start(self):
        """Start alert router."""
        if not self.running:
            self.running = True
            self.delivery_thread = threading.Thread(target=self._delivery_worker)
            self.delivery_thread.daemon = True
            self.delivery_thread.start()
            logger.info("Alert router started")
    
    def stop(self):
        """Stop alert router."""
        self.running = False
        if self.delivery_thread:
            self.delivery_thread.join()
        logger.info("Alert router stopped")
    
    def _delivery_worker(self):
        """Alert delivery worker."""
        while self.running:
            try:
                alert = self.delivery_queue.get(timeout=1)
                self._deliver_alert(alert)
                self.delivery_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Alert delivery error", error=str(e))
    
    def _deliver_alert(self, alert: Alert):
        """Deliver alert through all specified channels."""
        for channel in alert.channels:
            try:
                handler = self.delivery_handlers.get(channel)
                if handler:
                    success = handler(alert)
                    if success:
                        logger.info("Alert delivered", alert_id=alert.alert_id, channel=channel.value)
                    else:
                        logger.warning("Alert delivery failed", alert_id=alert.alert_id, channel=channel.value)
                        self._handle_delivery_failure(alert, channel)
                else:
                    logger.warning("No handler for channel", channel=channel.value)
            except Exception as e:
                logger.error("Channel delivery error", alert_id=alert.alert_id, 
                           channel=channel.value, error=str(e))
                self._handle_delivery_failure(alert, channel)
    
    def _deliver_email(self, alert: Alert) -> bool:
        """Deliver alert via email."""
        try:
            # This is a simplified email delivery
            # In production, you'd configure SMTP settings
            logger.info("Email alert prepared", 
                       recipients=alert.recipients,
                       subject=alert.title,
                       message=alert.message[:100] + "..." if len(alert.message) > 100 else alert.message)
            return True
        except Exception as e:
            logger.error("Email delivery error", error=str(e))
            return False
    
    def _deliver_sms(self, alert: Alert) -> bool:
        """Deliver alert via SMS."""
        try:
            # This is a simplified SMS delivery
            # In production, you'd integrate with SMS service
            logger.info("SMS alert prepared", 
                       recipients=alert.recipients,
                       message=alert.message[:160] + "..." if len(alert.message) > 160 else alert.message)
            return True
        except Exception as e:
            logger.error("SMS delivery error", error=str(e))
            return False
    
    def _deliver_webhook(self, alert: Alert) -> bool:
        """Deliver alert via webhook."""
        try:
            webhook_url = alert.metadata.get('webhook_url')
            if not webhook_url:
                return False
            
            payload = {
                'alert_id': alert.alert_id,
                'title': alert.title,
                'message': alert.message,
                'category': alert.category.value,
                'priority': alert.priority.value,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        
        except Exception as e:
            logger.error("Webhook delivery error", error=str(e))
            return False
    
    def _deliver_dashboard(self, alert: Alert) -> bool:
        """Deliver alert to dashboard."""
        try:
            # This would typically update a dashboard or send to WebSocket
            logger.info("Dashboard alert prepared", alert_id=alert.alert_id)
            return True
        except Exception as e:
            logger.error("Dashboard delivery error", error=str(e))
            return False
    
    def _deliver_push(self, alert: Alert) -> bool:
        """Deliver alert via push notification."""
        try:
            # This is a simplified push notification
            # In production, you'd integrate with push service
            logger.info("Push notification prepared", 
                       recipients=alert.recipients,
                       title=alert.title,
                       message=alert.message)
            return True
        except Exception as e:
            logger.error("Push delivery error", error=str(e))
            return False
    
    def _deliver_slack(self, alert: Alert) -> bool:
        """Deliver alert via Slack."""
        try:
            slack_webhook = alert.metadata.get('slack_webhook')
            if not slack_webhook:
                return False
            
            payload = {
                'text': f"*{alert.title}*\n{alert.message}",
                'attachments': [{
                    'color': self._get_priority_color(alert.priority),
                    'fields': [
                        {'title': 'Category', 'value': alert.category.value, 'short': True},
                        {'title': 'Priority', 'value': alert.priority.value, 'short': True},
                        {'title': 'Source', 'value': alert.source, 'short': True}
                    ]
                }]
            }
            
            response = requests.post(slack_webhook, json=payload, timeout=10)
            return response.status_code == 200
        
        except Exception as e:
            logger.error("Slack delivery error", error=str(e))
            return False
    
    def _deliver_telegram(self, alert: Alert) -> bool:
        """Deliver alert via Telegram."""
        try:
            telegram_bot_token = alert.metadata.get('telegram_bot_token')
            telegram_chat_id = alert.metadata.get('telegram_chat_id')
            
            if not telegram_bot_token or not telegram_chat_id:
                return False
            
            message = f"*{alert.title}*\n{alert.message}\n\nCategory: {alert.category.value}\nPriority: {alert.priority.value}"
            
            url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        
        except Exception as e:
            logger.error("Telegram delivery error", error=str(e))
            return False
    
    def _get_priority_color(self, priority: AlertPriority) -> str:
        """Get color for priority level."""
        colors = {
            AlertPriority.LOW: "#36a64f",
            AlertPriority.MEDIUM: "#ffa500",
            AlertPriority.HIGH: "#ff8c00",
            AlertPriority.CRITICAL: "#ff4500",
            AlertPriority.EMERGENCY: "#ff0000"
        }
        return colors.get(priority, "#808080")
    
    def _handle_delivery_failure(self, alert: Alert, channel: DeliveryChannel):
        """Handle delivery failure."""
        alert.retry_count += 1
        
        if alert.retry_count < alert.max_retries:
            # Re-queue for retry
            time.sleep(self.config.retry_delay)
            self.delivery_queue.put(alert)
            logger.info("Alert re-queued for retry", alert_id=alert.alert_id, 
                       channel=channel.value, retry_count=alert.retry_count)
        else:
            alert.status = AlertStatus.FAILED
            logger.error("Alert delivery failed after max retries", alert_id=alert.alert_id, 
                        channel=channel.value)

class AlertEscalator:
    """Alert escalation system."""
    
    def __init__(self, config: AlertConfig):
        """
        Initialize alert escalator.
        
        Args:
            config: Alert system configuration
        """
        self.config = config
        self.escalation_rules = {}
        self.escalation_timers = {}
        self.escalation_callbacks = []
    
    def add_escalation_rule(self, rule_id: str, rule_config: Dict[str, Any]):
        """Add escalation rule."""
        self.escalation_rules[rule_id] = rule_config
        logger.info("Escalation rule added", rule_id=rule_id)
    
    def start_escalation_timer(self, alert: Alert):
        """Start escalation timer for alert."""
        if not alert.metadata.get('escalation_rules'):
            return
        
        timer_key = f"{alert.alert_id}_{alert.escalation_level}"
        if timer_key in self.escalation_timers:
            return
        
        # Start timer
        timer = threading.Timer(self.config.escalation_timeout, 
                              self._escalate_alert, args=[alert])
        timer.daemon = True
        timer.start()
        
        self.escalation_timers[timer_key] = timer
        logger.info("Escalation timer started", alert_id=alert.alert_id, 
                   timeout=self.config.escalation_timeout)
    
    def cancel_escalation_timer(self, alert: Alert):
        """Cancel escalation timer for alert."""
        timer_key = f"{alert.alert_id}_{alert.escalation_level}"
        if timer_key in self.escalation_timers:
            timer = self.escalation_timers[timer_key]
            timer.cancel()
            del self.escalation_timers[timer_key]
            logger.info("Escalation timer cancelled", alert_id=alert.alert_id)
    
    def _escalate_alert(self, alert: Alert):
        """Escalate alert."""
        try:
            alert.escalation_level += 1
            alert.status = AlertStatus.ESCALATED
            
            # Apply escalation rules
            escalation_rules = alert.metadata.get('escalation_rules', [])
            for rule in escalation_rules:
                if rule.get('level') == alert.escalation_level:
                    self._apply_escalation_rule(alert, rule)
            
            # Call escalation callbacks
            for callback in self.escalation_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error("Escalation callback error", error=str(e))
            
            logger.warning("Alert escalated", alert_id=alert.alert_id, 
                          escalation_level=alert.escalation_level)
        
        except Exception as e:
            logger.error("Alert escalation error", alert_id=alert.alert_id, error=str(e))
    
    def _apply_escalation_rule(self, alert: Alert, rule: Dict[str, Any]):
        """Apply escalation rule to alert."""
        # Update recipients
        if 'recipients' in rule:
            alert.recipients.extend(rule['recipients'])
        
        # Update channels
        if 'channels' in rule:
            alert.channels.extend(rule['channels'])
        
        # Update priority
        if 'priority' in rule:
            alert.priority = AlertPriority(rule['priority'])
        
        # Add custom actions
        if 'actions' in rule:
            alert.metadata['escalation_actions'] = rule['actions']
    
    def add_escalation_callback(self, callback: Callable[[Alert], None]):
        """Add escalation callback."""
        self.escalation_callbacks.append(callback)

class AlertAnalytics:
    """Alert analytics and reporting system."""
    
    def __init__(self, config: AlertConfig):
        """
        Initialize alert analytics.
        
        Args:
            config: Alert system configuration
        """
        self.config = config
        self.alert_history = deque(maxlen=10000)
        self.analytics_data = defaultdict(lambda: {
            'total_alerts': 0,
            'by_category': defaultdict(int),
            'by_priority': defaultdict(int),
            'by_status': defaultdict(int),
            'delivery_success_rate': 0.0,
            'average_response_time': 0.0,
            'escalation_count': 0
        })
    
    def record_alert(self, alert: Alert):
        """Record alert for analytics."""
        self.alert_history.append(alert)
        self._update_analytics(alert)
    
    def get_analytics_summary(self, time_window: int = 86400) -> Dict[str, Any]:
        """Get analytics summary for time window."""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
        
        if not recent_alerts:
            return self._get_empty_analytics()
        
        summary = {
            'time_window_seconds': time_window,
            'total_alerts': len(recent_alerts),
            'by_category': defaultdict(int),
            'by_priority': defaultdict(int),
            'by_status': defaultdict(int),
            'delivery_success_rate': 0.0,
            'average_response_time': 0.0,
            'escalation_count': 0,
            'top_sources': [],
            'alert_trends': []
        }
        
        # Calculate metrics
        successful_deliveries = 0
        total_response_time = 0
        response_count = 0
        
        for alert in recent_alerts:
            summary['by_category'][alert.category.value] += 1
            summary['by_priority'][alert.priority.value] += 1
            summary['by_status'][alert.status.value] += 1
            
            if alert.status == AlertStatus.DELIVERED:
                successful_deliveries += 1
            
            if alert.acknowledged_at:
                response_time = (alert.acknowledged_at - alert.timestamp).total_seconds()
                total_response_time += response_time
                response_count += 1
            
            if alert.status == AlertStatus.ESCALATED:
                summary['escalation_count'] += 1
        
        # Calculate rates
        if recent_alerts:
            summary['delivery_success_rate'] = successful_deliveries / len(recent_alerts)
        
        if response_count > 0:
            summary['average_response_time'] = total_response_time / response_count
        
        # Top sources
        source_counts = defaultdict(int)
        for alert in recent_alerts:
            source_counts[alert.source] += 1
        
        summary['top_sources'] = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return summary
    
    def _update_analytics(self, alert: Alert):
        """Update analytics data."""
        analytics = self.analytics_data['overall']
        analytics['total_alerts'] += 1
        analytics['by_category'][alert.category.value] += 1
        analytics['by_priority'][alert.priority.value] += 1
        analytics['by_status'][alert.status.value] += 1
        
        if alert.status == AlertStatus.ESCALATED:
            analytics['escalation_count'] += 1
    
    def _get_empty_analytics(self) -> Dict[str, Any]:
        """Get empty analytics structure."""
        return {
            'time_window_seconds': 0,
            'total_alerts': 0,
            'by_category': {},
            'by_priority': {},
            'by_status': {},
            'delivery_success_rate': 0.0,
            'average_response_time': 0.0,
            'escalation_count': 0,
            'top_sources': [],
            'alert_trends': []
        }

class AlertSystem:
    """Main real-time alert system."""
    
    def __init__(self, config: AlertConfig):
        """
        Initialize alert system.
        
        Args:
            config: Alert system configuration
        """
        self.config = config
        self.aggregator = AlertAggregator(config)
        self.router = AlertRouter(config)
        self.escalator = AlertEscalator(config)
        self.analytics = AlertAnalytics(config)
        
        self.alerts = {}
        self.rules = {}
        self.running = False
        
        # Setup escalation callbacks
        self.escalator.add_escalation_callback(self._on_alert_escalation)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.rules[rule.rule_id] = rule
        logger.info("Alert rule added", rule_id=rule.rule_id, name=rule.name)
    
    def create_alert(self, title: str, message: str, category: AlertCategory, 
                    priority: AlertPriority, source: str, channels: List[DeliveryChannel] = None,
                    recipients: List[str] = None, metadata: Dict[str, Any] = None) -> Alert:
        """Create and send alert."""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            title=title,
            message=message,
            category=category,
            priority=priority,
            source=source,
            channels=channels or self.config.default_channels,
            recipients=recipients or self.config.default_recipients,
            metadata=metadata or {}
        )
        
        # Store alert
        self.alerts[alert.alert_id] = alert
        
        # Record for analytics
        self.analytics.record_alert(alert)
        
        # Start escalation timer
        self.escalator.start_escalation_timer(alert)
        
        # Route for delivery
        self.router.route_alert(alert)
        
        logger.info("Alert created", alert_id=alert.alert_id, title=title, 
                   category=category.value, priority=priority.value)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge alert."""
        if alert_id not in self.alerts:
            logger.warning("Alert not found for acknowledgment", alert_id=alert_id)
            return
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        
        # Cancel escalation timer
        self.escalator.cancel_escalation_timer(alert)
        
        logger.info("Alert acknowledged", alert_id=alert_id, acknowledged_by=acknowledged_by)
    
    def resolve_alert(self, alert_id: str, resolved_by: str):
        """Resolve alert."""
        if alert_id not in self.alerts:
            logger.warning("Alert not found for resolution", alert_id=alert_id)
            return
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.metadata['resolved_by'] = resolved_by
        alert.metadata['resolved_at'] = datetime.now()
        
        # Cancel escalation timer
        self.escalator.cancel_escalation_timer(alert)
        
        logger.info("Alert resolved", alert_id=alert_id, resolved_by=resolved_by)
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.alerts.get(alert_id)
    
    def get_alerts(self, status: AlertStatus = None, category: AlertCategory = None,
                  priority: AlertPriority = None) -> List[Alert]:
        """Get alerts with optional filtering."""
        alerts = list(self.alerts.values())
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_analytics(self, time_window: int = 86400) -> Dict[str, Any]:
        """Get alert analytics."""
        return self.analytics.get_analytics_summary(time_window)
    
    def start(self):
        """Start alert system."""
        if not self.running:
            self.running = True
            self.router.start()
            logger.info("Alert system started")
    
    def stop(self):
        """Stop alert system."""
        self.running = False
        self.router.stop()
        logger.info("Alert system stopped")
    
    def _on_alert_escalation(self, alert: Alert):
        """Handle alert escalation."""
        logger.warning("Alert escalated", alert_id=alert.alert_id, 
                      escalation_level=alert.escalation_level)
        
        # Re-route escalated alert
        self.router.route_alert(alert)

def create_alert_system(config: AlertConfig = None) -> AlertSystem:
    """
    Create a real-time alert system.
    
    Args:
        config: Alert system configuration
        
    Returns:
        AlertSystem instance
    """
    if config is None:
        config = AlertConfig()
    
    return AlertSystem(config)

if __name__ == "__main__":
    # Demo of alert system
    config = AlertConfig(
        default_channels=[DeliveryChannel.DASHBOARD, DeliveryChannel.EMAIL],
        default_recipients=["admin@example.com"],
        escalation_timeout=300,
        max_retries=3,
        retry_delay=60,
        aggregation_window=300,
        deduplication_enabled=True,
        analytics_enabled=True
    )
    
    alert_system = create_alert_system(config)
    
    # Add alert rule
    rule = AlertRule(
        rule_id="rule_001",
        name="High Priority Trading Alert",
        description="Alert for high priority trading events",
        category=AlertCategory.TRADING,
        priority=AlertPriority.HIGH,
        conditions={},
        channels=[DeliveryChannel.EMAIL, DeliveryChannel.SLACK],
        recipients=["trader@example.com", "manager@example.com"]
    )
    alert_system.add_alert_rule(rule)
    
    # Start alert system
    alert_system.start()
    
    print("Alert system created successfully!")
    print(f"Default channels: {[c.value for c in config.default_channels]}")
    print(f"Default recipients: {config.default_recipients}")
    
    # Create sample alert
    alert = alert_system.create_alert(
        title="High Volume Trading Detected",
        message="Unusual trading volume detected for AAPL",
        category=AlertCategory.TRADING,
        priority=AlertPriority.HIGH,
        source="trading_system",
        metadata={'symbol': 'AAPL', 'volume': 1000000}
    )
    
    print(f"Alert created: {alert.alert_id}")
    
    # Get analytics
    analytics = alert_system.get_analytics()
    print(f"Analytics: {analytics}")