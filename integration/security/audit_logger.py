"""
Audit Logging System for ML Stock Predictor Platform

This module provides comprehensive audit logging functionality
for tracking all security-relevant events in the trading system.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import hashlib
import uuid
from pathlib import Path
import threading
import queue
import time

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    timestamp: str
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    severity: str
    outcome: str
    metadata: Dict[str, Any]


class AuditLogger:
    """
    Comprehensive audit logging system for security events.
    
    Features:
    - Structured audit logging
    - Real-time event processing
    - Secure log storage
    - Event correlation
    - Compliance reporting
    """
    
    def __init__(self, log_dir: str = "logs/audit", max_file_size: int = 100 * 1024 * 1024):
        """
        Initialize the audit logger.
        
        Args:
            log_dir: Directory for audit logs
            max_file_size: Maximum size of log files in bytes
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self.event_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self._setup_logging()
        self._start_worker()
        
    def _setup_logging(self):
        """Setup audit log file handlers."""
        try:
            # Create audit log file
            log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
            
            # Configure file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            audit_logger = logging.getLogger('audit')
            audit_logger.addHandler(file_handler)
            audit_logger.setLevel(logging.INFO)
            
            self.audit_logger = audit_logger
            logger.info(f"Audit logging initialized: {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to setup audit logging: {e}")
            raise
    
    def _start_worker(self):
        """Start the background worker thread for processing audit events."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self.worker_thread.start()
        logger.info("Audit event processor started")
    
    def _process_events(self):
        """Background worker to process audit events."""
        while self.running:
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1)
                self._write_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")
    
    def _write_event(self, event: AuditEvent):
        """Write audit event to log file."""
        try:
            # Convert event to JSON
            event_dict = asdict(event)
            event_json = json.dumps(event_dict, default=str)
            
            # Write to audit log
            self.audit_logger.info(f"AUDIT_EVENT: {event_json}")
            
            # Check file size and rotate if needed
            self._check_file_rotation()
            
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")
    
    def _check_file_rotation(self):
        """Check if log file needs rotation."""
        try:
            log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
            if log_file.exists() and log_file.stat().st_size > self.max_file_size:
                # Create backup file
                backup_file = log_file.with_suffix(f'.log.{int(time.time())}')
                log_file.rename(backup_file)
                logger.info(f"Audit log rotated: {backup_file}")
                
                # Recreate log file
                self._setup_logging()
                
        except Exception as e:
            logger.error(f"Failed to rotate audit log: {e}")
    
    def log_event(self, 
                  event_type: str,
                  action: str,
                  resource: str,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  severity: str = "INFO",
                  outcome: str = "SUCCESS",
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (LOGIN, LOGOUT, DATA_ACCESS, etc.)
            action: Action performed
            resource: Resource accessed
            user_id: User ID (if applicable)
            session_id: Session ID (if applicable)
            ip_address: IP address of the request
            user_agent: User agent string
            details: Additional event details
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
            outcome: Event outcome (SUCCESS, FAILURE, DENIED)
            metadata: Additional metadata
        """
        try:
            # Create audit event
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                action=action,
                resource=resource,
                details=details or {},
                severity=severity,
                outcome=outcome,
                metadata=metadata or {}
            )
            
            # Add to processing queue
            self.event_queue.put(event)
            
            # Log to console for immediate visibility
            logger.info(f"Audit event queued: {event_type} - {action} - {resource}")
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    def log_login(self, user_id: str, ip_address: str, user_agent: str, 
                  success: bool = True, details: Optional[Dict[str, Any]] = None):
        """Log user login event."""
        self.log_event(
            event_type="AUTHENTICATION",
            action="LOGIN",
            resource="SYSTEM",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity="INFO" if success else "WARNING",
            outcome="SUCCESS" if success else "FAILURE"
        )
    
    def log_logout(self, user_id: str, session_id: str, ip_address: str):
        """Log user logout event."""
        self.log_event(
            event_type="AUTHENTICATION",
            action="LOGOUT",
            resource="SYSTEM",
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            severity="INFO",
            outcome="SUCCESS"
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str,
                       ip_address: str, success: bool = True, details: Optional[Dict[str, Any]] = None):
        """Log data access event."""
        self.log_event(
            event_type="DATA_ACCESS",
            action=action,
            resource=resource,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            severity="INFO" if success else "WARNING",
            outcome="SUCCESS" if success else "DENIED"
        )
    
    def log_trading_action(self, user_id: str, action: str, symbol: str, 
                          amount: float, ip_address: str, success: bool = True):
        """Log trading action event."""
        details = {
            "symbol": symbol,
            "amount": amount,
            "action_type": action
        }
        
        self.log_event(
            event_type="TRADING",
            action=action,
            resource=f"TRADE_{symbol}",
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            severity="INFO" if success else "ERROR",
            outcome="SUCCESS" if success else "FAILURE"
        )
    
    def log_configuration_change(self, user_id: str, config_key: str, 
                               old_value: Any, new_value: Any, ip_address: str):
        """Log configuration change event."""
        details = {
            "config_key": config_key,
            "old_value": str(old_value),
            "new_value": str(new_value)
        }
        
        self.log_event(
            event_type="CONFIGURATION",
            action="UPDATE",
            resource=f"CONFIG_{config_key}",
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            severity="INFO",
            outcome="SUCCESS"
        )
    
    def log_security_event(self, event_type: str, action: str, resource: str,
                          ip_address: str, severity: str = "WARNING", 
                          details: Optional[Dict[str, Any]] = None):
        """Log security-related event."""
        self.log_event(
            event_type="SECURITY",
            action=action,
            resource=resource,
            ip_address=ip_address,
            details=details,
            severity=severity,
            outcome="DETECTED"
        )
    
    def log_api_access(self, user_id: Optional[str], endpoint: str, method: str,
                      ip_address: str, status_code: int, response_time: float):
        """Log API access event."""
        details = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time
        }
        
        severity = "INFO"
        if status_code >= 400:
            severity = "WARNING"
        if status_code >= 500:
            severity = "ERROR"
        
        self.log_event(
            event_type="API_ACCESS",
            action=f"{method} {endpoint}",
            resource="API",
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            severity=severity,
            outcome="SUCCESS" if status_code < 400 else "FAILURE"
        )
    
    def get_events(self, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   event_type: Optional[str] = None,
                   user_id: Optional[str] = None,
                   severity: Optional[str] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve audit events with filtering.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            event_type: Event type filter
            user_id: User ID filter
            severity: Severity filter
            limit: Maximum number of events to return
            
        Returns:
            List of audit events
        """
        events = []
        try:
            # Read from today's log file
            log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
            
            if not log_file.exists():
                return events
            
            with open(log_file, 'r') as f:
                for line in f:
                    if "AUDIT_EVENT:" in line:
                        try:
                            # Extract JSON from log line
                            json_start = line.find("AUDIT_EVENT:") + len("AUDIT_EVENT:")
                            event_json = line[json_start:].strip()
                            event = json.loads(event_json)
                            
                            # Apply filters
                            if self._matches_filter(event, start_time, end_time, 
                                                  event_type, user_id, severity):
                                events.append(event)
                                
                                if len(events) >= limit:
                                    break
                                    
                        except json.JSONDecodeError:
                            continue
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to retrieve audit events: {e}")
        
        return events
    
    def _matches_filter(self, event: Dict[str, Any], 
                       start_time: Optional[datetime],
                       end_time: Optional[datetime],
                       event_type: Optional[str],
                       user_id: Optional[str],
                       severity: Optional[str]) -> bool:
        """Check if event matches the specified filters."""
        try:
            # Time filter
            if start_time or end_time:
                event_time = datetime.fromisoformat(event.get('timestamp', '').replace('Z', '+00:00'))
                if start_time and event_time < start_time:
                    return False
                if end_time and event_time > end_time:
                    return False
            
            # Event type filter
            if event_type and event.get('event_type') != event_type:
                return False
            
            # User ID filter
            if user_id and event.get('user_id') != user_id:
                return False
            
            # Severity filter
            if severity and event.get('severity') != severity:
                return False
            
            return True
            
        except Exception:
            return False
    
    def generate_report(self, 
                       start_time: datetime,
                       end_time: datetime,
                       report_type: str = "summary") -> Dict[str, Any]:
        """
        Generate audit report.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            report_type: Type of report (summary, detailed, security)
            
        Returns:
            Report data
        """
        try:
            events = self.get_events(start_time, end_time)
            
            if report_type == "summary":
                return self._generate_summary_report(events)
            elif report_type == "security":
                return self._generate_security_report(events)
            else:
                return self._generate_detailed_report(events)
                
        except Exception as e:
            logger.error(f"Failed to generate audit report: {e}")
            return {"error": str(e)}
    
    def _generate_summary_report(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary audit report."""
        report = {
            "total_events": len(events),
            "event_types": {},
            "severity_distribution": {},
            "outcome_distribution": {},
            "top_users": {},
            "top_resources": {}
        }
        
        for event in events:
            # Event types
            event_type = event.get('event_type', 'UNKNOWN')
            report["event_types"][event_type] = report["event_types"].get(event_type, 0) + 1
            
            # Severity distribution
            severity = event.get('severity', 'UNKNOWN')
            report["severity_distribution"][severity] = report["severity_distribution"].get(severity, 0) + 1
            
            # Outcome distribution
            outcome = event.get('outcome', 'UNKNOWN')
            report["outcome_distribution"][outcome] = report["outcome_distribution"].get(outcome, 0) + 1
            
            # Top users
            user_id = event.get('user_id')
            if user_id:
                report["top_users"][user_id] = report["top_users"].get(user_id, 0) + 1
            
            # Top resources
            resource = event.get('resource', 'UNKNOWN')
            report["top_resources"][resource] = report["top_resources"].get(resource, 0) + 1
        
        return report
    
    def _generate_security_report(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate security-focused audit report."""
        security_events = [e for e in events if e.get('event_type') == 'SECURITY']
        
        report = {
            "total_security_events": len(security_events),
            "security_events_by_severity": {},
            "suspicious_ips": {},
            "failed_authentication_attempts": 0,
            "data_access_violations": 0,
            "api_abuse_attempts": 0
        }
        
        for event in security_events:
            severity = event.get('severity', 'UNKNOWN')
            report["security_events_by_severity"][severity] = report["security_events_by_severity"].get(severity, 0) + 1
            
            ip_address = event.get('ip_address')
            if ip_address:
                report["suspicious_ips"][ip_address] = report["suspicious_ips"].get(ip_address, 0) + 1
            
            action = event.get('action', '')
            if 'LOGIN' in action and event.get('outcome') == 'FAILURE':
                report["failed_authentication_attempts"] += 1
            elif 'DATA_ACCESS' in action and event.get('outcome') == 'DENIED':
                report["data_access_violations"] += 1
            elif 'API' in action and event.get('outcome') == 'FAILURE':
                report["api_abuse_attempts"] += 1
        
        return report
    
    def _generate_detailed_report(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed audit report."""
        return {
            "total_events": len(events),
            "events": events[:1000],  # Limit to first 1000 events
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def shutdown(self):
        """Shutdown the audit logger."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Audit logger shutdown complete")


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# Example usage and testing
if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Test audit logging
    audit_logger = AuditLogger()
    
    # Log various events
    audit_logger.log_login("user123", "192.168.1.100", "Mozilla/5.0", True)
    audit_logger.log_data_access("user123", "trading_data", "READ", "192.168.1.100", True)
    audit_logger.log_trading_action("user123", "BUY", "AAPL", 100.0, "192.168.1.100", True)
    audit_logger.log_security_event("BRUTE_FORCE", "LOGIN_ATTEMPT", "AUTH", "192.168.1.101", "WARNING")
    
    # Wait for events to be processed
    time.sleep(2)
    
    # Generate report
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    report = audit_logger.generate_report(start_time, end_time, "summary")
    print(f"Audit Report: {json.dumps(report, indent=2)}")
    
    # Shutdown
    audit_logger.shutdown()
    
    print("Audit logging system test completed successfully!")