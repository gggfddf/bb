"""
Log Aggregator for Trading System

This module provides log aggregation and analysis functionality for the trading system,
collecting logs from multiple sources and providing analysis capabilities.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import re
import sqlite3
from pathlib import Path
import queue
import hashlib

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogSource(Enum):
    """Log source enumeration"""
    API = "api"
    DATABASE = "database"
    TRADING = "trading"
    SECURITY = "security"
    MONITORING = "monitoring"
    SYSTEM = "system"
    USER = "user"

@dataclass
class LogEntry:
    """Log entry structure"""
    log_id: str
    timestamp: datetime
    level: LogLevel
    source: LogSource
    component: str
    message: str
    details: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    request_id: Optional[str]
    duration: Optional[float]
    tags: List[str]

@dataclass
class LogAnalysis:
    """Log analysis result"""
    analysis_id: str
    timestamp: datetime
    analysis_type: str
    time_range: Dict[str, datetime]
    results: Dict[str, Any]
    summary: Dict[str, Any]

class LogAggregator:
    """
    Log aggregator for collecting and analyzing logs from multiple sources.
    """
    
    def __init__(self, db_path: str = "logs/aggregator.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Log processing
        self.log_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        # Analysis cache
        self.analysis_cache: Dict[str, LogAnalysis] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Statistics
        self.stats = {
            'total_logs': 0,
            'logs_by_level': {},
            'logs_by_source': {},
            'logs_by_component': {},
            'last_processed': None
        }
        
        # Initialize database
        self._init_database()
        
        # Start processing thread
        self.start()
        
        logger.info("Log aggregator initialized")
    
    def start(self):
        """Start the log aggregator"""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_logs, daemon=True)
            self.processing_thread.start()
            logger.info("Log aggregator started")
    
    def stop(self):
        """Stop the log aggregator"""
        if self.running:
            self.running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=5)
            logger.info("Log aggregator stopped")
    
    def add_log(self, log_entry: LogEntry):
        """
        Add a log entry to the aggregator.
        
        Args:
            log_entry: Log entry to add
        """
        try:
            self.log_queue.put(log_entry)
        except Exception as e:
            logger.error(f"Failed to add log entry: {e}")
    
    def add_log_from_dict(self, log_data: Dict[str, Any]):
        """
        Add a log entry from dictionary data.
        
        Args:
            log_data: Log data dictionary
        """
        try:
            log_entry = LogEntry(
                log_id=log_data.get('log_id', self._generate_log_id()),
                timestamp=datetime.fromisoformat(log_data['timestamp']) if isinstance(log_data['timestamp'], str) else log_data['timestamp'],
                level=LogLevel(log_data['level']),
                source=LogSource(log_data['source']),
                component=log_data['component'],
                message=log_data['message'],
                details=log_data.get('details', {}),
                user_id=log_data.get('user_id'),
                session_id=log_data.get('session_id'),
                ip_address=log_data.get('ip_address'),
                request_id=log_data.get('request_id'),
                duration=log_data.get('duration'),
                tags=log_data.get('tags', [])
            )
            self.add_log(log_entry)
        except Exception as e:
            logger.error(f"Failed to create log entry from dict: {e}")
    
    def get_logs(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                 level: Optional[LogLevel] = None, source: Optional[LogSource] = None,
                 component: Optional[str] = None, limit: int = 1000) -> List[LogEntry]:
        """
        Get logs with filters.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            level: Log level filter
            source: Log source filter
            component: Component filter
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM logs WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                if level:
                    query += " AND level = ?"
                    params.append(level.value)
                
                if source:
                    query += " AND source = ?"
                    params.append(source.value)
                
                if component:
                    query += " AND component = ?"
                    params.append(component)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_log_entry(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return []
    
    def analyze_logs(self, analysis_type: str, start_time: datetime, end_time: datetime,
                    filters: Optional[Dict[str, Any]] = None) -> LogAnalysis:
        """
        Analyze logs for a specific time range.
        
        Args:
            analysis_type: Type of analysis to perform
            start_time: Start time for analysis
            end_time: End time for analysis
            filters: Optional filters
            
        Returns:
            Log analysis result
        """
        # Check cache first
        cache_key = f"{analysis_type}_{start_time.isoformat()}_{end_time.isoformat()}"
        if cache_key in self.analysis_cache:
            cached_analysis = self.analysis_cache[cache_key]
            if (datetime.utcnow() - cached_analysis.timestamp).total_seconds() < self.cache_ttl:
                return cached_analysis
        
        try:
            if analysis_type == "error_analysis":
                results = self._analyze_errors(start_time, end_time, filters)
            elif analysis_type == "performance_analysis":
                results = self._analyze_performance(start_time, end_time, filters)
            elif analysis_type == "user_activity":
                results = self._analyze_user_activity(start_time, end_time, filters)
            elif analysis_type == "security_analysis":
                results = self._analyze_security(start_time, end_time, filters)
            else:
                results = self._analyze_general(start_time, end_time, filters)
            
            analysis = LogAnalysis(
                analysis_id=self._generate_analysis_id(),
                timestamp=datetime.utcnow(),
                analysis_type=analysis_type,
                time_range={'start': start_time, 'end': end_time},
                results=results,
                summary=self._create_summary(results, analysis_type)
            )
            
            # Cache the analysis
            self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze logs: {e}")
            return LogAnalysis(
                analysis_id=self._generate_analysis_id(),
                timestamp=datetime.utcnow(),
                analysis_type=analysis_type,
                time_range={'start': start_time, 'end': end_time},
                results={},
                summary={'error': str(e)}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregator statistics.
        
        Returns:
            Aggregator statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get total logs
                total_logs = conn.execute("SELECT COUNT(*) as count FROM logs").fetchone()['count']
                
                # Get logs by level
                level_stats = {}
                for row in conn.execute("SELECT level, COUNT(*) as count FROM logs GROUP BY level"):
                    level_stats[row['level']] = row['count']
                
                # Get logs by source
                source_stats = {}
                for row in conn.execute("SELECT source, COUNT(*) as count FROM logs GROUP BY source"):
                    source_stats[row['source']] = row['count']
                
                # Get logs by component
                component_stats = {}
                for row in conn.execute("SELECT component, COUNT(*) as count FROM logs GROUP BY component LIMIT 20"):
                    component_stats[row['component']] = row['count']
                
                return {
                    'total_logs': total_logs,
                    'logs_by_level': level_stats,
                    'logs_by_source': source_stats,
                    'logs_by_component': component_stats,
                    'last_processed': self.stats['last_processed'].isoformat() if self.stats['last_processed'] else None,
                    'queue_size': self.log_queue.qsize()
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def search_logs(self, search_term: str, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None, limit: int = 100) -> List[LogEntry]:
        """
        Search logs by text.
        
        Args:
            search_term: Search term
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of results
            
        Returns:
            List of matching log entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM logs WHERE (message LIKE ? OR details LIKE ?)"
                params = [f"%{search_term}%", f"%{search_term}%"]
                
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
                
                return [self._row_to_log_entry(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search logs: {e}")
            return []
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Clean up old logs.
        
        Args:
            days_to_keep: Number of days to keep logs
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                deleted_count = conn.execute(
                    "DELETE FROM logs WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                ).rowcount
                
                conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old log entries")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
    
    def _init_database(self):
        """Initialize the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS logs (
                        log_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        source TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        user_id TEXT,
                        session_id TEXT,
                        ip_address TEXT,
                        request_id TEXT,
                        duration REAL,
                        tags TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_source ON logs(source)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_component ON logs(component)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_user_id ON logs(user_id)")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _process_logs(self):
        """Process logs from the queue"""
        while self.running:
            try:
                # Process logs in batches
                logs_to_process = []
                
                # Get up to 100 logs from queue
                for _ in range(100):
                    try:
                        log_entry = self.log_queue.get(timeout=1)
                        logs_to_process.append(log_entry)
                    except queue.Empty:
                        break
                
                if logs_to_process:
                    self._store_logs(logs_to_process)
                    self.stats['last_processed'] = datetime.utcnow()
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in log processing: {e}")
                time.sleep(1)
    
    def _store_logs(self, log_entries: List[LogEntry]):
        """Store log entries in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for log_entry in log_entries:
                    conn.execute("""
                        INSERT INTO logs (
                            log_id, timestamp, level, source, component, message,
                            details, user_id, session_id, ip_address, request_id,
                            duration, tags
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        log_entry.log_id,
                        log_entry.timestamp.isoformat(),
                        log_entry.level.value,
                        log_entry.source.value,
                        log_entry.component,
                        log_entry.message,
                        json.dumps(log_entry.details),
                        log_entry.user_id,
                        log_entry.session_id,
                        log_entry.ip_address,
                        log_entry.request_id,
                        log_entry.duration,
                        json.dumps(log_entry.tags)
                    ))
                
                conn.commit()
                
                # Update statistics
                self.stats['total_logs'] += len(log_entries)
                
        except Exception as e:
            logger.error(f"Failed to store logs: {e}")
    
    def _row_to_log_entry(self, row) -> LogEntry:
        """Convert database row to log entry"""
        return LogEntry(
            log_id=row['log_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            level=LogLevel(row['level']),
            source=LogSource(row['source']),
            component=row['component'],
            message=row['message'],
            details=json.loads(row['details']) if row['details'] else {},
            user_id=row['user_id'],
            session_id=row['session_id'],
            ip_address=row['ip_address'],
            request_id=row['request_id'],
            duration=row['duration'],
            tags=json.loads(row['tags']) if row['tags'] else []
        )
    
    def _analyze_errors(self, start_time: datetime, end_time: datetime, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error logs"""
        error_logs = self.get_logs(start_time, end_time, level=LogLevel.ERROR)
        
        error_counts = {}
        error_messages = {}
        component_errors = {}
        
        for log in error_logs:
            # Count by error type
            error_type = log.details.get('error_type', 'unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Count by message
            error_messages[log.message] = error_messages.get(log.message, 0) + 1
            
            # Count by component
            component_errors[log.component] = component_errors.get(log.component, 0) + 1
        
        return {
            'total_errors': len(error_logs),
            'error_counts': error_counts,
            'error_messages': error_messages,
            'component_errors': component_errors
        }
    
    def _analyze_performance(self, start_time: datetime, end_time: datetime, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance logs"""
        performance_logs = self.get_logs(start_time, end_time)
        
        durations = []
        component_performance = {}
        
        for log in performance_logs:
            if log.duration is not None:
                durations.append(log.duration)
                
                if log.component not in component_performance:
                    component_performance[log.component] = []
                component_performance[log.component].append(log.duration)
        
        # Calculate statistics
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
        else:
            avg_duration = max_duration = min_duration = 0
        
        # Component performance
        component_stats = {}
        for component, comp_durations in component_performance.items():
            if comp_durations:
                component_stats[component] = {
                    'avg_duration': sum(comp_durations) / len(comp_durations),
                    'max_duration': max(comp_durations),
                    'min_duration': min(comp_durations),
                    'count': len(comp_durations)
                }
        
        return {
            'total_requests': len(durations),
            'avg_duration': avg_duration,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'component_performance': component_stats
        }
    
    def _analyze_user_activity(self, start_time: datetime, end_time: datetime, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user activity logs"""
        user_logs = self.get_logs(start_time, end_time)
        
        user_activity = {}
        session_activity = {}
        ip_activity = {}
        
        for log in user_logs:
            if log.user_id:
                if log.user_id not in user_activity:
                    user_activity[log.user_id] = {'count': 0, 'last_activity': None}
                user_activity[log.user_id]['count'] += 1
                user_activity[log.user_id]['last_activity'] = log.timestamp
            
            if log.session_id:
                session_activity[log.session_id] = session_activity.get(log.session_id, 0) + 1
            
            if log.ip_address:
                ip_activity[log.ip_address] = ip_activity.get(log.ip_address, 0) + 1
        
        return {
            'total_users': len(user_activity),
            'total_sessions': len(session_activity),
            'total_ips': len(ip_activity),
            'user_activity': user_activity,
            'session_activity': session_activity,
            'ip_activity': ip_activity
        }
    
    def _analyze_security(self, start_time: datetime, end_time: datetime, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze security logs"""
        security_logs = self.get_logs(start_time, end_time, source=LogSource.SECURITY)
        
        security_events = {}
        failed_attempts = {}
        suspicious_ips = {}
        
        for log in security_logs:
            event_type = log.details.get('event_type', 'unknown')
            security_events[event_type] = security_events.get(event_type, 0) + 1
            
            if 'failed' in log.message.lower() or log.level in [LogLevel.ERROR, LogLevel.WARNING]:
                if log.ip_address:
                    failed_attempts[log.ip_address] = failed_attempts.get(log.ip_address, 0) + 1
            
            if log.details.get('suspicious', False):
                if log.ip_address:
                    suspicious_ips[log.ip_address] = suspicious_ips.get(log.ip_address, 0) + 1
        
        return {
            'total_security_events': len(security_logs),
            'security_events': security_events,
            'failed_attempts': failed_attempts,
            'suspicious_ips': suspicious_ips
        }
    
    def _analyze_general(self, start_time: datetime, end_time: datetime, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """General log analysis"""
        logs = self.get_logs(start_time, end_time)
        
        level_counts = {}
        source_counts = {}
        component_counts = {}
        
        for log in logs:
            level_counts[log.level.value] = level_counts.get(log.level.value, 0) + 1
            source_counts[log.source.value] = source_counts.get(log.source.value, 0) + 1
            component_counts[log.component] = component_counts.get(log.component, 0) + 1
        
        return {
            'total_logs': len(logs),
            'level_counts': level_counts,
            'source_counts': source_counts,
            'component_counts': component_counts
        }
    
    def _create_summary(self, results: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Create summary from analysis results"""
        if analysis_type == "error_analysis":
            return {
                'total_errors': results.get('total_errors', 0),
                'most_common_error': max(results.get('error_counts', {}).items(), key=lambda x: x[1])[0] if results.get('error_counts') else None,
                'most_affected_component': max(results.get('component_errors', {}).items(), key=lambda x: x[1])[0] if results.get('component_errors') else None
            }
        elif analysis_type == "performance_analysis":
            return {
                'total_requests': results.get('total_requests', 0),
                'avg_duration': results.get('avg_duration', 0),
                'max_duration': results.get('max_duration', 0)
            }
        elif analysis_type == "user_activity":
            return {
                'total_users': results.get('total_users', 0),
                'total_sessions': results.get('total_sessions', 0),
                'total_ips': results.get('total_ips', 0)
            }
        elif analysis_type == "security_analysis":
            return {
                'total_security_events': results.get('total_security_events', 0),
                'total_failed_attempts': sum(results.get('failed_attempts', {}).values()),
                'total_suspicious_ips': len(results.get('suspicious_ips', {}))
            }
        else:
            return {
                'total_logs': results.get('total_logs', 0),
                'log_levels': len(results.get('level_counts', {})),
                'log_sources': len(results.get('source_counts', {}))
            }
    
    def _generate_log_id(self) -> str:
        """Generate a unique log ID"""
        return hashlib.md5(f"{datetime.utcnow().isoformat()}{threading.get_ident()}".encode()).hexdigest()
    
    def _generate_analysis_id(self) -> str:
        """Generate a unique analysis ID"""
        return hashlib.md5(f"analysis_{datetime.utcnow().isoformat()}".encode()).hexdigest()


# Global instance management
_log_aggregator_instance = None

def get_log_aggregator(db_path: str = "logs/aggregator.db") -> LogAggregator:
    """Get or create log aggregator instance"""
    global _log_aggregator_instance
    
    if _log_aggregator_instance is None:
        _log_aggregator_instance = LogAggregator(db_path)
    
    return _log_aggregator_instance


def init_log_aggregator(db_path: str = "logs/aggregator.db") -> LogAggregator:
    """Initialize log aggregator with custom database path"""
    global _log_aggregator_instance
    
    if _log_aggregator_instance:
        _log_aggregator_instance.stop()
    
    _log_aggregator_instance = LogAggregator(db_path)
    
    return _log_aggregator_instance