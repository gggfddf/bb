"""
Database Backup and Recovery Manager

This module provides comprehensive backup and recovery capabilities for the ML Stock Predictor Platform.
It handles automated backups, point-in-time recovery, backup verification, and disaster recovery procedures.
"""

import os
import subprocess
import shutil
import gzip
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import DatabaseSettings

logger = structlog.get_logger()

class BackupManager:
    """
    Comprehensive backup and recovery manager for the ML Stock Predictor Platform.
    
    Handles:
    - Automated database backups (full and incremental)
    - Backup verification and integrity checks
    - Point-in-time recovery
    - Backup retention and cleanup
    - Disaster recovery procedures
    - Backup monitoring and alerting
    """
    
    def __init__(self, db_settings: DatabaseSettings, backup_config: Dict[str, Any] = None):
        self.db_settings = db_settings
        self.backup_config = backup_config or self._get_default_backup_config()
        
        # Create backup directories
        self.backup_root = Path(self.backup_config["backup_root"])
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Backup subdirectories
        self.full_backup_dir = self.backup_root / "full"
        self.incremental_backup_dir = self.backup_root / "incremental"
        self.wal_backup_dir = self.backup_root / "wal"
        self.logs_dir = self.backup_root / "logs"
        
        for directory in [self.full_backup_dir, self.incremental_backup_dir, self.wal_backup_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Backup tracking
        self.backup_history = []
        self.backup_stats = {
            "total_backups": 0,
            "successful_backups": 0,
            "failed_backups": 0,
            "total_size_bytes": 0,
            "last_backup_time": None
        }
    
    def _get_default_backup_config(self) -> Dict[str, Any]:
        """Get default backup configuration."""
        return {
            "backup_root": "/var/backups/stock_predictor",
            "backup_retention_days": 30,
            "full_backup_interval_hours": 24,
            "incremental_backup_interval_hours": 6,
            "wal_backup_interval_minutes": 15,
            "compression_enabled": True,
            "encryption_enabled": False,
            "verification_enabled": True,
            "parallel_jobs": 4,
            "pg_dump_path": "pg_dump",
            "pg_restore_path": "pg_restore",
            "psql_path": "psql",
            "backup_format": "custom",  # custom, plain, directory
            "include_schema": True,
            "include_data": True,
            "include_indexes": True,
            "include_triggers": True,
            "exclude_tables": [],
            "notification_email": None,
            "backup_timeout_seconds": 3600
        }
    
    def create_full_backup(self) -> Dict[str, Any]:
        """
        Create a full database backup.
        
        Returns:
            Dict containing backup result information
        """
        try:
            logger.info("Starting full database backup...")
            
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"full_backup_{timestamp}.sql"
            backup_path = self.full_backup_dir / backup_filename
            
            # Build pg_dump command
            cmd = self._build_pg_dump_command(backup_path, backup_type="full")
            
            # Execute backup
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.backup_config["backup_timeout_seconds"]
            )
            end_time = datetime.now()
            
            if result.returncode == 0:
                # Compress backup if enabled
                if self.backup_config["compression_enabled"]:
                    backup_path = self._compress_backup(backup_path)
                
                # Verify backup if enabled
                if self.backup_config["verification_enabled"]:
                    verification_result = self._verify_backup(backup_path)
                else:
                    verification_result = {"verified": True, "message": "Verification skipped"}
                
                # Calculate backup size
                backup_size = backup_path.stat().st_size if backup_path.exists() else 0
                
                # Record backup
                backup_info = {
                    "backup_id": f"full_{timestamp}",
                    "backup_type": "full",
                    "filename": backup_path.name,
                    "path": str(backup_path),
                    "size_bytes": backup_size,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": (end_time - start_time).total_seconds(),
                    "status": "success",
                    "verification": verification_result,
                    "compressed": self.backup_config["compression_enabled"]
                }
                
                self._record_backup(backup_info)
                
                logger.info(f"Full backup completed successfully: {backup_path.name}")
                return backup_info
            else:
                error_msg = f"Backup failed: {result.stderr}"
                logger.error(error_msg)
                
                backup_info = {
                    "backup_id": f"full_{timestamp}",
                    "backup_type": "full",
                    "status": "failed",
                    "error": error_msg,
                    "start_time": start_time,
                    "end_time": end_time
                }
                
                self._record_backup(backup_info)
                return backup_info
                
        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def create_incremental_backup(self) -> Dict[str, Any]:
        """
        Create an incremental database backup.
        
        Returns:
            Dict containing backup result information
        """
        try:
            logger.info("Starting incremental database backup...")
            
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"incremental_backup_{timestamp}.sql"
            backup_path = self.incremental_backup_dir / backup_filename
            
            # Build pg_dump command for incremental backup
            cmd = self._build_pg_dump_command(backup_path, backup_type="incremental")
            
            # Execute backup
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.backup_config["backup_timeout_seconds"]
            )
            end_time = datetime.now()
            
            if result.returncode == 0:
                # Compress backup if enabled
                if self.backup_config["compression_enabled"]:
                    backup_path = self._compress_backup(backup_path)
                
                # Verify backup if enabled
                if self.backup_config["verification_enabled"]:
                    verification_result = self._verify_backup(backup_path)
                else:
                    verification_result = {"verified": True, "message": "Verification skipped"}
                
                # Calculate backup size
                backup_size = backup_path.stat().st_size if backup_path.exists() else 0
                
                # Record backup
                backup_info = {
                    "backup_id": f"incremental_{timestamp}",
                    "backup_type": "incremental",
                    "filename": backup_path.name,
                    "path": str(backup_path),
                    "size_bytes": backup_size,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": (end_time - start_time).total_seconds(),
                    "status": "success",
                    "verification": verification_result,
                    "compressed": self.backup_config["compression_enabled"]
                }
                
                self._record_backup(backup_info)
                
                logger.info(f"Incremental backup completed successfully: {backup_path.name}")
                return backup_info
            else:
                error_msg = f"Incremental backup failed: {result.stderr}"
                logger.error(error_msg)
                
                backup_info = {
                    "backup_id": f"incremental_{timestamp}",
                    "backup_type": "incremental",
                    "status": "failed",
                    "error": error_msg,
                    "start_time": start_time,
                    "end_time": end_time
                }
                
                self._record_backup(backup_info)
                return backup_info
                
        except Exception as e:
            logger.error(f"Incremental backup failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _build_pg_dump_command(self, backup_path: Path, backup_type: str) -> List[str]:
        """Build pg_dump command with appropriate options."""
        cmd = [
            self.backup_config["pg_dump_path"],
            "-h", self.db_settings.host,
            "-p", str(self.db_settings.port),
            "-U", self.db_settings.username,
            "-d", self.db_settings.database_name,
            "-f", str(backup_path)
        ]
        
        # Add format-specific options
        if self.backup_config["backup_format"] == "custom":
            cmd.extend(["-Fc"])  # Custom format
        elif self.backup_config["backup_format"] == "directory":
            cmd.extend(["-Fd"])  # Directory format
        else:
            cmd.extend(["-Fp"])  # Plain text format
        
        # Add content options
        if self.backup_config["include_schema"]:
            cmd.append("--schema-only")
        elif self.backup_config["include_data"]:
            cmd.append("--data-only")
        else:
            # Include both schema and data (default)
            pass
        
        if self.backup_config["include_indexes"]:
            cmd.append("--indexes")
        
        if self.backup_config["include_triggers"]:
            cmd.append("--triggers")
        
        # Exclude tables
        for table in self.backup_config["exclude_tables"]:
            cmd.extend(["--exclude-table", table])
        
        # Add performance options
        cmd.extend([
            "--verbose",
            "--no-password",
            f"--jobs={self.backup_config['parallel_jobs']}"
        ])
        
        # Add incremental backup options
        if backup_type == "incremental":
            # For incremental backup, we might want to backup only changed data
            # This is a simplified approach - in production, you might use WAL archiving
            cmd.extend(["--data-only"])
        
        return cmd
    
    def _compress_backup(self, backup_path: Path) -> Path:
        """Compress backup file using gzip."""
        try:
            compressed_path = backup_path.with_suffix(backup_path.suffix + ".gz")
            
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            backup_path.unlink()
            
            logger.info(f"Backup compressed: {compressed_path.name}")
            return compressed_path
            
        except Exception as e:
            logger.error(f"Failed to compress backup: {e}")
            return backup_path
    
    def _verify_backup(self, backup_path: Path) -> Dict[str, Any]:
        """Verify backup integrity."""
        try:
            # Check if file exists and has size
            if not backup_path.exists():
                return {"verified": False, "message": "Backup file not found"}
            
            file_size = backup_path.stat().st_size
            if file_size == 0:
                return {"verified": False, "message": "Backup file is empty"}
            
            # For compressed files, try to decompress
            if backup_path.suffix == ".gz":
                try:
                    with gzip.open(backup_path, 'rb') as f:
                        # Read first few bytes to check if it's valid gzip
                        f.read(1024)
                    return {"verified": True, "message": "Compressed backup verified"}
                except Exception as e:
                    return {"verified": False, "message": f"Invalid gzip file: {e}"}
            
            # For SQL files, check if it contains valid SQL
            if backup_path.suffix == ".sql":
                try:
                    with open(backup_path, 'r') as f:
                        content = f.read(1024)
                        if "PostgreSQL database dump" in content or "CREATE TABLE" in content:
                            return {"verified": True, "message": "SQL backup verified"}
                        else:
                            return {"verified": False, "message": "Invalid SQL backup file"}
                except Exception as e:
                    return {"verified": False, "message": f"Error reading SQL file: {e}"}
            
            return {"verified": True, "message": "Backup file verified"}
            
        except Exception as e:
            return {"verified": False, "message": f"Verification failed: {e}"}
    
    def _record_backup(self, backup_info: Dict[str, Any]):
        """Record backup information."""
        self.backup_history.append(backup_info)
        
        # Update statistics
        self.backup_stats["total_backups"] += 1
        
        if backup_info["status"] == "success":
            self.backup_stats["successful_backups"] += 1
            self.backup_stats["total_size_bytes"] += backup_info.get("size_bytes", 0)
            self.backup_stats["last_backup_time"] = backup_info["end_time"]
        else:
            self.backup_stats["failed_backups"] += 1
        
        # Save backup history to file
        self._save_backup_history()
    
    def _save_backup_history(self):
        """Save backup history to JSON file."""
        try:
            history_file = self.logs_dir / "backup_history.json"
            
            # Convert datetime objects to strings for JSON serialization
            serializable_history = []
            for backup in self.backup_history:
                serializable_backup = backup.copy()
                for key, value in serializable_backup.items():
                    if isinstance(value, datetime):
                        serializable_backup[key] = value.isoformat()
                serializable_history.append(serializable_backup)
            
            with open(history_file, 'w') as f:
                json.dump({
                    "backup_history": serializable_history,
                    "backup_stats": self.backup_stats,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backup history: {e}")
    
    def restore_backup(self, backup_path: str, target_database: str = None) -> Dict[str, Any]:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            target_database: Target database name (optional)
            
        Returns:
            Dict containing restore result information
        """
        try:
            logger.info(f"Starting database restore from: {backup_path}")
            
            backup_file = Path(backup_path)
            if not backup_file.exists():
                return {"status": "failed", "error": "Backup file not found"}
            
            # Determine target database
            if target_database is None:
                target_database = self.db_settings.database_name
            
            # Build pg_restore command
            cmd = self._build_pg_restore_command(backup_file, target_database)
            
            # Execute restore
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.backup_config["backup_timeout_seconds"]
            )
            end_time = datetime.now()
            
            if result.returncode == 0:
                restore_info = {
                    "backup_path": backup_path,
                    "target_database": target_database,
                    "status": "success",
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": (end_time - start_time).total_seconds()
                }
                
                logger.info(f"Database restore completed successfully")
                return restore_info
            else:
                error_msg = f"Restore failed: {result.stderr}"
                logger.error(error_msg)
                
                return {
                    "backup_path": backup_path,
                    "target_database": target_database,
                    "status": "failed",
                    "error": error_msg,
                    "start_time": start_time,
                    "end_time": end_time
                }
                
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _build_pg_restore_command(self, backup_path: Path, target_database: str) -> List[str]:
        """Build pg_restore command with appropriate options."""
        cmd = [
            self.backup_config["pg_restore_path"],
            "-h", self.db_settings.host,
            "-p", str(self.db_settings.port),
            "-U", self.db_settings.username,
            "-d", target_database,
            "--verbose",
            "--no-password",
            "--clean",  # Drop objects before recreating
            "--if-exists",  # Don't error if objects don't exist
            str(backup_path)
        ]
        
        # Add format-specific options
        if backup_path.suffix == ".gz":
            cmd.extend(["--format=custom"])
        elif self.backup_config["backup_format"] == "custom":
            cmd.extend(["--format=custom"])
        elif self.backup_config["backup_format"] == "directory":
            cmd.extend(["--format=directory"])
        else:
            cmd.extend(["--format=plain"])
        
        return cmd
    
    def cleanup_old_backups(self) -> Dict[str, Any]:
        """
        Clean up old backups based on retention policy.
        
        Returns:
            Dict containing cleanup result information
        """
        try:
            logger.info("Starting backup cleanup...")
            
            cutoff_date = datetime.now() - timedelta(days=self.backup_config["backup_retention_days"])
            deleted_files = []
            freed_space = 0
            
            # Clean up full backups
            for backup_file in self.full_backup_dir.glob("*.sql*"):
                if self._should_delete_backup(backup_file, cutoff_date):
                    file_size = backup_file.stat().st_size
                    backup_file.unlink()
                    deleted_files.append(str(backup_file))
                    freed_space += file_size
            
            # Clean up incremental backups
            for backup_file in self.incremental_backup_dir.glob("*.sql*"):
                if self._should_delete_backup(backup_file, cutoff_date):
                    file_size = backup_file.stat().st_size
                    backup_file.unlink()
                    deleted_files.append(str(backup_file))
                    freed_space += file_size
            
            # Clean up WAL backups
            for backup_file in self.wal_backup_dir.glob("*"):
                if self._should_delete_backup(backup_file, cutoff_date):
                    file_size = backup_file.stat().st_size
                    backup_file.unlink()
                    deleted_files.append(str(backup_file))
                    freed_space += file_size
            
            cleanup_info = {
                "status": "success",
                "deleted_files": deleted_files,
                "files_deleted": len(deleted_files),
                "freed_space_bytes": freed_space,
                "freed_space_mb": freed_space / (1024 * 1024),
                "cutoff_date": cutoff_date
            }
            
            logger.info(f"Backup cleanup completed: {len(deleted_files)} files deleted, {freed_space / (1024 * 1024):.2f} MB freed")
            return cleanup_info
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _should_delete_backup(self, backup_file: Path, cutoff_date: datetime) -> bool:
        """Check if backup file should be deleted based on retention policy."""
        try:
            # Extract timestamp from filename
            filename = backup_file.name
            
            # Try to parse timestamp from filename
            if "full_backup_" in filename:
                timestamp_str = filename.replace("full_backup_", "").split(".")[0]
            elif "incremental_backup_" in filename:
                timestamp_str = filename.replace("incremental_backup_", "").split(".")[0]
            else:
                # Use file modification time as fallback
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                return file_time < cutoff_date
            
            # Parse timestamp
            file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            return file_time < cutoff_date
            
        except Exception as e:
            logger.warning(f"Could not parse timestamp from {backup_file.name}: {e}")
            # Use file modification time as fallback
            file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
            return file_time < cutoff_date
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup status and statistics."""
        try:
            # Load backup history if not loaded
            if not self.backup_history:
                self._load_backup_history()
            
            # Calculate additional statistics
            recent_backups = [
                backup for backup in self.backup_history
                if backup.get("status") == "success" and
                backup.get("end_time") and
                isinstance(backup["end_time"], datetime) and
                backup["end_time"] > datetime.now() - timedelta(days=7)
            ]
            
            # Get disk usage
            total_size = sum(
                backup.get("size_bytes", 0) for backup in self.backup_history
                if backup.get("status") == "success"
            )
            
            # Check backup health
            health_status = self._check_backup_health()
            
            return {
                "backup_stats": self.backup_stats,
                "recent_backups": len(recent_backups),
                "total_backup_size_mb": total_size / (1024 * 1024),
                "backup_health": health_status,
                "backup_config": self.backup_config,
                "backup_directories": {
                    "full_backup_dir": str(self.full_backup_dir),
                    "incremental_backup_dir": str(self.incremental_backup_dir),
                    "wal_backup_dir": str(self.wal_backup_dir),
                    "logs_dir": str(self.logs_dir)
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get backup status: {e}")
            return {"error": str(e)}
    
    def _load_backup_history(self):
        """Load backup history from file."""
        try:
            history_file = self.logs_dir / "backup_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.backup_history = data.get("backup_history", [])
                    self.backup_stats = data.get("backup_stats", self.backup_stats)
                    
                    # Convert string timestamps back to datetime objects
                    for backup in self.backup_history:
                        for key, value in backup.items():
                            if key in ["start_time", "end_time"] and isinstance(value, str):
                                try:
                                    backup[key] = datetime.fromisoformat(value)
                                except:
                                    pass
        except Exception as e:
            logger.error(f"Failed to load backup history: {e}")
    
    def _check_backup_health(self) -> Dict[str, Any]:
        """Check backup system health."""
        try:
            health_status = {
                "overall_health": "good",
                "issues": [],
                "recommendations": []
            }
            
            # Check if recent backups exist
            recent_successful_backups = [
                backup for backup in self.backup_history
                if backup.get("status") == "success" and
                backup.get("end_time") and
                isinstance(backup["end_time"], datetime) and
                backup["end_time"] > datetime.now() - timedelta(hours=24)
            ]
            
            if not recent_successful_backups:
                health_status["issues"].append("No successful backups in the last 24 hours")
                health_status["overall_health"] = "poor"
                health_status["recommendations"].append("Check backup schedule and ensure backups are running")
            
            # Check backup success rate
            if self.backup_stats["total_backups"] > 0:
                success_rate = self.backup_stats["successful_backups"] / self.backup_stats["total_backups"]
                if success_rate < 0.8:
                    health_status["issues"].append(f"Low backup success rate: {success_rate:.2%}")
                    health_status["recommendations"].append("Investigate backup failures")
            
            # Check disk space
            total_size_mb = self.backup_stats["total_size_bytes"] / (1024 * 1024)
            if total_size_mb > 10000:  # 10 GB
                health_status["issues"].append(f"Large backup size: {total_size_mb:.2f} MB")
                health_status["recommendations"].append("Consider adjusting retention policy or compression")
            
            # Determine overall health
            if len(health_status["issues"]) > 2:
                health_status["overall_health"] = "poor"
            elif len(health_status["issues"]) > 0:
                health_status["overall_health"] = "fair"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to check backup health: {e}")
            return {"overall_health": "unknown", "error": str(e)}

# Convenience functions
def create_backup_manager(db_settings: DatabaseSettings, backup_config: Dict[str, Any] = None) -> BackupManager:
    """Create and return a backup manager instance."""
    return BackupManager(db_settings, backup_config)

def create_full_backup(db_settings: DatabaseSettings, backup_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a full database backup."""
    manager = BackupManager(db_settings, backup_config)
    return manager.create_full_backup()

def get_backup_status(db_settings: DatabaseSettings, backup_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get comprehensive backup status and statistics."""
    manager = BackupManager(db_settings, backup_config)
    return manager.get_backup_status()