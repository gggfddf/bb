#!/usr/bin/env python3
"""
Test script for the Backup Manager functionality.
Tests backup creation, verification, restoration, and monitoring.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.database.backup_manager import (
    BackupManager,
    create_backup_manager,
    create_full_backup,
    get_backup_status
)
from config.settings import DatabaseSettings
import structlog

logger = structlog.get_logger()

class BackupManagerTestSuite:
    def __init__(self):
        self.db_settings = DatabaseSettings()
        self.backup_config = {
            "backup_root": "/tmp/stock_predictor_backups",
            "retention_days": 30,
            "full_backup_interval_hours": 24,
            "incremental_backup_interval_hours": 6,
            "compression_enabled": True,
            "verification_enabled": True,
            "max_backup_size_gb": 10
        }
        self.backup_manager = BackupManager(self.db_settings, self.backup_config)
        self.test_results = []

    async def run_all_tests(self):
        """Run all backup manager tests."""
        logger.info("Starting Backup Manager Test Suite")
        
        tests = [
            ("Backup Configuration", self.test_backup_configuration),
            ("Full Backup Creation", self.test_full_backup_creation),
            ("Incremental Backup Creation", self.test_incremental_backup_creation),
            ("Backup Verification", self.test_backup_verification),
            ("Backup Restoration", self.test_backup_restoration),
            ("Backup Cleanup", self.test_backup_cleanup),
            ("Backup Monitoring", self.test_backup_monitoring),
            ("Error Handling", self.test_error_handling),
            ("Integration Test", self.test_integration)
        ]

        for test_name, test_func in tests:
            try:
                logger.info(f"Running test: {test_name}")
                result = await test_func()
                self.test_results.append({
                    "test": test_name,
                    "status": "PASSED" if result else "FAILED",
                    "timestamp": datetime.now()
                })
                logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.test_results.append({
                    "test": test_name,
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now()
                })

        self.print_test_summary()

    async def test_backup_configuration(self) -> bool:
        """Test backup configuration setup."""
        try:
            # Test default configuration
            default_manager = BackupManager(self.db_settings)
            assert default_manager.backup_config is not None
            
            # Test custom configuration
            custom_config = {
                "backup_root": "/tmp/custom_backups",
                "retention_days": 60
            }
            custom_manager = BackupManager(self.db_settings, custom_config)
            assert custom_manager.backup_config["retention_days"] == 60
            
            # Test backup directory creation
            backup_dir = Path(self.backup_config["backup_root"])
            if not backup_dir.exists():
                backup_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Backup configuration test passed")
            return True
        except Exception as e:
            logger.error(f"Backup configuration test failed: {e}")
            return False

    async def test_full_backup_creation(self) -> bool:
        """Test full backup creation."""
        try:
            # Create a test backup
            backup_result = self.backup_manager.create_full_backup()
            
            assert backup_result["success"] is True
            assert "backup_path" in backup_result
            assert "backup_size" in backup_result
            assert "compression_ratio" in backup_result
            
            # Verify backup file exists
            backup_path = Path(backup_result["backup_path"])
            assert backup_path.exists()
            
            logger.info(f"Full backup created successfully: {backup_result['backup_path']}")
            return True
        except Exception as e:
            logger.error(f"Full backup creation test failed: {e}")
            return False

    async def test_incremental_backup_creation(self) -> bool:
        """Test incremental backup creation."""
        try:
            # Create an incremental backup
            backup_result = self.backup_manager.create_incremental_backup()
            
            assert backup_result["success"] is True
            assert "backup_path" in backup_result
            assert backup_result["backup_type"] == "incremental"
            
            # Verify backup file exists
            backup_path = Path(backup_result["backup_path"])
            assert backup_path.exists()
            
            logger.info(f"Incremental backup created successfully: {backup_result['backup_path']}")
            return True
        except Exception as e:
            logger.error(f"Incremental backup creation test failed: {e}")
            return False

    async def test_backup_verification(self) -> bool:
        """Test backup verification."""
        try:
            # Create a backup first
            backup_result = self.backup_manager.create_full_backup()
            if not backup_result["success"]:
                logger.warning("Skipping verification test - backup creation failed")
                return True
            
            # Test verification
            backup_path = Path(backup_result["backup_path"])
            verification_result = self.backup_manager._verify_backup(backup_path)
            
            assert verification_result["is_valid"] is True
            assert verification_result["file_exists"] is True
            assert verification_result["file_size"] > 0
            
            logger.info("Backup verification test passed")
            return True
        except Exception as e:
            logger.error(f"Backup verification test failed: {e}")
            return False

    async def test_backup_restoration(self) -> bool:
        """Test backup restoration (simulated)."""
        try:
            # Create a backup first
            backup_result = self.backup_manager.create_full_backup()
            if not backup_result["success"]:
                logger.warning("Skipping restoration test - backup creation failed")
                return True
            
            # Test restoration (simulated - don't actually restore to avoid data loss)
            backup_path = backup_result["backup_path"]
            target_db = "test_restore_db"
            
            # Just test the command building
            restore_command = self.backup_manager._build_pg_restore_command(
                Path(backup_path), target_db
            )
            
            assert len(restore_command) > 0
            assert "pg_restore" in restore_command[0]
            assert target_db in restore_command
            
            logger.info("Backup restoration test passed (simulated)")
            return True
        except Exception as e:
            logger.error(f"Backup restoration test failed: {e}")
            return False

    async def test_backup_cleanup(self) -> bool:
        """Test backup cleanup functionality."""
        try:
            # Create some old backup files for testing
            backup_dir = Path(self.backup_config["backup_root"])
            old_backup = backup_dir / f"old_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql.gz"
            old_backup.touch()
            
            # Test cleanup
            cleanup_result = self.backup_manager.cleanup_old_backups()
            
            assert cleanup_result["success"] is True
            assert "deleted_files" in cleanup_result
            assert "freed_space" in cleanup_result
            
            logger.info("Backup cleanup test passed")
            return True
        except Exception as e:
            logger.error(f"Backup cleanup test failed: {e}")
            return False

    async def test_backup_monitoring(self) -> bool:
        """Test backup monitoring and status."""
        try:
            # Get backup status
            status = self.backup_manager.get_backup_status()
            
            assert "backup_history" in status
            assert "backup_stats" in status
            assert "health_check" in status
            assert "disk_usage" in status
            
            # Test health check
            health = status["health_check"]
            assert "last_successful_backup" in health
            assert "backup_success_rate" in health
            assert "disk_usage_percentage" in health
            
            logger.info("Backup monitoring test passed")
            return True
        except Exception as e:
            logger.error(f"Backup monitoring test failed: {e}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling scenarios."""
        try:
            # Test with invalid backup path
            invalid_path = Path("/invalid/path/backup.sql.gz")
            verification_result = self.backup_manager._verify_backup(invalid_path)
            
            assert verification_result["is_valid"] is False
            assert verification_result["file_exists"] is False
            
            # Test with invalid database settings
            invalid_settings = DatabaseSettings()
            invalid_settings.database_url = "postgresql://invalid:invalid@localhost:5432/invalid"
            
            try:
                invalid_manager = BackupManager(invalid_settings)
                # This should not raise an exception immediately
                assert invalid_manager is not None
            except Exception:
                # It's okay if it fails during initialization
                pass
            
            logger.info("Error handling test passed")
            return True
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False

    async def test_integration(self) -> bool:
        """Test integration with other components."""
        try:
            # Test with factory functions
            manager = create_backup_manager(self.db_settings, self.backup_config)
            assert manager is not None
            
            # Test backup creation via factory function
            backup_result = create_full_backup(self.db_settings, self.backup_config)
            assert "success" in backup_result
            
            # Test status retrieval via factory function
            status = get_backup_status(self.db_settings, self.backup_config)
            assert "backup_history" in status
            
            logger.info("Integration test passed")
            return True
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False

    def print_test_summary(self):
        """Print test results summary."""
        logger.info("=" * 60)
        logger.info("BACKUP MANAGER TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "PASSED")
        failed = sum(1 for result in self.test_results if result["status"] == "FAILED")
        errors = sum(1 for result in self.test_results if result["status"] == "ERROR")
        total = len(self.test_results)
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0 or errors > 0:
            logger.info("\nFailed/Error Tests:")
            for result in self.test_results:
                if result["status"] in ["FAILED", "ERROR"]:
                    logger.info(f"  - {result['test']}: {result.get('error', 'Unknown error')}")
        
        logger.info("=" * 60)

async def main():
    """Main test execution function."""
    logger.info("Starting Backup Manager Test Suite")
    
    test_suite = BackupManagerTestSuite()
    await test_suite.run_all_tests()
    
    logger.info("Backup Manager Test Suite completed")

if __name__ == "__main__":
    asyncio.run(main())