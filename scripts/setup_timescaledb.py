#!/usr/bin/env python3
"""
TimescaleDB Setup and Configuration Script

This script handles the complete setup and configuration of TimescaleDB for the ML Stock Predictor Platform.
It includes database initialization, hypertable creation, compression policies, retention policies, and testing.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.database.timescaledb_config import (
    TimescaleDBConfig,
    initialize_timescaledb_database,
    get_database_status
)
from config.settings import DatabaseSettings
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class TimescaleDBSetup:
    """Comprehensive TimescaleDB setup and configuration class."""
    
    def __init__(self, db_settings: DatabaseSettings):
        self.db_settings = db_settings
        self.config = TimescaleDBConfig(db_settings)
        self.setup_results = []
    
    async def run_complete_setup(self) -> bool:
        """Run complete TimescaleDB setup process."""
        logger.info("Starting complete TimescaleDB setup process")
        
        setup_steps = [
            ("Database Connection Test", self.test_database_connection),
            ("Database Initialization", self.initialize_database),
            ("Hypertable Verification", self.verify_hypertables),
            ("Compression Policy Verification", self.verify_compression_policies),
            ("Retention Policy Verification", self.verify_retention_policies),
            ("Performance Index Verification", self.verify_performance_indexes),
            ("Default Data Verification", self.verify_default_data),
            ("Database Optimization", self.optimize_database),
            ("Performance Testing", self.run_performance_tests),
        ]
        
        for step_name, step_func in setup_steps:
            try:
                logger.info(f"Running setup step: {step_name}")
                result = await step_func()
                self.setup_results.append((step_name, "SUCCESS" if result else "FAILED"))
                logger.info(f"Setup step {step_name}: {'SUCCESS' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Setup step {step_name} failed: {e}")
                self.setup_results.append((step_name, f"FAILED: {e}"))
        
        self.print_setup_summary()
        return all(result == "SUCCESS" for _, result in self.setup_results)
    
    async def test_database_connection(self) -> bool:
        """Test database connection and basic functionality."""
        try:
            logger.info("Testing database connection...")
            
            # Test basic connection
            with self.config.engine.connect() as connection:
                result = connection.execute("SELECT version();")
                version = result.fetchone()[0]
                logger.info(f"Database connection successful. PostgreSQL version: {version}")
            
            # Test TimescaleDB extension
            with self.config.engine.connect() as connection:
                result = connection.execute("SELECT installed_version FROM pg_available_extensions WHERE name = 'timescaledb';")
                timescale_version = result.fetchone()
                if timescale_version and timescale_version[0]:
                    logger.info(f"TimescaleDB extension found. Version: {timescale_version[0]}")
                else:
                    logger.warning("TimescaleDB extension not found. Please install TimescaleDB.")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def initialize_database(self) -> bool:
        """Initialize the database with all required components."""
        try:
            logger.info("Initializing database...")
            success = self.config.initialize_database()
            
            if success:
                logger.info("Database initialization completed successfully")
            else:
                logger.error("Database initialization failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    async def verify_hypertables(self) -> bool:
        """Verify that all required hypertables are created correctly."""
        try:
            logger.info("Verifying hypertables...")
            
            with self.config.engine.connect() as connection:
                # Check for required hypertables
                required_tables = ['stock_data', 'system_metrics', 'data_quality_logs']
                
                for table_name in required_tables:
                    result = connection.execute(f"""
                        SELECT hypertable_name 
                        FROM timescaledb_information.hypertables 
                        WHERE hypertable_name = '{table_name}';
                    """)
                    
                    if result.fetchone():
                        logger.info(f"Hypertable {table_name} verified")
                    else:
                        logger.error(f"Hypertable {table_name} not found")
                        return False
                
                # Get hypertable statistics
                result = connection.execute("""
                    SELECT hypertable_name, num_chunks, compression_enabled 
                    FROM timescaledb_information.hypertables;
                """)
                
                hypertable_stats = result.fetchall()
                for stat in hypertable_stats:
                    logger.info(f"Hypertable {stat[0]}: {stat[1]} chunks, compression: {stat[2]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Hypertable verification failed: {e}")
            return False
    
    async def verify_compression_policies(self) -> bool:
        """Verify compression policies are set up correctly."""
        try:
            logger.info("Verifying compression policies...")
            
            compression_stats = self.config.get_compression_stats()
            
            if "error" in compression_stats:
                logger.error(f"Failed to get compression stats: {compression_stats['error']}")
                return False
            
            # Check compression settings
            for setting in compression_stats.get("compression_settings", []):
                table_name = setting["hypertable_name"]
                enabled = setting["compression_enabled"]
                logger.info(f"Compression for {table_name}: {'enabled' if enabled else 'disabled'}")
            
            # Check compression jobs
            for job in compression_stats.get("compression_jobs", []):
                table_name = job["hypertable_name"]
                status = job["last_run_status"]
                logger.info(f"Compression job for {table_name}: {status}")
            
            # Check chunk statistics
            for stat in compression_stats.get("chunk_stats", []):
                table_name = stat["hypertable_name"]
                total = stat["total_chunks"]
                compressed = stat["compressed_chunks"]
                uncompressed = stat["uncompressed_chunks"]
                logger.info(f"Chunks for {table_name}: {total} total, {compressed} compressed, {uncompressed} uncompressed")
            
            return True
            
        except Exception as e:
            logger.error(f"Compression policy verification failed: {e}")
            return False
    
    async def verify_retention_policies(self) -> bool:
        """Verify retention policies are set up correctly."""
        try:
            logger.info("Verifying retention policies...")
            
            retention_stats = self.config.get_retention_stats()
            
            if "error" in retention_stats:
                logger.error(f"Failed to get retention stats: {retention_stats['error']}")
                return False
            
            # Check retention policies
            for policy in retention_stats.get("retention_policies", []):
                table_name = policy["hypertable_name"]
                drop_after = policy["drop_after"]
                logger.info(f"Retention policy for {table_name}: drop after {drop_after}")
            
            # Check retention jobs
            for job in retention_stats.get("retention_jobs", []):
                table_name = job["hypertable_name"]
                status = job["last_run_status"]
                logger.info(f"Retention job for {table_name}: {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Retention policy verification failed: {e}")
            return False
    
    async def verify_performance_indexes(self) -> bool:
        """Verify performance indexes are created correctly."""
        try:
            logger.info("Verifying performance indexes...")
            
            with self.config.engine.connect() as connection:
                # Check for required indexes
                required_indexes = [
                    'idx_stock_data_symbol_timeframe',
                    'idx_stock_data_exchange',
                    'idx_stock_data_source',
                    'idx_stock_data_recent',
                    'idx_stock_data_price_range',
                    'idx_stock_data_volume',
                    'idx_system_metrics_type',
                    'idx_system_metrics_value'
                ]
                
                for index_name in required_indexes:
                    result = connection.execute(f"""
                        SELECT indexname 
                        FROM pg_indexes 
                        WHERE indexname = '{index_name}';
                    """)
                    
                    if result.fetchone():
                        logger.info(f"Index {index_name} verified")
                    else:
                        logger.warning(f"Index {index_name} not found")
                
                # Get index statistics
                result = connection.execute("""
                    SELECT schemaname, tablename, indexname, indexdef
                    FROM pg_indexes 
                    WHERE schemaname = 'public' 
                    AND tablename IN ('stock_data', 'system_metrics', 'data_quality_logs')
                    ORDER BY tablename, indexname;
                """)
                
                indexes = result.fetchall()
                for index in indexes:
                    logger.info(f"Index {index[2]} on {index[1]}: {index[3][:100]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance index verification failed: {e}")
            return False
    
    async def verify_default_data(self) -> bool:
        """Verify default data sources and sample symbols are created."""
        try:
            logger.info("Verifying default data...")
            
            session = self.config.SessionLocal()
            
            # Check data sources
            from data_ingestion.models import DataSource, Symbol
            
            sources = session.query(DataSource).all()
            logger.info(f"Found {len(sources)} data sources:")
            for source in sources:
                logger.info(f"  - {source.name}: {source.description} ({'active' if source.is_active else 'inactive'})")
            
            # Check symbols
            symbols = session.query(Symbol).all()
            logger.info(f"Found {len(symbols)} symbols:")
            for symbol in symbols:
                logger.info(f"  - {symbol.symbol}: {symbol.name} ({symbol.exchange})")
            
            session.close()
            
            return len(sources) > 0 and len(symbols) > 0
            
        except Exception as e:
            logger.error(f"Default data verification failed: {e}")
            return False
    
    async def optimize_database(self) -> bool:
        """Run database optimization tasks."""
        try:
            logger.info("Running database optimization...")
            
            success = self.config.optimize_database()
            
            if success:
                logger.info("Database optimization completed successfully")
            else:
                logger.error("Database optimization failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
    
    async def run_performance_tests(self) -> bool:
        """Run basic performance tests to verify setup."""
        try:
            logger.info("Running performance tests...")
            
            with self.config.engine.connect() as connection:
                # Test 1: Simple query performance
                start_time = datetime.now()
                result = connection.execute("SELECT COUNT(*) FROM stock_data;")
                count = result.fetchone()[0]
                query_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Count query: {count} rows in {query_time:.3f} seconds")
                
                # Test 2: Time-range query performance
                start_time = datetime.now()
                result = connection.execute("""
                    SELECT symbol, COUNT(*) 
                    FROM stock_data 
                    WHERE timestamp > NOW() - INTERVAL '30 days'
                    GROUP BY symbol 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10;
                """)
                query_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Time-range query completed in {query_time:.3f} seconds")
                
                # Test 3: Aggregation query performance
                start_time = datetime.now()
                result = connection.execute("""
                    SELECT time_bucket('1 day', timestamp) as day, 
                           symbol, 
                           AVG(close) as avg_close,
                           MAX(high) as max_high,
                           MIN(low) as min_low
                    FROM stock_data 
                    WHERE timestamp > NOW() - INTERVAL '7 days'
                    GROUP BY day, symbol
                    ORDER BY day DESC, symbol
                    LIMIT 20;
                """)
                query_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Aggregation query completed in {query_time:.3f} seconds")
                
                # Test 4: Compression query performance
                start_time = datetime.now()
                result = connection.execute("""
                    SELECT hypertable_name, 
                           COUNT(*) as total_chunks,
                           COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks
                    FROM timescaledb_information.chunks
                    GROUP BY hypertable_name;
                """)
                query_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Compression query completed in {query_time:.3f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            return False
    
    def print_setup_summary(self):
        """Print comprehensive setup results summary."""
        logger.info("=" * 80)
        logger.info("TIMESCALEDB SETUP SUMMARY")
        logger.info("=" * 80)
        
        passed = sum(1 for _, result in self.setup_results if result == "SUCCESS")
        total = len(self.setup_results)
        
        for step_name, result in self.setup_results:
            status = "✅ SUCCESS" if result == "SUCCESS" else f"❌ {result}"
            logger.info(f"{step_name:<35} {status}")
        
        logger.info("=" * 80)
        logger.info(f"TOTAL: {passed}/{total} steps completed successfully")
        logger.info("=" * 80)
        
        if passed == total:
            logger.info("🎉 TIMESCALEDB SETUP COMPLETED SUCCESSFULLY!")
            logger.info("The database is ready for the ML Stock Predictor Platform.")
        else:
            logger.error("❌ SOME SETUP STEPS FAILED!")
            logger.error("Please review the errors and run the setup again.")
        
        # Print database status
        logger.info("\n" + "=" * 80)
        logger.info("DATABASE STATUS")
        logger.info("=" * 80)
        
        try:
            status = get_database_status(self.db_settings)
            for key, value in status.items():
                if isinstance(value, list):
                    logger.info(f"{key}: {len(value)} items")
                    for item in value[:3]:  # Show first 3 items
                        logger.info(f"  - {item}")
                else:
                    logger.info(f"{key}: {value}")
        except Exception as e:
            logger.error(f"Failed to get database status: {e}")

async def main():
    """Main setup execution function."""
    logger.info("Starting TimescaleDB Setup for ML Stock Predictor Platform")
    
    # Load database settings
    db_settings = DatabaseSettings()
    
    # Create setup instance
    setup = TimescaleDBSetup(db_settings)
    
    # Run complete setup
    success = await setup.run_complete_setup()
    
    if success:
        logger.info("TimescaleDB setup completed successfully!")
        return 0
    else:
        logger.error("TimescaleDB setup failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)