"""
TimescaleDB Configuration and Management

This module provides comprehensive TimescaleDB configuration including:
- Database initialization and setup
- Hypertable creation and management
- Compression policies
- Retention policies
- Performance optimization
- Monitoring and maintenance
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import structlog

from ..models import Base, StockData, Symbol, DataSource, DataCollectionJob, DataQualityLog, SystemMetrics, Alert
from config.settings import DatabaseSettings

logger = structlog.get_logger()

class TimescaleDBConfig:
    """
    TimescaleDB configuration and management class.
    
    Handles all TimescaleDB-specific operations including:
    - Database initialization
    - Hypertable creation and management
    - Compression and retention policies
    - Performance optimization
    - Monitoring and maintenance
    """
    
    def __init__(self, db_settings: DatabaseSettings):
        self.db_settings = db_settings
        self.engine = create_engine(
            db_settings.database_url,
            pool_size=db_settings.pool_size,
            max_overflow=db_settings.max_overflow,
            pool_timeout=db_settings.pool_timeout,
            pool_recycle=db_settings.pool_recycle
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # TimescaleDB specific settings
        self.chunk_time_interval = "1 day"  # Default chunk interval
        self.compression_enabled = True
        self.compression_after = "7 days"  # Compress chunks after 7 days
        self.retention_period = "365 days"  # Keep data for 1 year by default
        
        # Performance settings
        self.parallel_workers = 4
        self.max_background_workers = 8
        
    def initialize_database(self) -> bool:
        """
        Initialize the database with all required tables and TimescaleDB extensions.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing TimescaleDB database...")
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Base tables created successfully")
            
            # Enable TimescaleDB extension
            self._enable_timescaledb_extension()
            
            # Create hypertables
            self._create_hypertables()
            
            # Setup compression policies
            if self.compression_enabled:
                self._setup_compression_policies()
            
            # Setup retention policies
            self._setup_retention_policies()
            
            # Create indexes for performance
            self._create_performance_indexes()
            
            # Initialize default data
            self._initialize_default_data()
            
            logger.info("TimescaleDB database initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB database: {e}")
            return False
    
    def _enable_timescaledb_extension(self):
        """Enable TimescaleDB extension in the database."""
        try:
            with self.engine.connect() as connection:
                # Enable TimescaleDB extension
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
                connection.commit()
                logger.info("TimescaleDB extension enabled")
                
                # Check TimescaleDB version
                result = connection.execute(text("SELECT default_version, installed_version FROM pg_available_extensions WHERE name = 'timescaledb';"))
                version_info = result.fetchone()
                if version_info:
                    logger.info(f"TimescaleDB version: {version_info[1]}")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to enable TimescaleDB extension: {e}")
            raise
    
    def _create_hypertables(self):
        """Create hypertables for time-series data."""
        try:
            with self.engine.connect() as connection:
                # Create hypertable for stock_data
                connection.execute(text(f"""
                    SELECT create_hypertable('stock_data', 'timestamp', 
                        chunk_time_interval => INTERVAL '{self.chunk_time_interval}',
                        if_not_exists => TRUE);
                """))
                logger.info(f"Created hypertable for stock_data with chunk interval: {self.chunk_time_interval}")
                
                # Create hypertable for system_metrics
                connection.execute(text(f"""
                    SELECT create_hypertable('system_metrics', 'timestamp', 
                        chunk_time_interval => INTERVAL '{self.chunk_time_interval}',
                        if_not_exists => TRUE);
                """))
                logger.info(f"Created hypertable for system_metrics with chunk interval: {self.chunk_time_interval}")
                
                # Create hypertable for data_quality_logs
                connection.execute(text(f"""
                    SELECT create_hypertable('data_quality_logs', 'timestamp', 
                        chunk_time_interval => INTERVAL '{self.chunk_time_interval}',
                        if_not_exists => TRUE);
                """))
                logger.info(f"Created hypertable for data_quality_logs with chunk interval: {self.chunk_time_interval}")
                
                connection.commit()
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create hypertables: {e}")
            raise
    
    def _setup_compression_policies(self):
        """Setup compression policies for hypertables."""
        try:
            with self.engine.connect() as connection:
                # Enable compression on stock_data
                connection.execute(text(f"""
                    ALTER TABLE stock_data SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,timeframe',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
                
                # Add compression policy for stock_data
                connection.execute(text(f"""
                    SELECT add_compression_policy('stock_data', INTERVAL '{self.compression_after}');
                """))
                logger.info(f"Compression policy added for stock_data (compress after {self.compression_after})")
                
                # Enable compression on system_metrics
                connection.execute(text(f"""
                    ALTER TABLE system_metrics SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'metric_type',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
                
                # Add compression policy for system_metrics
                connection.execute(text(f"""
                    SELECT add_compression_policy('system_metrics', INTERVAL '{self.compression_after}');
                """))
                logger.info(f"Compression policy added for system_metrics (compress after {self.compression_after})")
                
                connection.commit()
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to setup compression policies: {e}")
            raise
    
    def _setup_retention_policies(self):
        """Setup retention policies for data cleanup."""
        try:
            with self.engine.connect() as connection:
                # Add retention policy for stock_data
                connection.execute(text(f"""
                    SELECT add_retention_policy('stock_data', INTERVAL '{self.retention_period}');
                """))
                logger.info(f"Retention policy added for stock_data (retain for {self.retention_period})")
                
                # Add retention policy for system_metrics (keep for 90 days)
                connection.execute(text(f"""
                    SELECT add_retention_policy('system_metrics', INTERVAL '90 days');
                """))
                logger.info("Retention policy added for system_metrics (retain for 90 days)")
                
                # Add retention policy for data_quality_logs (keep for 30 days)
                connection.execute(text(f"""
                    SELECT add_retention_policy('data_quality_logs', INTERVAL '30 days');
                """))
                logger.info("Retention policy added for data_quality_logs (retain for 30 days)")
                
                connection.commit()
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to setup retention policies: {e}")
            raise
    
    def _create_performance_indexes(self):
        """Create performance indexes for optimal query performance."""
        try:
            with self.engine.connect() as connection:
                # Create indexes for stock_data
                indexes = [
                    # Composite index for symbol and timeframe queries
                    "CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_timeframe ON stock_data (symbol, timeframe, timestamp DESC);",
                    
                    # Index for exchange-based queries
                    "CREATE INDEX IF NOT EXISTS idx_stock_data_exchange ON stock_data (exchange, timestamp DESC);",
                    
                    # Index for source-based queries
                    "CREATE INDEX IF NOT EXISTS idx_stock_data_source ON stock_data (source, timestamp DESC);",
                    
                    # Partial index for recent data
                    "CREATE INDEX IF NOT EXISTS idx_stock_data_recent ON stock_data (symbol, timestamp DESC) WHERE timestamp > NOW() - INTERVAL '30 days';",
                    
                    # Index for price range queries
                    "CREATE INDEX IF NOT EXISTS idx_stock_data_price_range ON stock_data (symbol, close, timestamp DESC);",
                    
                    # Index for volume queries
                    "CREATE INDEX IF NOT EXISTS idx_stock_data_volume ON stock_data (symbol, volume, timestamp DESC);"
                ]
                
                for index_sql in indexes:
                    connection.execute(text(index_sql))
                
                # Create indexes for system_metrics
                system_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_system_metrics_type ON system_metrics (metric_type, timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_system_metrics_value ON system_metrics (metric_type, metric_value, timestamp DESC);"
                ]
                
                for index_sql in system_indexes:
                    connection.execute(text(index_sql))
                
                connection.commit()
                logger.info("Performance indexes created successfully")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create performance indexes: {e}")
            raise
    
    def _initialize_default_data(self):
        """Initialize default data sources and sample symbols."""
        try:
            session = self.SessionLocal()
            
            # Add default data sources
            default_sources = [
                DataSource(name="yahoo_finance", description="Yahoo Finance API", is_active=True),
                DataSource(name="alpha_vantage", description="Alpha Vantage API", is_active=True),
                DataSource(name="websocket_stream", description="WebSocket real-time stream", is_active=True),
                DataSource(name="manual_import", description="Manual data import", is_active=False)
            ]
            
            for source in default_sources:
                existing = session.query(DataSource).filter(DataSource.name == source.name).first()
                if not existing:
                    session.add(source)
                    logger.info(f"Added default data source: {source.name}")
            
            # Add sample symbols
            sample_symbols = [
                Symbol(symbol="AAPL", name="Apple Inc.", exchange="NASDAQ", sector="Technology", industry="Consumer Electronics"),
                Symbol(symbol="MSFT", name="Microsoft Corporation", exchange="NASDAQ", sector="Technology", industry="Software"),
                Symbol(symbol="GOOGL", name="Alphabet Inc.", exchange="NASDAQ", sector="Technology", industry="Internet Services"),
                Symbol(symbol="AMZN", name="Amazon.com Inc.", exchange="NASDAQ", sector="Consumer Cyclical", industry="Internet Retail"),
                Symbol(symbol="TSLA", name="Tesla Inc.", exchange="NASDAQ", sector="Consumer Cyclical", industry="Auto Manufacturers"),
                Symbol(symbol="SPY", name="SPDR S&P 500 ETF Trust", exchange="NYSE", sector="ETF", industry="Exchange Traded Fund"),
                Symbol(symbol="QQQ", name="Invesco QQQ Trust", exchange="NASDAQ", sector="ETF", industry="Exchange Traded Fund"),
                Symbol(symbol="IWM", name="iShares Russell 2000 ETF", exchange="NYSE", sector="ETF", industry="Exchange Traded Fund")
            ]
            
            for symbol in sample_symbols:
                existing = session.query(Symbol).filter(Symbol.symbol == symbol.symbol).first()
                if not existing:
                    session.add(symbol)
                    logger.info(f"Added sample symbol: {symbol.symbol}")
            
            session.commit()
            logger.info("Default data initialized successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize default data: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information and statistics."""
        try:
            with self.engine.connect() as connection:
                info = {
                    "database_name": self.db_settings.database_name,
                    "host": self.db_settings.host,
                    "port": self.db_settings.port,
                    "pool_size": self.db_settings.pool_size,
                    "max_overflow": self.db_settings.max_overflow
                }
                
                # Get TimescaleDB version
                result = connection.execute(text("SELECT installed_version FROM pg_available_extensions WHERE name = 'timescaledb';"))
                version_info = result.fetchone()
                info["timescaledb_version"] = version_info[1] if version_info else "Not installed"
                
                # Get hypertable information
                result = connection.execute(text("""
                    SELECT hypertable_name, num_chunks, compression_enabled 
                    FROM timescaledb_information.hypertables;
                """))
                info["hypertables"] = [dict(row) for row in result.fetchall()]
                
                # Get compression statistics
                result = connection.execute(text("""
                    SELECT hypertable_name, total_chunks, number_compressed_chunks 
                    FROM timescaledb_information.compression_settings;
                """))
                info["compression_stats"] = [dict(row) for row in result.fetchall()]
                
                # Get table sizes
                result = connection.execute(text("""
                    SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
                """))
                info["table_sizes"] = [dict(row) for row in result.fetchall()]
                
                # Get chunk information
                result = connection.execute(text("""
                    SELECT hypertable_name, chunk_name, range_start, range_end, is_compressed
                    FROM timescaledb_information.chunks
                    ORDER BY range_start DESC
                    LIMIT 10;
                """))
                info["recent_chunks"] = [dict(row) for row in result.fetchall()]
                
                return info
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}
    
    def optimize_database(self) -> bool:
        """Run database optimization tasks."""
        try:
            logger.info("Starting database optimization...")
            
            with self.engine.connect() as connection:
                # Run VACUUM ANALYZE
                connection.execute(text("VACUUM ANALYZE;"))
                logger.info("VACUUM ANALYZE completed")
                
                # Update table statistics
                connection.execute(text("ANALYZE;"))
                logger.info("Table statistics updated")
                
                # Reindex if needed
                connection.execute(text("REINDEX DATABASE stock_predictor;"))
                logger.info("Database reindex completed")
                
                connection.commit()
                
            logger.info("Database optimization completed successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to optimize database: {e}")
            return False
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics and status."""
        try:
            with self.engine.connect() as connection:
                # Get compression settings
                result = connection.execute(text("""
                    SELECT hypertable_name, compression_enabled, 
                           timescaledb.compress_chunk_time_interval(hypertable_name::regclass) as chunk_interval
                    FROM timescaledb_information.hypertables;
                """))
                compression_settings = [dict(row) for row in result.fetchall()]
                
                # Get compression job status
                result = connection.execute(text("""
                    SELECT job_id, hypertable_name, last_run_started_at, last_run_status
                    FROM timescaledb_information.jobs
                    WHERE proc_name = 'policy_compression';
                """))
                compression_jobs = [dict(row) for row in result.fetchall()]
                
                # Get chunk compression status
                result = connection.execute(text("""
                    SELECT hypertable_name, 
                           COUNT(*) as total_chunks,
                           COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks,
                           COUNT(*) FILTER (WHERE NOT is_compressed) as uncompressed_chunks
                    FROM timescaledb_information.chunks
                    GROUP BY hypertable_name;
                """))
                chunk_stats = [dict(row) for row in result.fetchall()]
                
                return {
                    "compression_settings": compression_settings,
                    "compression_jobs": compression_jobs,
                    "chunk_stats": chunk_stats
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get compression stats: {e}")
            return {"error": str(e)}
    
    def get_retention_stats(self) -> Dict[str, Any]:
        """Get retention policy statistics."""
        try:
            with self.engine.connect() as connection:
                # Get retention policies
                result = connection.execute(text("""
                    SELECT hypertable_name, drop_after, schedule_interval
                    FROM timescaledb_information.drop_chunks_policies;
                """))
                retention_policies = [dict(row) for row in result.fetchall()]
                
                # Get retention job status
                result = connection.execute(text("""
                    SELECT job_id, hypertable_name, last_run_started_at, last_run_status
                    FROM timescaledb_information.jobs
                    WHERE proc_name = 'policy_drop_chunks';
                """))
                retention_jobs = [dict(row) for row in result.fetchall()]
                
                return {
                    "retention_policies": retention_policies,
                    "retention_jobs": retention_jobs
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get retention stats: {e}")
            return {"error": str(e)}
    
    def update_compression_settings(self, table_name: str, enabled: bool = True, 
                                  segment_by: str = None, order_by: str = None) -> bool:
        """Update compression settings for a specific table."""
        try:
            with self.engine.connect() as connection:
                if enabled:
                    # Enable compression
                    segment_clause = f", timescaledb.compress_segmentby = '{segment_by}'" if segment_by else ""
                    order_clause = f", timescaledb.compress_orderby = '{order_by}'" if order_by else ""
                    
                    sql = f"""
                        ALTER TABLE {table_name} SET (
                            timescaledb.compress{segment_clause}{order_clause}
                        );
                    """
                    connection.execute(text(sql))
                    logger.info(f"Compression enabled for {table_name}")
                else:
                    # Disable compression
                    connection.execute(text(f"ALTER TABLE {table_name} SET (timescaledb.compress = false);"))
                    logger.info(f"Compression disabled for {table_name}")
                
                connection.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to update compression settings for {table_name}: {e}")
            return False
    
    def update_retention_policy(self, table_name: str, retention_period: str) -> bool:
        """Update retention policy for a specific table."""
        try:
            with self.engine.connect() as connection:
                # Remove existing policy
                connection.execute(text(f"""
                    SELECT remove_retention_policy('{table_name}');
                """))
                
                # Add new policy
                connection.execute(text(f"""
                    SELECT add_retention_policy('{table_name}', INTERVAL '{retention_period}');
                """))
                
                connection.commit()
                logger.info(f"Retention policy updated for {table_name}: {retention_period}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to update retention policy for {table_name}: {e}")
            return False

# Convenience functions
def create_timescaledb_config(db_settings: DatabaseSettings) -> TimescaleDBConfig:
    """Create and return a TimescaleDB configuration instance."""
    return TimescaleDBConfig(db_settings)

def initialize_timescaledb_database(db_settings: DatabaseSettings) -> bool:
    """Initialize TimescaleDB database with all required setup."""
    config = TimescaleDBConfig(db_settings)
    return config.initialize_database()

def get_database_status(db_settings: DatabaseSettings) -> Dict[str, Any]:
    """Get comprehensive database status and statistics."""
    config = TimescaleDBConfig(db_settings)
    return config.get_database_info()