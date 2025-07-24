"""
Data Partitioning System

This module provides comprehensive data partitioning capabilities for the ML Stock Predictor Platform.
It handles automatic partitioning, data distribution, performance optimization, and partition management.
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

class DataPartitioningManager:
    """
    Comprehensive data partitioning manager for the ML Stock Predictor Platform.
    
    Handles:
    - Automatic time-based partitioning using TimescaleDB hypertables
    - Symbol-based partitioning for large datasets
    - Partition optimization and maintenance
    - Data distribution and load balancing
    - Partition monitoring and statistics
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
        
        # Partitioning configuration
        self.partitioning_config = {
            "stock_data": {
                "time_interval": "1 day",
                "compression_after": "7 days",
                "retention_period": "3 years",
                "enable_symbol_partitioning": True,
                "symbol_partition_threshold": 1000000  # 1M records per symbol
            },
            "system_metrics": {
                "time_interval": "1 hour",
                "compression_after": "1 day",
                "retention_period": "90 days",
                "enable_symbol_partitioning": False
            },
            "data_quality_logs": {
                "time_interval": "1 day",
                "compression_after": "7 days",
                "retention_period": "1 year",
                "enable_symbol_partitioning": True,
                "symbol_partition_threshold": 100000
            }
        }
        
        # Partition statistics
        self.partition_stats = {}
        
    def setup_partitioning(self) -> bool:
        """
        Setup partitioning for all time-series tables.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            logger.info("Setting up data partitioning...")
            
            # Setup partitioning for each table
            for table_name, config in self.partitioning_config.items():
                success = self._setup_table_partitioning(table_name, config)
                if not success:
                    logger.error(f"Failed to setup partitioning for {table_name}")
                    return False
            
            # Initialize partition statistics
            self._initialize_partition_stats()
            
            logger.info("Data partitioning setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup data partitioning: {e}")
            return False
    
    def _setup_table_partitioning(self, table_name: str, config: Dict[str, Any]) -> bool:
        """Setup partitioning for a specific table."""
        try:
            with self.engine.connect() as connection:
                # Check if table exists
                inspector = inspect(self.engine)
                if not inspector.has_table(table_name):
                    logger.warning(f"Table {table_name} does not exist, skipping partitioning setup")
                    return True
                
                # Create hypertable if it doesn't exist
                connection.execute(text(f"""
                    SELECT create_hypertable('{table_name}', 'timestamp', 
                        chunk_time_interval => INTERVAL '{config["time_interval"]}',
                        if_not_exists => TRUE);
                """))
                logger.info(f"Hypertable created/verified for {table_name}")
                
                # Setup compression if enabled
                if config.get("compression_after"):
                    self._setup_compression_policy(connection, table_name, config)
                
                # Setup retention policy
                if config.get("retention_period"):
                    self._setup_retention_policy(connection, table_name, config)
                
                # Setup symbol-based partitioning if enabled
                if config.get("enable_symbol_partitioning", False):
                    self._setup_symbol_partitioning(connection, table_name, config)
                
                connection.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to setup partitioning for {table_name}: {e}")
            return False
    
    def _setup_compression_policy(self, connection, table_name: str, config: Dict[str, Any]):
        """Setup compression policy for a table."""
        try:
            # Enable compression
            if table_name == "stock_data":
                connection.execute(text(f"""
                    ALTER TABLE {table_name} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,timeframe',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
            elif table_name == "system_metrics":
                connection.execute(text(f"""
                    ALTER TABLE {table_name} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'metric_type',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
            else:
                connection.execute(text(f"""
                    ALTER TABLE {table_name} SET (
                        timescaledb.compress,
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
            
            # Add compression policy
            connection.execute(text(f"""
                SELECT add_compression_policy('{table_name}', INTERVAL '{config["compression_after"]}');
            """))
            
            logger.info(f"Compression policy setup for {table_name}")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to setup compression policy for {table_name}: {e}")
            raise
    
    def _setup_retention_policy(self, connection, table_name: str, config: Dict[str, Any]):
        """Setup retention policy for a table."""
        try:
            connection.execute(text(f"""
                SELECT add_retention_policy('{table_name}', INTERVAL '{config["retention_period"]}');
            """))
            
            logger.info(f"Retention policy setup for {table_name}")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to setup retention policy for {table_name}: {e}")
            raise
    
    def _setup_symbol_partitioning(self, connection, table_name: str, config: Dict[str, Any]):
        """Setup symbol-based partitioning for large datasets."""
        try:
            # Create symbol-based indexes for better query performance
            if table_name == "stock_data":
                connection.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_partition 
                    ON {table_name} (symbol, timestamp DESC);
                """))
                
                # Create partial indexes for high-volume symbols
                connection.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_high_volume_symbols 
                    ON {table_name} (timestamp DESC) 
                    WHERE symbol IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA');
                """))
            
            elif table_name == "data_quality_logs":
                connection.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_partition 
                    ON {table_name} (symbol, timestamp DESC);
                """))
            
            logger.info(f"Symbol-based partitioning setup for {table_name}")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to setup symbol partitioning for {table_name}: {e}")
            raise
    
    def _initialize_partition_stats(self):
        """Initialize partition statistics tracking."""
        self.partition_stats = {
            "last_updated": datetime.now(),
            "tables": {}
        }
    
    def get_partition_statistics(self) -> Dict[str, Any]:
        """Get comprehensive partition statistics."""
        try:
            stats = {
                "last_updated": datetime.now(),
                "tables": {}
            }
            
            with self.engine.connect() as connection:
                for table_name in self.partitioning_config.keys():
                    table_stats = self._get_table_partition_stats(connection, table_name)
                    stats["tables"][table_name] = table_stats
            
            self.partition_stats = stats
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get partition statistics: {e}")
            return {"error": str(e)}
    
    def _get_table_partition_stats(self, connection, table_name: str) -> Dict[str, Any]:
        """Get partition statistics for a specific table."""
        try:
            stats = {
                "table_name": table_name,
                "hypertable_info": {},
                "chunk_info": {},
                "compression_info": {},
                "retention_info": {}
            }
            
            # Get hypertable information
            result = connection.execute(text(f"""
                SELECT hypertable_name, num_chunks, compression_enabled 
                FROM timescaledb_information.hypertables 
                WHERE hypertable_name = '{table_name}';
            """))
            
            hypertable_info = result.fetchone()
            if hypertable_info:
                stats["hypertable_info"] = {
                    "name": hypertable_info[0],
                    "num_chunks": hypertable_info[1],
                    "compression_enabled": hypertable_info[2]
                }
            
            # Get chunk information
            result = connection.execute(text(f"""
                SELECT COUNT(*) as total_chunks,
                       COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks,
                       COUNT(*) FILTER (WHERE NOT is_compressed) as uncompressed_chunks,
                       MIN(range_start) as earliest_chunk,
                       MAX(range_end) as latest_chunk
                FROM timescaledb_information.chunks 
                WHERE hypertable_name = '{table_name}';
            """))
            
            chunk_info = result.fetchone()
            if chunk_info:
                stats["chunk_info"] = {
                    "total_chunks": chunk_info[0],
                    "compressed_chunks": chunk_info[1],
                    "uncompressed_chunks": chunk_info[2],
                    "earliest_chunk": chunk_info[3],
                    "latest_chunk": chunk_info[4]
                }
            
            # Get compression information
            result = connection.execute(text(f"""
                SELECT hypertable_name, total_chunks, number_compressed_chunks 
                FROM timescaledb_information.compression_settings 
                WHERE hypertable_name = '{table_name}';
            """))
            
            compression_info = result.fetchone()
            if compression_info:
                stats["compression_info"] = {
                    "total_chunks": compression_info[1],
                    "compressed_chunks": compression_info[2],
                    "compression_ratio": compression_info[2] / compression_info[1] if compression_info[1] > 0 else 0
                }
            
            # Get retention policy information
            result = connection.execute(text(f"""
                SELECT hypertable_name, drop_after, schedule_interval 
                FROM timescaledb_information.drop_chunks_policies 
                WHERE hypertable_name = '{table_name}';
            """))
            
            retention_info = result.fetchone()
            if retention_info:
                stats["retention_info"] = {
                    "drop_after": retention_info[1],
                    "schedule_interval": retention_info[2]
                }
            
            return stats
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get partition stats for {table_name}: {e}")
            return {"error": str(e)}
    
    def optimize_partitions(self) -> bool:
        """Run partition optimization tasks."""
        try:
            logger.info("Starting partition optimization...")
            
            with self.engine.connect() as connection:
                # Run VACUUM on all hypertables
                for table_name in self.partitioning_config.keys():
                    connection.execute(text(f"VACUUM ANALYZE {table_name};"))
                    logger.info(f"VACUUM ANALYZE completed for {table_name}")
                
                # Update table statistics
                connection.execute(text("ANALYZE;"))
                logger.info("Table statistics updated")
                
                # Reindex if needed
                connection.execute(text("REINDEX DATABASE stock_predictor;"))
                logger.info("Database reindex completed")
                
                connection.commit()
            
            logger.info("Partition optimization completed successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to optimize partitions: {e}")
            return False
    
    def get_partition_performance_metrics(self) -> Dict[str, Any]:
        """Get partition performance metrics."""
        try:
            metrics = {
                "query_performance": {},
                "storage_metrics": {},
                "compression_metrics": {}
            }
            
            with self.engine.connect() as connection:
                # Query performance metrics
                for table_name in self.partitioning_config.keys():
                    # Test query performance
                    start_time = datetime.now()
                    result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
                    count = result.fetchone()[0]
                    query_time = (datetime.now() - start_time).total_seconds()
                    
                    metrics["query_performance"][table_name] = {
                        "total_records": count,
                        "count_query_time": query_time,
                        "records_per_second": count / query_time if query_time > 0 else 0
                    }
                
                # Storage metrics
                result = connection.execute(text("""
                    SELECT schemaname, tablename, 
                           pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                           pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    AND tablename IN ('stock_data', 'system_metrics', 'data_quality_logs')
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
                """))
                
                storage_data = result.fetchall()
                for row in storage_data:
                    table_name = row[1]
                    metrics["storage_metrics"][table_name] = {
                        "size_pretty": row[2],
                        "size_bytes": row[3]
                    }
                
                # Compression metrics
                result = connection.execute(text("""
                    SELECT hypertable_name, 
                           COUNT(*) as total_chunks,
                           COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks,
                           ROUND(
                               (COUNT(*) FILTER (WHERE is_compressed)::float / COUNT(*)::float) * 100, 2
                           ) as compression_percentage
                    FROM timescaledb_information.chunks
                    GROUP BY hypertable_name;
                """))
                
                compression_data = result.fetchall()
                for row in compression_data:
                    table_name = row[0]
                    metrics["compression_metrics"][table_name] = {
                        "total_chunks": row[1],
                        "compressed_chunks": row[2],
                        "compression_percentage": row[3]
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get partition performance metrics: {e}")
            return {"error": str(e)}
    
    def monitor_partition_health(self) -> Dict[str, Any]:
        """Monitor partition health and identify issues."""
        try:
            health_report = {
                "overall_health": "good",
                "issues": [],
                "recommendations": []
            }
            
            stats = self.get_partition_statistics()
            
            for table_name, table_stats in stats.get("tables", {}).items():
                if "error" in table_stats:
                    health_report["issues"].append(f"Error getting stats for {table_name}: {table_stats['error']}")
                    health_report["overall_health"] = "poor"
                    continue
                
                # Check chunk distribution
                chunk_info = table_stats.get("chunk_info", {})
                if chunk_info:
                    total_chunks = chunk_info.get("total_chunks", 0)
                    compressed_chunks = chunk_info.get("compressed_chunks", 0)
                    
                    if total_chunks > 0:
                        compression_ratio = compressed_chunks / total_chunks
                        
                        if compression_ratio < 0.5:
                            health_report["issues"].append(
                                f"Low compression ratio for {table_name}: {compression_ratio:.2%}"
                            )
                            health_report["recommendations"].append(
                                f"Consider adjusting compression policy for {table_name}"
                            )
                        
                        if total_chunks > 1000:
                            health_report["issues"].append(
                                f"Large number of chunks for {table_name}: {total_chunks}"
                            )
                            health_report["recommendations"].append(
                                f"Consider increasing chunk interval for {table_name}"
                            )
                
                # Check for missing retention policies
                retention_info = table_stats.get("retention_info", {})
                if not retention_info:
                    health_report["issues"].append(f"No retention policy for {table_name}")
                    health_report["recommendations"].append(f"Add retention policy for {table_name}")
            
            # Determine overall health
            if len(health_report["issues"]) > 5:
                health_report["overall_health"] = "poor"
            elif len(health_report["issues"]) > 2:
                health_report["overall_health"] = "fair"
            else:
                health_report["overall_health"] = "good"
            
            return health_report
            
        except Exception as e:
            logger.error(f"Failed to monitor partition health: {e}")
            return {"error": str(e), "overall_health": "unknown"}
    
    def add_custom_partitioning_rule(self, table_name: str, rule_config: Dict[str, Any]) -> bool:
        """Add custom partitioning rule for a table."""
        try:
            if table_name not in self.partitioning_config:
                self.partitioning_config[table_name] = {}
            
            self.partitioning_config[table_name].update(rule_config)
            
            # Apply the new configuration
            success = self._setup_table_partitioning(table_name, self.partitioning_config[table_name])
            
            if success:
                logger.info(f"Custom partitioning rule added for {table_name}")
            else:
                logger.error(f"Failed to apply custom partitioning rule for {table_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add custom partitioning rule for {table_name}: {e}")
            return False
    
    def get_partition_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for partition optimization."""
        try:
            recommendations = []
            stats = self.get_partition_statistics()
            performance_metrics = self.get_partition_performance_metrics()
            
            for table_name, table_stats in stats.get("tables", {}).items():
                if "error" in table_stats:
                    continue
                
                # Check chunk size
                chunk_info = table_stats.get("chunk_info", {})
                if chunk_info:
                    total_chunks = chunk_info.get("total_chunks", 0)
                    
                    if total_chunks > 1000:
                        recommendations.append({
                            "table": table_name,
                            "type": "chunk_optimization",
                            "priority": "high",
                            "description": f"Large number of chunks ({total_chunks}), consider increasing chunk interval",
                            "action": f"Increase chunk_time_interval for {table_name}"
                        })
                
                # Check compression
                compression_info = table_stats.get("compression_info", {})
                if compression_info:
                    compression_ratio = compression_info.get("compression_ratio", 0)
                    
                    if compression_ratio < 0.3:
                        recommendations.append({
                            "table": table_name,
                            "type": "compression_optimization",
                            "priority": "medium",
                            "description": f"Low compression ratio ({compression_ratio:.2%})",
                            "action": f"Review compression settings for {table_name}"
                        })
                
                # Check query performance
                query_perf = performance_metrics.get("query_performance", {}).get(table_name, {})
                if query_perf:
                    records_per_second = query_perf.get("records_per_second", 0)
                    
                    if records_per_second < 1000:
                        recommendations.append({
                            "table": table_name,
                            "type": "performance_optimization",
                            "priority": "high",
                            "description": f"Slow query performance ({records_per_second:.0f} records/sec)",
                            "action": f"Add indexes or optimize queries for {table_name}"
                        })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get partition recommendations: {e}")
            return []

# Convenience functions
def create_partitioning_manager(db_settings: DatabaseSettings) -> DataPartitioningManager:
    """Create and return a data partitioning manager instance."""
    return DataPartitioningManager(db_settings)

def setup_data_partitioning(db_settings: DatabaseSettings) -> bool:
    """Setup data partitioning for the database."""
    manager = DataPartitioningManager(db_settings)
    return manager.setup_partitioning()

def get_partitioning_status(db_settings: DatabaseSettings) -> Dict[str, Any]:
    """Get comprehensive partitioning status and statistics."""
    manager = DataPartitioningManager(db_settings)
    return {
        "statistics": manager.get_partition_statistics(),
        "performance_metrics": manager.get_partition_performance_metrics(),
        "health_report": manager.monitor_partition_health(),
        "recommendations": manager.get_partition_recommendations()
    }