#!/usr/bin/env python3
"""
Database setup script for the ML Stock Predictor Platform.
This script initializes TimescaleDB with the proper schema, hypertables, and initial data.
"""

import os
import sys
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from data_ingestion.models import Base, DataSource, Symbol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_database_engine():
    """Create and return a database engine."""
    try:
        engine = create_engine(
            settings.database.database_url,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            echo=False  # Set to True for SQL debugging
        )
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


def test_database_connection(engine):
    """Test the database connection."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            logger.info(f"Successfully connected to database: {version}")
            return True
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return False


def create_tables(engine):
    """Create all database tables."""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False


def setup_timescale_hypertable(engine):
    """Set up TimescaleDB hypertable for stock_data."""
    try:
        with engine.connect() as connection:
            # Check if TimescaleDB extension is available
            result = connection.execute(text("SELECT * FROM pg_extension WHERE extname = 'timescaledb';"))
            if not result.fetchone():
                logger.warning("TimescaleDB extension not found. Installing...")
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
                connection.commit()
            
            # Create hypertable for stock_data
            logger.info("Setting up TimescaleDB hypertable...")
            connection.execute(text("""
                SELECT create_hypertable('stock_data', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE);
            """))
            
            # Set up compression
            if settings.database.compression_enabled:
                logger.info("Setting up TimescaleDB compression...")
                connection.execute(text(f"""
                    ALTER TABLE stock_data SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,timeframe',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
                
                # Add compression policy
                connection.execute(text(f"""
                    SELECT add_compression_policy('stock_data', INTERVAL '{settings.database.compression_after}');
                """))
            
            # Add retention policy
            retention_days = settings.data_collection.data_retention_days
            connection.execute(text(f"""
                SELECT add_retention_policy('stock_data', INTERVAL '{retention_days} days');
            """))
            
            connection.commit()
            logger.info("TimescaleDB hypertable setup completed successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to setup TimescaleDB hypertable: {e}")
        return False


def create_indexes(engine):
    """Create additional indexes for performance optimization."""
    try:
        with engine.connect() as connection:
            logger.info("Creating additional indexes...")
            
            # Create indexes for efficient querying
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_timeframe ON stock_data (symbol, timeframe, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_stock_data_source_symbol ON stock_data (source, symbol, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_stock_data_quality_time ON stock_data (data_quality_score, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_symbols_active ON symbols (is_active, data_collection_enabled);",
                "CREATE INDEX IF NOT EXISTS idx_data_sources_active ON data_sources (is_active, circuit_breaker_status);",
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_time ON data_collection_jobs (status, created_at DESC);",
                "CREATE INDEX IF NOT EXISTS idx_alerts_severity_status ON alerts (severity, status, created_at DESC);",
            ]
            
            for index_sql in indexes:
                connection.execute(text(index_sql))
            
            connection.commit()
            logger.info("Additional indexes created successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")
        return False


def insert_initial_data_sources(engine):
    """Insert initial data sources."""
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        
        logger.info("Inserting initial data sources...")
        
        # Define initial data sources
        data_sources = [
            {
                "name": "yahoo_finance",
                "display_name": "Yahoo Finance",
                "source_type": "api",
                "base_url": "https://finance.yahoo.com",
                "api_key_required": False,
                "requests_per_minute": 60,
                "requests_per_second": 5,
                "is_active": settings.data_collection.yahoo_finance_enabled
            },
            {
                "name": "alpha_vantage",
                "display_name": "Alpha Vantage",
                "source_type": "api",
                "base_url": "https://www.alphavantage.co",
                "api_key_required": True,
                "requests_per_minute": 5,  # Free tier limit
                "requests_per_second": 1,
                "is_active": settings.data_collection.alpha_vantage_enabled
            },
            {
                "name": "polygon",
                "display_name": "Polygon.io",
                "source_type": "api",
                "base_url": "https://polygon.io",
                "api_key_required": True,
                "requests_per_minute": 5,  # Free tier limit
                "requests_per_second": 1,
                "is_active": settings.data_collection.polygon_enabled
            },
            {
                "name": "marketwatch",
                "display_name": "MarketWatch",
                "source_type": "scraping",
                "base_url": "https://www.marketwatch.com",
                "api_key_required": False,
                "requests_per_minute": 30,
                "requests_per_second": 2,
                "is_active": True
            },
            {
                "name": "investing",
                "display_name": "Investing.com",
                "source_type": "scraping",
                "base_url": "https://www.investing.com",
                "api_key_required": False,
                "requests_per_minute": 30,
                "requests_per_second": 2,
                "is_active": True
            }
        ]
        
        for source_data in data_sources:
            # Check if source already exists
            existing = session.query(DataSource).filter_by(name=source_data["name"]).first()
            if not existing:
                source = DataSource(**source_data)
                session.add(source)
                logger.info(f"Added data source: {source_data['display_name']}")
            else:
                logger.info(f"Data source already exists: {source_data['display_name']}")
        
        session.commit()
        session.close()
        logger.info("Initial data sources inserted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to insert initial data sources: {e}")
        return False


def insert_sample_symbols(engine):
    """Insert sample symbols for testing."""
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        
        logger.info("Inserting sample symbols...")
        
        # Define sample symbols (major stocks)
        sample_symbols = [
            {"symbol": "AAPL", "exchange": "NASDAQ", "company_name": "Apple Inc."},
            {"symbol": "MSFT", "exchange": "NASDAQ", "company_name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "exchange": "NASDAQ", "company_name": "Alphabet Inc."},
            {"symbol": "AMZN", "exchange": "NASDAQ", "company_name": "Amazon.com Inc."},
            {"symbol": "TSLA", "exchange": "NASDAQ", "company_name": "Tesla Inc."},
            {"symbol": "META", "exchange": "NASDAQ", "company_name": "Meta Platforms Inc."},
            {"symbol": "NVDA", "exchange": "NASDAQ", "company_name": "NVIDIA Corporation"},
            {"symbol": "JPM", "exchange": "NYSE", "company_name": "JPMorgan Chase & Co."},
            {"symbol": "JNJ", "exchange": "NYSE", "company_name": "Johnson & Johnson"},
            {"symbol": "V", "exchange": "NYSE", "company_name": "Visa Inc."},
            {"symbol": "WMT", "exchange": "NYSE", "company_name": "Walmart Inc."},
            {"symbol": "PG", "exchange": "NYSE", "company_name": "Procter & Gamble Co."},
            {"symbol": "UNH", "exchange": "NYSE", "company_name": "UnitedHealth Group Inc."},
            {"symbol": "HD", "exchange": "NYSE", "company_name": "The Home Depot Inc."},
            {"symbol": "MA", "exchange": "NYSE", "company_name": "Mastercard Inc."},
        ]
        
        for symbol_data in sample_symbols:
            # Check if symbol already exists
            existing = session.query(Symbol).filter_by(symbol=symbol_data["symbol"]).first()
            if not existing:
                symbol = Symbol(**symbol_data)
                session.add(symbol)
                logger.info(f"Added symbol: {symbol_data['symbol']}")
            else:
                logger.info(f"Symbol already exists: {symbol_data['symbol']}")
        
        session.commit()
        session.close()
        logger.info("Sample symbols inserted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to insert sample symbols: {e}")
        return False


def setup_database():
    """Main function to set up the database."""
    logger.info("Starting database setup...")
    
    try:
        # Create database engine
        engine = create_database_engine()
        
        # Test connection
        if not test_database_connection(engine):
            logger.error("Database connection test failed")
            return False
        
        # Create tables
        if not create_tables(engine):
            logger.error("Table creation failed")
            return False
        
        # Setup TimescaleDB hypertable
        if not setup_timescale_hypertable(engine):
            logger.error("TimescaleDB setup failed")
            return False
        
        # Create indexes
        if not create_indexes(engine):
            logger.error("Index creation failed")
            return False
        
        # Insert initial data
        if not insert_initial_data_sources(engine):
            logger.error("Data source insertion failed")
            return False
        
        if not insert_sample_symbols(engine):
            logger.error("Sample symbol insertion failed")
            return False
        
        logger.info("Database setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def main():
    """Main entry point."""
    logger.info("ML Stock Predictor Platform - Database Setup")
    logger.info("=" * 50)
    
    success = setup_database()
    
    if success:
        logger.info("Database setup completed successfully!")
        logger.info("You can now start the data collection system.")
    else:
        logger.error("Database setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()