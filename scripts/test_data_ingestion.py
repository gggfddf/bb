#!/usr/bin/env python3
"""
Test Data Ingestion System

This script demonstrates the data ingestion system working with the components
we've built for the ML Stock Predictor Platform.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import DatabaseSettings, DataCollectionSettings
from data_ingestion.orchestrator import DataIngestionOrchestrator
from data_ingestion.validation.data_validation_pipeline import DataValidationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_ingestion():
    """Test the data ingestion system with sample data."""
    
    logger.info("Starting data ingestion system test")
    
    try:
        # Initialize settings
        db_settings = DatabaseSettings()
        collection_settings = DataCollectionSettings()
        
        # Create database engine and session
        engine = create_engine(db_settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        # Initialize orchestrator
        orchestrator = DataIngestionOrchestrator(db_settings, collection_settings)
        
        # Initialize validation pipeline
        validation_pipeline = DataValidationPipeline(session)
        
        # Test symbols and timeframes
        test_symbols = ['AAPL', 'GOOGL', 'MSFT']
        test_timeframes = ['1d']  # Start with daily data for testing
        
        logger.info(f"Testing with symbols: {test_symbols}")
        logger.info(f"Testing with timeframes: {test_timeframes}")
        
        # Start data collection
        await orchestrator.start_collection(test_symbols, test_timeframes)
        
        # Wait for collection to complete (in a real scenario, this would run continuously)
        logger.info("Waiting for data collection to complete...")
        await asyncio.sleep(10)  # Wait 10 seconds for demonstration
        
        # Stop collection
        await orchestrator.stop_collection()
        
        # Get collection status
        status = orchestrator.get_collection_status()
        logger.info(f"Collection status: {status}")
        
        # Test data validation
        logger.info("Testing data validation pipeline...")
        
        # Get some sample data from the database for validation
        from data_ingestion.models import StockData
        
        sample_data = session.query(StockData).limit(100).all()
        
        if sample_data:
            # Convert to dictionary format for validation
            data_batch = []
            for record in sample_data:
                data_batch.append({
                    'timestamp': record.timestamp,
                    'symbol': record.symbol,
                    'open': float(record.open),
                    'high': float(record.high),
                    'low': float(record.low),
                    'close': float(record.close),
                    'volume': record.volume,
                    'exchange': record.exchange,
                    'source': record.source
                })
            
            # Run validation
            validation_result = await validation_pipeline.validate_data_batch(data_batch, 'test_source')
            logger.info(f"Validation result: {validation_result}")
            
            # Get validation summary
            summary = validation_pipeline.get_validation_summary()
            logger.info(f"Validation summary: {summary}")
        else:
            logger.warning("No data found in database for validation testing")
        
        # Test symbol info collection
        logger.info("Testing symbol info collection...")
        await orchestrator.collect_symbol_info(test_symbols)
        
        logger.info("Data ingestion system test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data ingestion test: {e}")
        raise
    finally:
        if 'session' in locals():
            session.close()


async def test_individual_components():
    """Test individual components of the data ingestion system."""
    
    logger.info("Testing individual data ingestion components")
    
    try:
        # Initialize settings
        db_settings = DatabaseSettings()
        collection_settings = DataCollectionSettings()
        
        # Create database engine and session
        engine = create_engine(db_settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        # Test rate limiter
        logger.info("Testing rate limiter...")
        from data_ingestion.utils.rate_limiter import RateLimiter
        rate_limiter = RateLimiter(requests_per_minute=60, requests_per_second=5)
        
        start_time = datetime.now()
        for i in range(10):
            await rate_limiter.wait()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Rate limiter test: 10 requests took {duration:.2f} seconds")
        
        # Test data validator
        logger.info("Testing data validator...")
        from data_ingestion.utils.data_validator import DataValidator
        data_validator = DataValidator()
        
        # Test valid data
        valid_data = {
            'timestamp': datetime.now(),
            'symbol': 'AAPL',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 152.0,
            'volume': 1000000
        }
        
        validation_result = data_validator.validate_stock_data(valid_data)
        logger.info(f"Valid data validation: {validation_result['is_valid']}")
        
        # Test invalid data
        invalid_data = {
            'timestamp': datetime.now(),
            'symbol': 'AAPL',
            'open': 150.0,
            'high': 145.0,  # High < Open (invalid)
            'low': 149.0,
            'close': 152.0,
            'volume': -1000  # Negative volume (invalid)
        }
        
        validation_result = data_validator.validate_stock_data(invalid_data)
        logger.info(f"Invalid data validation: {validation_result['is_valid']}")
        logger.info(f"Validation issues: {validation_result['issues']}")
        
        # Test error handler
        logger.info("Testing error handler...")
        from data_ingestion.utils.error_handler import ErrorHandler
        error_handler = ErrorHandler(max_retries=3, base_delay=1.0)
        
        @error_handler.retry_with_backoff()
        async def test_function():
            # Simulate a function that might fail
            import random
            if random.random() < 0.7:  # 70% chance of failure
                raise Exception("Simulated error")
            return "Success"
        
        try:
            result = await test_function()
            logger.info(f"Error handler test result: {result}")
        except Exception as e:
            logger.info(f"Error handler test: Function failed after retries: {e}")
        
        logger.info("Individual component tests completed!")
        
    except Exception as e:
        logger.error(f"Error during component testing: {e}")
        raise
    finally:
        if 'session' in locals():
            session.close()


async def main():
    """Main test function."""
    
    logger.info("=" * 60)
    logger.info("ML Stock Predictor Platform - Data Ingestion System Test")
    logger.info("=" * 60)
    
    try:
        # Test individual components first
        await test_individual_components()
        
        logger.info("-" * 40)
        
        # Test the full data ingestion system
        await test_data_ingestion()
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())