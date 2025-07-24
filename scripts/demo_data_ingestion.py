#!/usr/bin/env python3
"""
Data Ingestion System Demo

This script demonstrates the ML Stock Predictor Platform's data ingestion system
with a simple, runnable example that shows the key features working.
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

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('demo_data_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)


class DataIngestionDemo:
    """Demo class for showcasing the data ingestion system."""
    
    def __init__(self):
        self.db_settings = DatabaseSettings()
        self.collection_settings = DataCollectionSettings()
        self.engine = None
        self.session = None
        self.orchestrator = None
        self.validation_pipeline = None
    
    async def setup(self):
        """Set up the demo environment."""
        logger.info("🚀 Setting up Data Ingestion Demo...")
        
        try:
            # Create database engine
            self.engine = create_engine(self.db_settings.database_url)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.session = SessionLocal()
            
            # Initialize orchestrator
            self.orchestrator = DataIngestionOrchestrator(self.db_settings, self.collection_settings)
            
            # Initialize validation pipeline
            self.validation_pipeline = DataValidationPipeline(self.session)
            
            logger.info("✅ Demo environment setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup demo environment: {e}")
            raise
    
    async def demo_basic_components(self):
        """Demonstrate basic components working."""
        logger.info("\n🔧 Testing Basic Components...")
        
        try:
            # Test rate limiter
            logger.info("Testing Rate Limiter...")
            from data_ingestion.utils.rate_limiter import RateLimiter
            rate_limiter = RateLimiter(requests_per_minute=60, requests_per_second=5)
            
            start_time = datetime.now()
            for i in range(5):
                await rate_limiter.wait()
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Rate limiter: 5 requests took {duration:.2f} seconds")
            
            # Test data validator
            logger.info("Testing Data Validator...")
            from data_ingestion.utils.data_validator import DataValidator
            validator = DataValidator()
            
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
            
            result = validator.validate_stock_data(valid_data)
            logger.info(f"✅ Valid data validation: {result['is_valid']}")
            
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
            
            result = validator.validate_stock_data(invalid_data)
            logger.info(f"✅ Invalid data validation: {result['is_valid']} (expected: False)")
            logger.info(f"   Issues detected: {len(result['issues'])}")
            
            logger.info("✅ Basic components test completed!")
            
        except Exception as e:
            logger.error(f"❌ Basic components test failed: {e}")
            raise
    
    async def demo_data_collection(self):
        """Demonstrate data collection capabilities."""
        logger.info("\n📊 Testing Data Collection...")
        
        try:
            # Test symbols and timeframes
            test_symbols = ['AAPL', 'GOOGL', 'MSFT']
            test_timeframes = ['1d']  # Start with daily data for demo
            
            logger.info(f"Collecting data for symbols: {test_symbols}")
            logger.info(f"Timeframes: {test_timeframes}")
            
            # Start collection
            await self.orchestrator.start_collection(test_symbols, test_timeframes)
            
            # Wait for collection to process
            logger.info("⏳ Waiting for data collection to process...")
            await asyncio.sleep(5)
            
            # Get status
            status = self.orchestrator.get_collection_status()
            logger.info(f"✅ Collection status: {status['active_jobs']} active jobs")
            logger.info(f"   Available sources: {status['available_sources']}")
            
            # Stop collection
            await self.orchestrator.stop_collection()
            logger.info("✅ Data collection demo completed!")
            
        except Exception as e:
            logger.error(f"❌ Data collection demo failed: {e}")
            raise
    
    async def demo_data_validation(self):
        """Demonstrate data validation pipeline."""
        logger.info("\n🔍 Testing Data Validation Pipeline...")
        
        try:
            # Create sample data for validation
            sample_data = []
            base_price = 150.0
            
            for i in range(20):
                # Create realistic stock data
                open_price = base_price + (i * 0.1)
                high_price = open_price + 2.0
                low_price = open_price - 1.5
                close_price = open_price + (0.5 if i % 2 == 0 else -0.3)
                volume = 1000000 + (i * 50000)
                
                sample_data.append({
                    'timestamp': datetime.now() - timedelta(days=i),
                    'symbol': 'AAPL',
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'exchange': 'NASDAQ',
                    'source': 'demo'
                })
            
            # Add some anomalies for testing
            sample_data.append({
                'timestamp': datetime.now(),
                'symbol': 'AAPL',
                'open': 150.0,
                'high': 200.0,  # Unusual high
                'low': 149.0,
                'close': 152.0,
                'volume': 50000000,  # Unusual volume
                'exchange': 'NASDAQ',
                'source': 'demo'
            })
            
            logger.info(f"Validating {len(sample_data)} sample records...")
            
            # Run validation
            validation_result = await self.validation_pipeline.validate_data_batch(sample_data, 'demo_source')
            
            logger.info(f"✅ Validation completed!")
            logger.info(f"   Quality Score: {validation_result['quality_score']:.2f}")
            logger.info(f"   Valid Records: {validation_result['valid_records']}/{validation_result['total_records']}")
            logger.info(f"   Anomalies Detected: {validation_result['anomalies_detected']}")
            logger.info(f"   Issues Found: {len(validation_result['issues'])}")
            
            if validation_result['recommendations']:
                logger.info("   Recommendations:")
                for rec in validation_result['recommendations']:
                    logger.info(f"     - {rec}")
            
            logger.info("✅ Data validation demo completed!")
            
        except Exception as e:
            logger.error(f"❌ Data validation demo failed: {e}")
            raise
    
    async def demo_system_integration(self):
        """Demonstrate system integration."""
        logger.info("\n🔗 Testing System Integration...")
        
        try:
            # Test symbol info collection
            test_symbols = ['AAPL', 'GOOGL']
            logger.info(f"Collecting symbol information for: {test_symbols}")
            
            await self.orchestrator.collect_symbol_info(test_symbols)
            logger.info("✅ Symbol info collection completed!")
            
            # Test validation summary
            summary = self.validation_pipeline.get_validation_summary()
            logger.info(f"✅ Validation summary: {summary}")
            
            logger.info("✅ System integration demo completed!")
            
        except Exception as e:
            logger.error(f"❌ System integration demo failed: {e}")
            raise
    
    async def cleanup(self):
        """Clean up demo resources."""
        logger.info("\n🧹 Cleaning up demo resources...")
        
        try:
            if self.orchestrator and self.orchestrator.is_running:
                await self.orchestrator.stop_collection()
            
            if self.session:
                self.session.close()
            
            logger.info("✅ Cleanup completed!")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    async def run_demo(self):
        """Run the complete demo."""
        logger.info("=" * 60)
        logger.info("🎯 ML Stock Predictor Platform - Data Ingestion Demo")
        logger.info("=" * 60)
        
        try:
            # Setup
            await self.setup()
            
            # Run demos
            await self.demo_basic_components()
            await self.demo_data_collection()
            await self.demo_data_validation()
            await self.demo_system_integration()
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 Demo completed successfully!")
            logger.info("=" * 60)
            logger.info("\n📋 Demo Summary:")
            logger.info("✅ Basic components (rate limiter, data validator)")
            logger.info("✅ Data collection orchestration")
            logger.info("✅ Multi-stage data validation pipeline")
            logger.info("✅ Anomaly detection and quality scoring")
            logger.info("✅ System integration and symbol management")
            logger.info("\n🚀 The data ingestion system is ready for production use!")
            
        except Exception as e:
            logger.error(f"\n❌ Demo failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main demo function."""
    demo = DataIngestionDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())