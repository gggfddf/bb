#!/usr/bin/env python3
"""
Main application runner for the ML Stock Predictor Platform.
This script starts all the necessary components of the trading system.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings, validate_settings
from data_ingestion.orchestrator import DataIngestionOrchestrator
from feature_engineering.pipeline import FeatureEngineeringPipeline
from ml_models.trainer import ModelTrainer
from backtesting.engine import BacktestingEngine
from visualization.dashboard import Dashboard
from integration.orchestration.orchestrator import SystemOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class TradingSystemApp:
    """Main application class for the trading system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.orchestrator = None
        self.data_orchestrator = None
        self.feature_pipeline = None
        self.model_trainer = None
        self.backtesting_engine = None
        self.dashboard = None
        self.running = False
    
    async def initialize(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing ML Stock Predictor Platform...")
            
            # Validate settings
            validate_settings()
            logger.info("Settings validated successfully")
            
            # Initialize system orchestrator
            self.orchestrator = SystemOrchestrator()
            await self.orchestrator.initialize()
            logger.info("System orchestrator initialized")
            
            # Initialize data ingestion
            self.data_orchestrator = DataIngestionOrchestrator()
            await self.data_orchestrator.initialize()
            logger.info("Data ingestion orchestrator initialized")
            
            # Initialize feature engineering pipeline
            self.feature_pipeline = FeatureEngineeringPipeline()
            await self.feature_pipeline.initialize()
            logger.info("Feature engineering pipeline initialized")
            
            # Initialize model trainer
            self.model_trainer = ModelTrainer()
            await self.model_trainer.initialize()
            logger.info("Model trainer initialized")
            
            # Initialize backtesting engine
            self.backtesting_engine = BacktestingEngine()
            await self.backtesting_engine.initialize()
            logger.info("Backtesting engine initialized")
            
            # Initialize dashboard
            self.dashboard = Dashboard()
            await self.dashboard.initialize()
            logger.info("Dashboard initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
    
    async def start(self):
        """Start all system components."""
        try:
            logger.info("Starting ML Stock Predictor Platform...")
            
            # Start system orchestrator
            await self.orchestrator.start()
            logger.info("System orchestrator started")
            
            # Start data ingestion
            await self.data_orchestrator.start()
            logger.info("Data ingestion started")
            
            # Start feature engineering pipeline
            await self.feature_pipeline.start()
            logger.info("Feature engineering pipeline started")
            
            # Start model trainer
            await self.model_trainer.start()
            logger.info("Model trainer started")
            
            # Start backtesting engine
            await self.backtesting_engine.start()
            logger.info("Backtesting engine started")
            
            # Start dashboard
            await self.dashboard.start()
            logger.info("Dashboard started")
            
            self.running = True
            logger.info("ML Stock Predictor Platform started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            raise
    
    async def stop(self):
        """Stop all system components."""
        try:
            logger.info("Stopping ML Stock Predictor Platform...")
            
            self.running = False
            
            # Stop dashboard
            if self.dashboard:
                await self.dashboard.stop()
                logger.info("Dashboard stopped")
            
            # Stop backtesting engine
            if self.backtesting_engine:
                await self.backtesting_engine.stop()
                logger.info("Backtesting engine stopped")
            
            # Stop model trainer
            if self.model_trainer:
                await self.model_trainer.stop()
                logger.info("Model trainer stopped")
            
            # Stop feature engineering pipeline
            if self.feature_pipeline:
                await self.feature_pipeline.stop()
                logger.info("Feature engineering pipeline stopped")
            
            # Stop data ingestion
            if self.data_orchestrator:
                await self.data_orchestrator.stop()
                logger.info("Data ingestion stopped")
            
            # Stop system orchestrator
            if self.orchestrator:
                await self.orchestrator.stop()
                logger.info("System orchestrator stopped")
            
            logger.info("ML Stock Predictor Platform stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping application: {e}")
    
    async def run(self):
        """Run the application."""
        try:
            await self.initialize()
            await self.start()
            
            # Keep the application running
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            await self.stop()


async def main():
    """Main entry point."""
    app = TradingSystemApp()
    await app.run()


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Run the application
    asyncio.run(main())