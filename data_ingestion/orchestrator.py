"""
Data Ingestion Orchestrator

Coordinates all data collection sources, manages the collection pipeline,
and handles scheduling for the ML Stock Predictor Platform.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from .models import StockData, Symbol, DataSource, DataCollectionJob, DataQualityLog
from .scrapers.yahoo_finance_scraper import YahooFinanceScraper
from .utils.rate_limiter import RateLimiter
from .utils.data_validator import DataValidator
from .utils.error_handler import ErrorHandler, global_error_handler
from .utils.error_monitoring import start_error_monitoring, get_monitoring_summary
from config.settings import DatabaseSettings, DataCollectionSettings

logger = logging.getLogger(__name__)


class DataIngestionOrchestrator:
    """
    Main orchestrator for data ingestion across multiple sources.
    
    Responsibilities:
    - Coordinate data collection from multiple sources
    - Manage data quality and validation
    - Handle scheduling and job management
    - Provide monitoring and alerting
    """
    
    def __init__(self, db_settings: DatabaseSettings, collection_settings: DataCollectionSettings):
        self.db_settings = db_settings
        self.collection_settings = collection_settings
        self.engine = create_engine(db_settings.database_url)
        
        # Initialize utilities
        self.rate_limiter = RateLimiter()
        self.data_validator = DataValidator()
        self.error_handler = global_error_handler  # Use global error handler
        
        # Initialize scrapers
        self.scrapers: Dict[str, Any] = {}
        self._initialize_scrapers()
        
        # Collection state
        self.is_running = False
        self.current_jobs: Dict[str, DataCollectionJob] = {}
        
        # Error monitoring
        self.monitoring_active = False
        
    def _initialize_scrapers(self):
        """Initialize all available data scrapers."""
        try:
            # Create database session for scrapers
            session = Session(self.engine)
            
            # Initialize Yahoo Finance scraper
            if self.collection_settings.yahoo_finance_enabled:
                self.scrapers['yahoo_finance'] = YahooFinanceScraper(session)
                logger.info("Yahoo Finance scraper initialized")
            
            # Initialize Alpha Vantage scraper
            if self.collection_settings.alpha_vantage_api_key:
                from .scrapers.alpha_vantage_scraper import AlphaVantageScraper
                self.scrapers['alpha_vantage'] = AlphaVantageScraper(
                    session, 
                    api_key=self.collection_settings.alpha_vantage_api_key
                )
                logger.info("Alpha Vantage scraper initialized")
            
            # Initialize WebSocket streamer
            if self.collection_settings.websocket_enabled:
                from .streamers.websocket_streamer import WebSocketStreamer
                ws_config = {
                    'websocket_url': self.collection_settings.websocket_url,
                    'api_key': self.collection_settings.websocket_api_key,
                    'api_secret': self.collection_settings.websocket_api_secret
                }
                self.scrapers['websocket'] = WebSocketStreamer(session, ws_config)
                logger.info("WebSocket streamer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize scrapers: {e}")
            raise
    
    async def start_collection(self, symbols: List[str], timeframes: List[str] = None):
        """
        Start data collection for specified symbols and timeframes.
        
        Args:
            symbols: List of stock symbols to collect data for
            timeframes: List of timeframes to collect (1m, 5m, 15m, 1h, 1d, 1w, 1m)
        """
        if timeframes is None:
            timeframes = ['1d']  # Default to daily data
        
        self.is_running = True
        
        # Start error monitoring
        if not self.monitoring_active:
            await start_error_monitoring()
            self.monitoring_active = True
            logger.info("Error monitoring started")
        logger.info(f"Starting data collection for {len(symbols)} symbols across {len(timeframes)} timeframes")
        
        try:
            # Create collection jobs
            for symbol in symbols:
                for timeframe in timeframes:
                    job_id = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    job = DataCollectionJob(
                        job_id=job_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        status="running",
                        started_at=datetime.now()
                    )
                    self.current_jobs[job_id] = job
                    
                    # Start collection task
                    asyncio.create_task(self._collect_data_for_job(job))
            
            logger.info(f"Started {len(self.current_jobs)} collection jobs")
            
        except Exception as e:
            logger.error(f"Failed to start collection: {e}")
            self.is_running = False
            raise
    
    async def stop_collection(self):
        """Stop all running data collection jobs."""
        self.is_running = False
        logger.info("Stopping data collection...")
        
        # Update job statuses
        session = Session(self.engine)
        try:
            for job in self.current_jobs.values():
                job.status = "stopped"
                job.ended_at = datetime.now()
                session.add(job)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to update job statuses: {e}")
        finally:
            session.close()
        
        self.current_jobs.clear()
        logger.info("Data collection stopped")
    
    async def _collect_data_for_job(self, job: DataCollectionJob):
        """
        Collect data for a specific job (symbol + timeframe combination).
        
        Args:
            job: DataCollectionJob instance
        """
        logger.info(f"Starting collection for {job.symbol} ({job.timeframe})")
        
        try:
            # Determine collection parameters based on timeframe
            collection_params = self._get_collection_params(job.timeframe)
            
            # Collect from all available sources
            for source_name, scraper in self.scrapers.items():
                if not self.is_running:
                    break
                    
                try:
                    await self._collect_from_source(job, scraper, source_name, collection_params)
                except Exception as e:
                    logger.error(f"Failed to collect from {source_name} for {job.symbol}: {e}")
                    self._log_data_quality_issue(job, source_name, str(e))
                    # Record error in global error handler
                    await self.error_handler._handle_final_failure(e, f"collect_from_{source_name}", (job, scraper, source_name, collection_params), {})
            
            # Mark job as completed
            job.status = "completed"
            job.ended_at = datetime.now()
            
            session = Session(self.engine)
            try:
                session.add(job)
                session.commit()
            finally:
                session.close()
                
            logger.info(f"Completed collection for {job.symbol} ({job.timeframe})")
            
        except Exception as e:
            logger.error(f"Failed to collect data for {job.symbol} ({job.timeframe}): {e}")
            job.status = "failed"
            job.ended_at = datetime.now()
            job.error_message = str(e)
            
            session = Session(self.engine)
            try:
                session.add(job)
                session.commit()
            finally:
                session.close()
            
            # Record error in global error handler
            await self.error_handler._handle_final_failure(e, "collect_data_for_job", (job,), {})
    
    async def _collect_from_source(self, job: DataCollectionJob, scraper, source_name: str, params: Dict[str, Any]):
        """
        Collect data from a specific source.
        
        Args:
            job: DataCollectionJob instance
            scraper: Scraper instance
            source_name: Name of the data source
            params: Collection parameters
        """
        logger.debug(f"Collecting from {source_name} for {job.symbol}")
        
        # Get historical data
        if params.get('collect_historical', True):
            historical_data = await scraper.get_historical_data(
                symbol=job.symbol,
                start_date=params['start_date'],
                end_date=params['end_date'],
                interval=params['interval']
            )
            
            if historical_data:
                await self._process_and_store_data(historical_data, job, source_name)
        
        # Get real-time data if supported
        if params.get('collect_realtime', False) and hasattr(scraper, 'get_realtime_data'):
            realtime_data = await scraper.get_realtime_data(job.symbol)
            if realtime_data:
                await self._process_and_store_data([realtime_data], job, source_name)
    
    async def _process_and_store_data(self, data_list: List[Dict[str, Any]], job: DataCollectionJob, source_name: str):
        """
        Process and store collected data.
        
        Args:
            data_list: List of data dictionaries
            job: DataCollectionJob instance
            source_name: Name of the data source
        """
        session = Session(self.engine)
        try:
            stored_count = 0
            
            for data in data_list:
                # Validate data
                validation_result = self.data_validator.validate_stock_data(data)
                
                if validation_result['is_valid']:
                    # Create StockData record
                    stock_data = StockData(
                        timestamp=data['timestamp'],
                        symbol=data['symbol'],
                        exchange=data.get('exchange', 'UNKNOWN'),
                        open=data['open'],
                        high=data['high'],
                        low=data['low'],
                        close=data['close'],
                        volume=data.get('volume', 0),
                        timeframe=job.timeframe,
                        source=source_name
                    )
                    
                    session.add(stock_data)
                    stored_count += 1
                else:
                    # Log data quality issues
                    self._log_data_quality_issue(job, source_name, validation_result['issues'])
            
            session.commit()
            logger.info(f"Stored {stored_count} records for {job.symbol} from {source_name}")
            
        except Exception as e:
            logger.error(f"Failed to store data for {job.symbol}: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def _get_collection_params(self, timeframe: str) -> Dict[str, Any]:
        """
        Get collection parameters based on timeframe.
        
        Args:
            timeframe: Timeframe string (1m, 5m, 15m, 1h, 1d, 1w, 1m)
        
        Returns:
            Dictionary of collection parameters
        """
        now = datetime.now()
        
        # Define collection parameters for each timeframe
        params_map = {
            '1m': {
                'start_date': now - timedelta(days=7),  # 1 week of 1-minute data
                'end_date': now,
                'interval': '1m',
                'collect_historical': True,
                'collect_realtime': True
            },
            '5m': {
                'start_date': now - timedelta(days=30),  # 1 month of 5-minute data
                'end_date': now,
                'interval': '5m',
                'collect_historical': True,
                'collect_realtime': True
            },
            '15m': {
                'start_date': now - timedelta(days=90),  # 3 months of 15-minute data
                'end_date': now,
                'interval': '15m',
                'collect_historical': True,
                'collect_realtime': True
            },
            '1h': {
                'start_date': now - timedelta(days=365),  # 1 year of hourly data
                'end_date': now,
                'interval': '1h',
                'collect_historical': True,
                'collect_realtime': False
            },
            '1d': {
                'start_date': now - timedelta(days=1095),  # 3 years of daily data
                'end_date': now,
                'interval': '1d',
                'collect_historical': True,
                'collect_realtime': False
            },
            '1w': {
                'start_date': now - timedelta(days=3650),  # 10 years of weekly data
                'end_date': now,
                'interval': '1wk',
                'collect_historical': True,
                'collect_realtime': False
            },
            '1m': {
                'start_date': now - timedelta(days=3650),  # 10 years of monthly data
                'end_date': now,
                'interval': '1mo',
                'collect_historical': True,
                'collect_realtime': False
            }
        }
        
        return params_map.get(timeframe, params_map['1d'])
    
    def _log_data_quality_issue(self, job: DataCollectionJob, source_name: str, issues: str):
        """
        Log data quality issues.
        
        Args:
            job: DataCollectionJob instance
            source_name: Name of the data source
            issues: Description of the issues
        """
        session = Session(self.engine)
        try:
            quality_log = DataQualityLog(
                symbol=job.symbol,
                timeframe=job.timeframe,
                source=source_name,
                issue_type="data_quality",
                description=issues,
                severity="warning"
            )
            session.add(quality_log)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log data quality issue: {e}")
        finally:
            session.close()
    
    def get_collection_status(self) -> Dict[str, Any]:
        """
        Get current collection status.
        
        Returns:
            Dictionary with collection status information
        """
        return {
            'is_running': self.is_running,
            'active_jobs': len(self.current_jobs),
            'available_sources': list(self.scrapers.keys()),
            'jobs': {
                job_id: {
                    'symbol': job.symbol,
                    'timeframe': job.timeframe,
                    'status': job.status,
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'ended_at': job.ended_at.isoformat() if job.ended_at else None,
                    'error_message': job.error_message
                }
                for job_id, job in self.current_jobs.items()
            },
            'error_monitoring': {
                'active': self.monitoring_active,
                'error_summary': self.error_handler.get_error_summary(),
                'monitoring_summary': get_monitoring_summary()
            }
        }
    
    async def collect_symbol_info(self, symbols: List[str]):
        """
        Collect and store symbol information.
        
        Args:
            symbols: List of stock symbols
        """
        logger.info(f"Collecting symbol information for {len(symbols)} symbols")
        
        session = Session(self.engine)
        try:
            for symbol in symbols:
                for source_name, scraper in self.scrapers.items():
                    try:
                        if hasattr(scraper, 'get_symbol_info'):
                            symbol_info = await scraper.get_symbol_info(symbol)
                            if symbol_info:
                                # Store symbol information
                                symbol_record = Symbol(
                                    symbol=symbol,
                                    name=symbol_info.get('name', ''),
                                    exchange=symbol_info.get('exchange', ''),
                                    sector=symbol_info.get('sector', ''),
                                    industry=symbol_info.get('industry', ''),
                                    source=source_name
                                )
                                session.add(symbol_record)
                                logger.info(f"Stored symbol info for {symbol} from {source_name}")
                    except Exception as e:
                        logger.error(f"Failed to get symbol info for {symbol} from {source_name}: {e}")
            
            session.commit()
            logger.info("Symbol information collection completed")
            
        except Exception as e:
            logger.error(f"Failed to collect symbol information: {e}")
            session.rollback()
            raise
        finally:
            session.close()