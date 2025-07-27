"""
Celery tasks for data collection operations.
Handles distributed stock data collection from various sources.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from celery import current_task
import structlog

from ..orchestrator import DataIngestionOrchestrator
from ..utils.error_handler import global_error_handler
from config.settings import DataIngestionSettings

logger = structlog.get_logger()

@current_task.app.task(bind=True, name='data_ingestion.tasks.data_collection_tasks.collect_stock_data')
def collect_stock_data(self, symbols: Optional[List[str]] = None, 
                      timeframes: Optional[List[str]] = None,
                      sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Celery task to collect stock data from configured sources.
    
    Args:
        symbols: List of stock symbols to collect data for
        timeframes: List of timeframes to collect (1m, 5m, 15m, 1h, 1d, 1w, 1m)
        sources: List of data sources to use
        
    Returns:
        Dict containing collection results and statistics
    """
    task_id = self.request.id
    logger.info("Starting stock data collection task", task_id=task_id)
    
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Initializing data collection', 'progress': 0}
        )
        
        # Get settings
        settings = DataIngestionSettings()
        
        # Use default values if not provided
        if symbols is None:
            symbols = settings.default_symbols
        if timeframes is None:
            timeframes = settings.default_timeframes
        if sources is None:
            sources = settings.default_sources
        
        # Create orchestrator
        orchestrator = DataIngestionOrchestrator(settings)
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Created orchestrator', 'progress': 10}
        )
        
        # Start collection
        collection_job = {
            'symbols': symbols,
            'timeframes': timeframes,
            'sources': sources,
            'start_time': datetime.now(),
            'priority': 'normal'
        }
        
        # Run collection asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Start collection
            loop.run_until_complete(orchestrator.start_collection())
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Collection started', 'progress': 30}
            )
            
            # Collect data for the job
            result = loop.run_until_complete(
                orchestrator._collect_data_for_job(collection_job)
            )
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Collection completed', 'progress': 90}
            )
            
            # Get final status
            status = orchestrator.get_collection_status()
            
            # Stop collection
            loop.run_until_complete(orchestrator.stop_collection())
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Finalizing', 'progress': 100}
            )
            
            logger.info("Stock data collection task completed successfully", 
                       task_id=task_id, result=result)
            
            return {
                'success': True,
                'task_id': task_id,
                'collection_result': result,
                'status': status,
                'symbols_collected': len(symbols),
                'timeframes_collected': len(timeframes),
                'sources_used': len(sources),
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        # Record error
        global_error_handler._handle_final_failure(
            error=e,
            context="celery_task",
            task_id=task_id,
            operation="collect_stock_data"
        )
        
        logger.error("Stock data collection task failed", 
                    task_id=task_id, error=str(e))
        
        return {
            'success': False,
            'task_id': task_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@current_task.app.task(bind=True, name='data_ingestion.tasks.data_collection_tasks.collect_historical_data')
def collect_historical_data(self, symbol: str, start_date: str, end_date: str,
                          timeframe: str = '1d', source: str = 'yahoo') -> Dict[str, Any]:
    """
    Celery task to collect historical stock data.
    
    Args:
        symbol: Stock symbol to collect data for
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timeframe: Data timeframe (1m, 5m, 15m, 1h, 1d, 1w, 1m)
        source: Data source to use
        
    Returns:
        Dict containing collection results
    """
    task_id = self.request.id
    logger.info("Starting historical data collection task", 
               task_id=task_id, symbol=symbol, start_date=start_date, end_date=end_date)
    
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Initializing historical collection', 'progress': 0}
        )
        
        # Get settings
        settings = DataIngestionSettings()
        
        # Create orchestrator
        orchestrator = DataIngestionOrchestrator(settings)
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Created orchestrator', 'progress': 20}
        )
        
        # Create historical collection job
        collection_job = {
            'symbols': [symbol],
            'timeframes': [timeframe],
            'sources': [source],
            'start_date': start_date,
            'end_date': end_date,
            'start_time': datetime.now(),
            'priority': 'high'
        }
        
        # Run collection asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Start collection
            loop.run_until_complete(orchestrator.start_collection())
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Collection started', 'progress': 40}
            )
            
            # Collect historical data
            result = loop.run_until_complete(
                orchestrator._collect_data_for_job(collection_job)
            )
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Collection completed', 'progress': 90}
            )
            
            # Stop collection
            loop.run_until_complete(orchestrator.stop_collection())
            
            logger.info("Historical data collection task completed successfully", 
                       task_id=task_id, symbol=symbol, result=result)
            
            return {
                'success': True,
                'task_id': task_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'source': source,
                'start_date': start_date,
                'end_date': end_date,
                'collection_result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        # Record error
        global_error_handler._handle_final_failure(
            error=e,
            context="celery_task",
            task_id=task_id,
            operation="collect_historical_data",
            symbol=symbol
        )
        
        logger.error("Historical data collection task failed", 
                    task_id=task_id, symbol=symbol, error=str(e))
        
        return {
            'success': False,
            'task_id': task_id,
            'symbol': symbol,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@current_task.app.task(bind=True, name='data_ingestion.tasks.data_collection_tasks.validate_data_quality')
def validate_data_quality(self, symbol: str, timeframe: str, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Celery task to validate data quality for a specific symbol and timeframe.
    
    Args:
        symbol: Stock symbol to validate
        timeframe: Data timeframe to validate
        start_date: Start date for validation period (optional)
        end_date: End date for validation period (optional)
        
    Returns:
        Dict containing validation results
    """
    task_id = self.request.id
    logger.info("Starting data quality validation task", 
               task_id=task_id, symbol=symbol, timeframe=timeframe)
    
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Initializing validation', 'progress': 0}
        )
        
        # Get settings
        settings = DataIngestionSettings()
        
        # Create orchestrator
        orchestrator = DataIngestionOrchestrator(settings)
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Created orchestrator', 'progress': 20}
        )
        
        # Run validation asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Start collection (needed for database access)
            loop.run_until_complete(orchestrator.start_collection())
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Running validation', 'progress': 50}
            )
            
            # TODO: Implement data quality validation logic
            # This would query the database and run validation checks
            
            validation_result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'data_points_checked': 0,
                'quality_score': 1.0,
                'issues_found': [],
                'recommendations': []
            }
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Validation completed', 'progress': 90}
            )
            
            # Stop collection
            loop.run_until_complete(orchestrator.stop_collection())
            
            logger.info("Data quality validation task completed successfully", 
                       task_id=task_id, symbol=symbol, result=validation_result)
            
            return {
                'success': True,
                'task_id': task_id,
                'validation_result': validation_result,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        # Record error
        global_error_handler._handle_final_failure(
            error=e,
            context="celery_task",
            task_id=task_id,
            operation="validate_data_quality",
            symbol=symbol
        )
        
        logger.error("Data quality validation task failed", 
                    task_id=task_id, symbol=symbol, error=str(e))
        
        return {
            'success': False,
            'task_id': task_id,
            'symbol': symbol,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }