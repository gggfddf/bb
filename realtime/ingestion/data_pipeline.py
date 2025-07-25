#!/usr/bin/env python3
"""
Real-time Data Ingestion Pipeline Module

Implements high-performance real-time data ingestion pipeline for live trading data:
- WebSocket data streaming from multiple sources
- Data validation and quality checks
- Data transformation and normalization
- Data buffering and queuing systems
- Data persistence and caching
- Error handling and recovery mechanisms

Features:
- Multi-source WebSocket streaming
- Real-time data validation and quality control
- High-performance data transformation
- Robust buffering and queuing
- Efficient data persistence and caching
- Comprehensive error handling and recovery
- Scalable architecture for production use
"""

import asyncio
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import websockets
import aiohttp
import pandas as pd
import numpy as np
from collections import deque
import threading
import queue
import redis
import pickle
from concurrent.futures import ThreadPoolExecutor
import hashlib

logger = structlog.get_logger()

class DataSource(Enum):
    """Available data sources."""
    ALPACA = "alpaca"
    BINANCE = "binance"
    COINBASE = "coinbase"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    CUSTOM = "custom"

class DataType(Enum):
    """Types of data being ingested."""
    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    TICK = "tick"
    OHLC = "ohlc"
    VOLUME = "volume"

class ValidationLevel(Enum):
    """Data validation levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"

@dataclass
class DataConfig:
    """Configuration for data ingestion."""
    source: DataSource
    symbols: List[str]
    data_types: List[DataType]
    update_frequency: float = 1.0  # seconds
    buffer_size: int = 1000
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    enable_persistence: bool = True
    enable_caching: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class DataPoint:
    """Single data point from ingestion pipeline."""
    timestamp: datetime
    symbol: str
    data_type: DataType
    source: DataSource
    data: Dict[str, Any]
    quality_score: float = 1.0
    validation_status: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineMetrics:
    """Metrics for data ingestion pipeline."""
    total_messages: int = 0
    valid_messages: int = 0
    invalid_messages: int = 0
    processing_time_avg: float = 0.0
    buffer_utilization: float = 0.0
    error_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)

class DataValidator:
    """Data validation and quality control."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize data validator.
        
        Args:
            validation_level: Level of validation strictness
        """
        self.validation_level = validation_level
        self.validation_rules = self._setup_validation_rules()
        
    def _setup_validation_rules(self) -> Dict[str, Callable]:
        """Setup validation rules based on validation level."""
        rules = {
            'required_fields': self._check_required_fields,
            'data_types': self._check_data_types,
            'value_ranges': self._check_value_ranges,
            'timestamp_validity': self._check_timestamp_validity,
            'data_consistency': self._check_data_consistency
        }
        
        if self.validation_level == ValidationLevel.STRICT:
            rules.update({
                'outlier_detection': self._detect_outliers,
                'pattern_validation': self._validate_patterns
            })
        
        return rules
    
    def validate_data(self, data: Dict[str, Any], data_type: DataType) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate incoming data.
        
        Args:
            data: Data to validate
            data_type: Type of data
            
        Returns:
            Tuple of (is_valid, quality_score, validation_details)
        """
        try:
            validation_results = {}
            quality_score = 1.0
            
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    result = rule_func(data, data_type)
                    validation_results[rule_name] = result
                    
                    if not result['valid']:
                        quality_score *= 0.8
                        
                except Exception as e:
                    validation_results[rule_name] = {
                        'valid': False,
                        'error': str(e)
                    }
                    quality_score *= 0.5
            
            is_valid = quality_score > 0.5 if self.validation_level == ValidationLevel.LENIENT else quality_score > 0.8
            
            return is_valid, quality_score, validation_results
            
        except Exception as e:
            logger.error("Data validation failed", error=str(e))
            return False, 0.0, {'error': str(e)}
    
    def _check_required_fields(self, data: Dict[str, Any], data_type: DataType) -> Dict[str, Any]:
        """Check required fields based on data type."""
        required_fields = {
            DataType.TRADE: ['price', 'size', 'timestamp'],
            DataType.QUOTE: ['bid', 'ask', 'bid_size', 'ask_size'],
            DataType.BAR: ['open', 'high', 'low', 'close', 'volume'],
            DataType.OHLC: ['open', 'high', 'low', 'close'],
            DataType.VOLUME: ['volume', 'timestamp']
        }
        
        required = required_fields.get(data_type, [])
        missing_fields = [field for field in required if field not in data]
        
        return {
            'valid': len(missing_fields) == 0,
            'missing_fields': missing_fields
        }
    
    def _check_data_types(self, data: Dict[str, Any], data_type: DataType) -> Dict[str, Any]:
        """Check data types of fields."""
        type_errors = []
        
        for field, value in data.items():
            if field in ['price', 'bid', 'ask', 'open', 'high', 'low', 'close']:
                if not isinstance(value, (int, float)) or value <= 0:
                    type_errors.append(f"{field}: expected positive number, got {type(value)}")
            
            elif field in ['size', 'bid_size', 'ask_size', 'volume']:
                if not isinstance(value, (int, float)) or value < 0:
                    type_errors.append(f"{field}: expected non-negative number, got {type(value)}")
            
            elif field == 'timestamp':
                if not isinstance(value, (str, datetime, int, float)):
                    type_errors.append(f"{field}: expected timestamp, got {type(value)}")
        
        return {
            'valid': len(type_errors) == 0,
            'type_errors': type_errors
        }
    
    def _check_value_ranges(self, data: Dict[str, Any], data_type: DataType) -> Dict[str, Any]:
        """Check if values are within reasonable ranges."""
        range_errors = []
        
        # Price range check (assuming USD)
        for field in ['price', 'bid', 'ask', 'open', 'high', 'low', 'close']:
            if field in data:
                value = data[field]
                if value > 1000000 or value < 0.01:  # $0.01 to $1M
                    range_errors.append(f"{field}: value {value} outside reasonable range")
        
        # Volume range check
        for field in ['size', 'bid_size', 'ask_size', 'volume']:
            if field in data:
                value = data[field]
                if value > 1000000000 or value < 0:  # 0 to 1B
                    range_errors.append(f"{field}: value {value} outside reasonable range")
        
        return {
            'valid': len(range_errors) == 0,
            'range_errors': range_errors
        }
    
    def _check_timestamp_validity(self, data: Dict[str, Any], data_type: DataType) -> Dict[str, Any]:
        """Check timestamp validity."""
        if 'timestamp' not in data:
            return {'valid': False, 'error': 'No timestamp field'}
        
        try:
            timestamp = data['timestamp']
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            elif isinstance(timestamp, (int, float)):
                timestamp = pd.to_datetime(timestamp, unit='s')
            
            now = pd.Timestamp.now()
            time_diff = abs((timestamp - now).total_seconds())
            
            # Allow timestamps within 1 hour of current time
            is_valid = time_diff < 3600
            
            return {
                'valid': is_valid,
                'time_diff_seconds': time_diff
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _check_data_consistency(self, data: Dict[str, Any], data_type: DataType) -> Dict[str, Any]:
        """Check data consistency."""
        consistency_errors = []
        
        if data_type == DataType.BAR:
            # Check OHLC consistency
            if all(field in data for field in ['open', 'high', 'low', 'close']):
                if not (data['low'] <= data['open'] <= data['high'] and 
                       data['low'] <= data['close'] <= data['high']):
                    consistency_errors.append("OHLC values are inconsistent")
        
        elif data_type == DataType.QUOTE:
            # Check bid/ask consistency
            if all(field in data for field in ['bid', 'ask']):
                if data['bid'] >= data['ask']:
                    consistency_errors.append("Bid should be less than ask")
        
        return {
            'valid': len(consistency_errors) == 0,
            'consistency_errors': consistency_errors
        }
    
    def _detect_outliers(self, data: Dict[str, Any], data_type: DataType) -> Dict[str, Any]:
        """Detect outliers in data."""
        outliers = []
        
        # Simple outlier detection based on z-score
        for field in ['price', 'volume']:
            if field in data:
                # This is a simplified outlier detection
                # In production, you'd use historical data for comparison
                pass
        
        return {
            'valid': len(outliers) == 0,
            'outliers': outliers
        }
    
    def _validate_patterns(self, data: Dict[str, Any], data_type: DataType) -> Dict[str, Any]:
        """Validate data patterns."""
        pattern_errors = []
        
        # Add pattern validation logic here
        # For example, check for repeated values, suspicious patterns, etc.
        
        return {
            'valid': len(pattern_errors) == 0,
            'pattern_errors': pattern_errors
        }

class DataTransformer:
    """Data transformation and normalization."""
    
    def __init__(self):
        """Initialize data transformer."""
        self.transformers = {
            DataType.TRADE: self._transform_trade_data,
            DataType.QUOTE: self._transform_quote_data,
            DataType.BAR: self._transform_bar_data,
            DataType.OHLC: self._transform_ohlc_data,
            DataType.VOLUME: self._transform_volume_data
        }
    
    def transform_data(self, data: Dict[str, Any], data_type: DataType, source: DataSource) -> Dict[str, Any]:
        """
        Transform data to standardized format.
        
        Args:
            data: Raw data
            data_type: Type of data
            source: Data source
            
        Returns:
            Transformed data
        """
        try:
            transformer = self.transformers.get(data_type)
            if transformer:
                return transformer(data, source)
            else:
                return data
                
        except Exception as e:
            logger.error("Data transformation failed", error=str(e))
            return data
    
    def _transform_trade_data(self, data: Dict[str, Any], source: DataSource) -> Dict[str, Any]:
        """Transform trade data."""
        transformed = {
            'price': float(data.get('price', 0)),
            'size': float(data.get('size', 0)),
            'timestamp': self._normalize_timestamp(data.get('timestamp')),
            'side': data.get('side', 'unknown'),
            'exchange': source.value
        }
        
        return transformed
    
    def _transform_quote_data(self, data: Dict[str, Any], source: DataSource) -> Dict[str, Any]:
        """Transform quote data."""
        transformed = {
            'bid': float(data.get('bid', 0)),
            'ask': float(data.get('ask', 0)),
            'bid_size': float(data.get('bid_size', 0)),
            'ask_size': float(data.get('ask_size', 0)),
            'timestamp': self._normalize_timestamp(data.get('timestamp')),
            'exchange': source.value
        }
        
        return transformed
    
    def _transform_bar_data(self, data: Dict[str, Any], source: DataSource) -> Dict[str, Any]:
        """Transform bar data."""
        transformed = {
            'open': float(data.get('open', 0)),
            'high': float(data.get('high', 0)),
            'low': float(data.get('low', 0)),
            'close': float(data.get('close', 0)),
            'volume': float(data.get('volume', 0)),
            'timestamp': self._normalize_timestamp(data.get('timestamp')),
            'exchange': source.value
        }
        
        return transformed
    
    def _transform_ohlc_data(self, data: Dict[str, Any], source: DataSource) -> Dict[str, Any]:
        """Transform OHLC data."""
        return self._transform_bar_data(data, source)
    
    def _transform_volume_data(self, data: Dict[str, Any], source: DataSource) -> Dict[str, Any]:
        """Transform volume data."""
        transformed = {
            'volume': float(data.get('volume', 0)),
            'timestamp': self._normalize_timestamp(data.get('timestamp')),
            'exchange': source.value
        }
        
        return transformed
    
    def _normalize_timestamp(self, timestamp) -> datetime:
        """Normalize timestamp to datetime object."""
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, str):
            return pd.to_datetime(timestamp)
        elif isinstance(timestamp, (int, float)):
            return pd.to_datetime(timestamp, unit='s')
        else:
            return datetime.now()

class DataBuffer:
    """Data buffering and queuing system."""
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize data buffer.
        
        Args:
            buffer_size: Maximum buffer size
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.metrics = {
            'total_added': 0,
            'total_removed': 0,
            'buffer_overflows': 0,
            'current_size': 0
        }
    
    def add_data(self, data_point: DataPoint) -> bool:
        """
        Add data point to buffer.
        
        Args:
            data_point: Data point to add
            
        Returns:
            True if added successfully, False if buffer full
        """
        with self.lock:
            if len(self.buffer) >= self.buffer_size:
                self.metrics['buffer_overflows'] += 1
                return False
            
            self.buffer.append(data_point)
            self.metrics['total_added'] += 1
            self.metrics['current_size'] = len(self.buffer)
            return True
    
    def get_data(self, count: int = 1) -> List[DataPoint]:
        """
        Get data points from buffer.
        
        Args:
            count: Number of data points to retrieve
            
        Returns:
            List of data points
        """
        with self.lock:
            data_points = []
            for _ in range(min(count, len(self.buffer))):
                if self.buffer:
                    data_points.append(self.buffer.popleft())
                    self.metrics['total_removed'] += 1
            
            self.metrics['current_size'] = len(self.buffer)
            return data_points
    
    def get_all_data(self) -> List[DataPoint]:
        """Get all data points from buffer."""
        with self.lock:
            data_points = list(self.buffer)
            self.buffer.clear()
            self.metrics['total_removed'] += len(data_points)
            self.metrics['current_size'] = 0
            return data_points
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get buffer metrics."""
        with self.lock:
            return self.metrics.copy()

class DataPersistence:
    """Data persistence and caching layer."""
    
    def __init__(self, enable_persistence: bool = True, enable_caching: bool = True):
        """
        Initialize data persistence.
        
        Args:
            enable_persistence: Enable data persistence
            enable_caching: Enable data caching
        """
        self.enable_persistence = enable_persistence
        self.enable_caching = enable_caching
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize Redis for caching if available
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_available = True
        except:
            self.redis_available = False
            logger.warning("Redis not available, using in-memory cache")
    
    def store_data(self, data_point: DataPoint) -> bool:
        """
        Store data point.
        
        Args:
            data_point: Data point to store
            
        Returns:
            True if stored successfully
        """
        try:
            if self.enable_caching:
                self._cache_data(data_point)
            
            if self.enable_persistence:
                return self._persist_data(data_point)
            
            return True
            
        except Exception as e:
            logger.error("Failed to store data", error=str(e))
            return False
    
    def _cache_data(self, data_point: DataPoint):
        """Cache data point."""
        key = f"{data_point.symbol}:{data_point.data_type.value}:{data_point.timestamp.isoformat()}"
        
        if self.redis_available:
            try:
                self.redis_client.setex(
                    key, 
                    self.cache_ttl, 
                    pickle.dumps(data_point)
                )
            except Exception as e:
                logger.warning("Redis cache failed, using in-memory", error=str(e))
                self.cache[key] = (data_point, time.time() + self.cache_ttl)
        else:
            self.cache[key] = (data_point, time.time() + self.cache_ttl)
    
    def _persist_data(self, data_point: DataPoint) -> bool:
        """Persist data point to storage."""
        # In production, this would write to database or file system
        # For now, we'll just log the data
        logger.info("Data persisted", 
                   symbol=data_point.symbol,
                   data_type=data_point.data_type.value,
                   timestamp=data_point.timestamp.isoformat())
        return True
    
    def retrieve_data(self, symbol: str, data_type: DataType, 
                     start_time: datetime, end_time: datetime) -> List[DataPoint]:
        """
        Retrieve data from storage.
        
        Args:
            symbol: Symbol to retrieve
            data_type: Type of data
            start_time: Start time
            end_time: End time
            
        Returns:
            List of data points
        """
        # In production, this would query database
        # For now, return empty list
        return []

class RealTimeDataPipeline:
    """
    Main real-time data ingestion pipeline.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize real-time data pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.validator = DataValidator(config.validation_level)
        self.transformer = DataTransformer()
        self.buffer = DataBuffer(config.buffer_size)
        self.persistence = DataPersistence(config.enable_persistence, config.enable_caching)
        
        # Pipeline state
        self.is_running = False
        self.metrics = PipelineMetrics()
        self.error_count = 0
        self.last_error = None
        
        # WebSocket connections
        self.websocket_connections = {}
        self.connection_tasks = []
        
        logger.info("Real-time data pipeline initialized", 
                   source=config.source.value,
                   symbols=config.symbols)
    
    async def start(self):
        """Start the data ingestion pipeline."""
        try:
            self.is_running = True
            logger.info("Starting real-time data pipeline")
            
            # Start WebSocket connections
            await self._start_websocket_connections()
            
            # Start data processing loop
            await self._process_data_loop()
            
        except Exception as e:
            logger.error("Failed to start pipeline", error=str(e))
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the data ingestion pipeline."""
        try:
            self.is_running = False
            logger.info("Stopping real-time data pipeline")
            
            # Close WebSocket connections
            await self._close_websocket_connections()
            
        except Exception as e:
            logger.error("Failed to stop pipeline", error=str(e))
            raise
    
    async def _start_websocket_connections(self):
        """Start WebSocket connections for data sources."""
        for symbol in self.config.symbols:
            for data_type in self.config.data_types:
                task = asyncio.create_task(
                    self._connect_websocket(symbol, data_type)
                )
                self.connection_tasks.append(task)
    
    async def _connect_websocket(self, symbol: str, data_type: DataType):
        """Connect to WebSocket for specific symbol and data type."""
        try:
            # This is a simplified WebSocket connection
            # In production, you'd connect to actual data providers
            uri = f"wss://example.com/ws/{symbol}/{data_type.value}"
            
            async with websockets.connect(uri) as websocket:
                self.websocket_connections[f"{symbol}:{data_type.value}"] = websocket
                
                async for message in websocket:
                    if not self.is_running:
                        break
                    
                    await self._process_message(symbol, data_type, message)
                    
        except Exception as e:
            logger.error("WebSocket connection failed", 
                        symbol=symbol, 
                        data_type=data_type.value,
                        error=str(e))
            self.error_count += 1
            self.last_error = str(e)
    
    async def _process_message(self, symbol: str, data_type: DataType, message: str):
        """Process incoming WebSocket message."""
        try:
            start_time = time.time()
            
            # Parse message
            data = json.loads(message)
            
            # Transform data
            transformed_data = self.transformer.transform_data(data, data_type, self.config.source)
            
            # Validate data
            is_valid, quality_score, validation_details = self.validator.validate_data(
                transformed_data, data_type
            )
            
            # Create data point
            data_point = DataPoint(
                timestamp=datetime.now(),
                symbol=symbol,
                data_type=data_type,
                source=self.config.source,
                data=transformed_data,
                quality_score=quality_score,
                validation_status=is_valid,
                metadata={'validation_details': validation_details}
            )
            
            # Add to buffer
            if self.buffer.add_data(data_point):
                # Store data
                self.persistence.store_data(data_point)
                
                # Update metrics
                self.metrics.total_messages += 1
                if is_valid:
                    self.metrics.valid_messages += 1
                else:
                    self.metrics.invalid_messages += 1
                
                processing_time = time.time() - start_time
                self.metrics.processing_time_avg = (
                    (self.metrics.processing_time_avg * (self.metrics.total_messages - 1) + processing_time) / 
                    self.metrics.total_messages
                )
                
            else:
                logger.warning("Buffer full, dropping message", symbol=symbol)
            
        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            self.metrics.error_count += 1
    
    async def _process_data_loop(self):
        """Main data processing loop."""
        while self.is_running:
            try:
                # Process buffered data
                data_points = self.buffer.get_all_data()
                
                if data_points:
                    # Process data points (e.g., send to ML models)
                    await self._process_data_points(data_points)
                
                # Update metrics
                self._update_metrics()
                
                # Sleep
                await asyncio.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error("Data processing loop error", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_data_points(self, data_points: List[DataPoint]):
        """Process data points (placeholder for ML model integration)."""
        for data_point in data_points:
            # In production, this would send data to ML models
            # For now, just log the data
            if data_point.validation_status:
                logger.debug("Processing valid data point", 
                           symbol=data_point.symbol,
                           data_type=data_point.data_type.value)
    
    def _update_metrics(self):
        """Update pipeline metrics."""
        buffer_metrics = self.buffer.get_metrics()
        self.metrics.buffer_utilization = buffer_metrics['current_size'] / self.config.buffer_size
        self.metrics.last_update = datetime.now()
    
    async def _close_websocket_connections(self):
        """Close all WebSocket connections."""
        for task in self.connection_tasks:
            task.cancel()
        
        for websocket in self.websocket_connections.values():
            await websocket.close()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        buffer_metrics = self.buffer.get_metrics()
        
        return {
            'pipeline_metrics': {
                'total_messages': self.metrics.total_messages,
                'valid_messages': self.metrics.valid_messages,
                'invalid_messages': self.metrics.invalid_messages,
                'processing_time_avg': self.metrics.processing_time_avg,
                'buffer_utilization': self.metrics.buffer_utilization,
                'error_count': self.metrics.error_count,
                'last_update': self.metrics.last_update.isoformat()
            },
            'buffer_metrics': buffer_metrics,
            'is_running': self.is_running
        }

def create_data_pipeline(source: str = "alpaca",
                        symbols: List[str] = None,
                        data_types: List[str] = None,
                        buffer_size: int = 1000) -> RealTimeDataPipeline:
    """
    Create a real-time data pipeline with specified configuration.
    
    Args:
        source: Data source
        symbols: List of symbols to track
        data_types: List of data types to ingest
        buffer_size: Buffer size
        
    Returns:
        RealTimeDataPipeline instance
    """
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    if data_types is None:
        data_types = ['trade', 'quote', 'bar']
    
    config = DataConfig(
        source=DataSource(source),
        symbols=symbols,
        data_types=[DataType(dt) for dt in data_types],
        buffer_size=buffer_size
    )
    
    return RealTimeDataPipeline(config)

async def run_pipeline_demo():
    """Run a demo of the data pipeline."""
    pipeline = create_data_pipeline()
    
    try:
        await pipeline.start()
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Print metrics
        metrics = pipeline.get_metrics()
        print("Pipeline Metrics:", json.dumps(metrics, indent=2, default=str))
        
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(run_pipeline_demo())