#!/usr/bin/env python3
"""
Real-time Market Data Processing System

Implements comprehensive real-time market data processing:
- Real-time data ingestion and validation
- Data normalization and cleaning
- Real-time streaming and distribution
- Data quality monitoring
- Performance optimization
- Multi-source data integration

Features:
- Advanced real-time data processing pipeline
- Multi-source data integration and validation
- Real-time streaming and distribution
- Data quality monitoring and alerts
- Performance optimization and caching
- Integration with trading strategies
"""

import asyncio
import json
import time
import threading
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import queue

logger = structlog.get_logger()

class DataType(Enum):
    """Data type enumeration."""
    OHLCV = "ohlcv"
    TICK = "tick"
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    INDICATOR = "indicator"
    SIGNAL = "signal"

class DataQuality(Enum):
    """Data quality enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    data_type: DataType
    timestamp: datetime
    data: Dict[str, Any]
    quality: DataQuality
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataValidationResult:
    """Data validation result structure."""
    is_valid: bool
    quality_score: float
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    enable_validation: bool = True
    enable_normalization: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_queue_size: int = 10000
    batch_size: int = 100
    enable_real_time: bool = True
    enable_monitoring: bool = True

class DataValidator:
    """Market data validator."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize data validator.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        
    def validate_data(self, data: MarketData) -> DataValidationResult:
        """Validate market data."""
        try:
            issues = []
            warnings = []
            quality_score = 1.0
            
            # Basic validation
            if not data.symbol or not data.timestamp:
                issues.append("Missing required fields")
                quality_score -= 0.5
                
            # Data type specific validation
            if data.data_type == DataType.OHLCV:
                result = self._validate_ohlcv(data)
            elif data.data_type == DataType.TICK:
                result = self._validate_tick(data)
            elif data.data_type == DataType.ORDERBOOK:
                result = self._validate_orderbook(data)
            else:
                result = self._validate_generic(data)
                
            issues.extend(result.get('issues', []))
            warnings.extend(result.get('warnings', []))
            quality_score *= result.get('quality_score', 1.0)
            
            # Determine quality level
            if quality_score >= 0.9:
                quality = DataQuality.EXCELLENT
            elif quality_score >= 0.7:
                quality = DataQuality.GOOD
            elif quality_score >= 0.5:
                quality = DataQuality.FAIR
            elif quality_score >= 0.3:
                quality = DataQuality.POOR
            else:
                quality = DataQuality.INVALID
                
            return DataValidationResult(
                is_valid=len(issues) == 0,
                quality_score=quality_score,
                issues=issues,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return DataValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=[f"Validation error: {e}"],
                warnings=[]
            )
            
    def _validate_ohlcv(self, data: MarketData) -> Dict[str, Any]:
        """Validate OHLCV data."""
        issues = []
        warnings = []
        quality_score = 1.0
        
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        data_dict = data.data
        
        # Check required fields
        for field in required_fields:
            if field not in data_dict:
                issues.append(f"Missing required field: {field}")
                quality_score -= 0.2
                
        # Check data consistency
        if all(field in data_dict for field in required_fields):
            open_price = data_dict['open']
            high_price = data_dict['high']
            low_price = data_dict['low']
            close_price = data_dict['close']
            volume = data_dict['volume']
            
            # Check price consistency
            if low_price > high_price:
                issues.append("Low price greater than high price")
                quality_score -= 0.3
                
            if open_price < low_price or open_price > high_price:
                warnings.append("Open price outside high-low range")
                quality_score -= 0.1
                
            if close_price < low_price or close_price > high_price:
                warnings.append("Close price outside high-low range")
                quality_score -= 0.1
                
            # Check for zero or negative values
            if volume <= 0:
                warnings.append("Zero or negative volume")
                quality_score -= 0.1
                
        return {
            'issues': issues,
            'warnings': warnings,
            'quality_score': quality_score
        }
        
    def _validate_tick(self, data: MarketData) -> Dict[str, Any]:
        """Validate tick data."""
        issues = []
        warnings = []
        quality_score = 1.0
        
        required_fields = ['price', 'volume']
        data_dict = data.data
        
        # Check required fields
        for field in required_fields:
            if field not in data_dict:
                issues.append(f"Missing required field: {field}")
                quality_score -= 0.3
                
        # Check data consistency
        if all(field in data_dict for field in required_fields):
            price = data_dict['price']
            volume = data_dict['volume']
            
            if price <= 0:
                issues.append("Invalid price")
                quality_score -= 0.5
                
            if volume <= 0:
                warnings.append("Zero or negative volume")
                quality_score -= 0.1
                
        return {
            'issues': issues,
            'warnings': warnings,
            'quality_score': quality_score
        }
        
    def _validate_orderbook(self, data: MarketData) -> Dict[str, Any]:
        """Validate orderbook data."""
        issues = []
        warnings = []
        quality_score = 1.0
        
        required_fields = ['bids', 'asks']
        data_dict = data.data
        
        # Check required fields
        for field in required_fields:
            if field not in data_dict:
                issues.append(f"Missing required field: {field}")
                quality_score -= 0.3
                
        # Check data consistency
        if all(field in data_dict for field in required_fields):
            bids = data_dict['bids']
            asks = data_dict['asks']
            
            if not isinstance(bids, list) or not isinstance(asks, list):
                issues.append("Invalid orderbook format")
                quality_score -= 0.5
                
            # Check bid-ask spread
            if bids and asks:
                best_bid = max(bid[0] for bid in bids) if bids else 0
                best_ask = min(ask[0] for ask in asks) if asks else float('inf')
                
                if best_bid >= best_ask:
                    warnings.append("Invalid bid-ask spread")
                    quality_score -= 0.2
                    
        return {
            'issues': issues,
            'warnings': warnings,
            'quality_score': quality_score
        }
        
    def _validate_generic(self, data: MarketData) -> Dict[str, Any]:
        """Validate generic data."""
        issues = []
        warnings = []
        quality_score = 1.0
        
        if not data.data:
            issues.append("Empty data")
            quality_score -= 0.5
            
        return {
            'issues': issues,
            'warnings': warnings,
            'quality_score': quality_score
        }

class DataNormalizer:
    """Market data normalizer."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize data normalizer.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        
    def normalize_data(self, data: MarketData) -> MarketData:
        """Normalize market data."""
        try:
            if not self.config.enable_normalization:
                return data
                
            normalized_data = data.data.copy()
            
            # Data type specific normalization
            if data.data_type == DataType.OHLCV:
                normalized_data = self._normalize_ohlcv(normalized_data)
            elif data.data_type == DataType.TICK:
                normalized_data = self._normalize_tick(normalized_data)
            elif data.data_type == DataType.ORDERBOOK:
                normalized_data = self._normalize_orderbook(normalized_data)
            else:
                normalized_data = self._normalize_generic(normalized_data)
                
            # Create normalized data object
            normalized_market_data = MarketData(
                symbol=data.symbol,
                data_type=data.data_type,
                timestamp=data.timestamp,
                data=normalized_data,
                quality=data.quality,
                source=data.source,
                metadata={**data.metadata, 'normalized': True}
            )
            
            return normalized_market_data
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return data
            
    def _normalize_ohlcv(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OHLCV data."""
        normalized = data.copy()
        
        # Ensure numeric types
        for field in ['open', 'high', 'low', 'close', 'volume']:
            if field in normalized:
                try:
                    normalized[field] = float(normalized[field])
                except (ValueError, TypeError):
                    normalized[field] = 0.0
                    
        # Round to reasonable precision
        for field in ['open', 'high', 'low', 'close']:
            if field in normalized:
                normalized[field] = round(normalized[field], 6)
                
        return normalized
        
    def _normalize_tick(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tick data."""
        normalized = data.copy()
        
        # Ensure numeric types
        for field in ['price', 'volume']:
            if field in normalized:
                try:
                    normalized[field] = float(normalized[field])
                except (ValueError, TypeError):
                    normalized[field] = 0.0
                    
        # Round price to reasonable precision
        if 'price' in normalized:
            normalized['price'] = round(normalized['price'], 6)
            
        return normalized
        
    def _normalize_orderbook(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize orderbook data."""
        normalized = data.copy()
        
        # Normalize bids and asks
        for side in ['bids', 'asks']:
            if side in normalized and isinstance(normalized[side], list):
                normalized_side = []
                for order in normalized[side]:
                    if isinstance(order, (list, tuple)) and len(order) >= 2:
                        try:
                            price = float(order[0])
                            volume = float(order[1])
                            normalized_side.append([round(price, 6), volume])
                        except (ValueError, TypeError):
                            continue
                normalized[side] = normalized_side
                
        return normalized
        
    def _normalize_generic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize generic data."""
        # Basic normalization - ensure all values are serializable
        normalized = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                normalized[key] = value
            else:
                normalized[key] = str(value)
                
        return normalized

class MarketDataProcessor:
    """Main market data processor."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize market data processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.validator = DataValidator(config)
        self.normalizer = DataNormalizer(config)
        self.data_queue = queue.Queue(maxsize=config.max_queue_size)
        self.processed_data: List[MarketData] = []
        self.consumers: List[Callable] = []
        self.is_running = False
        self.processing_thread = None
        
    def add_consumer(self, consumer: Callable):
        """Add data consumer."""
        self.consumers.append(consumer)
        
    def process_data(self, data: MarketData) -> MarketData:
        """Process market data."""
        try:
            # Validate data
            if self.config.enable_validation:
                validation_result = self.validator.validate_data(data)
                if not validation_result.is_valid:
                    logger.warning(f"Invalid data for {data.symbol}: {validation_result.issues}")
                    data.quality = DataQuality.INVALID
                else:
                    data.quality = self._get_quality_from_score(validation_result.quality_score)
                    
            # Normalize data
            if self.config.enable_normalization:
                data = self.normalizer.normalize_data(data)
                
            # Store processed data
            self.processed_data.append(data)
            
            # Notify consumers
            self._notify_consumers(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return data
            
    def _get_quality_from_score(self, score: float) -> DataQuality:
        """Get quality level from score."""
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.FAIR
        elif score >= 0.3:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID
            
    def _notify_consumers(self, data: MarketData):
        """Notify all consumers of new data."""
        for consumer in self.consumers:
            try:
                consumer(data)
            except Exception as e:
                logger.error(f"Error in consumer: {e}")
                
    def start_processing(self):
        """Start real-time processing."""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.start()
            logger.info("Market data processor started")
            
    def stop_processing(self):
        """Stop real-time processing."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Market data processor stopped")
        
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Process queued data
                while not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    self.process_data(data)
                    
                # Sleep briefly
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                
    def add_data(self, data: MarketData):
        """Add data to processing queue."""
        try:
            self.data_queue.put_nowait(data)
        except queue.Full:
            logger.warning("Data queue full, dropping data")
            
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {
            'queue_size': self.data_queue.qsize(),
            'processed_count': len(self.processed_data),
            'consumer_count': len(self.consumers),
            'is_running': self.is_running
        }
        
        # Quality distribution
        quality_counts = defaultdict(int)
        for data in self.processed_data[-1000:]:  # Last 1000 items
            quality_counts[data.quality.value] += 1
            
        stats['quality_distribution'] = dict(quality_counts)
        
        return stats

def create_market_data_processor(config: ProcessingConfig = None) -> MarketDataProcessor:
    """Create a market data processor."""
    return MarketDataProcessor(config or ProcessingConfig())

# Demo of market data processing system
if __name__ == "__main__":
    # Create market data processor
    config = ProcessingConfig(
        enable_validation=True,
        enable_normalization=True,
        enable_caching=True,
        max_queue_size=1000,
        batch_size=100,
        enable_real_time=True
    )
    
    processor = create_market_data_processor(config)
    
    # Add consumer
    def data_consumer(data: MarketData):
        print(f"Received {data.data_type.value} data for {data.symbol}: {data.quality.value}")
    
    processor.add_consumer(data_consumer)
    
    # Create sample data
    ohlcv_data = MarketData(
        symbol="AAPL",
        data_type=DataType.OHLCV,
        timestamp=datetime.now(),
        data={
            'open': 150.0,
            'high': 152.0,
            'low': 149.5,
            'close': 151.5,
            'volume': 1000000
        },
        quality=DataQuality.GOOD,
        source="demo"
    )
    
    tick_data = MarketData(
        symbol="AAPL",
        data_type=DataType.TICK,
        timestamp=datetime.now(),
        data={
            'price': 151.25,
            'volume': 100
        },
        quality=DataQuality.GOOD,
        source="demo"
    )
    
    # Process data
    processor.start_processing()
    
    # Add data to queue
    processor.add_data(ohlcv_data)
    processor.add_data(tick_data)
    
    # Wait for processing
    time.sleep(1)
    
    # Get stats
    stats = processor.get_processing_stats()
    print(f"\nProcessing Statistics:")
    print(f"Queue size: {stats['queue_size']}")
    print(f"Processed count: {stats['processed_count']}")
    print(f"Quality distribution: {stats['quality_distribution']}")
    
    # Stop processing
    processor.stop_processing()
    
    print("Market data processing system completed successfully!")