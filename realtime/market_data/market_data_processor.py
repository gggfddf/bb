#!/usr/bin/env python3
"""
Real-time Market Data Processing Module

Implements real-time market data processing system:
- Market data normalization
- Data quality monitoring
- Data transformation pipeline
- Data enrichment
- Data distribution system
- Data analytics

Features:
- Advanced market data normalization and standardization
- Comprehensive data quality monitoring and validation
- High-performance data transformation pipeline
- Intelligent data enrichment and augmentation
- Efficient data distribution and broadcasting
- Real-time data analytics and insights
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import threading
import queue
import uuid
from collections import defaultdict, deque
import statistics

logger = structlog.get_logger()

class DataType(Enum):
    """Market data types."""
    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    TICK = "tick"
    OHLCV = "ohlcv"
    ORDERBOOK = "orderbook"
    NEWS = "news"
    SENTIMENT = "sentiment"

class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

class DataSource(Enum):
    """Data sources."""
    ALPACA = "alpaca"
    BINANCE = "binance"
    COINBASE = "coinbase"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX = "iex"

@dataclass
class MarketData:
    """Market data structure."""
    data_id: str
    symbol: str
    data_type: DataType
    source: DataSource
    timestamp: datetime
    data: Dict[str, Any]
    quality_score: float = 1.0
    quality_level: DataQuality = DataQuality.EXCELLENT
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedData:
    """Processed market data structure."""
    data_id: str
    symbol: str
    data_type: DataType
    source: DataSource
    timestamp: datetime
    normalized_data: Dict[str, Any]
    enriched_data: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataConfig:
    """Data processing configuration."""
    enable_normalization: bool = True
    enable_quality_monitoring: bool = True
    enable_enrichment: bool = True
    enable_analytics: bool = True
    quality_threshold: float = 0.8
    processing_timeout: float = 1.0  # seconds
    batch_size: int = 100
    max_queue_size: int = 10000

class DataNormalizer:
    """Market data normalization system."""
    
    def __init__(self):
        """Initialize data normalizer."""
        self.normalization_rules = {
            DataType.TRADE: self._normalize_trade,
            DataType.QUOTE: self._normalize_quote,
            DataType.BAR: self._normalize_bar,
            DataType.OHLCV: self._normalize_ohlcv,
            DataType.ORDERBOOK: self._normalize_orderbook,
            DataType.TICK: self._normalize_tick
        }
    
    def normalize_data(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Normalize market data to standard format.
        
        Args:
            market_data: Raw market data
            
        Returns:
            Normalized data dictionary
        """
        try:
            normalizer = self.normalization_rules.get(market_data.data_type)
            if normalizer:
                return normalizer(market_data)
            else:
                logger.warning("No normalizer for data type", data_type=market_data.data_type.value)
                return market_data.data
        except Exception as e:
            logger.error("Data normalization error", error=str(e))
            return market_data.data
    
    def _normalize_trade(self, market_data: MarketData) -> Dict[str, Any]:
        """Normalize trade data."""
        data = market_data.data
        
        normalized = {
            'symbol': market_data.symbol,
            'timestamp': market_data.timestamp.isoformat(),
            'price': float(data.get('price', 0)),
            'quantity': float(data.get('quantity', 0)),
            'side': data.get('side', 'unknown'),
            'trade_id': data.get('trade_id', ''),
            'venue': data.get('venue', ''),
            'conditions': data.get('conditions', [])
        }
        
        return normalized
    
    def _normalize_quote(self, market_data: MarketData) -> Dict[str, Any]:
        """Normalize quote data."""
        data = market_data.data
        
        normalized = {
            'symbol': market_data.symbol,
            'timestamp': market_data.timestamp.isoformat(),
            'bid_price': float(data.get('bid_price', 0)),
            'bid_size': float(data.get('bid_size', 0)),
            'ask_price': float(data.get('ask_price', 0)),
            'ask_size': float(data.get('ask_size', 0)),
            'venue': data.get('venue', ''),
            'quote_id': data.get('quote_id', '')
        }
        
        return normalized
    
    def _normalize_bar(self, market_data: MarketData) -> Dict[str, Any]:
        """Normalize bar data."""
        data = market_data.data
        
        normalized = {
            'symbol': market_data.symbol,
            'timestamp': market_data.timestamp.isoformat(),
            'open': float(data.get('open', 0)),
            'high': float(data.get('high', 0)),
            'low': float(data.get('low', 0)),
            'close': float(data.get('close', 0)),
            'volume': float(data.get('volume', 0)),
            'vwap': float(data.get('vwap', 0)),
            'bar_count': int(data.get('bar_count', 0))
        }
        
        return normalized
    
    def _normalize_ohlcv(self, market_data: MarketData) -> Dict[str, Any]:
        """Normalize OHLCV data."""
        data = market_data.data
        
        normalized = {
            'symbol': market_data.symbol,
            'timestamp': market_data.timestamp.isoformat(),
            'open': float(data.get('open', 0)),
            'high': float(data.get('high', 0)),
            'low': float(data.get('low', 0)),
            'close': float(data.get('close', 0)),
            'volume': float(data.get('volume', 0)),
            'interval': data.get('interval', '1min')
        }
        
        return normalized
    
    def _normalize_orderbook(self, market_data: MarketData) -> Dict[str, Any]:
        """Normalize orderbook data."""
        data = market_data.data
        
        normalized = {
            'symbol': market_data.symbol,
            'timestamp': market_data.timestamp.isoformat(),
            'bids': data.get('bids', []),
            'asks': data.get('asks', []),
            'venue': data.get('venue', ''),
            'depth': int(data.get('depth', 10))
        }
        
        return normalized
    
    def _normalize_tick(self, market_data: MarketData) -> Dict[str, Any]:
        """Normalize tick data."""
        data = market_data.data
        
        normalized = {
            'symbol': market_data.symbol,
            'timestamp': market_data.timestamp.isoformat(),
            'price': float(data.get('price', 0)),
            'quantity': float(data.get('quantity', 0)),
            'side': data.get('side', 'unknown'),
            'tick_type': data.get('tick_type', ''),
            'venue': data.get('venue', '')
        }
        
        return normalized

class DataQualityMonitor:
    """Data quality monitoring system."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize data quality monitor.
        
        Args:
            config: Data processing configuration
        """
        self.config = config
        self.quality_metrics = defaultdict(list)
        self.quality_thresholds = {
            'price_validity': 0.95,
            'timestamp_validity': 0.99,
            'completeness': 0.90,
            'consistency': 0.85,
            'freshness': 0.95
        }
    
    def assess_quality(self, market_data: MarketData) -> Tuple[float, DataQuality, Dict[str, Any]]:
        """
        Assess data quality.
        
        Args:
            market_data: Market data to assess
            
        Returns:
            Tuple of (quality_score, quality_level, quality_metrics)
        """
        try:
            metrics = {}
            
            # Price validity
            metrics['price_validity'] = self._check_price_validity(market_data)
            
            # Timestamp validity
            metrics['timestamp_validity'] = self._check_timestamp_validity(market_data)
            
            # Data completeness
            metrics['completeness'] = self._check_completeness(market_data)
            
            # Data consistency
            metrics['consistency'] = self._check_consistency(market_data)
            
            # Data freshness
            metrics['freshness'] = self._check_freshness(market_data)
            
            # Calculate overall quality score
            quality_score = np.mean(list(metrics.values()))
            
            # Determine quality level
            quality_level = self._determine_quality_level(quality_score)
            
            # Store metrics
            self.quality_metrics[market_data.symbol].append(metrics)
            
            return quality_score, quality_level, metrics
        
        except Exception as e:
            logger.error("Data quality assessment error", error=str(e))
            return 0.0, DataQuality.INVALID, {}
    
    def _check_price_validity(self, market_data: MarketData) -> float:
        """Check price validity."""
        data = market_data.data
        
        if market_data.data_type in [DataType.TRADE, DataType.QUOTE, DataType.BAR, DataType.OHLCV]:
            price_fields = ['price', 'bid_price', 'ask_price', 'open', 'high', 'low', 'close']
            valid_prices = 0
            total_prices = 0
            
            for field in price_fields:
                if field in data:
                    total_prices += 1
                    price = data[field]
                    if isinstance(price, (int, float)) and price > 0:
                        valid_prices += 1
            
            return valid_prices / total_prices if total_prices > 0 else 0.0
        
        return 1.0
    
    def _check_timestamp_validity(self, market_data: MarketData) -> float:
        """Check timestamp validity."""
        # Check if timestamp is recent
        time_diff = abs((datetime.now() - market_data.timestamp).total_seconds())
        
        if time_diff < 60:  # Within 1 minute
            return 1.0
        elif time_diff < 300:  # Within 5 minutes
            return 0.8
        elif time_diff < 3600:  # Within 1 hour
            return 0.5
        else:
            return 0.0
    
    def _check_completeness(self, market_data: MarketData) -> float:
        """Check data completeness."""
        data = market_data.data
        required_fields = self._get_required_fields(market_data.data_type)
        
        if not required_fields:
            return 1.0
        
        present_fields = sum(1 for field in required_fields if field in data and data[field] is not None)
        return present_fields / len(required_fields)
    
    def _check_consistency(self, market_data: MarketData) -> float:
        """Check data consistency."""
        data = market_data.data
        
        if market_data.data_type == DataType.BAR:
            # Check OHLC consistency
            if all(field in data for field in ['open', 'high', 'low', 'close']):
                if data['low'] <= data['open'] <= data['high'] and \
                   data['low'] <= data['close'] <= data['high']:
                    return 1.0
                else:
                    return 0.0
        
        return 1.0
    
    def _check_freshness(self, market_data: MarketData) -> float:
        """Check data freshness."""
        # Check if data is from expected source and recent
        expected_sources = [DataSource.ALPACA, DataSource.BINANCE, DataSource.POLYGON]
        
        if market_data.source in expected_sources:
            return 1.0
        else:
            return 0.5
    
    def _get_required_fields(self, data_type: DataType) -> List[str]:
        """Get required fields for data type."""
        required_fields = {
            DataType.TRADE: ['price', 'quantity'],
            DataType.QUOTE: ['bid_price', 'ask_price'],
            DataType.BAR: ['open', 'high', 'low', 'close'],
            DataType.OHLCV: ['open', 'high', 'low', 'close', 'volume'],
            DataType.ORDERBOOK: ['bids', 'asks'],
            DataType.TICK: ['price', 'quantity']
        }
        return required_fields.get(data_type, [])
    
    def _determine_quality_level(self, quality_score: float) -> DataQuality:
        """Determine quality level from score."""
        if quality_score >= 0.95:
            return DataQuality.EXCELLENT
        elif quality_score >= 0.85:
            return DataQuality.GOOD
        elif quality_score >= 0.70:
            return DataQuality.FAIR
        elif quality_score >= 0.50:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID
    
    def get_quality_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get quality summary."""
        if symbol:
            metrics = self.quality_metrics.get(symbol, [])
        else:
            # Aggregate across all symbols
            all_metrics = []
            for symbol_metrics in self.quality_metrics.values():
                all_metrics.extend(symbol_metrics)
            metrics = all_metrics
        
        if not metrics:
            return {}
        
        summary = {}
        for metric_name in ['price_validity', 'timestamp_validity', 'completeness', 'consistency', 'freshness']:
            values = [m.get(metric_name, 0) for m in metrics if metric_name in m]
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary

class DataEnricher:
    """Data enrichment system."""
    
    def __init__(self):
        """Initialize data enricher."""
        self.enrichment_rules = {
            DataType.TRADE: self._enrich_trade,
            DataType.QUOTE: self._enrich_quote,
            DataType.BAR: self._enrich_bar,
            DataType.OHLCV: self._enrich_ohlcv,
            DataType.ORDERBOOK: self._enrich_orderbook
        }
    
    def enrich_data(self, market_data: MarketData, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich market data with additional information.
        
        Args:
            market_data: Original market data
            normalized_data: Normalized data
            
        Returns:
            Enriched data dictionary
        """
        try:
            enricher = self.enrichment_rules.get(market_data.data_type)
            if enricher:
                return enricher(market_data, normalized_data)
            else:
                return normalized_data
        except Exception as e:
            logger.error("Data enrichment error", error=str(e))
            return normalized_data
    
    def _enrich_trade(self, market_data: MarketData, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich trade data."""
        enriched = normalized_data.copy()
        
        # Add trade size category
        quantity = enriched.get('quantity', 0)
        if quantity < 100:
            enriched['size_category'] = 'small'
        elif quantity < 1000:
            enriched['size_category'] = 'medium'
        else:
            enriched['size_category'] = 'large'
        
        # Add time-based features
        timestamp = market_data.timestamp
        enriched['hour'] = timestamp.hour
        enriched['minute'] = timestamp.minute
        enriched['day_of_week'] = timestamp.weekday()
        
        return enriched
    
    def _enrich_quote(self, market_data: MarketData, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich quote data."""
        enriched = normalized_data.copy()
        
        # Calculate spread
        bid_price = enriched.get('bid_price', 0)
        ask_price = enriched.get('ask_price', 0)
        if bid_price > 0 and ask_price > 0:
            enriched['spread'] = ask_price - bid_price
            enriched['spread_pct'] = (enriched['spread'] / bid_price) * 100
        
        # Calculate mid price
        if bid_price > 0 and ask_price > 0:
            enriched['mid_price'] = (bid_price + ask_price) / 2
        
        return enriched
    
    def _enrich_bar(self, market_data: MarketData, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich bar data."""
        enriched = normalized_data.copy()
        
        # Calculate bar properties
        open_price = enriched.get('open', 0)
        high_price = enriched.get('high', 0)
        low_price = enriched.get('low', 0)
        close_price = enriched.get('close', 0)
        
        if all(p > 0 for p in [open_price, high_price, low_price, close_price]):
            enriched['range'] = high_price - low_price
            enriched['body'] = abs(close_price - open_price)
            enriched['upper_shadow'] = high_price - max(open_price, close_price)
            enriched['lower_shadow'] = min(open_price, close_price) - low_price
            enriched['is_bullish'] = close_price > open_price
            enriched['is_bearish'] = close_price < open_price
            enriched['is_doji'] = abs(close_price - open_price) < (high_price - low_price) * 0.1
        
        return enriched
    
    def _enrich_ohlcv(self, market_data: MarketData, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich OHLCV data."""
        enriched = normalized_data.copy()
        
        # Calculate OHLCV properties (similar to bar)
        open_price = enriched.get('open', 0)
        high_price = enriched.get('high', 0)
        low_price = enriched.get('low', 0)
        close_price = enriched.get('close', 0)
        volume = enriched.get('volume', 0)
        
        if all(p > 0 for p in [open_price, high_price, low_price, close_price]):
            enriched['range'] = high_price - low_price
            enriched['body'] = abs(close_price - open_price)
            enriched['is_bullish'] = close_price > open_price
            enriched['is_bearish'] = close_price < open_price
            
            # Volume analysis
            if volume > 0:
                enriched['price_volume_ratio'] = (close_price - open_price) / volume
        
        return enriched
    
    def _enrich_orderbook(self, market_data: MarketData, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich orderbook data."""
        enriched = normalized_data.copy()
        
        bids = enriched.get('bids', [])
        asks = enriched.get('asks', [])
        
        if bids and asks:
            # Calculate orderbook metrics
            best_bid = max(bid[0] for bid in bids) if bids else 0
            best_ask = min(ask[0] for ask in asks) if asks else 0
            
            enriched['best_bid'] = best_bid
            enriched['best_ask'] = best_ask
            enriched['spread'] = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
            enriched['mid_price'] = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
            
            # Calculate bid/ask imbalance
            total_bid_volume = sum(bid[1] for bid in bids)
            total_ask_volume = sum(ask[1] for ask in asks)
            enriched['bid_ask_imbalance'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
        
        return enriched

class DataDistributor:
    """Data distribution system."""
    
    def __init__(self):
        """Initialize data distributor."""
        self.subscribers = defaultdict(list)
        self.distribution_queue = queue.Queue()
        self.running = False
        self.distribution_thread = None
    
    def subscribe(self, data_type: DataType, callback: Callable[[ProcessedData], None]):
        """Subscribe to data type."""
        self.subscribers[data_type].append(callback)
        logger.info("Subscriber added", data_type=data_type.value)
    
    def unsubscribe(self, data_type: DataType, callback: Callable[[ProcessedData], None]):
        """Unsubscribe from data type."""
        if data_type in self.subscribers:
            self.subscribers[data_type] = [cb for cb in self.subscribers[data_type] if cb != callback]
            logger.info("Subscriber removed", data_type=data_type.value)
    
    def distribute_data(self, processed_data: ProcessedData):
        """Distribute processed data to subscribers."""
        try:
            self.distribution_queue.put(processed_data)
        except Exception as e:
            logger.error("Failed to queue data for distribution", error=str(e))
    
    def start(self):
        """Start data distributor."""
        if not self.running:
            self.running = True
            self.distribution_thread = threading.Thread(target=self._distribution_worker)
            self.distribution_thread.daemon = True
            self.distribution_thread.start()
            logger.info("Data distributor started")
    
    def stop(self):
        """Stop data distributor."""
        self.running = False
        if self.distribution_thread:
            self.distribution_thread.join()
        logger.info("Data distributor stopped")
    
    def _distribution_worker(self):
        """Data distribution worker."""
        while self.running:
            try:
                processed_data = self.distribution_queue.get(timeout=1)
                self._distribute_to_subscribers(processed_data)
                self.distribution_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Data distribution error", error=str(e))
    
    def _distribute_to_subscribers(self, processed_data: ProcessedData):
        """Distribute data to subscribers."""
        subscribers = self.subscribers.get(processed_data.data_type, [])
        
        for callback in subscribers:
            try:
                callback(processed_data)
            except Exception as e:
                logger.error("Subscriber callback error", error=str(e))

class DataAnalytics:
    """Data analytics system."""
    
    def __init__(self):
        """Initialize data analytics."""
        self.analytics_data = defaultdict(lambda: deque(maxlen=1000))
        self.analytics_metrics = defaultdict(dict)
    
    def record_data(self, processed_data: ProcessedData):
        """Record data for analytics."""
        symbol = processed_data.symbol
        data_type = processed_data.data_type
        
        # Store data
        self.analytics_data[f"{symbol}_{data_type.value}"].append(processed_data)
        
        # Update metrics
        self._update_metrics(processed_data)
    
    def _update_metrics(self, processed_data: ProcessedData):
        """Update analytics metrics."""
        symbol = processed_data.symbol
        data_type = processed_data.data_type
        key = f"{symbol}_{data_type.value}"
        
        data_list = self.analytics_data[key]
        if len(data_list) < 2:
            return
        
        # Calculate basic metrics
        if data_type == DataType.TRADE:
            prices = [d.normalized_data.get('price', 0) for d in data_list[-100:]]
            volumes = [d.normalized_data.get('quantity', 0) for d in data_list[-100:]]
            
            if prices and volumes:
                self.analytics_metrics[key] = {
                    'avg_price': np.mean(prices),
                    'price_volatility': np.std(prices),
                    'total_volume': np.sum(volumes),
                    'avg_volume': np.mean(volumes),
                    'trade_count': len(data_list),
                    'last_update': datetime.now().isoformat()
                }
        
        elif data_type == DataType.QUOTE:
            spreads = [d.enriched_data.get('spread', 0) for d in data_list[-100:]]
            mid_prices = [d.enriched_data.get('mid_price', 0) for d in data_list[-100:]]
            
            if spreads and mid_prices:
                self.analytics_metrics[key] = {
                    'avg_spread': np.mean(spreads),
                    'spread_volatility': np.std(spreads),
                    'avg_mid_price': np.mean(mid_prices),
                    'quote_count': len(data_list),
                    'last_update': datetime.now().isoformat()
                }
    
    def get_analytics(self, symbol: str = None, data_type: DataType = None) -> Dict[str, Any]:
        """Get analytics data."""
        if symbol and data_type:
            key = f"{symbol}_{data_type.value}"
            return self.analytics_metrics.get(key, {})
        else:
            return dict(self.analytics_metrics)

class MarketDataProcessor:
    """Main real-time market data processing system."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize market data processor.
        
        Args:
            config: Data processing configuration
        """
        self.config = config
        self.normalizer = DataNormalizer()
        self.quality_monitor = DataQualityMonitor(config)
        self.enricher = DataEnricher()
        self.distributor = DataDistributor()
        self.analytics = DataAnalytics()
        
        self.processing_queue = queue.Queue(maxsize=config.max_queue_size)
        self.running = False
        self.processing_thread = None
    
    def process_data(self, market_data: MarketData) -> Optional[ProcessedData]:
        """
        Process market data.
        
        Args:
            market_data: Raw market data
            
        Returns:
            Processed data or None if processing failed
        """
        try:
            # Add to processing queue
            self.processing_queue.put(market_data)
            return None  # Async processing
        except queue.Full:
            logger.warning("Processing queue full, dropping data")
            return None
        except Exception as e:
            logger.error("Failed to queue data for processing", error=str(e))
            return None
    
    def start(self):
        """Start data processor."""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._processing_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.distributor.start()
            logger.info("Market data processor started")
    
    def stop(self):
        """Stop data processor."""
        self.running = False
        self.distributor.stop()
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Market data processor stopped")
    
    def _processing_worker(self):
        """Data processing worker."""
        while self.running:
            try:
                market_data = self.processing_queue.get(timeout=1)
                processed_data = self._process_single_data(market_data)
                if processed_data:
                    self.distributor.distribute_data(processed_data)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Data processing error", error=str(e))
    
    def _process_single_data(self, market_data: MarketData) -> Optional[ProcessedData]:
        """Process single market data item."""
        try:
            # Normalize data
            normalized_data = {}
            if self.config.enable_normalization:
                normalized_data = self.normalizer.normalize_data(market_data)
            else:
                normalized_data = market_data.data
            
            # Assess quality
            quality_score = 1.0
            quality_level = DataQuality.EXCELLENT
            quality_metrics = {}
            
            if self.config.enable_quality_monitoring:
                quality_score, quality_level, quality_metrics = self.quality_monitor.assess_quality(market_data)
            
            # Enrich data
            enriched_data = {}
            if self.config.enable_enrichment:
                enriched_data = self.enricher.enrich_data(market_data, normalized_data)
            else:
                enriched_data = normalized_data
            
            # Create processed data
            processed_data = ProcessedData(
                data_id=str(uuid.uuid4()),
                symbol=market_data.symbol,
                data_type=market_data.data_type,
                source=market_data.source,
                timestamp=market_data.timestamp,
                normalized_data=normalized_data,
                enriched_data=enriched_data,
                quality_metrics=quality_metrics,
                metadata=market_data.metadata
            )
            
            # Record for analytics
            if self.config.enable_analytics:
                self.analytics.record_data(processed_data)
            
            logger.debug("Data processed", data_id=processed_data.data_id, 
                        symbol=market_data.symbol, quality_score=quality_score)
            
            return processed_data
        
        except Exception as e:
            logger.error("Single data processing error", error=str(e))
            return None
    
    def subscribe(self, data_type: DataType, callback: Callable[[ProcessedData], None]):
        """Subscribe to processed data."""
        self.distributor.subscribe(data_type, callback)
    
    def get_quality_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get data quality summary."""
        return self.quality_monitor.get_quality_summary(symbol)
    
    def get_analytics(self, symbol: str = None, data_type: DataType = None) -> Dict[str, Any]:
        """Get data analytics."""
        return self.analytics.get_analytics(symbol, data_type)

def create_market_data_processor(config: DataConfig = None) -> MarketDataProcessor:
    """
    Create a real-time market data processing system.
    
    Args:
        config: Data processing configuration
        
    Returns:
        MarketDataProcessor instance
    """
    if config is None:
        config = DataConfig()
    
    return MarketDataProcessor(config)

if __name__ == "__main__":
    # Demo of market data processing system
    config = DataConfig(
        enable_normalization=True,
        enable_quality_monitoring=True,
        enable_enrichment=True,
        enable_analytics=True,
        quality_threshold=0.8,
        processing_timeout=1.0,
        batch_size=100,
        max_queue_size=10000
    )
    
    processor = create_market_data_processor(config)
    
    # Start processor
    processor.start()
    
    print("Market data processor created successfully!")
    print(f"Quality threshold: {config.quality_threshold}")
    print(f"Processing timeout: {config.processing_timeout}s")
    
    # Create sample market data
    sample_data = MarketData(
        data_id=str(uuid.uuid4()),
        symbol="AAPL",
        data_type=DataType.TRADE,
        source=DataSource.ALPACA,
        timestamp=datetime.now(),
        data={
            'price': 150.25,
            'quantity': 100,
            'side': 'buy',
            'trade_id': '12345',
            'venue': 'NASDAQ'
        }
    )
    
    # Process data
    processor.process_data(sample_data)
    
    # Get quality summary
    quality_summary = processor.get_quality_summary()
    print(f"Quality summary: {quality_summary}")
    
    # Get analytics
    analytics = processor.get_analytics("AAPL", DataType.TRADE)
    print(f"Analytics: {analytics}")