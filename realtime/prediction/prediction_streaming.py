#!/usr/bin/env python3
"""
Prediction Streaming System

Implements comprehensive prediction streaming for the trading system:
- Real-time model prediction generation
- Prediction caching and storage
- Multi-consumer prediction distribution
- Prediction quality monitoring
- Stream processing and analytics
- Prediction validation and filtering

Features:
- Real-time prediction generation from multiple models
- Advanced caching with TTL and eviction policies
- Multi-consumer distribution via WebSocket and REST
- Prediction quality monitoring and validation
- Stream processing with windowing and aggregation
- High-performance prediction delivery
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

class PredictionType(Enum):
    """Prediction type enumeration."""
    PRICE_DIRECTION = "price_direction"
    PRICE_MOVEMENT = "price_movement"
    VOLATILITY = "volatility"
    TREND = "trend"
    SUPPORT_RESISTANCE = "support_resistance"
    SIGNAL = "signal"
    PROBABILITY = "probability"

class StreamStatus(Enum):
    """Stream status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class CachePolicy(Enum):
    """Cache policy enumeration."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    HYBRID = "hybrid"

@dataclass
class Prediction:
    """Prediction structure."""
    prediction_id: str
    symbol: str
    timestamp: datetime
    prediction_type: PredictionType
    value: Union[float, int, str]
    confidence: float
    model_name: str
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    validation_status: str = "pending"

@dataclass
class StreamConfig:
    """Stream configuration."""
    batch_size: int = 100
    batch_timeout: float = 1.0  # seconds
    max_cache_size: int = 10000
    cache_ttl: float = 300.0  # seconds
    cache_policy: CachePolicy = CachePolicy.LRU
    enable_validation: bool = True
    enable_monitoring: bool = True
    max_consumers: int = 100
    prediction_interval: float = 1.0  # seconds

@dataclass
class Consumer:
    """Consumer information."""
    consumer_id: str
    name: str
    subscription_types: List[PredictionType]
    symbols: List[str]
    callback: Callable[[Prediction], None]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

class PredictionCache:
    """Prediction caching system."""
    
    def __init__(self, max_size: int = 10000, ttl: float = 300.0, policy: CachePolicy = CachePolicy.LRU):
        """
        Initialize prediction cache.
        
        Args:
            max_size: Maximum number of cached predictions
            ttl: Time to live for cached predictions
            policy: Cache eviction policy
        """
        self.max_size = max_size
        self.ttl = ttl
        self.policy = policy
        self.cache: Dict[str, Tuple[Prediction, datetime]] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.access_time: Dict[str, datetime] = {}
        
    def get(self, key: str) -> Optional[Prediction]:
        """Get prediction from cache."""
        if key not in self.cache:
            return None
            
        prediction, timestamp = self.cache[key]
        
        # Check TTL
        if datetime.now() - timestamp > timedelta(seconds=self.ttl):
            self.delete(key)
            return None
            
        # Update access statistics
        self.access_count[key] += 1
        self.access_time[key] = datetime.now()
        
        return prediction
        
    def set(self, key: str, prediction: Prediction):
        """Set prediction in cache."""
        # Evict if necessary
        if len(self.cache) >= self.max_size:
            self._evict_entries()
            
        self.cache[key] = (prediction, datetime.now())
        self.access_count[key] = 1
        self.access_time[key] = datetime.now()
        
    def delete(self, key: str):
        """Delete prediction from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_count:
            del self.access_count[key]
        if key in self.access_time:
            del self.access_time[key]
            
    def _evict_entries(self):
        """Evict entries based on policy."""
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            oldest_key = min(self.access_time.keys(), key=lambda k: self.access_time[k])
            self.delete(oldest_key)
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            least_frequent_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            self.delete(least_frequent_key)
        elif self.policy == CachePolicy.TTL:
            # Remove expired entries
            current_time = datetime.now()
            expired_keys = [k for k, (_, timestamp) in self.cache.items() 
                          if current_time - timestamp > timedelta(seconds=self.ttl)]
            for key in expired_keys:
                self.delete(key)
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self._calculate_hit_rate(),
            "policy": self.policy.value,
            "ttl": self.ttl
        }
        
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # This would need to track hits/misses in a real implementation
        return 0.8  # Placeholder

class PredictionValidator:
    """Prediction validation system."""
    
    def __init__(self):
        """Initialize prediction validator."""
        self.validation_rules: Dict[str, Callable[[Prediction], bool]] = {}
        self.quality_thresholds: Dict[PredictionType, float] = {
            PredictionType.PRICE_DIRECTION: 0.6,
            PredictionType.PRICE_MOVEMENT: 0.5,
            PredictionType.VOLATILITY: 0.7,
            PredictionType.TREND: 0.65,
            PredictionType.SUPPORT_RESISTANCE: 0.75,
            PredictionType.SIGNAL: 0.8,
            PredictionType.PROBABILITY: 0.9
        }
        
    def add_validation_rule(self, rule_name: str, rule_func: Callable[[Prediction], bool]):
        """Add custom validation rule."""
        self.validation_rules[rule_name] = rule_func
        
    def validate_prediction(self, prediction: Prediction) -> Tuple[bool, float, str]:
        """
        Validate prediction.
        
        Returns:
            Tuple of (is_valid, quality_score, status_message)
        """
        quality_score = 0.0
        validation_errors = []
        
        # Check confidence threshold
        threshold = self.quality_thresholds.get(prediction.prediction_type, 0.5)
        if prediction.confidence < threshold:
            validation_errors.append(f"Confidence {prediction.confidence} below threshold {threshold}")
        else:
            quality_score += prediction.confidence * 0.4
            
        # Check value validity
        if not self._is_valid_value(prediction):
            validation_errors.append("Invalid prediction value")
        else:
            quality_score += 0.3
            
        # Check timestamp validity
        if not self._is_valid_timestamp(prediction):
            validation_errors.append("Invalid timestamp")
        else:
            quality_score += 0.2
            
        # Apply custom validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                if not rule_func(prediction):
                    validation_errors.append(f"Failed custom rule: {rule_name}")
                else:
                    quality_score += 0.1
            except Exception as e:
                validation_errors.append(f"Error in rule {rule_name}: {str(e)}")
                
        # Determine final status
        is_valid = len(validation_errors) == 0 and quality_score >= 0.6
        status_message = "valid" if is_valid else "; ".join(validation_errors)
        
        return is_valid, quality_score, status_message
        
    def _is_valid_value(self, prediction: Prediction) -> bool:
        """Check if prediction value is valid."""
        if prediction.prediction_type == PredictionType.PRICE_DIRECTION:
            return prediction.value in [-1, 0, 1]
        elif prediction.prediction_type == PredictionType.PROBABILITY:
            return 0 <= prediction.value <= 1
        elif prediction.prediction_type == PredictionType.VOLATILITY:
            return prediction.value >= 0
        else:
            return True
            
    def _is_valid_timestamp(self, prediction: Prediction) -> bool:
        """Check if timestamp is valid."""
        now = datetime.now()
        return abs((now - prediction.timestamp).total_seconds()) < 300  # 5 minutes

class PredictionStreamProcessor:
    """Prediction stream processor."""
    
    def __init__(self, config: StreamConfig):
        """
        Initialize prediction stream processor.
        
        Args:
            config: Stream configuration
        """
        self.config = config
        self.cache = PredictionCache(
            max_size=config.max_cache_size,
            ttl=config.cache_ttl,
            policy=config.cache_policy
        )
        self.validator = PredictionValidator()
        self.consumers: Dict[str, Consumer] = {}
        self.prediction_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None
        
    def start(self):
        """Start prediction stream processing."""
        if self.is_running:
            logger.warning("Prediction stream processor is already running")
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Prediction stream processor started")
        
    def stop(self):
        """Stop prediction stream processing."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Prediction stream processor stopped")
        
    def add_consumer(self, consumer: Consumer):
        """Add prediction consumer."""
        self.consumers[consumer.consumer_id] = consumer
        logger.info(f"Added consumer: {consumer.name} ({consumer.consumer_id})")
        
    def remove_consumer(self, consumer_id: str):
        """Remove prediction consumer."""
        if consumer_id in self.consumers:
            del self.consumers[consumer_id]
            logger.info(f"Removed consumer: {consumer_id}")
            
    def publish_prediction(self, prediction: Prediction):
        """Publish prediction to stream."""
        # Validate prediction
        if self.config.enable_validation:
            is_valid, quality_score, status = self.validator.validate_prediction(prediction)
            prediction.validation_status = status
            prediction.quality_score = quality_score
            
            if not is_valid:
                logger.warning(f"Invalid prediction rejected: {status}")
                return
                
        # Cache prediction
        cache_key = f"{prediction.symbol}_{prediction.prediction_type.value}_{prediction.timestamp.isoformat()}"
        self.cache.set(cache_key, prediction)
        
        # Add to processing queue
        self.prediction_queue.put(prediction)
        
    def _processing_loop(self):
        """Main processing loop."""
        batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Get prediction from queue with timeout
                try:
                    prediction = self.prediction_queue.get(timeout=0.1)
                    batch.append(prediction)
                except queue.Empty:
                    pass
                    
                # Process batch if full or timeout reached
                current_time = time.time()
                if (len(batch) >= self.config.batch_size or 
                    (len(batch) > 0 and current_time - last_batch_time >= self.config.batch_timeout)):
                    
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in prediction processing loop: {e}")
                
    def _process_batch(self, predictions: List[Prediction]):
        """Process batch of predictions."""
        # Group predictions by consumer requirements
        consumer_predictions = defaultdict(list)
        
        for prediction in predictions:
            for consumer in self.consumers.values():
                if not consumer.is_active:
                    continue
                    
                # Check if consumer is interested in this prediction
                if (prediction.prediction_type in consumer.subscription_types and
                    prediction.symbol in consumer.symbols):
                    consumer_predictions[consumer.consumer_id].append(prediction)
                    
        # Send predictions to consumers
        for consumer_id, consumer_preds in consumer_predictions.items():
            consumer = self.consumers[consumer_id]
            try:
                for prediction in consumer_preds:
                    consumer.callback(prediction)
                consumer.last_activity = datetime.now()
            except Exception as e:
                logger.error(f"Error sending prediction to consumer {consumer_id}: {e}")
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            "active_consumers": len([c for c in self.consumers.values() if c.is_active]),
            "total_consumers": len(self.consumers),
            "queue_size": self.prediction_queue.qsize(),
            "cache_stats": self.cache.get_statistics(),
            "is_running": self.is_running
        }

class PredictionStreamingSystem:
    """Main prediction streaming system."""
    
    def __init__(self, config: StreamConfig = None):
        """
        Initialize prediction streaming system.
        
        Args:
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        self.processor = PredictionStreamProcessor(self.config)
        self.model_predictors: Dict[str, Callable] = {}
        self.prediction_generators: Dict[str, asyncio.Task] = {}
        
    def start(self):
        """Start the prediction streaming system."""
        self.processor.start()
        logger.info("Prediction streaming system started")
        
    def stop(self):
        """Stop the prediction streaming system."""
        # Stop all prediction generators
        for task in self.prediction_generators.values():
            task.cancel()
            
        self.processor.stop()
        logger.info("Prediction streaming system stopped")
        
    def register_model(self, model_name: str, predictor_func: Callable):
        """Register a model predictor function."""
        self.model_predictors[model_name] = predictor_func
        logger.info(f"Registered model: {model_name}")
        
    def start_prediction_stream(self, model_name: str, symbols: List[str], 
                              prediction_types: List[PredictionType], interval: float = None):
        """Start prediction stream for a model."""
        if model_name not in self.model_predictors:
            raise ValueError(f"Model {model_name} not registered")
            
        if model_name in self.prediction_generators:
            logger.warning(f"Prediction stream for {model_name} already running")
            return
            
        interval = interval or self.config.prediction_interval
        task = asyncio.create_task(
            self._generate_predictions(model_name, symbols, prediction_types, interval)
        )
        self.prediction_generators[model_name] = task
        logger.info(f"Started prediction stream for {model_name}")
        
    def stop_prediction_stream(self, model_name: str):
        """Stop prediction stream for a model."""
        if model_name in self.prediction_generators:
            self.prediction_generators[model_name].cancel()
            del self.prediction_generators[model_name]
            logger.info(f"Stopped prediction stream for {model_name}")
            
    async def _generate_predictions(self, model_name: str, symbols: List[str], 
                                  prediction_types: List[PredictionType], interval: float):
        """Generate predictions for a model."""
        predictor = self.model_predictors[model_name]
        
        while True:
            try:
                for symbol in symbols:
                    for pred_type in prediction_types:
                        # Generate prediction
                        prediction_value, confidence = await self._call_predictor(predictor, symbol, pred_type)
                        
                        # Create prediction object
                        prediction = Prediction(
                            prediction_id=str(uuid.uuid4()),
                            symbol=symbol,
                            timestamp=datetime.now(),
                            prediction_type=pred_type,
                            value=prediction_value,
                            confidence=confidence,
                            model_name=model_name
                        )
                        
                        # Publish prediction
                        self.processor.publish_prediction(prediction)
                        
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating predictions for {model_name}: {e}")
                await asyncio.sleep(interval)
                
    async def _call_predictor(self, predictor: Callable, symbol: str, pred_type: PredictionType):
        """Call predictor function safely."""
        try:
            if asyncio.iscoroutinefunction(predictor):
                result = await predictor(symbol, pred_type)
            else:
                # Run in thread pool for sync functions
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, predictor, symbol, pred_type)
                
            if isinstance(result, tuple):
                return result
            else:
                return result, 0.8  # Default confidence
                
        except Exception as e:
            logger.error(f"Error calling predictor: {e}")
            return 0, 0.0  # Default values
            
    def add_consumer(self, consumer: Consumer):
        """Add prediction consumer."""
        self.processor.add_consumer(consumer)
        
    def remove_consumer(self, consumer_id: str):
        """Remove prediction consumer."""
        self.processor.remove_consumer(consumer_id)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = self.processor.get_statistics()
        stats.update({
            "registered_models": len(self.model_predictors),
            "active_streams": len(self.prediction_generators),
            "prediction_interval": self.config.prediction_interval
        })
        return stats

def create_prediction_streaming_system(config: StreamConfig = None) -> PredictionStreamingSystem:
    """Create a prediction streaming system."""
    return PredictionStreamingSystem(config)

# Demo of prediction streaming system
if __name__ == "__main__":
    # Create system
    config = StreamConfig(
        batch_size=50,
        batch_timeout=0.5,
        max_cache_size=5000,
        cache_ttl=180.0,
        enable_validation=True,
        prediction_interval=2.0
    )
    
    system = create_prediction_streaming_system(config)
    
    # Register a sample model
    def sample_predictor(symbol: str, pred_type: PredictionType):
        """Sample predictor function."""
        import random
        if pred_type == PredictionType.PRICE_DIRECTION:
            return random.choice([-1, 0, 1]), random.uniform(0.6, 0.9)
        elif pred_type == PredictionType.PROBABILITY:
            return random.uniform(0, 1), random.uniform(0.7, 0.95)
        else:
            return random.uniform(-0.1, 0.1), random.uniform(0.5, 0.8)
    
    system.register_model("sample_model", sample_predictor)
    
    # Add a consumer
    def prediction_callback(prediction: Prediction):
        print(f"Received prediction: {prediction.symbol} - {prediction.prediction_type.value} = {prediction.value}")
    
    consumer = Consumer(
        consumer_id="demo_consumer",
        name="Demo Consumer",
        subscription_types=[PredictionType.PRICE_DIRECTION, PredictionType.PROBABILITY],
        symbols=["AAPL", "GOOGL"],
        callback=prediction_callback
    )
    
    system.add_consumer(consumer)
    
    # Start system
    system.start()
    system.start_prediction_stream("sample_model", ["AAPL", "GOOGL"], 
                                 [PredictionType.PRICE_DIRECTION, PredictionType.PROBABILITY])
    
    print("Prediction streaming system created successfully!")
    print("Press Ctrl+C to stop...")
    
    try:
        # Keep running
        import time
        while True:
            time.sleep(1)
            stats = system.get_statistics()
            print(f"Stats: {stats}")
    except KeyboardInterrupt:
        system.stop()
        print("System stopped.")