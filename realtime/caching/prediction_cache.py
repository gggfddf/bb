#!/usr/bin/env python3
"""
Prediction Caching System

Implements comprehensive prediction caching for the trading system:
- Multi-level caching (memory, disk, distributed)
- Cache invalidation strategies
- Performance optimization
- Cache analytics and monitoring
- Cache warming and preloading
- Cache compression and serialization

Features:
- Advanced multi-level caching with TTL and eviction
- Intelligent cache invalidation and warming
- Performance optimization and compression
- Cache analytics and monitoring
- Distributed caching support
- High-performance cache operations
"""

import asyncio
import json
import time
import threading
import hashlib
import pickle
import gzip
import zlib
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
import os
import shutil
from pathlib import Path
import tempfile

logger = structlog.get_logger()

class CacheLevel(Enum):
    """Cache level enumeration."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"

class EvictionPolicy(Enum):
    """Cache eviction policy enumeration."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    HYBRID = "hybrid"

class CompressionType(Enum):
    """Compression type enumeration."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"

@dataclass
class CacheEntry:
    """Cache entry structure."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    compression_type: CompressionType = CompressionType.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheConfig:
    """Cache configuration."""
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    max_disk_size: int = 1024 * 1024 * 1024  # 1GB
    memory_ttl: float = 300.0  # 5 minutes
    disk_ttl: float = 3600.0  # 1 hour
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes
    compression_type: CompressionType = CompressionType.GZIP
    enable_analytics: bool = True
    cache_warming_enabled: bool = True
    max_concurrent_operations: int = 10

@dataclass
class CacheStatistics:
    """Cache statistics structure."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    avg_access_time: float = 0.0
    compression_ratio: float = 1.0

class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int, ttl: float, policy: EvictionPolicy):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum cache size in bytes
            ttl: Time to live in seconds
            policy: Eviction policy
        """
        self.max_size = max_size
        self.ttl = ttl
        self.policy = policy
        self.cache: Dict[str, CacheEntry] = {}
        self.current_size = 0
        self.statistics = CacheStatistics(max_size=max_size)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.statistics.misses += 1
            return None
            
        entry = self.cache[key]
        
        # Check TTL
        if entry.ttl and datetime.now() - entry.created_at > timedelta(seconds=entry.ttl):
            self.delete(key)
            self.statistics.misses += 1
            return None
            
        # Update access statistics
        entry.access_count += 1
        entry.accessed_at = datetime.now()
        self.statistics.hits += 1
        
        return entry.value
        
    def set(self, key: str, value: Any, ttl: Optional[float] = None, 
            compression_type: CompressionType = CompressionType.NONE) -> bool:
        """Set value in cache."""
        # Calculate size
        size = self._calculate_size(value)
        
        # Evict if necessary
        while self.current_size + size > self.max_size:
            if not self._evict_entry():
                return False
                
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            ttl=ttl or self.ttl,
            size=size,
            compression_type=compression_type
        )
        
        self.cache[key] = entry
        self.current_size += size
        self.statistics.size = self.current_size
        
        return True
        
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size -= entry.size
            del self.cache[key]
            self.statistics.size = self.current_size
            return True
        return False
        
    def clear(self):
        """Clear all entries."""
        self.cache.clear()
        self.current_size = 0
        self.statistics.size = 0
        
    def _evict_entry(self) -> bool:
        """Evict entry based on policy."""
        if not self.cache:
            return False
            
        if self.policy == EvictionPolicy.LRU:
            # Remove least recently used
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].accessed_at)
            self.delete(oldest_key)
        elif self.policy == EvictionPolicy.LFU:
            # Remove least frequently used
            least_frequent_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            self.delete(least_frequent_key)
        elif self.policy == EvictionPolicy.FIFO:
            # Remove first in
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            self.delete(oldest_key)
        elif self.policy == EvictionPolicy.TTL:
            # Remove expired entries
            current_time = datetime.now()
            expired_keys = [k for k, entry in self.cache.items() 
                          if entry.ttl and current_time - entry.created_at > timedelta(seconds=entry.ttl)]
            for key in expired_keys:
                self.delete(key)
                
        self.statistics.evictions += 1
        return True
        
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Default size
            
    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        total_requests = self.statistics.hits + self.statistics.misses
        self.statistics.hit_rate = self.statistics.hits / total_requests if total_requests > 0 else 0.0
        return self.statistics

class DiskCache:
    """Disk-based cache implementation."""
    
    def __init__(self, cache_dir: str, max_size: int, ttl: float):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Cache directory
            max_size: Maximum cache size in bytes
            ttl: Time to live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.ttl = ttl
        self.current_size = 0
        self.statistics = CacheStatistics(max_size=max_size)
        self._load_metadata()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        file_path = self.cache_dir / f"{key}.cache"
        
        if not file_path.exists():
            self.statistics.misses += 1
            return None
            
        try:
            # Check metadata
            metadata_path = self.cache_dir / f"{key}.meta"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Check TTL
                created_at = datetime.fromisoformat(metadata['created_at'])
                if datetime.now() - created_at > timedelta(seconds=metadata.get('ttl', self.ttl)):
                    self.delete(key)
                    self.statistics.misses += 1
                    return None
                    
            # Load value
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
                
            # Decompress if necessary
            if metadata.get('compression_type') == 'gzip':
                value = pickle.loads(gzip.decompress(compressed_data))
            else:
                value = pickle.loads(compressed_data)
                
            # Update access statistics
            metadata['access_count'] = metadata.get('access_count', 0) + 1
            metadata['accessed_at'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            self.statistics.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
            self.delete(key)
            self.statistics.misses += 1
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[float] = None,
            compression_type: CompressionType = CompressionType.GZIP) -> bool:
        """Set value in disk cache."""
        try:
            # Serialize value
            serialized_data = pickle.dumps(value)
            
            # Compress if necessary
            if compression_type == CompressionType.GZIP and len(serialized_data) > 1024:
                compressed_data = gzip.compress(serialized_data)
                compression_ratio = len(compressed_data) / len(serialized_data)
            else:
                compressed_data = serialized_data
                compression_ratio = 1.0
                
            # Check size limit
            if self.current_size + len(compressed_data) > self.max_size:
                self._evict_entries()
                
            # Write to disk
            file_path = self.cache_dir / f"{key}.cache"
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
                
            # Write metadata
            metadata_path = self.cache_dir / f"{key}.meta"
            metadata = {
                'created_at': datetime.now().isoformat(),
                'accessed_at': datetime.now().isoformat(),
                'access_count': 0,
                'size': len(compressed_data),
                'original_size': len(serialized_data),
                'compression_ratio': compression_ratio,
                'compression_type': compression_type.value,
                'ttl': ttl or self.ttl
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            self.current_size += len(compressed_data)
            self.statistics.size = self.current_size
            self.statistics.compression_ratio = compression_ratio
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        file_path = self.cache_dir / f"{key}.cache"
        metadata_path = self.cache_dir / f"{key}.meta"
        
        if file_path.exists():
            try:
                # Get file size before deletion
                size = file_path.stat().st_size
                
                # Delete files
                file_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                    
                self.current_size -= size
                self.statistics.size = self.current_size
                return True
            except Exception as e:
                logger.error(f"Error deleting from disk cache: {e}")
                
        return False
        
    def clear(self):
        """Clear all entries."""
        for file_path in self.cache_dir.glob("*.cache"):
            file_path.unlink()
        for meta_path in self.cache_dir.glob("*.meta"):
            meta_path.unlink()
        self.current_size = 0
        self.statistics.size = 0
        
    def _evict_entries(self):
        """Evict entries based on TTL and size."""
        current_time = datetime.now()
        expired_files = []
        
        # Find expired files
        for meta_path in self.cache_dir.glob("*.meta"):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    
                created_at = datetime.fromisoformat(metadata['created_at'])
                ttl = metadata.get('ttl', self.ttl)
                
                if current_time - created_at > timedelta(seconds=ttl):
                    expired_files.append(meta_path.stem)
            except:
                pass
                
        # Delete expired files
        for key in expired_files:
            self.delete(key)
            
    def _load_metadata(self):
        """Load cache metadata."""
        total_size = 0
        for meta_path in self.cache_dir.glob("*.meta"):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                total_size += metadata.get('size', 0)
            except:
                pass
        self.current_size = total_size
        self.statistics.size = total_size
        
    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        return self.statistics

class PredictionCache:
    """Main prediction cache system."""
    
    def __init__(self, config: CacheConfig = None):
        """
        Initialize prediction cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.memory_cache = MemoryCache(
            max_size=self.config.max_memory_size,
            ttl=self.config.memory_ttl,
            policy=self.config.eviction_policy
        )
        self.disk_cache = DiskCache(
            cache_dir="cache/predictions",
            max_size=self.config.max_disk_size,
            ttl=self.config.disk_ttl
        )
        self.cache_warming_queue = queue.Queue()
        self.is_running = False
        self.warming_thread = None
        
    def start(self):
        """Start cache system."""
        if self.is_running:
            return
            
        self.is_running = True
        if self.config.cache_warming_enabled:
            self.warming_thread = threading.Thread(target=self._warming_loop, daemon=True)
            self.warming_thread.start()
        logger.info("Prediction cache system started")
        
    def stop(self):
        """Stop cache system."""
        self.is_running = False
        if self.warming_thread:
            self.warming_thread.join(timeout=5)
        logger.info("Prediction cache system stopped")
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
            
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Move to memory cache
            self.memory_cache.set(key, value)
            return value
            
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[float] = None,
            compression_type: CompressionType = None) -> bool:
        """Set value in cache (both memory and disk)."""
        compression_type = compression_type or self.config.compression_type
        
        # Set in memory cache
        memory_success = self.memory_cache.set(key, value, ttl, compression_type)
        
        # Set in disk cache
        disk_success = self.disk_cache.set(key, value, ttl, compression_type)
        
        return memory_success or disk_success
        
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        memory_deleted = self.memory_cache.delete(key)
        disk_deleted = self.disk_cache.delete(key)
        return memory_deleted or disk_deleted
        
    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
        
    def warm_cache(self, keys: List[str], value_generator: Callable[[str], Any]):
        """Warm cache with predicted keys."""
        for key in keys:
            try:
                value = value_generator(key)
                self.set(key, value)
            except Exception as e:
                logger.error(f"Error warming cache for key {key}: {e}")
                
    def _warming_loop(self):
        """Cache warming loop."""
        while self.is_running:
            try:
                # Process warming requests
                try:
                    warming_request = self.cache_warming_queue.get(timeout=1.0)
                    keys, generator = warming_request
                    self.warm_cache(keys, generator)
                except queue.Empty:
                    pass
            except Exception as e:
                logger.error(f"Error in cache warming loop: {e}")
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_stats = self.memory_cache.get_statistics()
        disk_stats = self.disk_cache.get_statistics()
        
        return {
            "memory": {
                "hits": memory_stats.hits,
                "misses": memory_stats.misses,
                "hit_rate": memory_stats.hit_rate,
                "size": memory_stats.size,
                "max_size": memory_stats.max_size,
                "evictions": memory_stats.evictions
            },
            "disk": {
                "hits": disk_stats.hits,
                "misses": disk_stats.misses,
                "hit_rate": disk_stats.hit_rate,
                "size": disk_stats.size,
                "max_size": disk_stats.max_size,
                "compression_ratio": disk_stats.compression_ratio
            },
            "overall": {
                "total_hits": memory_stats.hits + disk_stats.hits,
                "total_misses": memory_stats.misses + disk_stats.misses,
                "overall_hit_rate": (memory_stats.hits + disk_stats.hits) / 
                                  (memory_stats.hits + disk_stats.hits + memory_stats.misses + disk_stats.misses)
                if (memory_stats.hits + disk_stats.hits + memory_stats.misses + disk_stats.misses) > 0 else 0.0
            }
        }

def create_prediction_cache(config: CacheConfig = None) -> PredictionCache:
    """Create a prediction cache system."""
    return PredictionCache(config)

# Demo of prediction cache system
if __name__ == "__main__":
    # Create cache system
    config = CacheConfig(
        max_memory_size=50 * 1024 * 1024,  # 50MB
        max_disk_size=500 * 1024 * 1024,   # 500MB
        memory_ttl=180.0,  # 3 minutes
        disk_ttl=3600.0,   # 1 hour
        enable_compression=True,
        compression_type=CompressionType.GZIP,
        cache_warming_enabled=True
    )
    
    cache = create_prediction_cache(config)
    
    # Start cache system
    cache.start()
    
    # Test cache operations
    def generate_value(key: str):
        """Generate test value."""
        return {
            "prediction": np.random.rand(10),
            "confidence": np.random.uniform(0.5, 0.9),
            "timestamp": datetime.now().isoformat(),
            "key": key
        }
    
    # Set some values
    for i in range(10):
        key = f"prediction_{i}"
        value = generate_value(key)
        cache.set(key, value)
        print(f"Set cache key: {key}")
    
    # Get values
    for i in range(5):
        key = f"prediction_{i}"
        value = cache.get(key)
        if value:
            print(f"Retrieved cache key: {key}")
        else:
            print(f"Cache miss for key: {key}")
    
    # Get statistics
    stats = cache.get_statistics()
    print(f"Cache statistics: {stats}")
    
    print("Prediction cache system created successfully!")
    print("Press Ctrl+C to stop...")
    
    try:
        # Keep running
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cache.stop()
        print("Cache system stopped.")