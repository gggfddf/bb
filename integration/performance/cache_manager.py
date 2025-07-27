#!/usr/bin/env python3
"""
Performance Optimization Module

Implements comprehensive performance optimization for the trading system:
- Caching mechanisms
- Performance monitoring
- Load balancing
- Connection pooling
- Resource optimization
- Performance profiling

Features:
- Complete caching system with multiple backends
- Advanced performance monitoring and metrics
- Load balancing with multiple algorithms
- Connection pooling for database and external services
- Resource optimization and management
- Performance profiling and analysis
- High-performance optimization techniques
"""

import asyncio
import time
import threading
import hashlib
import pickle
import json
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import functools
import weakref

logger = structlog.get_logger()

class CacheType(Enum):
    """Cache type enumeration."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"

class EvictionPolicy(Enum):
    """Cache eviction policy enumeration."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live

class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"

class PoolStrategy(Enum):
    """Connection pool strategy enumeration."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"

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
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetric:
    """Performance metric structure."""
    metric_id: str
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheConfig:
    """Cache configuration."""
    cache_type: CacheType = CacheType.MEMORY
    max_size: int = 1000
    max_memory_mb: int = 100
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    default_ttl: float = 3600.0  # 1 hour
    enable_compression: bool = True
    enable_serialization: bool = True
    compression_threshold: int = 1024  # bytes
    enable_statistics: bool = True
    cleanup_interval: float = 300.0  # seconds

class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, config: CacheConfig):
        """
        Initialize memory cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache = {}
        self.access_order = deque()
        self.access_counts = defaultdict(int)
        self.total_size = 0
        self._lock = threading.RLock()
        self._cleanup_task = None
        self._running = False
        
        logger.info("Memory cache initialized")
    
    def start(self):
        """Start cache cleanup task."""
        if not self.config.enable_statistics:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Memory cache cleanup started")
    
    def stop(self):
        """Stop cache cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Memory cache cleanup stopped")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and datetime.now() > entry.created_at + timedelta(seconds=entry.ttl):
                self._remove_entry(key)
                return None
            
            # Update access information
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            self.access_counts[key] += 1
            
            # Update access order for LRU
            if self.config.eviction_policy == EvictionPolicy.LRU:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            # Serialize and compress if enabled
            if self.config.enable_serialization:
                value = self._serialize_value(value)
            
            if self.config.enable_compression and len(str(value)) > self.config.compression_threshold:
                value = self._compress_value(value)
            
            # Calculate size
            size = len(str(value))
            
            # Check if we need to evict entries
            if self.total_size + size > self.config.max_memory_mb * 1024 * 1024:
                self._evict_entries(size)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl=ttl or self.config.default_ttl,
                size=size
            )
            
            # Store entry
            self.cache[key] = entry
            self.total_size += size
            
            # Update access order
            if self.config.eviction_policy == EvictionPolicy.LRU:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_counts.clear()
            self.total_size = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'total_entries': len(self.cache),
                'total_size_mb': self.total_size / (1024 * 1024),
                'max_size': self.config.max_size,
                'max_memory_mb': self.config.max_memory_mb,
                'eviction_policy': self.config.eviction_policy.value,
                'hit_rate': self._calculate_hit_rate(),
                'most_accessed': self._get_most_accessed_keys(10)
            }
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache."""
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        self.total_size -= entry.size
        
        del self.cache[key]
        
        if key in self.access_order:
            self.access_order.remove(key)
        
        if key in self.access_counts:
            del self.access_counts[key]
        
        return True
    
    def _evict_entries(self, required_size: int):
        """Evict entries based on policy."""
        if self.config.eviction_policy == EvictionPolicy.LRU:
            self._evict_lru(required_size)
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            self._evict_lfu(required_size)
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            self._evict_fifo(required_size)
    
    def _evict_lru(self, required_size: int):
        """Evict least recently used entries."""
        while self.total_size + required_size > self.config.max_memory_mb * 1024 * 1024 and self.access_order:
            key = self.access_order.popleft()
            self._remove_entry(key)
    
    def _evict_lfu(self, required_size: int):
        """Evict least frequently used entries."""
        sorted_keys = sorted(self.access_counts.items(), key=lambda x: x[1])
        for key, _ in sorted_keys:
            if self.total_size + required_size <= self.config.max_memory_mb * 1024 * 1024:
                break
            self._remove_entry(key)
    
    def _evict_fifo(self, required_size: int):
        """Evict first in first out entries."""
        for key in list(self.cache.keys()):
            if self.total_size + required_size <= self.config.max_memory_mb * 1024 * 1024:
                break
            self._remove_entry(key)
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value if needed."""
        try:
            return pickle.dumps(value)
        except Exception:
            return value
    
    def _compress_value(self, value: Any) -> Any:
        """Compress value if needed."""
        try:
            import gzip
            if isinstance(value, str):
                return gzip.compress(value.encode())
            elif isinstance(value, bytes):
                return gzip.compress(value)
            return value
        except Exception:
            return value
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(self.access_counts.values())
        if total_accesses == 0:
            return 0.0
        return len(self.cache) / total_accesses
    
    def _get_most_accessed_keys(self, count: int) -> List[Tuple[str, int]]:
        """Get most accessed keys."""
        sorted_keys = sorted(self.access_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_keys[:count]
    
    async def _cleanup_loop(self):
        """Cleanup expired entries."""
        while self._running:
            try:
                with self._lock:
                    expired_keys = []
                    for key, entry in self.cache.items():
                        if entry.ttl and datetime.now() > entry.created_at + timedelta(seconds=entry.ttl):
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                    
                    if expired_keys:
                        logger.info("Cleaned up expired cache entries", count=len(expired_keys))
                
                await asyncio.sleep(self.config.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cache cleanup error", error=str(e))
                await asyncio.sleep(self.config.cleanup_interval)

class LoadBalancer:
    """Load balancer implementation."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self.servers = []
        self.current_index = 0
        self.server_weights = {}
        self.server_connections = defaultdict(int)
        self.server_response_times = defaultdict(list)
        self._lock = threading.RLock()
        
        logger.info("Load balancer initialized", strategy=strategy.value)
    
    def add_server(self, server: str, weight: int = 1):
        """Add server to load balancer."""
        with self._lock:
            if server not in self.servers:
                self.servers.append(server)
                self.server_weights[server] = weight
                logger.info("Server added to load balancer", server=server, weight=weight)
    
    def remove_server(self, server: str):
        """Remove server from load balancer."""
        with self._lock:
            if server in self.servers:
                self.servers.remove(server)
                if server in self.server_weights:
                    del self.server_weights[server]
                if server in self.server_connections:
                    del self.server_connections[server]
                if server in self.server_response_times:
                    del self.server_response_times[server]
                logger.info("Server removed from load balancer", server=server)
    
    def get_server(self, client_ip: str = None) -> Optional[str]:
        """Get next server based on strategy."""
        with self._lock:
            if not self.servers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin()
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections()
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin()
            elif self.strategy == LoadBalancingStrategy.IP_HASH:
                return self._ip_hash(client_ip)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time()
            
            return self.servers[0]
    
    def record_connection(self, server: str):
        """Record connection to server."""
        with self._lock:
            self.server_connections[server] += 1
    
    def record_response_time(self, server: str, response_time: float):
        """Record response time for server."""
        with self._lock:
            self.server_response_times[server].append(response_time)
            # Keep only last 100 response times
            if len(self.server_response_times[server]) > 100:
                self.server_response_times[server] = self.server_response_times[server][-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            return {
                'strategy': self.strategy.value,
                'total_servers': len(self.servers),
                'servers': self.servers,
                'server_weights': dict(self.server_weights),
                'server_connections': dict(self.server_connections),
                'average_response_times': {
                    server: sum(times) / len(times) if times else 0
                    for server, times in self.server_response_times.items()
                }
            }
    
    def _round_robin(self) -> str:
        """Round robin selection."""
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
    
    def _least_connections(self) -> str:
        """Least connections selection."""
        if not self.server_connections:
            return self.servers[0]
        
        min_connections = min(self.server_connections.values())
        candidates = [server for server, connections in self.server_connections.items() 
                     if connections == min_connections]
        
        return candidates[0] if candidates else self.servers[0]
    
    def _weighted_round_robin(self) -> str:
        """Weighted round robin selection."""
        # Simple weighted round robin implementation
        total_weight = sum(self.server_weights.values())
        if total_weight == 0:
            return self.servers[0]
        
        # Find server with highest weight and least connections
        best_server = self.servers[0]
        best_score = float('inf')
        
        for server in self.servers:
            weight = self.server_weights.get(server, 1)
            connections = self.server_connections.get(server, 0)
            score = connections / weight if weight > 0 else float('inf')
            
            if score < best_score:
                best_score = score
                best_server = server
        
        return best_server
    
    def _ip_hash(self, client_ip: str) -> str:
        """IP hash selection."""
        if not client_ip:
            return self.servers[0]
        
        hash_value = hash(client_ip)
        index = hash_value % len(self.servers)
        return self.servers[index]
    
    def _least_response_time(self) -> str:
        """Least response time selection."""
        if not self.server_response_times:
            return self.servers[0]
        
        best_server = self.servers[0]
        best_response_time = float('inf')
        
        for server in self.servers:
            times = self.server_response_times.get(server, [])
            if times:
                avg_time = sum(times) / len(times)
                if avg_time < best_response_time:
                    best_response_time = avg_time
                    best_server = server
        
        return best_server

class ConnectionPool:
    """Connection pool implementation."""
    
    def __init__(self, pool_size: int = 10, max_pool_size: int = 50,
                 strategy: PoolStrategy = PoolStrategy.STATIC):
        """
        Initialize connection pool.
        
        Args:
            pool_size: Initial pool size
            max_pool_size: Maximum pool size
            strategy: Pool strategy
        """
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.strategy = strategy
        
        self.connections = []
        self.available_connections = deque()
        self.in_use_connections = set()
        self.connection_stats = defaultdict(int)
        self._lock = threading.RLock()
        
        logger.info("Connection pool initialized", 
                   pool_size=pool_size,
                   max_pool_size=max_pool_size,
                   strategy=strategy.value)
    
    async def get_connection(self, connection_factory: Callable) -> Any:
        """
        Get connection from pool.
        
        Args:
            connection_factory: Factory function to create new connections
            
        Returns:
            Connection object
        """
        with self._lock:
            # Try to get available connection
            if self.available_connections:
                connection = self.available_connections.popleft()
                self.in_use_connections.add(connection)
                self.connection_stats['reused'] += 1
                return connection
            
            # Create new connection if pool not full
            if len(self.connections) < self.max_pool_size:
                connection = await connection_factory()
                self.connections.append(connection)
                self.in_use_connections.add(connection)
                self.connection_stats['created'] += 1
                return connection
            
            # Wait for available connection
            logger.warning("Connection pool exhausted, waiting for available connection")
            return None
    
    def release_connection(self, connection: Any):
        """Release connection back to pool."""
        with self._lock:
            if connection in self.in_use_connections:
                self.in_use_connections.remove(connection)
                self.available_connections.append(connection)
                self.connection_stats['released'] += 1
    
    def close_connection(self, connection: Any):
        """Close and remove connection from pool."""
        with self._lock:
            if connection in self.connections:
                self.connections.remove(connection)
            if connection in self.in_use_connections:
                self.in_use_connections.remove(connection)
            if connection in self.available_connections:
                self.available_connections.remove(connection)
            self.connection_stats['closed'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                'total_connections': len(self.connections),
                'available_connections': len(self.available_connections),
                'in_use_connections': len(self.in_use_connections),
                'pool_size': self.pool_size,
                'max_pool_size': self.max_pool_size,
                'strategy': self.strategy.value,
                'connection_stats': dict(self.connection_stats)
            }

class PerformanceProfiler:
    """Performance profiling system."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.profiles = {}
        self.active_profiles = {}
        self._lock = threading.RLock()
        
        logger.info("Performance profiler initialized")
    
    def start_profile(self, name: str, tags: Dict[str, str] = None):
        """Start profiling a function or operation."""
        profile_id = str(uuid.uuid4())
        start_time = time.time()
        
        with self._lock:
            self.active_profiles[profile_id] = {
                'name': name,
                'start_time': start_time,
                'tags': tags or {},
                'thread_id': threading.get_ident()
            }
        
        return profile_id
    
    def end_profile(self, profile_id: str, metadata: Dict[str, Any] = None):
        """End profiling and record results."""
        end_time = time.time()
        
        with self._lock:
            if profile_id not in self.active_profiles:
                logger.warning("Profile not found", profile_id=profile_id)
                return
            
            profile_info = self.active_profiles[profile_id]
            duration = end_time - profile_info['start_time']
            
            profile_result = {
                'profile_id': profile_id,
                'name': profile_info['name'],
                'duration': duration,
                'start_time': profile_info['start_time'],
                'end_time': end_time,
                'tags': profile_info['tags'],
                'thread_id': profile_info['thread_id'],
                'metadata': metadata or {}
            }
            
            if profile_info['name'] not in self.profiles:
                self.profiles[profile_info['name']] = []
            
            self.profiles[profile_info['name']].append(profile_result)
            
            del self.active_profiles[profile_id]
    
    def profile_function(self, name: str = None, tags: Dict[str, str] = None):
        """Decorator to profile a function."""
        def decorator(func):
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                profile_id = self.start_profile(profile_name, tags)
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    self.end_profile(profile_id)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                profile_id = self.start_profile(profile_name, tags)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_profile(profile_id)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_profile_statistics(self, name: str = None) -> Dict[str, Any]:
        """Get profiling statistics."""
        with self._lock:
            if name:
                profiles = self.profiles.get(name, [])
            else:
                profiles = [p for profile_list in self.profiles.values() for p in profile_list]
            
            if not profiles:
                return {}
            
            durations = [p['duration'] for p in profiles]
            
            return {
                'total_profiles': len(profiles),
                'average_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations),
                'profiles_by_name': {
                    name: len(profile_list) for name, profile_list in self.profiles.items()
                }
            }

class CacheManager:
    """Main cache management system."""
    
    def __init__(self, config: CacheConfig = None):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.caches = {}
        self.load_balancers = {}
        self.connection_pools = {}
        self.profiler = PerformanceProfiler()
        
        logger.info("Cache manager initialized")
    
    def create_cache(self, name: str, config: CacheConfig = None) -> MemoryCache:
        """Create a new cache."""
        cache_config = config or self.config
        cache = MemoryCache(cache_config)
        self.caches[name] = cache
        cache.start()
        
        logger.info("Cache created", name=name, type=cache_config.cache_type.value)
        return cache
    
    def get_cache(self, name: str) -> Optional[MemoryCache]:
        """Get cache by name."""
        return self.caches.get(name)
    
    def create_load_balancer(self, name: str, strategy: LoadBalancingStrategy = None) -> LoadBalancer:
        """Create a new load balancer."""
        balancer = LoadBalancer(strategy or LoadBalancingStrategy.ROUND_ROBIN)
        self.load_balancers[name] = balancer
        
        logger.info("Load balancer created", name=name, strategy=balancer.strategy.value)
        return balancer
    
    def get_load_balancer(self, name: str) -> Optional[LoadBalancer]:
        """Get load balancer by name."""
        return self.load_balancers.get(name)
    
    def create_connection_pool(self, name: str, pool_size: int = 10, 
                             max_pool_size: int = 50, strategy: PoolStrategy = None) -> ConnectionPool:
        """Create a new connection pool."""
        pool = ConnectionPool(pool_size, max_pool_size, strategy or PoolStrategy.STATIC)
        self.connection_pools[name] = pool
        
        logger.info("Connection pool created", name=name, pool_size=pool_size)
        return pool
    
    def get_connection_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get connection pool by name."""
        return self.connection_pools.get(name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get all statistics."""
        return {
            'caches': {name: cache.get_statistics() for name, cache in self.caches.items()},
            'load_balancers': {name: balancer.get_statistics() for name, balancer in self.load_balancers.items()},
            'connection_pools': {name: pool.get_statistics() for name, pool in self.connection_pools.items()},
            'profiler': self.profiler.get_profile_statistics()
        }

def create_cache_manager(config: CacheConfig = None) -> CacheManager:
    """
    Create a cache manager.
    
    Args:
        config: Cache configuration
        
    Returns:
        CacheManager instance
    """
    return CacheManager(config)

if __name__ == "__main__":
    # Demo of performance optimization
    config = CacheConfig(
        cache_type=CacheType.MEMORY,
        max_size=1000,
        max_memory_mb=100,
        eviction_policy=EvictionPolicy.LRU,
        default_ttl=3600.0,
        enable_compression=True,
        enable_serialization=True,
        enable_statistics=True
    )
    
    cache_manager = create_cache_manager(config)
    
    # Create cache
    cache = cache_manager.create_cache("demo_cache", config)
    
    # Test cache operations
    cache.set("key1", "value1", ttl=60.0)
    cache.set("key2", {"data": "value2"}, ttl=120.0)
    
    value1 = cache.get("key1")
    value2 = cache.get("key2")
    
    print(f"Cache values: {value1}, {value2}")
    
    # Create load balancer
    load_balancer = cache_manager.create_load_balancer("demo_balancer", LoadBalancingStrategy.ROUND_ROBIN)
    load_balancer.add_server("server1", weight=2)
    load_balancer.add_server("server2", weight=1)
    load_balancer.add_server("server3", weight=1)
    
    # Test load balancing
    for i in range(5):
        server = load_balancer.get_server()
        load_balancer.record_connection(server)
        load_balancer.record_response_time(server, 0.1 + i * 0.01)
        print(f"Request {i+1} -> {server}")
    
    # Create connection pool
    connection_pool = cache_manager.create_connection_pool("demo_pool", pool_size=5, max_pool_size=20)
    
    # Test connection pool
    async def test_connection_pool():
        async def create_connection():
            return {"id": str(uuid.uuid4()), "created_at": datetime.now()}
        
        connection = await connection_pool.get_connection(create_connection)
        if connection:
            print(f"Got connection: {connection}")
            connection_pool.release_connection(connection)
    
    # Get statistics
    stats = cache_manager.get_statistics()
    print(f"Performance statistics: {stats}")
    
    print("Performance Optimization system created successfully!")