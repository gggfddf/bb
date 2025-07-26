"""
Performance Profiler for Trading System

This module provides performance profiling functionality for analyzing
and optimizing system performance.
"""

import json
import logging
import threading
import time
import cProfile
import pstats
import io
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import functools
import traceback
import hashlib
import statistics

logger = logging.getLogger(__name__)

class ProfilerType(Enum):
    """Profiler type enumeration"""
    FUNCTION = "function"
    METHOD = "method"
    BLOCK = "block"
    CONTEXT = "context"

class ProfilerMode(Enum):
    """Profiler mode enumeration"""
    TIME = "time"
    MEMORY = "memory"
    CPU = "cpu"
    CALLS = "calls"
    COMPREHENSIVE = "comprehensive"

@dataclass
class ProfilerResult:
    """Profiler result structure"""
    profiler_id: str
    name: str
    profiler_type: ProfilerType
    mode: ProfilerMode
    start_time: datetime
    end_time: datetime
    duration: float
    memory_usage: Optional[float]
    cpu_usage: Optional[float]
    call_count: int
    stack_trace: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Performance metrics structure"""
    total_calls: int
    total_duration: float
    average_duration: float
    min_duration: float
    max_duration: float
    memory_peak: float
    cpu_peak: float
    error_count: int
    success_rate: float

class PerformanceProfiler:
    """
    Performance profiler for analyzing and optimizing system performance.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        
        # Profiling data
        self.profiler_results: Dict[str, ProfilerResult] = {}
        self.active_profilers: Dict[str, ProfilerResult] = {}
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Configuration
        self.enabled = True
        self.min_duration_threshold = 0.001  # 1ms
        self.max_results = 10000
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_profiles': 0,
            'active_profiles': 0,
            'total_duration': 0.0,
            'average_duration': 0.0
        }
        
        logger.info(f"Performance profiler initialized: {name}")
    
    def profile(self, name: Optional[str] = None, mode: ProfilerMode = ProfilerMode.TIME,
                profiler_type: ProfilerType = ProfilerType.FUNCTION):
        """
        Decorator for profiling functions and methods.
        
        Args:
            name: Profiler name (defaults to function name)
            mode: Profiling mode
            profiler_type: Profiler type
            
        Returns:
            Decorated function
        """
        def decorator(func):
            profiler_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                profiler_id = self._start_profiler(profiler_name, mode, profiler_type)
                
                try:
                    result = func(*args, **kwargs)
                    self._stop_profiler(profiler_id, success=True)
                    return result
                except Exception as e:
                    self._stop_profiler(profiler_id, success=False, error=str(e))
                    raise
            
            return wrapper
        
        return decorator
    
    def profile_block(self, name: str, mode: ProfilerMode = ProfilerMode.TIME):
        """
        Context manager for profiling code blocks.
        
        Args:
            name: Profiler name
            mode: Profiling mode
            
        Returns:
            Context manager
        """
        return ProfilerContext(self, name, mode, ProfilerType.BLOCK)
    
    def start_profiler(self, name: str, mode: ProfilerMode = ProfilerMode.TIME,
                      profiler_type: ProfilerType = ProfilerType.FUNCTION) -> str:
        """
        Start a profiler.
        
        Args:
            name: Profiler name
            mode: Profiling mode
            profiler_type: Profiler type
            
        Returns:
            Profiler ID
        """
        if not self.enabled:
            return ""
        
        return self._start_profiler(name, mode, profiler_type)
    
    def stop_profiler(self, profiler_id: str, success: bool = True, error: Optional[str] = None):
        """
        Stop a profiler.
        
        Args:
            profiler_id: Profiler ID
            success: Whether the operation was successful
            error: Error message if operation failed
        """
        if not self.enabled or not profiler_id:
            return
        
        self._stop_profiler(profiler_id, success, error)
    
    def get_profiler_result(self, profiler_id: str) -> Optional[ProfilerResult]:
        """
        Get profiler result by ID.
        
        Args:
            profiler_id: Profiler ID
            
        Returns:
            Profiler result if found
        """
        return self.profiler_results.get(profiler_id)
    
    def get_performance_metrics(self, name: str) -> Optional[PerformanceMetrics]:
        """
        Get performance metrics for a specific profiler.
        
        Args:
            name: Profiler name
            
        Returns:
            Performance metrics if found
        """
        return self.performance_metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all performance metrics"""
        return self.performance_metrics.copy()
    
    def get_slowest_operations(self, limit: int = 10) -> List[ProfilerResult]:
        """
        Get the slowest operations.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of slowest profiler results
        """
        sorted_results = sorted(
            self.profiler_results.values(),
            key=lambda x: x.duration,
            reverse=True
        )
        return sorted_results[:limit]
    
    def get_most_called_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most called operations.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of most called operations
        """
        call_counts = {}
        
        for result in self.profiler_results.values():
            if result.name not in call_counts:
                call_counts[result.name] = {
                    'name': result.name,
                    'call_count': 0,
                    'total_duration': 0.0,
                    'average_duration': 0.0
                }
            
            call_counts[result.name]['call_count'] += 1
            call_counts[result.name]['total_duration'] += result.duration
        
        # Calculate averages
        for stats in call_counts.values():
            stats['average_duration'] = stats['total_duration'] / stats['call_count']
        
        sorted_operations = sorted(
            call_counts.values(),
            key=lambda x: x['call_count'],
            reverse=True
        )
        return sorted_operations[:limit]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Performance summary
        """
        if not self.profiler_results:
            return {}
        
        durations = [r.duration for r in self.profiler_results.values()]
        memory_usage = [r.memory_usage for r in self.profiler_results.values() if r.memory_usage is not None]
        cpu_usage = [r.cpu_usage for r in self.profiler_results.values() if r.cpu_usage is not None]
        
        return {
            'total_profiles': len(self.profiler_results),
            'total_duration': sum(durations),
            'average_duration': statistics.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'duration_std': statistics.stdev(durations) if len(durations) > 1 else 0,
            'memory_usage_avg': statistics.mean(memory_usage) if memory_usage else 0,
            'memory_usage_max': max(memory_usage) if memory_usage else 0,
            'cpu_usage_avg': statistics.mean(cpu_usage) if cpu_usage else 0,
            'cpu_usage_max': max(cpu_usage) if cpu_usage else 0,
            'slowest_operation': max(self.profiler_results.values(), key=lambda x: x.duration).name,
            'most_called_operation': max(self.get_most_called_operations(1), key=lambda x: x['call_count'])['name'] if self.profiler_results else None
        }
    
    def clear_results(self):
        """Clear all profiler results"""
        with self._lock:
            self.profiler_results.clear()
            self.performance_metrics.clear()
            self.stats = {
                'total_profiles': 0,
                'active_profiles': 0,
                'total_duration': 0.0,
                'average_duration': 0.0
            }
    
    def enable(self):
        """Enable profiling"""
        self.enabled = True
        logger.info(f"Performance profiler enabled: {self.name}")
    
    def disable(self):
        """Disable profiling"""
        self.enabled = False
        logger.info(f"Performance profiler disabled: {self.name}")
    
    def _start_profiler(self, name: str, mode: ProfilerMode, profiler_type: ProfilerType) -> str:
        """Start a profiler"""
        try:
            profiler_id = self._generate_profiler_id()
            
            profiler_result = ProfilerResult(
                profiler_id=profiler_id,
                name=name,
                profiler_type=profiler_type,
                mode=mode,
                start_time=datetime.utcnow(),
                end_time=None,
                duration=0.0,
                memory_usage=None,
                cpu_usage=None,
                call_count=1,
                stack_trace=None,
                metadata={}
            )
            
            with self._lock:
                self.active_profilers[profiler_id] = profiler_result
                self.stats['active_profiles'] += 1
            
            logger.debug(f"Started profiler: {name} ({profiler_id})")
            return profiler_id
            
        except Exception as e:
            logger.error(f"Error starting profiler: {e}")
            return ""
    
    def _stop_profiler(self, profiler_id: str, success: bool, error: Optional[str] = None):
        """Stop a profiler"""
        try:
            with self._lock:
                if profiler_id not in self.active_profilers:
                    return
                
                profiler_result = self.active_profilers[profiler_id]
                profiler_result.end_time = datetime.utcnow()
                profiler_result.duration = (profiler_result.end_time - profiler_result.start_time).total_seconds()
                
                # Add metadata
                if error:
                    profiler_result.metadata['error'] = error
                    profiler_result.metadata['success'] = False
                else:
                    profiler_result.metadata['success'] = True
                
                # Add stack trace for slow operations
                if profiler_result.duration > self.min_duration_threshold:
                    profiler_result.stack_trace = ''.join(traceback.format_stack())
                
                # Remove from active profilers
                del self.active_profilers[profiler_id]
                self.stats['active_profiles'] -= 1
                
                # Add to results
                self.profiler_results[profiler_id] = profiler_result
                
                # Update statistics
                self.stats['total_profiles'] += 1
                self.stats['total_duration'] += profiler_result.duration
                self.stats['average_duration'] = self.stats['total_duration'] / self.stats['total_profiles']
                
                # Update performance metrics
                self._update_performance_metrics(profiler_result)
                
                # Limit results
                if len(self.profiler_results) > self.max_results:
                    self._cleanup_old_results()
            
            logger.debug(f"Stopped profiler: {profiler_result.name} ({profiler_id}) - {profiler_result.duration:.4f}s")
            
        except Exception as e:
            logger.error(f"Error stopping profiler: {e}")
    
    def _update_performance_metrics(self, profiler_result: ProfilerResult):
        """Update performance metrics for a profiler"""
        name = profiler_result.name
        
        if name not in self.performance_metrics:
            self.performance_metrics[name] = PerformanceMetrics(
                total_calls=0,
                total_duration=0.0,
                average_duration=0.0,
                min_duration=float('inf'),
                max_duration=0.0,
                memory_peak=0.0,
                cpu_peak=0.0,
                error_count=0,
                success_rate=0.0
            )
        
        metrics = self.performance_metrics[name]
        metrics.total_calls += 1
        metrics.total_duration += profiler_result.duration
        metrics.average_duration = metrics.total_duration / metrics.total_calls
        metrics.min_duration = min(metrics.min_duration, profiler_result.duration)
        metrics.max_duration = max(metrics.max_duration, profiler_result.duration)
        
        if profiler_result.memory_usage:
            metrics.memory_peak = max(metrics.memory_peak, profiler_result.memory_usage)
        
        if profiler_result.cpu_usage:
            metrics.cpu_peak = max(metrics.cpu_peak, profiler_result.cpu_usage)
        
        if not profiler_result.metadata.get('success', True):
            metrics.error_count += 1
        
        metrics.success_rate = (metrics.total_calls - metrics.error_count) / metrics.total_calls
    
    def _cleanup_old_results(self):
        """Clean up old profiler results"""
        # Remove oldest results to stay within limit
        sorted_results = sorted(
            self.profiler_results.items(),
            key=lambda x: x[1].start_time
        )
        
        excess_count = len(self.profiler_results) - self.max_results
        for i in range(excess_count):
            if i < len(sorted_results):
                del self.profiler_results[sorted_results[i][0]]
    
    def _generate_profiler_id(self) -> str:
        """Generate a unique profiler ID"""
        return hashlib.md5(f"{self.name}_{time.time()}_{threading.get_ident()}".encode()).hexdigest()


class ProfilerContext:
    """Context manager for profiling code blocks"""
    
    def __init__(self, profiler: PerformanceProfiler, name: str, mode: ProfilerMode, profiler_type: ProfilerType):
        self.profiler = profiler
        self.name = name
        self.mode = mode
        self.profiler_type = profiler_type
        self.profiler_id = None
    
    def __enter__(self):
        self.profiler_id = self.profiler.start_profiler(self.name, self.mode, self.profiler_type)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self.profiler.stop_profiler(self.profiler_id, success, error)


class CPUMemoryProfiler:
    """
    CPU and memory profiler using cProfile.
    """
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats = None
    
    def start(self):
        """Start CPU profiling"""
        self.profiler.enable()
    
    def stop(self):
        """Stop CPU profiling"""
        self.profiler.disable()
    
    def get_stats(self) -> str:
        """Get profiling statistics"""
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        return s.getvalue()
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """Get profiling statistics as dictionary"""
        stats = pstats.Stats(self.profiler)
        stats_dict = {}
        
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            func_name = f"{func[0]}:{func[1]}:{func[2]}"
            stats_dict[func_name] = {
                'call_count': cc,
                'primitive_call_count': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'callers': dict(callers)
            }
        
        return stats_dict


class PerformanceProfilerManager:
    """
    Manager for multiple performance profilers.
    """
    
    def __init__(self):
        self.profilers: Dict[str, PerformanceProfiler] = {}
        self._lock = threading.RLock()
        
        logger.info("Performance profiler manager initialized")
    
    def create_profiler(self, name: str) -> PerformanceProfiler:
        """
        Create a new performance profiler.
        
        Args:
            name: Profiler name
            
        Returns:
            Created performance profiler
        """
        with self._lock:
            if name in self.profilers:
                logger.warning(f"Performance profiler {name} already exists")
                return self.profilers[name]
            
            profiler = PerformanceProfiler(name)
            self.profilers[name] = profiler
            
            logger.info(f"Performance profiler created: {name}")
            return profiler
    
    def get_profiler(self, name: str) -> Optional[PerformanceProfiler]:
        """
        Get a performance profiler by name.
        
        Args:
            name: Profiler name
            
        Returns:
            Performance profiler if found
        """
        return self.profilers.get(name)
    
    def remove_profiler(self, name: str) -> bool:
        """
        Remove a performance profiler.
        
        Args:
            name: Profiler name
            
        Returns:
            True if profiler was removed
        """
        with self._lock:
            if name in self.profilers:
                del self.profilers[name]
                logger.info(f"Performance profiler removed: {name}")
                return True
            return False
    
    def get_all_profilers(self) -> Dict[str, PerformanceProfiler]:
        """Get all performance profilers"""
        return self.profilers.copy()
    
    def get_global_summary(self) -> Dict[str, Any]:
        """
        Get global performance summary across all profilers.
        
        Returns:
            Global performance summary
        """
        summary = {
            'total_profilers': len(self.profilers),
            'total_profiles': 0,
            'total_duration': 0.0,
            'average_duration': 0.0,
            'profiler_summaries': {}
        }
        
        for name, profiler in self.profilers.items():
            profiler_summary = profiler.get_performance_summary()
            summary['profiler_summaries'][name] = profiler_summary
            
            if profiler_summary:
                summary['total_profiles'] += profiler_summary.get('total_profiles', 0)
                summary['total_duration'] += profiler_summary.get('total_duration', 0.0)
        
        if summary['total_profiles'] > 0:
            summary['average_duration'] = summary['total_duration'] / summary['total_profiles']
        
        return summary


# Global instance management
_performance_profiler_manager_instance = None

def get_performance_profiler_manager() -> PerformanceProfilerManager:
    """Get or create performance profiler manager instance"""
    global _performance_profiler_manager_instance
    
    if _performance_profiler_manager_instance is None:
        _performance_profiler_manager_instance = PerformanceProfilerManager()
    
    return _performance_profiler_manager_instance


def init_performance_profiler_manager() -> PerformanceProfilerManager:
    """Initialize performance profiler manager"""
    global _performance_profiler_manager_instance
    
    _performance_profiler_manager_instance = PerformanceProfilerManager()
    
    return _performance_profiler_manager_instance


# Convenience functions
def profile_function(name: Optional[str] = None, mode: ProfilerMode = ProfilerMode.TIME):
    """Decorator for profiling functions"""
    profiler = get_performance_profiler_manager().get_profiler("default")
    if not profiler:
        profiler = get_performance_profiler_manager().create_profiler("default")
    
    return profiler.profile(name, mode, ProfilerType.FUNCTION)


def profile_method(name: Optional[str] = None, mode: ProfilerMode = ProfilerMode.TIME):
    """Decorator for profiling methods"""
    profiler = get_performance_profiler_manager().get_profiler("default")
    if not profiler:
        profiler = get_performance_profiler_manager().create_profiler("default")
    
    return profiler.profile(name, mode, ProfilerType.METHOD)


# Example usage:
# @profile_function("database_query")
# def database_operation():
#     # Database operation code
#     pass

# with profile_block("data_processing"):
#     # Code block to profile
#     pass