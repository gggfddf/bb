#!/usr/bin/env python3
"""
Real-time Chart Updates Module

Implements real-time chart updates and streaming functionality:
- WebSocket data streaming
- Real-time data feeds
- Chart update mechanisms
- Data buffering and caching
- Update frequency controls
- Connection management

Features:
- WebSocket-based data streaming
- Real-time chart updates with minimal latency
- Data buffering and caching for performance
- Configurable update frequencies
- Connection management and error handling
- Support for multiple data sources
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import asyncio
import threading
import time
import json
import queue
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = structlog.get_logger()

class UpdateFrequency(Enum):
    """Update frequency options."""
    REAL_TIME = "real_time"
    FAST = "fast"
    NORMAL = "normal"
    SLOW = "slow"
    MANUAL = "manual"

class ConnectionStatus(Enum):
    """Connection status."""
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class DataSource(Enum):
    """Data source types."""
    WEBSOCKET = "websocket"
    REST_API = "rest_api"
    FILE_STREAM = "file_stream"
    SIMULATED = "simulated"

@dataclass
class UpdateConfig:
    """Configuration for real-time updates."""
    frequency: UpdateFrequency = UpdateFrequency.NORMAL
    buffer_size: int = 1000
    max_points: int = 500
    enable_caching: bool = True
    cache_duration: int = 300  # seconds
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0

@dataclass
class DataPoint:
    """Data point for real-time updates."""
    timestamp: datetime
    value: float
    symbol: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataBuffer:
    """Data buffering and caching system."""
    
    def __init__(self, buffer_size: int = 1000, max_points: int = 500):
        """
        Initialize data buffer.
        
        Args:
            buffer_size: Maximum buffer size
            max_points: Maximum points to display
        """
        self.buffer_size = buffer_size
        self.max_points = max_points
        self.data_buffer = deque(maxlen=buffer_size)
        self.cache = {}
        self.cache_timestamps = {}
    
    def add_data_point(self, data_point: DataPoint):
        """Add data point to buffer."""
        self.data_buffer.append(data_point)
        
        # Update cache
        if data_point.symbol not in self.cache:
            self.cache[data_point.symbol] = deque(maxlen=self.max_points)
        
        self.cache[data_point.symbol].append(data_point)
        self.cache_timestamps[data_point.symbol] = datetime.now()
    
    def get_latest_data(self, symbol: str, n_points: int = None) -> List[DataPoint]:
        """Get latest data for symbol."""
        if symbol not in self.cache:
            return []
        
        data = list(self.cache[symbol])
        if n_points:
            return data[-n_points:]
        return data
    
    def get_data_range(self, symbol: str, start_time: datetime, end_time: datetime) -> List[DataPoint]:
        """Get data within time range."""
        if symbol not in self.cache:
            return []
        
        data = []
        for point in self.cache[symbol]:
            if start_time <= point.timestamp <= end_time:
                data.append(point)
        
        return data
    
    def clear_cache(self, symbol: str = None):
        """Clear cache for symbol or all symbols."""
        if symbol:
            if symbol in self.cache:
                del self.cache[symbol]
            if symbol in self.cache_timestamps:
                del self.cache_timestamps[symbol]
        else:
            self.cache.clear()
            self.cache_timestamps.clear()
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        stats = {
            'buffer_size': len(self.data_buffer),
            'max_buffer_size': self.buffer_size,
            'cached_symbols': list(self.cache.keys()),
            'cache_sizes': {symbol: len(data) for symbol, data in self.cache.items()}
        }
        return stats

class WebSocketDataStream:
    """WebSocket-based data streaming."""
    
    def __init__(self, url: str, symbols: List[str]):
        """
        Initialize WebSocket data stream.
        
        Args:
            url: WebSocket URL
            symbols: List of symbols to subscribe to
        """
        self.url = url
        self.symbols = symbols
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Optional WebSocket import
        try:
            import websockets
            self.websockets = websockets
            self.WEBSOCKET_AVAILABLE = True
        except ImportError:
            self.WEBSOCKET_AVAILABLE = False
            logger.warning("WebSocket library not available, using simulated data")
    
    async def connect(self):
        """Connect to WebSocket."""
        if not self.WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket not available, using simulated connection")
            self.connection_status = ConnectionStatus.CONNECTED
            self.is_connected = True
            return
        
        try:
            self.connection_status = ConnectionStatus.CONNECTING
            self.websocket = await self.websockets.connect(self.url)
            self.connection_status = ConnectionStatus.CONNECTED
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("WebSocket connected successfully", url=self.url)
        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            self.is_connected = False
            logger.error("WebSocket connection failed", error=str(e))
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        self.connection_status = ConnectionStatus.DISCONNECTED
        logger.info("WebSocket disconnected")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols."""
        if not self.is_connected:
            logger.warning("WebSocket not connected, cannot subscribe")
            return
        
        subscription_message = {
            'type': 'subscribe',
            'symbols': symbols
        }
        
        try:
            await self.websocket.send(json.dumps(subscription_message))
            logger.info("Subscribed to symbols", symbols=symbols)
        except Exception as e:
            logger.error("Subscription failed", error=str(e))
    
    async def receive_data(self) -> Optional[DataPoint]:
        """Receive data from WebSocket."""
        if not self.is_connected:
            return None
        
        try:
            if self.WEBSOCKET_AVAILABLE:
                message = await self.websocket.recv()
                data = json.loads(message)
                return self._parse_message(data)
            else:
                # Simulated data
                return self._generate_simulated_data()
        except Exception as e:
            logger.error("Error receiving data", error=str(e))
            return None
    
    def _parse_message(self, message: Dict[str, Any]) -> DataPoint:
        """Parse WebSocket message into DataPoint."""
        # This is a generic parser - adjust based on your data format
        timestamp = datetime.fromisoformat(message.get('timestamp', datetime.now().isoformat()))
        value = float(message.get('price', 0.0))
        symbol = message.get('symbol', 'UNKNOWN')
        
        return DataPoint(
            timestamp=timestamp,
            value=value,
            symbol=symbol,
            source='websocket',
            metadata=message
        )
    
    def _generate_simulated_data(self) -> DataPoint:
        """Generate simulated data for testing."""
        symbol = np.random.choice(self.symbols)
        value = 100 + np.random.randn() * 10
        timestamp = datetime.now()
        
        return DataPoint(
            timestamp=timestamp,
            value=value,
            symbol=symbol,
            source='simulated',
            metadata={'simulated': True}
        )

class RealTimeChartUpdater:
    """Real-time chart update mechanism."""
    
    def __init__(self, chart_figure: go.Figure, config: UpdateConfig):
        """
        Initialize real-time chart updater.
        
        Args:
            chart_figure: Plotly figure to update
            config: Update configuration
        """
        self.figure = chart_figure
        self.config = config
        self.data_buffer = DataBuffer(config.buffer_size, config.max_points)
        self.update_callbacks = []
        self.is_updating = False
        self.update_thread = None
        self.update_interval = self._get_update_interval()
    
    def _get_update_interval(self) -> float:
        """Get update interval based on frequency."""
        intervals = {
            UpdateFrequency.REAL_TIME: 0.1,
            UpdateFrequency.FAST: 0.5,
            UpdateFrequency.NORMAL: 1.0,
            UpdateFrequency.SLOW: 5.0,
            UpdateFrequency.MANUAL: None
        }
        return intervals.get(self.config.frequency, 1.0)
    
    def add_update_callback(self, callback: Callable):
        """Add callback function for updates."""
        self.update_callbacks.append(callback)
    
    def start_updates(self):
        """Start real-time updates."""
        if self.is_updating:
            logger.warning("Updates already running")
            return
        
        self.is_updating = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        logger.info("Real-time updates started")
    
    def stop_updates(self):
        """Stop real-time updates."""
        self.is_updating = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("Real-time updates stopped")
    
    def _update_loop(self):
        """Main update loop."""
        while self.is_updating:
            try:
                self._update_chart()
                
                # Call update callbacks
                for callback in self.update_callbacks:
                    try:
                        callback(self.figure)
                    except Exception as e:
                        logger.error("Update callback error", error=str(e))
                
                if self.update_interval:
                    time.sleep(self.update_interval)
                else:
                    break  # Manual mode
                    
            except Exception as e:
                logger.error("Update loop error", error=str(e))
                time.sleep(1.0)
    
    def _update_chart(self):
        """Update chart with latest data."""
        # Get latest data for each symbol
        symbols = list(self.data_buffer.cache.keys())
        
        for symbol in symbols:
            latest_data = self.data_buffer.get_latest_data(symbol, self.config.max_points)
            
            if not latest_data:
                continue
            
            # Extract time series data
            timestamps = [point.timestamp for point in latest_data]
            values = [point.value for point in latest_data]
            
            # Update or add trace
            trace_found = False
            for trace in self.figure.data:
                if hasattr(trace, 'name') and trace.name == symbol:
                    trace.x = timestamps
                    trace.y = values
                    trace_found = True
                    break
            
            if not trace_found:
                # Add new trace
                self.figure.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    name=symbol
                ))
    
    def add_data_point(self, data_point: DataPoint):
        """Add data point and trigger update."""
        self.data_buffer.add_data_point(data_point)
    
    def manual_update(self):
        """Trigger manual update."""
        self._update_chart()
        
        # Call update callbacks
        for callback in self.update_callbacks:
            try:
                callback(self.figure)
            except Exception as e:
                logger.error("Manual update callback error", error=str(e))

class ConnectionManager:
    """Connection management and error handling."""
    
    def __init__(self, max_reconnect_attempts: int = 5, reconnect_delay: float = 1.0):
        """
        Initialize connection manager.
        
        Args:
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Delay between reconnection attempts
        """
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.reconnect_attempts = 0
        self.last_connection_time = None
        self.connection_errors = []
    
    async def connect_with_retry(self, connection_func: Callable):
        """Connect with automatic retry."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.connection_status = ConnectionStatus.CONNECTING
                await connection_func()
                self.connection_status = ConnectionStatus.CONNECTED
                self.last_connection_time = datetime.now()
                self.reconnect_attempts = 0
                self.connection_errors.clear()
                logger.info("Connection established successfully")
                return True
            except Exception as e:
                self.reconnect_attempts += 1
                self.connection_status = ConnectionStatus.ERROR
                self.connection_errors.append({
                    'attempt': self.reconnect_attempts,
                    'error': str(e),
                    'timestamp': datetime.now()
                })
                
                logger.warning(f"Connection attempt {self.reconnect_attempts} failed", 
                             error=str(e), 
                             remaining_attempts=self.max_reconnect_attempts - self.reconnect_attempts)
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay * self.reconnect_attempts)
        
        logger.error("Max reconnection attempts reached")
        return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'status': self.connection_status.value,
            'reconnect_attempts': self.reconnect_attempts,
            'max_attempts': self.max_reconnect_attempts,
            'last_connection': self.last_connection_time,
            'error_count': len(self.connection_errors),
            'recent_errors': self.connection_errors[-5:] if self.connection_errors else []
        }

class RealTimeDataManager:
    """Main real-time data manager."""
    
    def __init__(self, config: UpdateConfig):
        """
        Initialize real-time data manager.
        
        Args:
            config: Update configuration
        """
        self.config = config
        self.data_streams = {}
        self.chart_updaters = {}
        self.connection_manager = ConnectionManager(
            config.reconnect_attempts, 
            config.reconnect_delay
        )
        self.is_running = False
    
    def add_data_stream(self, name: str, stream: WebSocketDataStream):
        """Add data stream."""
        self.data_streams[name] = stream
        logger.info("Data stream added", name=name)
    
    def add_chart_updater(self, name: str, updater: RealTimeChartUpdater):
        """Add chart updater."""
        self.chart_updaters[name] = updater
        logger.info("Chart updater added", name=name)
    
    async def start_all_streams(self):
        """Start all data streams."""
        self.is_running = True
        
        for name, stream in self.data_streams.items():
            try:
                await stream.connect()
                if stream.is_connected:
                    await stream.subscribe(stream.symbols)
                    logger.info("Data stream started", name=name)
            except Exception as e:
                logger.error(f"Failed to start stream {name}", error=str(e))
    
    def start_all_updaters(self):
        """Start all chart updaters."""
        for name, updater in self.chart_updaters.items():
            updater.start_updates()
            logger.info("Chart updater started", name=name)
    
    async def stop_all_streams(self):
        """Stop all data streams."""
        self.is_running = False
        
        for name, stream in self.data_streams.items():
            try:
                await stream.disconnect()
                logger.info("Data stream stopped", name=name)
            except Exception as e:
                logger.error(f"Failed to stop stream {name}", error=str(e))
    
    def stop_all_updaters(self):
        """Stop all chart updaters."""
        for name, updater in self.chart_updaters.items():
            updater.stop_updates()
            logger.info("Chart updater stopped", name=name)
    
    async def run_data_loop(self):
        """Main data processing loop."""
        while self.is_running:
            for name, stream in self.data_streams.items():
                if stream.is_connected:
                    try:
                        data_point = await stream.receive_data()
                        if data_point:
                            # Distribute data to all chart updaters
                            for updater_name, updater in self.chart_updaters.items():
                                updater.add_data_point(data_point)
                    except Exception as e:
                        logger.error(f"Error in data stream {name}", error=str(e))
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall status."""
        status = {
            'is_running': self.is_running,
            'data_streams': {},
            'chart_updaters': {},
            'connection_stats': self.connection_manager.get_connection_stats()
        }
        
        for name, stream in self.data_streams.items():
            status['data_streams'][name] = {
                'connected': stream.is_connected,
                'status': stream.connection_status.value,
                'symbols': stream.symbols
            }
        
        for name, updater in self.chart_updaters.items():
            status['chart_updaters'][name] = {
                'is_updating': updater.is_updating,
                'buffer_stats': updater.data_buffer.get_buffer_stats()
            }
        
        return status

def create_realtime_manager(config: UpdateConfig = None) -> RealTimeDataManager:
    """
    Create a real-time data manager.
    
    Args:
        config: Update configuration
        
    Returns:
        RealTimeDataManager instance
    """
    if config is None:
        config = UpdateConfig()
    
    return RealTimeDataManager(config)

if __name__ == "__main__":
    # Demo of real-time updates
    import asyncio
    
    # Create sample chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='BTC/USD'))
    
    # Create configuration
    config = UpdateConfig(frequency=UpdateFrequency.FAST)
    
    # Create real-time manager
    manager = create_realtime_manager(config)
    
    # Create data stream
    stream = WebSocketDataStream("wss://example.com", ["BTC/USD", "ETH/USD"])
    manager.add_data_stream("crypto_stream", stream)
    
    # Create chart updater
    updater = RealTimeChartUpdater(fig, config)
    manager.add_chart_updater("main_chart", updater)
    
    print("Real-time data manager created successfully!")
    print(f"Update frequency: {config.frequency.value}")
    print(f"Buffer size: {config.buffer_size}")
    print(f"Max points: {config.max_points}")