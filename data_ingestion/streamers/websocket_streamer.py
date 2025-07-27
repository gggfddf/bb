"""
WebSocket Data Streamer

Provides real-time stock data streaming via WebSocket connections for the ML Stock Predictor Platform.
This module handles live market data feeds and implements the same interface as other data sources.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import websockets
import aiohttp
from sqlalchemy.orm import Session

from ..utils.rate_limiter import RateLimiter
from ..utils.data_validator import DataValidator
from ..utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class WebSocketStreamer:
    """
    WebSocket-based real-time data streamer for stock market data.
    
    Features:
    - Real-time price updates
    - Volume and trade data
    - Multiple symbol subscriptions
    - Automatic reconnection
    - Data validation and storage
    """
    
    def __init__(self, session: Session, config: Dict[str, Any] = None):
        self.session = session
        self.config = config or {}
        
        # WebSocket connection state
        self.websocket = None
        self.is_connected = False
        self.is_running = False
        
        # Subscription management
        self.subscribed_symbols: List[str] = []
        self.data_callbacks: List[Callable] = []
        
        # Initialize utilities
        self.rate_limiter = RateLimiter(requests_per_minute=1000)  # High rate for real-time data
        self.data_validator = DataValidator()
        self.error_handler = ErrorHandler(max_retries=5, base_delay=1.0)
        
        # Configuration
        self.ws_url = self.config.get('websocket_url', 'wss://stream.data.alpaca.markets/v2/iex')
        self.api_key = self.config.get('api_key')
        self.api_secret = self.config.get('api_secret')
        self.reconnect_delay = self.config.get('reconnect_delay', 5)
        self.max_reconnect_attempts = self.config.get('max_reconnect_attempts', 10)
        
        # Data storage
        self.latest_data: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self) -> bool:
        """
        Establish WebSocket connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to WebSocket: {self.ws_url}")
            
            # Create connection with authentication if needed
            if self.api_key and self.api_secret:
                headers = {
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.api_secret
                }
                self.websocket = await websockets.connect(
                    self.ws_url,
                    extra_headers=headers
                )
            else:
                self.websocket = await websockets.connect(self.ws_url)
            
            self.is_connected = True
            logger.info("WebSocket connection established")
            
            # Send authentication message if required
            if self.api_key and self.api_secret:
                auth_message = {
                    "action": "auth",
                    "key": self.api_key,
                    "secret": self.api_secret
                }
                await self.websocket.send(json.dumps(auth_message))
                logger.info("Authentication message sent")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.is_connected = False
        self.is_running = False
        logger.info("WebSocket connection closed")
    
    async def subscribe_symbols(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time data for specified symbols.
        
        Args:
            symbols: List of stock symbols to subscribe to
        
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.is_connected:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Create subscription message
            subscription_message = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
                "bars": symbols
            }
            
            await self.websocket.send(json.dumps(subscription_message))
            self.subscribed_symbols.extend(symbols)
            
            logger.info(f"Subscribed to {len(symbols)} symbols: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            return False
    
    async def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time data for specified symbols.
        
        Args:
            symbols: List of stock symbols to unsubscribe from
        
        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.is_connected:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Create unsubscription message
            unsubscription_message = {
                "action": "unsubscribe",
                "trades": symbols,
                "quotes": symbols,
                "bars": symbols
            }
            
            await self.websocket.send(json.dumps(unsubscription_message))
            
            # Remove from subscribed symbols
            for symbol in symbols:
                if symbol in self.subscribed_symbols:
                    self.subscribed_symbols.remove(symbol)
            
            logger.info(f"Unsubscribed from {len(symbols)} symbols: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from symbols: {e}")
            return False
    
    async def start_streaming(self):
        """Start the WebSocket streaming loop."""
        if not self.is_connected:
            logger.error("WebSocket not connected")
            return
        
        self.is_running = True
        logger.info("Starting WebSocket streaming")
        
        reconnect_attempts = 0
        
        while self.is_running:
            try:
                async for message in self.websocket:
                    if not self.is_running:
                        break
                    
                    await self._process_message(message)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.is_connected = False
                
                if self.is_running and reconnect_attempts < self.max_reconnect_attempts:
                    reconnect_attempts += 1
                    logger.info(f"Attempting to reconnect (attempt {reconnect_attempts}/{self.max_reconnect_attempts})")
                    
                    await asyncio.sleep(self.reconnect_delay)
                    if await self.connect():
                        reconnect_attempts = 0
                        # Resubscribe to symbols
                        if self.subscribed_symbols:
                            await self.subscribe_symbols(self.subscribed_symbols)
                    else:
                        logger.error("Failed to reconnect")
                        break
                else:
                    logger.error("Max reconnection attempts reached")
                    break
                    
            except Exception as e:
                logger.error(f"Error in WebSocket streaming: {e}")
                if not self.is_running:
                    break
    
    async def stop_streaming(self):
        """Stop the WebSocket streaming loop."""
        self.is_running = False
        logger.info("Stopping WebSocket streaming")
    
    async def _process_message(self, message: str):
        """
        Process incoming WebSocket message.
        
        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)
            
            # Handle different message types
            if 'T' in data:  # Trade data
                await self._process_trade_data(data)
            elif 'Q' in data:  # Quote data
                await self._process_quote_data(data)
            elif 'b' in data:  # Bar data
                await self._process_bar_data(data)
            elif 'action' in data:  # Control message
                await self._process_control_message(data)
            else:
                logger.debug(f"Unknown message type: {data}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def _process_trade_data(self, data: Dict[str, Any]):
        """
        Process trade data from WebSocket.
        
        Args:
            data: Trade data dictionary
        """
        try:
            symbol = data.get('S', '')
            timestamp = datetime.fromtimestamp(data.get('t', 0) / 1000000000)
            
            trade_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'price': float(data.get('p', 0)),
                'size': int(data.get('s', 0)),
                'exchange': data.get('x', 'UNKNOWN'),
                'conditions': data.get('c', []),
                'type': 'trade',
                'source': 'websocket'
            }
            
            # Update latest data
            self.latest_data[symbol] = trade_data
            
            # Validate and store data
            validation_result = self.data_validator.validate_stock_data(trade_data)
            if validation_result['is_valid']:
                await self._store_data(trade_data)
            
            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(trade_data)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    async def _process_quote_data(self, data: Dict[str, Any]):
        """
        Process quote data from WebSocket.
        
        Args:
            data: Quote data dictionary
        """
        try:
            symbol = data.get('S', '')
            timestamp = datetime.fromtimestamp(data.get('t', 0) / 1000000000)
            
            quote_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'bid_price': float(data.get('bp', 0)),
                'bid_size': int(data.get('bs', 0)),
                'ask_price': float(data.get('ap', 0)),
                'ask_size': int(data.get('as', 0)),
                'exchange': data.get('x', 'UNKNOWN'),
                'conditions': data.get('c', []),
                'type': 'quote',
                'source': 'websocket'
            }
            
            # Update latest data
            self.latest_data[symbol] = quote_data
            
            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(quote_data)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing quote data: {e}")
    
    async def _process_bar_data(self, data: Dict[str, Any]):
        """
        Process bar data from WebSocket.
        
        Args:
            data: Bar data dictionary
        """
        try:
            symbol = data.get('S', '')
            timestamp = datetime.fromtimestamp(data.get('t', 0) / 1000000000)
            
            bar_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'open': float(data.get('o', 0)),
                'high': float(data.get('h', 0)),
                'low': float(data.get('l', 0)),
                'close': float(data.get('c', 0)),
                'volume': int(data.get('v', 0)),
                'exchange': data.get('x', 'UNKNOWN'),
                'type': 'bar',
                'source': 'websocket'
            }
            
            # Update latest data
            self.latest_data[symbol] = bar_data
            
            # Validate and store data
            validation_result = self.data_validator.validate_stock_data(bar_data)
            if validation_result['is_valid']:
                await self._store_data(bar_data)
            
            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(bar_data)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing bar data: {e}")
    
    async def _process_control_message(self, data: Dict[str, Any]):
        """
        Process control messages from WebSocket.
        
        Args:
            data: Control message dictionary
        """
        action = data.get('action', '')
        
        if action == 'authenticated':
            logger.info("WebSocket authentication successful")
        elif action == 'subscription':
            logger.info(f"Subscription status: {data}")
        elif action == 'error':
            logger.error(f"WebSocket error: {data}")
        else:
            logger.debug(f"Control message: {data}")
    
    async def _store_data(self, data: Dict[str, Any]):
        """
        Store data in the database.
        
        Args:
            data: Data dictionary to store
        """
        try:
            from ..models import StockData
            
            # Create StockData record for OHLC data
            if all(key in data for key in ['open', 'high', 'low', 'close']):
                stock_data = StockData(
                    timestamp=data['timestamp'],
                    symbol=data['symbol'],
                    exchange=data.get('exchange', 'UNKNOWN'),
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data.get('volume', 0),
                    timeframe='1m',  # Real-time data is typically 1-minute
                    source='websocket'
                )
                
                self.session.add(stock_data)
                self.session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store WebSocket data: {e}")
            self.session.rollback()
    
    def add_data_callback(self, callback: Callable):
        """
        Add a callback function to be called when new data is received.
        
        Args:
            callback: Async function to call with data
        """
        self.data_callbacks.append(callback)
        logger.info("Data callback added")
    
    def remove_data_callback(self, callback: Callable):
        """
        Remove a data callback function.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
            logger.info("Data callback removed")
    
    def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest data for a specific symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Latest data dictionary or None
        """
        return self.latest_data.get(symbol)
    
    def get_all_latest_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all latest data for all subscribed symbols.
        
        Returns:
            Dictionary of latest data by symbol
        """
        return self.latest_data.copy()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status.
        
        Returns:
            Connection status dictionary
        """
        return {
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'subscribed_symbols': self.subscribed_symbols.copy(),
            'latest_data_count': len(self.latest_data),
            'callback_count': len(self.data_callbacks)
        }