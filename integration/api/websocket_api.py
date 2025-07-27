"""
WebSocket API Endpoints for ML Stock Predictor Platform

This module provides comprehensive WebSocket endpoints for real-time data streaming,
trading signals, portfolio updates, and system notifications.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from flask import Flask, request, session
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from dataclasses import dataclass, asdict

from ..security.auth import get_api_auth, require_auth
from ..security.audit_logger import get_audit_logger

logger = logging.getLogger(__name__)

@dataclass
class WebSocketClient:
    """Client connection information"""
    sid: str
    user_id: Optional[str]
    username: Optional[str]
    rooms: List[str]
    connected_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str

@dataclass
class RealTimeData:
    """Real-time data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    change: float
    change_percent: float
    high: float
    low: float
    open_price: float
    previous_close: float

@dataclass
class TradingSignal:
    """Trading signal structure"""
    signal_id: str
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    timestamp: datetime
    price: float
    indicators: Dict[str, Any]
    strategy: str
    reasoning: str

@dataclass
class PortfolioUpdate:
    """Portfolio update structure"""
    user_id: str
    timestamp: datetime
    total_value: float
    daily_change: float
    daily_change_percent: float
    positions: List[Dict[str, Any]]
    cash_balance: float
    unrealized_pnl: float
    realized_pnl: float

class WebSocketAPI:
    """
    Comprehensive WebSocket API implementation for real-time trading system.
    """
    
    def __init__(self, app: Flask):
        self.app = app
        self.socketio = SocketIO(
            app, 
            cors_allowed_origins="*",
            async_mode='threading',
            logger=True,
            engineio_logger=True
        )
        
        # Client management
        self.clients: Dict[str, WebSocketClient] = {}
        self.rooms: Dict[str, List[str]] = {
            'market_data': [],
            'trading_signals': [],
            'portfolio_updates': [],
            'system_alerts': [],
            'price_alerts': []
        }
        
        # Data streams
        self.active_streams: Dict[str, bool] = {}
        self.stream_intervals: Dict[str, int] = {}
        
        # Security components
        self.api_auth = get_api_auth()
        self.audit_logger = get_audit_logger()
        
        self._register_events()
        logger.info("WebSocket API initialized successfully")

    def _register_events(self):
        """Register all WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            try:
                client = WebSocketClient(
                    sid=request.sid,
                    user_id=None,
                    username=None,
                    rooms=[],
                    connected_at=datetime.utcnow(),
                    last_activity=datetime.utcnow(),
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent', 'Unknown')
                )
                
                self.clients[request.sid] = client
                
                # Log connection
                self.audit_logger.log_api_access(
                    user_id=None,
                    endpoint="websocket_connect",
                    method="CONNECT",
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent'),
                    success=True
                )
                
                emit('connection_established', {
                    'message': 'Connected to WebSocket API',
                    'timestamp': datetime.utcnow().isoformat(),
                    'session_id': request.sid
                })
                
                logger.info(f"Client connected: {request.sid} from {request.remote_addr}")
                
            except Exception as e:
                logger.error(f"Error in connection handler: {e}")
                disconnect()

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            try:
                client = self.clients.get(request.sid)
                if client:
                    # Leave all rooms
                    for room in client.rooms:
                        leave_room(room)
                        if room in self.rooms:
                            self.rooms[room] = [c for c in self.rooms[room] if c != request.sid]
                    
                    # Log disconnection
                    self.audit_logger.log_api_access(
                        user_id=client.user_id,
                        endpoint="websocket_disconnect",
                        method="DISCONNECT",
                        ip_address=client.ip_address,
                        user_agent=client.user_agent,
                        success=True
                    )
                    
                    del self.clients[request.sid]
                    logger.info(f"Client disconnected: {request.sid}")
                
            except Exception as e:
                logger.error(f"Error in disconnection handler: {e}")

        @self.socketio.on('authenticate')
        def handle_authenticate(data):
            """Handle client authentication"""
            try:
                token = data.get('token')
                if not token:
                    emit('authentication_error', {'error': 'Token required'})
                    return
                
                # Verify token
                user_data = self.api_auth.verify_token(token)
                if not user_data:
                    emit('authentication_error', {'error': 'Invalid token'})
                    return
                
                # Update client info
                client = self.clients.get(request.sid)
                if client:
                    client.user_id = user_data['user_id']
                    client.username = user_data['username']
                    client.last_activity = datetime.utcnow()
                
                # Log authentication
                self.audit_logger.log_api_access(
                    user_id=user_data['user_id'],
                    endpoint="websocket_authenticate",
                    method="AUTH",
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent'),
                    success=True
                )
                
                emit('authentication_success', {
                    'message': 'Authentication successful',
                    'user_id': user_data['user_id'],
                    'username': user_data['username'],
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                logger.info(f"Client authenticated: {request.sid} as {user_data['username']}")
                
            except Exception as e:
                logger.error(f"Error in authentication handler: {e}")
                emit('authentication_error', {'error': 'Authentication failed'})

        @self.socketio.on('subscribe_market_data')
        def handle_subscribe_market_data(data):
            """Subscribe to real-time market data"""
            try:
                symbols = data.get('symbols', [])
                interval = data.get('interval', 1000)  # milliseconds
                
                if not symbols:
                    emit('subscription_error', {'error': 'Symbols required'})
                    return
                
                client = self.clients.get(request.sid)
                if not client or not client.user_id:
                    emit('subscription_error', {'error': 'Authentication required'})
                    return
                
                room = 'market_data'
                join_room(room)
                if request.sid not in self.rooms[room]:
                    self.rooms[room].append(request.sid)
                
                if room not in client.rooms:
                    client.rooms.append(room)
                
                # Start market data stream if not already running
                if not self.active_streams.get('market_data', False):
                    self._start_market_data_stream(symbols, interval)
                
                emit('subscription_success', {
                    'room': room,
                    'symbols': symbols,
                    'interval': interval,
                    'message': f'Subscribed to market data for {len(symbols)} symbols'
                })
                
                logger.info(f"Client {request.sid} subscribed to market data: {symbols}")
                
            except Exception as e:
                logger.error(f"Error in market data subscription: {e}")
                emit('subscription_error', {'error': 'Subscription failed'})

        @self.socketio.on('subscribe_trading_signals')
        def handle_subscribe_trading_signals(data):
            """Subscribe to trading signals"""
            try:
                symbols = data.get('symbols', [])
                signal_types = data.get('signal_types', ['buy', 'sell', 'hold'])
                
                client = self.clients.get(request.sid)
                if not client or not client.user_id:
                    emit('subscription_error', {'error': 'Authentication required'})
                    return
                
                room = 'trading_signals'
                join_room(room)
                if request.sid not in self.rooms[room]:
                    self.rooms[room].append(request.sid)
                
                if room not in client.rooms:
                    client.rooms.append(room)
                
                # Start trading signals stream if not already running
                if not self.active_streams.get('trading_signals', False):
                    self._start_trading_signals_stream(symbols, signal_types)
                
                emit('subscription_success', {
                    'room': room,
                    'symbols': symbols,
                    'signal_types': signal_types,
                    'message': f'Subscribed to trading signals for {len(symbols)} symbols'
                })
                
                logger.info(f"Client {request.sid} subscribed to trading signals: {symbols}")
                
            except Exception as e:
                logger.error(f"Error in trading signals subscription: {e}")
                emit('subscription_error', {'error': 'Subscription failed'})

        @self.socketio.on('subscribe_portfolio')
        def handle_subscribe_portfolio(data):
            """Subscribe to portfolio updates"""
            try:
                client = self.clients.get(request.sid)
                if not client or not client.user_id:
                    emit('subscription_error', {'error': 'Authentication required'})
                    return
                
                room = f'portfolio_{client.user_id}'
                join_room(room)
                if room not in self.rooms:
                    self.rooms[room] = []
                if request.sid not in self.rooms[room]:
                    self.rooms[room].append(request.sid)
                
                if room not in client.rooms:
                    client.rooms.append(room)
                
                # Start portfolio updates stream
                if not self.active_streams.get(f'portfolio_{client.user_id}', False):
                    self._start_portfolio_stream(client.user_id)
                
                emit('subscription_success', {
                    'room': room,
                    'message': 'Subscribed to portfolio updates'
                })
                
                logger.info(f"Client {request.sid} subscribed to portfolio updates")
                
            except Exception as e:
                logger.error(f"Error in portfolio subscription: {e}")
                emit('subscription_error', {'error': 'Subscription failed'})

        @self.socketio.on('subscribe_alerts')
        def handle_subscribe_alerts(data):
            """Subscribe to system alerts and notifications"""
            try:
                alert_types = data.get('alert_types', ['system', 'price', 'trading'])
                
                client = self.clients.get(request.sid)
                if not client or not client.user_id:
                    emit('subscription_error', {'error': 'Authentication required'})
                    return
                
                room = 'system_alerts'
                join_room(room)
                if request.sid not in self.rooms[room]:
                    self.rooms[room].append(request.sid)
                
                if room not in client.rooms:
                    client.rooms.append(room)
                
                emit('subscription_success', {
                    'room': room,
                    'alert_types': alert_types,
                    'message': f'Subscribed to {len(alert_types)} alert types'
                })
                
                logger.info(f"Client {request.sid} subscribed to alerts: {alert_types}")
                
            except Exception as e:
                logger.error(f"Error in alerts subscription: {e}")
                emit('subscription_error', {'error': 'Subscription failed'})

        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """Unsubscribe from a room"""
            try:
                room = data.get('room')
                if not room:
                    emit('unsubscription_error', {'error': 'Room required'})
                    return
                
                client = self.clients.get(request.sid)
                if client and room in client.rooms:
                    leave_room(room)
                    client.rooms.remove(room)
                    
                    if room in self.rooms and request.sid in self.rooms[room]:
                        self.rooms[room].remove(request.sid)
                
                emit('unsubscription_success', {
                    'room': room,
                    'message': f'Unsubscribed from {room}'
                })
                
                logger.info(f"Client {request.sid} unsubscribed from {room}")
                
            except Exception as e:
                logger.error(f"Error in unsubscription: {e}")
                emit('unsubscription_error', {'error': 'Unsubscription failed'})

        @self.socketio.on('ping')
        def handle_ping():
            """Handle ping for connection health check"""
            try:
                client = self.clients.get(request.sid)
                if client:
                    client.last_activity = datetime.utcnow()
                
                emit('pong', {
                    'timestamp': datetime.utcnow().isoformat(),
                    'session_id': request.sid
                })
                
            except Exception as e:
                logger.error(f"Error in ping handler: {e}")

        @self.socketio.on('client_message')
        def handle_client_message(data):
            """Handle general client messages"""
            try:
                client = self.clients.get(request.sid)
                if client:
                    client.last_activity = datetime.utcnow()
                
                message_type = data.get('type', 'general')
                message_data = data.get('data', {})
                
                # Log message
                self.audit_logger.log_api_access(
                    user_id=client.user_id if client else None,
                    endpoint="websocket_message",
                    method="MESSAGE",
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent'),
                    success=True,
                    details={'message_type': message_type, 'data': message_data}
                )
                
                emit('server_message', {
                    'type': 'response',
                    'message': 'Message received',
                    'data': message_data,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                logger.info(f"Received message from {request.sid}: {message_type}")
                
            except Exception as e:
                logger.error(f"Error in client message handler: {e}")
                emit('server_message', {
                    'type': 'error',
                    'message': 'Message processing failed',
                    'error': str(e)
                })

    def _start_market_data_stream(self, symbols: List[str], interval: int):
        """Start real-time market data stream"""
        self.active_streams['market_data'] = True
        
        def stream_market_data():
            while self.active_streams.get('market_data', False):
                try:
                    # Simulate real-time market data
                    for symbol in symbols:
                        # In production, this would fetch real market data
                        data = RealTimeData(
                            symbol=symbol,
                            timestamp=datetime.utcnow(),
                            price=100.0 + (hash(symbol) % 50),  # Simulated price
                            volume=1000000 + (hash(symbol) % 500000),
                            change=1.5,
                            change_percent=1.5,
                            high=105.0,
                            low=95.0,
                            open_price=100.0,
                            previous_close=98.5
                        )
                        
                        self.socketio.emit('market_data_update', asdict(data), room='market_data')
                    
                    # Wait for next update
                    import time
                    time.sleep(interval / 1000.0)
                    
                except Exception as e:
                    logger.error(f"Error in market data stream: {e}")
                    break
        
        # Start stream in background thread
        import threading
        thread = threading.Thread(target=stream_market_data, daemon=True)
        thread.start()

    def _start_trading_signals_stream(self, symbols: List[str], signal_types: List[str]):
        """Start real-time trading signals stream"""
        self.active_streams['trading_signals'] = True
        
        def stream_trading_signals():
            while self.active_streams.get('trading_signals', False):
                try:
                    # Simulate trading signals
                    for symbol in symbols:
                        signal = TradingSignal(
                            signal_id=f"signal_{datetime.utcnow().timestamp()}",
                            symbol=symbol,
                            signal_type=signal_types[hash(symbol) % len(signal_types)],
                            confidence=0.75 + (hash(symbol) % 25) / 100.0,
                            timestamp=datetime.utcnow(),
                            price=100.0 + (hash(symbol) % 50),
                            indicators={'rsi': 65, 'macd': 0.5, 'volume': 1200000},
                            strategy='ml_ensemble',
                            reasoning='Strong buy signal based on multiple indicators'
                        )
                        
                        self.socketio.emit('trading_signal', asdict(signal), room='trading_signals')
                    
                    # Wait for next update
                    import time
                    time.sleep(5)  # 5 second intervals
                    
                except Exception as e:
                    logger.error(f"Error in trading signals stream: {e}")
                    break
        
        # Start stream in background thread
        import threading
        thread = threading.Thread(target=stream_trading_signals, daemon=True)
        thread.start()

    def _start_portfolio_stream(self, user_id: str):
        """Start real-time portfolio updates stream"""
        self.active_streams[f'portfolio_{user_id}'] = True
        
        def stream_portfolio_updates():
            while self.active_streams.get(f'portfolio_{user_id}', False):
                try:
                    # Simulate portfolio updates
                    portfolio = PortfolioUpdate(
                        user_id=user_id,
                        timestamp=datetime.utcnow(),
                        total_value=100000.0 + (hash(user_id) % 50000),
                        daily_change=1500.0,
                        daily_change_percent=1.5,
                        positions=[
                            {
                                'symbol': 'AAPL',
                                'quantity': 100,
                                'avg_price': 150.0,
                                'current_price': 155.0,
                                'unrealized_pnl': 500.0
                            }
                        ],
                        cash_balance=25000.0,
                        unrealized_pnl=500.0,
                        realized_pnl=2500.0
                    )
                    
                    self.socketio.emit('portfolio_update', asdict(portfolio), room=f'portfolio_{user_id}')
                    
                    # Wait for next update
                    import time
                    time.sleep(10)  # 10 second intervals
                    
                except Exception as e:
                    logger.error(f"Error in portfolio stream: {e}")
                    break
        
        # Start stream in background thread
        import threading
        thread = threading.Thread(target=stream_portfolio_updates, daemon=True)
        thread.start()

    def broadcast_system_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """Broadcast system alert to all connected clients"""
        try:
            alert_data = {
                'type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.socketio.emit('system_alert', alert_data, room='system_alerts')
            logger.info(f"System alert broadcast: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"Error broadcasting system alert: {e}")

    def send_price_alert(self, user_id: str, symbol: str, price: float, condition: str):
        """Send price alert to specific user"""
        try:
            alert_data = {
                'type': 'price_alert',
                'symbol': symbol,
                'price': price,
                'condition': condition,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.socketio.emit('price_alert', alert_data, room=f'portfolio_{user_id}')
            logger.info(f"Price alert sent to user {user_id}: {symbol} {condition} {price}")
            
        except Exception as e:
            logger.error(f"Error sending price alert: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        try:
            active_clients = len(self.clients)
            total_rooms = len(self.rooms)
            room_stats = {}
            
            for room, clients in self.rooms.items():
                room_stats[room] = len(clients)
            
            return {
                'active_clients': active_clients,
                'total_rooms': total_rooms,
                'room_stats': room_stats,
                'active_streams': list(self.active_streams.keys()),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {}

    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the WebSocket server"""
        try:
            logger.info(f"Starting WebSocket server on {host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")


# Global instance management
_websocket_api_instance = None

def get_websocket_api(app: Flask = None) -> WebSocketAPI:
    """Get or create WebSocket API instance"""
    global _websocket_api_instance
    
    if _websocket_api_instance is None and app:
        _websocket_api_instance = WebSocketAPI(app)
    
    return _websocket_api_instance


# Example usage and testing
if __name__ == "__main__":
    app = Flask(__name__)
    ws_api = WebSocketAPI(app)
    ws_api.run()