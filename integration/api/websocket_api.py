"""
WebSocket API Endpoints for ML Stock Predictor Platform

This module provides WebSocket endpoints for real-time data streaming
and trading system notifications.
"""

import logging
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketAPI:
    """
    WebSocket API implementation for real-time data and notifications.
    """
    def __init__(self, app: Flask):
        self.app = app
        self.socketio = SocketIO(app, cors_allowed_origins="*")
        self._register_events()
        logger.info("WebSocket API initialized successfully")

    def _register_events(self):
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            emit('server_message', {'message': 'Connected to WebSocket API', 'timestamp': datetime.utcnow().isoformat()})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")

        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            room = data.get('room')
            if room:
                join_room(room)
                emit('server_message', {'message': f'Subscribed to {room}'}, room=room)
                logger.info(f"Client {request.sid} subscribed to {room}")

        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            room = data.get('room')
            if room:
                leave_room(room)
                emit('server_message', {'message': f'Unsubscribed from {room}'}, room=room)
                logger.info(f"Client {request.sid} unsubscribed from {room}")

        @self.socketio.on('client_message')
        def handle_client_message(data):
            logger.info(f"Received message from {request.sid}: {data}")
            emit('server_message', {'message': 'Message received', 'data': data}, room=request.sid)

    def run(self, host='0.0.0.0', port=5001):
        self.socketio.run(self.app, host=host, port=port, debug=True)

# Example usage and testing
if __name__ == "__main__":
    app = Flask(__name__)
    ws_api = WebSocketAPI(app)
    ws_api.run()