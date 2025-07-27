"""
RESTful API Endpoints for ML Stock Predictor Platform

This module provides comprehensive RESTful API endpoints
for the trading system, including data access, trading operations,
and system management.
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import uuid
import traceback

# Import security components
from integration.security.auth_manager import get_auth_manager
from integration.security.access_control import get_access_control_manager, PermissionLevel, ResourceType
from integration.security.audit_logger import get_audit_logger

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Standard API response structure."""
    success: bool
    data: Optional[Any] = None
    message: str = ""
    error_code: Optional[str] = None
    timestamp: str = ""
    request_id: str = ""


class RESTfulAPI:
    """
    RESTful API implementation for the trading system.
    
    Features:
    - Authentication and authorization
    - Rate limiting and throttling
    - Request/response validation
    - Error handling and logging
    - Audit trail
    """
    
    def __init__(self, app: Flask):
        """
        Initialize the RESTful API.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        self.auth_manager = get_auth_manager()
        self.access_control = get_access_control_manager()
        self.audit_logger = get_audit_logger()
        
        # Initialize rate limiter
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
        
        # Setup CORS
        CORS(self.app)
        
        # Register routes
        self._register_routes()
        
        logger.info("RESTful API initialized successfully")
    
    def _register_routes(self):
        """Register all API routes."""
        
        # Health check endpoint
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return self._create_response(
                success=True,
                data={"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()},
                message="Service is healthy"
            )
        
        # Authentication endpoints
        @self.app.route('/api/auth/login', methods=['POST'])
        @self.limiter.limit("5 per minute")
        def login():
            return self._handle_login()
        
        @self.app.route('/api/auth/logout', methods=['POST'])
        @self.auth_required
        def logout():
            return self._handle_logout()
        
        @self.app.route('/api/auth/refresh', methods=['POST'])
        def refresh_token():
            return self._handle_token_refresh()
        
        # Data endpoints
        @self.app.route('/api/data/stocks', methods=['GET'])
        @self.auth_required
        @self.rate_limit("100 per hour")
        def get_stocks():
            return self._handle_get_stocks()
        
        @self.app.route('/api/data/stock/<symbol>', methods=['GET'])
        @self.auth_required
        @self.rate_limit("200 per hour")
        def get_stock_data(symbol):
            return self._handle_get_stock_data(symbol)
        
        @self.app.route('/api/data/indicators/<symbol>', methods=['GET'])
        @self.auth_required
        @self.rate_limit("100 per hour")
        def get_indicators(symbol):
            return self._handle_get_indicators(symbol)
        
        # Trading endpoints
        @self.app.route('/api/trading/portfolio', methods=['GET'])
        @self.auth_required
        @self.rate_limit("50 per hour")
        def get_portfolio():
            return self._handle_get_portfolio()
        
        @self.app.route('/api/trading/orders', methods=['GET', 'POST'])
        @self.auth_required
        @self.rate_limit("20 per hour")
        def manage_orders():
            if request.method == 'GET':
                return self._handle_get_orders()
            else:
                return self._handle_create_order()
        
        @self.app.route('/api/trading/orders/<order_id>', methods=['GET', 'PUT', 'DELETE'])
        @self.auth_required
        @self.rate_limit("50 per hour")
        def manage_order(order_id):
            if request.method == 'GET':
                return self._handle_get_order(order_id)
            elif request.method == 'PUT':
                return self._handle_update_order(order_id)
            else:
                return self._handle_cancel_order(order_id)
        
        # Analysis endpoints
        @self.app.route('/api/analysis/predictions/<symbol>', methods=['GET'])
        @self.auth_required
        @self.rate_limit("30 per hour")
        def get_predictions(symbol):
            return self._handle_get_predictions(symbol)
        
        @self.app.route('/api/analysis/backtest', methods=['POST'])
        @self.auth_required
        @self.rate_limit("10 per hour")
        def run_backtest():
            return self._handle_backtest()
        
        # System endpoints
        @self.app.route('/api/system/status', methods=['GET'])
        @self.auth_required
        @self.rate_limit("100 per hour")
        def get_system_status():
            return self._handle_get_system_status()
        
        @self.app.route('/api/system/config', methods=['GET', 'PUT'])
        @self.auth_required
        @self.rate_limit("20 per hour")
        def manage_config():
            if request.method == 'GET':
                return self._handle_get_config()
            else:
                return self._handle_update_config()
        
        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return self._create_response(
                success=False,
                message="Endpoint not found",
                error_code="ENDPOINT_NOT_FOUND"
            ), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return self._create_response(
                success=False,
                message="Internal server error",
                error_code="INTERNAL_ERROR"
            ), 500
    
    def auth_required(self, f):
        """Decorator for authentication requirement."""
        def decorated_function(*args, **kwargs):
            try:
                # Get authorization header
                auth_header = request.headers.get('Authorization')
                if not auth_header:
                    return self._create_response(
                        success=False,
                        message="Authorization header required",
                        error_code="AUTH_REQUIRED"
                    ), 401
                
                # Extract token
                if not auth_header.startswith('Bearer '):
                    return self._create_response(
                        success=False,
                        message="Invalid authorization format",
                        error_code="INVALID_AUTH_FORMAT"
                    ), 401
                
                token = auth_header[7:]  # Remove 'Bearer ' prefix
                
                # Validate token
                user_id = self.auth_manager.validate_token(token)
                if not user_id:
                    return self._create_response(
                        success=False,
                        message="Invalid or expired token",
                        error_code="INVALID_TOKEN"
                    ), 401
                
                # Add user_id to request context
                request.user_id = user_id
                
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                return self._create_response(
                    success=False,
                    message="Authentication failed",
                    error_code="AUTH_FAILED"
                ), 401
        
        return decorated_function
    
    def rate_limit(self, limit_string):
        """Decorator for rate limiting."""
        def decorator(f):
            return self.limiter.limit(limit_string)(f)
        return decorator
    
    def _create_response(self, success: bool, data: Optional[Any] = None,
                        message: str = "", error_code: Optional[str] = None) -> Response:
        """Create a standardized API response."""
        response = APIResponse(
            success=success,
            data=data,
            message=message,
            error_code=error_code,
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=str(uuid.uuid4())
        )
        
        return jsonify(asdict(response))
    
    def _log_api_access(self, user_id: Optional[str], endpoint: str, method: str,
                       status_code: int, response_time: float):
        """Log API access for audit purposes."""
        try:
            self.audit_logger.log_api_access(
                user_id=user_id,
                endpoint=endpoint,
                method=method,
                ip_address=request.remote_addr,
                status_code=status_code,
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"Failed to log API access: {e}")
    
    def _handle_login(self):
        """Handle user login."""
        start_time = time.time()
        
        try:
            data = request.get_json()
            if not data:
                return self._create_response(
                    success=False,
                    message="Invalid request data",
                    error_code="INVALID_REQUEST"
                ), 400
            
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                return self._create_response(
                    success=False,
                    message="Username and password required",
                    error_code="MISSING_CREDENTIALS"
                ), 400
            
            # Authenticate user
            auth_result = self.auth_manager.authenticate_user(username, password)
            
            if auth_result['success']:
                response_data = {
                    'token': auth_result['token'],
                    'user_id': auth_result['user_id'],
                    'expires_at': auth_result['expires_at']
                }
                
                response = self._create_response(
                    success=True,
                    data=response_data,
                    message="Login successful"
                )
                
                # Log successful login
                self.audit_logger.log_login(
                    user_id=auth_result['user_id'],
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent', ''),
                    success=True
                )
                
            else:
                response = self._create_response(
                    success=False,
                    message="Invalid credentials",
                    error_code="INVALID_CREDENTIALS"
                )
                
                # Log failed login
                self.audit_logger.log_login(
                    user_id=username,
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent', ''),
                    success=False
                )
            
            response_time = time.time() - start_time
            self._log_api_access(None, '/api/auth/login', 'POST', 
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return self._create_response(
                success=False,
                message="Login failed",
                error_code="LOGIN_FAILED"
            ), 500
    
    def _handle_logout(self):
        """Handle user logout."""
        start_time = time.time()
        
        try:
            user_id = getattr(request, 'user_id', None)
            
            # Invalidate token
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header[7:]
                self.auth_manager.invalidate_token(token)
            
            response = self._create_response(
                success=True,
                message="Logout successful"
            )
            
            # Log logout
            if user_id:
                self.audit_logger.log_logout(
                    user_id=user_id,
                    session_id=token if 'token' in locals() else None,
                    ip_address=request.remote_addr
                )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, '/api/auth/logout', 'POST',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return self._create_response(
                success=False,
                message="Logout failed",
                error_code="LOGOUT_FAILED"
            ), 500
    
    def _handle_token_refresh(self):
        """Handle token refresh."""
        start_time = time.time()
        
        try:
            data = request.get_json()
            if not data or 'refresh_token' not in data:
                return self._create_response(
                    success=False,
                    message="Refresh token required",
                    error_code="MISSING_REFRESH_TOKEN"
                ), 400
            
            refresh_token = data['refresh_token']
            refresh_result = self.auth_manager.refresh_token(refresh_token)
            
            if refresh_result['success']:
                response_data = {
                    'token': refresh_result['token'],
                    'expires_at': refresh_result['expires_at']
                }
                
                response = self._create_response(
                    success=True,
                    data=response_data,
                    message="Token refreshed successfully"
                )
            else:
                response = self._create_response(
                    success=False,
                    message="Invalid refresh token",
                    error_code="INVALID_REFRESH_TOKEN"
                )
            
            response_time = time.time() - start_time
            self._log_api_access(None, '/api/auth/refresh', 'POST',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return self._create_response(
                success=False,
                message="Token refresh failed",
                error_code="REFRESH_FAILED"
            ), 500
    
    def _handle_get_stocks(self):
        """Handle getting available stocks."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, "data.stocks", 
                                                      PermissionLevel.READ, ResourceType.DATA):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Mock data - replace with actual data retrieval
            stocks_data = [
                {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
                {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive"},
                {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"}
            ]
            
            response = self._create_response(
                success=True,
                data=stocks_data,
                message="Stocks retrieved successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, '/api/data/stocks', 'GET',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Get stocks error: {e}")
            return self._create_response(
                success=False,
                message="Failed to retrieve stocks",
                error_code="STOCKS_RETRIEVAL_FAILED"
            ), 500
    
    def _handle_get_stock_data(self, symbol):
        """Handle getting stock data."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, f"data.stock.{symbol}", 
                                                      PermissionLevel.READ, ResourceType.DATA):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Get query parameters
            timeframe = request.args.get('timeframe', '1d')
            limit = int(request.args.get('limit', 100))
            
            # Mock data - replace with actual data retrieval
            stock_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": [
                    {
                        "timestamp": "2024-01-01T09:30:00Z",
                        "open": 150.0,
                        "high": 152.0,
                        "low": 149.0,
                        "close": 151.0,
                        "volume": 1000000
                    }
                    # Add more data points as needed
                ]
            }
            
            response = self._create_response(
                success=True,
                data=stock_data,
                message=f"Stock data for {symbol} retrieved successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, f'/api/data/stock/{symbol}', 'GET',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Get stock data error: {e}")
            return self._create_response(
                success=False,
                message=f"Failed to retrieve data for {symbol}",
                error_code="STOCK_DATA_RETRIEVAL_FAILED"
            ), 500
    
    def _handle_get_indicators(self, symbol):
        """Handle getting technical indicators."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, f"data.indicators.{symbol}", 
                                                      PermissionLevel.READ, ResourceType.DATA):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Mock data - replace with actual indicator calculation
            indicators_data = {
                "symbol": symbol,
                "indicators": {
                    "rsi": 65.5,
                    "macd": {"macd": 0.5, "signal": 0.3, "histogram": 0.2},
                    "bollinger_bands": {"upper": 155.0, "middle": 150.0, "lower": 145.0},
                    "sma_20": 148.5,
                    "ema_12": 149.2
                }
            }
            
            response = self._create_response(
                success=True,
                data=indicators_data,
                message=f"Indicators for {symbol} retrieved successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, f'/api/data/indicators/{symbol}', 'GET',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Get indicators error: {e}")
            return self._create_response(
                success=False,
                message=f"Failed to retrieve indicators for {symbol}",
                error_code="INDICATORS_RETRIEVAL_FAILED"
            ), 500
    
    def _handle_get_portfolio(self):
        """Handle getting user portfolio."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, "trading.portfolio", 
                                                      PermissionLevel.READ, ResourceType.TRADING):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Mock data - replace with actual portfolio retrieval
            portfolio_data = {
                "user_id": user_id,
                "total_value": 50000.0,
                "cash": 10000.0,
                "positions": [
                    {
                        "symbol": "AAPL",
                        "quantity": 100,
                        "avg_price": 150.0,
                        "current_price": 155.0,
                        "market_value": 15500.0,
                        "unrealized_pnl": 500.0
                    }
                ]
            }
            
            response = self._create_response(
                success=True,
                data=portfolio_data,
                message="Portfolio retrieved successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, '/api/trading/portfolio', 'GET',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Get portfolio error: {e}")
            return self._create_response(
                success=False,
                message="Failed to retrieve portfolio",
                error_code="PORTFOLIO_RETRIEVAL_FAILED"
            ), 500
    
    def _handle_get_orders(self):
        """Handle getting user orders."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, "trading.orders", 
                                                      PermissionLevel.READ, ResourceType.TRADING):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Mock data - replace with actual order retrieval
            orders_data = [
                {
                    "order_id": "ORD001",
                    "symbol": "AAPL",
                    "side": "BUY",
                    "quantity": 50,
                    "price": 150.0,
                    "status": "FILLED",
                    "created_at": "2024-01-01T10:00:00Z"
                }
            ]
            
            response = self._create_response(
                success=True,
                data=orders_data,
                message="Orders retrieved successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, '/api/trading/orders', 'GET',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Get orders error: {e}")
            return self._create_response(
                success=False,
                message="Failed to retrieve orders",
                error_code="ORDERS_RETRIEVAL_FAILED"
            ), 500
    
    def _handle_create_order(self):
        """Handle creating a new order."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, "trading.orders", 
                                                      PermissionLevel.WRITE, ResourceType.TRADING):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            data = request.get_json()
            if not data:
                return self._create_response(
                    success=False,
                    message="Invalid order data",
                    error_code="INVALID_ORDER_DATA"
                ), 400
            
            # Validate required fields
            required_fields = ['symbol', 'side', 'quantity']
            for field in required_fields:
                if field not in data:
                    return self._create_response(
                        success=False,
                        message=f"Missing required field: {field}",
                        error_code="MISSING_FIELD"
                    ), 400
            
            # Mock order creation - replace with actual order processing
            order_data = {
                "order_id": f"ORD{int(time.time())}",
                "user_id": user_id,
                "symbol": data['symbol'],
                "side": data['side'],
                "quantity": data['quantity'],
                "price": data.get('price'),
                "status": "PENDING",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            response = self._create_response(
                success=True,
                data=order_data,
                message="Order created successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, '/api/trading/orders', 'POST',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Create order error: {e}")
            return self._create_response(
                success=False,
                message="Failed to create order",
                error_code="ORDER_CREATION_FAILED"
            ), 500
    
    def _handle_get_order(self, order_id):
        """Handle getting a specific order."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, f"trading.order.{order_id}", 
                                                      PermissionLevel.READ, ResourceType.TRADING):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Mock data - replace with actual order retrieval
            order_data = {
                "order_id": order_id,
                "user_id": user_id,
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 50,
                "price": 150.0,
                "status": "FILLED",
                "created_at": "2024-01-01T10:00:00Z",
                "filled_at": "2024-01-01T10:05:00Z"
            }
            
            response = self._create_response(
                success=True,
                data=order_data,
                message=f"Order {order_id} retrieved successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, f'/api/trading/orders/{order_id}', 'GET',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Get order error: {e}")
            return self._create_response(
                success=False,
                message=f"Failed to retrieve order {order_id}",
                error_code="ORDER_RETRIEVAL_FAILED"
            ), 500
    
    def _handle_update_order(self, order_id):
        """Handle updating an order."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, f"trading.order.{order_id}", 
                                                      PermissionLevel.WRITE, ResourceType.TRADING):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            data = request.get_json()
            if not data:
                return self._create_response(
                    success=False,
                    message="Invalid update data",
                    error_code="INVALID_UPDATE_DATA"
                ), 400
            
            # Mock order update - replace with actual order processing
            order_data = {
                "order_id": order_id,
                "user_id": user_id,
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": data.get('quantity', 50),
                "price": data.get('price', 150.0),
                "status": "UPDATED",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            response = self._create_response(
                success=True,
                data=order_data,
                message=f"Order {order_id} updated successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, f'/api/trading/orders/{order_id}', 'PUT',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Update order error: {e}")
            return self._create_response(
                success=False,
                message=f"Failed to update order {order_id}",
                error_code="ORDER_UPDATE_FAILED"
            ), 500
    
    def _handle_cancel_order(self, order_id):
        """Handle canceling an order."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, f"trading.order.{order_id}", 
                                                      PermissionLevel.WRITE, ResourceType.TRADING):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Mock order cancellation - replace with actual order processing
            order_data = {
                "order_id": order_id,
                "user_id": user_id,
                "status": "CANCELLED",
                "cancelled_at": datetime.now(timezone.utc).isoformat()
            }
            
            response = self._create_response(
                success=True,
                data=order_data,
                message=f"Order {order_id} cancelled successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, f'/api/trading/orders/{order_id}', 'DELETE',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return self._create_response(
                success=False,
                message=f"Failed to cancel order {order_id}",
                error_code="ORDER_CANCELLATION_FAILED"
            ), 500
    
    def _handle_get_predictions(self, symbol):
        """Handle getting ML predictions."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, f"analysis.predictions.{symbol}", 
                                                      PermissionLevel.READ, ResourceType.DATA):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Mock predictions - replace with actual ML model predictions
            predictions_data = {
                "symbol": symbol,
                "predictions": [
                    {
                        "timestamp": "2024-01-02T09:30:00Z",
                        "predicted_price": 156.0,
                        "confidence": 0.85,
                        "model": "LSTM"
                    }
                ]
            }
            
            response = self._create_response(
                success=True,
                data=predictions_data,
                message=f"Predictions for {symbol} retrieved successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, f'/api/analysis/predictions/{symbol}', 'GET',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Get predictions error: {e}")
            return self._create_response(
                success=False,
                message=f"Failed to retrieve predictions for {symbol}",
                error_code="PREDICTIONS_RETRIEVAL_FAILED"
            ), 500
    
    def _handle_backtest(self):
        """Handle running backtest."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, "analysis.backtest", 
                                                      PermissionLevel.WRITE, ResourceType.DATA):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            data = request.get_json()
            if not data:
                return self._create_response(
                    success=False,
                    message="Invalid backtest parameters",
                    error_code="INVALID_BACKTEST_PARAMS"
                ), 400
            
            # Mock backtest - replace with actual backtesting
            backtest_data = {
                "backtest_id": f"BT{int(time.time())}",
                "symbol": data.get('symbol', 'AAPL'),
                "strategy": data.get('strategy', 'SMA_CROSSOVER'),
                "start_date": data.get('start_date'),
                "end_date": data.get('end_date'),
                "results": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.05,
                    "win_rate": 0.65
                }
            }
            
            response = self._create_response(
                success=True,
                data=backtest_data,
                message="Backtest completed successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, '/api/analysis/backtest', 'POST',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return self._create_response(
                success=False,
                message="Failed to run backtest",
                error_code="BACKTEST_FAILED"
            ), 500
    
    def _handle_get_system_status(self):
        """Handle getting system status."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, "system.status", 
                                                      PermissionLevel.READ, ResourceType.SYSTEM):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Mock system status - replace with actual system monitoring
            status_data = {
                "status": "healthy",
                "uptime": "24h 30m 15s",
                "version": "1.0.0",
                "services": {
                    "data_ingestion": "running",
                    "ml_models": "running",
                    "trading_engine": "running",
                    "api_gateway": "running"
                },
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            response = self._create_response(
                success=True,
                data=status_data,
                message="System status retrieved successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, '/api/system/status', 'GET',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Get system status error: {e}")
            return self._create_response(
                success=False,
                message="Failed to retrieve system status",
                error_code="SYSTEM_STATUS_RETRIEVAL_FAILED"
            ), 500
    
    def _handle_get_config(self):
        """Handle getting system configuration."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, "system.config", 
                                                      PermissionLevel.READ, ResourceType.SYSTEM):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            # Mock config - replace with actual configuration retrieval
            config_data = {
                "trading": {
                    "max_position_size": 10000,
                    "risk_per_trade": 0.02,
                    "default_timeframe": "1h"
                },
                "ml_models": {
                    "prediction_horizon": 24,
                    "confidence_threshold": 0.7,
                    "retrain_frequency": "daily"
                },
                "api": {
                    "rate_limit": 100,
                    "timeout": 30,
                    "max_requests_per_minute": 60
                }
            }
            
            response = self._create_response(
                success=True,
                data=config_data,
                message="Configuration retrieved successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, '/api/system/config', 'GET',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Get config error: {e}")
            return self._create_response(
                success=False,
                message="Failed to retrieve configuration",
                error_code="CONFIG_RETRIEVAL_FAILED"
            ), 500
    
    def _handle_update_config(self):
        """Handle updating system configuration."""
        start_time = time.time()
        user_id = getattr(request, 'user_id', None)
        
        try:
            # Check permission
            if not self.access_control.check_permission(user_id, "system.config", 
                                                      PermissionLevel.WRITE, ResourceType.SYSTEM):
                return self._create_response(
                    success=False,
                    message="Insufficient permissions",
                    error_code="INSUFFICIENT_PERMISSIONS"
                ), 403
            
            data = request.get_json()
            if not data:
                return self._create_response(
                    success=False,
                    message="Invalid configuration data",
                    error_code="INVALID_CONFIG_DATA"
                ), 400
            
            # Mock config update - replace with actual configuration management
            config_data = {
                "updated_by": user_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "changes": data
            }
            
            response = self._create_response(
                success=True,
                data=config_data,
                message="Configuration updated successfully"
            )
            
            response_time = time.time() - start_time
            self._log_api_access(user_id, '/api/system/config', 'PUT',
                               response.status_code, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Update config error: {e}")
            return self._create_response(
                success=False,
                message="Failed to update configuration",
                error_code="CONFIG_UPDATE_FAILED"
            ), 500


# Example usage and testing
if __name__ == "__main__":
    # Initialize Flask app
    app = Flask(__name__)
    
    # Initialize RESTful API
    api = RESTfulAPI(app)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)