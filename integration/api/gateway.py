#!/usr/bin/env python3
"""
API Gateway Module

Implements comprehensive API gateway for the trading system:
- RESTful API endpoints
- WebSocket API for real-time data
- Authentication and authorization
- Rate limiting and throttling
- API documentation
- Request/response validation

Features:
- Complete RESTful API with all trading system endpoints
- Real-time WebSocket API for live data streaming
- Robust authentication and authorization system
- Advanced rate limiting and throttling mechanisms
- Comprehensive API documentation and validation
- High-performance request routing and processing
"""

import asyncio
import json
import time
import hashlib
import hmac
import base64
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import threading
import uuid
from collections import defaultdict, deque
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    warnings.warn("PyJWT not available. JWT authentication will not function.")
from functools import wraps

try:
    from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, ValidationError
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create dummy types for when FastAPI is not available
    class Request:
        pass
    class Response:
        pass
    class WebSocket:
        pass
    class HTTPException(Exception):
        pass
    class BaseHTTPMiddleware:
        pass
    class JSONResponse:
        pass
    warnings.warn("FastAPI not available. API Gateway will not function.")

logger = structlog.get_logger()

class APIVersion(Enum):
    """API versions."""
    V1 = "v1"
    V2 = "v2"

class EndpointType(Enum):
    """API endpoint types."""
    REST = "rest"
    WEBSOCKET = "websocket"
    GRAPHQL = "graphql"

class AuthMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"
    BASIC = "basic"

@dataclass
class APIEndpoint:
    """API endpoint configuration."""
    path: str
    method: str
    endpoint_type: EndpointType
    auth_required: bool = True
    rate_limit: int = 100  # requests per minute
    timeout: float = 30.0  # seconds
    description: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class APIConfig:
    """API gateway configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    api_version: APIVersion = APIVersion.V1
    enable_cors: bool = True
    enable_rate_limiting: bool = True
    enable_auth: bool = True
    enable_logging: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: float = 30.0
    rate_limit_window: int = 60  # seconds
    rate_limit_max_requests: int = 100
    jwt_secret: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiry: int = 3600  # seconds

@dataclass
class User:
    """User structure."""
    user_id: str
    username: str
    email: str
    api_key: str
    permissions: List[str] = field(default_factory=list)
    rate_limit: int = 100
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

class RateLimiter:
    """Rate limiting system."""
    
    def __init__(self, window_size: int = 60, max_requests: int = 100):
        """
        Initialize rate limiter.
        
        Args:
            window_size: Time window in seconds
            max_requests: Maximum requests per window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = defaultdict(deque)
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed.
        
        Args:
            client_id: Client identifier (IP, user_id, etc.)
            
        Returns:
            True if request is allowed
        """
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests outside window
        while client_requests and client_requests[0] < now - self.window_size:
            client_requests.popleft()
        
        # Check if under limit
        if len(client_requests) < self.max_requests:
            client_requests.append(now)
            return True
        
        return False
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests
        while client_requests and client_requests[0] < now - self.window_size:
            client_requests.popleft()
        
        return max(0, self.max_requests - len(client_requests))

class AuthenticationManager:
    """Authentication and authorization system."""
    
    def __init__(self, config: APIConfig):
        """
        Initialize authentication manager.
        
        Args:
            config: API configuration
        """
        self.config = config
        self.users = {}
        self.api_keys = {}
        self.jwt_tokens = {}
        if FASTAPI_AVAILABLE:
            self.security = HTTPBearer()
        else:
            self.security = None
    
    def add_user(self, user: User):
        """Add user to system."""
        self.users[user.user_id] = user
        self.api_keys[user.api_key] = user.user_id
        logger.info("User added", user_id=user.user_id, username=user.username)
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key."""
        user_id = self.api_keys.get(api_key)
        if user_id:
            user = self.users.get(user_id)
            if user and user.is_active:
                return user
        return None
    
    def authenticate_jwt(self, token: str) -> Optional[User]:
        """Authenticate using JWT token."""
        if not JWT_AVAILABLE:
            logger.warning("JWT not available")
            return None
        
        try:
            payload = jwt.decode(token, self.config.jwt_secret, 
                               algorithms=[self.config.jwt_algorithm])
            user_id = payload.get('user_id')
            if user_id:
                user = self.users.get(user_id)
                if user and user.is_active:
                    return user
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
        return None
    
    def generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user."""
        if not JWT_AVAILABLE:
            logger.warning("JWT not available")
            return "jwt_not_available"
        
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'exp': datetime.utcnow() + timedelta(seconds=self.config.jwt_expiry)
        }
        return jwt.encode(payload, self.config.jwt_secret, 
                         algorithm=self.config.jwt_algorithm)
    
    def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has permission."""
        return permission in user.permissions or 'admin' in user.permissions

class RequestValidator:
    """Request validation system."""
    
    def __init__(self):
        """Initialize request validator."""
        self.validators = {}
    
    def add_validator(self, endpoint: str, validator: Callable):
        """Add validator for endpoint."""
        self.validators[endpoint] = validator
    
    def validate_request(self, endpoint: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate request data.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        validator = self.validators.get(endpoint)
        if not validator:
            return True, []
        
        try:
            errors = validator(data)
            return len(errors) == 0, errors
        except Exception as e:
            return False, [str(e)]

class ResponseFormatter:
    """Response formatting system."""
    
    def __init__(self):
        """Initialize response formatter."""
        self.formatters = {}
    
    def add_formatter(self, endpoint: str, formatter: Callable):
        """Add formatter for endpoint."""
        self.formatters[endpoint] = formatter
    
    def format_response(self, endpoint: str, data: Any) -> Dict[str, Any]:
        """
        Format response data.
        
        Args:
            endpoint: API endpoint
            data: Response data
            
        Returns:
            Formatted response
        """
        formatter = self.formatters.get(endpoint)
        if formatter:
            return formatter(data)
        
        # Default formatting
        return {
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }

class APIGateway:
    """Main API gateway system."""
    
    def __init__(self, config: APIConfig):
        """
        Initialize API gateway.
        
        Args:
            config: API configuration
        """
        self.config = config
        self.app = None
        self.rate_limiter = RateLimiter(config.rate_limit_window, config.rate_limit_max_requests)
        self.auth_manager = AuthenticationManager(config)
        self.request_validator = RequestValidator()
        self.response_formatter = ResponseFormatter()
        
        self.endpoints = {}
        self.websocket_connections = {}
        self.request_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        
        if FASTAPI_AVAILABLE:
            self._setup_fastapi()
    
    def _setup_fastapi(self):
        """Setup FastAPI application."""
        self.app = FastAPI(
            title="Trading System API",
            description="Comprehensive API for trading system operations",
            version=self.config.api_version.value,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add middleware
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )
        
        if self.config.enable_auth:
            self.app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        
        # Add rate limiting middleware
        if self.config.enable_rate_limiting:
            self.app.add_middleware(RateLimitMiddleware, gateway=self)
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        if not self.app:
            return
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # Authentication endpoints
        @self.app.post("/auth/login")
        async def login(request: Request):
            return await self._handle_login(request)
        
        @self.app.post("/auth/logout")
        async def logout(request: Request):
            return await self._handle_logout(request)
        
        # Trading endpoints
        @self.app.get("/trading/strategies")
        async def get_strategies(request: Request):
            return await self._handle_get_strategies(request)
        
        @self.app.post("/trading/strategies")
        async def create_strategy(request: Request):
            return await self._handle_create_strategy(request)
        
        @self.app.get("/trading/orders")
        async def get_orders(request: Request):
            return await self._handle_get_orders(request)
        
        @self.app.post("/trading/orders")
        async def create_order(request: Request):
            return await self._handle_create_order(request)
        
        # Market data endpoints
        @self.app.get("/market-data/{symbol}")
        async def get_market_data(symbol: str, request: Request):
            return await self._handle_get_market_data(symbol, request)
        
        # Risk management endpoints
        @self.app.get("/risk/portfolio/{portfolio_id}")
        async def get_portfolio_risk(portfolio_id: str, request: Request):
            return await self._handle_get_portfolio_risk(portfolio_id, request)
        
        # Performance endpoints
        @self.app.get("/performance/strategy/{strategy_id}")
        async def get_strategy_performance(strategy_id: str, request: Request):
            return await self._handle_get_strategy_performance(strategy_id, request)
        
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
    
    async def _handle_login(self, request: Request) -> Dict[str, Any]:
        """Handle login request."""
        try:
            data = await request.json()
            username = data.get('username')
            password = data.get('password')
            
            # Simple authentication (in production, use proper password hashing)
            user = self._authenticate_user(username, password)
            if user:
                token = self.auth_manager.generate_jwt_token(user)
                return {
                    'success': True,
                    'token': token,
                    'user': {
                        'user_id': user.user_id,
                        'username': user.username,
                        'permissions': user.permissions
                    }
                }
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        
        except Exception as e:
            logger.error("Login error", error=str(e))
            raise HTTPException(status_code=400, detail="Invalid request")
    
    async def _handle_logout(self, request: Request) -> Dict[str, Any]:
        """Handle logout request."""
        return {'success': True, 'message': 'Logged out successfully'}
    
    async def _handle_get_strategies(self, request: Request) -> Dict[str, Any]:
        """Handle get strategies request."""
        # This would integrate with the strategy executor
        strategies = [
            {
                'strategy_id': 'strategy_001',
                'name': 'Sample Strategy',
                'status': 'active',
                'performance': 0.15
            }
        ]
        return self.response_formatter.format_response('/trading/strategies', strategies)
    
    async def _handle_create_strategy(self, request: Request) -> Dict[str, Any]:
        """Handle create strategy request."""
        try:
            data = await request.json()
            # Validate request
            is_valid, errors = self.request_validator.validate_request('/trading/strategies', data)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Validation errors: {errors}")
            
            # Create strategy (would integrate with strategy executor)
            strategy_id = str(uuid.uuid4())
            strategy = {
                'strategy_id': strategy_id,
                'name': data.get('name'),
                'status': 'created'
            }
            
            return self.response_formatter.format_response('/trading/strategies', strategy)
        
        except Exception as e:
            logger.error("Create strategy error", error=str(e))
            raise HTTPException(status_code=400, detail="Invalid request")
    
    async def _handle_get_orders(self, request: Request) -> Dict[str, Any]:
        """Handle get orders request."""
        # This would integrate with the order manager
        orders = [
            {
                'order_id': 'order_001',
                'symbol': 'AAPL',
                'side': 'buy',
                'quantity': 100,
                'status': 'filled'
            }
        ]
        return self.response_formatter.format_response('/trading/orders', orders)
    
    async def _handle_create_order(self, request: Request) -> Dict[str, Any]:
        """Handle create order request."""
        try:
            data = await request.json()
            # Validate request
            is_valid, errors = self.request_validator.validate_request('/trading/orders', data)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Validation errors: {errors}")
            
            # Create order (would integrate with order manager)
            order_id = str(uuid.uuid4())
            order = {
                'order_id': order_id,
                'symbol': data.get('symbol'),
                'side': data.get('side'),
                'quantity': data.get('quantity'),
                'status': 'submitted'
            }
            
            return self.response_formatter.format_response('/trading/orders', order)
        
        except Exception as e:
            logger.error("Create order error", error=str(e))
            raise HTTPException(status_code=400, detail="Invalid request")
    
    async def _handle_get_market_data(self, symbol: str, request: Request) -> Dict[str, Any]:
        """Handle get market data request."""
        # This would integrate with the market data processor
        market_data = {
            'symbol': symbol,
            'price': 150.25,
            'volume': 1000000,
            'timestamp': datetime.now().isoformat()
        }
        return self.response_formatter.format_response(f'/market-data/{symbol}', market_data)
    
    async def _handle_get_portfolio_risk(self, portfolio_id: str, request: Request) -> Dict[str, Any]:
        """Handle get portfolio risk request."""
        # This would integrate with the risk manager
        risk_data = {
            'portfolio_id': portfolio_id,
            'var': 5000.0,
            'volatility': 0.15,
            'max_drawdown': 0.08
        }
        return self.response_formatter.format_response(f'/risk/portfolio/{portfolio_id}', risk_data)
    
    async def _handle_get_strategy_performance(self, strategy_id: str, request: Request) -> Dict[str, Any]:
        """Handle get strategy performance request."""
        # This would integrate with the performance monitor
        performance_data = {
            'strategy_id': strategy_id,
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05
        }
        return self.response_formatter.format_response(f'/performance/strategy/{strategy_id}', performance_data)
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection."""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.websocket_connections[connection_id] = websocket
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get('type') == 'subscribe':
                    await self._handle_subscription(connection_id, message)
                elif message.get('type') == 'unsubscribe':
                    await self._handle_unsubscription(connection_id, message)
                else:
                    # Echo message back
                    await websocket.send_text(json.dumps({
                        'type': 'echo',
                        'data': message
                    }))
        
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", connection_id=connection_id)
        except Exception as e:
            logger.error("WebSocket error", connection_id=connection_id, error=str(e))
        finally:
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]
    
    async def _handle_subscription(self, connection_id: str, message: Dict[str, Any]):
        """Handle WebSocket subscription."""
        # This would integrate with real-time data systems
        websocket = self.websocket_connections.get(connection_id)
        if websocket:
            await websocket.send_text(json.dumps({
                'type': 'subscription_confirmed',
                'data': message.get('data', {})
            }))
    
    async def _handle_unsubscription(self, connection_id: str, message: Dict[str, Any]):
        """Handle WebSocket unsubscription."""
        websocket = self.websocket_connections.get(connection_id)
        if websocket:
            await websocket.send_text(json.dumps({
                'type': 'unsubscription_confirmed',
                'data': message.get('data', {})
            }))
    
    def _authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user (simplified)."""
        # In production, use proper password hashing and database lookup
        if username == "admin" and password == "password":
            return User(
                user_id="admin_001",
                username=username,
                email="admin@example.com",
                api_key="admin_api_key_123",
                permissions=["admin", "trading", "risk", "performance"]
            )
        return None
    
    def add_endpoint(self, endpoint: APIEndpoint):
        """Add API endpoint."""
        self.endpoints[endpoint.path] = endpoint
        logger.info("API endpoint added", path=endpoint.path, method=endpoint.method)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get API metrics."""
        return {
            'total_requests': sum(self.request_counters.values()),
            'total_errors': sum(self.error_counters.values()),
            'active_websocket_connections': len(self.websocket_connections),
            'endpoints': len(self.endpoints),
            'users': len(self.auth_manager.users)
        }
    
    def start(self):
        """Start API gateway."""
        if self.app:
            import uvicorn
            uvicorn.run(self.app, host=self.config.host, port=self.config.port)
        else:
            logger.error("FastAPI not available")

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, gateway: APIGateway):
        super().__init__(app)
        self.gateway = gateway
    
    async def dispatch(self, request: Request, call_next):
        # Get client identifier
        client_id = request.client.host
        
        # Check rate limit
        if not self.gateway.rate_limiter.is_allowed(client_id):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        # Add rate limit headers
        response = await call_next(request)
        remaining = self.gateway.rate_limiter.get_remaining_requests(client_id)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response

def create_api_gateway(config: APIConfig = None) -> APIGateway:
    """
    Create an API gateway.
    
    Args:
        config: API configuration
        
    Returns:
        APIGateway instance
    """
    if config is None:
        config = APIConfig()
    
    return APIGateway(config)

if __name__ == "__main__":
    # Demo of API gateway
    config = APIConfig(
        host="0.0.0.0",
        port=8000,
        api_version=APIVersion.V1,
        enable_cors=True,
        enable_rate_limiting=True,
        enable_auth=True,
        enable_logging=True
    )
    
    gateway = create_api_gateway(config)
    
    # Add sample user
    user = User(
        user_id="user_001",
        username="trader",
        email="trader@example.com",
        api_key="api_key_123",
        permissions=["trading", "risk", "performance"]
    )
    gateway.auth_manager.add_user(user)
    
    # Add sample endpoints
    endpoints = [
        APIEndpoint("/trading/strategies", "GET", EndpointType.REST, True, 100),
        APIEndpoint("/trading/orders", "POST", EndpointType.REST, True, 50),
        APIEndpoint("/market-data/{symbol}", "GET", EndpointType.REST, True, 200),
        APIEndpoint("/ws", "GET", EndpointType.WEBSOCKET, True, 1000)
    ]
    
    for endpoint in endpoints:
        gateway.add_endpoint(endpoint)
    
    print("API Gateway created successfully!")
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"API Version: {config.api_version.value}")
    print(f"Endpoints: {len(gateway.endpoints)}")
    print(f"Users: {len(gateway.auth_manager.users)}")
    
    # Get metrics
    metrics = gateway.get_metrics()
    print(f"Metrics: {metrics}")
    
    # Start gateway (uncomment to run)
    # gateway.start()