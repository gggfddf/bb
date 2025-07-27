"""
API Authentication Module

This module provides authentication and authorization functionality for the API gateway,
integrating with the security components for user authentication, session management,
and access control.
"""

import os
import time
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from functools import wraps
from flask import request, jsonify, current_app, g
from werkzeug.security import check_password_hash, generate_password_hash

from ..security.auth_manager import AuthManager
from ..security.access_control import AccessControlManager, PermissionLevel
from ..security.audit_logger import AuditLogger


class APIAuth:
    """
    API Authentication handler that integrates with security components
    """
    
    def __init__(self, auth_manager: AuthManager, access_control: AccessControlManager, 
                 audit_logger: AuditLogger):
        self.auth_manager = auth_manager
        self.access_control = access_control
        self.audit_logger = audit_logger
        self.secret_key = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
        self.token_expiry_hours = int(os.getenv('JWT_EXPIRY_HOURS', '24'))
        
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user credentials and return user info if valid
        """
        try:
            # Get user from auth manager
            user = self.auth_manager.get_user_by_username(username)
            if not user:
                self.audit_logger.log_login(
                    username=username,
                    success=False,
                    reason="User not found",
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent')
                )
                return None
                
            # Verify password
            if not check_password_hash(user['password_hash'], password):
                self.audit_logger.log_login(
                    username=username,
                    success=False,
                    reason="Invalid password",
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent')
                )
                return None
                
            # Check if user is active
            if not user.get('is_active', True):
                self.audit_logger.log_login(
                    username=username,
                    success=False,
                    reason="Account disabled",
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent')
                )
                return None
                
            # Log successful login
            self.audit_logger.log_login(
                username=username,
                success=True,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
            
            return {
                'user_id': user['user_id'],
                'username': user['username'],
                'email': user.get('email'),
                'roles': user.get('roles', []),
                'permissions': user.get('permissions', [])
            }
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type="authentication_error",
                action="authenticate_user",
                resource="auth_system",
                details={"error": str(e), "username": username},
                severity="error"
            )
            return None
    
    def generate_token(self, user_data: Dict[str, Any]) -> str:
        """
        Generate JWT token for authenticated user
        """
        try:
            payload = {
                'user_id': user_data['user_id'],
                'username': user_data['username'],
                'roles': user_data.get('roles', []),
                'permissions': user_data.get('permissions', []),
                'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                'iat': datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            
            # Log token generation
            self.audit_logger.log_security_event(
                event_type="token_generated",
                action="generate_token",
                resource="auth_system",
                user_id=user_data['user_id'],
                details={"username": user_data['username']},
                severity="info"
            )
            
            return token
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type="token_generation_error",
                action="generate_token",
                resource="auth_system",
                details={"error": str(e), "user_id": user_data.get('user_id')},
                severity="error"
            )
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token and return user data if valid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if token is expired
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                return None
                
            # Get current user data to ensure it's still valid
            user = self.auth_manager.get_user_by_id(payload['user_id'])
            if not user or not user.get('is_active', True):
                return None
                
            return {
                'user_id': payload['user_id'],
                'username': payload['username'],
                'roles': payload.get('roles', []),
                'permissions': payload.get('permissions', [])
            }
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type="token_verification_error",
                action="verify_token",
                resource="auth_system",
                details={"error": str(e)},
                severity="error"
            )
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh JWT token if it's still valid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if token is close to expiry (within 1 hour)
            expiry_time = datetime.fromtimestamp(payload['exp'])
            if datetime.utcnow() > expiry_time - timedelta(hours=1):
                return None
                
            # Generate new token
            user_data = {
                'user_id': payload['user_id'],
                'username': payload['username'],
                'roles': payload.get('roles', []),
                'permissions': payload.get('permissions', [])
            }
            
            new_token = self.generate_token(user_data)
            
            # Log token refresh
            self.audit_logger.log_security_event(
                event_type="token_refreshed",
                action="refresh_token",
                resource="auth_system",
                user_id=payload['user_id'],
                details={"username": payload['username']},
                severity="info"
            )
            
            return new_token
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type="token_refresh_error",
                action="refresh_token",
                resource="auth_system",
                details={"error": str(e)},
                severity="error"
            )
            return None
    
    def check_permission(self, user_id: str, resource: str, 
                        required_level: PermissionLevel) -> bool:
        """
        Check if user has required permission for resource
        """
        try:
            has_permission = self.access_control.check_permission(
                user_id=user_id,
                resource=resource,
                required_level=required_level
            )
            
            # Log permission check
            self.audit_logger.log_data_access(
                user_id=user_id,
                resource=resource,
                action="permission_check",
                success=has_permission,
                details={"required_level": required_level.value}
            )
            
            return has_permission
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type="permission_check_error",
                action="check_permission",
                resource=resource,
                user_id=user_id,
                details={"error": str(e), "required_level": required_level.value},
                severity="error"
            )
            return False
    
    def logout_user(self, user_id: str, token: str) -> bool:
        """
        Logout user and invalidate token
        """
        try:
            # Add token to blacklist (in production, use Redis or database)
            # For now, we'll just log the logout
            self.audit_logger.log_security_event(
                event_type="user_logout",
                action="logout",
                resource="auth_system",
                user_id=user_id,
                details={"token": token[:20] + "..."},  # Log partial token for security
                severity="info"
            )
            
            return True
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type="logout_error",
                action="logout",
                resource="auth_system",
                user_id=user_id,
                details={"error": str(e)},
                severity="error"
            )
            return False


# Decorators for Flask routes
def require_auth(f):
    """
    Decorator to require authentication for API endpoints
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'Authorization header required'}), 401
        
        try:
            # Extract token from Bearer token
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
            else:
                token = auth_header
                
            # Get API auth instance
            api_auth = get_api_auth()
            user_data = api_auth.verify_token(token)
            
            if not user_data:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            # Store user data in Flask g for use in route
            g.current_user = user_data
            
            return f(*args, **kwargs)
            
        except Exception as e:
            return jsonify({'error': 'Authentication failed'}), 401
    
    return decorated_function


def require_permission(resource: str, permission_level: PermissionLevel):
    """
    Decorator to require specific permission for API endpoints
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # First check authentication
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                return jsonify({'error': 'Authorization header required'}), 401
            
            try:
                # Extract token
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                else:
                    token = auth_header
                    
                # Get API auth instance
                api_auth = get_api_auth()
                user_data = api_auth.verify_token(token)
                
                if not user_data:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                # Check permission
                if not api_auth.check_permission(
                    user_id=user_data['user_id'],
                    resource=resource,
                    required_level=permission_level
                ):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                # Store user data in Flask g
                g.current_user = user_data
                
                return f(*args, **kwargs)
                
            except Exception as e:
                return jsonify({'error': 'Authorization failed'}), 401
        
        return decorated_function
    return decorator


def require_role(required_roles: list):
    """
    Decorator to require specific roles for API endpoints
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # First check authentication
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                return jsonify({'error': 'Authorization header required'}), 401
            
            try:
                # Extract token
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                else:
                    token = auth_header
                    
                # Get API auth instance
                api_auth = get_api_auth()
                user_data = api_auth.verify_token(token)
                
                if not user_data:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                # Check if user has required role
                user_roles = user_data.get('roles', [])
                if not any(role in user_roles for role in required_roles):
                    return jsonify({'error': 'Insufficient role permissions'}), 403
                
                # Store user data in Flask g
                g.current_user = user_data
                
                return f(*args, **kwargs)
                
            except Exception as e:
                return jsonify({'error': 'Authorization failed'}), 401
        
        return decorated_function
    return decorator


# Global instance management
_api_auth_instance = None

def get_api_auth() -> APIAuth:
    """
    Get or create API auth instance
    """
    global _api_auth_instance
    
    if _api_auth_instance is None:
        from ..security.auth_manager import get_auth_manager
        from ..security.access_control import get_access_control_manager
        from ..security.audit_logger import get_audit_logger
        
        auth_manager = get_auth_manager()
        access_control = get_access_control_manager()
        audit_logger = get_audit_logger()
        
        _api_auth_instance = APIAuth(auth_manager, access_control, audit_logger)
    
    return _api_auth_instance


def init_api_auth(auth_manager: AuthManager, access_control: AccessControlManager, 
                  audit_logger: AuditLogger):
    """
    Initialize API auth with custom instances
    """
    global _api_auth_instance
    _api_auth_instance = APIAuth(auth_manager, access_control, audit_logger)