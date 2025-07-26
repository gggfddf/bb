"""
Access Control System for ML Stock Predictor Platform

This module provides comprehensive access control functionality
for managing user permissions and resource access in the trading system.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Set, Union
from dataclasses import dataclass, asdict
import hashlib
import uuid
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for access control."""
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3
    SUPER_ADMIN = 4


class ResourceType(Enum):
    """Types of resources that can be protected."""
    SYSTEM = "system"
    DATA = "data"
    API = "api"
    TRADING = "trading"
    CONFIGURATION = "configuration"
    USER_MANAGEMENT = "user_management"
    REPORTS = "reports"
    SECURITY = "security"


@dataclass
class Permission:
    """Permission definition."""
    resource: str
    resource_type: ResourceType
    permission_level: PermissionLevel
    granted_at: str
    granted_by: str
    expires_at: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None


@dataclass
class Role:
    """Role definition."""
    role_id: str
    name: str
    description: str
    permissions: List[Permission]
    created_at: str
    created_by: str
    is_active: bool = True


@dataclass
class UserAccess:
    """User access information."""
    user_id: str
    roles: List[str]
    permissions: List[Permission]
    last_access: str
    is_active: bool = True


class AccessControlManager:
    """
    Comprehensive access control manager for the trading system.
    
    Features:
    - Role-based access control (RBAC)
    - Permission-based access control (PBAC)
    - Resource-level permissions
    - Time-based access control
    - Condition-based access control
    - Access auditing and monitoring
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the access control manager.
        
        Args:
            config_file: Optional configuration file path
        """
        self.roles: Dict[str, Role] = {}
        self.user_access: Dict[str, UserAccess] = {}
        self.resource_permissions: Dict[str, Dict[str, PermissionLevel]] = {}
        self.access_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.lock = threading.RLock()
        
        self._load_default_roles()
        if config_file:
            self._load_config(config_file)
        
        # Start cache cleanup thread
        self._start_cache_cleanup()
        
        logger.info("Access control manager initialized")
    
    def _load_default_roles(self):
        """Load default roles and permissions."""
        try:
            # Super Admin role
            super_admin_permissions = [
                Permission(
                    resource="*",
                    resource_type=ResourceType.SYSTEM,
                    permission_level=PermissionLevel.SUPER_ADMIN,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                )
            ]
            
            self.roles["super_admin"] = Role(
                role_id="super_admin",
                name="Super Administrator",
                description="Full system access with all permissions",
                permissions=super_admin_permissions,
                created_at=datetime.now(timezone.utc).isoformat(),
                created_by="system"
            )
            
            # Admin role
            admin_permissions = [
                Permission(
                    resource="system.*",
                    resource_type=ResourceType.SYSTEM,
                    permission_level=PermissionLevel.ADMIN,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                ),
                Permission(
                    resource="user_management.*",
                    resource_type=ResourceType.USER_MANAGEMENT,
                    permission_level=PermissionLevel.ADMIN,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                ),
                Permission(
                    resource="reports.*",
                    resource_type=ResourceType.REPORTS,
                    permission_level=PermissionLevel.ADMIN,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                )
            ]
            
            self.roles["admin"] = Role(
                role_id="admin",
                name="Administrator",
                description="System administration with user and report management",
                permissions=admin_permissions,
                created_at=datetime.now(timezone.utc).isoformat(),
                created_by="system"
            )
            
            # Trader role
            trader_permissions = [
                Permission(
                    resource="trading.*",
                    resource_type=ResourceType.TRADING,
                    permission_level=PermissionLevel.WRITE,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                ),
                Permission(
                    resource="data.*",
                    resource_type=ResourceType.DATA,
                    permission_level=PermissionLevel.READ,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                ),
                Permission(
                    resource="api.trading",
                    resource_type=ResourceType.API,
                    permission_level=PermissionLevel.WRITE,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                )
            ]
            
            self.roles["trader"] = Role(
                role_id="trader",
                name="Trader",
                description="Trading operations with data access",
                permissions=trader_permissions,
                created_at=datetime.now(timezone.utc).isoformat(),
                created_by="system"
            )
            
            # Analyst role
            analyst_permissions = [
                Permission(
                    resource="data.*",
                    resource_type=ResourceType.DATA,
                    permission_level=PermissionLevel.READ,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                ),
                Permission(
                    resource="reports.view",
                    resource_type=ResourceType.REPORTS,
                    permission_level=PermissionLevel.READ,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                ),
                Permission(
                    resource="api.data",
                    resource_type=ResourceType.API,
                    permission_level=PermissionLevel.READ,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                )
            ]
            
            self.roles["analyst"] = Role(
                role_id="analyst",
                name="Analyst",
                description="Data analysis and reporting access",
                permissions=analyst_permissions,
                created_at=datetime.now(timezone.utc).isoformat(),
                created_by="system"
            )
            
            # Viewer role
            viewer_permissions = [
                Permission(
                    resource="data.public",
                    resource_type=ResourceType.DATA,
                    permission_level=PermissionLevel.READ,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                ),
                Permission(
                    resource="reports.public",
                    resource_type=ResourceType.REPORTS,
                    permission_level=PermissionLevel.READ,
                    granted_at=datetime.now(timezone.utc).isoformat(),
                    granted_by="system"
                )
            ]
            
            self.roles["viewer"] = Role(
                role_id="viewer",
                name="Viewer",
                description="Public data and report viewing",
                permissions=viewer_permissions,
                created_at=datetime.now(timezone.utc).isoformat(),
                created_by="system"
            )
            
            logger.info("Default roles loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load default roles: {e}")
            raise
    
    def _load_config(self, config_file: str):
        """Load access control configuration from file."""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Load custom roles
                if 'roles' in config:
                    for role_data in config['roles']:
                        role = Role(**role_data)
                        self.roles[role.role_id] = role
                
                # Load user access
                if 'user_access' in config:
                    for user_data in config['user_access']:
                        user_access = UserAccess(**user_data)
                        self.user_access[user_access.user_id] = user_access
                
                logger.info(f"Access control configuration loaded from {config_file}")
                
        except Exception as e:
            logger.error(f"Failed to load access control configuration: {e}")
    
    def _start_cache_cleanup(self):
        """Start background thread for cache cleanup."""
        def cleanup_cache():
            while True:
                try:
                    time.sleep(60)  # Run every minute
                    self._cleanup_expired_cache()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_cache, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, cache_data in self.access_cache.items():
            if current_time - cache_data.get('timestamp', 0) > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.access_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def create_role(self, role_id: str, name: str, description: str, 
                   permissions: List[Permission], created_by: str) -> Role:
        """
        Create a new role.
        
        Args:
            role_id: Unique role identifier
            name: Role name
            description: Role description
            permissions: List of permissions
            created_by: User who created the role
            
        Returns:
            Created role
        """
        with self.lock:
            if role_id in self.roles:
                raise ValueError(f"Role {role_id} already exists")
            
            role = Role(
                role_id=role_id,
                name=name,
                description=description,
                permissions=permissions,
                created_at=datetime.now(timezone.utc).isoformat(),
                created_by=created_by
            )
            
            self.roles[role_id] = role
            logger.info(f"Role created: {role_id} by {created_by}")
            return role
    
    def update_role(self, role_id: str, name: Optional[str] = None,
                   description: Optional[str] = None,
                   permissions: Optional[List[Permission]] = None,
                   is_active: Optional[bool] = None) -> Role:
        """
        Update an existing role.
        
        Args:
            role_id: Role identifier
            name: New role name
            description: New role description
            permissions: New permissions list
            is_active: Active status
            
        Returns:
            Updated role
        """
        with self.lock:
            if role_id not in self.roles:
                raise ValueError(f"Role {role_id} does not exist")
            
            role = self.roles[role_id]
            
            if name is not None:
                role.name = name
            if description is not None:
                role.description = description
            if permissions is not None:
                role.permissions = permissions
            if is_active is not None:
                role.is_active = is_active
            
            # Clear cache for users with this role
            self._clear_user_cache_by_role(role_id)
            
            logger.info(f"Role updated: {role_id}")
            return role
    
    def delete_role(self, role_id: str):
        """
        Delete a role.
        
        Args:
            role_id: Role identifier
        """
        with self.lock:
            if role_id not in self.roles:
                raise ValueError(f"Role {role_id} does not exist")
            
            # Check if role is assigned to any users
            for user_id, user_access in self.user_access.items():
                if role_id in user_access.roles:
                    raise ValueError(f"Cannot delete role {role_id} - assigned to user {user_id}")
            
            del self.roles[role_id]
            logger.info(f"Role deleted: {role_id}")
    
    def assign_role_to_user(self, user_id: str, role_id: str, assigned_by: str):
        """
        Assign a role to a user.
        
        Args:
            user_id: User identifier
            role_id: Role identifier
            assigned_by: User who assigned the role
        """
        with self.lock:
            if role_id not in self.roles:
                raise ValueError(f"Role {role_id} does not exist")
            
            if not self.roles[role_id].is_active:
                raise ValueError(f"Role {role_id} is not active")
            
            if user_id not in self.user_access:
                self.user_access[user_id] = UserAccess(
                    user_id=user_id,
                    roles=[],
                    permissions=[],
                    last_access=datetime.now(timezone.utc).isoformat()
                )
            
            if role_id not in self.user_access[user_id].roles:
                self.user_access[user_id].roles.append(role_id)
                
                # Add role permissions to user
                role_permissions = self.roles[role_id].permissions
                self.user_access[user_id].permissions.extend(role_permissions)
                
                # Clear user cache
                self._clear_user_cache(user_id)
                
                logger.info(f"Role {role_id} assigned to user {user_id} by {assigned_by}")
    
    def remove_role_from_user(self, user_id: str, role_id: str, removed_by: str):
        """
        Remove a role from a user.
        
        Args:
            user_id: User identifier
            role_id: Role identifier
            removed_by: User who removed the role
        """
        with self.lock:
            if user_id not in self.user_access:
                raise ValueError(f"User {user_id} does not exist")
            
            if role_id not in self.user_access[user_id].roles:
                raise ValueError(f"User {user_id} does not have role {role_id}")
            
            # Remove role from user
            self.user_access[user_id].roles.remove(role_id)
            
            # Rebuild user permissions from remaining roles
            self._rebuild_user_permissions(user_id)
            
            # Clear user cache
            self._clear_user_cache(user_id)
            
            logger.info(f"Role {role_id} removed from user {user_id} by {removed_by}")
    
    def _rebuild_user_permissions(self, user_id: str):
        """Rebuild user permissions from assigned roles."""
        user_access = self.user_access[user_id]
        user_access.permissions = []
        
        for role_id in user_access.roles:
            if role_id in self.roles and self.roles[role_id].is_active:
                user_access.permissions.extend(self.roles[role_id].permissions)
    
    def check_permission(self, user_id: str, resource: str, 
                        required_level: PermissionLevel,
                        resource_type: Optional[ResourceType] = None) -> bool:
        """
        Check if user has permission to access a resource.
        
        Args:
            user_id: User identifier
            resource: Resource to access
            required_level: Required permission level
            resource_type: Resource type (optional)
            
        Returns:
            True if user has permission, False otherwise
        """
        try:
            # Check cache first
            cache_key = f"{user_id}:{resource}:{required_level.value}"
            if cache_key in self.access_cache:
                cache_data = self.access_cache[cache_key]
                if time.time() - cache_data['timestamp'] < self.cache_ttl:
                    return cache_data['result']
            
            # Check user access
            if user_id not in self.user_access:
                result = False
            else:
                user_access = self.user_access[user_id]
                
                if not user_access.is_active:
                    result = False
                else:
                    # Update last access time
                    user_access.last_access = datetime.now(timezone.utc).isoformat()
                    
                    # Check permissions
                    result = self._check_user_permissions(user_access, resource, 
                                                        required_level, resource_type)
            
            # Cache result
            self.access_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking permission for user {user_id}: {e}")
            return False
    
    def _check_user_permissions(self, user_access: UserAccess, resource: str,
                               required_level: PermissionLevel,
                               resource_type: Optional[ResourceType]) -> bool:
        """Check user permissions for a specific resource."""
        max_permission = PermissionLevel.NONE
        
        for permission in user_access.permissions:
            # Check if permission applies to this resource
            if self._permission_matches_resource(permission, resource, resource_type):
                # Check if permission is not expired
                if not self._is_permission_expired(permission):
                    # Check conditions
                    if self._check_permission_conditions(permission):
                        max_permission = max(max_permission, permission.permission_level)
        
        return max_permission.value >= required_level.value
    
    def _permission_matches_resource(self, permission: Permission, resource: str,
                                   resource_type: Optional[ResourceType]) -> bool:
        """Check if permission matches the resource."""
        # Check resource type if specified
        if resource_type and permission.resource_type != resource_type:
            return False
        
        # Check resource pattern
        if permission.resource == "*":
            return True
        elif permission.resource.endswith(".*"):
            prefix = permission.resource[:-2]
            return resource.startswith(prefix)
        else:
            return permission.resource == resource
    
    def _is_permission_expired(self, permission: Permission) -> bool:
        """Check if permission has expired."""
        if not permission.expires_at:
            return False
        
        try:
            expires_at = datetime.fromisoformat(permission.expires_at.replace('Z', '+00:00'))
            return datetime.now(timezone.utc) > expires_at
        except Exception:
            return False
    
    def _check_permission_conditions(self, permission: Permission) -> bool:
        """Check permission conditions."""
        if not permission.conditions:
            return True
        
        # Implement condition checking logic here
        # For now, return True (conditions are met)
        return True
    
    def grant_permission(self, user_id: str, resource: str, 
                        permission_level: PermissionLevel,
                        resource_type: ResourceType,
                        granted_by: str,
                        expires_at: Optional[str] = None,
                        conditions: Optional[Dict[str, Any]] = None):
        """
        Grant a specific permission to a user.
        
        Args:
            user_id: User identifier
            resource: Resource to grant access to
            permission_level: Permission level
            resource_type: Resource type
            granted_by: User who granted the permission
            expires_at: Expiration time (optional)
            conditions: Permission conditions (optional)
        """
        with self.lock:
            if user_id not in self.user_access:
                raise ValueError(f"User {user_id} does not exist")
            
            permission = Permission(
                resource=resource,
                resource_type=resource_type,
                permission_level=permission_level,
                granted_at=datetime.now(timezone.utc).isoformat(),
                granted_by=granted_by,
                expires_at=expires_at,
                conditions=conditions
            )
            
            self.user_access[user_id].permissions.append(permission)
            self._clear_user_cache(user_id)
            
            logger.info(f"Permission granted to user {user_id} for {resource} by {granted_by}")
    
    def revoke_permission(self, user_id: str, resource: str, revoked_by: str):
        """
        Revoke a specific permission from a user.
        
        Args:
            user_id: User identifier
            resource: Resource to revoke access from
            revoked_by: User who revoked the permission
        """
        with self.lock:
            if user_id not in self.user_access:
                raise ValueError(f"User {user_id} does not exist")
            
            user_permissions = self.user_access[user_id].permissions
            original_count = len(user_permissions)
            
            # Remove permissions for this resource
            user_permissions[:] = [p for p in user_permissions if p.resource != resource]
            
            if len(user_permissions) < original_count:
                self._clear_user_cache(user_id)
                logger.info(f"Permission revoked from user {user_id} for {resource} by {revoked_by}")
            else:
                logger.warning(f"No permission found for user {user_id} on resource {resource}")
    
    def get_user_permissions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all permissions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user permissions
        """
        if user_id not in self.user_access:
            return []
        
        user_access = self.user_access[user_id]
        permissions = []
        
        for permission in user_access.permissions:
            if not self._is_permission_expired(permission):
                permissions.append(asdict(permission))
        
        return permissions
    
    def get_user_roles(self, user_id: str) -> List[str]:
        """
        Get all roles for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of role IDs
        """
        if user_id not in self.user_access:
            return []
        
        return self.user_access[user_id].roles.copy()
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """
        List all roles.
        
        Returns:
            List of role information
        """
        roles = []
        for role in self.roles.values():
            roles.append(asdict(role))
        return roles
    
    def list_users(self) -> List[Dict[str, Any]]:
        """
        List all users with their access information.
        
        Returns:
            List of user access information
        """
        users = []
        for user_access in self.user_access.values():
            users.append(asdict(user_access))
        return users
    
    def _clear_user_cache(self, user_id: str):
        """Clear cache entries for a specific user."""
        keys_to_remove = [key for key in self.access_cache.keys() if key.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.access_cache[key]
    
    def _clear_user_cache_by_role(self, role_id: str):
        """Clear cache entries for users with a specific role."""
        for user_id, user_access in self.user_access.items():
            if role_id in user_access.roles:
                self._clear_user_cache(user_id)
    
    def export_config(self, file_path: str):
        """
        Export access control configuration to file.
        
        Args:
            file_path: Output file path
        """
        try:
            config = {
                'roles': [asdict(role) for role in self.roles.values()],
                'user_access': [asdict(user_access) for user_access in self.user_access.values()],
                'exported_at': datetime.now(timezone.utc).isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"Access control configuration exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export access control configuration: {e}")
            raise


# Global access control manager instance
_access_control_manager = None


def get_access_control_manager() -> AccessControlManager:
    """Get the global access control manager instance."""
    global _access_control_manager
    if _access_control_manager is None:
        _access_control_manager = AccessControlManager()
    return _access_control_manager


# Example usage and testing
if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Test access control
    acm = AccessControlManager()
    
    # Create a test user
    acm.assign_role_to_user("test_user", "trader", "system")
    
    # Test permissions
    can_trade = acm.check_permission("test_user", "trading.execute", PermissionLevel.WRITE)
    can_admin = acm.check_permission("test_user", "system.config", PermissionLevel.ADMIN)
    
    print(f"User can trade: {can_trade}")
    print(f"User can admin: {can_admin}")
    
    # Grant additional permission
    acm.grant_permission("test_user", "reports.advanced", PermissionLevel.READ, 
                        ResourceType.REPORTS, "admin")
    
    # Test new permission
    can_view_advanced_reports = acm.check_permission("test_user", "reports.advanced", 
                                                   PermissionLevel.READ)
    print(f"User can view advanced reports: {can_view_advanced_reports}")
    
    # List user permissions
    permissions = acm.get_user_permissions("test_user")
    print(f"User permissions: {json.dumps(permissions, indent=2)}")
    
    print("Access control system test completed successfully!")