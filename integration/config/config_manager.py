#!/usr/bin/env python3
"""
Configuration Management Module

Implements comprehensive configuration management for the trading system:
- Configuration loading and validation
- Environment-specific configurations
- Configuration hot-reloading
- Configuration encryption
- Configuration templates
- Configuration versioning

Features:
- Complete configuration management and validation
- Environment-specific configuration loading
- Hot-reloading of configuration changes
- Secure configuration encryption and decryption
- Configuration templates and defaults
- Configuration versioning and rollback
- Multi-format configuration support (JSON, YAML, INI)
"""

import os
import json
import yaml
import configparser
import hashlib
import threading
import asyncio
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
from pathlib import Path
import copy
import base64

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    warnings.warn("Cryptography not available. Configuration encryption will not function.")

logger = structlog.get_logger()

class ConfigFormat(Enum):
    """Configuration format enumeration."""
    JSON = "json"
    YAML = "yaml"
    INI = "ini"
    ENV = "env"

class ConfigEnvironment(Enum):
    """Configuration environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class ConfigValidationLevel(Enum):
    """Configuration validation level enumeration."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    CUSTOM = "custom"

@dataclass
class ConfigSection:
    """Configuration section structure."""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    required: bool = False
    encrypted: bool = False
    validation_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfigTemplate:
    """Configuration template structure."""
    template_id: str
    name: str
    description: str
    sections: List[ConfigSection] = field(default_factory=list)
    version: str = "1.0.0"
    environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ConfigVersion:
    """Configuration version structure."""
    version_id: str
    config_hash: str
    config_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    author: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class ConfigManagerConfig:
    """Configuration manager configuration."""
    config_dir: str = "config"
    default_environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT
    default_format: ConfigFormat = ConfigFormat.JSON
    validation_level: ConfigValidationLevel = ConfigValidationLevel.BASIC
    enable_hot_reload: bool = True
    enable_encryption: bool = True
    enable_versioning: bool = True
    max_versions: int = 10
    encryption_key: Optional[str] = None
    watch_interval: float = 5.0  # seconds
    config_file_pattern: str = "*.{format}"

class ConfigurationValidator:
    """Configuration validation system."""
    
    def __init__(self, validation_level: ConfigValidationLevel = ConfigValidationLevel.BASIC):
        """
        Initialize configuration validator.
        
        Args:
            validation_level: Validation level
        """
        self.validation_level = validation_level
        self.validators = {}
        self.custom_rules = {}
    
    def add_validator(self, field_name: str, validator: Callable[[Any], bool]):
        """Add custom validator for a field."""
        self.validators[field_name] = validator
    
    def add_custom_rule(self, rule_name: str, rule_func: Callable[[Dict[str, Any]], List[str]]):
        """Add custom validation rule."""
        self.custom_rules[rule_name] = rule_func
    
    def validate_config(self, config_data: Dict[str, Any], 
                       template: Optional[ConfigTemplate] = None) -> Tuple[bool, List[str]]:
        """
        Validate configuration data.
        
        Args:
            config_data: Configuration data to validate
            template: Configuration template for validation
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if self.validation_level == ConfigValidationLevel.NONE:
            return True, []
        
        # Basic validation
        if self.validation_level in [ConfigValidationLevel.BASIC, ConfigValidationLevel.STRICT]:
            errors.extend(self._basic_validation(config_data))
        
        # Template-based validation
        if template and self.validation_level in [ConfigValidationLevel.STRICT, ConfigValidationLevel.CUSTOM]:
            errors.extend(self._template_validation(config_data, template))
        
        # Custom validation
        if self.validation_level == ConfigValidationLevel.CUSTOM:
            errors.extend(self._custom_validation(config_data))
        
        return len(errors) == 0, errors
    
    def _basic_validation(self, config_data: Dict[str, Any]) -> List[str]:
        """Perform basic validation."""
        errors = []
        
        # Check for required fields
        required_fields = ['environment', 'version']
        for field in required_fields:
            if field not in config_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate environment
        if 'environment' in config_data:
            try:
                ConfigEnvironment(config_data['environment'])
            except ValueError:
                errors.append(f"Invalid environment: {config_data['environment']}")
        
        # Validate version format
        if 'version' in config_data:
            version = config_data['version']
            if not isinstance(version, str) or not version.strip():
                errors.append("Version must be a non-empty string")
        
        return errors
    
    def _template_validation(self, config_data: Dict[str, Any], 
                           template: ConfigTemplate) -> List[str]:
        """Perform template-based validation."""
        errors = []
        
        # Check required sections
        for section in template.sections:
            if section.required and section.name not in config_data:
                errors.append(f"Missing required section: {section.name}")
            elif section.name in config_data:
                # Validate section data
                section_errors = self._validate_section(config_data[section.name], section)
                errors.extend([f"{section.name}.{error}" for error in section_errors])
        
        return errors
    
    def _validate_section(self, section_data: Dict[str, Any], 
                         section_template: ConfigSection) -> List[str]:
        """Validate a configuration section."""
        errors = []
        
        if not isinstance(section_data, dict):
            errors.append("Section data must be a dictionary")
            return errors
        
        # Validate required fields
        for field_name, field_rules in section_template.validation_rules.items():
            if field_rules.get('required', False) and field_name not in section_data:
                errors.append(f"Missing required field: {field_name}")
            elif field_name in section_data:
                # Validate field value
                field_errors = self._validate_field(section_data[field_name], field_rules)
                errors.extend(field_errors)
        
        return errors
    
    def _validate_field(self, field_value: Any, field_rules: Dict[str, Any]) -> List[str]:
        """Validate a configuration field."""
        errors = []
        
        # Type validation
        expected_type = field_rules.get('type')
        if expected_type:
            if expected_type == 'string' and not isinstance(field_value, str):
                errors.append("Field must be a string")
            elif expected_type == 'integer' and not isinstance(field_value, int):
                errors.append("Field must be an integer")
            elif expected_type == 'float' and not isinstance(field_value, (int, float)):
                errors.append("Field must be a number")
            elif expected_type == 'boolean' and not isinstance(field_value, bool):
                errors.append("Field must be a boolean")
            elif expected_type == 'list' and not isinstance(field_value, list):
                errors.append("Field must be a list")
            elif expected_type == 'dict' and not isinstance(field_value, dict):
                errors.append("Field must be a dictionary")
        
        # Range validation
        if 'min' in field_rules and field_value < field_rules['min']:
            errors.append(f"Field value must be >= {field_rules['min']}")
        if 'max' in field_rules and field_value > field_rules['max']:
            errors.append(f"Field value must be <= {field_rules['max']}")
        
        # Pattern validation
        if 'pattern' in field_rules and isinstance(field_value, str):
            import re
            if not re.match(field_rules['pattern'], field_value):
                errors.append(f"Field value must match pattern: {field_rules['pattern']}")
        
        # Enum validation
        if 'enum' in field_rules and field_value not in field_rules['enum']:
            errors.append(f"Field value must be one of: {field_rules['enum']}")
        
        # Custom validator
        if 'validator' in field_rules:
            validator = self.validators.get(field_rules['validator'])
            if validator and not validator(field_value):
                errors.append(f"Field failed custom validation: {field_rules['validator']}")
        
        return errors
    
    def _custom_validation(self, config_data: Dict[str, Any]) -> List[str]:
        """Perform custom validation."""
        errors = []
        
        for rule_name, rule_func in self.custom_rules.items():
            try:
                rule_errors = rule_func(config_data)
                errors.extend(rule_errors)
            except Exception as e:
                errors.append(f"Custom validation rule '{rule_name}' failed: {str(e)}")
        
        return errors

class ConfigurationEncryption:
    """Configuration encryption system."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize configuration encryption.
        
        Args:
            encryption_key: Encryption key (if None, will generate one)
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography not available. Encryption disabled.")
            self.fernet = None
            return
        
        if encryption_key:
            self.fernet = self._create_fernet_from_key(encryption_key)
        else:
            self.fernet = Fernet.generate_key()
            self.fernet = Fernet(self.fernet)
        
        logger.info("Configuration encryption initialized")
    
    def _create_fernet_from_key(self, key: str):
        """Create Fernet instance from key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return None
        
        # Generate a key from the provided key
        salt = b'config_salt_123'  # In production, use a secure random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key_bytes = kdf.derive(key.encode())
        return Fernet(base64.urlsafe_b64encode(key_bytes))
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value."""
        if not self.fernet:
            return value
        
        try:
            encrypted = self.fernet.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value."""
        if not self.fernet:
            return encrypted_value
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            return encrypted_value
    
    def encrypt_config_section(self, section_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt a configuration section."""
        if not self.fernet:
            return section_data
        
        encrypted_section = {}
        for key, value in section_data.items():
            if isinstance(value, str):
                encrypted_section[key] = self.encrypt_value(value)
            elif isinstance(value, dict):
                encrypted_section[key] = self.encrypt_config_section(value)
            else:
                encrypted_section[key] = value
        
        return encrypted_section
    
    def decrypt_config_section(self, section_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt a configuration section."""
        if not self.fernet:
            return section_data
        
        decrypted_section = {}
        for key, value in section_data.items():
            if isinstance(value, str):
                decrypted_section[key] = self.decrypt_value(value)
            elif isinstance(value, dict):
                decrypted_section[key] = self.decrypt_config_section(value)
            else:
                decrypted_section[key] = value
        
        return decrypted_section

class ConfigurationVersioning:
    """Configuration versioning system."""
    
    def __init__(self, max_versions: int = 10):
        """
        Initialize configuration versioning.
        
        Args:
            max_versions: Maximum number of versions to keep
        """
        self.max_versions = max_versions
        self.versions = deque(maxlen=max_versions)
        self._lock = threading.RLock()
    
    def add_version(self, config_data: Dict[str, Any], author: str = "", 
                   description: str = "", tags: List[str] = None) -> str:
        """
        Add a new configuration version.
        
        Args:
            config_data: Configuration data
            author: Version author
            description: Version description
            tags: Version tags
            
        Returns:
            Version ID
        """
        with self._lock:
            # Calculate config hash
            config_hash = self._calculate_config_hash(config_data)
            
            # Create version
            version_id = str(uuid.uuid4())
            version = ConfigVersion(
                version_id=version_id,
                config_hash=config_hash,
                config_data=copy.deepcopy(config_data),
                author=author,
                description=description,
                tags=tags or []
            )
            
            # Add to versions
            self.versions.append(version)
            
            logger.info("Configuration version added", 
                       version_id=version_id,
                       config_hash=config_hash[:8])
            
            return version_id
    
    def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """Get a specific version."""
        with self._lock:
            for version in self.versions:
                if version.version_id == version_id:
                    return version
            return None
    
    def get_latest_version(self) -> Optional[ConfigVersion]:
        """Get the latest version."""
        with self._lock:
            return self.versions[-1] if self.versions else None
    
    def get_versions_by_tag(self, tag: str) -> List[ConfigVersion]:
        """Get versions by tag."""
        with self._lock:
            return [v for v in self.versions if tag in v.tags]
    
    def rollback_to_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Rollback to a specific version."""
        version = self.get_version(version_id)
        if version:
            logger.info("Rolling back to version", version_id=version_id)
            return copy.deepcopy(version.config_data)
        return None
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Calculate hash of configuration data."""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

class ConfigurationManager:
    """Main configuration management system."""
    
    def __init__(self, config: ConfigManagerConfig = None):
        """
        Initialize configuration manager.
        
        Args:
            config: Configuration manager configuration
        """
        self.config = config or ConfigManagerConfig()
        self.validator = ConfigurationValidator(self.config.validation_level)
        self.encryption = ConfigurationEncryption(self.config.encryption_key)
        self.versioning = ConfigurationVersioning(self.config.max_versions)
        
        self.config_data = {}
        self.templates = {}
        self.watchers = {}
        self.callbacks = defaultdict(list)
        
        self._lock = threading.RLock()
        self._file_watcher_task = None
        self._running = False
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config.config_dir, exist_ok=True)
        
        logger.info("Configuration manager initialized")
    
    def load_config(self, environment: ConfigEnvironment = None, 
                   config_format: ConfigFormat = None) -> Dict[str, Any]:
        """
        Load configuration for environment.
        
        Args:
            environment: Configuration environment
            config_format: Configuration format
            
        Returns:
            Configuration data
        """
        environment = environment or self.config.default_environment
        config_format = config_format or self.config.default_format
        
        config_file = self._get_config_file_path(environment, config_format)
        
        if not os.path.exists(config_file):
            logger.warning("Configuration file not found", file=config_file)
            return self._get_default_config(environment)
        
        try:
            config_data = self._load_config_file(config_file, config_format)
            
            # Validate configuration
            is_valid, errors = self.validator.validate_config(config_data)
            if not is_valid:
                logger.error("Configuration validation failed", errors=errors)
                raise ValueError(f"Configuration validation failed: {errors}")
            
            # Decrypt encrypted sections
            config_data = self._decrypt_config(config_data)
            
            # Store configuration
            with self._lock:
                self.config_data = config_data
            
            # Add version
            if self.config.enable_versioning:
                self.versioning.add_version(config_data, description=f"Loaded {environment.value} config")
            
            logger.info("Configuration loaded", 
                       environment=environment.value,
                       file=config_file)
            
            return config_data
            
        except Exception as e:
            logger.error("Failed to load configuration", 
                        environment=environment.value,
                        error=str(e))
            raise
    
    def save_config(self, config_data: Dict[str, Any], 
                   environment: ConfigEnvironment = None,
                   config_format: ConfigFormat = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_data: Configuration data
            environment: Configuration environment
            config_format: Configuration format
            
        Returns:
            True if successful
        """
        environment = environment or self.config.default_environment
        config_format = config_format or self.config.default_format
        
        config_file = self._get_config_file_path(environment, config_format)
        
        try:
            # Validate configuration
            is_valid, errors = self.validator.validate_config(config_data)
            if not is_valid:
                logger.error("Configuration validation failed", errors=errors)
                return False
            
            # Encrypt sensitive sections
            config_data_to_save = self._encrypt_config(config_data)
            
            # Save to file
            self._save_config_file(config_file, config_data_to_save, config_format)
            
            # Update stored configuration
            with self._lock:
                self.config_data = config_data
            
            # Add version
            if self.config.enable_versioning:
                self.versioning.add_version(config_data, description=f"Saved {environment.value} config")
            
            logger.info("Configuration saved", 
                       environment=environment.value,
                       file=config_file)
            
            return True
            
        except Exception as e:
            logger.error("Failed to save configuration", 
                        environment=environment.value,
                        error=str(e))
            return False
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        with self._lock:
            if key is None:
                return self.config_data
            
            # Support dot notation for nested keys
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
    
    def set_config(self, key: str, value: Any) -> bool:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            value: Configuration value
            
        Returns:
            True if successful
        """
        with self._lock:
            # Support dot notation for nested keys
            keys = key.split('.')
            config = self.config_data
            
            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set value
            config[keys[-1]] = value
            
            logger.info("Configuration value set", key=key)
            return True
    
    def add_template(self, template: ConfigTemplate):
        """Add configuration template."""
        with self._lock:
            self.templates[template.template_id] = template
            logger.info("Configuration template added", template_id=template.template_id)
    
    def get_template(self, template_id: str) -> Optional[ConfigTemplate]:
        """Get configuration template."""
        with self._lock:
            return self.templates.get(template_id)
    
    def create_config_from_template(self, template_id: str, 
                                  environment: ConfigEnvironment = None) -> Dict[str, Any]:
        """Create configuration from template."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        environment = environment or self.config.default_environment
        
        # Create base configuration
        config_data = {
            'environment': environment.value,
            'version': template.version,
            'template_id': template_id,
            'created_at': datetime.now().isoformat()
        }
        
        # Add template sections
        for section in template.sections:
            config_data[section.name] = section.data.copy()
        
        return config_data
    
    def add_config_watcher(self, callback: Callable[[Dict[str, Any]], None]):
        """Add configuration change watcher."""
        with self._lock:
            self.callbacks['config_changed'].append(callback)
    
    def start_file_watcher(self):
        """Start file watching for hot reload."""
        if not self.config.enable_hot_reload:
            return
        
        self._running = True
        self._file_watcher_task = asyncio.create_task(self._file_watcher())
        logger.info("Configuration file watcher started")
    
    def stop_file_watcher(self):
        """Stop file watching."""
        self._running = False
        if self._file_watcher_task:
            self._file_watcher_task.cancel()
        logger.info("Configuration file watcher stopped")
    
    def _get_config_file_path(self, environment: ConfigEnvironment, 
                             config_format: ConfigFormat) -> str:
        """Get configuration file path."""
        filename = f"config.{environment.value}.{config_format.value}"
        return os.path.join(self.config.config_dir, filename)
    
    def _load_config_file(self, file_path: str, config_format: ConfigFormat) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(file_path, 'r') as f:
            if config_format == ConfigFormat.JSON:
                return json.load(f)
            elif config_format == ConfigFormat.YAML:
                return yaml.safe_load(f)
            elif config_format == ConfigFormat.INI:
                config = configparser.ConfigParser()
                config.read(file_path)
                return self._ini_to_dict(config)
            else:
                raise ValueError(f"Unsupported config format: {config_format}")
    
    def _save_config_file(self, file_path: str, config_data: Dict[str, Any], 
                         config_format: ConfigFormat):
        """Save configuration to file."""
        with open(file_path, 'w') as f:
            if config_format == ConfigFormat.JSON:
                json.dump(config_data, f, indent=2)
            elif config_format == ConfigFormat.YAML:
                yaml.dump(config_data, f, default_flow_style=False)
            elif config_format == ConfigFormat.INI:
                config = self._dict_to_ini(config_data)
                config.write(f)
            else:
                raise ValueError(f"Unsupported config format: {config_format}")
    
    def _ini_to_dict(self, config: configparser.ConfigParser) -> Dict[str, Any]:
        """Convert INI config to dictionary."""
        result = {}
        for section in config.sections():
            result[section] = dict(config[section])
        return result
    
    def _dict_to_ini(self, config_data: Dict[str, Any]) -> configparser.ConfigParser:
        """Convert dictionary to INI config."""
        config = configparser.ConfigParser()
        for section, data in config_data.items():
            if isinstance(data, dict):
                config[section] = data
            else:
                config['DEFAULT'][section] = str(data)
        return config
    
    def _get_default_config(self, environment: ConfigEnvironment) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'environment': environment.value,
            'version': '1.0.0',
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'trading_system',
                'user': 'trading_user'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': environment == ConfigEnvironment.DEVELOPMENT
            },
            'logging': {
                'level': 'INFO',
                'format': 'json'
            }
        }
    
    def _encrypt_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt configuration data."""
        if not self.config.enable_encryption:
            return config_data
        
        encrypted_config = {}
        for key, value in config_data.items():
            if key in ['database', 'api']:  # Encrypt sensitive sections
                encrypted_config[key] = self.encryption.encrypt_config_section(value)
            else:
                encrypted_config[key] = value
        
        return encrypted_config
    
    def _decrypt_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt configuration data."""
        if not self.config.enable_encryption:
            return config_data
        
        decrypted_config = {}
        for key, value in config_data.items():
            if key in ['database', 'api']:  # Decrypt sensitive sections
                decrypted_config[key] = self.encryption.decrypt_config_section(value)
            else:
                decrypted_config[key] = value
        
        return decrypted_config
    
    async def _file_watcher(self):
        """File watching task for hot reload."""
        last_modified = {}
        
        while self._running:
            try:
                # Check all config files
                for environment in ConfigEnvironment:
                    for config_format in ConfigFormat:
                        if config_format == ConfigFormat.ENV:
                            continue
                        
                        file_path = self._get_config_file_path(environment, config_format)
                        if os.path.exists(file_path):
                            current_mtime = os.path.getmtime(file_path)
                            last_mtime = last_modified.get(file_path, 0)
                            
                            if current_mtime > last_mtime:
                                last_modified[file_path] = current_mtime
                                
                                # Reload configuration
                                try:
                                    new_config = self.load_config(environment, config_format)
                                    
                                    # Notify watchers
                                    for callback in self.callbacks['config_changed']:
                                        try:
                                            callback(new_config)
                                        except Exception as e:
                                            logger.error("Config change callback error", error=str(e))
                                    
                                    logger.info("Configuration reloaded", file=file_path)
                                    
                                except Exception as e:
                                    logger.error("Failed to reload configuration", 
                                               file=file_path, error=str(e))
                
                await asyncio.sleep(self.config.watch_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("File watcher error", error=str(e))
                await asyncio.sleep(self.config.watch_interval)

def create_config_manager(config: ConfigManagerConfig = None) -> ConfigurationManager:
    """
    Create a configuration manager.
    
    Args:
        config: Configuration manager configuration
        
    Returns:
        ConfigurationManager instance
    """
    return ConfigurationManager(config)

if __name__ == "__main__":
    # Demo of configuration manager
    config = ConfigManagerConfig(
        config_dir="config",
        default_environment=ConfigEnvironment.DEVELOPMENT,
        default_format=ConfigFormat.JSON,
        validation_level=ConfigValidationLevel.BASIC,
        enable_hot_reload=True,
        enable_encryption=True,
        enable_versioning=True
    )
    
    manager = create_config_manager(config)
    
    # Create sample template
    template = ConfigTemplate(
        template_id="trading_system_v1",
        name="Trading System v1.0",
        description="Basic trading system configuration template",
        sections=[
            ConfigSection("database", {
                "host": "localhost",
                "port": 5432,
                "name": "trading_system"
            }, required=True),
            ConfigSection("api", {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": True
            }, required=True)
        ],
        environment=ConfigEnvironment.DEVELOPMENT
    )
    
    manager.add_template(template)
    
    # Create configuration from template
    config_data = manager.create_config_from_template("trading_system_v1")
    
    # Save configuration
    success = manager.save_config(config_data)
    
    print("Configuration Manager created successfully!")
    print(f"Templates: {len(manager.templates)}")
    print(f"Configuration saved: {success}")
    
    # Load configuration
    loaded_config = manager.load_config()
    print(f"Configuration loaded: {loaded_config is not None}")
    
    # Get configuration value
    db_host = manager.get_config("database.host")
    print(f"Database host: {db_host}")
    
    # Get versioning info
    latest_version = manager.versioning.get_latest_version()
    if latest_version:
        print(f"Latest version: {latest_version.version_id}")