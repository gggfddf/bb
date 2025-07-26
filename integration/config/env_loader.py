"""
Environment Configuration Loader for Trading System

This module provides environment-specific configuration loading functionality,
supporting multiple environments (development, staging, production) and
environment variable overrides.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import configparser

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentConfig:
    """Environment configuration structure"""
    environment: str
    config_data: Dict[str, Any]
    loaded_at: datetime
    source_files: List[str]
    overrides: Dict[str, Any]

class EnvironmentLoader:
    """
    Environment-specific configuration loader.
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.config_cache: Dict[str, EnvironmentConfig] = {}
        
        # Supported file formats
        self.supported_formats = ['.json', '.yaml', '.yml', '.ini', '.env']
        
        # Environment-specific directories
        self.env_dirs = {
            'development': 'dev',
            'staging': 'staging',
            'production': 'prod',
            'testing': 'test'
        }
        
        logger.info(f"Environment loader initialized for environment: {self.environment}")
    
    def load_environment_config(self, environment: Optional[str] = None) -> EnvironmentConfig:
        """
        Load configuration for a specific environment.
        
        Args:
            environment: Environment name (defaults to current environment)
            
        Returns:
            Environment configuration
        """
        env = environment or self.environment
        
        # Check cache first
        if env in self.config_cache:
            logger.debug(f"Using cached configuration for environment: {env}")
            return self.config_cache[env]
        
        try:
            config_data = {}
            source_files = []
            overrides = {}
            
            # Load base configuration
            base_config = self._load_base_config()
            config_data.update(base_config)
            source_files.append('base')
            
            # Load environment-specific configuration
            env_config = self._load_environment_specific_config(env)
            config_data.update(env_config)
            source_files.append(f'environment_{env}')
            
            # Load local overrides
            local_config = self._load_local_config()
            config_data.update(local_config)
            if local_config:
                source_files.append('local')
            
            # Apply environment variable overrides
            env_overrides = self._load_environment_variables()
            config_data.update(env_overrides)
            overrides.update(env_overrides)
            
            # Create environment config
            env_config_obj = EnvironmentConfig(
                environment=env,
                config_data=config_data,
                loaded_at=datetime.utcnow(),
                source_files=source_files,
                overrides=overrides
            )
            
            # Cache the configuration
            self.config_cache[env] = env_config_obj
            
            logger.info(f"Configuration loaded for environment: {env}")
            return env_config_obj
            
        except Exception as e:
            logger.error(f"Failed to load configuration for environment {env}: {e}")
            raise
    
    def reload_config(self, environment: Optional[str] = None) -> EnvironmentConfig:
        """
        Reload configuration for an environment.
        
        Args:
            environment: Environment name (defaults to current environment)
            
        Returns:
            Reloaded environment configuration
        """
        env = environment or self.environment
        
        # Clear cache for this environment
        if env in self.config_cache:
            del self.config_cache[env]
        
        return self.load_environment_config(env)
    
    def get_config_value(self, key: str, default: Any = None, environment: Optional[str] = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            environment: Environment name (defaults to current environment)
            
        Returns:
            Configuration value
        """
        env_config = self.load_environment_config(environment)
        return self._get_nested_value(env_config.config_data, key, default)
    
    def set_config_value(self, key: str, value: Any, environment: Optional[str] = None) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            environment: Environment name (defaults to current environment)
            
        Returns:
            True if value was set successfully
        """
        try:
            env_config = self.load_environment_config(environment)
            self._set_nested_value(env_config.config_data, key, value)
            
            # Update cache
            self.config_cache[environment or self.environment] = env_config
            
            logger.debug(f"Configuration value set: {key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set configuration value {key}: {e}")
            return False
    
    def get_environment_info(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about an environment configuration.
        
        Args:
            environment: Environment name (defaults to current environment)
            
        Returns:
            Environment information
        """
        env_config = self.load_environment_config(environment)
        
        return {
            'environment': env_config.environment,
            'loaded_at': env_config.loaded_at.isoformat(),
            'source_files': env_config.source_files,
            'override_count': len(env_config.overrides),
            'config_keys': list(env_config.config_data.keys()),
            'cache_hit': True
        }
    
    def list_environments(self) -> List[str]:
        """
        List available environments.
        
        Returns:
            List of available environment names
        """
        environments = []
        
        # Check for environment-specific directories
        for env_name, dir_name in self.env_dirs.items():
            env_dir = self.config_dir / dir_name
            if env_dir.exists():
                environments.append(env_name)
        
        # Check for environment-specific files
        for file_path in self.config_dir.glob('*.json'):
            if file_path.stem.startswith('config_'):
                env_name = file_path.stem.replace('config_', '')
                if env_name not in environments:
                    environments.append(env_name)
        
        return sorted(environments)
    
    def export_config(self, environment: Optional[str] = None, format: str = 'json') -> str:
        """
        Export configuration to a string.
        
        Args:
            environment: Environment name (defaults to current environment)
            format: Export format ('json', 'yaml', 'ini')
            
        Returns:
            Configuration as string
        """
        env_config = self.load_environment_config(environment)
        
        if format.lower() == 'json':
            return json.dumps(env_config.config_data, indent=2)
        elif format.lower() in ['yaml', 'yml']:
            return yaml.dump(env_config.config_data, default_flow_style=False)
        elif format.lower() == 'ini':
            return self._config_to_ini(env_config.config_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        config_data = {}
        
        # Load base configuration files
        base_files = [
            self.config_dir / 'config.json',
            self.config_dir / 'config.yaml',
            self.config_dir / 'config.yml',
            self.config_dir / 'base.json',
            self.config_dir / 'base.yaml',
            self.config_dir / 'base.yml'
        ]
        
        for file_path in base_files:
            if file_path.exists():
                try:
                    file_config = self._load_config_file(file_path)
                    config_data.update(file_config)
                    logger.debug(f"Loaded base config from: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load base config from {file_path}: {e}")
        
        return config_data
    
    def _load_environment_specific_config(self, environment: str) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        config_data = {}
        
        # Try environment-specific directory
        env_dir_name = self.env_dirs.get(environment, environment)
        env_dir = self.config_dir / env_dir_name
        
        if env_dir.exists() and env_dir.is_dir():
            # Load all config files in environment directory
            for file_path in env_dir.glob('*'):
                if file_path.suffix in self.supported_formats:
                    try:
                        file_config = self._load_config_file(file_path)
                        config_data.update(file_config)
                        logger.debug(f"Loaded env config from: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load env config from {file_path}: {e}")
        
        # Try environment-specific files
        env_files = [
            self.config_dir / f'config_{environment}.json',
            self.config_dir / f'config_{environment}.yaml',
            self.config_dir / f'config_{environment}.yml',
            self.config_dir / f'{environment}.json',
            self.config_dir / f'{environment}.yaml',
            self.config_dir / f'{environment}.yml'
        ]
        
        for file_path in env_files:
            if file_path.exists():
                try:
                    file_config = self._load_config_file(file_path)
                    config_data.update(file_config)
                    logger.debug(f"Loaded env config from: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load env config from {file_path}: {e}")
        
        return config_data
    
    def _load_local_config(self) -> Dict[str, Any]:
        """Load local configuration overrides"""
        config_data = {}
        
        # Load local configuration files
        local_files = [
            self.config_dir / 'local.json',
            self.config_dir / 'local.yaml',
            self.config_dir / 'local.yml',
            self.config_dir / '.env.local',
            Path('.env.local'),
            Path('.env')
        ]
        
        for file_path in local_files:
            if file_path.exists():
                try:
                    if file_path.suffix == '.env':
                        file_config = self._load_env_file(file_path)
                    else:
                        file_config = self._load_config_file(file_path)
                    config_data.update(file_config)
                    logger.debug(f"Loaded local config from: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load local config from {file_path}: {e}")
        
        return config_data
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config_data = {}
        
        # Common environment variable prefixes
        prefixes = ['APP_', 'CONFIG_', 'TRADING_']
        
        for key, value in os.environ.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    # Convert key to nested structure
                    config_key = key[len(prefix):].lower()
                    nested_key = config_key.replace('_', '.')
                    self._set_nested_value(config_data, nested_key, value)
                    break
        
        return config_data
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    return json.load(f)
                elif file_path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif file_path.suffix == '.ini':
                    return self._load_ini_file(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_path.suffix}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            return {}
    
    def _load_ini_file(self, file_path: Path) -> Dict[str, Any]:
        """Load INI configuration file"""
        config_data = {}
        config = configparser.ConfigParser()
        
        try:
            config.read(file_path)
            
            for section in config.sections():
                config_data[section] = {}
                for key, value in config[section].items():
                    # Try to convert value to appropriate type
                    config_data[section][key] = self._convert_value(value)
            
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to load INI file {file_path}: {e}")
            return {}
    
    def _load_env_file(self, file_path: Path) -> Dict[str, Any]:
        """Load environment file"""
        config_data = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        config_data[key] = self._convert_value(value)
            
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to load env file {file_path}: {e}")
            return {}
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type"""
        # Try to convert to boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try to convert to integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _get_nested_value(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get nested value using dot notation"""
        keys = key.split('.')
        value = data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: Any):
        """Set nested value using dot notation"""
        keys = key.split('.')
        current = data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _config_to_ini(self, config_data: Dict[str, Any]) -> str:
        """Convert configuration to INI format"""
        config = configparser.ConfigParser()
        
        def add_section(section_name: str, section_data: Dict[str, Any]):
            if not config.has_section(section_name):
                config.add_section(section_name)
            
            for key, value in section_data.items():
                if isinstance(value, dict):
                    add_section(f"{section_name}.{key}", value)
                else:
                    config.set(section_name, key, str(value))
        
        for section_name, section_data in config_data.items():
            if isinstance(section_data, dict):
                add_section(section_name, section_data)
            else:
                if not config.has_section('DEFAULT'):
                    config.add_section('DEFAULT')
                config.set('DEFAULT', section_name, str(section_data))
        
        # Convert to string
        from io import StringIO
        output = StringIO()
        config.write(output)
        return output.getvalue()


# Global instance management
_env_loader_instance = None

def get_env_loader(config_dir: str = "config") -> EnvironmentLoader:
    """Get or create environment loader instance"""
    global _env_loader_instance
    
    if _env_loader_instance is None:
        _env_loader_instance = EnvironmentLoader(config_dir)
    
    return _env_loader_instance


def init_env_loader(config_dir: str = "config") -> EnvironmentLoader:
    """Initialize environment loader with custom config directory"""
    global _env_loader_instance
    
    _env_loader_instance = EnvironmentLoader(config_dir)
    
    return _env_loader_instance