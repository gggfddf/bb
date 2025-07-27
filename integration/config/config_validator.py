"""
Configuration Validator for Trading System

This module provides comprehensive configuration validation functionality,
including schema validation, value validation, and dependency checking.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import jsonschema
from jsonschema import Draft7Validator, ValidationError
import yaml

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation level enumeration"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ConfigType(Enum):
    """Configuration type enumeration"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"
    EMAIL = "email"
    URL = "url"
    IP_ADDRESS = "ip_address"
    PORT = "port"
    PATH = "path"
    DATETIME = "datetime"
    DURATION = "duration"

@dataclass
class ValidationRule:
    """Configuration validation rule"""
    field_path: str
    rule_type: str
    parameters: Dict[str, Any]
    message: str
    level: ValidationLevel = ValidationLevel.ERROR

@dataclass
class ValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    info: List[Dict[str, Any]]
    validated_config: Dict[str, Any]

class ConfigValidator:
    """
    Configuration validator for validating configuration schemas and values.
    """
    
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self.validation_rules: List[ValidationRule] = []
        
        # Built-in validators
        self._register_builtin_validators()
        
        # Built-in schemas
        self._register_builtin_schemas()
        
        logger.info("Configuration validator initialized")
    
    def register_schema(self, schema_name: str, schema: Dict[str, Any]) -> bool:
        """
        Register a JSON schema for validation.
        
        Args:
            schema_name: Name of the schema
            schema: JSON schema definition
            
        Returns:
            True if schema was registered successfully
        """
        try:
            # Validate schema itself
            Draft7Validator.check_schema(schema)
            
            self.schemas[schema_name] = schema
            logger.info(f"Schema registered: {schema_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register schema {schema_name}: {e}")
            return False
    
    def register_custom_validator(self, validator_name: str, validator_func: Callable) -> bool:
        """
        Register a custom validator function.
        
        Args:
            validator_name: Name of the validator
            validator_func: Validator function
            
        Returns:
            True if validator was registered successfully
        """
        try:
            self.custom_validators[validator_name] = validator_func
            logger.info(f"Custom validator registered: {validator_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register custom validator {validator_name}: {e}")
            return False
    
    def add_validation_rule(self, rule: ValidationRule) -> bool:
        """
        Add a validation rule.
        
        Args:
            rule: Validation rule
            
        Returns:
            True if rule was added successfully
        """
        try:
            self.validation_rules.append(rule)
            logger.info(f"Validation rule added: {rule.field_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add validation rule: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any], schema_name: Optional[str] = None) -> ValidationResult:
        """
        Validate configuration against schema and rules.
        
        Args:
            config: Configuration to validate
            schema_name: Optional schema name to validate against
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        info = []
        
        try:
            # Schema validation
            if schema_name and schema_name in self.schemas:
                schema_errors = self._validate_schema(config, self.schemas[schema_name])
                errors.extend(schema_errors)
            
            # Custom validation rules
            rule_errors, rule_warnings, rule_info = self._validate_rules(config)
            errors.extend(rule_errors)
            warnings.extend(rule_warnings)
            info.extend(rule_info)
            
            # Built-in validations
            builtin_errors, builtin_warnings, builtin_info = self._validate_builtin(config)
            errors.extend(builtin_errors)
            warnings.extend(builtin_warnings)
            info.extend(builtin_info)
            
            is_valid = len([e for e in errors if e.get('level') == ValidationLevel.ERROR.value]) == 0
            
            result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                info=info,
                validated_config=config
            )
            
            if is_valid:
                logger.info("Configuration validation passed")
            else:
                logger.warning(f"Configuration validation failed with {len(errors)} errors")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during configuration validation: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[{'message': f'Validation error: {e}', 'level': ValidationLevel.ERROR.value}],
                warnings=[],
                info=[],
                validated_config=config
            )
    
    def validate_field(self, field_path: str, value: Any, field_type: ConfigType, 
                      constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Validate a single field.
        
        Args:
            field_path: Path to the field
            value: Field value
            field_type: Expected field type
            constraints: Optional constraints
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            # Type validation
            type_errors = self._validate_type(field_path, value, field_type)
            errors.extend(type_errors)
            
            # Constraint validation
            if constraints:
                constraint_errors = self._validate_constraints(field_path, value, constraints)
                errors.extend(constraint_errors)
            
            return errors
            
        except Exception as e:
            logger.error(f"Error validating field {field_path}: {e}")
            return [{'field': field_path, 'message': f'Validation error: {e}', 'level': ValidationLevel.ERROR.value}]
    
    def _validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate configuration against JSON schema"""
        errors = []
        
        try:
            validator = Draft7Validator(schema)
            for error in validator.iter_errors(config):
                errors.append({
                    'field': '.'.join(str(p) for p in error.path),
                    'message': error.message,
                    'level': ValidationLevel.ERROR.value,
                    'schema_path': '.'.join(str(p) for p in error.schema_path)
                })
        except Exception as e:
            errors.append({
                'field': 'schema',
                'message': f'Schema validation error: {e}',
                'level': ValidationLevel.ERROR.value
            })
        
        return errors
    
    def _validate_rules(self, config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate configuration against custom rules"""
        errors = []
        warnings = []
        info = []
        
        for rule in self.validation_rules:
            try:
                field_value = self._get_field_value(config, rule.field_path)
                
                if rule.rule_type in self.custom_validators:
                    validator_func = self.custom_validators[rule.rule_type]
                    is_valid = validator_func(field_value, rule.parameters)
                    
                    if not is_valid:
                        validation_result = {
                            'field': rule.field_path,
                            'message': rule.message,
                            'level': rule.level.value,
                            'rule_type': rule.rule_type
                        }
                        
                        if rule.level == ValidationLevel.ERROR:
                            errors.append(validation_result)
                        elif rule.level == ValidationLevel.WARNING:
                            warnings.append(validation_result)
                        else:
                            info.append(validation_result)
                            
            except Exception as e:
                logger.error(f"Error validating rule {rule.field_path}: {e}")
                errors.append({
                    'field': rule.field_path,
                    'message': f'Rule validation error: {e}',
                    'level': ValidationLevel.ERROR.value
                })
        
        return errors, warnings, info
    
    def _validate_builtin(self, config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply built-in validations"""
        errors = []
        warnings = []
        info = []
        
        # Database configuration validation
        if 'database' in config:
            db_errors = self._validate_database_config(config['database'])
            errors.extend(db_errors)
        
        # API configuration validation
        if 'api' in config:
            api_errors = self._validate_api_config(config['api'])
            errors.extend(api_errors)
        
        # Security configuration validation
        if 'security' in config:
            security_errors = self._validate_security_config(config['security'])
            errors.extend(security_errors)
        
        # Logging configuration validation
        if 'logging' in config:
            logging_errors = self._validate_logging_config(config['logging'])
            errors.extend(logging_errors)
        
        return errors, warnings, info
    
    def _validate_type(self, field_path: str, value: Any, field_type: ConfigType) -> List[Dict[str, Any]]:
        """Validate field type"""
        errors = []
        
        try:
            if field_type == ConfigType.STRING and not isinstance(value, str):
                errors.append({
                    'field': field_path,
                    'message': f'Expected string, got {type(value).__name__}',
                    'level': ValidationLevel.ERROR.value
                })
            elif field_type == ConfigType.INTEGER and not isinstance(value, int):
                errors.append({
                    'field': field_path,
                    'message': f'Expected integer, got {type(value).__name__}',
                    'level': ValidationLevel.ERROR.value
                })
            elif field_type == ConfigType.FLOAT and not isinstance(value, (int, float)):
                errors.append({
                    'field': field_path,
                    'message': f'Expected float, got {type(value).__name__}',
                    'level': ValidationLevel.ERROR.value
                })
            elif field_type == ConfigType.BOOLEAN and not isinstance(value, bool):
                errors.append({
                    'field': field_path,
                    'message': f'Expected boolean, got {type(value).__name__}',
                    'level': ValidationLevel.ERROR.value
                })
            elif field_type == ConfigType.ARRAY and not isinstance(value, list):
                errors.append({
                    'field': field_path,
                    'message': f'Expected array, got {type(value).__name__}',
                    'level': ValidationLevel.ERROR.value
                })
            elif field_type == ConfigType.OBJECT and not isinstance(value, dict):
                errors.append({
                    'field': field_path,
                    'message': f'Expected object, got {type(value).__name__}',
                    'level': ValidationLevel.ERROR.value
                })
            elif field_type == ConfigType.EMAIL and isinstance(value, str):
                if not self._is_valid_email(value):
                    errors.append({
                        'field': field_path,
                        'message': 'Invalid email format',
                        'level': ValidationLevel.ERROR.value
                    })
            elif field_type == ConfigType.URL and isinstance(value, str):
                if not self._is_valid_url(value):
                    errors.append({
                        'field': field_path,
                        'message': 'Invalid URL format',
                        'level': ValidationLevel.ERROR.value
                    })
            elif field_type == ConfigType.IP_ADDRESS and isinstance(value, str):
                if not self._is_valid_ip_address(value):
                    errors.append({
                        'field': field_path,
                        'message': 'Invalid IP address format',
                        'level': ValidationLevel.ERROR.value
                    })
            elif field_type == ConfigType.PORT and isinstance(value, int):
                if not (1 <= value <= 65535):
                    errors.append({
                        'field': field_path,
                        'message': 'Port must be between 1 and 65535',
                        'level': ValidationLevel.ERROR.value
                    })
            
        except Exception as e:
            errors.append({
                'field': field_path,
                'message': f'Type validation error: {e}',
                'level': ValidationLevel.ERROR.value
            })
        
        return errors
    
    def _validate_constraints(self, field_path: str, value: Any, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate field constraints"""
        errors = []
        
        try:
            # Min/Max constraints
            if 'min' in constraints and value < constraints['min']:
                errors.append({
                    'field': field_path,
                    'message': f'Value must be >= {constraints["min"]}',
                    'level': ValidationLevel.ERROR.value
                })
            
            if 'max' in constraints and value > constraints['max']:
                errors.append({
                    'field': field_path,
                    'message': f'Value must be <= {constraints["max"]}',
                    'level': ValidationLevel.ERROR.value
                })
            
            # Length constraints
            if 'min_length' in constraints and len(str(value)) < constraints['min_length']:
                errors.append({
                    'field': field_path,
                    'message': f'Length must be >= {constraints["min_length"]}',
                    'level': ValidationLevel.ERROR.value
                })
            
            if 'max_length' in constraints and len(str(value)) > constraints['max_length']:
                errors.append({
                    'field': field_path,
                    'message': f'Length must be <= {constraints["max_length"]}',
                    'level': ValidationLevel.ERROR.value
                })
            
            # Pattern constraints
            if 'pattern' in constraints and isinstance(value, str):
                if not re.match(constraints['pattern'], value):
                    errors.append({
                        'field': field_path,
                        'message': f'Value does not match pattern: {constraints["pattern"]}',
                        'level': ValidationLevel.ERROR.value
                    })
            
            # Enum constraints
            if 'enum' in constraints and value not in constraints['enum']:
                errors.append({
                    'field': field_path,
                    'message': f'Value must be one of: {constraints["enum"]}',
                    'level': ValidationLevel.ERROR.value
                })
            
            # Required constraints
            if constraints.get('required', False) and value is None:
                errors.append({
                    'field': field_path,
                    'message': 'Field is required',
                    'level': ValidationLevel.ERROR.value
                })
            
        except Exception as e:
            errors.append({
                'field': field_path,
                'message': f'Constraint validation error: {e}',
                'level': ValidationLevel.ERROR.value
            })
        
        return errors
    
    def _validate_database_config(self, db_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate database configuration"""
        errors = []
        
        required_fields = ['host', 'port', 'database', 'username']
        for field in required_fields:
            if field not in db_config:
                errors.append({
                    'field': f'database.{field}',
                    'message': f'Database {field} is required',
                    'level': ValidationLevel.ERROR.value
                })
        
        if 'port' in db_config:
            port_errors = self.validate_field('database.port', db_config['port'], ConfigType.PORT)
            errors.extend(port_errors)
        
        if 'host' in db_config:
            host_errors = self.validate_field('database.host', db_config['host'], ConfigType.IP_ADDRESS)
            errors.extend(host_errors)
        
        return errors
    
    def _validate_api_config(self, api_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate API configuration"""
        errors = []
        
        if 'port' in api_config:
            port_errors = self.validate_field('api.port', api_config['port'], ConfigType.PORT)
            errors.extend(port_errors)
        
        if 'host' in api_config:
            host_errors = self.validate_field('api.host', api_config['host'], ConfigType.IP_ADDRESS)
            errors.extend(host_errors)
        
        return errors
    
    def _validate_security_config(self, security_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate security configuration"""
        errors = []
        
        if 'jwt_secret' in security_config:
            if len(security_config['jwt_secret']) < 32:
                errors.append({
                    'field': 'security.jwt_secret',
                    'message': 'JWT secret must be at least 32 characters long',
                    'level': ValidationLevel.ERROR.value
                })
        
        return errors
    
    def _validate_logging_config(self, logging_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate logging configuration"""
        errors = []
        
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if 'level' in logging_config and logging_config['level'] not in valid_levels:
            errors.append({
                'field': 'logging.level',
                'message': f'Log level must be one of: {valid_levels}',
                'level': ValidationLevel.ERROR.value
            })
        
        return errors
    
    def _get_field_value(self, config: Dict[str, Any], field_path: str) -> Any:
        """Get field value by path"""
        keys = field_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Field {field_path} not found")
        
        return value
    
    def _is_valid_email(self, email: str) -> bool:
        """Check if email is valid"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return re.match(pattern, url) is not None
    
    def _is_valid_ip_address(self, ip: str) -> bool:
        """Check if IP address is valid"""
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if not re.match(pattern, ip):
            return False
        
        parts = ip.split('.')
        return all(0 <= int(part) <= 255 for part in parts)
    
    def _register_builtin_validators(self):
        """Register built-in custom validators"""
        
        def validate_positive(value, params):
            return value > 0
        
        def validate_non_negative(value, params):
            return value >= 0
        
        def validate_percentage(value, params):
            return 0 <= value <= 100
        
        def validate_timeout(value, params):
            return 0 < value <= 3600  # 1 hour max
        
        self.register_custom_validator('positive', validate_positive)
        self.register_custom_validator('non_negative', validate_non_negative)
        self.register_custom_validator('percentage', validate_percentage)
        self.register_custom_validator('timeout', validate_timeout)
    
    def _register_builtin_schemas(self):
        """Register built-in schemas"""
        
        # Database schema
        database_schema = {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "database": {"type": "string"},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "pool_size": {"type": "integer", "minimum": 1, "maximum": 100}
            },
            "required": ["host", "port", "database", "username"]
        }
        
        # API schema
        api_schema = {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "debug": {"type": "boolean"},
                "cors_origins": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["host", "port"]
        }
        
        self.register_schema("database", database_schema)
        self.register_schema("api", api_schema)


# Global instance management
_config_validator_instance = None

def get_config_validator() -> ConfigValidator:
    """Get or create config validator instance"""
    global _config_validator_instance
    
    if _config_validator_instance is None:
        _config_validator_instance = ConfigValidator()
    
    return _config_validator_instance


def init_config_validator() -> ConfigValidator:
    """Initialize config validator"""
    global _config_validator_instance
    
    if _config_validator_instance:
        _config_validator_instance = ConfigValidator()
    
    return _config_validator_instance