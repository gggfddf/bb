"""
Configuration Encryption System for Trading System

This module provides configuration encryption functionality for securing
sensitive configuration values like passwords, API keys, and secrets.
"""

import os
import base64
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import hmac

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

class EncryptionType(Enum):
    """Encryption type enumeration"""
    FERNET = "fernet"
    AES = "aes"
    CUSTOM = "custom"

@dataclass
class EncryptedValue:
    """Encrypted value structure"""
    encrypted_data: str
    encryption_type: EncryptionType
    salt: Optional[str]
    iv: Optional[str]
    created_at: datetime
    metadata: Dict[str, Any]

class ConfigEncryption:
    """
    Configuration encryption system for securing sensitive configuration values.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.getenv('CONFIG_MASTER_KEY')
        self.fernet_key = None
        self.aes_key = None
        
        # Encryption settings
        self.default_encryption_type = EncryptionType.FERNET
        self.key_derivation_salt = os.getenv('CONFIG_KEY_SALT', 'default_salt_change_in_production')
        
        # Initialize encryption keys
        self._initialize_keys()
        
        logger.info("Configuration encryption system initialized")
    
    def encrypt_value(self, value: str, encryption_type: Optional[EncryptionType] = None) -> EncryptedValue:
        """
        Encrypt a configuration value.
        
        Args:
            value: Value to encrypt
            encryption_type: Type of encryption to use
            
        Returns:
            Encrypted value
        """
        if not value:
            raise ValueError("Value cannot be empty")
        
        encryption_type = encryption_type or self.default_encryption_type
        
        try:
            if encryption_type == EncryptionType.FERNET:
                return self._encrypt_fernet(value)
            elif encryption_type == EncryptionType.AES:
                return self._encrypt_aes(value)
            elif encryption_type == EncryptionType.CUSTOM:
                return self._encrypt_custom(value)
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            raise
    
    def decrypt_value(self, encrypted_value: EncryptedValue) -> str:
        """
        Decrypt a configuration value.
        
        Args:
            encrypted_value: Encrypted value to decrypt
            
        Returns:
            Decrypted value
        """
        try:
            if encrypted_value.encryption_type == EncryptionType.FERNET:
                return self._decrypt_fernet(encrypted_value)
            elif encrypted_value.encryption_type == EncryptionType.AES:
                return self._decrypt_aes(encrypted_value)
            elif encrypted_value.encryption_type == EncryptionType.CUSTOM:
                return self._decrypt_custom(encrypted_value)
            else:
                raise ValueError(f"Unsupported encryption type: {encrypted_value.encryption_type}")
                
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            raise
    
    def encrypt_config_section(self, config_section: Dict[str, Any], 
                              sensitive_keys: List[str]) -> Dict[str, Any]:
        """
        Encrypt sensitive values in a configuration section.
        
        Args:
            config_section: Configuration section to encrypt
            sensitive_keys: List of keys to encrypt
            
        Returns:
            Configuration section with encrypted values
        """
        encrypted_config = config_section.copy()
        
        for key in sensitive_keys:
            if key in encrypted_config and encrypted_config[key]:
                value = str(encrypted_config[key])
                encrypted_value = self.encrypt_value(value)
                encrypted_config[key] = asdict(encrypted_value)
        
        return encrypted_config
    
    def decrypt_config_section(self, config_section: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted values in a configuration section.
        
        Args:
            config_section: Configuration section to decrypt
            
        Returns:
            Configuration section with decrypted values
        """
        decrypted_config = config_section.copy()
        
        for key, value in decrypted_config.items():
            if isinstance(value, dict) and 'encrypted_data' in value:
                try:
                    encrypted_value = EncryptedValue(
                        encrypted_data=value['encrypted_data'],
                        encryption_type=EncryptionType(value['encryption_type']),
                        salt=value.get('salt'),
                        iv=value.get('iv'),
                        created_at=datetime.fromisoformat(value['created_at']),
                        metadata=value.get('metadata', {})
                    )
                    decrypted_config[key] = self.decrypt_value(encrypted_value)
                except Exception as e:
                    logger.warning(f"Failed to decrypt value for key {key}: {e}")
                    # Keep encrypted value if decryption fails
        
        return decrypted_config
    
    def is_encrypted(self, value: Any) -> bool:
        """
        Check if a value is encrypted.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is encrypted
        """
        if isinstance(value, dict):
            return 'encrypted_data' in value and 'encryption_type' in value
        return False
    
    def get_encryption_info(self, encrypted_value: EncryptedValue) -> Dict[str, Any]:
        """
        Get information about an encrypted value.
        
        Args:
            encrypted_value: Encrypted value
            
        Returns:
            Encryption information
        """
        return {
            'encryption_type': encrypted_value.encryption_type.value,
            'created_at': encrypted_value.created_at.isoformat(),
            'metadata': encrypted_value.metadata,
            'has_salt': encrypted_value.salt is not None,
            'has_iv': encrypted_value.iv is not None
        }
    
    def rotate_keys(self, new_master_key: str) -> bool:
        """
        Rotate encryption keys.
        
        Args:
            new_master_key: New master key
            
        Returns:
            True if keys were rotated successfully
        """
        try:
            old_master_key = self.master_key
            self.master_key = new_master_key
            
            # Reinitialize keys with new master key
            self._initialize_keys()
            
            logger.info("Encryption keys rotated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate encryption keys: {e}")
            # Restore old key
            self.master_key = old_master_key
            return False
    
    def _initialize_keys(self):
        """Initialize encryption keys"""
        if not self.master_key:
            raise ValueError("Master key is required for encryption")
        
        try:
            # Generate Fernet key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.key_derivation_salt.encode(),
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            self.fernet_key = Fernet(key)
            
            # Generate AES key
            kdf_aes = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=(self.key_derivation_salt + "_aes").encode(),
                iterations=100000,
                backend=default_backend()
            )
            self.aes_key = kdf_aes.derive(self.master_key.encode())
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption keys: {e}")
            raise
    
    def _encrypt_fernet(self, value: str) -> EncryptedValue:
        """Encrypt value using Fernet"""
        encrypted_data = self.fernet_key.encrypt(value.encode())
        
        return EncryptedValue(
            encrypted_data=base64.urlsafe_b64encode(encrypted_data).decode(),
            encryption_type=EncryptionType.FERNET,
            salt=None,
            iv=None,
            created_at=datetime.utcnow(),
            metadata={'method': 'fernet'}
        )
    
    def _decrypt_fernet(self, encrypted_value: EncryptedValue) -> str:
        """Decrypt value using Fernet"""
        encrypted_data = base64.urlsafe_b64decode(encrypted_value.encrypted_data.encode())
        decrypted_data = self.fernet_key.decrypt(encrypted_data)
        return decrypted_data.decode()
    
    def _encrypt_aes(self, value: str) -> EncryptedValue:
        """Encrypt value using AES"""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        # Generate a random IV
        iv = os.urandom(12)
        
        # Create AESGCM cipher
        aesgcm = AESGCM(self.aes_key)
        
        # Encrypt the data
        encrypted_data = aesgcm.encrypt(iv, value.encode(), None)
        
        return EncryptedValue(
            encrypted_data=base64.urlsafe_b64encode(encrypted_data).decode(),
            encryption_type=EncryptionType.AES,
            salt=None,
            iv=base64.urlsafe_b64encode(iv).decode(),
            created_at=datetime.utcnow(),
            metadata={'method': 'aes_gcm'}
        )
    
    def _decrypt_aes(self, encrypted_value: EncryptedValue) -> str:
        """Decrypt value using AES"""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        # Decode IV and encrypted data
        iv = base64.urlsafe_b64decode(encrypted_value.iv.encode())
        encrypted_data = base64.urlsafe_b64decode(encrypted_value.encrypted_data.encode())
        
        # Create AESGCM cipher
        aesgcm = AESGCM(self.aes_key)
        
        # Decrypt the data
        decrypted_data = aesgcm.decrypt(iv, encrypted_data, None)
        return decrypted_data.decode()
    
    def _encrypt_custom(self, value: str) -> EncryptedValue:
        """Custom encryption method"""
        # Generate a random salt
        salt = os.urandom(16)
        
        # Create a custom key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(self.master_key.encode())
        
        # Generate a random IV
        iv = os.urandom(16)
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Pad the data
        padded_data = self._pad_data(value.encode())
        
        # Encrypt
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptedValue(
            encrypted_data=base64.urlsafe_b64encode(encrypted_data).decode(),
            encryption_type=EncryptionType.CUSTOM,
            salt=base64.urlsafe_b64encode(salt).decode(),
            iv=base64.urlsafe_b64encode(iv).decode(),
            created_at=datetime.utcnow(),
            metadata={'method': 'custom_aes_cbc'}
        )
    
    def _decrypt_custom(self, encrypted_value: EncryptedValue) -> str:
        """Custom decryption method"""
        # Decode salt, IV, and encrypted data
        salt = base64.urlsafe_b64decode(encrypted_value.salt.encode())
        iv = base64.urlsafe_b64decode(encrypted_value.iv.encode())
        encrypted_data = base64.urlsafe_b64decode(encrypted_value.encrypted_data.encode())
        
        # Recreate key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(self.master_key.encode())
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Decrypt
        decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Unpad the data
        unpadded_data = self._unpad_data(decrypted_data)
        return unpadded_data.decode()
    
    def _pad_data(self, data: bytes) -> bytes:
        """Pad data for AES encryption"""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, data: bytes) -> bytes:
        """Unpad data after AES decryption"""
        padding_length = data[-1]
        return data[:-padding_length]


class ConfigEncryptionManager:
    """
    Manager for configuration encryption operations.
    """
    
    def __init__(self, encryption: ConfigEncryption):
        self.encryption = encryption
        self.encrypted_sections: Dict[str, List[str]] = {}
        
        logger.info("Configuration encryption manager initialized")
    
    def register_sensitive_section(self, section_name: str, sensitive_keys: List[str]):
        """
        Register a configuration section with sensitive keys.
        
        Args:
            section_name: Name of the configuration section
            sensitive_keys: List of sensitive keys in the section
        """
        self.encrypted_sections[section_name] = sensitive_keys
        logger.info(f"Registered sensitive section: {section_name} with {len(sensitive_keys)} keys")
    
    def encrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt sensitive values in a configuration.
        
        Args:
            config: Configuration to encrypt
            
        Returns:
            Configuration with encrypted values
        """
        encrypted_config = config.copy()
        
        for section_name, sensitive_keys in self.encrypted_sections.items():
            if section_name in encrypted_config:
                encrypted_config[section_name] = self.encryption.encrypt_config_section(
                    encrypted_config[section_name], sensitive_keys
                )
        
        return encrypted_config
    
    def decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted values in a configuration.
        
        Args:
            config: Configuration to decrypt
            
        Returns:
            Configuration with decrypted values
        """
        decrypted_config = config.copy()
        
        for section_name in self.encrypted_sections.keys():
            if section_name in decrypted_config:
                decrypted_config[section_name] = self.encryption.decrypt_config_section(
                    decrypted_config[section_name]
                )
        
        return decrypted_config
    
    def get_encryption_stats(self) -> Dict[str, Any]:
        """
        Get encryption statistics.
        
        Returns:
            Encryption statistics
        """
        total_sections = len(self.encrypted_sections)
        total_keys = sum(len(keys) for keys in self.encrypted_sections.values())
        
        return {
            'total_sections': total_sections,
            'total_sensitive_keys': total_keys,
            'sections': list(self.encrypted_sections.keys())
        }


# Global instance management
_config_encryption_instance = None

def get_config_encryption(master_key: Optional[str] = None) -> ConfigEncryption:
    """Get or create config encryption instance"""
    global _config_encryption_instance
    
    if _config_encryption_instance is None:
        _config_encryption_instance = ConfigEncryption(master_key)
    
    return _config_encryption_instance


def init_config_encryption(master_key: str) -> ConfigEncryption:
    """Initialize config encryption with master key"""
    global _config_encryption_instance
    
    _config_encryption_instance = ConfigEncryption(master_key)
    
    return _config_encryption_instance


def get_config_encryption_manager(master_key: Optional[str] = None) -> ConfigEncryptionManager:
    """Get or create config encryption manager instance"""
    encryption = get_config_encryption(master_key)
    return ConfigEncryptionManager(encryption)