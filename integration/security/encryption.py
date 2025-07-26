"""
Encryption and Decryption System for ML Stock Predictor Platform

This module provides comprehensive encryption and decryption functionality
for securing sensitive data in the trading system.
"""

import os
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import json
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    Comprehensive encryption and decryption manager for the trading system.
    
    Supports:
    - Symmetric encryption (AES-256)
    - Asymmetric encryption (RSA)
    - Password-based key derivation
    - Data hashing and verification
    - Secure key management
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize the encryption manager.
        
        Args:
            master_key: Optional master key for key derivation
        """
        self.master_key = master_key or os.getenv('MASTER_ENCRYPTION_KEY')
        self.fernet_key = None
        self.rsa_private_key = None
        self.rsa_public_key = None
        self._initialize_keys()
        
    def _initialize_keys(self):
        """Initialize encryption keys."""
        try:
            if self.master_key:
                # Derive Fernet key from master key
                salt = b'ml_stock_predictor_salt'
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
                self.fernet_key = Fernet(key)
            else:
                # Generate new Fernet key
                self.fernet_key = Fernet.generate_key()
                fernet = Fernet(self.fernet_key)
                
            # Generate RSA key pair
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            self.rsa_public_key = self.rsa_private_key.public_key()
            
            logger.info("Encryption keys initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption keys: {e}")
            raise
    
    def encrypt_symmetric(self, data: Union[str, bytes, Dict]) -> Dict[str, str]:
        """
        Encrypt data using symmetric encryption (AES-256).
        
        Args:
            data: Data to encrypt (string, bytes, or dictionary)
            
        Returns:
            Dictionary containing encrypted data and metadata
        """
        try:
            # Convert data to JSON string if it's a dictionary
            if isinstance(data, dict):
                data_str = json.dumps(data)
            elif isinstance(data, bytes):
                data_str = data.decode('utf-8')
            else:
                data_str = str(data)
            
            # Encrypt the data
            encrypted_data = self.fernet_key.encrypt(data_str.encode())
            
            # Create metadata
            metadata = {
                'encrypted_data': base64.b64encode(encrypted_data).decode(),
                'encryption_type': 'symmetric',
                'algorithm': 'AES-256',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
            
            logger.info(f"Data encrypted successfully using symmetric encryption")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_symmetric(self, encrypted_metadata: Dict[str, str]) -> Union[str, Dict]:
        """
        Decrypt data using symmetric encryption.
        
        Args:
            encrypted_metadata: Dictionary containing encrypted data and metadata
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_data = base64.b64decode(encrypted_metadata['encrypted_data'])
            decrypted_data = self.fernet_key.decrypt(encrypted_data)
            
            # Try to parse as JSON, otherwise return as string
            try:
                result = json.loads(decrypted_data.decode())
            except json.JSONDecodeError:
                result = decrypted_data.decode()
            
            logger.info("Data decrypted successfully using symmetric encryption")
            return result
            
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def encrypt_asymmetric(self, data: Union[str, bytes, Dict]) -> Dict[str, str]:
        """
        Encrypt data using asymmetric encryption (RSA).
        
        Args:
            data: Data to encrypt
            
        Returns:
            Dictionary containing encrypted data and metadata
        """
        try:
            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode()
            elif isinstance(data, str):
                data_bytes = data.encode()
            else:
                data_bytes = data
            
            # Encrypt with public key
            encrypted_data = self.rsa_public_key.encrypt(
                data_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            metadata = {
                'encrypted_data': base64.b64encode(encrypted_data).decode(),
                'encryption_type': 'asymmetric',
                'algorithm': 'RSA-2048',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
            
            logger.info("Data encrypted successfully using asymmetric encryption")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_asymmetric(self, encrypted_metadata: Dict[str, str]) -> Union[str, Dict]:
        """
        Decrypt data using asymmetric encryption.
        
        Args:
            encrypted_metadata: Dictionary containing encrypted data and metadata
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_data = base64.b64decode(encrypted_metadata['encrypted_data'])
            
            decrypted_data = self.rsa_private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Try to parse as JSON, otherwise return as string
            try:
                result = json.loads(decrypted_data.decode())
            except json.JSONDecodeError:
                result = decrypted_data.decode()
            
            logger.info("Data decrypted successfully using asymmetric encryption")
            return result
            
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def hash_data(self, data: Union[str, bytes, Dict], salt: Optional[str] = None) -> Dict[str, str]:
        """
        Create a secure hash of data.
        
        Args:
            data: Data to hash
            salt: Optional salt for additional security
            
        Returns:
            Dictionary containing hash and metadata
        """
        try:
            # Convert data to string
            if isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True)
            elif isinstance(data, bytes):
                data_str = data.decode()
            else:
                data_str = str(data)
            
            # Add salt if provided
            if salt:
                data_str = salt + data_str
            
            # Create hash
            hash_obj = hashlib.sha256()
            hash_obj.update(data_str.encode())
            data_hash = hash_obj.hexdigest()
            
            metadata = {
                'hash': data_hash,
                'algorithm': 'SHA-256',
                'salt': salt,
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
            
            logger.info("Data hashed successfully")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to hash data: {e}")
            raise
    
    def verify_hash(self, data: Union[str, bytes, Dict], hash_metadata: Dict[str, str]) -> bool:
        """
        Verify data against a hash.
        
        Args:
            data: Data to verify
            hash_metadata: Dictionary containing hash and metadata
            
        Returns:
            True if hash matches, False otherwise
        """
        try:
            # Create hash of input data
            input_hash_metadata = self.hash_data(data, hash_metadata.get('salt'))
            
            # Compare hashes
            is_valid = input_hash_metadata['hash'] == hash_metadata['hash']
            
            logger.info(f"Hash verification {'successful' if is_valid else 'failed'}")
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to verify hash: {e}")
            return False
    
    def generate_secure_token(self, data: Dict[str, Any], expiry_hours: int = 24) -> str:
        """
        Generate a secure token for data.
        
        Args:
            data: Data to include in token
            expiry_hours: Token expiry time in hours
            
        Returns:
            Secure token string
        """
        try:
            # Add expiry timestamp
            data['expires_at'] = (datetime.utcnow() + timedelta(hours=expiry_hours)).isoformat()
            
            # Encrypt the data
            encrypted_metadata = self.encrypt_symmetric(data)
            
            # Create token
            token_data = {
                'encrypted_payload': encrypted_metadata['encrypted_data'],
                'created_at': datetime.utcnow().isoformat(),
                'expires_at': data['expires_at']
            }
            
            # Encode token
            token = base64.urlsafe_b64encode(json.dumps(token_data).encode()).decode()
            
            logger.info(f"Secure token generated successfully, expires in {expiry_hours} hours")
            return token
            
        except Exception as e:
            logger.error(f"Failed to generate secure token: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a secure token.
        
        Args:
            token: Token to verify
            
        Returns:
            Decoded data if valid, None if invalid or expired
        """
        try:
            # Decode token
            token_data = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
            
            # Check expiry
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if datetime.utcnow() > expires_at:
                logger.warning("Token has expired")
                return None
            
            # Decrypt payload
            encrypted_metadata = {
                'encrypted_data': token_data['encrypted_payload'],
                'encryption_type': 'symmetric',
                'algorithm': 'AES-256',
                'timestamp': token_data['created_at'],
                'version': '1.0'
            }
            
            data = self.decrypt_symmetric(encrypted_metadata)
            
            logger.info("Token verified successfully")
            return data
            
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return None
    
    def export_public_key(self) -> str:
        """
        Export the public key for sharing.
        
        Returns:
            PEM-encoded public key
        """
        try:
            pem = self.rsa_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return pem.decode()
        except Exception as e:
            logger.error(f"Failed to export public key: {e}")
            raise
    
    def import_public_key(self, public_key_pem: str):
        """
        Import a public key.
        
        Args:
            public_key_pem: PEM-encoded public key
        """
        try:
            self.rsa_public_key = serialization.load_pem_public_key(
                public_key_pem.encode()
            )
            logger.info("Public key imported successfully")
        except Exception as e:
            logger.error(f"Failed to import public key: {e}")
            raise


class DataEncryption:
    """
    High-level data encryption utilities for common use cases.
    """
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
    
    def encrypt_user_credentials(self, username: str, password: str) -> Dict[str, str]:
        """
        Encrypt user credentials securely.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Encrypted credentials
        """
        credentials = {
            'username': username,
            'password': password,
            'created_at': datetime.utcnow().isoformat()
        }
        
        return self.encryption_manager.encrypt_symmetric(credentials)
    
    def encrypt_trading_data(self, trading_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Encrypt sensitive trading data.
        
        Args:
            trading_data: Trading data to encrypt
            
        Returns:
            Encrypted trading data
        """
        return self.encryption_manager.encrypt_symmetric(trading_data)
    
    def encrypt_api_keys(self, api_keys: Dict[str, str]) -> Dict[str, str]:
        """
        Encrypt API keys.
        
        Args:
            api_keys: Dictionary of API keys
            
        Returns:
            Encrypted API keys
        """
        return self.encryption_manager.encrypt_symmetric(api_keys)
    
    def encrypt_configuration(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Encrypt configuration data.
        
        Args:
            config: Configuration data to encrypt
            
        Returns:
            Encrypted configuration
        """
        return self.encryption_manager.encrypt_symmetric(config)


# Global encryption manager instance
_encryption_manager = None
_data_encryption = None


def get_encryption_manager() -> EncryptionManager:
    """Get the global encryption manager instance."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def get_data_encryption() -> DataEncryption:
    """Get the global data encryption instance."""
    global _data_encryption
    if _data_encryption is None:
        _data_encryption = DataEncryption(get_encryption_manager())
    return _data_encryption


# Example usage and testing
if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Test encryption functionality
    encryption_mgr = EncryptionManager("test_master_key")
    data_encryption = DataEncryption(encryption_mgr)
    
    # Test symmetric encryption
    test_data = {"user_id": 123, "action": "buy", "amount": 1000}
    encrypted = encryption_mgr.encrypt_symmetric(test_data)
    decrypted = encryption_mgr.decrypt_symmetric(encrypted)
    print(f"Original: {test_data}")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {test_data == decrypted}")
    
    # Test hashing
    hash_metadata = encryption_mgr.hash_data(test_data, "test_salt")
    is_valid = encryption_mgr.verify_hash(test_data, hash_metadata)
    print(f"Hash valid: {is_valid}")
    
    # Test token generation
    token = encryption_mgr.generate_secure_token(test_data, 1)
    token_data = encryption_mgr.verify_token(token)
    print(f"Token valid: {token_data is not None}")
    
    print("Encryption system test completed successfully!")