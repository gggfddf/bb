"""
Configuration settings for the ML Stock Predictor Platform.
"""

from typing import List, Optional
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # TimescaleDB/PostgreSQL settings
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/stock_predictor",
        description="Database connection URL"
    )
    timescale_url: str = Field(
        default="postgresql://user:password@localhost:5432/stock_predictor",
        description="TimescaleDB connection URL"
    )
    
    # Connection pool settings
    pool_size: int = Field(default=10, description="Database connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Connection pool timeout in seconds")
    
    # TimescaleDB specific settings
    chunk_time_interval: str = Field(
        default="1 day",
        description="TimescaleDB chunk time interval"
    )
    compression_enabled: bool = Field(
        default=True,
        description="Enable TimescaleDB compression"
    )
    compression_after: str = Field(
        default="7 days",
        description="Compress data after this interval"
    )
    
    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    
    # Celery settings
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1",
        description="Celery result backend URL"
    )
    
    # Cache settings
    cache_ttl: int = Field(
        default=300,
        description="Default cache TTL in seconds"
    )
    prediction_cache_ttl: int = Field(
        default=60,
        description="Prediction cache TTL in seconds"
    )
    
    class Config:
        env_prefix = "REDIS_"


class DataCollectionSettings(BaseSettings):
    """Data collection configuration settings."""
    
    # Data sources
    yahoo_finance_enabled: bool = Field(
        default=True,
        description="Enable Yahoo Finance data collection"
    )
    alpha_vantage_enabled: bool = Field(
        default=True,
        description="Enable Alpha Vantage data collection"
    )
    polygon_enabled: bool = Field(
        default=True,
        description="Enable Polygon.io data collection"
    )
    
    # API keys (for free tier access)
    yahoo_finance_api_key: Optional[str] = Field(
        default=None,
        description="Yahoo Finance API key"
    )
    alpha_vantage_api_key: Optional[str] = Field(
        default=None,
        description="Alpha Vantage API key"
    )
    polygon_api_key: Optional[str] = Field(
        default=None,
        description="Polygon.io API key"
    )
    
    # Rate limiting
    requests_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute per source"
    )
    requests_per_second: int = Field(
        default=5,
        description="Maximum requests per second per source"
    )
    
    # Data collection intervals
    real_time_interval: int = Field(
        default=1,
        description="Real-time data collection interval in seconds"
    )
    historical_interval: int = Field(
        default=3600,
        description="Historical data collection interval in seconds"
    )
    
    # Data retention
    data_retention_days: int = Field(
        default=1095,  # 3 years
        description="Data retention period in days"
    )
    
    # WebSocket streaming settings
    websocket_enabled: bool = Field(
        default=False,
        description="Enable WebSocket streaming"
    )
    websocket_url: str = Field(
        default="wss://stream.data.alpaca.markets/v2/iex",
        description="WebSocket streaming URL"
    )
    websocket_api_key: Optional[str] = Field(
        default=None,
        description="WebSocket API key"
    )
    websocket_api_secret: Optional[str] = Field(
        default=None,
        description="WebSocket API secret"
    )
    
    # Data collection parameters
    default_timeframes: List[str] = Field(
        default=["1d", "1h", "15m", "5m", "1m"],
        description="Default timeframes for data collection"
    )
    batch_size: int = Field(
        default=1000,
        description="Batch size for data processing"
    )
    collection_interval_minutes: int = Field(
        default=5,
        description="Data collection interval in minutes"
    )
    
    class Config:
        env_prefix = "DATA_"


class MLModelSettings(BaseSettings):
    """Machine learning model configuration settings."""
    
    # Model storage
    model_cache_dir: str = Field(
        default="./models",
        description="Directory for storing trained models"
    )
    model_versioning: bool = Field(
        default=True,
        description="Enable model versioning"
    )
    
    # Training settings
    training_batch_size: int = Field(
        default=32,
        description="Training batch size"
    )
    training_epochs: int = Field(
        default=100,
        description="Maximum training epochs"
    )
    early_stopping_patience: int = Field(
        default=10,
        description="Early stopping patience"
    )
    
    # Model parameters
    lstm_units: int = Field(
        default=50,
        description="LSTM units"
    )
    lstm_layers: int = Field(
        default=2,
        description="Number of LSTM layers"
    )
    lstm_dropout: float = Field(
        default=0.2,
        description="LSTM dropout rate"
    )
    
    # Random Forest parameters
    rf_n_estimators: int = Field(
        default=100,
        description="Random Forest number of estimators"
    )
    rf_max_depth: int = Field(
        default=10,
        description="Random Forest maximum depth"
    )
    
    # XGBoost parameters
    xgb_learning_rate: float = Field(
        default=0.1,
        description="XGBoost learning rate"
    )
    xgb_max_depth: int = Field(
        default=6,
        description="XGBoost maximum depth"
    )
    
    class Config:
        env_prefix = "ML_"


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    port: int = Field(
        default=8000,
        description="API server port"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # Security settings
    secret_key: str = Field(
        default="your-secret-key-here",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(
        default=100,
        description="API rate limit per minute"
    )
    
    # CORS settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    class Config:
        env_prefix = "API_"


class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration settings."""
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: str = Field(
        default="./logs/app.log",
        description="Log file path"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    
    # Prometheus metrics
    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=9090,
        description="Prometheus metrics port"
    )
    
    # Health checks
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    
    # Alerting
    alert_webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for alerts"
    )
    alert_email: Optional[str] = Field(
        default=None,
        description="Email for alerts"
    )
    
    class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(
        default="development",
        description="Application environment"
    )
    app_name: str = Field(
        default="ML Stock Predictor Platform",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    # Database
    database: DatabaseSettings = DatabaseSettings()
    
    # Redis
    redis: RedisSettings = RedisSettings()
    
    # Data Collection
    data_collection: DataCollectionSettings = DataCollectionSettings()
    
    # ML Models
    ml_models: MLModelSettings = MLModelSettings()
    
    # API
    api: APISettings = APISettings()
    
    # Monitoring
    monitoring: MonitoringSettings = MonitoringSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def validate_settings() -> None:
    """Validate all settings and raise errors if invalid."""
    # Validate database URL
    if not settings.database.database_url:
        raise ValueError("Database URL is required")
    
    # Validate Redis URL
    if not settings.redis.redis_url:
        raise ValueError("Redis URL is required")
    
    # Validate API keys if services are enabled
    if settings.data_collection.alpha_vantage_enabled and not settings.data_collection.alpha_vantage_api_key:
        raise ValueError("Alpha Vantage API key is required when Alpha Vantage is enabled")
    
    if settings.data_collection.polygon_enabled and not settings.data_collection.polygon_api_key:
        raise ValueError("Polygon API key is required when Polygon is enabled")


# Validate settings on import
try:
    validate_settings()
except ValueError as e:
    print(f"Configuration error: {e}")
    raise