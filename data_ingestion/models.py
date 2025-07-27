"""
Database models for the ML Stock Predictor Platform data ingestion system.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import (
    Column, String, DateTime, Numeric, BigInteger, 
    Index, Text, Boolean, Integer, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class StockData(Base):
    """
    Main table for storing stock price data with TimescaleDB hypertable.
    """
    __tablename__ = "stock_data"
    
    # Primary key and timestamps
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Stock identification
    symbol = Column(String(10), nullable=False, index=True)
    exchange = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(5), nullable=False, index=True)  # 1m, 5m, 15m, 1h, 1d, 1w, 1m
    
    # Price data
    open_price = Column(Numeric(10, 4), nullable=True)
    high_price = Column(Numeric(10, 4), nullable=True)
    low_price = Column(Numeric(10, 4), nullable=True)
    close_price = Column(Numeric(10, 4), nullable=True)
    volume = Column(BigInteger, nullable=True)
    
    # Additional data
    adjusted_close = Column(Numeric(10, 4), nullable=True)
    dividend_amount = Column(Numeric(10, 4), nullable=True)
    split_coefficient = Column(Numeric(10, 6), nullable=True)
    
    # Data source and quality
    source = Column(String(50), nullable=False, index=True)
    data_quality_score = Column(Numeric(3, 2), nullable=True)  # 0.00 to 1.00
    is_validated = Column(Boolean, default=False, index=True)
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_stock_data_symbol_time', 'symbol', 'timestamp'),
        Index('idx_stock_data_timeframe_time', 'timeframe', 'timestamp'),
        Index('idx_stock_data_source_time', 'source', 'timestamp'),
        Index('idx_stock_data_quality', 'data_quality_score'),
    )


class Symbol(Base):
    """
    Table for storing symbol metadata and information.
    """
    __tablename__ = "symbols"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    exchange = Column(String(10), nullable=False, index=True)
    
    # Company information
    company_name = Column(String(200), nullable=True)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    market_cap = Column(BigInteger, nullable=True)
    
    # Trading information
    is_active = Column(Boolean, default=True, index=True)
    listing_date = Column(DateTime(timezone=True), nullable=True)
    delisting_date = Column(DateTime(timezone=True), nullable=True)
    
    # Data collection settings
    data_collection_enabled = Column(Boolean, default=True, index=True)
    last_data_update = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    stock_data = relationship("StockData", backref="symbol_info")


class DataSource(Base):
    """
    Table for tracking data sources and their status.
    """
    __tablename__ = "data_sources"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    
    # Source configuration
    source_type = Column(String(20), nullable=False, index=True)  # api, scraping, websocket
    base_url = Column(String(200), nullable=True)
    api_key_required = Column(Boolean, default=False)
    
    # Rate limiting
    requests_per_minute = Column(Integer, default=60)
    requests_per_second = Column(Integer, default=5)
    
    # Status tracking
    is_active = Column(Boolean, default=True, index=True)
    last_successful_request = Column(DateTime(timezone=True), nullable=True)
    last_failed_request = Column(DateTime(timezone=True), nullable=True)
    consecutive_failures = Column(Integer, default=0)
    total_requests = Column(BigInteger, default=0)
    successful_requests = Column(BigInteger, default=0)
    
    # Error tracking
    last_error_message = Column(Text, nullable=True)
    circuit_breaker_status = Column(String(20), default='closed')  # closed, open, half_open
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class DataCollectionJob(Base):
    """
    Table for tracking data collection jobs and their status.
    """
    __tablename__ = "data_collection_jobs"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Job configuration
    job_type = Column(String(50), nullable=False, index=True)  # historical, real_time, batch
    source_id = Column(BigInteger, ForeignKey('data_sources.id'), nullable=False)
    symbol = Column(String(10), nullable=True, index=True)
    timeframe = Column(String(5), nullable=True, index=True)
    
    # Job status
    status = Column(String(20), nullable=False, index=True)  # pending, running, completed, failed
    progress = Column(Numeric(5, 2), default=0.00)  # 0.00 to 100.00
    
    # Job execution
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Results
    records_collected = Column(BigInteger, default=0)
    records_processed = Column(BigInteger, default=0)
    records_failed = Column(BigInteger, default=0)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    source = relationship("DataSource", backref="jobs")


class DataQualityLog(Base):
    """
    Table for logging data quality issues and validation results.
    """
    __tablename__ = "data_quality_logs"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Data identification
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    timeframe = Column(String(5), nullable=False, index=True)
    source = Column(String(50), nullable=False, index=True)
    
    # Quality assessment
    quality_score = Column(Numeric(3, 2), nullable=False)  # 0.00 to 1.00
    validation_status = Column(String(20), nullable=False, index=True)  # passed, failed, warning
    
    # Issue details
    issue_type = Column(String(50), nullable=True, index=True)  # missing_data, outlier, invalid_ohlc, etc.
    issue_description = Column(Text, nullable=True)
    field_name = Column(String(50), nullable=True)
    expected_value = Column(Text, nullable=True)
    actual_value = Column(Text, nullable=True)
    
    # Resolution
    is_resolved = Column(Boolean, default=False, index=True)
    resolution_method = Column(String(50), nullable=True)  # interpolation, removal, manual_fix
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SystemMetrics(Base):
    """
    Table for storing system performance and health metrics.
    """
    __tablename__ = "system_metrics"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Performance metrics
    cpu_usage_percent = Column(Numeric(5, 2), nullable=True)
    memory_usage_percent = Column(Numeric(5, 2), nullable=True)
    disk_usage_percent = Column(Numeric(5, 2), nullable=True)
    network_bytes_sent = Column(BigInteger, nullable=True)
    network_bytes_received = Column(BigInteger, nullable=True)
    
    # Data collection metrics
    data_ingestion_rate = Column(Integer, nullable=True)  # records per second
    active_connections = Column(Integer, nullable=True)
    queue_size = Column(Integer, nullable=True)
    
    # Error metrics
    error_rate = Column(Numeric(5, 2), nullable=True)  # percentage
    failed_requests = Column(Integer, nullable=True)
    successful_requests = Column(Integer, nullable=True)
    
    # Database metrics
    db_connection_count = Column(Integer, nullable=True)
    db_query_time_avg = Column(Numeric(10, 3), nullable=True)  # milliseconds
    db_query_count = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Alert(Base):
    """
    Table for storing system alerts and notifications.
    """
    __tablename__ = "alerts"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Alert information
    alert_type = Column(String(50), nullable=False, index=True)  # error, warning, info
    severity = Column(String(20), nullable=False, index=True)  # critical, high, medium, low
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Alert context
    component = Column(String(50), nullable=True, index=True)  # data_collection, api, database, etc.
    source = Column(String(50), nullable=True, index=True)
    symbol = Column(String(10), nullable=True, index=True)
    
    # Alert status
    status = Column(String(20), default='active', index=True)  # active, acknowledged, resolved
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    
    # Alert metadata
    metadata = Column(Text, nullable=True)  # JSON string for additional data
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# Pydantic models for API responses
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from decimal import Decimal


class StockDataResponse(BaseModel):
    """Pydantic model for stock data API responses."""
    
    symbol: str
    exchange: str
    timestamp: datetime
    timeframe: str
    open_price: Optional[Decimal]
    high_price: Optional[Decimal]
    low_price: Optional[Decimal]
    close_price: Optional[Decimal]
    volume: Optional[int]
    adjusted_close: Optional[Decimal]
    source: str
    data_quality_score: Optional[Decimal]
    
    class Config:
        from_attributes = True


class SymbolResponse(BaseModel):
    """Pydantic model for symbol API responses."""
    
    symbol: str
    exchange: str
    company_name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[int]
    is_active: bool
    data_collection_enabled: bool
    last_data_update: Optional[datetime]
    
    class Config:
        from_attributes = True


class DataSourceResponse(BaseModel):
    """Pydantic model for data source API responses."""
    
    name: str
    display_name: str
    source_type: str
    is_active: bool
    last_successful_request: Optional[datetime]
    consecutive_failures: int
    total_requests: int
    successful_requests: int
    circuit_breaker_status: str
    
    class Config:
        from_attributes = True


class DataCollectionJobResponse(BaseModel):
    """Pydantic model for data collection job API responses."""
    
    job_id: str
    job_type: str
    symbol: Optional[str]
    timeframe: Optional[str]
    status: str
    progress: Decimal
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[int]
    records_collected: int
    records_processed: int
    records_failed: int
    
    class Config:
        from_attributes = True


class SystemMetricsResponse(BaseModel):
    """Pydantic model for system metrics API responses."""
    
    timestamp: datetime
    cpu_usage_percent: Optional[Decimal]
    memory_usage_percent: Optional[Decimal]
    disk_usage_percent: Optional[Decimal]
    data_ingestion_rate: Optional[int]
    error_rate: Optional[Decimal]
    db_connection_count: Optional[int]
    
    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    """Pydantic model for alert API responses."""
    
    alert_type: str
    severity: str
    title: str
    message: str
    component: Optional[str]
    source: Optional[str]
    symbol: Optional[str]
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True