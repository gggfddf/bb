"""
Yahoo Finance scraper for collecting stock data.
This module implements web scraping and API access to Yahoo Finance.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session

from config.settings import settings
from data_ingestion.models import StockData, Symbol, DataSource
from data_ingestion.utils.rate_limiter import RateLimiter
from data_ingestion.utils.data_validator import DataValidator
from data_ingestion.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class YahooFinanceScraper:
    """
    Yahoo Finance data scraper with rate limiting and error handling.
    """
    
    def __init__(self, session: Session):
        self.session = session
        self.rate_limiter = RateLimiter(
            requests_per_minute=settings.data_collection.requests_per_minute,
            requests_per_second=settings.data_collection.requests_per_second
        )
        self.data_validator = DataValidator()
        self.error_handler = ErrorHandler()
        self.source_name = "yahoo_finance"
        
        # Initialize data source record
        self._init_data_source()
    
    def _init_data_source(self):
        """Initialize or update the data source record."""
        try:
            source = self.session.query(DataSource).filter_by(name=self.source_name).first()
            if not source:
                source = DataSource(
                    name=self.source_name,
                    display_name="Yahoo Finance",
                    source_type="api",
                    base_url="https://finance.yahoo.com",
                    api_key_required=False,
                    requests_per_minute=settings.data_collection.requests_per_minute,
                    requests_per_second=settings.data_collection.requests_per_second,
                    is_active=True
                )
                self.session.add(source)
                self.session.commit()
                logger.info(f"Initialized data source: {self.source_name}")
        except Exception as e:
            logger.error(f"Failed to initialize data source: {e}")
            self.session.rollback()
    
    def _update_source_status(self, success: bool, error_message: str = None):
        """Update the data source status."""
        try:
            source = self.session.query(DataSource).filter_by(name=self.source_name).first()
            if source:
                if success:
                    source.last_successful_request = datetime.utcnow()
                    source.consecutive_failures = 0
                    source.circuit_breaker_status = 'closed'
                else:
                    source.last_failed_request = datetime.utcnow()
                    source.consecutive_failures += 1
                    source.last_error_message = error_message
                    
                    # Circuit breaker logic
                    if source.consecutive_failures >= 5:
                        source.circuit_breaker_status = 'open'
                
                source.total_requests += 1
                if success:
                    source.successful_requests += 1
                
                self.session.commit()
        except Exception as e:
            logger.error(f"Failed to update source status: {e}")
            self.session.rollback()
    
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
        
        Returns:
            List of stock data records
        """
        try:
            # Rate limiting
            await self.rate_limiter.wait()
            
            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            # Use yfinance to get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                self._update_source_status(False, "No data returned")
                return []
            
            # Convert to list of dictionaries
            records = []
            for timestamp, row in data.iterrows():
                record = {
                    'symbol': symbol,
                    'exchange': 'NASDAQ',  # Default, will be updated from database
                    'timestamp': timestamp.to_pydatetime(),
                    'timeframe': interval,
                    'open_price': Decimal(str(row['Open'])) if pd.notna(row['Open']) else None,
                    'high_price': Decimal(str(row['High'])) if pd.notna(row['High']) else None,
                    'low_price': Decimal(str(row['Low'])) if pd.notna(row['Low']) else None,
                    'close_price': Decimal(str(row['Close'])) if pd.notna(row['Close']) else None,
                    'volume': int(row['Volume']) if pd.notna(row['Volume']) else None,
                    'adjusted_close': Decimal(str(row['Adj Close'])) if pd.notna(row['Adj Close']) else None,
                    'source': self.source_name,
                    'data_quality_score': None,  # Will be calculated by validator
                    'is_validated': False
                }
                
                # Validate data
                validation_result = self.data_validator.validate_stock_data(record)
                record['data_quality_score'] = validation_result['quality_score']
                record['is_validated'] = validation_result['is_valid']
                
                records.append(record)
            
            logger.info(f"Retrieved {len(records)} records for {symbol}")
            self._update_source_status(True)
            return records
            
        except Exception as e:
            error_msg = f"Failed to get historical data for {symbol}: {str(e)}"
            logger.error(error_msg)
            self._update_source_status(False, error_msg)
            return []
    
    async def get_real_time_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time data for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Real-time stock data record
        """
        try:
            # Rate limiting
            await self.rate_limiter.wait()
            
            logger.info(f"Fetching real-time data for {symbol}")
            
            # Use yfinance to get real-time data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            current_price = info.get('regularMarketPrice')
            if not current_price:
                logger.warning(f"No current price available for {symbol}")
                self._update_source_status(False, "No current price available")
                return None
            
            # Get market data
            record = {
                'symbol': symbol,
                'exchange': info.get('exchange', 'NASDAQ'),
                'timestamp': datetime.utcnow(),
                'timeframe': '1m',  # Real-time data
                'open_price': Decimal(str(info.get('regularMarketOpen', 0))) if info.get('regularMarketOpen') else None,
                'high_price': Decimal(str(info.get('dayHigh', 0))) if info.get('dayHigh') else None,
                'low_price': Decimal(str(info.get('dayLow', 0))) if info.get('dayLow') else None,
                'close_price': Decimal(str(current_price)),
                'volume': int(info.get('volume', 0)) if info.get('volume') else None,
                'adjusted_close': Decimal(str(current_price)),  # Use current price as adjusted close
                'source': self.source_name,
                'data_quality_score': None,
                'is_validated': False
            }
            
            # Validate data
            validation_result = self.data_validator.validate_stock_data(record)
            record['data_quality_score'] = validation_result['quality_score']
            record['is_validated'] = validation_result['is_valid']
            
            logger.info(f"Retrieved real-time data for {symbol}: ${current_price}")
            self._update_source_status(True)
            return record
            
        except Exception as e:
            error_msg = f"Failed to get real-time data for {symbol}: {str(e)}"
            logger.error(error_msg)
            self._update_source_status(False, error_msg)
            return None
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information and metadata.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Symbol information dictionary
        """
        try:
            # Rate limiting
            await self.rate_limiter.wait()
            
            logger.info(f"Fetching symbol info for {symbol}")
            
            # Use yfinance to get symbol info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                logger.warning(f"No info available for {symbol}")
                self._update_source_status(False, "No symbol info available")
                return None
            
            symbol_info = {
                'symbol': symbol,
                'exchange': info.get('exchange', 'NASDAQ'),
                'company_name': info.get('longName', info.get('shortName')),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': int(info.get('marketCap', 0)) if info.get('marketCap') else None,
                'is_active': True,
                'data_collection_enabled': True
            }
            
            logger.info(f"Retrieved symbol info for {symbol}: {symbol_info['company_name']}")
            self._update_source_status(True)
            return symbol_info
            
        except Exception as e:
            error_msg = f"Failed to get symbol info for {symbol}: {str(e)}"
            logger.error(error_msg)
            self._update_source_status(False, error_msg)
            return None
    
    async def collect_historical_data_batch(
        self, 
        symbols: List[str], 
        days_back: int = 30,
        interval: str = "1d"
    ) -> Dict[str, int]:
        """
        Collect historical data for multiple symbols in batch.
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to go back
            interval: Data interval
        
        Returns:
            Dictionary with collection statistics
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        stats = {
            'total_symbols': len(symbols),
            'successful_symbols': 0,
            'failed_symbols': 0,
            'total_records': 0,
            'failed_records': 0
        }
        
        logger.info(f"Starting batch collection for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                records = await self.get_historical_data(symbol, start_date, end_date, interval)
                
                if records:
                    # Save to database
                    saved_count = self._save_records(records)
                    stats['successful_symbols'] += 1
                    stats['total_records'] += saved_count
                    logger.info(f"Saved {saved_count} records for {symbol}")
                else:
                    stats['failed_symbols'] += 1
                    logger.warning(f"No records saved for {symbol}")
                
                # Small delay between symbols
                await asyncio.sleep(0.1)
                
            except Exception as e:
                stats['failed_symbols'] += 1
                logger.error(f"Failed to collect data for {symbol}: {e}")
        
        logger.info(f"Batch collection completed: {stats}")
        return stats
    
    def _save_records(self, records: List[Dict[str, Any]]) -> int:
        """
        Save records to the database.
        
        Args:
            records: List of stock data records
        
        Returns:
            Number of records saved
        """
        try:
            saved_count = 0
            for record_data in records:
                try:
                    # Check if record already exists
                    existing = self.session.query(StockData).filter_by(
                        symbol=record_data['symbol'],
                        timestamp=record_data['timestamp'],
                        timeframe=record_data['timeframe'],
                        source=record_data['source']
                    ).first()
                    
                    if not existing:
                        stock_data = StockData(**record_data)
                        self.session.add(stock_data)
                        saved_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to save record for {record_data['symbol']}: {e}")
                    continue
            
            self.session.commit()
            return saved_count
            
        except Exception as e:
            logger.error(f"Failed to save records: {e}")
            self.session.rollback()
            return 0
    
    def get_active_symbols(self) -> List[str]:
        """
        Get list of active symbols from the database.
        
        Returns:
            List of active symbol strings
        """
        try:
            symbols = self.session.query(Symbol).filter_by(
                is_active=True,
                data_collection_enabled=True
            ).all()
            return [symbol.symbol for symbol in symbols]
        except Exception as e:
            logger.error(f"Failed to get active symbols: {e}")
            return []
    
    async def update_symbol_info(self, symbol: str) -> bool:
        """
        Update symbol information in the database.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            True if successful, False otherwise
        """
        try:
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info:
                return False
            
            # Update or create symbol record
            existing_symbol = self.session.query(Symbol).filter_by(symbol=symbol).first()
            if existing_symbol:
                for key, value in symbol_info.items():
                    if hasattr(existing_symbol, key):
                        setattr(existing_symbol, key, value)
                existing_symbol.updated_at = datetime.utcnow()
            else:
                new_symbol = Symbol(**symbol_info)
                self.session.add(new_symbol)
            
            self.session.commit()
            logger.info(f"Updated symbol info for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update symbol info for {symbol}: {e}")
            self.session.rollback()
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get data collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            source = self.session.query(DataSource).filter_by(name=self.source_name).first()
            if not source:
                return {}
            
            # Get recent data count
            recent_data = self.session.query(StockData).filter(
                StockData.source == self.source_name,
                StockData.created_at >= datetime.utcnow() - timedelta(hours=1)
            ).count()
            
            return {
                'source_name': source.name,
                'display_name': source.display_name,
                'is_active': source.is_active,
                'total_requests': source.total_requests,
                'successful_requests': source.successful_requests,
                'success_rate': (source.successful_requests / source.total_requests * 100) if source.total_requests > 0 else 0,
                'consecutive_failures': source.consecutive_failures,
                'circuit_breaker_status': source.circuit_breaker_status,
                'last_successful_request': source.last_successful_request,
                'last_failed_request': source.last_failed_request,
                'recent_data_count': recent_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


# Example usage and testing
async def test_yahoo_finance_scraper():
    """Test function for the Yahoo Finance scraper."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create database session
    engine = create_engine(settings.database.database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Create scraper
        scraper = YahooFinanceScraper(session)
        
        # Test symbol info
        symbol_info = await scraper.get_symbol_info("AAPL")
        print(f"Symbol info: {symbol_info}")
        
        # Test historical data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        historical_data = await scraper.get_historical_data("AAPL", start_date, end_date, "1d")
        print(f"Historical data records: {len(historical_data)}")
        
        # Test real-time data
        real_time_data = await scraper.get_real_time_data("AAPL")
        print(f"Real-time data: {real_time_data}")
        
        # Get stats
        stats = scraper.get_collection_stats()
        print(f"Collection stats: {stats}")
        
    finally:
        session.close()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_yahoo_finance_scraper())