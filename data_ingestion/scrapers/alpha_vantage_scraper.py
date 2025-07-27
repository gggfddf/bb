"""
Alpha Vantage Data Scraper

Provides data collection from Alpha Vantage API for the ML Stock Predictor Platform.
This scraper implements the same interface as other scrapers for consistency.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import pandas as pd
from sqlalchemy.orm import Session

from ..utils.rate_limiter import RateLimiter
from ..utils.data_validator import DataValidator
from ..utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class AlphaVantageScraper:
    """
    Alpha Vantage data scraper for stock market data.
    
    Features:
    - Historical intraday data (1min, 5min, 15min, 30min, 60min)
    - Daily, weekly, monthly data
    - Real-time quotes
    - Company information
    - Rate limiting and error handling
    """
    
    def __init__(self, session: Session, api_key: str = None):
        self.session = session
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
        # Initialize utilities
        self.rate_limiter = RateLimiter(requests_per_minute=5)  # Alpha Vantage free tier limit
        self.data_validator = DataValidator()
        self.error_handler = ErrorHandler(max_retries=3, base_delay=2.0)
        
        # API endpoints
        self.endpoints = {
            'intraday': 'TIME_SERIES_INTRADAY',
            'daily': 'TIME_SERIES_DAILY',
            'weekly': 'TIME_SERIES_WEEKLY',
            'monthly': 'TIME_SERIES_MONTHLY',
            'quote': 'GLOBAL_QUOTE',
            'company': 'OVERVIEW'
        }
        
        # Interval mapping
        self.interval_mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '60m': '60min',
            '1h': '60min',
            '1d': 'daily',
            '1w': 'weekly',
            '1m': 'monthly'
        }
    
    @ErrorHandler.retry_with_backoff()
    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d") -> List[Dict[str, Any]]:
        """
        Get historical stock data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data interval (1m, 5m, 15m, 30m, 60m, 1d, 1w, 1m)
        
        Returns:
            List of stock data dictionaries
        """
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided, skipping data collection")
            return []
        
        await self.rate_limiter.wait()
        
        try:
            # Map interval to Alpha Vantage format
            av_interval = self.interval_mapping.get(interval, 'daily')
            
            # Determine endpoint based on interval
            if av_interval in ['1min', '5min', '15min', '30min', '60min']:
                endpoint = self.endpoints['intraday']
                params = {
                    'function': endpoint,
                    'symbol': symbol,
                    'interval': av_interval,
                    'apikey': self.api_key,
                    'outputsize': 'full'  # Get full history
                }
            elif av_interval == 'daily':
                endpoint = self.endpoints['daily']
                params = {
                    'function': endpoint,
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'full'
                }
            elif av_interval == 'weekly':
                endpoint = self.endpoints['weekly']
                params = {
                    'function': endpoint,
                    'symbol': symbol,
                    'apikey': self.api_key
                }
            elif av_interval == 'monthly':
                endpoint = self.endpoints['monthly']
                params = {
                    'function': endpoint,
                    'symbol': symbol,
                    'apikey': self.api_key
                }
            else:
                logger.error(f"Unsupported interval: {interval}")
                return []
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Alpha Vantage API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    
                    # Check for API errors
                    if 'Error Message' in data:
                        logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                        return []
                    
                    if 'Note' in data:
                        logger.warning(f"Alpha Vantage API note: {data['Note']}")
                        return []
                    
                    # Parse response data
                    return self._parse_alpha_vantage_response(data, symbol, av_interval, start_date, end_date)
        
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise
    
    def _parse_alpha_vantage_response(self, data: Dict[str, Any], symbol: str, interval: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Parse Alpha Vantage API response into standardized format.
        
        Args:
            data: Raw API response
            symbol: Stock symbol
            interval: Data interval
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            List of parsed stock data dictionaries
        """
        parsed_data = []
        
        # Extract time series data
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.error("No time series data found in Alpha Vantage response")
            return []
        
        time_series_data = data[time_series_key]
        
        for timestamp_str, values in time_series_data.items():
            try:
                # Parse timestamp
                if interval in ['1min', '5min', '15min', '30min', '60min']:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                else:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d')
                
                # Filter by date range
                if timestamp < start_date or timestamp > end_date:
                    continue
                
                # Extract OHLCV data
                stock_data = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'open': float(values.get('1. open', 0)),
                    'high': float(values.get('2. high', 0)),
                    'low': float(values.get('3. low', 0)),
                    'close': float(values.get('4. close', 0)),
                    'volume': int(values.get('5. volume', 0)),
                    'exchange': 'UNKNOWN',  # Alpha Vantage doesn't provide exchange info
                    'source': 'alpha_vantage'
                }
                
                parsed_data.append(stock_data)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse data point for {symbol} at {timestamp_str}: {e}")
                continue
        
        # Sort by timestamp
        parsed_data.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Parsed {len(parsed_data)} data points for {symbol} from Alpha Vantage")
        return parsed_data
    
    @ErrorHandler.retry_with_backoff()
    async def get_realtime_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Real-time stock data dictionary or None
        """
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided, skipping real-time data")
            return None
        
        await self.rate_limiter.wait()
        
        try:
            params = {
                'function': self.endpoints['quote'],
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Alpha Vantage API error: {response.status}")
                        return None
                    
                    data = await response.json()
                    
                    # Check for API errors
                    if 'Error Message' in data:
                        logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                        return None
                    
                    if 'Note' in data:
                        logger.warning(f"Alpha Vantage API note: {data['Note']}")
                        return None
                    
                    # Parse quote data
                    return self._parse_quote_response(data, symbol)
        
        except Exception as e:
            logger.error(f"Failed to get real-time data for {symbol}: {e}")
            raise
    
    def _parse_quote_response(self, data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """
        Parse Alpha Vantage quote response.
        
        Args:
            data: Raw API response
            symbol: Stock symbol
        
        Returns:
            Parsed quote data dictionary or None
        """
        if 'Global Quote' not in data:
            logger.error("No global quote data found in Alpha Vantage response")
            return None
        
        quote_data = data['Global Quote']
        
        try:
            # Extract quote information
            quote = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'open': float(quote_data.get('02. open', 0)),
                'high': float(quote_data.get('03. high', 0)),
                'low': float(quote_data.get('04. low', 0)),
                'close': float(quote_data.get('05. price', 0)),
                'volume': int(quote_data.get('06. volume', 0)),
                'exchange': quote_data.get('01. exchange', 'UNKNOWN'),
                'source': 'alpha_vantage'
            }
            
            logger.info(f"Retrieved real-time quote for {symbol}: ${quote['close']}")
            return quote
            
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse quote data for {symbol}: {e}")
            return None
    
    @ErrorHandler.retry_with_backoff()
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company information from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Company information dictionary or None
        """
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided, skipping symbol info")
            return None
        
        await self.rate_limiter.wait()
        
        try:
            params = {
                'function': self.endpoints['company'],
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Alpha Vantage API error: {response.status}")
                        return None
                    
                    data = await response.json()
                    
                    # Check for API errors
                    if 'Error Message' in data:
                        logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                        return None
                    
                    if 'Note' in data:
                        logger.warning(f"Alpha Vantage API note: {data['Note']}")
                        return None
                    
                    # Parse company information
                    return self._parse_company_response(data, symbol)
        
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            raise
    
    def _parse_company_response(self, data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """
        Parse Alpha Vantage company overview response.
        
        Args:
            data: Raw API response
            symbol: Stock symbol
        
        Returns:
            Company information dictionary or None
        """
        try:
            company_info = {
                'symbol': symbol,
                'name': data.get('Name', ''),
                'exchange': data.get('Exchange', ''),
                'sector': data.get('Sector', ''),
                'industry': data.get('Industry', ''),
                'description': data.get('Description', ''),
                'market_cap': data.get('MarketCapitalization', ''),
                'pe_ratio': data.get('PERatio', ''),
                'dividend_yield': data.get('DividendYield', ''),
                'source': 'alpha_vantage'
            }
            
            logger.info(f"Retrieved company info for {symbol}: {company_info['name']}")
            return company_info
            
        except Exception as e:
            logger.error(f"Failed to parse company data for {symbol}: {e}")
            return None
    
    async def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols (not directly supported by Alpha Vantage).
        This would need to be implemented with a separate symbol list source.
        
        Returns:
            List of available symbols
        """
        # Alpha Vantage doesn't provide a direct endpoint for available symbols
        # This would need to be implemented with a separate data source
        # For now, return an empty list
        logger.warning("Alpha Vantage doesn't provide available symbols endpoint")
        return []
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get current rate limit information.
        
        Returns:
            Rate limit information dictionary
        """
        return {
            'requests_per_minute': self.rate_limiter.requests_per_minute,
            'requests_per_second': self.rate_limiter.requests_per_second,
            'burst_size': self.rate_limiter.burst_size,
            'current_queue_size': len(self.rate_limiter.request_times)
        }