"""
Data validator utility for validating stock data quality and integrity.
"""

import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validator for stock data quality and integrity.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_rules = {
            'price_range': (0.01, 1000000.0),  # Min, max price
            'volume_range': (0, 10000000000),   # Min, max volume
            'price_change_threshold': 0.5,      # 50% max price change
            'volume_change_threshold': 10.0,    # 1000% max volume change
            'missing_data_threshold': 0.1,      # 10% max missing data
            'outlier_threshold': 3.0,           # 3 standard deviations
        }
    
    def validate_stock_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single stock data record.
        
        Args:
            data: Stock data record to validate
        
        Returns:
            Validation result with quality score and issues
        """
        issues = []
        quality_score = 1.0
        
        try:
            # Basic data type validation
            type_issues = self._validate_data_types(data)
            issues.extend(type_issues)
            
            # OHLC relationship validation
            ohlc_issues = self._validate_ohlc_relationships(data)
            issues.extend(ohlc_issues)
            
            # Price range validation
            price_issues = self._validate_price_ranges(data)
            issues.extend(price_issues)
            
            # Volume validation
            volume_issues = self._validate_volume(data)
            issues.extend(volume_issues)
            
            # Timestamp validation
            timestamp_issues = self._validate_timestamp(data)
            issues.extend(timestamp_issues)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(issues)
            
            return {
                'is_valid': len(issues) == 0,
                'quality_score': quality_score,
                'issues': issues,
                'issue_count': len(issues)
            }
            
        except Exception as e:
            logger.error(f"Error validating stock data: {e}")
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'issues': [f'Validation error: {str(e)}'],
                'issue_count': 1
            }
    
    def validate_data_batch(self, data_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of stock data records.
        
        Args:
            data_batch: List of stock data records
        
        Returns:
            Batch validation result
        """
        if not data_batch:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'issues': ['Empty data batch'],
                'issue_count': 1,
                'total_records': 0,
                'valid_records': 0,
                'invalid_records': 0
            }
        
        batch_issues = []
        valid_records = 0
        invalid_records = 0
        quality_scores = []
        
        for record in data_batch:
            validation_result = self.validate_stock_data(record)
            
            if validation_result['is_valid']:
                valid_records += 1
            else:
                invalid_records += 1
                batch_issues.extend(validation_result['issues'])
            
            quality_scores.append(validation_result['quality_score'])
        
        # Calculate batch statistics
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0
        total_records = len(data_batch)
        
        # Check for batch-level issues
        batch_issues.extend(self._validate_batch_consistency(data_batch))
        
        return {
            'is_valid': invalid_records == 0,
            'quality_score': avg_quality_score,
            'issues': batch_issues,
            'issue_count': len(batch_issues),
            'total_records': total_records,
            'valid_records': valid_records,
            'invalid_records': invalid_records,
            'validity_rate': valid_records / total_records if total_records > 0 else 0.0
        }
    
    def _validate_data_types(self, data: Dict[str, Any]) -> List[str]:
        """Validate data types of fields."""
        issues = []
        
        # Required fields
        required_fields = ['symbol', 'timestamp', 'timeframe', 'source']
        for field in required_fields:
            if field not in data or data[field] is None:
                issues.append(f'Missing required field: {field}')
        
        # Symbol validation
        if 'symbol' in data and data['symbol']:
            if not isinstance(data['symbol'], str) or len(data['symbol']) > 10:
                issues.append('Invalid symbol format')
        
        # Timestamp validation
        if 'timestamp' in data and data['timestamp']:
            if not isinstance(data['timestamp'], datetime):
                issues.append('Invalid timestamp format')
        
        # Price fields validation
        price_fields = ['open_price', 'high_price', 'low_price', 'close_price', 'adjusted_close']
        for field in price_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], (int, float, Decimal)):
                    issues.append(f'Invalid price format for {field}')
        
        # Volume validation
        if 'volume' in data and data['volume'] is not None:
            if not isinstance(data['volume'], (int, float)):
                issues.append('Invalid volume format')
        
        return issues
    
    def _validate_ohlc_relationships(self, data: Dict[str, Any]) -> List[str]:
        """Validate OHLC price relationships."""
        issues = []
        
        # Check if we have all OHLC values
        ohlc_fields = ['open_price', 'high_price', 'low_price', 'close_price']
        ohlc_values = {}
        
        for field in ohlc_fields:
            if field in data and data[field] is not None:
                ohlc_values[field] = float(data[field])
        
        if len(ohlc_values) < 4:
            issues.append('Missing OHLC values')
            return issues
        
        # Validate relationships
        high = ohlc_values['high_price']
        low = ohlc_values['low_price']
        open_price = ohlc_values['open_price']
        close_price = ohlc_values['close_price']
        
        # High should be >= Low
        if high < low:
            issues.append('High price is less than low price')
        
        # High should be >= Open and Close
        if high < open_price or high < close_price:
            issues.append('High price is less than open or close price')
        
        # Low should be <= Open and Close
        if low > open_price or low > close_price:
            issues.append('Low price is greater than open or close price')
        
        return issues
    
    def _validate_price_ranges(self, data: Dict[str, Any]) -> List[str]:
        """Validate price ranges."""
        issues = []
        min_price, max_price = self.validation_rules['price_range']
        
        price_fields = ['open_price', 'high_price', 'low_price', 'close_price', 'adjusted_close']
        
        for field in price_fields:
            if field in data and data[field] is not None:
                price = float(data[field])
                
                if price < min_price:
                    issues.append(f'{field} is below minimum price threshold')
                
                if price > max_price:
                    issues.append(f'{field} is above maximum price threshold')
        
        return issues
    
    def _validate_volume(self, data: Dict[str, Any]) -> List[str]:
        """Validate volume data."""
        issues = []
        
        if 'volume' in data and data['volume'] is not None:
            volume = float(data['volume'])
            min_volume, max_volume = self.validation_rules['volume_range']
            
            if volume < min_volume:
                issues.append('Volume is below minimum threshold')
            
            if volume > max_volume:
                issues.append('Volume is above maximum threshold')
        
        return issues
    
    def _validate_timestamp(self, data: Dict[str, Any]) -> List[str]:
        """Validate timestamp data."""
        issues = []
        
        if 'timestamp' in data and data['timestamp']:
            timestamp = data['timestamp']
            
            # Check if timestamp is in the future
            if timestamp > datetime.utcnow() + timedelta(minutes=5):
                issues.append('Timestamp is in the future')
            
            # Check if timestamp is too old (more than 10 years)
            if timestamp < datetime.utcnow() - timedelta(days=3650):
                issues.append('Timestamp is too old')
            
            # Check if timestamp is during market hours (basic check)
            if not self._is_market_hours(timestamp):
                issues.append('Timestamp is outside market hours')
        
        return issues
    
    def _validate_batch_consistency(self, data_batch: List[Dict[str, Any]]) -> List[str]:
        """Validate consistency across a batch of data."""
        issues = []
        
        if len(data_batch) < 2:
            return issues
        
        # Check for duplicate timestamps
        timestamps = [record.get('timestamp') for record in data_batch if record.get('timestamp')]
        if len(timestamps) != len(set(timestamps)):
            issues.append('Duplicate timestamps found in batch')
        
        # Check for consistent symbol
        symbols = [record.get('symbol') for record in data_batch if record.get('symbol')]
        if len(set(symbols)) > 1:
            issues.append('Multiple symbols found in single batch')
        
        # Check for consistent timeframe
        timeframes = [record.get('timeframe') for record in data_batch if record.get('timeframe')]
        if len(set(timeframes)) > 1:
            issues.append('Multiple timeframes found in single batch')
        
        return issues
    
    def _calculate_quality_score(self, issues: List[str]) -> float:
        """
        Calculate quality score based on issues found.
        
        Args:
            issues: List of validation issues
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not issues:
            return 1.0
        
        # Weight different types of issues
        critical_issues = ['Missing required field', 'Invalid OHLC relationships', 'High price is less than low price']
        major_issues = ['Invalid price format', 'Invalid volume format', 'Timestamp is in the future']
        
        score = 1.0
        
        for issue in issues:
            if any(critical in issue for critical in critical_issues):
                score -= 0.3  # Critical issues heavily penalize score
            elif any(major in issue for major in major_issues):
                score -= 0.1  # Major issues moderately penalize score
            else:
                score -= 0.05  # Minor issues slightly penalize score
        
        return max(0.0, score)
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is during market hours.
        Basic implementation - can be enhanced for holidays, etc.
        """
        # Convert to US Eastern time (simplified)
        # In production, use proper timezone handling
        
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        weekday = timestamp.weekday()  # Monday = 0, Sunday = 6
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Weekend check
        if weekday >= 5:  # Saturday or Sunday
            return False
        
        # Market hours check (simplified)
        market_start = 9 * 60 + 30  # 9:30 AM
        market_end = 16 * 60  # 4:00 PM
        current_time = hour * 60 + minute
        
        return market_start <= current_time <= market_end
    
    def detect_outliers(self, data_series: List[float], method: str = 'zscore') -> List[int]:
        """
        Detect outliers in a data series.
        
        Args:
            data_series: List of numerical values
            method: Outlier detection method ('zscore', 'iqr')
        
        Returns:
            List of outlier indices
        """
        if len(data_series) < 3:
            return []
        
        outliers = []
        
        if method == 'zscore':
            # Z-score method
            mean_val = statistics.mean(data_series)
            std_val = statistics.stdev(data_series)
            
            if std_val == 0:
                return []
            
            threshold = self.validation_rules['outlier_threshold']
            
            for i, value in enumerate(data_series):
                z_score = abs((value - mean_val) / std_val)
                if z_score > threshold:
                    outliers.append(i)
        
        elif method == 'iqr':
            # Interquartile range method
            sorted_data = sorted(data_series)
            q1 = statistics.quantiles(sorted_data, n=4)[0]
            q3 = statistics.quantiles(sorted_data, n=4)[2]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, value in enumerate(data_series):
                if value < lower_bound or value > upper_bound:
                    outliers.append(i)
        
        return outliers
    
    def get_validation_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of validation results.
        
        Args:
            validation_results: List of validation result dictionaries
        
        Returns:
            Summary statistics
        """
        if not validation_results:
            return {
                'total_records': 0,
                'valid_records': 0,
                'invalid_records': 0,
                'avg_quality_score': 0.0,
                'common_issues': [],
                'issue_frequency': {}
            }
        
        total_records = len(validation_results)
        valid_records = sum(1 for result in validation_results if result['is_valid'])
        invalid_records = total_records - valid_records
        
        quality_scores = [result['quality_score'] for result in validation_results]
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0
        
        # Analyze common issues
        all_issues = []
        for result in validation_results:
            all_issues.extend(result.get('issues', []))
        
        issue_frequency = {}
        for issue in all_issues:
            issue_frequency[issue] = issue_frequency.get(issue, 0) + 1
        
        # Get most common issues
        common_issues = sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        common_issues = [issue for issue, count in common_issues]
        
        return {
            'total_records': total_records,
            'valid_records': valid_records,
            'invalid_records': invalid_records,
            'validity_rate': valid_records / total_records if total_records > 0 else 0.0,
            'avg_quality_score': avg_quality_score,
            'common_issues': common_issues,
            'issue_frequency': issue_frequency,
            'total_issues': len(all_issues)
        }


# Example usage
def test_data_validator():
    """Test the data validator functionality."""
    validator = DataValidator()
    
    # Test valid data
    valid_data = {
        'symbol': 'AAPL',
        'timestamp': datetime.utcnow(),
        'timeframe': '1d',
        'open_price': Decimal('150.00'),
        'high_price': Decimal('155.00'),
        'low_price': Decimal('149.00'),
        'close_price': Decimal('152.00'),
        'volume': 1000000,
        'source': 'yahoo_finance'
    }
    
    result = validator.validate_stock_data(valid_data)
    print(f"Valid data result: {result}")
    
    # Test invalid data
    invalid_data = {
        'symbol': 'AAPL',
        'timestamp': datetime.utcnow(),
        'timeframe': '1d',
        'open_price': Decimal('150.00'),
        'high_price': Decimal('145.00'),  # High < Open
        'low_price': Decimal('149.00'),
        'close_price': Decimal('152.00'),
        'volume': -1000,  # Negative volume
        'source': 'yahoo_finance'
    }
    
    result = validator.validate_stock_data(invalid_data)
    print(f"Invalid data result: {result}")


if __name__ == "__main__":
    test_data_validator()