"""
Data Validation Pipeline

Comprehensive data validation system for the ML Stock Predictor Platform.
Implements multi-stage validation, anomaly detection, and data quality monitoring.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..models import StockData, DataQualityLog, Symbol
from ..utils.data_validator import DataValidator
from ..utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class DataValidationPipeline:
    """
    Comprehensive data validation pipeline for stock market data.
    
    Features:
    - Multi-stage validation (raw data, business rules, statistical)
    - Anomaly detection using statistical methods
    - Data quality scoring and reporting
    - Automated data cleaning and correction
    - Historical data consistency checks
    """
    
    def __init__(self, session: Session):
        self.session = session
        self.data_validator = DataValidator()
        self.error_handler = ErrorHandler()
        
        # Validation thresholds
        self.thresholds = {
            'price_change_max': 0.5,  # 50% max price change
            'volume_spike_threshold': 10.0,  # 10x volume spike
            'missing_data_threshold': 0.1,  # 10% missing data
            'outlier_std_threshold': 3.0,  # 3 standard deviations
            'consistency_threshold': 0.95  # 95% consistency
        }
        
        # Validation results storage
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        
    async def validate_data_batch(self, data_batch: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
        """
        Validate a batch of data records.
        
        Args:
            data_batch: List of data dictionaries to validate
            source: Data source identifier
        
        Returns:
            Validation results dictionary
        """
        logger.info(f"Starting validation for {len(data_batch)} records from {source}")
        
        validation_result = {
            'source': source,
            'total_records': len(data_batch),
            'valid_records': 0,
            'invalid_records': 0,
            'anomalies_detected': 0,
            'quality_score': 0.0,
            'issues': [],
            'anomalies': [],
            'recommendations': []
        }
        
        try:
            # Stage 1: Basic data validation
            basic_validation = await self._basic_validation(data_batch)
            validation_result['valid_records'] = basic_validation['valid_count']
            validation_result['invalid_records'] = basic_validation['invalid_count']
            validation_result['issues'].extend(basic_validation['issues'])
            
            # Stage 2: Business rule validation
            business_validation = await self._business_rule_validation(data_batch)
            validation_result['issues'].extend(business_validation['issues'])
            
            # Stage 3: Statistical validation and anomaly detection
            statistical_validation = await self._statistical_validation(data_batch)
            validation_result['anomalies_detected'] = statistical_validation['anomaly_count']
            validation_result['anomalies'].extend(statistical_validation['anomalies'])
            
            # Stage 4: Historical consistency check
            consistency_check = await self._historical_consistency_check(data_batch)
            validation_result['issues'].extend(consistency_check['issues'])
            
            # Calculate overall quality score
            validation_result['quality_score'] = self._calculate_quality_score(validation_result)
            
            # Generate recommendations
            validation_result['recommendations'] = self._generate_recommendations(validation_result)
            
            # Store validation results
            self.validation_results[source] = validation_result
            
            # Log validation results
            await self._log_validation_results(validation_result)
            
            logger.info(f"Validation completed for {source}: {validation_result['quality_score']:.2f} quality score")
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            validation_result['issues'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    async def _basic_validation(self, data_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform basic data validation using the DataValidator utility.
        
        Args:
            data_batch: List of data dictionaries
        
        Returns:
            Basic validation results
        """
        valid_count = 0
        invalid_count = 0
        issues = []
        
        for data in data_batch:
            validation_result = self.data_validator.validate_stock_data(data)
            
            if validation_result['is_valid']:
                valid_count += 1
            else:
                invalid_count += 1
                issues.extend(validation_result['issues'])
        
        return {
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'issues': issues
        }
    
    async def _business_rule_validation(self, data_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply business rules validation.
        
        Args:
            data_batch: List of data dictionaries
        
        Returns:
            Business rule validation results
        """
        issues = []
        
        for i, data in enumerate(data_batch):
            # Check OHLC relationships
            if not self._validate_ohlc_relationships(data):
                issues.append(f"Record {i}: Invalid OHLC relationships")
            
            # Check price changes
            if i > 0:
                price_change = self._calculate_price_change(data_batch[i-1], data)
                if abs(price_change) > self.thresholds['price_change_max']:
                    issues.append(f"Record {i}: Excessive price change ({price_change:.2%})")
            
            # Check volume spikes
            if i > 0:
                volume_ratio = data.get('volume', 0) / max(data_batch[i-1].get('volume', 1), 1)
                if volume_ratio > self.thresholds['volume_spike_threshold']:
                    issues.append(f"Record {i}: Volume spike detected ({volume_ratio:.1f}x)")
        
        return {'issues': issues}
    
    def _validate_ohlc_relationships(self, data: Dict[str, Any]) -> bool:
        """
        Validate OHLC price relationships.
        
        Args:
            data: Stock data dictionary
        
        Returns:
            True if OHLC relationships are valid
        """
        try:
            open_price = data.get('open', 0)
            high_price = data.get('high', 0)
            low_price = data.get('low', 0)
            close_price = data.get('close', 0)
            
            # Basic OHLC validation
            if not (low_price <= open_price <= high_price and low_price <= close_price <= high_price):
                return False
            
            # Check for zero or negative prices
            if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_price_change(self, prev_data: Dict[str, Any], curr_data: Dict[str, Any]) -> float:
        """
        Calculate price change between two data points.
        
        Args:
            prev_data: Previous data point
            curr_data: Current data point
        
        Returns:
            Price change as a percentage
        """
        try:
            prev_close = prev_data.get('close', 0)
            curr_close = curr_data.get('close', 0)
            
            if prev_close == 0:
                return 0.0
            
            return (curr_close - prev_close) / prev_close
            
        except Exception:
            return 0.0
    
    async def _statistical_validation(self, data_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform statistical validation and anomaly detection.
        
        Args:
            data_batch: List of data dictionaries
        
        Returns:
            Statistical validation results
        """
        if len(data_batch) < 10:  # Need minimum data for statistical analysis
            return {'anomaly_count': 0, 'anomalies': []}
        
        anomalies = []
        anomaly_count = 0
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data_batch)
        
        # Price anomaly detection
        price_anomalies = self._detect_price_anomalies(df)
        anomalies.extend(price_anomalies)
        anomaly_count += len(price_anomalies)
        
        # Volume anomaly detection
        volume_anomalies = self._detect_volume_anomalies(df)
        anomalies.extend(volume_anomalies)
        anomaly_count += len(volume_anomalies)
        
        # Pattern anomaly detection
        pattern_anomalies = self._detect_pattern_anomalies(df)
        anomalies.extend(pattern_anomalies)
        anomaly_count += len(pattern_anomalies)
        
        return {
            'anomaly_count': anomaly_count,
            'anomalies': anomalies
        }
    
    def _detect_price_anomalies(self, df: pd.DataFrame) -> List[str]:
        """
        Detect price anomalies using statistical methods.
        
        Args:
            df: DataFrame with stock data
        
        Returns:
            List of detected price anomalies
        """
        anomalies = []
        
        try:
            # Calculate price changes
            df['price_change'] = df['close'].pct_change()
            
            # Detect outliers using Z-score
            price_changes = df['price_change'].dropna()
            if len(price_changes) > 0:
                z_scores = np.abs((price_changes - price_changes.mean()) / price_changes.std())
                outlier_indices = z_scores[z_scores > self.thresholds['outlier_std_threshold']].index
                
                for idx in outlier_indices:
                    anomalies.append(f"Price anomaly at index {idx}: {price_changes[idx]:.4f} change")
            
            # Detect extreme price levels
            price_mean = df['close'].mean()
            price_std = df['close'].std()
            
            extreme_high = price_mean + (3 * price_std)
            extreme_low = price_mean - (3 * price_std)
            
            extreme_indices = df[(df['close'] > extreme_high) | (df['close'] < extreme_low)].index
            for idx in extreme_indices:
                anomalies.append(f"Extreme price at index {idx}: ${df.loc[idx, 'close']:.2f}")
                
        except Exception as e:
            logger.error(f"Error in price anomaly detection: {e}")
        
        return anomalies
    
    def _detect_volume_anomalies(self, df: pd.DataFrame) -> List[str]:
        """
        Detect volume anomalies using statistical methods.
        
        Args:
            df: DataFrame with stock data
        
        Returns:
            List of detected volume anomalies
        """
        anomalies = []
        
        try:
            # Calculate volume changes
            df['volume_change'] = df['volume'].pct_change()
            
            # Detect volume spikes
            volume_changes = df['volume_change'].dropna()
            if len(volume_changes) > 0:
                z_scores = np.abs((volume_changes - volume_changes.mean()) / volume_changes.std())
                spike_indices = z_scores[z_scores > self.thresholds['outlier_std_threshold']].index
                
                for idx in spike_indices:
                    anomalies.append(f"Volume spike at index {idx}: {volume_changes[idx]:.2%} change")
            
            # Detect zero or very low volume
            low_volume_indices = df[df['volume'] < 100].index
            for idx in low_volume_indices:
                anomalies.append(f"Low volume at index {idx}: {df.loc[idx, 'volume']} shares")
                
        except Exception as e:
            logger.error(f"Error in volume anomaly detection: {e}")
        
        return anomalies
    
    def _detect_pattern_anomalies(self, df: pd.DataFrame) -> List[str]:
        """
        Detect pattern anomalies in the data.
        
        Args:
            df: DataFrame with stock data
        
        Returns:
            List of detected pattern anomalies
        """
        anomalies = []
        
        try:
            # Detect repeated values (potential data quality issues)
            for column in ['open', 'high', 'low', 'close']:
                repeated_counts = df[column].value_counts()
                suspicious_repeats = repeated_counts[repeated_counts > len(df) * 0.1]  # More than 10% repeats
                
                for value, count in suspicious_repeats.items():
                    anomalies.append(f"Suspicious repeated {column} value: {value} ({count} times)")
            
            # Detect missing data patterns
            missing_data = df.isnull().sum()
            for column, missing_count in missing_data.items():
                if missing_count > len(df) * self.thresholds['missing_data_threshold']:
                    anomalies.append(f"High missing data in {column}: {missing_count}/{len(df)} records")
                    
        except Exception as e:
            logger.error(f"Error in pattern anomaly detection: {e}")
        
        return anomalies
    
    async def _historical_consistency_check(self, data_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check consistency with historical data in the database.
        
        Args:
            data_batch: List of data dictionaries
        
        Returns:
            Consistency check results
        """
        issues = []
        
        try:
            if not data_batch:
                return {'issues': issues}
            
            symbol = data_batch[0].get('symbol')
            if not symbol:
                return {'issues': issues}
            
            # Get historical data for comparison
            historical_data = await self._get_historical_data(symbol, limit=100)
            
            if not historical_data:
                return {'issues': issues}
            
            # Compare with historical patterns
            for data in data_batch:
                consistency_score = self._calculate_consistency_score(data, historical_data)
                if consistency_score < self.thresholds['consistency_threshold']:
                    issues.append(f"Low consistency for {symbol}: {consistency_score:.2f}")
                    break  # Only report once per symbol
        
        except Exception as e:
            logger.error(f"Error in historical consistency check: {e}")
            issues.append(f"Consistency check error: {str(e)}")
        
        return {'issues': issues}
    
    async def _get_historical_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical data from database for consistency checking.
        
        Args:
            symbol: Stock symbol
            limit: Number of records to retrieve
        
        Returns:
            List of historical data dictionaries
        """
        try:
            query = text("""
                SELECT timestamp, open, high, low, close, volume
                FROM stock_data
                WHERE symbol = :symbol
                ORDER BY timestamp DESC
                LIMIT :limit
            """)
            
            result = self.session.execute(query, {'symbol': symbol, 'limit': limit})
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return []
    
    def _calculate_consistency_score(self, data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency score between current data and historical data.
        
        Args:
            data: Current data point
            historical_data: Historical data for comparison
        
        Returns:
            Consistency score (0-1)
        """
        try:
            if not historical_data:
                return 0.0
            
            # Calculate historical statistics
            historical_prices = [d['close'] for d in historical_data]
            historical_volumes = [d['volume'] for d in historical_data]
            
            price_mean = np.mean(historical_prices)
            price_std = np.std(historical_prices)
            volume_mean = np.mean(historical_volumes)
            volume_std = np.std(historical_volumes)
            
            # Calculate current data's position relative to historical distribution
            current_price = data.get('close', 0)
            current_volume = data.get('volume', 0)
            
            if price_std == 0 or volume_std == 0:
                return 0.5  # Neutral score if no variation in historical data
            
            price_z_score = abs((current_price - price_mean) / price_std)
            volume_z_score = abs((current_volume - volume_mean) / volume_std)
            
            # Convert Z-scores to consistency scores (higher Z-score = lower consistency)
            price_consistency = max(0, 1 - (price_z_score / 3))  # 3 std devs = 0 consistency
            volume_consistency = max(0, 1 - (volume_z_score / 3))
            
            # Overall consistency is average of price and volume consistency
            overall_consistency = (price_consistency + volume_consistency) / 2
            
            return overall_consistency
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0.0
    
    def _calculate_quality_score(self, validation_result: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score.
        
        Args:
            validation_result: Validation results dictionary
        
        Returns:
            Quality score (0-1)
        """
        try:
            total_records = validation_result['total_records']
            if total_records == 0:
                return 0.0
            
            # Base score from valid records
            valid_ratio = validation_result['valid_records'] / total_records
            
            # Penalty for anomalies
            anomaly_penalty = min(0.2, validation_result['anomalies_detected'] / total_records)
            
            # Penalty for issues
            issue_penalty = min(0.3, len(validation_result['issues']) / total_records)
            
            # Calculate final score
            quality_score = valid_ratio - anomaly_penalty - issue_penalty
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _generate_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on validation results.
        
        Args:
            validation_result: Validation results dictionary
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Quality score recommendations
        quality_score = validation_result['quality_score']
        if quality_score < 0.5:
            recommendations.append("Data quality is poor - consider data source replacement")
        elif quality_score < 0.8:
            recommendations.append("Data quality needs improvement - implement additional validation")
        else:
            recommendations.append("Data quality is good - continue monitoring")
        
        # Anomaly recommendations
        if validation_result['anomalies_detected'] > 0:
            recommendations.append(f"Review {validation_result['anomalies_detected']} detected anomalies")
        
        # Issue recommendations
        if len(validation_result['issues']) > 0:
            recommendations.append("Address validation issues to improve data quality")
        
        return recommendations
    
    async def _log_validation_results(self, validation_result: Dict[str, Any]):
        """
        Log validation results to the database.
        
        Args:
            validation_result: Validation results dictionary
        """
        try:
            # Log overall validation result
            quality_log = DataQualityLog(
                symbol='BATCH',  # For batch validation
                timeframe='BATCH',
                source=validation_result['source'],
                issue_type="validation_summary",
                description=f"Quality score: {validation_result['quality_score']:.2f}, "
                           f"Valid: {validation_result['valid_records']}, "
                           f"Invalid: {validation_result['invalid_records']}, "
                           f"Anomalies: {validation_result['anomalies_detected']}",
                severity="info" if validation_result['quality_score'] > 0.8 else "warning"
            )
            self.session.add(quality_log)
            
            # Log individual issues
            for issue in validation_result['issues'][:10]:  # Limit to first 10 issues
                issue_log = DataQualityLog(
                    symbol='BATCH',
                    timeframe='BATCH',
                    source=validation_result['source'],
                    issue_type="validation_issue",
                    description=issue,
                    severity="warning"
                )
                self.session.add(issue_log)
            
            # Log anomalies
            for anomaly in validation_result['anomalies'][:10]:  # Limit to first 10 anomalies
                anomaly_log = DataQualityLog(
                    symbol='BATCH',
                    timeframe='BATCH',
                    source=validation_result['source'],
                    issue_type="anomaly_detected",
                    description=anomaly,
                    severity="warning"
                )
                self.session.add(anomaly_log)
            
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Error logging validation results: {e}")
            self.session.rollback()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation results.
        
        Returns:
            Validation summary dictionary
        """
        if not self.validation_results:
            return {'message': 'No validation results available'}
        
        total_records = sum(result['total_records'] for result in self.validation_results.values())
        total_valid = sum(result['valid_records'] for result in self.validation_results.values())
        total_anomalies = sum(result['anomalies_detected'] for result in self.validation_results.values())
        
        avg_quality_score = np.mean([result['quality_score'] for result in self.validation_results.values()])
        
        return {
            'total_sources': len(self.validation_results),
            'total_records': total_records,
            'total_valid_records': total_valid,
            'total_anomalies': total_anomalies,
            'average_quality_score': avg_quality_score,
            'sources': list(self.validation_results.keys())
        }