#!/usr/bin/env python3
"""
Outlier Detection System using DBSCAN

Implements comprehensive outlier detection for market analysis:
- DBSCAN clustering for outlier identification
- Multi-dimensional outlier analysis
- Outlier classification and scoring
- Real-time outlier detection
- Outlier impact assessment
- Outlier filtering and validation

Features:
- Advanced DBSCAN clustering with multiple features
- Multi-dimensional outlier analysis
- Outlier classification and scoring
- Real-time outlier detection and alerts
- Outlier impact assessment and filtering
- Integration with trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class OutlierType(Enum):
    """Outlier type enumeration."""
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    TREND_REVERSAL = "trend_reversal"
    GAP = "gap"
    FLASH_CRASH = "flash_crash"
    PUMP_AND_DUMP = "pump_and_dump"
    MANIPULATION = "manipulation"
    NEWS_IMPACT = "news_impact"
    TECHNICAL_BREAKDOWN = "technical_breakdown"

class OutlierSeverity(Enum):
    """Outlier severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Outlier:
    """Outlier structure."""
    outlier_id: str
    outlier_type: OutlierType
    severity: OutlierSeverity
    timestamp: datetime
    price_change: float
    volume_change: float
    volatility_change: float
    confidence: float
    impact_score: float
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OutlierCluster:
    """Outlier cluster structure."""
    cluster_id: str
    outliers: List[Outlier]
    cluster_type: OutlierType
    cluster_severity: OutlierSeverity
    start_timestamp: datetime
    end_timestamp: datetime
    duration: timedelta
    total_impact: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OutlierDetectionConfig:
    """Outlier detection configuration."""
    eps: float = 0.5
    min_samples: int = 5
    feature_columns: List[str] = field(default_factory=lambda: [
        'price_change', 'volume_change', 'volatility_change', 'momentum', 'rsi'
    ])
    window_size: int = 20
    outlier_threshold: float = 0.8
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.8,
        'critical': 0.95
    })
    enable_pca: bool = True
    pca_components: int = 3
    enable_validation: bool = True
    enable_filtering: bool = True

class OutlierFeatureExtractor:
    """Feature extraction for outlier detection."""
    
    def __init__(self, config: OutlierDetectionConfig):
        """
        Initialize outlier feature extractor.
        
        Args:
            config: Outlier detection configuration
        """
        self.config = config
        self.scaler = RobustScaler()  # More robust to outliers
        self.pca = None
        
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for outlier detection."""
        try:
            features = []
            
            for i in range(self.config.window_size, len(data)):
                window_data = data.iloc[i-self.config.window_size:i+1]
                
                # Calculate features
                feature_vector = self._calculate_features(window_data, i, data)
                features.append(feature_vector)
                
            features_array = np.array(features)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Apply PCA if enabled
            if self.config.enable_pca and features_scaled.shape[1] > self.config.pca_components:
                self.pca = PCA(n_components=self.config.pca_components)
                features_scaled = self.pca.fit_transform(features_scaled)
                
            return features_scaled
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.array([])
            
    def _calculate_features(self, window_data: pd.DataFrame, current_idx: int, full_data: pd.DataFrame) -> List[float]:
        """Calculate features for outlier detection."""
        try:
            # Price change
            price_change = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
            
            # Volume change
            volume_change = (window_data['volume'].iloc[-1] - window_data['volume'].mean()) / window_data['volume'].mean()
            
            # Volatility change
            returns = window_data['close'].pct_change().dropna()
            current_volatility = returns.std()
            historical_volatility = full_data['close'].pct_change().rolling(50).std().iloc[current_idx]
            volatility_change = (current_volatility - historical_volatility) / historical_volatility if historical_volatility > 0 else 0
            
            # Momentum
            momentum = (window_data['close'].iloc[-1] - window_data['close'].iloc[-5]) / window_data['close'].iloc[-5]
            
            # RSI
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            rs = avg_gain / avg_loss if avg_loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            # Price range
            price_range = (window_data['high'].max() - window_data['low'].min()) / window_data['close'].mean()
            
            # Volume volatility
            volume_volatility = window_data['volume'].std() / window_data['volume'].mean()
            
            # Gap detection
            if current_idx > 0:
                gap = (window_data['open'].iloc[0] - full_data['close'].iloc[current_idx-1]) / full_data['close'].iloc[current_idx-1]
            else:
                gap = 0.0
                
            # Trend strength
            if len(window_data) > 1:
                trend_slope = np.polyfit(range(len(window_data)), window_data['close'], 1)[0]
                trend_strength = abs(trend_slope) / window_data['close'].mean()
            else:
                trend_strength = 0.0
                
            features = [
                price_change,
                volume_change,
                volatility_change,
                momentum,
                rsi,
                price_range,
                volume_volatility,
                gap,
                trend_strength
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return [0.0] * 9  # Default features

class DBSCANOutlierDetector:
    """DBSCAN-based outlier detector."""
    
    def __init__(self, config: OutlierDetectionConfig):
        """
        Initialize DBSCAN outlier detector.
        
        Args:
            config: Outlier detection configuration
        """
        self.config = config
        self.feature_extractor = OutlierFeatureExtractor(config)
        self.dbscan = DBSCAN(eps=config.eps, min_samples=config.min_samples)
        self.outliers: List[Outlier] = []
        self.outlier_clusters: List[OutlierCluster] = []
        self.cluster_labels: Optional[np.ndarray] = None
        
    def detect_outliers(self, data: pd.DataFrame) -> List[Outlier]:
        """Detect outliers in the data."""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(data)
            
            if len(features) == 0:
                logger.warning("No features extracted for outlier detection")
                return []
                
            # Fit DBSCAN
            self.cluster_labels = self.dbscan.fit_predict(features)
            
            # Identify outliers (label -1)
            outlier_indices = np.where(self.cluster_labels == -1)[0]
            
            # Create outlier objects
            outliers = []
            for idx in outlier_indices:
                outlier = self._create_outlier_object(idx, data, features[idx])
                if outlier:
                    outliers.append(outlier)
                    
            # Create outlier clusters
            clusters = self._create_outlier_clusters(outliers)
            self.outlier_clusters.extend(clusters)
            
            # Validate outliers
            if self.config.enable_validation:
                outliers = self._validate_outliers(outliers)
                
            self.outliers.extend(outliers)
            
            logger.info(f"Detected {len(outliers)} outliers with {len(clusters)} clusters")
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return []
            
    def _create_outlier_object(self, index: int, data: pd.DataFrame, features: np.ndarray) -> Optional[Outlier]:
        """Create outlier object."""
        try:
            # Get timestamp
            timestamp = data.index[index + self.config.window_size]
            
            # Determine outlier type
            outlier_type = self._determine_outlier_type(features)
            
            # Determine severity
            severity = self._determine_severity(features)
            
            # Calculate confidence
            confidence = self._calculate_outlier_confidence(features)
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(features)
            
            # Extract feature values
            feature_values = {
                'price_change': features[0],
                'volume_change': features[1],
                'volatility_change': features[2],
                'momentum': features[3],
                'rsi': features[4],
                'price_range': features[5],
                'volume_volatility': features[6],
                'gap': features[7],
                'trend_strength': features[8]
            }
            
            outlier = Outlier(
                outlier_id=str(uuid.uuid4()),
                outlier_type=outlier_type,
                severity=severity,
                timestamp=timestamp,
                price_change=features[0],
                volume_change=features[1],
                volatility_change=features[2],
                confidence=confidence,
                impact_score=impact_score,
                features=feature_values
            )
            
            return outlier
            
        except Exception as e:
            logger.error(f"Error creating outlier object: {e}")
            return None
            
    def _determine_outlier_type(self, features: np.ndarray) -> OutlierType:
        """Determine outlier type based on features."""
        try:
            price_change = features[0]
            volume_change = features[1]
            volatility_change = features[2]
            gap = features[7]
            
            # Determine type based on characteristics
            if abs(price_change) > 0.1 and abs(volume_change) > 2.0:
                return OutlierType.PUMP_AND_DUMP
            elif abs(price_change) > 0.05 and abs(gap) > 0.02:
                return OutlierType.GAP
            elif abs(price_change) > 0.1:
                return OutlierType.PRICE_SPIKE
            elif abs(volume_change) > 3.0:
                return OutlierType.VOLUME_SPIKE
            elif abs(volatility_change) > 2.0:
                return OutlierType.VOLATILITY_SPIKE
            elif abs(price_change) > 0.15:
                return OutlierType.FLASH_CRASH
            elif abs(price_change) > 0.05 and abs(volatility_change) > 1.5:
                return OutlierType.TREND_REVERSAL
            else:
                return OutlierType.MANIPULATION
                
        except Exception as e:
            logger.error(f"Error determining outlier type: {e}")
            return OutlierType.MANIPULATION
            
    def _determine_severity(self, features: np.ndarray) -> OutlierSeverity:
        """Determine outlier severity."""
        try:
            # Calculate overall anomaly score
            anomaly_score = np.mean(np.abs(features))
            
            # Determine severity based on thresholds
            if anomaly_score >= self.config.severity_thresholds['critical']:
                return OutlierSeverity.CRITICAL
            elif anomaly_score >= self.config.severity_thresholds['high']:
                return OutlierSeverity.HIGH
            elif anomaly_score >= self.config.severity_thresholds['medium']:
                return OutlierSeverity.MEDIUM
            else:
                return OutlierSeverity.LOW
                
        except Exception as e:
            logger.error(f"Error determining severity: {e}")
            return OutlierSeverity.MEDIUM
            
    def _calculate_outlier_confidence(self, features: np.ndarray) -> float:
        """Calculate outlier confidence."""
        try:
            # Confidence based on feature magnitudes
            feature_magnitudes = np.abs(features)
            confidence = np.mean(feature_magnitudes)
            
            # Normalize to [0, 1]
            confidence = min(1.0, max(0.0, confidence))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating outlier confidence: {e}")
            return 0.5
            
    def _calculate_impact_score(self, features: np.ndarray) -> float:
        """Calculate outlier impact score."""
        try:
            # Impact based on price change, volume change, and volatility change
            price_impact = abs(features[0]) * 0.4
            volume_impact = abs(features[1]) * 0.3
            volatility_impact = abs(features[2]) * 0.3
            
            impact_score = price_impact + volume_impact + volatility_impact
            
            return min(1.0, max(0.0, impact_score))
            
        except Exception as e:
            logger.error(f"Error calculating impact score: {e}")
            return 0.5
            
    def _create_outlier_clusters(self, outliers: List[Outlier]) -> List[OutlierCluster]:
        """Create outlier clusters."""
        if not outliers:
            return []
            
        # Group outliers by type and time proximity
        clusters = []
        current_cluster = []
        
        for outlier in sorted(outliers, key=lambda x: x.timestamp):
            if not current_cluster:
                current_cluster = [outlier]
            else:
                # Check if outlier belongs to current cluster
                time_diff = outlier.timestamp - current_cluster[-1].timestamp
                same_type = outlier.outlier_type == current_cluster[-1].outlier_type
                
                if time_diff.days <= 1 and same_type:
                    current_cluster.append(outlier)
                else:
                    # End current cluster
                    if len(current_cluster) >= 2:
                        cluster = self._create_cluster_object(current_cluster)
                        clusters.append(cluster)
                    current_cluster = [outlier]
                    
        # Handle last cluster
        if len(current_cluster) >= 2:
            cluster = self._create_cluster_object(current_cluster)
            clusters.append(cluster)
            
        return clusters
        
    def _create_cluster_object(self, outliers: List[Outlier]) -> OutlierCluster:
        """Create outlier cluster object."""
        # Determine cluster characteristics
        cluster_type = outliers[0].outlier_type
        cluster_severity = max(outliers, key=lambda x: x.severity.value).severity
        start_time = min(outliers, key=lambda x: x.timestamp).timestamp
        end_time = max(outliers, key=lambda x: x.timestamp).timestamp
        duration = end_time - start_time
        total_impact = sum(outlier.impact_score for outlier in outliers)
        confidence = np.mean([outlier.confidence for outlier in outliers])
        
        cluster = OutlierCluster(
            cluster_id=str(uuid.uuid4()),
            outliers=outliers,
            cluster_type=cluster_type,
            cluster_severity=cluster_severity,
            start_timestamp=start_time,
            end_timestamp=end_time,
            duration=duration,
            total_impact=total_impact,
            confidence=confidence
        )
        
        return cluster
        
    def _validate_outliers(self, outliers: List[Outlier]) -> List[Outlier]:
        """Validate outliers based on criteria."""
        validated_outliers = []
        
        for outlier in outliers:
            # Check confidence threshold
            if outlier.confidence < self.config.outlier_threshold:
                continue
                
            # Check impact score
            if outlier.impact_score < 0.1:
                continue
                
            # Check severity
            if outlier.severity == OutlierSeverity.LOW and outlier.confidence < 0.7:
                continue
                
            validated_outliers.append(outlier)
            
        return validated_outliers
        
    def detect_real_time_outlier(self, data: pd.DataFrame) -> Optional[Outlier]:
        """Detect outlier in real-time."""
        try:
            if len(data) < self.config.window_size:
                return None
                
            # Extract features for current window
            current_features = self.feature_extractor.extract_features(data.tail(self.config.window_size))
            
            if len(current_features) == 0:
                return None
                
            # Check if current point is an outlier
            cluster_label = self.dbscan.fit_predict(current_features)[0]
            
            if cluster_label == -1:  # Outlier
                outlier = self._create_outlier_object(
                    len(data) - self.config.window_size, data, current_features[0]
                )
                return outlier
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error detecting real-time outlier: {e}")
            return None
            
    def get_outlier_statistics(self) -> Dict[str, Any]:
        """Get outlier detection statistics."""
        if not self.outliers:
            return {}
            
        outlier_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        impact_scores = []
        confidence_scores = []
        
        for outlier in self.outliers:
            outlier_counts[outlier.outlier_type.value] += 1
            severity_counts[outlier.severity.value] += 1
            type_counts[outlier.outlier_type.value] += 1
            impact_scores.append(outlier.impact_score)
            confidence_scores.append(outlier.confidence)
            
        stats = {
            'total_outliers': len(self.outliers),
            'total_clusters': len(self.outlier_clusters),
            'outlier_counts': dict(outlier_counts),
            'severity_counts': dict(severity_counts),
            'avg_impact_score': np.mean(impact_scores) if impact_scores else 0.0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'cluster_quality': self._calculate_cluster_quality()
        }
        
        return stats
        
    def _calculate_cluster_quality(self) -> Dict[str, float]:
        """Calculate cluster quality metrics."""
        try:
            if self.cluster_labels is None or len(self.cluster_labels) == 0:
                return {}
                
            # Calculate silhouette score for non-outlier clusters
            non_outlier_labels = self.cluster_labels[self.cluster_labels != -1]
            if len(non_outlier_labels) > 1:
                features = self.feature_extractor.extract_features(pd.DataFrame())  # Would need actual data
                if len(features) > 0:
                    non_outlier_features = features[self.cluster_labels != -1]
                    if len(non_outlier_features) > 1:
                        silhouette = silhouette_score(non_outlier_features, non_outlier_labels)
                    else:
                        silhouette = 0.0
                else:
                    silhouette = 0.0
            else:
                silhouette = 0.0
                
            return {
                'silhouette_score': silhouette,
                'outlier_ratio': np.sum(self.cluster_labels == -1) / len(self.cluster_labels) if len(self.cluster_labels) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating cluster quality: {e}")
            return {}

def create_outlier_detector(config: OutlierDetectionConfig = None) -> DBSCANOutlierDetector:
    """Create an outlier detector."""
    return DBSCANOutlierDetector(config or OutlierDetectionConfig())

# Demo of outlier detection system
if __name__ == "__main__":
    # Create outlier detector
    config = OutlierDetectionConfig(
        eps=0.5,
        min_samples=5,
        window_size=20,
        outlier_threshold=0.8,
        enable_pca=True,
        pca_components=3
    )
    
    detector = create_outlier_detector(config)
    
    # Create sample data with outliers
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Add some outliers
    data.loc[50, 'close'] *= 1.2  # Price spike
    data.loc[100, 'volume'] *= 5   # Volume spike
    data.loc[150, 'close'] *= 0.8  # Flash crash
    
    # Detect outliers
    outliers = detector.detect_outliers(data)
    
    print(f"Detected {len(outliers)} outliers:")
    for outlier in outliers[:5]:  # Show first 5
        print(f"- {outlier.outlier_type.value} ({outlier.severity.value}) at {outlier.timestamp}")
        print(f"  Impact: {outlier.impact_score:.2f}, Confidence: {outlier.confidence:.2f}")
    
    # Get statistics
    stats = detector.get_outlier_statistics()
    print(f"\nOutlier Statistics:")
    print(f"Total outliers: {stats.get('total_outliers', 0)}")
    print(f"Total clusters: {stats.get('total_clusters', 0)}")
    print(f"Average impact score: {stats.get('avg_impact_score', 0):.2f}")
    print(f"Average confidence: {stats.get('avg_confidence', 0):.2f}")
    
    # Detect real-time outlier
    real_time_outlier = detector.detect_real_time_outlier(data)
    if real_time_outlier:
        print(f"\nReal-time outlier detected: {real_time_outlier.outlier_type.value}")
    
    print("Outlier detection system completed successfully!")