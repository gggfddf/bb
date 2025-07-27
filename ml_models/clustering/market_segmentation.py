#!/usr/bin/env python3
"""
Market Segmentation System using Hierarchical Clustering

Implements comprehensive market segmentation for trading analysis:
- Hierarchical clustering for market segmentation
- Multi-dimensional segment analysis
- Segment behavior profiling
- Real-time segment classification
- Segment performance tracking
- Segment-based strategy adaptation

Features:
- Advanced hierarchical clustering with multiple features
- Multi-dimensional market segment analysis
- Segment behavior profiling and classification
- Real-time segment classification
- Segment performance tracking and optimization
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class SegmentType(Enum):
    """Market segment type enumeration."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"
    LIQUID = "liquid"
    ILLIQUID = "illiquid"
    CORRELATED = "correlated"
    DECORRELATED = "decorrelated"

class SegmentBehavior(Enum):
    """Segment behavior enumeration."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    VOLATILE = "volatile"
    STABLE = "stable"

@dataclass
class MarketSegment:
    """Market segment structure."""
    segment_id: str
    segment_type: SegmentType
    behavior: SegmentBehavior
    cluster_id: int
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None
    duration: Optional[timedelta] = None
    features: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SegmentTransition:
    """Segment transition structure."""
    transition_id: str
    from_segment: SegmentType
    to_segment: SegmentType
    transition_type: str
    timestamp: datetime
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SegmentationConfig:
    """Market segmentation configuration."""
    n_clusters: int = 5
    linkage_method: str = "ward"
    feature_columns: List[str] = field(default_factory=lambda: [
        'volatility', 'volume_profile', 'trend_strength', 'momentum', 'correlation'
    ])
    window_size: int = 30
    min_segment_duration: int = 10
    max_segment_duration: int = 200
    transition_threshold: float = 0.7
    enable_pca: bool = True
    pca_components: int = 3
    enable_validation: bool = True
    enable_visualization: bool = True

class SegmentFeatureExtractor:
    """Feature extraction for market segmentation."""
    
    def __init__(self, config: SegmentationConfig):
        """
        Initialize segment feature extractor.
        
        Args:
            config: Segmentation configuration
        """
        self.config = config
        self.scaler = StandardScaler()
        self.pca = None
        
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for market segmentation."""
        try:
            features = []
            
            for i in range(self.config.window_size, len(data)):
                window_data = data.iloc[i-self.config.window_size:i+1]
                
                # Calculate features
                feature_vector = self._calculate_features(window_data)
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
            
    def _calculate_features(self, window_data: pd.DataFrame) -> List[float]:
        """Calculate features for market segmentation."""
        try:
            # Volatility
            returns = window_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Volume profile
            volume_mean = window_data['volume'].mean()
            volume_std = window_data['volume'].std()
            volume_profile = volume_std / volume_mean if volume_mean > 0 else 0
            
            # Trend strength
            if len(window_data) > 1:
                trend_slope = np.polyfit(range(len(window_data)), window_data['close'], 1)[0]
                trend_strength = abs(trend_slope) / window_data['close'].mean()
            else:
                trend_strength = 0.0
                
            # Momentum
            momentum = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
            
            # Correlation with market (simplified)
            correlation = np.random.uniform(-1, 1)  # Placeholder
            
            # Price range
            price_range = (window_data['high'].max() - window_data['low'].min()) / window_data['close'].mean()
            
            # Volume volatility
            volume_volatility = window_data['volume'].std() / window_data['volume'].mean()
            
            # RSI
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            rs = avg_gain / avg_loss if avg_loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            # Liquidity measure
            liquidity = volume_mean / price_range if price_range > 0 else 0
            
            features = [
                volatility,
                volume_profile,
                trend_strength,
                momentum,
                correlation,
                price_range,
                volume_volatility,
                rsi,
                liquidity
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return [0.0] * 9  # Default features

class HierarchicalMarketSegmenter:
    """Hierarchical clustering-based market segmenter."""
    
    def __init__(self, config: SegmentationConfig):
        """
        Initialize hierarchical market segmenter.
        
        Args:
            config: Segmentation configuration
        """
        self.config = config
        self.feature_extractor = SegmentFeatureExtractor(config)
        self.clustering = AgglomerativeClustering(
            n_clusters=config.n_clusters,
            linkage=config.linkage_method
        )
        self.segments: List[MarketSegment] = []
        self.transitions: List[SegmentTransition] = []
        self.cluster_labels: Optional[np.ndarray] = None
        
    def segment_market(self, data: pd.DataFrame) -> List[MarketSegment]:
        """Segment the market using hierarchical clustering."""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(data)
            
            if len(features) == 0:
                logger.warning("No features extracted for market segmentation")
                return []
                
            # Fit hierarchical clustering
            self.cluster_labels = self.clustering.fit_predict(features)
            
            # Create segments
            segments = self._create_segments(data, features, self.cluster_labels)
            
            # Detect transitions
            transitions = self._detect_transitions(segments)
            self.transitions.extend(transitions)
            
            # Validate segments
            if self.config.enable_validation:
                segments = self._validate_segments(segments)
                
            self.segments.extend(segments)
            
            logger.info(f"Created {len(segments)} market segments with {len(transitions)} transitions")
            return segments
            
        except Exception as e:
            logger.error(f"Error segmenting market: {e}")
            return []
            
    def _create_segments(self, data: pd.DataFrame, features: np.ndarray, labels: np.ndarray) -> List[MarketSegment]:
        """Create market segments from clustering results."""
        segments = []
        
        # Group consecutive same labels
        current_label = labels[0]
        start_idx = self.config.window_size
        segment_start = data.index[start_idx]
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                # End current segment
                segment_end = data.index[start_idx + i - 1]
                duration = segment_end - segment_start
                
                if duration.days >= self.config.min_segment_duration:
                    segment = self._create_segment_object(
                        current_label, segment_start, segment_end,
                        features[start_idx-self.config.window_size:i], data.iloc[start_idx:i]
                    )
                    segments.append(segment)
                    
                # Start new segment
                current_label = labels[i]
                start_idx = start_idx + i
                segment_start = data.index[start_idx]
                
        # Handle last segment
        if start_idx < len(data):
            segment_end = data.index[-1]
            duration = segment_end - segment_start
            
            if duration.days >= self.config.min_segment_duration:
                segment = self._create_segment_object(
                    current_label, segment_start, segment_end,
                    features[start_idx-self.config.window_size:], data.iloc[start_idx:]
                )
                segments.append(segment)
                
        return segments
        
    def _create_segment_object(self, cluster_id: int, start_time: datetime, end_time: datetime,
                              features: np.ndarray, data: pd.DataFrame) -> MarketSegment:
        """Create a market segment object."""
        # Determine segment type
        segment_type = self._determine_segment_type(cluster_id, features, data)
        
        # Determine behavior
        behavior = self._determine_segment_behavior(features, data)
        
        # Calculate segment features
        segment_features = self._calculate_segment_features(features, data)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(data)
        
        # Calculate confidence
        confidence = self._calculate_segment_confidence(cluster_id, features)
        
        segment = MarketSegment(
            segment_id=str(uuid.uuid4()),
            segment_type=segment_type,
            behavior=behavior,
            cluster_id=cluster_id,
            start_timestamp=start_time,
            end_timestamp=end_time,
            duration=end_time - start_time,
            features=segment_features,
            performance_metrics=performance_metrics,
            confidence=confidence
        )
        
        return segment
        
    def _determine_segment_type(self, cluster_id: int, features: np.ndarray, data: pd.DataFrame) -> SegmentType:
        """Determine segment type based on cluster characteristics."""
        try:
            # Extract key features
            volatility = features[0]
            volume_profile = features[1]
            trend_strength = features[2]
            momentum = features[3]
            correlation = features[4]
            liquidity = features[8]
            
            # Determine type based on characteristics
            if volatility > 0.05:
                return SegmentType.HIGH_VOLATILITY
            elif volatility < 0.02:
                return SegmentType.LOW_VOLATILITY
            elif trend_strength > 0.1:
                return SegmentType.TRENDING
            elif abs(momentum) < 0.02:
                return SegmentType.MEAN_REVERTING
            elif momentum > 0.05:
                return SegmentType.MOMENTUM
            elif momentum < -0.05:
                return SegmentType.CONTRARIAN
            elif liquidity > 1000:
                return SegmentType.LIQUID
            elif liquidity < 100:
                return SegmentType.ILLIQUID
            elif abs(correlation) > 0.7:
                return SegmentType.CORRELATED
            else:
                return SegmentType.DECORRELATED
                
        except Exception as e:
            logger.error(f"Error determining segment type: {e}")
            return SegmentType.LOW_VOLATILITY
            
    def _determine_segment_behavior(self, features: np.ndarray, data: pd.DataFrame) -> SegmentBehavior:
        """Determine segment behavior."""
        try:
            volatility = features[0]
            momentum = features[3]
            volume_profile = features[1]
            
            # Determine behavior based on characteristics
            if volatility > 0.05 and abs(momentum) > 0.1:
                return SegmentBehavior.AGGRESSIVE
            elif volatility < 0.02 and abs(momentum) < 0.02:
                return SegmentBehavior.CONSERVATIVE
            elif volatility > 0.05:
                return SegmentBehavior.VOLATILE
            elif volatility < 0.02:
                return SegmentBehavior.STABLE
            else:
                return SegmentBehavior.MODERATE
                
        except Exception as e:
            logger.error(f"Error determining segment behavior: {e}")
            return SegmentBehavior.MODERATE
            
    def _calculate_segment_features(self, features: np.ndarray, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate segment features."""
        try:
            feature_names = [
                'volatility', 'volume_profile', 'trend_strength', 'momentum',
                'correlation', 'price_range', 'volume_volatility', 'rsi', 'liquidity'
            ]
            
            segment_features = {}
            for i, name in enumerate(feature_names):
                if i < features.shape[1]:
                    segment_features[name] = float(features[:, i].mean())
                    
            return segment_features
            
        except Exception as e:
            logger.error(f"Error calculating segment features: {e}")
            return {}
            
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate segment performance metrics."""
        try:
            returns = data['close'].pct_change().dropna()
            
            metrics = {
                'total_return': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
                'mean_return': returns.mean() * 100,
                'volatility': returns.std() * 100,
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'win_rate': (returns > 0).mean() * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
            
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min() * 100
        except:
            return 0.0
            
    def _calculate_segment_confidence(self, cluster_id: int, features: np.ndarray) -> float:
        """Calculate segment confidence."""
        try:
            # Confidence based on feature consistency
            feature_std = np.std(features, axis=0)
            feature_mean = np.mean(features, axis=0)
            
            # Lower standard deviation indicates higher confidence
            cv = feature_std / (np.abs(feature_mean) + 1e-8)
            confidence = 1.0 - np.mean(cv)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating segment confidence: {e}")
            return 0.5
            
    def _detect_transitions(self, segments: List[MarketSegment]) -> List[SegmentTransition]:
        """Detect segment transitions."""
        transitions = []
        
        for i in range(1, len(segments)):
            from_segment = segments[i-1]
            to_segment = segments[i]
            
            # Calculate transition characteristics
            transition_type = self._determine_transition_type(from_segment, to_segment)
            confidence = self._calculate_transition_confidence(from_segment, to_segment)
            
            # Create transition features
            features = self._calculate_transition_features(from_segment, to_segment)
            
            transition = SegmentTransition(
                transition_id=str(uuid.uuid4()),
                from_segment=from_segment.segment_type,
                to_segment=to_segment.segment_type,
                transition_type=transition_type,
                timestamp=to_segment.start_timestamp,
                confidence=confidence,
                features=features
            )
            
            transitions.append(transition)
            
        return transitions
        
    def _determine_transition_type(self, from_segment: MarketSegment, to_segment: MarketSegment) -> str:
        """Determine transition type."""
        # Check if it's a regime change
        if (from_segment.segment_type in [SegmentType.HIGH_VOLATILITY, SegmentType.LOW_VOLATILITY] and
            to_segment.segment_type in [SegmentType.LOW_VOLATILITY, SegmentType.HIGH_VOLATILITY]):
            return "regime_change"
        elif (from_segment.segment_type in [SegmentType.TRENDING, SegmentType.MEAN_REVERTING] and
              to_segment.segment_type in [SegmentType.MEAN_REVERTING, SegmentType.TRENDING]):
            return "trend_change"
        else:
            return "behavior_change"
            
    def _calculate_transition_confidence(self, from_segment: MarketSegment, to_segment: MarketSegment) -> float:
        """Calculate transition confidence."""
        # Confidence based on segment confidences and duration
        from_confidence = from_segment.confidence
        to_confidence = to_segment.confidence
        
        # Longer segments have higher confidence
        from_duration_factor = min(1.0, from_segment.duration.days / 30.0)
        to_duration_factor = min(1.0, to_segment.duration.days / 30.0)
        
        confidence = (from_confidence + to_confidence) / 2 * (from_duration_factor + to_duration_factor) / 2
        return min(1.0, max(0.0, confidence))
        
    def _calculate_transition_features(self, from_segment: MarketSegment, to_segment: MarketSegment) -> Dict[str, float]:
        """Calculate transition features."""
        features = {
            'duration_change': (to_segment.duration - from_segment.duration).total_seconds() / 86400,  # days
            'confidence_change': to_segment.confidence - from_segment.confidence,
            'volatility_change': to_segment.features.get('volatility', 0) - from_segment.features.get('volatility', 0),
            'momentum_change': to_segment.features.get('momentum', 0) - from_segment.features.get('momentum', 0)
        }
        return features
        
    def _validate_segments(self, segments: List[MarketSegment]) -> List[MarketSegment]:
        """Validate segments based on criteria."""
        validated_segments = []
        
        for segment in segments:
            # Check minimum duration
            if segment.duration.days < self.config.min_segment_duration:
                continue
                
            # Check maximum duration
            if segment.duration.days > self.config.max_segment_duration:
                continue
                
            # Check confidence threshold
            if segment.confidence < 0.3:
                continue
                
            validated_segments.append(segment)
            
        return validated_segments
        
    def classify_current_segment(self, data: pd.DataFrame) -> Optional[MarketSegment]:
        """Classify current market segment."""
        try:
            if len(data) < self.config.window_size:
                return None
                
            # Extract features for current window
            current_features = self.feature_extractor.extract_features(data.tail(self.config.window_size))
            
            if len(current_features) == 0:
                return None
                
            # Predict cluster
            cluster_id = self.clustering.fit_predict(current_features)[0]
            
            # Create segment object
            segment = self._create_segment_object(
                cluster_id, data.index[-1], data.index[-1],
                current_features, data.tail(self.config.window_size)
            )
            
            return segment
            
        except Exception as e:
            logger.error(f"Error classifying current segment: {e}")
            return None
            
    def get_segment_statistics(self) -> Dict[str, Any]:
        """Get market segmentation statistics."""
        if not self.segments:
            return {}
            
        segment_counts = defaultdict(int)
        behavior_counts = defaultdict(int)
        type_counts = defaultdict(int)
        performance_scores = []
        confidence_scores = []
        
        for segment in self.segments:
            segment_counts[segment.segment_type.value] += 1
            behavior_counts[segment.behavior.value] += 1
            type_counts[segment.segment_type.value] += 1
            performance_scores.append(segment.performance_metrics.get('total_return', 0))
            confidence_scores.append(segment.confidence)
            
        stats = {
            'total_segments': len(self.segments),
            'total_transitions': len(self.transitions),
            'segment_counts': dict(segment_counts),
            'behavior_counts': dict(behavior_counts),
            'avg_performance': np.mean(performance_scores) if performance_scores else 0.0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'cluster_quality': self._calculate_cluster_quality()
        }
        
        return stats
        
    def _calculate_cluster_quality(self) -> Dict[str, float]:
        """Calculate cluster quality metrics."""
        try:
            if self.cluster_labels is None or len(self.cluster_labels) == 0:
                return {}
                
            # Calculate silhouette score
            features = self.feature_extractor.extract_features(pd.DataFrame())  # Would need actual data
            if len(features) > 0:
                silhouette = silhouette_score(features, self.cluster_labels)
                calinski = calinski_harabasz_score(features, self.cluster_labels)
            else:
                silhouette = 0.0
                calinski = 0.0
                
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski
            }
            
        except Exception as e:
            logger.error(f"Error calculating cluster quality: {e}")
            return {}

def create_market_segmenter(config: SegmentationConfig = None) -> HierarchicalMarketSegmenter:
    """Create a market segmenter."""
    return HierarchicalMarketSegmenter(config or SegmentationConfig())

# Demo of market segmentation system
if __name__ == "__main__":
    # Create market segmenter
    config = SegmentationConfig(
        n_clusters=5,
        linkage_method="ward",
        window_size=30,
        min_segment_duration=10,
        max_segment_duration=200,
        enable_pca=True,
        pca_components=3
    )
    
    segmenter = create_market_segmenter(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(300).cumsum() + 100,
        'high': np.random.randn(300).cumsum() + 102,
        'low': np.random.randn(300).cumsum() + 98,
        'close': np.random.randn(300).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 300)
    }, index=dates)
    
    # Segment market
    segments = segmenter.segment_market(data)
    
    print(f"Created {len(segments)} market segments:")
    for segment in segments[:5]:  # Show first 5
        print(f"- {segment.segment_type.value} ({segment.behavior.value})")
        print(f"  Duration: {segment.duration.days} days, Confidence: {segment.confidence:.2f}")
    
    # Get statistics
    stats = segmenter.get_segment_statistics()
    print(f"\nMarket Segmentation Statistics:")
    print(f"Total segments: {stats.get('total_segments', 0)}")
    print(f"Total transitions: {stats.get('total_transitions', 0)}")
    print(f"Segment counts: {stats.get('segment_counts', {})}")
    print(f"Average performance: {stats.get('avg_performance', 0):.2f}%")
    
    # Classify current segment
    current_segment = segmenter.classify_current_segment(data)
    if current_segment:
        print(f"\nCurrent segment: {current_segment.segment_type.value}")
    
    print("Market segmentation system completed successfully!")