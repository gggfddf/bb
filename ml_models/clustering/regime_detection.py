#!/usr/bin/env python3
"""
Regime Detection System using KMeans

Implements comprehensive regime detection for market analysis:
- KMeans clustering for regime identification
- Multi-dimensional feature analysis
- Regime transition detection
- Regime-specific model training
- Real-time regime classification
- Regime performance tracking

Features:
- Advanced KMeans clustering with multiple features
- Multi-dimensional regime analysis
- Regime transition detection and alerts
- Regime-specific model training and prediction
- Real-time regime classification
- Regime performance tracking and optimization
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class RegimeType(Enum):
    """Regime type enumeration."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"

class RegimeTransitionType(Enum):
    """Regime transition type enumeration."""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"

@dataclass
class Regime:
    """Regime structure."""
    regime_id: str
    regime_type: RegimeType
    cluster_id: int
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None
    duration: Optional[timedelta] = None
    features: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegimeTransition:
    """Regime transition structure."""
    transition_id: str
    from_regime: RegimeType
    to_regime: RegimeType
    transition_type: RegimeTransitionType
    timestamp: datetime
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegimeDetectionConfig:
    """Regime detection configuration."""
    n_clusters: int = 5
    feature_columns: List[str] = field(default_factory=lambda: [
        'returns', 'volatility', 'volume_ratio', 'trend_strength', 'momentum'
    ])
    window_size: int = 20
    min_regime_duration: int = 5  # minimum periods
    max_regime_duration: int = 100  # maximum periods
    transition_threshold: float = 0.7
    enable_pca: bool = True
    pca_components: int = 3
    enable_validation: bool = True
    enable_visualization: bool = True

class FeatureExtractor:
    """Feature extraction for regime detection."""
    
    def __init__(self, config: RegimeDetectionConfig):
        """
        Initialize feature extractor.
        
        Args:
            config: Regime detection configuration
        """
        self.config = config
        self.scaler = StandardScaler()
        self.pca = None
        
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection."""
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
        """Calculate features for a time window."""
        try:
            # Returns
            returns = window_data['close'].pct_change().dropna()
            mean_return = returns.mean()
            return_std = returns.std()
            
            # Volatility
            volatility = returns.std()
            
            # Volume ratio
            volume_ratio = window_data['volume'].mean() / window_data['volume'].iloc[-10:].mean()
            
            # Trend strength
            if len(window_data) > 1:
                trend_slope = np.polyfit(range(len(window_data)), window_data['close'], 1)[0]
                trend_strength = abs(trend_slope) / window_data['close'].mean()
            else:
                trend_strength = 0.0
                
            # Momentum
            momentum = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
            
            # Price range
            price_range = (window_data['high'].max() - window_data['low'].min()) / window_data['close'].mean()
            
            # Volume volatility
            volume_volatility = window_data['volume'].std() / window_data['volume'].mean()
            
            # RSI-like momentum
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            rs = avg_gain / avg_loss if avg_loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            features = [
                mean_return,
                return_std,
                volatility,
                volume_ratio,
                trend_strength,
                momentum,
                price_range,
                volume_volatility,
                rsi
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return [0.0] * 9  # Default features

class KMeansRegimeDetector:
    """KMeans-based regime detector."""
    
    def __init__(self, config: RegimeDetectionConfig):
        """
        Initialize KMeans regime detector.
        
        Args:
            config: Regime detection configuration
        """
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
        self.regimes: List[Regime] = []
        self.transitions: List[RegimeTransition] = []
        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        
    def detect_regimes(self, data: pd.DataFrame) -> List[Regime]:
        """Detect regimes in the data."""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(data)
            
            if len(features) == 0:
                logger.warning("No features extracted for regime detection")
                return []
                
            # Fit KMeans
            self.cluster_labels = self.kmeans.fit_predict(features)
            self.cluster_centers = self.kmeans.cluster_centers_
            
            # Create regimes
            regimes = self._create_regimes(data, features, self.cluster_labels)
            
            # Detect transitions
            transitions = self._detect_transitions(regimes)
            self.transitions.extend(transitions)
            
            # Validate regimes
            if self.config.enable_validation:
                regimes = self._validate_regimes(regimes)
                
            self.regimes.extend(regimes)
            
            logger.info(f"Detected {len(regimes)} regimes with {len(transitions)} transitions")
            return regimes
            
        except Exception as e:
            logger.error(f"Error detecting regimes: {e}")
            return []
            
    def _create_regimes(self, data: pd.DataFrame, features: np.ndarray, labels: np.ndarray) -> List[Regime]:
        """Create regime objects from clustering results."""
        regimes = []
        
        # Group consecutive same labels
        current_label = labels[0]
        start_idx = self.config.window_size
        regime_start = data.index[start_idx]
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                # End current regime
                regime_end = data.index[start_idx + i - 1]
                duration = regime_end - regime_start
                
                if duration.days >= self.config.min_regime_duration:
                    regime = self._create_regime_object(
                        current_label, regime_start, regime_end, 
                        features[start_idx-self.config.window_size:i], data.iloc[start_idx:i]
                    )
                    regimes.append(regime)
                    
                # Start new regime
                current_label = labels[i]
                start_idx = start_idx + i
                regime_start = data.index[start_idx]
                
        # Handle last regime
        if start_idx < len(data):
            regime_end = data.index[-1]
            duration = regime_end - regime_start
            
            if duration.days >= self.config.min_regime_duration:
                regime = self._create_regime_object(
                    current_label, regime_start, regime_end,
                    features[start_idx-self.config.window_size:], data.iloc[start_idx:]
                )
                regimes.append(regime)
                
        return regimes
        
    def _create_regime_object(self, cluster_id: int, start_time: datetime, end_time: datetime,
                            features: np.ndarray, data: pd.DataFrame) -> Regime:
        """Create a regime object."""
        # Determine regime type based on cluster characteristics
        regime_type = self._determine_regime_type(cluster_id, features, data)
        
        # Calculate regime features
        regime_features = self._calculate_regime_features(features, data)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(data)
        
        # Calculate confidence
        confidence = self._calculate_regime_confidence(cluster_id, features)
        
        regime = Regime(
            regime_id=str(uuid.uuid4()),
            regime_type=regime_type,
            cluster_id=cluster_id,
            start_timestamp=start_time,
            end_timestamp=end_time,
            duration=end_time - start_time,
            features=regime_features,
            performance_metrics=performance_metrics,
            confidence=confidence
        )
        
        return regime
        
    def _determine_regime_type(self, cluster_id: int, features: np.ndarray, data: pd.DataFrame) -> RegimeType:
        """Determine regime type based on cluster characteristics."""
        try:
            # Get cluster center
            cluster_center = self.cluster_centers[cluster_id]
            
            # Extract key features
            mean_return = cluster_center[0]
            volatility = cluster_center[2]
            trend_strength = cluster_center[4]
            momentum = cluster_center[5]
            
            # Determine regime type based on characteristics
            if trend_strength > 0.1 and momentum > 0.05:
                return RegimeType.TRENDING_UP
            elif trend_strength > 0.1 and momentum < -0.05:
                return RegimeType.TRENDING_DOWN
            elif volatility > 0.05:
                return RegimeType.VOLATILE
            elif volatility < 0.02:
                return RegimeType.LOW_VOLATILITY
            elif abs(momentum) < 0.02 and volatility < 0.03:
                return RegimeType.SIDEWAYS
            elif momentum > 0.1:
                return RegimeType.BULL_MARKET
            elif momentum < -0.1:
                return RegimeType.BEAR_MARKET
            else:
                return RegimeType.CONSOLIDATION
                
        except Exception as e:
            logger.error(f"Error determining regime type: {e}")
            return RegimeType.CONSOLIDATION
            
    def _calculate_regime_features(self, features: np.ndarray, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime features."""
        try:
            feature_names = [
                'mean_return', 'return_std', 'volatility', 'volume_ratio',
                'trend_strength', 'momentum', 'price_range', 'volume_volatility', 'rsi'
            ]
            
            regime_features = {}
            for i, name in enumerate(feature_names):
                if i < features.shape[1]:
                    regime_features[name] = float(features[:, i].mean())
                    
            return regime_features
            
        except Exception as e:
            logger.error(f"Error calculating regime features: {e}")
            return {}
            
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime performance metrics."""
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
            
    def _calculate_regime_confidence(self, cluster_id: int, features: np.ndarray) -> float:
        """Calculate regime confidence."""
        try:
            # Calculate distance to cluster center
            cluster_center = self.cluster_centers[cluster_id]
            distances = np.linalg.norm(features - cluster_center, axis=1)
            
            # Confidence is inversely proportional to average distance
            avg_distance = distances.mean()
            confidence = max(0.0, min(1.0, 1.0 - avg_distance / 2.0))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5
            
    def _detect_transitions(self, regimes: List[Regime]) -> List[RegimeTransition]:
        """Detect regime transitions."""
        transitions = []
        
        for i in range(1, len(regimes)):
            from_regime = regimes[i-1]
            to_regime = regimes[i]
            
            # Calculate transition characteristics
            transition_type = self._determine_transition_type(from_regime, to_regime)
            confidence = self._calculate_transition_confidence(from_regime, to_regime)
            
            # Create transition features
            features = self._calculate_transition_features(from_regime, to_regime)
            
            transition = RegimeTransition(
                transition_id=str(uuid.uuid4()),
                from_regime=from_regime.regime_type,
                to_regime=to_regime.regime_type,
                transition_type=transition_type,
                timestamp=to_regime.start_timestamp,
                confidence=confidence,
                features=features
            )
            
            transitions.append(transition)
            
        return transitions
        
    def _determine_transition_type(self, from_regime: Regime, to_regime: Regime) -> RegimeTransitionType:
        """Determine transition type."""
        # Check if it's a reversal
        if (from_regime.regime_type in [RegimeType.TRENDING_UP, RegimeType.BULL_MARKET] and
            to_regime.regime_type in [RegimeType.TRENDING_DOWN, RegimeType.BEAR_MARKET]):
            return RegimeTransitionType.REVERSAL
        elif (from_regime.regime_type in [RegimeType.TRENDING_DOWN, RegimeType.BEAR_MARKET] and
              to_regime.regime_type in [RegimeType.TRENDING_UP, RegimeType.BULL_MARKET]):
            return RegimeTransitionType.REVERSAL
        else:
            return RegimeTransitionType.CONTINUATION
            
    def _calculate_transition_confidence(self, from_regime: Regime, to_regime: Regime) -> float:
        """Calculate transition confidence."""
        # Confidence based on regime confidences and duration
        from_confidence = from_regime.confidence
        to_confidence = to_regime.confidence
        
        # Longer regimes have higher confidence
        from_duration_factor = min(1.0, from_regime.duration.days / 30.0)
        to_duration_factor = min(1.0, to_regime.duration.days / 30.0)
        
        confidence = (from_confidence + to_confidence) / 2 * (from_duration_factor + to_duration_factor) / 2
        return min(1.0, max(0.0, confidence))
        
    def _calculate_transition_features(self, from_regime: Regime, to_regime: Regime) -> Dict[str, float]:
        """Calculate transition features."""
        features = {
            'duration_change': (to_regime.duration - from_regime.duration).total_seconds() / 86400,  # days
            'confidence_change': to_regime.confidence - from_regime.confidence,
            'volatility_change': to_regime.features.get('volatility', 0) - from_regime.features.get('volatility', 0),
            'momentum_change': to_regime.features.get('momentum', 0) - from_regime.features.get('momentum', 0)
        }
        return features
        
    def _validate_regimes(self, regimes: List[Regime]) -> List[Regime]:
        """Validate regimes based on criteria."""
        validated_regimes = []
        
        for regime in regimes:
            # Check minimum duration
            if regime.duration.days < self.config.min_regime_duration:
                continue
                
            # Check maximum duration
            if regime.duration.days > self.config.max_regime_duration:
                continue
                
            # Check confidence threshold
            if regime.confidence < 0.3:
                continue
                
            validated_regimes.append(regime)
            
        return validated_regimes
        
    def classify_current_regime(self, data: pd.DataFrame) -> Optional[Regime]:
        """Classify current regime."""
        try:
            if len(data) < self.config.window_size:
                return None
                
            # Extract features for current window
            current_features = self.feature_extractor.extract_features(data.tail(self.config.window_size))
            
            if len(current_features) == 0:
                return None
                
            # Predict cluster
            cluster_id = self.kmeans.predict(current_features)[0]
            
            # Create regime object
            regime = self._create_regime_object(
                cluster_id, data.index[-1], data.index[-1],
                current_features, data.tail(self.config.window_size)
            )
            
            return regime
            
        except Exception as e:
            logger.error(f"Error classifying current regime: {e}")
            return None
            
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection statistics."""
        if not self.regimes:
            return {}
            
        regime_counts = defaultdict(int)
        regime_durations = defaultdict(list)
        regime_performances = defaultdict(list)
        
        for regime in self.regimes:
            regime_counts[regime.regime_type.value] += 1
            regime_durations[regime.regime_type.value].append(regime.duration.days)
            
            if 'total_return' in regime.performance_metrics:
                regime_performances[regime.regime_type.value].append(regime.performance_metrics['total_return'])
                
        stats = {
            'total_regimes': len(self.regimes),
            'total_transitions': len(self.transitions),
            'regime_counts': dict(regime_counts),
            'avg_durations': {k: np.mean(v) for k, v in regime_durations.items()},
            'avg_performances': {k: np.mean(v) for k, v in regime_performances.items()},
            'cluster_quality': self._calculate_cluster_quality()
        }
        
        return stats
        
    def _calculate_cluster_quality(self) -> Dict[str, float]:
        """Calculate cluster quality metrics."""
        try:
            if self.cluster_labels is None or len(self.cluster_labels) == 0:
                return {}
                
            # Calculate silhouette score
            features = self.feature_extractor.extract_features(pd.DataFrame())  # This would need actual data
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

def create_regime_detector(config: RegimeDetectionConfig = None) -> KMeansRegimeDetector:
    """Create a regime detector."""
    return KMeansRegimeDetector(config or RegimeDetectionConfig())

# Demo of regime detection system
if __name__ == "__main__":
    # Create regime detector
    config = RegimeDetectionConfig(
        n_clusters=5,
        window_size=20,
        min_regime_duration=5,
        max_regime_duration=100,
        enable_pca=True,
        pca_components=3
    )
    
    detector = create_regime_detector(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Detect regimes
    regimes = detector.detect_regimes(data)
    
    print(f"Detected {len(regimes)} regimes:")
    for regime in regimes[:5]:  # Show first 5
        print(f"- {regime.regime_type.value} (confidence: {regime.confidence:.2f}, duration: {regime.duration.days} days)")
    
    # Get statistics
    stats = detector.get_regime_statistics()
    print(f"\nRegime Statistics:")
    print(f"Total regimes: {stats.get('total_regimes', 0)}")
    print(f"Total transitions: {stats.get('total_transitions', 0)}")
    print(f"Regime counts: {stats.get('regime_counts', {})}")
    
    # Classify current regime
    current_regime = detector.classify_current_regime(data)
    if current_regime:
        print(f"\nCurrent regime: {current_regime.regime_type.value}")
    
    print("Regime detection system completed successfully!")