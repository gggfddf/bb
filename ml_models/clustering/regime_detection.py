"""
Regime Detection using KMeans Clustering

A system for detecting market regimes using KMeans clustering.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import structlog

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = structlog.get_logger()

@dataclass
class RegimeInfo:
    """Information about a detected regime."""
    regime_id: int
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    confidence: float
    features: Dict[str, float]

class KMeansRegimeDetector:
    """Regime detector using KMeans clustering."""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.kmeans = None
        self.regime_mapping = {}
    
    def extract_features(self, prices: np.ndarray, returns: np.ndarray, 
                        volumes: Optional[np.ndarray] = None, window_size: int = 20) -> np.ndarray:
        """Extract features for regime detection."""
        features = []
        
        for i in range(len(prices) - window_size + 1):
            price_window = prices[i:i + window_size]
            return_window = returns[i:i + window_size]
            
            # Calculate features
            price_mean = np.mean(price_window)
            price_std = np.std(price_window)
            return_mean = np.mean(return_window)
            return_std = np.std(return_window)
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            feature_vector = [price_mean, price_std, return_mean, return_std, price_trend]
            
            if volumes is not None:
                volume_window = volumes[i:i + window_size]
                volume_mean = np.mean(volume_window)
                feature_vector.append(volume_mean)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def detect_regimes(self, features: np.ndarray) -> List[RegimeInfo]:
        """Detect regimes using KMeans clustering."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for regime detection")
            return []
        
        # Fit KMeans
        self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        cluster_labels = self.kmeans.fit_predict(features)
        
        # Analyze clusters and assign regime types
        regimes = []
        for i in range(self.n_regimes):
            cluster_mask = cluster_labels == i
            cluster_features = features[cluster_mask]
            
            if len(cluster_features) == 0:
                continue
            
            # Determine regime type based on features
            avg_return = np.mean(cluster_features[:, 2])  # return_mean
            avg_volatility = np.mean(cluster_features[:, 3])  # return_std
            avg_trend = np.mean(cluster_features[:, 4])  # price_trend
            
            if avg_trend > 0.01 and avg_return > 0:
                regime_type = 'bull'
            elif avg_trend < -0.01 and avg_return < 0:
                regime_type = 'bear'
            elif avg_volatility > 0.02:
                regime_type = 'volatile'
            else:
                regime_type = 'sideways'
            
            regime = RegimeInfo(
                regime_id=i,
                regime_type=regime_type,
                confidence=len(cluster_features) / len(features),
                features={
                    'avg_return': avg_return,
                    'avg_volatility': avg_volatility,
                    'avg_trend': avg_trend
                }
            )
            regimes.append(regime)
        
        return regimes

# Convenience function
def create_regime_detector(n_regimes: int = 4) -> KMeansRegimeDetector:
    """Create a regime detector."""
    return KMeansRegimeDetector(n_regimes)