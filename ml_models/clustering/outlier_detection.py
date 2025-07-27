"""
Outlier Detection using DBSCAN

A system for detecting outliers in financial data using DBSCAN clustering.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import structlog

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = structlog.get_logger()

@dataclass
class OutlierInfo:
    """Information about detected outliers."""
    outlier_id: int
    outlier_score: float
    outlier_type: str  # 'price', 'volume', 'return'
    index: int
    features: Dict[str, float]

class DBSCANOutlierDetector:
    """Outlier detector using DBSCAN clustering."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = None
    
    def detect_price_outliers(self, prices: np.ndarray, window_size: int = 20) -> List[OutlierInfo]:
        """Detect price outliers using DBSCAN."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for outlier detection")
            return []
        
        # Extract features
        features = []
        for i in range(len(prices) - window_size + 1):
            window = prices[i:i + window_size]
            features.append([
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                window[-1] - window[0]  # price change
            ])
        
        features = np.array(features)
        
        # Normalize features
        features_normalized = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        # Apply DBSCAN
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.dbscan.fit_predict(features_normalized)
        
        # Identify outliers (label = -1)
        outliers = []
        for i, label in enumerate(labels):
            if label == -1:
                outlier = OutlierInfo(
                    outlier_id=i,
                    outlier_score=1.0,  # DBSCAN doesn't provide scores
                    outlier_type='price',
                    index=i + window_size - 1,
                    features={
                        'mean': features[i, 0],
                        'std': features[i, 1],
                        'max': features[i, 2],
                        'min': features[i, 3],
                        'change': features[i, 4]
                    }
                )
                outliers.append(outlier)
        
        return outliers
    
    def detect_return_outliers(self, returns: np.ndarray, window_size: int = 20) -> List[OutlierInfo]:
        """Detect return outliers using DBSCAN."""
        if not SKLEARN_AVAILABLE:
            return []
        
        # Extract features
        features = []
        for i in range(len(returns) - window_size + 1):
            window = returns[i:i + window_size]
            features.append([
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                np.sum(np.abs(window))  # total absolute return
            ])
        
        features = np.array(features)
        
        # Normalize features
        features_normalized = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        # Apply DBSCAN
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.dbscan.fit_predict(features_normalized)
        
        # Identify outliers
        outliers = []
        for i, label in enumerate(labels):
            if label == -1:
                outlier = OutlierInfo(
                    outlier_id=i,
                    outlier_score=1.0,
                    outlier_type='return',
                    index=i + window_size - 1,
                    features={
                        'mean': features[i, 0],
                        'std': features[i, 1],
                        'max': features[i, 2],
                        'min': features[i, 3],
                        'total_abs': features[i, 4]
                    }
                )
                outliers.append(outlier)
        
        return outliers

# Convenience function
def create_outlier_detector(eps: float = 0.5, min_samples: int = 5) -> DBSCANOutlierDetector:
    """Create an outlier detector."""
    return DBSCANOutlierDetector(eps, min_samples)