"""
Feature Scaling Module

Implements comprehensive feature scaling and normalization techniques:
- Standard Scaler (Z-score normalization)
- Min-Max Scaler
- Robust Scaler
- Power Transformer
- Quantile Transformer
- Custom scaling methods

Features:
- Multiple scaling algorithms
- Automatic scaling method selection
- Outlier handling
- Scaling validation
- Inverse transformation
- Batch processing support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import structlog
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    PowerTransformer, QuantileTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats

logger = structlog.get_logger()

class ScalingMethod(Enum):
    """Scaling method enumeration."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    POWER = "power"
    QUANTILE = "quantile"
    CUSTOM = "custom"

class OutlierHandling(Enum):
    """Outlier handling methods."""
    NONE = "none"
    CLIP = "clip"
    REMOVE = "remove"
    WINSORIZE = "winsorize"

@dataclass
class ScalingConfig:
    """Configuration for feature scaling."""
    method: ScalingMethod = ScalingMethod.STANDARD
    outlier_handling: OutlierHandling = OutlierHandling.NONE
    outlier_threshold: float = 3.0
    clip_range: Tuple[float, float] = (-3.0, 3.0)
    winsorize_limits: Tuple[float, float] = (0.05, 0.95)
    handle_zeros: bool = True
    handle_infinite: bool = True
    handle_missing: bool = True
    random_state: int = 42

@dataclass
class ScalingResult:
    """Result of feature scaling operation."""
    scaled_data: np.ndarray
    scaler: Any
    scaling_params: Dict[str, Any]
    outlier_info: Dict[str, Any]
    metadata: Dict[str, Any]

class FeatureScaler:
    """
    Comprehensive feature scaling implementation.
    """
    
    def __init__(self, config: ScalingConfig = None):
        """
        Initialize feature scaler.
        
        Args:
            config: Scaling configuration
        """
        self.config = config or ScalingConfig()
        self.scaler = None
        self.is_fitted = False
        self.scaling_params = {}
        self.outlier_info = {}
    
    def fit(self, data: np.ndarray) -> 'FeatureScaler':
        """
        Fit the scaler to the data.
        
        Args:
            data: Input data array
            
        Returns:
            Self for chaining
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Handle outliers
        data_clean, outlier_info = self._handle_outliers(data)
        self.outlier_info = outlier_info
        
        # Create and fit scaler
        self.scaler = self._create_scaler()
        self.scaler.fit(data_clean)
        
        # Store scaling parameters
        self.scaling_params = self._extract_scaling_params()
        
        self.is_fitted = True
        logger.info(f"Feature scaler fitted with method: {self.config.method.value}")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            data: Input data array
            
        Returns:
            Scaled data array
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transformation")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Handle outliers in transform data
        data_clean, _ = self._handle_outliers(data)
        
        # Transform data
        scaled_data = self.scaler.transform(data_clean)
        
        return scaled_data
    
    def fit_transform(self, data: np.ndarray) -> ScalingResult:
        """
        Fit scaler and transform data.
        
        Args:
            data: Input data array
            
        Returns:
            ScalingResult with scaled data and metadata
        """
        self.fit(data)
        scaled_data = self.transform(data)
        
        metadata = {
            "scaling_method": self.config.method.value,
            "outlier_handling": self.config.outlier_handling.value,
            "data_shape": data.shape,
            "scaled_shape": scaled_data.shape,
            "is_fitted": self.is_fitted
        }
        
        return ScalingResult(
            scaled_data=scaled_data,
            scaler=self.scaler,
            scaling_params=self.scaling_params,
            outlier_info=self.outlier_info,
            metadata=metadata
        )
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            scaled_data: Scaled data array
            
        Returns:
            Original scale data array
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transformation")
        
        if scaled_data.ndim == 1:
            scaled_data = scaled_data.reshape(-1, 1)
        
        # Inverse transform
        original_data = self.scaler.inverse_transform(scaled_data)
        
        return original_data
    
    def _create_scaler(self) -> BaseEstimator:
        """Create scaler based on configuration."""
        if self.config.method == ScalingMethod.STANDARD:
            return StandardScaler()
        elif self.config.method == ScalingMethod.MINMAX:
            return MinMaxScaler()
        elif self.config.method == ScalingMethod.ROBUST:
            return RobustScaler()
        elif self.config.method == ScalingMethod.POWER:
            return PowerTransformer(method='yeo-johnson', standardize=True)
        elif self.config.method == ScalingMethod.QUANTILE:
            return QuantileTransformer(output_distribution='normal', random_state=self.config.random_state)
        else:
            raise ValueError(f"Unsupported scaling method: {self.config.method}")
    
    def _handle_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Handle outliers in the data."""
        outlier_info = {
            "outliers_detected": 0,
            "outlier_indices": [],
            "outlier_values": [],
            "handling_method": self.config.outlier_handling.value
        }
        
        if self.config.outlier_handling == OutlierHandling.NONE:
            return data, outlier_info
        
        data_clean = data.copy()
        
        for col in range(data.shape[1]):
            column_data = data[:, col]
            
            # Detect outliers using Z-score
            z_scores = np.abs(stats.zscore(column_data, nan_policy='omit'))
            outlier_mask = z_scores > self.config.outlier_threshold
            
            outlier_indices = np.where(outlier_mask)[0]
            outlier_values = column_data[outlier_mask]
            
            outlier_info["outliers_detected"] += len(outlier_indices)
            outlier_info["outlier_indices"].extend(outlier_indices.tolist())
            outlier_info["outlier_values"].extend(outlier_values.tolist())
            
            if self.config.outlier_handling == OutlierHandling.CLIP:
                # Clip outliers to threshold
                lower_bound = np.percentile(column_data, 25) - 1.5 * (np.percentile(column_data, 75) - np.percentile(column_data, 25))
                upper_bound = np.percentile(column_data, 75) + 1.5 * (np.percentile(column_data, 75) - np.percentile(column_data, 25))
                data_clean[:, col] = np.clip(column_data, lower_bound, upper_bound)
            
            elif self.config.outlier_handling == OutlierHandling.REMOVE:
                # Remove outliers (replace with NaN)
                data_clean[outlier_mask, col] = np.nan
            
            elif self.config.outlier_handling == OutlierHandling.WINSORIZE:
                # Winsorize outliers
                lower_limit = np.percentile(column_data, self.config.winsorize_limits[0] * 100)
                upper_limit = np.percentile(column_data, self.config.winsorize_limits[1] * 100)
                data_clean[:, col] = np.clip(column_data, lower_limit, upper_limit)
        
        return data_clean, outlier_info
    
    def _extract_scaling_params(self) -> Dict[str, Any]:
        """Extract scaling parameters from fitted scaler."""
        params = {}
        
        if hasattr(self.scaler, 'mean_'):
            params['mean'] = self.scaler.mean_
        if hasattr(self.scaler, 'scale_'):
            params['scale'] = self.scaler.scale_
        if hasattr(self.scaler, 'min_'):
            params['min'] = self.scaler.min_
        if hasattr(self.scaler, 'data_min_'):
            params['data_min'] = self.scaler.data_min_
        if hasattr(self.scaler, 'data_max_'):
            params['data_max'] = self.scaler.data_max_
        if hasattr(self.scaler, 'center_'):
            params['center'] = self.scaler.center_
        if hasattr(self.scaler, 'scale_'):
            params['scale'] = self.scaler.scale_
        
        return params

class AdaptiveScaler(BaseEstimator, TransformerMixin):
    """
    Adaptive scaler that automatically selects the best scaling method.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize adaptive scaler.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.best_method = None
        self.best_scaler = None
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'AdaptiveScaler':
        """
        Fit adaptive scaler by testing multiple methods.
        
        Args:
            data: Input data array
            
        Returns:
            Self for chaining
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Test different scaling methods
        methods = [
            ScalingMethod.STANDARD,
            ScalingMethod.MINMAX,
            ScalingMethod.ROBUST,
            ScalingMethod.POWER,
            ScalingMethod.QUANTILE
        ]
        
        best_score = float('inf')
        best_method = ScalingMethod.STANDARD
        
        for method in methods:
            try:
                config = ScalingConfig(method=method, random_state=self.random_state)
                scaler = FeatureScaler(config)
                result = scaler.fit_transform(data)
                
                # Evaluate scaling quality (lower is better)
                score = self._evaluate_scaling_quality(result.scaled_data)
                
                if score < best_score:
                    best_score = score
                    best_method = method
                    self.best_scaler = scaler
                    
            except Exception as e:
                logger.warning(f"Failed to test scaling method {method.value}: {e}")
                continue
        
        self.best_method = best_method
        self.is_fitted = True
        
        logger.info(f"Adaptive scaler selected method: {best_method.value}")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using best scaling method.
        
        Args:
            data: Input data array
            
        Returns:
            Scaled data array
        """
        if not self.is_fitted:
            raise ValueError("Adaptive scaler must be fitted before transformation")
        
        return self.best_scaler.transform(data)
    
    def _evaluate_scaling_quality(self, scaled_data: np.ndarray) -> float:
        """
        Evaluate the quality of scaling.
        
        Args:
            scaled_data: Scaled data array
            
        Returns:
            Quality score (lower is better)
        """
        # Calculate multiple quality metrics
        metrics = []
        
        # 1. Normality test (lower p-value is better)
        for col in range(scaled_data.shape[1]):
            if not np.all(np.isnan(scaled_data[:, col])):
                _, p_value = stats.normaltest(scaled_data[:, col])
                metrics.append(1 - p_value)  # Convert to score where lower is better
        
        # 2. Skewness (closer to 0 is better)
        skewness = np.abs(stats.skew(scaled_data, nan_policy='omit'))
        metrics.append(np.mean(skewness))
        
        # 3. Kurtosis (closer to 3 is better for normal distribution)
        kurtosis = np.abs(stats.kurtosis(scaled_data, nan_policy='omit') - 3)
        metrics.append(np.mean(kurtosis))
        
        # 4. Range (prefer reasonable range)
        data_range = np.ptp(scaled_data, axis=0)
        range_score = np.mean(np.abs(data_range - 6))  # Prefer range around 6 (3 std devs)
        metrics.append(range_score)
        
        return np.mean(metrics)

class BatchScaler:
    """
    Batch processing scaler for large datasets.
    """
    
    def __init__(self, config: ScalingConfig, batch_size: int = 1000):
        """
        Initialize batch scaler.
        
        Args:
            config: Scaling configuration
            batch_size: Size of batches for processing
        """
        self.config = config
        self.batch_size = batch_size
        self.scaler = FeatureScaler(config)
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'BatchScaler':
        """
        Fit scaler using batch processing.
        
        Args:
            data: Input data array
            
        Returns:
            Self for chaining
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Use first batch to fit the scaler
        first_batch = data[:min(self.batch_size, len(data))]
        self.scaler.fit(first_batch)
        self.is_fitted = True
        
        logger.info(f"Batch scaler fitted with {len(first_batch)} samples")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using batch processing.
        
        Args:
            data: Input data array
            
        Returns:
            Scaled data array
        """
        if not self.is_fitted:
            raise ValueError("Batch scaler must be fitted before transformation")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        scaled_data = np.zeros_like(data)
        
        # Process data in batches
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            scaled_batch = self.scaler.transform(batch)
            scaled_data[i:i + len(batch)] = scaled_batch
        
        return scaled_data

# Convenience functions
def scale_features(data: np.ndarray, method: str = "standard", 
                  outlier_handling: str = "none") -> ScalingResult:
    """
    Convenience function for feature scaling.
    
    Args:
        data: Input data array
        method: Scaling method
        outlier_handling: Outlier handling method
        
    Returns:
        ScalingResult
    """
    config = ScalingConfig(
        method=ScalingMethod(method),
        outlier_handling=OutlierHandling(outlier_handling)
    )
    
    scaler = FeatureScaler(config)
    return scaler.fit_transform(data)

def adaptive_scale_features(data: np.ndarray) -> ScalingResult:
    """
    Convenience function for adaptive feature scaling.
    
    Args:
        data: Input data array
        
    Returns:
        ScalingResult
    """
    scaler = AdaptiveScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    
    metadata = {
        "scaling_method": scaler.best_method.value,
        "is_adaptive": True,
        "data_shape": data.shape,
        "scaled_shape": scaled_data.shape
    }
    
    return ScalingResult(
        scaled_data=scaled_data,
        scaler=scaler.best_scaler,
        scaling_params=scaler.best_scaler.scaling_params,
        outlier_info=scaler.best_scaler.outlier_info,
        metadata=metadata
    )

def batch_scale_features(data: np.ndarray, batch_size: int = 1000,
                        method: str = "standard") -> ScalingResult:
    """
    Convenience function for batch feature scaling.
    
    Args:
        data: Input data array
        batch_size: Batch size for processing
        method: Scaling method
        
    Returns:
        ScalingResult
    """
    config = ScalingConfig(method=ScalingMethod(method))
    scaler = BatchScaler(config, batch_size)
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    
    metadata = {
        "scaling_method": method,
        "batch_size": batch_size,
        "is_batch": True,
        "data_shape": data.shape,
        "scaled_shape": scaled_data.shape
    }
    
    return ScalingResult(
        scaled_data=scaled_data,
        scaler=scaler.scaler,
        scaling_params=scaler.scaler.scaling_params,
        outlier_info=scaler.scaler.outlier_info,
        metadata=metadata
    )