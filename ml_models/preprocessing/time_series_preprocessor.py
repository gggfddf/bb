"""
Time Series Preprocessing Pipeline

A comprehensive preprocessing pipeline for time series data including
sequence creation, sliding window approach, data augmentation, normalization,
and validation split for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import random

logger = structlog.get_logger()

class ScalingMethod(Enum):
    """Supported scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"

class AugmentationMethod(Enum):
    """Supported augmentation methods."""
    NOISE = "noise"
    TIME_WARPING = "time_warping"
    SCALING = "scaling"
    ROTATION = "rotation"
    MIXUP = "mixup"

@dataclass
class PreprocessingConfig:
    """Configuration for time series preprocessing."""
    # Sequence parameters
    sequence_length: int = 60
    step_size: int = 1
    overlap: float = 0.0
    
    # Scaling parameters
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    fit_scaler_on_train: bool = True
    
    # Augmentation parameters
    augmentation_methods: List[AugmentationMethod] = field(default_factory=list)
    augmentation_factor: float = 1.0
    noise_std: float = 0.01
    scaling_range: Tuple[float, float] = (0.9, 1.1)
    
    # Validation parameters
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42
    
    # Additional parameters
    remove_outliers: bool = False
    outlier_threshold: float = 3.0
    fill_missing: bool = True
    fill_method: str = "forward"

class TimeSeriesPreprocessor:
    """
    Comprehensive time series preprocessing pipeline.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.scaler = None
        self.is_fitted = False
        
        # Set random seeds
        np.random.seed(config.random_state)
        random.seed(config.random_state)
        
        logger.info("Time series preprocessor initialized",
                   sequence_length=config.sequence_length,
                   scaling_method=config.scaling_method.value,
                   augmentation_methods=[m.value for m in config.augmentation_methods])
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences using sliding window approach.
        
        Args:
            data: Input time series data (n_samples, n_features)
            targets: Target values (n_samples,)
            
        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        
        # Calculate step size based on overlap
        if self.config.overlap > 0:
            step = max(1, int(self.config.sequence_length * (1 - self.config.overlap)))
        else:
            step = self.config.step_size
        
        for i in range(0, len(data) - self.config.sequence_length + 1, step):
            X.append(data[i:(i + self.config.sequence_length)])
            if targets is not None:
                y.append(targets[i + self.config.sequence_length - 1])
        
        X = np.array(X)
        y = np.array(y) if targets is not None else None
        
        logger.info("Sequences created",
                   n_sequences=len(X),
                   sequence_shape=X.shape,
                   step_size=step)
        
        return X, y
    
    def remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Remove outliers using z-score method.
        
        Args:
            data: Input data
            
        Returns:
            Data with outliers removed
        """
        if not self.config.remove_outliers:
            return data
        
        # Calculate z-scores
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        
        # Create mask for non-outliers
        mask = np.all(z_scores < self.config.outlier_threshold, axis=1)
        
        cleaned_data = data[mask]
        
        logger.info("Outliers removed",
                   original_samples=len(data),
                   cleaned_samples=len(cleaned_data),
                   removed_samples=len(data) - len(cleaned_data))
        
        return cleaned_data
    
    def fill_missing_values(self, data: np.ndarray) -> np.ndarray:
        """
        Fill missing values in the data.
        
        Args:
            data: Input data
            
        Returns:
            Data with missing values filled
        """
        if not self.config.fill_missing:
            return data
        
        df = pd.DataFrame(data)
        
        if self.config.fill_method == "forward":
            df = df.fillna(method='ffill')
        elif self.config.fill_method == "backward":
            df = df.fillna(method='bfill')
        elif self.config.fill_method == "interpolate":
            df = df.interpolate()
        elif self.config.fill_method == "mean":
            df = df.fillna(df.mean())
        else:
            df = df.fillna(0)
        
        # Fill any remaining NaNs with 0
        df = df.fillna(0)
        
        return df.values
    
    def fit_scaler(self, data: np.ndarray):
        """
        Fit the scaler on training data.
        
        Args:
            data: Training data
        """
        if self.config.scaling_method == ScalingMethod.STANDARD:
            self.scaler = StandardScaler()
        elif self.config.scaling_method == ScalingMethod.MINMAX:
            self.scaler = MinMaxScaler()
        elif self.config.scaling_method == ScalingMethod.ROBUST:
            self.scaler = RobustScaler()
        else:
            self.scaler = None
            return
        
        # Reshape data for fitting (flatten sequences)
        if len(data.shape) == 3:
            data_reshaped = data.reshape(-1, data.shape[-1])
        else:
            data_reshaped = data
        
        self.scaler.fit(data_reshaped)
        self.is_fitted = True
        
        logger.info("Scaler fitted",
                   scaling_method=self.config.scaling_method.value,
                   data_shape=data.shape)
    
    def scale_data(self, data: np.ndarray) -> np.ndarray:
        """
        Scale the data using fitted scaler.
        
        Args:
            data: Input data
            
        Returns:
            Scaled data
        """
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        
        # Reshape for scaling
        if len(data.shape) == 3:
            data_reshaped = data.reshape(-1, data.shape[-1])
        else:
            data_reshaped = data
        
        # Scale data
        scaled_data = self.scaler.transform(data_reshaped)
        
        # Reshape back
        scaled_data = scaled_data.reshape(original_shape)
        
        return scaled_data
    
    def augment_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augment the data using various methods.
        
        Args:
            X: Input sequences
            y: Target values
            
        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        if not self.config.augmentation_methods:
            return X, y
        
        augmented_X = [X]
        augmented_y = [y] if y is not None else [None]
        
        n_augmentations = int(len(X) * self.config.augmentation_factor)
        
        for method in self.config.augmentation_methods:
            if method == AugmentationMethod.NOISE:
                aug_X = self._add_noise(X, n_augmentations)
            elif method == AugmentationMethod.TIME_WARPING:
                aug_X = self._time_warping(X, n_augmentations)
            elif method == AugmentationMethod.SCALING:
                aug_X = self._scaling_augmentation(X, n_augmentations)
            elif method == AugmentationMethod.ROTATION:
                aug_X = self._rotation_augmentation(X, n_augmentations)
            elif method == AugmentationMethod.MIXUP:
                aug_X, aug_y = self._mixup_augmentation(X, y, n_augmentations)
                augmented_X.append(aug_X)
                if aug_y is not None:
                    augmented_y.append(aug_y)
                continue
            else:
                continue
            
            augmented_X.append(aug_X)
            if y is not None:
                augmented_y.append(y[:len(aug_X)])
        
        # Combine all augmentations
        final_X = np.concatenate(augmented_X, axis=0)
        final_y = np.concatenate([y for y in augmented_y if y is not None], axis=0) if y is not None else None
        
        logger.info("Data augmentation completed",
                   original_samples=len(X),
                   augmented_samples=len(final_X),
                   methods=[m.value for m in self.config.augmentation_methods])
        
        return final_X, final_y
    
    def _add_noise(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Add Gaussian noise to sequences."""
        indices = np.random.choice(len(X), n_samples, replace=True)
        augmented = X[indices].copy()
        
        noise = np.random.normal(0, self.config.noise_std, augmented.shape)
        augmented += noise
        
        return augmented
    
    def _time_warping(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Apply time warping to sequences."""
        indices = np.random.choice(len(X), n_samples, replace=True)
        augmented = X[indices].copy()
        
        for i in range(len(augmented)):
            # Random time warping
            warp_factor = np.random.uniform(0.8, 1.2)
            seq_len = len(augmented[i])
            new_len = int(seq_len * warp_factor)
            
            if new_len != seq_len:
                # Resample the sequence
                indices_old = np.linspace(0, seq_len - 1, seq_len)
                indices_new = np.linspace(0, seq_len - 1, new_len)
                
                for j in range(augmented[i].shape[1]):
                    augmented[i, :, j] = np.interp(indices_old, indices_new, 
                                                 augmented[i, :new_len, j])
        
        return augmented
    
    def _scaling_augmentation(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Apply scaling augmentation to sequences."""
        indices = np.random.choice(len(X), n_samples, replace=True)
        augmented = X[indices].copy()
        
        for i in range(len(augmented)):
            scale_factor = np.random.uniform(*self.config.scaling_range)
            augmented[i] *= scale_factor
        
        return augmented
    
    def _rotation_augmentation(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Apply rotation augmentation to sequences."""
        indices = np.random.choice(len(X), n_samples, replace=True)
        augmented = X[indices].copy()
        
        for i in range(len(augmented)):
            # Random rotation angle
            angle = np.random.uniform(-np.pi/6, np.pi/6)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Apply rotation to each time step
            for t in range(len(augmented[i])):
                if augmented[i].shape[1] >= 2:
                    x, y = augmented[i, t, 0], augmented[i, t, 1]
                    augmented[i, t, 0] = x * cos_a - y * sin_a
                    augmented[i, t, 1] = x * sin_a + y * cos_a
        
        return augmented
    
    def _mixup_augmentation(self, X: np.ndarray, y: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply mixup augmentation."""
        indices1 = np.random.choice(len(X), n_samples, replace=True)
        indices2 = np.random.choice(len(X), n_samples, replace=True)
        
        alpha = np.random.beta(0.2, 0.2, n_samples)
        alpha = alpha.reshape(-1, 1, 1)
        
        augmented_X = alpha * X[indices1] + (1 - alpha) * X[indices2]
        
        if y is not None:
            augmented_y = alpha.flatten() * y[indices1] + (1 - alpha.flatten()) * y[indices2]
        else:
            augmented_y = None
        
        return augmented_X, augmented_y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input sequences
            y: Target values
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        
        # Calculate split indices
        test_size = int(n_samples * self.config.test_split)
        val_size = int(n_samples * self.config.validation_split)
        train_size = n_samples - test_size - val_size
        
        # Split data
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size] if y is not None else None
        y_val = y[train_size:train_size + val_size] if y is not None else None
        y_test = y[train_size + val_size:] if y is not None else None
        
        logger.info("Data split completed",
                   train_samples=len(X_train),
                   val_samples=len(X_val),
                   test_samples=len(X_test))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess(self, data: np.ndarray, targets: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Input time series data
            targets: Target values
            
        Returns:
            Dictionary containing preprocessed data splits
        """
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Remove outliers
        data = self.remove_outliers(data)
        
        # Step 2: Fill missing values
        data = self.fill_missing_values(data)
        
        # Step 3: Create sequences
        X, y = self.create_sequences(data, targets)
        
        # Step 4: Fit scaler on training data
        if self.config.fit_scaler_on_train:
            # Split data first for scaler fitting
            X_temp, X_val_temp, X_test_temp, y_temp, y_val_temp, y_test_temp = self.split_data(X, y)
            self.fit_scaler(X_temp)
        else:
            self.fit_scaler(X)
        
        # Step 5: Scale data
        X = self.scale_data(X)
        
        # Step 6: Augment data
        X, y = self.augment_data(X, y)
        
        # Step 7: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 8: Scale validation and test data if not already scaled
        if not self.config.fit_scaler_on_train:
            X_val = self.scale_data(X_val)
            X_test = self.scale_data(X_test)
        
        result = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        logger.info("Preprocessing pipeline completed",
                   train_shape=X_train.shape,
                   val_shape=X_val.shape,
                   test_shape=X_test.shape)
        
        return result
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing pipeline."""
        return {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'augmentation_methods': [m.value for m in self.config.augmentation_methods]
        }

# Convenience functions
def create_preprocessor(sequence_length: int = 60,
                       scaling_method: Union[ScalingMethod, str] = ScalingMethod.STANDARD,
                       augmentation_methods: List[Union[AugmentationMethod, str]] = None,
                       **kwargs) -> TimeSeriesPreprocessor:
    """Create a preprocessor with default configuration."""
    if augmentation_methods is None:
        augmentation_methods = []
    
    # Convert string to enum if needed
    if isinstance(scaling_method, str):
        scaling_method = ScalingMethod(scaling_method)
    
    # Convert string augmentation methods to enums
    processed_augmentation_methods = []
    for method in augmentation_methods:
        if isinstance(method, str):
            processed_augmentation_methods.append(AugmentationMethod(method))
        else:
            processed_augmentation_methods.append(method)
    
    config = PreprocessingConfig(
        sequence_length=sequence_length,
        scaling_method=scaling_method,
        augmentation_methods=processed_augmentation_methods,
        **kwargs
    )
    return TimeSeriesPreprocessor(config)

def create_advanced_preprocessor(sequence_length: int = 60,
                               augmentation_factor: float = 0.5,
                               **kwargs) -> TimeSeriesPreprocessor:
    """Create a preprocessor with advanced augmentation."""
    augmentation_methods = [
        AugmentationMethod.NOISE,
        AugmentationMethod.SCALING,
        AugmentationMethod.TIME_WARPING
    ]
    
    return create_preprocessor(
        sequence_length=sequence_length,
        augmentation_methods=augmentation_methods,
        augmentation_factor=augmentation_factor,
        **kwargs
    )