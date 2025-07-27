"""
Feature Combination Generators

Implements comprehensive feature combination and generation techniques:
- Mathematical combinations (addition, multiplication, division, etc.)
- Statistical combinations (rolling statistics, ratios, differences)
- Technical indicator combinations
- Polynomial features
- Interaction features
- Custom feature generators

Features:
- Multiple combination strategies
- Automatic feature generation
- Feature selection and filtering
- Memory-efficient processing
- Validation and quality assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import structlog
from itertools import combinations, product
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from scipy import stats

logger = structlog.get_logger()

class CombinationType(Enum):
    """Feature combination types."""
    MATHEMATICAL = "mathematical"
    STATISTICAL = "statistical"
    TECHNICAL = "technical"
    POLYNOMIAL = "polynomial"
    INTERACTION = "interaction"
    CUSTOM = "custom"

class MathematicalOperation(Enum):
    """Mathematical operations for feature combination."""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    LOG = "log"
    SQRT = "sqrt"
    ABS = "abs"

@dataclass
class CombinationConfig:
    """Configuration for feature combination."""
    combination_type: CombinationType = CombinationType.MATHEMATICAL
    max_features: int = 100
    min_correlation: float = 0.1
    max_correlation: float = 0.95
    polynomial_degree: int = 2
    interaction_only: bool = False
    include_bias: bool = False
    random_state: int = 42

@dataclass
class CombinationResult:
    """Result of feature combination operation."""
    combined_features: np.ndarray
    feature_names: List[str]
    combination_info: Dict[str, Any]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]

class FeatureCombinationGenerator:
    """
    Comprehensive feature combination generator.
    """
    
    def __init__(self, config: CombinationConfig = None):
        """
        Initialize feature combination generator.
        
        Args:
            config: Combination configuration
        """
        self.config = config or CombinationConfig()
        self.feature_names = []
        self.combination_info = {}
        self.is_fitted = False
    
    def generate_combinations(self, data: np.ndarray, feature_names: List[str] = None) -> CombinationResult:
        """
        Generate feature combinations.
        
        Args:
            data: Input feature data
            feature_names: Names of input features
            
        Returns:
            CombinationResult with combined features
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]
        
        if len(self.feature_names) != data.shape[1]:
            raise ValueError("Number of feature names must match number of features")
        
        # Generate combinations based on type
        if self.config.combination_type == CombinationType.MATHEMATICAL:
            combined_data, combined_names = self._generate_mathematical_combinations(data)
        elif self.config.combination_type == CombinationType.STATISTICAL:
            combined_data, combined_names = self._generate_statistical_combinations(data)
        elif self.config.combination_type == CombinationType.TECHNICAL:
            combined_data, combined_names = self._generate_technical_combinations(data)
        elif self.config.combination_type == CombinationType.POLYNOMIAL:
            combined_data, combined_names = self._generate_polynomial_combinations(data)
        elif self.config.combination_type == CombinationType.INTERACTION:
            combined_data, combined_names = self._generate_interaction_combinations(data)
        else:
            raise ValueError(f"Unsupported combination type: {self.config.combination_type}")
        
        # Filter and select best features
        filtered_data, filtered_names = self._filter_combinations(combined_data, combined_names, data)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(filtered_data, data)
        
        # Store combination information
        self.combination_info = {
            "original_features": len(self.feature_names),
            "generated_combinations": len(combined_names),
            "selected_combinations": len(filtered_names),
            "combination_type": self.config.combination_type.value
        }
        
        self.is_fitted = True
        
        metadata = {
            "combination_type": self.config.combination_type.value,
            "original_shape": data.shape,
            "combined_shape": filtered_data.shape,
            "is_fitted": self.is_fitted
        }
        
        return CombinationResult(
            combined_features=filtered_data,
            feature_names=filtered_names,
            combination_info=self.combination_info,
            quality_metrics=quality_metrics,
            metadata=metadata
        )
    
    def _generate_mathematical_combinations(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Generate mathematical combinations of features."""
        combinations_list = []
        names_list = []
        
        # Generate all pairs of features
        for i, j in combinations(range(data.shape[1]), 2):
            feature1 = data[:, i]
            feature2 = data[:, j]
            name1 = self.feature_names[i]
            name2 = self.feature_names[j]
            
            # Addition
            add_feature = feature1 + feature2
            combinations_list.append(add_feature)
            names_list.append(f"{name1}_plus_{name2}")
            
            # Subtraction
            sub_feature = feature1 - feature2
            combinations_list.append(sub_feature)
            names_list.append(f"{name1}_minus_{name2}")
            
            # Multiplication
            mul_feature = feature1 * feature2
            combinations_list.append(mul_feature)
            names_list.append(f"{name1}_times_{name2}")
            
            # Division (with safety check)
            div_feature = np.where(feature2 != 0, feature1 / feature2, 0)
            combinations_list.append(div_feature)
            names_list.append(f"{name1}_div_{name2}")
            
            # Power
            pow_feature = np.power(feature1, feature2)
            combinations_list.append(pow_feature)
            names_list.append(f"{name1}_pow_{name2}")
        
        # Single feature transformations
        for i in range(data.shape[1]):
            feature = data[:, i]
            name = self.feature_names[i]
            
            # Log transformation
            log_feature = np.where(feature > 0, np.log(feature), 0)
            combinations_list.append(log_feature)
            names_list.append(f"log_{name}")
            
            # Square root
            sqrt_feature = np.where(feature >= 0, np.sqrt(feature), 0)
            combinations_list.append(sqrt_feature)
            names_list.append(f"sqrt_{name}")
            
            # Absolute value
            abs_feature = np.abs(feature)
            combinations_list.append(abs_feature)
            names_list.append(f"abs_{name}")
        
        return np.column_stack(combinations_list), names_list
    
    def _generate_statistical_combinations(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Generate statistical combinations of features."""
        combinations_list = []
        names_list = []
        
        # Rolling statistics
        window_sizes = [5, 10, 20]
        
        for i in range(data.shape[1]):
            feature = data[:, i]
            name = self.feature_names[i]
            
            for window in window_sizes:
                if len(feature) >= window:
                    # Rolling mean
                    rolling_mean = pd.Series(feature).rolling(window=window, min_periods=1).mean().values
                    combinations_list.append(rolling_mean)
                    names_list.append(f"{name}_rolling_mean_{window}")
                    
                    # Rolling std
                    rolling_std = pd.Series(feature).rolling(window=window, min_periods=1).std().values
                    combinations_list.append(rolling_std)
                    names_list.append(f"{name}_rolling_std_{window}")
                    
                    # Rolling median
                    rolling_median = pd.Series(feature).rolling(window=window, min_periods=1).median().values
                    combinations_list.append(rolling_median)
                    names_list.append(f"{name}_rolling_median_{window}")
        
        # Feature ratios and differences
        for i, j in combinations(range(data.shape[1]), 2):
            feature1 = data[:, i]
            feature2 = data[:, j]
            name1 = self.feature_names[i]
            name2 = self.feature_names[j]
            
            # Ratio
            ratio = np.where(feature2 != 0, feature1 / feature2, 0)
            combinations_list.append(ratio)
            names_list.append(f"{name1}_ratio_{name2}")
            
            # Difference
            diff = feature1 - feature2
            combinations_list.append(diff)
            names_list.append(f"{name1}_diff_{name2}")
            
            # Z-score difference
            z1 = (feature1 - np.mean(feature1)) / np.std(feature1) if np.std(feature1) > 0 else 0
            z2 = (feature2 - np.mean(feature2)) / np.std(feature2) if np.std(feature2) > 0 else 0
            z_diff = z1 - z2
            combinations_list.append(z_diff)
            names_list.append(f"{name1}_z_diff_{name2}")
        
        return np.column_stack(combinations_list), names_list
    
    def _generate_technical_combinations(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Generate technical indicator combinations."""
        combinations_list = []
        names_list = []
        
        # Momentum indicators
        for i in range(data.shape[1]):
            feature = data[:, i]
            name = self.feature_names[i]
            
            # Rate of change
            roc = np.zeros_like(feature)
            roc[1:] = (feature[1:] - feature[:-1]) / feature[:-1]
            combinations_list.append(roc)
            names_list.append(f"{name}_roc")
            
            # Momentum
            momentum = np.zeros_like(feature)
            momentum[4:] = feature[4:] - feature[:-4]
            combinations_list.append(momentum)
            names_list.append(f"{name}_momentum")
            
            # Acceleration
            acceleration = np.zeros_like(feature)
            acceleration[2:] = feature[2:] - 2 * feature[1:-1] + feature[:-2]
            combinations_list.append(acceleration)
            names_list.append(f"{name}_acceleration")
        
        # Volatility measures
        for i in range(data.shape[1]):
            feature = data[:, i]
            name = self.feature_names[i]
            
            # Rolling volatility
            returns = np.diff(feature) / feature[:-1]
            volatility = pd.Series(returns).rolling(window=20, min_periods=1).std().values
            volatility = np.concatenate([[0], volatility])  # Pad to match original length
            combinations_list.append(volatility)
            names_list.append(f"{name}_volatility")
        
        # Cross-feature technical combinations
        for i, j in combinations(range(data.shape[1]), 2):
            feature1 = data[:, i]
            feature2 = data[:, j]
            name1 = self.feature_names[i]
            name2 = self.feature_names[j]
            
            # Relative strength
            rel_strength = feature1 / feature2
            combinations_list.append(rel_strength)
            names_list.append(f"{name1}_rel_strength_{name2}")
            
            # Spread
            spread = feature1 - feature2
            combinations_list.append(spread)
            names_list.append(f"{name1}_spread_{name2}")
        
        return np.column_stack(combinations_list), names_list
    
    def _generate_polynomial_combinations(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Generate polynomial feature combinations."""
        poly = PolynomialFeatures(
            degree=self.config.polynomial_degree,
            interaction_only=self.config.interaction_only,
            include_bias=self.config.include_bias
        )
        
        # Fit and transform
        poly_features = poly.fit_transform(data)
        
        # Generate feature names
        feature_names = []
        if self.config.include_bias:
            feature_names.append("bias")
        
        # Original features
        feature_names.extend(self.feature_names)
        
        # Interaction features
        if self.config.polynomial_degree >= 2:
            for i, j in combinations(range(len(self.feature_names)), 2):
                feature_names.append(f"{self.feature_names[i]}_{self.feature_names[j]}")
        
        # Higher degree features
        if self.config.polynomial_degree >= 3:
            for i, j, k in combinations(range(len(self.feature_names)), 3):
                feature_names.append(f"{self.feature_names[i]}_{self.feature_names[j]}_{self.feature_names[k]}")
        
        return poly_features, feature_names
    
    def _generate_interaction_combinations(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Generate interaction feature combinations."""
        combinations_list = []
        names_list = []
        
        # Two-way interactions
        for i, j in combinations(range(data.shape[1]), 2):
            feature1 = data[:, i]
            feature2 = data[:, j]
            name1 = self.feature_names[i]
            name2 = self.feature_names[j]
            
            # Product interaction
            interaction = feature1 * feature2
            combinations_list.append(interaction)
            names_list.append(f"{name1}_x_{name2}")
            
            # Ratio interaction
            ratio_interaction = np.where(feature2 != 0, feature1 / feature2, 0)
            combinations_list.append(ratio_interaction)
            names_list.append(f"{name1}_div_{name2}")
            
            # Sum interaction
            sum_interaction = feature1 + feature2
            combinations_list.append(sum_interaction)
            names_list.append(f"{name1}_plus_{name2}")
        
        # Three-way interactions
        for i, j, k in combinations(range(data.shape[1]), 3):
            feature1 = data[:, i]
            feature2 = data[:, j]
            feature3 = data[:, k]
            name1 = self.feature_names[i]
            name2 = self.feature_names[j]
            name3 = self.feature_names[k]
            
            # Triple interaction
            triple_interaction = feature1 * feature2 * feature3
            combinations_list.append(triple_interaction)
            names_list.append(f"{name1}_x_{name2}_x_{name3}")
        
        return np.column_stack(combinations_list), names_list
    
    def _filter_combinations(self, combined_data: np.ndarray, combined_names: List[str], 
                           original_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Filter and select the best feature combinations."""
        if combined_data.shape[1] <= self.config.max_features:
            return combined_data, combined_names
        
        # Calculate feature importance scores
        importance_scores = self._calculate_feature_importance(combined_data, original_data)
        
        # Select top features
        top_indices = np.argsort(importance_scores)[-self.config.max_features:]
        
        filtered_data = combined_data[:, top_indices]
        filtered_names = [combined_names[i] for i in top_indices]
        
        return filtered_data, filtered_names
    
    def _calculate_feature_importance(self, combined_data: np.ndarray, original_data: np.ndarray) -> np.ndarray:
        """Calculate importance scores for combined features."""
        importance_scores = np.zeros(combined_data.shape[1])
        
        for i in range(combined_data.shape[1]):
            feature = combined_data[:, i]
            
            # Calculate multiple importance metrics
            metrics = []
            
            # 1. Variance
            variance = np.var(feature)
            metrics.append(variance)
            
            # 2. Correlation with original features
            correlations = []
            for j in range(original_data.shape[1]):
                corr = np.corrcoef(feature, original_data[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                metrics.append(avg_correlation)
            
            # 3. Skewness (prefer normal distribution)
            skewness = abs(stats.skew(feature))
            metrics.append(1 / (1 + skewness))
            
            # 4. Kurtosis (prefer normal distribution)
            kurtosis = abs(stats.kurtosis(feature))
            metrics.append(1 / (1 + kurtosis))
            
            # Combine metrics
            importance_scores[i] = np.mean(metrics)
        
        return importance_scores
    
    def _calculate_quality_metrics(self, combined_data: np.ndarray, original_data: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for combined features."""
        metrics = {}
        
        # Feature count
        metrics["total_features"] = combined_data.shape[1]
        metrics["original_features"] = original_data.shape[1]
        metrics["feature_expansion_ratio"] = combined_data.shape[1] / original_data.shape[1]
        
        # Variance metrics
        variances = np.var(combined_data, axis=0)
        metrics["mean_variance"] = np.mean(variances)
        metrics["variance_std"] = np.std(variances)
        
        # Correlation metrics
        correlations = []
        for i in range(combined_data.shape[1]):
            for j in range(i + 1, combined_data.shape[1]):
                corr = np.corrcoef(combined_data[:, i], combined_data[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            metrics["mean_correlation"] = np.mean(correlations)
            metrics["max_correlation"] = np.max(correlations)
            metrics["correlation_std"] = np.std(correlations)
        
        # Distribution metrics
        skewness = stats.skew(combined_data, axis=0)
        kurtosis = stats.kurtosis(combined_data, axis=0)
        
        metrics["mean_skewness"] = np.mean(np.abs(skewness))
        metrics["mean_kurtosis"] = np.mean(np.abs(kurtosis))
        
        return metrics

class CustomFeatureGenerator:
    """
    Custom feature generator for domain-specific features.
    """
    
    def __init__(self, feature_functions: Dict[str, Callable] = None):
        """
        Initialize custom feature generator.
        
        Args:
            feature_functions: Dictionary of custom feature functions
        """
        self.feature_functions = feature_functions or {}
        self.is_fitted = False
    
    def add_feature_function(self, name: str, function: Callable):
        """Add a custom feature function."""
        self.feature_functions[name] = function
    
    def generate_custom_features(self, data: np.ndarray, feature_names: List[str] = None) -> CombinationResult:
        """
        Generate custom features using provided functions.
        
        Args:
            data: Input feature data
            feature_names: Names of input features
            
        Returns:
            CombinationResult with custom features
        """
        if not self.feature_functions:
            raise ValueError("No custom feature functions defined")
        
        custom_features = []
        custom_names = []
        
        for name, func in self.feature_functions.items():
            try:
                feature = func(data, feature_names)
                if feature.ndim == 1:
                    feature = feature.reshape(-1, 1)
                
                custom_features.append(feature)
                custom_names.append(name)
                
            except Exception as e:
                logger.warning(f"Failed to generate custom feature {name}: {e}")
                continue
        
        if not custom_features:
            raise ValueError("No custom features were successfully generated")
        
        combined_data = np.column_stack(custom_features)
        
        metadata = {
            "custom_features": len(custom_names),
            "feature_functions": list(self.feature_functions.keys()),
            "is_custom": True
        }
        
        return CombinationResult(
            combined_features=combined_data,
            feature_names=custom_names,
            combination_info={"custom_features": len(custom_names)},
            quality_metrics={},
            metadata=metadata
        )

# Convenience functions
def generate_feature_combinations(data: np.ndarray, feature_names: List[str] = None,
                                combination_type: str = "mathematical",
                                max_features: int = 100) -> CombinationResult:
    """
    Convenience function for feature combination generation.
    
    Args:
        data: Input feature data
        feature_names: Names of input features
        combination_type: Type of combination
        max_features: Maximum number of features to generate
        
    Returns:
        CombinationResult
    """
    config = CombinationConfig(
        combination_type=CombinationType(combination_type),
        max_features=max_features
    )
    
    generator = FeatureCombinationGenerator(config)
    return generator.generate_combinations(data, feature_names)

def generate_polynomial_features(data: np.ndarray, degree: int = 2,
                               interaction_only: bool = False) -> CombinationResult:
    """
    Convenience function for polynomial feature generation.
    
    Args:
        data: Input feature data
        degree: Polynomial degree
        interaction_only: Whether to include only interaction terms
        
    Returns:
        CombinationResult
    """
    config = CombinationConfig(
        combination_type=CombinationType.POLYNOMIAL,
        polynomial_degree=degree,
        interaction_only=interaction_only
    )
    
    generator = FeatureCombinationGenerator(config)
    return generator.generate_combinations(data)

def generate_custom_features(data: np.ndarray, feature_functions: Dict[str, Callable],
                           feature_names: List[str] = None) -> CombinationResult:
    """
    Convenience function for custom feature generation.
    
    Args:
        data: Input feature data
        feature_functions: Dictionary of custom feature functions
        feature_names: Names of input features
        
    Returns:
        CombinationResult
    """
    generator = CustomFeatureGenerator(feature_functions)
    return generator.generate_custom_features(data, feature_names)