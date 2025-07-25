#!/usr/bin/env python3
"""
Ensemble Model Frameworks Module

Implements comprehensive ensemble model frameworks for combining multiple ML models and strategies:
- Stacking ensemble with meta-learning
- Blending ensemble with weighted combinations
- Dynamic ensemble selection
- Ensemble diversity measures
- Ensemble performance evaluation
- Adaptive ensemble methods

Features:
- Stacking with meta-learning algorithms
- Blending with dynamic weight optimization
- Dynamic ensemble selection based on performance
- Diversity measurement and optimization
- Comprehensive performance evaluation
- Adaptive ensemble methods for changing market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import pickle

logger = structlog.get_logger()

class EnsembleMethod(Enum):
    """Available ensemble methods."""
    STACKING = "stacking"
    BLENDING = "blending"
    VOTING = "voting"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"

class DiversityMetric(Enum):
    """Diversity metrics for ensemble evaluation."""
    Q_STATISTIC = "q_statistic"
    DISAGREEMENT = "disagreement"
    DOUBLE_FAULT = "double_fault"
    CORRELATION = "correlation"
    ENTROPY = "entropy"

@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    method: EnsembleMethod
    base_models: List[BaseEstimator]
    meta_model: Optional[BaseEstimator] = None
    cv_folds: int = 5
    diversity_metric: DiversityMetric = DiversityMetric.Q_STATISTIC
    adaptive_threshold: float = 0.1
    dynamic_window: int = 100
    enable_diversity_optimization: bool = True

@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    model_weights: Optional[np.ndarray] = None
    diversity_score: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiversityMetrics:
    """Diversity metrics for ensemble."""
    q_statistic: float = 0.0
    disagreement: float = 0.0
    double_fault: float = 0.0
    correlation: float = 0.0
    entropy: float = 0.0
    overall_diversity: float = 0.0

class BaseEnsemble:
    """Base class for ensemble methods."""
    
    def __init__(self, config: EnsembleConfig):
        """
        Initialize base ensemble.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.base_models = config.base_models
        self.meta_model = config.meta_model
        self.is_fitted = False
        self.diversity_metrics = DiversityMetrics()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEnsemble':
        """
        Fit the ensemble model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def calculate_diversity(self, X: np.ndarray, y: np.ndarray) -> DiversityMetrics:
        """
        Calculate diversity metrics for the ensemble.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Diversity metrics
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before calculating diversity")
        
        # Get predictions from all base models
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        base_predictions = np.array(base_predictions)
        
        # Calculate diversity metrics
        metrics = DiversityMetrics()
        
        # Q-statistic
        metrics.q_statistic = self._calculate_q_statistic(base_predictions, y)
        
        # Disagreement measure
        metrics.disagreement = self._calculate_disagreement(base_predictions)
        
        # Double fault measure
        metrics.double_fault = self._calculate_double_fault(base_predictions, y)
        
        # Correlation
        metrics.correlation = self._calculate_correlation(base_predictions)
        
        # Entropy
        metrics.entropy = self._calculate_entropy(base_predictions)
        
        # Overall diversity (weighted average)
        metrics.overall_diversity = (
            metrics.q_statistic + metrics.disagreement + 
            metrics.double_fault + metrics.correlation + metrics.entropy
        ) / 5.0
        
        self.diversity_metrics = metrics
        return metrics
    
    def _calculate_q_statistic(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Q-statistic diversity measure."""
        n_models = predictions.shape[0]
        n_samples = predictions.shape[1]
        
        q_total = 0
        count = 0
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Calculate Q-statistic for pair of models
                n11 = np.sum((predictions[i] == y_true) & (predictions[j] == y_true))
                n10 = np.sum((predictions[i] == y_true) & (predictions[j] != y_true))
                n01 = np.sum((predictions[i] != y_true) & (predictions[j] == y_true))
                n00 = np.sum((predictions[i] != y_true) & (predictions[j] != y_true))
                
                if n11 * n00 + n01 * n10 != 0:
                    q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
                    q_total += q
                    count += 1
        
        return q_total / count if count > 0 else 0.0
    
    def _calculate_disagreement(self, predictions: np.ndarray) -> float:
        """Calculate disagreement diversity measure."""
        n_models = predictions.shape[0]
        n_samples = predictions.shape[1]
        
        disagreement_total = 0
        count = 0
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.sum(predictions[i] != predictions[j]) / n_samples
                disagreement_total += disagreement
                count += 1
        
        return disagreement_total / count if count > 0 else 0.0
    
    def _calculate_double_fault(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate double fault diversity measure."""
        n_models = predictions.shape[0]
        n_samples = predictions.shape[1]
        
        double_fault_total = 0
        count = 0
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                double_fault = np.sum((predictions[i] != y_true) & (predictions[j] != y_true)) / n_samples
                double_fault_total += double_fault
                count += 1
        
        return double_fault_total / count if count > 0 else 0.0
    
    def _calculate_correlation(self, predictions: np.ndarray) -> float:
        """Calculate correlation diversity measure."""
        n_models = predictions.shape[0]
        
        correlation_total = 0
        count = 0
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                correlation = np.corrcoef(predictions[i], predictions[j])[0, 1]
                if not np.isnan(correlation):
                    correlation_total += correlation
                    count += 1
        
        return correlation_total / count if count > 0 else 0.0
    
    def _calculate_entropy(self, predictions: np.ndarray) -> float:
        """Calculate entropy diversity measure."""
        n_models = predictions.shape[0]
        n_samples = predictions.shape[1]
        
        entropy_total = 0
        
        for sample_idx in range(n_samples):
            # Count predictions for each class
            unique, counts = np.unique(predictions[:, sample_idx], return_counts=True)
            probabilities = counts / n_models
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropy_total += entropy
        
        return entropy_total / n_samples

class StackingEnsemble(BaseEnsemble):
    """Stacking ensemble with meta-learning."""
    
    def __init__(self, config: EnsembleConfig):
        """
        Initialize stacking ensemble.
        
        Args:
            config: Ensemble configuration
        """
        super().__init__(config)
        if self.meta_model is None:
            self.meta_model = LogisticRegression() if self._is_classification() else LinearRegression()
    
    def _is_classification(self) -> bool:
        """Check if this is a classification problem."""
        return hasattr(self.base_models[0], 'predict_proba')
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting stacking ensemble", n_models=len(self.base_models))
        
        # Fit base models
        for i, model in enumerate(self.base_models):
            logger.debug("Fitting base model", model_index=i, model_type=type(model).__name__)
            model.fit(X, y)
        
        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Fit meta-model
        logger.debug("Fitting meta-model", meta_model_type=type(self.meta_model).__name__)
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        logger.info("Stacking ensemble fitted successfully")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking ensemble.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Generate meta-features for prediction
        meta_features = self._generate_meta_features_prediction(X)
        
        # Make prediction using meta-model
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (for classification).
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if not self._is_classification():
            raise ValueError("predict_proba only available for classification")
        
        meta_features = self._generate_meta_features_prediction(X)
        return self.meta_model.predict_proba(meta_features)
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate meta-features using cross-validation."""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        meta_features = np.zeros((n_samples, n_models))
        
        # Use stratified k-fold for classification, regular k-fold for regression
        if self._is_classification():
            kf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for i, model in enumerate(self.base_models):
                # Fit model on training fold
                model_copy = joblib.load(joblib.dump(model)[1])  # Deep copy
                model_copy.fit(X_train, y_train)
                
                # Generate predictions for validation fold
                if self._is_classification():
                    meta_features[val_idx, i] = model_copy.predict_proba(X_val)[:, 1]
                else:
                    meta_features[val_idx, i] = model_copy.predict(X_val)
        
        return meta_features
    
    def _generate_meta_features_prediction(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features for prediction."""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        meta_features = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.base_models):
            if self._is_classification():
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model.predict(X)
        
        return meta_features

class BlendingEnsemble(BaseEnsemble):
    """Blending ensemble with weighted combinations."""
    
    def __init__(self, config: EnsembleConfig):
        """
        Initialize blending ensemble.
        
        Args:
            config: Ensemble configuration
        """
        super().__init__(config)
        self.weights = None
        if self.meta_model is None:
            self.meta_model = LogisticRegression() if self._is_classification() else LinearRegression()
    
    def _is_classification(self) -> bool:
        """Check if this is a classification problem."""
        return hasattr(self.base_models[0], 'predict_proba')
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BlendingEnsemble':
        """
        Fit the blending ensemble.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting blending ensemble", n_models=len(self.base_models))
        
        # Split data for blending
        split_idx = int(0.7 * len(X))
        X_train, X_blend = X[:split_idx], X[split_idx:]
        y_train, y_blend = y[:split_idx], y[split_idx:]
        
        # Fit base models on training data
        for i, model in enumerate(self.base_models):
            logger.debug("Fitting base model", model_index=i, model_type=type(model).__name__)
            model.fit(X_train, y_train)
        
        # Generate blending features
        blend_features = self._generate_blend_features(X_blend)
        
        # Fit meta-model on blending data
        logger.debug("Fitting meta-model", meta_model_type=type(self.meta_model).__name__)
        self.meta_model.fit(blend_features, y_blend)
        
        # Calculate optimal weights
        self.weights = self._calculate_optimal_weights(blend_features, y_blend)
        
        self.is_fitted = True
        logger.info("Blending ensemble fitted successfully")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using blending ensemble.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        blend_features = self._generate_blend_features(X)
        return self.meta_model.predict(blend_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (for classification).
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if not self._is_classification():
            raise ValueError("predict_proba only available for classification")
        
        blend_features = self._generate_blend_features(X)
        return self.meta_model.predict_proba(blend_features)
    
    def _generate_blend_features(self, X: np.ndarray) -> np.ndarray:
        """Generate blending features from base models."""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        blend_features = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.base_models):
            if self._is_classification():
                blend_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                blend_features[:, i] = model.predict(X)
        
        return blend_features
    
    def _calculate_optimal_weights(self, blend_features: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate optimal weights for base models."""
        from scipy.optimize import minimize
        
        def objective(weights):
            weighted_pred = np.sum(blend_features * weights.reshape(1, -1), axis=1)
            return mean_squared_error(y, weighted_pred)
        
        # Constraint: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(len(self.base_models))]
        
        # Initial weights (equal)
        initial_weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        # Optimize
        result = minimize(objective, initial_weights, constraints=constraints, bounds=bounds)
        
        return result.x

class DynamicEnsemble(BaseEnsemble):
    """Dynamic ensemble selection based on performance."""
    
    def __init__(self, config: EnsembleConfig):
        """
        Initialize dynamic ensemble.
        
        Args:
            config: Ensemble configuration
        """
        super().__init__(config)
        self.performance_history = []
        self.model_performances = {}
        self.selection_threshold = config.adaptive_threshold
        self.window_size = config.dynamic_window
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DynamicEnsemble':
        """
        Fit the dynamic ensemble.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting dynamic ensemble", n_models=len(self.base_models))
        
        # Fit all base models
        for i, model in enumerate(self.base_models):
            logger.debug("Fitting base model", model_index=i, model_type=type(model).__name__)
            model.fit(X, y)
        
        # Initialize performance tracking
        for i in range(len(self.base_models)):
            self.model_performances[i] = []
        
        self.is_fitted = True
        logger.info("Dynamic ensemble fitted successfully")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using dynamic ensemble selection.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Select best models for this sample
            selected_models = self._select_models(X[i:i+1])
            
            # Get predictions from selected models
            model_predictions = []
            for model_idx in selected_models:
                model = self.base_models[model_idx]
                pred = model.predict(X[i:i+1])[0]
                model_predictions.append(pred)
            
            # Combine predictions (simple average for now)
            predictions[i] = np.mean(model_predictions)
        
        return predictions
    
    def update_performance(self, X: np.ndarray, y: np.ndarray):
        """
        Update model performance history.
        
        Args:
            X: Features
            y: True targets
        """
        if not self.is_fitted:
            return
        
        # Get predictions from all models
        for i, model in enumerate(self.base_models):
            pred = model.predict(X)
            performance = self._calculate_performance(pred, y)
            self.model_performances[i].append(performance)
            
            # Keep only recent performance history
            if len(self.model_performances[i]) > self.window_size:
                self.model_performances[i] = self.model_performances[i][-self.window_size:]
    
    def _select_models(self, X: np.ndarray) -> List[int]:
        """Select best models for given input."""
        if not self.model_performances:
            # If no performance history, use all models
            return list(range(len(self.base_models)))
        
        # Calculate recent average performance for each model
        recent_performances = {}
        for model_idx, performances in self.model_performances.items():
            if performances:
                recent_performances[model_idx] = np.mean(performances[-10:])  # Last 10 performances
            else:
                recent_performances[model_idx] = 0.0
        
        # Select models above threshold
        max_performance = max(recent_performances.values())
        threshold = max_performance - self.selection_threshold
        
        selected_models = [
            model_idx for model_idx, performance in recent_performances.items()
            if performance >= threshold
        ]
        
        # Ensure at least one model is selected
        if not selected_models:
            selected_models = [max(recent_performances, key=recent_performances.get)]
        
        return selected_models
    
    def _calculate_performance(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate performance metric."""
        # Use appropriate metric based on problem type
        if len(np.unique(y_true)) <= 10:  # Classification
            return accuracy_score(y_true, predictions)
        else:  # Regression
            return -mean_squared_error(y_true, predictions)  # Negative because higher is better

class AdaptiveEnsemble(BaseEnsemble):
    """Adaptive ensemble methods for changing market conditions."""
    
    def __init__(self, config: EnsembleConfig):
        """
        Initialize adaptive ensemble.
        
        Args:
            config: Ensemble configuration
        """
        super().__init__(config)
        self.market_regime = 'normal'
        self.regime_models = {}
        self.regime_performances = {}
        self.adaptation_threshold = config.adaptive_threshold
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveEnsemble':
        """
        Fit the adaptive ensemble.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting adaptive ensemble", n_models=len(self.base_models))
        
        # Fit base models
        for i, model in enumerate(self.base_models):
            logger.debug("Fitting base model", model_index=i, model_type=type(model).__name__)
            model.fit(X, y)
        
        # Initialize regime-specific models
        self._initialize_regime_models(X, y)
        
        self.is_fitted = True
        logger.info("Adaptive ensemble fitted successfully")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using adaptive ensemble.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Detect current market regime
        current_regime = self._detect_market_regime(X)
        
        # Use regime-specific ensemble
        if current_regime in self.regime_models:
            return self.regime_models[current_regime].predict(X)
        else:
            # Fallback to base ensemble
            return self._base_predict(X)
    
    def adapt_to_regime(self, X: np.ndarray, y: np.ndarray):
        """
        Adapt ensemble to current market regime.
        
        Args:
            X: Recent features
            y: Recent targets
        """
        if not self.is_fitted:
            return
        
        # Detect regime change
        new_regime = self._detect_market_regime(X)
        
        if new_regime != self.market_regime:
            logger.info("Market regime change detected", 
                       old_regime=self.market_regime, 
                       new_regime=new_regime)
            
            # Update regime-specific models
            self._update_regime_models(new_regime, X, y)
            self.market_regime = new_regime
    
    def _initialize_regime_models(self, X: np.ndarray, y: np.ndarray):
        """Initialize models for different market regimes."""
        # Define market regimes
        regimes = ['bull', 'bear', 'volatile', 'sideways']
        
        for regime in regimes:
            # Create a blending ensemble for each regime
            config = EnsembleConfig(
                method=EnsembleMethod.BLENDING,
                base_models=self.base_models.copy(),
                cv_folds=self.config.cv_folds
            )
            self.regime_models[regime] = BlendingEnsemble(config)
            self.regime_models[regime].fit(X, y)
            self.regime_performances[regime] = []
    
    def _detect_market_regime(self, X: np.ndarray) -> str:
        """Detect current market regime from features."""
        # Simple regime detection based on feature statistics
        # In production, you'd use more sophisticated methods
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Calculate basic statistics
        mean_val = np.mean(X)
        std_val = np.std(X)
        
        # Simple rule-based regime detection
        if std_val > 0.5:
            return 'volatile'
        elif mean_val > 0.1:
            return 'bull'
        elif mean_val < -0.1:
            return 'bear'
        else:
            return 'sideways'
    
    def _update_regime_models(self, regime: str, X: np.ndarray, y: np.ndarray):
        """Update models for specific regime."""
        if regime in self.regime_models:
            # Retrain regime-specific model with recent data
            self.regime_models[regime].fit(X, y)
    
    def _base_predict(self, X: np.ndarray) -> np.ndarray:
        """Base prediction using simple averaging."""
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)

def create_ensemble_framework(method: str = "stacking",
                            base_models: List[BaseEstimator] = None,
                            meta_model: BaseEstimator = None,
                            cv_folds: int = 5) -> BaseEnsemble:
    """
    Create an ensemble framework with specified configuration.
    
    Args:
        method: Ensemble method
        base_models: List of base models
        meta_model: Meta-model for ensemble
        cv_folds: Number of cross-validation folds
        
    Returns:
        Ensemble framework instance
    """
    if base_models is None:
        base_models = [
            RandomForestClassifier(n_estimators=100, random_state=42),
            RandomForestClassifier(n_estimators=200, random_state=42),
            RandomForestClassifier(n_estimators=300, random_state=42)
        ]
    
    config = EnsembleConfig(
        method=EnsembleMethod(method),
        base_models=base_models,
        meta_model=meta_model,
        cv_folds=cv_folds
    )
    
    if method == "stacking":
        return StackingEnsemble(config)
    elif method == "blending":
        return BlendingEnsemble(config)
    elif method == "dynamic":
        return DynamicEnsemble(config)
    elif method == "adaptive":
        return AdaptiveEnsemble(config)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

def evaluate_ensemble_performance(ensemble: BaseEnsemble, 
                                X_test: np.ndarray, 
                                y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate ensemble performance.
    
    Args:
        ensemble: Ensemble to evaluate
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Performance metrics
    """
    # Make predictions
    y_pred = ensemble.predict(X_test)
    
    # Calculate metrics
    metrics = {}
    
    if len(np.unique(y_test)) <= 10:  # Classification
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
    else:  # Regression
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
    
    # Calculate diversity
    diversity_metrics = ensemble.calculate_diversity(X_test, y_test)
    metrics['diversity'] = diversity_metrics.overall_diversity
    
    return metrics

if __name__ == "__main__":
    # Demo of ensemble frameworks
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different ensemble methods
    methods = ['stacking', 'blending', 'dynamic', 'adaptive']
    
    for method in methods:
        print(f"\nTesting {method.upper()} ensemble:")
        
        # Create ensemble
        ensemble = create_ensemble_framework(method=method)
        
        # Fit and evaluate
        ensemble.fit(X_train, y_train)
        metrics = evaluate_ensemble_performance(ensemble, X_test, y_test)
        
        print(f"Performance metrics: {metrics}")