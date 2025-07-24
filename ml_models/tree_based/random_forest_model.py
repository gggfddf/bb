"""
Random Forest Model Implementation

A comprehensive Random Forest implementation for stock prediction with:
- Feature importance analysis
- Hyperparameter tuning
- Ensemble methods
- Out-of-bag validation
- Model evaluation integration

Features:
- Custom Random Forest wrapper with enhanced functionality
- Feature importance visualization and analysis
- Hyperparameter optimization with multiple strategies
- Out-of-bag validation and error estimation
- Integration with model evaluation framework
- Ensemble methods for improved predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = structlog.get_logger()

class ModelType(Enum):
    """Model types for Random Forest."""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"

@dataclass
class RandomForestConfig:
    """Configuration for Random Forest model."""
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[Union[str, int, float]] = 'sqrt'
    bootstrap: bool = True
    oob_score: bool = True
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0
    warm_start: bool = False
    max_samples: Optional[Union[int, float]] = None

@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    feature_names: List[str]
    importance_scores: np.ndarray
    importance_std: np.ndarray
    permutation_importance: Optional[np.ndarray] = None
    shap_values: Optional[np.ndarray] = None
    top_features: List[str] = field(default_factory=list)
    importance_ranking: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

@dataclass
class HyperparameterTuningResult:
    """Container for hyperparameter tuning results."""
    best_params: Dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    tuning_history: List[Dict[str, Any]] = field(default_factory=list)
    optimization_time: float = 0.0
    n_iterations: int = 0

class RandomForestModel:
    """Enhanced Random Forest model with additional functionality."""
    
    def __init__(self, 
                 config: RandomForestConfig,
                 model_type: ModelType = ModelType.CLASSIFIER,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize Random Forest model.
        
        Args:
            config: Random Forest configuration
            model_type: Type of model (classifier or regressor)
            feature_names: Names of features for analysis
        """
        self.config = config
        self.model_type = model_type
        self.feature_names = feature_names or []
        self.model = None
        self.is_fitted = False
        self.feature_importance_result = None
        self.tuning_result = None
        self.training_history = []
        
        # Initialize the appropriate model
        if model_type == ModelType.CLASSIFIER:
            self.model = RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                max_features=config.max_features,
                bootstrap=config.bootstrap,
                oob_score=config.oob_score,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbose=config.verbose,
                warm_start=config.warm_start,
                max_samples=config.max_samples
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                max_features=config.max_features,
                bootstrap=config.bootstrap,
                oob_score=config.oob_score,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbose=config.verbose,
                warm_start=config.warm_start,
                max_samples=config.max_samples
            )
        
        logger.info("Random Forest model initialized",
                   model_type=model_type.value,
                   n_estimators=config.n_estimators,
                   max_depth=config.max_depth)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestModel':
        """
        Fit the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional arguments for fit method
        
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Random Forest model", 
                   X_shape=X.shape,
                   y_shape=y.shape)
        
        start_time = datetime.now()
        
        try:
            # Fit the model
            self.model.fit(X, y, **kwargs)
            self.is_fitted = True
            
            # Record training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store training history
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'X_shape': X.shape,
                'y_shape': y.shape,
                'training_time': training_time,
                'oob_score': getattr(self.model, 'oob_score_', None)
            })
            
            logger.info("Random Forest model fitted successfully",
                       training_time=training_time,
                       oob_score=getattr(self.model, 'oob_score_', None))
            
        except Exception as e:
            logger.error("Error fitting Random Forest model", error=str(e))
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model_type == ModelType.CLASSIFIER:
            return self.model.predict_proba(X)
        else:
            raise ValueError("predict_proba is only available for classifiers")
    
    def analyze_feature_importance(self, 
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 method: str = 'default',
                                 top_n: int = 10) -> FeatureImportanceResult:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Method for importance calculation ('default', 'permutation', 'shap')
            top_n: Number of top features to return
        
        Returns:
            FeatureImportanceResult object
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing feature importance")
        
        logger.info("Analyzing feature importance", method=method)
        
        # Get default feature importance
        importance_scores = self.model.feature_importances_
        importance_std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        
        # Create feature names if not provided
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate permutation importance if requested
        permutation_importance = None
        if method in ['permutation', 'both']:
            try:
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(
                    self.model, X, y, n_repeats=10, random_state=42
                )
                permutation_importance = perm_importance.importances_mean
            except Exception as e:
                logger.warning("Could not calculate permutation importance", error=str(e))
        
        # Calculate SHAP values if requested
        shap_values = None
        if method in ['shap', 'both']:
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values)
            except Exception as e:
                logger.warning("Could not calculate SHAP values", error=str(e))
        
        # Create importance ranking
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores,
            'importance_std': importance_std
        })
        
        if permutation_importance is not None:
            importance_df['permutation_importance'] = permutation_importance
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Get top features
        top_features = importance_df.head(top_n)['feature'].tolist()
        
        result = FeatureImportanceResult(
            feature_names=self.feature_names,
            importance_scores=importance_scores,
            importance_std=importance_std,
            permutation_importance=permutation_importance,
            shap_values=shap_values,
            top_features=top_features,
            importance_ranking=importance_df
        )
        
        self.feature_importance_result = result
        
        logger.info("Feature importance analysis completed",
                   top_features=top_features[:5],
                   max_importance=max(importance_scores))
        
        return result
    
    def tune_hyperparameters(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           param_grid: Optional[Dict[str, List]] = None,
                           method: str = 'grid',
                           cv: int = 5,
                           n_iter: int = 100,
                           scoring: str = 'accuracy') -> HyperparameterTuningResult:
        """
        Tune hyperparameters using grid search or random search.
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid for tuning
            method: Tuning method ('grid' or 'random')
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            scoring: Scoring metric
        
        Returns:
            HyperparameterTuningResult object
        """
        logger.info("Starting hyperparameter tuning", method=method)
        
        start_time = datetime.now()
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        try:
            if method == 'grid':
                # Grid search
                search = GridSearchCV(
                    self.model, param_grid, cv=cv, scoring=scoring,
                    n_jobs=-1, verbose=1
                )
                search.fit(X, y)
                
            elif method == 'random':
                # Random search
                search = RandomizedSearchCV(
                    self.model, param_grid, n_iter=n_iter, cv=cv,
                    scoring=scoring, n_jobs=-1, verbose=1,
                    random_state=42
                )
                search.fit(X, y)
            
            else:
                raise ValueError("Method must be 'grid' or 'random'")
            
            # Update model with best parameters
            self.model = search.best_estimator_
            self.is_fitted = True
            
            # Calculate tuning time
            tuning_time = (datetime.now() - start_time).total_seconds()
            
            # Create result object
            result = HyperparameterTuningResult(
                best_params=search.best_params_,
                best_score=search.best_score_,
                cv_results=pd.DataFrame(search.cv_results_),
                optimization_time=tuning_time,
                n_iterations=len(search.cv_results_)
            )
            
            self.tuning_result = result
            
            logger.info("Hyperparameter tuning completed",
                       best_score=search.best_score_,
                       best_params=search.best_params_,
                       tuning_time=tuning_time)
            
            return result
            
        except Exception as e:
            logger.error("Error during hyperparameter tuning", error=str(e))
            raise
    
    def get_oob_score(self) -> Optional[float]:
        """Get out-of-bag score if available."""
        if self.is_fitted and hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        return None
    
    def get_estimator_count(self) -> int:
        """Get number of estimators in the forest."""
        return len(self.model.estimators_)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.is_fitted:
            return self.model.feature_importances_
        return np.array([])
    
    def select_features(self, 
                       X: np.ndarray,
                       y: np.ndarray,
                       threshold: Union[str, float] = 'median') -> Tuple[np.ndarray, List[str]]:
        """
        Select important features based on feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            threshold: Threshold for feature selection
        
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before feature selection")
        
        # Create feature selector
        selector = SelectFromModel(self.model, threshold=threshold)
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support()
        selected_feature_names = [
            self.feature_names[i] for i, selected in enumerate(selected_indices) if selected
        ]
        
        logger.info("Feature selection completed",
                   original_features=X.shape[1],
                   selected_features=X_selected.shape[1],
                   threshold=threshold)
        
        return X_selected, selected_feature_names
    
    def plot_feature_importance(self, 
                              top_n: int = 20,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if self.feature_importance_result is None:
            raise ValueError("Feature importance must be analyzed first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get top features
        top_features = self.feature_importance_result.importance_ranking.head(top_n)
        
        # Create bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], 
                xerr=top_features['importance_std'], capsize=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {
            'model_type': self.model_type.value,
            'is_fitted': self.is_fitted,
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'oob_score': self.get_oob_score(),
            'training_history': self.training_history
        }
        
        if self.feature_importance_result:
            summary['top_features'] = self.feature_importance_result.top_features[:5]
            summary['max_importance'] = max(self.feature_importance_result.importance_scores)
        
        if self.tuning_result:
            summary['best_score'] = self.tuning_result.best_score
            summary['best_params'] = self.tuning_result.best_params
            summary['tuning_time'] = self.tuning_result.optimization_time
        
        return summary

class EnsembleRandomForest:
    """Ensemble of Random Forest models for improved predictions."""
    
    def __init__(self, 
                 base_configs: List[RandomForestConfig],
                 model_type: ModelType = ModelType.CLASSIFIER,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize ensemble of Random Forest models.
        
        Args:
            base_configs: List of configurations for base models
            model_type: Type of models
            feature_names: Names of features
        """
        self.base_configs = base_configs
        self.model_type = model_type
        self.feature_names = feature_names
        self.models = []
        self.is_fitted = False
        
        # Create base models
        for config in base_configs:
            model = RandomForestModel(config, model_type, feature_names)
            self.models.append(model)
        
        logger.info("Ensemble Random Forest initialized",
                   n_models=len(self.models))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleRandomForest':
        """Fit all models in the ensemble."""
        logger.info("Fitting ensemble models")
        
        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i+1}/{len(self.models)}")
            model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, method: str = 'voting') -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            method: Ensemble method ('voting', 'averaging')
        
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if method == 'voting':
            # Majority voting for classification
            predictions = []
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)
            
            # Take majority vote
            predictions_array = np.array(predictions)
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=predictions_array
            )
            
        elif method == 'averaging':
            # Average probabilities for classification
            probabilities = []
            for model in self.models:
                prob = model.predict_proba(X)
                probabilities.append(prob)
            
            avg_prob = np.mean(probabilities, axis=0)
            ensemble_pred = np.argmax(avg_prob, axis=1)
            
        else:
            raise ValueError("Method must be 'voting' or 'averaging'")
        
        return ensemble_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        probabilities = []
        for model in self.models:
            prob = model.predict_proba(X)
            probabilities.append(prob)
        
        return np.mean(probabilities, axis=0)
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get ensemble summary."""
        summary = {
            'n_models': len(self.models),
            'is_fitted': self.is_fitted,
            'model_type': self.model_type.value
        }
        
        if self.is_fitted:
            oob_scores = [model.get_oob_score() for model in self.models]
            summary['oob_scores'] = oob_scores
            summary['mean_oob_score'] = np.mean(oob_scores)
            summary['std_oob_score'] = np.std(oob_scores)
        
        return summary

# Convenience functions
def create_random_forest_classifier(n_estimators: int = 100,
                                  max_depth: Optional[int] = None,
                                  feature_names: Optional[List[str]] = None) -> RandomForestModel:
    """Create a Random Forest classifier with default settings."""
    config = RandomForestConfig(n_estimators=n_estimators, max_depth=max_depth)
    return RandomForestModel(config, ModelType.CLASSIFIER, feature_names)

def create_random_forest_regressor(n_estimators: int = 100,
                                 max_depth: Optional[int] = None,
                                 feature_names: Optional[List[str]] = None) -> RandomForestModel:
    """Create a Random Forest regressor with default settings."""
    config = RandomForestConfig(n_estimators=n_estimators, max_depth=max_depth)
    return RandomForestModel(config, ModelType.REGRESSOR, feature_names)

def create_ensemble_classifier(n_models: int = 3,
                             feature_names: Optional[List[str]] = None) -> EnsembleRandomForest:
    """Create an ensemble of Random Forest classifiers."""
    configs = []
    for i in range(n_models):
        config = RandomForestConfig(
            n_estimators=100 + i * 50,
            max_depth=10 + i * 5,
            random_state=42 + i
        )
        configs.append(config)
    
    return EnsembleRandomForest(configs, ModelType.CLASSIFIER, feature_names)