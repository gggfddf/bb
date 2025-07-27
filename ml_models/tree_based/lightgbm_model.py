"""
LightGBM Model Implementation

A comprehensive LightGBM implementation for stock prediction with:
- Categorical feature handling
- Feature importance analysis
- Hyperparameter tuning
- Cross-validation
- Early stopping
- Model evaluation integration

Features:
- Custom LightGBM wrapper with enhanced functionality
- Categorical feature encoding and handling
- Feature importance visualization and analysis
- Hyperparameter optimization with multiple strategies
- Cross-validation with early stopping
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
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = structlog.get_logger()

class ModelType(Enum):
    """Model types for LightGBM."""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"

@dataclass
class LightGBMConfig:
    """Configuration for LightGBM model."""
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = -1
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1
    early_stopping_rounds: Optional[int] = None
    categorical_feature: Optional[Union[str, List[int]]] = 'auto'
    objective: Optional[str] = None
    metric: Optional[str] = None
    boosting_type: str = 'gbdt'
    min_split_gain: float = 0.0
    min_child_weight: float = 1e-3
    subsample_freq: int = 0
    colsample_bynode: float = 1.0
    extra_trees: bool = False
    path_smooth: float = 0.0

@dataclass
class CategoricalFeatureInfo:
    """Information about categorical features."""
    feature_names: List[str]
    feature_indices: List[int]
    encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    cardinalities: Dict[str, int] = field(default_factory=dict)
    encoding_method: str = 'label'

@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    feature_names: List[str]
    importance_scores: np.ndarray
    importance_std: np.ndarray
    importance_types: Dict[str, np.ndarray] = field(default_factory=dict)
    top_features: List[str] = field(default_factory=list)
    importance_ranking: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    categorical_features: Optional[CategoricalFeatureInfo] = None

@dataclass
class HyperparameterTuningResult:
    """Container for hyperparameter tuning results."""
    best_params: Dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    tuning_history: List[Dict[str, Any]] = field(default_factory=list)
    optimization_time: float = 0.0
    n_iterations: int = 0

@dataclass
class TrainingHistory:
    """Container for training history."""
    training_time: float = 0.0
    best_iteration: Optional[int] = None
    best_score: Optional[float] = None
    train_scores: List[float] = field(default_factory=list)
    val_scores: List[float] = field(default_factory=list)
    early_stopped: bool = False
    evals_result: Optional[Dict] = None

class LightGBMModel:
    """Enhanced LightGBM model with additional functionality."""
    
    def __init__(self, 
                 config: LightGBMConfig,
                 model_type: ModelType = ModelType.CLASSIFIER,
                 feature_names: Optional[List[str]] = None,
                 categorical_features: Optional[List[str]] = None):
        """
        Initialize LightGBM model.
        
        Args:
            config: LightGBM configuration
            model_type: Type of model (classifier or regressor)
            feature_names: Names of features for analysis
            categorical_features: Names of categorical features
        """
        self.config = config
        self.model_type = model_type
        self.feature_names = feature_names or []
        self.categorical_features = categorical_features or []
        self.model = None
        self.is_fitted = False
        self.feature_importance_result = None
        self.tuning_result = None
        self.training_history = TrainingHistory()
        self.categorical_info = None
        
        # Set objective and metric based on model type
        if model_type == ModelType.CLASSIFIER:
            if config.objective is None:
                config.objective = 'binary'
            if config.metric is None:
                config.metric = 'binary_logloss'
        else:
            if config.objective is None:
                config.objective = 'regression'
            if config.metric is None:
                config.metric = 'rmse'
        
        # Initialize categorical feature handling
        if categorical_features:
            self._setup_categorical_features()
        
        logger.info("LightGBM model initialized",
                   model_type=model_type.value,
                   n_estimators=config.n_estimators,
                   learning_rate=config.learning_rate,
                   categorical_features=len(self.categorical_features))
    
    def _setup_categorical_features(self):
        """Setup categorical feature handling."""
        categorical_indices = []
        encoders = {}
        cardinalities = {}
        
        for feature_name in self.categorical_features:
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                categorical_indices.append(feature_idx)
                encoders[feature_name] = LabelEncoder()
                cardinalities[feature_name] = 0  # Will be set during fitting
        
        self.categorical_info = CategoricalFeatureInfo(
            feature_names=self.categorical_features,
            feature_indices=categorical_indices,
            encoders=encoders,
            cardinalities=cardinalities
        )
        
        # Update config with categorical features
        if categorical_indices:
            self.config.categorical_feature = categorical_indices
    
    def _encode_categorical_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Encode categorical features.
        
        Args:
            X: Feature matrix
            fit: Whether to fit encoders (True for training, False for prediction)
        
        Returns:
            Encoded feature matrix
        """
        if not self.categorical_info:
            return X
        
        X_encoded = X.copy()
        
        for i, feature_name in enumerate(self.categorical_info.feature_names):
            feature_idx = self.categorical_info.feature_indices[i]
            if feature_idx < X.shape[1]:
                feature_data = X[:, feature_idx]
                
                # Handle missing values
                feature_data = np.where(feature_data == None, 'MISSING', feature_data)
                feature_data = np.where(feature_data == '', 'EMPTY', feature_data)
                
                if fit:
                    # Fit encoder
                    encoded_values = self.categorical_info.encoders[feature_name].fit_transform(feature_data)
                    self.categorical_info.cardinalities[feature_name] = len(
                        self.categorical_info.encoders[feature_name].classes_
                    )
                else:
                    # Transform with fitted encoder
                    try:
                        encoded_values = self.categorical_info.encoders[feature_name].transform(feature_data)
                    except ValueError:
                        # Handle unseen categories
                        encoded_values = np.full(len(feature_data), -1)
                
                X_encoded[:, feature_idx] = encoded_values
        
        return X_encoded
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs) -> 'LightGBMModel':
        """
        Fit the LightGBM model.
        
        Args:
            X: Feature matrix
            y: Target variable
            X_val: Validation feature matrix (optional)
            y_val: Validation target variable (optional)
            **kwargs: Additional arguments for fit method
        
        Returns:
            Self for method chaining
        """
        logger.info("Fitting LightGBM model", 
                   X_shape=X.shape,
                   y_shape=y.shape,
                   categorical_features=len(self.categorical_features))
        
        start_time = datetime.now()
        
        try:
            # Encode categorical features
            X_encoded = self._encode_categorical_features(X, fit=True)
            
            # Create dataset
            train_data = lgb.Dataset(X_encoded, label=y, feature_name=self.feature_names)
            
            # Prepare validation data if provided
            valid_data = None
            if X_val is not None and y_val is not None:
                X_val_encoded = self._encode_categorical_features(X_val, fit=False)
                valid_data = lgb.Dataset(X_val_encoded, label=y_val, feature_name=self.feature_names)
            
            # Prepare parameters
            params = {
                'objective': self.config.objective,
                'metric': self.config.metric,
                'boosting_type': self.config.boosting_type,
                'num_leaves': self.config.num_leaves,
                'learning_rate': self.config.learning_rate,
                'feature_fraction': self.config.colsample_bytree,
                'bagging_fraction': self.config.subsample,
                'bagging_freq': self.config.subsample_freq,
                'verbose': self.config.verbose,
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs,
                'min_child_samples': self.config.min_child_samples,
                'min_child_weight': self.config.min_child_weight,
                'min_split_gain': self.config.min_split_gain,
                'reg_alpha': self.config.reg_alpha,
                'reg_lambda': self.config.reg_lambda,
                'path_smooth': self.config.path_smooth,
                'extra_trees': self.config.extra_trees
            }
            
            # Add categorical features if present
            if self.categorical_info and self.categorical_info.feature_indices:
                params['categorical_feature'] = self.categorical_info.feature_indices
            
            # Train model
            callbacks = []
            if self.config.early_stopping_rounds and valid_data:
                callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds))
                callbacks.append(lgb.record_evaluation(self.training_history.evals_result))
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.config.n_estimators,
                valid_sets=[valid_data] if valid_data else None,
                callbacks=callbacks,
                **kwargs
            )
            
            self.is_fitted = True
            
            # Record training time
            training_time = (datetime.now() - start_time).total_seconds()
            self.training_history.training_time = training_time
            
            # Record best iteration and score
            if hasattr(self.model, 'best_iteration'):
                self.training_history.best_iteration = self.model.best_iteration
                self.training_history.early_stopped = True
            else:
                self.training_history.best_iteration = self.config.n_estimators
            
            if self.training_history.evals_result:
                # Extract best score from evaluation results
                for metric_name, scores in self.training_history.evals_result.items():
                    if 'valid' in metric_name:
                        self.training_history.val_scores = scores
                        if self.training_history.best_iteration:
                            self.training_history.best_score = scores[self.training_history.best_iteration - 1]
                        break
            
            logger.info("LightGBM model fitted successfully",
                       training_time=training_time,
                       best_iteration=self.training_history.best_iteration,
                       best_score=self.training_history.best_score)
            
        except Exception as e:
            logger.error("Error fitting LightGBM model", error=str(e))
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
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X, fit=False)
        
        return self.model.predict(X_encoded)
    
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
            # Encode categorical features
            X_encoded = self._encode_categorical_features(X, fit=False)
            
            # Get raw predictions
            raw_preds = self.model.predict(X_encoded, pred_leaf=False, pred_contrib=False)
            
            # Convert to probabilities for binary classification
            if self.config.objective == 'binary':
                probs = 1 / (1 + np.exp(-raw_preds))
                return np.column_stack([1 - probs, probs])
            else:
                # For multiclass, LightGBM returns probabilities directly
                return raw_preds
        else:
            raise ValueError("predict_proba is only available for classifiers")
    
    def get_feature_importance(self, importance_type: str = 'split') -> np.ndarray:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('split', 'gain', 'shap')
        
        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model.feature_importance(importance_type=importance_type)
    
    def analyze_feature_importance(self, 
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 importance_type: str = 'split',
                                 top_n: int = 10) -> FeatureImportanceResult:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            importance_type: Type of importance ('split', 'gain', 'shap')
            top_n: Number of top features to return
        
        Returns:
            FeatureImportanceResult object
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing feature importance")
        
        logger.info("Analyzing feature importance", importance_type=importance_type)
        
        # Get feature importance for different types
        importance_types = {}
        for imp_type in ['split', 'gain']:
            try:
                importance_types[imp_type] = self.model.feature_importance(importance_type=imp_type)
            except Exception as e:
                logger.warning(f"Could not calculate {imp_type} importance", error=str(e))
                importance_types[imp_type] = np.zeros(len(self.feature_names))
        
        # Use the requested importance type as primary
        importance_scores = importance_types.get(importance_type, importance_types.get('split', np.zeros(len(self.feature_names))))
        
        # Create feature names if not provided
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create importance ranking
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })
        
        # Add other importance types
        for imp_type, scores in importance_types.items():
            if imp_type != importance_type:
                importance_df[f'{imp_type}_importance'] = scores
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Get top features
        top_features = importance_df.head(top_n)['feature'].tolist()
        
        result = FeatureImportanceResult(
            feature_names=self.feature_names,
            importance_scores=importance_scores,
            importance_std=np.zeros(len(importance_scores)),  # LightGBM doesn't provide std
            importance_types=importance_types,
            top_features=top_features,
            importance_ranking=importance_df,
            categorical_features=self.categorical_info
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
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [15, 31, 63],
                'min_child_samples': [10, 20, 50],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        
        try:
            # Encode categorical features
            X_encoded = self._encode_categorical_features(X, fit=True)
            
            # Create base model
            base_params = {
                'objective': self.config.objective,
                'metric': self.config.metric,
                'boosting_type': self.config.boosting_type,
                'verbose': -1,
                'random_state': self.config.random_state,
                'n_jobs': 1  # Use 1 for CV to avoid conflicts
            }
            
            # Add categorical features if present
            if self.categorical_info and self.categorical_info.feature_indices:
                base_params['categorical_feature'] = self.categorical_info.feature_indices
            
            # Create LightGBM estimator for sklearn compatibility
            if self.model_type == ModelType.CLASSIFIER:
                base_model = lgb.LGBMClassifier(**base_params)
            else:
                base_model = lgb.LGBMRegressor(**base_params)
            
            if method == 'grid':
                # Grid search
                search = GridSearchCV(
                    base_model, param_grid, cv=cv, scoring=scoring,
                    n_jobs=1, verbose=1
                )
                search.fit(X_encoded, y)
                
            elif method == 'random':
                # Random search
                search = RandomizedSearchCV(
                    base_model, param_grid, n_iter=n_iter, cv=cv,
                    scoring=scoring, n_jobs=1, verbose=1,
                    random_state=42
                )
                search.fit(X_encoded, y)
            
            else:
                raise ValueError("Method must be 'grid' or 'random'")
            
            # Update model with best parameters
            self.model = search.best_estimator_.booster_
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
                       optimization_time=tuning_time)
            
        except Exception as e:
            logger.error("Error during hyperparameter tuning", error=str(e))
            raise
        
        return result
    
    def cross_validate_with_early_stopping(self,
                                         X: np.ndarray,
                                         y: np.ndarray,
                                         cv: int = 5,
                                         scoring: str = 'accuracy',
                                         early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        Perform cross-validation with early stopping.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            scoring: Scoring metric
            early_stopping_rounds: Number of rounds for early stopping
        
        Returns:
            Dictionary with cross-validation results
        """
        logger.info("Starting cross-validation with early stopping")
        
        from sklearn.model_selection import TimeSeriesSplit
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv)
        
        scores = []
        best_iterations = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{cv}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and fit model for this fold
            fold_config = LightGBMConfig(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                early_stopping_rounds=early_stopping_rounds,
                **{k: v for k, v in self.config.__dict__.items() 
                   if k not in ['n_estimators', 'learning_rate', 'early_stopping_rounds']}
            )
            
            fold_model = LightGBMModel(
                fold_config, 
                self.model_type, 
                self.feature_names,
                self.categorical_features
            )
            
            try:
                fold_model.fit(X_train, y_train, X_val, y_val)
                
                # Make predictions
                y_pred = fold_model.predict(X_val)
                
                # Calculate score
                if scoring == 'accuracy':
                    score = accuracy_score(y_val, y_pred)
                elif scoring == 'precision':
                    score = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                elif scoring == 'recall':
                    score = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                elif scoring == 'f1':
                    score = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                else:
                    score = accuracy_score(y_val, y_pred)
                
                scores.append(score)
                best_iterations.append(fold_model.training_history.best_iteration or self.config.n_estimators)
                
                logger.info(f"Fold {fold + 1} completed",
                           score=score,
                           best_iteration=fold_model.training_history.best_iteration)
                
            except Exception as e:
                logger.warning(f"Error in fold {fold + 1}", error=str(e))
                scores.append(0.0)
                best_iterations.append(self.config.n_estimators)
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'best_iterations': best_iterations,
            'mean_best_iteration': np.mean(best_iterations)
        }
    
    def select_features(self, 
                       X: np.ndarray, 
                       y: np.ndarray, 
                       threshold: Union[str, float] = 'median') -> Tuple[np.ndarray, List[str]]:
        """
        Select features based on importance scores.
        
        Args:
            X: Feature matrix
            y: Target variable
            threshold: Threshold for feature selection ('median', 'mean', or float)
        
        Returns:
            Tuple of (selected_features_matrix, selected_feature_names)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before feature selection")
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Determine threshold
        if threshold == 'median':
            thresh = np.median(importance)
        elif threshold == 'mean':
            thresh = np.mean(importance)
        else:
            thresh = threshold
        
        # Select features
        selected_indices = np.where(importance >= thresh)[0]
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        logger.info("Feature selection completed",
                   original_features=len(self.feature_names),
                   selected_features=len(selected_features),
                   threshold=thresh)
        
        return X[:, selected_indices], selected_features
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        summary = {
            'model_type': self.model_type.value,
            'is_fitted': self.is_fitted,
            'n_estimators': self.config.n_estimators,
            'learning_rate': self.config.learning_rate,
            'num_leaves': self.config.num_leaves,
            'max_depth': self.config.max_depth,
            'feature_count': len(self.feature_names),
            'categorical_features': len(self.categorical_features),
            'best_iteration': self.training_history.best_iteration,
            'best_score': self.training_history.best_score,
            'early_stopped': self.training_history.early_stopped,
            'training_time': self.training_history.training_time
        }
        
        # Add feature importance info
        if self.feature_importance_result:
            summary['top_features'] = self.feature_importance_result.top_features[:5]
            summary['max_importance'] = max(self.feature_importance_result.importance_scores)
        
        # Add tuning info
        if self.tuning_result:
            summary['tuning_score'] = self.tuning_result.best_score
            summary['tuning_time'] = self.tuning_result.optimization_time
        
        return summary
    
    def plot_feature_importance(self, top_n: int = 20) -> Optional[plt.Figure]:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
        
        Returns:
            Matplotlib figure
        """
        if not self.feature_importance_result:
            logger.warning("No feature importance result available")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get top features
            top_features = self.feature_importance_result.top_features[:top_n]
            top_importance = [self.feature_importance_result.importance_scores[
                self.feature_names.index(f)] for f in top_features]
            
            # Create bar plot
            bars = ax.barh(range(len(top_features)), top_importance)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {len(top_features)} Feature Importance')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, top_importance)):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error("Error plotting feature importance", error=str(e))
            return None
    
    def plot_training_history(self) -> Optional[plt.Figure]:
        """
        Plot training history.
        
        Returns:
            Matplotlib figure
        """
        if not self.training_history.evals_result:
            logger.warning("No training history available")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for metric_name, scores in self.training_history.evals_result.items():
                ax.plot(scores, label=metric_name)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score')
            ax.set_title('Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error("Error plotting training history", error=str(e))
            return None

# Convenience functions for easy usage
def create_lightgbm_classifier(n_estimators: int = 100,
                              learning_rate: float = 0.1,
                              num_leaves: int = 31,
                              feature_names: Optional[List[str]] = None,
                              categorical_features: Optional[List[str]] = None) -> LightGBMModel:
    """Create a LightGBM classifier with default parameters."""
    config = LightGBMConfig(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves
    )
    return LightGBMModel(config, ModelType.CLASSIFIER, feature_names, categorical_features)

def create_lightgbm_regressor(n_estimators: int = 100,
                             learning_rate: float = 0.1,
                             num_leaves: int = 31,
                             feature_names: Optional[List[str]] = None,
                             categorical_features: Optional[List[str]] = None) -> LightGBMModel:
    """Create a LightGBM regressor with default parameters."""
    config = LightGBMConfig(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves
    )
    return LightGBMModel(config, ModelType.REGRESSOR, feature_names, categorical_features)

def create_lightgbm_with_early_stopping(n_estimators: int = 1000,
                                       learning_rate: float = 0.1,
                                       early_stopping_rounds: int = 50,
                                       feature_names: Optional[List[str]] = None,
                                       categorical_features: Optional[List[str]] = None) -> LightGBMModel:
    """Create a LightGBM model with early stopping."""
    config = LightGBMConfig(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds
    )
    return LightGBMModel(config, ModelType.CLASSIFIER, feature_names, categorical_features)