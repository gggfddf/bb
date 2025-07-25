"""
XGBoost Model Implementation

A comprehensive XGBoost implementation for stock prediction with:
- Early stopping for optimal training
- Feature importance analysis
- Hyperparameter optimization
- Cross-validation integration
- Model evaluation integration

Features:
- Custom XGBoost wrapper with enhanced functionality
- Early stopping with validation data
- Feature importance visualization and analysis
- Hyperparameter optimization with multiple strategies
- Cross-validation with early stopping
- Integration with model evaluation framework
- Advanced training monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = structlog.get_logger()

class ModelType(Enum):
    """Model types for XGBoost."""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"

@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model."""
    # Core parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0
    
    # Regularization
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_weight: int = 1
    
    # Early stopping
    early_stopping_rounds: Optional[int] = 10
    eval_metric: str = 'logloss'
    
    # Other parameters
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 0
    tree_method: str = 'auto'
    booster: str = 'gbtree'
    objective: Optional[str] = None

@dataclass
class TrainingHistory:
    """Container for training history."""
    train_scores: List[float] = field(default_factory=list)
    val_scores: List[float] = field(default_factory=list)
    best_iteration: Optional[int] = None
    training_time: float = 0.0
    early_stopped: bool = False

@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    feature_names: List[str]
    importance_scores: np.ndarray
    importance_types: Dict[str, np.ndarray] = field(default_factory=dict)
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

class XGBoostModel:
    """Enhanced XGBoost model with additional functionality."""
    
    def __init__(self, 
                 config: XGBoostConfig,
                 model_type: ModelType = ModelType.CLASSIFIER,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize XGBoost model.
        
        Args:
            config: XGBoost configuration
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
        self.training_history = TrainingHistory()
        
        # Set objective based on model type
        if config.objective is None:
            if model_type == ModelType.CLASSIFIER:
                config.objective = 'binary:logistic'
            else:
                config.objective = 'reg:squarederror'
        
        # Initialize the appropriate model
        if model_type == ModelType.CLASSIFIER:
            self.model = xgb.XGBClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                colsample_bylevel=config.colsample_bylevel,
                colsample_bynode=config.colsample_bynode,
                reg_alpha=config.reg_alpha,
                reg_lambda=config.reg_lambda,
                min_child_weight=config.min_child_weight,
                eval_metric=config.eval_metric,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbosity=config.verbosity,
                tree_method=config.tree_method,
                booster=config.booster,
                objective=config.objective
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                colsample_bylevel=config.colsample_bylevel,
                colsample_bynode=config.colsample_bynode,
                reg_alpha=config.reg_alpha,
                reg_lambda=config.reg_lambda,
                min_child_weight=config.min_child_weight,
                eval_metric=config.eval_metric,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbosity=config.verbosity,
                tree_method=config.tree_method,
                booster=config.booster,
                objective=config.objective
            )
        
        logger.info("XGBoost model initialized",
                   model_type=model_type.value,
                   n_estimators=config.n_estimators,
                   learning_rate=config.learning_rate,
                   max_depth=config.max_depth)
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs) -> 'XGBoostModel':
        """
        Fit the XGBoost model with early stopping.
        
        Args:
            X: Training feature matrix
            y: Training target variable
            X_val: Validation feature matrix (optional)
            y_val: Validation target variable (optional)
            **kwargs: Additional arguments for fit method
        
        Returns:
            Self for method chaining
        """
        logger.info("Fitting XGBoost model", 
                   X_shape=X.shape,
                   y_shape=y.shape,
                   has_validation=X_val is not None)
        
        start_time = datetime.now()
        
        try:
            # Prepare validation data if provided
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            
            # Fit the model with early stopping
            self.model.fit(
                X, y,
                eval_set=eval_set,
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False,
                **kwargs
            )
            
            self.is_fitted = True
            
            # Record training time
            training_time = (datetime.now() - start_time).total_seconds()
            self.training_history.training_time = training_time
            
            # Record early stopping information
            if hasattr(self.model, 'best_iteration'):
                self.training_history.best_iteration = self.model.best_iteration
                self.training_history.early_stopped = True
            
            # Record evaluation scores if available
            if eval_set is not None:
                train_scores = self.model.evals_result().get('validation_0', {})
                if self.config.eval_metric in train_scores:
                    self.training_history.val_scores = train_scores[self.config.eval_metric]
            
            logger.info("XGBoost model fitted successfully",
                       training_time=training_time,
                       best_iteration=self.training_history.best_iteration,
                       early_stopped=self.training_history.early_stopped)
            
        except Exception as e:
            logger.error("Error fitting XGBoost model", error=str(e))
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
                                 importance_type: str = 'weight',
                                 top_n: int = 10) -> FeatureImportanceResult:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
            top_n: Number of top features to return
        
        Returns:
            FeatureImportanceResult object
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing feature importance")
        
        logger.info("Analyzing feature importance", importance_type=importance_type)
        
        # Create feature names if not provided
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Get feature importance scores
        importance_scores = self.model.feature_importances_
        
        # Get detailed importance information
        importance_types = {}
        try:
            # Get importance for different types
            for imp_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
                try:
                    imp_scores = self.model.get_booster().get_score(importance_type=imp_type)
                    # Convert to array format
                    imp_array = np.zeros(len(self.feature_names))
                    for i, feature in enumerate(self.feature_names):
                        if feature in imp_scores:
                            imp_array[i] = imp_scores[feature]
                    importance_types[imp_type] = imp_array
                except Exception as e:
                    logger.warning(f"Could not get {imp_type} importance", error=str(e))
        except Exception as e:
            logger.warning("Could not get detailed importance types", error=str(e))
        
        # Create importance ranking
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })
        
        # Add other importance types if available
        for imp_type, imp_scores in importance_types.items():
            importance_df[f'{imp_type}_importance'] = imp_scores
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Get top features
        top_features = importance_df.head(top_n)['feature'].tolist()
        
        result = FeatureImportanceResult(
            feature_names=self.feature_names,
            importance_scores=importance_scores,
            importance_types=importance_types,
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
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None,
                           param_grid: Optional[Dict[str, List]] = None,
                           method: str = 'grid',
                           cv: int = 5,
                           n_iter: int = 100,
                           scoring: str = 'accuracy',
                           early_stopping_rounds: int = 10) -> HyperparameterTuningResult:
        """
        Tune hyperparameters using grid search or random search with early stopping.
        
        Args:
            X: Feature matrix
            y: Target variable
            X_val: Validation data (optional)
            y_val: Validation targets (optional)
            param_grid: Parameter grid for tuning
            method: Tuning method ('grid' or 'random')
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            scoring: Scoring metric
            early_stopping_rounds: Early stopping rounds
        
        Returns:
            HyperparameterTuningResult object
        """
        logger.info("Starting hyperparameter tuning", method=method)
        
        start_time = datetime.now()
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [0, 0.1, 1.0]
            }
        
        try:
            # Create a custom scoring function that supports early stopping
            def custom_scorer(estimator, X, y):
                if X_val is not None and y_val is not None:
                    # Use validation data for early stopping
                    estimator.fit(X, y, eval_set=[(X_val, y_val)], 
                                early_stopping_rounds=early_stopping_rounds, verbose=False)
                else:
                    estimator.fit(X, y)
                
                if self.model_type == ModelType.CLASSIFIER:
                    return accuracy_score(y, estimator.predict(X))
                else:
                    from sklearn.metrics import mean_squared_error
                    return -mean_squared_error(y, estimator.predict(X))  # Negative for maximization
            
            if method == 'grid':
                # Grid search
                search = GridSearchCV(
                    self.model, param_grid, cv=cv, scoring=custom_scorer,
                    n_jobs=-1, verbose=1
                )
                search.fit(X, y)
                
            elif method == 'random':
                # Random search
                search = RandomizedSearchCV(
                    self.model, param_grid, n_iter=n_iter, cv=cv,
                    scoring=custom_scorer, n_jobs=-1, verbose=1,
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
    
    def cross_validate_with_early_stopping(self,
                                         X: np.ndarray,
                                         y: np.ndarray,
                                         cv: int = 5,
                                         scoring: str = 'accuracy',
                                         early_stopping_rounds: int = 10) -> Dict[str, List[float]]:
        """
        Perform cross-validation with early stopping.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            scoring: Scoring metric
            early_stopping_rounds: Early stopping rounds
        
        Returns:
            Dictionary with cross-validation results
        """
        logger.info("Performing cross-validation with early stopping")
        
        from sklearn.model_selection import KFold
        
        scores = []
        best_iterations = []
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Training fold {fold + 1}/{cv}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create a fresh model for this fold
            fold_model = xgb.XGBClassifier(**self.model.get_params()) if self.model_type == ModelType.CLASSIFIER else xgb.XGBRegressor(**self.model.get_params())
            
            # Fit with early stopping
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            
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
            
            # Record best iteration
            if hasattr(fold_model, 'best_iteration'):
                best_iterations.append(fold_model.best_iteration)
            else:
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
    
    def get_best_iteration(self) -> Optional[int]:
        """Get the best iteration from early stopping."""
        if self.is_fitted and hasattr(self.model, 'best_iteration'):
            return self.model.best_iteration
        return None
    
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
                              importance_type: str = 'weight',
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            importance_type: Type of importance to plot
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
        ax.barh(y_pos, top_features['importance'])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances ({importance_type})')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 6)) -> Optional[plt.Figure]:
        """
        Plot training history if available.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure or None if no history available
        """
        if not self.training_history.val_scores:
            logger.warning("No training history available for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        iterations = range(1, len(self.training_history.val_scores) + 1)
        ax.plot(iterations, self.training_history.val_scores, 'b-', label='Validation Score')
        
        if self.training_history.best_iteration:
            ax.axvline(x=self.training_history.best_iteration, color='r', linestyle='--', 
                      label=f'Best Iteration ({self.training_history.best_iteration})')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_title('XGBoost Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {
            'model_type': self.model_type.value,
            'is_fitted': self.is_fitted,
            'n_estimators': self.config.n_estimators,
            'learning_rate': self.config.learning_rate,
            'max_depth': self.config.max_depth,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'best_iteration': self.get_best_iteration(),
            'early_stopped': self.training_history.early_stopped,
            'training_time': self.training_history.training_time
        }
        
        if self.feature_importance_result:
            summary['top_features'] = self.feature_importance_result.top_features[:5]
            summary['max_importance'] = max(self.feature_importance_result.importance_scores)
        
        if self.tuning_result:
            summary['best_score'] = self.tuning_result.best_score
            summary['best_params'] = self.tuning_result.best_params
            summary['tuning_time'] = self.tuning_result.optimization_time
        
        return summary

# Convenience functions
def create_xgboost_classifier(n_estimators: int = 100,
                             learning_rate: float = 0.1,
                             max_depth: int = 6,
                             feature_names: Optional[List[str]] = None) -> XGBoostModel:
    """Create an XGBoost classifier with default settings."""
    config = XGBoostConfig(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    return XGBoostModel(config, ModelType.CLASSIFIER, feature_names)

def create_xgboost_regressor(n_estimators: int = 100,
                            learning_rate: float = 0.1,
                            max_depth: int = 6,
                            feature_names: Optional[List[str]] = None) -> XGBoostModel:
    """Create an XGBoost regressor with default settings."""
    config = XGBoostConfig(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    return XGBoostModel(config, ModelType.REGRESSOR, feature_names)

def create_xgboost_with_early_stopping(n_estimators: int = 1000,
                                      learning_rate: float = 0.1,
                                      early_stopping_rounds: int = 50,
                                      feature_names: Optional[List[str]] = None) -> XGBoostModel:
    """Create an XGBoost model optimized for early stopping."""
    config = XGBoostConfig(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds
    )
    return XGBoostModel(config, ModelType.CLASSIFIER, feature_names)