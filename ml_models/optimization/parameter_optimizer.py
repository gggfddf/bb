"""
Parameter Optimization Framework

A comprehensive parameter optimization framework for ML models that includes:
- Grid search optimization
- Random search optimization
- Bayesian optimization
- Hyperparameter tuning pipeline
- Cross-validation integration
- Model comparison and selection

Features:
- Unified interface for all tree-based models
- Multiple optimization strategies
- Automated hyperparameter tuning
- Cross-validation with early stopping
- Model performance comparison
- Optimization result analysis and visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

logger = structlog.get_logger()

class OptimizationMethod(Enum):
    """Optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

class ModelType(Enum):
    """Supported model types."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    cv_folds: int = 5
    scoring: str = 'accuracy'
    n_iter: int = 100
    n_trials: int = 50
    timeout: Optional[int] = None
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1
    early_stopping_rounds: Optional[int] = 50
    study_name: Optional[str] = None
    storage: Optional[str] = None

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    optimization_time: float
    method: OptimizationMethod
    model_type: ModelType
    n_trials: int
    study: Optional[Any] = None
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ModelComparisonResult:
    """Container for model comparison results."""
    model_results: Dict[str, OptimizationResult]
    comparison_df: pd.DataFrame
    best_model: str
    best_score: float
    comparison_time: float

class ParameterOptimizer:
    """Comprehensive parameter optimization framework."""

    def __init__(self, config: OptimizationConfig):
        """
        Initialize parameter optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.results = {}
        self.comparison_results = None

        # Default parameter grids for different models
        self.default_param_grids = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            ModelType.XGBOOST: {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            },
            ModelType.LIGHTGBM: {
                'n_estimators': [50, 100, 200, 300],
                'num_leaves': [15, 31, 63, 127],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_samples': [10, 20, 50],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
        }

        logger.info("Parameter optimizer initialized",
                   method=config.method.value,
                   cv_folds=config.cv_folds,
                   scoring=config.scoring)

    def optimize_random_forest(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              param_grid: Optional[Dict[str, List]] = None,
                              **kwargs) -> OptimizationResult:
        """
        Optimize Random Forest parameters.

        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid (optional)
            **kwargs: Additional arguments

        Returns:
            OptimizationResult object
        """
        logger.info("Starting Random Forest optimization")

        from ml_models.tree_based.random_forest_model import RandomForestModel, RandomForestConfig

        start_time = datetime.now()

        # Use default parameter grid if not provided
        if param_grid is None:
            param_grid = self.default_param_grids[ModelType.RANDOM_FOREST]

        try:
            if self.config.method == OptimizationMethod.GRID_SEARCH:
                result = self._grid_search_optimization(
                    X, y, param_grid, ModelType.RANDOM_FOREST, **kwargs
                )
            elif self.config.method == OptimizationMethod.RANDOM_SEARCH:
                result = self._random_search_optimization(
                    X, y, param_grid, ModelType.RANDOM_FOREST, **kwargs
                )
            elif self.config.method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
                result = self._bayesian_optimization(
                    X, y, ModelType.RANDOM_FOREST, **kwargs
                )
            else:
                raise ValueError(f"Unsupported optimization method: {self.config.method}")

            optimization_time = (datetime.now() - start_time).total_seconds()
            result.optimization_time = optimization_time

            self.results[ModelType.RANDOM_FOREST.value] = result

            logger.info("Random Forest optimization completed",
                       best_score=result.best_score,
                       optimization_time=optimization_time)

            return result

        except Exception as e:
            logger.error("Error in Random Forest optimization", error=str(e))
            raise

    def optimize_xgboost(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        param_grid: Optional[Dict[str, List]] = None,
                        **kwargs) -> OptimizationResult:
        """
        Optimize XGBoost parameters.

        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid (optional)
            **kwargs: Additional arguments

        Returns:
            OptimizationResult object
        """
        logger.info("Starting XGBoost optimization")

        from ml_models.tree_based.xgboost_model import XGBoostModel, XGBoostConfig

        start_time = datetime.now()

        # Use default parameter grid if not provided
        if param_grid is None:
            param_grid = self.default_param_grids[ModelType.XGBOOST]

        try:
            if self.config.method == OptimizationMethod.GRID_SEARCH:
                result = self._grid_search_optimization(
                    X, y, param_grid, ModelType.XGBOOST, **kwargs
                )
            elif self.config.method == OptimizationMethod.RANDOM_SEARCH:
                result = self._random_search_optimization(
                    X, y, param_grid, ModelType.XGBOOST, **kwargs
                )
            elif self.config.method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
                result = self._bayesian_optimization(
                    X, y, ModelType.XGBOOST, **kwargs
                )
            else:
                raise ValueError(f"Unsupported optimization method: {self.config.method}")

            optimization_time = (datetime.now() - start_time).total_seconds()
            result.optimization_time = optimization_time

            self.results[ModelType.XGBOOST.value] = result

            logger.info("XGBoost optimization completed",
                       best_score=result.best_score,
                       optimization_time=optimization_time)

            return result

        except Exception as e:
            logger.error("Error in XGBoost optimization", error=str(e))
            raise

    def optimize_lightgbm(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         param_grid: Optional[Dict[str, List]] = None,
                         **kwargs) -> OptimizationResult:
        """
        Optimize LightGBM parameters.

        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid (optional)
            **kwargs: Additional arguments

        Returns:
            OptimizationResult object
        """
        logger.info("Starting LightGBM optimization")

        from ml_models.tree_based.lightgbm_model import LightGBMModel, LightGBMConfig

        start_time = datetime.now()

        # Use default parameter grid if not provided
        if param_grid is None:
            param_grid = self.default_param_grids[ModelType.LIGHTGBM]

        try:
            if self.config.method == OptimizationMethod.GRID_SEARCH:
                result = self._grid_search_optimization(
                    X, y, param_grid, ModelType.LIGHTGBM, **kwargs
                )
            elif self.config.method == OptimizationMethod.RANDOM_SEARCH:
                result = self._random_search_optimization(
                    X, y, param_grid, ModelType.LIGHTGBM, **kwargs
                )
            elif self.config.method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
                result = self._bayesian_optimization(
                    X, y, ModelType.LIGHTGBM, **kwargs
                )
            else:
                raise ValueError(f"Unsupported optimization method: {self.config.method}")

            optimization_time = (datetime.now() - start_time).total_seconds()
            result.optimization_time = optimization_time

            self.results[ModelType.LIGHTGBM.value] = result

            logger.info("LightGBM optimization completed",
                       best_score=result.best_score,
                       optimization_time=optimization_time)

            return result

        except Exception as e:
            logger.error("Error in LightGBM optimization", error=str(e))
            raise

    def _grid_search_optimization(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 param_grid: Dict[str, List],
                                 model_type: ModelType,
                                 **kwargs) -> OptimizationResult:
        """
        Perform grid search optimization.

        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid
            model_type: Type of model
            **kwargs: Additional arguments

        Returns:
            OptimizationResult object
        """
        logger.info("Performing grid search optimization")

        # Create base model
        base_model = self._create_base_model(model_type, **kwargs)

        # Use TimeSeriesSplit for time series data
        cv = TimeSeriesSplit(n_splits=self.config.cv_folds)

        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=self.config.scoring,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            return_train_score=True
        )

        grid_search.fit(X, y)

        # Create result object
        result = OptimizationResult(
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            cv_results=pd.DataFrame(grid_search.cv_results_),
            optimization_time=0.0,  # Will be set by calling method
            method=self.config.method,
            model_type=model_type,
            n_trials=len(grid_search.cv_results_)
        )

        return result

    def _random_search_optimization(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  param_grid: Dict[str, List],
                                  model_type: ModelType,
                                  **kwargs) -> OptimizationResult:
        """
        Perform random search optimization.

        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid
            model_type: Type of model
            **kwargs: Additional arguments

        Returns:
            OptimizationResult object
        """
        logger.info("Performing random search optimization")

        # Create base model
        base_model = self._create_base_model(model_type, **kwargs)

        # Use TimeSeriesSplit for time series data
        cv = TimeSeriesSplit(n_splits=self.config.cv_folds)

        # Perform random search
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=self.config.n_iter,
            cv=cv,
            scoring=self.config.scoring,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            random_state=self.config.random_state,
            return_train_score=True
        )

        random_search.fit(X, y)

        # Create result object
        result = OptimizationResult(
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            cv_results=pd.DataFrame(random_search.cv_results_),
            optimization_time=0.0,  # Will be set by calling method
            method=self.config.method,
            model_type=model_type,
            n_trials=len(random_search.cv_results_)
        )

        return result

    def _bayesian_optimization(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              model_type: ModelType,
                              **kwargs) -> OptimizationResult:
        """
        Perform Bayesian optimization using Optuna.

        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model
            **kwargs: Additional arguments

        Returns:
            OptimizationResult object
        """
        logger.info("Performing Bayesian optimization")

        # Create study
        study_name = self.config.study_name or f"{model_type.value}_optimization"
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner(),
            study_name=study_name,
            storage=self.config.storage
        )

        # Define objective function
        def objective(trial):
            # Get hyperparameters for this trial
            params = self._get_trial_params(trial, model_type)
            
            # Create model with trial parameters
            model = self._create_model_with_params(model_type, params, **kwargs)
            
            # Use TimeSeriesSplit for cross-validation
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                try:
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_val)
                    
                    # Calculate score
                    score = self._calculate_score(y_val, y_pred, self.config.scoring)
                    scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    scores.append(0.0)
            
            return np.mean(scores)

        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )

        # Create result object
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            cv_results=pd.DataFrame(study.trials_dataframe()),
            optimization_time=0.0,  # Will be set by calling method
            method=self.config.method,
            model_type=model_type,
            n_trials=len(study.trials),
            study=study
        )

        return result

    def _create_base_model(self, model_type: ModelType, **kwargs):
        """Create base model for sklearn compatibility."""
        if model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if kwargs.get('task_type') == 'regression':
                return RandomForestRegressor(random_state=self.config.random_state)
            else:
                return RandomForestClassifier(random_state=self.config.random_state)
        
        elif model_type == ModelType.XGBOOST:
            import xgboost as xgb
            if kwargs.get('task_type') == 'regression':
                return xgb.XGBRegressor(random_state=self.config.random_state)
            else:
                return xgb.XGBClassifier(random_state=self.config.random_state)
        
        elif model_type == ModelType.LIGHTGBM:
            import lightgbm as lgb
            if kwargs.get('task_type') == 'regression':
                return lgb.LGBMRegressor(random_state=self.config.random_state)
            else:
                return lgb.LGBMClassifier(random_state=self.config.random_state)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_model_with_params(self, model_type: ModelType, params: Dict[str, Any], **kwargs):
        """Create model with specific parameters."""
        if model_type == ModelType.RANDOM_FOREST:
            from ml_models.tree_based.random_forest_model import RandomForestModel, RandomForestConfig
            config = RandomForestConfig(**params)
            return RandomForestModel(config, **kwargs)
        
        elif model_type == ModelType.XGBOOST:
            from ml_models.tree_based.xgboost_model import XGBoostModel, XGBoostConfig
            config = XGBoostConfig(**params)
            return XGBoostModel(config, **kwargs)
        
        elif model_type == ModelType.LIGHTGBM:
            from ml_models.tree_based.lightgbm_model import LightGBMModel, LightGBMConfig
            config = LightGBMConfig(**params)
            return LightGBMModel(config, **kwargs)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _get_trial_params(self, trial, model_type: ModelType) -> Dict[str, Any]:
        """Get parameters for Optuna trial."""
        if model_type == ModelType.RANDOM_FOREST:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        
        elif model_type == ModelType.XGBOOST:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
            }
        
        elif model_type == ModelType.LIGHTGBM:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
            }
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
        """Calculate score based on scoring metric."""
        if scoring == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif scoring == 'precision':
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
        elif scoring == 'recall':
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
        elif scoring == 'f1':
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        elif scoring == 'mse':
            return -mean_squared_error(y_true, y_pred)  # Negative for maximization
        elif scoring == 'r2':
            return r2_score(y_true, y_pred)
        else:
            return accuracy_score(y_true, y_pred)

    def compare_models(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      models: List[ModelType],
                      **kwargs) -> ModelComparisonResult:
        """
        Compare multiple models using the same optimization method.

        Args:
            X: Feature matrix
            y: Target variable
            models: List of models to compare
            **kwargs: Additional arguments

        Returns:
            ModelComparisonResult object
        """
        logger.info("Starting model comparison", models=[m.value for m in models])

        start_time = datetime.now()
        model_results = {}

        # Optimize each model
        for model_type in models:
            if model_type == ModelType.RANDOM_FOREST:
                result = self.optimize_random_forest(X, y, **kwargs)
            elif model_type == ModelType.XGBOOST:
                result = self.optimize_xgboost(X, y, **kwargs)
            elif model_type == ModelType.LIGHTGBM:
                result = self.optimize_lightgbm(X, y, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            model_results[model_type.value] = result

        # Create comparison dataframe
        comparison_data = []
        for model_name, result in model_results.items():
            comparison_data.append({
                'model': model_name,
                'best_score': result.best_score,
                'optimization_time': result.optimization_time,
                'n_trials': result.n_trials,
                'method': result.method.value
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('best_score', ascending=False)

        # Find best model
        best_model = comparison_df.iloc[0]['model']
        best_score = comparison_df.iloc[0]['best_score']

        comparison_time = (datetime.now() - start_time).total_seconds()

        # Create result object
        self.comparison_results = ModelComparisonResult(
            model_results=model_results,
            comparison_df=comparison_df,
            best_model=best_model,
            best_score=best_score,
            comparison_time=comparison_time
        )

        logger.info("Model comparison completed",
                   best_model=best_model,
                   best_score=best_score,
                   comparison_time=comparison_time)

        return self.comparison_results

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results."""
        summary = {
            'total_models': len(self.results),
            'optimization_method': self.config.method.value,
            'scoring': self.config.scoring,
            'cv_folds': self.config.cv_folds
        }

        if self.results:
            summary['model_results'] = {}
            for model_name, result in self.results.items():
                summary['model_results'][model_name] = {
                    'best_score': result.best_score,
                    'optimization_time': result.optimization_time,
                    'n_trials': result.n_trials,
                    'best_params': result.best_params
                }

        if self.comparison_results:
            summary['comparison'] = {
                'best_model': self.comparison_results.best_model,
                'best_score': self.comparison_results.best_score,
                'comparison_time': self.comparison_results.comparison_time
            }

        return summary

    def plot_optimization_results(self, model_name: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot optimization results.

        Args:
            model_name: Specific model to plot (optional)

        Returns:
            Matplotlib figure
        """
        if not self.results:
            logger.warning("No optimization results available")
            return None

        try:
            if model_name and model_name not in self.results:
                logger.warning(f"Model {model_name} not found in results")
                return None

            models_to_plot = [model_name] if model_name else list(self.results.keys())

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Parameter Optimization Results', fontsize=16)

            for i, model_name in enumerate(models_to_plot):
                result = self.results[model_name]

                # Plot 1: Score distribution
                if result.method == OptimizationMethod.BAYESIAN_OPTIMIZATION and result.study:
                    scores = [trial.value for trial in result.study.trials if trial.value is not None]
                    axes[0, 0].hist(scores, bins=20, alpha=0.7, label=model_name)
                    axes[0, 0].set_title('Score Distribution')
                    axes[0, 0].set_xlabel('Score')
                    axes[0, 0].set_ylabel('Frequency')
                    axes[0, 0].legend()

                # Plot 2: Optimization history
                if result.method == OptimizationMethod.BAYESIAN_OPTIMIZATION and result.study:
                    optuna.visualization.matplotlib.plot_optimization_history(result.study, ax=axes[0, 1])
                    axes[0, 1].set_title('Optimization History')

                # Plot 3: Parameter importance
                if result.method == OptimizationMethod.BAYESIAN_OPTIMIZATION and result.study:
                    optuna.visualization.matplotlib.plot_param_importances(result.study, ax=axes[1, 0])
                    axes[1, 0].set_title('Parameter Importance')

                # Plot 4: Parameter relationships
                if result.method == OptimizationMethod.BAYESIAN_OPTIMIZATION and result.study:
                    optuna.visualization.matplotlib.plot_parallel_coordinate(result.study, ax=axes[1, 1])
                    axes[1, 1].set_title('Parameter Relationships')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error("Error plotting optimization results", error=str(e))
            return None

    def plot_model_comparison(self) -> Optional[plt.Figure]:
        """Plot model comparison results."""
        if not self.comparison_results:
            logger.warning("No comparison results available")
            return None

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Comparison Results', fontsize=16)

            # Plot 1: Best scores comparison
            models = self.comparison_results.comparison_df['model']
            scores = self.comparison_results.comparison_df['best_score']
            bars = axes[0, 0].bar(models, scores)
            axes[0, 0].set_title('Best Scores Comparison')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{score:.3f}', ha='center', va='bottom')

            # Plot 2: Optimization time comparison
            times = self.comparison_results.comparison_df['optimization_time']
            bars = axes[0, 1].bar(models, times)
            axes[0, 1].set_title('Optimization Time Comparison')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, time in zip(bars, times):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{time:.1f}s', ha='center', va='bottom')

            # Plot 3: Number of trials comparison
            trials = self.comparison_results.comparison_df['n_trials']
            bars = axes[1, 0].bar(models, trials)
            axes[1, 0].set_title('Number of Trials Comparison')
            axes[1, 0].set_ylabel('Number of Trials')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, trial in zip(bars, trials):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{trial}', ha='center', va='bottom')

            # Plot 4: Score vs Time scatter plot
            axes[1, 1].scatter(times, scores, s=100, alpha=0.7)
            for i, model in enumerate(models):
                axes[1, 1].annotate(model, (times[i], scores[i]), 
                                  xytext=(5, 5), textcoords='offset points')
            axes[1, 1].set_xlabel('Optimization Time (seconds)')
            axes[1, 1].set_ylabel('Best Score')
            axes[1, 1].set_title('Score vs Optimization Time')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error("Error plotting model comparison", error=str(e))
            return None

# Convenience functions
def create_parameter_optimizer(method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                              cv_folds: int = 5,
                              scoring: str = 'accuracy',
                              n_iter: int = 100,
                              n_trials: int = 50) -> ParameterOptimizer:
    """Create a parameter optimizer with default settings."""
    config = OptimizationConfig(
        method=method,
        cv_folds=cv_folds,
        scoring=scoring,
        n_iter=n_iter,
        n_trials=n_trials
    )
    return ParameterOptimizer(config)

def optimize_all_models(X: np.ndarray,
                       y: np.ndarray,
                       method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                       **kwargs) -> ModelComparisonResult:
    """Optimize all supported models and compare results."""
    optimizer = create_parameter_optimizer(method=method, **kwargs)
    models = [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]
    return optimizer.compare_models(X, y, models)