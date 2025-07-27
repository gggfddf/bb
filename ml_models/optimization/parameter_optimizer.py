"""
Unified Parameter Optimization Framework

Supports grid search, random search, and is compatible with Random Forest, XGBoost, and LightGBM models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import structlog
import time

logger = structlog.get_logger()

class SearchMethod(Enum):
    GRID = "grid"
    RANDOM = "random"
    # BAYESIAN = "bayesian"  # Placeholder for future extension

@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    search_method: str
    optimization_time: float
    n_iterations: int
    model_name: str
    scoring: str
    search_history: List[Dict[str, Any]] = field(default_factory=list)

class ParameterOptimizer:
    """
    Unified parameter optimizer for tree-based models.
    Supports grid search and random search.
    """
    def __init__(self,
                 model,
                 param_grid: Dict[str, List[Any]],
                 method: SearchMethod = SearchMethod.GRID,
                 cv: int = 5,
                 n_iter: int = 50,
                 scoring: str = 'accuracy',
                 random_state: int = 42,
                 verbose: int = 1):
        self.model = model
        self.param_grid = param_grid
        self.method = method
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        self.searcher = None
        self.result: Optional[OptimizationResult] = None

    def optimize(self, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        logger.info("Starting parameter optimization",
                    model=str(self.model),
                    method=self.method.value,
                    scoring=self.scoring)
        start_time = time.time()
        if self.method == SearchMethod.GRID:
            self.searcher = GridSearchCV(
                self.model,
                self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=1,
                verbose=self.verbose
            )
        elif self.method == SearchMethod.RANDOM:
            self.searcher = RandomizedSearchCV(
                self.model,
                self.param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=1,
                verbose=self.verbose,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported search method: {self.method}")

        self.searcher.fit(X, y)
        elapsed = time.time() - start_time
        logger.info("Parameter optimization completed",
                    best_score=self.searcher.best_score_,
                    best_params=self.searcher.best_params_,
                    time=elapsed)
        self.result = OptimizationResult(
            best_params=self.searcher.best_params_,
            best_score=self.searcher.best_score_,
            cv_results=pd.DataFrame(self.searcher.cv_results_),
            search_method=self.method.value,
            optimization_time=elapsed,
            n_iterations=len(self.searcher.cv_results_["params"]),
            model_name=type(self.model).__name__,
            scoring=self.scoring
        )
        return self.result

    def get_best_model(self):
        if self.searcher is None:
            raise ValueError("No searcher has been run yet.")
        return self.searcher.best_estimator_

# Convenience function for unified API

def optimize_model(model, X, y, param_grid, method="grid", cv=5, n_iter=50, scoring="accuracy", random_state=42, verbose=1):
    method_enum = SearchMethod(method)
    optimizer = ParameterOptimizer(
        model=model,
        param_grid=param_grid,
        method=method_enum,
        cv=cv,
        n_iter=n_iter,
        scoring=scoring,
        random_state=random_state,
        verbose=verbose
    )
    return optimizer.optimize(X, y)