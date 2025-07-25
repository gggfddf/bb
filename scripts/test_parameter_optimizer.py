#!/usr/bin/env python3
"""
Test script for the Unified Parameter Optimizer.
Tests optimization for Random Forest, XGBoost, and LightGBM models, validates best parameter selection, and checks edge cases.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml_models.optimization.parameter_optimizer import (
    optimize_model, ParameterOptimizer, SearchMethod
)
from sklearn.ensemble import RandomForestClassifier
import structlog

logger = structlog.get_logger()

try:
    from xgboost import XGBClassifier
    import lightgbm as lgb
except ImportError:
    XGBClassifier = None
    lgb = None

class ParameterOptimizerTestSuite:
    def __init__(self):
        self.X, self.y = self._generate_data()

    def _generate_data(self, n_samples=200, n_features=8, n_classes=2):
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, size=n_samples)
        return X, y

    async def test_random_forest_grid(self):
        logger.info("Testing Random Forest Grid Search")
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [2, 4],
        }
        result = optimize_model(model, self.X, self.y, param_grid, method="grid", cv=3, scoring="accuracy", verbose=0)
        assert isinstance(result.best_params, dict)
        assert 'n_estimators' in result.best_params
        logger.info("Random Forest Grid Search Passed", best_params=result.best_params)
        return True

    async def test_random_forest_random(self):
        logger.info("Testing Random Forest Random Search")
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [10, 20, 30],
            'max_depth': [2, 4, 6],
        }
        result = optimize_model(model, self.X, self.y, param_grid, method="random", n_iter=3, cv=3, scoring="accuracy", verbose=0)
        assert isinstance(result.best_params, dict)
        logger.info("Random Forest Random Search Passed", best_params=result.best_params)
        return True

    async def test_xgboost_grid(self):
        if XGBClassifier is None:
            logger.warning("XGBoost not installed, skipping test.")
            return True
        logger.info("Testing XGBoost Grid Search")
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [2, 4],
        }
        result = optimize_model(model, self.X, self.y, param_grid, method="grid", cv=3, scoring="accuracy", verbose=0)
        assert isinstance(result.best_params, dict)
        logger.info("XGBoost Grid Search Passed", best_params=result.best_params)
        return True

    async def test_lightgbm_grid(self):
        if lgb is None:
            logger.warning("LightGBM not installed, skipping test.")
            return True
        logger.info("Testing LightGBM Grid Search")
        model = lgb.LGBMClassifier(random_state=42)
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [2, 4],
        }
        result = optimize_model(model, self.X, self.y, param_grid, method="grid", cv=3, scoring="accuracy", verbose=0)
        assert isinstance(result.best_params, dict)
        logger.info("LightGBM Grid Search Passed", best_params=result.best_params)
        return True

    async def test_invalid_method(self):
        logger.info("Testing Invalid Search Method")
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [10, 20]}
        try:
            optimize_model(model, self.X, self.y, param_grid, method="invalid", cv=3, scoring="accuracy", verbose=0)
        except ValueError as e:
            logger.info("Invalid method correctly raised ValueError", error=str(e))
            return True
        return False

    async def run_all_tests(self):
        results = []
        results.append(await self.test_random_forest_grid())
        results.append(await self.test_random_forest_random())
        results.append(await self.test_xgboost_grid())
        results.append(await self.test_lightgbm_grid())
        results.append(await self.test_invalid_method())
        return all(results)

async def main():
    logger.info("Starting Parameter Optimizer Test Suite")
    test_suite = ParameterOptimizerTestSuite()
    success = await test_suite.run_all_tests()
    if success:
        logger.info("Parameter Optimizer Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Parameter Optimizer Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())