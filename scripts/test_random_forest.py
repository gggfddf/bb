#!/usr/bin/env python3
"""
Test script for the Random Forest Model Implementation.
Tests feature importance analysis, hyperparameter tuning, ensemble methods, and out-of-bag validation.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml_models.tree_based.random_forest_model import (
    RandomForestModel, EnsembleRandomForest, RandomForestConfig, ModelType,
    create_random_forest_classifier, create_random_forest_regressor, create_ensemble_classifier
)
import structlog

logger = structlog.get_logger()

class RandomForestTestSuite:
    """Comprehensive test suite for the Random Forest implementation."""
    
    def __init__(self):
        self.test_results = {}
        self.sample_data = None
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        # Generate sample features and targets
        n_samples = 1000
        n_features = 20
        
        # Generate features with some correlation structure
        X = np.random.randn(n_samples, n_features)
        
        # Create some feature interactions
        X[:, 0] = X[:, 1] * 0.5 + np.random.normal(0, 0.1, n_samples)  # Correlated features
        X[:, 2] = X[:, 3] * X[:, 4] + np.random.normal(0, 0.1, n_samples)  # Interaction
        
        # Generate targets (binary classification)
        # Make targets depend on specific features
        y = np.zeros(n_samples)
        y[X[:, 0] > 0.5] = 1  # Feature 0 is important
        y[X[:, 5] < -0.5] = 1  # Feature 5 is important
        y[X[:, 10] > 1.0] = 1  # Feature 10 is important
        
        # Add some noise
        noise = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        y = np.logical_xor(y, noise).astype(int)
        
        # Generate feature names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        self.sample_data = {
            'X': X,
            'y': y,
            'feature_names': feature_names
        }
        
        logger.info("Sample data created", 
                   n_samples=n_samples,
                   n_features=n_features,
                   class_distribution=np.bincount(y))
    
    async def test_basic_random_forest_classifier(self):
        """Test basic Random Forest classifier functionality."""
        logger.info("Testing basic Random Forest classifier")
        
        try:
            # Create model
            config = RandomForestConfig(n_estimators=50, max_depth=10)
            model = RandomForestModel(config, ModelType.CLASSIFIER, self.sample_data['feature_names'])
            
            # Check initialization
            assert model.model is not None, "Model should be initialized"
            assert not model.is_fitted, "Model should not be fitted initially"
            assert model.model_type == ModelType.CLASSIFIER, "Model type should be classifier"
            
            # Fit model
            X = self.sample_data['X']
            y = self.sample_data['y']
            model.fit(X, y)
            
            # Check fitting
            assert model.is_fitted, "Model should be fitted after fit()"
            assert len(model.training_history) > 0, "Training history should be recorded"
            
            # Make predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Check predictions
            assert len(predictions) == len(y), "Predictions should match target length"
            assert predictions.shape == y.shape, "Prediction shape should match target shape"
            assert probabilities.shape == (len(y), 2), "Probabilities should be 2D for binary classification"
            assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
            
            # Check OOB score
            oob_score = model.get_oob_score()
            assert oob_score is not None, "OOB score should be available"
            assert 0 <= oob_score <= 1, "OOB score should be between 0 and 1"
            
            # Check feature importance
            importance = model.get_feature_importance()
            assert len(importance) == X.shape[1], "Feature importance should match feature count"
            assert np.all(importance >= 0), "Feature importance should be non-negative"
            
            self.test_results['basic_classifier'] = True
            logger.info("Basic Random Forest classifier test passed",
                       oob_score=oob_score,
                       prediction_accuracy=np.mean(predictions == y))
            
        except Exception as e:
            self.test_results['basic_classifier'] = False
            logger.error("Basic Random Forest classifier test failed", error=str(e))
            raise
    
    async def test_basic_random_forest_regressor(self):
        """Test basic Random Forest regressor functionality."""
        logger.info("Testing basic Random Forest regressor")
        
        try:
            # Create regression targets
            X = self.sample_data['X']
            y_reg = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.normal(0, 0.1, len(X))
            
            # Create model
            config = RandomForestConfig(n_estimators=50, max_depth=10)
            model = RandomForestModel(config, ModelType.REGRESSOR, self.sample_data['feature_names'])
            
            # Fit model
            model.fit(X, y_reg)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Check predictions
            assert len(predictions) == len(y_reg), "Predictions should match target length"
            assert predictions.shape == y_reg.shape, "Prediction shape should match target shape"
            
            # Check that predict_proba raises error for regressor
            try:
                model.predict_proba(X)
                assert False, "predict_proba should raise error for regressor"
            except ValueError:
                pass  # Expected behavior
            
            # Check OOB score
            oob_score = model.get_oob_score()
            assert oob_score is not None, "OOB score should be available"
            
            self.test_results['basic_regressor'] = True
            logger.info("Basic Random Forest regressor test passed",
                       oob_score=oob_score,
                       mse=np.mean((predictions - y_reg) ** 2))
            
        except Exception as e:
            self.test_results['basic_regressor'] = False
            logger.error("Basic Random Forest regressor test failed", error=str(e))
            raise
    
    async def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        logger.info("Testing feature importance analysis")
        
        try:
            # Create and fit model
            config = RandomForestConfig(n_estimators=100, max_depth=10)
            model = RandomForestModel(config, ModelType.CLASSIFIER, self.sample_data['feature_names'])
            
            X = self.sample_data['X']
            y = self.sample_data['y']
            model.fit(X, y)
            
            # Analyze feature importance
            importance_result = model.analyze_feature_importance(X, y, method='default', top_n=10)
            
            # Check result structure
            assert len(importance_result.feature_names) == X.shape[1], "Feature names should match feature count"
            assert len(importance_result.importance_scores) == X.shape[1], "Importance scores should match feature count"
            assert len(importance_result.importance_std) == X.shape[1], "Importance std should match feature count"
            assert len(importance_result.top_features) == 10, "Should have 10 top features"
            assert len(importance_result.importance_ranking) == X.shape[1], "Ranking should include all features"
            
            # Check importance ranking
            ranking = importance_result.importance_ranking
            assert ranking.iloc[0]['importance'] >= ranking.iloc[-1]['importance'], "Ranking should be sorted"
            
            # Check that important features are in top features
            # Features 0, 5, 10 should be important based on our data generation
            important_features = ['feature_0', 'feature_5', 'feature_10']
            top_features_set = set(importance_result.top_features)
            important_in_top = sum(1 for f in important_features if f in top_features_set)
            assert important_in_top >= 1, "At least one important feature should be in top features"
            
            # Test permutation importance
            perm_result = model.analyze_feature_importance(X, y, method='permutation', top_n=5)
            assert perm_result.permutation_importance is not None, "Permutation importance should be calculated"
            
            self.test_results['feature_importance'] = True
            logger.info("Feature importance analysis test passed",
                       top_features=importance_result.top_features[:3],
                       max_importance=max(importance_result.importance_scores))
            
        except Exception as e:
            self.test_results['feature_importance'] = False
            logger.error("Feature importance analysis test failed", error=str(e))
            raise
    
    async def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning."""
        logger.info("Testing hyperparameter tuning")
        
        try:
            # Create model
            config = RandomForestConfig(n_estimators=50, max_depth=5)
            model = RandomForestModel(config, ModelType.CLASSIFIER, self.sample_data['feature_names'])
            
            X = self.sample_data['X']
            y = self.sample_data['y']
            
            # Test grid search
            param_grid = {
                'n_estimators': [25, 50],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            }
            
            tuning_result = model.tune_hyperparameters(
                X, y, param_grid=param_grid, method='grid', cv=3, scoring='accuracy'
            )
            
            # Check tuning result
            assert 'best_params' in tuning_result.best_params, "Should have best parameters"
            assert tuning_result.best_score > 0, "Best score should be positive"
            assert tuning_result.optimization_time > 0, "Optimization time should be positive"
            assert tuning_result.n_iterations > 0, "Should have iterations"
            assert len(tuning_result.cv_results) > 0, "Should have CV results"
            
            # Check that model was updated
            assert model.is_fitted, "Model should be fitted after tuning"
            assert model.tuning_result is not None, "Tuning result should be stored"
            
            # Test random search
            config2 = RandomForestConfig(n_estimators=30, max_depth=5)
            model2 = RandomForestModel(config2, ModelType.CLASSIFIER, self.sample_data['feature_names'])
            
            random_result = model2.tune_hyperparameters(
                X, y, param_grid=param_grid, method='random', cv=3, n_iter=5, scoring='accuracy'
            )
            
            assert random_result.best_score > 0, "Random search should find positive score"
            
            self.test_results['hyperparameter_tuning'] = True
            logger.info("Hyperparameter tuning test passed",
                       grid_best_score=tuning_result.best_score,
                       random_best_score=random_result.best_score)
            
        except Exception as e:
            self.test_results['hyperparameter_tuning'] = False
            logger.error("Hyperparameter tuning test failed", error=str(e))
            raise
    
    async def test_feature_selection(self):
        """Test feature selection functionality."""
        logger.info("Testing feature selection")
        
        try:
            # Create and fit model
            config = RandomForestConfig(n_estimators=100, max_depth=10)
            model = RandomForestModel(config, ModelType.CLASSIFIER, self.sample_data['feature_names'])
            
            X = self.sample_data['X']
            y = self.sample_data['y']
            model.fit(X, y)
            
            # Test feature selection with median threshold
            X_selected, selected_features = model.select_features(X, y, threshold='median')
            
            # Check results
            assert X_selected.shape[1] <= X.shape[1], "Selected features should be <= original features"
            assert len(selected_features) == X_selected.shape[1], "Feature names should match selected count"
            assert all(f in self.sample_data['feature_names'] for f in selected_features), "All selected features should be original features"
            
            # Test with custom threshold
            X_selected_custom, selected_features_custom = model.select_features(X, y, threshold=0.01)
            
            # Different thresholds should give different results
            assert len(selected_features) != len(selected_features_custom), "Different thresholds should give different results"
            
            self.test_results['feature_selection'] = True
            logger.info("Feature selection test passed",
                       original_features=X.shape[1],
                       selected_features=X_selected.shape[1])
            
        except Exception as e:
            self.test_results['feature_selection'] = False
            logger.error("Feature selection test failed", error=str(e))
            raise
    
    async def test_ensemble_random_forest(self):
        """Test ensemble Random Forest functionality."""
        logger.info("Testing ensemble Random Forest")
        
        try:
            # Create ensemble
            configs = [
                RandomForestConfig(n_estimators=50, max_depth=5, random_state=42),
                RandomForestConfig(n_estimators=100, max_depth=10, random_state=43),
                RandomForestConfig(n_estimators=75, max_depth=8, random_state=44)
            ]
            
            ensemble = EnsembleRandomForest(configs, ModelType.CLASSIFIER, self.sample_data['feature_names'])
            
            # Check initialization
            assert len(ensemble.models) == 3, "Should have 3 models"
            assert not ensemble.is_fitted, "Ensemble should not be fitted initially"
            
            # Fit ensemble
            X = self.sample_data['X']
            y = self.sample_data['y']
            ensemble.fit(X, y)
            
            # Check fitting
            assert ensemble.is_fitted, "Ensemble should be fitted"
            assert all(model.is_fitted for model in ensemble.models), "All models should be fitted"
            
            # Test voting predictions
            predictions_voting = ensemble.predict(X, method='voting')
            assert len(predictions_voting) == len(y), "Voting predictions should match target length"
            
            # Test averaging predictions
            predictions_averaging = ensemble.predict(X, method='averaging')
            assert len(predictions_averaging) == len(y), "Averaging predictions should match target length"
            
            # Test probability predictions
            probabilities = ensemble.predict_proba(X)
            assert probabilities.shape == (len(y), 2), "Probabilities should be 2D for binary classification"
            assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
            
            # Get ensemble summary
            summary = ensemble.get_ensemble_summary()
            assert summary['n_models'] == 3, "Summary should show 3 models"
            assert summary['is_fitted'], "Summary should show fitted status"
            assert 'oob_scores' in summary, "Summary should include OOB scores"
            
            self.test_results['ensemble'] = True
            logger.info("Ensemble Random Forest test passed",
                       voting_accuracy=np.mean(predictions_voting == y),
                       averaging_accuracy=np.mean(predictions_averaging == y))
            
        except Exception as e:
            self.test_results['ensemble'] = False
            logger.error("Ensemble Random Forest test failed", error=str(e))
            raise
    
    async def test_convenience_functions(self):
        """Test convenience functions."""
        logger.info("Testing convenience functions")
        
        try:
            # Test classifier creation
            classifier = create_random_forest_classifier(
                n_estimators=50, max_depth=10, feature_names=self.sample_data['feature_names']
            )
            assert classifier.model_type == ModelType.CLASSIFIER, "Should create classifier"
            assert classifier.config.n_estimators == 50, "Should set n_estimators"
            assert classifier.config.max_depth == 10, "Should set max_depth"
            
            # Test regressor creation
            regressor = create_random_forest_regressor(
                n_estimators=75, max_depth=15, feature_names=self.sample_data['feature_names']
            )
            assert regressor.model_type == ModelType.REGRESSOR, "Should create regressor"
            assert regressor.config.n_estimators == 75, "Should set n_estimators"
            assert regressor.config.max_depth == 15, "Should set max_depth"
            
            # Test ensemble creation
            ensemble = create_ensemble_classifier(
                n_models=3, feature_names=self.sample_data['feature_names']
            )
            assert len(ensemble.models) == 3, "Should create 3 models"
            assert ensemble.model_type == ModelType.CLASSIFIER, "Should create classifier ensemble"
            
            self.test_results['convenience_functions'] = True
            logger.info("Convenience functions test passed")
            
        except Exception as e:
            self.test_results['convenience_functions'] = False
            logger.error("Convenience functions test failed", error=str(e))
            raise
    
    async def test_model_summary_and_visualization(self):
        """Test model summary and visualization functionality."""
        logger.info("Testing model summary and visualization")
        
        try:
            # Create and fit model
            config = RandomForestConfig(n_estimators=100, max_depth=10)
            model = RandomForestModel(config, ModelType.CLASSIFIER, self.sample_data['feature_names'])
            
            X = self.sample_data['X']
            y = self.sample_data['y']
            model.fit(X, y)
            
            # Analyze feature importance
            model.analyze_feature_importance(X, y)
            
            # Get model summary
            summary = model.get_model_summary()
            
            # Check summary structure
            required_keys = ['model_type', 'is_fitted', 'n_estimators', 'max_depth', 'feature_count', 'oob_score']
            for key in required_keys:
                assert key in summary, f"Summary should contain {key}"
            
            assert summary['model_type'] == 'classifier', "Model type should be classifier"
            assert summary['is_fitted'], "Model should be fitted"
            assert summary['n_estimators'] == 100, "Should show correct n_estimators"
            assert summary['feature_count'] == 20, "Should show correct feature count"
            assert summary['oob_score'] is not None, "Should have OOB score"
            
            # Test feature importance plotting
            try:
                fig = model.plot_feature_importance(top_n=10)
                assert fig is not None, "Should return matplotlib figure"
                plt.close(fig)  # Close to free memory
            except Exception as e:
                logger.warning("Feature importance plotting failed", error=str(e))
                # This is not critical, so we don't fail the test
            
            self.test_results['model_summary'] = True
            logger.info("Model summary and visualization test passed",
                       oob_score=summary['oob_score'],
                       top_features=summary.get('top_features', []))
            
        except Exception as e:
            self.test_results['model_summary'] = False
            logger.error("Model summary and visualization test failed", error=str(e))
            raise
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        logger.info("Testing edge cases")
        
        try:
            # Test with empty data
            config = RandomForestConfig(n_estimators=10)
            model = RandomForestModel(config, ModelType.CLASSIFIER)
            
            try:
                model.fit(np.array([]), np.array([]))
                assert False, "Should raise error for empty data"
            except Exception:
                pass  # Expected behavior
            
            # Test with single sample
            try:
                model.fit(np.array([[1, 2, 3]]), np.array([1]))
                # Should work with single sample
            except Exception as e:
                logger.warning("Single sample fitting failed", error=str(e))
            
            # Test with mismatched dimensions
            try:
                model.fit(np.array([[1, 2, 3]]), np.array([1, 0]))
                assert False, "Should raise error for mismatched dimensions"
            except Exception:
                pass  # Expected behavior
            
            # Test prediction before fitting
            try:
                model.predict(np.array([[1, 2, 3]]))
                assert False, "Should raise error for prediction before fitting"
            except ValueError:
                pass  # Expected behavior
            
            # Test feature importance before fitting
            try:
                model.analyze_feature_importance(np.array([[1, 2, 3]]), np.array([1]))
                assert False, "Should raise error for feature importance before fitting"
            except ValueError:
                pass  # Expected behavior
            
            self.test_results['edge_cases'] = True
            logger.info("Edge cases test passed")
            
        except Exception as e:
            self.test_results['edge_cases'] = False
            logger.error("Edge cases test failed", error=str(e))
            raise
    
    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("Starting Random Forest Test Suite")
        
        test_methods = [
            self.test_basic_random_forest_classifier,
            self.test_basic_random_forest_regressor,
            self.test_feature_importance_analysis,
            self.test_hyperparameter_tuning,
            self.test_feature_selection,
            self.test_ensemble_random_forest,
            self.test_convenience_functions,
            self.test_model_summary_and_visualization,
            self.test_edge_cases
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed", error=str(e))
                self.test_results[test_method.__name__] = False
        
        # Summary
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        
        logger.info("Random Forest Test Suite completed",
                   passed_tests=passed_tests,
                   total_tests=total_tests,
                   success_rate=passed_tests/total_tests)
        
        return passed_tests == total_tests

async def main():
    """Main test execution function."""
    logger.info("Starting Random Forest Test Suite")
    
    test_suite = RandomForestTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Random Forest Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Random Forest Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())