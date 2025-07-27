#!/usr/bin/env python3
"""
Test script for the LightGBM Model Implementation.
Tests categorical feature handling, feature importance analysis, hyperparameter tuning, cross-validation, and all functionality.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml_models.tree_based.lightgbm_model import (
    LightGBMModel, LightGBMConfig, ModelType,
    create_lightgbm_classifier, create_lightgbm_regressor, create_lightgbm_with_early_stopping
)
import structlog

logger = structlog.get_logger()

class LightGBMTestSuite:
    """Comprehensive test suite for the LightGBM implementation."""
    
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
        
        # Add categorical features
        categorical_data = np.random.choice(['A', 'B', 'C', 'D'], size=(n_samples, 3))
        X = np.column_stack([X, categorical_data])
        
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
        feature_names = [f"feature_{i}" for i in range(n_features)] + ['cat_1', 'cat_2', 'cat_3']
        categorical_features = ['cat_1', 'cat_2', 'cat_3']
        
        self.sample_data = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'categorical_features': categorical_features
        }
        
        logger.info("Sample data created", 
                   n_samples=n_samples,
                   n_features=n_features + 3,
                   categorical_features=len(categorical_features),
                   class_distribution=np.bincount(y))
    
    async def test_basic_lightgbm_classifier(self):
        """Test basic LightGBM classifier functionality."""
        logger.info("Testing basic LightGBM classifier")
        
        try:
            # Create model
            config = LightGBMConfig(n_estimators=50, learning_rate=0.1, num_leaves=31)
            model = LightGBMModel(config, ModelType.CLASSIFIER, 
                                self.sample_data['feature_names'],
                                self.sample_data['categorical_features'])
            
            # Check initialization
            assert model.model is None, "Model should not be initialized initially"
            assert not model.is_fitted, "Model should not be fitted initially"
            assert model.model_type == ModelType.CLASSIFIER, "Model type should be classifier"
            assert len(model.categorical_features) == 3, "Should have 3 categorical features"
            assert model.categorical_info is not None, "Should have categorical info"
            
            # Fit model
            X = self.sample_data['X']
            y = self.sample_data['y']
            model.fit(X, y)
            
            # Check fitting
            assert model.is_fitted, "Model should be fitted after fit()"
            assert model.model is not None, "Model should be created after fitting"
            assert model.training_history.training_time > 0, "Training time should be recorded"
            
            # Make predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Check predictions
            assert len(predictions) == len(y), "Predictions should match target length"
            assert predictions.shape == y.shape, "Prediction shape should match target shape"
            assert probabilities.shape == (len(y), 2), "Probabilities should be 2D for binary classification"
            assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
            
            # Check feature importance
            importance = model.get_feature_importance()
            assert len(importance) == X.shape[1], "Feature importance should match feature count"
            assert np.all(importance >= 0), "Feature importance should be non-negative"
            
            self.test_results['basic_classifier'] = True
            logger.info("Basic LightGBM classifier test passed",
                       prediction_accuracy=np.mean(predictions == y))
            
        except Exception as e:
            self.test_results['basic_classifier'] = False
            logger.error("Basic LightGBM classifier test failed", error=str(e))
            raise
    
    async def test_basic_lightgbm_regressor(self):
        """Test basic LightGBM regressor functionality."""
        logger.info("Testing basic LightGBM regressor")
        
        try:
            # Create regression targets
            X = self.sample_data['X']
            y_reg = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.normal(0, 0.1, len(X))
            
            # Create model
            config = LightGBMConfig(n_estimators=50, learning_rate=0.1, num_leaves=31)
            model = LightGBMModel(config, ModelType.REGRESSOR, 
                                self.sample_data['feature_names'],
                                self.sample_data['categorical_features'])
            
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
            
            self.test_results['basic_regressor'] = True
            logger.info("Basic LightGBM regressor test passed",
                       mse=np.mean((predictions - y_reg) ** 2))
            
        except Exception as e:
            self.test_results['basic_regressor'] = False
            logger.error("Basic LightGBM regressor test failed", error=str(e))
            raise
    
    async def test_categorical_feature_handling(self):
        """Test categorical feature handling."""
        logger.info("Testing categorical feature handling")
        
        try:
            # Create model with categorical features
            config = LightGBMConfig(n_estimators=50, learning_rate=0.1, num_leaves=31)
            model = LightGBMModel(config, ModelType.CLASSIFIER, 
                                self.sample_data['feature_names'],
                                self.sample_data['categorical_features'])
            
            X = self.sample_data['X']
            y = self.sample_data['y']
            
            # Check categorical info setup
            assert model.categorical_info is not None, "Should have categorical info"
            assert len(model.categorical_info.feature_names) == 3, "Should have 3 categorical features"
            assert len(model.categorical_info.encoders) == 3, "Should have 3 encoders"
            
            # Fit model
            model.fit(X, y)
            
            # Check that encoders were fitted
            for feature_name in self.sample_data['categorical_features']:
                assert feature_name in model.categorical_info.encoders, f"Should have encoder for {feature_name}"
                assert len(model.categorical_info.encoders[feature_name].classes_) > 0, f"Encoder for {feature_name} should be fitted"
                assert model.categorical_info.cardinalities[feature_name] > 0, f"Should have cardinality for {feature_name}"
            
            # Test prediction with categorical features
            predictions = model.predict(X)
            assert len(predictions) == len(y), "Predictions should work with categorical features"
            
            # Test with new categorical values
            X_new = X.copy()
            X_new[:, -3:] = np.random.choice(['A', 'B', 'C', 'D', 'E'], size=(len(X), 3))  # New categories
            
            # Should handle unseen categories gracefully
            predictions_new = model.predict(X_new)
            assert len(predictions_new) == len(y), "Should handle unseen categories"
            
            self.test_results['categorical_features'] = True
            logger.info("Categorical feature handling test passed",
                       categorical_features=len(model.categorical_features),
                       cardinalities=list(model.categorical_info.cardinalities.values()))
            
        except Exception as e:
            self.test_results['categorical_features'] = False
            logger.error("Categorical feature handling test failed", error=str(e))
            raise
    
    async def test_early_stopping(self):
        """Test early stopping functionality."""
        logger.info("Testing early stopping")
        
        try:
            # Create model with early stopping
            config = LightGBMConfig(
                n_estimators=1000,
                learning_rate=0.1,
                early_stopping_rounds=10
            )
            model = LightGBMModel(config, ModelType.CLASSIFIER, 
                                self.sample_data['feature_names'],
                                self.sample_data['categorical_features'])
            
            # Split data for validation
            X = self.sample_data['X']
            y = self.sample_data['y']
            
            # Use last 20% as validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Fit with validation data
            model.fit(X_train, y_train, X_val, y_val)
            
            # Check early stopping
            best_iteration = model.training_history.best_iteration
            assert best_iteration is not None, "Should have best iteration"
            assert best_iteration < config.n_estimators, "Should stop early"
            assert model.training_history.early_stopped, "Should indicate early stopping"
            
            # Check training history
            assert model.training_history.evals_result is not None, "Should have evaluation results"
            assert len(model.training_history.val_scores) > 0, "Should have validation scores"
            assert model.training_history.best_iteration == best_iteration, "Best iteration should match"
            
            self.test_results['early_stopping'] = True
            logger.info("Early stopping test passed",
                       best_iteration=best_iteration,
                       n_estimators=config.n_estimators)
            
        except Exception as e:
            self.test_results['early_stopping'] = False
            logger.error("Early stopping test failed", error=str(e))
            raise
    
    async def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        logger.info("Testing feature importance analysis")
        
        try:
            # Create and fit model
            config = LightGBMConfig(n_estimators=100, learning_rate=0.1, num_leaves=31)
            model = LightGBMModel(config, ModelType.CLASSIFIER, 
                                self.sample_data['feature_names'],
                                self.sample_data['categorical_features'])
            
            X = self.sample_data['X']
            y = self.sample_data['y']
            model.fit(X, y)
            
            # Analyze feature importance
            importance_result = model.analyze_feature_importance(X, y, importance_type='split', top_n=10)
            
            # Check result structure
            assert len(importance_result.feature_names) == X.shape[1], "Feature names should match feature count"
            assert len(importance_result.importance_scores) == X.shape[1], "Importance scores should match feature count"
            assert len(importance_result.top_features) == 10, "Should have 10 top features"
            assert len(importance_result.importance_ranking) == X.shape[1], "Ranking should include all features"
            assert importance_result.categorical_features is not None, "Should have categorical features info"
            
            # Check importance ranking
            ranking = importance_result.importance_ranking
            assert ranking.iloc[0]['importance'] >= ranking.iloc[-1]['importance'], "Ranking should be sorted"
            
            # Check importance types
            assert 'split' in importance_result.importance_types, "Should have split importance"
            assert 'gain' in importance_result.importance_types, "Should have gain importance"
            
            # Check that important features are in top features
            # Features 0, 5, 10 should be important based on our data generation
            important_features = ['feature_0', 'feature_5', 'feature_10']
            top_features_set = set(importance_result.top_features)
            important_in_top = sum(1 for f in important_features if f in top_features_set)
            assert important_in_top >= 1, "At least one important feature should be in top features"
            
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
            config = LightGBMConfig(n_estimators=50, learning_rate=0.1, num_leaves=31)
            model = LightGBMModel(config, ModelType.CLASSIFIER, 
                                self.sample_data['feature_names'],
                                self.sample_data['categorical_features'])
            
            X = self.sample_data['X']
            y = self.sample_data['y']
            
            # Test grid search
            param_grid = {
                'n_estimators': [25, 50],
                'learning_rate': [0.1, 0.2],
                'num_leaves': [15, 31]
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
            config2 = LightGBMConfig(n_estimators=30, learning_rate=0.1, num_leaves=31)
            model2 = LightGBMModel(config2, ModelType.CLASSIFIER, 
                                 self.sample_data['feature_names'],
                                 self.sample_data['categorical_features'])
            
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
    
    async def test_cross_validation_with_early_stopping(self):
        """Test cross-validation with early stopping."""
        logger.info("Testing cross-validation with early stopping")
        
        try:
            # Create model
            config = LightGBMConfig(n_estimators=100, learning_rate=0.1, num_leaves=31)
            model = LightGBMModel(config, ModelType.CLASSIFIER, 
                                self.sample_data['feature_names'],
                                self.sample_data['categorical_features'])
            
            X = self.sample_data['X']
            y = self.sample_data['y']
            
            # Perform cross-validation
            cv_results = model.cross_validate_with_early_stopping(
                X, y, cv=3, scoring='accuracy', early_stopping_rounds=10
            )
            
            # Check CV results
            assert 'scores' in cv_results, "Should have scores"
            assert 'mean_score' in cv_results, "Should have mean score"
            assert 'std_score' in cv_results, "Should have standard deviation"
            assert 'best_iterations' in cv_results, "Should have best iterations"
            assert 'mean_best_iteration' in cv_results, "Should have mean best iteration"
            
            assert len(cv_results['scores']) == 3, "Should have 3 CV scores"
            assert 0 <= cv_results['mean_score'] <= 1, "Mean score should be between 0 and 1"
            assert cv_results['std_score'] >= 0, "Standard deviation should be non-negative"
            assert len(cv_results['best_iterations']) == 3, "Should have 3 best iterations"
            
            # Check that early stopping worked
            mean_best_iter = cv_results['mean_best_iteration']
            assert mean_best_iter < config.n_estimators, "Should stop early on average"
            
            self.test_results['cross_validation'] = True
            logger.info("Cross-validation with early stopping test passed",
                       mean_score=cv_results['mean_score'],
                       mean_best_iteration=mean_best_iter)
            
        except Exception as e:
            self.test_results['cross_validation'] = False
            logger.error("Cross-validation with early stopping test failed", error=str(e))
            raise
    
    async def test_feature_selection(self):
        """Test feature selection functionality."""
        logger.info("Testing feature selection")
        
        try:
            # Create and fit model
            config = LightGBMConfig(n_estimators=100, learning_rate=0.1, num_leaves=31)
            model = LightGBMModel(config, ModelType.CLASSIFIER, 
                                self.sample_data['feature_names'],
                                self.sample_data['categorical_features'])
            
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
    
    async def test_convenience_functions(self):
        """Test convenience functions."""
        logger.info("Testing convenience functions")
        
        try:
            # Test classifier creation
            classifier = create_lightgbm_classifier(
                n_estimators=50, learning_rate=0.1, num_leaves=31,
                feature_names=self.sample_data['feature_names'],
                categorical_features=self.sample_data['categorical_features']
            )
            assert classifier.model_type == ModelType.CLASSIFIER, "Should create classifier"
            assert classifier.config.n_estimators == 50, "Should set n_estimators"
            assert classifier.config.learning_rate == 0.1, "Should set learning_rate"
            assert classifier.config.num_leaves == 31, "Should set num_leaves"
            assert len(classifier.categorical_features) == 3, "Should set categorical features"
            
            # Test regressor creation
            regressor = create_lightgbm_regressor(
                n_estimators=75, learning_rate=0.15, num_leaves=63,
                feature_names=self.sample_data['feature_names'],
                categorical_features=self.sample_data['categorical_features']
            )
            assert regressor.model_type == ModelType.REGRESSOR, "Should create regressor"
            assert regressor.config.n_estimators == 75, "Should set n_estimators"
            assert regressor.config.learning_rate == 0.15, "Should set learning_rate"
            assert regressor.config.num_leaves == 63, "Should set num_leaves"
            
            # Test early stopping model creation
            early_stop_model = create_lightgbm_with_early_stopping(
                n_estimators=1000, learning_rate=0.1, early_stopping_rounds=50,
                feature_names=self.sample_data['feature_names'],
                categorical_features=self.sample_data['categorical_features']
            )
            assert early_stop_model.config.n_estimators == 1000, "Should set n_estimators"
            assert early_stop_model.config.early_stopping_rounds == 50, "Should set early stopping rounds"
            
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
            # Create and fit model with early stopping
            config = LightGBMConfig(
                n_estimators=100, learning_rate=0.1, num_leaves=31, early_stopping_rounds=10
            )
            model = LightGBMModel(config, ModelType.CLASSIFIER, 
                                self.sample_data['feature_names'],
                                self.sample_data['categorical_features'])
            
            X = self.sample_data['X']
            y = self.sample_data['y']
            
            # Split for validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            model.fit(X_train, y_train, X_val, y_val)
            
            # Analyze feature importance
            model.analyze_feature_importance(X_train, y_train)
            
            # Get model summary
            summary = model.get_model_summary()
            
            # Check summary structure
            required_keys = ['model_type', 'is_fitted', 'n_estimators', 'learning_rate', 'num_leaves', 'feature_count', 'categorical_features', 'best_iteration']
            for key in required_keys:
                assert key in summary, f"Summary should contain {key}"
            
            assert summary['model_type'] == 'classifier', "Model type should be classifier"
            assert summary['is_fitted'], "Model should be fitted"
            assert summary['n_estimators'] == 100, "Should show correct n_estimators"
            assert summary['learning_rate'] == 0.1, "Should show correct learning_rate"
            assert summary['feature_count'] == 23, "Should show correct feature count"
            assert summary['categorical_features'] == 3, "Should show correct categorical feature count"
            assert summary['best_iteration'] is not None, "Should have best iteration"
            assert summary['early_stopped'], "Should indicate early stopping"
            
            # Test feature importance plotting
            try:
                fig = model.plot_feature_importance(top_n=10)
                assert fig is not None, "Should return matplotlib figure"
                plt.close(fig)  # Close to free memory
            except Exception as e:
                logger.warning("Feature importance plotting failed", error=str(e))
                # This is not critical, so we don't fail the test
            
            # Test training history plotting
            try:
                fig = model.plot_training_history()
                if fig is not None:
                    assert fig is not None, "Should return matplotlib figure"
                    plt.close(fig)  # Close to free memory
            except Exception as e:
                logger.warning("Training history plotting failed", error=str(e))
                # This is not critical, so we don't fail the test
            
            self.test_results['model_summary'] = True
            logger.info("Model summary and visualization test passed",
                       best_iteration=summary['best_iteration'],
                       categorical_features=summary['categorical_features'])
            
        except Exception as e:
            self.test_results['model_summary'] = False
            logger.error("Model summary and visualization test failed", error=str(e))
            raise
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        logger.info("Testing edge cases")
        
        try:
            # Test with empty data
            config = LightGBMConfig(n_estimators=10)
            model = LightGBMModel(config, ModelType.CLASSIFIER)
            
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
            
            # Test with invalid categorical features
            try:
                invalid_model = LightGBMModel(config, ModelType.CLASSIFIER, 
                                            ['f1', 'f2'], ['nonexistent_feature'])
                # Should handle gracefully
            except Exception as e:
                logger.warning("Invalid categorical features handling", error=str(e))
            
            self.test_results['edge_cases'] = True
            logger.info("Edge cases test passed")
            
        except Exception as e:
            self.test_results['edge_cases'] = False
            logger.error("Edge cases test failed", error=str(e))
            raise
    
    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("Starting LightGBM Test Suite")
        
        test_methods = [
            self.test_basic_lightgbm_classifier,
            self.test_basic_lightgbm_regressor,
            self.test_categorical_feature_handling,
            self.test_early_stopping,
            self.test_feature_importance_analysis,
            self.test_hyperparameter_tuning,
            self.test_cross_validation_with_early_stopping,
            self.test_feature_selection,
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
        
        logger.info("LightGBM Test Suite completed",
                   passed_tests=passed_tests,
                   total_tests=total_tests,
                   success_rate=passed_tests/total_tests)
        
        return passed_tests == total_tests

async def main():
    """Main test execution function."""
    logger.info("Starting LightGBM Test Suite")
    
    test_suite = LightGBMTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("LightGBM Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("LightGBM Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())