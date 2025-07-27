#!/usr/bin/env python3
"""
Test script for the Model Evaluation Framework.
Tests accuracy metrics, financial metrics, risk metrics, cross-validation, and model comparison.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml_models.evaluation.model_evaluator import (
    AccuracyMetricsCalculator, FinancialMetricsCalculator, RiskMetricsCalculator,
    TimeSeriesCrossValidator, ModelComparisonFramework, ModelEvaluator,
    evaluate_single_model, compare_models
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import structlog

logger = structlog.get_logger()

class ModelEvaluatorTestSuite:
    """Comprehensive test suite for the model evaluation framework."""
    
    def __init__(self):
        self.test_results = {}
        self.sample_data = None
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        # Generate sample features and targets
        n_samples = 1000
        n_features = 10
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate targets (binary classification)
        y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        
        # Generate returns for financial metrics
        returns = np.random.normal(0.001, 0.02, n_samples)
        
        self.sample_data = {
            'X': X,
            'y': y,
            'returns': returns
        }
        
        logger.info("Sample data created", 
                   n_samples=n_samples,
                   n_features=n_features,
                   class_distribution=np.bincount(y))
    
    async def test_accuracy_metrics(self):
        """Test accuracy metrics calculation."""
        logger.info("Testing accuracy metrics")
        
        try:
            # Create sample predictions
            y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
            y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.7, 0.9, 0.1, 0.8])
            
            # Calculate metrics
            metrics = AccuracyMetricsCalculator.calculate_accuracy_metrics(y_true, y_pred, y_prob)
            
            # Basic assertions
            assert 'accuracy' in metrics, "Should have accuracy metric"
            assert 'precision' in metrics, "Should have precision metric"
            assert 'recall' in metrics, "Should have recall metric"
            assert 'f1_score' in metrics, "Should have F1 score metric"
            assert 'auc_roc' in metrics, "Should have AUC-ROC metric"
            
            # Check metric ranges
            assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"
            assert 0 <= metrics['precision'] <= 1, "Precision should be between 0 and 1"
            assert 0 <= metrics['recall'] <= 1, "Recall should be between 0 and 1"
            assert 0 <= metrics['f1_score'] <= 1, "F1 score should be between 0 and 1"
            assert 0 <= metrics['auc_roc'] <= 1, "AUC-ROC should be between 0 and 1"
            
            # Test confusion matrix
            cm = AccuracyMetricsCalculator.calculate_confusion_matrix(y_true, y_pred)
            assert cm.shape == (2, 2), "Confusion matrix should be 2x2 for binary classification"
            
            # Test classification report
            report = AccuracyMetricsCalculator.generate_classification_report(y_true, y_pred)
            assert isinstance(report, str), "Classification report should be a string"
            assert len(report) > 0, "Classification report should not be empty"
            
            # Test edge cases
            empty_metrics = AccuracyMetricsCalculator.calculate_accuracy_metrics(
                np.array([]), np.array([])
            )
            assert len(empty_metrics) > 0, "Should handle empty arrays"
            
            self.test_results['accuracy_metrics'] = True
            logger.info("Accuracy metrics test passed",
                       accuracy=metrics['accuracy'],
                       precision=metrics['precision'],
                       recall=metrics['recall'],
                       f1_score=metrics['f1_score'])
            
        except Exception as e:
            self.test_results['accuracy_metrics'] = False
            logger.error("Accuracy metrics test failed", error=str(e))
            raise
    
    async def test_financial_metrics(self):
        """Test financial metrics calculation."""
        logger.info("Testing financial metrics")
        
        try:
            # Create sample data
            y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
            returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, -0.01, 0.03])
            
            # Calculate financial metrics
            metrics = FinancialMetricsCalculator.calculate_financial_metrics(
                y_true, y_pred, returns, risk_free_rate=0.02
            )
            
            # Check required metrics
            required_metrics = [
                'strategy_total_return', 'buy_hold_total_return', 'excess_return',
                'strategy_annualized_return', 'buy_hold_annualized_return',
                'strategy_volatility', 'buy_hold_volatility', 'sharpe_ratio',
                'sortino_ratio', 'max_drawdown', 'calmar_ratio', 'win_rate', 'profit_factor'
            ]
            
            for metric in required_metrics:
                assert metric in metrics, f"Should have {metric} metric"
                assert isinstance(metrics[metric], (int, float)), f"{metric} should be numeric"
            
            # Check logical relationships
            assert metrics['excess_return'] == metrics['strategy_total_return'] - metrics['buy_hold_total_return'], \
                "Excess return should equal strategy return minus buy-hold return"
            
            assert 0 <= metrics['win_rate'] <= 1, "Win rate should be between 0 and 1"
            assert metrics['profit_factor'] >= 0, "Profit factor should be non-negative"
            
            # Test edge cases
            zero_returns = np.zeros_like(returns)
            zero_metrics = FinancialMetricsCalculator.calculate_financial_metrics(
                y_true, y_pred, zero_returns
            )
            assert len(zero_metrics) > 0, "Should handle zero returns"
            
            self.test_results['financial_metrics'] = True
            logger.info("Financial metrics test passed",
                       strategy_return=metrics['strategy_total_return'],
                       sharpe_ratio=metrics['sharpe_ratio'],
                       max_drawdown=metrics['max_drawdown'],
                       win_rate=metrics['win_rate'])
            
        except Exception as e:
            self.test_results['financial_metrics'] = False
            logger.error("Financial metrics test failed", error=str(e))
            raise
    
    async def test_risk_metrics(self):
        """Test risk metrics calculation."""
        logger.info("Testing risk metrics")
        
        try:
            # Create sample returns
            returns = np.random.normal(0.001, 0.02, 1000)
            
            # Calculate risk metrics
            metrics = RiskMetricsCalculator.calculate_risk_metrics(returns, confidence_level=0.05)
            
            # Check required metrics
            required_metrics = [
                'volatility', 'var', 'cvar', 'max_drawdown', 'downside_deviation',
                'skewness', 'kurtosis', 'ulcer_index'
            ]
            
            for metric in required_metrics:
                assert metric in metrics, f"Should have {metric} metric"
                assert isinstance(metrics[metric], (int, float)), f"{metric} should be numeric"
            
            # Check logical relationships
            assert metrics['volatility'] >= 0, "Volatility should be non-negative"
            assert metrics['cvar'] <= metrics['var'], "CVaR should be <= VaR"
            assert metrics['max_drawdown'] <= 0, "Maximum drawdown should be <= 0"
            assert metrics['downside_deviation'] >= 0, "Downside deviation should be non-negative"
            assert metrics['ulcer_index'] >= 0, "Ulcer index should be non-negative"
            
            # Test edge cases
            empty_metrics = RiskMetricsCalculator.calculate_risk_metrics(np.array([]))
            assert len(empty_metrics) == 0, "Should handle empty returns"
            
            constant_returns = np.ones(100) * 0.01
            constant_metrics = RiskMetricsCalculator.calculate_risk_metrics(constant_returns)
            assert constant_metrics['volatility'] == 0, "Constant returns should have zero volatility"
            
            self.test_results['risk_metrics'] = True
            logger.info("Risk metrics test passed",
                       volatility=metrics['volatility'],
                       var=metrics['var'],
                       cvar=metrics['cvar'],
                       max_drawdown=metrics['max_drawdown'])
            
        except Exception as e:
            self.test_results['risk_metrics'] = False
            logger.error("Risk metrics test failed", error=str(e))
            raise
    
    async def test_time_series_cross_validation(self):
        """Test time series cross-validation."""
        logger.info("Testing time series cross-validation")
        
        try:
            # Create sample data
            X = np.random.randn(100, 5)
            y = np.random.choice([0, 1], size=100)
            
            # Create a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            # Perform cross-validation
            cv_validator = TimeSeriesCrossValidator(n_splits=3)
            cv_scores = cv_validator.cross_validate(model, X, y, scoring='accuracy')
            
            # Check CV results
            assert 'scores' in cv_scores, "Should have scores"
            assert 'mean_score' in cv_scores, "Should have mean score"
            assert 'std_score' in cv_scores, "Should have standard deviation"
            assert 'min_score' in cv_scores, "Should have minimum score"
            assert 'max_score' in cv_scores, "Should have maximum score"
            
            assert len(cv_scores['scores']) == 3, "Should have 3 CV scores"
            assert 0 <= cv_scores['mean_score'] <= 1, "Mean score should be between 0 and 1"
            assert cv_scores['std_score'] >= 0, "Standard deviation should be non-negative"
            assert cv_scores['min_score'] <= cv_scores['max_score'], "Min should be <= max"
            
            # Test different scoring metrics
            cv_scores_f1 = cv_validator.cross_validate(model, X, y, scoring='f1')
            assert 'mean_score' in cv_scores_f1, "Should work with F1 scoring"
            
            self.test_results['time_series_cv'] = True
            logger.info("Time series cross-validation test passed",
                       mean_score=cv_scores['mean_score'],
                       std_score=cv_scores['std_score'],
                       n_splits=len(cv_scores['scores']))
            
        except Exception as e:
            self.test_results['time_series_cv'] = False
            logger.error("Time series cross-validation test failed", error=str(e))
            raise
    
    async def test_model_comparison_framework(self):
        """Test model comparison framework."""
        logger.info("Testing model comparison framework")
        
        try:
            # Create comparison framework
            framework = ModelComparisonFramework()
            
            # Create sample evaluation results
            from ml_models.evaluation.model_evaluator import ModelEvaluationResult
            
            result1 = ModelEvaluationResult(
                model_name="Model1",
                accuracy_metrics={'f1_score': 0.8, 'accuracy': 0.75},
                financial_metrics={'sharpe_ratio': 1.2, 'max_drawdown': -0.1}
            )
            
            result2 = ModelEvaluationResult(
                model_name="Model2",
                accuracy_metrics={'f1_score': 0.85, 'accuracy': 0.8},
                financial_metrics={'sharpe_ratio': 1.5, 'max_drawdown': -0.08}
            )
            
            # Add results to framework
            framework.add_model_result("Model1", result1)
            framework.add_model_result("Model2", result2)
            
            # Compare models
            comparison_df = framework.compare_models('f1_score')
            assert len(comparison_df) == 2, "Should have 2 models in comparison"
            assert 'model_name' in comparison_df.columns, "Should have model_name column"
            
            # Check sorting
            best_model = framework.get_best_model('f1_score')
            assert best_model == "Model2", "Model2 should be best by F1 score"
            
            # Generate comparison report
            report = framework.generate_comparison_report()
            assert isinstance(report, str), "Report should be a string"
            assert len(report) > 0, "Report should not be empty"
            assert "Model Comparison Report" in report, "Report should have title"
            
            # Test empty framework
            empty_framework = ModelComparisonFramework()
            empty_comparison = empty_framework.compare_models()
            assert len(empty_comparison) == 0, "Empty framework should return empty comparison"
            
            self.test_results['model_comparison'] = True
            logger.info("Model comparison framework test passed",
                       n_models=len(comparison_df),
                       best_model=best_model)
            
        except Exception as e:
            self.test_results['model_comparison'] = False
            logger.error("Model comparison framework test failed", error=str(e))
            raise
    
    async def test_model_evaluator(self):
        """Test the main ModelEvaluator class."""
        logger.info("Testing ModelEvaluator")
        
        try:
            # Create evaluator
            evaluator = ModelEvaluator(risk_free_rate=0.02, confidence_level=0.05)
            
            # Create sample data
            X = self.sample_data['X']
            y = self.sample_data['y']
            returns = self.sample_data['returns']
            
            # Create and fit a model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Evaluate model
            result = evaluator.evaluate_model(
                model, X, y, returns, "TestModel", perform_cv=True
            )
            
            # Check result structure
            assert result.model_name == "TestModel", "Model name should match"
            assert len(result.accuracy_metrics) > 0, "Should have accuracy metrics"
            assert len(result.financial_metrics) > 0, "Should have financial metrics"
            assert len(result.risk_metrics) > 0, "Should have risk metrics"
            assert len(result.cross_validation_scores) > 0, "Should have CV scores"
            
            # Check that predictions were made
            assert len(result.predictions) == len(y), "Should have predictions for all samples"
            
            # Check feature importance
            assert result.feature_importance is not None, "Should have feature importance"
            assert len(result.feature_importance) == X.shape[1], "Feature importance should match feature count"
            
            # Test model comparison
            comparison_df = evaluator.compare_all_models('f1_score')
            assert len(comparison_df) == 1, "Should have one model in comparison"
            
            best_model = evaluator.get_best_model('f1_score')
            assert best_model == "TestModel", "TestModel should be the best model"
            
            # Generate comparison report
            report = evaluator.generate_comparison_report()
            assert isinstance(report, str), "Report should be a string"
            assert len(report) > 0, "Report should not be empty"
            
            self.test_results['model_evaluator'] = True
            logger.info("ModelEvaluator test passed",
                       accuracy=result.accuracy_metrics.get('accuracy', 0),
                       f1_score=result.accuracy_metrics.get('f1_score', 0),
                       sharpe_ratio=result.financial_metrics.get('sharpe_ratio', 0))
            
        except Exception as e:
            self.test_results['model_evaluator'] = False
            logger.error("ModelEvaluator test failed", error=str(e))
            raise
    
    async def test_convenience_functions(self):
        """Test convenience functions."""
        logger.info("Testing convenience functions")
        
        try:
            # Create sample data
            X = self.sample_data['X']
            y = self.sample_data['y']
            returns = self.sample_data['returns']
            
            # Test evaluate_single_model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            result = evaluate_single_model(model, X, y, returns, "SingleModel")
            assert result.model_name == "SingleModel", "Model name should match"
            assert len(result.accuracy_metrics) > 0, "Should have accuracy metrics"
            
            # Test compare_models
            models = {
                "RF": RandomForestClassifier(n_estimators=10, random_state=42),
                "LR": LogisticRegression(random_state=42)
            }
            
            # Fit models
            for name, model in models.items():
                model.fit(X, y)
            
            comparison_df = compare_models(models, X, y, returns)
            assert len(comparison_df) == 2, "Should have 2 models in comparison"
            assert 'model_name' in comparison_df.columns, "Should have model_name column"
            
            self.test_results['convenience_functions'] = True
            logger.info("Convenience functions test passed",
                       single_model_result=result.model_name,
                       n_models_comparison=len(comparison_df))
            
        except Exception as e:
            self.test_results['convenience_functions'] = False
            logger.error("Convenience functions test failed", error=str(e))
            raise
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        logger.info("Testing edge cases")
        
        try:
            # Test with empty data
            empty_result = evaluate_single_model(
                RandomForestClassifier(), np.array([]), np.array([])
            )
            assert empty_result.model_name == "Model", "Should handle empty data"
            
            # Test with single sample
            single_result = evaluate_single_model(
                RandomForestClassifier(), np.array([[1, 2, 3]]), np.array([1])
            )
            assert single_result.model_name == "Model", "Should handle single sample"
            
            # Test with mismatched lengths
            try:
                mismatched_result = evaluate_single_model(
                    RandomForestClassifier(), np.array([[1, 2, 3]]), np.array([1, 0])
                )
                # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected behavior
            
            # Test with invalid model
            try:
                invalid_result = evaluate_single_model(
                    "not_a_model", np.array([[1, 2, 3]]), np.array([1])
                )
                # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected behavior
            
            self.test_results['edge_cases'] = True
            logger.info("Edge cases test passed")
            
        except Exception as e:
            self.test_results['edge_cases'] = False
            logger.error("Edge cases test failed", error=str(e))
            raise
    
    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("Starting Model Evaluator Test Suite")
        
        test_methods = [
            self.test_accuracy_metrics,
            self.test_financial_metrics,
            self.test_risk_metrics,
            self.test_time_series_cross_validation,
            self.test_model_comparison_framework,
            self.test_model_evaluator,
            self.test_convenience_functions,
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
        
        logger.info("Model Evaluator Test Suite completed",
                   passed_tests=passed_tests,
                   total_tests=total_tests,
                   success_rate=passed_tests/total_tests)
        
        return passed_tests == total_tests

async def main():
    """Main test execution function."""
    logger.info("Starting Model Evaluator Test Suite")
    
    test_suite = ModelEvaluatorTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Model Evaluator Test Suite completed successfully")
        sys.exit(0)
    else:
        logger.error("Model Evaluator Test Suite failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())