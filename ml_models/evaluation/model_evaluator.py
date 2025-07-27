"""
Model Evaluation Framework

A comprehensive evaluation framework for ML models in stock prediction.
Implements accuracy metrics, financial metrics, risk metrics, and model comparison tools.

Features:
- Accuracy metrics (precision, recall, F1, confusion matrix)
- Financial metrics (Sharpe ratio, Sortino ratio, returns)
- Risk metrics (max drawdown, VaR, volatility)
- Time series cross-validation
- Model comparison and ranking
- Performance visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = structlog.get_logger()

class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    FINANCIAL = "financial"
    RISK = "risk"
    TIME_SERIES = "time_series"

@dataclass
class ModelEvaluationResult:
    """Container for model evaluation results."""
    model_name: str
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    financial_metrics: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    time_series_metrics: Dict[str, float] = field(default_factory=dict)
    cross_validation_scores: Dict[str, List[float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_importance: Optional[pd.Series] = None
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)

class AccuracyMetricsCalculator:
    """Calculate accuracy-based metrics for classification models."""
    
    @staticmethod
    def calculate_accuracy_metrics(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
        
        Returns:
            Dictionary of accuracy metrics
        """
        metrics = {}
        
        # Basic accuracy metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        try:
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        except Exception as e:
            logger.warning("Could not calculate macro metrics", error=str(e))
            metrics['precision_macro'] = 0.0
            metrics['recall_macro'] = 0.0
            metrics['f1_macro'] = 0.0
        
        # AUC-ROC if probabilities are available
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            except Exception as e:
                logger.warning("Could not calculate AUC-ROC", error=str(e))
                metrics['auc_roc'] = 0.0
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics['true_positives'] = cm[1, 1] if cm.shape == (2, 2) else 0
            metrics['true_negatives'] = cm[0, 0] if cm.shape == (2, 2) else 0
            metrics['false_positives'] = cm[0, 1] if cm.shape == (2, 2) else 0
            metrics['false_negatives'] = cm[1, 0] if cm.shape == (2, 2) else 0
        except Exception as e:
            logger.warning("Could not calculate confusion matrix metrics", error=str(e))
            metrics['true_positives'] = 0
            metrics['true_negatives'] = 0
            metrics['false_positives'] = 0
            metrics['false_negatives'] = 0
        
        return metrics
    
    @staticmethod
    def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        try:
            return confusion_matrix(y_true, y_pred)
        except Exception as e:
            logger.warning("Could not calculate confusion matrix", error=str(e))
            return np.array([])
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report."""
        try:
            return classification_report(y_true, y_pred, zero_division=0)
        except Exception as e:
            logger.warning("Could not generate classification report", error=str(e))
            return "Classification report not available"

class FinancialMetricsCalculator:
    """Calculate financial performance metrics."""
    
    @staticmethod
    def calculate_financial_metrics(y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  returns: np.ndarray,
                                  risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate financial performance metrics.
        
        Args:
            y_true: True labels (1 for positive, 0 for negative)
            y_pred: Predicted labels
            returns: Actual returns for each prediction
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        
        Returns:
            Dictionary of financial metrics
        """
        metrics = {}
        
        if len(y_true) != len(y_pred) or len(y_pred) != len(returns):
            logger.error("Length mismatch in financial metrics calculation")
            return metrics
        
        # Strategy returns (only trade when prediction is positive)
        strategy_returns = returns * (y_pred == 1)
        
        # Buy and hold returns
        buy_hold_returns = returns
        
        # Calculate cumulative returns
        strategy_cumulative = np.cumprod(1 + strategy_returns)
        buy_hold_cumulative = np.cumprod(1 + buy_hold_returns)
        
        # Total returns
        metrics['strategy_total_return'] = strategy_cumulative[-1] - 1 if len(strategy_cumulative) > 0 else 0
        metrics['buy_hold_total_return'] = buy_hold_cumulative[-1] - 1 if len(buy_hold_cumulative) > 0 else 0
        metrics['excess_return'] = metrics['strategy_total_return'] - metrics['buy_hold_total_return']
        
        # Annualized returns (assuming daily data)
        if len(strategy_returns) > 0:
            metrics['strategy_annualized_return'] = (1 + metrics['strategy_total_return']) ** (252 / len(strategy_returns)) - 1
            metrics['buy_hold_annualized_return'] = (1 + metrics['buy_hold_total_return']) ** (252 / len(buy_hold_returns)) - 1
        else:
            metrics['strategy_annualized_return'] = 0
            metrics['buy_hold_annualized_return'] = 0
        
        # Volatility
        if len(strategy_returns) > 1:
            metrics['strategy_volatility'] = np.std(strategy_returns) * np.sqrt(252)
            metrics['buy_hold_volatility'] = np.std(buy_hold_returns) * np.sqrt(252)
        else:
            metrics['strategy_volatility'] = 0
            metrics['buy_hold_volatility'] = 0
        
        # Sharpe ratio
        if metrics['strategy_volatility'] > 0:
            excess_return = metrics['strategy_annualized_return'] - risk_free_rate
            metrics['sharpe_ratio'] = excess_return / metrics['strategy_volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            if downside_deviation > 0:
                excess_return = metrics['strategy_annualized_return'] - risk_free_rate
                metrics['sortino_ratio'] = excess_return / downside_deviation
            else:
                metrics['sortino_ratio'] = 0
        else:
            metrics['sortino_ratio'] = 0
        
        # Maximum drawdown
        if len(strategy_cumulative) > 0:
            running_max = np.maximum.accumulate(strategy_cumulative)
            drawdown = (strategy_cumulative - running_max) / running_max
            metrics['max_drawdown'] = np.min(drawdown)
        else:
            metrics['max_drawdown'] = 0
        
        # Calmar ratio
        if abs(metrics['max_drawdown']) > 0:
            metrics['calmar_ratio'] = metrics['strategy_annualized_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
        
        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        total_trades = np.sum(y_pred == 1)
        metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return metrics

class RiskMetricsCalculator:
    """Calculate risk metrics for trading strategies."""
    
    @staticmethod
    def calculate_risk_metrics(returns: np.ndarray, 
                             confidence_level: float = 0.05) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Strategy returns
            confidence_level: Confidence level for VaR/CVaR
        
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        if len(returns) == 0:
            return metrics
        
        # Volatility
        metrics['volatility'] = np.std(returns) * np.sqrt(252)
        
        # Value at Risk (VaR)
        metrics['var'] = np.percentile(returns, confidence_level * 100)
        
        # Conditional Value at Risk (CVaR)
        var_threshold = metrics['var']
        tail_returns = returns[returns <= var_threshold]
        metrics['cvar'] = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            metrics['downside_deviation'] = np.std(downside_returns) * np.sqrt(252)
        else:
            metrics['downside_deviation'] = 0
        
        # Skewness and kurtosis
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        
        # Ulcer Index
        if len(drawdown) > 0:
            metrics['ulcer_index'] = np.sqrt(np.mean(drawdown ** 2))
        else:
            metrics['ulcer_index'] = 0
        
        return metrics

class TimeSeriesCrossValidator:
    """Implement time series cross-validation."""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    def cross_validate(self, 
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      scoring: str = 'accuracy') -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            model: ML model with fit/predict methods
            X: Feature matrix
            y: Target variable
            scoring: Scoring metric
        
        Returns:
            Dictionary with cross-validation scores
        """
        scores = []
        
        for train_idx, test_idx in self.tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate score
                if scoring == 'accuracy':
                    score = accuracy_score(y_test, y_pred)
                elif scoring == 'precision':
                    score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                elif scoring == 'recall':
                    score = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                elif scoring == 'f1':
                    score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                else:
                    score = accuracy_score(y_test, y_pred)
                
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Error in cross-validation fold: {str(e)}")
                scores.append(0.0)
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }

class ModelComparisonFramework:
    """Framework for comparing multiple models."""
    
    def __init__(self):
        self.results = {}
        self.comparison_metrics = {}
    
    def add_model_result(self, model_name: str, result: ModelEvaluationResult):
        """Add model evaluation result to comparison."""
        self.results[model_name] = result
    
    def compare_models(self, metric: str = 'f1_score') -> pd.DataFrame:
        """
        Compare models based on specified metric.
        
        Args:
            metric: Metric to compare on
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, result in self.results.items():
            row = {'model_name': model_name}
            
            # Add accuracy metrics
            for key, value in result.accuracy_metrics.items():
                row[f'accuracy_{key}'] = value
            
            # Add financial metrics
            for key, value in result.financial_metrics.items():
                row[f'financial_{key}'] = value
            
            # Add risk metrics
            for key, value in result.risk_metrics.items():
                row[f'risk_{key}'] = value
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric
        metric_col = None
        for col in df.columns:
            if metric in col:
                metric_col = col
                break
        
        if metric_col and metric_col in df.columns:
            df = df.sort_values(metric_col, ascending=False)
        
        return df
    
    def get_best_model(self, metric: str = 'f1_score') -> str:
        """Get the best performing model based on metric."""
        comparison_df = self.compare_models(metric)
        
        if len(comparison_df) == 0:
            return ""
        
        return comparison_df.iloc[0]['model_name']
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report."""
        if not self.results:
            return "No models to compare"
        
        report = "Model Comparison Report\n"
        report += "=" * 50 + "\n\n"
        
        # Compare on different metrics
        metrics_to_compare = ['f1_score', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
        
        for metric in metrics_to_compare:
            comparison_df = self.compare_models(metric)
            if len(comparison_df) > 0:
                report += f"Top 3 models by {metric}:\n"
                for i in range(min(3, len(comparison_df))):
                    model_name = comparison_df.iloc[i]['model_name']
                    metric_col = None
                    for col in comparison_df.columns:
                        if metric in col:
                            metric_col = col
                            break
                    if metric_col:
                        value = comparison_df.iloc[i][metric_col]
                        report += f"  {i+1}. {model_name}: {value:.4f}\n"
                report += "\n"
        
        return report

class ModelEvaluator:
    """Main model evaluation framework."""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.comparison_framework = ModelComparisonFramework()
    
    def evaluate_model(self, 
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      returns: Optional[np.ndarray] = None,
                      model_name: str = "Model",
                      perform_cv: bool = True) -> ModelEvaluationResult:
        """
        Comprehensive model evaluation.
        
        Args:
            model: ML model to evaluate
            X: Feature matrix
            y: Target variable
            returns: Returns for financial metrics (optional)
            model_name: Name of the model
            perform_cv: Whether to perform cross-validation
        
        Returns:
            ModelEvaluationResult object
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        try:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        except Exception as e:
            logger.error(f"Error making predictions for {model_name}: {str(e)}")
            return ModelEvaluationResult(model_name=model_name)
        
        # Calculate accuracy metrics
        accuracy_metrics = AccuracyMetricsCalculator.calculate_accuracy_metrics(y, y_pred, y_prob)
        
        # Calculate confusion matrix
        confusion_matrix = AccuracyMetricsCalculator.calculate_confusion_matrix(y, y_pred)
        
        # Calculate financial metrics if returns provided
        financial_metrics = {}
        if returns is not None and len(returns) == len(y):
            financial_metrics = FinancialMetricsCalculator.calculate_financial_metrics(
                y, y_pred, returns, self.risk_free_rate
            )
        
        # Calculate risk metrics if returns provided
        risk_metrics = {}
        if returns is not None:
            risk_metrics = RiskMetricsCalculator.calculate_risk_metrics(
                returns, self.confidence_level
            )
        
        # Perform cross-validation
        cv_scores = {}
        if perform_cv:
            cv_validator = TimeSeriesCrossValidator()
            cv_scores = cv_validator.cross_validate(model, X, y)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            feature_importance = pd.Series(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        
        # Create result object
        result = ModelEvaluationResult(
            model_name=model_name,
            accuracy_metrics=accuracy_metrics,
            financial_metrics=financial_metrics,
            risk_metrics=risk_metrics,
            cross_validation_scores=cv_scores,
            confusion_matrix=confusion_matrix,
            feature_importance=feature_importance,
            predictions=y_pred,
            probabilities=y_prob,
            metadata={
                'risk_free_rate': self.risk_free_rate,
                'confidence_level': self.confidence_level,
                'evaluation_date': datetime.now().isoformat()
            }
        )
        
        # Add to comparison framework
        self.comparison_framework.add_model_result(model_name, result)
        
        logger.info(f"Model evaluation completed for {model_name}")
        
        return result
    
    def compare_all_models(self, metric: str = 'f1_score') -> pd.DataFrame:
        """Compare all evaluated models."""
        return self.comparison_framework.compare_models(metric)
    
    def get_best_model(self, metric: str = 'f1_score') -> str:
        """Get the best performing model."""
        return self.comparison_framework.get_best_model(metric)
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report."""
        return self.comparison_framework.generate_comparison_report()

# Convenience functions
def evaluate_single_model(model: Any,
                         X: np.ndarray,
                         y: np.ndarray,
                         returns: Optional[np.ndarray] = None,
                         model_name: str = "Model") -> ModelEvaluationResult:
    """Evaluate a single model with default settings."""
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, X, y, returns, model_name)

def compare_models(models: Dict[str, Any],
                  X: np.ndarray,
                  y: np.ndarray,
                  returns: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Compare multiple models."""
    evaluator = ModelEvaluator()
    
    for name, model in models.items():
        evaluator.evaluate_model(model, X, y, returns, name)
    
    return evaluator.compare_all_models()