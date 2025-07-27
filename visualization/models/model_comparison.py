#!/usr/bin/env python3
"""
Model Comparison Interfaces Module

Implements comprehensive model comparison interfaces:
- Model performance comparison
- Ensemble model visualization
- Feature importance comparison
- Prediction accuracy charts
- Model selection interfaces
- Model validation displays

Features:
- Comprehensive model performance comparison
- Ensemble model visualization and analysis
- Feature importance comparison across models
- Prediction accuracy and error analysis
- Interactive model selection interfaces
- Model validation and robustness displays
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

logger = structlog.get_logger()

class ModelType(Enum):
    """Model types for comparison."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"

class ComparisonMetric(Enum):
    """Comparison metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    model_type: ModelType
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    training_time: float = 0.0
    prediction_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ModelPrediction:
    """Model prediction results."""
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)

class ModelPerformanceComparison:
    """Model performance comparison component."""
    
    def __init__(self):
        """Initialize model performance comparison."""
        self.model_performances = {}
        self.model_predictions = {}
    
    def add_model_performance(self, performance: ModelPerformance):
        """Add model performance data."""
        self.model_performances[performance.model_name] = performance
    
    def add_model_prediction(self, prediction: ModelPrediction):
        """Add model prediction data."""
        self.model_predictions[prediction.model_name] = prediction
    
    def calculate_performance_metrics(self, model_name: str, y_true: np.ndarray, 
                                    y_pred: np.ndarray, y_prob: np.ndarray = None) -> ModelPerformance:
        """Calculate comprehensive performance metrics."""
        # Determine if classification or regression
        unique_values = len(np.unique(y_true))
        is_classification = unique_values <= 10
        
        performance = ModelPerformance(model_name=model_name)
        
        if is_classification:
            performance.model_type = ModelType.CLASSIFICATION
            performance.accuracy = accuracy_score(y_true, y_pred)
            performance.precision = precision_score(y_true, y_pred, average='weighted')
            performance.recall = recall_score(y_true, y_pred, average='weighted')
            performance.f1_score = f1_score(y_true, y_pred, average='weighted')
        else:
            performance.model_type = ModelType.REGRESSION
            performance.mse = mean_squared_error(y_true, y_pred)
            performance.mae = mean_absolute_error(y_true, y_pred)
            performance.r2_score = r2_score(y_true, y_pred)
        
        return performance
    
    def create_performance_comparison_chart(self) -> go.Figure:
        """Create model performance comparison chart."""
        if not self.model_performances:
            return go.Figure()
        
        model_names = list(self.model_performances.keys())
        
        # Determine metrics to compare based on model types
        classification_models = [name for name, perf in self.model_performances.items() 
                               if perf.model_type == ModelType.CLASSIFICATION]
        regression_models = [name for name, perf in self.model_performances.items() 
                           if perf.model_type == ModelType.REGRESSION]
        
        if classification_models and regression_models:
            # Mixed model types - create separate charts
            return self._create_mixed_performance_chart()
        elif classification_models:
            return self._create_classification_performance_chart()
        else:
            return self._create_regression_performance_chart()
    
    def _create_classification_performance_chart(self) -> go.Figure:
        """Create classification performance comparison chart."""
        model_names = [name for name, perf in self.model_performances.items() 
                      if perf.model_type == ModelType.CLASSIFICATION]
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = {metric: [getattr(self.model_performances[name], metric) 
                                 for name in model_names] for metric in metrics}
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(x=model_names, y=metric_values[metric], name=metric.title(),
                       marker_color=colors[i % len(colors)]),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Classification Model Performance Comparison",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _create_regression_performance_chart(self) -> go.Figure:
        """Create regression performance comparison chart."""
        model_names = [name for name, perf in self.model_performances.items() 
                      if perf.model_type == ModelType.REGRESSION]
        
        metrics = ['mse', 'mae', 'r2_score']
        metric_values = {metric: [getattr(self.model_performances[name], metric) 
                                 for name in model_names] for metric in metrics}
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Mean Squared Error', 'Mean Absolute Error', 'R² Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['red', 'orange', 'green']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(x=model_names, y=metric_values[metric], name=metric.upper(),
                       marker_color=colors[i]),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Regression Model Performance Comparison",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_mixed_performance_chart(self) -> go.Figure:
        """Create mixed model type performance comparison chart."""
        classification_models = [name for name, perf in self.model_performances.items() 
                               if perf.model_type == ModelType.CLASSIFICATION]
        regression_models = [name for name, perf in self.model_performances.items() 
                           if perf.model_type == ModelType.REGRESSION]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Classification Models', 'Regression Models'),
            vertical_spacing=0.1
        )
        
        # Classification models
        if classification_models:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for i, metric in enumerate(metrics):
                values = [getattr(self.model_performances[name], metric) 
                         for name in classification_models]
                fig.add_trace(
                    go.Bar(x=classification_models, y=values, name=metric.title()),
                    row=1, col=1
                )
        
        # Regression models
        if regression_models:
            metrics = ['mse', 'mae', 'r2_score']
            for i, metric in enumerate(metrics):
                values = [getattr(self.model_performances[name], metric) 
                         for name in regression_models]
                fig.add_trace(
                    go.Bar(x=regression_models, y=values, name=metric.upper()),
                    row=2, col=1
                )
        
        fig.update_layout(
            title="Mixed Model Performance Comparison",
            height=800,
            showlegend=True
        )
        
        return fig

class EnsembleModelVisualization:
    """Ensemble model visualization component."""
    
    def __init__(self):
        """Initialize ensemble model visualization."""
        self.ensemble_data = {}
        self.base_model_performances = {}
    
    def add_ensemble_data(self, ensemble_name: str, base_models: List[str],
                         ensemble_predictions: np.ndarray, base_predictions: Dict[str, np.ndarray],
                         y_true: np.ndarray):
        """Add ensemble model data."""
        self.ensemble_data[ensemble_name] = {
            'base_models': base_models,
            'ensemble_predictions': ensemble_predictions,
            'base_predictions': base_predictions,
            'y_true': y_true
        }
    
    def add_base_model_performance(self, model_name: str, performance: ModelPerformance):
        """Add base model performance."""
        self.base_model_performances[model_name] = performance
    
    def create_ensemble_comparison_chart(self, ensemble_name: str) -> go.Figure:
        """Create ensemble vs base models comparison chart."""
        if ensemble_name not in self.ensemble_data:
            return go.Figure()
        
        data = self.ensemble_data[ensemble_name]
        base_models = data['base_models']
        
        # Calculate accuracies
        ensemble_accuracy = accuracy_score(data['y_true'], data['ensemble_predictions'])
        base_accuracies = []
        
        for model in base_models:
            if model in data['base_predictions']:
                acc = accuracy_score(data['y_true'], data['base_predictions'][model])
                base_accuracies.append(acc)
            else:
                base_accuracies.append(0.0)
        
        # Create comparison chart
        fig = go.Figure()
        
        # Base model accuracies
        fig.add_trace(go.Bar(
            x=base_models,
            y=base_accuracies,
            name='Base Models',
            marker_color='lightblue'
        ))
        
        # Ensemble accuracy
        fig.add_trace(go.Bar(
            x=['Ensemble'],
            y=[ensemble_accuracy],
            name='Ensemble',
            marker_color='red'
        ))
        
        fig.update_layout(
            title=f"Ensemble vs Base Models - {ensemble_name}",
            xaxis_title="Models",
            yaxis_title="Accuracy",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_ensemble_agreement_chart(self, ensemble_name: str) -> go.Figure:
        """Create ensemble agreement visualization."""
        if ensemble_name not in self.ensemble_data:
            return go.Figure()
        
        data = self.ensemble_data[ensemble_name]
        base_predictions = data['base_predictions']
        
        # Calculate agreement matrix
        n_samples = len(data['y_true'])
        n_models = len(base_predictions)
        agreement_matrix = np.zeros((n_samples, n_models))
        
        for i, (model_name, predictions) in enumerate(base_predictions.items()):
            agreement_matrix[:, i] = predictions
        
        # Calculate agreement score for each sample
        agreement_scores = []
        for i in range(n_samples):
            predictions = agreement_matrix[i, :]
            # Count most common prediction
            unique, counts = np.unique(predictions, return_counts=True)
            max_count = np.max(counts)
            agreement_score = max_count / n_models
            agreement_scores.append(agreement_score)
        
        # Create agreement distribution chart
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=agreement_scores,
            nbinsx=20,
            name='Agreement Distribution',
            marker_color='green'
        ))
        
        fig.update_layout(
            title=f"Ensemble Agreement Distribution - {ensemble_name}",
            xaxis_title="Agreement Score",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig

class FeatureImportanceComparison:
    """Feature importance comparison component."""
    
    def __init__(self):
        """Initialize feature importance comparison."""
        self.feature_importances = {}
    
    def add_feature_importance(self, model_name: str, feature_names: List[str], 
                              importance_scores: np.ndarray):
        """Add feature importance data for a model."""
        self.feature_importances[model_name] = {
            'feature_names': feature_names,
            'importance_scores': importance_scores
        }
    
    def create_feature_importance_comparison(self, top_n: int = 10) -> go.Figure:
        """Create feature importance comparison chart."""
        if not self.feature_importances:
            return go.Figure()
        
        model_names = list(self.feature_importances.keys())
        
        # Get common features
        all_features = set()
        for data in self.feature_importances.values():
            all_features.update(data['feature_names'])
        
        common_features = sorted(list(all_features))
        
        # Create comparison matrix
        comparison_matrix = np.zeros((len(common_features), len(model_names)))
        
        for i, model_name in enumerate(model_names):
            data = self.feature_importances[model_name]
            for j, feature in enumerate(common_features):
                if feature in data['feature_names']:
                    idx = data['feature_names'].index(feature)
                    comparison_matrix[j, i] = data['importance_scores'][idx]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=comparison_matrix,
            x=model_names,
            y=common_features,
            colorscale='Viridis',
            text=np.round(comparison_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Importance Comparison Across Models",
            height=max(400, len(common_features) * 20),
            width=max(600, len(model_names) * 100)
        )
        
        return fig
    
    def create_top_features_chart(self, top_n: int = 10) -> go.Figure:
        """Create top features comparison chart."""
        if not self.feature_importances:
            return go.Figure()
        
        model_names = list(self.feature_importances.keys())
        
        # Get top features for each model
        top_features_data = {}
        for model_name in model_names:
            data = self.feature_importances[model_name]
            # Sort by importance and get top N
            sorted_indices = np.argsort(data['importance_scores'])[::-1]
            top_indices = sorted_indices[:top_n]
            
            top_features_data[model_name] = {
                'features': [data['feature_names'][i] for i in top_indices],
                'scores': [data['importance_scores'][i] for i in top_indices]
            }
        
        # Create subplot for each model
        n_models = len(model_names)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=model_names,
            specs=[[{"type": "bar"}] * n_models]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, model_name in enumerate(model_names):
            data = top_features_data[model_name]
            
            fig.add_trace(
                go.Bar(x=data['features'], y=data['scores'],
                       name=model_name, marker_color=colors[i % len(colors)]),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=f"Top {top_n} Features by Model",
            height=400,
            showlegend=False
        )
        
        return fig

class PredictionAccuracyCharts:
    """Prediction accuracy charts component."""
    
    def __init__(self):
        """Initialize prediction accuracy charts."""
        self.prediction_data = {}
    
    def add_prediction_data(self, model_name: str, prediction: ModelPrediction):
        """Add prediction data for a model."""
        self.prediction_data[model_name] = prediction
    
    def create_confusion_matrix_chart(self, model_name: str) -> go.Figure:
        """Create confusion matrix chart for classification."""
        if model_name not in self.prediction_data:
            return go.Figure()
        
        prediction = self.prediction_data[model_name]
        
        # Calculate confusion matrix
        cm = confusion_matrix(prediction.y_true, prediction.y_pred)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            height=400
        )
        
        return fig
    
    def create_prediction_vs_actual_chart(self, model_name: str) -> go.Figure:
        """Create prediction vs actual chart."""
        if model_name not in self.prediction_data:
            return go.Figure()
        
        prediction = self.prediction_data[model_name]
        
        # Determine if classification or regression
        unique_values = len(np.unique(prediction.y_true))
        is_classification = unique_values <= 10
        
        if is_classification:
            return self._create_classification_prediction_chart(model_name)
        else:
            return self._create_regression_prediction_chart(model_name)
    
    def _create_classification_prediction_chart(self, model_name: str) -> go.Figure:
        """Create classification prediction chart."""
        prediction = self.prediction_data[model_name]
        
        # Calculate accuracy over time
        window_size = 50
        accuracies = []
        for i in range(window_size, len(prediction.y_true)):
            window_true = prediction.y_true[i-window_size:i]
            window_pred = prediction.y_pred[i-window_size:i]
            acc = accuracy_score(window_true, window_pred)
            accuracies.append(acc)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=accuracies,
            mode='lines',
            name='Rolling Accuracy',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title=f"Prediction Accuracy Over Time - {model_name}",
            xaxis_title="Time Window",
            yaxis_title="Accuracy",
            height=400
        )
        
        return fig
    
    def _create_regression_prediction_chart(self, model_name: str) -> go.Figure:
        """Create regression prediction chart."""
        prediction = self.prediction_data[model_name]
        
        fig = go.Figure()
        
        # Scatter plot of predicted vs actual
        fig.add_trace(go.Scatter(
            x=prediction.y_true,
            y=prediction.y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(size=5, color='blue', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(np.min(prediction.y_true), np.min(prediction.y_pred))
        max_val = max(np.max(prediction.y_true), np.max(prediction.y_pred))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"Predicted vs Actual - {model_name}",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=400
        )
        
        return fig

class ModelSelectionInterface:
    """Model selection interface component."""
    
    def __init__(self):
        """Initialize model selection interface."""
        self.available_models = {}
        self.selected_models = set()
        self.selection_criteria = {}
    
    def add_model(self, model_name: str, model_type: ModelType, 
                 performance: ModelPerformance):
        """Add model to selection interface."""
        self.available_models[model_name] = {
            'type': model_type,
            'performance': performance
        }
    
    def set_selection_criteria(self, criteria: Dict[str, float]):
        """Set model selection criteria."""
        self.selection_criteria = criteria
    
    def select_models_by_criteria(self) -> List[str]:
        """Select models based on criteria."""
        selected = []
        
        for model_name, model_data in self.available_models.items():
            performance = model_data['performance']
            meets_criteria = True
            
            for criterion, threshold in self.selection_criteria.items():
                if hasattr(performance, criterion):
                    value = getattr(performance, criterion)
                    if criterion in ['mse', 'mae', 'max_drawdown']:
                        # Lower is better
                        if value > threshold:
                            meets_criteria = False
                    else:
                        # Higher is better
                        if value < threshold:
                            meets_criteria = False
            
            if meets_criteria:
                selected.append(model_name)
        
        self.selected_models = set(selected)
        return selected
    
    def create_model_selection_chart(self) -> go.Figure:
        """Create model selection visualization."""
        if not self.available_models:
            return go.Figure()
        
        model_names = list(self.available_models.keys())
        model_types = [self.available_models[name]['type'].value for name in model_names]
        
        # Create scatter plot of performance metrics
        accuracies = []
        training_times = []
        colors = []
        
        for i, model_name in enumerate(model_names):
            performance = self.available_models[model_name]['performance']
            
            # Use appropriate metric based on model type
            if performance.model_type == ModelType.CLASSIFICATION:
                metric_value = performance.accuracy
            else:
                metric_value = performance.r2_score
            
            accuracies.append(metric_value)
            training_times.append(performance.training_time)
            
            # Color based on selection status
            if model_name in self.selected_models:
                colors.append('red')
            else:
                colors.append('blue')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=training_times,
            y=accuracies,
            mode='markers+text',
            text=model_names,
            textposition="top center",
            marker=dict(size=10, color=colors),
            name='Models'
        ))
        
        fig.update_layout(
            title="Model Selection Interface",
            xaxis_title="Training Time (seconds)",
            yaxis_title="Performance Metric",
            height=500
        )
        
        return fig

class ModelValidationDisplay:
    """Model validation display component."""
    
    def __init__(self):
        """Initialize model validation display."""
        self.validation_results = {}
        self.cross_validation_scores = {}
    
    def add_validation_result(self, model_name: str, validation_scores: List[float],
                            validation_type: str = "cross_validation"):
        """Add validation results for a model."""
        self.validation_results[model_name] = {
            'scores': validation_scores,
            'type': validation_type,
            'mean_score': np.mean(validation_scores),
            'std_score': np.std(validation_scores)
        }
    
    def create_validation_comparison_chart(self) -> go.Figure:
        """Create validation comparison chart."""
        if not self.validation_results:
            return go.Figure()
        
        model_names = list(self.validation_results.keys())
        mean_scores = [self.validation_results[name]['mean_score'] for name in model_names]
        std_scores = [self.validation_results[name]['std_score'] for name in model_names]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=mean_scores,
            error_y=dict(type='data', array=std_scores, visible=True),
            name='Mean Validation Score',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Model Validation Comparison",
            xaxis_title="Models",
            yaxis_title="Validation Score",
            height=400
        )
        
        return fig
    
    def create_validation_distribution_chart(self, model_name: str) -> go.Figure:
        """Create validation score distribution chart."""
        if model_name not in self.validation_results:
            return go.Figure()
        
        data = self.validation_results[model_name]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data['scores'],
            nbinsx=20,
            name='Validation Scores',
            marker_color='green'
        ))
        
        # Add mean line
        fig.add_vline(x=data['mean_score'], line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {data['mean_score']:.3f}")
        
        fig.update_layout(
            title=f"Validation Score Distribution - {model_name}",
            xaxis_title="Validation Score",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig

class ModelComparisonManager:
    """Main model comparison manager."""
    
    def __init__(self):
        """Initialize model comparison manager."""
        self.performance_comparison = ModelPerformanceComparison()
        self.ensemble_visualization = EnsembleModelVisualization()
        self.feature_importance = FeatureImportanceComparison()
        self.prediction_accuracy = PredictionAccuracyCharts()
        self.model_selection = ModelSelectionInterface()
        self.model_validation = ModelValidationDisplay()
        
        self.models = []
    
    def add_model(self, model_name: str, model_type: ModelType, 
                 performance: ModelPerformance, prediction: ModelPrediction = None):
        """Add model to comparison manager."""
        if model_name not in self.models:
            self.models.append(model_name)
        
        # Add to all components
        self.performance_comparison.add_model_performance(performance)
        self.model_selection.add_model(model_name, model_type, performance)
        
        if prediction:
            self.prediction_accuracy.add_prediction_data(model_name, prediction)
    
    def add_ensemble_data(self, ensemble_name: str, base_models: List[str],
                         ensemble_predictions: np.ndarray, base_predictions: Dict[str, np.ndarray],
                         y_true: np.ndarray):
        """Add ensemble model data."""
        self.ensemble_visualization.add_ensemble_data(
            ensemble_name, base_models, ensemble_predictions, base_predictions, y_true
        )
    
    def add_feature_importance(self, model_name: str, feature_names: List[str], 
                              importance_scores: np.ndarray):
        """Add feature importance data."""
        self.feature_importance.add_feature_importance(model_name, feature_names, importance_scores)
    
    def add_validation_result(self, model_name: str, validation_scores: List[float],
                            validation_type: str = "cross_validation"):
        """Add validation results."""
        self.model_validation.add_validation_result(model_name, validation_scores, validation_type)
    
    def create_comprehensive_comparison_report(self) -> Dict[str, go.Figure]:
        """Create comprehensive model comparison report."""
        report = {
            'performance_comparison': self.performance_comparison.create_performance_comparison_chart(),
            'feature_importance_comparison': self.feature_importance.create_feature_importance_comparison(),
            'top_features': self.feature_importance.create_top_features_chart(),
            'model_selection': self.model_selection.create_model_selection_chart(),
            'validation_comparison': self.model_validation.create_validation_comparison_chart()
        }
        
        # Add prediction accuracy charts for each model
        for model in self.models:
            if model in self.prediction_accuracy.prediction_data:
                report[f'prediction_accuracy_{model}'] = self.prediction_accuracy.create_prediction_vs_actual_chart(model)
                report[f'confusion_matrix_{model}'] = self.prediction_accuracy.create_confusion_matrix_chart(model)
        
        # Add ensemble charts
        for ensemble_name in self.ensemble_visualization.ensemble_data.keys():
            report[f'ensemble_comparison_{ensemble_name}'] = self.ensemble_visualization.create_ensemble_comparison_chart(ensemble_name)
            report[f'ensemble_agreement_{ensemble_name}'] = self.ensemble_visualization.create_ensemble_agreement_chart(ensemble_name)
        
        return report

def create_model_comparison_manager() -> ModelComparisonManager:
    """
    Create a model comparison manager.
    
    Returns:
        ModelComparisonManager instance
    """
    return ModelComparisonManager()

if __name__ == "__main__":
    # Demo of model comparison
    manager = create_model_comparison_manager()
    
    # Add sample models
    models = ["Random Forest", "XGBoost", "Neural Network", "SVM"]
    model_types = [ModelType.CLASSIFICATION] * len(models)
    
    for i, (model_name, model_type) in enumerate(zip(models, model_types)):
        # Generate sample performance
        performance = ModelPerformance(
            model_name=model_name,
            model_type=model_type,
            accuracy=np.random.uniform(0.7, 0.95),
            precision=np.random.uniform(0.7, 0.95),
            recall=np.random.uniform(0.7, 0.95),
            f1_score=np.random.uniform(0.7, 0.95),
            training_time=np.random.uniform(1, 10),
            prediction_time=np.random.uniform(0.001, 0.1)
        )
        
        # Generate sample predictions
        y_true = np.random.randint(0, 2, 1000)
        y_pred = np.random.randint(0, 2, 1000)
        prediction = ModelPrediction(model_name, y_true, y_pred)
        
        manager.add_model(model_name, model_type, performance, prediction)
        
        # Add feature importance
        feature_names = [f"feature_{j}" for j in range(20)]
        importance_scores = np.random.rand(20)
        manager.add_feature_importance(model_name, feature_names, importance_scores)
        
        # Add validation results
        validation_scores = np.random.uniform(0.7, 0.95, 10)
        manager.add_validation_result(model_name, validation_scores.tolist())
    
    print("Model comparison manager created successfully!")
    print(f"Models: {manager.models}")
    
    # Create comprehensive report
    report = manager.create_comprehensive_comparison_report()
    print(f"Generated {len(report)} comparison charts")