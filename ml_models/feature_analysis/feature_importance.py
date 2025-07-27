#!/usr/bin/env python3
"""
Feature Importance Analysis Module

Implements comprehensive feature importance analysis for ML models:
- Permutation importance analysis
- SHAP (SHapley Additive exPlanations) analysis
- Feature correlation analysis
- Recursive feature elimination
- Feature stability analysis
- Feature selection frameworks

Features:
- Permutation importance with statistical significance
- SHAP values for model interpretability
- Feature correlation and multicollinearity detection
- Recursive feature elimination with cross-validation
- Feature stability across different datasets
- Automated feature selection pipelines
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import pickle

logger = structlog.get_logger()

class ImportanceMethod(Enum):
    """Feature importance calculation methods."""
    PERMUTATION = "permutation"
    SHAP = "shap"
    CORRELATION = "correlation"
    RFE = "rfe"
    STABILITY = "stability"
    SELECTION = "selection"

class SelectionCriterion(Enum):
    """Feature selection criteria."""
    F_SCORE = "f_score"
    MUTUAL_INFO = "mutual_info"
    CHI2 = "chi2"
    ANOVA = "anova"
    RECURSIVE = "recursive"

@dataclass
class FeatureImportance:
    """Feature importance result."""
    feature_name: str
    importance_score: float
    rank: int
    method: str
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureAnalysisConfig:
    """Configuration for feature analysis."""
    methods: List[ImportanceMethod]
    n_permutations: int = 100
    cv_folds: int = 5
    random_state: int = 42
    n_features_to_select: int = 10
    stability_threshold: float = 0.8
    correlation_threshold: float = 0.8
    enable_visualization: bool = True

@dataclass
class FeatureAnalysisResult:
    """Result from feature analysis."""
    feature_importances: List[FeatureImportance]
    selected_features: List[str]
    correlation_matrix: Optional[pd.DataFrame] = None
    stability_scores: Optional[Dict[str, float]] = None
    selection_summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PermutationImportanceAnalyzer:
    """Permutation importance analysis."""
    
    def __init__(self, n_permutations: int = 100, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize permutation importance analyzer.
        
        Args:
            n_permutations: Number of permutations
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_permutations = n_permutations
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def calculate_importance(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str] = None) -> List[FeatureImportance]:
        """
        Calculate permutation importance for features.
        
        Args:
            model: Trained model
            X: Features
            y: Targets
            feature_names: Names of features
            
        Returns:
            List of feature importance results
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info("Calculating permutation importance", 
                   n_features=len(feature_names),
                   n_permutations=self.n_permutations)
        
        # Get baseline score
        baseline_score = self._get_baseline_score(model, X, y)
        
        # Calculate importance for each feature
        importances = []
        for i, feature_name in enumerate(feature_names):
            importance_score, confidence_interval, p_value = self._calculate_feature_importance(
                model, X, y, i, baseline_score
            )
            
            importance = FeatureImportance(
                feature_name=feature_name,
                importance_score=importance_score,
                rank=0,  # Will be set later
                method="permutation",
                confidence_interval=confidence_interval,
                p_value=p_value
            )
            importances.append(importance)
        
        # Sort by importance and assign ranks
        importances.sort(key=lambda x: x.importance_score, reverse=True)
        for i, importance in enumerate(importances):
            importance.rank = i + 1
        
        logger.info("Permutation importance calculation completed", 
                   top_feature=importances[0].feature_name if importances else None)
        
        return importances
    
    def _get_baseline_score(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        """Get baseline model score."""
        if hasattr(model, 'predict_proba'):
            # Classification
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='accuracy')
        else:
            # Regression
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='neg_mean_squared_error')
            scores = -scores  # Convert back to positive
        
        return np.mean(scores)
    
    def _calculate_feature_importance(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, 
                                    feature_idx: int, baseline_score: float) -> Tuple[float, Tuple[float, float], float]:
        """Calculate importance for a single feature."""
        np.random.seed(self.random_state)
        
        # Store original feature values
        original_feature = X[:, feature_idx].copy()
        
        # Calculate scores with permuted feature
        permuted_scores = []
        for _ in range(self.n_permutations):
            # Permute the feature
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = np.random.permutation(original_feature)
            
            # Calculate score with permuted feature
            if hasattr(model, 'predict_proba'):
                score = cross_val_score(model, X_permuted, y, cv=self.cv_folds, scoring='accuracy')
            else:
                score = cross_val_score(model, X_permuted, y, cv=self.cv_folds, scoring='neg_mean_squared_error')
                score = -score
            
            permuted_scores.append(np.mean(score))
        
        # Calculate importance as difference from baseline
        importance_score = baseline_score - np.mean(permuted_scores)
        
        # Calculate confidence interval
        confidence_interval = np.percentile(permuted_scores, [2.5, 97.5])
        
        # Calculate p-value (proportion of permutations with better score)
        p_value = np.mean(np.array(permuted_scores) >= baseline_score)
        
        return importance_score, tuple(confidence_interval), p_value

class SHAPImportanceAnalyzer:
    """SHAP (SHapley Additive exPlanations) importance analysis."""
    
    def __init__(self):
        """Initialize SHAP importance analyzer."""
        self.shap_available = self._check_shap_availability()
    
    def _check_shap_availability(self) -> bool:
        """Check if SHAP is available."""
        try:
            import shap
            return True
        except ImportError:
            logger.warning("SHAP not available, SHAP analysis will be skipped")
            return False
    
    def calculate_importance(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                           feature_names: List[str] = None) -> List[FeatureImportance]:
        """
        Calculate SHAP importance for features.
        
        Args:
            model: Trained model
            X: Features
            y: Targets
            feature_names: Names of features
            
        Returns:
            List of feature importance results
        """
        if not self.shap_available:
            logger.warning("SHAP not available, returning empty results")
            return []
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info("Calculating SHAP importance", n_features=len(feature_names))
        
        try:
            import shap
            
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                # Classification
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.LinearExplainer(model, X)
            else:
                # Regression
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.LinearExplainer(model, X)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # For classification, use the positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance results
            importances = []
            for i, (feature_name, shap_value) in enumerate(zip(feature_names, mean_shap_values)):
                importance = FeatureImportance(
                    feature_name=feature_name,
                    importance_score=shap_value,
                    rank=0,  # Will be set later
                    method="shap"
                )
                importances.append(importance)
            
            # Sort by importance and assign ranks
            importances.sort(key=lambda x: x.importance_score, reverse=True)
            for i, importance in enumerate(importances):
                importance.rank = i + 1
            
            logger.info("SHAP importance calculation completed", 
                       top_feature=importances[0].feature_name if importances else None)
            
            return importances
            
        except Exception as e:
            logger.error("SHAP importance calculation failed", error=str(e))
            return []

class FeatureCorrelationAnalyzer:
    """Feature correlation analysis."""
    
    def __init__(self, correlation_threshold: float = 0.8):
        """
        Initialize feature correlation analyzer.
        
        Args:
            correlation_threshold: Threshold for high correlation
        """
        self.correlation_threshold = correlation_threshold
    
    def analyze_correlations(self, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyze feature correlations.
        
        Args:
            X: Features
            feature_names: Names of features
            
        Returns:
            Correlation analysis results
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info("Analyzing feature correlations", n_features=len(feature_names))
        
        # Calculate correlation matrix
        df = pd.DataFrame(X, columns=feature_names)
        correlation_matrix = df.corr()
        
        # Find highly correlated feature pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= self.correlation_threshold:
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Calculate feature importance based on correlation with target
        # (This is a simplified approach - in practice you'd have the target variable)
        feature_importances = []
        for i, feature_name in enumerate(feature_names):
            # For demonstration, use variance as a proxy for importance
            importance_score = np.var(X[:, i])
            
            importance = FeatureImportance(
                feature_name=feature_name,
                importance_score=importance_score,
                rank=0,
                method="correlation"
            )
            feature_importances.append(importance)
        
        # Sort by importance and assign ranks
        feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
        for i, importance in enumerate(feature_importances):
            importance.rank = i + 1
        
        result = {
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_correlations,
            'feature_importances': feature_importances,
            'multicollinearity_detected': len(high_correlations) > 0
        }
        
        logger.info("Feature correlation analysis completed", 
                   high_correlations=len(high_correlations),
                   multicollinearity=result['multicollinearity_detected'])
        
        return result

class RecursiveFeatureEliminationAnalyzer:
    """Recursive feature elimination analysis."""
    
    def __init__(self, n_features_to_select: int = 10, cv_folds: int = 5):
        """
        Initialize recursive feature elimination analyzer.
        
        Args:
            n_features_to_select: Number of features to select
            cv_folds: Number of cross-validation folds
        """
        self.n_features_to_select = n_features_to_select
        self.cv_folds = cv_folds
    
    def select_features(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                       feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Perform recursive feature elimination.
        
        Args:
            model: Base model for feature selection
            X: Features
            y: Targets
            feature_names: Names of features
            
        Returns:
            Feature selection results
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info("Performing recursive feature elimination", 
                   n_features=len(feature_names),
                   n_to_select=self.n_features_to_select)
        
        # Create RFE object
        rfe = RFE(
            estimator=model,
            n_features_to_select=self.n_features_to_select,
            step=1
        )
        
        # Fit RFE
        rfe.fit(X, y)
        
        # Get selected features
        selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
        
        # Get feature rankings
        feature_rankings = []
        for i, feature_name in enumerate(feature_names):
            ranking = FeatureImportance(
                feature_name=feature_name,
                importance_score=1.0 / (rfe.ranking_[i] + 1),  # Convert ranking to score
                rank=rfe.ranking_[i],
                method="rfe"
            )
            feature_rankings.append(ranking)
        
        # Sort by rank
        feature_rankings.sort(key=lambda x: x.rank)
        
        result = {
            'selected_features': selected_features,
            'feature_rankings': feature_rankings,
            'n_selected': len(selected_features),
            'rfe_support': rfe.support_.tolist(),
            'rfe_ranking': rfe.ranking_.tolist()
        }
        
        logger.info("Recursive feature elimination completed", 
                   selected_features=selected_features)
        
        return result

class FeatureStabilityAnalyzer:
    """Feature stability analysis across different datasets."""
    
    def __init__(self, n_bootstrap: int = 100, stability_threshold: float = 0.8):
        """
        Initialize feature stability analyzer.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            stability_threshold: Threshold for stability
        """
        self.n_bootstrap = n_bootstrap
        self.stability_threshold = stability_threshold
    
    def analyze_stability(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyze feature stability across bootstrap samples.
        
        Args:
            model: Model to analyze
            X: Features
            y: Targets
            feature_names: Names of features
            
        Returns:
            Stability analysis results
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info("Analyzing feature stability", 
                   n_features=len(feature_names),
                   n_bootstrap=self.n_bootstrap)
        
        n_samples, n_features = X.shape
        feature_selection_frequency = np.zeros(n_features)
        
        # Bootstrap sampling and feature selection
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Fit model and get feature importance
            model_copy = joblib.load(joblib.dump(model)[1])  # Deep copy
            model_copy.fit(X_boot, y_boot)
            
            # Get feature importance (assuming tree-based model)
            if hasattr(model_copy, 'feature_importances_'):
                importances = model_copy.feature_importances_
            else:
                # For non-tree models, use permutation importance
                perm_analyzer = PermutationImportanceAnalyzer(n_permutations=10)
                importances = [imp.importance_score for imp in perm_analyzer.calculate_importance(
                    model_copy, X_boot, y_boot, feature_names
                )]
            
            # Select top features
            top_features = np.argsort(importances)[-10:]  # Top 10 features
            feature_selection_frequency[top_features] += 1
        
        # Calculate stability scores
        stability_scores = feature_selection_frequency / self.n_bootstrap
        
        # Create feature importance results
        feature_importances = []
        for i, (feature_name, stability_score) in enumerate(zip(feature_names, stability_scores)):
            importance = FeatureImportance(
                feature_name=feature_name,
                importance_score=stability_score,
                rank=0,
                method="stability",
                metadata={'selection_frequency': feature_selection_frequency[i]}
            )
            feature_importances.append(importance)
        
        # Sort by stability and assign ranks
        feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
        for i, importance in enumerate(feature_importances):
            importance.rank = i + 1
        
        # Identify stable features
        stable_features = [
            feature_name for feature_name, score in zip(feature_names, stability_scores)
            if score >= self.stability_threshold
        ]
        
        result = {
            'stability_scores': dict(zip(feature_names, stability_scores)),
            'feature_importances': feature_importances,
            'stable_features': stable_features,
            'n_stable': len(stable_features),
            'selection_frequencies': dict(zip(feature_names, feature_selection_frequency))
        }
        
        logger.info("Feature stability analysis completed", 
                   stable_features=stable_features,
                   n_stable=len(stable_features))
        
        return result

class FeatureSelectionPipeline:
    """Automated feature selection pipeline."""
    
    def __init__(self, config: FeatureAnalysisConfig):
        """
        Initialize feature selection pipeline.
        
        Args:
            config: Feature analysis configuration
        """
        self.config = config
        self.permutation_analyzer = PermutationImportanceAnalyzer(
            n_permutations=config.n_permutations,
            cv_folds=config.cv_folds,
            random_state=config.random_state
        )
        self.shap_analyzer = SHAPImportanceAnalyzer()
        self.correlation_analyzer = FeatureCorrelationAnalyzer(
            correlation_threshold=config.correlation_threshold
        )
        self.rfe_analyzer = RecursiveFeatureEliminationAnalyzer(
            n_features_to_select=config.n_features_to_select,
            cv_folds=config.cv_folds
        )
        self.stability_analyzer = FeatureStabilityAnalyzer(
            n_bootstrap=100,
            stability_threshold=config.stability_threshold
        )
    
    def analyze_features(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                        feature_names: List[str] = None) -> FeatureAnalysisResult:
        """
        Perform comprehensive feature analysis.
        
        Args:
            model: Model to analyze
            X: Features
            y: Targets
            feature_names: Names of features
            
        Returns:
            Comprehensive feature analysis results
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info("Starting comprehensive feature analysis", 
                   n_features=len(feature_names),
                   methods=[method.value for method in self.config.methods])
        
        all_importances = []
        correlation_result = None
        stability_result = None
        selection_summary = {}
        
        # Perform analysis based on configured methods
        for method in self.config.methods:
            try:
                if method == ImportanceMethod.PERMUTATION:
                    importances = self.permutation_analyzer.calculate_importance(
                        model, X, y, feature_names
                    )
                    all_importances.extend(importances)
                    selection_summary['permutation'] = {
                        'n_features_analyzed': len(importances),
                        'top_features': [imp.feature_name for imp in importances[:5]]
                    }
                
                elif method == ImportanceMethod.SHAP:
                    importances = self.shap_analyzer.calculate_importance(
                        model, X, y, feature_names
                    )
                    all_importances.extend(importances)
                    selection_summary['shap'] = {
                        'n_features_analyzed': len(importances),
                        'top_features': [imp.feature_name for imp in importances[:5]] if importances else []
                    }
                
                elif method == ImportanceMethod.CORRELATION:
                    correlation_result = self.correlation_analyzer.analyze_correlations(X, feature_names)
                    all_importances.extend(correlation_result['feature_importances'])
                    selection_summary['correlation'] = {
                        'n_features_analyzed': len(correlation_result['feature_importances']),
                        'high_correlations': len(correlation_result['high_correlations']),
                        'multicollinearity_detected': correlation_result['multicollinearity_detected']
                    }
                
                elif method == ImportanceMethod.RFE:
                    rfe_result = self.rfe_analyzer.select_features(model, X, y, feature_names)
                    all_importances.extend(rfe_result['feature_rankings'])
                    selection_summary['rfe'] = {
                        'n_features_analyzed': len(rfe_result['feature_rankings']),
                        'n_selected': rfe_result['n_selected'],
                        'selected_features': rfe_result['selected_features']
                    }
                
                elif method == ImportanceMethod.STABILITY:
                    stability_result = self.stability_analyzer.analyze_stability(model, X, y, feature_names)
                    all_importances.extend(stability_result['feature_importances'])
                    selection_summary['stability'] = {
                        'n_features_analyzed': len(stability_result['feature_importances']),
                        'n_stable': stability_result['n_stable'],
                        'stable_features': stability_result['stable_features']
                    }
                
            except Exception as e:
                logger.error(f"Error in {method.value} analysis", error=str(e))
        
        # Select final features based on consensus
        selected_features = self._select_consensus_features(all_importances, feature_names)
        
        # Create result
        result = FeatureAnalysisResult(
            feature_importances=all_importances,
            selected_features=selected_features,
            correlation_matrix=correlation_result['correlation_matrix'] if correlation_result else None,
            stability_scores=stability_result['stability_scores'] if stability_result else None,
            selection_summary=selection_summary
        )
        
        logger.info("Feature analysis completed", 
                   selected_features=selected_features,
                   n_selected=len(selected_features))
        
        return result
    
    def _select_consensus_features(self, all_importances: List[FeatureImportance], 
                                 feature_names: List[str]) -> List[str]:
        """Select features based on consensus across methods."""
        # Group importances by feature
        feature_scores = {}
        for importance in all_importances:
            feature_name = importance.feature_name
            if feature_name not in feature_scores:
                feature_scores[feature_name] = []
            feature_scores[feature_name].append(importance.importance_score)
        
        # Calculate average importance for each feature
        avg_scores = {}
        for feature_name, scores in feature_scores.items():
            avg_scores[feature_name] = np.mean(scores)
        
        # Select top features
        sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, score in sorted_features[:self.config.n_features_to_select]]
        
        return selected_features

def create_feature_analyzer(methods: List[str] = None,
                          n_features_to_select: int = 10,
                          cv_folds: int = 5) -> FeatureSelectionPipeline:
    """
    Create a feature analysis pipeline.
    
    Args:
        methods: List of analysis methods
        n_features_to_select: Number of features to select
        cv_folds: Number of cross-validation folds
        
    Returns:
        FeatureSelectionPipeline instance
    """
    if methods is None:
        methods = ['permutation', 'correlation', 'rfe', 'stability']
    
    config = FeatureAnalysisConfig(
        methods=[ImportanceMethod(method) for method in methods],
        n_features_to_select=n_features_to_select,
        cv_folds=cv_folds
    )
    
    return FeatureSelectionPipeline(config)

if __name__ == "__main__":
    # Demo of feature importance analysis
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create feature analyzer
    analyzer = create_feature_analyzer()
    
    # Analyze features
    result = analyzer.analyze_features(model, X_train, y_train, feature_names)
    
    print(f"Selected features: {result.selected_features}")
    print(f"Number of selected features: {len(result.selected_features)}")
    print(f"Selection summary: {result.selection_summary}")