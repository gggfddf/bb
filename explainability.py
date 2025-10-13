"""
Explainability module - provides interpretable insights into model predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available for plotting")


class ModelExplainer:
    """
    Provides interpretability for ML models using various techniques.
    """
    
    def __init__(self):
        self.shap_values = None
        self.explainer = None
        
    def explain_tree_model(self, model: Any, X: pd.DataFrame,
                          feature_names: Optional[List[str]] = None) -> Dict:
        """
        Explain tree-based model predictions using SHAP.
        
        Args:
            model: Trained tree model (LightGBM, XGBoost, CatBoost)
            X: Input features
            feature_names: List of feature names
            
        Returns:
            Dictionary with explanation results
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP not available")
            return {}
        
        if feature_names is None:
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        try:
            # Create SHAP explainer
            self.explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            self.shap_values = self.explainer.shap_values(X)
            
            # Handle different output shapes
            if isinstance(self.shap_values, list):
                # Multi-class case
                shap_values_to_use = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
            else:
                shap_values_to_use = self.shap_values
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values_to_use).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Get top features
            top_features = importance_df.head(20)
            
            return {
                'shap_values': self.shap_values,
                'feature_importance': importance_df,
                'top_features': top_features,
                'explainer': self.explainer,
            }
        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}")
            return {}
    
    def get_instance_explanation(self, instance_idx: int,
                                 feature_names: List[str]) -> pd.DataFrame:
        """
        Get explanation for a specific instance.
        
        Args:
            instance_idx: Index of instance to explain
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature contributions
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Run explain_tree_model first.")
        
        # Handle multi-class
        if isinstance(self.shap_values, list):
            shap_values_to_use = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        else:
            shap_values_to_use = self.shap_values
        
        # Get SHAP values for this instance
        instance_shap = shap_values_to_use[instance_idx]
        
        explanation_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': instance_shap
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return explanation_df
    
    def plot_shap_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """Plot SHAP summary plot."""
        if not SHAP_AVAILABLE or not PLOTTING_AVAILABLE:
            warnings.warn("SHAP or plotting libraries not available")
            return
        
        if self.shap_values is None:
            raise ValueError("No SHAP values computed")
        
        plt.figure(figsize=(10, 8))
        
        # Handle multi-class
        if isinstance(self.shap_values, list):
            shap_values_to_use = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        else:
            shap_values_to_use = self.shap_values
        
        shap.summary_plot(shap_values_to_use, X, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()


class AttentionVisualizer:
    """
    Visualizes attention weights for sequence models.
    """
    
    @staticmethod
    def extract_attention_weights(model: Any, X: np.ndarray) -> np.ndarray:
        """
        Extract attention weights from Transformer model.
        
        This is a placeholder - actual implementation depends on model architecture.
        """
        warnings.warn("Attention extraction not implemented for this model type")
        return np.array([])
    
    @staticmethod
    def plot_attention_heatmap(attention_weights: np.ndarray,
                               timestamps: List[str],
                               save_path: Optional[str] = None):
        """Plot attention weights as heatmap."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting libraries not available")
            return
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights, xticklabels=timestamps, yticklabels=timestamps,
                   cmap='viridis', cbar=True)
        plt.title('Attention Weights')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()


class PatternLibraryExplainer:
    """
    Creates human-readable explanations for discovered patterns.
    """
    
    @staticmethod
    def describe_pattern(pattern_data: Dict, pattern_name: str) -> str:
        """
        Generate natural language description of a pattern.
        
        Args:
            pattern_data: Dictionary with pattern statistics
            pattern_name: Name of the pattern
            
        Returns:
            Human-readable description
        """
        description_parts = [f"Pattern: {pattern_name}"]
        
        # Occurrence frequency
        occurrences = pattern_data.get('occurrences') or pattern_data.get('size', 0)
        description_parts.append(f"Observed {occurrences} times in the dataset")
        
        # Best performing horizon
        best_horizon = None
        best_win_rate = 0
        best_return = 0
        
        for key, value in pattern_data.items():
            if key.startswith('horizon_') and isinstance(value, dict):
                horizon = key.split('_')[1]
                win_rate = value.get('win_rate', 0)
                mean_return = value.get('mean_return', 0)
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_horizon = horizon
                    best_return = mean_return
        
        if best_horizon:
            description_parts.append(
                f"Best predictive power at {best_horizon}-day horizon: "
                f"{best_win_rate:.1%} win rate, {best_return:.2%} mean return"
            )
        
        return ". ".join(description_parts) + "."
    
    @staticmethod
    def create_pattern_report(pattern_library: Dict) -> pd.DataFrame:
        """
        Create a comprehensive report of all patterns.
        
        Args:
            pattern_library: Dictionary of patterns and their statistics
            
        Returns:
            DataFrame with pattern report
        """
        reports = []
        
        for pattern_name, pattern_data in pattern_library.items():
            # Get description
            description = PatternLibraryExplainer.describe_pattern(pattern_data, pattern_name)
            
            # Extract key metrics
            occurrences = pattern_data.get('occurrences') or pattern_data.get('size', 0)
            
            # Find best horizon
            best_metrics = {}
            for key, value in pattern_data.items():
                if key.startswith('horizon_') and isinstance(value, dict):
                    horizon = key.split('_')[1]
                    sharpe = value.get('sharpe_ratio', -np.inf)
                    
                    if not best_metrics or sharpe > best_metrics.get('sharpe', -np.inf):
                        best_metrics = {
                            'horizon': horizon,
                            'win_rate': value.get('win_rate', 0),
                            'mean_return': value.get('mean_return', 0),
                            'sharpe': sharpe,
                        }
            
            reports.append({
                'pattern_name': pattern_name,
                'description': description,
                'occurrences': occurrences,
                'best_horizon_days': best_metrics.get('horizon', 'N/A'),
                'win_rate': best_metrics.get('win_rate', 0),
                'mean_return': best_metrics.get('mean_return', 0),
                'sharpe_ratio': best_metrics.get('sharpe', 0),
            })
        
        return pd.DataFrame(reports).sort_values('sharpe_ratio', ascending=False)


class SignalExplainer:
    """
    Explains trading signals and predictions.
    """
    
    @staticmethod
    def explain_signal(prediction: float, features: Dict[str, float],
                      feature_importance: pd.DataFrame,
                      threshold: float = 0.5) -> Dict:
        """
        Explain a trading signal.
        
        Args:
            prediction: Model prediction (probability or score)
            features: Dictionary of feature values
            feature_importance: DataFrame with feature importance
            threshold: Decision threshold
            
        Returns:
            Dictionary with signal explanation
        """
        signal = 'BUY' if prediction >= threshold else 'HOLD/SELL'
        confidence = abs(prediction - 0.5) * 2  # Scale to 0-1
        
        # Get top contributing features
        top_features = feature_importance.head(10)
        
        contributing_features = []
        for _, row in top_features.iterrows():
            feat_name = row['feature']
            if feat_name in features:
                contributing_features.append({
                    'feature': feat_name,
                    'value': features[feat_name],
                    'importance': row['importance']
                })
        
        explanation = {
            'signal': signal,
            'confidence': confidence,
            'prediction_score': prediction,
            'threshold': threshold,
            'top_contributing_features': contributing_features,
        }
        
        return explanation
    
    @staticmethod
    def generate_signal_narrative(explanation: Dict) -> str:
        """
        Generate natural language narrative for a signal.
        
        Args:
            explanation: Dictionary from explain_signal
            
        Returns:
            Human-readable narrative
        """
        signal = explanation['signal']
        confidence = explanation['confidence']
        
        narrative_parts = [
            f"Signal: {signal}",
            f"Confidence: {confidence:.1%}",
        ]
        
        if explanation['top_contributing_features']:
            narrative_parts.append("Key factors:")
            for feat in explanation['top_contributing_features'][:5]:
                narrative_parts.append(
                    f"  - {feat['feature']}: {feat['value']:.3f} (importance: {feat['importance']:.3f})"
                )
        
        return "\n".join(narrative_parts)


class FeatureAttributor:
    """
    Attributes predictions to specific features and time periods.
    """
    
    @staticmethod
    def attribute_to_time_windows(shap_values: np.ndarray,
                                  feature_names: List[str],
                                  window_sizes: List[int]) -> Dict[int, float]:
        """
        Attribute prediction to different time windows.
        
        Args:
            shap_values: SHAP values for an instance
            feature_names: List of feature names
            window_sizes: List of window sizes used in features
            
        Returns:
            Dictionary mapping window size to attribution score
        """
        window_attribution = {w: 0.0 for w in window_sizes}
        
        for i, feat_name in enumerate(feature_names):
            shap_val = shap_values[i]
            
            # Extract window size from feature name (e.g., "momentum_20" -> 20)
            for window in window_sizes:
                if f'_{window}' in feat_name or f'{window}d' in feat_name:
                    window_attribution[window] += abs(shap_val)
                    break
        
        return window_attribution
    
    @staticmethod
    def identify_key_time_periods(shap_values: np.ndarray,
                                  feature_names: List[str],
                                  top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Identify which time periods (via features) are most important.
        
        Returns:
            List of (feature_name, attribution_score) tuples
        """
        attributions = [(name, abs(val)) for name, val in zip(feature_names, shap_values)]
        attributions.sort(key=lambda x: x[1], reverse=True)
        
        return attributions[:top_k]


def create_prediction_report(prediction: float, 
                            features: pd.Series,
                            shap_values: np.ndarray,
                            pattern_matches: List[str],
                            cycle_info: Dict) -> str:
    """
    Create comprehensive prediction report combining all explanation methods.
    
    Args:
        prediction: Model prediction
        features: Feature values for this instance
        shap_values: SHAP values for this instance
        pattern_matches: List of matched pattern names
        cycle_info: Information about current cycle/regime
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 60,
        "PREDICTION REPORT",
        "=" * 60,
        "",
        f"Prediction Score: {prediction:.3f}",
        f"Signal: {'BUY' if prediction >= 0.5 else 'HOLD/SELL'}",
        "",
        "TOP CONTRIBUTING FACTORS:",
    ]
    
    # Top SHAP features
    if shap_values is not None and len(shap_values) > 0:
        feat_names = features.index.tolist()
        top_features = FeatureAttributor.identify_key_time_periods(shap_values, feat_names, top_k=5)
        
        for feat_name, attribution in top_features:
            feat_value = features[feat_name] if feat_name in features.index else 'N/A'
            report_lines.append(f"  - {feat_name}: {feat_value:.3f} (attribution: {attribution:.3f})")
    
    # Matched patterns
    if pattern_matches:
        report_lines.extend([
            "",
            "MATCHED PATTERNS:",
        ])
        for pattern in pattern_matches[:5]:
            report_lines.append(f"  - {pattern}")
    
    # Cycle/regime info
    if cycle_info:
        report_lines.extend([
            "",
            "MARKET REGIME:",
        ])
        for key, value in cycle_info.items():
            report_lines.append(f"  - {key}: {value}")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)
