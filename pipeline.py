"""
Main pipeline orchestrator - coordinates all components of the system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import pickle
import json
from pathlib import Path
import warnings

from config import SystemConfig, DEFAULT_CONFIG
from data_preprocessing import DataPreprocessor, create_date_features
from feature_engineering import FeatureEngineer
from label_generation import LabelGenerator
from models import BaselineTreeModel, evaluate_predictions
try:
    from models import SequenceModel
except ImportError:
    SequenceModel = None
from pattern_discovery import PatternDiscovery, ShapeBasedMatcher
from cycle_detection import CycleDetector, RegimeAnalyzer
from explainability import ModelExplainer, PatternLibraryExplainer, create_prediction_report
from evaluation import WalkForwardValidator, Backtester, generate_evaluation_report


class StockPatternPipeline:
    """
    End-to-end pipeline for stock pattern detection and prediction.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config.data)
        self.feature_engineer = FeatureEngineer(self.config.feature)
        self.label_generator = LabelGenerator(self.config.label)
        self.pattern_discovery = PatternDiscovery(self.config.pattern)
        self.cycle_detector = CycleDetector(self.config.cycle)
        self.explainer = ModelExplainer()
        
        # Models
        self.models = {}
        self.pattern_library = {}
        self.cycle_info = {}
        
        # Results
        self.processed_data = None
        self.features = None
        self.labels = None
        
    def load_and_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Load and preprocess data.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Preprocessed DataFrame
        """
        print("=" * 70)
        print("STEP 1: Data Preprocessing")
        print("=" * 70)
        
        df, metadata = self.preprocessor.preprocess_pipeline(data)
        
        print(f"Loaded {metadata['n_rows_initial']} rows")
        print(f"Date range: {metadata['date_range'][0].date()} to {metadata['date_range'][1].date()}")
        print(f"Has volume: {metadata['has_volume']}")
        print(f"Outliers detected: {metadata['n_outliers']}")
        print(f"Final rows: {metadata['n_rows_after_cleaning']}")
        
        # Add date features
        df = create_date_features(df)
        
        self.processed_data = df
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with features
        """
        print("\n" + "=" * 70)
        print("STEP 2: Feature Engineering")
        print("=" * 70)
        
        df = self.feature_engineer.generate_all_features(df)
        
        feature_names = self.feature_engineer.get_feature_names(df)
        print(f"Generated {len(feature_names)} features")
        
        self.features = df
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate training labels.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with labels
        """
        print("\n" + "=" * 70)
        print("STEP 3: Label Generation")
        print("=" * 70)
        
        df = self.label_generator.generate_all_labels(df)
        
        # Get statistics
        stats = self.label_generator.get_label_statistics(df)
        
        print("\nLabel Statistics:")
        for label_name, label_stats in stats.items():
            print(f"\n{label_name}:")
            for key, value in label_stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        self.labels = df
        return df
    
    def discover_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Discover recurring patterns.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Pattern library dictionary
        """
        print("\n" + "=" * 70)
        print("STEP 4: Pattern Discovery")
        print("=" * 70)
        
        pattern_library = self.pattern_discovery.build_pattern_library(df)
        
        # Filter significant patterns
        significant_patterns = self.pattern_discovery.filter_significant_patterns()
        
        print(f"\nDiscovered {len(pattern_library)} patterns")
        print(f"Significant patterns: {len(significant_patterns)}")
        
        # Get summary
        summary = self.pattern_discovery.get_pattern_summary()
        if len(summary) > 0:
            print("\nTop 5 Patterns by Sharpe Ratio:")
            print(summary.head(5).to_string(index=False))
        
        self.pattern_library = pattern_library
        return pattern_library
    
    def detect_cycles(self, df: pd.DataFrame) -> Dict:
        """
        Detect market cycles and regimes.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Cycle detection results
        """
        print("\n" + "=" * 70)
        print("STEP 5: Cycle Detection")
        print("=" * 70)
        
        cycle_info = self.cycle_detector.run_all_cycle_detection(df)
        
        # Summarize cycles
        cycle_summary = self.cycle_detector.summarize_cycles()
        if len(cycle_summary) > 0:
            print("\nDetected Cycles:")
            print(cycle_summary.to_string(index=False))
        
        # Regime statistics
        if 'market_regime' in df.columns:
            regime_stats = RegimeAnalyzer.calculate_regime_statistics(df)
            print("\nMarket Regime Statistics:")
            for regime, stats in regime_stats.items():
                print(f"\n{regime}:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        
        self.cycle_info = cycle_info
        return cycle_info
    
    def train_models(self, df: pd.DataFrame, 
                    target_label: str = 'label_up_10d',
                    model_types: Optional[List[str]] = None) -> Dict:
        """
        Train prediction models.
        
        Args:
            df: DataFrame with features and labels
            target_label: Label column to predict
            model_types: List of model types to train
            
        Returns:
            Dictionary of trained models
        """
        print("\n" + "=" * 70)
        print("STEP 6: Model Training")
        print("=" * 70)
        
        if model_types is None:
            model_types = ['lightgbm']  # Default to tree model
        
        # Get features and labels
        feature_cols = self.feature_engineer.get_feature_names(df)
        
        # Remove rows with missing target
        valid_mask = df[target_label].notna()
        df_valid = df[valid_mask].copy()
        
        X = df_valid[feature_cols].fillna(0)  # Fill NaN in features
        y = df_valid[target_label]
        
        print(f"\nTraining on {len(X)} samples with {len(feature_cols)} features")
        print(f"Target: {target_label}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Split data (simple train/test for now)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # Train models
        for model_type in model_types:
            print(f"\nTraining {model_type} model...")
            
            if model_type in ['lightgbm', 'xgboost', 'catboost']:
                model = BaselineTreeModel(self.config.model, model_type=model_type)
                
                # Calculate class weights for imbalanced data
                if self.config.model.use_class_weights:
                    class_counts = y_train.value_counts()
                    total = len(y_train)
                    class_weights = {cls: total / (len(class_counts) * count) 
                                   for cls, count in class_counts.items()}
                else:
                    class_weights = None
                
                # Train
                history = model.train(X_train, y_train, X_test, y_test, class_weights)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_binary = (y_pred > 0.5).astype(int) if len(y_pred.shape) == 1 else np.argmax(y_pred, axis=1)
                
                metrics = evaluate_predictions(y_test.values, y_pred_binary)
                
                print(f"\nTest Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                # Feature importance
                feature_importance = model.get_feature_importance(feature_cols)
                print(f"\nTop 10 Features:")
                print(feature_importance.head(10).to_string(index=False))
                
                # Explain model
                explanation = self.explainer.explain_tree_model(model.model, X_test, feature_cols)
                
                self.models[model_type] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_importance': feature_importance,
                    'explanation': explanation,
                }
        
        return self.models
    
    def backtest_strategy(self, df: pd.DataFrame, 
                         predictions: pd.Series,
                         threshold: float = 0.5) -> Any:
        """
        Backtest trading strategy.
        
        Args:
            df: DataFrame with OHLCV data
            predictions: Model predictions
            threshold: Decision threshold
            
        Returns:
            BacktestResult object
        """
        print("\n" + "=" * 70)
        print("STEP 7: Backtesting")
        print("=" * 70)
        
        # Convert predictions to signals
        signals = pd.Series(0, index=predictions.index)
        signals[predictions >= threshold] = 1
        signals[predictions < (1 - threshold)] = -1
        
        # Run backtest
        backtester = Backtester(self.config.evaluation)
        result = backtester.backtest(df.loc[signals.index], signals, predictions)
        
        # Generate report
        report = generate_evaluation_report(result)
        print("\n" + report)
        
        return result
    
    def run_full_pipeline(self, data: pd.DataFrame,
                         target_label: str = 'label_up_10d',
                         run_backtest: bool = True) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            data: Raw OHLCV data
            target_label: Label to predict
            run_backtest: Whether to run backtesting
            
        Returns:
            Dictionary with all results
        """
        print("\n" + "=" * 70)
        print("STOCK PATTERN DETECTION AND MARKET CYCLE ANALYSIS")
        print("=" * 70)
        
        # Step 1: Preprocess
        df = self.load_and_preprocess(data)
        
        # Step 2: Feature engineering
        df = self.engineer_features(df)
        
        # Step 3: Label generation
        df = self.generate_labels(df)
        
        # Step 4: Pattern discovery
        patterns = self.discover_patterns(df)
        
        # Step 5: Cycle detection
        cycles = self.detect_cycles(df)
        
        # Step 6: Train models
        models = self.train_models(df, target_label=target_label)
        
        # Step 7: Backtest (optional)
        backtest_result = None
        if run_backtest and models:
            # Get predictions from best model
            model_name = list(models.keys())[0]
            model = models[model_name]['model']
            
            feature_cols = self.feature_engineer.get_feature_names(df)
            X = df[feature_cols].fillna(0)
            
            predictions = model.predict(X, return_proba=True)
            predictions = pd.Series(predictions, index=df.index)
            
            backtest_result = self.backtest_strategy(df, predictions)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        
        return {
            'data': df,
            'patterns': patterns,
            'cycles': cycles,
            'models': models,
            'backtest': backtest_result,
        }
    
    def save_pipeline(self, save_dir: str):
        """Save trained models and pattern library."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model_data in self.models.items():
            model_path = save_path / f'model_{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_data['model'], f)
            
            # Save feature importance
            importance_path = save_path / f'feature_importance_{model_name}.csv'
            model_data['feature_importance'].to_csv(importance_path, index=False)
        
        # Save pattern library
        pattern_path = save_path / 'pattern_library.json'
        # Convert numpy arrays to lists for JSON serialization
        pattern_library_serializable = {}
        for key, value in self.pattern_library.items():
            pattern_library_serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    pattern_library_serializable[key][k] = v.tolist()
                else:
                    pattern_library_serializable[key][k] = v
        
        with open(pattern_path, 'w') as f:
            json.dump(pattern_library_serializable, f, indent=2)
        
        # Save cycle info
        cycle_path = save_path / 'cycle_info.json'
        # Convert numpy arrays to lists for JSON serialization
        cycle_info_serializable = {}
        for key, value in self.cycle_info.items():
            if isinstance(value, dict):
                cycle_info_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        cycle_info_serializable[key][k] = v.tolist()
                    else:
                        cycle_info_serializable[key][k] = v
            else:
                cycle_info_serializable[key] = value
        
        with open(cycle_path, 'w') as f:
            json.dump(cycle_info_serializable, f, indent=2)
        
        print(f"\nPipeline saved to {save_dir}")
    
    def load_pipeline(self, load_dir: str):
        """Load trained models and pattern library."""
        load_path = Path(load_dir)
        
        # Load models
        for model_file in load_path.glob('model_*.pkl'):
            model_name = model_file.stem.replace('model_', '')
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            self.models[model_name] = {'model': model}
        
        # Load pattern library
        pattern_path = load_path / 'pattern_library.json'
        if pattern_path.exists():
            with open(pattern_path, 'r') as f:
                self.pattern_library = json.load(f)
        
        # Load cycle info
        cycle_path = load_path / 'cycle_info.json'
        if cycle_path.exists():
            with open(cycle_path, 'r') as f:
                self.cycle_info = json.load(f)
        
        print(f"\nPipeline loaded from {load_dir}")
    
    def predict_and_explain(self, data: pd.DataFrame,
                           model_name: str = 'lightgbm') -> Dict:
        """
        Make predictions with explanations on new data.
        
        Args:
            data: New OHLCV data
            model_name: Model to use for prediction
            
        Returns:
            Dictionary with predictions and explanations
        """
        # Preprocess
        df, _ = self.preprocessor.preprocess_pipeline(data)
        df = create_date_features(df)
        
        # Engineer features
        df = self.feature_engineer.generate_all_features(df)
        
        # Get features
        feature_cols = self.feature_engineer.get_feature_names(df)
        X = df[feature_cols].fillna(0)
        
        # Predict
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train model first.")
        
        model = self.models[model_name]['model']
        predictions = model.predict(X, return_proba=True)
        
        # Get explanations
        explanation = self.explainer.explain_tree_model(model.model, X, feature_cols)
        
        # Generate reports for each prediction
        reports = []
        for i in range(len(predictions)):
            if i < len(X):
                instance_explanation = self.explainer.get_instance_explanation(i, feature_cols)
                
                report = create_prediction_report(
                    prediction=predictions[i],
                    features=X.iloc[i],
                    shap_values=instance_explanation['shap_value'].values if 'shap_value' in instance_explanation else None,
                    pattern_matches=[],
                    cycle_info={}
                )
                reports.append(report)
        
        return {
            'predictions': predictions,
            'explanation': explanation,
            'reports': reports,
        }


def create_sample_data() -> pd.DataFrame:
    """
    Create sample stock data for testing.
    
    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Generate synthetic price data with trend and noise
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 2, n)
    cyclical = 10 * np.sin(np.linspace(0, 8 * np.pi, n))
    
    close = trend + noise + cyclical
    close = np.maximum(close, 1)  # Ensure positive
    
    # OHLC
    open_price = close * (1 + np.random.normal(0, 0.01, n))
    high = np.maximum(open_price, close) * (1 + abs(np.random.normal(0, 0.01, n)))
    low = np.minimum(open_price, close) * (1 - abs(np.random.normal(0, 0.01, n)))
    
    # Volume
    volume = np.random.lognormal(15, 0.5, n)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    return df


if __name__ == '__main__':
    # Example usage
    print("Creating sample data...")
    data = create_sample_data()
    
    print("\nInitializing pipeline...")
    pipeline = StockPatternPipeline()
    
    print("\nRunning full pipeline...")
    results = pipeline.run_full_pipeline(data, target_label='label_up_10d', run_backtest=True)
    
    print("\nSaving pipeline...")
    pipeline.save_pipeline('./outputs')
    
    print("\nDone!")
