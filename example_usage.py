"""
Example usage of the Stock Pattern Detection and Market Cycle Analysis system.
"""

import pandas as pd
import numpy as np
from pipeline import StockPatternPipeline, create_sample_data
from config import SystemConfig


def example_1_basic_usage():
    """
    Example 1: Basic usage with default configuration.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)
    
    # Create or load your data
    # Expected format: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data = create_sample_data()
    
    print(f"\nData shape: {data.shape}")
    print(f"\nFirst few rows:")
    print(data.head())
    
    # Initialize pipeline with default config
    pipeline = StockPatternPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        data=data,
        target_label='label_up_10d',  # Predict 10-day upward moves
        run_backtest=True
    )
    
    # Save trained models and patterns
    pipeline.save_pipeline('./outputs/example1')
    
    print("\n✓ Basic example complete!")
    return results


def example_2_custom_configuration():
    """
    Example 2: Using custom configuration.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 70)
    
    # Create custom configuration
    config = SystemConfig()
    
    # Modify feature engineering settings
    config.feature.lookback_windows = [5, 10, 20, 40, 60]  # Custom windows
    config.feature.use_volume_features = True
    config.feature.use_shape_embeddings = True
    
    # Modify label settings
    config.label.prediction_horizons = [5, 10, 15, 20]
    config.label.atr_multiplier_up = 2.0  # More conservative threshold
    
    # Modify model settings
    config.model.tree_model_type = 'lightgbm'
    config.model.tree_n_estimators = 1000
    config.model.tree_learning_rate = 0.03
    
    # Initialize with custom config
    pipeline = StockPatternPipeline(config=config)
    
    # Load data
    data = create_sample_data()
    
    # Run pipeline
    results = pipeline.run_full_pipeline(data, target_label='label_up_15d')
    
    print("\n✓ Custom configuration example complete!")
    return results


def example_3_step_by_step():
    """
    Example 3: Step-by-step pipeline execution with intermediate inspections.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("=" * 70)
    
    pipeline = StockPatternPipeline()
    data = create_sample_data()
    
    # Step 1: Preprocess
    df = pipeline.load_and_preprocess(data)
    print(f"\nAfter preprocessing: {df.shape}")
    
    # Step 2: Feature Engineering
    df = pipeline.engineer_features(df)
    feature_cols = pipeline.feature_engineer.get_feature_names(df)
    print(f"\nGenerated {len(feature_cols)} features")
    print(f"Sample features: {feature_cols[:5]}")
    
    # Step 3: Label Generation
    df = pipeline.generate_labels(df)
    label_cols = pipeline.label_generator.get_label_columns(df, label_type='move')
    print(f"\nGenerated {len(label_cols)} move labels")
    
    # Step 4: Pattern Discovery
    patterns = pipeline.discover_patterns(df)
    print(f"\nDiscovered {len(patterns)} patterns")
    
    # Step 5: Cycle Detection
    cycles = pipeline.detect_cycles(df)
    print(f"\nCycle detection complete")
    if 'autocorrelation' in cycles:
        print(f"  - Significant ACF lags: {cycles['autocorrelation'].get('significant_acf_lags', [])[:5]}")
    
    # Step 6: Train Models
    models = pipeline.train_models(df, target_label='label_up_10d')
    print(f"\nTrained {len(models)} models")
    
    print("\n✓ Step-by-step example complete!")
    return df, patterns, cycles, models


def example_4_real_world_data():
    """
    Example 4: Using real-world data (requires yfinance).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Real-World Data")
    print("=" * 70)
    
    try:
        import yfinance as yf
        
        # Download stock data
        print("Downloading SPY data from Yahoo Finance...")
        ticker = yf.Ticker("SPY")
        data = ticker.history(period="5y")
        
        # Reset index to have Date as column
        data = data.reset_index()
        data = data.rename(columns={'index': 'Date'})
        
        print(f"\nDownloaded {len(data)} days of data")
        
        # Initialize pipeline
        pipeline = StockPatternPipeline()
        
        # Run pipeline
        results = pipeline.run_full_pipeline(
            data=data,
            target_label='label_up_10d',
            run_backtest=True
        )
        
        # Save results
        pipeline.save_pipeline('./outputs/spy_analysis')
        
        print("\n✓ Real-world data example complete!")
        return results
        
    except ImportError:
        print("\n⚠ yfinance not installed. Install with: pip install yfinance")
        print("Using synthetic data instead...")
        return example_1_basic_usage()


def example_5_pattern_analysis():
    """
    Example 5: Deep dive into pattern analysis.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Pattern Analysis")
    print("=" * 70)
    
    pipeline = StockPatternPipeline()
    data = create_sample_data()
    
    # Preprocess and generate features
    df = pipeline.load_and_preprocess(data)
    df = pipeline.engineer_features(df)
    
    # Discover patterns
    patterns = pipeline.discover_patterns(df)
    
    # Get pattern summary
    summary = pipeline.pattern_discovery.get_pattern_summary()
    print("\nPattern Summary:")
    print(summary.to_string(index=False))
    
    # Filter significant patterns
    significant = pipeline.pattern_discovery.filter_significant_patterns(
        min_occurrences=5,
        min_confidence=0.55
    )
    
    print(f"\nFound {len(significant)} significant patterns")
    
    # Create pattern report
    from explainability import PatternLibraryExplainer
    report = PatternLibraryExplainer.create_pattern_report(significant)
    print("\nTop Patterns Report:")
    print(report.head(10).to_string(index=False))
    
    print("\n✓ Pattern analysis example complete!")
    return patterns, significant


def example_6_prediction_and_explanation():
    """
    Example 6: Make predictions with explanations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Predictions with Explanations")
    print("=" * 70)
    
    # Train pipeline
    pipeline = StockPatternPipeline()
    data = create_sample_data()
    
    print("Training pipeline...")
    results = pipeline.run_full_pipeline(data, target_label='label_up_10d', run_backtest=False)
    
    # Create new data for prediction
    new_data = create_sample_data()  # In practice, this would be new/recent data
    new_data = new_data.tail(30)  # Last 30 days
    
    print("\nMaking predictions on new data...")
    predictions = pipeline.predict_and_explain(new_data, model_name='lightgbm')
    
    print(f"\nGenerated {len(predictions['predictions'])} predictions")
    
    # Show sample prediction report
    if predictions['reports']:
        print("\nSample Prediction Report:")
        print(predictions['reports'][0])
    
    print("\n✓ Prediction and explanation example complete!")
    return predictions


def example_7_walk_forward_validation():
    """
    Example 7: Rigorous walk-forward cross-validation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Walk-Forward Cross-Validation")
    print("=" * 70)
    
    from evaluation import WalkForwardValidator
    from config import EvaluationConfig
    
    # Create data
    data = create_sample_data()
    
    # Prepare pipeline
    pipeline = StockPatternPipeline()
    df = pipeline.load_and_preprocess(data)
    df = pipeline.engineer_features(df)
    df = pipeline.generate_labels(df)
    
    # Get features and labels
    feature_cols = pipeline.feature_engineer.get_feature_names(df)
    X = df[feature_cols].fillna(0)
    y = df['label_up_10d'].fillna(0)
    
    # Initialize validator
    eval_config = EvaluationConfig()
    eval_config.train_size_years = 2
    eval_config.validation_size_months = 3
    eval_config.test_size_months = 3
    eval_config.step_size_months = 3
    
    validator = WalkForwardValidator(eval_config)
    
    # Define training and prediction functions
    def train_func(X_train, y_train, X_val, y_val):
        from models import BaselineTreeModel
        from config import ModelConfig
        model = BaselineTreeModel(ModelConfig(), model_type='lightgbm')
        model.train(X_train, y_train, X_val, y_val)
        return model
    
    def predict_func(model, X):
        return (model.predict(X) > 0.5).astype(int)
    
    def metric_func(y_true, y_pred):
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
    
    # Run cross-validation
    print("\nRunning walk-forward cross-validation...")
    cv_results = validator.cross_validate(df, X, y, train_func, predict_func, metric_func)
    
    print(f"\nCompleted {len(cv_results['splits_info'])} folds")
    print("\nAverage Test Metrics:")
    for key, value in cv_results['avg_test_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Walk-forward validation example complete!")
    return cv_results


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("STOCK PATTERN DETECTION - EXAMPLE USAGE")
    print("=" * 70)
    
    # Run examples
    print("\nRunning examples...")
    
    # Example 1: Basic usage
    results_1 = example_1_basic_usage()
    
    # Example 2: Custom configuration
    # results_2 = example_2_custom_configuration()
    
    # Example 3: Step-by-step
    # df, patterns, cycles, models = example_3_step_by_step()
    
    # Example 5: Pattern analysis
    # patterns, significant = example_5_pattern_analysis()
    
    # Example 6: Predictions with explanations
    # predictions = example_6_prediction_and_explanation()
    
    # Uncomment to run other examples
    # results_4 = example_4_real_world_data()
    # cv_results = example_7_walk_forward_validation()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
