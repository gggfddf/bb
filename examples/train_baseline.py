"""
Example: Train baseline LightGBM model with walk-forward validation.

This script demonstrates the complete pipeline:
1. Load/generate OHLCV data
2. Generate features
3. Generate labels
4. Train model with walk-forward CV
5. Evaluate performance
6. Backtest strategy
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import DataLoader
from src.features.engineering import FeatureEngine
from src.labels.generator import LabelGenerator
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator, BacktestEngine
from src.utils.helpers import create_sample_data, plot_results, plot_feature_importance, plot_backtest_results


def main():
    """Run complete training pipeline."""
    
    print("="*80)
    print("ML PATTERN DISCOVERY PIPELINE - BASELINE MODEL")
    print("="*80)
    
    # ============================================================================
    # 1. Load Data
    # ============================================================================
    print("\n[1/7] Loading data...")
    
    # For this example, we'll create sample data
    # In production, you would load from CSV or API:
    # loader = DataLoader()
    # df = loader.load_from_csv('data/AAPL.csv')
    
    df = create_sample_data(n_days=1500, start_date='2019-01-01')
    
    # Load and preprocess
    loader = DataLoader(adjusted=True, min_history=100)
    df = loader.load_from_dataframe(df)
    
    print(f"  Loaded {len(df)} days of data")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # ============================================================================
    # 2. Generate Features
    # ============================================================================
    print("\n[2/7] Generating features...")
    
    feature_engine = FeatureEngine(
        windows=[3, 5, 10, 20, 40, 80],
        include_volume=True
    )
    
    df = feature_engine.generate_features(df)
    feature_cols = feature_engine.get_feature_names(df)
    
    print(f"  Generated {len(feature_cols)} features")
    print(f"  Feature categories:")
    
    # Count features by category
    categories = {}
    for col in feature_cols:
        prefix = col.split('_')[0]
        categories[prefix] = categories.get(prefix, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count}")
    
    # ============================================================================
    # 3. Generate Labels
    # ============================================================================
    print("\n[3/7] Generating labels...")
    
    label_generator = LabelGenerator(atr_window=20)
    
    df = label_generator.generate_labels(
        df,
        horizons=[3, 5, 10, 20],
        atr_multiplier=2.0
    )
    
    label_cols = label_generator.get_label_columns(df)
    print(f"  Generated {len(label_cols)} label columns")
    
    # Show label statistics
    label_stats = label_generator.get_label_stats(df)
    print("\n  Label Statistics:")
    print(label_stats.to_string(index=False))
    
    # ============================================================================
    # 4. Prepare Training Data
    # ============================================================================
    print("\n[4/7] Preparing training data...")
    
    # For this example, we'll predict 10-day major moves (binary: UP or not)
    target_label = 'label_10d_up'
    
    # Remove rows with NaN in features or labels
    valid_features = [col for col in feature_cols if col in df.columns]
    df_clean = df[valid_features + [target_label]].dropna()
    
    print(f"  Clean data: {len(df_clean)} samples")
    print(f"  Target label: {target_label}")
    print(f"  Positive class rate: {df_clean[target_label].mean():.2%}")
    
    # ============================================================================
    # 5. Train Model with Walk-Forward CV
    # ============================================================================
    print("\n[5/7] Training model with walk-forward validation...")
    
    trainer = ModelTrainer(
        model_type='lightgbm',
        walk_forward=True,
        verbose=1
    )
    
    # LightGBM parameters (optional - uses sensible defaults if not provided)
    lgb_params = {
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1
    }
    
    results = trainer.train(
        df=df_clean,
        feature_cols=valid_features,
        label_col=target_label,
        train_size=504,  # ~2 years
        test_size=252,   # ~1 year
        step_size=126,   # ~6 months
        lgb_params=lgb_params
    )
    
    print(f"\n  Walk-forward splits: {results['n_splits']}")
    print(f"  Total test predictions: {len(results['predictions'])}")
    
    # ============================================================================
    # 6. Evaluate Model
    # ============================================================================
    print("\n[6/7] Evaluating model...")
    
    evaluator = ModelEvaluator()
    
    # Classification metrics
    classification_metrics = evaluator.evaluate_classification(
        y_true=results['y_test'],
        y_pred=results['predictions'],
        y_pred_proba=results['probabilities']
    )
    
    print("\n  Classification Metrics:")
    for metric, value in sorted(classification_metrics.items()):
        print(f"    {metric:25s}: {value:.4f}")
    
    # Get feature importance
    print("\n  Top 20 Most Important Features:")
    importance_df = trainer.get_feature_importance(top_n=20)
    print(importance_df.to_string(index=False))
    
    # ============================================================================
    # 7. Backtest Strategy
    # ============================================================================
    print("\n[7/7] Backtesting strategy...")
    
    # Create signals from predictions (aligned with test indices)
    test_indices = results['test_indices']
    signals = pd.Series(results['predictions'], index=test_indices)
    
    # Get forward returns for test period
    # Note: In practice, you'd use next-day returns to avoid lookahead
    df_clean['forward_return'] = df_clean['Close'].pct_change().shift(-1)
    
    backtest_engine = BacktestEngine(
        transaction_cost=0.001,  # 0.1% per trade
        slippage=0.0005          # 0.05% slippage
    )
    
    # Backtest
    backtest_df = backtest_engine.backtest_signals(
        df=df_clean.loc[test_indices],
        signals=signals,
        returns_col='forward_return',
        position_size=1.0
    )
    
    # Analyze backtest
    backtest_metrics = backtest_engine.analyze_backtest(backtest_df)
    
    print("\n  Backtest Metrics:")
    for metric, value in sorted(backtest_metrics.items()):
        if isinstance(value, float):
            if 'rate' in metric or 'ratio' in metric:
                print(f"    {metric:25s}: {value:.4f}")
            else:
                print(f"    {metric:25s}: {value:.2%}")
        else:
            print(f"    {metric:25s}: {value}")
    
    # ============================================================================
    # Save Results
    # ============================================================================
    print("\n" + "="*80)
    print("Saving results...")
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_dir / 'baseline_model.pkl'
    trainer.save_model(model_path)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': test_indices,
        'actual': results['y_test'],
        'predicted': results['predictions'],
        'probability': results['probabilities']
    })
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Save metrics
    all_metrics = {**classification_metrics, **backtest_metrics}
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
    
    # Save feature importance
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    print(f"  Results saved to {output_dir}/")
    
    # ============================================================================
    # Generate Plots (optional)
    # ============================================================================
    print("\nGenerating plots...")
    
    try:
        # Plot predictions on test data
        test_df = df_clean.loc[test_indices].copy()
        test_df['signal'] = signals
        
        plot_results(
            test_df,
            predictions=test_df['signal'],
            title='Model Predictions on Test Set',
            save_path=output_dir / 'predictions_plot.png'
        )
        
        # Plot feature importance
        plot_feature_importance(
            importance_df,
            top_n=20,
            title='Top 20 Feature Importance',
            save_path=output_dir / 'feature_importance.png'
        )
        
        # Plot backtest results
        plot_backtest_results(
            backtest_df,
            title='Backtest Performance',
            save_path=output_dir / 'backtest_plot.png'
        )
        
        print(f"  Plots saved to {output_dir}/")
        
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  Model Type: LightGBM with Walk-Forward CV")
    print(f"  Data: {len(df)} days ({df.index[0]} to {df.index[-1]})")
    print(f"  Features: {len(valid_features)}")
    print(f"  Target: {target_label}")
    print(f"  Test Samples: {len(results['predictions'])}")
    print(f"  AUC: {classification_metrics.get('auc', 0):.4f}")
    print(f"  Precision@Top5%: {classification_metrics.get('precision@top5pct', 0):.4f}")
    print(f"  Sharpe Ratio: {backtest_metrics.get('sharpe', 0):.4f}")
    print(f"  CAGR: {backtest_metrics.get('cagr', 0):.2%}")
    print(f"  Max Drawdown: {backtest_metrics.get('max_drawdown', 0):.2%}")
    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
