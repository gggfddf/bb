"""
Basic integration tests to verify the pipeline works.
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngine
from src.labels.generator import LabelGenerator
from src.utils.helpers import create_sample_data


def test_data_loader():
    """Test data loading and preprocessing."""
    print("Testing DataLoader...")
    
    # Create sample data
    df = create_sample_data(n_days=500)
    
    # Load and preprocess
    loader = DataLoader(adjusted=True, min_history=50)
    df = loader.load_from_dataframe(df)
    
    # Check required columns
    assert 'Close' in df.columns
    assert 'Open' in df.columns
    assert 'High' in df.columns
    assert 'Low' in df.columns
    assert 'log_return' in df.columns
    
    print("  ✓ DataLoader works correctly")


def test_feature_engineering():
    """Test feature generation."""
    print("Testing FeatureEngine...")
    
    # Create and load data
    df = create_sample_data(n_days=500)
    loader = DataLoader()
    df = loader.load_from_dataframe(df)
    
    # Generate features
    feature_engine = FeatureEngine(windows=[5, 10, 20])
    df = feature_engine.generate_features(df)
    
    # Check features were created
    feature_cols = feature_engine.get_feature_names(df)
    assert len(feature_cols) > 0
    
    # Check specific features exist
    assert any('price_' in col for col in feature_cols)
    assert any('vol_' in col for col in feature_cols)
    
    print(f"  ✓ Generated {len(feature_cols)} features")


def test_label_generation():
    """Test label generation."""
    print("Testing LabelGenerator...")
    
    # Create and load data
    df = create_sample_data(n_days=500)
    loader = DataLoader()
    df = loader.load_from_dataframe(df)
    
    # Generate labels
    label_generator = LabelGenerator(atr_window=20)
    df = label_generator.generate_labels(df, horizons=[5, 10], atr_multiplier=2.0)
    
    # Check labels were created
    label_cols = label_generator.get_label_columns(df)
    assert len(label_cols) > 0
    assert 'label_5d_up' in df.columns
    assert 'label_10d_up' in df.columns
    
    # Check label statistics
    stats = label_generator.get_label_stats(df)
    assert len(stats) > 0
    
    print(f"  ✓ Generated {len(label_cols)} labels")


def test_full_pipeline():
    """Test complete pipeline end-to-end."""
    print("Testing full pipeline...")
    
    # Create data
    df = create_sample_data(n_days=500)
    
    # Load
    loader = DataLoader()
    df = loader.load_from_dataframe(df)
    
    # Features
    feature_engine = FeatureEngine(windows=[5, 10, 20])
    df = feature_engine.generate_features(df)
    feature_cols = feature_engine.get_feature_names(df)
    
    # Labels
    label_generator = LabelGenerator(atr_window=20)
    df = label_generator.generate_labels(df, horizons=[10], atr_multiplier=2.0)
    
    # Check we have data ready for training
    df_clean = df[feature_cols + ['label_10d_up']].dropna()
    
    assert len(df_clean) > 100
    assert len(feature_cols) > 10
    
    print(f"  ✓ Pipeline complete: {len(df_clean)} samples, {len(feature_cols)} features")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60 + "\n")
    
    try:
        test_data_loader()
        test_feature_engineering()
        test_label_generation()
        test_full_pipeline()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
