"""
Basic test to verify the system can be imported and initialized.
Run with: python test_basic.py
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config")
    except Exception as e:
        print(f"✗ config: {e}")
        return False
    
    try:
        import data_preprocessing
        print("✓ data_preprocessing")
    except Exception as e:
        print(f"✗ data_preprocessing: {e}")
        return False
    
    try:
        import feature_engineering
        print("✓ feature_engineering")
    except Exception as e:
        print(f"✗ feature_engineering: {e}")
        return False
    
    try:
        import label_generation
        print("✓ label_generation")
    except Exception as e:
        print(f"✗ label_generation: {e}")
        return False
    
    try:
        import models
        print("✓ models")
    except Exception as e:
        print(f"✗ models: {e}")
        return False
    
    try:
        import pattern_discovery
        print("✓ pattern_discovery")
    except Exception as e:
        print(f"✗ pattern_discovery: {e}")
        return False
    
    try:
        import cycle_detection
        print("✓ cycle_detection")
    except Exception as e:
        print(f"✗ cycle_detection: {e}")
        return False
    
    try:
        import explainability
        print("✓ explainability")
    except Exception as e:
        print(f"✗ explainability: {e}")
        return False
    
    try:
        import evaluation
        print("✓ evaluation")
    except Exception as e:
        print(f"✗ evaluation: {e}")
        return False
    
    try:
        import pipeline
        print("✓ pipeline")
    except Exception as e:
        print(f"✗ pipeline: {e}")
        return False
    
    return True


def test_initialization():
    """Test that pipeline can be initialized."""
    print("\nTesting pipeline initialization...")
    
    try:
        from pipeline import StockPatternPipeline
        from config import SystemConfig
        
        # Test with default config
        pipeline = StockPatternPipeline()
        print("✓ Pipeline initialized with default config")
        
        # Test with custom config
        config = SystemConfig()
        pipeline = StockPatternPipeline(config=config)
        print("✓ Pipeline initialized with custom config")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        traceback.print_exc()
        return False


def test_sample_data():
    """Test that sample data can be created."""
    print("\nTesting sample data creation...")
    
    try:
        from pipeline import create_sample_data
        
        data = create_sample_data()
        print(f"✓ Created sample data with {len(data)} rows")
        
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                print(f"✗ Missing column: {col}")
                return False
        
        print("✓ Sample data has all required columns")
        return True
    except Exception as e:
        print(f"✗ Sample data creation failed: {e}")
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test basic preprocessing."""
    print("\nTesting preprocessing...")
    
    try:
        from pipeline import create_sample_data, StockPatternPipeline
        
        data = create_sample_data()
        pipeline = StockPatternPipeline()
        
        df = pipeline.load_and_preprocess(data)
        print(f"✓ Preprocessed data: {len(df)} rows")
        
        # Check for ATR columns
        atr_cols = [col for col in df.columns if 'ATR' in col]
        if atr_cols:
            print(f"✓ ATR columns created: {atr_cols}")
        else:
            print("⚠ No ATR columns found")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature engineering."""
    print("\nTesting feature engineering...")
    
    try:
        from pipeline import create_sample_data, StockPatternPipeline
        
        data = create_sample_data()
        pipeline = StockPatternPipeline()
        
        df = pipeline.load_and_preprocess(data)
        df = pipeline.engineer_features(df)
        
        feature_names = pipeline.feature_engineer.get_feature_names(df)
        print(f"✓ Generated {len(feature_names)} features")
        
        # Show sample features
        print(f"  Sample features: {feature_names[:5]}")
        
        return True
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        traceback.print_exc()
        return False


def test_label_generation():
    """Test label generation."""
    print("\nTesting label generation...")
    
    try:
        from pipeline import create_sample_data, StockPatternPipeline
        
        data = create_sample_data()
        pipeline = StockPatternPipeline()
        
        df = pipeline.load_and_preprocess(data)
        df = pipeline.engineer_features(df)
        df = pipeline.generate_labels(df)
        
        label_cols = pipeline.label_generator.get_label_columns(df, label_type='move')
        print(f"✓ Generated {len(label_cols)} move labels")
        
        # Show sample labels
        print(f"  Sample labels: {label_cols[:3]}")
        
        return True
    except Exception as e:
        print(f"✗ Label generation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("STOCK PATTERN DETECTION SYSTEM - BASIC TESTS")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Initialization", test_initialization),
        ("Sample Data", test_sample_data),
        ("Preprocessing", test_preprocessing),
        ("Feature Engineering", test_feature_engineering),
        ("Label Generation", test_label_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nPassed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
