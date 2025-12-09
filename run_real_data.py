"""
Run the complete pipeline with real US stock data from Yahoo Finance.
"""

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
from pipeline import StockPatternPipeline
from config import SystemConfig

def main():
    print("=" * 70)
    print("STOCK PATTERN DETECTION - REAL US MARKET DATA")
    print("=" * 70)
    
    # Download SPY data (S&P 500 ETF)
    print("\n📊 Downloading SPY (S&P 500) data from Yahoo Finance...")
    ticker = yf.Ticker("SPY")
    
    # Get maximum available data but focus on recent for faster processing
    data = ticker.history(period="2y")  # 2 years for enough history
    data = data.reset_index()
    
    print(f"✓ Downloaded {len(data)} days of data")
    print(f"  Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nLast few rows:")
    print(data.tail())
    
    # Configure pipeline for faster processing with smaller dataset
    print("\n⚙️  Configuring pipeline...")
    config = SystemConfig()
    config.feature.lookback_windows = [5, 10, 20, 40]  # Reduced for speed
    config.label.prediction_horizons = [5, 10, 20]  # Focus on key horizons
    config.model.tree_n_estimators = 200  # Reduced for speed
    config.pattern.mp_window_size = 15  # Smaller for faster pattern discovery
    config.pattern.mp_top_k_motifs = 5  # Top 5 patterns
    
    # Initialize pipeline
    print("\n🔧 Initializing pipeline...")
    pipeline = StockPatternPipeline(config=config)
    
    # Run full pipeline
    print("\n🚀 Running complete analysis pipeline...")
    print("   This will take a few minutes...\n")
    
    try:
        results = pipeline.run_full_pipeline(
            data=data,
            target_label='label_up_10d',
            run_backtest=True
        )
        
        # Save results
        print("\n💾 Saving results...")
        pipeline.save_pipeline('./outputs/spy_analysis')
        
        # Display summary
        print("\n" + "=" * 70)
        print("📈 RESULTS SUMMARY")
        print("=" * 70)
        
        # Data summary
        df = results['data']
        print(f"\n✓ Processed {len(df)} days of data")
        print(f"✓ Generated {len(pipeline.feature_engineer.get_feature_names(df))} features")
        
        # Pattern summary
        if results['patterns']:
            print(f"\n✓ Discovered {len(results['patterns'])} patterns")
            pattern_summary = pipeline.pattern_discovery.get_pattern_summary()
            if len(pattern_summary) > 0:
                print("\nTop 3 Patterns by Sharpe Ratio:")
                print(pattern_summary.head(3)[['pattern_name', 'occurrences', 'best_horizon_days', 'win_rate', 'sharpe_ratio']].to_string(index=False))
        
        # Cycle summary
        if results['cycles']:
            print(f"\n✓ Cycle detection complete")
            if 'autocorrelation' in results['cycles']:
                acf = results['cycles']['autocorrelation']
                if acf.get('peak_lags'):
                    print(f"  Significant ACF lags: {acf['peak_lags'][:5]}")
            if 'spectral' in results['cycles']:
                spec = results['cycles']['spectral']
                if spec.get('dominant_periods'):
                    periods = [f"{p:.0f}" for p in spec['dominant_periods'][:3]]
                    print(f"  Dominant periods: {', '.join(periods)} days")
        
        # Model summary
        if results['models']:
            print(f"\n✓ Trained {len(results['models'])} model(s)")
            for model_name, model_data in results['models'].items():
                print(f"\n  Model: {model_name}")
                metrics = model_data['metrics']
                for key, value in metrics.items():
                    print(f"    {key}: {value:.4f}")
                
                print(f"\n  Top 5 Features:")
                top_features = model_data['feature_importance'].head(5)
                for _, row in top_features.iterrows():
                    print(f"    {row['feature']}: {row['importance']:.2f}")
        
        # Backtest summary
        if results['backtest']:
            bt = results['backtest']
            print(f"\n✓ Backtest complete")
            print("\n  Performance Metrics:")
            for key, value in bt.metrics.items():
                if isinstance(value, (int, float)):
                    if 'rate' in key or 'return' in key or 'drawdown' in key or 'expectancy' in key:
                        print(f"    {key}: {value:.2%}")
                    else:
                        print(f"    {key}: {value:.4f}")
            
            print(f"\n  Completed {bt.metrics.get('num_trades', 0)} trades")
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nResults saved to: ./outputs/spy_analysis/")
        print("\nFiles created:")
        print("  - model_lightgbm.pkl")
        print("  - feature_importance_lightgbm.csv")
        print("  - pattern_library.json")
        print("  - cycle_info.json")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    results = main()
