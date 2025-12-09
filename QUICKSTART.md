# Quick Start Guide

## 30-Second Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the example
python example_usage.py

# Check the results
ls outputs/
```

## 5-Minute Tutorial

### Step 1: Prepare Your Data

Your data should be a CSV or DataFrame with these columns:
```
Date, Open, High, Low, Close, Volume
```

Example:
```python
import pandas as pd

# Load your data
data = pd.read_csv('AAPL.csv')

# Or download from Yahoo Finance
import yfinance as yf
data = yf.Ticker("AAPL").history(period="5y").reset_index()
```

### Step 2: Run the Pipeline

```python
from pipeline import StockPatternPipeline

# Initialize
pipeline = StockPatternPipeline()

# Run complete analysis
results = pipeline.run_full_pipeline(
    data=data,
    target_label='label_up_10d',  # Predict 10-day upward moves
    run_backtest=True
)
```

### Step 3: View Results

```python
# Backtest metrics
print(results['backtest'].metrics)

# Pattern summary
from explainability import PatternLibraryExplainer
pattern_report = PatternLibraryExplainer.create_pattern_report(results['patterns'])
print(pattern_report)

# Feature importance
model = results['models']['lightgbm']
print(model['feature_importance'].head(10))
```

### Step 4: Save and Load

```python
# Save trained pipeline
pipeline.save_pipeline('./my_models')

# Later, load it
pipeline.load_pipeline('./my_models')

# Make predictions on new data
predictions = pipeline.predict_and_explain(new_data)
```

## Common Use Cases

### Use Case 1: Find Best Patterns
```python
patterns = pipeline.discover_patterns(df)
significant = pipeline.pattern_discovery.filter_significant_patterns(
    min_occurrences=10,
    min_confidence=0.60
)
print(f"Found {len(significant)} high-quality patterns")
```

### Use Case 2: Detect Market Regime
```python
cycles = pipeline.detect_cycles(df)
if 'hmm' in cycles:
    current_regime = cycles['hmm']['regime_names'][-1]
    print(f"Current market regime: {current_regime}")
```

### Use Case 3: Optimize Trading Threshold
```python
from evaluation import Backtester

backtester = Backtester(pipeline.config.evaluation)
optimization = backtester.optimize_threshold(df, predictions)
best_threshold = optimization['best_threshold']
print(f"Optimal threshold: {best_threshold:.3f}")
```

### Use Case 4: Custom Configuration
```python
from config import SystemConfig

config = SystemConfig()
config.label.atr_multiplier_up = 2.0  # More conservative
config.model.tree_n_estimators = 1000  # More trees
config.feature.lookback_windows = [5, 10, 20, 40]  # Specific windows

pipeline = StockPatternPipeline(config=config)
```

## Key Parameters to Adjust

### For Better Accuracy
- Increase `tree_n_estimators` (500 → 1000)
- Use more features: `lookback_windows = [3,5,10,20,40,60,80]`
- Longer training period: `train_size_years = 5`

### For Faster Training
- Decrease `tree_n_estimators` (500 → 100)
- Fewer features: `lookback_windows = [10, 20, 40]`
- Shorter training period: `train_size_years = 2`

### For More Conservative Signals
- Increase `atr_multiplier_up` (1.5 → 2.0)
- Higher `pattern_confidence_threshold` (0.6 → 0.7)
- Stricter `breakout_volume_multiplier` (1.5 → 2.0)

### For More Aggressive Signals
- Decrease `atr_multiplier_up` (1.5 → 1.0)
- Lower `pattern_confidence_threshold` (0.6 → 0.5)
- Relaxed `breakout_volume_multiplier` (1.5 → 1.2)

## Troubleshooting

**Problem**: ImportError for numpy/pandas
```bash
pip install numpy pandas scipy
```

**Problem**: No module named 'lightgbm'
```bash
pip install lightgbm
```

**Problem**: SHAP not available
```bash
pip install shap
```

**Problem**: Memory error
```python
# Reduce data or features
config.feature.lookback_windows = [10, 20, 40]  # Fewer features
data = data.tail(1000)  # Use less data
```

**Problem**: Training too slow
```python
config.model.tree_n_estimators = 100  # Fewer trees
config.evaluation.train_size_years = 2  # Less training data
```

## Example Outputs

### Pattern Report
```
Pattern: motif_3
Observed 47 times in the dataset
Best predictive power at 10-day horizon: 63.8% win rate, 3.2% mean return
```

### Backtest Summary
```
Total Return:      34.52%
Sharpe Ratio:      0.44
Win Rate:          58.33%
Max Drawdown:      -12.34%
```

### Top Features
```
momentum_20        1243.56
volatility_10       987.32
rsi_14             856.91
```

## Next Steps

1. ✅ Run `python example_usage.py`
2. ✅ Check outputs in `./outputs/`
3. ✅ Try with your own data
4. ✅ Adjust configuration
5. ✅ Review backtest results
6. ✅ Analyze discovered patterns

## Full Documentation

- **README.md** - Complete system overview
- **INSTALLATION.md** - Detailed setup guide
- **PROJECT_SUMMARY.md** - Technical implementation details
- **example_usage.py** - 7 different examples

## Getting Help

1. Check error messages carefully
2. Review INSTALLATION.md for setup issues
3. Look at example_usage.py for usage patterns
4. Verify data format matches requirements

---

**Ready to analyze stocks? Start with:** `python example_usage.py`
