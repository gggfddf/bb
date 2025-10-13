# Examples

This directory contains example scripts demonstrating how to use the ML Pattern Discovery system.

## Available Examples

### 1. `train_baseline.py` - Complete Training Pipeline

A comprehensive example showing the entire workflow:

```bash
python train_baseline.py
```

**What it does:**
1. Loads/generates OHLCV data
2. Generates 100+ features across multiple timeframes
3. Creates ATR-based labels for major moves
4. Trains LightGBM with walk-forward cross-validation
5. Evaluates with classification and economic metrics
6. Backtests the trading strategy
7. Saves results, plots, and models to `results/`

**Output:**
- `results/baseline_model.pkl` - Trained model
- `results/predictions.csv` - Model predictions
- `results/metrics.csv` - All evaluation metrics
- `results/feature_importance.csv` - Feature rankings
- `results/*.png` - Visualization plots

## Usage with Your Own Data

To use your own data instead of sample data, modify the data loading section in any example:

```python
# Replace this:
df = create_sample_data(n_days=1500, start_date='2019-01-01')

# With this:
loader = DataLoader(adjusted=True, min_history=100)
df = loader.load_from_csv('path/to/your/data.csv')
```

### Data Format

Your CSV should have these columns:
- `Date` (YYYY-MM-DD format)
- `Open` (opening price)
- `High` (highest price)
- `Low` (lowest price)
- `Close` (closing price)
- `Volume` (trading volume, optional)
- `Adj Close` (adjusted close, optional)

Example CSV:
```
Date,Open,High,Low,Close,Volume
2020-01-02,100.50,102.30,100.00,101.80,1500000
2020-01-03,101.90,103.50,101.50,103.20,1800000
...
```

## Customizing the Pipeline

### Change Target Label

```python
# Predict 5-day moves instead of 10-day
target_label = 'label_5d_up'
```

### Adjust Feature Windows

```python
feature_engine = FeatureEngine(
    windows=[5, 10, 20, 50, 100],  # Custom windows
    include_volume=True
)
```

### Modify Label Thresholds

```python
label_generator = LabelGenerator(atr_window=20)
df = label_generator.generate_labels(
    df,
    horizons=[5, 10, 15],      # Custom horizons
    atr_multiplier=1.5         # More sensitive threshold
)
```

### Change Walk-Forward Parameters

```python
results = trainer.train(
    df=df_clean,
    feature_cols=valid_features,
    label_col=target_label,
    train_size=252*3,   # 3 years training
    test_size=252,      # 1 year testing
    step_size=63,       # 3 months rolling
    lgb_params=custom_params
)
```

## Tips for Production Use

1. **Start Simple**: Run the baseline example first to understand the pipeline
2. **Validate Carefully**: Always use walk-forward validation, never random splits
3. **Monitor Performance**: Track metrics over time as market regimes change
4. **Retrain Regularly**: Models degrade; retrain monthly or quarterly
5. **Transaction Costs**: Include realistic costs (commissions, slippage, spread)
6. **Position Sizing**: Start conservative; this is a signals system, not a complete trading system

## Troubleshooting

**Problem:** "No valid data after removing NaN"
- **Solution:** Increase `min_history` or use more data (features need lookback)

**Problem:** "Insufficient data for walk-forward"
- **Solution:** Reduce `train_size` or `test_size`, or use more historical data

**Problem:** Poor AUC (<0.52)
- **Solution:** Normal for markets; try different features, labels, or assets

**Problem:** High Sharpe but unrealistic
- **Solution:** Check for lookahead bias, ensure transaction costs are included

## Next Steps

After running the examples:

1. **Analyze Feature Importance**: Which features matter most?
2. **Experiment with Labels**: Try different horizons and thresholds
3. **Test on Multiple Assets**: Does it work across stocks/indices?
4. **Add Custom Features**: Domain knowledge can improve performance
5. **Implement Advanced Models**: Try sequence models or ensembles

For more details, see the main [README.md](../README.md).
