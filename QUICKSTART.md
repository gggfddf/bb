# Quick Start Guide

Get up and running in 5 minutes!

## Installation

```bash
# Clone repository
git clone <repository-url>
cd ml-pattern-discovery

# Install dependencies
pip install -r requirements.txt

# Verify installation
python tests/test_pipeline.py
```

## Run Your First Model

```bash
cd examples
python train_baseline.py
```

This will:
- Generate sample stock data
- Create 100+ features
- Train a LightGBM model
- Evaluate performance
- Save results to `results/`

**Expected runtime:** 2-5 minutes  
**Expected AUC:** 0.55-0.65

## View Results

After running, check:

```bash
results/
├── baseline_model.pkl          # Trained model
├── predictions.csv             # Model predictions
├── metrics.csv                 # Performance metrics
├── feature_importance.csv      # Top features
├── predictions_plot.png        # Signals visualization
├── feature_importance.png      # Feature rankings
└── backtest_plot.png          # Strategy performance
```

## Use Your Own Data

Replace the sample data in `train_baseline.py`:

```python
# Instead of:
df = create_sample_data(n_days=1500)

# Use your CSV:
loader = DataLoader(adjusted=True)
df = loader.load_from_csv('data/AAPL.csv')
```

**CSV format required:**
```
Date,Open,High,Low,Close,Volume
2020-01-02,100.50,102.30,100.00,101.80,1500000
2020-01-03,101.90,103.50,101.50,103.20,1800000
```

## Simple Example (Python)

```python
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngine
from src.labels.generator import LabelGenerator
from src.models.trainer import ModelTrainer

# 1. Load data
loader = DataLoader()
df = loader.load_from_csv('data/stock.csv')

# 2. Generate features
engine = FeatureEngine(windows=[5, 10, 20, 40])
df = engine.generate_features(df)
features = engine.get_feature_names(df)

# 3. Generate labels
labeler = LabelGenerator(atr_window=20)
df = labeler.generate_labels(df, horizons=[10], atr_multiplier=2.0)

# 4. Train model
trainer = ModelTrainer(walk_forward=True)
results = trainer.train(
    df=df,
    feature_cols=features,
    label_col='label_10d_up',
    train_size=504,
    test_size=252
)

# 5. Evaluate
from src.evaluation.metrics import ModelEvaluator
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_classification(
    y_true=results['y_test'],
    y_pred=results['predictions'],
    y_pred_proba=results['probabilities']
)

print(f"AUC: {metrics['auc']:.3f}")
print(f"Precision@5%: {metrics['precision@top5pct']:.3f}")
```

## Next Steps

1. ✅ **Read the full [README.md](README.md)** - Understand the system
2. 📊 **Explore [FORMULAS.md](FORMULAS.md)** - See all feature/label definitions
3. 🔧 **Customize features** - Add your own in `src/features/engineering.py`
4. 🎯 **Try different labels** - Experiment with horizons and thresholds
5. 📈 **Test on real data** - Use actual stock data from your broker/API

## Common Issues

**Problem:** Module not found  
**Fix:** Run from project root: `cd /path/to/ml-pattern-discovery`

**Problem:** Not enough data  
**Fix:** Need at least 600+ days for walk-forward CV

**Problem:** Low AUC  
**Fix:** Financial markets are noisy - AUC > 0.55 is good!

## Need Help?

- 📖 **Full docs**: See [README.md](README.md)
- 📝 **Examples**: Check [examples/README.md](examples/README.md)
- 🔢 **Formulas**: Read [FORMULAS.md](FORMULAS.md)
- 🐛 **Issues**: Open a GitHub issue

---

**Ready to build?** Start with `python examples/train_baseline.py`
