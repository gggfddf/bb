# ML Pattern Discovery for Stock Market Data

A production-ready machine learning system for discovering and learning patterns in daily OHLCV stock data. This system can detect consolidations, breakouts, reversals, bull zones, cycles, and multi-day pattern precursors.

## 🎯 Overview

This project implements a comprehensive ML pipeline that:

- **Ingests daily OHLCV data** (Open, High, Low, Close, Volume)
- **Discovers recurring patterns** through unsupervised and supervised learning
- **Learns time-based cycles** and estimates lead times for major moves
- **Predicts future price movements** with interpretable signals
- **Provides professional evaluation** with walk-forward validation and realistic backtesting

## 🏗️ Architecture

```
src/
├── data/           # Data loading and preprocessing
├── features/       # Multi-horizon feature engineering
├── labels/         # ATR-based label generation
├── models/         # Model training with walk-forward CV
├── evaluation/     # Metrics and backtesting
└── utils/          # Visualization and helpers

examples/
└── train_baseline.py   # Complete pipeline example
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

### Run Example Pipeline

```bash
cd examples
python train_baseline.py
```

This will:
1. Generate sample OHLCV data (or load your own)
2. Generate 100+ features across multiple timeframes
3. Create ATR-based labels for major moves
4. Train LightGBM model with walk-forward validation
5. Evaluate with classification and economic metrics
6. Backtest the trading strategy
7. Save results and plots to `results/`

## 📊 Features

### Multi-Horizon Feature Engineering

The system generates **100+ features** across multiple lookback windows (3, 5, 10, 20, 40, 80 days):

**Price Features:**
- Momentum, moving averages (SMA/EMA), distance from MA
- High/low ranges, pullback depth, rally height
- Normalized price positions

**Volume Features:**
- Volume surges, z-scores, ratios
- On-Balance Volume (OBV) and slopes
- Volume-weighted indicators

**Structural Indicators:**
- Bollinger Bands (width and position)
- RSI, MACD, Stochastic oscillators
- Volatility (ATR, standard deviation)

**Pattern Features:**
- Consecutive up/down days
- Local extrema (fractal tops/bottoms)
- Range expansion indicators

**Time/Cycle Features:**
- Days since local high/low
- Autocorrelation at multiple lags
- Seasonality (day of week, month)

### ATR-Based Label Generation

**Scale-invariant labeling** using Average True Range (ATR):

1. **Major Move Labels** (multi-horizon: 3, 5, 10, 20 days)
   - UP: max_return ≥ threshold before -threshold
   - DOWN: min_return ≤ -threshold before +threshold
   - Threshold = k × ATR (default k=2.0)

2. **Pattern Labels**
   - Consolidation: narrow range for N days
   - Breakout: price > consolidation_high + β×ATR
   - False Breakout: breakout that fails to hold
   - Reversal: significant directional change

3. **Regime Labels**
   - Bull/Bear/Neutral zones based on returns and volatility

### Model Training

**LightGBM Baseline** with professional features:
- Walk-forward cross-validation (no lookahead bias)
- Automatic class imbalance handling
- Early stopping and regularization
- Feature importance analysis
- Model persistence

### Evaluation Metrics

**Classification Metrics:**
- AUC, Precision@k (top 5%, 10%, 20%)
- F1, Recall, Balanced Accuracy

**Economic Metrics:**
- CAGR, Sharpe Ratio, Sortino Ratio
- Maximum Drawdown
- Win Rate, Average Win/Loss
- Expectancy, Information Ratio

## 💡 Usage Examples

### Load and Prepare Data

```python
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngine
from src.labels.generator import LabelGenerator

# Load data
loader = DataLoader(adjusted=True, min_history=100)
df = loader.load_from_csv('data/AAPL.csv')

# Generate features
feature_engine = FeatureEngine(windows=[3, 5, 10, 20, 40, 80])
df = feature_engine.generate_features(df)
feature_cols = feature_engine.get_feature_names(df)

# Generate labels
label_generator = LabelGenerator(atr_window=20)
df = label_generator.generate_labels(df, horizons=[3, 5, 10, 20], atr_multiplier=2.0)
```

### Train Model

```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(model_type='lightgbm', walk_forward=True, verbose=1)

results = trainer.train(
    df=df,
    feature_cols=feature_cols,
    label_col='label_10d_up',
    train_size=504,   # 2 years
    test_size=252,    # 1 year
    step_size=126     # 6 months rolling
)
```

### Evaluate Performance

```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator()

# Classification metrics
classification_metrics = evaluator.evaluate_classification(
    y_true=results['y_test'],
    y_pred=results['predictions'],
    y_pred_proba=results['probabilities']
)

# Economic metrics (backtest)
economic_metrics = evaluator.evaluate_economic(
    returns=df.loc[test_indices, 'return'],
    predictions=results['predictions'],
    transaction_cost=0.001
)

evaluator.print_evaluation_report(classification_metrics, economic_metrics)
```

### Backtest Strategy

```python
from src.evaluation.metrics import BacktestEngine

backtest_engine = BacktestEngine(transaction_cost=0.001, slippage=0.0005)

backtest_df = backtest_engine.backtest_signals(
    df=df,
    signals=predictions,
    returns_col='return'
)

metrics = backtest_engine.analyze_backtest(backtest_df)
```

## 🎓 Design Principles

### No Lookahead Bias
- Features computed strictly from t ≤ current
- Walk-forward validation with chronological splits
- Realistic execution assumptions (next-day open)

### Scale Invariance
- ATR-based thresholds adapt to volatility
- Works across different price ranges and assets
- Normalized features where appropriate

### Robustness
- Multiple validation windows
- Transaction costs and slippage included
- Class imbalance handling
- Regularization to prevent overfitting

### Interpretability
- Feature importance from tree models
- Clear pattern definitions
- Economic metrics (not just statistical)

## 📈 Expected Performance

On well-behaved data, you can expect:

- **AUC:** 0.55-0.65 (better than random, realistic for markets)
- **Precision@Top5%:** 40-60% (enrichment in top predictions)
- **Sharpe Ratio:** 0.5-1.5 (after costs)
- **Win Rate:** 45-55%

**Note:** Financial markets are noisy and non-stationary. Results vary by asset, timeframe, and market regime. Always validate on out-of-sample data.

## 🔧 Customization

### Add Custom Features

```python
class CustomFeatureEngine(FeatureEngine):
    def _add_custom_features(self, df):
        # Add your custom features
        df['my_feature'] = ...
        return df
```

### Custom Label Strategies

```python
label_generator = LabelGenerator(atr_window=20)
# Adjust multipliers and horizons
df = label_generator.generate_labels(
    df, 
    horizons=[5, 15, 30], 
    atr_multiplier=1.5
)
```

### Different Models

The trainer supports different models by changing `model_type`:
- `lightgbm` (default, recommended)
- `xgboost`
- `catboost`

## 📝 Label Definitions Reference

### Major Move Label (H days)
```
For each day t:
  threshold = atr_multiplier × ATR_t
  Look forward up to H days
  If max(return) ≥ +threshold before hitting -threshold → Label = 1 (UP)
  If min(return) ≤ -threshold before hitting +threshold → Label = -1 (DOWN)
  Otherwise → Label = 0 (NONE)
```

### Consolidation Label
```
For window W:
  range = max(high) - min(low)
  If range ≤ width_multiplier × ATR → Label = 1
```

### Breakout Label
```
recent_high = max(high[-W:])
breakout_level = recent_high + threshold × ATR
If close > breakout_level AND volume_surge → Label = 1
If falls back below breakout_level within M days → False Breakout
```

## 🛣️ Roadmap & Extensions

**Implemented:**
- ✅ Data loader with preprocessing
- ✅ Multi-horizon feature engineering
- ✅ ATR-based label generation
- ✅ LightGBM baseline with walk-forward CV
- ✅ Comprehensive evaluation metrics
- ✅ Backtesting engine

**Future Enhancements:**
- [ ] Matrix Profile for motif discovery
- [ ] SHAP explanations for predictions
- [ ] Sequence models (LSTM/Transformer)
- [ ] Multi-task learning (joint prediction)
- [ ] Ensemble methods (stacking)
- [ ] Hyperparameter optimization (Optuna)
- [ ] Live trading integration
- [ ] Dashboard for monitoring

## 📚 References

This system is based on professional quantitative trading practices:

1. **ATR-based thresholds:** Scale-invariant labeling across assets
2. **Walk-forward validation:** Industry standard for time series
3. **Economic metrics:** Focus on tradeable performance, not just accuracy
4. **Gradient boosting:** Proven baseline for tabular financial data

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional feature engineering methods
- Alternative label strategies
- Advanced models (deep learning, ensembles)
- Pattern discovery algorithms
- Visualization improvements

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading stocks involves risk, including the loss of principal. Past performance does not guarantee future results. Always do your own research and consult with a financial advisor before making investment decisions.

## 📄 License

MIT License - See LICENSE file for details

---

**Built with:** Python, LightGBM, pandas, numpy, scikit-learn

**Author:** ML Pattern Discovery Team

For questions or issues, please open a GitHub issue.
