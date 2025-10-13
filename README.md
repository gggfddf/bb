# Stock Pattern Detection and Market Cycle Analysis System

A comprehensive, production-ready system for detecting recurring price/volume patterns, learning market cycles, and forecasting stock movements using machine learning.

## 🎯 Features

### Core Capabilities

- **Multi-Scale Feature Engineering**: 100+ features across multiple timeframes (3, 5, 10, 20, 40, 80 days)
  - Price momentum, volatility, and technical indicators
  - Volume analysis and divergence detection
  - Pattern shape embeddings and SAX representation
  - Multi-scale trend alignment and regime detection

- **ATR-Based Labeling**: Scale-invariant labels using Average True Range
  - Major directional moves with configurable thresholds
  - Breakout and false breakout detection
  - Trend reversals and momentum bursts
  - Market regime classification (bull/bear/sideways)

- **Hybrid Modeling**:
  - **Baseline**: LightGBM, XGBoost, CatBoost with full explainability
  - **Sequence Models**: LSTM, CNN, Transformer for temporal dependencies
  - Multi-task learning for joint prediction of moves, patterns, and magnitudes

- **Unsupervised Pattern Discovery**:
  - Matrix Profile for motif detection
  - HDBSCAN and K-means clustering on normalized subsequences
  - Pattern library with predictive power statistics
  - DTW-based shape matching

- **Cycle Detection**:
  - Autocorrelation and partial autocorrelation analysis
  - Spectral analysis (FFT) for dominant frequencies
  - Wavelet transform for time-localized cycles
  - Hidden Markov Models for regime identification

- **Full Explainability**:
  - SHAP values for feature attribution
  - Pattern prototype visualization
  - Signal narratives and human-readable reports
  - Feature importance across time windows

- **Rigorous Evaluation**:
  - Walk-forward cross-validation (no lookahead)
  - Backtesting with realistic transaction costs
  - Comprehensive metrics (Sharpe, Sortino, max drawdown, win rate)
  - Threshold optimization

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- numpy, pandas, scipy
- lightgbm, xgboost, catboost
- torch (for deep learning models)
- scikit-learn
- stumpy (Matrix Profile)
- hdbscan (clustering)
- PyWavelets, hmmlearn (cycle detection)
- shap (explainability)
- matplotlib, seaborn (visualization)

## 🚀 Quick Start

### Basic Usage

```python
from pipeline import StockPatternPipeline, create_sample_data

# Load your OHLCV data
# data = pd.read_csv('your_stock_data.csv')
data = create_sample_data()  # For testing

# Initialize pipeline
pipeline = StockPatternPipeline()

# Run complete analysis
results = pipeline.run_full_pipeline(
    data=data,
    target_label='label_up_10d',  # Predict 10-day moves
    run_backtest=True
)

# Save trained models
pipeline.save_pipeline('./outputs')
```

### Using Real Stock Data

```python
import yfinance as yf
from pipeline import StockPatternPipeline

# Download data
ticker = yf.Ticker("AAPL")
data = ticker.history(period="5y")
data = data.reset_index()

# Run analysis
pipeline = StockPatternPipeline()
results = pipeline.run_full_pipeline(data)
```

### Custom Configuration

```python
from config import SystemConfig
from pipeline import StockPatternPipeline

# Customize settings
config = SystemConfig()
config.feature.lookback_windows = [5, 10, 20, 40, 60]
config.label.atr_multiplier_up = 2.0  # More conservative
config.model.tree_n_estimators = 1000

# Initialize with custom config
pipeline = StockPatternPipeline(config=config)
```

## 📊 System Architecture

```
Data Input (OHLCV)
    ↓
[1] Data Preprocessing
    - Validation & cleaning
    - Outlier detection
    - ATR calculation
    - Normalization
    ↓
[2] Feature Engineering (100+ features)
    - Price features (momentum, volatility, MAs)
    - Volume features (OBV, VWAP, surges)
    - Technical indicators (RSI, MACD, Bollinger)
    - Pattern shapes (SAX, embeddings)
    - Multi-scale & relational features
    ↓
[3] Label Generation
    - Major moves (ATR-based)
    - Breakouts & false breakouts
    - Reversals
    - Market regimes
    ↓
[4] Pattern Discovery        [5] Cycle Detection
    - Matrix Profile              - Autocorrelation
    - Clustering                  - Spectral analysis
    - Motif library              - Wavelets
                                 - HMM regimes
    ↓                            ↓
[6] Model Training
    - Tree models (LightGBM/XGBoost/CatBoost)
    - Sequence models (LSTM/CNN/Transformer)
    - Class balancing & focal loss
    ↓
[7] Explainability          [8] Evaluation
    - SHAP attribution          - Walk-forward CV
    - Pattern reports           - Backtesting
    - Signal narratives         - Performance metrics
    ↓
Predictions + Explanations
```

## 🔬 Key Components

### 1. Data Preprocessing (`data_preprocessing.py`)
- OHLCV validation and quality checks
- Outlier detection (z-score and absolute return thresholds)
- Missing data handling
- ATR calculation across multiple windows
- Rolling normalization

### 2. Feature Engineering (`feature_engineering.py`)
- **Price features**: momentum, volatility, MA distances
- **Volume features**: z-scores, surges, OBV, VWAP
- **Structural**: Bollinger Bands, RSI, Stochastic, MACD
- **Multi-scale**: trend alignment, volatility regimes
- **Shape**: normalized patterns, SAX symbols
- **Relational**: risk-adjusted returns, divergences
- **Time**: days since highs/lows, autocorrelations

### 3. Label Generation (`label_generation.py`)
All labels use ATR-based thresholds for scale invariance:
- **Major moves**: Label=1 if price moves +k×ATR before -k×ATR within H days
- **Consolidation**: Price range < α×ATR for L days
- **Breakout**: Close > consolidation_high + β×ATR with volume confirmation
- **False breakout**: Breakout that fails within M days
- **Reversal**: Prior move > m×ATR followed by n×ATR reversal

### 4. Models (`models.py`)

**Baseline Tree Models:**
- LightGBM, XGBoost, CatBoost
- Class weighting for imbalanced data
- Early stopping and hyperparameter tuning

**Sequence Models:**
- LSTM: Captures long-term dependencies
- 1D CNN: Pattern recognition in subsequences
- Transformer: Attention-based temporal modeling
- Focal loss for rare event prediction

### 5. Pattern Discovery (`pattern_discovery.py`)
- **Matrix Profile**: Find recurring motifs with configurable similarity
- **Clustering**: HDBSCAN or K-means on normalized shapes
- **Predictive analysis**: Statistics on how patterns predict future moves
- **Pattern library**: Curated collection with win rates, Sharpe ratios, lead times

### 6. Cycle Detection (`cycle_detection.py`)
- **Autocorrelation**: Identify periodic lags
- **Spectral**: FFT to find dominant frequencies/periods
- **Wavelets**: Time-localized frequency analysis
- **HMM**: Gaussian HMM for regime switching (bull/bear/sideways)
- **Duration analysis**: Average time in each regime

### 7. Explainability (`explainability.py`)
- **SHAP**: TreeSHAP for feature importance and instance explanations
- **Attention**: Visualization for sequence models
- **Pattern library**: Human-readable pattern descriptions
- **Signal explanations**: Top contributing features with narratives
- **Comprehensive reports**: Multi-factor prediction summaries

### 8. Evaluation (`evaluation.py`)
- **Walk-forward CV**: Time-series safe validation (no lookahead)
- **Backtesting**: Realistic simulation with:
  - Bid-ask spread
  - Slippage
  - Commission
  - Execution delay
  - Position sizing
- **Metrics**: CAGR, Sharpe, Sortino, max drawdown, win rate, expectancy
- **Threshold optimization**: Find optimal decision boundary

## 📈 Example Outputs

### Backtest Report
```
======================================================================
BACKTEST EVALUATION REPORT
======================================================================

PERFORMANCE METRICS:
  Total Return:           34.52%
  CAGR:                    8.12%
  Volatility:             18.45%
  Sharpe Ratio:            0.44
  Sortino Ratio:           0.67
  Max Drawdown:          -12.34%

TRADE STATISTICS:
  Number of Trades:          156
  Win Rate:               58.33%
  Avg Trade Return:        0.22%
  Avg Win:                 2.45%
  Avg Loss:               -1.89%
  Profit Factor:           1.30
  Expectancy:              0.43%
======================================================================
```

### Pattern Summary
```
pattern_name          occurrences  best_horizon_days  win_rate  mean_return  sharpe_ratio
motif_3                        47                 10     0.638        0.032          1.89
cluster_12                     82                  5     0.622        0.019          1.45
motif_1                        34                 20     0.618        0.041          1.38
```

### Feature Importance (Top 10)
```
feature                        importance
momentum_20                      1243.56
volatility_10                     987.32
rsi_14                           856.91
price_to_sma_40                  734.28
volume_surge_20                  689.45
macd_histogram                   623.17
bb_position_20                   591.83
stoch_k_14                       542.29
shape_slope_20                   498.76
autocorr_lag_5                   467.52
```

## 🎓 Advanced Usage

### Walk-Forward Cross-Validation

```python
from evaluation import WalkForwardValidator
from config import EvaluationConfig

config = EvaluationConfig()
config.train_size_years = 3
config.validation_size_months = 6
config.test_size_months = 6
config.step_size_months = 3

validator = WalkForwardValidator(config)
cv_results = validator.cross_validate(df, X, y, train_func, predict_func, metric_func)
```

### Pattern Analysis

```python
# Discover patterns
patterns = pipeline.discover_patterns(df)

# Filter significant ones
significant = pipeline.pattern_discovery.filter_significant_patterns(
    min_occurrences=10,
    min_confidence=0.60
)

# Get detailed report
from explainability import PatternLibraryExplainer
report = PatternLibraryExplainer.create_pattern_report(significant)
```

### Making Predictions with Explanations

```python
# Load new data
new_data = pd.read_csv('recent_prices.csv')

# Get predictions with explanations
predictions = pipeline.predict_and_explain(new_data, model_name='lightgbm')

# Print prediction report
print(predictions['reports'][0])
```

## 📚 Pattern Definitions

### Consolidation
- **Definition**: Price range ≤ k×ATR_median for L days (L ∈ [10,30])
- **Parameters**: k=0.75-1.2, typically k=1.0
- **Example**: Stock trades in narrow range for 15 days

### Confirmed Breakout
- **Definition**: Close > consolidation_high + β×ATR
- **Volume**: Volume > median_volume × γ (γ=1.5)
- **Confirmation**: Sustained for M days (M=2-3)
- **Example**: Price breaks above 20-day range with 50% volume increase

### False Breakout
- **Definition**: After breakout, price returns below breakout_level - δ×ATR within N days
- **Parameters**: δ=0.5, N=10
- **Example**: Breakout fails and price falls back into range

### Momentum Burst
- **Definition**: 3-day cumulative return > 2σ with volume surge
- **Volume**: Volume > 1.5× median
- **Example**: Rapid 3-day rally with high volume

### V-Reversal
- **Definition**: Sharp decline (>5%) followed by sharp recovery (>5%)
- **Shape**: V-shaped price pattern
- **Example**: Quick drop and immediate bounce

## 🔧 Configuration Options

### Feature Engineering
```python
config.feature.lookback_windows = [3, 5, 10, 20, 40, 80]
config.feature.use_volume_features = True
config.feature.use_shape_embeddings = True
config.feature.sax_alphabet_size = 10
```

### Label Generation
```python
config.label.prediction_horizons = [3, 5, 10, 20]
config.label.atr_multiplier_up = 1.5
config.label.consolidation_min_days = 10
config.label.breakout_volume_multiplier = 1.5
```

### Model Training
```python
config.model.tree_model_type = 'lightgbm'  # or 'xgboost', 'catboost'
config.model.tree_n_estimators = 500
config.model.tree_max_depth = 7
config.model.use_focal_loss = True  # for imbalanced data
```

### Backtesting
```python
config.evaluation.initial_capital = 100000.0
config.evaluation.position_size_pct = 0.05  # 5% per trade
config.evaluation.bid_ask_spread_bps = 5.0
config.evaluation.commission_pct = 0.001  # 0.1%
config.evaluation.execution_delay_days = 1
```

## 📊 Data Format

Expected input format:
```python
data = pd.DataFrame({
    'Date': pd.date_range('2020-01-01', periods=100),
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]  # Optional but recommended
})
```

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Running Examples
```bash
python example_usage.py
```

### Running Pipeline
```bash
python pipeline.py
```

## 📄 License

This project is provided as-is for educational and research purposes.

## ⚠️ Disclaimer

This system is for educational and research purposes only. **Not financial advice.** Trading stocks involves risk. Past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional pattern definitions
- More sophisticated cycle detection methods
- Real-time data ingestion
- Portfolio optimization
- Multi-asset support
- GPU acceleration for deep learning

## 📞 Support

For issues, questions, or suggestions, please open an issue in the repository.

## 🎯 Roadmap

- [ ] Real-time prediction API
- [ ] Portfolio-level backtesting
- [ ] Multi-asset pattern transfer learning
- [ ] Reinforcement learning for strategy optimization
- [ ] Web dashboard for visualization
- [ ] Automated hyperparameter tuning
- [ ] Model monitoring and drift detection

---

**Built with modern ML best practices for production-ready stock market analysis.**
