# Silver Price ML Predictor - Pattern Learning Edition

A machine learning project that downloads daily silver price data from yfinance and predicts future prices using **pattern-based learning** focused on price movements, time cycles, and market behavior.

## 🎯 Features

### Pattern-Based Learning (NEW!)
- **Price Movement Patterns**: Learns from actual price movements, not indicators
- **Time Cycle Analysis**: Understands correction time, consolidation time, reversal time
- **Pattern Recognition**: Detects peaks, troughs, and cycle positions
- **Momentum Exhaustion**: Identifies overbought/oversold patterns
- **Movement Relationships**: Analyzes relationships between rallies, corrections, and consolidations

### Traditional Indicators (Original)
- Downloads 1-day interval silver futures data (SI=F) from Yahoo Finance
- Creates technical indicators (SMA, EMA, RSI, Volatility, etc.)
- Trains Random Forest and Linear Regression models
- Predicts next day's silver price
- Visualizes model performance with comprehensive charts

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Pattern-Based Model (Recommended)
Run the pattern-learning model that focuses on price movements and time cycles:

```bash
python3 silver_ml_pattern_analysis.py
```

### Traditional Indicator Model
Run the traditional model with technical indicators:

```bash
python3 silver_ml_analysis.py
```

## Output

The script will:
1. Download 1 year of daily silver price data
2. Create technical indicators and features
3. Train a Random Forest model
4. Display model performance metrics (R², RMSE, MAE)
5. Generate visualizations saved as `silver_ml_results.png`
6. Predict the next day's silver price

## Data Source

- **Ticker**: SI=F (Silver Futures)
- **Interval**: 1 day
- **Default Period**: Maximum available (up to 50 years)

## Model Features

### Pattern-Based Model (30 Features)
Learns from pure price action and time patterns:

**Price Patterns:**
- Price change percentage and movement direction
- Peak and trough detection
- Drawdown from high and rise from low

**Correction Patterns:**
- In correction flag and duration
- Correction time tracking

**Consolidation Patterns:**
- Range ratio and consolidation detection
- Consolidation duration tracking

**Reversal Patterns:**
- Trend direction and reversal detection
- Days since last reversal

**Momentum Patterns:**
- 5-day, 10-day, 20-day momentum
- Overbought/oversold detection
- Price-momentum divergence

**Time Cycles:**
- Day of week, month patterns
- Cycle position (days since peak)
- Consecutive up/down days
- Rally vs decline ratio

### Traditional Model (15 Features)
Uses classic technical indicators:
- Open, High, Low, Close prices
- Volume and volume changes
- Simple Moving Averages (5-day, 20-day)
- Exponential Moving Averages (5-day, 20-day)
- Relative Strength Index (RSI)
- Price volatility
- High-Low spread

## Customization

You can modify the script to:
- Change the time period: `analyzer.download_silver_data(period="2y", interval="1d")`
- Use Linear Regression: `analyzer.train_model(model_type='linear')`
- Adjust model parameters in the `SilverMLAnalysis` class