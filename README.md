# Silver Price ML Predictor

A machine learning project that downloads daily silver price data from yfinance and predicts future prices using various ML algorithms.

## Features

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

Run the main script to download data, train the model, and make predictions:

```bash
python silver_ml_analysis.py
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
- **Default Period**: 1 year

## Model Features

The model uses the following features:
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