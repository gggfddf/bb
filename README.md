# GOLD SMA5 Reversion Strategy - Python Implementation

This is a Python implementation of the Pine Script "GOLD SMA5 Reversion Strategy (Long Only)" that can work with CSV data and provides comprehensive backtesting and visualization.

## Features

- **Exact Pine Script Logic**: Implements the same SMA5 reversion strategy with identical parameters
- **CSV Data Support**: Load OHLCV data from CSV files with flexible column naming
- **Comprehensive Visualization**: 
  - Price chart with SMA overlay
  - Entry/exit signals
  - Distance from SMA indicator
  - Equity curve
- **Performance Metrics**: Detailed trade analysis and performance statistics
- **Sample Data Generator**: Create realistic test data for backtesting

## Strategy Logic

- **Entry Condition**: When price is at least 0.2% below SMA5
- **Exit Condition**: After holding for exactly 20 candles
- **Position Size**: 100% of equity (can be modified)
- **Direction**: Long only

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Sample Data (Optional)
```bash
python sample_data_generator.py
```

### 2. Run Strategy with CSV Data
```bash
python gold_sma5_strategy.py --csv sample_gold_data.csv
```

### 3. Run with Custom Parameters
```bash
python gold_sma5_strategy.py --csv your_data.csv --sma-length 5 --hold-candles 20 --entry-distance 0.002 --capital 10000
```

### 4. Run with Sample Data (No CSV needed)
```bash
python gold_sma5_strategy.py
```

## CSV Data Format

The script accepts CSV files with OHLCV data. Column names are flexible and case-insensitive:

**Required columns:**
- Date/date/DATE/timestamp/time/Time
- Open/open/OPEN
- High/high/HIGH  
- Low/low/LOW
- Close/close/CLOSE

**Optional columns:**
- Volume/volume/VOLUME

Example CSV format:
```csv
Date,Open,High,Low,Close,Volume
2020-01-01,1800.50,1810.25,1795.75,1805.00,5000
2020-01-02,1805.00,1815.50,1800.25,1812.75,5200
...
```

## Output

The script generates:

1. **Console Output**: 
   - Performance summary
   - Trade details
   - Entry/exit signals

2. **Visualization**: 
   - `strategy_results.png` - Comprehensive charts

3. **Data Export**: 
   - `strategy_results.csv` - Detailed results with all indicators

## Parameters

- `--sma-length`: SMA period (default: 5)
- `--hold-candles`: Candles to hold position (default: 20)  
- `--entry-distance`: Minimum distance from SMA to enter (default: 0.002 = 0.2%)
- `--capital`: Initial capital (default: 10000)

## Performance Metrics

- Total return
- Win rate
- Number of trades
- Average win/loss
- Equity curve
- Trade-by-trade analysis

## Example Output

```
STRATEGY PERFORMANCE SUMMARY
============================================================
Initial Capital: $10,000.00
Final Equity: $12,450.00
Total Return: 24.50%
Total Trades: 45
Winning Trades: 28
Losing Trades: 17
Win Rate: 62.22%
Average Win: 1.85%
Average Loss: -1.12%
```

## Files

- `gold_sma5_strategy.py` - Main strategy implementation
- `sample_data_generator.py` - Generate test data
- `requirements.txt` - Python dependencies
- `README.md` - This documentation