# Feature and Label Formulas Reference

Complete mathematical reference for all features and labels in the system.

## Table of Contents
- [Price Features](#price-features)
- [Volatility Features](#volatility-features)
- [Volume Features](#volume-features)
- [Momentum Features](#momentum-features)
- [Structural Indicators](#structural-indicators)
- [Pattern Features](#pattern-features)
- [Label Definitions](#label-definitions)

---

## Price Features

For each window W ∈ {3, 5, 10, 20, 40, 80}:

### Simple Moving Average (SMA)
```
SMA_t,W = (1/W) × Σ(i=0 to W-1) Close_{t-i}
```

### Exponential Moving Average (EMA)
```
α = 2/(W+1)
EMA_t = α × Close_t + (1-α) × EMA_{t-1}
```

### Distance from SMA (normalized by ATR)
```
dist_SMA_t,W = (Close_t - SMA_t,W) / ATR_t,W
```

### Price Momentum (ratio)
```
momentum_t,W = Close_t / Close_{t-W}
```

### Price Change (percentage)
```
change_t,W = (Close_t - Close_{t-W}) / Close_{t-W}
```

### Distance from High/Low
```
from_high_t,W = (Close_t - max(High_{t-W:t})) / Close_t
from_low_t,W = (Close_t - min(Low_{t-W:t})) / Close_t
```

---

## Volatility Features

### True Range (TR)
```
TR_t = max(High_t - Low_t, 
           |High_t - Close_{t-1}|, 
           |Low_t - Close_{t-1}|)
```

### Average True Range (ATR)
```
ATR_t,W = (1/W) × Σ(i=0 to W-1) TR_{t-i}
```

### Standard Deviation of Returns
```
log_return_t = ln(Close_t / Close_{t-1})
σ_t,W = √[(1/W) × Σ(i=0 to W-1) (log_return_{t-i} - μ_W)²]
```

### Bollinger Band Width
```
BB_width_t,W = (2 × σ_t,W) / SMA_t,W
```

### Bollinger Band Position
```
BB_position_t,W = (Close_t - SMA_t,W) / (2 × σ_t,W)
```
Values: -1 (lower band) to +1 (upper band)

### Range Expansion
```
avg_range_W = (1/W) × Σ(i=0 to W-1) (High_{t-i} - Low_{t-i})
range_expansion_t,W = (High_t - Low_t) / avg_range_W
```

---

## Volume Features

### Volume Ratio
```
volume_ratio_t,W = Volume_t / [(1/W) × Σ(i=0 to W-1) Volume_{t-i}]
```

### Volume Z-Score
```
μ_vol,W = (1/W) × Σ(i=0 to W-1) Volume_{t-i}
σ_vol,W = √[(1/W) × Σ(i=0 to W-1) (Volume_{t-i} - μ_vol,W)²]
volume_zscore_t,W = (Volume_t - μ_vol,W) / σ_vol,W
```

### Volume Surge
```
volume_surge_t,W = Volume_t / median(Volume_{t-W:t})
```

### On-Balance Volume (OBV)
```
OBV_t = OBV_{t-1} + sign(Close_t - Close_{t-1}) × Volume_t

where sign(x) = {
  +1  if x > 0
   0  if x = 0
  -1  if x < 0
}
```

### OBV Slope
```
OBV_slope_t,W = (OBV_t - OBV_{t-W}) / W
```

---

## Momentum Features

### Rate of Change (ROC)
```
ROC_t,W = [(Close_t - Close_{t-W}) / Close_{t-W}] × 100
```

### Relative Strength Index (RSI)
```
For W = 14 or 20:

gain_t = max(Close_t - Close_{t-1}, 0)
loss_t = max(Close_{t-1} - Close_t, 0)

avg_gain_W = (1/W) × Σ(i=0 to W-1) gain_{t-i}
avg_loss_W = (1/W) × Σ(i=0 to W-1) loss_{t-i}

RS = avg_gain_W / avg_loss_W

RSI_t = 100 - (100 / (1 + RS))
```
Values: 0-100 (>70 overbought, <30 oversold)

### Cumulative Return
```
cum_return_t,W = Σ(i=0 to W-1) log_return_{t-i}
```

---

## Structural Indicators

### MACD (Moving Average Convergence Divergence)
```
EMA_12 = EMA(Close, span=12)
EMA_26 = EMA(Close, span=26)

MACD_t = EMA_12_t - EMA_26_t
MACD_signal_t = EMA(MACD, span=9)
MACD_histogram_t = MACD_t - MACD_signal_t
```

### Stochastic Oscillator
```
For W = 14:

lowest_low_W = min(Low_{t-W:t})
highest_high_W = max(High_{t-W:t})

%K_t = 100 × (Close_t - lowest_low_W) / (highest_high_W - lowest_low_W)
%D_t = SMA(%K, 3)
```
Values: 0-100 (>80 overbought, <20 oversold)

---

## Pattern Features

### Consecutive Up/Down Days
```
up_day_t = {1 if Close_t > Close_{t-1}, else 0}
down_day_t = {1 if Close_t < Close_{t-1}, else 0}

consecutive_ups_t = count of consecutive up_days ending at t
consecutive_downs_t = count of consecutive down_days ending at t
```

### Pullback Depth
```
For W ∈ {10, 20, 40}:

recent_high_W = max(High_{t-W:t})
pullback_depth_t,W = (Close_t - recent_high_W) / recent_high_W
```
Negative values indicate distance below recent high.

### Rally Height
```
recent_low_W = min(Low_{t-W:t})
rally_height_t,W = (Close_t - recent_low_W) / recent_low_W
```
Positive values indicate distance above recent low.

### Local Extrema (Fractal)
```
local_high_t = {
  1 if High_t > High_{t-1} AND High_t > High_{t-2}
  0 otherwise
}

local_low_t = {
  1 if Low_t < Low_{t-1} AND Low_t < Low_{t-2}
  0 otherwise
}
```

---

## Label Definitions

### Major Move Label (ATR-Based)

**Parameters:**
- H: horizon (days to look forward)
- k: ATR multiplier (default 2.0)
- W_atr: ATR window (default 20)

**Algorithm:**
```
For each day t:
  
  threshold = k × ATR_t,W_atr
  
  Look forward H days: {t+1, t+2, ..., t+H}
  
  returns = Close_{t+i} - Close_t  for i ∈ {1..H}
  
  up_cross = first i where returns_i ≥ +threshold
  down_cross = first i where returns_i ≤ -threshold
  
  if up_cross exists AND (down_cross doesn't exist OR up_cross < down_cross):
    label = 1 (UP)
    magnitude = returns[up_cross] / ATR_t
    days_to_move = up_cross
  
  elif down_cross exists AND (up_cross doesn't exist OR down_cross < up_cross):
    label = -1 (DOWN)
    magnitude = returns[down_cross] / ATR_t
    days_to_move = down_cross
  
  else:
    label = 0 (NONE)
```

**Binary Version:**
```
label_H_up = {1 if label == 1, else 0}
label_H_down = {1 if label == -1, else 0}
```

---

### Consolidation Label

**Parameters:**
- L: minimum consolidation days (default 10)
- α: width multiplier (default 1.2)

**Definition:**
```
For day t, looking back L days:

range_L = max(High_{t-L:t}) - min(Low_{t-L:t})
threshold = α × ATR_t

consolidation_t = {
  1 if range_L ≤ threshold
  0 otherwise
}

consolidation_width_t = range_L / ATR_t
```

---

### Breakout Label

**Parameters:**
- W: lookback window (default 20)
- β: breakout threshold (default 0.5)
- γ: volume threshold (default 1.5)
- M: confirmation days (default 3)

**Definition:**
```
For day t:

consolidation_high = max(High_{t-W:t-1})
breakout_level = consolidation_high + β × ATR_t
volume_threshold = γ × median(Volume_{t-W:t-1})

# Breakout condition
if Close_t > breakout_level AND Volume_t > volume_threshold:
  breakout_t = 1
  
  # Check for false breakout
  future_prices = Close_{t+1:t+M}
  fail_level = breakout_level - 0.5 × ATR_t
  
  if any(future_prices < fail_level):
    false_breakout_t = 1
    confirmed_breakout_t = 0
  else:
    false_breakout_t = 0
    confirmed_breakout_t = 1
else:
  breakout_t = 0
```

---

### Reversal Label

**Parameters:**
- m: move threshold (default 2.0)
- n: reversal threshold (default 2.0)
- W_prior: window for prior move (default 10)
- H_rev: reversal horizon (default 10)

**Definition:**
```
For day t:

prior_move = Close_t - Close_{t-W_prior}
threshold = m × ATR_t

# Check for prior upward move followed by reversal
if prior_move > threshold:
  future_min = min(Close_{t:t+H_rev})
  reversal_magnitude = future_min - Close_t
  
  if reversal_magnitude < -n × ATR_t:
    reversal_t = 1
    reversal_type_t = 1  # Top reversal

# Check for prior downward move followed by reversal
elif prior_move < -threshold:
  future_max = max(Close_{t:t+H_rev})
  reversal_magnitude = future_max - Close_t
  
  if reversal_magnitude > n × ATR_t:
    reversal_t = 1
    reversal_type_t = -1  # Bottom reversal

else:
  reversal_t = 0
```

---

### Regime Label

**Parameters:**
- W: window (default 40)
- θ_return: return threshold (default 0.10 = 10%)
- θ_vol: volatility threshold (default 0.30 = 30% annualized)

**Definition:**
```
For day t:

rolling_return_t = (Close_t / Close_{t-W}) - 1
rolling_vol_t = σ(log_return_{t-W:t}) × √252  # Annualized

regime_t = {
  +1 (BULL)    if rolling_return_t > θ_return AND rolling_vol_t < θ_vol
  -1 (BEAR)    if rolling_return_t < -θ_return OR rolling_vol_t > 1.5×θ_vol
   0 (NEUTRAL) otherwise
}
```

---

## Notes on Implementation

1. **ATR-based thresholds** ensure scale invariance across different price levels and assets
2. **Multiple horizons** (3, 5, 10, 20 days) capture different trading timeframes
3. **Volume confirmation** reduces false signals in breakout detection
4. **Lookback windows** {3, 5, 10, 20, 40, 80} span short-term to medium-term patterns
5. All features avoid **lookahead bias** - only use data up to time t

---

## Quick Reference: Key Thresholds

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| ATR multiplier (k) | 2.0 | 1.0-3.0 | Major move sensitivity |
| ATR window | 20 | 14-30 | Volatility measurement period |
| Horizons (H) | [3,5,10,20] | 1-60 | Prediction timeframes |
| Consolidation days (L) | 10 | 5-30 | Min consolidation period |
| Breakout threshold (β) | 0.5 | 0.3-1.0 | Breakout sensitivity |
| Volume surge (γ) | 1.5 | 1.2-2.0 | Volume confirmation |

---

For implementation details, see:
- Features: `src/features/engineering.py`
- Labels: `src/labels/generator.py`
