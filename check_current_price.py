import yfinance as yf
import pandas as pd

# Check current silver price
silver = yf.Ticker("SI=F")
data = silver.history(period="5d")

print("Last 5 days of Silver data:")
print(data)
print(f"\nLatest Close: ${data['Close'].iloc[-1]:.2f}")
print(f"Latest Date: {data.index[-1]}")

# Also check with max period to see the actual latest date
data_max = silver.history(period="max")
print(f"\nWith 'max' period - Latest Date: {data_max.index[-1]}")
print(f"With 'max' period - Latest Close: ${data_max['Close'].iloc[-1]:.2f}")
print(f"Total records with 'max': {len(data_max)}")
