import yfinance as yf
from datetime import datetime

print(f"Checking intraday silver prices...")
print(f"Current time: {datetime.now()}\n")

# Try to get latest intraday data
silver = yf.Ticker("SI=F")

# Try 1-day with 1-minute interval for most recent data
print("1. Trying 1-minute interval data (last hour):")
try:
    data_1m = silver.history(period='1d', interval='1m')
    if len(data_1m) > 0:
        print(f"   Total 1-min bars: {len(data_1m)}")
        print(f"   Latest timestamp: {data_1m.index[-1]}")
        print(f"   Latest Close: ${data_1m['Close'].iloc[-1]:.2f}")
        print(f"\n   Last 5 bars:")
        print(data_1m[['Close']].tail())
    else:
        print("   No 1-minute data available")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Trying 5-minute interval data:")
try:
    data_5m = silver.history(period='1d', interval='5m')
    if len(data_5m) > 0:
        print(f"   Total 5-min bars: {len(data_5m)}")
        print(f"   Latest timestamp: {data_5m.index[-1]}")
        print(f"   Latest Close: ${data_5m['Close'].iloc[-1]:.2f}")
        print(f"\n   Last 3 bars:")
        print(data_5m[['Close']].tail(3))
    else:
        print("   No 5-minute data available")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Trying hourly data (last 5 days):")
try:
    data_1h = silver.history(period='5d', interval='1h')
    if len(data_1h) > 0:
        print(f"   Total hourly bars: {len(data_1h)}")
        print(f"   Latest timestamp: {data_1h.index[-1]}")
        print(f"   Latest Close: ${data_1h['Close'].iloc[-1]:.2f}")
        print(f"\n   Last 3 hours:")
        print(data_1h[['Close']].tail(3))
    else:
        print("   No hourly data available")
except Exception as e:
    print(f"   Error: {e}")

# Check fast_info for real-time
print("\n4. Checking fast_info:")
try:
    print(f"   Last Price: ${silver.fast_info.get('last_price', 'N/A')}")
    print(f"   Previous Close: ${silver.fast_info.get('previous_close', 'N/A')}")
except Exception as e:
    print(f"   Error: {e}")
