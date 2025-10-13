import yfinance as yf
from datetime import datetime

# Try different silver tickers
tickers = {
    'SI=F': 'Silver Futures (COMEX)',
    'SLV': 'iShares Silver Trust ETF',
    'SILVER': 'Silver Spot Price',
    'XAG=F': 'Silver Spot (Yahoo)',
    'XAGUSD=X': 'Silver/USD Forex'
}

print(f"Current time: {datetime.now()}\n")
print("=" * 70)

for ticker, name in tickers.items():
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period='5d')
        if len(hist) > 0:
            latest_price = hist['Close'].iloc[-1]
            latest_date = hist.index[-1]
            print(f"{ticker:12} ({name})")
            print(f"  Latest Price: ${latest_price:.2f}")
            print(f"  Latest Date:  {latest_date}")
            print(f"  Records: {len(hist)}")
            print()
    except Exception as e:
        print(f"{ticker:12} - Error: {e}\n")

print("=" * 70)

# Get real-time quote if available
print("\nTrying to get real-time info for SI=F:")
silver = yf.Ticker("SI=F")
info = silver.info
if 'regularMarketPrice' in info:
    print(f"Regular Market Price: ${info['regularMarketPrice']:.2f}")
if 'currentPrice' in info:
    print(f"Current Price: ${info['currentPrice']:.2f}")
if 'previousClose' in info:
    print(f"Previous Close: ${info['previousClose']:.2f}")
