#!/usr/bin/env python3
"""
Sample data generator for testing the GOLD SMA5 strategy
Creates realistic OHLCV data for gold prices
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(start_date='2020-01-01', end_date='2023-12-31', 
                        initial_price=1800, volatility=0.02, trend=0.0001):
    """
    Generate realistic sample OHLCV data for gold prices
    
    Parameters:
    - start_date: Start date for the data
    - end_date: End date for the data  
    - initial_price: Starting price for gold
    - volatility: Daily volatility (default 2%)
    - trend: Daily trend (default 0.01%)
    """
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price series with trend and volatility
    np.random.seed(42)  # For reproducible results
    price = initial_price
    prices = []
    
    for i in range(len(dates)):
        # Add trend and random walk
        change = np.random.normal(trend, volatility)
        price *= (1 + change)
        prices.append(price)
    
    # Generate OHLC data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        daily_vol = abs(np.random.normal(0, volatility * 0.5))
        
        open_price = close_price * (1 + np.random.normal(0, daily_vol * 0.3))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, daily_vol)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, daily_vol)))
        
        # Ensure OHLC consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume (higher volume on larger moves)
        price_change = abs(close_price - open_price) / open_price
        base_volume = 5000
        volume = int(base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 2.0))
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    # Generate sample data
    print("Generating sample gold price data...")
    df = generate_sample_data()
    
    # Save to CSV
    output_file = '/workspace/sample_gold_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Sample data generated and saved to: {output_file}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Show last few rows
    print("\nLast 5 rows:")
    print(df.tail())

if __name__ == "__main__":
    main()