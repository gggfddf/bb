#!/usr/bin/env python3
"""
GOLD SMA5 Reversion Strategy (Long Only) - Python Implementation
Based on Pine Script strategy with same logic and parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
import os

class GoldSMA5Strategy:
    def __init__(self, sma_length=5, hold_candles=20, entry_distance=0.002, initial_capital=10000):
        """
        Initialize the GOLD SMA5 Reversion Strategy
        
        Parameters:
        - sma_length: Length of SMA (default: 5)
        - hold_candles: Number of candles to hold position (default: 20)
        - entry_distance: Minimum distance from SMA to enter (default: 0.002 = 0.2%)
        - initial_capital: Starting capital (default: 10000)
        """
        self.sma_length = sma_length
        self.hold_candles = hold_candles
        self.entry_distance = entry_distance
        self.initial_capital = initial_capital
        
        # Strategy state
        self.entry_bar_index = None
        self.position = 0  # 0 = no position, 1 = long
        self.trades = []
        self.equity_curve = []
        
    def load_csv_data(self, csv_file):
        """
        Load OHLCV data from CSV file
        Expected columns: Date, Open, High, Low, Close, Volume (or similar)
        """
        try:
            # Try different common date column names
            date_columns = ['Date', 'date', 'DATE', 'timestamp', 'time', 'Time']
            
            df = pd.read_csv(csv_file)
            
            # Find date column
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                # If no date column found, create one
                df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1D')
                date_col = 'Date'
            
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            
            # Standardize column names (case insensitive)
            df.columns = df.columns.str.lower()
            
            # Ensure we have OHLC data
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Sort by date
            df = df.sort_index()
            
            print(f"Loaded {len(df)} candles from {csv_file}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    
    def calculate_sma(self, prices):
        """Calculate Simple Moving Average"""
        return prices.rolling(window=self.sma_length).mean()
    
    def run_strategy(self, df):
        """
        Run the SMA5 reversion strategy on the data
        """
        # Calculate SMA
        df['sma'] = self.calculate_sma(df['close'])
        
        # Calculate distance from SMA
        df['distance_from_sma'] = (df['sma'] - df['close']) / df['sma']
        
        # Initialize strategy columns
        df['entry_signal'] = False
        df['exit_signal'] = False
        df['position'] = 0
        df['equity'] = self.initial_capital
        
        current_equity = self.initial_capital
        entry_price = 0
        entry_bar = 0
        
        for i in range(len(df)):
            current_bar = i
            current_price = df['close'].iloc[i]
            
            # Check entry condition
            entry_condition = (df['distance_from_sma'].iloc[i] >= self.entry_distance and 
                             self.position == 0 and 
                             not pd.isna(df['sma'].iloc[i]))
            
            # Check exit condition
            exit_condition = (self.position == 1 and 
                            (current_bar - entry_bar) >= self.hold_candles)
            
            if entry_condition:
                # Enter long position
                self.position = 1
                entry_price = current_price
                entry_bar = current_bar
                df.loc[df.index[i], 'entry_signal'] = True
                df.loc[df.index[i], 'position'] = 1
                
                print(f"Entry at {df.index[i]}: Price = {current_price:.2f}, SMA = {df['sma'].iloc[i]:.2f}")
                
            elif exit_condition:
                # Exit position
                exit_price = current_price
                pnl = (exit_price - entry_price) / entry_price
                current_equity *= (1 + pnl)
                
                # Record trade
                trade = {
                    'entry_date': df.index[entry_bar],
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl * 100,
                    'hold_candles': current_bar - entry_bar
                }
                self.trades.append(trade)
                
                self.position = 0
                df.loc[df.index[i], 'exit_signal'] = True
                df.loc[df.index[i], 'position'] = 0
                
                print(f"Exit at {df.index[i]}: Price = {exit_price:.2f}, PnL = {pnl*100:.2f}%")
            
            # Update position and equity
            df.loc[df.index[i], 'position'] = self.position
            df.loc[df.index[i], 'equity'] = current_equity
        
        return df
    
    def calculate_performance_metrics(self, df):
        """Calculate strategy performance metrics"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        total_return = (df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_return,
            'final_equity': df['equity'].iloc[-1]
        }
        
        return metrics
    
    def plot_results(self, df, save_plot=True):
        """
        Create comprehensive visualization of the strategy results
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price, SMA, and Entry/Exit Signals
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1, color='blue')
        ax1.plot(df.index, df['sma'], label=f'SMA{self.sma_length}', linewidth=2, color='orange')
        
        # Entry signals
        entry_points = df[df['entry_signal'] == True]
        if not entry_points.empty:
            ax1.scatter(entry_points.index, entry_points['close'], 
                       color='green', marker='^', s=100, label='Entry Signal', zorder=5)
        
        # Exit signals
        exit_points = df[df['exit_signal'] == True]
        if not exit_points.empty:
            ax1.scatter(exit_points.index, exit_points['close'], 
                       color='red', marker='v', s=100, label='Exit Signal', zorder=5)
        
        ax1.set_title('GOLD SMA5 Reversion Strategy - Price Chart with Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distance from SMA
        ax2.plot(df.index, df['distance_from_sma'] * 100, label='Distance from SMA (%)', color='purple')
        ax2.axhline(y=self.entry_distance * 100, color='red', linestyle='--', 
                   label=f'Entry Threshold ({self.entry_distance*100:.1f}%)')
        ax2.set_title('Distance from SMA')
        ax2.set_ylabel('Distance (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Equity Curve
        ax3.plot(df.index, df['equity'], label='Equity Curve', color='green', linewidth=2)
        ax3.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
        ax3.set_title('Strategy Equity Curve')
        ax3.set_ylabel('Equity ($)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('/workspace/strategy_results.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'strategy_results.png'")
        
        plt.show()
    
    def print_performance_summary(self, metrics):
        """Print detailed performance summary"""
        print("\n" + "="*60)
        print("STRATEGY PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Equity: ${metrics['final_equity']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Average Win: {metrics['avg_win']:.2f}%")
        print(f"Average Loss: {metrics['avg_loss']:.2f}%")
        
        if self.trades:
            print(f"\nTrade Details:")
            print("-" * 80)
            print(f"{'Entry Date':<12} {'Exit Date':<12} {'Entry Price':<12} {'Exit Price':<12} {'PnL %':<10} {'Hold Days':<10}")
            print("-" * 80)
            
            for trade in self.trades:
                print(f"{trade['entry_date'].strftime('%Y-%m-%d'):<12} "
                      f"{trade['exit_date'].strftime('%Y-%m-%d'):<12} "
                      f"{trade['entry_price']:<12.2f} "
                      f"{trade['exit_price']:<12.2f} "
                      f"{trade['pnl_pct']:<10.2f} "
                      f"{trade['hold_candles']:<10}")

def main():
    parser = argparse.ArgumentParser(description='GOLD SMA5 Reversion Strategy')
    parser.add_argument('--csv', type=str, help='Path to CSV file with OHLCV data')
    parser.add_argument('--sma-length', type=int, default=5, help='SMA length (default: 5)')
    parser.add_argument('--hold-candles', type=int, default=20, help='Hold candles (default: 20)')
    parser.add_argument('--entry-distance', type=float, default=0.002, help='Entry distance (default: 0.002)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital (default: 10000)')
    
    args = parser.parse_args()
    
    # Initialize strategy
    strategy = GoldSMA5Strategy(
        sma_length=args.sma_length,
        hold_candles=args.hold_candles,
        entry_distance=args.entry_distance,
        initial_capital=args.capital
    )
    
    # Load data
    if args.csv:
        df = strategy.load_csv_data(args.csv)
    else:
        # Create sample data if no CSV provided
        print("No CSV file provided. Creating sample data...")
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        price = 1800  # Starting price
        prices = []
        for _ in range(len(dates)):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price *= (1 + change)
            prices.append(price)
        
        df = pd.DataFrame({
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Run strategy
    print("Running strategy...")
    df = strategy.run_strategy(df)
    
    # Calculate performance metrics
    metrics = strategy.calculate_performance_metrics(df)
    
    # Print results
    strategy.print_performance_summary(metrics)
    
    # Create visualization
    strategy.plot_results(df)
    
    # Save results to CSV
    output_file = '/workspace/strategy_results.csv'
    df.to_csv(output_file)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()