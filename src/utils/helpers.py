"""
Utility functions for data generation, visualization, and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from datetime import datetime, timedelta


def create_sample_data(n_days: int = 1000, 
                      start_date: str = '2020-01-01',
                      ticker: str = 'SAMPLE') -> pd.DataFrame:
    """
    Create sample OHLCV data for testing.
    
    Generates realistic-looking price data with trends, volatility, and patterns.
    
    Args:
        n_days: Number of days to generate
        start_date: Starting date
        ticker: Ticker symbol
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate price with trend and noise
    trend = np.linspace(100, 150, n_days)
    volatility = 2.0
    noise = np.random.randn(n_days) * volatility
    
    # Add some cycles
    cycle = 10 * np.sin(np.linspace(0, 8*np.pi, n_days))
    
    close = trend + noise + cycle
    close = np.maximum(close, 10)  # Ensure positive prices
    
    # Generate OHLC from close
    daily_volatility = volatility * np.random.rand(n_days)
    
    high = close + daily_volatility * np.abs(np.random.randn(n_days))
    low = close - daily_volatility * np.abs(np.random.randn(n_days))
    
    # Open is between yesterday's close and today's close
    open_prices = np.zeros(n_days)
    open_prices[0] = close[0]
    for i in range(1, n_days):
        open_prices[i] = close[i-1] + np.random.randn() * volatility * 0.5
    
    # Ensure OHLC relationships
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))
    
    # Generate volume (inversely correlated with price changes)
    returns = np.diff(close, prepend=close[0])
    volume = 1e6 + 5e5 * np.abs(returns) + np.random.randn(n_days) * 1e5
    volume = np.maximum(volume, 1e4)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(int)
    })
    
    df = df.set_index('Date')
    
    return df


def plot_results(df: pd.DataFrame,
                predictions: Optional[pd.Series] = None,
                title: str = 'Results',
                save_path: Optional[str] = None) -> None:
    """
    Plot price data with optional prediction signals.
    
    Args:
        df: DataFrame with OHLCV data
        predictions: Optional series with predictions/signals
        title: Plot title
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Price plot
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='Close', linewidth=1.5)
    
    if predictions is not None:
        # Highlight prediction signals
        buy_signals = predictions == 1
        sell_signals = predictions == -1
        
        if buy_signals.any():
            ax1.scatter(df.index[buy_signals], df['Close'][buy_signals],
                       marker='^', color='green', s=100, label='Buy Signal', zorder=5)
        
        if sell_signals.any():
            ax1.scatter(df.index[sell_signals], df['Close'][sell_signals],
                       marker='v', color='red', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_ylabel('Price')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume plot
    ax2 = axes[1]
    ax2.bar(df.index, df['Volume'], alpha=0.5, color='blue')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # Returns plot
    ax3 = axes[2]
    if 'return' in df.columns:
        returns = df['return']
    else:
        returns = df['Close'].pct_change()
    
    ax3.bar(df.index, returns, alpha=0.5, color=['green' if r > 0 else 'red' for r in returns])
    ax3.set_ylabel('Returns')
    ax3.set_xlabel('Date')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 20,
                           title: str = 'Feature Importance',
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_backtest_results(backtest_df: pd.DataFrame,
                         title: str = 'Backtest Results',
                         save_path: Optional[str] = None) -> None:
    """
    Plot backtest performance.
    
    Args:
        backtest_df: DataFrame from BacktestEngine.backtest_signals
        title: Plot title
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Cumulative returns
    ax1 = axes[0]
    ax1.plot(backtest_df.index, backtest_df['cumulative_return'], 
            label='Strategy', linewidth=2)
    ax1.plot(backtest_df.index, backtest_df['cumulative_benchmark'],
            label='Buy & Hold', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(backtest_df.index, backtest_df['drawdown'], 0,
                     alpha=0.5, color='red')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)
    
    # Position
    ax3 = axes[2]
    ax3.plot(backtest_df.index, backtest_df['position'], linewidth=1.5)
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def calculate_signal_statistics(df: pd.DataFrame,
                                signals: pd.Series,
                                forward_returns_col: str = 'return') -> pd.DataFrame:
    """
    Calculate statistics for signals.
    
    Args:
        df: DataFrame with price data
        signals: Series with signal values (1, 0, -1)
        forward_returns_col: Column with forward returns
        
    Returns:
        DataFrame with signal statistics
    """
    stats = []
    
    for signal_value in sorted(signals.unique()):
        mask = signals == signal_value
        
        if mask.sum() == 0:
            continue
        
        signal_returns = df.loc[mask, forward_returns_col]
        
        stats.append({
            'signal': signal_value,
            'count': mask.sum(),
            'mean_return': signal_returns.mean(),
            'median_return': signal_returns.median(),
            'std_return': signal_returns.std(),
            'win_rate': (signal_returns > 0).mean(),
            'avg_win': signal_returns[signal_returns > 0].mean() if (signal_returns > 0).any() else 0,
            'avg_loss': signal_returns[signal_returns < 0].mean() if (signal_returns < 0).any() else 0
        })
    
    return pd.DataFrame(stats)
