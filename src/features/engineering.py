"""
Multi-horizon feature engineering for OHLCV data.

Implements comprehensive feature generation across multiple lookback windows
including price, volume, structural, and pattern-based features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings


class FeatureEngine:
    """
    Generates multi-horizon features from OHLCV data.
    
    Feature categories:
    - Price features: returns, momentum, volatility, distance from MA
    - Volume features: volume surges, OBV, z-scores
    - Structural features: Bollinger Bands, RSI, MACD
    - Pattern features: consecutive runs, pullback depth, range expansion
    - Time/cycle features: days since extrema, autocorrelations
    """
    
    def __init__(self, windows: Optional[List[int]] = None,
                 include_volume: bool = True):
        """
        Initialize FeatureEngine.
        
        Args:
            windows: List of lookback windows (default: [3, 5, 10, 20, 40, 80])
            include_volume: Whether to generate volume-based features
        """
        self.windows = windows or [3, 5, 10, 20, 40, 80]
        self.include_volume = include_volume
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the given OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all generated features
        """
        df = df.copy()
        
        # Basic features
        df = self._add_basic_features(df)
        
        # Multi-window features
        for window in self.windows:
            df = self._add_price_features(df, window)
            
            if self.include_volume and 'Volume' in df.columns:
                df = self._add_volume_features(df, window)
            
            df = self._add_volatility_features(df, window)
            df = self._add_momentum_features(df, window)
        
        # Structural indicators
        df = self._add_structural_features(df)
        
        # Pattern features
        df = self._add_pattern_features(df)
        
        # Time/cycle features
        df = self._add_time_features(df)
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features."""
        # Log returns (if not already present)
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Simple return
        df['return'] = df['Close'].pct_change()
        
        # Intraday range
        df['range'] = (df['High'] - df['Low']) / df['Close']
        df['range_hl'] = df['High'] - df['Low']
        
        # Body (open-close)
        df['body'] = (df['Close'] - df['Open']) / df['Open']
        
        # Upper/lower shadows
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        
        # True range
        df['true_range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add price-based features for a given window."""
        prefix = f'price_{window}d'
        
        # Moving averages
        df[f'{prefix}_sma'] = df['Close'].rolling(window).mean()
        df[f'{prefix}_ema'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Distance from moving average (normalized by ATR)
        if 'true_range' in df.columns:
            atr = df['true_range'].rolling(window).mean()
            df[f'{prefix}_dist_sma'] = (df['Close'] - df[f'{prefix}_sma']) / atr
            df[f'{prefix}_dist_ema'] = (df['Close'] - df[f'{prefix}_ema']) / atr
        
        # Price momentum (ratio)
        df[f'{prefix}_momentum'] = df['Close'] / df['Close'].shift(window)
        
        # Price change
        df[f'{prefix}_change'] = (df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)
        
        # High/Low over window
        df[f'{prefix}_high'] = df['High'].rolling(window).max()
        df[f'{prefix}_low'] = df['Low'].rolling(window).min()
        
        # Distance from high/low
        df[f'{prefix}_from_high'] = (df['Close'] - df[f'{prefix}_high']) / df['Close']
        df[f'{prefix}_from_low'] = (df['Close'] - df[f'{prefix}_low']) / df['Close']
        
        # Mean return over window
        df[f'{prefix}_mean_return'] = df['return'].rolling(window).mean()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add volatility features for a given window."""
        prefix = f'vol_{window}d'
        
        # Standard deviation of returns
        df[f'{prefix}_std'] = df['log_return'].rolling(window).std()
        
        # ATR (Average True Range)
        df[f'{prefix}_atr'] = df['true_range'].rolling(window).mean()
        
        # Normalized ATR
        df[f'{prefix}_atr_norm'] = df[f'{prefix}_atr'] / df['Close']
        
        # Range expansion
        avg_range = df['range_hl'].rolling(window).mean()
        df[f'{prefix}_range_expansion'] = df['range_hl'] / avg_range
        
        # Bollinger Band width
        sma = df['Close'].rolling(window).mean()
        std = df['Close'].rolling(window).std()
        df[f'{prefix}_bb_width'] = (2 * std) / sma
        
        # Price position in Bollinger Bands
        df[f'{prefix}_bb_position'] = (df['Close'] - sma) / (2 * std)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add momentum indicators for a given window."""
        prefix = f'mom_{window}d'
        
        # ROC (Rate of Change)
        df[f'{prefix}_roc'] = (df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window) * 100
        
        # RSI (Relative Strength Index) - only for specific windows
        if window in [14, 20]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / loss
            df[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))
        
        # Cumulative return
        df[f'{prefix}_cum_return'] = df['log_return'].rolling(window).sum()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add volume-based features for a given window."""
        prefix = f'vol_{window}d'
        
        # Volume moving average
        df[f'{prefix}_volume_ma'] = df['Volume'].rolling(window).mean()
        
        # Volume ratio
        df[f'{prefix}_volume_ratio'] = df['Volume'] / df[f'{prefix}_volume_ma']
        
        # Volume z-score
        vol_mean = df['Volume'].rolling(window).mean()
        vol_std = df['Volume'].rolling(window).std()
        df[f'{prefix}_volume_zscore'] = (df['Volume'] - vol_mean) / vol_std
        
        # Volume surge
        vol_median = df['Volume'].rolling(window).median()
        df[f'{prefix}_volume_surge'] = df['Volume'] / vol_median
        
        # On-Balance Volume (OBV)
        obv = (np.sign(df['return']) * df['Volume']).cumsum()
        df[f'{prefix}_obv'] = obv
        df[f'{prefix}_obv_ma'] = obv.rolling(window).mean()
        df[f'{prefix}_obv_slope'] = (obv - obv.shift(window)) / window
        
        return df
    
    def _add_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add structural indicators (MACD, Stochastic, etc.)."""
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stochastic_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['stochastic_d'] = df['stochastic_k'].rolling(3).mean()
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern-based features."""
        # Consecutive up/down days
        df['up_day'] = (df['Close'] > df['Close'].shift(1)).astype(int)
        df['down_day'] = (df['Close'] < df['Close'].shift(1)).astype(int)
        
        # Count consecutive runs
        df['consecutive_ups'] = df['up_day'].groupby(
            (df['up_day'] != df['up_day'].shift()).cumsum()
        ).cumsum()
        
        df['consecutive_downs'] = df['down_day'].groupby(
            (df['down_day'] != df['down_day'].shift()).cumsum()
        ).cumsum()
        
        # Reset to 0 when pattern breaks
        df.loc[df['up_day'] == 0, 'consecutive_ups'] = 0
        df.loc[df['down_day'] == 0, 'consecutive_downs'] = 0
        
        # Pullback depth from recent high
        for window in [10, 20, 40]:
            recent_high = df['High'].rolling(window).max()
            df[f'pullback_depth_{window}d'] = (df['Close'] - recent_high) / recent_high
            
            recent_low = df['Low'].rolling(window).min()
            df[f'rally_height_{window}d'] = (df['Close'] - recent_low) / recent_low
        
        # Local extrema detection (fractal tops/bottoms)
        # A fractal top is when the high is the highest of 5 bars (2 before, current, 2 after)
        # We can approximate by looking at 3-bar patterns
        df['local_high'] = ((df['High'] > df['High'].shift(1)) & 
                            (df['High'] > df['High'].shift(2))).astype(int)
        df['local_low'] = ((df['Low'] < df['Low'].shift(1)) & 
                           (df['Low'] < df['Low'].shift(2))).astype(int)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based and cycle features."""
        # Days since last local high/low
        df['days_since_local_high'] = self._days_since_event(df['local_high'])
        df['days_since_local_low'] = self._days_since_event(df['local_low'])
        
        # Autocorrelation features (at specific lags)
        for lag in [1, 5, 10, 20]:
            df[f'autocorr_lag{lag}'] = df['log_return'].rolling(40).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=False
            )
        
        # Day of week and month (if datetime index)
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
        
        return df
    
    @staticmethod
    def _days_since_event(event_series: pd.Series) -> pd.Series:
        """Calculate days since last event occurred."""
        result = pd.Series(index=event_series.index, dtype=float)
        days_counter = 0
        
        for idx in event_series.index:
            if event_series[idx] == 1:
                days_counter = 0
            else:
                days_counter += 1
            result[idx] = days_counter
        
        return result
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of all generated feature names (excluding OHLCV).
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'Adj Close', 'is_missing', 'is_outlier']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
