"""
Feature engineering module - creates multi-scale features from OHLCV data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import stats
from config import FeatureConfig


class FeatureEngineer:
    """
    Generates comprehensive features across multiple timeframes.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.windows = config.lookback_windows
        
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all feature types.
        
        Args:
            df: DataFrame with OHLCV data and ATR already calculated
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        # Price features
        df = self._create_price_features(df)
        
        # Volume features
        if self.config.use_volume_features and 'Volume' in df.columns:
            df = self._create_volume_features(df)
        
        # Structural/technical features
        df = self._create_structural_features(df)
        
        # Multi-scale features
        df = self._create_multiscale_features(df)
        
        # Pattern shape features
        if self.config.use_shape_embeddings:
            df = self._create_shape_features(df)
        
        # Relational features
        df = self._create_relational_features(df)
        
        # Time features
        df = self._create_time_features(df)
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features across multiple windows."""
        df = df.copy()
        
        # Daily returns (already calculated in preprocessing)
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for window in self.windows:
            # Momentum (cumulative return)
            df[f'momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
            
            # Volatility (std of log returns)
            df[f'volatility_{window}'] = df['log_return'].rolling(
                window=window, min_periods=max(1, window//2)
            ).std()
            
            # Moving averages
            df[f'sma_{window}'] = df['Close'].rolling(
                window=window, min_periods=max(1, window//2)
            ).mean()
            
            df[f'ema_{window}'] = df['Close'].ewm(
                span=window, adjust=False, min_periods=max(1, window//2)
            ).mean()
            
            # Distance from MA (normalized by ATR)
            atr_col = 'ATR_14' if 'ATR_14' in df.columns else 'ATR_20'
            if atr_col in df.columns:
                df[f'price_to_sma_{window}'] = (
                    (df['Close'] - df[f'sma_{window}']) / (df[atr_col] + 1e-8)
                )
            
            # Price range features
            df[f'high_low_range_{window}'] = (
                (df['High'].rolling(window).max() - df['Low'].rolling(window).min()) 
                / df['Close']
            )
            
            # Range expansion
            avg_range = (df['High'] - df['Low']).rolling(window).mean()
            df[f'range_expansion_{window}'] = (
                (df['High'] - df['Low']) / (avg_range + 1e-8)
            )
            
            # Percentage from high/low
            df[f'pct_from_high_{window}'] = (
                df['Close'] / df['High'].rolling(window).max() - 1
            )
            df[f'pct_from_low_{window}'] = (
                df['Close'] / df['Low'].rolling(window).min() - 1
            )
            
        # Intraday features
        df['intraday_range'] = (df['High'] - df['Low']) / df['Close']
        df['intraday_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        df['open_to_close'] = (df['Close'] - df['Open']) / df['Open']
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        df = df.copy()
        
        for window in self.windows:
            # Volume z-score
            vol_mean = df['Volume'].rolling(window).mean()
            vol_std = df['Volume'].rolling(window).std()
            df[f'volume_zscore_{window}'] = (
                (df['Volume'] - vol_mean) / (vol_std + 1e-8)
            )
            
            # Volume surge
            vol_median = df['Volume'].rolling(window).median()
            df[f'volume_surge_{window}'] = df['Volume'] / (vol_median + 1e-8)
            
            # On-Balance Volume (OBV)
            obv = (np.sign(df['log_return']) * df['Volume']).fillna(0).cumsum()
            df[f'obv_slope_{window}'] = obv.diff(window) / window
            
            # Volume-weighted features
            df[f'vwap_{window}'] = (
                (df['Close'] * df['Volume']).rolling(window).sum() /
                (df['Volume'].rolling(window).sum() + 1e-8)
            )
            df[f'price_to_vwap_{window}'] = df['Close'] / (df[f'vwap_{window}'] + 1e-8) - 1
            
        # Price-Volume correlation
        for window in [10, 20, 40]:
            df[f'price_volume_corr_{window}'] = (
                df['log_return'].rolling(window).corr(df['Volume'].pct_change())
            )
        
        return df
    
    def _create_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create structural/technical indicator features."""
        df = df.copy()
        
        for window in self.windows:
            # Bollinger Bands
            sma = df['Close'].rolling(window).mean()
            std = df['Close'].rolling(window).std()
            df[f'bb_width_{window}'] = (2 * std) / (sma + 1e-8)
            df[f'bb_position_{window}'] = (df['Close'] - sma) / (2 * std + 1e-8)
            
            # RSI
            df[f'rsi_{window}'] = self._calculate_rsi(df['Close'], window)
            
            # Stochastic oscillator
            high_max = df['High'].rolling(window).max()
            low_min = df['Low'].rolling(window).min()
            df[f'stoch_k_{window}'] = (
                (df['Close'] - low_min) / (high_max - low_min + 1e-8) * 100
            )
            df[f'stoch_d_{window}'] = df[f'stoch_k_{window}'].rolling(3).mean()
            
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Consecutive up/down days
        df['consecutive_up'] = self._count_consecutive(df['log_return'] > 0)
        df['consecutive_down'] = self._count_consecutive(df['log_return'] < 0)
        
        # Pullback depth from local maxima/minima
        for window in [10, 20, 40]:
            rolling_max = df['Close'].rolling(window).max()
            rolling_min = df['Close'].rolling(window).min()
            df[f'pullback_from_high_{window}'] = (
                (df['Close'] - rolling_max) / (rolling_max + 1e-8)
            )
            df[f'rally_from_low_{window}'] = (
                (df['Close'] - rolling_min) / (rolling_min + 1e-8)
            )
        
        # Fractal indicators (local extrema)
        df['local_high_5'] = self._detect_local_extrema(df['High'], 5, mode='high')
        df['local_low_5'] = self._detect_local_extrema(df['Low'], 5, mode='low')
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _count_consecutive(condition: pd.Series) -> pd.Series:
        """Count consecutive True values."""
        # Create groups where condition changes
        groups = (condition != condition.shift()).cumsum()
        # Count within each group, but only for True conditions
        consecutive = condition.groupby(groups).cumsum()
        return consecutive.where(condition, 0)
    
    @staticmethod
    def _detect_local_extrema(series: pd.Series, window: int, mode: str = 'high') -> pd.Series:
        """Detect local extrema (peaks or troughs)."""
        result = pd.Series(0, index=series.index)
        
        for i in range(window, len(series) - window):
            window_slice = series.iloc[i-window:i+window+1]
            center_val = series.iloc[i]
            
            if mode == 'high':
                if center_val == window_slice.max():
                    result.iloc[i] = 1
            else:  # mode == 'low'
                if center_val == window_slice.min():
                    result.iloc[i] = 1
        
        return result
    
    def _create_multiscale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that compare across different timescales."""
        df = df.copy()
        
        # Trend alignment (all MAs pointing same direction)
        windows = [5, 10, 20, 40]
        sma_cols = [f'sma_{w}' for w in windows if f'sma_{w}' in df.columns]
        
        if len(sma_cols) >= 2:
            # Check if short MA > long MA (uptrend alignment)
            df['trend_alignment'] = 0
            for i in range(len(sma_cols) - 1):
                df['trend_alignment'] += (df[sma_cols[i]] > df[sma_cols[i+1]]).astype(int)
            df['trend_alignment'] = df['trend_alignment'] / (len(sma_cols) - 1)
        
        # Volatility regime (current vs historical)
        if 'volatility_20' in df.columns:
            vol_percentile = df['volatility_20'].rolling(252).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 1 else 50
            ) / 100
            df['volatility_regime'] = vol_percentile
        
        # Momentum consistency across timeframes
        momentum_cols = [f'momentum_{w}' for w in self.windows if f'momentum_{w}' in df.columns]
        if len(momentum_cols) >= 2:
            # Count how many momentum indicators are positive
            df['momentum_consistency'] = sum(
                (df[col] > 0).astype(int) for col in momentum_cols
            ) / len(momentum_cols)
        
        return df
    
    def _create_shape_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features representing price pattern shapes.
        Uses normalized subsequences.
        """
        df = df.copy()
        window = self.config.shape_window_size
        
        # Normalized price shape (z-score within window)
        close_norm = df['Close'].rolling(window).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x) == window else np.nan
        )
        df[f'shape_position_{window}'] = close_norm
        
        # Shape slope (linear regression slope)
        df[f'shape_slope_{window}'] = df['Close'].rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
        )
        
        # Shape curvature (quadratic term)
        df[f'shape_curvature_{window}'] = df['Close'].rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 2)[0] if len(x) == window else np.nan
        )
        
        # SAX representation (symbolic)
        if self.config.sax_alphabet_size > 0:
            df = self._create_sax_features(df, window)
        
        return df
    
    def _create_sax_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Create Symbolic Aggregate Approximation (SAX) features.
        Converts time series to symbolic strings for pattern matching.
        """
        df = df.copy()
        
        # Simplified SAX: discretize normalized returns into symbols
        alphabet_size = self.config.sax_alphabet_size
        
        # Use quantiles to create alphabet bins
        returns_norm = df['log_return'].rolling(window).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x) == window else np.nan
        )
        
        # Discretize into alphabet_size bins
        df['sax_symbol'] = pd.cut(
            returns_norm, 
            bins=alphabet_size, 
            labels=False
        )
        
        return df
    
    def _create_relational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that relate different aspects of price/volume.
        """
        df = df.copy()
        
        # Volatility-adjusted returns
        for window in [5, 10, 20]:
            if f'volatility_{window}' in df.columns:
                df[f'risk_adjusted_return_{window}'] = (
                    df[f'momentum_{window}'] / (df[f'volatility_{window}'] + 1e-8)
                )
        
        # Volume-price divergence
        if 'Volume' in df.columns:
            for window in [10, 20]:
                price_trend = df['Close'].rolling(window).apply(
                    lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
                )
                volume_trend = df['Volume'].rolling(window).apply(
                    lambda x: 1 if x.mean() > df['Volume'].rolling(window*2).mean().iloc[-1] else -1
                )
                df[f'price_volume_divergence_{window}'] = (price_trend != volume_trend).astype(int)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based cycle features."""
        df = df.copy()
        
        # Days since events
        for window in [20, 40, 60]:
            # Days since high
            rolling_high_idx = df['High'].rolling(window).apply(
                lambda x: len(x) - x.argmax() - 1, raw=False
            )
            df[f'days_since_high_{window}'] = rolling_high_idx
            
            # Days since low
            rolling_low_idx = df['Low'].rolling(window).apply(
                lambda x: len(x) - x.argmin() - 1, raw=False
            )
            df[f'days_since_low_{window}'] = rolling_low_idx
        
        # Autocorrelation features
        for lag in [1, 5, 10, 20]:
            df[f'autocorr_lag_{lag}'] = df['log_return'].rolling(60).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of all generated feature names (excluding OHLCV and intermediate columns).
        """
        # Base columns to exclude
        base_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date', 
                     'return', 'log_return', 'is_outlier', 'outlier_zscore', 
                     'outlier_return', 'gap', 'large_gap', 'market_regime',
                     'Dividends', 'Stock Splits', 'Capital Gains'}
        
        # Get all columns that are features (exclude non-numeric and label columns)
        feature_cols = []
        for col in df.columns:
            if col not in base_cols and not col.startswith('label_') and not col.startswith('in_') and \
               not col.startswith('consolidation_') and not col.startswith('false_') and \
               not col.startswith('trend_') and not col.startswith('bull_') and \
               not col.startswith('bear_') and not col.startswith('sideways_') and \
               not col.startswith('momentum_burst') and not col.startswith('v_reversal'):
                # Only include numeric columns
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    feature_cols.append(col)
        
        return feature_cols
