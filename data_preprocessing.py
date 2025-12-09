"""
Data ingestion and preprocessing module for OHLCV stock data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import warnings
from config import DataConfig


class DataPreprocessor:
    """
    Handles data ingestion, cleaning, and preprocessing for OHLCV data.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def load_data(self, data: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Load and validate OHLCV data.
        
        Args:
            data: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                  Volume is optional
            validate: Whether to run validation checks
            
        Returns:
            Validated and cleaned DataFrame
        """
        df = data.copy()
        
        # Ensure Date is datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have Date column or DatetimeIndex")
            
        df = df.sort_index()
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check if Volume exists
        has_volume = 'Volume' in df.columns
        if not has_volume:
            warnings.warn("Volume data not available. Volume-based features will be disabled.")
            df['Volume'] = np.nan
            
        # Filter by date range if specified
        if self.config.start_date:
            df = df[df.index >= pd.to_datetime(self.config.start_date)]
        if self.config.end_date:
            df = df[df.index <= pd.to_datetime(self.config.end_date)]
            
        if validate:
            df = self._validate_data(df)
            
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and integrity.
        """
        # Check minimum data length
        if len(df) < self.config.min_trading_days:
            raise ValueError(
                f"Insufficient data: {len(df)} days < {self.config.min_trading_days} required"
            )
        
        # Check for missing data
        missing_pct = df[['Open', 'High', 'Low', 'Close']].isnull().sum().sum() / (len(df) * 4)
        if missing_pct > self.config.max_missing_pct:
            warnings.warn(
                f"High percentage of missing data: {missing_pct:.2%} > {self.config.max_missing_pct:.2%}"
            )
        
        # Validate OHLC relationships
        invalid_high = df['High'] < df['Low']
        invalid_high |= df['High'] < df['Open']
        invalid_high |= df['High'] < df['Close']
        
        if invalid_high.any():
            warnings.warn(f"Found {invalid_high.sum()} rows with invalid High values")
            df = df[~invalid_high]
        
        # Check for non-positive prices
        for col in ['Open', 'High', 'Low', 'Close']:
            non_positive = df[col] <= 0
            if non_positive.any():
                warnings.warn(f"Found {non_positive.sum()} non-positive {col} values")
                df = df[~non_positive]
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and flag outliers due to corporate actions or data errors.
        
        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()
        
        # Calculate returns
        df['return'] = df['Close'].pct_change()
        
        # Z-score based outlier detection
        returns_zscore = np.abs(df['return'] - df['return'].mean()) / df['return'].std()
        df['outlier_zscore'] = returns_zscore > self.config.outlier_std_threshold
        
        # Absolute return threshold
        df['outlier_return'] = np.abs(df['return']) > self.config.outlier_return_threshold
        
        # Combined outlier flag
        df['is_outlier'] = df['outlier_zscore'] | df['outlier_return']
        
        # Gap detection (overnight gaps)
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['large_gap'] = np.abs(df['gap']) > 0.1  # 10% gap
        
        return df
    
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing data points.
        
        Args:
            method: 'forward_fill', 'interpolate', or 'drop'
        """
        df = df.copy()
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear', limit=5)
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return df
    
    def normalize_prices(self, df: pd.DataFrame, 
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize price data using specified method.
        
        Args:
            columns: Columns to normalize. If None, normalizes OHLC.
        """
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close']
            
        df = df.copy()
        
        if self.config.normalization_method == 'zscore':
            for col in columns:
                if col in df.columns:
                    df[f'{col}_norm'] = self._rolling_zscore(
                        df[col], 
                        window=self.config.rolling_norm_window
                    )
        elif self.config.normalization_method == 'minmax':
            for col in columns:
                if col in df.columns:
                    df[f'{col}_norm'] = self._rolling_minmax(
                        df[col],
                        window=self.config.rolling_norm_window
                    )
        elif self.config.normalization_method == 'robust':
            for col in columns:
                if col in df.columns:
                    df[f'{col}_norm'] = self._rolling_robust_scale(
                        df[col],
                        window=self.config.rolling_norm_window
                    )
        
        return df
    
    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score."""
        mean = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
        return (series - mean) / (std + 1e-8)
    
    @staticmethod
    def _rolling_minmax(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling min-max normalization."""
        rolling_min = series.rolling(window=window, min_periods=1).min()
        rolling_max = series.rolling(window=window, min_periods=1).max()
        return (series - rolling_min) / (rolling_max - rolling_min + 1e-8)
    
    @staticmethod
    def _rolling_robust_scale(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling robust scaling (median and IQR)."""
        median = series.rolling(window=window, min_periods=1).median()
        q75 = series.rolling(window=window, min_periods=1).quantile(0.75)
        q25 = series.rolling(window=window, min_periods=1).quantile(0.25)
        iqr = q75 - q25
        return (series - median) / (iqr + 1e-8)
    
    def calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns for price series."""
        df = df.copy()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df
    
    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            window: ATR period (default 14 days)
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range components
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the rolling mean of TR
        atr = tr.rolling(window=window, min_periods=1).mean()
        
        return atr
    
    def preprocess_pipeline(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Complete preprocessing pipeline.
        
        Returns:
            Tuple of (processed_df, metadata_dict)
        """
        metadata = {}
        
        # Load and validate
        df = self.load_data(data, validate=True)
        metadata['n_rows_initial'] = len(df)
        metadata['date_range'] = (df.index.min(), df.index.max())
        metadata['has_volume'] = not df['Volume'].isna().all()
        
        # Detect outliers
        df = self.detect_outliers(df)
        metadata['n_outliers'] = df['is_outlier'].sum()
        
        # Handle missing data
        df = self.handle_missing_data(df, method='forward_fill')
        metadata['n_rows_after_cleaning'] = len(df)
        
        # Calculate log returns
        df = self.calculate_log_returns(df)
        
        # Calculate ATR (multiple windows for features)
        for window in [14, 20, 40]:
            df[f'ATR_{window}'] = self.calculate_atr(df, window=window)
        
        # Normalize prices (optional, for certain features)
        df = self.normalize_prices(df)
        
        return df, metadata


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features (day of week, month, etc.).
    """
    df = df.copy()
    
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    
    # Binary flags
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    
    return df
