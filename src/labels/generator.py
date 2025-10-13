"""
Label generation for supervised learning with ATR-based thresholds.

Implements multiple labeling strategies:
- Major directional moves (binary/multi-class)
- Breakout/consolidation patterns
- Reversal detection
- Cycle-based regime labels
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict


class LabelGenerator:
    """
    Generates labels for supervised learning using ATR-based thresholds.
    
    Key features:
    - Scale-invariant thresholds using ATR
    - Multiple prediction horizons
    - Pattern-based labels (breakout, consolidation, reversal)
    - Regime detection (bull/bear zones)
    """
    
    def __init__(self, atr_window: int = 20):
        """
        Initialize LabelGenerator.
        
        Args:
            atr_window: Window for ATR calculation (default: 20)
        """
        self.atr_window = atr_window
    
    def generate_labels(self, df: pd.DataFrame,
                       horizons: Optional[List[int]] = None,
                       atr_multiplier: float = 2.0) -> pd.DataFrame:
        """
        Generate all labels for the given data.
        
        Args:
            df: DataFrame with OHLCV data and features
            horizons: List of prediction horizons in days (default: [3, 5, 10, 20])
            atr_multiplier: Multiplier for ATR threshold (default: 2.0)
            
        Returns:
            DataFrame with all generated labels
        """
        df = df.copy()
        horizons = horizons or [3, 5, 10, 20]
        
        # Calculate ATR if not present
        if f'vol_{self.atr_window}d_atr' not in df.columns:
            if 'true_range' in df.columns:
                df[f'vol_{self.atr_window}d_atr'] = df['true_range'].rolling(self.atr_window).mean()
            else:
                # Calculate true range
                df['true_range'] = np.maximum(
                    df['High'] - df['Low'],
                    np.maximum(
                        abs(df['High'] - df['Close'].shift(1)),
                        abs(df['Low'] - df['Close'].shift(1))
                    )
                )
                df[f'vol_{self.atr_window}d_atr'] = df['true_range'].rolling(self.atr_window).mean()
        
        atr_col = f'vol_{self.atr_window}d_atr'
        
        # Generate major move labels for each horizon
        for horizon in horizons:
            df = self._add_major_move_label(df, horizon, atr_col, atr_multiplier)
        
        # Pattern-based labels
        df = self._add_consolidation_label(df, atr_col)
        df = self._add_breakout_label(df, atr_col)
        df = self._add_reversal_label(df, atr_col)
        
        # Regime labels
        df = self._add_regime_label(df)
        
        return df
    
    def _add_major_move_label(self, df: pd.DataFrame, horizon: int,
                             atr_col: str, atr_multiplier: float) -> pd.DataFrame:
        """
        Generate major move label for a specific horizon.
        
        Label is:
        - 1 (UP) if max_future_return >= threshold before hitting -threshold
        - -1 (DOWN) if min_future_return <= -threshold before hitting +threshold
        - 0 (NONE) otherwise
        
        Args:
            df: DataFrame with data
            horizon: Number of days to look forward
            atr_col: Column name for ATR
            atr_multiplier: Multiplier for ATR to set threshold
            
        Returns:
            DataFrame with label column added
        """
        label_col = f'label_{horizon}d_major_move'
        magnitude_col = f'label_{horizon}d_move_magnitude'
        days_to_move_col = f'label_{horizon}d_days_to_move'
        
        # Initialize arrays
        labels = np.zeros(len(df), dtype=int)
        magnitudes = np.zeros(len(df), dtype=float)
        days_to_move = np.zeros(len(df), dtype=float)
        
        for i in range(len(df) - horizon):
            # Get current price and ATR
            current_price = df['Close'].iloc[i]
            current_atr = df[atr_col].iloc[i]
            
            if pd.isna(current_atr) or current_atr == 0:
                labels[i] = 0
                continue
            
            # Calculate threshold
            threshold = atr_multiplier * current_atr
            
            # Look forward up to horizon days
            future_prices = df['Close'].iloc[i+1:i+1+horizon].values
            
            if len(future_prices) == 0:
                continue
            
            # Calculate returns
            returns = future_prices - current_price
            
            # Find first crossing of threshold (up or down)
            up_cross_idx = np.where(returns >= threshold)[0]
            down_cross_idx = np.where(returns <= -threshold)[0]
            
            if len(up_cross_idx) > 0 and len(down_cross_idx) > 0:
                # Both occurred - which came first?
                if up_cross_idx[0] < down_cross_idx[0]:
                    labels[i] = 1
                    magnitudes[i] = returns[up_cross_idx[0]] / current_atr
                    days_to_move[i] = up_cross_idx[0] + 1
                else:
                    labels[i] = -1
                    magnitudes[i] = returns[down_cross_idx[0]] / current_atr
                    days_to_move[i] = down_cross_idx[0] + 1
            elif len(up_cross_idx) > 0:
                labels[i] = 1
                magnitudes[i] = returns[up_cross_idx[0]] / current_atr
                days_to_move[i] = up_cross_idx[0] + 1
            elif len(down_cross_idx) > 0:
                labels[i] = -1
                magnitudes[i] = returns[down_cross_idx[0]] / current_atr
                days_to_move[i] = down_cross_idx[0] + 1
            else:
                # No significant move
                labels[i] = 0
                magnitudes[i] = np.max(np.abs(returns)) / current_atr if len(returns) > 0 else 0
        
        df[label_col] = labels
        df[magnitude_col] = magnitudes
        df[days_to_move_col] = days_to_move
        
        # Also create binary version (UP vs not UP)
        df[f'label_{horizon}d_up'] = (labels == 1).astype(int)
        df[f'label_{horizon}d_down'] = (labels == -1).astype(int)
        
        return df
    
    def _add_consolidation_label(self, df: pd.DataFrame, atr_col: str,
                                 min_days: int = 10,
                                 width_multiplier: float = 1.2) -> pd.DataFrame:
        """
        Label consolidation periods.
        
        Consolidation is when price stays within a narrow range for min_days.
        
        Args:
            df: DataFrame with data
            atr_col: Column name for ATR
            min_days: Minimum days for consolidation
            width_multiplier: Max width as multiple of ATR
            
        Returns:
            DataFrame with consolidation label
        """
        label_col = 'label_consolidation'
        width_col = 'label_consolidation_width'
        
        labels = np.zeros(len(df), dtype=int)
        widths = np.zeros(len(df), dtype=float)
        
        for i in range(min_days, len(df)):
            # Look back min_days
            lookback = df.iloc[i-min_days:i+1]
            
            price_range = lookback['High'].max() - lookback['Low'].min()
            current_atr = df[atr_col].iloc[i]
            
            if pd.isna(current_atr) or current_atr == 0:
                continue
            
            threshold = width_multiplier * current_atr
            
            if price_range <= threshold:
                labels[i] = 1
                widths[i] = price_range / current_atr
        
        df[label_col] = labels
        df[width_col] = widths
        
        return df
    
    def _add_breakout_label(self, df: pd.DataFrame, atr_col: str,
                           lookback: int = 20,
                           breakout_threshold: float = 0.5,
                           volume_threshold: float = 1.5,
                           confirmation_days: int = 3) -> pd.DataFrame:
        """
        Label breakouts and false breakouts.
        
        Breakout: Close > recent high + threshold*ATR with volume surge
        False breakout: Falls back below breakout level within confirmation period
        
        Args:
            df: DataFrame with data
            atr_col: Column name for ATR
            lookback: Days to look back for consolidation high
            breakout_threshold: ATR multiplier for breakout threshold
            volume_threshold: Volume multiplier for surge detection
            confirmation_days: Days to confirm breakout
            
        Returns:
            DataFrame with breakout labels
        """
        breakout_col = 'label_breakout'
        false_breakout_col = 'label_false_breakout'
        confirmed_breakout_col = 'label_confirmed_breakout'
        
        breakouts = np.zeros(len(df), dtype=int)
        false_breakouts = np.zeros(len(df), dtype=int)
        confirmed_breakouts = np.zeros(len(df), dtype=int)
        
        for i in range(lookback, len(df) - confirmation_days):
            # Get recent high
            recent_high = df['High'].iloc[i-lookback:i].max()
            current_close = df['Close'].iloc[i]
            current_atr = df[atr_col].iloc[i]
            
            if pd.isna(current_atr) or current_atr == 0:
                continue
            
            # Check for breakout
            breakout_level = recent_high + breakout_threshold * current_atr
            
            if current_close > breakout_level:
                # Check volume surge
                has_volume_surge = True
                if 'Volume' in df.columns:
                    recent_volume_median = df['Volume'].iloc[i-lookback:i].median()
                    current_volume = df['Volume'].iloc[i]
                    has_volume_surge = current_volume > volume_threshold * recent_volume_median
                
                if has_volume_surge:
                    breakouts[i] = 1
                    
                    # Check if it's a false breakout
                    future_prices = df['Close'].iloc[i+1:i+1+confirmation_days]
                    if len(future_prices) > 0 and (future_prices < breakout_level - 0.5*current_atr).any():
                        false_breakouts[i] = 1
                    else:
                        confirmed_breakouts[i] = 1
        
        df[breakout_col] = breakouts
        df[false_breakout_col] = false_breakouts
        df[confirmed_breakout_col] = confirmed_breakouts
        
        return df
    
    def _add_reversal_label(self, df: pd.DataFrame, atr_col: str,
                           move_threshold: float = 2.0,
                           reversal_threshold: float = 2.0,
                           reversal_horizon: int = 10) -> pd.DataFrame:
        """
        Label reversal points.
        
        Reversal: After significant move, direction changes significantly.
        
        Args:
            df: DataFrame with data
            atr_col: Column name for ATR
            move_threshold: ATR multiplier for significant prior move
            reversal_threshold: ATR multiplier for reversal magnitude
            reversal_horizon: Days to look forward for reversal
            
        Returns:
            DataFrame with reversal label
        """
        label_col = 'label_reversal'
        reversal_type_col = 'label_reversal_type'  # 1 for top, -1 for bottom
        
        labels = np.zeros(len(df), dtype=int)
        reversal_types = np.zeros(len(df), dtype=int)
        
        for i in range(20, len(df) - reversal_horizon):
            current_atr = df[atr_col].iloc[i]
            
            if pd.isna(current_atr) or current_atr == 0:
                continue
            
            # Check for prior upward move
            prior_10d_return = df['Close'].iloc[i] - df['Close'].iloc[i-10]
            
            if prior_10d_return > move_threshold * current_atr:
                # Look for reversal (downward)
                future_return = df['Close'].iloc[i:i+reversal_horizon].min() - df['Close'].iloc[i]
                
                if future_return < -reversal_threshold * current_atr:
                    labels[i] = 1
                    reversal_types[i] = 1  # Top/peak reversal
            
            # Check for prior downward move
            elif prior_10d_return < -move_threshold * current_atr:
                # Look for reversal (upward)
                future_return = df['Close'].iloc[i:i+reversal_horizon].max() - df['Close'].iloc[i]
                
                if future_return > reversal_threshold * current_atr:
                    labels[i] = 1
                    reversal_types[i] = -1  # Bottom reversal
        
        df[label_col] = labels
        df[reversal_type_col] = reversal_types
        
        return df
    
    def _add_regime_label(self, df: pd.DataFrame,
                         bull_return_threshold: float = 0.10,
                         bull_vol_threshold: float = 0.30,
                         window: int = 40) -> pd.DataFrame:
        """
        Label market regime (bull/bear/neutral).
        
        Bull zone: positive returns and moderate volatility
        Bear zone: negative returns or high volatility
        
        Args:
            df: DataFrame with data
            bull_return_threshold: Return threshold for bull zone
            bull_vol_threshold: Max volatility for bull zone
            window: Window for regime calculation
            
        Returns:
            DataFrame with regime label
        """
        # Calculate rolling metrics
        rolling_return = (df['Close'] / df['Close'].shift(window)) - 1
        rolling_vol = df['log_return'].rolling(window).std() * np.sqrt(252)  # Annualized
        
        # Define regimes
        # 1 = bull, 0 = neutral, -1 = bear
        regime = np.zeros(len(df), dtype=int)
        
        bull_mask = (rolling_return > bull_return_threshold) & (rolling_vol < bull_vol_threshold)
        bear_mask = (rolling_return < -bull_return_threshold) | (rolling_vol > bull_vol_threshold * 1.5)
        
        regime[bull_mask] = 1
        regime[bear_mask] = -1
        
        df['label_regime'] = regime
        df['label_regime_return'] = rolling_return
        df['label_regime_volatility'] = rolling_vol
        
        return df
    
    def get_label_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of all label columns.
        
        Args:
            df: DataFrame with labels
            
        Returns:
            List of label column names
        """
        label_cols = [col for col in df.columns if col.startswith('label_')]
        return label_cols
    
    def get_label_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics about labels (class balance, etc.).
        
        Args:
            df: DataFrame with labels
            
        Returns:
            DataFrame with label statistics
        """
        label_cols = self.get_label_columns(df)
        
        stats = []
        for col in label_cols:
            if df[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
                value_counts = df[col].value_counts()
                stats.append({
                    'label': col,
                    'total_samples': len(df[~df[col].isna()]),
                    'positive_samples': (df[col] == 1).sum(),
                    'negative_samples': (df[col] == -1).sum() if (df[col] == -1).any() else 0,
                    'zero_samples': (df[col] == 0).sum() if (df[col] == 0).any() else 0,
                    'positive_rate': (df[col] == 1).mean(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                })
        
        return pd.DataFrame(stats)
