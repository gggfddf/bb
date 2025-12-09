"""
Label generation module - creates training labels for pattern detection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from config import LabelConfig


class LabelGenerator:
    """
    Generates labels for supervised learning based on future price movements.
    All labels use ATR-based thresholds for scale invariance.
    """
    
    def __init__(self, config: LabelConfig):
        self.config = config
        
    def generate_all_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all label types.
        
        Args:
            df: DataFrame with OHLCV data and ATR already calculated
            
        Returns:
            DataFrame with all labels added
        """
        df = df.copy()
        
        # Major directional move labels
        df = self._create_major_move_labels(df)
        
        # Breakout and consolidation labels
        df = self._create_breakout_labels(df)
        
        # Reversal labels
        df = self._create_reversal_labels(df)
        
        # Cycle/regime labels
        df = self._create_cycle_labels(df)
        
        # Pattern-specific labels
        df = self._create_pattern_labels(df)
        
        return df
    
    def _create_major_move_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels for major directional moves using ATR-based thresholds.
        
        Label logic:
        - For horizon H, label is 1 (UP) if max_future_return >= k*ATR before -k*ATR
        - Label is -1 (DOWN) if min_future_return <= -k*ATR before +k*ATR
        - Label is 0 (NONE) otherwise
        """
        df = df.copy()
        
        # Use ATR_14 as default, fallback to ATR_20
        atr_col = 'ATR_14' if 'ATR_14' in df.columns else 'ATR_20'
        if atr_col not in df.columns:
            raise ValueError("ATR column not found. Run preprocessing first.")
        
        atr = df[atr_col]
        close = df['Close']
        
        for horizon in self.config.prediction_horizons:
            label_col = f'label_move_{horizon}d'
            prob_col = f'prob_move_{horizon}d'
            magnitude_col = f'magnitude_move_{horizon}d'
            
            # Initialize labels
            df[label_col] = 0  # 0 = no major move
            df[prob_col] = 0.0
            df[magnitude_col] = 0.0
            
            # Calculate for each row
            for i in range(len(df) - horizon):
                current_close = close.iloc[i]
                current_atr = atr.iloc[i]
                
                if pd.isna(current_atr) or current_atr == 0:
                    continue
                
                # Future prices within horizon
                future_prices = close.iloc[i+1:i+horizon+1]
                
                # Calculate returns from current close
                future_returns = (future_prices - current_close) / current_close
                
                # Thresholds
                up_threshold = self.config.atr_multiplier_up * current_atr / current_close
                down_threshold = -self.config.atr_multiplier_down * current_atr / current_close
                
                # Find first breach
                up_breach_mask = future_returns >= up_threshold
                down_breach_mask = future_returns <= down_threshold
                
                up_breach_idx = up_breach_mask.idxmax() if up_breach_mask.any() else None
                down_breach_idx = down_breach_mask.idxmax() if down_breach_mask.any() else None
                
                # Determine label based on which threshold is breached first
                if up_breach_idx is not None and down_breach_idx is not None:
                    if future_prices.index.get_loc(up_breach_idx) < future_prices.index.get_loc(down_breach_idx):
                        df.loc[df.index[i], label_col] = 1  # UP
                        df.loc[df.index[i], magnitude_col] = future_returns.max()
                    else:
                        df.loc[df.index[i], label_col] = -1  # DOWN
                        df.loc[df.index[i], magnitude_col] = future_returns.min()
                elif up_breach_idx is not None:
                    df.loc[df.index[i], label_col] = 1  # UP
                    df.loc[df.index[i], magnitude_col] = future_returns.max()
                elif down_breach_idx is not None:
                    df.loc[df.index[i], label_col] = -1  # DOWN
                    df.loc[df.index[i], magnitude_col] = future_returns.min()
                else:
                    # No threshold breached
                    df.loc[df.index[i], magnitude_col] = future_returns.iloc[-1] if len(future_returns) > 0 else 0
            
            # Convert to binary classification (up vs not-up) for probability
            df[f'label_up_{horizon}d'] = (df[label_col] == 1).astype(int)
            df[f'label_down_{horizon}d'] = (df[label_col] == -1).astype(int)
        
        return df
    
    def _create_breakout_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels for consolidation, breakout, and false breakout patterns.
        """
        df = df.copy()
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df.get('Volume', pd.Series(index=df.index))
        atr = df.get('ATR_14', df.get('ATR_20'))
        
        # Initialize labels
        df['in_consolidation'] = 0
        df['consolidation_breakout'] = 0
        df['false_breakout'] = 0
        df['consolidation_width'] = np.nan
        df['consolidation_days'] = 0
        
        min_days = self.config.consolidation_min_days
        max_days = self.config.consolidation_max_days
        
        for i in range(max_days, len(df)):
            # Check for consolidation in lookback window
            for lookback in range(min_days, max_days + 1):
                window_start = i - lookback
                window_close = close.iloc[window_start:i]
                window_high = high.iloc[window_start:i]
                window_low = low.iloc[window_start:i]
                window_atr = atr.iloc[i]
                
                if pd.isna(window_atr) or window_atr == 0:
                    continue
                
                # Calculate consolidation width
                range_width = window_high.max() - window_low.min()
                median_atr = atr.iloc[window_start:i].median()
                
                threshold = self.config.consolidation_width_multiplier * median_atr
                
                # Check if in consolidation
                if range_width <= threshold:
                    df.loc[df.index[i], 'in_consolidation'] = 1
                    df.loc[df.index[i], 'consolidation_width'] = range_width / median_atr
                    df.loc[df.index[i], 'consolidation_days'] = lookback
                    
                    consolidation_high = window_high.max()
                    consolidation_low = window_low.min()
                    
                    # Check for breakout in next few days
                    if i < len(df) - self.config.breakout_confirmation_days:
                        future_window = slice(i, i + self.config.breakout_confirmation_days)
                        future_close = close.iloc[future_window]
                        future_volume = volume.iloc[future_window] if not volume.isna().all() else None
                        
                        # Breakout threshold
                        breakout_threshold = consolidation_high + self.config.breakout_price_threshold * window_atr
                        
                        # Check if price breaks above
                        if (future_close > breakout_threshold).any():
                            # Check volume confirmation if available
                            volume_confirmed = True
                            if future_volume is not None and not future_volume.isna().all():
                                median_vol = volume.iloc[window_start:i].median()
                                if median_vol > 0:
                                    volume_confirmed = (
                                        future_volume > median_vol * self.config.breakout_volume_multiplier
                                    ).any()
                            
                            if volume_confirmed:
                                df.loc[df.index[i], 'consolidation_breakout'] = 1
                                
                                # Check for false breakout
                                if i < len(df) - self.config.false_breakout_window:
                                    future_prices = close.iloc[i:i + self.config.false_breakout_window]
                                    fallback_threshold = breakout_threshold - self.config.false_breakout_threshold * window_atr
                                    
                                    if (future_prices < fallback_threshold).any():
                                        df.loc[df.index[i], 'false_breakout'] = 1
                    
                    break  # Use first valid consolidation found
        
        return df
    
    def _create_reversal_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels for trend reversals.
        
        Reversal definition:
        - Prior move of magnitude > m * ATR
        - Direction change of n * ATR within H days
        """
        df = df.copy()
        
        close = df['Close']
        atr = df.get('ATR_14', df.get('ATR_20'))
        
        df['trend_reversal'] = 0
        df['reversal_strength'] = 0.0
        
        window = self.config.reversal_window
        
        for i in range(window * 2, len(df) - window):
            current_atr = atr.iloc[i]
            
            if pd.isna(current_atr) or current_atr == 0:
                continue
            
            # Check prior move
            prior_prices = close.iloc[i-window:i]
            prior_move = close.iloc[i] - close.iloc[i-window]
            prior_move_atr = abs(prior_move) / current_atr
            
            if prior_move_atr < self.config.reversal_prior_move_multiplier:
                continue
            
            # Determine prior direction
            prior_direction = 1 if prior_move > 0 else -1
            
            # Check for reversal in next window
            if i < len(df) - window:
                future_prices = close.iloc[i:i+window]
                future_move = future_prices.iloc[-1] - close.iloc[i]
                future_move_atr = abs(future_move) / current_atr
                
                if future_move_atr >= self.config.reversal_change_multiplier:
                    future_direction = 1 if future_move > 0 else -1
                    
                    # Reversal if direction changed
                    if future_direction != prior_direction:
                        df.loc[df.index[i], 'trend_reversal'] = 1
                        df.loc[df.index[i], 'reversal_strength'] = future_move_atr
        
        return df
    
    def _create_cycle_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create regime/cycle labels (bull zone, bear zone, sideways).
        """
        df = df.copy()
        
        # Calculate rolling returns for regime detection
        returns_40d = df['Close'].pct_change(40)
        volatility_40d = df['log_return'].rolling(40).std() if 'log_return' in df.columns else None
        
        df['market_regime'] = 'sideways'  # default
        df['bull_zone'] = 0
        df['bear_zone'] = 0
        df['sideways_zone'] = 1
        
        # Define regime thresholds (can be adjusted)
        bull_return_threshold = 0.10  # 10% gain over 40 days
        bear_return_threshold = -0.10  # 10% loss over 40 days
        
        # Simple regime classification
        bull_mask = returns_40d > bull_return_threshold
        bear_mask = returns_40d < bear_return_threshold
        
        df.loc[bull_mask, 'market_regime'] = 'bull'
        df.loc[bull_mask, 'bull_zone'] = 1
        df.loc[bull_mask, 'sideways_zone'] = 0
        
        df.loc[bear_mask, 'market_regime'] = 'bear'
        df.loc[bear_mask, 'bear_zone'] = 1
        df.loc[bear_mask, 'sideways_zone'] = 0
        
        # Calculate days in current regime
        regime_changes = (df['market_regime'] != df['market_regime'].shift()).cumsum()
        df['days_in_regime'] = df.groupby(regime_changes).cumcount() + 1
        
        return df
    
    def _create_pattern_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels for specific chart patterns.
        """
        df = df.copy()
        
        # Momentum burst pattern
        df = self._label_momentum_burst(df)
        
        # V-shaped reversal
        df = self._label_v_reversal(df)
        
        return df
    
    def _label_momentum_burst(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label momentum burst: rapid 3-day move > 2*sigma with volume surge.
        """
        df = df.copy()
        
        # 3-day cumulative return
        returns_3d = df['Close'].pct_change(3)
        
        # Rolling statistics
        returns_std = df['log_return'].rolling(20).std() if 'log_return' in df.columns else None
        volume_median = df['Volume'].rolling(20).median() if 'Volume' in df.columns else None
        
        df['momentum_burst'] = 0
        
        if returns_std is not None:
            threshold = 2 * returns_std * np.sqrt(3)  # 2-sigma for 3 days
            momentum_condition = abs(returns_3d) > threshold
            
            # Volume confirmation if available
            if volume_median is not None and not volume_median.isna().all():
                volume_surge = df['Volume'] > volume_median * 1.5
                momentum_condition = momentum_condition & volume_surge
            
            df.loc[momentum_condition, 'momentum_burst'] = 1
        
        return df
    
    def _label_v_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label V-shaped reversal: sharp decline followed by sharp recovery.
        """
        df = df.copy()
        
        close = df['Close']
        window = 10
        
        df['v_reversal'] = 0
        
        for i in range(window, len(df) - window):
            # Check for V-shape: decline then rise
            first_half = close.iloc[i-window:i]
            second_half = close.iloc[i:i+window]
            
            if len(first_half) == 0 or len(second_half) == 0:
                continue
            
            decline = (first_half.iloc[-1] - first_half.iloc[0]) / first_half.iloc[0]
            recovery = (second_half.iloc[-1] - second_half.iloc[0]) / second_half.iloc[0]
            
            # V-reversal: significant decline followed by recovery
            if decline < -0.05 and recovery > 0.05:  # 5% moves
                # Check if i is the trough
                if close.iloc[i] == close.iloc[i-window:i+window].min():
                    df.loc[df.index[i], 'v_reversal'] = 1
        
        return df
    
    def get_label_columns(self, df: pd.DataFrame, label_type: str = 'all') -> list:
        """
        Get list of label column names.
        
        Args:
            label_type: 'all', 'move', 'breakout', 'reversal', 'cycle', 'pattern'
        """
        if label_type == 'all':
            # All columns starting with 'label_' or specific pattern labels
            return [col for col in df.columns if 
                    col.startswith('label_') or 
                    col in ['in_consolidation', 'consolidation_breakout', 'false_breakout',
                           'trend_reversal', 'bull_zone', 'bear_zone', 'momentum_burst', 'v_reversal']]
        
        elif label_type == 'move':
            return [col for col in df.columns if col.startswith('label_move_') or col.startswith('label_up_')]
        
        elif label_type == 'breakout':
            return ['in_consolidation', 'consolidation_breakout', 'false_breakout']
        
        elif label_type == 'reversal':
            return ['trend_reversal', 'v_reversal']
        
        elif label_type == 'cycle':
            return ['bull_zone', 'bear_zone', 'sideways_zone', 'market_regime']
        
        elif label_type == 'pattern':
            return ['momentum_burst', 'v_reversal', 'consolidation_breakout']
        
        return []
    
    def get_label_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate statistics for each label type.
        """
        stats = {}
        
        # Major move labels
        for horizon in self.config.prediction_horizons:
            label_col = f'label_move_{horizon}d'
            if label_col in df.columns:
                value_counts = df[label_col].value_counts()
                total = len(df[df[label_col].notna()])
                stats[label_col] = {
                    'total_samples': total,
                    'up_count': value_counts.get(1, 0),
                    'down_count': value_counts.get(-1, 0),
                    'neutral_count': value_counts.get(0, 0),
                    'up_pct': value_counts.get(1, 0) / total if total > 0 else 0,
                    'down_pct': value_counts.get(-1, 0) / total if total > 0 else 0,
                }
        
        # Breakout labels
        for label in ['in_consolidation', 'consolidation_breakout', 'false_breakout']:
            if label in df.columns:
                count = df[label].sum()
                total = len(df[df[label].notna()])
                stats[label] = {
                    'count': count,
                    'frequency': count / total if total > 0 else 0,
                }
        
        # Reversal labels
        if 'trend_reversal' in df.columns:
            count = df['trend_reversal'].sum()
            total = len(df[df['trend_reversal'].notna()])
            stats['trend_reversal'] = {
                'count': count,
                'frequency': count / total if total > 0 else 0,
                'avg_strength': df[df['trend_reversal'] == 1]['reversal_strength'].mean() if count > 0 else 0,
            }
        
        return stats
