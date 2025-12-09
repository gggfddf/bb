"""
Cycle detection module - identifies periodicities and market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import signal, stats
from scipy.fft import fft, fftfreq

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets not available. Install with: pip install PyWavelets")

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not available. Install with: pip install hmmlearn")

from config import CycleConfig


class CycleDetector:
    """
    Detects cycles and periodicities in time series data.
    """
    
    def __init__(self, config: CycleConfig):
        self.config = config
        self.detected_cycles = {}
        
    def detect_autocorrelation_cycles(self, series: pd.Series) -> Dict:
        """
        Detect cycles using autocorrelation and partial autocorrelation.
        
        Args:
            series: Time series to analyze
            
        Returns:
            Dictionary with significant lags and correlation values
        """
        series_clean = series.dropna()
        
        if len(series_clean) < self.config.max_lag:
            warnings.warn(f"Series too short for max_lag={self.config.max_lag}")
            return {}
        
        # Calculate autocorrelation
        acf_values = []
        for lag in range(1, self.config.max_lag + 1):
            if lag < len(series_clean):
                acf = series_clean.autocorr(lag=lag)
                acf_values.append(acf)
            else:
                acf_values.append(np.nan)
        
        acf_values = np.array(acf_values)
        
        # Calculate partial autocorrelation (simplified approach)
        pacf_values = self._calculate_pacf(series_clean.values, self.config.max_lag)
        
        # Find significant lags
        threshold = self.config.significant_threshold
        significant_acf_lags = np.where(np.abs(acf_values) > threshold)[0] + 1
        significant_pacf_lags = np.where(np.abs(pacf_values) > threshold)[0] + 1
        
        # Find peaks in ACF (potential cycle periods)
        peaks, properties = signal.find_peaks(acf_values, height=threshold)
        peak_lags = peaks + 1
        
        results = {
            'acf': acf_values,
            'pacf': pacf_values,
            'significant_acf_lags': significant_acf_lags.tolist(),
            'significant_pacf_lags': significant_pacf_lags.tolist(),
            'peak_lags': peak_lags.tolist(),
            'peak_values': acf_values[peaks].tolist() if len(peaks) > 0 else [],
        }
        
        self.detected_cycles['autocorrelation'] = results
        return results
    
    @staticmethod
    def _calculate_pacf(series: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Calculate partial autocorrelation function.
        Simplified implementation using Yule-Walker equations.
        """
        pacf = np.zeros(max_lag)
        
        # Remove mean
        series_centered = series - series.mean()
        
        for k in range(1, max_lag + 1):
            if k == 1:
                # PACF at lag 1 equals ACF at lag 1
                pacf[k-1] = np.corrcoef(series_centered[:-1], series_centered[1:])[0, 1]
            else:
                # Use Yule-Walker equations
                # This is a simplified approach
                gamma = np.correlate(series_centered, series_centered, mode='full')
                gamma = gamma[len(series_centered)-1:]
                
                if k < len(gamma):
                    R = np.array([gamma[i] for i in range(k)])
                    r = np.array([gamma[i] for i in range(1, k+1)])
                    
                    # Create autocorrelation matrix
                    R_matrix = np.zeros((k, k))
                    for i in range(k):
                        for j in range(k):
                            R_matrix[i, j] = gamma[abs(i - j)]
                    
                    try:
                        phi = np.linalg.solve(R_matrix, r)
                        pacf[k-1] = phi[-1]
                    except:
                        pacf[k-1] = 0
        
        return pacf
    
    def detect_spectral_cycles(self, series: pd.Series) -> Dict:
        """
        Detect cycles using Fourier Transform (spectral analysis).
        
        Args:
            series: Time series to analyze
            
        Returns:
            Dictionary with dominant frequencies and periods
        """
        series_clean = series.dropna().values
        
        if len(series_clean) < 10:
            return {}
        
        # Detrend the series
        detrended = signal.detrend(series_clean)
        
        # Apply FFT
        N = len(detrended)
        yf = fft(detrended)
        xf = fftfreq(N, 1)  # Assuming daily data (sample spacing = 1)
        
        # Only positive frequencies
        positive_freq_idx = xf > 0
        xf_positive = xf[positive_freq_idx]
        yf_positive = np.abs(yf[positive_freq_idx])
        
        # Convert frequencies to periods
        periods = 1 / xf_positive
        
        # Filter periods within valid range
        valid_idx = (periods >= self.config.fft_min_period) & (periods <= self.config.fft_max_period)
        periods_valid = periods[valid_idx]
        power_valid = yf_positive[valid_idx]
        
        if len(periods_valid) == 0:
            return {}
        
        # Find dominant periods (peaks in power spectrum)
        peaks, properties = signal.find_peaks(power_valid, prominence=np.std(power_valid))
        
        if len(peaks) > 0:
            # Sort by power
            sorted_idx = np.argsort(power_valid[peaks])[::-1]
            top_periods = periods_valid[peaks[sorted_idx]][:10]  # Top 10 periods
            top_power = power_valid[peaks[sorted_idx]][:10]
        else:
            top_periods = []
            top_power = []
        
        results = {
            'periods': periods_valid.tolist(),
            'power': power_valid.tolist(),
            'dominant_periods': top_periods.tolist(),
            'dominant_power': top_power.tolist(),
        }
        
        self.detected_cycles['spectral'] = results
        return results
    
    def detect_wavelet_cycles(self, series: pd.Series) -> Dict:
        """
        Detect time-localized cycles using Continuous Wavelet Transform.
        
        Args:
            series: Time series to analyze
            
        Returns:
            Dictionary with wavelet analysis results
        """
        if not PYWT_AVAILABLE:
            warnings.warn("PyWavelets not available")
            return {}
        
        series_clean = series.dropna().values
        
        if len(series_clean) < 10:
            return {}
        
        # Continuous Wavelet Transform
        wavelet = self.config.wavelet_type
        scales = self.config.wavelet_scales
        
        # Limit scales to reasonable range
        max_scale = min(len(series_clean) // 4, len(scales))
        scales_used = scales[:max_scale]
        
        try:
            coefficients, frequencies = pywt.cwt(
                series_clean,
                scales_used,
                wavelet
            )
            
            # Calculate power
            power = np.abs(coefficients) ** 2
            
            # Find dominant scale (period) at each time point
            dominant_scales = scales_used[np.argmax(power, axis=0)]
            
            # Average power across time for each scale
            avg_power = power.mean(axis=1)
            
            # Find peaks in average power
            peaks, _ = signal.find_peaks(avg_power)
            
            if len(peaks) > 0:
                dominant_periods = scales_used[peaks]
            else:
                dominant_periods = []
            
            results = {
                'scales': scales_used.tolist(),
                'avg_power': avg_power.tolist(),
                'dominant_periods': dominant_periods.tolist(),
                'power_shape': power.shape,
            }
            
            self.detected_cycles['wavelet'] = results
            return results
        except Exception as e:
            warnings.warn(f"Wavelet transform failed: {e}")
            return {}
    
    def detect_regime_hmm(self, df: pd.DataFrame, 
                         features: Optional[List[str]] = None) -> Dict:
        """
        Detect market regimes using Hidden Markov Model.
        
        Args:
            df: DataFrame with price data and features
            features: List of feature columns to use (default: returns and volatility)
            
        Returns:
            Dictionary with regime labels and model info
        """
        if not HMM_AVAILABLE:
            warnings.warn("hmmlearn not available")
            return {}
        
        # Default features: returns and volatility
        if features is None:
            if 'log_return' not in df.columns:
                df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            df['volatility'] = df['log_return'].rolling(20).std()
            features = ['log_return', 'volatility']
        
        # Prepare data
        X = df[features].dropna().values
        
        if len(X) < self.config.hmm_n_states * 10:
            warnings.warn("Insufficient data for HMM")
            return {}
        
        # Fit Gaussian HMM
        try:
            model = hmm.GaussianHMM(
                n_components=self.config.hmm_n_states,
                covariance_type='full',
                n_iter=self.config.hmm_n_iter,
                random_state=42
            )
            
            model.fit(X)
            
            # Predict regimes
            states = model.predict(X)
            
            # Map states to regime names based on characteristics
            regime_names = self._interpret_hmm_states(X, states, self.config.hmm_n_states)
            
            # Calculate regime durations
            regime_durations = self._calculate_regime_durations(states)
            
            results = {
                'states': states.tolist(),
                'regime_names': regime_names,
                'n_states': self.config.hmm_n_states,
                'means': model.means_.tolist(),
                'transition_matrix': model.transmat_.tolist(),
                'regime_durations': regime_durations,
            }
            
            self.detected_cycles['hmm'] = results
            return results
        except Exception as e:
            warnings.warn(f"HMM fitting failed: {e}")
            return {}
    
    @staticmethod
    def _interpret_hmm_states(X: np.ndarray, states: np.ndarray, 
                             n_states: int) -> List[str]:
        """
        Interpret HMM states as market regimes.
        
        Assumes X has [returns, volatility] as columns.
        """
        # Calculate mean returns and volatility for each state
        state_stats = []
        for state in range(n_states):
            mask = states == state
            if mask.sum() > 0:
                mean_return = X[mask, 0].mean() if X.shape[1] > 0 else 0
                mean_vol = X[mask, 1].mean() if X.shape[1] > 1 else 0
                state_stats.append({
                    'state': state,
                    'mean_return': mean_return,
                    'mean_volatility': mean_vol
                })
        
        # Sort states by mean return
        state_stats.sort(key=lambda x: x['mean_return'])
        
        # Assign names
        regime_map = {}
        if n_states == 2:
            regime_map[state_stats[0]['state']] = 'bear'
            regime_map[state_stats[1]['state']] = 'bull'
        elif n_states == 3:
            regime_map[state_stats[0]['state']] = 'bear'
            regime_map[state_stats[1]['state']] = 'sideways'
            regime_map[state_stats[2]['state']] = 'bull'
        else:
            for i, stat in enumerate(state_stats):
                regime_map[stat['state']] = f'regime_{i}'
        
        return [regime_map[s] for s in states]
    
    @staticmethod
    def _calculate_regime_durations(states: np.ndarray) -> Dict:
        """Calculate average duration of each regime."""
        regime_changes = np.concatenate([[0], np.where(np.diff(states) != 0)[0] + 1, [len(states)]])
        durations = np.diff(regime_changes)
        regime_labels = states[regime_changes[:-1]]
        
        duration_stats = {}
        for regime in np.unique(states):
            regime_durations = durations[regime_labels == regime]
            if len(regime_durations) > 0:
                duration_stats[int(regime)] = {
                    'mean': float(regime_durations.mean()),
                    'median': float(np.median(regime_durations)),
                    'std': float(regime_durations.std()),
                    'min': int(regime_durations.min()),
                    'max': int(regime_durations.max()),
                }
        
        return duration_stats
    
    def run_all_cycle_detection(self, df: pd.DataFrame,
                               price_col: str = 'Close') -> Dict:
        """
        Run all cycle detection methods.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            
        Returns:
            Dictionary with all cycle detection results
        """
        series = df[price_col]
        
        print("Detecting autocorrelation cycles...")
        autocorr_results = self.detect_autocorrelation_cycles(series)
        
        print("Detecting spectral cycles...")
        spectral_results = self.detect_spectral_cycles(series)
        
        print("Detecting wavelet cycles...")
        wavelet_results = self.detect_wavelet_cycles(series)
        
        print("Detecting HMM regimes...")
        hmm_results = self.detect_regime_hmm(df)
        
        return {
            'autocorrelation': autocorr_results,
            'spectral': spectral_results,
            'wavelet': wavelet_results,
            'hmm': hmm_results,
        }
    
    def summarize_cycles(self) -> pd.DataFrame:
        """
        Summarize detected cycles across all methods.
        
        Returns:
            DataFrame with cycle summary
        """
        summaries = []
        
        # Autocorrelation cycles
        if 'autocorrelation' in self.detected_cycles:
            acf_data = self.detected_cycles['autocorrelation']
            for lag in acf_data.get('peak_lags', []):
                summaries.append({
                    'method': 'autocorrelation',
                    'period_days': lag,
                    'strength': acf_data['acf'][lag-1] if lag-1 < len(acf_data['acf']) else None,
                })
        
        # Spectral cycles
        if 'spectral' in self.detected_cycles:
            spec_data = self.detected_cycles['spectral']
            for period, power in zip(spec_data.get('dominant_periods', []),
                                    spec_data.get('dominant_power', [])):
                summaries.append({
                    'method': 'spectral',
                    'period_days': period,
                    'strength': power,
                })
        
        # Wavelet cycles
        if 'wavelet' in self.detected_cycles:
            wav_data = self.detected_cycles['wavelet']
            for period in wav_data.get('dominant_periods', []):
                summaries.append({
                    'method': 'wavelet',
                    'period_days': period,
                    'strength': None,  # Could add power values
                })
        
        if summaries:
            return pd.DataFrame(summaries).sort_values('period_days')
        else:
            return pd.DataFrame()


class RegimeAnalyzer:
    """
    Analyzes market regimes and their characteristics.
    """
    
    @staticmethod
    def calculate_regime_statistics(df: pd.DataFrame, 
                                    regime_col: str = 'market_regime') -> Dict:
        """
        Calculate statistics for each market regime.
        
        Args:
            df: DataFrame with regime labels
            regime_col: Column name for regime labels
            
        Returns:
            Dictionary with regime statistics
        """
        if regime_col not in df.columns:
            return {}
        
        stats = {}
        
        for regime in df[regime_col].unique():
            mask = df[regime_col] == regime
            regime_data = df[mask]
            
            if 'log_return' in regime_data.columns:
                returns = regime_data['log_return'].dropna()
            else:
                returns = regime_data['Close'].pct_change().dropna()
            
            stats[regime] = {
                'n_days': len(regime_data),
                'frequency': len(regime_data) / len(df),
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe': returns.mean() / (returns.std() + 1e-8) * np.sqrt(252),
                'max_drawdown': (regime_data['Close'] / regime_data['Close'].cummax() - 1).min(),
            }
        
        return stats
    
    @staticmethod
    def detect_regime_transitions(df: pd.DataFrame,
                                  regime_col: str = 'market_regime') -> pd.DataFrame:
        """
        Detect regime transition points.
        
        Returns:
            DataFrame with transition information
        """
        if regime_col not in df.columns:
            return pd.DataFrame()
        
        # Find where regime changes
        regime_changes = df[regime_col] != df[regime_col].shift(1)
        transition_indices = df[regime_changes].index
        
        transitions = []
        for i in range(1, len(transition_indices)):
            idx = transition_indices[i]
            prev_idx = transition_indices[i-1]
            
            transitions.append({
                'date': idx,
                'from_regime': df.loc[prev_idx, regime_col],
                'to_regime': df.loc[idx, regime_col],
                'duration_days': (idx - prev_idx).days if hasattr(idx, 'days') else i - (i-1),
            })
        
        return pd.DataFrame(transitions)
