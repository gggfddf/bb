"""
Unsupervised pattern discovery module using Matrix Profile, clustering, and motif detection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
from collections import defaultdict

try:
    import stumpy
    STUMPY_AVAILABLE = True
except ImportError:
    STUMPY_AVAILABLE = False
    warnings.warn("stumpy not available for Matrix Profile. Install with: pip install stumpy")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import hdbscan
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    warnings.warn("scikit-learn or hdbscan not available")

from config import PatternConfig


class PatternDiscovery:
    """
    Discovers recurring patterns and motifs in time series data.
    """
    
    def __init__(self, config: PatternConfig):
        self.config = config
        self.motifs = []
        self.clusters = None
        self.pattern_library = defaultdict(dict)
        
    def discover_motifs_matrix_profile(self, series: pd.Series,
                                       return_top_k: Optional[int] = None) -> List[Dict]:
        """
        Discover recurring motifs using Matrix Profile.
        
        Args:
            series: Time series to analyze
            return_top_k: Number of top motifs to return (default: config value)
            
        Returns:
            List of dictionaries containing motif information
        """
        if not STUMPY_AVAILABLE:
            warnings.warn("Matrix Profile requires stumpy. Skipping motif discovery.")
            return []
        
        window_size = self.config.mp_window_size
        top_k = return_top_k or self.config.mp_top_k_motifs
        
        # Remove NaN values
        series_clean = series.dropna()
        
        if len(series_clean) < window_size * 2:
            warnings.warn(f"Series too short for window size {window_size}")
            return []
        
        # Compute matrix profile
        mp = stumpy.stump(series_clean.values, m=window_size)
        
        # Extract motifs
        motifs_discovered = []
        motif_distances, motif_indices = stumpy.motifs(
            series_clean.values,
            mp[:, 0],
            max_distance=np.percentile(mp[:, 0], 25),  # Bottom 25% of distances
            max_matches=self.config.mp_max_neighbors
        )
        
        for i, (distances, indices) in enumerate(zip(motif_distances, motif_indices)):
            if i >= top_k:
                break
            
            # Extract subsequences for this motif
            subsequences = []
            for idx in indices:
                if idx + window_size <= len(series_clean):
                    subseq = series_clean.iloc[idx:idx + window_size].values
                    subsequences.append(subseq)
            
            if len(subsequences) > 0:
                # Calculate prototype (median)
                prototype = np.median(subsequences, axis=0)
                
                motif_info = {
                    'motif_id': i,
                    'indices': indices,
                    'distances': distances,
                    'prototype': prototype,
                    'occurrences': len(indices),
                    'window_size': window_size,
                    'mean_distance': np.mean(distances)
                }
                
                motifs_discovered.append(motif_info)
        
        self.motifs = motifs_discovered
        return motifs_discovered
    
    def cluster_subsequences(self, series: pd.Series, window_size: int,
                            method: str = 'hdbscan') -> Dict:
        """
        Cluster normalized subsequences to find pattern groups.
        
        Args:
            series: Time series to analyze
            window_size: Size of subsequences
            method: Clustering method ('hdbscan' or 'kmeans')
            
        Returns:
            Dictionary with cluster assignments and centroids
        """
        if not CLUSTERING_AVAILABLE:
            warnings.warn("Clustering libraries not available")
            return {}
        
        # Create sliding windows
        subsequences = []
        indices = []
        
        series_clean = series.dropna()
        
        for i in range(len(series_clean) - window_size + 1):
            subseq = series_clean.iloc[i:i + window_size].values
            
            # Normalize subsequence (z-score)
            subseq_normalized = (subseq - subseq.mean()) / (subseq.std() + 1e-8)
            
            subsequences.append(subseq_normalized)
            indices.append(series_clean.index[i])
        
        if len(subsequences) == 0:
            return {}
        
        X = np.array(subsequences)
        
        # Apply clustering
        if method == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_samples
            )
            labels = clusterer.fit_predict(X)
        elif method == 'kmeans':
            kmeans = KMeans(
                n_clusters=self.config.kmeans_n_clusters,
                random_state=42
            )
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate centroids for each cluster
        unique_labels = set(labels)
        if -1 in unique_labels:  # Remove noise label for HDBSCAN
            unique_labels.remove(-1)
        
        cluster_info = {}
        for label in unique_labels:
            mask = labels == label
            cluster_subsequences = X[mask]
            
            cluster_info[int(label)] = {
                'centroid': cluster_subsequences.mean(axis=0),
                'size': mask.sum(),
                'indices': [indices[i] for i, m in enumerate(mask) if m],
                'std': cluster_subsequences.std(axis=0).mean()
            }
        
        self.clusters = {
            'labels': labels,
            'cluster_info': cluster_info,
            'indices': indices,
            'window_size': window_size
        }
        
        return self.clusters
    
    def find_pattern_occurrences(self, series: pd.Series, pattern: np.ndarray,
                                threshold: float = 0.1) -> List[int]:
        """
        Find occurrences of a specific pattern in the time series.
        
        Args:
            series: Time series to search
            pattern: Pattern to find (normalized)
            threshold: Distance threshold for matching
            
        Returns:
            List of indices where pattern occurs
        """
        window_size = len(pattern)
        series_clean = series.dropna()
        
        occurrences = []
        
        for i in range(len(series_clean) - window_size + 1):
            subseq = series_clean.iloc[i:i + window_size].values
            
            # Normalize subsequence
            subseq_normalized = (subseq - subseq.mean()) / (subseq.std() + 1e-8)
            
            # Calculate distance (Euclidean)
            distance = np.sqrt(np.sum((subseq_normalized - pattern) ** 2))
            
            if distance <= threshold:
                occurrences.append(i)
        
        return occurrences
    
    def analyze_pattern_predictive_power(self, series: pd.Series, 
                                        pattern_indices: List[int],
                                        future_horizon: int = 5) -> Dict:
        """
        Analyze if a pattern predicts future moves.
        
        Args:
            series: Price series
            pattern_indices: Indices where pattern occurs
            future_horizon: Days to look ahead
            
        Returns:
            Dictionary with statistical analysis
        """
        future_returns = []
        max_future_returns = []
        min_future_returns = []
        
        for idx in pattern_indices:
            if idx + future_horizon < len(series):
                current_price = series.iloc[idx]
                future_prices = series.iloc[idx+1:idx+future_horizon+1]
                
                if len(future_prices) > 0 and current_price > 0:
                    future_ret = (future_prices.iloc[-1] - current_price) / current_price
                    max_ret = (future_prices.max() - current_price) / current_price
                    min_ret = (future_prices.min() - current_price) / current_price
                    
                    future_returns.append(future_ret)
                    max_future_returns.append(max_ret)
                    min_future_returns.append(min_ret)
        
        if len(future_returns) == 0:
            return {}
        
        analysis = {
            'n_occurrences': len(pattern_indices),
            'n_valid_predictions': len(future_returns),
            'mean_return': np.mean(future_returns),
            'median_return': np.median(future_returns),
            'std_return': np.std(future_returns),
            'mean_max_return': np.mean(max_future_returns),
            'mean_min_return': np.mean(min_future_returns),
            'win_rate': np.mean([r > 0 for r in future_returns]),
            'sharpe_ratio': np.mean(future_returns) / (np.std(future_returns) + 1e-8) * np.sqrt(252 / future_horizon),
        }
        
        return analysis
    
    def build_pattern_library(self, df: pd.DataFrame,
                             price_col: str = 'Close') -> Dict:
        """
        Build a comprehensive pattern library with statistics.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            
        Returns:
            Pattern library dictionary
        """
        series = df[price_col]
        
        # Discover motifs
        print("Discovering motifs with Matrix Profile...")
        motifs = self.discover_motifs_matrix_profile(series)
        
        # Analyze each motif
        for motif in motifs:
            motif_id = motif['motif_id']
            
            # Analyze predictive power at multiple horizons
            for horizon in [3, 5, 10, 20]:
                analysis = self.analyze_pattern_predictive_power(
                    series,
                    motif['indices'],
                    future_horizon=horizon
                )
                
                self.pattern_library[f'motif_{motif_id}'][f'horizon_{horizon}'] = analysis
            
            # Store prototype
            self.pattern_library[f'motif_{motif_id}']['prototype'] = motif['prototype']
            self.pattern_library[f'motif_{motif_id}']['occurrences'] = motif['occurrences']
        
        # Cluster subsequences
        print("Clustering subsequences...")
        clusters = self.cluster_subsequences(
            series,
            window_size=self.config.mp_window_size,
            method=self.config.clustering_method
        )
        
        if clusters:
            for cluster_id, cluster_data in clusters.get('cluster_info', {}).items():
                # Analyze predictive power of each cluster
                for horizon in [3, 5, 10, 20]:
                    # Convert date indices to integer indices
                    date_indices = cluster_data['indices']
                    int_indices = [series.index.get_loc(idx) for idx in date_indices if idx in series.index]
                    
                    analysis = self.analyze_pattern_predictive_power(
                        series,
                        int_indices,
                        future_horizon=horizon
                    )
                    
                    self.pattern_library[f'cluster_{cluster_id}'][f'horizon_{horizon}'] = analysis
                
                # Store centroid
                self.pattern_library[f'cluster_{cluster_id}']['centroid'] = cluster_data['centroid']
                self.pattern_library[f'cluster_{cluster_id}']['size'] = cluster_data['size']
        
        return dict(self.pattern_library)
    
    def filter_significant_patterns(self, min_occurrences: Optional[int] = None,
                                   min_confidence: Optional[float] = None) -> Dict:
        """
        Filter pattern library to only significant patterns.
        
        Args:
            min_occurrences: Minimum number of pattern occurrences
            min_confidence: Minimum win rate
            
        Returns:
            Filtered pattern library
        """
        min_occ = min_occurrences or self.config.min_pattern_occurrences
        min_conf = min_confidence or self.config.pattern_confidence_threshold
        
        filtered = {}
        
        for pattern_name, pattern_data in self.pattern_library.items():
            # Check occurrences
            occurrences = pattern_data.get('occurrences') or pattern_data.get('size', 0)
            
            if occurrences < min_occ:
                continue
            
            # Check confidence (win rate) for at least one horizon
            has_good_confidence = False
            for key, value in pattern_data.items():
                if key.startswith('horizon_') and isinstance(value, dict):
                    win_rate = value.get('win_rate', 0)
                    if win_rate >= min_conf:
                        has_good_confidence = True
                        break
            
            if has_good_confidence:
                filtered[pattern_name] = pattern_data
        
        return filtered
    
    def get_pattern_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of all patterns in library.
        
        Returns:
            DataFrame with pattern summary
        """
        summaries = []
        
        for pattern_name, pattern_data in self.pattern_library.items():
            summary = {
                'pattern_name': pattern_name,
                'occurrences': pattern_data.get('occurrences') or pattern_data.get('size', 0),
            }
            
            # Get best horizon statistics
            best_sharpe = -np.inf
            best_horizon = None
            
            for key, value in pattern_data.items():
                if key.startswith('horizon_') and isinstance(value, dict):
                    horizon = key.split('_')[1]
                    sharpe = value.get('sharpe_ratio', -np.inf)
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_horizon = horizon
                        summary.update({
                            'best_horizon': horizon,
                            'mean_return': value.get('mean_return', 0),
                            'win_rate': value.get('win_rate', 0),
                            'sharpe_ratio': sharpe,
                        })
            
            summaries.append(summary)
        
        if len(summaries) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(summaries).sort_values('sharpe_ratio', ascending=False)


class ShapeBasedMatcher:
    """
    Match shapes/patterns using DTW and other distance metrics.
    """
    
    @staticmethod
    def euclidean_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Calculate Euclidean distance between normalized sequences."""
        # Normalize both sequences
        seq1_norm = (seq1 - seq1.mean()) / (seq1.std() + 1e-8)
        seq2_norm = (seq2 - seq2.mean()) / (seq2.std() + 1e-8)
        
        return np.sqrt(np.sum((seq1_norm - seq2_norm) ** 2))
    
    @staticmethod
    def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Calculate Dynamic Time Warping distance.
        Simple implementation without external dependencies.
        """
        n, m = len(seq1), len(seq2)
        
        # Normalize sequences
        seq1_norm = (seq1 - seq1.mean()) / (seq1.std() + 1e-8)
        seq2_norm = (seq2 - seq2.mean()) / (seq2.std() + 1e-8)
        
        # Initialize DTW matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1_norm[i-1] - seq2_norm[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        return dtw_matrix[n, m]
    
    @staticmethod
    def find_similar_shapes(series: pd.Series, template: np.ndarray,
                           top_k: int = 10, method: str = 'euclidean') -> List[Tuple[int, float]]:
        """
        Find subsequences most similar to template.
        
        Args:
            series: Time series to search
            template: Template pattern to match
            top_k: Number of top matches to return
            method: Distance method ('euclidean' or 'dtw')
            
        Returns:
            List of (index, distance) tuples
        """
        window_size = len(template)
        distances = []
        
        series_clean = series.dropna()
        
        for i in range(len(series_clean) - window_size + 1):
            subseq = series_clean.iloc[i:i + window_size].values
            
            if method == 'euclidean':
                dist = ShapeBasedMatcher.euclidean_distance(subseq, template)
            elif method == 'dtw':
                dist = ShapeBasedMatcher.dtw_distance(subseq, template)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            distances.append((i, dist))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:top_k]
