"""
Configuration file for stock pattern detection and market cycle analysis system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Multi-horizon lookback windows (in days)
    lookback_windows: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 40, 80])
    
    # Volume features enabled
    use_volume_features: bool = True
    
    # Pattern shape embedding configuration
    use_shape_embeddings: bool = True
    shape_window_size: int = 20
    
    # SAX (Symbolic Aggregate Approximation) parameters
    sax_alphabet_size: int = 10
    sax_word_length: int = 10


@dataclass
class LabelConfig:
    """Configuration for label generation."""
    # Horizons to predict (in days)
    prediction_horizons: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    
    # ATR multiplier for move detection
    atr_multiplier_up: float = 1.5
    atr_multiplier_down: float = 1.5
    
    # Consolidation parameters
    consolidation_min_days: int = 10
    consolidation_max_days: int = 30
    consolidation_width_multiplier: float = 1.2  # * ATR
    
    # Breakout parameters
    breakout_confirmation_days: int = 3
    breakout_price_threshold: float = 0.5  # * ATR above consolidation high
    breakout_volume_multiplier: float = 1.5  # * median volume
    
    # False breakout parameters
    false_breakout_window: int = 10
    false_breakout_threshold: float = 0.5  # * ATR below breakout level
    
    # Reversal parameters
    reversal_prior_move_multiplier: float = 2.0  # * ATR
    reversal_change_multiplier: float = 2.0  # * ATR
    reversal_window: int = 10


@dataclass
class ModelConfig:
    """Configuration for modeling."""
    # Baseline tree model
    tree_model_type: str = 'lightgbm'  # 'xgboost', 'lightgbm', 'catboost'
    tree_n_estimators: int = 500
    tree_max_depth: int = 7
    tree_learning_rate: float = 0.05
    
    # Sequence model parameters
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 15
    learning_rate: float = 0.001
    
    # Class imbalance handling
    use_class_weights: bool = True
    use_focal_loss: bool = True
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0


@dataclass
class CycleConfig:
    """Configuration for cycle detection."""
    # Autocorrelation parameters
    max_lag: int = 100
    significant_threshold: float = 0.3
    
    # Spectral analysis
    fft_min_period: int = 5
    fft_max_period: int = 252  # trading days in a year
    
    # Wavelet parameters
    wavelet_type: str = 'morl'  # Morlet wavelet
    wavelet_scales: np.ndarray = field(default_factory=lambda: np.arange(1, 128))
    
    # HMM parameters
    hmm_n_states: int = 3  # bull, bear, sideways
    hmm_n_iter: int = 100


@dataclass
class PatternConfig:
    """Configuration for pattern discovery."""
    # Matrix Profile parameters
    mp_window_size: int = 20
    mp_top_k_motifs: int = 10
    mp_max_neighbors: int = 5
    
    # Clustering parameters
    clustering_method: str = 'hdbscan'  # 'hdbscan', 'kmeans'
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5
    kmeans_n_clusters: int = 15
    
    # Pattern library
    min_pattern_occurrences: int = 5
    pattern_confidence_threshold: float = 0.6


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and backtesting."""
    # Cross-validation
    cv_method: str = 'walk_forward'  # 'walk_forward', 'rolling'
    train_size_years: int = 5
    validation_size_months: int = 6
    test_size_months: int = 6
    step_size_months: int = 3
    
    # Backtesting
    initial_capital: float = 100000.0
    position_size_pct: float = 0.05  # 5% per position
    max_positions: int = 10
    
    # Transaction costs
    bid_ask_spread_bps: float = 5.0  # 5 basis points
    slippage_bps: float = 2.0  # 2 basis points
    commission_pct: float = 0.001  # 0.1%
    
    # Execution
    execution_delay_days: int = 1  # trade at next day open
    
    # Metrics
    min_sharpe_threshold: float = 1.0
    max_drawdown_threshold: float = 0.20  # 20%


@dataclass
class DataConfig:
    """Configuration for data handling."""
    # Date range (None means use all available)
    start_date: str = None
    end_date: str = None
    
    # Data quality
    min_trading_days: int = 252  # minimum 1 year of data
    max_missing_pct: float = 0.05  # max 5% missing data
    
    # Outlier detection
    outlier_std_threshold: float = 10.0  # z-score threshold
    outlier_return_threshold: float = 0.5  # 50% single-day return
    
    # Normalization
    normalization_method: str = 'zscore'  # 'zscore', 'minmax', 'robust'
    rolling_norm_window: int = 252  # 1 year


@dataclass
class SystemConfig:
    """Main system configuration."""
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Logging and monitoring
    log_level: str = 'INFO'
    save_dir: str = './outputs'
    experiment_name: str = 'stock_pattern_detection'
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Parallel processing
    n_jobs: int = -1  # use all cores


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()
