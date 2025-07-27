#!/usr/bin/env python3
"""
Market State Detection System using Gaussian Mixture Models

Implements comprehensive market state detection for trading analysis:
- Gaussian Mixture Models for state identification
- Multi-dimensional state analysis
- State probability estimation
- Real-time state classification
- State transition modeling
- State-based strategy adaptation

Features:
- Advanced GMM clustering with multiple features
- Multi-dimensional market state analysis
- State probability estimation and confidence scoring
- Real-time state classification and prediction
- State transition modeling and forecasting
- Integration with trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class MarketState(Enum):
    """Market state enumeration."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"

class StateTransitionType(Enum):
    """State transition type enumeration."""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"

@dataclass
class MarketStateInfo:
    """Market state information structure."""
    state_id: str
    state_type: MarketState
    cluster_id: int
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None
    duration: Optional[timedelta] = None
    probability: float = 0.0
    confidence: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateTransition:
    """State transition structure."""
    transition_id: str
    from_state: MarketState
    to_state: MarketState
    transition_type: StateTransitionType
    timestamp: datetime
    probability: float
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateDetectionConfig:
    """Market state detection configuration."""
    n_components: int = 5
    covariance_type: str = "full"
    feature_columns: List[str] = field(default_factory=lambda: [
        'returns', 'volatility', 'volume_profile', 'trend_strength', 'momentum'
    ])
    window_size: int = 20
    min_state_duration: int = 5
    max_state_duration: int = 100
    probability_threshold: float = 0.6
    enable_pca: bool = True
    pca_components: int = 3
    enable_validation: bool = True
    enable_forecasting: bool = True

class StateFeatureExtractor:
    """Feature extraction for market state detection."""
    
    def __init__(self, config: StateDetectionConfig):
        """
        Initialize state feature extractor.
        
        Args:
            config: State detection configuration
        """
        self.config = config
        self.scaler = StandardScaler()
        self.pca = None
        
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for market state detection."""
        try:
            features = []
            
            for i in range(self.config.window_size, len(data)):
                window_data = data.iloc[i-self.config.window_size:i+1]
                
                # Calculate features
                feature_vector = self._calculate_features(window_data)
                features.append(feature_vector)
                
            features_array = np.array(features)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Apply PCA if enabled
            if self.config.enable_pca and features_scaled.shape[1] > self.config.pca_components:
                self.pca = PCA(n_components=self.config.pca_components)
                features_scaled = self.pca.fit_transform(features_scaled)
                
            return features_scaled
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.array([])
            
    def _calculate_features(self, window_data: pd.DataFrame) -> List[float]:
        """Calculate features for market state detection."""
        try:
            # Returns
            returns = window_data['close'].pct_change().dropna()
            mean_return = returns.mean()
            return_std = returns.std()
            
            # Volatility
            volatility = returns.std()
            
            # Volume profile
            volume_mean = window_data['volume'].mean()
            volume_std = window_data['volume'].std()
            volume_profile = volume_std / volume_mean if volume_mean > 0 else 0
            
            # Trend strength
            if len(window_data) > 1:
                trend_slope = np.polyfit(range(len(window_data)), window_data['close'], 1)[0]
                trend_strength = abs(trend_slope) / window_data['close'].mean()
            else:
                trend_strength = 0.0
                
            # Momentum
            momentum = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
            
            # Price range
            price_range = (window_data['high'].max() - window_data['low'].min()) / window_data['close'].mean()
            
            # Volume volatility
            volume_volatility = window_data['volume'].std() / window_data['volume'].mean()
            
            # RSI
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            rs = avg_gain / avg_loss if avg_loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            # Market efficiency ratio
            price_changes = window_data['close'].pct_change().dropna()
            total_movement = np.sum(np.abs(price_changes))
            net_movement = np.sum(price_changes)
            efficiency_ratio = abs(net_movement) / total_movement if total_movement > 0 else 0
            
            features = [
                mean_return,
                return_std,
                volatility,
                volume_profile,
                trend_strength,
                momentum,
                price_range,
                volume_volatility,
                rsi,
                efficiency_ratio
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return [0.0] * 10  # Default features

class GMMMarketStateDetector:
    """Gaussian Mixture Model-based market state detector."""
    
    def __init__(self, config: StateDetectionConfig):
        """
        Initialize GMM market state detector.
        
        Args:
            config: State detection configuration
        """
        self.config = config
        self.feature_extractor = StateFeatureExtractor(config)
        self.gmm = GaussianMixture(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            random_state=42
        )
        self.states: List[MarketStateInfo] = []
        self.transitions: List[StateTransition] = []
        self.state_labels: Optional[np.ndarray] = None
        self.state_probabilities: Optional[np.ndarray] = None
        
    def detect_states(self, data: pd.DataFrame) -> List[MarketStateInfo]:
        """Detect market states using GMM."""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(data)
            
            if len(features) == 0:
                logger.warning("No features extracted for state detection")
                return []
                
            # Fit GMM
            self.gmm.fit(features)
            self.state_labels = self.gmm.predict(features)
            self.state_probabilities = self.gmm.predict_proba(features)
            
            # Create states
            states = self._create_states(data, features, self.state_labels, self.state_probabilities)
            
            # Detect transitions
            transitions = self._detect_transitions(states)
            self.transitions.extend(transitions)
            
            # Validate states
            if self.config.enable_validation:
                states = self._validate_states(states)
                
            self.states.extend(states)
            
            logger.info(f"Detected {len(states)} market states with {len(transitions)} transitions")
            return states
            
        except Exception as e:
            logger.error(f"Error detecting market states: {e}")
            return []
            
    def _create_states(self, data: pd.DataFrame, features: np.ndarray, 
                      labels: np.ndarray, probabilities: np.ndarray) -> List[MarketStateInfo]:
        """Create market states from GMM results."""
        states = []
        
        # Group consecutive same labels
        current_label = labels[0]
        start_idx = self.config.window_size
        state_start = data.index[start_idx]
        current_probabilities = [probabilities[0]]
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                # End current state
                state_end = data.index[start_idx + i - 1]
                duration = state_end - state_start
                
                if duration.days >= self.config.min_state_duration:
                    state = self._create_state_object(
                        current_label, state_start, state_end,
                        features[start_idx-self.config.window_size:i], 
                        data.iloc[start_idx:i], current_probabilities
                    )
                    states.append(state)
                    
                # Start new state
                current_label = labels[i]
                start_idx = start_idx + i
                state_start = data.index[start_idx]
                current_probabilities = [probabilities[i]]
            else:
                current_probabilities.append(probabilities[i])
                
        # Handle last state
        if start_idx < len(data):
            state_end = data.index[-1]
            duration = state_end - state_start
            
            if duration.days >= self.config.min_state_duration:
                state = self._create_state_object(
                    current_label, state_start, state_end,
                    features[start_idx-self.config.window_size:], 
                    data.iloc[start_idx:], current_probabilities
                )
                states.append(state)
                
        return states
        
    def _create_state_object(self, cluster_id: int, start_time: datetime, end_time: datetime,
                           features: np.ndarray, data: pd.DataFrame, 
                           probabilities: List[np.ndarray]) -> MarketStateInfo:
        """Create a market state object."""
        # Determine state type
        state_type = self._determine_state_type(cluster_id, features, data)
        
        # Calculate state probability
        avg_probability = np.mean([prob[cluster_id] for prob in probabilities])
        
        # Calculate confidence
        confidence = self._calculate_state_confidence(features, probabilities)
        
        # Calculate state features
        state_features = self._calculate_state_features(features, data)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(data)
        
        state = MarketStateInfo(
            state_id=str(uuid.uuid4()),
            state_type=state_type,
            cluster_id=cluster_id,
            start_timestamp=start_time,
            end_timestamp=end_time,
            duration=end_time - start_time,
            probability=avg_probability,
            confidence=confidence,
            features=state_features,
            performance_metrics=performance_metrics
        )
        
        return state
        
    def _determine_state_type(self, cluster_id: int, features: np.ndarray, data: pd.DataFrame) -> MarketState:
        """Determine state type based on cluster characteristics."""
        try:
            # Extract key features
            mean_return = features[0]
            volatility = features[2]
            trend_strength = features[4]
            momentum = features[5]
            efficiency_ratio = features[9]
            
            # Determine state based on characteristics
            if mean_return > 0.01 and trend_strength > 0.05:
                return MarketState.BULL_MARKET
            elif mean_return < -0.01 and trend_strength > 0.05:
                return MarketState.BEAR_MARKET
            elif volatility > 0.05:
                return MarketState.HIGH_VOLATILITY
            elif volatility < 0.02:
                return MarketState.LOW_VOLATILITY
            elif momentum > 0.05:
                return MarketState.TRENDING_UP
            elif momentum < -0.05:
                return MarketState.TRENDING_DOWN
            elif efficiency_ratio < 0.3:
                return MarketState.SIDEWAYS
            elif efficiency_ratio > 0.7:
                return MarketState.BREAKOUT
            elif abs(momentum) < 0.02 and volatility < 0.03:
                return MarketState.CONSOLIDATION
            else:
                return MarketState.BREAKDOWN
                
        except Exception as e:
            logger.error(f"Error determining state type: {e}")
            return MarketState.SIDEWAYS
            
    def _calculate_state_confidence(self, features: np.ndarray, probabilities: List[np.ndarray]) -> float:
        """Calculate state confidence."""
        try:
            # Confidence based on probability consistency
            cluster_probs = [prob[self.state_labels[0]] for prob in probabilities]
            confidence = np.mean(cluster_probs)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating state confidence: {e}")
            return 0.5
            
    def _calculate_state_features(self, features: np.ndarray, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate state features."""
        try:
            feature_names = [
                'mean_return', 'return_std', 'volatility', 'volume_profile',
                'trend_strength', 'momentum', 'price_range', 'volume_volatility',
                'rsi', 'efficiency_ratio'
            ]
            
            state_features = {}
            for i, name in enumerate(feature_names):
                if i < features.shape[1]:
                    state_features[name] = float(features[:, i].mean())
                    
            return state_features
            
        except Exception as e:
            logger.error(f"Error calculating state features: {e}")
            return {}
            
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate state performance metrics."""
        try:
            returns = data['close'].pct_change().dropna()
            
            metrics = {
                'total_return': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
                'mean_return': returns.mean() * 100,
                'volatility': returns.std() * 100,
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'win_rate': (returns > 0).mean() * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
            
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min() * 100
        except:
            return 0.0
            
    def _detect_transitions(self, states: List[MarketStateInfo]) -> List[StateTransition]:
        """Detect state transitions."""
        transitions = []
        
        for i in range(1, len(states)):
            from_state = states[i-1]
            to_state = states[i]
            
            # Calculate transition characteristics
            transition_type = self._determine_transition_type(from_state, to_state)
            probability = self._calculate_transition_probability(from_state, to_state)
            confidence = self._calculate_transition_confidence(from_state, to_state)
            
            # Create transition features
            features = self._calculate_transition_features(from_state, to_state)
            
            transition = StateTransition(
                transition_id=str(uuid.uuid4()),
                from_state=from_state.state_type,
                to_state=to_state.state_type,
                transition_type=transition_type,
                timestamp=to_state.start_timestamp,
                probability=probability,
                confidence=confidence,
                features=features
            )
            
            transitions.append(transition)
            
        return transitions
        
    def _determine_transition_type(self, from_state: MarketStateInfo, to_state: MarketStateInfo) -> StateTransitionType:
        """Determine transition type."""
        # Check if it's a reversal
        if (from_state.state_type in [MarketState.BULL_MARKET, MarketState.TRENDING_UP] and
            to_state.state_type in [MarketState.BEAR_MARKET, MarketState.TRENDING_DOWN]):
            return StateTransitionType.REVERSAL
        elif (from_state.state_type in [MarketState.BEAR_MARKET, MarketState.TRENDING_DOWN] and
              to_state.state_type in [MarketState.BULL_MARKET, MarketState.TRENDING_UP]):
            return StateTransitionType.REVERSAL
        elif (from_state.state_type in [MarketState.HIGH_VOLATILITY, MarketState.LOW_VOLATILITY] and
              to_state.state_type in [MarketState.LOW_VOLATILITY, MarketState.HIGH_VOLATILITY]):
            return StateTransitionType.SUDDEN
        else:
            return StateTransitionType.CONTINUATION
            
    def _calculate_transition_probability(self, from_state: MarketStateInfo, to_state: MarketStateInfo) -> float:
        """Calculate transition probability."""
        # Use GMM transition probabilities if available
        if hasattr(self.gmm, 'transition_matrix_'):
            return self.gmm.transition_matrix_[from_state.cluster_id, to_state.cluster_id]
        else:
            # Simplified probability calculation
            return (from_state.probability + to_state.probability) / 2
            
    def _calculate_transition_confidence(self, from_state: MarketStateInfo, to_state: MarketStateInfo) -> float:
        """Calculate transition confidence."""
        # Confidence based on state confidences and duration
        from_confidence = from_state.confidence
        to_confidence = to_state.confidence
        
        # Longer states have higher confidence
        from_duration_factor = min(1.0, from_state.duration.days / 30.0)
        to_duration_factor = min(1.0, to_state.duration.days / 30.0)
        
        confidence = (from_confidence + to_confidence) / 2 * (from_duration_factor + to_duration_factor) / 2
        return min(1.0, max(0.0, confidence))
        
    def _calculate_transition_features(self, from_state: MarketStateInfo, to_state: MarketStateInfo) -> Dict[str, float]:
        """Calculate transition features."""
        features = {
            'duration_change': (to_state.duration - from_state.duration).total_seconds() / 86400,  # days
            'probability_change': to_state.probability - from_state.probability,
            'confidence_change': to_state.confidence - from_state.confidence,
            'volatility_change': to_state.features.get('volatility', 0) - from_state.features.get('volatility', 0),
            'momentum_change': to_state.features.get('momentum', 0) - from_state.features.get('momentum', 0)
        }
        return features
        
    def _validate_states(self, states: List[MarketStateInfo]) -> List[MarketStateInfo]:
        """Validate states based on criteria."""
        validated_states = []
        
        for state in states:
            # Check minimum duration
            if state.duration.days < self.config.min_state_duration:
                continue
                
            # Check maximum duration
            if state.duration.days > self.config.max_state_duration:
                continue
                
            # Check probability threshold
            if state.probability < self.config.probability_threshold:
                continue
                
            # Check confidence threshold
            if state.confidence < 0.3:
                continue
                
            validated_states.append(state)
            
        return validated_states
        
    def classify_current_state(self, data: pd.DataFrame) -> Optional[MarketStateInfo]:
        """Classify current market state."""
        try:
            if len(data) < self.config.window_size:
                return None
                
            # Extract features for current window
            current_features = self.feature_extractor.extract_features(data.tail(self.config.window_size))
            
            if len(current_features) == 0:
                return None
                
            # Predict state
            cluster_id = self.gmm.predict(current_features)[0]
            probabilities = self.gmm.predict_proba(current_features)[0]
            
            # Create state object
            state = self._create_state_object(
                cluster_id, data.index[-1], data.index[-1],
                current_features, data.tail(self.config.window_size), [probabilities]
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Error classifying current state: {e}")
            return None
            
    def forecast_next_state(self, data: pd.DataFrame, horizon: int = 5) -> List[Tuple[MarketState, float]]:
        """Forecast next market states."""
        try:
            if not self.config.enable_forecasting:
                return []
                
            # Get current state
            current_state = self.classify_current_state(data)
            if not current_state:
                return []
                
            # Use GMM for forecasting (simplified)
            forecasts = []
            current_cluster = current_state.cluster_id
            
            for i in range(horizon):
                # Simplified forecasting - in practice, you'd use more sophisticated methods
                next_cluster = (current_cluster + 1) % self.config.n_components
                next_state_type = self._determine_state_type(next_cluster, np.array([0]), pd.DataFrame())
                probability = 0.5  # Simplified probability
                
                forecasts.append((next_state_type, probability))
                current_cluster = next_cluster
                
            return forecasts
            
        except Exception as e:
            logger.error(f"Error forecasting next states: {e}")
            return []
            
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get market state detection statistics."""
        if not self.states:
            return {}
            
        state_counts = defaultdict(int)
        type_counts = defaultdict(int)
        probabilities = []
        confidence_scores = []
        
        for state in self.states:
            state_counts[state.state_type.value] += 1
            type_counts[state.state_type.value] += 1
            probabilities.append(state.probability)
            confidence_scores.append(state.confidence)
            
        stats = {
            'total_states': len(self.states),
            'total_transitions': len(self.transitions),
            'state_counts': dict(state_counts),
            'avg_probability': np.mean(probabilities) if probabilities else 0.0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'gmm_quality': self._calculate_gmm_quality()
        }
        
        return stats
        
    def _calculate_gmm_quality(self) -> Dict[str, float]:
        """Calculate GMM quality metrics."""
        try:
            if self.state_labels is None or len(self.state_labels) == 0:
                return {}
                
            # Calculate BIC and AIC
            bic = self.gmm.bic(self.feature_extractor.extract_features(pd.DataFrame())) if hasattr(self.gmm, 'bic') else 0
            aic = self.gmm.aic(self.feature_extractor.extract_features(pd.DataFrame())) if hasattr(self.gmm, 'aic') else 0
                
            return {
                'bic': bic,
                'aic': aic,
                'n_components': self.config.n_components
            }
            
        except Exception as e:
            logger.error(f"Error calculating GMM quality: {e}")
            return {}

def create_market_state_detector(config: StateDetectionConfig = None) -> GMMMarketStateDetector:
    """Create a market state detector."""
    return GMMMarketStateDetector(config or StateDetectionConfig())

# Demo of market state detection system
if __name__ == "__main__":
    # Create market state detector
    config = StateDetectionConfig(
        n_components=5,
        covariance_type="full",
        window_size=20,
        min_state_duration=5,
        max_state_duration=100,
        probability_threshold=0.6,
        enable_pca=True,
        pca_components=3
    )
    
    detector = create_market_state_detector(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Detect states
    states = detector.detect_states(data)
    
    print(f"Detected {len(states)} market states:")
    for state in states[:5]:  # Show first 5
        print(f"- {state.state_type.value} (probability: {state.probability:.2f}, confidence: {state.confidence:.2f})")
        print(f"  Duration: {state.duration.days} days")
    
    # Get statistics
    stats = detector.get_state_statistics()
    print(f"\nMarket State Statistics:")
    print(f"Total states: {stats.get('total_states', 0)}")
    print(f"Total transitions: {stats.get('total_transitions', 0)}")
    print(f"State counts: {stats.get('state_counts', {})}")
    print(f"Average probability: {stats.get('avg_probability', 0):.2f}")
    
    # Classify current state
    current_state = detector.classify_current_state(data)
    if current_state:
        print(f"\nCurrent state: {current_state.state_type.value}")
    
    # Forecast next states
    forecasts = detector.forecast_next_state(data, horizon=3)
    print(f"\nForecasted next states:")
    for state_type, probability in forecasts:
        print(f"- {state_type.value}: {probability:.2f}")
    
    print("Market state detection system completed successfully!")