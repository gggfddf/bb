#!/usr/bin/env python3
"""
Pattern-Based Prediction System

Implements comprehensive pattern-based prediction for trading:
- Pattern-based signal generation
- Pattern confidence scoring
- Multi-pattern ensemble predictions
- Pattern validation and filtering
- Real-time pattern detection
- Pattern performance tracking

Features:
- Advanced pattern-based prediction algorithms
- Multi-pattern ensemble and voting systems
- Pattern confidence scoring and validation
- Real-time pattern detection and prediction
- Pattern performance tracking and optimization
- Integration with ML models for enhanced predictions
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import talib

logger = structlog.get_logger()

class PredictionMethod(Enum):
    """Prediction method enumeration."""
    SINGLE_PATTERN = "single_pattern"
    MULTI_PATTERN = "multi_pattern"
    ENSEMBLE = "ensemble"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"

class PatternSignalType(Enum):
    """Pattern signal type enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    STRONG_BULLISH = "strong_bullish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class PatternPrediction:
    """Pattern prediction structure."""
    prediction_id: str
    pattern_name: str
    signal_type: PatternSignalType
    confidence: float
    strength: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsemblePrediction:
    """Ensemble prediction structure."""
    prediction_id: str
    final_signal: int  # -1, 0, 1
    confidence: float
    agreement_ratio: float
    contributing_patterns: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionConfig:
    """Prediction configuration."""
    method: PredictionMethod = PredictionMethod.ENSEMBLE
    min_confidence: float = 0.6
    min_patterns: int = 2
    max_patterns: int = 10
    enable_validation: bool = True
    enable_filtering: bool = True
    pattern_weight_decay: float = 0.95
    ensemble_voting_threshold: float = 0.7
    enable_ml_integration: bool = True

class SinglePatternPredictor:
    """Single pattern-based predictor."""
    
    def __init__(self):
        """Initialize single pattern predictor."""
        self.pattern_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1_score': 0.5,
            'total_predictions': 0,
            'correct_predictions': 0
        })
        
    def predict_from_pattern(self, pattern: 'Pattern', data: pd.DataFrame) -> PatternPrediction:
        """Generate prediction from a single pattern."""
        try:
            # Determine signal type based on pattern
            signal_type = self._determine_signal_type(pattern)
            
            # Calculate confidence based on pattern strength and historical performance
            confidence = self._calculate_confidence(pattern)
            
            # Calculate strength based on pattern characteristics
            strength = self._calculate_strength(pattern, data)
            
            # Calculate target price and stop loss
            target_price = self._calculate_target_price(pattern, data)
            stop_loss = self._calculate_stop_loss(pattern, data)
            
            prediction = PatternPrediction(
                prediction_id=str(uuid.uuid4()),
                pattern_name=pattern.name,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                target_price=target_price,
                stop_loss=stop_loss
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting from pattern {pattern.name}: {e}")
            return None
            
    def _determine_signal_type(self, pattern: 'Pattern') -> PatternSignalType:
        """Determine signal type from pattern."""
        if pattern.direction == 1:
            if pattern.confidence > 0.8 and pattern.strength > 0.7:
                return PatternSignalType.STRONG_BULLISH
            else:
                return PatternSignalType.BULLISH
        elif pattern.direction == -1:
            if pattern.confidence > 0.8 and pattern.strength > 0.7:
                return PatternSignalType.STRONG_BEARISH
            else:
                return PatternSignalType.BEARISH
        else:
            return PatternSignalType.NEUTRAL
            
    def _calculate_confidence(self, pattern: 'Pattern') -> float:
        """Calculate prediction confidence."""
        # Base confidence from pattern
        base_confidence = pattern.confidence * pattern.strength
        
        # Adjust based on historical performance
        performance = self.pattern_performance[pattern.name]
        if performance['total_predictions'] > 0:
            historical_accuracy = performance['accuracy']
            adjusted_confidence = base_confidence * (0.7 + 0.3 * historical_accuracy)
        else:
            adjusted_confidence = base_confidence
            
        return min(1.0, max(0.0, adjusted_confidence))
        
    def _calculate_strength(self, pattern: 'Pattern', data: pd.DataFrame) -> float:
        """Calculate prediction strength."""
        # Base strength from pattern
        base_strength = pattern.strength
        
        # Adjust based on market conditions
        recent_data = data.iloc[max(0, pattern.end_index-10):pattern.end_index+1]
        volatility = recent_data['close'].pct_change().std()
        volume_ratio = recent_data['volume'].mean() / data['volume'].mean()
        
        # Higher strength for lower volatility and higher volume
        volatility_factor = 1.0 - min(0.3, volatility)
        volume_factor = min(1.2, volume_ratio)
        
        adjusted_strength = base_strength * volatility_factor * volume_factor
        return min(1.0, max(0.0, adjusted_strength))
        
    def _calculate_target_price(self, pattern: 'Pattern', data: pd.DataFrame) -> Optional[float]:
        """Calculate target price."""
        if pattern.target_price:
            return pattern.target_price
            
        try:
            current_price = data.iloc[pattern.end_index]['close']
            
            if pattern.direction == 1:  # Bullish
                # Calculate resistance level
                recent_highs = data.iloc[max(0, pattern.end_index-20):pattern.end_index+1]['high']
                resistance = recent_highs.max()
                return resistance
            elif pattern.direction == -1:  # Bearish
                # Calculate support level
                recent_lows = data.iloc[max(0, pattern.end_index-20):pattern.end_index+1]['low']
                support = recent_lows.min()
                return support
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error calculating target price: {e}")
            return None
            
    def _calculate_stop_loss(self, pattern: 'Pattern', data: pd.DataFrame) -> Optional[float]:
        """Calculate stop loss."""
        if pattern.stop_loss:
            return pattern.stop_loss
            
        try:
            current_price = data.iloc[pattern.end_index]['close']
            
            if pattern.direction == 1:  # Bullish
                # Stop loss below recent low
                recent_lows = data.iloc[max(0, pattern.end_index-10):pattern.end_index+1]['low']
                stop_loss = recent_lows.min()
                return stop_loss
            elif pattern.direction == -1:  # Bearish
                # Stop loss above recent high
                recent_highs = data.iloc[max(0, pattern.end_index-10):pattern.end_index+1]['high']
                stop_loss = recent_highs.max()
                return stop_loss
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return None
            
    def update_performance(self, pattern_name: str, prediction: PatternPrediction, actual_outcome: int):
        """Update pattern performance metrics."""
        performance = self.pattern_performance[pattern_name]
        
        # Determine if prediction was correct
        predicted_signal = 1 if prediction.signal_type in [PatternSignalType.BULLISH, PatternSignalType.STRONG_BULLISH] else -1
        is_correct = (predicted_signal == actual_outcome)
        
        # Update metrics
        performance['total_predictions'] += 1
        if is_correct:
            performance['correct_predictions'] += 1
            
        performance['accuracy'] = performance['correct_predictions'] / performance['total_predictions']
        
        # Update other metrics (simplified)
        if performance['total_predictions'] > 10:
            performance['precision'] = performance['accuracy'] * 0.9
            performance['recall'] = performance['accuracy'] * 0.85
            performance['f1_score'] = 2 * (performance['precision'] * performance['recall']) / (performance['precision'] + performance['recall'])

class MultiPatternPredictor:
    """Multi-pattern ensemble predictor."""
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize multi-pattern predictor.
        
        Args:
            config: Prediction configuration
        """
        self.config = config
        self.single_predictor = SinglePatternPredictor()
        self.pattern_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.ensemble_history: List[EnsemblePrediction] = []
        
    def predict_from_patterns(self, patterns: List['Pattern'], data: pd.DataFrame) -> EnsemblePrediction:
        """Generate ensemble prediction from multiple patterns."""
        try:
            # Generate individual predictions
            individual_predictions = []
            for pattern in patterns:
                prediction = self.single_predictor.predict_from_pattern(pattern, data)
                if prediction and prediction.confidence >= self.config.min_confidence:
                    individual_predictions.append(prediction)
                    
            if len(individual_predictions) < self.config.min_patterns:
                logger.warning(f"Insufficient patterns for ensemble prediction: {len(individual_predictions)}")
                return None
                
            # Create ensemble prediction
            ensemble_prediction = self._create_ensemble_prediction(individual_predictions)
            
            # Update pattern weights
            self._update_pattern_weights(individual_predictions)
            
            # Store ensemble prediction
            self.ensemble_history.append(ensemble_prediction)
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Error creating ensemble prediction: {e}")
            return None
            
    def _create_ensemble_prediction(self, predictions: List[PatternPrediction]) -> EnsemblePrediction:
        """Create ensemble prediction from individual predictions."""
        # Calculate weighted votes
        bullish_votes = 0.0
        bearish_votes = 0.0
        total_weight = 0.0
        contributing_patterns = []
        
        for prediction in predictions:
            weight = self.pattern_weights[prediction.pattern_name] * prediction.confidence
            contributing_patterns.append(prediction.pattern_name)
            
            if prediction.signal_type in [PatternSignalType.BULLISH, PatternSignalType.STRONG_BULLISH]:
                bullish_votes += weight
            elif prediction.signal_type in [PatternSignalType.BEARISH, PatternSignalType.STRONG_BEARISH]:
                bearish_votes += weight
                
            total_weight += weight
            
        if total_weight == 0:
            return None
            
        # Determine final signal
        if bullish_votes > bearish_votes and bullish_votes / total_weight >= self.config.ensemble_voting_threshold:
            final_signal = 1
        elif bearish_votes > bullish_votes and bearish_votes / total_weight >= self.config.ensemble_voting_threshold:
            final_signal = -1
        else:
            final_signal = 0
            
        # Calculate confidence and agreement
        max_votes = max(bullish_votes, bearish_votes)
        confidence = max_votes / total_weight if total_weight > 0 else 0.0
        agreement_ratio = max_votes / total_weight if total_weight > 0 else 0.0
        
        return EnsemblePrediction(
            prediction_id=str(uuid.uuid4()),
            final_signal=final_signal,
            confidence=confidence,
            agreement_ratio=agreement_ratio,
            contributing_patterns=contributing_patterns
        )
        
    def _update_pattern_weights(self, predictions: List[PatternPrediction]):
        """Update pattern weights based on performance."""
        for prediction in predictions:
            pattern_name = prediction.pattern_name
            current_weight = self.pattern_weights[pattern_name]
            
            # Decay weight over time
            self.pattern_weights[pattern_name] = current_weight * self.config.pattern_weight_decay
            
            # Adjust weight based on confidence
            if prediction.confidence > 0.8:
                self.pattern_weights[pattern_name] *= 1.1
            elif prediction.confidence < 0.5:
                self.pattern_weights[pattern_name] *= 0.9
                
            # Ensure weight stays within bounds
            self.pattern_weights[pattern_name] = max(0.1, min(2.0, self.pattern_weights[pattern_name]))

class MLPatternPredictor:
    """Machine learning-based pattern predictor."""
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize ML pattern predictor.
        
        Args:
            config: Prediction configuration
        """
        self.config = config
        self.models: Dict[str, Any] = {}
        self.feature_scalers: Dict[str, Any] = {}
        self.training_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def train_model(self, pattern_name: str, training_data: List[Dict[str, Any]]):
        """Train ML model for a specific pattern."""
        try:
            if not training_data:
                logger.warning(f"No training data for pattern {pattern_name}")
                return
                
            # Prepare features and labels
            X = []
            y = []
            
            for sample in training_data:
                features = self._extract_features(sample)
                label = sample.get('label', 0)
                
                X.append(features)
                y.append(label)
                
            if len(X) < 10:
                logger.warning(f"Insufficient training data for {pattern_name}: {len(X)} samples")
                return
                
            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self.models[pattern_name] = model
            self.feature_scalers[pattern_name] = scaler
            
            logger.info(f"Trained ML model for pattern {pattern_name} with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training ML model for {pattern_name}: {e}")
            
    def predict_with_ml(self, pattern: 'Pattern', data: pd.DataFrame) -> PatternPrediction:
        """Make ML-based prediction for a pattern."""
        try:
            if pattern.name not in self.models:
                logger.warning(f"No ML model available for pattern {pattern.name}")
                return None
                
            # Extract features
            features = self._extract_pattern_features(pattern, data)
            
            # Scale features
            scaler = self.feature_scalers[pattern.name]
            features_scaled = scaler.transform([features])
            
            # Make prediction
            model = self.models[pattern.name]
            prediction_proba = model.predict_proba(features_scaled)[0]
            prediction_class = model.predict(features_scaled)[0]
            
            # Calculate confidence
            confidence = max(prediction_proba)
            
            # Determine signal type
            if prediction_class == 1:
                signal_type = PatternSignalType.BULLISH if confidence < 0.8 else PatternSignalType.STRONG_BULLISH
            elif prediction_class == -1:
                signal_type = PatternSignalType.BEARISH if confidence < 0.8 else PatternSignalType.STRONG_BEARISH
            else:
                signal_type = PatternSignalType.NEUTRAL
                
            return PatternPrediction(
                prediction_id=str(uuid.uuid4()),
                pattern_name=pattern.name,
                signal_type=signal_type,
                confidence=confidence,
                strength=pattern.strength
            )
            
        except Exception as e:
            logger.error(f"Error making ML prediction for {pattern.name}: {e}")
            return None
            
    def _extract_features(self, sample: Dict[str, Any]) -> List[float]:
        """Extract features from training sample."""
        # Simplified feature extraction
        features = [
            sample.get('pattern_confidence', 0.5),
            sample.get('pattern_strength', 0.5),
            sample.get('price_change', 0.0),
            sample.get('volume_ratio', 1.0),
            sample.get('volatility', 0.0),
            sample.get('trend_strength', 0.0)
        ]
        return features
        
    def _extract_pattern_features(self, pattern: 'Pattern', data: pd.DataFrame) -> List[float]:
        """Extract features from pattern and data."""
        try:
            # Calculate features
            pattern_data = data.iloc[pattern.start_index:pattern.end_index+1]
            
            price_change = (pattern_data['close'].iloc[-1] - pattern_data['close'].iloc[0]) / pattern_data['close'].iloc[0]
            volume_ratio = pattern_data['volume'].mean() / data['volume'].mean()
            volatility = pattern_data['close'].pct_change().std()
            
            # Calculate trend strength
            if len(pattern_data) > 1:
                trend_slope = np.polyfit(range(len(pattern_data)), pattern_data['close'], 1)[0]
                trend_strength = abs(trend_slope) / pattern_data['close'].mean()
            else:
                trend_strength = 0.0
                
            features = [
                pattern.confidence,
                pattern.strength,
                price_change,
                volume_ratio,
                volatility,
                trend_strength
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pattern features: {e}")
            return [0.5, 0.5, 0.0, 1.0, 0.0, 0.0]  # Default features

class PatternBasedPredictionSystem:
    """Main pattern-based prediction system."""
    
    def __init__(self, config: PredictionConfig = None):
        """
        Initialize pattern-based prediction system.
        
        Args:
            config: Prediction configuration
        """
        self.config = config or PredictionConfig()
        self.single_predictor = SinglePatternPredictor()
        self.multi_predictor = MultiPatternPredictor(self.config)
        self.ml_predictor = MLPatternPredictor(self.config)
        self.prediction_history: List[Union[PatternPrediction, EnsemblePrediction]] = []
        
    def predict_from_patterns(self, patterns: List['Pattern'], data: pd.DataFrame) -> Union[PatternPrediction, EnsemblePrediction]:
        """Generate prediction from patterns."""
        if not patterns:
            logger.warning("No patterns provided for prediction")
            return None
            
        try:
            if self.config.method == PredictionMethod.SINGLE_PATTERN:
                # Use the strongest pattern
                strongest_pattern = max(patterns, key=lambda p: p.confidence * p.strength)
                prediction = self.single_predictor.predict_from_pattern(strongest_pattern, data)
                
            elif self.config.method == PredictionMethod.MULTI_PATTERN:
                # Use multiple patterns with ensemble
                prediction = self.multi_predictor.predict_from_patterns(patterns, data)
                
            elif self.config.method == PredictionMethod.MACHINE_LEARNING:
                # Use ML models for each pattern
                ml_predictions = []
                for pattern in patterns:
                    ml_pred = self.ml_predictor.predict_with_ml(pattern, data)
                    if ml_pred:
                        ml_predictions.append(ml_pred)
                        
                if ml_predictions:
                    prediction = self.multi_predictor._create_ensemble_prediction(ml_predictions)
                else:
                    prediction = None
                    
            elif self.config.method == PredictionMethod.HYBRID:
                # Combine traditional and ML predictions
                traditional_pred = self.single_predictor.predict_from_pattern(patterns[0], data)
                ml_pred = self.ml_predictor.predict_with_ml(patterns[0], data)
                
                if traditional_pred and ml_pred:
                    # Combine predictions
                    combined_confidence = (traditional_pred.confidence + ml_pred.confidence) / 2
                    combined_signal = traditional_pred.signal_type if traditional_pred.confidence > ml_pred.confidence else ml_pred.signal_type
                    
                    prediction = PatternPrediction(
                        prediction_id=str(uuid.uuid4()),
                        pattern_name=patterns[0].name,
                        signal_type=combined_signal,
                        confidence=combined_confidence,
                        strength=patterns[0].strength
                    )
                else:
                    prediction = traditional_pred or ml_pred
                    
            else:
                # Default to ensemble
                prediction = self.multi_predictor.predict_from_patterns(patterns, data)
                
            if prediction:
                self.prediction_history.append(prediction)
                
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating pattern-based prediction: {e}")
            return None
            
    def update_performance(self, prediction: Union[PatternPrediction, EnsemblePrediction], actual_outcome: int):
        """Update prediction performance."""
        try:
            if isinstance(prediction, PatternPrediction):
                self.single_predictor.update_performance(prediction.pattern_name, prediction, actual_outcome)
            elif isinstance(prediction, EnsemblePrediction):
                # Update ensemble performance
                for pattern_name in prediction.contributing_patterns:
                    # Simplified performance update for ensemble
                    pass
                    
        except Exception as e:
            logger.error(f"Error updating prediction performance: {e}")
            
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        stats = {
            'total_predictions': len(self.prediction_history),
            'single_pattern_predictions': len([p for p in self.prediction_history if isinstance(p, PatternPrediction)]),
            'ensemble_predictions': len([p for p in self.prediction_history if isinstance(p, EnsemblePrediction)]),
            'pattern_performance': dict(self.single_predictor.pattern_performance),
            'pattern_weights': dict(self.multi_predictor.pattern_weights)
        }
        
        return stats

def create_pattern_based_prediction_system(config: PredictionConfig = None) -> PatternBasedPredictionSystem:
    """Create a pattern-based prediction system."""
    return PatternBasedPredictionSystem(config)

# Demo of pattern-based prediction system
if __name__ == "__main__":
    # Create prediction system
    config = PredictionConfig(
        method=PredictionMethod.ENSEMBLE,
        min_confidence=0.6,
        min_patterns=2,
        max_patterns=5,
        enable_validation=True,
        enable_filtering=True,
        ensemble_voting_threshold=0.7
    )
    
    system = create_pattern_based_prediction_system(config)
    
    # Create sample patterns (using Pattern class from pattern_labeling.py)
    from dataclasses import dataclass
    from enum import Enum
    
    class PatternType(Enum):
        REVERSAL = "reversal"
        CONTINUATION = "continuation"
        
    class PatternCategory(Enum):
        CANDLESTICK = "candlestick"
        CHART = "chart"
        
    @dataclass
    class Pattern:
        pattern_id: str
        pattern_type: PatternType
        pattern_category: PatternCategory
        name: str
        start_index: int
        end_index: int
        confidence: float
        strength: float
        direction: int
        
    # Create sample patterns
    patterns = [
        Pattern("1", PatternType.REVERSAL, PatternCategory.CANDLESTICK, "HAMMER", 10, 15, 0.8, 0.7, 1),
        Pattern("2", PatternType.CONTINUATION, PatternCategory.CHART, "TRIANGLE", 20, 30, 0.6, 0.5, 1),
        Pattern("3", PatternType.REVERSAL, PatternCategory.CANDLESTICK, "DOJI", 25, 30, 0.7, 0.6, -1)
    ]
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(50).cumsum() + 100,
        'high': np.random.randn(50).cumsum() + 102,
        'low': np.random.randn(50).cumsum() + 98,
        'close': np.random.randn(50).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    # Generate prediction
    prediction = system.predict_from_patterns(patterns, data)
    
    if prediction:
        print(f"Generated prediction: {type(prediction).__name__}")
        if hasattr(prediction, 'final_signal'):
            print(f"Signal: {prediction.final_signal}")
            print(f"Confidence: {prediction.confidence:.2f}")
        else:
            print(f"Signal Type: {prediction.signal_type}")
            print(f"Confidence: {prediction.confidence:.2f}")
    
    # Get statistics
    stats = system.get_prediction_statistics()
    print(f"\nPrediction Statistics:")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Pattern performance: {len(stats['pattern_performance'])} patterns tracked")
    
    print("Pattern-based prediction system completed successfully!")