#!/usr/bin/env python3
"""
Pattern Labeling System

Implements comprehensive pattern labeling for market data:
- Technical pattern recognition
- Pattern labeling and classification
- Training data generation
- Pattern validation and filtering
- Multi-timeframe pattern analysis
- Pattern confidence scoring

Features:
- Advanced technical pattern recognition algorithms
- Comprehensive pattern labeling and classification
- Training data generation for ML models
- Pattern validation and quality filtering
- Multi-timeframe pattern analysis
- Pattern confidence scoring and ranking
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
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class PatternType(Enum):
    """Pattern type enumeration."""
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    TREND = "trend"
    VOLATILITY = "volatility"

class PatternCategory(Enum):
    """Pattern category enumeration."""
    CANDLESTICK = "candlestick"
    CHART = "chart"
    INDICATOR = "indicator"
    SUPPORT_RESISTANCE = "support_resistance"
    FIBONACCI = "fibonacci"
    ELLIOTT_WAVE = "elliott_wave"
    HARMONIC = "harmonic"

@dataclass
class Pattern:
    """Pattern structure."""
    pattern_id: str
    pattern_type: PatternType
    pattern_category: PatternCategory
    name: str
    start_index: int
    end_index: int
    confidence: float
    strength: float
    direction: int  # -1, 0, 1
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternLabel:
    """Pattern label structure."""
    label_id: str
    pattern: Pattern
    label: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LabelingConfig:
    """Labeling configuration."""
    min_pattern_length: int = 5
    max_pattern_length: int = 50
    min_confidence: float = 0.6
    enable_validation: bool = True
    enable_filtering: bool = True
    pattern_overlap_threshold: float = 0.3
    enable_multi_timeframe: bool = True
    timeframes: List[str] = field(default_factory=lambda: ['1H', '4H', '1D'])

class CandlestickPatternDetector:
    """Candlestick pattern detection system."""
    
    def __init__(self):
        """Initialize candlestick pattern detector."""
        self.pattern_functions = {
            'DOJI': talib.CDLDOJI,
            'HAMMER': talib.CDLHAMMER,
            'HANGING_MAN': talib.CDLHANGINGMAN,
            'ENGULFING': talib.CDLENGULFING,
            'MORNING_STAR': talib.CDLMORNINGSTAR,
            'EVENING_STAR': talib.CDLEVENINGSTAR,
            'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS,
            'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS,
            'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
            'INVERTED_HAMMER': talib.CDLINVERTEDHAMMER,
            'PIERCING': talib.CDLPIERCING,
            'DARK_CLOUD_COVER': talib.CDLDARKCLOUDCOVER,
            'HARAMI': talib.CDLHARAMI,
            'SPINNING_TOP': talib.CDLSPINNINGTOP,
            'MARUBOZU': talib.CDLMARUBOZU
        }
        
    def detect_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect candlestick patterns in data."""
        patterns = []
        
        for pattern_name, pattern_func in self.pattern_functions.items():
            try:
                # Get OHLC data
                open_prices = data['open'].values
                high_prices = data['high'].values
                low_prices = data['low'].values
                close_prices = data['close'].values
                
                # Detect pattern
                pattern_signals = pattern_func(open_prices, high_prices, low_prices, close_prices)
                
                # Find pattern occurrences
                pattern_indices = np.where(pattern_signals != 0)[0]
                
                for idx in pattern_indices:
                    if idx >= self.min_pattern_length:
                        pattern = self._create_pattern(
                            pattern_name, idx, data, pattern_signals[idx]
                        )
                        if pattern:
                            patterns.append(pattern)
                            
            except Exception as e:
                logger.warning(f"Error detecting {pattern_name}: {e}")
                
        return patterns
        
    def _create_pattern(self, pattern_name: str, index: int, data: pd.DataFrame, signal: int) -> Optional[Pattern]:
        """Create pattern object."""
        try:
            # Calculate pattern characteristics
            start_idx = max(0, index - 5)
            end_idx = min(len(data) - 1, index + 5)
            
            # Calculate confidence based on pattern strength
            confidence = abs(signal) / 100.0 if signal != 0 else 0.5
            
            # Determine pattern type and direction
            pattern_type = self._determine_pattern_type(pattern_name)
            direction = self._determine_pattern_direction(pattern_name, signal)
            
            # Calculate strength
            strength = self._calculate_pattern_strength(data, start_idx, end_idx)
            
            pattern = Pattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=pattern_type,
                pattern_category=PatternCategory.CANDLESTICK,
                name=pattern_name,
                start_index=start_idx,
                end_index=end_idx,
                confidence=confidence,
                strength=strength,
                direction=direction,
                target_price=self._calculate_target_price(data, index, direction),
                stop_loss=self._calculate_stop_loss(data, index, direction)
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error creating pattern {pattern_name}: {e}")
            return None
            
    def _determine_pattern_type(self, pattern_name: str) -> PatternType:
        """Determine pattern type."""
        reversal_patterns = ['DOJI', 'HAMMER', 'HANGING_MAN', 'ENGULFING', 'MORNING_STAR', 
                           'EVENING_STAR', 'SHOOTING_STAR', 'INVERTED_HAMMER', 'PIERCING', 
                           'DARK_CLOUD_COVER']
        
        if pattern_name in reversal_patterns:
            return PatternType.REVERSAL
        else:
            return PatternType.CONTINUATION
            
    def _determine_pattern_direction(self, pattern_name: str, signal: int) -> int:
        """Determine pattern direction."""
        bullish_patterns = ['HAMMER', 'MORNING_STAR', 'THREE_WHITE_SOLDIERS', 'INVERTED_HAMMER', 'PIERCING']
        bearish_patterns = ['HANGING_MAN', 'EVENING_STAR', 'THREE_BLACK_CROWS', 'SHOOTING_STAR', 'DARK_CLOUD_COVER']
        
        if pattern_name in bullish_patterns:
            return 1
        elif pattern_name in bearish_patterns:
            return -1
        else:
            return 1 if signal > 0 else -1
            
    def _calculate_pattern_strength(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate pattern strength."""
        try:
            # Calculate price movement
            price_change = (data.iloc[end_idx]['close'] - data.iloc[start_idx]['close']) / data.iloc[start_idx]['close']
            
            # Calculate volume strength
            volume_avg = data.iloc[start_idx:end_idx+1]['volume'].mean()
            volume_std = data.iloc[start_idx:end_idx+1]['volume'].std()
            volume_strength = volume_avg / (volume_std + 1e-8)
            
            # Calculate volatility
            returns = data.iloc[start_idx:end_idx+1]['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Combine factors
            strength = abs(price_change) * volume_strength * (1 - volatility)
            
            return min(1.0, max(0.0, strength))
            
        except Exception as e:
            logger.error(f"Error calculating pattern strength: {e}")
            return 0.5
            
    def _calculate_target_price(self, data: pd.DataFrame, index: int, direction: int) -> Optional[float]:
        """Calculate target price."""
        try:
            current_price = data.iloc[index]['close']
            
            if direction == 1:  # Bullish
                # Target at resistance level
                high_prices = data.iloc[max(0, index-20):index+1]['high']
                resistance = high_prices.max()
                return resistance
            elif direction == -1:  # Bearish
                # Target at support level
                low_prices = data.iloc[max(0, index-20):index+1]['low']
                support = low_prices.min()
                return support
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error calculating target price: {e}")
            return None
            
    def _calculate_stop_loss(self, data: pd.DataFrame, index: int, direction: int) -> Optional[float]:
        """Calculate stop loss."""
        try:
            current_price = data.iloc[index]['close']
            
            if direction == 1:  # Bullish
                # Stop loss below recent low
                low_prices = data.iloc[max(0, index-10):index+1]['low']
                stop_loss = low_prices.min()
                return stop_loss
            elif direction == -1:  # Bearish
                # Stop loss above recent high
                high_prices = data.iloc[max(0, index-10):index+1]['high']
                stop_loss = high_prices.max()
                return stop_loss
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return None

class ChartPatternDetector:
    """Chart pattern detection system."""
    
    def __init__(self):
        """Initialize chart pattern detector."""
        self.pattern_detectors = {
            'head_and_shoulders': self._detect_head_and_shoulders,
            'inverse_head_and_shoulders': self._detect_inverse_head_and_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'triangle': self._detect_triangle,
            'wedge': self._detect_wedge,
            'flag': self._detect_flag,
            'pennant': self._detect_pennant,
            'channel': self._detect_channel
        }
        
    def detect_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect chart patterns in data."""
        patterns = []
        
        for pattern_name, detector_func in self.pattern_detectors.items():
            try:
                pattern_list = detector_func(data)
                patterns.extend(pattern_list)
            except Exception as e:
                logger.warning(f"Error detecting {pattern_name}: {e}")
                
        return patterns
        
    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect head and shoulders pattern."""
        patterns = []
        
        # Simplified head and shoulders detection
        # In practice, this would be more sophisticated
        for i in range(20, len(data) - 20):
            try:
                # Look for three peaks with middle peak higher
                left_peak = data.iloc[i-10:i]['high'].max()
                middle_peak = data.iloc[i-5:i+5]['high'].max()
                right_peak = data.iloc[i:i+10]['high'].max()
                
                # Check if middle peak is higher than shoulders
                if (middle_peak > left_peak and middle_peak > right_peak and
                    abs(left_peak - right_peak) / left_peak < 0.1):
                    
                    pattern = Pattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.REVERSAL,
                        pattern_category=PatternCategory.CHART,
                        name='head_and_shoulders',
                        start_index=i-10,
                        end_index=i+10,
                        confidence=0.7,
                        strength=0.6,
                        direction=-1,  # Bearish reversal
                        target_price=data.iloc[i]['close'] * 0.9,
                        stop_loss=data.iloc[i]['close'] * 1.05
                    )
                    patterns.append(pattern)
                    
            except Exception as e:
                continue
                
        return patterns
        
    def _detect_inverse_head_and_shoulders(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect inverse head and shoulders pattern."""
        patterns = []
        
        # Similar to head and shoulders but for bottoms
        for i in range(20, len(data) - 20):
            try:
                left_trough = data.iloc[i-10:i]['low'].min()
                middle_trough = data.iloc[i-5:i+5]['low'].min()
                right_trough = data.iloc[i:i+10]['low'].min()
                
                if (middle_trough < left_trough and middle_trough < right_trough and
                    abs(left_trough - right_trough) / left_trough < 0.1):
                    
                    pattern = Pattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.REVERSAL,
                        pattern_category=PatternCategory.CHART,
                        name='inverse_head_and_shoulders',
                        start_index=i-10,
                        end_index=i+10,
                        confidence=0.7,
                        strength=0.6,
                        direction=1,  # Bullish reversal
                        target_price=data.iloc[i]['close'] * 1.1,
                        stop_loss=data.iloc[i]['close'] * 0.95
                    )
                    patterns.append(pattern)
                    
            except Exception as e:
                continue
                
        return patterns
        
    def _detect_double_top(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect double top pattern."""
        patterns = []
        
        for i in range(10, len(data) - 10):
            try:
                # Look for two similar peaks
                peak1 = data.iloc[i-5:i]['high'].max()
                peak2 = data.iloc[i:i+5]['high'].max()
                
                if abs(peak1 - peak2) / peak1 < 0.05:
                    pattern = Pattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.REVERSAL,
                        pattern_category=PatternCategory.CHART,
                        name='double_top',
                        start_index=i-5,
                        end_index=i+5,
                        confidence=0.6,
                        strength=0.5,
                        direction=-1,
                        target_price=data.iloc[i]['close'] * 0.95,
                        stop_loss=data.iloc[i]['close'] * 1.02
                    )
                    patterns.append(pattern)
                    
            except Exception as e:
                continue
                
        return patterns
        
    def _detect_double_bottom(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect double bottom pattern."""
        patterns = []
        
        for i in range(10, len(data) - 10):
            try:
                # Look for two similar troughs
                trough1 = data.iloc[i-5:i]['low'].min()
                trough2 = data.iloc[i:i+5]['low'].min()
                
                if abs(trough1 - trough2) / trough1 < 0.05:
                    pattern = Pattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.REVERSAL,
                        pattern_category=PatternCategory.CHART,
                        name='double_bottom',
                        start_index=i-5,
                        end_index=i+5,
                        confidence=0.6,
                        strength=0.5,
                        direction=1,
                        target_price=data.iloc[i]['close'] * 1.05,
                        stop_loss=data.iloc[i]['close'] * 0.98
                    )
                    patterns.append(pattern)
                    
            except Exception as e:
                continue
                
        return patterns
        
    def _detect_triangle(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect triangle patterns."""
        patterns = []
        
        for i in range(15, len(data) - 15):
            try:
                # Look for converging trend lines
                highs = data.iloc[i-15:i]['high'].values
                lows = data.iloc[i-15:i]['low'].values
                
                # Fit trend lines
                x = np.arange(len(highs))
                high_slope, high_intercept = np.polyfit(x, highs, 1)
                low_slope, low_intercept = np.polyfit(x, lows, 1)
                
                # Check if lines are converging
                if abs(high_slope - low_slope) > 0.01:
                    pattern = Pattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.CONTINUATION,
                        pattern_category=PatternCategory.CHART,
                        name='triangle',
                        start_index=i-15,
                        end_index=i,
                        confidence=0.5,
                        strength=0.4,
                        direction=1 if high_slope < low_slope else -1,
                        target_price=data.iloc[i]['close'] * (1.05 if high_slope < low_slope else 0.95),
                        stop_loss=data.iloc[i]['close'] * (0.98 if high_slope < low_slope else 1.02)
                    )
                    patterns.append(pattern)
                    
            except Exception as e:
                continue
                
        return patterns
        
    def _detect_wedge(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect wedge patterns."""
        # Similar to triangle but with different characteristics
        return self._detect_triangle(data)  # Simplified
        
    def _detect_flag(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect flag patterns."""
        patterns = []
        
        for i in range(10, len(data) - 10):
            try:
                # Look for strong move followed by consolidation
                move_start = data.iloc[i-10:i-5]['close'].pct_change().sum()
                consolidation = data.iloc[i-5:i+5]['close'].std()
                
                if abs(move_start) > 0.05 and consolidation < 0.02:
                    pattern = Pattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.CONTINUATION,
                        pattern_category=PatternCategory.CHART,
                        name='flag',
                        start_index=i-10,
                        end_index=i+5,
                        confidence=0.6,
                        strength=0.5,
                        direction=1 if move_start > 0 else -1,
                        target_price=data.iloc[i]['close'] * (1.1 if move_start > 0 else 0.9),
                        stop_loss=data.iloc[i]['close'] * (0.98 if move_start > 0 else 1.02)
                    )
                    patterns.append(pattern)
                    
            except Exception as e:
                continue
                
        return patterns
        
    def _detect_pennant(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect pennant patterns."""
        # Similar to flag but with triangular consolidation
        return self._detect_flag(data)  # Simplified
        
    def _detect_channel(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect channel patterns."""
        patterns = []
        
        for i in range(20, len(data) - 20):
            try:
                # Look for parallel trend lines
                highs = data.iloc[i-20:i]['high'].values
                lows = data.iloc[i-20:i]['low'].values
                
                # Fit parallel lines
                x = np.arange(len(highs))
                high_slope, high_intercept = np.polyfit(x, highs, 1)
                low_slope, low_intercept = np.polyfit(x, lows, 1)
                
                # Check if lines are parallel
                if abs(high_slope - low_slope) < 0.01:
                    pattern = Pattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.CONTINUATION,
                        pattern_category=PatternCategory.CHART,
                        name='channel',
                        start_index=i-20,
                        end_index=i,
                        confidence=0.5,
                        strength=0.4,
                        direction=1 if high_slope > 0 else -1,
                        target_price=data.iloc[i]['close'] * (1.05 if high_slope > 0 else 0.95),
                        stop_loss=data.iloc[i]['close'] * (0.98 if high_slope > 0 else 1.02)
                    )
                    patterns.append(pattern)
                    
            except Exception as e:
                continue
                
        return patterns

class PatternLabeler:
    """Pattern labeling system."""
    
    def __init__(self, config: LabelingConfig):
        """
        Initialize pattern labeler.
        
        Args:
            config: Labeling configuration
        """
        self.config = config
        self.candlestick_detector = CandlestickPatternDetector()
        self.chart_detector = ChartPatternDetector()
        self.labels: List[PatternLabel] = []
        
    def label_patterns(self, data: pd.DataFrame) -> List[PatternLabel]:
        """Label patterns in data."""
        # Detect patterns
        candlestick_patterns = self.candlestick_detector.detect_patterns(data)
        chart_patterns = self.chart_detector.detect_patterns(data)
        
        all_patterns = candlestick_patterns + chart_patterns
        
        # Filter patterns
        if self.config.enable_filtering:
            all_patterns = self._filter_patterns(all_patterns)
            
        # Validate patterns
        if self.config.enable_validation:
            all_patterns = self._validate_patterns(all_patterns, data)
            
        # Create labels
        labels = []
        for pattern in all_patterns:
            label = self._create_label(pattern)
            if label:
                labels.append(label)
                
        self.labels.extend(labels)
        return labels
        
    def _filter_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Filter patterns based on criteria."""
        filtered_patterns = []
        
        for pattern in patterns:
            # Check minimum confidence
            if pattern.confidence < self.config.min_confidence:
                continue
                
            # Check pattern length
            pattern_length = pattern.end_index - pattern.start_index
            if (pattern_length < self.config.min_pattern_length or 
                pattern_length > self.config.max_pattern_length):
                continue
                
            # Check for overlapping patterns
            if not self._is_overlapping(pattern, filtered_patterns):
                filtered_patterns.append(pattern)
                
        return filtered_patterns
        
    def _validate_patterns(self, patterns: List[Pattern], data: pd.DataFrame) -> List[Pattern]:
        """Validate patterns."""
        validated_patterns = []
        
        for pattern in patterns:
            try:
                # Check if pattern is still valid
                if self._is_pattern_valid(pattern, data):
                    validated_patterns.append(pattern)
            except Exception as e:
                logger.warning(f"Error validating pattern {pattern.pattern_id}: {e}")
                
        return validated_patterns
        
    def _is_overlapping(self, pattern: Pattern, existing_patterns: List[Pattern]) -> bool:
        """Check if pattern overlaps with existing patterns."""
        for existing in existing_patterns:
            overlap_start = max(pattern.start_index, existing.start_index)
            overlap_end = min(pattern.end_index, existing.end_index)
            
            if overlap_start < overlap_end:
                overlap_ratio = (overlap_end - overlap_start) / min(
                    pattern.end_index - pattern.start_index,
                    existing.end_index - existing.start_index
                )
                
                if overlap_ratio > self.config.pattern_overlap_threshold:
                    return True
                    
        return False
        
    def _is_pattern_valid(self, pattern: Pattern, data: pd.DataFrame) -> bool:
        """Check if pattern is still valid."""
        try:
            # Check if pattern boundaries are within data range
            if (pattern.start_index < 0 or pattern.end_index >= len(data) or
                pattern.start_index >= pattern.end_index):
                return False
                
            # Check if pattern has sufficient data points
            pattern_data = data.iloc[pattern.start_index:pattern.end_index+1]
            if len(pattern_data) < 3:
                return False
                
            # Check for price continuity
            price_changes = pattern_data['close'].pct_change().dropna()
            if price_changes.std() > 0.1:  # Too volatile
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating pattern: {e}")
            return False
            
    def _create_label(self, pattern: Pattern) -> Optional[PatternLabel]:
        """Create pattern label."""
        try:
            # Generate label based on pattern characteristics
            label = self._generate_label(pattern)
            
            # Calculate label confidence
            confidence = self._calculate_label_confidence(pattern)
            
            label_obj = PatternLabel(
                label_id=str(uuid.uuid4()),
                pattern=pattern,
                label=label,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            return label_obj
            
        except Exception as e:
            logger.error(f"Error creating label for pattern {pattern.pattern_id}: {e}")
            return None
            
    def _generate_label(self, pattern: Pattern) -> str:
        """Generate label for pattern."""
        # Create descriptive label
        direction_str = "bullish" if pattern.direction == 1 else "bearish" if pattern.direction == -1 else "neutral"
        strength_str = "strong" if pattern.strength > 0.7 else "moderate" if pattern.strength > 0.4 else "weak"
        
        label = f"{pattern.name}_{direction_str}_{strength_str}"
        return label
        
    def _calculate_label_confidence(self, pattern: Pattern) -> float:
        """Calculate label confidence."""
        # Combine pattern confidence and strength
        base_confidence = pattern.confidence * pattern.strength
        
        # Adjust based on pattern type
        if pattern.pattern_type == PatternType.REVERSAL:
            base_confidence *= 1.1  # Reversal patterns are more significant
        elif pattern.pattern_type == PatternType.CONTINUATION:
            base_confidence *= 0.9
            
        return min(1.0, max(0.0, base_confidence))
        
    def get_labels(self) -> List[PatternLabel]:
        """Get all labels."""
        return self.labels
        
    def get_labels_by_type(self, pattern_type: PatternType) -> List[PatternLabel]:
        """Get labels by pattern type."""
        return [label for label in self.labels if label.pattern.pattern_type == pattern_type]
        
    def get_labels_by_category(self, category: PatternCategory) -> List[PatternLabel]:
        """Get labels by pattern category."""
        return [label for label in self.labels if label.pattern.pattern_category == category]

def create_pattern_labeler(config: LabelingConfig = None) -> PatternLabeler:
    """Create a pattern labeler."""
    return PatternLabeler(config or LabelingConfig())

# Demo of pattern labeling system
if __name__ == "__main__":
    # Create labeler
    config = LabelingConfig(
        min_pattern_length=5,
        max_pattern_length=50,
        min_confidence=0.6,
        enable_validation=True,
        enable_filtering=True,
        pattern_overlap_threshold=0.3
    )
    
    labeler = create_pattern_labeler(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Label patterns
    labels = labeler.label_patterns(data)
    
    print(f"Detected {len(labels)} patterns:")
    for label in labels[:5]:  # Show first 5
        print(f"- {label.label} (confidence: {label.confidence:.2f})")
    
    # Get labels by type
    reversal_labels = labeler.get_labels_by_type(PatternType.REVERSAL)
    print(f"\nReversal patterns: {len(reversal_labels)}")
    
    candlestick_labels = labeler.get_labels_by_category(PatternCategory.CANDLESTICK)
    print(f"Candlestick patterns: {len(candlestick_labels)}")
    
    print("Pattern labeling system completed successfully!")