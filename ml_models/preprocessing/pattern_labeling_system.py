"""
Pattern Labeling System for Financial Time Series

A comprehensive system for labeling patterns in financial time series data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()

class PatternType(Enum):
    """Supported pattern types."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

@dataclass
class PatternLabel:
    """Pattern label with metadata."""
    pattern_type: PatternType
    confidence: float
    start_index: int
    end_index: int
    metadata: Dict[str, Any]

class PatternLabelingSystem:
    """System for labeling patterns in time series data."""
    
    def __init__(self):
        self.patterns = []
    
    def label_trend_patterns(self, prices: np.ndarray, window_size: int = 20) -> List[PatternLabel]:
        """Label trend patterns in price data."""
        labels = []
        
        for i in range(len(prices) - window_size + 1):
            window = prices[i:i + window_size]
            
            # Calculate trend
            slope = np.polyfit(range(len(window)), window, 1)[0]
            
            if slope > 0.01:
                pattern_type = PatternType.TREND_UP
                confidence = min(abs(slope) * 100, 1.0)
            elif slope < -0.01:
                pattern_type = PatternType.TREND_DOWN
                confidence = min(abs(slope) * 100, 1.0)
            else:
                pattern_type = PatternType.SIDEWAYS
                confidence = 0.5
            
            label = PatternLabel(
                pattern_type=pattern_type,
                confidence=confidence,
                start_index=i,
                end_index=i + window_size - 1,
                metadata={'slope': slope}
            )
            labels.append(label)
        
        return labels
    
    def label_breakout_patterns(self, prices: np.ndarray, window_size: int = 20) -> List[PatternLabel]:
        """Label breakout patterns."""
        labels = []
        
        for i in range(len(prices) - window_size + 1):
            window = prices[i:i + window_size]
            
            # Calculate support and resistance
            support = np.min(window[:-1])
            resistance = np.max(window[:-1])
            
            current_price = window[-1]
            
            if current_price > resistance:
                pattern_type = PatternType.BREAKOUT
                confidence = 0.8
            elif current_price < support:
                pattern_type = PatternType.BREAKOUT
                confidence = 0.8
            else:
                continue
            
            label = PatternLabel(
                pattern_type=pattern_type,
                confidence=confidence,
                start_index=i,
                end_index=i + window_size - 1,
                metadata={'support': support, 'resistance': resistance}
            )
            labels.append(label)
        
        return labels

# Convenience function
def create_pattern_labeling_system() -> PatternLabelingSystem:
    """Create a pattern labeling system."""
    return PatternLabelingSystem()