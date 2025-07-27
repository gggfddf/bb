"""
CNN Pattern Recognition for Candlestick Patterns

A comprehensive CNN implementation using TensorFlow/Keras for recognizing
candlestick patterns in financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. CNN functionality will be limited.")

logger = structlog.get_logger()

class PatternType(Enum):
    """Supported candlestick pattern types."""
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING = "engulfing"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    HANGING_MAN = "hanging_man"
    INV_HAMMER = "inverted_hammer"

@dataclass
class CNNConfig:
    """Configuration for CNN pattern recognition model."""
    # Architecture parameters
    input_height: int = 64
    input_width: int = 64
    input_channels: int = 3
    n_classes: int = len(PatternType)
    
    # CNN parameters
    n_conv_layers: int = 4
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    conv_strides: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    dropout_rate: float = 0.3
    
    # Dense layers
    dense_units: List[int] = field(default_factory=lambda: [512, 256, 128])
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    # Early stopping parameters
    patience: int = 10
    min_delta: float = 0.001
    restore_best_weights: bool = True

@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any
    history: Dict[str, List[float]]
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    training_time: float = 0.0
    config: CNNConfig = None

class CandlestickPatternDetector:
    """Detects candlestick patterns in OHLC data."""
    
    @staticmethod
    def detect_doji(open_prices: np.ndarray, high_prices: np.ndarray, 
                   low_prices: np.ndarray, close_prices: np.ndarray, 
                   threshold: float = 0.1) -> np.ndarray:
        """Detect doji patterns."""
        body_size = np.abs(close_prices - open_prices)
        total_range = high_prices - low_prices
        
        # Doji: small body relative to total range
        doji_mask = body_size <= (total_range * threshold)
        return doji_mask
    
    @staticmethod
    def detect_hammer(open_prices: np.ndarray, high_prices: np.ndarray,
                     low_prices: np.ndarray, close_prices: np.ndarray,
                     body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> np.ndarray:
        """Detect hammer patterns."""
        body_size = np.abs(close_prices - open_prices)
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices
        
        # Hammer: small body, long lower shadow, short upper shadow
        hammer_mask = (
            (body_size <= (body_size + upper_shadow + lower_shadow) * body_ratio) &
            (lower_shadow >= body_size * shadow_ratio) &
            (upper_shadow <= body_size * 0.5)
        )
        return hammer_mask
    
    @staticmethod
    def detect_shooting_star(open_prices: np.ndarray, high_prices: np.ndarray,
                           low_prices: np.ndarray, close_prices: np.ndarray,
                           body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> np.ndarray:
        """Detect shooting star patterns."""
        body_size = np.abs(close_prices - open_prices)
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices
        
        # Shooting star: small body, long upper shadow, short lower shadow
        shooting_star_mask = (
            (body_size <= (body_size + upper_shadow + lower_shadow) * body_ratio) &
            (upper_shadow >= body_size * shadow_ratio) &
            (lower_shadow <= body_size * 0.5)
        )
        return shooting_star_mask
    
    @staticmethod
    def detect_engulfing(open_prices: np.ndarray, high_prices: np.ndarray,
                        low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect engulfing patterns."""
        # Bullish engulfing: current candle completely engulfs previous bearish candle
        # Bearish engulfing: current candle completely engulfs previous bullish candle
        
        bullish_engulfing = (
            (close_prices[1:] > open_prices[1:]) &  # Current candle is bullish
            (close_prices[:-1] < open_prices[:-1]) &  # Previous candle is bearish
            (open_prices[1:] < close_prices[:-1]) &  # Current open < previous close
            (close_prices[1:] > open_prices[:-1])    # Current close > previous open
        )
        
        bearish_engulfing = (
            (close_prices[1:] < open_prices[1:]) &  # Current candle is bearish
            (close_prices[:-1] > open_prices[:-1]) &  # Previous candle is bullish
            (open_prices[1:] > close_prices[:-1]) &  # Current open > previous close
            (close_prices[1:] < open_prices[:-1])    # Current close < previous open
        )
        
        # Pad with False for the first element
        engulfing_mask = np.concatenate([[False], bullish_engulfing | bearish_engulfing])
        return engulfing_mask

class ImageRepresentationBuilder:
    """Converts OHLC data to image-like representations for CNN."""
    
    def __init__(self, height: int = 64, width: int = 64):
        self.height = height
        self.width = width
    
    def create_candlestick_image(self, open_prices: np.ndarray, high_prices: np.ndarray,
                                low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """
        Create image representation of candlestick patterns.
        
        Args:
            open_prices: Opening prices
            high_prices: High prices
            low_prices: Low prices
            close_prices: Closing prices
            
        Returns:
            Image array of shape (height, width, channels)
        """
        # Normalize prices to [0, 1] range
        min_price = min(low_prices.min(), open_prices.min(), close_prices.min())
        max_price = max(high_prices.max(), open_prices.max(), close_prices.max())
        price_range = max_price - min_price
        
        if price_range == 0:
            price_range = 1
        
        # Normalize all prices
        open_norm = (open_prices - min_price) / price_range
        high_norm = (high_prices - min_price) / price_range
        low_norm = (low_prices - min_price) / price_range
        close_norm = (close_prices - min_price) / price_range
        
        # Create image
        image = np.zeros((self.height, self.width, 3))
        
        # Scale to image dimensions
        n_candles = len(open_prices)
        candle_width = max(1, self.width // n_candles)
        
        for i in range(min(n_candles, self.width // candle_width)):
            x_start = i * candle_width
            x_end = min((i + 1) * candle_width, self.width)
            
            # Calculate y positions
            open_y = int((1 - open_norm[i]) * (self.height - 1))
            high_y = int((1 - high_norm[i]) * (self.height - 1))
            low_y = int((1 - low_norm[i]) * (self.height - 1))
            close_y = int((1 - close_norm[i]) * (self.height - 1))
            
            # Determine if bullish or bearish
            is_bullish = close_norm[i] > open_norm[i]
            
            # Draw wick (high to low)
            wick_color = [0.8, 0.8, 0.8]  # Gray
            for y in range(min(high_y, low_y), max(high_y, low_y) + 1):
                if 0 <= y < self.height:
                    image[y, x_start:x_end] = wick_color
            
            # Draw body
            body_start = min(open_y, close_y)
            body_end = max(open_y, close_y)
            
            if is_bullish:
                body_color = [0.2, 0.8, 0.2]  # Green
            else:
                body_color = [0.8, 0.2, 0.2]  # Red
            
            for y in range(body_start, body_end + 1):
                if 0 <= y < self.height:
                    image[y, x_start:x_end] = body_color
        
        return image
    
    def create_ohlc_image(self, open_prices: np.ndarray, high_prices: np.ndarray,
                         low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """
        Create OHLC image representation.
        
        Args:
            open_prices: Opening prices
            high_prices: High prices
            low_prices: Low prices
            close_prices: Closing prices
            
        Returns:
            Image array of shape (height, width, channels)
        """
        # Create separate channels for OHLC
        image = np.zeros((self.height, self.width, 4))
        
        # Normalize each series
        for i, prices in enumerate([open_prices, high_prices, low_prices, close_prices]):
            min_val, max_val = prices.min(), prices.max()
            if max_val == min_val:
                normalized = np.zeros_like(prices)
            else:
                normalized = (prices - min_val) / (max_val - min_val)
            
            # Scale to image height
            y_positions = ((1 - normalized) * (self.height - 1)).astype(int)
            
            # Draw lines
            for j in range(len(y_positions) - 1):
                x_start = int(j * self.width / len(y_positions))
                x_end = int((j + 1) * self.width / len(y_positions))
                
                y_start = max(0, min(self.height - 1, y_positions[j]))
                y_end = max(0, min(self.height - 1, y_positions[j + 1]))
                
                # Draw line
                if x_start != x_end:
                    for x in range(x_start, x_end):
                        if 0 <= x < self.width:
                            image[y_start:y_end + 1, x, i] = 1.0
        
        # Convert to 3 channels (RGB)
        rgb_image = np.zeros((self.height, self.width, 3))
        rgb_image[:, :, 0] = image[:, :, 0]  # Open -> Red
        rgb_image[:, :, 1] = image[:, :, 3]  # Close -> Green
        rgb_image[:, :, 2] = (image[:, :, 1] + image[:, :, 2]) / 2  # High/Low -> Blue
        
        return rgb_image

class CNNPatternRecognition:
    """
    CNN model for candlestick pattern recognition.
    """
    
    def __init__(self, config: CNNConfig):
        """
        Initialize CNN pattern recognition model.
        
        Args:
            config: CNN configuration
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. CNN model will be limited to configuration and data preparation.")
            
        self.config = config
        self.model = None
        self.pattern_detector = CandlestickPatternDetector()
        self.image_builder = ImageRepresentationBuilder(config.input_height, config.input_width)
        
        logger.info("CNN pattern recognition model initialized",
                   input_shape=(config.input_height, config.input_width, config.input_channels),
                   n_classes=config.n_classes)
    
    def build_model(self):
        """
        Build CNN model architecture.
        
        Returns:
            Compiled model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required to build CNN model")
            
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.config.input_height, self.config.input_width, self.config.input_channels)))
        
        # Convolutional layers
        for i in range(self.config.n_conv_layers):
            model.add(layers.Conv2D(
                filters=self.config.conv_filters[i],
                kernel_size=(self.config.conv_kernel_sizes[i], self.config.conv_kernel_sizes[i]),
                strides=(self.config.conv_strides[i], self.config.conv_strides[i]),
                padding='same',
                activation='relu'
            ))
            
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D(pool_size=(self.config.pool_sizes[i], self.config.pool_sizes[i])))
            model.add(layers.Dropout(self.config.dropout_rate))
        
        # Flatten
        model.add(layers.Flatten())
        
        # Dense layers
        for units in self.config.dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.config.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(self.config.n_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        logger.info("CNN model built successfully",
                   total_params=model.count_params())
        
        return model
    
    def create_pattern_labels(self, open_prices: np.ndarray, high_prices: np.ndarray,
                            low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """
        Create pattern labels for the data.
        
        Args:
            open_prices: Opening prices
            high_prices: High prices
            low_prices: Low prices
            close_prices: Closing prices
            
        Returns:
            Pattern labels array
        """
        # Detect patterns
        doji = self.pattern_detector.detect_doji(open_prices, high_prices, low_prices, close_prices)
        hammer = self.pattern_detector.detect_hammer(open_prices, high_prices, low_prices, close_prices)
        shooting_star = self.pattern_detector.detect_shooting_star(open_prices, high_prices, low_prices, close_prices)
        engulfing = self.pattern_detector.detect_engulfing(open_prices, high_prices, low_prices, close_prices)
        
        # Create labels (simplified - just using first 4 patterns)
        labels = np.zeros(len(open_prices))
        labels[doji] = 0
        labels[hammer] = 1
        labels[shooting_star] = 2
        labels[engulfing] = 3
        
        return labels.astype(int)
    
    def prepare_data(self, open_prices: np.ndarray, high_prices: np.ndarray,
                    low_prices: np.ndarray, close_prices: np.ndarray,
                    window_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for CNN training.
        
        Args:
            open_prices: Opening prices
            high_prices: High prices
            low_prices: Low prices
            close_prices: Closing prices
            window_size: Size of sliding window
            
        Returns:
            Tuple of (X, y) where X is images and y is labels
        """
        X, y = [], []
        
        # Create pattern labels
        pattern_labels = self.create_pattern_labels(open_prices, high_prices, low_prices, close_prices)
        
        for i in range(len(open_prices) - window_size + 1):
            # Extract window
            window_open = open_prices[i:i + window_size]
            window_high = high_prices[i:i + window_size]
            window_low = low_prices[i:i + window_size]
            window_close = close_prices[i:i + window_size]
            
            # Create image representation
            image = self.image_builder.create_candlestick_image(
                window_open, window_high, window_low, window_close
            )
            
            X.append(image)
            
            # Use the pattern label at the end of the window
            label = pattern_labels[i + window_size - 1]
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert to categorical
        y = tf.keras.utils.to_categorical(y, num_classes=self.config.n_classes)
        
        logger.info("Data prepared for CNN",
                   n_samples=len(X),
                   image_shape=X.shape,
                   label_shape=y.shape)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> TrainingResult:
        """
        Train the CNN model.
        
        Args:
            X: Training images
            y: Training labels
            validation_data: Optional validation data
            
        Returns:
            TrainingResult object
        """
        if self.model is None:
            self.build_model()
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                restore_best_weights=self.config.restore_best_weights,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        logger.info("Starting CNN training",
                   epochs=self.config.epochs,
                   batch_size=self.config.batch_size,
                   validation_split=self.config.validation_split)
        
        start_time = datetime.now()
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Find best epoch
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = min(history.history['val_loss'])
        best_val_accuracy = max(history.history['val_accuracy'])
        
        result = TrainingResult(
            model=self.model,
            history=history.history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            training_time=training_time,
            config=self.config
        )
        
        logger.info("CNN training completed",
                   best_epoch=best_epoch,
                   best_val_loss=best_val_loss,
                   best_val_accuracy=best_val_accuracy,
                   training_time=training_time)
        
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input images
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test images
            y: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        results = self.model.evaluate(X, y, verbose=0)
        
        return {
            'loss': results[0],
            'accuracy': results[1]
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info("Model saved", filepath=filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        logger.info("Model loaded", filepath=filepath)

# Convenience functions
def create_cnn_pattern_recognition(input_height: int = 64,
                                 input_width: int = 64,
                                 n_classes: int = len(PatternType),
                                 **kwargs) -> CNNPatternRecognition:
    """Create a CNN pattern recognition model with default configuration."""
    config = CNNConfig(
        input_height=input_height,
        input_width=input_width,
        n_classes=n_classes,
        **kwargs
    )
    return CNNPatternRecognition(config)

def create_simple_cnn(input_height: int = 32,
                     input_width: int = 32,
                     n_classes: int = 4,
                     **kwargs) -> CNNPatternRecognition:
    """Create a simple CNN for pattern recognition."""
    config = CNNConfig(
        input_height=input_height,
        input_width=input_width,
        n_classes=n_classes,
        n_conv_layers=2,
        conv_filters=[16, 32],
        dense_units=[64],
        **kwargs
    )
    return CNNPatternRecognition(config)