"""
Transformer Model Implementation for Sequence Prediction

A comprehensive Transformer implementation using TensorFlow/Keras for time series
and sequence prediction tasks in stock market analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime
import warnings
import math

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. Transformer functionality will be limited.")

logger = structlog.get_logger()

class TaskType(Enum):
    """Supported task types for Transformer."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    # Architecture parameters
    sequence_length: int = 60
    n_features: int = 1
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout_rate: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    # Task parameters
    task_type: TaskType = TaskType.REGRESSION
    n_classes: int = 2
    
    # Early stopping parameters
    patience: int = 10
    min_delta: float = 0.001
    restore_best_weights: bool = True
    
    # Additional parameters
    use_positional_encoding: bool = True
    max_position: int = 1000

@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any
    history: Dict[str, List[float]]
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: Optional[float] = None
    training_time: float = 0.0
    config: TransformerConfig = None

class PositionalEncoding:
    """Positional encoding layer for Transformer."""
    
    def __init__(self, d_model: int, max_position: int = 1000, **kwargs):
        self.d_model = d_model
        self.max_position = max_position
        
        # Create positional encoding matrix
        pe = np.zeros((max_position, d_model))
        position = np.arange(0, max_position, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        try:
            self.pe = tf.constant(pe, dtype=tf.float32)
        except:
            self.pe = pe
    
    def call(self, x):
        try:
            seq_len = tf.shape(x)[1]
            return x + self.pe[:seq_len, :]
        except:
            # Fallback if TensorFlow is not available
            seq_len = x.shape[1]
            return x + self.pe[:seq_len, :]

class MultiHeadAttention:
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, **kwargs):
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        
        self.d_k = d_model // n_heads
        
        if TENSORFLOW_AVAILABLE:
            self.wq = layers.Dense(d_model)
            self.wk = layers.Dense(d_model)
            self.wv = layers.Dense(d_model)
            self.wo = layers.Dense(d_model)
        else:
            self.wq = None
            self.wk = None
            self.wv = None
            self.wo = None
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Scaled dot-product attention."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, (batch_size, -1, self.n_heads, self.d_k))
        k = tf.reshape(k, (batch_size, -1, self.n_heads, self.d_k))
        v = tf.reshape(v, (batch_size, -1, self.n_heads, self.d_k))
        
        # Transpose for attention computation
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        # Apply attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape back
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.wo(concat_attention)
        return output, attention_weights

class TransformerBlock:
    """Transformer block with attention and feed-forward layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=True, mask=None):
        # Multi-head attention
        attn_output, _ = self.attention(x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class SequenceDataPreparator:
    """Handles sequence data preparation for Transformer models."""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        
    def create_sequences(self, data: np.ndarray, targets: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences from time series data.
        
        Args:
            data: Input time series data (n_samples, n_features)
            targets: Target values (n_samples,)
            
        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            if targets is not None:
                y.append(targets[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y) if targets is not None else None
        
        return X, y
    
    def prepare_classification_data(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for classification tasks.
        
        Args:
            data: Input time series data
            targets: Binary classification targets
            
        Returns:
            Tuple of (X, y) for classification
        """
        X, y = self.create_sequences(data, targets)
        
        # Ensure targets are categorical for classification
        if len(y.shape) == 1:
            try:
                y = tf.keras.utils.to_categorical(y)
            except:
                # Fallback if TensorFlow is not available
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            
        return X, y
    
    def prepare_regression_data(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for regression tasks.
        
        Args:
            data: Input time series data
            targets: Continuous targets
            
        Returns:
            Tuple of (X, y) for regression
        """
        return self.create_sequences(data, targets)

class TransformerModel:
    """
    Transformer model implementation for sequence prediction.
    """
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize Transformer model.
        
        Args:
            config: Transformer configuration
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Transformer model will be limited to configuration and data preparation.")
            
        self.config = config
        self.model = None
        self.data_preparator = SequenceDataPreparator(config.sequence_length)
        
        logger.info("Transformer model initialized",
                   sequence_length=config.sequence_length,
                   n_features=config.n_features,
                   d_model=config.d_model,
                   n_heads=config.n_heads,
                   task_type=config.task_type.value)
    
    def build_model(self):
        """
        Build Transformer model architecture.
        
        Returns:
            Compiled model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required to build Transformer model")
            
        # Input layer
        inputs = layers.Input(shape=(self.config.sequence_length, self.config.n_features))
        
        # Project to d_model dimensions
        x = layers.Dense(self.config.d_model)(inputs)
        
        # Add positional encoding
        if self.config.use_positional_encoding:
            x = PositionalEncoding(self.config.d_model, self.config.max_position)(x)
        
        # Transformer blocks
        for _ in range(self.config.n_layers):
            x = TransformerBlock(
                self.config.d_model,
                self.config.n_heads,
                self.config.d_ff,
                self.config.dropout_rate
            )(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        if self.config.task_type == TaskType.CLASSIFICATION:
            outputs = layers.Dense(self.config.n_classes, activation='softmax')(x)
        else:
            outputs = layers.Dense(1, activation='linear')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        if self.config.task_type == TaskType.CLASSIFICATION:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='mse',
                metrics=['mae']
            )
        
        self.model = model
        
        logger.info("Transformer model built successfully",
                   total_params=model.count_params(),
                   task_type=self.config.task_type.value)
        
        return model
    
    def prepare_data(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            data: Input time series data
            targets: Target values
            
        Returns:
            Tuple of (X, y) ready for training
        """
        if self.config.task_type == TaskType.CLASSIFICATION:
            return self.data_preparator.prepare_classification_data(data, targets)
        else:
            return self.data_preparator.prepare_regression_data(data, targets)
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> TrainingResult:
        """
        Train the Transformer model.
        
        Args:
            X: Training sequences
            y: Training targets
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
        
        logger.info("Starting Transformer training",
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
        best_val_accuracy = max(history.history.get('val_accuracy', [0]))
        
        result = TrainingResult(
            model=self.model,
            history=history.history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            training_time=training_time,
            config=self.config
        )
        
        logger.info("Transformer training completed",
                   best_epoch=best_epoch,
                   best_val_loss=best_val_loss,
                   training_time=training_time)
        
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input sequences
            
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
            X: Test sequences
            y: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        results = self.model.evaluate(X, y, verbose=0)
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            return {
                'loss': results[0],
                'accuracy': results[1]
            }
        else:
            return {
                'loss': results[0],
                'mae': results[1]
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
def create_transformer_model(sequence_length: int = 60,
                           n_features: int = 1,
                           d_model: int = 128,
                           n_heads: int = 8,
                           task_type: TaskType = TaskType.REGRESSION,
                           **kwargs) -> TransformerModel:
    """Create a Transformer model with default configuration."""
    config = TransformerConfig(
        sequence_length=sequence_length,
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        task_type=task_type,
        **kwargs
    )
    return TransformerModel(config)

def create_classification_transformer(sequence_length: int = 60,
                                   n_features: int = 1,
                                   n_classes: int = 2,
                                   d_model: int = 128,
                                   n_heads: int = 8,
                                   **kwargs) -> TransformerModel:
    """Create a Transformer model for classification tasks."""
    return create_transformer_model(
        sequence_length=sequence_length,
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        task_type=TaskType.CLASSIFICATION,
        n_classes=n_classes,
        **kwargs
    )

def create_regression_transformer(sequence_length: int = 60,
                                n_features: int = 1,
                                d_model: int = 128,
                                n_heads: int = 8,
                                **kwargs) -> TransformerModel:
    """Create a Transformer model for regression tasks."""
    return create_transformer_model(
        sequence_length=sequence_length,
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        task_type=TaskType.REGRESSION,
        **kwargs
    )