"""
GRU Model Implementation for Sequence Prediction

A comprehensive GRU implementation using TensorFlow/Keras for time series
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

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. GRU functionality will be limited.")

logger = structlog.get_logger()

class TaskType(Enum):
    """Supported task types for GRU."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

@dataclass
class GRUConfig:
    """Configuration for GRU model."""
    # Architecture parameters
    sequence_length: int = 60
    n_features: int = 1
    n_gru_layers: int = 2
    gru_units: List[int] = field(default_factory=lambda: [50, 50])
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.2
    
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
    use_bidirectional: bool = False
    return_sequences: bool = False

@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any
    history: Dict[str, List[float]]
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: Optional[float] = None
    training_time: float = 0.0
    config: GRUConfig = None

class SequenceDataPreparator:
    """Handles sequence data preparation for GRU models."""
    
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

class GRUModel:
    """
    GRU model implementation for sequence prediction.
    """
    
    def __init__(self, config: GRUConfig):
        """
        Initialize GRU model.
        
        Args:
            config: GRU configuration
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. GRU model will be limited to configuration and data preparation.")
            
        self.config = config
        self.model = None
        self.data_preparator = SequenceDataPreparator(config.sequence_length)
        
        logger.info("GRU model initialized",
                   sequence_length=config.sequence_length,
                   n_features=config.n_features,
                   task_type=config.task_type.value)
    
    def build_model(self):
        """
        Build GRU model architecture.
        
        Returns:
            Compiled model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required to build GRU model")
            
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.config.sequence_length, self.config.n_features)))
        
        # GRU layers
        for i, units in enumerate(self.config.gru_units):
            return_sequences = (i < len(self.config.gru_units) - 1) or self.config.return_sequences
            
            if self.config.use_bidirectional:
                gru_layer = layers.Bidirectional(
                    layers.GRU(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.config.dropout_rate,
                        recurrent_dropout=self.config.recurrent_dropout
                    )
                )
            else:
                gru_layer = layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout
                )
            
            model.add(gru_layer)
            
            # Add dropout after each GRU layer (except the last one if return_sequences=False)
            if return_sequences:
                model.add(layers.Dropout(self.config.dropout_rate))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.config.dropout_rate))
        
        # Output layer
        if self.config.task_type == TaskType.CLASSIFICATION:
            model.add(layers.Dense(self.config.n_classes, activation='softmax'))
        else:
            model.add(layers.Dense(1, activation='linear'))
        
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
        
        logger.info("GRU model built successfully",
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
        Train the GRU model.
        
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
        
        logger.info("Starting GRU training",
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
        
        logger.info("GRU training completed",
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

class ModelComparison:
    """Framework for comparing GRU and LSTM models."""
    
    def __init__(self):
        self.results = {}
    
    def compare_models(self, 
                      data: np.ndarray, 
                      targets: np.ndarray,
                      sequence_length: int = 60,
                      n_features: int = 1,
                      task_type: TaskType = TaskType.REGRESSION,
                      **kwargs) -> Dict[str, Any]:
        """
        Compare GRU and LSTM models on the same dataset.
        
        Args:
            data: Input time series data
            targets: Target values
            sequence_length: Length of sequences
            n_features: Number of features
            task_type: Type of task (regression/classification)
            **kwargs: Additional parameters for models
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting model comparison")
        
        # Create GRU model
        gru_config = GRUConfig(
            sequence_length=sequence_length,
            n_features=n_features,
            task_type=task_type,
            **kwargs
        )
        gru_model = GRUModel(gru_config)
        
        # Create LSTM model (import here to avoid circular imports)
        try:
            from ml_models.deep_learning.lstm_model import LSTMModel, LSTMConfig
            lstm_config = LSTMConfig(
                sequence_length=sequence_length,
                n_features=n_features,
                task_type=task_type,
                **kwargs
            )
            lstm_model = LSTMModel(lstm_config)
        except ImportError:
            logger.warning("LSTM model not available for comparison")
            lstm_model = None
        
        # Prepare data
        X, y = gru_model.prepare_data(data, targets)
        
        # Train GRU model
        try:
            gru_result = gru_model.train(X, y)
            self.results['gru'] = {
                'model': gru_model,
                'result': gru_result,
                'config': gru_config
            }
        except Exception as e:
            logger.error("GRU training failed", error=str(e))
            self.results['gru'] = None
        
        # Train LSTM model if available
        if lstm_model is not None:
            try:
                lstm_result = lstm_model.train(X, y)
                self.results['lstm'] = {
                    'model': lstm_model,
                    'result': lstm_result,
                    'config': lstm_config
                }
            except Exception as e:
                logger.error("LSTM training failed", error=str(e))
                self.results['lstm'] = None
        
        # Generate comparison summary
        comparison = self._generate_comparison_summary()
        
        logger.info("Model comparison completed", comparison=comparison)
        
        return comparison
    
    def _generate_comparison_summary(self) -> Dict[str, Any]:
        """Generate summary of model comparison."""
        summary = {
            'models_tested': list(self.results.keys()),
            'best_model': None,
            'comparison_metrics': {}
        }
        
        if 'gru' in self.results and self.results['gru'] is not None:
            gru_result = self.results['gru']['result']
            summary['comparison_metrics']['gru'] = {
                'best_val_loss': gru_result.best_val_loss,
                'best_val_accuracy': gru_result.best_val_accuracy,
                'training_time': gru_result.training_time,
                'best_epoch': gru_result.best_epoch
            }
        
        if 'lstm' in self.results and self.results['lstm'] is not None:
            lstm_result = self.results['lstm']['result']
            summary['comparison_metrics']['lstm'] = {
                'best_val_loss': lstm_result.best_val_loss,
                'best_val_accuracy': lstm_result.best_val_accuracy,
                'training_time': lstm_result.training_time,
                'best_epoch': lstm_result.best_epoch
            }
        
        # Determine best model
        if 'gru' in summary['comparison_metrics'] and 'lstm' in summary['comparison_metrics']:
            gru_loss = summary['comparison_metrics']['gru']['best_val_loss']
            lstm_loss = summary['comparison_metrics']['lstm']['best_val_loss']
            
            if gru_loss < lstm_loss:
                summary['best_model'] = 'gru'
            else:
                summary['best_model'] = 'lstm'
        elif 'gru' in summary['comparison_metrics']:
            summary['best_model'] = 'gru'
        elif 'lstm' in summary['comparison_metrics']:
            summary['best_model'] = 'lstm'
        
        return summary

# Convenience functions
def create_gru_model(sequence_length: int = 60,
                    n_features: int = 1,
                    task_type: TaskType = TaskType.REGRESSION,
                    **kwargs) -> GRUModel:
    """Create a GRU model with default configuration."""
    config = GRUConfig(
        sequence_length=sequence_length,
        n_features=n_features,
        task_type=task_type,
        **kwargs
    )
    return GRUModel(config)

def create_classification_gru(sequence_length: int = 60,
                            n_features: int = 1,
                            n_classes: int = 2,
                            **kwargs) -> GRUModel:
    """Create a GRU model for classification tasks."""
    return create_gru_model(
        sequence_length=sequence_length,
        n_features=n_features,
        task_type=TaskType.CLASSIFICATION,
        n_classes=n_classes,
        **kwargs
    )

def create_regression_gru(sequence_length: int = 60,
                         n_features: int = 1,
                         **kwargs) -> GRUModel:
    """Create a GRU model for regression tasks."""
    return create_gru_model(
        sequence_length=sequence_length,
        n_features=n_features,
        task_type=TaskType.REGRESSION,
        **kwargs
    )