"""
Machine learning models for pattern detection and forecasting.
Includes baseline tree models and advanced sequence models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Tree models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available")

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from config import ModelConfig


class BaselineTreeModel:
    """
    Baseline gradient boosted tree model for pattern detection.
    Supports XGBoost, LightGBM, and CatBoost.
    """
    
    def __init__(self, config: ModelConfig, model_type: str = 'lightgbm'):
        self.config = config
        self.model_type = model_type
        self.model = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              class_weights: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train the tree model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weights: Class weights for imbalanced data
            
        Returns:
            Training history/metrics
        """
        # Determine task type
        is_binary = len(np.unique(y_train)) == 2
        is_multiclass = len(np.unique(y_train)) > 2 and not np.issubdtype(y_train.dtype, np.floating)
        is_regression = np.issubdtype(y_train.dtype, np.floating)
        
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return self._train_lightgbm(X_train, y_train, X_val, y_val, 
                                       is_binary, is_regression, class_weights)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return self._train_xgboost(X_train, y_train, X_val, y_val,
                                      is_binary, is_regression, class_weights)
        elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
            return self._train_catboost(X_train, y_train, X_val, y_val,
                                       is_binary, is_regression, class_weights)
        else:
            raise ValueError(f"Model type {self.model_type} not available")
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val,
                       is_binary, is_regression, class_weights):
        """Train LightGBM model."""
        params = {
            'objective': 'regression' if is_regression else 'binary' if is_binary else 'multiclass',
            'num_leaves': 2 ** self.config.tree_max_depth - 1,
            'learning_rate': self.config.tree_learning_rate,
            'n_estimators': self.config.tree_n_estimators,
            'max_depth': self.config.tree_max_depth,
            'verbose': -1,
            'random_state': 42,
        }
        
        if not is_regression and not is_binary:
            params['num_class'] = len(np.unique(y_train))
        
        # Handle class weights
        if class_weights and not is_regression:
            sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train])
        else:
            sample_weights = None
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        valid_sets = [train_data]
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
        
        # Train
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            callbacks=callbacks
        )
        
        return {'best_iteration': self.model.best_iteration}
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val,
                      is_binary, is_regression, class_weights):
        """Train XGBoost model."""
        params = {
            'objective': 'reg:squarederror' if is_regression else 
                        'binary:logistic' if is_binary else 'multi:softprob',
            'max_depth': self.config.tree_max_depth,
            'learning_rate': self.config.tree_learning_rate,
            'n_estimators': self.config.tree_n_estimators,
            'random_state': 42,
            'verbosity': 0,
        }
        
        if not is_regression and not is_binary:
            params['num_class'] = len(np.unique(y_train))
        
        # Handle class weights
        if class_weights and not is_regression:
            sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train])
        else:
            sample_weights = None
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        
        eval_set = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_set.append((dval, 'val'))
        
        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.tree_n_estimators,
            evals=eval_set,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        return {'best_iteration': self.model.best_iteration}
    
    def _train_catboost(self, X_train, y_train, X_val, y_val,
                       is_binary, is_regression, class_weights):
        """Train CatBoost model."""
        if is_regression:
            self.model = CatBoostRegressor(
                iterations=self.config.tree_n_estimators,
                depth=self.config.tree_max_depth,
                learning_rate=self.config.tree_learning_rate,
                verbose=False,
                random_state=42
            )
        else:
            self.model = CatBoostClassifier(
                iterations=self.config.tree_n_estimators,
                depth=self.config.tree_max_depth,
                learning_rate=self.config.tree_learning_rate,
                verbose=False,
                random_state=42
            )
        
        # Handle class weights
        if class_weights and not is_regression:
            class_weights_list = [class_weights.get(i, 1.0) for i in range(len(class_weights))]
            self.model.set_params(class_weights=class_weights_list)
        
        # Train
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        return {'best_iteration': self.model.get_best_iteration()}
    
    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == 'lightgbm':
            pred = self.model.predict(X)
        elif self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X)
            pred = self.model.predict(dmatrix)
        else:  # catboost
            pred = self.model.predict(X) if not return_proba else self.model.predict_proba(X)
        
        return pred
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == 'lightgbm':
            importance = self.model.feature_importance(importance_type='gain')
        elif self.model_type == 'xgboost':
            importance = self.model.get_score(importance_type='gain')
            importance = [importance.get(f'f{i}', 0) for i in range(len(feature_names))]
        else:  # catboost
            importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df


if TORCH_AVAILABLE:
    
    class TimeSeriesDataset(Dataset):
        """PyTorch dataset for time series data."""
        
        def __init__(self, sequences: np.ndarray, labels: np.ndarray):
            self.sequences = torch.FloatTensor(sequences)
            self.labels = torch.FloatTensor(labels)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]
    
    
    class LSTMModel(nn.Module):
        """LSTM model for sequence-based prediction."""
        
        def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                     output_size: int, dropout: float = 0.2):
            super(LSTMModel, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Use last hidden state
            last_hidden = h_n[-1]  # Shape: (batch, hidden_size)
            
            out = self.dropout(last_hidden)
            out = self.fc(out)
            
            return out
    
    
    class CNNModel(nn.Module):
        """1D CNN model for pattern recognition in sequences."""
        
        def __init__(self, input_size: int, num_classes: int,
                     filters: List[int], kernel_sizes: List[int]):
            super(CNNModel, self).__init__()
            
            self.conv_layers = nn.ModuleList()
            in_channels = input_size
            
            for out_channels, kernel_size in zip(filters, kernel_sizes):
                self.conv_layers.append(nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ))
                in_channels = out_channels
            
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(filters[-1], num_classes)
        
        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            # Transpose for Conv1d: (batch, input_size, seq_len)
            x = x.transpose(1, 2)
            
            for conv in self.conv_layers:
                x = F.relu(conv(x))
            
            # Global max pooling
            x = self.pool(x)  # Shape: (batch, filters[-1], 1)
            x = x.squeeze(2)  # Shape: (batch, filters[-1])
            
            out = self.fc(x)
            return out
    
    
    class TransformerModel(nn.Module):
        """Transformer model for time series."""
        
        def __init__(self, input_size: int, d_model: int, nhead: int,
                     num_layers: int, output_size: int, dropout: float = 0.1,
                     max_seq_len: int = 100):
            super(TransformerModel, self).__init__()
            
            self.d_model = d_model
            self.input_projection = nn.Linear(input_size, d_model)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
            
            # Transformer encoder
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layers,
                num_layers=num_layers
            )
            
            self.fc = nn.Linear(d_model, output_size)
        
        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            x = self.input_projection(x)  # (batch, seq_len, d_model)
            x = self.pos_encoder(x)
            
            # Transformer
            transformer_out = self.transformer_encoder(x)
            
            # Use mean pooling across sequence
            pooled = transformer_out.mean(dim=1)  # (batch, d_model)
            
            out = self.fc(pooled)
            return out
    
    
    class PositionalEncoding(nn.Module):
        """Positional encoding for Transformer."""
        
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
    
    
    class SequenceModel:
        """Wrapper for sequence-based deep learning models."""
        
        def __init__(self, config: ModelConfig, model_type: str = 'lstm'):
            self.config = config
            self.model_type = model_type
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        def build_model(self, input_size: int, output_size: int, seq_len: int):
            """Build the model architecture."""
            if self.model_type == 'lstm':
                self.model = LSTMModel(
                    input_size=input_size,
                    hidden_size=self.config.lstm_hidden_size,
                    num_layers=self.config.lstm_num_layers,
                    output_size=output_size,
                    dropout=self.config.lstm_dropout
                )
            elif self.model_type == 'cnn':
                self.model = CNNModel(
                    input_size=input_size,
                    num_classes=output_size,
                    filters=self.config.cnn_filters,
                    kernel_sizes=self.config.cnn_kernel_sizes
                )
            elif self.model_type == 'transformer':
                self.model = TransformerModel(
                    input_size=input_size,
                    d_model=self.config.transformer_d_model,
                    nhead=self.config.transformer_nhead,
                    num_layers=self.config.transformer_num_layers,
                    output_size=output_size,
                    dropout=self.config.transformer_dropout,
                    max_seq_len=seq_len
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.model = self.model.to(self.device)
        
        def train(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None) -> Dict[str, List]:
            """
            Train the sequence model.
            
            Args:
                X_train: Training sequences (batch, seq_len, features)
                y_train: Training labels
                X_val: Validation sequences
                y_val: Validation labels
            
            Returns:
                Training history
            """
            if self.model is None:
                raise ValueError("Model not built. Call build_model() first.")
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            val_loader = None
            if X_val is not None and y_val is not None:
                val_dataset = TimeSeriesDataset(X_val, y_val)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )
            
            # Loss and optimizer
            is_classification = len(y_train.shape) == 1 or y_train.shape[1] == 1
            
            if is_classification:
                if self.config.use_focal_loss:
                    criterion = FocalLoss(
                        alpha=self.config.focal_loss_alpha,
                        gamma=self.config.focal_loss_gamma
                    )
                else:
                    criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.MSELoss()
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            history = {'train_loss': [], 'val_loss': []}
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                # Train
                self.model.train()
                train_losses = []
                
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                avg_train_loss = np.mean(train_losses)
                history['train_loss'].append(avg_train_loss)
                
                # Validation
                if val_loader is not None:
                    self.model.eval()
                    val_losses = []
                    
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch = X_batch.to(self.device)
                            y_batch = y_batch.to(self.device)
                            
                            outputs = self.model(X_batch)
                            loss = criterion(outputs, y_batch)
                            val_losses.append(loss.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    history['val_loss'].append(avg_val_loss)
                    
                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config.early_stopping_patience:
                        break
            
            return history
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions."""
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor)
            
            return predictions.cpu().numpy()
    
    
    class FocalLoss(nn.Module):
        """Focal Loss for handling class imbalance."""
        
        def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            return focal_loss.mean()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate model predictions with various metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Classification metrics
    if len(np.unique(y_true)) <= 10:  # Assume classification
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle binary vs multiclass
        average = 'binary' if len(np.unique(y_true)) == 2 else 'macro'
        
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                    metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                pass
    
    return metrics
