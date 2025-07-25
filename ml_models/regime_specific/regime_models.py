"""
Regime-Specific Models

Models that are specifically trained for different market regimes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class RegimeModel:
    """Model trained for a specific regime."""
    regime_type: str
    model: Any
    performance: Dict[str, float]
    training_samples: int

class RegimeSpecificModels:
    """Collection of models trained for different regimes."""
    
    def __init__(self):
        self.regime_models = {}
        self.regime_classifier = None
    
    def train_regime_model(self, regime_type: str, X: np.ndarray, y: np.ndarray, 
                          model_type: str = 'random_forest') -> RegimeModel:
        """Train a model for a specific regime."""
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'linear':
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            model.fit(X, y)
            
            # Evaluate performance
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))
            
            performance = {
                'mse': mse,
                'mae': mae,
                'r2': model.score(X, y) if hasattr(model, 'score') else 0.0
            }
            
            regime_model = RegimeModel(
                regime_type=regime_type,
                model=model,
                performance=performance,
                training_samples=len(X)
            )
            
            self.regime_models[regime_type] = regime_model
            
            logger.info(f"Trained {model_type} model for {regime_type} regime",
                       performance=performance,
                       samples=len(X))
            
            return regime_model
            
        except ImportError:
            logger.warning("Scikit-learn not available for regime-specific models")
            return None
    
    def predict_regime_specific(self, regime_type: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a regime-specific model."""
        if regime_type not in self.regime_models:
            raise ValueError(f"No model trained for regime: {regime_type}")
        
        model = self.regime_models[regime_type].model
        return model.predict(X)
    
    def get_regime_performance(self, regime_type: str) -> Dict[str, float]:
        """Get performance metrics for a specific regime model."""
        if regime_type not in self.regime_models:
            return {}
        
        return self.regime_models[regime_type].performance
    
    def list_regimes(self) -> List[str]:
        """List all available regimes."""
        return list(self.regime_models.keys())

# Convenience function
def create_regime_specific_models() -> RegimeSpecificModels:
    """Create regime-specific models."""
    return RegimeSpecificModels()