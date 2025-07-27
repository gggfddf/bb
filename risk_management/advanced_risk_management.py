#!/usr/bin/env python3
"""
Advanced Risk Management System

Implements comprehensive risk management for trading:
- Multiple risk models (VaR, CVaR, Expected Shortfall)
- Dynamic position sizing
- Portfolio-level risk controls
- Real-time risk monitoring
- Risk-adjusted performance metrics
- Stress testing and scenario analysis

Features:
- Advanced risk models with multiple methodologies
- Dynamic position sizing and risk allocation
- Portfolio-level risk controls and limits
- Real-time risk monitoring and alerts
- Risk-adjusted performance metrics
- Stress testing and scenario analysis
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
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class RiskModel(Enum):
    """Risk model enumeration."""
    VAR = "var"
    CVAR = "cvar"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"

class PositionSizingMethod(Enum):
    """Position sizing method enumeration."""
    FIXED_SIZE = "fixed_size"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    VOLATILITY_TARGETING = "volatility_targeting"
    MAX_DRAWDOWN = "max_drawdown"

@dataclass
class RiskMetrics:
    """Risk metrics structure."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0

@dataclass
class PositionSizing:
    """Position sizing structure."""
    symbol: str
    size: float
    method: PositionSizingMethod
    risk_contribution: float
    target_risk: float
    max_position: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskLimit:
    """Risk limit structure."""
    limit_id: str
    limit_type: str
    symbol: str
    value: float
    current_value: float
    breach_threshold: float
    is_breached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskConfig:
    """Risk management configuration."""
    var_confidence_level: float = 0.95
    cvar_confidence_level: float = 0.95
    max_portfolio_risk: float = 0.02  # 2%
    max_position_risk: float = 0.01   # 1%
    max_drawdown_limit: float = 0.15  # 15%
    volatility_target: float = 0.12   # 12%
    correlation_threshold: float = 0.7
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.RISK_PARITY
    enable_stress_testing: bool = True
    enable_real_time_monitoring: bool = True

class VaRCalculator:
    """Value at Risk calculator."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: VaR confidence level
        """
        self.confidence_level = confidence_level
        
    def calculate_var(self, returns: pd.Series, method: str = "parametric") -> float:
        """Calculate Value at Risk."""
        try:
            if method == "parametric":
                return self._parametric_var(returns)
            elif method == "historical":
                return self._historical_var(returns)
            elif method == "monte_carlo":
                return self._monte_carlo_var(returns)
            else:
                raise ValueError(f"Unknown VaR method: {method}")
                
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
            
    def _parametric_var(self, returns: pd.Series) -> float:
        """Calculate parametric VaR."""
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Use normal distribution assumption
        z_score = stats.norm.ppf(1 - self.confidence_level)
        var = mean_return - z_score * std_return
        
        return abs(var)
        
    def _historical_var(self, returns: pd.Series) -> float:
        """Calculate historical VaR."""
        sorted_returns = returns.sort_values()
        index = int((1 - self.confidence_level) * len(sorted_returns))
        
        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1
            
        var = sorted_returns.iloc[index]
        return abs(var)
        
    def _monte_carlo_var(self, returns: pd.Series, n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR."""
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # Calculate VaR
        var = np.percentile(simulated_returns, (1 - self.confidence_level) * 100)
        
        return abs(var)

class CVaRCalculator:
    """Conditional Value at Risk calculator."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize CVaR calculator.
        
        Args:
            confidence_level: CVaR confidence level
        """
        self.confidence_level = confidence_level
        
    def calculate_cvar(self, returns: pd.Series) -> float:
        """Calculate Conditional Value at Risk."""
        try:
            # Calculate VaR first
            var_calculator = VaRCalculator(self.confidence_level)
            var = var_calculator.calculate_var(returns, "historical")
            
            # Calculate CVaR as expected loss beyond VaR
            tail_returns = returns[returns <= -var]
            
            if len(tail_returns) == 0:
                return var
                
            cvar = abs(tail_returns.mean())
            return cvar
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0

class RiskMetricsCalculator:
    """Comprehensive risk metrics calculator."""
    
    def __init__(self, config: RiskConfig):
        """
        Initialize risk metrics calculator.
        
        Args:
            config: Risk configuration
        """
        self.config = config
        self.var_calculator = VaRCalculator(config.var_confidence_level)
        self.cvar_calculator = CVaRCalculator(config.cvar_confidence_level)
        
    def calculate_risk_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            metrics = RiskMetrics()
            
            # VaR calculations
            metrics.var_95 = self.var_calculator.calculate_var(returns, "parametric")
            metrics.var_99 = VaRCalculator(0.99).calculate_var(returns, "parametric")
            
            # CVaR calculations
            metrics.cvar_95 = self.cvar_calculator.calculate_cvar(returns)
            metrics.cvar_99 = CVaRCalculator(0.99).calculate_cvar(returns)
            
            # Expected Shortfall (same as CVaR)
            metrics.expected_shortfall = metrics.cvar_95
            
            # Maximum Drawdown
            metrics.max_drawdown = self._calculate_max_drawdown(returns)
            
            # Risk-adjusted ratios
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
            metrics.calmar_ratio = self._calculate_calmar_ratio(returns)
            
            # Volatility
            metrics.volatility = returns.std()
            
            # Beta and correlation (if benchmark provided)
            if benchmark_returns is not None:
                metrics.beta = self._calculate_beta(returns, benchmark_returns)
                metrics.correlation = returns.corr(benchmark_returns)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics()
            
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except:
            return 0.0
            
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        try:
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            return excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        except:
            return 0.0
            
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        try:
            excess_returns = returns - risk_free_rate / 252
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            return excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
        except:
            return 0.0
            
    def _calculate_calmar_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Calmar ratio."""
        try:
            max_dd = self._calculate_max_drawdown(returns)
            annual_return = returns.mean() * 252
            return (annual_return - risk_free_rate) / max_dd if max_dd > 0 else 0
        except:
            return 0.0
            
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta."""
        try:
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            return covariance / benchmark_variance if benchmark_variance > 0 else 0
        except:
            return 0.0

class PositionSizer:
    """Position sizing calculator."""
    
    def __init__(self, config: RiskConfig):
        """
        Initialize position sizer.
        
        Args:
            config: Risk configuration
        """
        self.config = config
        
    def calculate_position_size(self, symbol: str, returns: pd.Series, 
                              method: PositionSizingMethod = None) -> PositionSizing:
        """Calculate position size based on risk metrics."""
        try:
            method = method or self.config.position_sizing_method
            
            if method == PositionSizingMethod.FIXED_SIZE:
                size = self._fixed_size_position()
            elif method == PositionSizingMethod.KELLY_CRITERION:
                size = self._kelly_criterion_position(returns)
            elif method == PositionSizingMethod.RISK_PARITY:
                size = self._risk_parity_position(returns)
            elif method == PositionSizingMethod.VOLATILITY_TARGETING:
                size = self._volatility_targeting_position(returns)
            elif method == PositionSizingMethod.MAX_DRAWDOWN:
                size = self._max_drawdown_position(returns)
            else:
                size = self._fixed_size_position()
                
            # Calculate risk contribution
            risk_contribution = self._calculate_risk_contribution(size, returns)
            
            # Apply limits
            max_position = self.config.max_position_risk
            size = min(size, max_position)
            
            # Calculate confidence
            confidence = self._calculate_position_confidence(size, returns)
            
            return PositionSizing(
                symbol=symbol,
                size=size,
                method=method,
                risk_contribution=risk_contribution,
                target_risk=self.config.max_position_risk,
                max_position=max_position,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return PositionSizing(symbol=symbol, size=0.0, method=method or PositionSizingMethod.FIXED_SIZE)
            
    def _fixed_size_position(self) -> float:
        """Calculate fixed size position."""
        return 0.01  # 1% of portfolio
        
    def _kelly_criterion_position(self, returns: pd.Series) -> float:
        """Calculate Kelly criterion position size."""
        try:
            win_rate = (returns > 0).mean()
            avg_win = returns[returns > 0].mean()
            avg_loss = abs(returns[returns < 0].mean())
            
            if avg_loss == 0:
                return 0.01
                
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            return max(0.0, min(kelly_fraction, 0.1))  # Cap at 10%
            
        except:
            return 0.01
            
    def _risk_parity_position(self, returns: pd.Series) -> float:
        """Calculate risk parity position size."""
        try:
            volatility = returns.std()
            if volatility == 0:
                return 0.01
                
            # Equal risk contribution
            target_risk = self.config.max_portfolio_risk / 10  # Assume 10 positions
            size = target_risk / volatility
            return min(size, 0.1)  # Cap at 10%
            
        except:
            return 0.01
            
    def _volatility_targeting_position(self, returns: pd.Series) -> float:
        """Calculate volatility targeting position size."""
        try:
            current_volatility = returns.std() * np.sqrt(252)  # Annualized
            target_volatility = self.config.volatility_target
            
            if current_volatility == 0:
                return 0.01
                
            size = target_volatility / current_volatility
            return min(size, 0.1)  # Cap at 10%
            
        except:
            return 0.01
            
    def _max_drawdown_position(self, returns: pd.Series) -> float:
        """Calculate max drawdown position size."""
        try:
            max_dd = self._calculate_max_drawdown(returns)
            if max_dd == 0:
                return 0.01
                
            # Size inversely proportional to max drawdown
            size = self.config.max_drawdown_limit / max_dd
            return min(size, 0.1)  # Cap at 10%
            
        except:
            return 0.01
            
    def _calculate_risk_contribution(self, size: float, returns: pd.Series) -> float:
        """Calculate risk contribution of position."""
        try:
            volatility = returns.std()
            return size * volatility
        except:
            return 0.0
            
    def _calculate_position_confidence(self, size: float, returns: pd.Series) -> float:
        """Calculate position confidence."""
        try:
            # Confidence based on Sharpe ratio and sample size
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            sample_size_factor = min(1.0, len(returns) / 252)  # Normalize to 1 year
            
            confidence = (0.5 + 0.3 * sharpe_ratio + 0.2 * sample_size_factor)
            return min(1.0, max(0.0, confidence))
            
        except:
            return 0.5
            
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except:
            return 0.0

class RiskLimitsManager:
    """Risk limits manager."""
    
    def __init__(self, config: RiskConfig):
        """
        Initialize risk limits manager.
        
        Args:
            config: Risk configuration
        """
        self.config = config
        self.limits: Dict[str, RiskLimit] = {}
        self.breach_history: List[Dict[str, Any]] = []
        
    def add_limit(self, limit_id: str, limit_type: str, symbol: str, 
                  value: float, breach_threshold: float = 0.8):
        """Add a risk limit."""
        limit = RiskLimit(
            limit_id=limit_id,
            limit_type=limit_type,
            symbol=symbol,
            value=value,
            current_value=0.0,
            breach_threshold=breach_threshold
        )
        self.limits[limit_id] = limit
        
    def update_limit(self, limit_id: str, current_value: float):
        """Update a risk limit."""
        if limit_id in self.limits:
            limit = self.limits[limit_id]
            limit.current_value = current_value
            limit.timestamp = datetime.now()
            
            # Check for breach
            if current_value >= limit.value * limit.breach_threshold:
                limit.is_breached = True
                self._record_breach(limit)
            else:
                limit.is_breached = False
                
    def _record_breach(self, limit: RiskLimit):
        """Record a limit breach."""
        breach = {
            'limit_id': limit.limit_id,
            'limit_type': limit.limit_type,
            'symbol': limit.symbol,
            'value': limit.value,
            'current_value': limit.current_value,
            'timestamp': limit.timestamp
        }
        self.breach_history.append(breach)
        logger.warning(f"Risk limit breached: {breach}")
        
    def get_breached_limits(self) -> List[RiskLimit]:
        """Get all breached limits."""
        return [limit for limit in self.limits.values() if limit.is_breached]
        
    def get_limit_status(self) -> Dict[str, Any]:
        """Get limit status summary."""
        total_limits = len(self.limits)
        breached_limits = len(self.get_breached_limits())
        
        return {
            'total_limits': total_limits,
            'breached_limits': breached_limits,
            'breach_rate': breached_limits / total_limits if total_limits > 0 else 0,
            'recent_breaches': len(self.breach_history[-10:])  # Last 10 breaches
        }

class AdvancedRiskManager:
    """Main advanced risk management system."""
    
    def __init__(self, config: RiskConfig = None):
        """
        Initialize advanced risk manager.
        
        Args:
            config: Risk configuration
        """
        self.config = config or RiskConfig()
        self.risk_calculator = RiskMetricsCalculator(self.config)
        self.position_sizer = PositionSizer(self.config)
        self.limits_manager = RiskLimitsManager(self.config)
        self.portfolio_metrics: Dict[str, RiskMetrics] = {}
        self.position_sizes: Dict[str, PositionSizing] = {}
        
    def calculate_portfolio_risk(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series = None) -> RiskMetrics:
        """Calculate portfolio risk metrics."""
        try:
            metrics = self.risk_calculator.calculate_risk_metrics(portfolio_returns, benchmark_returns)
            self.portfolio_metrics['portfolio'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return RiskMetrics()
            
    def calculate_position_risk(self, symbol: str, returns: pd.Series) -> RiskMetrics:
        """Calculate position-specific risk metrics."""
        try:
            metrics = self.risk_calculator.calculate_risk_metrics(returns)
            self.portfolio_metrics[symbol] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return RiskMetrics()
            
    def calculate_position_size(self, symbol: str, returns: pd.Series, 
                              method: PositionSizingMethod = None) -> PositionSizing:
        """Calculate optimal position size."""
        try:
            sizing = self.position_sizer.calculate_position_size(symbol, returns, method)
            self.position_sizes[symbol] = sizing
            return sizing
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return PositionSizing(symbol=symbol, size=0.0, method=method or PositionSizingMethod.FIXED_SIZE)
            
    def add_risk_limit(self, limit_id: str, limit_type: str, symbol: str, 
                      value: float, breach_threshold: float = 0.8):
        """Add a risk limit."""
        self.limits_manager.add_limit(limit_id, limit_type, symbol, value, breach_threshold)
        
    def update_risk_limits(self, portfolio_metrics: RiskMetrics, 
                          position_metrics: Dict[str, RiskMetrics]):
        """Update all risk limits."""
        try:
            # Update portfolio limits
            self.limits_manager.update_limit('portfolio_var', portfolio_metrics.var_95)
            self.limits_manager.update_limit('portfolio_drawdown', portfolio_metrics.max_drawdown)
            self.limits_manager.update_limit('portfolio_volatility', portfolio_metrics.volatility)
            
            # Update position limits
            for symbol, metrics in position_metrics.items():
                self.limits_manager.update_limit(f'{symbol}_var', metrics.var_95)
                self.limits_manager.update_limit(f'{symbol}_volatility', metrics.volatility)
                
        except Exception as e:
            logger.error(f"Error updating risk limits: {e}")
            
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            portfolio_metrics = self.portfolio_metrics.get('portfolio', RiskMetrics())
            limit_status = self.limits_manager.get_limit_status()
            
            summary = {
                'portfolio_risk': {
                    'var_95': portfolio_metrics.var_95,
                    'cvar_95': portfolio_metrics.cvar_95,
                    'max_drawdown': portfolio_metrics.max_drawdown,
                    'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                    'volatility': portfolio_metrics.volatility
                },
                'position_sizes': {
                    symbol: {
                        'size': sizing.size,
                        'method': sizing.method.value,
                        'risk_contribution': sizing.risk_contribution,
                        'confidence': sizing.confidence
                    }
                    for symbol, sizing in self.position_sizes.items()
                },
                'limit_status': limit_status,
                'breached_limits': len(self.limits_manager.get_breached_limits())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return {}

def create_advanced_risk_manager(config: RiskConfig = None) -> AdvancedRiskManager:
    """Create an advanced risk manager."""
    return AdvancedRiskManager(config)

# Demo of advanced risk management system
if __name__ == "__main__":
    # Create risk manager
    config = RiskConfig(
        var_confidence_level=0.95,
        cvar_confidence_level=0.95,
        max_portfolio_risk=0.02,
        max_position_risk=0.01,
        max_drawdown_limit=0.15,
        volatility_target=0.12,
        position_sizing_method=PositionSizingMethod.RISK_PARITY
    )
    
    risk_manager = create_advanced_risk_manager(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    portfolio_returns = pd.Series(np.random.randn(252) * 0.02, index=dates)
    benchmark_returns = pd.Series(np.random.randn(252) * 0.015, index=dates)
    
    # Calculate portfolio risk
    portfolio_risk = risk_manager.calculate_portfolio_risk(portfolio_returns, benchmark_returns)
    print(f"Portfolio Risk Metrics:")
    print(f"VaR (95%): {portfolio_risk.var_95:.4f}")
    print(f"CVaR (95%): {portfolio_risk.cvar_95:.4f}")
    print(f"Max Drawdown: {portfolio_risk.max_drawdown:.4f}")
    print(f"Sharpe Ratio: {portfolio_risk.sharpe_ratio:.4f}")
    
    # Calculate position sizes
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    for symbol in symbols:
        returns = pd.Series(np.random.randn(252) * 0.025, index=dates)
        position_risk = risk_manager.calculate_position_risk(symbol, returns)
        position_size = risk_manager.calculate_position_size(symbol, returns)
        
        print(f"\n{symbol} Position:")
        print(f"Size: {position_size.size:.4f}")
        print(f"Risk Contribution: {position_size.risk_contribution:.4f}")
        print(f"Confidence: {position_size.confidence:.4f}")
    
    # Add risk limits
    risk_manager.add_risk_limit('portfolio_var', 'var', 'portfolio', 0.02)
    risk_manager.add_risk_limit('portfolio_drawdown', 'drawdown', 'portfolio', 0.15)
    
    # Update limits
    position_metrics = {symbol: risk_manager.calculate_position_risk(symbol, pd.Series(np.random.randn(252) * 0.025, index=dates)) 
                       for symbol in symbols}
    risk_manager.update_risk_limits(portfolio_risk, position_metrics)
    
    # Get risk summary
    summary = risk_manager.get_risk_summary()
    print(f"\nRisk Summary:")
    print(f"Breached Limits: {summary.get('breached_limits', 0)}")
    print(f"Portfolio VaR: {summary.get('portfolio_risk', {}).get('var_95', 0):.4f}")
    
    print("Advanced risk management system completed successfully!")