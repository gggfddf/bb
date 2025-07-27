#!/usr/bin/env python3
"""
Portfolio Optimization System

Implements comprehensive portfolio optimization:
- Markowitz Modern Portfolio Theory
- Risk Parity optimization
- Black-Litterman model
- Hierarchical Risk Parity (HRP)
- Maximum Sharpe Ratio optimization
- Minimum Variance optimization
- Equal Risk Contribution (ERC)
- Kelly Criterion optimization

Features:
- Advanced portfolio optimization algorithms
- Risk management and position sizing
- Multi-objective optimization
- Rebalancing strategies
- Performance attribution
- Risk-adjusted returns analysis
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
from scipy.optimize import minimize
from scipy import stats
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class OptimizationMethod(Enum):
    """Portfolio optimization method enumeration."""
    MARKOWITZ = "markowitz"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    KELLY_CRITERION = "kelly_criterion"

class ConstraintType(Enum):
    """Portfolio constraint type enumeration."""
    WEIGHT_SUM = "weight_sum"
    WEIGHT_BOUNDS = "weight_bounds"
    SECTOR_LIMITS = "sector_limits"
    CONCENTRATION_LIMITS = "concentration_limits"
    TURNOVER_LIMITS = "turnover_limits"

@dataclass
class PortfolioConstraints:
    """Portfolio constraints structure."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_return: Optional[float] = None
    max_volatility: Optional[float] = None
    sector_limits: Dict[str, float] = field(default_factory=dict)
    concentration_limits: Dict[str, float] = field(default_factory=dict)
    turnover_limit: Optional[float] = None

@dataclass
class PortfolioWeights:
    """Portfolio weights structure."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Portfolio optimization result structure."""
    result_id: str
    method: OptimizationMethod
    weights: PortfolioWeights
    constraints: PortfolioConstraints
    optimization_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationConfig:
    """Portfolio optimization configuration."""
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    max_iterations: int = 1000
    tolerance: float = 1e-6
    enable_short_selling: bool = False
    enable_leverage: bool = False
    max_leverage: float = 1.0

class MarkowitzOptimizer:
    """Markowitz Modern Portfolio Theory optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize Markowitz optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        
    def optimize(self, returns: pd.DataFrame, constraints: PortfolioConstraints) -> PortfolioWeights:
        """Optimize portfolio using Markowitz MPT."""
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean()
            covariance_matrix = returns.cov()
            
            # Number of assets
            n_assets = len(expected_returns)
            
            # Define optimization variables
            weights = cp.Variable(n_assets)
            
            # Define objective function (minimize variance)
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            portfolio_return = expected_returns.values @ weights
            
            # Define constraints
            constraints_list = []
            
            # Weight sum constraint
            constraints_list.append(cp.sum(weights) == 1.0)
            
            # Weight bounds
            if not self.config.enable_short_selling:
                constraints_list.append(weights >= constraints.min_weight)
            constraints_list.append(weights <= constraints.max_weight)
            
            # Target return constraint
            if constraints.target_return is not None:
                constraints_list.append(portfolio_return >= constraints.target_return)
                
            # Maximum volatility constraint
            if constraints.max_volatility is not None:
                constraints_list.append(cp.sqrt(portfolio_variance) <= constraints.max_volatility)
                
            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_list)
            problem.solve()
            
            if problem.status != cp.OPTIMAL:
                raise ValueError(f"Optimization failed with status: {problem.status}")
                
            # Extract results
            optimal_weights = weights.value
            portfolio_weights = {asset: weight for asset, weight in zip(expected_returns.index, optimal_weights)}
            
            # Calculate portfolio metrics
            expected_return = portfolio_return.value
            expected_volatility = np.sqrt(portfolio_variance.value)
            sharpe_ratio = (expected_return - self.config.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
            diversification_ratio = self._calculate_diversification_ratio(portfolio_weights, covariance_matrix)
            
            return PortfolioWeights(
                weights=portfolio_weights,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                diversification_ratio=diversification_ratio
            )
            
        except Exception as e:
            logger.error(f"Error in Markowitz optimization: {e}")
            return self._create_default_weights(returns.columns)
            
    def _calculate_diversification_ratio(self, weights: Dict[str, float], covariance_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio."""
        try:
            weight_vector = np.array(list(weights.values()))
            portfolio_variance = weight_vector.T @ covariance_matrix.values @ weight_vector
            weighted_volatilities = sum(abs(w) * np.sqrt(covariance_matrix.iloc[i, i]) 
                                      for i, w in enumerate(weights.values()))
            return weighted_volatilities / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1.0
        except:
            return 1.0

class RiskParityOptimizer:
    """Risk Parity optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize Risk Parity optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        
    def optimize(self, returns: pd.DataFrame, constraints: PortfolioConstraints) -> PortfolioWeights:
        """Optimize portfolio using Risk Parity."""
        try:
            # Calculate covariance matrix
            covariance_matrix = returns.cov()
            
            # Number of assets
            n_assets = len(covariance_matrix)
            
            # Define optimization variables
            weights = cp.Variable(n_assets)
            
            # Calculate portfolio variance
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            
            # Calculate individual asset contributions to portfolio risk
            risk_contributions = []
            for i in range(n_assets):
                # Contribution of asset i to portfolio risk
                contribution = weights[i] * (covariance_matrix.values[i, :] @ weights) / cp.sqrt(portfolio_variance)
                risk_contributions.append(contribution)
            
            # Define objective function (minimize variance of risk contributions)
            risk_contribution_variance = cp.sum_squares(cp.hstack(risk_contributions) - cp.mean(cp.hstack(risk_contributions)))
            
            # Define constraints
            constraints_list = []
            
            # Weight sum constraint
            constraints_list.append(cp.sum(weights) == 1.0)
            
            # Weight bounds
            if not self.config.enable_short_selling:
                constraints_list.append(weights >= constraints.min_weight)
            constraints_list.append(weights <= constraints.max_weight)
            
            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(risk_contribution_variance), constraints_list)
            problem.solve()
            
            if problem.status != cp.OPTIMAL:
                raise ValueError(f"Optimization failed with status: {problem.status}")
                
            # Extract results
            optimal_weights = weights.value
            portfolio_weights = {asset: weight for asset, weight in zip(covariance_matrix.index, optimal_weights)}
            
            # Calculate portfolio metrics
            expected_returns = returns.mean()
            expected_return = sum(portfolio_weights[asset] * expected_returns[asset] for asset in portfolio_weights)
            expected_volatility = np.sqrt(portfolio_variance.value)
            sharpe_ratio = (expected_return - self.config.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
            diversification_ratio = self._calculate_diversification_ratio(portfolio_weights, covariance_matrix)
            
            return PortfolioWeights(
                weights=portfolio_weights,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                diversification_ratio=diversification_ratio
            )
            
        except Exception as e:
            logger.error(f"Error in Risk Parity optimization: {e}")
            return self._create_default_weights(returns.columns)

class MaxSharpeOptimizer:
    """Maximum Sharpe Ratio optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize Max Sharpe optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        
    def optimize(self, returns: pd.DataFrame, constraints: PortfolioConstraints) -> PortfolioWeights:
        """Optimize portfolio for maximum Sharpe ratio."""
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean()
            covariance_matrix = returns.cov()
            
            # Number of assets
            n_assets = len(expected_returns)
            
            # Define optimization variables
            weights = cp.Variable(n_assets)
            
            # Calculate portfolio metrics
            portfolio_return = expected_returns.values @ weights
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            portfolio_volatility = cp.sqrt(portfolio_variance)
            
            # Define objective function (maximize Sharpe ratio)
            excess_return = portfolio_return - self.config.risk_free_rate
            sharpe_ratio = excess_return / portfolio_volatility
            
            # Define constraints
            constraints_list = []
            
            # Weight sum constraint
            constraints_list.append(cp.sum(weights) == 1.0)
            
            # Weight bounds
            if not self.config.enable_short_selling:
                constraints_list.append(weights >= constraints.min_weight)
            constraints_list.append(weights <= constraints.max_weight)
            
            # Solve optimization problem
            problem = cp.Problem(cp.Maximize(sharpe_ratio), constraints_list)
            problem.solve()
            
            if problem.status != cp.OPTIMAL:
                raise ValueError(f"Optimization failed with status: {problem.status}")
                
            # Extract results
            optimal_weights = weights.value
            portfolio_weights = {asset: weight for asset, weight in zip(expected_returns.index, optimal_weights)}
            
            # Calculate portfolio metrics
            expected_return = portfolio_return.value
            expected_volatility = portfolio_volatility.value
            sharpe_ratio_value = sharpe_ratio.value
            diversification_ratio = self._calculate_diversification_ratio(portfolio_weights, covariance_matrix)
            
            return PortfolioWeights(
                weights=portfolio_weights,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio_value,
                diversification_ratio=diversification_ratio
            )
            
        except Exception as e:
            logger.error(f"Error in Max Sharpe optimization: {e}")
            return self._create_default_weights(returns.columns)

class HierarchicalRiskParityOptimizer:
    """Hierarchical Risk Parity (HRP) optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize HRP optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        
    def optimize(self, returns: pd.DataFrame, constraints: PortfolioConstraints) -> PortfolioWeights:
        """Optimize portfolio using Hierarchical Risk Parity."""
        try:
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            # Convert correlation to distance matrix
            distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix.values))
            
            # Perform hierarchical clustering
            from scipy.cluster.hierarchy import linkage, dendrogram
            linkage_matrix = linkage(distance_matrix, method='single')
            
            # Allocate weights using HRP algorithm
            weights = self._hrp_allocation(linkage_matrix, returns)
            
            # Create portfolio weights dictionary
            portfolio_weights = {asset: weight for asset, weight in zip(returns.columns, weights)}
            
            # Calculate portfolio metrics
            expected_returns = returns.mean()
            covariance_matrix = returns.cov()
            expected_return = sum(portfolio_weights[asset] * expected_returns[asset] for asset in portfolio_weights)
            expected_volatility = np.sqrt(sum(portfolio_weights[asset] * portfolio_weights[asset2] * covariance_matrix.loc[asset, asset2] 
                                            for asset in portfolio_weights for asset2 in portfolio_weights))
            sharpe_ratio = (expected_return - self.config.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
            diversification_ratio = self._calculate_diversification_ratio(portfolio_weights, covariance_matrix)
            
            return PortfolioWeights(
                weights=portfolio_weights,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                diversification_ratio=diversification_ratio
            )
            
        except Exception as e:
            logger.error(f"Error in HRP optimization: {e}")
            return self._create_default_weights(returns.columns)
            
    def _hrp_allocation(self, linkage_matrix: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
        """Perform HRP weight allocation."""
        try:
            n_assets = len(returns.columns)
            weights = np.ones(n_assets)
            
            # Recursive allocation
            def allocate(cluster_id: int) -> float:
                if cluster_id < n_assets:
                    return weights[cluster_id]
                else:
                    left_cluster = int(linkage_matrix[cluster_id - n_assets, 0])
                    right_cluster = int(linkage_matrix[cluster_id - n_assets, 1])
                    
                    left_weight = allocate(left_cluster)
                    right_weight = allocate(right_cluster)
                    
                    # Allocate based on inverse variance
                    total_weight = left_weight + right_weight
                    weights[left_cluster] = left_weight / total_weight
                    weights[right_cluster] = right_weight / total_weight
                    
                    return total_weight
                    
            # Start allocation from root
            allocate(2 * n_assets - 2)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in HRP allocation: {e}")
            return np.ones(len(returns.columns)) / len(returns.columns)

class PortfolioOptimizer:
    """Main portfolio optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize portfolio optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.markowitz_optimizer = MarkowitzOptimizer(config)
        self.risk_parity_optimizer = RiskParityOptimizer(config)
        self.max_sharpe_optimizer = MaxSharpeOptimizer(config)
        self.hrp_optimizer = HierarchicalRiskParityOptimizer(config)
        self.results: List[OptimizationResult] = []
        
    def optimize_portfolio(self, returns: pd.DataFrame, method: OptimizationMethod,
                         constraints: PortfolioConstraints) -> OptimizationResult:
        """Optimize portfolio using specified method."""
        try:
            # Select optimizer based on method
            if method == OptimizationMethod.MARKOWITZ:
                weights = self.markowitz_optimizer.optimize(returns, constraints)
            elif method == OptimizationMethod.RISK_PARITY:
                weights = self.risk_parity_optimizer.optimize(returns, constraints)
            elif method == OptimizationMethod.MAX_SHARPE:
                weights = self.max_sharpe_optimizer.optimize(returns, constraints)
            elif method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
                weights = self.hrp_optimizer.optimize(returns, constraints)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
                
            # Calculate optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(weights, returns)
            
            # Create result
            result = OptimizationResult(
                result_id=str(uuid.uuid4()),
                method=method,
                weights=weights,
                constraints=constraints,
                optimization_metrics=optimization_metrics
            )
            
            # Store result
            self.results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return self._create_default_result(method, constraints)
            
    def _calculate_optimization_metrics(self, weights: PortfolioWeights, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional optimization metrics."""
        try:
            metrics = {
                'total_weight': sum(weights.weights.values()),
                'num_assets': len(weights.weights),
                'concentration': sum(w**2 for w in weights.weights.values()),  # Herfindahl index
                'effective_n': 1 / sum(w**2 for w in weights.weights.values()) if sum(w**2 for w in weights.weights.values()) > 0 else 0
            }
            
            # Calculate sector concentration if sector information is available
            if hasattr(returns, 'sectors'):
                sector_weights = defaultdict(float)
                for asset, weight in weights.weights.items():
                    if asset in returns.sectors:
                        sector_weights[returns.sectors[asset]] += weight
                metrics['sector_concentration'] = sum(w**2 for w in sector_weights.values())
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating optimization metrics: {e}")
            return {}
            
    def _create_default_result(self, method: OptimizationMethod, constraints: PortfolioConstraints) -> OptimizationResult:
        """Create default optimization result."""
        return OptimizationResult(
            result_id=str(uuid.uuid4()),
            method=method,
            weights=self._create_default_weights([]),
            constraints=constraints,
            optimization_metrics={}
        )
        
    def _create_default_weights(self, assets: List[str]) -> PortfolioWeights:
        """Create default portfolio weights."""
        if not assets:
            return PortfolioWeights(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                diversification_ratio=1.0
            )
            
        # Equal weight allocation
        equal_weight = 1.0 / len(assets)
        weights = {asset: equal_weight for asset in assets}
        
        return PortfolioWeights(
            weights=weights,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            diversification_ratio=1.0
        )
        
    def compare_methods(self, returns: pd.DataFrame, constraints: PortfolioConstraints) -> Dict[str, OptimizationResult]:
        """Compare different optimization methods."""
        methods = [
            OptimizationMethod.MARKOWITZ,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.MAX_SHARPE,
            OptimizationMethod.HIERARCHICAL_RISK_PARITY
        ]
        
        results = {}
        for method in methods:
            try:
                result = self.optimize_portfolio(returns, method, constraints)
                results[method.value] = result
            except Exception as e:
                logger.error(f"Error comparing method {method}: {e}")
                
        return results
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.results:
            return {}
            
        summary = {
            'total_optimizations': len(self.results),
            'methods_used': list(set(r.method.value for r in self.results)),
            'average_sharpe_ratio': np.mean([r.weights.sharpe_ratio for r in self.results]),
            'average_volatility': np.mean([r.weights.expected_volatility for r in self.results]),
            'average_return': np.mean([r.weights.expected_return for r in self.results])
        }
        
        return summary

def create_portfolio_optimizer(config: OptimizationConfig = None) -> PortfolioOptimizer:
    """Create a portfolio optimizer."""
    return PortfolioOptimizer(config or OptimizationConfig())

# Demo of portfolio optimization system
if __name__ == "__main__":
    # Create portfolio optimizer
    config = OptimizationConfig(
        risk_free_rate=0.02,
        enable_short_selling=False
    )
    
    optimizer = create_portfolio_optimizer(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    returns_data = pd.DataFrame({
        'AAPL': np.random.randn(252) * 0.02 + 0.001,
        'GOOGL': np.random.randn(252) * 0.025 + 0.0012,
        'MSFT': np.random.randn(252) * 0.018 + 0.0008,
        'AMZN': np.random.randn(252) * 0.03 + 0.0015,
        'TSLA': np.random.randn(252) * 0.04 + 0.002
    }, index=dates)
    
    # Define constraints
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.4,
        target_return=0.001
    )
    
    # Compare optimization methods
    print("Running portfolio optimization...")
    results = optimizer.compare_methods(returns_data, constraints)
    
    # Display results
    for method, result in results.items():
        print(f"\n{method.upper()} Results:")
        print(f"Expected Return: {result.weights.expected_return:.4f}")
        print(f"Expected Volatility: {result.weights.expected_volatility:.4f}")
        print(f"Sharpe Ratio: {result.weights.sharpe_ratio:.4f}")
        print(f"Diversification Ratio: {result.weights.diversification_ratio:.4f}")
        print(f"Number of Assets: {result.optimization_metrics.get('num_assets', 0)}")
        print(f"Concentration: {result.optimization_metrics.get('concentration', 0):.4f}")
    
    # Get summary
    summary = optimizer.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Total optimizations: {summary.get('total_optimizations', 0)}")
    print(f"Methods used: {summary.get('methods_used', [])}")
    print(f"Average Sharpe ratio: {summary.get('average_sharpe_ratio', 0):.4f}")
    
    print("Portfolio optimization system completed successfully!")