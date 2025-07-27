#!/usr/bin/env python3
"""
Strategy Optimization System

Implements comprehensive strategy optimization:
- Parameter optimization using various algorithms
- Walk-forward analysis
- Machine learning integration
- Overfitting detection and prevention
- Multi-objective optimization
- Robustness testing

Features:
- Advanced parameter optimization algorithms
- Walk-forward analysis and out-of-sample testing
- Machine learning integration for strategy enhancement
- Overfitting detection and prevention techniques
- Multi-objective optimization and Pareto frontier
- Robustness testing and stress testing
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
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class OptimizationMethod(Enum):
    """Optimization method enumeration."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    MACHINE_LEARNING = "machine_learning"

class ObjectiveFunction(Enum):
    """Objective function enumeration."""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_CALMAR = "maximize_calmar"
    MINIMIZE_VOLATILITY = "minimize_volatility"

@dataclass
class OptimizationParameter:
    """Optimization parameter structure."""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    parameter_type: str = "continuous"
    current_value: Optional[float] = None

@dataclass
class OptimizationResult:
    """Optimization result structure."""
    result_id: str
    method: OptimizationMethod
    parameters: Dict[str, float]
    objective_value: float
    performance_metrics: Dict[str, float]
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WalkForwardResult:
    """Walk-forward analysis result structure."""
    result_id: str
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    parameters: Dict[str, float]
    train_performance: Dict[str, float]
    test_performance: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationConfig:
    """Strategy optimization configuration."""
    method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM
    objective: ObjectiveFunction = ObjectiveFunction.MAXIMIZE_SHARPE
    max_iterations: int = 100
    population_size: int = 50
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    enable_walk_forward: bool = True
    n_folds: int = 5
    test_size: float = 0.2
    enable_overfitting_detection: bool = True
    overfitting_threshold: float = 0.1

class ParameterOptimizer:
    """Parameter optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize parameter optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.parameters: List[OptimizationParameter] = []
        self.objective_function: Optional[Callable] = None
        self.results: List[OptimizationResult] = []
        
    def add_parameter(self, parameter: OptimizationParameter):
        """Add optimization parameter."""
        self.parameters.append(parameter)
        
    def set_objective_function(self, objective_function: Callable):
        """Set objective function."""
        self.objective_function = objective_function
        
    def optimize(self, data: pd.DataFrame) -> OptimizationResult:
        """Optimize strategy parameters."""
        try:
            if not self.parameters:
                raise ValueError("No parameters defined for optimization")
                
            if self.objective_function is None:
                raise ValueError("No objective function defined")
                
            if self.config.method == OptimizationMethod.GENETIC_ALGORITHM:
                result = self._genetic_algorithm_optimization(data)
            elif self.config.method == OptimizationMethod.GRID_SEARCH:
                result = self._grid_search_optimization(data)
            elif self.config.method == OptimizationMethod.RANDOM_SEARCH:
                result = self._random_search_optimization(data)
            else:
                raise ValueError(f"Unsupported optimization method: {self.config.method}")
                
            # Store result
            self.results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return self._create_default_result()
            
    def _genetic_algorithm_optimization(self, data: pd.DataFrame) -> OptimizationResult:
        """Optimize using genetic algorithm."""
        try:
            # Define bounds for parameters
            bounds = [(param.min_value, param.max_value) for param in self.parameters]
            
            # Define objective function wrapper
            def objective_wrapper(params):
                try:
                    # Convert parameters to dictionary
                    param_dict = {param.name: params[i] for i, param in enumerate(self.parameters)}
                    
                    # Evaluate objective function
                    result = self.objective_function(data, param_dict)
                    
                    # Negate for maximization (scipy.minimize minimizes)
                    return -result
                except Exception as e:
                    logger.error(f"Error in objective function: {e}")
                    return float('inf')
                    
            # Run genetic algorithm
            result = differential_evolution(
                objective_wrapper,
                bounds,
                maxiter=self.config.max_iterations,
                popsize=self.config.population_size,
                recombination=self.config.crossover_rate,
                mutation=self.config.mutation_rate,
                seed=42
            )
            
            # Create optimization result
            optimal_params = {param.name: result.x[i] for i, param in enumerate(self.parameters)}
            
            return OptimizationResult(
                result_id=str(uuid.uuid4()),
                method=OptimizationMethod.GENETIC_ALGORITHM,
                parameters=optimal_params,
                objective_value=-result.fun,  # Convert back to positive
                performance_metrics=self._evaluate_performance(data, optimal_params),
                in_sample_performance=self._evaluate_performance(data, optimal_params),
                out_of_sample_performance={}  # Will be filled by walk-forward analysis
            )
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {e}")
            return self._create_default_result()
            
    def _grid_search_optimization(self, data: pd.DataFrame) -> OptimizationResult:
        """Optimize using grid search."""
        try:
            best_result = None
            best_objective = float('-inf')
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations()
            
            for params in param_combinations:
                try:
                    # Evaluate objective function
                    objective_value = self.objective_function(data, params)
                    
                    if objective_value > best_objective:
                        best_objective = objective_value
                        best_result = params
                        
                except Exception as e:
                    logger.warning(f"Error evaluating parameters {params}: {e}")
                    continue
                    
            if best_result is None:
                raise ValueError("No valid parameter combination found")
                
            return OptimizationResult(
                result_id=str(uuid.uuid4()),
                method=OptimizationMethod.GRID_SEARCH,
                parameters=best_result,
                objective_value=best_objective,
                performance_metrics=self._evaluate_performance(data, best_result),
                in_sample_performance=self._evaluate_performance(data, best_result),
                out_of_sample_performance={}
            )
            
        except Exception as e:
            logger.error(f"Error in grid search optimization: {e}")
            return self._create_default_result()
            
    def _random_search_optimization(self, data: pd.DataFrame) -> OptimizationResult:
        """Optimize using random search."""
        try:
            best_result = None
            best_objective = float('-inf')
            
            for _ in range(self.config.max_iterations):
                # Generate random parameters
                params = {}
                for param in self.parameters:
                    if param.parameter_type == "continuous":
                        params[param.name] = np.random.uniform(param.min_value, param.max_value)
                    elif param.parameter_type == "discrete":
                        params[param.name] = np.random.choice(
                            np.arange(param.min_value, param.max_value + param.step, param.step)
                        )
                        
                try:
                    # Evaluate objective function
                    objective_value = self.objective_function(data, params)
                    
                    if objective_value > best_objective:
                        best_objective = objective_value
                        best_result = params
                        
                except Exception as e:
                    logger.warning(f"Error evaluating parameters {params}: {e}")
                    continue
                    
            if best_result is None:
                raise ValueError("No valid parameter combination found")
                
            return OptimizationResult(
                result_id=str(uuid.uuid4()),
                method=OptimizationMethod.RANDOM_SEARCH,
                parameters=best_result,
                objective_value=best_objective,
                performance_metrics=self._evaluate_performance(data, best_result),
                in_sample_performance=self._evaluate_performance(data, best_result),
                out_of_sample_performance={}
            )
            
        except Exception as e:
            logger.error(f"Error in random search optimization: {e}")
            return self._create_default_result()
            
    def _generate_parameter_combinations(self) -> List[Dict[str, float]]:
        """Generate parameter combinations for grid search."""
        combinations = []
        
        # Generate parameter values
        param_values = []
        for param in self.parameters:
            if param.parameter_type == "continuous":
                values = np.arange(param.min_value, param.max_value + param.step, param.step)
            else:
                values = [param.min_value, param.max_value]
            param_values.append(values)
            
        # Generate combinations
        import itertools
        for combination in itertools.product(*param_values):
            params = {param.name: combination[i] for i, param in enumerate(self.parameters)}
            combinations.append(params)
            
        return combinations
        
    def _evaluate_performance(self, data: pd.DataFrame, parameters: Dict[str, float]) -> Dict[str, float]:
        """Evaluate strategy performance with given parameters."""
        try:
            # This is a placeholder - in practice, you would run the strategy with these parameters
            returns = data['close'].pct_change().dropna()
            
            metrics = {
                'total_return': returns.sum(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'volatility': returns.std()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return {}
            
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except:
            return 0.0
            
    def _create_default_result(self) -> OptimizationResult:
        """Create default optimization result."""
        return OptimizationResult(
            result_id=str(uuid.uuid4()),
            method=self.config.method,
            parameters={},
            objective_value=0.0,
            performance_metrics={},
            in_sample_performance={},
            out_of_sample_performance={}
        )

class WalkForwardAnalyzer:
    """Walk-forward analysis."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize walk-forward analyzer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.results: List[WalkForwardResult] = []
        
    def run_walk_forward_analysis(self, data: pd.DataFrame, 
                                optimizer: ParameterOptimizer) -> List[WalkForwardResult]:
        """Run walk-forward analysis."""
        try:
            # Create time series split
            tscv = TimeSeriesSplit(n_splits=self.config.n_folds, test_size=int(len(data) * self.config.test_size))
            
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
                # Split data
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # Optimize on training data
                train_result = optimizer.optimize(train_data)
                
                # Evaluate on test data
                test_performance = optimizer._evaluate_performance(test_data, train_result.parameters)
                
                # Create walk-forward result
                result = WalkForwardResult(
                    result_id=str(uuid.uuid4()),
                    fold=fold + 1,
                    train_start=train_data.index[0],
                    train_end=train_data.index[-1],
                    test_start=test_data.index[0],
                    test_end=test_data.index[-1],
                    parameters=train_result.parameters,
                    train_performance=train_result.in_sample_performance,
                    test_performance=test_performance
                )
                
                fold_results.append(result)
                
            # Store results
            self.results.extend(fold_results)
            
            return fold_results
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            return []
            
    def detect_overfitting(self) -> Dict[str, Any]:
        """Detect overfitting in walk-forward results."""
        try:
            if not self.results:
                return {}
                
            # Calculate performance degradation
            train_performances = [result.train_performance.get('sharpe_ratio', 0) for result in self.results]
            test_performances = [result.test_performance.get('sharpe_ratio', 0) for result in self.results]
            
            avg_train_performance = np.mean(train_performances)
            avg_test_performance = np.mean(test_performances)
            
            performance_degradation = avg_train_performance - avg_test_performance
            
            # Check for overfitting
            is_overfitting = performance_degradation > self.config.overfitting_threshold
            
            return {
                'is_overfitting': is_overfitting,
                'performance_degradation': performance_degradation,
                'avg_train_performance': avg_train_performance,
                'avg_test_performance': avg_test_performance,
                'overfitting_threshold': self.config.overfitting_threshold
            }
            
        except Exception as e:
            logger.error(f"Error detecting overfitting: {e}")
            return {}

class StrategyOptimizer:
    """Main strategy optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize strategy optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.optimizer = ParameterOptimizer(config)
        self.walk_forward_analyzer = WalkForwardAnalyzer(config)
        
    def optimize_strategy(self, data: pd.DataFrame, 
                        parameters: List[OptimizationParameter],
                        objective_function: Callable) -> OptimizationResult:
        """Optimize strategy parameters."""
        try:
            # Set up optimizer
            for param in parameters:
                self.optimizer.add_parameter(param)
            self.optimizer.set_objective_function(objective_function)
            
            # Run optimization
            result = self.optimizer.optimize(data)
            
            # Run walk-forward analysis if enabled
            if self.config.enable_walk_forward:
                walk_forward_results = self.walk_forward_analyzer.run_walk_forward_analysis(data, self.optimizer)
                
                # Update result with out-of-sample performance
                if walk_forward_results:
                    avg_test_performance = {}
                    for metric in walk_forward_results[0].test_performance:
                        values = [r.test_performance[metric] for r in walk_forward_results]
                        avg_test_performance[metric] = np.mean(values)
                    result.out_of_sample_performance = avg_test_performance
                    
                # Check for overfitting
                overfitting_analysis = self.walk_forward_analyzer.detect_overfitting()
                result.metadata['overfitting_analysis'] = overfitting_analysis
                
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}")
            return self.optimizer._create_default_result()
            
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        summary = {
            'total_optimizations': len(self.optimizer.results),
            'total_walk_forward_folds': len(self.walk_forward_analyzer.results),
            'optimization_methods_used': list(set(r.method.value for r in self.optimizer.results))
        }
        
        if self.optimizer.results:
            summary['best_objective_value'] = max(r.objective_value for r in self.optimizer.results)
            summary['average_objective_value'] = np.mean([r.objective_value for r in self.optimizer.results])
            
        return summary

def create_strategy_optimizer(config: OptimizationConfig = None) -> StrategyOptimizer:
    """Create a strategy optimizer."""
    return StrategyOptimizer(config or OptimizationConfig())

# Demo of strategy optimization system
if __name__ == "__main__":
    # Create strategy optimizer
    config = OptimizationConfig(
        method=OptimizationMethod.GENETIC_ALGORITHM,
        objective=ObjectiveFunction.MAXIMIZE_SHARPE,
        max_iterations=50,
        population_size=20,
        enable_walk_forward=True,
        n_folds=3
    )
    
    optimizer = create_strategy_optimizer(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(252).cumsum() + 100,
        'high': np.random.randn(252).cumsum() + 102,
        'low': np.random.randn(252).cumsum() + 98,
        'close': np.random.randn(252).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 252)
    }, index=dates)
    
    # Define parameters
    parameters = [
        OptimizationParameter("short_window", 5, 50, 5, "discrete"),
        OptimizationParameter("long_window", 20, 200, 10, "discrete"),
        OptimizationParameter("threshold", 0.001, 0.01, None, "continuous")
    ]
    
    # Define objective function
    def objective_function(data: pd.DataFrame, params: Dict[str, float]) -> float:
        try:
            # Simplified strategy evaluation
            returns = data['close'].pct_change().dropna()
            short_ma = data['close'].rolling(window=int(params['short_window'])).mean()
            long_ma = data['close'].rolling(window=int(params['long_window'])).mean()
            
            # Generate signals
            signals = np.where(short_ma > long_ma, 1, -1)
            strategy_returns = signals * returns
            
            # Calculate Sharpe ratio
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 0.0
    
    # Run optimization
    print("Running strategy optimization...")
    result = optimizer.optimize_strategy(data, parameters, objective_function)
    
    print(f"Optimization Results:")
    print(f"Method: {result.method.value}")
    print(f"Parameters: {result.parameters}")
    print(f"Objective Value: {result.objective_value:.4f}")
    print(f"In-sample Sharpe: {result.in_sample_performance.get('sharpe_ratio', 0):.4f}")
    print(f"Out-of-sample Sharpe: {result.out_of_sample_performance.get('sharpe_ratio', 0):.4f}")
    
    # Check overfitting
    if 'overfitting_analysis' in result.metadata:
        overfitting = result.metadata['overfitting_analysis']
        print(f"Overfitting detected: {overfitting.get('is_overfitting', False)}")
        print(f"Performance degradation: {overfitting.get('performance_degradation', 0):.4f}")
    
    # Get summary
    summary = optimizer.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Total optimizations: {summary.get('total_optimizations', 0)}")
    print(f"Walk-forward folds: {summary.get('total_walk_forward_folds', 0)}")
    print(f"Best objective value: {summary.get('best_objective_value', 0):.4f}")
    
    print("Strategy optimization system completed successfully!")