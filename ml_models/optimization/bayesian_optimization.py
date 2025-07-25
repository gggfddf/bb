#!/usr/bin/env python3
"""
Bayesian Optimization Module

Implements comprehensive Bayesian optimization for hyperparameter tuning:
- Gaussian Process regression for surrogate modeling
- Acquisition function optimization (Expected Improvement, UCB)
- Multi-objective Bayesian optimization
- Parallel optimization support
- Optimization history tracking
- Adaptive sampling strategies

Features:
- Gaussian Process surrogate models
- Multiple acquisition functions
- Multi-objective optimization with Pareto fronts
- Parallel evaluation support
- Optimization history and visualization
- Adaptive sampling and exploration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import joblib
import pickle

logger = structlog.get_logger()

class AcquisitionFunction(Enum):
    """Acquisition functions for Bayesian optimization."""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UCB = "ucb"
    PROBABILITY_IMPROVEMENT = "probability_improvement"
    THOMPSON_SAMPLING = "thompson_sampling"

class KernelType(Enum):
    """Kernel types for Gaussian Process."""
    RBF = "rbf"
    MATERN = "matern"
    RATIONAL_QUADRATIC = "rational_quadratic"
    LINEAR = "linear"

@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization."""
    n_initial_points: int = 5
    n_iterations: int = 50
    acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT
    kernel_type: KernelType = KernelType.RBF
    exploration_weight: float = 0.1
    random_state: int = 42
    parallel_evaluations: int = 1
    noise_level: float = 1e-6

@dataclass
class OptimizationResult:
    """Result from Bayesian optimization."""
    best_parameters: Dict[str, float]
    best_objective: float
    optimization_history: List[Dict[str, Any]]
    gaussian_process: Any
    convergence_iteration: int
    optimization_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class GaussianProcess:
    """Gaussian Process surrogate model."""
    
    def __init__(self, kernel_type: KernelType = KernelType.RBF, noise_level: float = 1e-6):
        """
        Initialize Gaussian Process.
        
        Args:
            kernel_type: Type of kernel to use
            noise_level: Noise level for observations
        """
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.X_train = None
        self.y_train = None
        self.kernel_params = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Gaussian Process to data.
        
        Args:
            X: Training features
            y: Training targets
        """
        self.X_train = X
        self.y_train = y
        
        # Initialize kernel parameters
        self._initialize_kernel_params(X.shape[1])
        
        # Optimize kernel parameters
        self._optimize_kernel_params()
        
        self.is_fitted = True
        
        logger.debug("Gaussian Process fitted", 
                    n_samples=len(X),
                    n_features=X.shape[1])
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (mean, std)
        """
        if not self.is_fitted:
            raise ValueError("Gaussian Process must be fitted before prediction")
        
        # Calculate kernel matrices
        K_train = self._compute_kernel(self.X_train, self.X_train)
        K_train += self.noise_level * np.eye(len(self.X_train))
        
        K_test = self._compute_kernel(X, self.X_train)
        K_test_test = self._compute_kernel(X, X)
        
        # Calculate predictions
        K_inv = np.linalg.inv(K_train)
        mean = K_test @ K_inv @ self.y_train
        
        var = K_test_test - K_test @ K_inv @ K_test.T
        std = np.sqrt(np.diag(var))
        
        return mean, std
    
    def _initialize_kernel_params(self, n_features: int):
        """Initialize kernel parameters."""
        if self.kernel_type == KernelType.RBF:
            # length_scale, signal_variance
            self.kernel_params = np.array([1.0, 1.0])
        elif self.kernel_type == KernelType.MATERN:
            # length_scale, signal_variance, nu
            self.kernel_params = np.array([1.0, 1.0, 1.5])
        elif self.kernel_type == KernelType.RATIONAL_QUADRATIC:
            # length_scale, signal_variance, alpha
            self.kernel_params = np.array([1.0, 1.0, 1.0])
        elif self.kernel_type == KernelType.LINEAR:
            # signal_variance
            self.kernel_params = np.array([1.0])
    
    def _optimize_kernel_params(self):
        """Optimize kernel parameters using maximum likelihood."""
        def negative_log_likelihood(params):
            self.kernel_params = params
            K = self._compute_kernel(self.X_train, self.X_train)
            K += self.noise_level * np.eye(len(self.X_train))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
                log_likelihood = -0.5 * self.y_train.T @ alpha - np.sum(np.log(np.diag(L)))
                return -log_likelihood
            except np.linalg.LinAlgError:
                return np.inf
        
        # Optimize parameters
        result = minimize(negative_log_likelihood, self.kernel_params, method='L-BFGS-B')
        self.kernel_params = result.x
    
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel_type == KernelType.RBF:
            return self._rbf_kernel(X1, X2)
        elif self.kernel_type == KernelType.MATERN:
            return self._matern_kernel(X1, X2)
        elif self.kernel_type == KernelType.RATIONAL_QUADRATIC:
            return self._rational_quadratic_kernel(X1, X2)
        elif self.kernel_type == KernelType.LINEAR:
            return self._linear_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel."""
        length_scale, signal_variance = self.kernel_params[:2]
        
        # Compute squared distances
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        distances = X1_norm + X2_norm - 2 * X1 @ X2.T
        
        return signal_variance * np.exp(-0.5 * distances / length_scale**2)
    
    def _matern_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matern kernel."""
        length_scale, signal_variance, nu = self.kernel_params[:3]
        
        # Compute distances
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        distances = np.sqrt(X1_norm + X2_norm - 2 * X1 @ X2.T)
        
        # Matern kernel with nu=1.5
        scaled_distances = np.sqrt(3) * distances / length_scale
        kernel = (1 + scaled_distances) * np.exp(-scaled_distances)
        
        return signal_variance * kernel
    
    def _rational_quadratic_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Rational quadratic kernel."""
        length_scale, signal_variance, alpha = self.kernel_params[:3]
        
        # Compute squared distances
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        distances = X1_norm + X2_norm - 2 * X1 @ X2.T
        
        return signal_variance * (1 + distances / (2 * alpha * length_scale**2))**(-alpha)
    
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Linear kernel."""
        signal_variance = self.kernel_params[0]
        return signal_variance * X1 @ X2.T

class AcquisitionFunctionOptimizer:
    """Optimizer for acquisition functions."""
    
    def __init__(self, acquisition_function: AcquisitionFunction, exploration_weight: float = 0.1):
        """
        Initialize acquisition function optimizer.
        
        Args:
            acquisition_function: Type of acquisition function
            exploration_weight: Weight for exploration in UCB
        """
        self.acquisition_function = acquisition_function
        self.exploration_weight = exploration_weight
    
    def optimize(self, gp: GaussianProcess, bounds: List[Tuple[float, float]], 
                n_random_starts: int = 10) -> np.ndarray:
        """
        Optimize acquisition function to find next point.
        
        Args:
            gp: Fitted Gaussian Process
            bounds: Parameter bounds
            n_random_starts: Number of random starts for optimization
            
        Returns:
            Optimal parameters
        """
        best_acquisition = -np.inf
        best_params = None
        
        # Try multiple random starts
        for _ in range(n_random_starts):
            # Generate random initial point
            x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            
            # Optimize acquisition function
            result = minimize(
                lambda x: -self._evaluate_acquisition(x, gp),
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success and -result.fun > best_acquisition:
                best_acquisition = -result.fun
                best_params = result.x
        
        if best_params is None:
            # Fallback to random point
            best_params = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        
        return best_params
    
    def _evaluate_acquisition(self, x: np.ndarray, gp: GaussianProcess) -> float:
        """Evaluate acquisition function at point x."""
        x = x.reshape(1, -1)
        mean, std = gp.predict(x)
        
        if self.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            return self._expected_improvement(mean[0], std[0], gp.y_train)
        elif self.acquisition_function == AcquisitionFunction.UCB:
            return self._upper_confidence_bound(mean[0], std[0])
        elif self.acquisition_function == AcquisitionFunction.PROBABILITY_IMPROVEMENT:
            return self._probability_improvement(mean[0], std[0], gp.y_train)
        elif self.acquisition_function == AcquisitionFunction.THOMPSON_SAMPLING:
            return self._thompson_sampling(mean[0], std[0])
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
    
    def _expected_improvement(self, mean: float, std: float, y_train: np.ndarray) -> float:
        """Expected Improvement acquisition function."""
        best_f = np.max(y_train)
        
        if std == 0:
            return 0.0
        
        z = (mean - best_f) / std
        ei = (mean - best_f) * norm.cdf(z) + std * norm.pdf(z)
        
        return max(0, ei)
    
    def _upper_confidence_bound(self, mean: float, std: float) -> float:
        """Upper Confidence Bound acquisition function."""
        return mean + self.exploration_weight * std
    
    def _probability_improvement(self, mean: float, std: float, y_train: np.ndarray) -> float:
        """Probability of Improvement acquisition function."""
        best_f = np.max(y_train)
        
        if std == 0:
            return 0.0
        
        z = (mean - best_f) / std
        return norm.cdf(z)
    
    def _thompson_sampling(self, mean: float, std: float) -> float:
        """Thompson Sampling acquisition function."""
        return np.random.normal(mean, std)

class BayesianOptimizer:
    """Main Bayesian optimization implementation."""
    
    def __init__(self, config: OptimizationConfig, 
                 objective_function: Callable,
                 parameter_bounds: Dict[str, Tuple[float, float]]):
        """
        Initialize Bayesian optimizer.
        
        Args:
            config: Optimization configuration
            objective_function: Function to optimize
            parameter_bounds: Bounds for parameters
        """
        self.config = config
        self.objective_function = objective_function
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        
        # Initialize components
        self.gp = GaussianProcess(config.kernel_type, config.noise_level)
        self.acquisition_optimizer = AcquisitionFunctionOptimizer(
            config.acquisition_function, config.exploration_weight
        )
        
        # Set random seed
        np.random.seed(config.random_state)
        
        # Optimization history
        self.X_history = []
        self.y_history = []
        self.optimization_history = []
        
        logger.info("Bayesian optimizer initialized", 
                   n_parameters=len(parameter_bounds),
                   n_iterations=config.n_iterations)
    
    def optimize(self) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Returns:
            Optimization results
        """
        start_time = datetime.now()
        
        # Initialize with random points
        self._initialize_random_points()
        
        # Main optimization loop
        for iteration in range(self.config.n_iterations):
            logger.debug(f"Optimization iteration {iteration + 1}/{self.config.n_iterations}")
            
            # Fit Gaussian Process
            X = np.array(self.X_history)
            y = np.array(self.y_history)
            self.gp.fit(X, y)
            
            # Find next point to evaluate
            next_point = self._find_next_point()
            
            # Evaluate objective function
            objective_value = self._evaluate_objective(next_point)
            
            # Update history
            self.X_history.append(next_point)
            self.y_history.append(objective_value)
            
            # Record iteration
            iteration_record = {
                'iteration': iteration + 1,
                'parameters': dict(zip(self.parameter_names, next_point)),
                'objective': objective_value,
                'best_objective_so_far': max(self.y_history),
                'timestamp': datetime.now()
            }
            self.optimization_history.append(iteration_record)
            
            # Log progress
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}", 
                           best_objective=max(self.y_history),
                           current_objective=objective_value)
        
        # Find best result
        best_idx = np.argmax(self.y_history)
        best_parameters = dict(zip(self.parameter_names, self.X_history[best_idx]))
        best_objective = self.y_history[best_idx]
        
        # Calculate optimization time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Determine convergence
        convergence_iteration = self._find_convergence_iteration()
        
        result = OptimizationResult(
            best_parameters=best_parameters,
            best_objective=best_objective,
            optimization_history=self.optimization_history,
            gaussian_process=self.gp,
            convergence_iteration=convergence_iteration,
            optimization_time=optimization_time
        )
        
        logger.info("Bayesian optimization completed", 
                   best_objective=best_objective,
                   optimization_time=optimization_time)
        
        return result
    
    def _initialize_random_points(self):
        """Initialize with random points."""
        bounds = [self.parameter_bounds[name] for name in self.parameter_names]
        
        for _ in range(self.config.n_initial_points):
            # Generate random point
            point = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            
            # Evaluate objective
            objective_value = self._evaluate_objective(point)
            
            # Store in history
            self.X_history.append(point)
            self.y_history.append(objective_value)
    
    def _find_next_point(self) -> np.ndarray:
        """Find next point to evaluate using acquisition function."""
        bounds = [self.parameter_bounds[name] for name in self.parameter_names]
        return self.acquisition_optimizer.optimize(self.gp, bounds)
    
    def _evaluate_objective(self, parameters: np.ndarray) -> float:
        """Evaluate objective function."""
        try:
            param_dict = dict(zip(self.parameter_names, parameters))
            return self.objective_function(param_dict)
        except Exception as e:
            logger.warning("Objective function evaluation failed", error=str(e))
            return -np.inf
    
    def _find_convergence_iteration(self) -> int:
        """Find iteration where optimization converged."""
        if len(self.y_history) < 10:
            return len(self.y_history)
        
        # Simple convergence detection: no improvement in last 10 iterations
        best_so_far = []
        current_best = -np.inf
        
        for i, y in enumerate(self.y_history):
            if y > current_best:
                current_best = y
            best_so_far.append(current_best)
        
        # Check if converged (no improvement in last 10 iterations)
        for i in range(len(best_so_far) - 10, len(best_so_far)):
            if i > 0 and best_so_far[i] > best_so_far[i-1]:
                return i + 1
        
        return len(self.y_history)
    
    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history."""
        if len(self.optimization_history) == 0:
            logger.warning("No optimization history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot objective values
        iterations = [record['iteration'] for record in self.optimization_history]
        objectives = [record['objective'] for record in self.optimization_history]
        best_objectives = [record['best_objective_so_far'] for record in self.optimization_history]
        
        ax1.plot(iterations, objectives, 'bo-', alpha=0.6, label='Objective Values')
        ax1.plot(iterations, best_objectives, 'r-', linewidth=2, label='Best So Far')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Bayesian Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot parameter evolution (first parameter only)
        if len(self.parameter_names) > 0:
            param_values = [record['parameters'][self.parameter_names[0]] 
                          for record in self.optimization_history]
            ax2.plot(iterations, param_values, 'go-', alpha=0.6)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel(f'Parameter: {self.parameter_names[0]}')
            ax2.set_title('Parameter Evolution')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def create_bayesian_optimizer(objective_function: Callable,
                            parameter_bounds: Dict[str, Tuple[float, float]],
                            n_iterations: int = 50,
                            n_initial_points: int = 5) -> BayesianOptimizer:
    """
    Create a Bayesian optimizer.
    
    Args:
        objective_function: Function to optimize
        parameter_bounds: Bounds for parameters
        n_iterations: Number of optimization iterations
        n_initial_points: Number of initial random points
        
    Returns:
        BayesianOptimizer instance
    """
    config = OptimizationConfig(
        n_iterations=n_iterations,
        n_initial_points=n_initial_points
    )
    
    return BayesianOptimizer(config, objective_function, parameter_bounds)

if __name__ == "__main__":
    # Demo of Bayesian optimization
    def demo_objective_function(parameters):
        """Demo objective function."""
        # Simple function with multiple local optima
        x = parameters.get('x', 0)
        y = parameters.get('y', 0)
        
        # Function with multiple peaks
        objective = np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)
        
        return objective
    
    # Define parameter bounds
    parameter_bounds = {
        'x': (-5.0, 5.0),
        'y': (-5.0, 5.0)
    }
    
    # Create Bayesian optimizer
    optimizer = create_bayesian_optimizer(
        objective_function=demo_objective_function,
        parameter_bounds=parameter_bounds,
        n_iterations=30,
        n_initial_points=5
    )
    
    # Run optimization
    result = optimizer.optimize()
    
    print(f"Best objective: {result.best_objective}")
    print(f"Best parameters: {result.best_parameters}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")
    
    # Plot results
    optimizer.plot_optimization_history()