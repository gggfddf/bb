#!/usr/bin/env python3
"""
Genetic Algorithm for Strategy Optimization Module

Implements a comprehensive genetic algorithm framework for optimizing trading strategies:
- Chromosome representation for strategy parameters
- Fitness functions for strategy evaluation
- Selection, crossover, and mutation operators
- Population management and evolution
- Multi-objective optimization support
- Adaptive genetic algorithm parameters

Features:
- Flexible chromosome representation
- Multiple fitness functions (Sharpe ratio, returns, drawdown)
- Various selection methods (tournament, roulette wheel, rank-based)
- Crossover operators (single-point, two-point, uniform)
- Mutation operators (Gaussian, uniform, swap)
- Population diversity maintenance
- Multi-objective optimization with Pareto fronts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import joblib
import pickle

logger = structlog.get_logger()

class SelectionMethod(Enum):
    """Selection methods for genetic algorithm."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    ELITISM = "elitism"

class CrossoverMethod(Enum):
    """Crossover methods for genetic algorithm."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"

class MutationMethod(Enum):
    """Mutation methods for genetic algorithm."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    SWAP = "swap"
    INVERSION = "inversion"

class FitnessMetric(Enum):
    """Fitness metrics for strategy evaluation."""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"

@dataclass
class Chromosome:
    """Chromosome representation for strategy parameters."""
    genes: np.ndarray
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 100
    generations: int = 50
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.SINGLE_POINT
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_rate: float = 0.1
    tournament_size: int = 3
    random_state: int = 42
    multi_objective: bool = False
    fitness_metrics: List[FitnessMetric] = None

@dataclass
class GeneticAlgorithmResult:
    """Result from genetic algorithm optimization."""
    best_chromosome: Chromosome
    best_fitness: float
    population_history: List[List[Chromosome]]
    fitness_history: List[float]
    convergence_generation: int
    final_population: List[Chromosome]
    optimization_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class StrategyParameterEncoder:
    """Encoder for strategy parameters to chromosome representation."""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        """
        Initialize parameter encoder.
        
        Args:
            parameter_bounds: Dictionary mapping parameter names to (min, max) bounds
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.n_parameters = len(parameter_bounds)
        
    def encode_parameters(self, parameters: Dict[str, float]) -> np.ndarray:
        """
        Encode parameters to chromosome.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Chromosome as numpy array
        """
        chromosome = np.zeros(self.n_parameters)
        
        for i, param_name in enumerate(self.parameter_names):
            if param_name in parameters:
                min_val, max_val = self.parameter_bounds[param_name]
                # Normalize to [0, 1]
                normalized_val = (parameters[param_name] - min_val) / (max_val - min_val)
                chromosome[i] = np.clip(normalized_val, 0, 1)
        
        return chromosome
    
    def decode_chromosome(self, chromosome: np.ndarray) -> Dict[str, float]:
        """
        Decode chromosome to parameters.
        
        Args:
            chromosome: Chromosome as numpy array
            
        Returns:
            Dictionary of parameter values
        """
        parameters = {}
        
        for i, param_name in enumerate(self.parameter_names):
            min_val, max_val = self.parameter_bounds[param_name]
            # Denormalize from [0, 1]
            normalized_val = np.clip(chromosome[i], 0, 1)
            parameters[param_name] = min_val + normalized_val * (max_val - min_val)
        
        return parameters

class FitnessFunction:
    """Fitness function for strategy evaluation."""
    
    def __init__(self, metrics: List[FitnessMetric] = None, weights: List[float] = None):
        """
        Initialize fitness function.
        
        Args:
            metrics: List of fitness metrics to use
            weights: Weights for each metric (must sum to 1)
        """
        if metrics is None:
            metrics = [FitnessMetric.SHARPE_RATIO]
        
        if weights is None:
            weights = [1.0 / len(metrics)] * len(metrics)
        
        self.metrics = metrics
        self.weights = weights
        
        # Validate weights sum to 1
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
    
    def evaluate_fitness(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Evaluate fitness of a strategy.
        
        Args:
            returns: Array of strategy returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Fitness score
        """
        if len(returns) == 0:
            return -np.inf
        
        fitness_components = []
        
        for metric in self.metrics:
            if metric == FitnessMetric.SHARPE_RATIO:
                component = self._calculate_sharpe_ratio(returns, risk_free_rate)
            elif metric == FitnessMetric.TOTAL_RETURN:
                component = self._calculate_total_return(returns)
            elif metric == FitnessMetric.MAX_DRAWDOWN:
                component = self._calculate_max_drawdown(returns)
            elif metric == FitnessMetric.CALMAR_RATIO:
                component = self._calculate_calmar_ratio(returns)
            elif metric == FitnessMetric.SORTINO_RATIO:
                component = self._calculate_sortino_ratio(returns, risk_free_rate)
            elif metric == FitnessMetric.WIN_RATE:
                component = self._calculate_win_rate(returns)
            elif metric == FitnessMetric.PROFIT_FACTOR:
                component = self._calculate_profit_factor(returns)
            else:
                component = 0.0
            
            fitness_components.append(component)
        
        # Calculate weighted fitness
        fitness = sum(w * c for w, c in zip(self.weights, fitness_components))
        
        return fitness
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return."""
        return np.prod(1 + returns) - 1
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return -np.min(drawdown)  # Return positive value
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        total_return = self._calculate_total_return(returns)
        max_dd = self._calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        
        return total_return / max_dd
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate."""
        if len(returns) == 0:
            return 0.0
        
        winning_trades = np.sum(returns > 0)
        return winning_trades / len(returns)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf if len(positive_returns) > 0 else 1.0
        
        gross_profit = np.sum(positive_returns)
        gross_loss = abs(np.sum(negative_returns))
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss

class SelectionOperator:
    """Selection operators for genetic algorithm."""
    
    def __init__(self, method: SelectionMethod, tournament_size: int = 3):
        """
        Initialize selection operator.
        
        Args:
            method: Selection method
            tournament_size: Size of tournament for tournament selection
        """
        self.method = method
        self.tournament_size = tournament_size
    
    def select(self, population: List[Chromosome], n_parents: int) -> List[Chromosome]:
        """
        Select parents from population.
        
        Args:
            population: Current population
            n_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        if self.method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population, n_parents)
        elif self.method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection(population, n_parents)
        elif self.method == SelectionMethod.RANK_BASED:
            return self._rank_based_selection(population, n_parents)
        elif self.method == SelectionMethod.ELITISM:
            return self._elitism_selection(population, n_parents)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
    
    def _tournament_selection(self, population: List[Chromosome], n_parents: int) -> List[Chromosome]:
        """Tournament selection."""
        parents = []
        
        for _ in range(n_parents):
            # Select random individuals for tournament
            tournament = random.sample(population, self.tournament_size)
            # Select best individual from tournament
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def _roulette_wheel_selection(self, population: List[Chromosome], n_parents: int) -> List[Chromosome]:
        """Roulette wheel selection."""
        # Calculate selection probabilities
        total_fitness = sum(chrom.fitness for chrom in population)
        
        if total_fitness == 0:
            # If all fitness values are 0, use uniform selection
            return random.sample(population, n_parents)
        
        probabilities = [chrom.fitness / total_fitness for chrom in population]
        
        # Select parents
        parents = []
        for _ in range(n_parents):
            selected = random.choices(population, weights=probabilities, k=1)[0]
            parents.append(selected)
        
        return parents
    
    def _rank_based_selection(self, population: List[Chromosome], n_parents: int) -> List[Chromosome]:
        """Rank-based selection."""
        # Sort population by fitness
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Calculate rank-based probabilities
        n = len(sorted_population)
        probabilities = [(2 * (n - i)) / (n * (n + 1)) for i in range(n)]
        
        # Select parents
        parents = []
        for _ in range(n_parents):
            selected = random.choices(sorted_population, weights=probabilities, k=1)[0]
            parents.append(selected)
        
        return parents
    
    def _elitism_selection(self, population: List[Chromosome], n_parents: int) -> List[Chromosome]:
        """Elitism selection (select best individuals)."""
        # Sort population by fitness and select best
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_population[:n_parents]

class CrossoverOperator:
    """Crossover operators for genetic algorithm."""
    
    def __init__(self, method: CrossoverMethod, crossover_rate: float = 0.8):
        """
        Initialize crossover operator.
        
        Args:
            method: Crossover method
            crossover_rate: Probability of crossover
        """
        self.method = method
        self.crossover_rate = crossover_rate
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return deepcopy(parent1), deepcopy(parent2)
        
        if self.method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.method}")
    
    def _single_point_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Single-point crossover."""
        n_genes = len(parent1.genes)
        crossover_point = random.randint(1, n_genes - 1)
        
        # Create offspring
        offspring1 = Chromosome(
            genes=np.concatenate([parent1.genes[:crossover_point], parent2.genes[crossover_point:]]),
            generation=parent1.generation + 1
        )
        
        offspring2 = Chromosome(
            genes=np.concatenate([parent2.genes[:crossover_point], parent1.genes[crossover_point:]]),
            generation=parent2.generation + 1
        )
        
        return offspring1, offspring2
    
    def _two_point_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Two-point crossover."""
        n_genes = len(parent1.genes)
        point1, point2 = sorted(random.sample(range(1, n_genes), 2))
        
        # Create offspring
        offspring1_genes = parent1.genes.copy()
        offspring1_genes[point1:point2] = parent2.genes[point1:point2]
        
        offspring2_genes = parent2.genes.copy()
        offspring2_genes[point1:point2] = parent1.genes[point1:point2]
        
        offspring1 = Chromosome(genes=offspring1_genes, generation=parent1.generation + 1)
        offspring2 = Chromosome(genes=offspring2_genes, generation=parent2.generation + 1)
        
        return offspring1, offspring2
    
    def _uniform_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Uniform crossover."""
        n_genes = len(parent1.genes)
        mask = np.random.random(n_genes) < 0.5
        
        offspring1_genes = np.where(mask, parent1.genes, parent2.genes)
        offspring2_genes = np.where(mask, parent2.genes, parent1.genes)
        
        offspring1 = Chromosome(genes=offspring1_genes, generation=parent1.generation + 1)
        offspring2 = Chromosome(genes=offspring2_genes, generation=parent2.generation + 1)
        
        return offspring1, offspring2
    
    def _arithmetic_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Arithmetic crossover."""
        alpha = random.random()
        
        offspring1_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        offspring2_genes = alpha * parent2.genes + (1 - alpha) * parent1.genes
        
        offspring1 = Chromosome(genes=offspring1_genes, generation=parent1.generation + 1)
        offspring2 = Chromosome(genes=offspring2_genes, generation=parent2.generation + 1)
        
        return offspring1, offspring2

class MutationOperator:
    """Mutation operators for genetic algorithm."""
    
    def __init__(self, method: MutationMethod, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """
        Initialize mutation operator.
        
        Args:
            method: Mutation method
            mutation_rate: Probability of mutation per gene
            mutation_strength: Strength of mutation (for Gaussian)
        """
        self.method = method
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutate a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated_genes = chromosome.genes.copy()
        
        if self.method == MutationMethod.GAUSSIAN:
            mutated_genes = self._gaussian_mutation(mutated_genes)
        elif self.method == MutationMethod.UNIFORM:
            mutated_genes = self._uniform_mutation(mutated_genes)
        elif self.method == MutationMethod.SWAP:
            mutated_genes = self._swap_mutation(mutated_genes)
        elif self.method == MutationMethod.INVERSION:
            mutated_genes = self._inversion_mutation(mutated_genes)
        else:
            raise ValueError(f"Unknown mutation method: {self.method}")
        
        # Ensure genes are within [0, 1] bounds
        mutated_genes = np.clip(mutated_genes, 0, 1)
        
        return Chromosome(
            genes=mutated_genes,
            generation=chromosome.generation,
            age=chromosome.age + 1
        )
    
    def _gaussian_mutation(self, genes: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mask = np.random.random(len(genes)) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_strength, len(genes))
        genes[mask] += noise[mask]
        return genes
    
    def _uniform_mutation(self, genes: np.ndarray) -> np.ndarray:
        """Uniform mutation."""
        mask = np.random.random(len(genes)) < self.mutation_rate
        genes[mask] = np.random.random(np.sum(mask))
        return genes
    
    def _swap_mutation(self, genes: np.ndarray) -> np.ndarray:
        """Swap mutation."""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(genes)), 2)
            genes[i], genes[j] = genes[j], genes[i]
        return genes
    
    def _inversion_mutation(self, genes: np.ndarray) -> np.ndarray:
        """Inversion mutation."""
        if random.random() < self.mutation_rate:
            i, j = sorted(random.sample(range(len(genes)), 2))
            genes[i:j+1] = genes[i:j+1][::-1]
        return genes

class GeneticAlgorithm:
    """Main genetic algorithm implementation."""
    
    def __init__(self, config: GeneticAlgorithmConfig, 
                 parameter_encoder: StrategyParameterEncoder,
                 fitness_function: FitnessFunction,
                 strategy_evaluator: Callable):
        """
        Initialize genetic algorithm.
        
        Args:
            config: Genetic algorithm configuration
            parameter_encoder: Parameter encoder
            fitness_function: Fitness function
            strategy_evaluator: Function to evaluate strategy with parameters
        """
        self.config = config
        self.parameter_encoder = parameter_encoder
        self.fitness_function = fitness_function
        self.strategy_evaluator = strategy_evaluator
        
        # Initialize operators
        self.selection_operator = SelectionOperator(config.selection_method, config.tournament_size)
        self.crossover_operator = CrossoverOperator(config.crossover_method, config.crossover_rate)
        self.mutation_operator = MutationOperator(config.mutation_method, config.mutation_rate)
        
        # Set random seed
        random.seed(config.random_state)
        np.random.seed(config.random_state)
        
        # Population
        self.population = []
        self.generation = 0
        
        logger.info("Genetic algorithm initialized", 
                   population_size=config.population_size,
                   generations=config.generations)
    
    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.config.population_size):
            # Generate random chromosome
            genes = np.random.random(self.parameter_encoder.n_parameters)
            chromosome = Chromosome(genes=genes, generation=0)
            self.population.append(chromosome)
        
        logger.info("Population initialized", size=len(self.population))
    
    def evaluate_population(self):
        """Evaluate fitness of all individuals in population."""
        for chromosome in self.population:
            try:
                # Decode parameters
                parameters = self.parameter_encoder.decode_chromosome(chromosome.genes)
                
                # Evaluate strategy
                returns = self.strategy_evaluator(parameters)
                
                # Calculate fitness
                chromosome.fitness = self.fitness_function.evaluate_fitness(returns)
                
            except Exception as e:
                logger.warning("Strategy evaluation failed", error=str(e))
                chromosome.fitness = -np.inf
    
    def evolve(self) -> GeneticAlgorithmResult:
        """
        Run genetic algorithm evolution.
        
        Returns:
            Optimization results
        """
        start_time = datetime.now()
        
        # Initialize population
        self.initialize_population()
        
        # Track history
        population_history = []
        fitness_history = []
        best_fitness = -np.inf
        convergence_generation = 0
        
        logger.info("Starting genetic algorithm evolution")
        
        for generation in range(self.config.generations):
            self.generation = generation
            
            # Evaluate population
            self.evaluate_population()
            
            # Track best fitness
            current_best_fitness = max(chrom.fitness for chrom in self.population)
            fitness_history.append(current_best_fitness)
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                convergence_generation = generation
            
            # Store population history
            population_history.append(deepcopy(self.population))
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}/{self.config.generations}", 
                           best_fitness=best_fitness,
                           avg_fitness=np.mean([chrom.fitness for chrom in self.population]))
            
            # Check for convergence
            if generation > 20 and len(set(fitness_history[-20:])) == 1:
                logger.info("Convergence detected, stopping early")
                break
            
            # Create next generation
            self._create_next_generation()
        
        # Find best chromosome
        best_chromosome = max(self.population, key=lambda x: x.fitness)
        
        # Calculate optimization time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = GeneticAlgorithmResult(
            best_chromosome=best_chromosome,
            best_fitness=best_fitness,
            population_history=population_history,
            fitness_history=fitness_history,
            convergence_generation=convergence_generation,
            final_population=self.population,
            optimization_time=optimization_time
        )
        
        logger.info("Genetic algorithm completed", 
                   best_fitness=best_fitness,
                   optimization_time=optimization_time)
        
        return result
    
    def _create_next_generation(self):
        """Create next generation of individuals."""
        new_population = []
        
        # Elitism: keep best individuals
        n_elite = int(self.config.elitism_rate * self.config.population_size)
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n_elite]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Select parents
            parents = self.selection_operator.select(self.population, 2)
            
            # Crossover
            offspring1, offspring2 = self.crossover_operator.crossover(parents[0], parents[1])
            
            # Mutate
            offspring1 = self.mutation_operator.mutate(offspring1)
            offspring2 = self.mutation_operator.mutate(offspring2)
            
            # Add to new population
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        self.population = new_population[:self.config.population_size]

def create_genetic_algorithm(parameter_bounds: Dict[str, Tuple[float, float]],
                           strategy_evaluator: Callable,
                           population_size: int = 100,
                           generations: int = 50) -> GeneticAlgorithm:
    """
    Create a genetic algorithm for strategy optimization.
    
    Args:
        parameter_bounds: Dictionary mapping parameter names to (min, max) bounds
        strategy_evaluator: Function that evaluates strategy and returns returns
        population_size: Size of population
        generations: Number of generations
        
    Returns:
        GeneticAlgorithm instance
    """
    # Create components
    parameter_encoder = StrategyParameterEncoder(parameter_bounds)
    fitness_function = FitnessFunction()
    
    # Create configuration
    config = GeneticAlgorithmConfig(
        population_size=population_size,
        generations=generations
    )
    
    return GeneticAlgorithm(config, parameter_encoder, fitness_function, strategy_evaluator)

if __name__ == "__main__":
    # Demo of genetic algorithm
    def demo_strategy_evaluator(parameters):
        """Demo strategy evaluator that returns random returns."""
        # Simulate strategy returns based on parameters
        n_days = 252
        base_return = 0.0001
        volatility = 0.02
        
        # Use parameters to influence returns
        param_sum = sum(parameters.values())
        returns = np.random.normal(base_return * param_sum, volatility, n_days)
        
        return returns
    
    # Define parameter bounds
    parameter_bounds = {
        'rsi_period': (10, 30),
        'rsi_overbought': (60, 80),
        'rsi_oversold': (20, 40),
        'stop_loss': (0.01, 0.05),
        'take_profit': (0.02, 0.10)
    }
    
    # Create genetic algorithm
    ga = create_genetic_algorithm(
        parameter_bounds=parameter_bounds,
        strategy_evaluator=demo_strategy_evaluator,
        population_size=50,
        generations=20
    )
    
    # Run optimization
    result = ga.evolve()
    
    print(f"Best fitness: {result.best_fitness}")
    print(f"Best parameters: {ga.parameter_encoder.decode_chromosome(result.best_chromosome.genes)}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")