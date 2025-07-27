#!/usr/bin/env python3
"""
Statistical Significance Testing Module

Implements comprehensive statistical testing for trading strategies:
- Hypothesis testing (t-tests, chi-square tests, ANOVA)
- Monte Carlo simulation for strategy validation
- Bootstrap resampling for confidence intervals
- Permutation tests for strategy comparison
- Statistical power analysis
- Multiple testing correction methods

Features:
- Robust statistical testing framework
- Monte Carlo simulation with customizable parameters
- Bootstrap resampling for non-parametric confidence intervals
- Permutation tests for strategy comparison
- Power analysis for sample size determination
- Multiple testing correction (Bonferroni, FDR, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, mannwhitneyu
from sklearn.utils import resample
from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

class TestType(Enum):
    """Types of statistical tests."""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    PERMUTATION = "permutation"

class CorrectionMethod(Enum):
    """Multiple testing correction methods."""
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    FDR = "fdr"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"

@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_type: TestType
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    alpha: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""
    n_simulations: int
    mean_return: float
    std_return: float
    p_value: float
    confidence_interval: Tuple[float, float]
    empirical_distribution: np.ndarray
    observed_statistic: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BootstrapResult:
    """Result of bootstrap resampling."""
    n_bootstrap: int
    mean_estimate: float
    std_estimate: float
    confidence_interval: Tuple[float, float]
    bias: float
    bootstrap_distribution: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PowerAnalysisResult:
    """Result of statistical power analysis."""
    effect_size: float
    alpha: float
    power: float
    sample_size: int
    alternative: str
    test_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class StatisticalTester:
    """
    Main class for statistical significance testing.
    """
    
    def __init__(self, 
                 default_alpha: float = 0.05,
                 n_monte_carlo: int = 10000,
                 n_bootstrap: int = 1000,
                 random_state: int = 42):
        """
        Initialize statistical tester.
        
        Args:
            default_alpha: Default significance level
            n_monte_carlo: Number of Monte Carlo simulations
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.default_alpha = default_alpha
        self.n_monte_carlo = n_monte_carlo
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        
        # Storage for results
        self.test_results: List[StatisticalTestResult] = []
        self.monte_carlo_results: List[MonteCarloResult] = []
        self.bootstrap_results: List[BootstrapResult] = []
        self.power_results: List[PowerAnalysisResult] = []
    
    def run_hypothesis_test(self, 
                           data1: np.ndarray,
                           data2: np.ndarray = None,
                           test_type: TestType = TestType.T_TEST,
                           alpha: float = None,
                           alternative: str = 'two-sided') -> StatisticalTestResult:
        """
        Run hypothesis test between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset (optional for one-sample tests)
            test_type: Type of statistical test
            alpha: Significance level
            alternative: Alternative hypothesis ('two-sided', 'greater', 'less')
            
        Returns:
            StatisticalTestResult with test results
        """
        try:
            alpha = alpha or self.default_alpha
            logger.info("Running hypothesis test", test_type=test_type.value, alpha=alpha)
            
            if test_type == TestType.T_TEST:
                result = self._run_t_test(data1, data2, alpha, alternative)
            elif test_type == TestType.CHI_SQUARE:
                result = self._run_chi_square_test(data1, data2, alpha)
            elif test_type == TestType.ANOVA:
                result = self._run_anova_test(data1, data2, alpha)
            elif test_type == TestType.MANN_WHITNEY:
                result = self._run_mann_whitney_test(data1, data2, alpha, alternative)
            elif test_type == TestType.WILCOXON:
                result = self._run_wilcoxon_test(data1, data2, alpha, alternative)
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
            
            # Store result
            self.test_results.append(result)
            
            logger.info("Hypothesis test completed",
                       test_type=test_type.value,
                       p_value=result.p_value,
                       is_significant=result.is_significant)
            
            return result
            
        except Exception as e:
            logger.error("Hypothesis test failed", error=str(e))
            raise
    
    def _run_t_test(self, 
                   data1: np.ndarray,
                   data2: np.ndarray,
                   alpha: float,
                   alternative: str) -> StatisticalTestResult:
        """Run t-test between two datasets."""
        # Perform t-test
        statistic, p_value = ttest_ind(data1, data2, alternative=alternative)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                             (len(data2) - 1) * np.var(data2, ddof=1)) / 
                            (len(data1) + len(data2) - 2))
        effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        # Calculate confidence interval
        diff = np.mean(data1) - np.mean(data2)
        se = pooled_std * np.sqrt(1/len(data1) + 1/len(data2))
        ci_lower = diff - stats.t.ppf(1 - alpha/2, len(data1) + len(data2) - 2) * se
        ci_upper = diff + stats.t.ppf(1 - alpha/2, len(data1) + len(data2) - 2) * se
        
        return StatisticalTestResult(
            test_type=TestType.T_TEST,
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < alpha,
            alpha=alpha,
            metadata={'alternative': alternative}
        )
    
    def _run_chi_square_test(self, 
                           data1: np.ndarray,
                           data2: np.ndarray,
                           alpha: float) -> StatisticalTestResult:
        """Run chi-square test between two datasets."""
        # Create contingency table
        contingency_table = np.array([data1, data2])
        
        # Perform chi-square test
        statistic, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate effect size (Cramer's V)
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape) - 1
        effect_size = np.sqrt(statistic / (n * min_dim))
        
        return StatisticalTestResult(
            test_type=TestType.CHI_SQUARE,
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(0, 1),  # Chi-square doesn't have direct CI
            is_significant=p_value < alpha,
            alpha=alpha,
            metadata={'dof': dof}
        )
    
    def _run_anova_test(self, 
                       data1: np.ndarray,
                       data2: np.ndarray,
                       alpha: float) -> StatisticalTestResult:
        """Run ANOVA test between two datasets."""
        # Perform ANOVA
        statistic, p_value = f_oneway(data1, data2)
        
        # Calculate effect size (eta-squared)
        ss_between = len(data1) * (np.mean(data1) - np.mean(np.concatenate([data1, data2]))) ** 2 + \
                    len(data2) * (np.mean(data2) - np.mean(np.concatenate([data1, data2]))) ** 2
        ss_total = np.sum((np.concatenate([data1, data2]) - np.mean(np.concatenate([data1, data2]))) ** 2)
        effect_size = ss_between / ss_total
        
        return StatisticalTestResult(
            test_type=TestType.ANOVA,
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(0, 1),  # ANOVA doesn't have direct CI
            is_significant=p_value < alpha,
            alpha=alpha
        )
    
    def _run_mann_whitney_test(self, 
                              data1: np.ndarray,
                              data2: np.ndarray,
                              alpha: float,
                              alternative: str) -> StatisticalTestResult:
        """Run Mann-Whitney U test between two datasets."""
        # Perform Mann-Whitney U test
        statistic, p_value = mannwhitneyu(data1, data2, alternative=alternative)
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(data1), len(data2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return StatisticalTestResult(
            test_type=TestType.MANN_WHITNEY,
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(0, 1),  # Non-parametric test
            is_significant=p_value < alpha,
            alpha=alpha,
            metadata={'alternative': alternative}
        )
    
    def _run_wilcoxon_test(self, 
                          data1: np.ndarray,
                          data2: np.ndarray,
                          alpha: float,
                          alternative: str) -> StatisticalTestResult:
        """Run Wilcoxon signed-rank test between two datasets."""
        # Perform Wilcoxon test
        statistic, p_value = stats.wilcoxon(data1, data2, alternative=alternative)
        
        # Calculate effect size (rank-biserial correlation)
        n = len(data1)
        effect_size = abs(statistic) / (n * (n + 1) / 2)
        
        return StatisticalTestResult(
            test_type=TestType.WILCOXON,
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(0, 1),  # Non-parametric test
            is_significant=p_value < alpha,
            alpha=alpha,
            metadata={'alternative': alternative}
        )
    
    def run_monte_carlo_simulation(self, 
                                 observed_statistic: float,
                                 null_distribution: np.ndarray,
                                 n_simulations: int = None) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for strategy validation.
        
        Args:
            observed_statistic: Observed test statistic
            null_distribution: Null distribution for comparison
            n_simulations: Number of simulations
            
        Returns:
            MonteCarloResult with simulation results
        """
        try:
            n_simulations = n_simulations or self.n_monte_carlo
            logger.info("Starting Monte Carlo simulation", n_simulations=n_simulations)
            
            # Generate empirical distribution
            empirical_dist = np.random.choice(null_distribution, size=n_simulations, replace=True)
            
            # Calculate p-value
            p_value = np.mean(empirical_dist >= observed_statistic)
            
            # Calculate confidence interval
            ci_lower = np.percentile(empirical_dist, 2.5)
            ci_upper = np.percentile(empirical_dist, 97.5)
            
            result = MonteCarloResult(
                n_simulations=n_simulations,
                mean_return=np.mean(empirical_dist),
                std_return=np.std(empirical_dist),
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                empirical_distribution=empirical_dist,
                observed_statistic=observed_statistic
            )
            
            # Store result
            self.monte_carlo_results.append(result)
            
            logger.info("Monte Carlo simulation completed",
                       p_value=p_value,
                       mean_return=result.mean_return)
            
            return result
            
        except Exception as e:
            logger.error("Monte Carlo simulation failed", error=str(e))
            raise
    
    def run_bootstrap_resampling(self, 
                               data: np.ndarray,
                               statistic_function: Callable,
                               n_bootstrap: int = None,
                               confidence_level: float = 0.95) -> BootstrapResult:
        """
        Run bootstrap resampling for confidence intervals.
        
        Args:
            data: Input data
            statistic_function: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            BootstrapResult with bootstrap results
        """
        try:
            n_bootstrap = n_bootstrap or self.n_bootstrap
            logger.info("Starting bootstrap resampling", n_bootstrap=n_bootstrap)
            
            # Calculate observed statistic
            observed_statistic = statistic_function(data)
            
            # Generate bootstrap samples
            bootstrap_stats = []
            for _ in range(n_bootstrap):
                bootstrap_sample = resample(data, replace=True, n_samples=len(data))
                bootstrap_stat = statistic_function(bootstrap_sample)
                bootstrap_stats.append(bootstrap_stat)
            
            bootstrap_stats = np.array(bootstrap_stats)
            
            # Calculate bootstrap statistics
            mean_estimate = np.mean(bootstrap_stats)
            std_estimate = np.std(bootstrap_stats)
            bias = mean_estimate - observed_statistic
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
            ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
            
            result = BootstrapResult(
                n_bootstrap=n_bootstrap,
                mean_estimate=mean_estimate,
                std_estimate=std_estimate,
                confidence_interval=(ci_lower, ci_upper),
                bias=bias,
                bootstrap_distribution=bootstrap_stats
            )
            
            # Store result
            self.bootstrap_results.append(result)
            
            logger.info("Bootstrap resampling completed",
                       mean_estimate=mean_estimate,
                       bias=bias)
            
            return result
            
        except Exception as e:
            logger.error("Bootstrap resampling failed", error=str(e))
            raise
    
    def run_permutation_test(self, 
                           data1: np.ndarray,
                           data2: np.ndarray,
                           statistic_function: Callable,
                           n_permutations: int = 1000) -> StatisticalTestResult:
        """
        Run permutation test for strategy comparison.
        
        Args:
            data1: First dataset
            data2: Second dataset
            statistic_function: Function to calculate test statistic
            n_permutations: Number of permutations
            
        Returns:
            StatisticalTestResult with permutation test results
        """
        try:
            logger.info("Starting permutation test", n_permutations=n_permutations)
            
            # Calculate observed statistic
            observed_statistic = statistic_function(data1, data2)
            
            # Generate permutation statistics
            combined_data = np.concatenate([data1, data2])
            permutation_stats = []
            
            for _ in range(n_permutations):
                # Shuffle data
                shuffled_data = np.random.permutation(combined_data)
                perm_data1 = shuffled_data[:len(data1)]
                perm_data2 = shuffled_data[len(data1):]
                
                # Calculate statistic
                perm_stat = statistic_function(perm_data1, perm_data2)
                permutation_stats.append(perm_stat)
            
            permutation_stats = np.array(permutation_stats)
            
            # Calculate p-value
            p_value = np.mean(permutation_stats >= observed_statistic)
            
            # Calculate effect size
            effect_size = (observed_statistic - np.mean(permutation_stats)) / np.std(permutation_stats)
            
            result = StatisticalTestResult(
                test_type=TestType.PERMUTATION,
                test_statistic=observed_statistic,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(np.percentile(permutation_stats, 2.5), 
                                   np.percentile(permutation_stats, 97.5)),
                is_significant=p_value < self.default_alpha,
                alpha=self.default_alpha,
                metadata={'n_permutations': n_permutations}
            )
            
            # Store result
            self.test_results.append(result)
            
            logger.info("Permutation test completed",
                       p_value=p_value,
                       is_significant=result.is_significant)
            
            return result
            
        except Exception as e:
            logger.error("Permutation test failed", error=str(e))
            raise
    
    def run_power_analysis(self, 
                          effect_size: float,
                          alpha: float = None,
                          power: float = 0.8,
                          sample_size: int = None,
                          test_type: str = 't-test',
                          alternative: str = 'two-sided') -> PowerAnalysisResult:
        """
        Run statistical power analysis.
        
        Args:
            effect_size: Expected effect size
            alpha: Significance level
            power: Desired power
            sample_size: Sample size (if None, will be calculated)
            test_type: Type of test
            alternative: Alternative hypothesis
            
        Returns:
            PowerAnalysisResult with power analysis results
        """
        try:
            alpha = alpha or self.default_alpha
            logger.info("Starting power analysis", effect_size=effect_size, alpha=alpha, power=power)
            
            if sample_size is None:
                # Calculate required sample size
                if test_type == 't-test':
                    sample_size = stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power)
                    sample_size = int((2 * sample_size**2) / effect_size**2)
                else:
                    # Simplified calculation for other tests
                    sample_size = int(50 / effect_size**2)
            
            # Calculate actual power
            if test_type == 't-test':
                actual_power = stats.norm.cdf(stats.norm.ppf(alpha/2) + effect_size * np.sqrt(sample_size/2))
            else:
                actual_power = power  # Simplified
            
            result = PowerAnalysisResult(
                effect_size=effect_size,
                alpha=alpha,
                power=actual_power,
                sample_size=sample_size,
                alternative=alternative,
                test_type=test_type
            )
            
            # Store result
            self.power_results.append(result)
            
            logger.info("Power analysis completed",
                       sample_size=sample_size,
                       actual_power=actual_power)
            
            return result
            
        except Exception as e:
            logger.error("Power analysis failed", error=str(e))
            raise
    
    def apply_multiple_testing_correction(self, 
                                        p_values: List[float],
                                        method: CorrectionMethod = CorrectionMethod.BONFERRONI,
                                        alpha: float = None) -> Dict[str, Any]:
        """
        Apply multiple testing correction.
        
        Args:
            p_values: List of p-values
            method: Correction method
            alpha: Significance level
            
        Returns:
            Dictionary with corrected p-values and significance
        """
        try:
            alpha = alpha or self.default_alpha
            p_values = np.array(p_values)
            n_tests = len(p_values)
            
            if method == CorrectionMethod.BONFERRONI:
                corrected_p_values = p_values * n_tests
                corrected_p_values = np.minimum(corrected_p_values, 1.0)
                
            elif method == CorrectionMethod.HOLM:
                # Sort p-values and apply Holm correction
                sorted_indices = np.argsort(p_values)
                corrected_p_values = np.zeros_like(p_values)
                
                for i, idx in enumerate(sorted_indices):
                    corrected_p_values[idx] = p_values[idx] * (n_tests - i)
                
                corrected_p_values = np.minimum(corrected_p_values, 1.0)
                
            elif method == CorrectionMethod.FDR:
                # Benjamini-Hochberg FDR correction
                sorted_indices = np.argsort(p_values)
                corrected_p_values = np.zeros_like(p_values)
                
                for i, idx in enumerate(sorted_indices):
                    corrected_p_values[idx] = p_values[idx] * n_tests / (i + 1)
                
                corrected_p_values = np.minimum(corrected_p_values, 1.0)
                
            else:
                raise ValueError(f"Unsupported correction method: {method}")
            
            # Determine significance
            is_significant = corrected_p_values < alpha
            
            return {
                'original_p_values': p_values,
                'corrected_p_values': corrected_p_values,
                'is_significant': is_significant,
                'method': method.value,
                'alpha': alpha,
                'n_tests': n_tests
            }
            
        except Exception as e:
            logger.error("Multiple testing correction failed", error=str(e))
            raise
    
    def plot_test_results(self, save_path: str = None) -> plt.Figure:
        """Plot statistical test results."""
        if not self.test_results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Statistical Test Results', fontsize=16)
        
        # Plot 1: P-values distribution
        p_values = [result.p_value for result in self.test_results]
        test_types = [result.test_type.value for result in self.test_results]
        
        ax1 = axes[0, 0]
        ax1.hist(p_values, bins=20, alpha=0.7)
        ax1.axvline(self.default_alpha, color='red', linestyle='--', label=f'α={self.default_alpha}')
        ax1.set_title('P-values Distribution')
        ax1.set_xlabel('P-value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Effect sizes
        effect_sizes = [result.effect_size for result in self.test_results]
        
        ax2 = axes[0, 1]
        ax2.bar(range(len(effect_sizes)), effect_sizes)
        ax2.set_title('Effect Sizes by Test')
        ax2.set_xlabel('Test Index')
        ax2.set_ylabel('Effect Size')
        ax2.grid(True)
        
        # Plot 3: Significance by test type
        significant_counts = {}
        total_counts = {}
        
        for result in self.test_results:
            test_type = result.test_type.value
            if test_type not in total_counts:
                total_counts[test_type] = 0
                significant_counts[test_type] = 0
            
            total_counts[test_type] += 1
            if result.is_significant:
                significant_counts[test_type] += 1
        
        test_types = list(total_counts.keys())
        significance_rates = [significant_counts[t] / total_counts[t] for t in test_types]
        
        ax3 = axes[1, 0]
        ax3.bar(test_types, significance_rates)
        ax3.set_title('Significance Rate by Test Type')
        ax3.set_ylabel('Significance Rate')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True)
        
        # Plot 4: Bootstrap results (if available)
        ax4 = axes[1, 1]
        if self.bootstrap_results:
            bootstrap_means = [result.mean_estimate for result in self.bootstrap_results]
            bootstrap_stds = [result.std_estimate for result in self.bootstrap_results]
            
            ax4.errorbar(range(len(bootstrap_means)), bootstrap_means, 
                        yerr=bootstrap_stds, fmt='o')
            ax4.set_title('Bootstrap Estimates')
            ax4.set_xlabel('Bootstrap Index')
            ax4.set_ylabel('Estimate')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'No bootstrap results available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Bootstrap Results')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def create_statistical_tester(default_alpha: float = 0.05,
                            n_monte_carlo: int = 10000,
                            n_bootstrap: int = 1000) -> StatisticalTester:
    """
    Create a statistical tester with specified parameters.
    
    Args:
        default_alpha: Default significance level
        n_monte_carlo: Number of Monte Carlo simulations
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        StatisticalTester instance
    """
    return StatisticalTester(
        default_alpha=default_alpha,
        n_monte_carlo=n_monte_carlo,
        n_bootstrap=n_bootstrap
    )

def run_quick_statistical_test(data1: np.ndarray,
                             data2: np.ndarray,
                             test_type: str = "t_test",
                             alpha: float = 0.05) -> Dict[str, Any]:
    """
    Quick function to run a statistical test.
    
    Args:
        data1: First dataset
        data2: Second dataset
        test_type: Type of test
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    test_type_enum = TestType(test_type)
    tester = StatisticalTester()
    result = tester.run_hypothesis_test(data1, data2, test_type_enum, alpha)
    
    return {
        'test_type': result.test_type.value,
        'p_value': result.p_value,
        'is_significant': result.is_significant,
        'effect_size': result.effect_size,
        'confidence_interval': result.confidence_interval
    }