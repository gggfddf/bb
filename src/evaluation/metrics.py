"""
Evaluation metrics and backtesting for trading models.

Implements comprehensive evaluation including:
- Classification metrics (AUC, precision@k, F1)
- Economic metrics (Sharpe, Sortino, CAGR, Max DD)
- Walk-forward validation
- Realistic backtesting with transaction costs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, balanced_accuracy_score, classification_report
)
import warnings


class ModelEvaluator:
    """
    Evaluates model predictions with comprehensive metrics.
    
    Provides both statistical and economic performance metrics.
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        pass
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for AUC, precision@k)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Remove NaN values
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_pred_proba is not None:
            y_pred_proba = y_pred_proba[mask]
        
        if len(y_true) == 0:
            warnings.warn("No valid predictions to evaluate", UserWarning)
            return metrics
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle binary vs multi-class
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) <= 2
        
        if is_binary:
            # Binary classification metrics
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                try:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                except ValueError:
                    pass
        else:
            # Multi-class metrics
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            if y_pred_proba is not None and len(unique_classes) > 1:
                try:
                    metrics['auc_ovr'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', average='macro')
                except ValueError:
                    pass
        
        # Precision@k (top percentiles)
        if y_pred_proba is not None:
            for k in [5, 10, 20]:
                precision_at_k = self._precision_at_k(y_true, y_pred_proba, k)
                if precision_at_k is not None:
                    metrics[f'precision@top{k}pct'] = precision_at_k
        
        return metrics
    
    def _precision_at_k(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       k: int) -> Optional[float]:
        """
        Calculate precision at top k percentile of predictions.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            k: Percentile threshold (e.g., 5 for top 5%)
            
        Returns:
            Precision at top k% or None if not enough samples
        """
        if len(y_true) < 20:  # Need reasonable sample size
            return None
        
        # Get threshold for top k%
        threshold = np.percentile(y_pred_proba, 100 - k)
        
        # Get predictions in top k%
        top_k_mask = y_pred_proba >= threshold
        
        if top_k_mask.sum() == 0:
            return None
        
        # Calculate precision in top k%
        precision = y_true[top_k_mask].mean()
        
        return precision
    
    def evaluate_economic(self, returns: pd.Series, predictions: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Evaluate economic performance of trading strategy.
        
        Args:
            returns: Actual returns (forward returns)
            predictions: Model predictions (1 = long, 0/-1 = neutral/short)
            benchmark_returns: Benchmark returns for comparison
            transaction_cost: Transaction cost per trade (default 0.1%)
            
        Returns:
            Dictionary of economic metrics
        """
        metrics = {}
        
        # Align and clean data
        df = pd.DataFrame({
            'return': returns,
            'prediction': predictions
        }).dropna()
        
        if len(df) == 0:
            warnings.warn("No valid data for economic evaluation", UserWarning)
            return metrics
        
        # Calculate strategy returns
        # Assume prediction is 1 for long, 0 for neutral, -1 for short
        df['position'] = df['prediction']
        df['strategy_return'] = df['position'] * df['return']
        
        # Apply transaction costs
        df['position_change'] = df['position'].diff().abs()
        df['cost'] = df['position_change'] * transaction_cost
        df['strategy_return_net'] = df['strategy_return'] - df['cost']
        
        # Cumulative returns
        df['cumulative_return'] = (1 + df['strategy_return_net']).cumprod()
        
        # Performance metrics
        total_return = df['cumulative_return'].iloc[-1] - 1
        n_periods = len(df)
        
        # Annualized return (assuming daily data)
        if n_periods > 0:
            trading_days_per_year = 252
            years = n_periods / trading_days_per_year
            metrics['total_return'] = total_return
            metrics['cagr'] = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        if len(df['strategy_return_net']) > 1:
            metrics['volatility'] = df['strategy_return_net'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if 'volatility' in metrics and metrics['volatility'] > 0:
            metrics['sharpe'] = metrics['cagr'] / metrics['volatility']
        
        # Sortino ratio (downside deviation)
        downside_returns = df['strategy_return_net'][df['strategy_return_net'] < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std() * np.sqrt(252)
            if downside_std > 0:
                metrics['sortino'] = metrics['cagr'] / downside_std
        
        # Maximum drawdown
        running_max = df['cumulative_return'].cummax()
        drawdown = (df['cumulative_return'] - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Win rate
        winning_trades = (df['strategy_return_net'] > 0).sum()
        total_trades = (df['position'] != 0).sum()
        if total_trades > 0:
            metrics['win_rate'] = winning_trades / total_trades
        
        # Average trade
        trade_returns = df[df['position'] != 0]['strategy_return_net']
        if len(trade_returns) > 0:
            metrics['avg_trade'] = trade_returns.mean()
            metrics['avg_win'] = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0
            metrics['avg_loss'] = trade_returns[trade_returns < 0].mean() if (trade_returns < 0).any() else 0
        
        # Expectancy
        if 'win_rate' in metrics and 'avg_win' in metrics and 'avg_loss' in metrics:
            metrics['expectancy'] = (metrics['win_rate'] * metrics['avg_win'] + 
                                    (1 - metrics['win_rate']) * metrics['avg_loss'])
        
        # Number of trades
        metrics['num_trades'] = total_trades
        
        # Benchmark comparison
        if benchmark_returns is not None:
            benchmark_aligned = benchmark_returns.reindex(df.index).fillna(0)
            benchmark_cum = (1 + benchmark_aligned).cumprod()
            benchmark_total = benchmark_cum.iloc[-1] - 1
            
            metrics['benchmark_return'] = benchmark_total
            metrics['excess_return'] = total_return - benchmark_total
            
            # Information ratio
            active_returns = df['strategy_return_net'] - benchmark_aligned
            if len(active_returns) > 1:
                tracking_error = active_returns.std() * np.sqrt(252)
                if tracking_error > 0:
                    metrics['information_ratio'] = (metrics['cagr'] - 
                                                   (benchmark_total + 1) ** (1/years) - 1) / tracking_error
        
        return metrics
    
    def print_evaluation_report(self, classification_metrics: Dict[str, float],
                               economic_metrics: Dict[str, float]) -> None:
        """
        Print formatted evaluation report.
        
        Args:
            classification_metrics: Classification metrics dictionary
            economic_metrics: Economic metrics dictionary
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print("\nCLASSIFICATION METRICS:")
        print("-"*60)
        for metric, value in sorted(classification_metrics.items()):
            print(f"  {metric:30s}: {value:8.4f}")
        
        print("\nECONOMIC METRICS:")
        print("-"*60)
        for metric, value in sorted(economic_metrics.items()):
            if 'rate' in metric or 'ratio' in metric or metric in ['sharpe', 'sortino', 'information_ratio']:
                print(f"  {metric:30s}: {value:8.4f}")
            else:
                print(f"  {metric:30s}: {value:8.2%}")
        
        print("="*60 + "\n")


class BacktestEngine:
    """
    Backtesting engine with walk-forward validation.
    
    Implements realistic backtesting with:
    - Walk-forward cross-validation
    - Transaction costs and slippage
    - Position sizing
    - No lookahead bias
    """
    
    def __init__(self, transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize BacktestEngine.
        
        Args:
            transaction_cost: Cost per trade (default 0.1%)
            slippage: Slippage per trade (default 0.05%)
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
    
    def walk_forward_split(self, df: pd.DataFrame,
                          train_size: int = 252 * 2,  # 2 years
                          test_size: int = 252,  # 1 year
                          step_size: Optional[int] = None) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Generate walk-forward train/test splits.
        
        Args:
            df: DataFrame with time index
            train_size: Number of periods for training
            test_size: Number of periods for testing
            step_size: Step size for rolling window (default: test_size)
            
        Returns:
            List of (train_index, test_index) tuples
        """
        if step_size is None:
            step_size = test_size
        
        splits = []
        total_size = len(df)
        
        start = 0
        while start + train_size + test_size <= total_size:
            train_end = start + train_size
            test_end = train_end + test_size
            
            train_idx = df.index[start:train_end]
            test_idx = df.index[train_end:test_end]
            
            splits.append((train_idx, test_idx))
            
            start += step_size
        
        return splits
    
    def backtest_signals(self, df: pd.DataFrame,
                        signals: pd.Series,
                        returns_col: str = 'return',
                        position_size: float = 1.0) -> pd.DataFrame:
        """
        Backtest trading signals.
        
        Args:
            df: DataFrame with price data
            signals: Trading signals (1 = long, 0 = neutral, -1 = short)
            returns_col: Column name for returns
            position_size: Position size (fraction of capital)
            
        Returns:
            DataFrame with backtest results
        """
        results = df.copy()
        results['signal'] = signals
        results['position'] = results['signal'] * position_size
        
        # Calculate returns
        results['strategy_return'] = results['position'] * results[returns_col]
        
        # Transaction costs
        results['position_change'] = results['position'].diff().abs()
        results['transaction_cost'] = results['position_change'] * (self.transaction_cost + self.slippage)
        results['strategy_return_net'] = results['strategy_return'] - results['transaction_cost']
        
        # Cumulative performance
        results['cumulative_return'] = (1 + results['strategy_return_net']).cumprod()
        results['cumulative_benchmark'] = (1 + results[returns_col]).cumprod()
        
        # Drawdown
        running_max = results['cumulative_return'].cummax()
        results['drawdown'] = (results['cumulative_return'] - running_max) / running_max
        
        return results
    
    def analyze_backtest(self, backtest_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze backtest results and return summary statistics.
        
        Args:
            backtest_results: Output from backtest_signals
            
        Returns:
            Dictionary of backtest statistics
        """
        evaluator = ModelEvaluator()
        
        # Create predictions from signals
        predictions = backtest_results['signal'].fillna(0)
        returns = backtest_results['return']
        
        # Get economic metrics
        metrics = evaluator.evaluate_economic(
            returns=returns,
            predictions=predictions,
            transaction_cost=self.transaction_cost
        )
        
        # Add backtest-specific metrics
        metrics['total_days'] = len(backtest_results)
        metrics['days_in_market'] = (backtest_results['position'] != 0).sum()
        metrics['market_exposure'] = metrics['days_in_market'] / metrics['total_days']
        
        return metrics
