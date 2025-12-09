"""
Evaluation and backtesting framework with walk-forward cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass

from config import EvaluationConfig


@dataclass
class BacktestResult:
    """Container for backtest results."""
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: Dict[str, float]
    positions: pd.DataFrame


class WalkForwardValidator:
    """
    Implements walk-forward cross-validation for time series.
    Prevents lookahead bias and provides realistic performance estimates.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def create_splits(self, df: pd.DataFrame) -> List[Dict[str, pd.Index]]:
        """
        Create train/validation/test splits using walk-forward approach.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            List of dictionaries with 'train', 'val', 'test' indices
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        splits = []
        
        start_date = df.index.min()
        end_date = df.index.max()
        
        # Convert config to timedeltas
        train_delta = timedelta(days=self.config.train_size_years * 365)
        val_delta = timedelta(days=self.config.validation_size_months * 30)
        test_delta = timedelta(days=self.config.test_size_months * 30)
        step_delta = timedelta(days=self.config.step_size_months * 30)
        
        current_date = start_date + train_delta
        
        while current_date + val_delta + test_delta <= end_date:
            train_start = current_date - train_delta
            train_end = current_date
            val_start = train_end
            val_end = val_start + val_delta
            test_start = val_end
            test_end = test_start + test_delta
            
            # Get indices for each split
            train_idx = df[(df.index >= train_start) & (df.index < train_end)].index
            val_idx = df[(df.index >= val_start) & (df.index < val_end)].index
            test_idx = df[(df.index >= test_start) & (df.index < test_end)].index
            
            if len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0:
                splits.append({
                    'train': train_idx,
                    'val': val_idx,
                    'test': test_idx,
                    'train_period': (train_start, train_end),
                    'val_period': (val_start, val_end),
                    'test_period': (test_start, test_end),
                })
            
            current_date += step_delta
        
        return splits
    
    def cross_validate(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                      train_func: Callable, predict_func: Callable,
                      metric_func: Callable) -> Dict:
        """
        Perform walk-forward cross-validation.
        
        Args:
            df: Full DataFrame with datetime index
            X: Features
            y: Labels
            train_func: Function to train model (takes X_train, y_train)
            predict_func: Function to make predictions (takes model, X)
            metric_func: Function to calculate metrics (takes y_true, y_pred)
            
        Returns:
            Dictionary with CV results
        """
        splits = self.create_splits(df)
        
        results = {
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': [],
            'models': [],
            'splits_info': splits,
        }
        
        for i, split in enumerate(splits):
            print(f"Processing fold {i+1}/{len(splits)}: "
                  f"Train {split['train_period'][0].date()} to {split['train_period'][1].date()}, "
                  f"Test {split['test_period'][0].date()} to {split['test_period'][1].date()}")
            
            # Get data for this split
            X_train = X.loc[split['train']]
            y_train = y.loc[split['train']]
            X_val = X.loc[split['val']]
            y_val = y.loc[split['val']]
            X_test = X.loc[split['test']]
            y_test = y.loc[split['test']]
            
            # Train model
            model = train_func(X_train, y_train, X_val, y_val)
            
            # Make predictions
            y_train_pred = predict_func(model, X_train)
            y_val_pred = predict_func(model, X_val)
            y_test_pred = predict_func(model, X_test)
            
            # Calculate metrics
            train_metrics = metric_func(y_train, y_train_pred)
            val_metrics = metric_func(y_val, y_val_pred)
            test_metrics = metric_func(y_test, y_test_pred)
            
            results['train_metrics'].append(train_metrics)
            results['val_metrics'].append(val_metrics)
            results['test_metrics'].append(test_metrics)
            results['models'].append(model)
        
        # Aggregate metrics
        results['avg_train_metrics'] = self._aggregate_metrics(results['train_metrics'])
        results['avg_val_metrics'] = self._aggregate_metrics(results['val_metrics'])
        results['avg_test_metrics'] = self._aggregate_metrics(results['test_metrics'])
        
        return results
    
    @staticmethod
    def _aggregate_metrics(metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across folds."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m and m[key] is not None]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated


class Backtester:
    """
    Backtesting engine with realistic transaction costs.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def backtest(self, df: pd.DataFrame, signals: pd.Series,
                predictions: Optional[pd.Series] = None) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            signals: Series with trading signals (1=buy, 0=hold, -1=sell)
            predictions: Optional prediction scores/probabilities
            
        Returns:
            BacktestResult object
        """
        # Initialize
        capital = self.config.initial_capital
        position_size = self.config.position_size_pct
        max_positions = self.config.max_positions
        
        # Tracking
        equity_curve = pd.Series(index=df.index, dtype=float)
        equity_curve.iloc[0] = capital
        
        positions = pd.DataFrame(index=df.index, columns=['position', 'shares', 'value'])
        positions['position'] = 0.0
        positions['shares'] = 0.0
        positions['value'] = 0.0
        
        trades = []
        
        current_position = 0
        entry_price = 0
        entry_date = None
        
        for i in range(1, len(df)):
            date = df.index[i]
            prev_date = df.index[i-1]
            
            # Get prices (use Open for execution with delay)
            if self.config.execution_delay_days == 1:
                execution_price = df['Open'].iloc[i]
            else:
                execution_price = df['Close'].iloc[i]
            
            signal = signals.iloc[i-1] if i-1 < len(signals) else 0  # Use previous day's signal
            
            # Calculate transaction costs
            spread_cost = execution_price * (self.config.bid_ask_spread_bps / 10000)
            slippage_cost = execution_price * (self.config.slippage_bps / 10000)
            total_cost_pct = self.config.commission_pct
            
            # Trading logic
            if signal > 0 and current_position == 0:  # Buy signal
                # Calculate position size
                position_value = capital * position_size
                shares = position_value / (execution_price + spread_cost + slippage_cost)
                
                # Deduct costs
                transaction_cost = position_value * total_cost_pct
                capital -= position_value + transaction_cost
                
                # Record position
                current_position = shares
                entry_price = execution_price + spread_cost + slippage_cost
                entry_date = date
                
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'shares': shares,
                    'signal': signal,
                    'type': 'LONG_ENTRY'
                })
                
            elif signal < 0 and current_position > 0:  # Sell signal
                # Calculate exit value
                exit_price = execution_price - spread_cost - slippage_cost
                position_value = current_position * exit_price
                
                # Deduct costs
                transaction_cost = position_value * total_cost_pct
                capital += position_value - transaction_cost
                
                # Calculate trade metrics
                trade_return = (exit_price - entry_price) / entry_price
                trade_pnl = (exit_price - entry_price) * current_position
                
                # Update last trade
                if trades and trades[-1]['type'] == 'LONG_ENTRY':
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'pnl': trade_pnl,
                        'holding_days': (date - entry_date).days if hasattr(date, 'days') else 1,
                        'type': 'COMPLETED_TRADE'
                    })
                
                # Close position
                current_position = 0
                entry_price = 0
                entry_date = None
            
            # Update position tracking
            positions.loc[date, 'position'] = current_position
            positions.loc[date, 'shares'] = current_position
            
            if current_position > 0:
                positions.loc[date, 'value'] = current_position * df['Close'].iloc[i]
            else:
                positions.loc[date, 'value'] = 0
            
            # Update equity
            total_equity = capital + positions.loc[date, 'value']
            equity_curve.iloc[i] = total_equity
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, pd.DataFrame(trades))
        
        return BacktestResult(
            trades=pd.DataFrame(trades),
            equity_curve=equity_curve,
            metrics=metrics,
            positions=positions
        )
    
    def _calculate_metrics(self, equity_curve: pd.Series, 
                          trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Remove NaN values
        equity_clean = equity_curve.dropna()
        
        if len(equity_clean) < 2:
            return {}
        
        # Returns
        returns = equity_clean.pct_change().dropna()
        total_return = (equity_clean.iloc[-1] / equity_clean.iloc[0]) - 1
        
        # CAGR
        years = len(equity_clean) / 252  # Assuming daily data
        cagr = (equity_clean.iloc[-1] / equity_clean.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (returns.mean() * 252) / (volatility + 1e-8)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        sortino = (returns.mean() * 252) / downside_std
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if len(trades) > 0 and 'return' in trades.columns:
            completed_trades = trades[trades['type'] == 'COMPLETED_TRADE']
            
            if len(completed_trades) > 0:
                win_rate = (completed_trades['return'] > 0).mean()
                avg_win = completed_trades[completed_trades['return'] > 0]['return'].mean() if (completed_trades['return'] > 0).any() else 0
                avg_loss = completed_trades[completed_trades['return'] < 0]['return'].mean() if (completed_trades['return'] < 0).any() else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                avg_trade_return = completed_trades['return'].mean()
                
                # Expectancy
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
            else:
                win_rate = avg_win = avg_loss = profit_factor = avg_trade_return = expectancy = 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_trade_return = expectancy = 0
        
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'expectancy': expectancy,
            'num_trades': len(trades),
            'final_equity': equity_clean.iloc[-1],
        }
        
        return metrics
    
    def optimize_threshold(self, df: pd.DataFrame, 
                          predictions: pd.Series,
                          thresholds: Optional[List[float]] = None) -> Dict:
        """
        Optimize decision threshold for trading signals.
        
        Args:
            df: DataFrame with OHLCV data
            predictions: Prediction scores/probabilities
            thresholds: List of thresholds to test
            
        Returns:
            Dictionary with optimization results
        """
        if thresholds is None:
            thresholds = np.linspace(0.3, 0.7, 21)
        
        results = []
        
        for threshold in thresholds:
            # Convert predictions to signals
            signals = pd.Series(0, index=predictions.index)
            signals[predictions >= threshold] = 1
            signals[predictions < (1 - threshold)] = -1
            
            # Backtest
            backtest_result = self.backtest(df, signals, predictions)
            
            results.append({
                'threshold': threshold,
                'sharpe': backtest_result.metrics.get('sharpe_ratio', 0),
                'total_return': backtest_result.metrics.get('total_return', 0),
                'max_drawdown': backtest_result.metrics.get('max_drawdown', 0),
                'win_rate': backtest_result.metrics.get('win_rate', 0),
            })
        
        results_df = pd.DataFrame(results)
        
        # Find best threshold
        best_idx = results_df['sharpe'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        
        return {
            'optimization_results': results_df,
            'best_threshold': best_threshold,
            'best_metrics': results_df.loc[best_idx].to_dict(),
        }


class MetricsCalculator:
    """
    Calculates various evaluation metrics.
    """
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
        }
        
        # Handle binary vs multiclass
        n_classes = len(np.unique(y_true))
        average = 'binary' if n_classes == 2 else 'macro'
        
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                if n_classes == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                    metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy (for returns)
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        directional_accuracy = (direction_true == direction_pred).mean()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
        }


def generate_evaluation_report(backtest_result: BacktestResult,
                               save_path: Optional[str] = None) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        backtest_result: BacktestResult object
        save_path: Optional path to save report
        
    Returns:
        Report string
    """
    metrics = backtest_result.metrics
    
    report_lines = [
        "=" * 70,
        "BACKTEST EVALUATION REPORT",
        "=" * 70,
        "",
        "PERFORMANCE METRICS:",
        f"  Total Return:      {metrics.get('total_return', 0):>10.2%}",
        f"  CAGR:             {metrics.get('cagr', 0):>10.2%}",
        f"  Volatility:       {metrics.get('volatility', 0):>10.2%}",
        f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):>10.2f}",
        f"  Sortino Ratio:    {metrics.get('sortino_ratio', 0):>10.2f}",
        f"  Max Drawdown:     {metrics.get('max_drawdown', 0):>10.2%}",
        "",
        "TRADE STATISTICS:",
        f"  Number of Trades: {metrics.get('num_trades', 0):>10}",
        f"  Win Rate:         {metrics.get('win_rate', 0):>10.2%}",
        f"  Avg Trade Return: {metrics.get('avg_trade_return', 0):>10.2%}",
        f"  Avg Win:          {metrics.get('avg_win', 0):>10.2%}",
        f"  Avg Loss:         {metrics.get('avg_loss', 0):>10.2%}",
        f"  Profit Factor:    {metrics.get('profit_factor', 0):>10.2f}",
        f"  Expectancy:       {metrics.get('expectancy', 0):>10.2%}",
        "",
        f"  Final Equity:     ${metrics.get('final_equity', 0):>10,.2f}",
        "=" * 70,
    ]
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report
