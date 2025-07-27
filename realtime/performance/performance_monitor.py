#!/usr/bin/env python3
"""
Live Performance Monitoring Module

Implements live performance monitoring system for real-time tracking:
- Real-time performance tracking
- Performance metrics calculation
- Live performance dashboard
- Performance alerts
- Performance reporting
- Performance analytics

Features:
- Real-time performance tracking and monitoring
- Comprehensive performance metrics calculation
- Live performance dashboard and visualization
- Intelligent performance alerting system
- Automated performance reporting
- Advanced performance analytics and insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import threading
import queue
import time
import uuid
import json
from collections import defaultdict, deque

logger = structlog.get_logger()

class PerformanceMetric(Enum):
    """Performance metrics."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    ALPHA = "alpha"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"
    RECOVERY_FACTOR = "recovery_factor"

class AlertLevel(Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Performance alert types."""
    PERFORMANCE_DROP = "performance_drop"
    DRAWDOWN_BREACH = "drawdown_breach"
    SHARPE_DECLINE = "sharpe_decline"
    VOLATILITY_SPIKE = "volatility_spike"
    UNDERPERFORMANCE = "underperformance"
    OVERPERFORMANCE = "overperformance"

@dataclass
class PerformanceData:
    """Performance data structure."""
    strategy_id: str
    timestamp: datetime
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    alpha: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    recovery_factor: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert structure."""
    alert_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    message: str
    strategy_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""
    update_interval: float = 1.0  # seconds
    lookback_period: int = 252  # days
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    enable_alerts: bool = True
    enable_reporting: bool = True
    enable_analytics: bool = True

class MetricsCalculator:
    """Performance metrics calculation engine."""
    
    def __init__(self, lookback_period: int = 252):
        """
        Initialize metrics calculator.
        
        Args:
            lookback_period: Lookback period for calculations
        """
        self.lookback_period = lookback_period
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_performance_metrics(self, returns: np.ndarray, 
                                    benchmark_returns: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Array of returns
            benchmark_returns: Array of benchmark returns (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        if len(returns) == 0:
            return self._get_empty_metrics()
        
        try:
            metrics = {}
            
            # Basic return metrics
            metrics['total_return'] = self._calculate_total_return(returns)
            metrics['volatility'] = self._calculate_volatility(returns)
            
            # Risk-adjusted metrics
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
            metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns)
            
            # Drawdown metrics
            metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
            metrics['recovery_factor'] = self._calculate_recovery_factor(returns)
            
            # Win/loss metrics
            win_rate, profit_factor, avg_win, avg_loss = self._calculate_win_loss_metrics(returns)
            metrics['win_rate'] = win_rate
            metrics['profit_factor'] = profit_factor
            metrics['average_win'] = avg_win
            metrics['average_loss'] = avg_loss
            
            # Benchmark metrics (if benchmark provided)
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                beta, alpha = self._calculate_beta_alpha(returns, benchmark_returns)
                metrics['beta'] = beta
                metrics['alpha'] = alpha
            else:
                metrics['beta'] = 1.0
                metrics['alpha'] = 0.0
            
            return metrics
        
        except Exception as e:
            logger.error("Performance metrics calculation error", error=str(e))
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Get empty metrics dictionary."""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'beta': 1.0,
            'alpha': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'recovery_factor': 0.0
        }
    
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return."""
        return np.prod(1 + returns) - 1
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        return np.std(returns) * np.sqrt(252)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        mean_excess_return = np.mean(excess_returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        return mean_excess_return / volatility if volatility > 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        mean_excess_return = np.mean(excess_returns) * 252
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        
        return mean_excess_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0
        
        total_return = self._calculate_total_return(returns)
        max_drawdown = abs(self._calculate_max_drawdown(returns))
        
        return total_return / max_drawdown if max_drawdown > 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        return np.min(drawdown)
    
    def _calculate_recovery_factor(self, returns: np.ndarray) -> float:
        """Calculate recovery factor."""
        if len(returns) == 0:
            return 0.0
        
        total_return = self._calculate_total_return(returns)
        max_drawdown = abs(self._calculate_max_drawdown(returns))
        
        return total_return / max_drawdown if max_drawdown > 0 else 0.0
    
    def _calculate_win_loss_metrics(self, returns: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate win/loss metrics."""
        if len(returns) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        win_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0.0
        average_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0.0
        average_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0.0
        
        profit_factor = abs(average_win * len(winning_returns) / (average_loss * len(losing_returns))) if average_loss != 0 else 0.0
        
        return win_rate, profit_factor, average_win, average_loss
    
    def _calculate_beta_alpha(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> Tuple[float, float]:
        """Calculate beta and alpha."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 1.0, 0.0
        
        # Align returns to same length
        min_length = min(len(returns), len(benchmark_returns))
        aligned_returns = returns[-min_length:]
        aligned_benchmark = benchmark_returns[-min_length:]
        
        # Calculate beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Calculate alpha
        mean_return = np.mean(aligned_returns) * 252
        mean_benchmark = np.mean(aligned_benchmark) * 252
        alpha = mean_return - (self.risk_free_rate + beta * (mean_benchmark - self.risk_free_rate))
        
        return beta, alpha

class PerformanceTracker:
    """Real-time performance tracking system."""
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize performance tracker.
        
        Args:
            config: Performance monitoring configuration
        """
        self.config = config
        self.metrics_calculator = MetricsCalculator(config.lookback_period)
        
        self.strategies = {}
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.benchmark_data = {}
        
        self.running = False
        self.tracking_thread = None
    
    def add_strategy(self, strategy_id: str, initial_capital: float = 100000.0):
        """Add strategy for performance tracking."""
        self.strategies[strategy_id] = {
            'initial_capital': initial_capital,
            'current_capital': initial_capital,
            'returns': [],
            'trades': [],
            'last_update': datetime.now()
        }
        logger.info("Strategy added for performance tracking", strategy_id=strategy_id)
    
    def update_strategy_performance(self, strategy_id: str, returns: List[float], 
                                  trades: List[Dict[str, Any]] = None):
        """Update strategy performance data."""
        if strategy_id not in self.strategies:
            logger.warning("Strategy not found", strategy_id=strategy_id)
            return
        
        strategy_data = self.strategies[strategy_id]
        
        # Update returns
        if returns:
            strategy_data['returns'].extend(returns)
            # Keep only recent returns
            if len(strategy_data['returns']) > self.config.lookback_period:
                strategy_data['returns'] = strategy_data['returns'][-self.config.lookback_period:]
        
        # Update trades
        if trades:
            strategy_data['trades'].extend(trades)
        
        # Update current capital
        if strategy_data['returns']:
            total_return = np.prod(1 + np.array(strategy_data['returns'])) - 1
            strategy_data['current_capital'] = strategy_data['initial_capital'] * (1 + total_return)
        
        strategy_data['last_update'] = datetime.now()
        
        # Calculate and store performance metrics
        self._calculate_and_store_metrics(strategy_id)
    
    def _calculate_and_store_metrics(self, strategy_id: str):
        """Calculate and store performance metrics."""
        if strategy_id not in self.strategies:
            return
        
        strategy_data = self.strategies[strategy_id]
        returns = np.array(strategy_data['returns'])
        
        if len(returns) == 0:
            return
        
        # Get benchmark returns if available
        benchmark_returns = self.benchmark_data.get(strategy_id)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_performance_metrics(returns, benchmark_returns)
        
        # Create performance data
        performance_data = PerformanceData(
            strategy_id=strategy_id,
            timestamp=datetime.now(),
            **metrics
        )
        
        # Store in history
        self.performance_history[strategy_id].append(performance_data)
        
        logger.debug("Performance metrics calculated", strategy_id=strategy_id, 
                    total_return=metrics['total_return'], sharpe_ratio=metrics['sharpe_ratio'])
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[PerformanceData]:
        """Get latest performance data for strategy."""
        if strategy_id not in self.performance_history:
            return None
        
        history = self.performance_history[strategy_id]
        return history[-1] if history else None
    
    def get_performance_history(self, strategy_id: str, days: int = 30) -> List[PerformanceData]:
        """Get performance history for strategy."""
        if strategy_id not in self.performance_history:
            return []
        
        history = list(self.performance_history[strategy_id])
        return history[-days:] if days > 0 else history
    
    def add_benchmark_data(self, strategy_id: str, benchmark_returns: List[float]):
        """Add benchmark data for strategy."""
        self.benchmark_data[strategy_id] = benchmark_returns
        logger.info("Benchmark data added", strategy_id=strategy_id)
    
    def start(self):
        """Start performance tracking."""
        if not self.running:
            self.running = True
            self.tracking_thread = threading.Thread(target=self._tracking_worker)
            self.tracking_thread.daemon = True
            self.tracking_thread.start()
            logger.info("Performance tracker started")
    
    def stop(self):
        """Stop performance tracking."""
        self.running = False
        if self.tracking_thread:
            self.tracking_thread.join()
        logger.info("Performance tracker stopped")
    
    def _tracking_worker(self):
        """Performance tracking worker."""
        while self.running:
            try:
                # Update performance for all strategies
                for strategy_id in self.strategies.keys():
                    self._calculate_and_store_metrics(strategy_id)
                
                time.sleep(self.config.update_interval)
            
            except Exception as e:
                logger.error("Performance tracking error", error=str(e))

class PerformanceAlertManager:
    """Performance alerting system."""
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize performance alert manager.
        
        Args:
            config: Performance monitoring configuration
        """
        self.config = config
        self.alert_thresholds = config.alert_thresholds
        self.alerts = []
        self.alert_callbacks = []
    
    def set_alert_threshold(self, metric: str, threshold: float, alert_type: AlertType = AlertType.PERFORMANCE_DROP):
        """Set alert threshold for metric."""
        self.alert_thresholds[metric] = {
            'threshold': threshold,
            'type': alert_type
        }
        logger.info("Alert threshold set", metric=metric, threshold=threshold)
    
    def check_performance_alerts(self, performance_data: PerformanceData) -> List[PerformanceAlert]:
        """Check performance data against alert thresholds."""
        alerts = []
        
        for metric_name, threshold_config in self.alert_thresholds.items():
            threshold = threshold_config['threshold']
            alert_type = threshold_config['type']
            
            if hasattr(performance_data, metric_name):
                current_value = getattr(performance_data, metric_name)
                
                # Check if threshold is breached
                if self._is_threshold_breached(metric_name, current_value, threshold, alert_type):
                    alert = self._create_alert(performance_data.strategy_id, metric_name, 
                                             current_value, threshold, alert_type)
                    alerts.append(alert)
        
        # Add alerts to history
        self.alerts.extend(alerts)
        
        # Call alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error("Performance alert callback error", error=str(e))
        
        return alerts
    
    def _is_threshold_breached(self, metric_name: str, current_value: float, 
                              threshold: float, alert_type: AlertType) -> bool:
        """Check if threshold is breached."""
        if alert_type in [AlertType.PERFORMANCE_DROP, AlertType.SHARPE_DECLINE, AlertType.UNDERPERFORMANCE]:
            return current_value < threshold
        elif alert_type in [AlertType.VOLATILITY_SPIKE, AlertType.DRAWDOWN_BREACH]:
            return current_value > threshold
        elif alert_type == AlertType.OVERPERFORMANCE:
            return current_value > threshold
        else:
            return False
    
    def _create_alert(self, strategy_id: str, metric_name: str, current_value: float,
                     threshold: float, alert_type: AlertType) -> PerformanceAlert:
        """Create performance alert."""
        # Determine alert level based on severity
        alert_level = self._determine_alert_level(metric_name, current_value, threshold)
        
        # Create alert message
        message = self._create_alert_message(metric_name, current_value, threshold, alert_type)
        
        alert = PerformanceAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            alert_level=alert_level,
            message=message,
            strategy_id=strategy_id,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold
        )
        
        return alert
    
    def _determine_alert_level(self, metric_name: str, current_value: float, threshold: float) -> AlertLevel:
        """Determine alert level based on severity."""
        # Calculate deviation from threshold
        if threshold != 0:
            deviation = abs(current_value - threshold) / abs(threshold)
        else:
            deviation = abs(current_value)
        
        if deviation > 0.5:  # 50% deviation
            return AlertLevel.CRITICAL
        elif deviation > 0.25:  # 25% deviation
            return AlertLevel.ERROR
        elif deviation > 0.1:  # 10% deviation
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _create_alert_message(self, metric_name: str, current_value: float, 
                            threshold: float, alert_type: AlertType) -> str:
        """Create alert message."""
        if alert_type in [AlertType.PERFORMANCE_DROP, AlertType.SHARPE_DECLINE, AlertType.UNDERPERFORMANCE]:
            return f"{metric_name} ({current_value:.2f}) below threshold ({threshold:.2f})"
        elif alert_type in [AlertType.VOLATILITY_SPIKE, AlertType.DRAWDOWN_BREACH]:
            return f"{metric_name} ({current_value:.2f}) above threshold ({threshold:.2f})"
        elif alert_type == AlertType.OVERPERFORMANCE:
            return f"{metric_name} ({current_value:.2f}) above threshold ({threshold:.2f})"
        else:
            return f"{metric_name} threshold breached: {current_value:.2f} vs {threshold:.2f}"
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add performance alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_alerts(self, strategy_id: str = None, alert_level: AlertLevel = None) -> List[PerformanceAlert]:
        """Get alerts with optional filtering."""
        alerts = self.alerts
        
        if strategy_id:
            alerts = [alert for alert in alerts if alert.strategy_id == strategy_id]
        
        if alert_level:
            alerts = [alert for alert in alerts if alert.alert_level == alert_level]
        
        return alerts

class PerformanceMonitor:
    """Main live performance monitoring system."""
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize performance monitor.
        
        Args:
            config: Performance monitoring configuration
        """
        self.config = config
        self.tracker = PerformanceTracker(config)
        self.alert_manager = PerformanceAlertManager(config)
        
        self.running = False
        self.monitoring_thread = None
        
        # Setup alert callbacks
        self.alert_manager.add_alert_callback(self._on_performance_alert)
    
    def add_strategy(self, strategy_id: str, initial_capital: float = 100000.0):
        """Add strategy for monitoring."""
        self.tracker.add_strategy(strategy_id, initial_capital)
    
    def update_strategy_performance(self, strategy_id: str, returns: List[float], 
                                  trades: List[Dict[str, Any]] = None):
        """Update strategy performance."""
        self.tracker.update_strategy_performance(strategy_id, returns, trades)
        
        # Check for alerts
        performance_data = self.tracker.get_strategy_performance(strategy_id)
        if performance_data:
            alerts = self.alert_manager.check_performance_alerts(performance_data)
            if alerts:
                logger.info(f"Performance alerts generated for strategy {strategy_id}", 
                          alert_count=len(alerts))
    
    def set_alert_threshold(self, metric: str, threshold: float, alert_type: AlertType = AlertType.PERFORMANCE_DROP):
        """Set performance alert threshold."""
        self.alert_manager.set_alert_threshold(metric, threshold, alert_type)
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[PerformanceData]:
        """Get strategy performance data."""
        return self.tracker.get_strategy_performance(strategy_id)
    
    def get_performance_summary(self, strategy_id: str = None) -> Dict[str, Any]:
        """Get performance summary."""
        if strategy_id:
            strategies = [strategy_id] if strategy_id in self.tracker.strategies else []
        else:
            strategies = list(self.tracker.strategies.keys())
        
        summary = {
            'total_strategies': len(strategies),
            'performance_data': {},
            'alerts': len(self.alert_manager.alerts),
            'timestamp': datetime.now().isoformat()
        }
        
        for sid in strategies:
            performance_data = self.tracker.get_strategy_performance(sid)
            if performance_data:
                summary['performance_data'][sid] = {
                    'total_return': performance_data.total_return,
                    'sharpe_ratio': performance_data.sharpe_ratio,
                    'max_drawdown': performance_data.max_drawdown,
                    'volatility': performance_data.volatility,
                    'win_rate': performance_data.win_rate,
                    'profit_factor': performance_data.profit_factor
                }
        
        return summary
    
    def start(self):
        """Start performance monitoring."""
        if not self.running:
            self.running = True
            self.tracker.start()
            logger.info("Performance monitor started")
    
    def stop(self):
        """Stop performance monitoring."""
        self.running = False
        self.tracker.stop()
        logger.info("Performance monitor stopped")
    
    def _on_performance_alert(self, alert: PerformanceAlert):
        """Handle performance alert."""
        logger.warning("Performance alert", 
                      strategy_id=alert.strategy_id,
                      alert_type=alert.alert_type.value,
                      alert_level=alert.alert_level.value,
                      message=alert.message)

def create_performance_monitor(config: PerformanceConfig = None) -> PerformanceMonitor:
    """
    Create a live performance monitoring system.
    
    Args:
        config: Performance monitoring configuration
        
    Returns:
        PerformanceMonitor instance
    """
    if config is None:
        config = PerformanceConfig()
    
    return PerformanceMonitor(config)

if __name__ == "__main__":
    # Demo of performance monitoring system
    config = PerformanceConfig(
        update_interval=5.0,
        lookback_period=252,
        enable_alerts=True,
        enable_reporting=True,
        enable_analytics=True
    )
    
    monitor = create_performance_monitor(config)
    
    # Add sample strategy
    monitor.add_strategy("strategy_001", initial_capital=100000.0)
    
    # Set alert thresholds
    monitor.set_alert_threshold("sharpe_ratio", 1.0, AlertType.SHARPE_DECLINE)
    monitor.set_alert_threshold("max_drawdown", -0.1, AlertType.DRAWDOWN_BREACH)
    monitor.set_alert_threshold("volatility", 0.25, AlertType.VOLATILITY_SPIKE)
    
    # Start monitoring
    monitor.start()
    
    print("Performance monitoring system created successfully!")
    print(f"Update interval: {config.update_interval} seconds")
    print(f"Lookback period: {config.lookback_period} days")
    print(f"Alert thresholds: {len(monitor.alert_manager.alert_thresholds)}")
    
    # Simulate performance updates
    sample_returns = np.random.normal(0.001, 0.02, 50)
    monitor.update_strategy_performance("strategy_001", sample_returns.tolist())
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"Performance summary: {summary}")