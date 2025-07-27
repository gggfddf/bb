#!/usr/bin/env python3
"""
Real-time Risk Management Module

Implements comprehensive real-time risk management system:
- Portfolio risk monitoring
- Position limit management
- VaR (Value at Risk) calculation
- Stress testing framework
- Correlation monitoring
- Risk alerting system

Features:
- Real-time portfolio risk monitoring and assessment
- Advanced position limit management and enforcement
- Comprehensive VaR calculation with multiple methodologies
- Dynamic stress testing and scenario analysis
- Real-time correlation monitoring and analysis
- Intelligent risk alerting and notification system
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
import asyncio
import time
import uuid
from collections import defaultdict
from scipy import stats
from scipy.stats import norm, t

logger = structlog.get_logger()

class RiskMetric(Enum):
    """Risk metrics."""
    VAR = "var"
    CVAR = "cvar"
    VOLATILITY = "volatility"
    BETA = "beta"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"

class RiskLevel(Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Risk alert types."""
    VAR_BREACH = "var_breach"
    POSITION_LIMIT = "position_limit"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN_BREACH = "drawdown_breach"
    CONCENTRATION_BREACH = "concentration_breach"

@dataclass
class RiskLimits:
    """Risk limits configuration."""
    portfolio_var_limit: float
    position_var_limit: float
    max_position_size: float
    max_concentration: float
    max_correlation: float
    max_drawdown: float
    max_volatility: float
    var_confidence_level: float = 0.95
    lookback_period: int = 252  # days

@dataclass
class RiskMetrics:
    """Risk metrics data."""
    portfolio_id: str
    timestamp: datetime
    var: float
    cvar: float
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: pd.DataFrame
    concentration_risk: float
    stress_test_results: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskAlert:
    """Risk alert structure."""
    alert_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    portfolio_id: str
    metric_value: float
    limit_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Position structure for risk calculation."""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    weight: float
    returns: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)

class VaRCalculator:
    """VaR calculation engine."""
    
    def __init__(self, confidence_level: float = 0.95, lookback_period: int = 252):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: Confidence level for VaR calculation
            lookback_period: Lookback period for historical data
        """
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.var_methods = {
            'historical': self._calculate_historical_var,
            'parametric': self._calculate_parametric_var,
            'monte_carlo': self._calculate_monte_carlo_var
        }
    
    def calculate_portfolio_var(self, positions: List[Position], 
                              method: str = 'historical') -> Tuple[float, float]:
        """
        Calculate portfolio VaR.
        
        Args:
            positions: List of positions
            method: VaR calculation method
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        if not positions:
            return 0.0, 0.0
        
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(positions)
            
            # Calculate VaR using specified method
            var_func = self.var_methods.get(method, self._calculate_historical_var)
            var = var_func(portfolio_returns)
            
            # Calculate CVaR
            cvar = self._calculate_cvar(portfolio_returns, var)
            
            return var, cvar
        
        except Exception as e:
            logger.error("Portfolio VaR calculation error", error=str(e))
            return 0.0, 0.0
    
    def _calculate_portfolio_returns(self, positions: List[Position]) -> np.ndarray:
        """Calculate portfolio returns from positions."""
        if not positions:
            return np.array([])
        
        # Align returns to same length
        min_length = min(len(pos.returns) for pos in positions)
        aligned_returns = []
        
        for position in positions:
            if len(position.returns) >= min_length:
                aligned_returns.append(position.returns[-min_length:])
        
        if not aligned_returns:
            return np.array([])
        
        # Calculate weighted portfolio returns
        weights = np.array([pos.weight for pos in positions[:len(aligned_returns)]])
        weights = weights / np.sum(weights)  # Normalize weights
        
        portfolio_returns = np.sum([ret * weight for ret, weight in zip(aligned_returns, weights)], axis=0)
        
        return portfolio_returns
    
    def _calculate_historical_var(self, returns: np.ndarray) -> float:
        """Calculate historical VaR."""
        if len(returns) == 0:
            return 0.0
        
        var_percentile = (1 - self.confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        return abs(var)
    
    def _calculate_parametric_var(self, returns: np.ndarray) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-score for confidence level
        z_score = norm.ppf(1 - self.confidence_level)
        
        var = mean_return - z_score * std_return
        
        return abs(var)
    
    def _calculate_monte_carlo_var(self, returns: np.ndarray, n_simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate random returns
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # Calculate VaR
        var_percentile = (1 - self.confidence_level) * 100
        var = np.percentile(simulated_returns, var_percentile)
        
        return abs(var)
    
    def _calculate_cvar(self, returns: np.ndarray, var: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        
        # Returns below VaR
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        cvar = np.mean(tail_returns)
        
        return abs(cvar)

class StressTester:
    """Stress testing framework."""
    
    def __init__(self):
        """Initialize stress tester."""
        self.scenarios = {
            'market_crash': self._market_crash_scenario,
            'volatility_spike': self._volatility_spike_scenario,
            'correlation_breakdown': self._correlation_breakdown_scenario,
            'interest_rate_shock': self._interest_rate_shock_scenario,
            'liquidity_crisis': self._liquidity_crisis_scenario
        }
    
    def run_stress_tests(self, positions: List[Position], 
                        scenarios: List[str] = None) -> Dict[str, float]:
        """
        Run stress tests on portfolio.
        
        Args:
            positions: List of positions
            scenarios: List of scenario names to run
            
        Returns:
            Dictionary of scenario results
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())
        
        results = {}
        
        for scenario in scenarios:
            if scenario in self.scenarios:
                try:
                    scenario_func = self.scenarios[scenario]
                    result = scenario_func(positions)
                    results[scenario] = result
                except Exception as e:
                    logger.error(f"Stress test error for {scenario}", error=str(e))
                    results[scenario] = 0.0
        
        return results
    
    def _market_crash_scenario(self, positions: List[Position]) -> float:
        """Market crash scenario: 20% decline across all assets."""
        total_value = sum(pos.market_value for pos in positions)
        if total_value == 0:
            return 0.0
        
        # Simulate 20% decline
        crash_loss = total_value * 0.20
        
        return crash_loss
    
    def _volatility_spike_scenario(self, positions: List[Position]) -> float:
        """Volatility spike scenario: 3x increase in volatility."""
        if not positions:
            return 0.0
        
        # Calculate current portfolio volatility
        portfolio_returns = self._calculate_portfolio_returns(positions)
        current_vol = np.std(portfolio_returns)
        
        # Simulate 3x volatility increase
        new_vol = current_vol * 3
        vol_impact = new_vol * np.sqrt(252)  # Annualized impact
        
        return vol_impact
    
    def _correlation_breakdown_scenario(self, positions: List[Position]) -> float:
        """Correlation breakdown scenario: diversification fails."""
        if len(positions) < 2:
            return 0.0
        
        # Calculate current correlation benefit
        returns_matrix = np.array([pos.returns[-252:] if len(pos.returns) >= 252 else pos.returns 
                                  for pos in positions])
        
        if returns_matrix.size == 0:
            return 0.0
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_matrix)
        
        # Simulate correlation breakdown (correlations approach 1)
        breakdown_corr = np.ones_like(corr_matrix) * 0.8
        np.fill_diagonal(breakdown_corr, 1.0)
        
        # Calculate impact
        correlation_impact = np.sum(np.abs(corr_matrix - breakdown_corr)) / 2
        
        return correlation_impact
    
    def _interest_rate_shock_scenario(self, positions: List[Position]) -> float:
        """Interest rate shock scenario: 2% rate increase."""
        # This would typically affect bond positions more
        # For simplicity, assume 5% impact on total portfolio
        total_value = sum(pos.market_value for pos in positions)
        
        return total_value * 0.05
    
    def _liquidity_crisis_scenario(self, positions: List[Position]) -> float:
        """Liquidity crisis scenario: 50% haircut on illiquid positions."""
        # Assume 20% of portfolio is illiquid
        total_value = sum(pos.market_value for pos in positions)
        illiquid_value = total_value * 0.20
        
        haircut_loss = illiquid_value * 0.50
        
        return haircut_loss
    
    def _calculate_portfolio_returns(self, positions: List[Position]) -> np.ndarray:
        """Calculate portfolio returns."""
        if not positions:
            return np.array([])
        
        # Align returns to same length
        min_length = min(len(pos.returns) for pos in positions)
        aligned_returns = []
        
        for position in positions:
            if len(position.returns) >= min_length:
                aligned_returns.append(position.returns[-min_length:])
        
        if not aligned_returns:
            return np.array([])
        
        # Calculate weighted portfolio returns
        weights = np.array([pos.weight for pos in positions[:len(aligned_returns)]])
        weights = weights / np.sum(weights)
        
        portfolio_returns = np.sum([ret * weight for ret, weight in zip(aligned_returns, weights)], axis=0)
        
        return portfolio_returns

class CorrelationMonitor:
    """Correlation monitoring system."""
    
    def __init__(self, lookback_period: int = 60):
        """
        Initialize correlation monitor.
        
        Args:
            lookback_period: Lookback period for correlation calculation
        """
        self.lookback_period = lookback_period
        self.correlation_history = {}
        self.correlation_alerts = []
    
    def calculate_correlations(self, positions: List[Position]) -> pd.DataFrame:
        """Calculate correlation matrix for positions."""
        if len(positions) < 2:
            return pd.DataFrame()
        
        try:
            # Align returns to same length
            min_length = min(len(pos.returns) for pos in positions)
            if min_length < 30:  # Need minimum data for correlation
                return pd.DataFrame()
            
            # Create returns matrix
            symbols = [pos.symbol for pos in positions]
            returns_data = {}
            
            for position in positions:
                if len(position.returns) >= min_length:
                    returns_data[position.symbol] = position.returns[-min_length:]
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            # Calculate correlation matrix
            returns_df = pd.DataFrame(returns_data)
            corr_matrix = returns_df.corr()
            
            # Store correlation history
            timestamp = datetime.now()
            self.correlation_history[timestamp] = corr_matrix.copy()
            
            # Keep only recent history
            if len(self.correlation_history) > 100:
                oldest_key = min(self.correlation_history.keys())
                del self.correlation_history[oldest_key]
            
            return corr_matrix
        
        except Exception as e:
            logger.error("Correlation calculation error", error=str(e))
            return pd.DataFrame()
    
    def detect_correlation_spikes(self, corr_matrix: pd.DataFrame, 
                                threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Detect correlation spikes above threshold."""
        spikes = []
        
        if corr_matrix.empty:
            return spikes
        
        try:
            # Get upper triangle of correlation matrix
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find correlations above threshold
            high_correlations = upper_triangle[upper_triangle > threshold]
            
            for (symbol1, symbol2), correlation in high_correlations.items():
                spike = {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'correlation': correlation,
                    'timestamp': datetime.now(),
                    'threshold': threshold
                }
                spikes.append(spike)
                
                # Add to alerts
                self.correlation_alerts.append(spike)
            
            return spikes
        
        except Exception as e:
            logger.error("Correlation spike detection error", error=str(e))
            return []
    
    def get_correlation_trends(self, symbol1: str, symbol2: str, 
                             days: int = 30) -> List[float]:
        """Get correlation trends for specific pair."""
        correlations = []
        
        for timestamp, corr_matrix in self.correlation_history.items():
            if symbol1 in corr_matrix.index and symbol2 in corr_matrix.columns:
                corr_value = corr_matrix.loc[symbol1, symbol2]
                if not pd.isna(corr_value):
                    correlations.append(corr_value)
        
        return correlations[-days:] if correlations else []

class RiskManager:
    """Main real-time risk management system."""
    
    def __init__(self, risk_limits: RiskLimits):
        """
        Initialize risk manager.
        
        Args:
            risk_limits: Risk limits configuration
        """
        self.risk_limits = risk_limits
        self.var_calculator = VaRCalculator(
            confidence_level=risk_limits.var_confidence_level,
            lookback_period=risk_limits.lookback_period
        )
        self.stress_tester = StressTester()
        self.correlation_monitor = CorrelationMonitor()
        
        self.portfolios = {}
        self.risk_alerts = []
        self.alert_callbacks = []
        
        self.running = False
        self.monitoring_thread = None
    
    def add_portfolio(self, portfolio_id: str, positions: List[Position]):
        """Add portfolio for risk monitoring."""
        self.portfolios[portfolio_id] = positions
        logger.info("Portfolio added for risk monitoring", portfolio_id=portfolio_id)
    
    def update_portfolio(self, portfolio_id: str, positions: List[Position]):
        """Update portfolio positions."""
        self.portfolios[portfolio_id] = positions
    
    def calculate_portfolio_risk(self, portfolio_id: str) -> Optional[RiskMetrics]:
        """Calculate comprehensive risk metrics for portfolio."""
        if portfolio_id not in self.portfolios:
            return None
        
        positions = self.portfolios[portfolio_id]
        
        try:
            # Calculate VaR
            var, cvar = self.var_calculator.calculate_portfolio_var(positions)
            
            # Calculate other risk metrics
            volatility = self._calculate_volatility(positions)
            beta = self._calculate_beta(positions)
            sharpe_ratio = self._calculate_sharpe_ratio(positions)
            max_drawdown = self._calculate_max_drawdown(positions)
            
            # Calculate correlation matrix
            correlation_matrix = self.correlation_monitor.calculate_correlations(positions)
            
            # Calculate concentration risk
            concentration_risk = self._calculate_concentration_risk(positions)
            
            # Run stress tests
            stress_test_results = self.stress_tester.run_stress_tests(positions)
            
            # Create risk metrics
            risk_metrics = RiskMetrics(
                portfolio_id=portfolio_id,
                timestamp=datetime.now(),
                var=var,
                cvar=cvar,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                correlation_matrix=correlation_matrix,
                concentration_risk=concentration_risk,
                stress_test_results=stress_test_results
            )
            
            return risk_metrics
        
        except Exception as e:
            logger.error("Portfolio risk calculation error", portfolio_id=portfolio_id, error=str(e))
            return None
    
    def check_risk_limits(self, portfolio_id: str) -> List[RiskAlert]:
        """Check portfolio against risk limits."""
        risk_metrics = self.calculate_portfolio_risk(portfolio_id)
        if not risk_metrics:
            return []
        
        alerts = []
        
        # Check VaR limit
        if risk_metrics.var > self.risk_limits.portfolio_var_limit:
            alert = RiskAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.VAR_BREACH,
                risk_level=RiskLevel.HIGH,
                message=f"Portfolio VaR {risk_metrics.var:.2f} exceeds limit {self.risk_limits.portfolio_var_limit:.2f}",
                portfolio_id=portfolio_id,
                metric_value=risk_metrics.var,
                limit_value=self.risk_limits.portfolio_var_limit
            )
            alerts.append(alert)
        
        # Check concentration risk
        if risk_metrics.concentration_risk > self.risk_limits.max_concentration:
            alert = RiskAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.CONCENTRATION_BREACH,
                risk_level=RiskLevel.MEDIUM,
                message=f"Concentration risk {risk_metrics.concentration_risk:.2f} exceeds limit {self.risk_limits.max_concentration:.2f}",
                portfolio_id=portfolio_id,
                metric_value=risk_metrics.concentration_risk,
                limit_value=self.risk_limits.max_concentration
            )
            alerts.append(alert)
        
        # Check drawdown
        if abs(risk_metrics.max_drawdown) > self.risk_limits.max_drawdown:
            alert = RiskAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.DRAWDOWN_BREACH,
                risk_level=RiskLevel.CRITICAL,
                message=f"Max drawdown {risk_metrics.max_drawdown:.2f} exceeds limit {self.risk_limits.max_drawdown:.2f}",
                portfolio_id=portfolio_id,
                metric_value=abs(risk_metrics.max_drawdown),
                limit_value=self.risk_limits.max_drawdown
            )
            alerts.append(alert)
        
        # Check volatility
        if risk_metrics.volatility > self.risk_limits.max_volatility:
            alert = RiskAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.VOLATILITY_SPIKE,
                risk_level=RiskLevel.MEDIUM,
                message=f"Volatility {risk_metrics.volatility:.2f} exceeds limit {self.risk_limits.max_volatility:.2f}",
                portfolio_id=portfolio_id,
                metric_value=risk_metrics.volatility,
                limit_value=self.risk_limits.max_volatility
            )
            alerts.append(alert)
        
        # Check correlation spikes
        correlation_spikes = self.correlation_monitor.detect_correlation_spikes(
            risk_metrics.correlation_matrix, 
            threshold=self.risk_limits.max_correlation
        )
        
        for spike in correlation_spikes:
            alert = RiskAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.CORRELATION_SPIKE,
                risk_level=RiskLevel.MEDIUM,
                message=f"High correlation {spike['correlation']:.2f} between {spike['symbol1']} and {spike['symbol2']}",
                portfolio_id=portfolio_id,
                metric_value=spike['correlation'],
                limit_value=self.risk_limits.max_correlation
            )
            alerts.append(alert)
        
        # Add alerts to history
        self.risk_alerts.extend(alerts)
        
        # Call alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error("Risk alert callback error", error=str(e))
        
        return alerts
    
    def _calculate_volatility(self, positions: List[Position]) -> float:
        """Calculate portfolio volatility."""
        if not positions:
            return 0.0
        
        try:
            portfolio_returns = self._calculate_portfolio_returns(positions)
            if len(portfolio_returns) == 0:
                return 0.0
            
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            return volatility
        
        except Exception as e:
            logger.error("Volatility calculation error", error=str(e))
            return 0.0
    
    def _calculate_beta(self, positions: List[Position]) -> float:
        """Calculate portfolio beta."""
        if not positions:
            return 0.0
        
        try:
            # This is a simplified beta calculation
            # In production, you'd use market index returns
            portfolio_returns = self._calculate_portfolio_returns(positions)
            if len(portfolio_returns) == 0:
                return 0.0
            
            # Assume market returns (simplified)
            market_returns = np.random.normal(0.0001, 0.015, len(portfolio_returns))
            
            # Calculate beta
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            beta = covariance / market_variance if market_variance > 0 else 0.0
            
            return beta
        
        except Exception as e:
            logger.error("Beta calculation error", error=str(e))
            return 0.0
    
    def _calculate_sharpe_ratio(self, positions: List[Position]) -> float:
        """Calculate Sharpe ratio."""
        if not positions:
            return 0.0
        
        try:
            portfolio_returns = self._calculate_portfolio_returns(positions)
            if len(portfolio_returns) == 0:
                return 0.0
            
            mean_return = np.mean(portfolio_returns) * 252  # Annualized
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            
            # Assume risk-free rate of 2%
            risk_free_rate = 0.02
            
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0.0
            
            return sharpe_ratio
        
        except Exception as e:
            logger.error("Sharpe ratio calculation error", error=str(e))
            return 0.0
    
    def _calculate_max_drawdown(self, positions: List[Position]) -> float:
        """Calculate maximum drawdown."""
        if not positions:
            return 0.0
        
        try:
            portfolio_returns = self._calculate_portfolio_returns(positions)
            if len(portfolio_returns) == 0:
                return 0.0
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            max_drawdown = np.min(drawdown)
            
            return max_drawdown
        
        except Exception as e:
            logger.error("Max drawdown calculation error", error=str(e))
            return 0.0
    
    def _calculate_concentration_risk(self, positions: List[Position]) -> float:
        """Calculate concentration risk (Herfindahl index)."""
        if not positions:
            return 0.0
        
        try:
            total_value = sum(pos.market_value for pos in positions)
            if total_value == 0:
                return 0.0
            
            # Calculate weights
            weights = [pos.market_value / total_value for pos in positions]
            
            # Calculate Herfindahl index
            concentration = sum(weight ** 2 for weight in weights)
            
            return concentration
        
        except Exception as e:
            logger.error("Concentration risk calculation error", error=str(e))
            return 0.0
    
    def _calculate_portfolio_returns(self, positions: List[Position]) -> np.ndarray:
        """Calculate portfolio returns."""
        if not positions:
            return np.array([])
        
        # Align returns to same length
        min_length = min(len(pos.returns) for pos in positions)
        aligned_returns = []
        
        for position in positions:
            if len(position.returns) >= min_length:
                aligned_returns.append(position.returns[-min_length:])
        
        if not aligned_returns:
            return np.array([])
        
        # Calculate weighted portfolio returns
        weights = np.array([pos.weight for pos in positions[:len(aligned_returns)]])
        weights = weights / np.sum(weights)
        
        portfolio_returns = np.sum([ret * weight for ret, weight in zip(aligned_returns, weights)], axis=0)
        
        return portfolio_returns
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Add risk alert callback."""
        self.alert_callbacks.append(callback)
    
    def start(self):
        """Start risk monitoring."""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_worker)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Risk manager started")
    
    def stop(self):
        """Stop risk monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Risk manager stopped")
    
    def _monitoring_worker(self):
        """Risk monitoring worker."""
        while self.running:
            try:
                # Check all portfolios
                for portfolio_id in self.portfolios.keys():
                    alerts = self.check_risk_limits(portfolio_id)
                    if alerts:
                        logger.info(f"Risk alerts generated for portfolio {portfolio_id}", 
                                  alert_count=len(alerts))
                
                time.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                logger.error("Risk monitoring error", error=str(e))
    
    def get_risk_summary(self, portfolio_id: str = None) -> Dict[str, Any]:
        """Get risk summary for portfolio(s)."""
        if portfolio_id:
            portfolios = [portfolio_id] if portfolio_id in self.portfolios else []
        else:
            portfolios = list(self.portfolios.keys())
        
        summary = {
            'total_portfolios': len(portfolios),
            'risk_metrics': {},
            'alerts': len(self.risk_alerts),
            'timestamp': datetime.now().isoformat()
        }
        
        for pid in portfolios:
            risk_metrics = self.calculate_portfolio_risk(pid)
            if risk_metrics:
                summary['risk_metrics'][pid] = {
                    'var': risk_metrics.var,
                    'cvar': risk_metrics.cvar,
                    'volatility': risk_metrics.volatility,
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'concentration_risk': risk_metrics.concentration_risk
                }
        
        return summary

def create_risk_manager(risk_limits: RiskLimits) -> RiskManager:
    """
    Create a real-time risk management system.
    
    Args:
        risk_limits: Risk limits configuration
        
    Returns:
        RiskManager instance
    """
    return RiskManager(risk_limits)

if __name__ == "__main__":
    # Demo of risk management system
    risk_limits = RiskLimits(
        portfolio_var_limit=10000.0,
        position_var_limit=5000.0,
        max_position_size=100000.0,
        max_concentration=0.3,
        max_correlation=0.8,
        max_drawdown=0.15,
        max_volatility=0.25
    )
    
    risk_manager = create_risk_manager(risk_limits)
    
    # Add sample portfolio
    sample_positions = [
        Position(
            symbol="AAPL",
            quantity=100,
            average_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            weight=0.4,
            returns=np.random.normal(0.001, 0.02, 252)
        ),
        Position(
            symbol="GOOGL",
            quantity=10,
            average_price=2800.0,
            current_price=2850.0,
            market_value=28500.0,
            weight=0.6,
            returns=np.random.normal(0.001, 0.025, 252)
        )
    ]
    
    risk_manager.add_portfolio("portfolio_001", sample_positions)
    
    # Start risk monitoring
    risk_manager.start()
    
    print("Risk management system created successfully!")
    print(f"Risk limits: VaR={risk_limits.portfolio_var_limit}, Max Drawdown={risk_limits.max_drawdown}")
    print(f"Portfolio: {len(sample_positions)} positions")
    
    # Calculate risk metrics
    risk_metrics = risk_manager.calculate_portfolio_risk("portfolio_001")
    if risk_metrics:
        print(f"Portfolio VaR: ${risk_metrics.var:.2f}")
        print(f"Portfolio CVaR: ${risk_metrics.cvar:.2f}")
        print(f"Volatility: {risk_metrics.volatility:.2%}")
        print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    
    # Check risk limits
    alerts = risk_manager.check_risk_limits("portfolio_001")
    print(f"Risk alerts: {len(alerts)}")