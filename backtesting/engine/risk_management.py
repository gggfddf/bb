"""
Risk Management Framework

A comprehensive risk management framework for trading strategies that includes:
- Position sizing algorithms
- Stop-loss and take-profit mechanisms
- Risk limits and portfolio constraints
- Real-time risk monitoring
- Portfolio-level risk management

Features:
- Multiple position sizing strategies (fixed, Kelly, volatility-based)
- Dynamic stop-loss and take-profit levels
- Portfolio-level risk limits and constraints
- Real-time risk monitoring and alerts
- Integration with backtesting engine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
from scipy import stats

logger = structlog.get_logger()

class RiskMetric(Enum):
    """Risk metrics for monitoring."""
    VAR = "var"
    CVAR = "cvar"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    BETA = "beta"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"

class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    OPTIMAL_F = "optimal_f"

class StopLossType(Enum):
    """Stop-loss types."""
    FIXED = "fixed"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"

@dataclass
class RiskLimit:
    """Risk limit configuration."""
    max_position_size: float = 0.1  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    max_drawdown: float = 0.15  # 15% max drawdown
    max_var: float = 0.03  # 3% VaR limit
    max_concentration: float = 0.25  # 25% max concentration
    max_leverage: float = 2.0  # 2x max leverage
    min_correlation: float = -0.7  # Minimum correlation threshold

@dataclass
class PositionSizingResult:
    """Result of position sizing calculation."""
    position_size: float
    allocation_pct: float
    risk_amount: float
    method: PositionSizingMethod
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StopLossResult:
    """Result of stop-loss calculation."""
    stop_price: float
    stop_type: StopLossType
    trigger_condition: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskMetrics:
    """Current risk metrics."""
    portfolio_value: float
    total_risk: float
    var_95: float
    cvar_95: float
    volatility: float
    current_drawdown: float
    max_drawdown: float
    beta: float
    sharpe_ratio: float
    concentration_risk: float
    leverage: float
    timestamp: datetime = field(default_factory=datetime.now)

class PositionSizer:
    """Position sizing algorithms."""
    
    def __init__(self, portfolio_value: float, risk_limits: RiskLimit):
        self.portfolio_value = portfolio_value
        self.risk_limits = risk_limits
    
    def calculate_fixed_size(self, 
                           allocation_pct: float,
                           current_price: float) -> PositionSizingResult:
        """
        Calculate fixed position size.
        
        Args:
            allocation_pct: Percentage of portfolio to allocate
            current_price: Current asset price
        
        Returns:
            PositionSizingResult object
        """
        # Ensure allocation doesn't exceed limits
        allocation_pct = min(allocation_pct, self.risk_limits.max_position_size)
        
        # Calculate position size
        position_value = self.portfolio_value * allocation_pct
        position_size = position_value / current_price
        
        return PositionSizingResult(
            position_size=position_size,
            allocation_pct=allocation_pct,
            risk_amount=position_value,
            method=PositionSizingMethod.FIXED,
            parameters={'allocation_pct': allocation_pct}
        )
    
    def calculate_kelly_size(self,
                           win_rate: float,
                           avg_win: float,
                           avg_loss: float,
                           current_price: float) -> PositionSizingResult:
        """
        Calculate Kelly Criterion position size.
        
        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            current_price: Current asset price
        
        Returns:
            PositionSizingResult object
        """
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        if avg_loss == 0:
            kelly_fraction = 0.0
        else:
            b = avg_win / abs(avg_loss)
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        
        # Apply fractional Kelly (usually 1/4 or 1/2)
        fractional_kelly = kelly_fraction * 0.25
        
        # Ensure within limits
        allocation_pct = min(max(fractional_kelly, 0), self.risk_limits.max_position_size)
        
        position_value = self.portfolio_value * allocation_pct
        position_size = position_value / current_price
        
        return PositionSizingResult(
            position_size=position_size,
            allocation_pct=allocation_pct,
            risk_amount=position_value,
            method=PositionSizingMethod.KELLY,
            parameters={
                'kelly_fraction': kelly_fraction,
                'fractional_kelly': fractional_kelly,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
        )
    
    def calculate_volatility_size(self,
                                volatility: float,
                                target_volatility: float,
                                current_price: float) -> PositionSizingResult:
        """
        Calculate volatility-based position size.
        
        Args:
            volatility: Asset volatility
            target_volatility: Target portfolio volatility
            current_price: Current asset price
        
        Returns:
            PositionSizingResult object
        """
        if volatility == 0:
            allocation_pct = 0.0
        else:
            # Volatility targeting: position size inversely proportional to volatility
            allocation_pct = target_volatility / volatility
            allocation_pct = min(allocation_pct, self.risk_limits.max_position_size)
        
        position_value = self.portfolio_value * allocation_pct
        position_size = position_value / current_price
        
        return PositionSizingResult(
            position_size=position_size,
            allocation_pct=allocation_pct,
            risk_amount=position_value,
            method=PositionSizingMethod.VOLATILITY,
            parameters={
                'volatility': volatility,
                'target_volatility': target_volatility
            }
        )
    
    def calculate_optimal_f_size(self,
                               returns: np.ndarray,
                               current_price: float) -> PositionSizingResult:
        """
        Calculate Optimal f position size (Ralph Vince).
        
        Args:
            returns: Historical returns
            current_price: Current asset price
        
        Returns:
            PositionSizingResult object
        """
        if len(returns) == 0:
            return PositionSizingResult(
                position_size=0.0,
                allocation_pct=0.0,
                risk_amount=0.0,
                method=PositionSizingMethod.OPTIMAL_F
            )
        
        # Calculate TWR (Terminal Wealth Relative) for different f values
        f_values = np.linspace(0.01, 0.99, 99)
        twr_values = []
        
        for f in f_values:
            twr = 1.0
            for ret in returns:
                twr *= (1 + f * ret)
            twr_values.append(twr)
        
        # Find optimal f (maximum TWR)
        optimal_f = f_values[np.argmax(twr_values)]
        
        # Apply fractional optimal f
        fractional_f = optimal_f * 0.25
        allocation_pct = min(fractional_f, self.risk_limits.max_position_size)
        
        position_value = self.portfolio_value * allocation_pct
        position_size = position_value / current_price
        
        return PositionSizingResult(
            position_size=position_size,
            allocation_pct=allocation_pct,
            risk_amount=position_value,
            method=PositionSizingMethod.OPTIMAL_F,
            parameters={
                'optimal_f': optimal_f,
                'fractional_f': fractional_f,
                'max_twr': max(twr_values)
            }
        )

class StopLossManager:
    """Stop-loss and take-profit management."""
    
    def __init__(self, risk_limits: RiskLimit):
        self.risk_limits = risk_limits
    
    def calculate_fixed_stop(self,
                           entry_price: float,
                           position_type: str,
                           stop_pct: float) -> StopLossResult:
        """
        Calculate fixed percentage stop-loss.
        
        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'
            stop_pct: Stop percentage
        
        Returns:
            StopLossResult object
        """
        if position_type == 'long':
            stop_price = entry_price * (1 - stop_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.FIXED,
            trigger_condition=f"{position_type} position reaches {stop_pct:.2%} loss",
            parameters={'stop_pct': stop_pct}
        )
    
    def calculate_trailing_stop(self,
                              entry_price: float,
                              current_price: float,
                              position_type: str,
                              trail_pct: float,
                              highest_price: float = None,
                              lowest_price: float = None) -> StopLossResult:
        """
        Calculate trailing stop-loss.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            position_type: 'long' or 'short'
            trail_pct: Trailing percentage
            highest_price: Highest price since entry (for long)
            lowest_price: Lowest price since entry (for short)
        
        Returns:
            StopLossResult object
        """
        if position_type == 'long':
            if highest_price is None:
                highest_price = current_price
            else:
                highest_price = max(highest_price, current_price)
            
            stop_price = highest_price * (1 - trail_pct)
        else:
            if lowest_price is None:
                lowest_price = current_price
            else:
                lowest_price = min(lowest_price, current_price)
            
            stop_price = lowest_price * (1 + trail_pct)
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.TRAILING,
            trigger_condition=f"{position_type} position trails by {trail_pct:.2%}",
            parameters={
                'trail_pct': trail_pct,
                'highest_price': highest_price,
                'lowest_price': lowest_price
            }
        )
    
    def calculate_atr_stop(self,
                          entry_price: float,
                          position_type: str,
                          atr: float,
                          atr_multiplier: float = 2.0) -> StopLossResult:
        """
        Calculate ATR-based stop-loss.
        
        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'
            atr: Average True Range
            atr_multiplier: ATR multiplier
        
        Returns:
            StopLossResult object
        """
        atr_distance = atr * atr_multiplier
        
        if position_type == 'long':
            stop_price = entry_price - atr_distance
        else:
            stop_price = entry_price + atr_distance
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.ATR_BASED,
            trigger_condition=f"{position_type} position stops at {atr_multiplier}x ATR",
            parameters={
                'atr': atr,
                'atr_multiplier': atr_multiplier,
                'atr_distance': atr_distance
            }
        )
    
    def calculate_time_stop(self,
                           entry_time: datetime,
                           max_hold_days: int,
                           current_time: datetime = None) -> StopLossResult:
        """
        Calculate time-based stop-loss.
        
        Args:
            entry_time: Entry time
            max_hold_days: Maximum holding period in days
            current_time: Current time
        
        Returns:
            StopLossResult object
        """
        if current_time is None:
            current_time = datetime.now()
        
        time_elapsed = (current_time - entry_time).days
        
        # Return a special stop result for time-based stops
        return StopLossResult(
            stop_price=0.0,  # Not applicable for time stops
            stop_type=StopLossType.TIME_BASED,
            trigger_condition=f"Position held for {time_elapsed}/{max_hold_days} days",
            parameters={
                'entry_time': entry_time,
                'max_hold_days': max_hold_days,
                'time_elapsed': time_elapsed
            }
        )

class PortfolioRiskManager:
    """Portfolio-level risk management."""
    
    def __init__(self, risk_limits: RiskLimit):
        self.risk_limits = risk_limits
        self.positions = {}
        self.risk_history = []
    
    def add_position(self, symbol: str, position_data: Dict[str, Any]):
        """Add or update position."""
        self.positions[symbol] = position_data
    
    def remove_position(self, symbol: str):
        """Remove position."""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def calculate_portfolio_risk(self, 
                               returns: pd.DataFrame,
                               weights: Dict[str, float]) -> RiskMetrics:
        """
        Calculate portfolio risk metrics.
        
        Args:
            returns: Returns dataframe with symbols as columns
            weights: Position weights dictionary
        
        Returns:
            RiskMetrics object
        """
        if len(returns) == 0 or len(weights) == 0:
            return RiskMetrics(
                portfolio_value=0.0,
                total_risk=0.0,
                var_95=0.0,
                cvar_95=0.0,
                volatility=0.0,
                current_drawdown=0.0,
                max_drawdown=0.0,
                beta=0.0,
                sharpe_ratio=0.0,
                concentration_risk=0.0,
                leverage=0.0
            )
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns.index)
        for symbol, weight in weights.items():
            if symbol in returns.columns:
                portfolio_returns += returns[symbol] * weight
        
        # Calculate risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0
        max_drawdown = drawdown.min()
        
        # Beta (assuming market returns available)
        beta = 1.0  # Default value, would need market returns for actual calculation
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_returns = portfolio_returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0.0
        
        # Concentration risk
        concentration_risk = max(weights.values()) if weights else 0.0
        
        # Leverage
        leverage = sum(abs(weight) for weight in weights.values())
        
        # Portfolio value (simplified)
        portfolio_value = 1000000.0  # Would be calculated from actual positions
        
        # Total risk (simplified)
        total_risk = volatility
        
        metrics = RiskMetrics(
            portfolio_value=portfolio_value,
            total_risk=total_risk,
            var_95=var_95,
            cvar_95=cvar_95,
            volatility=volatility,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            concentration_risk=concentration_risk,
            leverage=leverage
        )
        
        # Store in history
        self.risk_history.append(metrics)
        
        return metrics
    
    def check_risk_limits(self, metrics: RiskMetrics) -> Dict[str, bool]:
        """
        Check if risk metrics exceed limits.
        
        Args:
            metrics: Current risk metrics
        
        Returns:
            Dictionary of limit violations
        """
        violations = {}
        violations['max_drawdown'] = bool(abs(metrics.current_drawdown) > self.risk_limits.max_drawdown)
        violations['max_var'] = bool(abs(metrics.var_95) > self.risk_limits.max_var)
        violations['max_concentration'] = bool(metrics.concentration_risk > self.risk_limits.max_concentration)
        violations['max_leverage'] = bool(metrics.leverage > self.risk_limits.max_leverage)
        return violations
    
    def generate_risk_report(self) -> str:
        """Generate comprehensive risk report."""
        if not self.risk_history:
            return "No risk history available"
        
        latest_metrics = self.risk_history[-1]
        violations = self.check_risk_limits(latest_metrics)
        
        report = "Portfolio Risk Report\n"
        report += "=" * 50 + "\n\n"
        
        # Current metrics
        report += f"Portfolio Value: ${latest_metrics.portfolio_value:,.2f}\n"
        report += f"Total Risk: {latest_metrics.total_risk:.4f}\n"
        report += f"Volatility: {latest_metrics.volatility:.4f}\n"
        report += f"VaR (95%): {latest_metrics.var_95:.4f}\n"
        report += f"CVaR (95%): {latest_metrics.cvar_95:.4f}\n"
        report += f"Current Drawdown: {latest_metrics.current_drawdown:.4f}\n"
        report += f"Max Drawdown: {latest_metrics.max_drawdown:.4f}\n"
        report += f"Sharpe Ratio: {latest_metrics.sharpe_ratio:.4f}\n"
        report += f"Concentration Risk: {latest_metrics.concentration_risk:.4f}\n"
        report += f"Leverage: {latest_metrics.leverage:.4f}\n\n"
        
        # Limit violations
        report += "Risk Limit Violations:\n"
        for limit, violated in violations.items():
            status = "VIOLATED" if violated else "OK"
            report += f"  {limit}: {status}\n"
        
        return report

class RiskMonitor:
    """Real-time risk monitoring system."""
    
    def __init__(self, risk_limits: RiskLimit, alert_threshold: float = 0.8):
        self.risk_limits = risk_limits
        self.alert_threshold = alert_threshold
        self.alerts = []
        self.monitoring_active = False
    
    def start_monitoring(self):
        """Start risk monitoring."""
        self.monitoring_active = True
        logger.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring."""
        self.monitoring_active = False
        logger.info("Risk monitoring stopped")
    
    def check_risk_alerts(self, metrics: RiskMetrics) -> List[str]:
        """
        Check for risk alerts.
        
        Args:
            metrics: Current risk metrics
        
        Returns:
            List of alert messages
        """
        alerts = []
        
        # Check drawdown alert
        drawdown_ratio = abs(metrics.current_drawdown) / self.risk_limits.max_drawdown
        if drawdown_ratio > self.alert_threshold:
            alerts.append(f"Drawdown alert: {metrics.current_drawdown:.2%} ({drawdown_ratio:.1%} of limit)")
        
        # Check VaR alert
        var_ratio = abs(metrics.var_95) / self.risk_limits.max_var
        if var_ratio > self.alert_threshold:
            alerts.append(f"VaR alert: {metrics.var_95:.2%} ({var_ratio:.1%} of limit)")
        
        # Check concentration alert
        concentration_ratio = metrics.concentration_risk / self.risk_limits.max_concentration
        if concentration_ratio > self.alert_threshold:
            alerts.append(f"Concentration alert: {metrics.concentration_risk:.2%} ({concentration_ratio:.1%} of limit)")
        
        # Check leverage alert
        leverage_ratio = metrics.leverage / self.risk_limits.max_leverage
        if leverage_ratio > self.alert_threshold:
            alerts.append(f"Leverage alert: {metrics.leverage:.2f}x ({leverage_ratio:.1%} of limit)")
        
        # Store alerts
        for alert in alerts:
            self.alerts.append({
                'timestamp': datetime.now(),
                'message': alert,
                'metrics': metrics
            })
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]

class RiskManagementFramework:
    """Main risk management framework."""
    
    def __init__(self, 
                 portfolio_value: float,
                 risk_limits: RiskLimit,
                 alert_threshold: float = 0.8):
        self.portfolio_value = portfolio_value
        self.risk_limits = risk_limits
        self.position_sizer = PositionSizer(portfolio_value, risk_limits)
        self.stop_loss_manager = StopLossManager(risk_limits)
        self.portfolio_risk_manager = PortfolioRiskManager(risk_limits)
        self.risk_monitor = RiskMonitor(risk_limits, alert_threshold)
        
        logger.info("Risk management framework initialized",
                   portfolio_value=portfolio_value,
                   max_drawdown_limit=risk_limits.max_drawdown,
                   max_var_limit=risk_limits.max_var)
    
    def calculate_position_size(self,
                              method: PositionSizingMethod,
                              **kwargs) -> PositionSizingResult:
        """
        Calculate position size using specified method.
        
        Args:
            method: Position sizing method
            **kwargs: Method-specific parameters
        
        Returns:
            PositionSizingResult object
        """
        if method == PositionSizingMethod.FIXED:
            return self.position_sizer.calculate_fixed_size(
                kwargs.get('allocation_pct', 0.05),
                kwargs.get('current_price', 100.0)
            )
        elif method == PositionSizingMethod.KELLY:
            return self.position_sizer.calculate_kelly_size(
                kwargs.get('win_rate', 0.5),
                kwargs.get('avg_win', 0.02),
                kwargs.get('avg_loss', -0.01),
                kwargs.get('current_price', 100.0)
            )
        elif method == PositionSizingMethod.VOLATILITY:
            return self.position_sizer.calculate_volatility_size(
                kwargs.get('volatility', 0.2),
                kwargs.get('target_volatility', 0.15),
                kwargs.get('current_price', 100.0)
            )
        elif method == PositionSizingMethod.OPTIMAL_F:
            return self.position_sizer.calculate_optimal_f_size(
                kwargs.get('returns', np.array([])),
                kwargs.get('current_price', 100.0)
            )
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
    
    def calculate_stop_loss(self,
                           stop_type: StopLossType,
                           **kwargs) -> StopLossResult:
        """
        Calculate stop-loss using specified method.
        
        Args:
            stop_type: Stop-loss type
            **kwargs: Method-specific parameters
        
        Returns:
            StopLossResult object
        """
        if stop_type == StopLossType.FIXED:
            return self.stop_loss_manager.calculate_fixed_stop(
                kwargs.get('entry_price', 100.0),
                kwargs.get('position_type', 'long'),
                kwargs.get('stop_pct', 0.05)
            )
        elif stop_type == StopLossType.TRAILING:
            return self.stop_loss_manager.calculate_trailing_stop(
                kwargs.get('entry_price', 100.0),
                kwargs.get('current_price', 100.0),
                kwargs.get('position_type', 'long'),
                kwargs.get('trail_pct', 0.05),
                kwargs.get('highest_price'),
                kwargs.get('lowest_price')
            )
        elif stop_type == StopLossType.ATR_BASED:
            return self.stop_loss_manager.calculate_atr_stop(
                kwargs.get('entry_price', 100.0),
                kwargs.get('position_type', 'long'),
                kwargs.get('atr', 2.0),
                kwargs.get('atr_multiplier', 2.0)
            )
        elif stop_type == StopLossType.TIME_BASED:
            return self.stop_loss_manager.calculate_time_stop(
                kwargs.get('entry_time', datetime.now()),
                kwargs.get('max_hold_days', 30)
            )
        else:
            raise ValueError(f"Unknown stop-loss type: {stop_type}")
    
    def update_portfolio_risk(self,
                            returns: pd.DataFrame,
                            weights: Dict[str, float]) -> RiskMetrics:
        """
        Update portfolio risk metrics.
        
        Args:
            returns: Returns dataframe
            weights: Position weights
        
        Returns:
            RiskMetrics object
        """
        metrics = self.portfolio_risk_manager.calculate_portfolio_risk(returns, weights)
        
        # Check for alerts
        alerts = self.risk_monitor.check_risk_alerts(metrics)
        if alerts:
            logger.warning("Risk alerts triggered", alerts=alerts)
        
        return metrics
    
    def check_risk_limits(self, metrics: RiskMetrics) -> Dict[str, bool]:
        """Check risk limit violations."""
        return self.portfolio_risk_manager.check_risk_limits(metrics)
    
    def generate_risk_report(self) -> str:
        """Generate comprehensive risk report."""
        return self.portfolio_risk_manager.generate_risk_report()
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent risk alerts."""
        return self.risk_monitor.get_recent_alerts(hours)

# Convenience functions
def create_risk_framework(portfolio_value: float = 1000000.0,
                         max_drawdown: float = 0.15,
                         max_var: float = 0.03) -> RiskManagementFramework:
    """Create a risk management framework with default settings."""
    risk_limits = RiskLimit(
        max_drawdown=max_drawdown,
        max_var=max_var
    )
    return RiskManagementFramework(portfolio_value, risk_limits)

def calculate_position_size_fixed(portfolio_value: float,
                                allocation_pct: float,
                                current_price: float) -> float:
    """Calculate fixed position size."""
    framework = create_risk_framework(portfolio_value)
    result = framework.calculate_position_size(
        PositionSizingMethod.FIXED,
        allocation_pct=allocation_pct,
        current_price=current_price
    )
    return result.position_size

def calculate_stop_loss_fixed(entry_price: float,
                            position_type: str,
                            stop_pct: float) -> float:
    """Calculate fixed stop-loss price."""
    framework = create_risk_framework()
    result = framework.calculate_stop_loss(
        StopLossType.FIXED,
        entry_price=entry_price,
        position_type=position_type,
        stop_pct=stop_pct
    )
    return result.stop_price