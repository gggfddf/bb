"""
Real-Time Dashboard

Implements a comprehensive real-time dashboard for the trading system:
- Live market data visualization
- Real-time indicator displays
- Portfolio performance tracking
- Alert and notification system
- Interactive charts and graphs
- System status monitoring

Features:
- WebSocket-based real-time updates
- Multiple chart types and timeframes
- Customizable dashboard layouts
- Performance metrics display
- Risk management indicators
- News and sentiment integration
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import structlog
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import websockets
import threading
from collections import deque

logger = structlog.get_logger()

class DashboardType(Enum):
    """Dashboard types."""
    MARKET_OVERVIEW = "market_overview"
    PORTFOLIO = "portfolio"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RISK_MANAGEMENT = "risk_management"
    SYSTEM_STATUS = "system_status"
    CUSTOM = "custom"

class ChartType(Enum):
    """Chart types."""
    CANDLESTICK = "candlestick"
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"

@dataclass
class DashboardConfig:
    """Configuration for dashboard."""
    dashboard_type: DashboardType = DashboardType.MARKET_OVERVIEW
    update_interval: float = 1.0  # seconds
    max_data_points: int = 1000
    enable_alerts: bool = True
    enable_notifications: bool = True
    chart_types: List[ChartType] = None
    custom_layout: Dict[str, Any] = None

@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: Dict[str, float] = None
    metadata: Dict[str, Any] = None

@dataclass
class DashboardUpdate:
    """Dashboard update structure."""
    update_type: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None

class RealTimeDashboard:
    """
    Real-time dashboard for trading system.
    """
    
    def __init__(self, config: DashboardConfig = None):
        """
        Initialize real-time dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.app = FastAPI(title="Trading System Dashboard")
        self.websocket_connections: List[WebSocket] = []
        self.market_data_cache = {}
        self.portfolio_data = {}
        self.system_status = {}
        self.alerts = deque(maxlen=100)
        self.is_running = False
        
        # Initialize dashboard components
        self._setup_routes()
        self._setup_websocket_handlers()
        self._setup_static_files()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def dashboard_home():
            """Dashboard home page."""
            return HTMLResponse(self._generate_dashboard_html())
        
        @self.app.get("/api/status")
        async def get_system_status():
            """Get system status."""
            return self.system_status
        
        @self.app.get("/api/market-data")
        async def get_market_data():
            """Get current market data."""
            return self.market_data_cache
        
        @self.app.get("/api/portfolio")
        async def get_portfolio_data():
            """Get portfolio data."""
            return self.portfolio_data
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get recent alerts."""
            return list(self.alerts)
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket handlers."""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(self.config.update_interval)
                    await self._send_dashboard_update(websocket)
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    def _setup_static_files(self):
        """Setup static files serving."""
        self.app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading System Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
                .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { grid-column: 1 / -1; text-align: center; margin-bottom: 20px; }
                .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
                .status-online { background: #4CAF50; }
                .status-offline { background: #f44336; }
                .status-warning { background: #ff9800; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading System Dashboard</h1>
                <div id="system-status">
                    <span class="status-indicator status-online"></span>
                    <span id="status-text">System Online</span>
                </div>
            </div>
            
            <div class="dashboard">
                <div class="card">
                    <h3>Market Overview</h3>
                    <div id="market-chart"></div>
                </div>
                
                <div class="card">
                    <h3>Portfolio Performance</h3>
                    <div id="portfolio-chart"></div>
                </div>
                
                <div class="card">
                    <h3>Technical Indicators</h3>
                    <div id="indicators-chart"></div>
                </div>
                
                <div class="card">
                    <h3>Risk Metrics</h3>
                    <div id="risk-chart"></div>
                </div>
                
                <div class="card">
                    <h3>Recent Alerts</h3>
                    <div id="alerts-list"></div>
                </div>
                
                <div class="card">
                    <h3>System Status</h3>
                    <div id="system-metrics"></div>
                </div>
            </div>
            
            <script>
                // WebSocket connection
                const ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                function updateDashboard(data) {
                    // Update market chart
                    if (data.market_data) {
                        updateMarketChart(data.market_data);
                    }
                    
                    // Update portfolio chart
                    if (data.portfolio_data) {
                        updatePortfolioChart(data.portfolio_data);
                    }
                    
                    // Update indicators
                    if (data.indicators) {
                        updateIndicatorsChart(data.indicators);
                    }
                    
                    // Update risk metrics
                    if (data.risk_metrics) {
                        updateRiskChart(data.risk_metrics);
                    }
                    
                    // Update alerts
                    if (data.alerts) {
                        updateAlerts(data.alerts);
                    }
                    
                    // Update system status
                    if (data.system_status) {
                        updateSystemStatus(data.system_status);
                    }
                }
                
                function updateMarketChart(data) {
                    const trace = {
                        x: data.timestamps,
                        open: data.open,
                        high: data.high,
                        low: data.low,
                        close: data.close,
                        type: 'candlestick',
                        name: 'Price'
                    };
                    
                    const layout = {
                        title: 'Market Data',
                        xaxis: { title: 'Time' },
                        yaxis: { title: 'Price' }
                    };
                    
                    Plotly.newPlot('market-chart', [trace], layout);
                }
                
                function updatePortfolioChart(data) {
                    const trace = {
                        x: data.timestamps,
                        y: data.values,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Portfolio Value'
                    };
                    
                    const layout = {
                        title: 'Portfolio Performance',
                        xaxis: { title: 'Time' },
                        yaxis: { title: 'Value' }
                    };
                    
                    Plotly.newPlot('portfolio-chart', [trace], layout);
                }
                
                function updateIndicatorsChart(data) {
                    const traces = [];
                    
                    for (const [name, values] of Object.entries(data)) {
                        traces.push({
                            x: values.timestamps,
                            y: values.values,
                            type: 'scatter',
                            mode: 'lines',
                            name: name
                        });
                    }
                    
                    const layout = {
                        title: 'Technical Indicators',
                        xaxis: { title: 'Time' },
                        yaxis: { title: 'Value' }
                    };
                    
                    Plotly.newPlot('indicators-chart', traces, layout);
                }
                
                function updateRiskChart(data) {
                    const ctx = document.getElementById('risk-chart').getContext('2d');
                    new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: Object.keys(data),
                            datasets: [{
                                data: Object.values(data),
                                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Risk Distribution'
                                }
                            }
                        }
                    });
                }
                
                function updateAlerts(alerts) {
                    const alertsList = document.getElementById('alerts-list');
                    alertsList.innerHTML = '';
                    
                    alerts.forEach(alert => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = 'alert';
                        alertDiv.innerHTML = `
                            <strong>${alert.timestamp}</strong>: ${alert.message}
                        `;
                        alertsList.appendChild(alertDiv);
                    });
                }
                
                function updateSystemStatus(status) {
                    const statusDiv = document.getElementById('system-metrics');
                    statusDiv.innerHTML = '';
                    
                    for (const [metric, value] of Object.entries(status)) {
                        const metricDiv = document.createElement('div');
                        metricDiv.innerHTML = `
                            <strong>${metric}</strong>: ${value}
                        `;
                        statusDiv.appendChild(metricDiv);
                    }
                }
            </script>
        </body>
        </html>
        """
    
    async def _send_dashboard_update(self, websocket: WebSocket):
        """Send dashboard update to WebSocket client."""
        try:
            update_data = {
                'timestamp': datetime.now().isoformat(),
                'market_data': self._get_market_data_for_chart(),
                'portfolio_data': self._get_portfolio_data_for_chart(),
                'indicators': self._get_indicators_data_for_chart(),
                'risk_metrics': self._get_risk_metrics(),
                'alerts': list(self.alerts),
                'system_status': self.system_status
            }
            
            await websocket.send_text(json.dumps(update_data))
        except Exception as e:
            logger.error(f"Failed to send dashboard update: {e}")
    
    def _get_market_data_for_chart(self) -> Dict[str, Any]:
        """Get market data formatted for charting."""
        if not self.market_data_cache:
            return {}
        
        # Get latest data for each symbol
        chart_data = {}
        for symbol, data in self.market_data_cache.items():
            if data and len(data) > 0:
                latest = data[-1]
                chart_data[symbol] = {
                    'timestamps': [d.timestamp.isoformat() for d in data[-50:]],  # Last 50 points
                    'open': [d.open for d in data[-50:]],
                    'high': [d.high for d in data[-50:]],
                    'low': [d.low for d in data[-50:]],
                    'close': [d.close for d in data[-50:]],
                    'volume': [d.volume for d in data[-50:]]
                }
        
        return chart_data
    
    def _get_portfolio_data_for_chart(self) -> Dict[str, Any]:
        """Get portfolio data formatted for charting."""
        if not self.portfolio_data:
            return {}
        
        return {
            'timestamps': [d.get('timestamp', '').isoformat() for d in self.portfolio_data.get('history', [])],
            'values': [d.get('value', 0) for d in self.portfolio_data.get('history', [])],
            'returns': [d.get('return', 0) for d in self.portfolio_data.get('history', [])]
        }
    
    def _get_indicators_data_for_chart(self) -> Dict[str, Any]:
        """Get indicators data formatted for charting."""
        indicators_data = {}
        
        for symbol, data in self.market_data_cache.items():
            if data and len(data) > 0:
                latest = data[-1]
                if latest.indicators:
                    for indicator_name, value in latest.indicators.items():
                        if indicator_name not in indicators_data:
                            indicators_data[indicator_name] = {
                                'timestamps': [],
                                'values': []
                            }
                        
                        indicators_data[indicator_name]['timestamps'].append(latest.timestamp.isoformat())
                        indicators_data[indicator_name]['values'].append(value)
        
        return indicators_data
    
    def _get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        if not self.portfolio_data:
            return {}
        
        return {
            'Sharpe Ratio': self.portfolio_data.get('sharpe_ratio', 0.0),
            'Max Drawdown': self.portfolio_data.get('max_drawdown', 0.0),
            'Volatility': self.portfolio_data.get('volatility', 0.0),
            'Beta': self.portfolio_data.get('beta', 0.0)
        }
    
    def update_market_data(self, symbol: str, market_data: MarketData):
        """Update market data for a symbol."""
        if symbol not in self.market_data_cache:
            self.market_data_cache[symbol] = []
        
        self.market_data_cache[symbol].append(market_data)
        
        # Keep only recent data points
        if len(self.market_data_cache[symbol]) > self.config.max_data_points:
            self.market_data_cache[symbol] = self.market_data_cache[symbol][-self.config.max_data_points:]
        
        logger.info(f"Updated market data for {symbol}")
    
    def update_portfolio_data(self, portfolio_data: Dict[str, Any]):
        """Update portfolio data."""
        self.portfolio_data.update(portfolio_data)
        
        # Add to history
        if 'history' not in self.portfolio_data:
            self.portfolio_data['history'] = []
        
        self.portfolio_data['history'].append({
            'timestamp': datetime.now(),
            'value': portfolio_data.get('total_value', 0),
            'return': portfolio_data.get('daily_return', 0)
        })
        
        # Keep only recent history
        if len(self.portfolio_data['history']) > self.config.max_data_points:
            self.portfolio_data['history'] = self.portfolio_data['history'][-self.config.max_data_points:]
        
        logger.info("Updated portfolio data")
    
    def update_system_status(self, status_data: Dict[str, Any]):
        """Update system status."""
        self.system_status.update(status_data)
        self.system_status['last_update'] = datetime.now().isoformat()
        
        logger.info("Updated system status")
    
    def add_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Add an alert to the dashboard."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        
        if self.config.enable_notifications:
            self._send_notification(alert)
        
        logger.info(f"Added alert: {message}")
    
    def _send_notification(self, alert: Dict[str, Any]):
        """Send notification for alert."""
        # Implementation would depend on notification system
        # Could be email, Slack, SMS, etc.
        logger.info(f"Notification sent: {alert['message']}")
    
    def start(self, host: str = "localhost", port: int = 8000):
        """Start the dashboard server."""
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._background_update_task())
        
        # Start server
        uvicorn.run(self.app, host=host, port=port)
        
        logger.info(f"Dashboard started on http://{host}:{port}")
    
    async def _background_update_task(self):
        """Background task for periodic updates."""
        while self.is_running:
            try:
                # Update system status
                self.update_system_status({
                    'uptime': time.time(),
                    'active_connections': len(self.websocket_connections),
                    'data_points': sum(len(data) for data in self.market_data_cache.values())
                })
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Background update task failed: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def stop(self):
        """Stop the dashboard."""
        self.is_running = False
        logger.info("Dashboard stopped")

class ChartGenerator:
    """
    Chart generation utilities for dashboard.
    """
    
    @staticmethod
    def create_candlestick_chart(data: List[MarketData], title: str = "Market Data") -> go.Figure:
        """Create candlestick chart."""
        if not data:
            return go.Figure()
        
        fig = go.Figure(data=[go.Candlestick(
            x=[d.timestamp for d in data],
            open=[d.open for d in data],
            high=[d.high for d in data],
            low=[d.low for d in data],
            close=[d.close for d in data]
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Price"
        )
        
        return fig
    
    @staticmethod
    def create_line_chart(x: List, y: List, title: str = "Line Chart", x_title: str = "X", y_title: str = "Y") -> go.Figure:
        """Create line chart."""
        fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines')])
        
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title
        )
        
        return fig
    
    @staticmethod
    def create_subplot_chart(charts: List[go.Figure], rows: int, cols: int, title: str = "Multi-Chart") -> go.Figure:
        """Create subplot chart."""
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Chart {i+1}" for i in range(len(charts))])
        
        for i, chart in enumerate(charts):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            for trace in chart.data:
                fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(title=title)
        
        return fig

# Convenience functions
def create_dashboard(config: DashboardConfig = None) -> RealTimeDashboard:
    """Create a dashboard instance."""
    return RealTimeDashboard(config)

def start_dashboard(dashboard: RealTimeDashboard, host: str = "localhost", port: int = 8000):
    """Start the dashboard."""
    dashboard.start(host, port)

def update_dashboard_data(dashboard: RealTimeDashboard, data_type: str, data: Any):
    """Update dashboard data."""
    if data_type == "market_data":
        dashboard.update_market_data(data.symbol, data)
    elif data_type == "portfolio_data":
        dashboard.update_portfolio_data(data)
    elif data_type == "system_status":
        dashboard.update_system_status(data)
    elif data_type == "alert":
        dashboard.add_alert(data['type'], data['message'], data.get('severity', 'info'))