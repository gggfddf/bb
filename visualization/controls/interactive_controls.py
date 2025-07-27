#!/usr/bin/env python3
"""
Interactive Chart Controls Module

Implements comprehensive interactive chart controls for enhanced user experience:
- Zoom and pan controls
- Time range selectors
- Indicator toggle controls
- Chart type switchers
- Annotation tools
- Chart state management

Features:
- Interactive zoom and pan functionality
- Dynamic time range selection
- Real-time indicator toggling
- Chart type switching capabilities
- Annotation and drawing tools
- State management and persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button, Slider, CheckButtons, RadioButtons
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
logger = structlog.get_logger()

# Optional imports for Dash functionality
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash not available, interactive controls will be limited")

class ControlType(Enum):
    """Types of interactive controls."""
    ZOOM_PAN = "zoom_pan"
    TIME_RANGE = "time_range"
    INDICATOR_TOGGLE = "indicator_toggle"
    CHART_TYPE = "chart_type"
    ANNOTATION = "annotation"
    STATE_MANAGEMENT = "state_management"

class ChartType(Enum):
    """Available chart types."""
    CANDLESTICK = "candlestick"
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"

@dataclass
class ControlConfig:
    """Configuration for interactive controls."""
    control_type: ControlType
    enabled: bool = True
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None

@dataclass
class ChartState:
    """State management for charts."""
    zoom_level: float = 1.0
    pan_offset: Tuple[float, float] = (0.0, 0.0)
    time_range: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now() - timedelta(days=30), datetime.now()))
    visible_indicators: List[str] = field(default_factory=list)
    chart_type: ChartType = ChartType.CANDLESTICK
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    theme: str = "light"

class ZoomPanController:
    """Zoom and pan controls for charts."""
    
    def __init__(self, chart_figure: go.Figure):
        """
        Initialize zoom and pan controller.
        
        Args:
            chart_figure: Plotly figure to control
        """
        self.figure = chart_figure
        self.zoom_level = 1.0
        self.pan_offset = (0.0, 0.0)
        self.original_x_range = None
        self.original_y_range = None
        
        # Store original ranges
        if self.figure.data:
            x_values = []
            y_values = []
            for trace in self.figure.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    x_values.extend(trace.x)
                if hasattr(trace, 'y') and trace.y is not None:
                    y_values.extend(trace.y)
            
            if x_values:
                self.original_x_range = (min(x_values), max(x_values))
            if y_values:
                self.original_y_range = (min(y_values), max(y_values))
    
    def zoom_in(self, factor: float = 1.5):
        """Zoom in by specified factor."""
        self.zoom_level *= factor
        self._apply_zoom_pan()
    
    def zoom_out(self, factor: float = 1.5):
        """Zoom out by specified factor."""
        self.zoom_level /= factor
        self._apply_zoom_pan()
    
    def pan(self, dx: float, dy: float):
        """Pan chart by specified offsets."""
        self.pan_offset = (self.pan_offset[0] + dx, self.pan_offset[1] + dy)
        self._apply_zoom_pan()
    
    def reset_view(self):
        """Reset to original view."""
        self.zoom_level = 1.0
        self.pan_offset = (0.0, 0.0)
        self._apply_zoom_pan()
    
    def _apply_zoom_pan(self):
        """Apply zoom and pan transformations."""
        if not self.original_x_range or not self.original_y_range:
            return
        
        # Calculate new ranges
        x_range = self.original_x_range
        y_range = self.original_y_range
        
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        
        x_span = (x_range[1] - x_range[0]) / self.zoom_level
        y_span = (y_range[1] - y_range[0]) / self.zoom_level
        
        new_x_range = (x_center - x_span/2 + self.pan_offset[0], 
                      x_center + x_span/2 + self.pan_offset[0])
        new_y_range = (y_center - y_span/2 + self.pan_offset[1], 
                      y_center + y_span/2 + self.pan_offset[1])
        
        # Update figure layout
        self.figure.update_layout(
            xaxis=dict(range=new_x_range),
            yaxis=dict(range=new_y_range)
        )

class TimeRangeSelector:
    """Time range selection controls."""
    
    def __init__(self, data: pd.DataFrame, date_column: str = 'date'):
        """
        Initialize time range selector.
        
        Args:
            data: DataFrame containing time series data
            date_column: Name of the date column
        """
        self.data = data
        self.date_column = date_column
        self.current_range = None
        
        # Extract date range
        if date_column in data.columns:
            dates = pd.to_datetime(data[date_column])
            self.min_date = dates.min()
            self.max_date = dates.max()
            self.current_range = (self.min_date, self.max_date)
    
    def set_range(self, start_date: datetime, end_date: datetime):
        """Set time range."""
        if start_date < self.min_date:
            start_date = self.min_date
        if end_date > self.max_date:
            end_date = self.max_date
        
        self.current_range = (start_date, end_date)
    
    def get_filtered_data(self) -> pd.DataFrame:
        """Get data filtered by current time range."""
        if not self.current_range:
            return self.data
        
        mask = (self.data[self.date_column] >= self.current_range[0]) & \
               (self.data[self.date_column] <= self.current_range[1])
        
        return self.data[mask]
    
    def create_range_buttons(self) -> List[Dict[str, Any]]:
        """Create predefined range buttons."""
        ranges = [
            {'label': '1D', 'days': 1},
            {'label': '1W', 'days': 7},
            {'label': '1M', 'days': 30},
            {'label': '3M', 'days': 90},
            {'label': '6M', 'days': 180},
            {'label': '1Y', 'days': 365},
            {'label': 'ALL', 'days': None}
        ]
        
        buttons = []
        for range_info in ranges:
            if range_info['days']:
                end_date = self.max_date
                start_date = end_date - timedelta(days=range_info['days'])
            else:
                start_date = self.min_date
                end_date = self.max_date
            
            buttons.append({
                'label': range_info['label'],
                'start_date': start_date,
                'end_date': end_date
            })
        
        return buttons

class IndicatorToggleController:
    """Indicator toggle controls."""
    
    def __init__(self, indicators: Dict[str, Dict[str, Any]]):
        """
        Initialize indicator toggle controller.
        
        Args:
            indicators: Dictionary of indicator configurations
        """
        self.indicators = indicators
        self.visible_indicators = set()
        self.indicator_traces = {}
    
    def toggle_indicator(self, indicator_name: str, visible: bool):
        """Toggle indicator visibility."""
        if visible:
            self.visible_indicators.add(indicator_name)
        else:
            self.visible_indicators.discard(indicator_name)
    
    def get_visible_indicators(self) -> List[str]:
        """Get list of visible indicators."""
        return list(self.visible_indicators)
    
    def add_indicator_trace(self, indicator_name: str, trace_data: Dict[str, Any]):
        """Add indicator trace data."""
        self.indicator_traces[indicator_name] = trace_data
    
    def update_chart_indicators(self, figure: go.Figure):
        """Update chart with current indicator visibility."""
        # Remove all indicator traces
        traces_to_remove = []
        for i, trace in enumerate(figure.data):
            if hasattr(trace, 'name') and trace.name in self.indicator_traces:
                traces_to_remove.append(i)
        
        # Remove traces in reverse order to maintain indices
        for i in reversed(traces_to_remove):
            figure.data = list(figure.data[:i]) + list(figure.data[i+1:])
        
        # Add visible indicator traces
        for indicator_name in self.visible_indicators:
            if indicator_name in self.indicator_traces:
                trace_data = self.indicator_traces[indicator_name]
                figure.add_trace(go.Scatter(**trace_data))

class ChartTypeSwitcher:
    """Chart type switching controls."""
    
    def __init__(self, available_types: List[ChartType] = None):
        """
        Initialize chart type switcher.
        
        Args:
            available_types: List of available chart types
        """
        if available_types is None:
            available_types = [ChartType.CANDLESTICK, ChartType.LINE, ChartType.BAR]
        
        self.available_types = available_types
        self.current_type = available_types[0]
    
    def switch_chart_type(self, chart_type: ChartType):
        """Switch to specified chart type."""
        if chart_type in self.available_types:
            self.current_type = chart_type
    
    def get_current_type(self) -> ChartType:
        """Get current chart type."""
        return self.current_type
    
    def create_chart(self, data: pd.DataFrame, x_column: str, y_column: str) -> go.Figure:
        """Create chart with current type."""
        if self.current_type == ChartType.CANDLESTICK:
            return self._create_candlestick_chart(data)
        elif self.current_type == ChartType.LINE:
            return self._create_line_chart(data, x_column, y_column)
        elif self.current_type == ChartType.BAR:
            return self._create_bar_chart(data, x_column, y_column)
        elif self.current_type == ChartType.SCATTER:
            return self._create_scatter_chart(data, x_column, y_column)
        else:
            return self._create_line_chart(data, x_column, y_column)
    
    def _create_candlestick_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create candlestick chart."""
        fig = go.Figure(data=[go.Candlestick(
            x=data['date'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        )])
        
        fig.update_layout(
            title='Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white'
        )
        
        return fig
    
    def _create_line_chart(self, data: pd.DataFrame, x_column: str, y_column: str) -> go.Figure:
        """Create line chart."""
        fig = go.Figure(data=[go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode='lines',
            name=y_column
        )])
        
        fig.update_layout(
            title='Line Chart',
            xaxis_title=x_column,
            yaxis_title=y_column,
            template='plotly_white'
        )
        
        return fig
    
    def _create_bar_chart(self, data: pd.DataFrame, x_column: str, y_column: str) -> go.Figure:
        """Create bar chart."""
        fig = go.Figure(data=[go.Bar(
            x=data[x_column],
            y=data[y_column],
            name=y_column
        )])
        
        fig.update_layout(
            title='Bar Chart',
            xaxis_title=x_column,
            yaxis_title=y_column,
            template='plotly_white'
        )
        
        return fig
    
    def _create_scatter_chart(self, data: pd.DataFrame, x_column: str, y_column: str) -> go.Figure:
        """Create scatter chart."""
        fig = go.Figure(data=[go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode='markers',
            name=y_column
        )])
        
        fig.update_layout(
            title='Scatter Chart',
            xaxis_title=x_column,
            yaxis_title=y_column,
            template='plotly_white'
        )
        
        return fig

class AnnotationTool:
    """Annotation and drawing tools."""
    
    def __init__(self):
        """Initialize annotation tool."""
        self.annotations = []
        self.current_annotation = None
        self.annotation_mode = None
    
    def add_text_annotation(self, x: float, y: float, text: str, 
                           color: str = 'red', size: int = 12):
        """Add text annotation."""
        annotation = {
            'type': 'text',
            'x': x,
            'y': y,
            'text': text,
            'color': color,
            'size': size,
            'timestamp': datetime.now()
        }
        self.annotations.append(annotation)
    
    def add_line_annotation(self, x1: float, y1: float, x2: float, y2: float,
                           color: str = 'blue', width: int = 2):
        """Add line annotation."""
        annotation = {
            'type': 'line',
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'color': color,
            'width': width,
            'timestamp': datetime.now()
        }
        self.annotations.append(annotation)
    
    def add_arrow_annotation(self, x: float, y: float, ax: float, ay: float,
                            text: str = '', color: str = 'green'):
        """Add arrow annotation."""
        annotation = {
            'type': 'arrow',
            'x': x,
            'y': y,
            'ax': ax,
            'ay': ay,
            'text': text,
            'color': color,
            'timestamp': datetime.now()
        }
        self.annotations.append(annotation)
    
    def remove_annotation(self, index: int):
        """Remove annotation by index."""
        if 0 <= index < len(self.annotations):
            self.annotations.pop(index)
    
    def clear_annotations(self):
        """Clear all annotations."""
        self.annotations.clear()
    
    def apply_annotations_to_chart(self, figure: go.Figure):
        """Apply annotations to chart."""
        for annotation in self.annotations:
            if annotation['type'] == 'text':
                figure.add_annotation(
                    x=annotation['x'],
                    y=annotation['y'],
                    text=annotation['text'],
                    showarrow=False,
                    font=dict(color=annotation['color'], size=annotation['size'])
                )
            elif annotation['type'] == 'line':
                figure.add_shape(
                    type="line",
                    x0=annotation['x1'],
                    y0=annotation['y1'],
                    x1=annotation['x2'],
                    y1=annotation['y2'],
                    line=dict(color=annotation['color'], width=annotation['width'])
                )
            elif annotation['type'] == 'arrow':
                figure.add_annotation(
                    x=annotation['x'],
                    y=annotation['y'],
                    ax=annotation['ax'],
                    ay=annotation['ay'],
                    text=annotation['text'],
                    arrowcolor=annotation['color'],
                    arrowhead=2
                )

class ChartStateManager:
    """Chart state management and persistence."""
    
    def __init__(self):
        """Initialize chart state manager."""
        self.current_state = ChartState()
        self.state_history = []
        self.max_history_size = 10
    
    def save_state(self, state: ChartState):
        """Save current state."""
        self.state_history.append(self.current_state)
        self.current_state = state
        
        # Limit history size
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)
    
    def restore_state(self, state: ChartState):
        """Restore to specified state."""
        self.current_state = state
    
    def undo(self) -> Optional[ChartState]:
        """Undo last state change."""
        if self.state_history:
            previous_state = self.state_history.pop()
            current_state = self.current_state
            self.current_state = previous_state
            return current_state
        return None
    
    def get_current_state(self) -> ChartState:
        """Get current state."""
        return self.current_state
    
    def export_state(self) -> Dict[str, Any]:
        """Export state to dictionary."""
        return {
            'zoom_level': self.current_state.zoom_level,
            'pan_offset': self.current_state.pan_offset,
            'time_range': self.current_state.time_range,
            'visible_indicators': self.current_state.visible_indicators,
            'chart_type': self.current_state.chart_type.value,
            'theme': self.current_state.theme
        }
    
    def import_state(self, state_dict: Dict[str, Any]):
        """Import state from dictionary."""
        self.current_state.zoom_level = state_dict.get('zoom_level', 1.0)
        self.current_state.pan_offset = state_dict.get('pan_offset', (0.0, 0.0))
        self.current_state.time_range = state_dict.get('time_range', 
                                                      (datetime.now() - timedelta(days=30), datetime.now()))
        self.current_state.visible_indicators = state_dict.get('visible_indicators', [])
        self.current_state.chart_type = ChartType(state_dict.get('chart_type', 'candlestick'))
        self.current_state.theme = state_dict.get('theme', 'light')

class InteractiveChartController:
    """Main interactive chart controller."""
    
    def __init__(self, chart_figure: go.Figure, data: pd.DataFrame):
        """
        Initialize interactive chart controller.
        
        Args:
            chart_figure: Plotly figure to control
            data: Data for the chart
        """
        self.figure = chart_figure
        self.data = data
        
        # Initialize controllers
        self.zoom_pan_controller = ZoomPanController(chart_figure)
        self.time_range_selector = TimeRangeSelector(data)
        self.indicator_controller = IndicatorToggleController({})
        self.chart_type_switcher = ChartTypeSwitcher()
        self.annotation_tool = AnnotationTool()
        self.state_manager = ChartStateManager()
    
    def create_control_panel(self):
        """Create control panel for Dash app."""
        if not DASH_AVAILABLE:
            logger.warning("Dash not available, returning None for control panel")
            return None
            
        return html.Div([
            # Zoom and Pan Controls
            html.Div([
                html.H4("Zoom & Pan"),
                dbc.Button("Zoom In", id="zoom-in-btn", color="primary", size="sm"),
                dbc.Button("Zoom Out", id="zoom-out-btn", color="primary", size="sm"),
                dbc.Button("Reset", id="reset-view-btn", color="secondary", size="sm"),
            ], className="mb-3"),
            
            # Time Range Controls
            html.Div([
                html.H4("Time Range"),
                dcc.DatePickerRange(
                    id="date-range-picker",
                    start_date=self.time_range_selector.min_date,
                    end_date=self.time_range_selector.max_date,
                    display_format='YYYY-MM-DD'
                ),
                html.Div([
                    dbc.Button(btn['label'], id=f"range-btn-{i}", 
                              color="outline-primary", size="sm", className="me-1")
                    for i, btn in enumerate(self.time_range_selector.create_range_buttons())
                ])
            ], className="mb-3"),
            
            # Chart Type Controls
            html.Div([
                html.H4("Chart Type"),
                dcc.Dropdown(
                    id="chart-type-dropdown",
                    options=[
                        {'label': 'Candlestick', 'value': 'candlestick'},
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Bar', 'value': 'bar'},
                        {'label': 'Scatter', 'value': 'scatter'}
                    ],
                    value='candlestick'
                )
            ], className="mb-3"),
            
            # Indicator Controls
            html.Div([
                html.H4("Indicators"),
                dcc.Checklist(
                    id="indicator-checklist",
                    options=[
                        {'label': 'Moving Average', 'value': 'ma'},
                        {'label': 'RSI', 'value': 'rsi'},
                        {'label': 'MACD', 'value': 'macd'},
                        {'label': 'Bollinger Bands', 'value': 'bb'}
                    ],
                    value=[]
                )
            ], className="mb-3"),
            
            # Annotation Controls
            html.Div([
                html.H4("Annotations"),
                dbc.Button("Add Text", id="add-text-btn", color="success", size="sm"),
                dbc.Button("Add Line", id="add-line-btn", color="success", size="sm"),
                dbc.Button("Clear All", id="clear-annotations-btn", color="danger", size="sm"),
            ], className="mb-3"),
            
            # State Management
            html.Div([
                html.H4("State Management"),
                dbc.Button("Save State", id="save-state-btn", color="info", size="sm"),
                dbc.Button("Undo", id="undo-btn", color="warning", size="sm"),
            ])
        ])
    
    def update_chart(self):
        """Update chart with current state."""
        # Apply time range filter
        filtered_data = self.time_range_selector.get_filtered_data()
        
        # Update chart type
        self.figure = self.chart_type_switcher.create_chart(
            filtered_data, 'date', 'close'
        )
        
        # Apply indicators
        self.indicator_controller.update_chart_indicators(self.figure)
        
        # Apply annotations
        self.annotation_tool.apply_annotations_to_chart(self.figure)
        
        # Apply zoom and pan
        self.zoom_pan_controller._apply_zoom_pan()

def create_interactive_controller(chart_figure: go.Figure, 
                                data: pd.DataFrame) -> InteractiveChartController:
    """
    Create an interactive chart controller.
    
    Args:
        chart_figure: Plotly figure to control
        data: Data for the chart
        
    Returns:
        InteractiveChartController instance
    """
    return InteractiveChartController(chart_figure, data)

if __name__ == "__main__":
    # Demo of interactive controls
    import plotly.graph_objects as go
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    prices = np.random.randn(len(dates)).cumsum() + 100
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices + np.random.randn(len(dates)),
        'high': prices + abs(np.random.randn(len(dates))),
        'low': prices - abs(np.random.randn(len(dates)))
    })
    
    # Create sample chart
    fig = go.Figure(data=[go.Scatter(x=data['date'], y=data['close'])])
    
    # Create controller
    controller = create_interactive_controller(fig, data)
    
    print("Interactive chart controller created successfully!")
    print(f"Available chart types: {[t.value for t in controller.chart_type_switcher.available_types]}")
    print(f"Time range: {controller.time_range_selector.min_date} to {controller.time_range_selector.max_date}")