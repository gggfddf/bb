# ML Stock Predictor Platform

A comprehensive, self-contained machine learning platform for stock price prediction with 40+ technical indicators, advanced ML models, and real-time prediction capabilities.

## 🚀 Project Overview

This platform is designed to be completely API-independent, building its own data collection system, feature engineering pipeline, and ML prediction models. The system automatically discovers profitable trading strategies through machine learning without relying on predefined rules or external APIs.

### Key Features

- **Self-Contained Data Collection**: Web scraping and WebSocket streaming for real-time data
- **40+ Technical Indicators**: Complete implementation of all major technical analysis indicators
- **Advanced ML Models**: Tree-based models, LSTM, GRU, Transformers, and CNN for pattern recognition
- **Real-time Prediction**: Sub-2-second prediction latency with streaming capabilities
- **Comprehensive Backtesting**: Vectorized backtesting engine with realistic trade simulation
- **Interactive Dashboard**: Real-time visualization with candlestick charts and performance metrics
- **Production Ready**: Docker containerization, Kubernetes orchestration, and monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML Stock Predictor Platform                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   Data      │  │  Feature    │  │     ML      │  │Backtest │ │
│  │ Ingestion   │  │Engineering  │  │   Models    │  │ Engine  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │  Storage    │  │Visualization│  │   Real-time │  │  API    │ │
│  │  Layer      │  │  Dashboard  │  │ Prediction  │  │Gateway  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Project Structure

```
ml-stock-predictor/
├── data_ingestion/          # Data collection and storage
├── feature_engineering/      # Technical indicators and feature processing
├── ml_models/              # Machine learning models
├── backtesting/            # Backtesting engine
├── visualization/          # Dashboard and charts
├── api/                   # REST API and WebSocket endpoints
├── deployment/            # Docker and Kubernetes configs
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
└── config/                # Configuration files
```

## 🛠️ Technology Stack

### Backend
- **Python 3.9+**: Core programming language
- **FastAPI**: High-performance web framework
- **Celery**: Distributed task queue
- **Redis**: Caching and message broker

### Database
- **TimescaleDB**: Time-series database (PostgreSQL extension)
- **PostgreSQL**: Primary database

### Machine Learning
- **scikit-learn**: Traditional ML algorithms
- **TensorFlow/Keras**: Deep learning models
- **XGBoost**: Gradient boosting
- **LightGBM**: Light gradient boosting
- **ta-lib**: Technical analysis library

### Data Processing
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **polars**: Fast data processing

### Frontend
- **React.js**: User interface
- **D3.js**: Data visualization
- **Chart.js**: Interactive charts

### Deployment
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Prometheus**: Monitoring
- **Grafana**: Visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL with TimescaleDB extension
- Redis

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ml-stock-predictor.git
   cd ml-stock-predictor
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start services with Docker Compose**
   ```bash
   docker-compose up -d
   ```

6. **Run database migrations**
   ```bash
   python scripts/setup_database.py
   ```

7. **Start the application**
   ```bash
   python main.py
   ```

### Development Setup

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run tests**
   ```bash
   pytest tests/
   ```

3. **Start development server**
   ```bash
   uvicorn api.main:app --reload
   ```

## 📊 Technical Indicators

The platform implements 40+ technical indicators:

### Momentum Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)
- TSI (True Strength Index)

### Trend Indicators
- SMA/EMA (Simple/Exponential Moving Averages)
- Ichimoku Cloud
- Parabolic SAR
- ADX (Average Directional Index)
- SuperTrend
- Hull Moving Average

### Volatility Indicators
- Bollinger Bands
- ATR (Average True Range)
- Keltner Channel
- Donchian Channel
- Ulcer Index

### Volume Indicators
- VWAP (Volume Weighted Average Price)
- OBV (On-Balance Volume)
- MFI (Money Flow Index)
- Chaikin Oscillator
- Chaikin Money Flow
- Accumulation/Distribution Line

### Advanced Indicators
- Fibonacci Retracements
- Pivot Points
- Volume Profile
- Beta/Correlation Analysis
- And many more...

## 🤖 Machine Learning Models

### Tree-Based Models
- Random Forest
- XGBoost
- LightGBM

### Deep Learning Models
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer-based models
- CNN for pattern recognition

### Clustering Models
- K-Means for regime detection
- DBSCAN for outlier detection

## 📈 Backtesting Features

- **Vectorized Backtesting**: Fast performance calculation
- **Realistic Trade Simulation**: Slippage, commissions, execution delays
- **Risk Management**: Position sizing, stop-loss, risk limits
- **Performance Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Walk-Forward Analysis**: Out-of-sample validation
- **Multi-Timeframe Analysis**: Cross-timeframe strategy testing

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/stock_predictor
TIMESCALE_URL=postgresql://user:password@localhost:5432/stock_predictor

# Redis
REDIS_URL=redis://localhost:6379

# API Keys (for data sources)
YAHOO_FINANCE_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key

# Model Configuration
MODEL_CACHE_DIR=./models
PREDICTION_CACHE_TTL=300

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### Model Configuration

Models can be configured through YAML files in the `config/` directory:

```yaml
# config/models.yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  lstm:
    units: 50
    layers: 2
    dropout: 0.2
    batch_size: 32
```

## 📊 Performance Metrics

### Technical Metrics
- **Prediction Accuracy**: > 60% for directional prediction
- **Sharpe Ratio**: > 1.5 for profitable strategies
- **Maximum Drawdown**: < 20%
- **Response Time**: < 2 seconds for real-time predictions

### System Metrics
- **Data Ingestion**: > 1000 records/second
- **Backtesting Speed**: < 1 minute for 1 year of data
- **System Uptime**: > 99.9%

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Module interaction testing
- **End-to-End Tests**: Full system workflow testing
- **Performance Tests**: Load and stress testing

## 📚 Documentation

- [Architecture Guide](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Model Development Guide](docs/models.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Build production image
docker build -t ml-stock-predictor:latest .
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/k8s/

# Monitor deployment
kubectl get pods -n stock-predictor
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/ml-stock-predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/ml-stock-predictor/discussions)
- **Documentation**: [Project Wiki](https://github.com/your-username/ml-stock-predictor/wiki)

## 🗺️ Roadmap

### v0.1 (8 weeks) - Foundation
- ✅ Data ingestion system
- ✅ All 40 technical indicators
- ✅ Basic feature engineering

### v0.2 (13 weeks) - ML Core
- 🔄 ML models for single indicators
- 🔄 Backtesting engine
- 🔄 Performance metrics

### v0.3 (17 weeks) - Advanced Analytics
- 📋 Multi-indicator analysis
- 📋 Visualization dashboard
- 📋 Strategy optimization

### v1.0 (20 weeks) - Production
- 📋 Real-time prediction system
- 📋 Production deployment
- 📋 Complete API gateway

---

**Note**: This is a sophisticated ML platform. Please ensure you understand the risks involved in algorithmic trading and use this system responsibly.