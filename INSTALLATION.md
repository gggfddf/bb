# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Step-by-Step Installation

### 1. Clone or Download the Repository

```bash
cd /workspace  # or your preferred directory
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Some dependencies may take time to install (PyTorch, LightGBM, etc.)

### 4. Verify Installation

Run the basic test suite:

```bash
python test_basic.py
```

You should see all tests pass:
```
✓ PASS: Imports
✓ PASS: Initialization
✓ PASS: Sample Data
✓ PASS: Preprocessing
✓ PASS: Feature Engineering
✓ PASS: Label Generation
```

### 5. Run Example

```bash
python example_usage.py
```

This will:
- Create sample stock data
- Run the complete analysis pipeline
- Generate pattern library
- Detect market cycles
- Train models
- Run backtest
- Save results to `./outputs/`

## Minimal Installation (Core Only)

If you want to install only core dependencies without deep learning or advanced features:

```bash
pip install numpy pandas scipy scikit-learn lightgbm matplotlib seaborn
```

This will allow you to use:
- Data preprocessing
- Feature engineering
- Label generation
- Tree-based models (LightGBM)
- Basic evaluation

Optional features requiring additional packages:
- **Sequence Models (LSTM/CNN/Transformer)**: `pip install torch`
- **XGBoost**: `pip install xgboost`
- **CatBoost**: `pip install catboost`
- **Matrix Profile**: `pip install stumpy`
- **HDBSCAN Clustering**: `pip install hdbscan`
- **Wavelet Analysis**: `pip install PyWavelets`
- **HMM**: `pip install hmmlearn`
- **SHAP Explainability**: `pip install shap`
- **Real Stock Data**: `pip install yfinance`

## Troubleshooting

### Issue: PyTorch Installation Fails

Try installing a specific version:
```bash
# CPU version
pip install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Or skip PyTorch if you don't need sequence models
```

### Issue: HDBSCAN Compilation Error

Some systems require additional build tools. You can skip this package if you don't need HDBSCAN clustering:
```bash
# The system will fall back to K-means clustering
```

### Issue: ImportError for Optional Dependencies

The system gracefully handles missing optional dependencies with warnings. Core functionality will still work.

### Issue: Memory Error During Training

Reduce data size or model complexity:
```python
config.model.tree_n_estimators = 100  # Default is 500
config.feature.lookback_windows = [5, 10, 20]  # Reduce from default
```

## Platform-Specific Notes

### Windows

- Some packages may require Microsoft Visual C++ Build Tools
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### macOS

- Ensure Xcode command line tools are installed:
  ```bash
  xcode-select --install
  ```

### Linux

- Most distributions work out of the box
- May need build-essential:
  ```bash
  sudo apt-get install build-essential
  ```

## Docker Installation (Alternative)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "example_usage.py"]
```

Build and run:
```bash
docker build -t stock-pattern-detection .
docker run -v $(pwd)/outputs:/app/outputs stock-pattern-detection
```

## Testing Installation

Quick verification:

```python
# test_install.py
from pipeline import StockPatternPipeline, create_sample_data

data = create_sample_data()
pipeline = StockPatternPipeline()
df = pipeline.load_and_preprocess(data)

print(f"✓ Installation successful! Loaded {len(df)} rows of data.")
```

Run with:
```bash
python test_install.py
```

## Next Steps

After successful installation:

1. Read `README.md` for system overview
2. Check `example_usage.py` for usage examples
3. Review `config.py` to understand configuration options
4. Start with your own data!

## Support

If you encounter issues:

1. Check that all dependencies installed correctly: `pip list`
2. Verify Python version: `python --version` (should be 3.8+)
3. Try the basic test: `python test_basic.py`
4. Check for import errors in the error messages

## Performance Optimization

For production use:

1. **GPU Acceleration**: Install PyTorch with CUDA support
2. **Parallel Processing**: Adjust `config.n_jobs = -1` (uses all cores)
3. **Memory**: Use data chunking for very large datasets
4. **Model Size**: Tune `tree_n_estimators` and `tree_max_depth`

---

**Ready to start? Run:** `python example_usage.py`
