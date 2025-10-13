import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set style for better-looking plots
sns.set_style("whitegrid")

class SilverMLAnalysis:
    def __init__(self):
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def download_silver_data(self, period="1y", interval="1d"):
        """
        Download silver data from yfinance
        Silver ticker: SLV (iShares Silver Trust) or SI=F (Silver Futures)
        """
        print(f"Downloading silver data with {interval} interval for period: {period}...")
        
        # Using SI=F for silver futures
        silver = yf.Ticker("SI=F")
        self.data = silver.history(period=period, interval=interval)
        
        print(f"Downloaded {len(self.data)} records")
        print(f"\nData range: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        
        return self.data
    
    def prepare_features(self):
        """
        Create features for ML model
        """
        if self.data is None:
            raise ValueError("No data available. Please download data first.")
        
        df = self.data.copy()
        
        # Create technical indicators and features
        df['Returns'] = df['Close'].pct_change()
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Price_Change'] = df['Close'].diff()
        df['High_Low_Spread'] = df['High'] - df['Low']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Target: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        # Select features
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'Returns', 'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20',
                          'Volatility', 'Price_Change', 'High_Low_Spread', 
                          'Volume_Change', 'RSI']
        
        X = df[feature_columns]
        y = df['Target']
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def train_model(self, model_type='random_forest'):
        """
        Train ML model for silver price prediction
        """
        X, y = self.prepare_features()
        
        # Split data (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        # Choose model
        if model_type == 'linear':
            self.model = LinearRegression()
            print("\nTraining Linear Regression model...")
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            print("\nTraining Random Forest model...")
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        
        # Evaluate
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        
        print("\nTraining Set Metrics:")
        print(f"R² Score: {r2_score(self.y_train, train_predictions):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(self.y_train, train_predictions)):.4f}")
        print(f"MAE: {mean_absolute_error(self.y_train, train_predictions):.4f}")
        
        print("\nTest Set Metrics:")
        print(f"R² Score: {r2_score(self.y_test, test_predictions):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(self.y_test, test_predictions)):.4f}")
        print(f"MAE: {mean_absolute_error(self.y_test, test_predictions):.4f}")
        
        return test_predictions
    
    def visualize_results(self, predictions):
        """
        Create visualizations of the model results
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Actual vs Predicted prices
        axes[0, 0].plot(self.y_test.values, label='Actual', alpha=0.7)
        axes[0, 0].plot(predictions, label='Predicted', alpha=0.7)
        axes[0, 0].set_title('Actual vs Predicted Silver Prices')
        axes[0, 0].set_xlabel('Test Sample Index')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Scatter plot
        axes[0, 1].scatter(self.y_test.values, predictions, alpha=0.5)
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 
                       'r--', lw=2)
        axes[0, 1].set_title('Prediction Scatter Plot')
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].grid(True)
        
        # 3. Prediction errors
        errors = self.y_test.values - predictions
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Distribution of Prediction Errors')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].grid(True)
        
        # 4. Feature importance (if Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 1].set_title('Top 10 Feature Importances')
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].invert_yaxis()
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type',
                          ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('silver_ml_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'silver_ml_results.png'")
        plt.show()
    
    def predict_next_day(self):
        """
        Predict the next day's silver price
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Get the latest data point
        X, _ = self.prepare_features()
        latest_features = X.iloc[-1:].values
        
        # Make prediction
        prediction = self.model.predict(latest_features)[0]
        current_price = self.data['Close'].iloc[-1]
        
        print("\n" + "="*50)
        print("NEXT DAY PREDICTION")
        print("="*50)
        print(f"Current Silver Price: ${current_price:.2f}")
        print(f"Predicted Next Day Price: ${prediction:.2f}")
        print(f"Expected Change: ${prediction - current_price:.2f} ({((prediction - current_price) / current_price * 100):.2f}%)")
        
        return prediction


def main():
    # Create instance
    analyzer = SilverMLAnalysis()
    
    # Download silver data (max available daily data, up to 50 years)
    analyzer.download_silver_data(period="max", interval="1d")
    
    # Train model
    predictions = analyzer.train_model(model_type='random_forest')
    
    # Visualize results
    analyzer.visualize_results(predictions)
    
    # Predict next day
    analyzer.predict_next_day()
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)


if __name__ == "__main__":
    main()
