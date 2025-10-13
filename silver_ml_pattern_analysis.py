import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.signal import find_peaks

# Set style for better-looking plots
sns.set_style("whitegrid")

class SilverPatternMLAnalysis:
    def __init__(self):
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def download_silver_data(self, period="max", interval="1d"):
        """Download silver data from yfinance"""
        print(f"Downloading silver data with {interval} interval for period: {period}...")
        
        silver = yf.Ticker("SI=F")
        self.data = silver.history(period=period, interval=interval)
        
        print(f"Downloaded {len(self.data)} records")
        print(f"\nData range: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        
        return self.data
    
    def identify_price_movements(self, df):
        """Identify different types of price movements and their durations"""
        
        # Calculate price changes
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100
        
        # Identify movement direction
        df['Movement_Direction'] = np.where(df['Price_Change_Pct'] > 0, 1, 
                                           np.where(df['Price_Change_Pct'] < 0, -1, 0))
        
        # Find peaks and troughs for cycle detection
        peaks, _ = find_peaks(df['Close'].values, distance=5)
        troughs, _ = find_peaks(-df['Close'].values, distance=5)
        
        df['Is_Peak'] = 0
        df['Is_Trough'] = 0
        df.iloc[peaks, df.columns.get_loc('Is_Peak')] = 1
        df.iloc[troughs, df.columns.get_loc('Is_Trough')] = 1
        
        return df
    
    def calculate_correction_patterns(self, df):
        """Calculate correction time, duration, and magnitude"""
        
        # Rolling window to detect corrections (price drops after rise)
        window = 20
        df['Price_High_20d'] = df['Close'].rolling(window).max()
        df['Price_Low_20d'] = df['Close'].rolling(window).min()
        df['Drawdown_from_High'] = ((df['Close'] - df['Price_High_20d']) / df['Price_High_20d']) * 100
        df['Rise_from_Low'] = ((df['Close'] - df['Price_Low_20d']) / df['Price_Low_20d']) * 100
        
        # Correction detection (drop > 3% from recent high)
        df['In_Correction'] = (df['Drawdown_from_High'] < -3).astype(int)
        
        # Calculate time in correction
        df['Correction_Duration'] = 0
        correction_count = 0
        for i in range(1, len(df)):
            if df.iloc[i]['In_Correction'] == 1:
                correction_count += 1
                df.iloc[i, df.columns.get_loc('Correction_Duration')] = correction_count
            else:
                correction_count = 0
        
        return df
    
    def calculate_consolidation_patterns(self, df):
        """Identify consolidation periods (low volatility, sideways movement)"""
        
        window = 10
        # Volatility measure
        df['Price_Range'] = df['High'] - df['Low']
        df['Avg_Range_10d'] = df['Price_Range'].rolling(window).mean()
        df['Range_Ratio'] = df['Price_Range'] / df['Avg_Range_10d']
        
        # Consolidation: low volatility + small price changes
        df['Is_Consolidating'] = ((df['Range_Ratio'] < 0.7) & 
                                  (abs(df['Price_Change_Pct']) < 1)).astype(int)
        
        # Duration of consolidation
        df['Consolidation_Duration'] = 0
        consol_count = 0
        for i in range(1, len(df)):
            if df.iloc[i]['Is_Consolidating'] == 1:
                consol_count += 1
                df.iloc[i, df.columns.get_loc('Consolidation_Duration')] = consol_count
            else:
                consol_count = 0
        
        return df
    
    def calculate_reversal_patterns(self, df):
        """Detect reversal patterns and time to reversal"""
        
        # Trend detection using price position relative to moving averages
        df['MA_10'] = df['Close'].rolling(10).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        
        df['Trend'] = np.where(df['Close'] > df['MA_10'], 1,
                              np.where(df['Close'] < df['MA_10'], -1, 0))
        
        # Detect trend reversals
        df['Trend_Change'] = df['Trend'].diff()
        df['Is_Reversal'] = (abs(df['Trend_Change']) >= 2).astype(int)
        
        # Time since last reversal
        df['Days_Since_Reversal'] = 0
        days_count = 0
        for i in range(len(df)):
            if df.iloc[i]['Is_Reversal'] == 1:
                days_count = 0
            else:
                days_count += 1
            df.iloc[i, df.columns.get_loc('Days_Since_Reversal')] = days_count
        
        return df
    
    def calculate_momentum_exhaustion(self, df):
        """Detect oversold/overbought conditions and exhaustion patterns"""
        
        # Calculate momentum
        df['Momentum_5d'] = df['Close'].pct_change(5) * 100
        df['Momentum_10d'] = df['Close'].pct_change(10) * 100
        df['Momentum_20d'] = df['Close'].pct_change(20) * 100
        
        # Exhaustion detection (extreme momentum)
        df['Is_Overbought'] = (df['Momentum_10d'] > 10).astype(int)
        df['Is_Oversold'] = (df['Momentum_10d'] < -10).astype(int)
        
        # Momentum divergence (price up but momentum slowing)
        df['Momentum_Change'] = df['Momentum_10d'].diff()
        df['Price_Momentum_Divergence'] = ((df['Price_Change_Pct'] > 0) & 
                                           (df['Momentum_Change'] < 0)).astype(int)
        
        return df
    
    def calculate_cycle_patterns(self, df):
        """Analyze time cycles and pattern cycles"""
        
        # Day of week and month patterns
        df['Day_of_Week'] = df.index.dayofweek
        df['Day_of_Month'] = df.index.day
        df['Month'] = df.index.month
        
        # Price cycle patterns (time between peaks/troughs)
        df['Cycle_Position'] = 0
        last_peak_idx = 0
        for i in range(len(df)):
            if df.iloc[i]['Is_Peak'] == 1:
                last_peak_idx = i
            if last_peak_idx > 0:
                df.iloc[i, df.columns.get_loc('Cycle_Position')] = i - last_peak_idx
        
        # Wave patterns (consecutive up/down days)
        df['Consecutive_Up_Days'] = 0
        df['Consecutive_Down_Days'] = 0
        up_count = 0
        down_count = 0
        for i in range(1, len(df)):
            if df.iloc[i]['Price_Change_Pct'] > 0:
                up_count += 1
                down_count = 0
            elif df.iloc[i]['Price_Change_Pct'] < 0:
                down_count += 1
                up_count = 0
            else:
                up_count = 0
                down_count = 0
            
            df.iloc[i, df.columns.get_loc('Consecutive_Up_Days')] = up_count
            df.iloc[i, df.columns.get_loc('Consecutive_Down_Days')] = down_count
        
        return df
    
    def calculate_movement_relationships(self, df):
        """Calculate relationships between different movement types"""
        
        # Ratio of correction time to rally time
        df['Rally_Days'] = (df['Movement_Direction'] == 1).astype(int).rolling(20).sum()
        df['Decline_Days'] = (df['Movement_Direction'] == -1).astype(int).rolling(20).sum()
        df['Rally_Decline_Ratio'] = df['Rally_Days'] / (df['Decline_Days'] + 1)
        
        # Average move size in corrections vs rallies
        df['Avg_Up_Move'] = df[df['Price_Change_Pct'] > 0]['Price_Change_Pct'].rolling(10).mean()
        df['Avg_Down_Move'] = abs(df[df['Price_Change_Pct'] < 0]['Price_Change_Pct'].rolling(10).mean())
        df['Avg_Up_Move'] = df['Avg_Up_Move'].ffill()
        df['Avg_Down_Move'] = df['Avg_Down_Move'].ffill()
        
        # Time to next peak/trough (lookahead feature for training)
        df['Distance_to_Next_Peak'] = 0
        df['Distance_to_Next_Trough'] = 0
        
        return df
    
    def prepare_pattern_features(self):
        """Create all pattern-based features for ML model"""
        
        if self.data is None:
            raise ValueError("No data available. Please download data first.")
        
        df = self.data.copy()
        
        print("\nCalculating price movement patterns...")
        df = self.identify_price_movements(df)
        
        print("Calculating correction patterns...")
        df = self.calculate_correction_patterns(df)
        
        print("Calculating consolidation patterns...")
        df = self.calculate_consolidation_patterns(df)
        
        print("Calculating reversal patterns...")
        df = self.calculate_reversal_patterns(df)
        
        print("Calculating momentum exhaustion patterns...")
        df = self.calculate_momentum_exhaustion(df)
        
        print("Calculating cycle patterns...")
        df = self.calculate_cycle_patterns(df)
        
        print("Calculating movement relationships...")
        df = self.calculate_movement_relationships(df)
        
        # Target: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        
        # Additional target: Direction of next move
        df['Next_Direction'] = np.where(df['Target'] > df['Close'], 1,
                                       np.where(df['Target'] < df['Close'], -1, 0))
        
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop NaN values
        df = df.dropna()
        
        # Select pattern-based features
        feature_columns = [
            # Price patterns
            'Price_Change_Pct', 'Movement_Direction', 'Is_Peak', 'Is_Trough',
            # Correction patterns
            'Drawdown_from_High', 'Rise_from_Low', 'In_Correction', 'Correction_Duration',
            # Consolidation patterns
            'Range_Ratio', 'Is_Consolidating', 'Consolidation_Duration',
            # Reversal patterns
            'Trend', 'Is_Reversal', 'Days_Since_Reversal',
            # Momentum patterns
            'Momentum_5d', 'Momentum_10d', 'Momentum_20d',
            'Is_Overbought', 'Is_Oversold', 'Price_Momentum_Divergence',
            # Cycle patterns
            'Day_of_Week', 'Day_of_Month', 'Month', 'Cycle_Position',
            'Consecutive_Up_Days', 'Consecutive_Down_Days',
            # Movement relationships
            'Rally_Decline_Ratio', 'Avg_Up_Move', 'Avg_Down_Move',
            # Current price for context
            'Close'
        ]
        
        X = df[feature_columns]
        y = df['Target']
        
        print(f"\nPattern feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"\nFeatures used for pattern learning:")
        for i, col in enumerate(feature_columns, 1):
            print(f"  {i}. {col}")
        
        return X, y
    
    def train_model(self, model_type='gradient_boosting'):
        """Train ML model on pattern features"""
        
        X, y = self.prepare_pattern_features()
        
        # Split data (80% train, 20% test) - time series split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        # Choose model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
            print("\nTraining Random Forest model on price patterns...")
        else:
            self.model = GradientBoostingRegressor(n_estimators=200, max_depth=7, 
                                                   learning_rate=0.1, random_state=42)
            print("\nTraining Gradient Boosting model on price patterns...")
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        
        # Evaluate
        print("\n" + "="*60)
        print("PATTERN-BASED ML MODEL PERFORMANCE")
        print("="*60)
        
        print("\nTraining Set Metrics:")
        print(f"R² Score: {r2_score(self.y_train, train_predictions):.4f}")
        print(f"RMSE: ${np.sqrt(mean_squared_error(self.y_train, train_predictions)):.4f}")
        print(f"MAE: ${mean_absolute_error(self.y_train, train_predictions):.4f}")
        
        print("\nTest Set Metrics:")
        print(f"R² Score: {r2_score(self.y_test, test_predictions):.4f}")
        print(f"RMSE: ${np.sqrt(mean_squared_error(self.y_test, test_predictions)):.4f}")
        print(f"MAE: ${mean_absolute_error(self.y_test, test_predictions):.4f}")
        
        # Direction accuracy
        test_direction_actual = np.where(self.y_test.values > self.X_test['Close'].values, 1, -1)
        test_direction_pred = np.where(test_predictions > self.X_test['Close'].values, 1, -1)
        direction_accuracy = np.mean(test_direction_actual == test_direction_pred) * 100
        
        print(f"\nDirection Prediction Accuracy: {direction_accuracy:.2f}%")
        
        return test_predictions
    
    def visualize_pattern_analysis(self, predictions):
        """Create visualizations of pattern-based predictions"""
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        
        # 1. Actual vs Predicted prices
        axes[0, 0].plot(self.y_test.values[-200:], label='Actual', alpha=0.7, linewidth=2)
        axes[0, 0].plot(predictions[-200:], label='Predicted', alpha=0.7, linewidth=2)
        axes[0, 0].set_title('Pattern-Based: Actual vs Predicted Silver Prices (Last 200 days)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Test Sample Index')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Prediction accuracy scatter
        axes[0, 1].scatter(self.y_test.values, predictions, alpha=0.5)
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 
                       'r--', lw=2)
        axes[0, 1].set_title('Prediction Accuracy Scatter Plot', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Actual Price ($)')
        axes[0, 1].set_ylabel('Predicted Price ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction errors
        errors = self.y_test.values - predictions
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Prediction Error ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
            axes[1, 1].set_title('Top 15 Pattern Feature Importances', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].invert_yaxis()
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # 5. Correction pattern analysis
        X_combined = pd.concat([self.X_train, self.X_test])
        axes[2, 0].plot(X_combined['Correction_Duration'].values[-500:], 
                       color='red', alpha=0.6, linewidth=1)
        axes[2, 0].set_title('Correction Duration Pattern (Last 500 days)', fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('Days')
        axes[2, 0].set_ylabel('Days in Correction')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Cycle position analysis
        axes[2, 1].plot(X_combined['Cycle_Position'].values[-500:], 
                       color='green', alpha=0.6, linewidth=1)
        axes[2, 1].set_title('Price Cycle Position (Last 500 days)', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Days')
        axes[2, 1].set_ylabel('Days Since Peak')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('silver_pattern_ml_results.png', dpi=300, bbox_inches='tight')
        print("\nPattern analysis visualization saved as 'silver_pattern_ml_results.png'")
        
    def predict_next_day_pattern(self):
        """Predict next day with pattern analysis insights"""
        
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Get the latest data point
        X, _ = self.prepare_pattern_features()
        latest_features = X.iloc[-1:]
        
        # Make prediction
        prediction = self.model.predict(latest_features)[0]
        current_price = latest_features['Close'].values[0]
        
        # Get pattern insights
        in_correction = latest_features['In_Correction'].values[0]
        correction_duration = latest_features['Correction_Duration'].values[0]
        is_consolidating = latest_features['Is_Consolidating'].values[0]
        consolidation_duration = latest_features['Consolidation_Duration'].values[0]
        is_overbought = latest_features['Is_Overbought'].values[0]
        is_oversold = latest_features['Is_Oversold'].values[0]
        days_since_reversal = latest_features['Days_Since_Reversal'].values[0]
        consecutive_up = latest_features['Consecutive_Up_Days'].values[0]
        consecutive_down = latest_features['Consecutive_Down_Days'].values[0]
        momentum_10d = latest_features['Momentum_10d'].values[0]
        
        print("\n" + "="*60)
        print("PATTERN-BASED NEXT DAY PREDICTION & ANALYSIS")
        print("="*60)
        
        print(f"\n📊 CURRENT PRICE: ${current_price:.2f}")
        print(f"🎯 PREDICTED NEXT DAY PRICE: ${prediction:.2f}")
        print(f"📈 EXPECTED CHANGE: ${prediction - current_price:.2f} ({((prediction - current_price) / current_price * 100):.2f}%)")
        
        print(f"\n🔍 PATTERN ANALYSIS:")
        print(f"   {'✓ IN CORRECTION' if in_correction else '✗ Not in correction'} (Duration: {int(correction_duration)} days)")
        print(f"   {'✓ CONSOLIDATING' if is_consolidating else '✗ Not consolidating'} (Duration: {int(consolidation_duration)} days)")
        print(f"   {'⚠️  OVERBOUGHT' if is_overbought else '✗ Not overbought'}")
        print(f"   {'⚠️  OVERSOLD' if is_oversold else '✗ Not oversold'}")
        
        print(f"\n⏱️  TIME CYCLE ANALYSIS:")
        print(f"   Days Since Reversal: {int(days_since_reversal)}")
        print(f"   Consecutive Up Days: {int(consecutive_up)}")
        print(f"   Consecutive Down Days: {int(consecutive_down)}")
        print(f"   10-Day Momentum: {momentum_10d:.2f}%")
        
        # Pattern interpretation
        print(f"\n💡 INTERPRETATION:")
        if is_overbought and consecutive_up > 3:
            print("   ⚠️  Strong uptrend but showing overbought signals - potential reversal ahead")
        elif is_oversold and consecutive_down > 3:
            print("   ⚠️  Strong downtrend but showing oversold signals - potential bounce expected")
        elif is_consolidating:
            print("   📊 Market in consolidation phase - breakout may be imminent")
        elif in_correction:
            print("   📉 Currently in correction phase - monitoring for reversal signals")
        else:
            print("   ➡️  Market in normal price discovery mode")
        
        return prediction


def main():
    # Create instance
    analyzer = SilverPatternMLAnalysis()
    
    # Download silver data (max available daily data)
    analyzer.download_silver_data(period="max", interval="1d")
    
    # Train pattern-based model
    predictions = analyzer.train_model(model_type='gradient_boosting')
    
    # Visualize pattern analysis
    analyzer.visualize_pattern_analysis(predictions)
    
    # Predict next day with pattern insights
    analyzer.predict_next_day_pattern()
    
    print("\n" + "="*60)
    print("✅ PATTERN-BASED ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
