# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 2: Download Historical Data
# -----------------------------
def download_data(ticker="^GSPC", start="2010-01-01", end="2025-11-01"):
    """Download historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    data.to_csv("sp500.csv")
    return data

# -----------------------------
# Step 3: Feature Engineering
# -----------------------------
def create_features(data):
    """Add rolling averages and target variable."""
    # Target: 1 if next day's close is higher than today, else 0
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Rolling averages
    windows = [5, 10, 20]
    for window in windows:
        data[f'Close_{window}'] = data['Close'].rolling(window).mean()
    
    # Drop rows with NaN values created by rolling
    data.dropna(inplace=True)
    return data

# -----------------------------
# Step 4: Train ML Model
# -----------------------------
def train_model(data):
    """Train a Random Forest classifier and return trained model."""
    features = ['Close_5', 'Close_10', 'Close_20']
    X = data[features]
    y = data['Target']
    
    # Train-test split (keep time series order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model, X_test, y_test

# -----------------------------
# Step 5: Backtesting Strategy
# -----------------------------
def backtest(data, model):
    """Simple backtest using model predictions."""
    features = ['Close_5', 'Close_10', 'Close_20']
    X = data[features]
    
    data['Pred'] = model.predict(X)
    data['Strategy_Return'] = data['Pred'].shift(1) * data['Close'].pct_change()
    data['Market_Return'] = data['Close'].pct_change()
    
    data.fillna(0, inplace=True)
    
    # Cumulative returns
    cumulative_strategy = (1 + data['Strategy_Return']).cumprod()
    cumulative_market = (1 + data['Market_Return']).cumprod()
    
    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(cumulative_strategy, label='Strategy Cumulative Return')
    plt.plot(cumulative_market, label='Market Cumulative Return')
    plt.title('Backtesting: Strategy vs Market')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()

# -----------------------------
# Step 6: Main Execution
# -----------------------------
if __name__ == "__main__":
    print("Downloading historical S&P 500 data...")
    data = download_data()
    
    print("Creating features and target...")
    data = create_features(data)
    
    print("Training Random Forest model...")
    model, X_test, y_test = train_model(data)
    
    print("Running backtest...")
    backtest(data, model)
