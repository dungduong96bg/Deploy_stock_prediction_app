"""
Example Usage Script for Stock Market Prediction
Demonstrates how to use the LSTM-DNN model with real stock data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Import the predictor (assuming stock_predictor.py is in the same directory)
# from stock_predictor import StockMarketPredictor

# For demonstration, we'll include a simplified version here
# In practice, import from the main file

def load_stock_data_from_csv(filepath):
    """
    Load stock data from CSV file

    Expected columns: Date, Open, High, Low, Close, Volume
    """
    try:
        data = pd.read_csv(filepath, parse_dates=['Date'])
        print(f"✓ Loaded {len(data)} rows from {filepath}")
        return data
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data using yfinance

    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    - start_date: Start date (YYYY-MM-DD)
    - end_date: End date (YYYY-MM-DD)
    """
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data.reset_index()
        print(f"✓ Downloaded {len(data)} rows for {ticker}")
        return data
    except ImportError:
        print("✗ yfinance not installed. Install with: pip install yfinance")
        return None
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return None


def generate_sample_data(days=1000):
    """
    Generate sample stock data for testing
    """
    print(f"Generating {days} days of sample data...")

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Simulate realistic stock price movement
    np.random.seed(42)
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    prices = initial_price * np.exp(np.cumsum(returns))

    # Add some trend
    trend = np.linspace(0, 50, days)
    prices = prices + trend

    # Create OHLC data
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * np.random.uniform(0.98, 1.0, days),
        'High': prices * np.random.uniform(1.0, 1.03, days),
        'Low': prices * np.random.uniform(0.97, 1.0, days),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, days)
    })

    print("✓ Sample data generated successfully")
    return data


def calculate_technical_indicators(data):
    """
    Add technical indicators to the dataset
    """
    print("Calculating technical indicators...")

    # Moving Averages
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    # Exponential Moving Average
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

    # Daily Returns
    data['Returns'] = data['Close'].pct_change()

    # Volatility (20-day rolling standard deviation)
    data['Volatility'] = data['Returns'].rolling(window=20).std()

    # Drop NaN values
    data = data.dropna()

    print("✓ Technical indicators calculated")
    return data


def visualize_data(data, ticker="Stock"):
    """
    Visualize stock data and technical indicators
    """
    print("Creating visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Price and Moving Averages
    axes[0].plot(data['Date'], data['Close'], label='Close Price', linewidth=2)
    if 'MA_20' in data.columns:
        axes[0].plot(data['Date'], data['MA_20'], label='MA 20', alpha=0.7)
    if 'MA_50' in data.columns:
        axes[0].plot(data['Date'], data['MA_50'], label='MA 50', alpha=0.7)
    axes[0].set_title(f'{ticker} - Price and Moving Averages', fontsize=14)
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Volume
    axes[1].bar(data['Date'], data['Volume'], alpha=0.5, color='blue')
    axes[1].set_title(f'{ticker} - Trading Volume', fontsize=14)
    axes[1].set_ylabel('Volume')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: RSI
    if 'RSI' in data.columns:
        axes[2].plot(data['Date'], data['RSI'], label='RSI', color='purple')
        axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[2].set_title(f'{ticker} - Relative Strength Index', fontsize=14)
        axes[2].set_ylabel('RSI')
        axes[2].set_ylim(0, 100)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    axes[2].set_xlabel('Date')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main execution function
    """
    print("=" * 70)
    print("Stock Market Prediction - LSTM-DNN Hybrid Model")
    print("=" * 70)
    print()

    # Step 1: Get data
    print("Step 1: Loading Data")
    print("-" * 70)

    # Option 1: Generate sample data (for testing)
    data = generate_sample_data(days=1500)

    # Option 2: Load from CSV (uncomment to use)
    # data = load_stock_data_from_csv('stock_data.csv')

    # Option 3: Download from Yahoo Finance (uncomment to use)
    # data = download_stock_data('AAPL', '2020-01-01', '2024-01-01')

    if data is None:
        print("Failed to load data. Exiting.")
        return

    print(f"\nData Summary:")
    print(f"  Date Range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"  Total Days: {len(data)}")
    print(f"  Columns: {', '.join(data.columns)}")
    print()

    # Step 2: Add technical indicators (optional)
    print("Step 2: Feature Engineering")
    print("-" * 70)
    data_with_indicators = calculate_technical_indicators(data.copy())
    print()

    # Step 3: Visualize data
    print("Step 3: Data Visualization")
    print("-" * 70)
    visualize_data(data, "Sample Stock")
    print()

    # Step 4: Prepare for model training
    print("Step 4: Model Training Setup")
    print("-" * 70)
    print("To train the model, use the following code:")
    print()
    print("from stock_predictor import StockMarketPredictor")
    print()
    print("# Initialize predictor")
    print("predictor = StockMarketPredictor(lookback=60, lstm_units=128, dnn_units=64)")
    print()
    print("# Prepare data")
    print("X_train, X_test, y_train, y_test, _ = predictor.prepare_data(data)")
    print()
    print("# Train model")
    print("history = predictor.train(X_train, y_train, X_test, y_test, epochs=100)")
    print()
    print("# Evaluate")
    print("metrics, predictions = predictor.evaluate(X_test, y_test)")
    print()
    print("# Print results")
    print("print(f'R² Score: {metrics[\"R2\"]:.5f}')")
    print("print(f'MAE: {metrics[\"MAE\"]:.5f}')")
    print("print(f'MSE: {metrics[\"MSE\"]:.5f}')")
    print()

    # Step 5: Export data
    print("Step 5: Export Data (Optional)")
    print("-" * 70)
    output_file = 'processed_stock_data.csv'
    data_with_indicators.to_csv(output_file, index=False)
    print(f"✓ Processed data saved to {output_file}")
    print()

    print("=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("1. Review the generated visualizations")
    print("2. Adjust hyperparameters if needed")
    print("3. Run the training code above")
    print("4. Evaluate model performance")
    print("5. Make predictions on new data")
    print()


if __name__ == "__main__":
    main()