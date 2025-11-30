"""
Stock Market Prediction using LSTM-DNN Hybrid Model
Based on: "Enhancing Stock Market Prediction: A Robust LSTM-DNN Model Analysis on 26 Real-Life Datasets"
IEEE Access Journal, 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')


class StockMarketPredictor:
    """
    Hybrid LSTM-DNN Model for Stock Market Prediction
    """

    def __init__(self, lookback=60, lstm_units=128, dnn_units=64):
        """
        Initialize the predictor

        Parameters:
        - lookback: number of previous time steps to use
        - lstm_units: number of LSTM units
        - dnn_units: number of DNN units
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dnn_units = dnn_units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None

    def prepare_data(self, data, target_col='Close'):
        """
        Prepare and preprocess data for training

        Parameters:
        - data: pandas DataFrame with stock data
        - target_col: column name for prediction target

        Returns:
        - X_train, X_test, y_train, y_test, scaled_data
        """
        # Extract features
        if target_col not in data.columns:
            raise ValueError(f"{target_col} column not found in data")

        # Select features (using OHLCV - Open, High, Low, Close, Volume)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        features = [f for f in features if f in data.columns]

        # Scale the data
        scaled_data = self.scaler.fit_transform(data[features])

        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i])
            # Target is the Close price (assuming it's index 3)
            target_idx = features.index(target_col)
            y.append(scaled_data[i, target_idx])

        X, y = np.array(X), np.array(y)

        # Split into train and test sets (80-20 split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test, scaled_data

    def build_lstm_dnn_model(self, input_shape):
        """
        Build the hybrid LSTM-DNN model

        Parameters:
        - input_shape: shape of input data (lookback, features)

        Returns:
        - compiled Keras model
        """
        # Input layer
        inputs = Input(shape=input_shape)

        # LSTM Branch
        lstm_out = LSTM(self.lstm_units, return_sequences=True)(inputs)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(self.lstm_units // 2, return_sequences=False)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)

        # DNN Branch (parallel processing)
        flatten_input = tf.keras.layers.Flatten()(inputs)
        dnn_out = Dense(self.dnn_units, activation='relu')(flatten_input)
        dnn_out = Dropout(0.3)(dnn_out)
        dnn_out = Dense(self.dnn_units // 2, activation='relu')(dnn_out)
        dnn_out = Dropout(0.3)(dnn_out)

        # Merge LSTM and DNN outputs
        merged = Concatenate()([lstm_out, dnn_out])

        # Final dense layers
        dense_out = Dense(32, activation='relu')(merged)
        dense_out = Dropout(0.2)(dense_out)
        output = Dense(1, activation='linear')(dense_out)

        # Create model
        model = Model(inputs=inputs, outputs=output)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the model

        Parameters:
        - X_train, y_train: training data
        - X_val, y_val: validation data
        - epochs: number of training epochs
        - batch_size: batch size for training

        Returns:
        - training history
        """
        # Build model if not already built
        if self.model is None:
            self.model = self.build_lstm_dnn_model((X_train.shape[1], X_train.shape[2]))

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        return self.history

    def predict(self, X):
        """
        Make predictions

        Parameters:
        - X: input data

        Returns:
        - predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Parameters:
        - X_test, y_test: test data

        Returns:
        - dictionary with evaluation metrics
        """
        predictions = self.predict(X_test).flatten()

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

        return metrics, predictions

    def plot_results(self, y_true, y_pred, title="Stock Price Prediction"):
        """
        Plot actual vs predicted prices

        Parameters:
        - y_true: actual prices
        - y_pred: predicted prices
        - title: plot title
        """
        plt.figure(figsize=(14, 5))
        plt.plot(y_true, label='Actual Price', color='blue', linewidth=2)
        plt.plot(y_pred, label='Predicted Price', color='red', linewidth=2, linestyle='--')
        plt.title(title, fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Normalized Price', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Generate sample data (replace with actual stock data)
    print("Stock Market Prediction using LSTM-DNN Hybrid Model")
    print("=" * 60)

    # Sample data generation (in practice, load from CSV or API)
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)

    # Simulate stock price with trend and noise
    trend = np.linspace(100, 200, len(dates))
    seasonality = 10 * np.sin(np.linspace(0, 20 * np.pi, len(dates)))
    noise = np.random.normal(0, 5, len(dates))
    close_price = trend + seasonality + noise

    data = pd.DataFrame({
        'Date': dates,
        'Open': close_price - np.random.uniform(0, 2, len(dates)),
        'High': close_price + np.random.uniform(0, 3, len(dates)),
        'Low': close_price - np.random.uniform(0, 3, len(dates)),
        'Close': close_price,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })

    print(f"\nDataset shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

    # Initialize predictor
    predictor = StockMarketPredictor(lookback=60, lstm_units=128, dnn_units=64)

    # Prepare data
    X_train, X_test, y_train, y_test, scaled_data = predictor.prepare_data(data)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    print("\nTraining model...")
    history = predictor.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)

    # Evaluate
    print("\nEvaluating model...")
    metrics, predictions = predictor.evaluate(X_test, y_test)

    print("\nModel Performance Metrics:")
    print(f"R² Score: {metrics['R2']:.5f}")
    print(f"MAE: {metrics['MAE']:.5f}")
    print(f"MSE: {metrics['MSE']:.5f}")
    print(f"RMSE: {metrics['RMSE']:.5f}")

    # Plot results
    predictor.plot_training_history()
    predictor.plot_results(y_test, predictions, "LSTM-DNN Stock Price Prediction")

    print("\nModel training complete!")