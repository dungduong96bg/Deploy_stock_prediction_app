#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Price Prediction Pipeline
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score, max_error
from tensorflow.keras.layers import LSTM, Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tabulate import tabulate

warnings.filterwarnings('ignore')


class StockPricePredictionPipeline:
    """Pipeline for stock price prediction using LSTM and DNN"""

    def __init__(self, data_path, symbol,window_size=100, save_dir='.'):
        self.data_path = data_path
        self.window_size = window_size
        self.data = None
        self.model = None
        self.history = None
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.symbol = symbol
        self.save_dir = save_dir

    def load_data(self):
        """Step 1: Load and prepare data"""
        print("=" * 50)
        print("STEP 1: LOADING DATA")
        print("=" * 50)

        self.data = pd.read_csv(self.data_path)
        self.data = self.data[self.data['Symbol'] == self.symbol]
        print("Data loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print("\nFirst few rows:")
        print(self.data.head())

        # Drop Date column
        self.data.drop(['Date'], axis=1, inplace=True)
        print("\nColumns:", self.data.columns.tolist())

        return self

    def explore_data(self):
        """Step 2: Exploratory Data Analysis"""
        print("\n" + "=" * 50)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 50)

        print(f"\nData shape: {self.data.shape}")
        print(f"Data size: {self.data.size}")
        print(f"\nUnique values:\n{self.data.nunique()}")
        print(f"\nData types:\n{self.data.dtypes}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        print("\nDescriptive statistics:")
        print(self.data.describe().transpose())

        return self

    def visualize_data(self):
        """Step 3: Visualize data"""
        print("\n" + "=" * 50)
        print("STEP 3: DATA VISUALIZATION")
        print("=" * 50)

        # Plot all columns
        self.data.plot(legend=True, subplots=True, figsize=(12, 10))
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_data_overview.png'))
        plt.close()

        # Plot trading data
        cols_plot = ['Open', 'High', 'Low', 'Close', 'Volume']
        axes = self.data[cols_plot].plot(marker='.', alpha=0.7, linestyle='None',
                                         figsize=(11, 9), subplots=True)
        for ax in axes:
            ax.set_ylabel('Daily trade')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_trading_data.png'))
        plt.close()

        # Plot close price
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Close'], label="Close price")
        plt.xlabel("Timestamp")
        plt.ylabel("Closing price")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_close_price.png'))
        plt.close()

        return self

    def prepare_features(self):
        """Step 4: Feature engineering and data preparation"""
        print("\n" + "=" * 50)
        print("STEP 4: FEATURE ENGINEERING")
        print("=" * 50)

        self.data.reset_index(drop=True, inplace=True)

        X = []
        Y = []

        print(f"Creating sequences with window size: {self.window_size}")

        for i in range(1, len(self.data) - self.window_size - 1, 1):
            first = self.data.iloc[i, 2]  # Close price column
            temp = []
            temp2 = []

            # Create window of normalized prices
            for j in range(self.window_size):
                temp.append((self.data.iloc[i + j, 2] - first) / first)

            # Target value (next price after window)
            temp2.append((self.data.iloc[i + self.window_size, 2] - first) / first)

            X.append(np.array(temp).reshape(self.window_size, 1))
            Y.append(np.array(temp2).reshape(1, 1))

        print(f"Total sequences created: {len(X)}")

        return X, Y

    def split_data(self, X, Y, test_size=0.2):
        """Step 5: Split data into train and test sets"""
        print("\n" + "=" * 50)
        print("STEP 5: TRAIN/TEST SPLIT")
        print("=" * 50)

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, shuffle=True
        )

        self.train_X = np.array(x_train)
        self.test_X = np.array(x_test)
        self.train_Y = np.array(y_train)
        self.test_Y = np.array(y_test)

        # Reshape for model input
        self.train_X = self.train_X.reshape(self.train_X.shape[0], 1, self.window_size, 1)
        self.test_X = self.test_X.reshape(self.test_X.shape[0], 1, self.window_size, 1)

        print(f"Training samples: {len(self.train_X)}")
        print(f"Testing samples: {len(self.test_X)}")
        print(f"Train X shape: {self.train_X.shape}")
        print(f"Test X shape: {self.test_X.shape}")

        return self

    def build_model(self):
        """Step 6: Build LSTM + DNN model"""
        print("\n" + "=" * 50)
        print("STEP 6: MODEL BUILDING")
        print("=" * 50)

        # LSTM layers
        lstm_1 = LSTM(16, activation='tanh', return_sequences=True,
                      input_shape=(self.train_X.shape[1], self.window_size))
        lstm_2 = LSTM(32, activation='tanh')
        dense_layer = Dense(64, activation='relu')

        # Create sequential model
        self.model = Sequential([lstm_1, lstm_2, dense_layer])

        # DNN layers
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))

        self.model.add(Dense(64))
        self.model.add(Activation('relu'))

        self.model.add(Dense(64))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())

        # Output layer
        self.model.add(Dense(1, activation='linear'))

        # Compile model
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

        print("\nModel architecture:")
        print(self.model.summary())

        return self

    def train_model(self, epochs=100, batch_size=64):
        """Step 7: Train the model"""
        print("\n" + "=" * 50)
        print("STEP 7: MODEL TRAINING")
        print("=" * 50)

        print(f"Training for {epochs} epochs with batch size {batch_size}...")

        self.history = self.model.fit(
            self.train_X, self.train_Y,
            validation_data=(self.test_X, self.test_Y),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=False
        )

        print("\nTraining completed!")

        return self

    def visualize_training(self):
        """Step 8: Visualize training history"""
        print("\n" + "=" * 50)
        print("STEP 8: TRAINING VISUALIZATION")
        print("=" * 50)

        # Loss plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['loss'], label='train loss')
        plt.plot(self.history.history['val_loss'], label='val loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        # MSE plot
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['mse'], label='train mse')
        plt.plot(self.history.history['val_mse'], label='val mse')
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.title("Mean Squared Error")

        # MAE plot
        plt.subplot(1, 3, 3)
        plt.plot(self.history.history['mae'], label='train mae')
        plt.plot(self.history.history['val_mae'], label='val mae')
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.title("Mean Absolute Error")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_training_history.png'))
        plt.close()

        return self

    def evaluate_model(self):
        """Step 9: Evaluate model performance"""
        print("\n" + "=" * 50)
        print("STEP 9: MODEL EVALUATION")
        print("=" * 50)

        # Model evaluation
        evaluation = self.model.evaluate(self.test_X, self.test_Y)
        print(f"\nTest Loss: {evaluation[0]:.6f}")
        print(f"Test MSE: {evaluation[1]:.6f}")
        print(f"Test MAE: {evaluation[2]:.6f}")

        # Predictions
        yhat_probs = self.model.predict(self.test_X, verbose=0)
        yhat_probs = yhat_probs[:, 0]

        # Calculate metrics
        var = explained_variance_score(self.test_Y.reshape(-1, 1), yhat_probs)
        r2 = r2_score(self.test_Y.reshape(-1, 1), yhat_probs)
        max_err = max_error(self.test_Y.reshape(-1, 1), yhat_probs)

        print(f"\nExplained Variance: {var:.6f}")
        print(f"R2 Score: {r2:.6f}")
        print(f"Max Error: {max_err:.6f}")

        return yhat_probs

    #@staticmethod
    def visualize_predictions(self, predicted, test_label):
        """Step 10: Visualize predictions"""
        print("\n" + "=" * 50)
        print("STEP 10: PREDICTION VISUALIZATION")
        print("=" * 50)

        plt.figure(figsize=(12, 6))
        plt.plot(predicted, color='green', label='Predicted Stock Price')
        plt.plot(test_label, color='yellow', label='Real Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_predictions.png'))
        plt.close()

        # Create comparison table
        predicted_flat = [val[0] if isinstance(val, (list, np.ndarray)) else val
                          for val in predicted]
        test_label_flat = [val[0] if isinstance(val, (list, np.ndarray)) else val
                           for val in test_label]

        res = pd.DataFrame({
            'Actual Price': test_label_flat,
            'Predicted Price': predicted_flat,
        })

        print("\nPrediction Results (first 20 rows):")
        print(res.head(20))

        print("\n" + tabulate(res.head(20), headers='keys', tablefmt='pretty'))

        return res

    def save_model_architecture(self, filename='model.png'):
        """Save model architecture diagram"""
        plot_model(
            self.model,
            to_file=os.path.join(self.save_dir, filename),
            show_shapes=True,
            show_layer_names=True,
            dpi=96
        )
        print(f"\nModel architecture saved to {filename}")

        return self

    def run_pipeline(self, epochs=100, batch_size=64):
        """Execute complete pipeline"""
        print("\n" + "=" * 50)
        print("STOCK PRICE PREDICTION PIPELINE")
        print("=" * 50)

        # Execute all steps
        self.load_data()
        self.explore_data()
        self.visualize_data()

        X, Y = self.prepare_features()
        self.split_data(X, Y)

        self.build_model()
        self.train_model(epochs=epochs, batch_size=batch_size)
        self.visualize_training()

        predictions = self.evaluate_model()

        # For visualization, use test_Y as test_label
        results = self.visualize_predictions(predictions, self.test_Y.reshape(-1))

        self.save_model_architecture()

        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        return self
