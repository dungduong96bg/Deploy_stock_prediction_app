"""
LSTM-DNN Hybrid Model for Stock Market Prediction
Core model architecture implementation
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Flatten, Concatenate, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class LSTMDNNModel:
    """
    Hybrid LSTM-DNN Architecture for Time Series Prediction

    Architecture:
    1. LSTM Branch: Captures temporal dependencies
    2. DNN Branch: Extracts features from flattened input
    3. Merged Layer: Combines both branches
    4. Output Layer: Final prediction
    """

    def __init__(self,
                 input_shape,
                 lstm_units=[128, 64],
                 dnn_units=[64, 32],
                 dropout_lstm=0.2,
                 dropout_dnn=0.3,
                 learning_rate=0.001):
        """
        Initialize the LSTM-DNN hybrid model

        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (timesteps, features)
        lstm_units : list
            Number of units in each LSTM layer
        dnn_units : list
            Number of units in each DNN layer
        dropout_lstm : float
            Dropout rate for LSTM layers
        dropout_dnn : float
            Dropout rate for DNN layers
        learning_rate : float
            Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dnn_units = dnn_units
        self.dropout_lstm = dropout_lstm
        self.dropout_dnn = dropout_dnn
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self):
        """
        Build the hybrid LSTM-DNN model

        Returns:
        --------
        model : keras.Model
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape, name='input_layer')

        # ==================== LSTM BRANCH ====================
        # First LSTM layer with return sequences
        lstm_out = LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            name='lstm_layer_1'
        )(inputs)
        lstm_out = Dropout(self.dropout_lstm, name='lstm_dropout_1')(lstm_out)

        # Second LSTM layer
        lstm_out = LSTM(
            units=self.lstm_units[1],
            return_sequences=False,
            name='lstm_layer_2'
        )(lstm_out)
        lstm_out = Dropout(self.dropout_lstm, name='lstm_dropout_2')(lstm_out)

        # Optional: Batch Normalization for LSTM output
        lstm_out = BatchNormalization(name='lstm_batch_norm')(lstm_out)

            # ==================== DNN BRANCH ====================
        # Flatten the input for DNN processing
        dnn_input = Flatten(name='flatten_layer')(inputs)

        # First Dense layer
        dnn_out = Dense(
            units=self.dnn_units[0],
            activation='relu',
            name='dnn_layer_1'
        )(dnn_input)
        dnn_out = Dropout(self.dropout_dnn, name='dnn_dropout_1')(dnn_out)

        # Second Dense layer
        dnn_out = Dense(
            units=self.dnn_units[1],
            activation='relu',
            name='dnn_layer_2'
        )(dnn_out)
        dnn_out = Dropout(self.dropout_dnn, name='dnn_dropout_2')(dnn_out)

        # Optional: Batch Normalization for DNN output
        dnn_out = BatchNormalization(name='dnn_batch_norm')(dnn_out)

        # ==================== MERGE BRANCHES ====================
        # Concatenate LSTM and DNN outputs
        merged = Concatenate(name='merge_layer')([lstm_out, dnn_out])

        # ==================== FINAL LAYERS ====================
        # Dense layer after merging
        final_dense = Dense(
            units=32,
            activation='relu',
            name='final_dense'
        )(merged)
        final_dense = Dropout(0.2, name='final_dropout')(final_dense)

        # Output layer (single value prediction)
        output = Dense(
            units=1,
            activation='linear',
            name='output_layer'
        )(final_dense)

        # Create the model
        self.model = Model(inputs=inputs, outputs=output, name='LSTM_DNN_Hybrid')

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )

        return self.model

    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()

    def get_callbacks(self, patience_early=15, patience_lr=5, checkpoint_path=None):
        """
        Get training callbacks

        Parameters:
        -----------
        patience_early : int
            Patience for early stopping
        patience_lr : int
            Patience for learning rate reduction
        checkpoint_path : str
            Path to save best model checkpoints

        Returns:
        --------
        callbacks : list
            List of Keras callbacks
        """
        callbacks = []

        # Early Stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience_early,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)

        # Reduce Learning Rate on Plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # Model Checkpoint (optional)
        if checkpoint_path:
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)

        return callbacks

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=32, callbacks=None, verbose=1):
        """
        Train the model

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training targets
        X_val : numpy.ndarray
            Validation features
        y_val : numpy.ndarray
            Validation targets
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        callbacks : list
            List of Keras callbacks
        verbose : int
            Verbosity mode

        Returns:
        --------
        history : keras.callbacks.History
            Training history
        """
        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Get callbacks if not provided
        if callbacks is None:
            callbacks = self.get_callbacks()

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X):
        """
        Make predictions

        Parameters:
        -----------
        X : numpy.ndarray
            Input features

        Returns:
        --------
        predictions : numpy.ndarray
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not built yet! Call build_model() first.")

        return self.model.predict(X)

    def save_model(self, filepath):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_weights(self, filepath):
        """Load model weights"""
        if self.model is None:
            self.build_model()
        self.model.load_weights(filepath)
        print(f"Weights loaded from {filepath}")


# ==================== ALTERNATIVE ARCHITECTURES ====================

def build_simple_lstm_dnn(input_shape, lstm_units=128, dnn_units=64):
    """
    Simplified LSTM-DNN model (single layer per branch)

    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    lstm_units : int
        Number of LSTM units
    dnn_units : int
        Number of DNN units

    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)

    # LSTM Branch
    lstm_out = LSTM(lstm_units)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)

    # DNN Branch
    dnn_out = Flatten()(inputs)
    dnn_out = Dense(dnn_units, activation='relu')(dnn_out)
    dnn_out = Dropout(0.3)(dnn_out)

    # Merge and Output
    merged = Concatenate()([lstm_out, dnn_out])
    output = Dense(1, activation='linear')(merged)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

    return model


def build_deep_lstm_dnn(input_shape, lstm_units=[128, 64, 32], dnn_units=[128, 64, 32]):
    """
    Deep LSTM-DNN model (multiple layers per branch)

    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    lstm_units : list
        Number of units in each LSTM layer
    dnn_units : list
        Number of units in each DNN layer

    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)

    # Deep LSTM Branch
    lstm_out = inputs
    for i, units in enumerate(lstm_units[:-1]):
        lstm_out = LSTM(units, return_sequences=True, name=f'lstm_{i + 1}')(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(lstm_units[-1], name=f'lstm_{len(lstm_units)}')(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)

    # Deep DNN Branch
    dnn_out = Flatten()(inputs)
    for i, units in enumerate(dnn_units):
        dnn_out = Dense(units, activation='relu', name=f'dnn_{i + 1}')(dnn_out)
        dnn_out = Dropout(0.3)(dnn_out)

    # Merge and Output
    merged = Concatenate()([lstm_out, dnn_out])
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    output = Dense(1, activation='linear')(merged)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

    return model


def build_attention_lstm_dnn(input_shape):
    """
    LSTM-DNN model with attention mechanism

    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)

    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    from tensorflow.keras.layers import Attention, Permute, Multiply

    inputs = Input(shape=input_shape)

    # LSTM with Attention
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)

    # Apply attention
    lstm_attention = Multiply()([lstm_out, attention])
    lstm_attention = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(lstm_attention)

    # DNN Branch
    dnn_out = Flatten()(inputs)
    dnn_out = Dense(64, activation='relu')(dnn_out)
    dnn_out = Dropout(0.3)(dnn_out)

    # Merge and Output
    merged = Concatenate()([lstm_attention, dnn_out])
    merged = Dense(32, activation='relu')(merged)
    output = Dense(1, activation='linear')(merged)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

    return model


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("LSTM-DNN Hybrid Model Architecture")
    print("=" * 70)
    print()

    # Example configuration
    timesteps = 60  # Look back 60 days
    features = 5  # OHLCV data
    input_shape = (timesteps, features)

    # Initialize the model
    print("1. Creating LSTM-DNN Hybrid Model...")
    lstm_dnn = LSTMDNNModel(
        input_shape=input_shape,
        lstm_units=[128, 64],
        dnn_units=[64, 32],
        dropout_lstm=0.2,
        dropout_dnn=0.3,
        learning_rate=0.001
    )

    # Build the model
    model = lstm_dnn.build_model()

    # Display model architecture
    print("\n2. Model Architecture:")
    print("-" * 70)
    lstm_dnn.get_model_summary()

    # Generate sample data for demonstration
    print("\n3. Generating Sample Data...")
    print("-" * 70)
    n_samples = 1000
    X_train = np.random.randn(n_samples, timesteps, features)
    y_train = np.random.randn(n_samples, 1)
    X_val = np.random.randn(200, timesteps, features)
    y_val = np.random.randn(200, 1)
    print(f"Training samples: {X_train.shape}")
    print(f"Validation samples: {X_val.shape}")

    # Get callbacks
    callbacks = lstm_dnn.get_callbacks(
        patience_early=10,
        patience_lr=5,
        checkpoint_path='best_model.h5'
    )

    print("\n4. Model Ready for Training!")
    print("-" * 70)
    print("Use the following code to train:")
    print()
    print("history = lstm_dnn.train(")
    print("    X_train, y_train,")
    print("    X_val, y_val,")
    print("    epochs=100,")
    print("    batch_size=32")
    print(")")
    print()

    # Train for a few epochs as demonstration
    print("5. Running Quick Training Demo (5 epochs)...")
    print("-" * 70)
    history = lstm_dnn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=32,
        callbacks=callbacks
    )

    # Make predictions
    print("\n6. Making Predictions...")
    print("-" * 70)
    predictions = lstm_dnn.predict(X_val[:10])
    print(f"Sample predictions shape: {predictions.shape}")
    print(f"First 5 predictions: {predictions[:5].flatten()}")

    print("\n" + "=" * 70)
    print("LSTM-DNN Model Demo Complete!")
    print("=" * 70)