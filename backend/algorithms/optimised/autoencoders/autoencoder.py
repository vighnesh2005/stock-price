"""
Autoencoder for Stock Price Prediction

Optimized implementation using TensorFlow/Keras for feature extraction and stock price prediction.
Uses autoencoder for dimensionality reduction and feature learning, then applies regression
for stock price prediction based on OHLCV data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ...model_interface import ModelInterface
from ...stock_indicators import StockIndicators
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging

logger = logging.getLogger(__name__)

class AutoencoderModel(ModelInterface):
    """
    Autoencoder model for stock price prediction.
    
    Uses autoencoder for feature extraction and dimensionality reduction,
    then applies linear regression on the encoded features for prediction.
    """
    
    def __init__(self, encoding_dim: int = 24, hidden_layers: List[int] = [64, 48, 32], 
                 dropout_rate: float = 0.3, **kwargs):
        """
        Initialize Autoencoder model.
        
        Args:
            encoding_dim: Dimension of the encoded representation
            hidden_layers: List of hidden layer sizes for encoder/decoder
            dropout_rate: Dropout rate for regularization
        """
        super().__init__('Autoencoder', **kwargs)
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Models
        self.autoencoder = None
        self.encoder = None
        self.regressor = None
        self.scaler = None
        self.feature_columns = None
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _build_autoencoder(self, input_dim: int) -> Tuple[Model, Model]:
        """Build autoencoder and encoder models."""
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for hidden_size in self.hidden_layers:
            encoded = Dense(hidden_size, activation='relu')(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(self.dropout_rate)(encoded)
        
        # Bottleneck
        encoded = Dense(self.encoding_dim, activation='relu', name='encoded')(encoded)
        
        # Decoder
        decoded = encoded
        for hidden_size in reversed(self.hidden_layers):
            decoded = Dense(hidden_size, activation='relu')(decoded)
            decoded = BatchNormalization()(decoded)
            decoded = Dropout(self.dropout_rate)(decoded)
        
        # Output layer - use linear activation for regression (not sigmoid)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Create models
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        encoder = Model(inputs=input_layer, outputs=encoded)
        
        return autoencoder, encoder
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the autoencoder and regression model.
        
        Args:
            X: Input features (OHLC data with technical indicators)
            y: Target values (stock prices)
        """
        self.validate_input(X, y)
        
        try:
            # Scale features using StandardScaler for better performance with regression
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Build autoencoder
            self.autoencoder, self.encoder = self._build_autoencoder(X.shape[1])
            
            # Compile autoencoder
            self.autoencoder.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            # Train autoencoder
            logger.info("Training autoencoder...")
            history = self.autoencoder.fit(
                X_scaled, X_scaled,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # CRITICAL: Prevent time-series data leakage
            )
            
            # Extract encoded features
            encoded_features = self.encoder.predict(X_scaled)
            
            # Train non-linear regression model on encoded features
            logger.info("Training non-linear regression model on encoded features...")
            self.regressor = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
            self.regressor.fit(encoded_features, y)
            
            # Set trained flag before calculating metrics
            self.is_trained = True
            
            # Calculate training metrics
            y_pred = self.predict(X)
            self.training_metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2_score': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'reconstruction_loss': history.history['loss'][-1],
                'val_reconstruction_loss': history.history['val_loss'][-1]
            }
            logger.info(f"Autoencoder training completed. R² Score: {self.training_metrics['r2_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training autoencoder: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained autoencoder and regression model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.validate_input(X)
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Extract encoded features
            encoded_features = self.encoder.predict(X_scaled)
            
            # Make predictions
            predictions = self.regressor.predict(encoded_features)
            
            # Convert numpy float32 to Python float for JSON serialization
            return predictions.astype(float)
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def get_encoded_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract encoded features from input data.
        
        Args:
            X: Input features
            
        Returns:
            Encoded features
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled)
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for each sample.
        
        Args:
            X: Input features
            
        Returns:
            Reconstruction errors
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled)
        return np.mean(np.square(X_scaled - reconstructed), axis=1)
    
    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            # Save autoencoder and encoder
            self.autoencoder.save(f"{path}_autoencoder.keras")
            self.encoder.save(f"{path}_encoder.keras")
            
            # Save other components
            joblib.dump({
                'regressor': self.regressor,
                'scaler': self.scaler,
                'encoding_dim': self.encoding_dim,
                'hidden_layers': self.hidden_layers,
                'dropout_rate': self.dropout_rate,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'model_params': self.model_params
            }, f"{path}_metadata.pkl")
            
            logger.info(f"Autoencoder model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving autoencoder model: {str(e)}")
            raise
    
    def load(self, path: str) -> 'ModelInterface':
        """Load a previously saved model from disk."""
        from tensorflow.keras.models import load_model
        
        try:
            # Support both new .keras and legacy .h5 formats
            import os
            keras_path = f"{path}_autoencoder.keras"
            h5_path = f"{path}_autoencoder.h5"
            
            if os.path.exists(keras_path):
                self.autoencoder = load_model(keras_path, compile=False)
                self.encoder = load_model(f"{path}_encoder.keras", compile=False)
            elif os.path.exists(h5_path):
                self.autoencoder = load_model(h5_path, compile=False)
                self.encoder = load_model(f"{path}_encoder.h5", compile=False)
            else:
                raise FileNotFoundError(f"Could not find model file at {keras_path} or {h5_path}")
            
            # Load metadata
            metadata = joblib.load(f"{path}_metadata.pkl")
            self.regressor = metadata['regressor']
            self.scaler = metadata['scaler']
            self.encoding_dim = metadata['encoding_dim']
            self.hidden_layers = metadata['hidden_layers']
            self.dropout_rate = metadata['dropout_rate']
            self.feature_columns = metadata['feature_columns']
            self.training_metrics = metadata['training_metrics']
            self.model_params = metadata['model_params']
            
            self.is_trained = True
            logger.info(f"Autoencoder model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading autoencoder model: {str(e)}")
            raise
        
        return self
