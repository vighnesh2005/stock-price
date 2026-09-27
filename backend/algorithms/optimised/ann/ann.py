"""
Artificial Neural Network (ANN) for Stock Price Prediction

Optimized implementation using TensorFlow/Keras for stock price prediction.
Uses multi-layer Perceptron with Dropout, Batch Normalization, and RobustScaler.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import sys
import os
import joblib
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

# Import model_interface and stock_indicators safely for module and direct execution
try:
    from ...model_interface import ModelInterface
    from ...stock_indicators import StockIndicators
except (ImportError, ValueError):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from model_interface import ModelInterface
    from stock_indicators import StockIndicators


class ANNModel(ModelInterface):
    """
    Optimized Artificial Neural Network model for stock price prediction.
    
    Uses Dense, BatchNormalization, and Dropout layers scaled with RobustScaler.
    """

    def __init__(self, model_name: str = "Optimized ANN", **kwargs):
        super().__init__(model_name, **kwargs)
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANNModel":
        """
        Train the ANN model on stock data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - stock prices
            
        Returns:
            self: Returns self for method chaining
        """
        self.validate_input(X, y)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y_clean = np.nan_to_num(y, nan=np.nanmean(y), posinf=1e6, neginf=-1e6)

        X_scaled = self.scaler.fit_transform(X_clean)

        self.model = Sequential([
            Dense(
                128,
                activation="relu",
                input_shape=(X_scaled.shape[1],)
            ),
            BatchNormalization(),
            Dropout(0.3),

            Dense(
                64,
                activation="relu"
            ),
            BatchNormalization(),
            Dropout(0.2),

            Dense(
                1,
                activation="linear"
            )
        ])

        self.model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_scaled,
            y_clean,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            shuffle=False,
            callbacks=[early_stopping],
            verbose=0
        )

        self.is_trained = True

        y_pred = self.predict(X_clean)
        mse = float(mean_squared_error(y_clean, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_clean, y_pred))
        mae = float(mean_absolute_error(y_clean, y_pred))

        self.set_training_metrics({
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "mae": mae,
            "loss": float(min(history.history["loss"])),
            "val_loss": float(min(history.history["val_loss"]))
        })

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new stock data.
        
        Args:
            X: Features to predict on (n_samples, n_features)
            
        Returns:
            predictions: Predicted stock prices (n_samples,)
        """
        if self.model is None or not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        self.validate_input(X)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X_clean)

        predictions = self.model.predict(
            X_scaled,
            verbose=0
        )

        return predictions.flatten()

    def save(self, path: str) -> None:
        """Save the trained model and scaler/metadata to disk."""
        if self.model is None or not self.is_trained:
            raise ValueError("Cannot save an untrained model")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if path.endswith('.keras'):
            model_path = path
            meta_path = path.rsplit('.keras', 1)[0] + '_metadata.pkl'
        elif path.endswith('.pkl'):
            model_path = path.rsplit('.pkl', 1)[0] + '.keras'
            meta_path = path
        else:
            model_path = f"{path}.keras"
            meta_path = f"{path}_metadata.pkl"

        self.model.save(model_path)

        joblib.dump({
            'scaler': self.scaler,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'feature_columns': getattr(self, 'feature_columns', None)
        }, meta_path)

    def load(self, path: str) -> "ANNModel":
        """Load a previously saved model and scaler/metadata from disk."""
        if path.endswith('.keras'):
            model_path = path
            meta_path = path.rsplit('.keras', 1)[0] + '_metadata.pkl'
        elif path.endswith('.pkl'):
            model_path = path.rsplit('.pkl', 1)[0] + '.keras'
            meta_path = path
        else:
            model_path = f"{path}.keras"
            meta_path = f"{path}_metadata.pkl"

        if os.path.exists(model_path):
            self.model = load_model(model_path)
        elif os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {path}")

        if os.path.exists(meta_path):
            data = joblib.load(meta_path)
            self.scaler = data.get('scaler', RobustScaler())
            self.training_metrics = data.get('training_metrics', {})
            self.model_params = data.get('model_params', {})
            self.feature_columns = data.get('feature_columns', None)

        self.is_trained = True
        return self


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    prices = 100 + np.cumsum(np.random.normal(0, 1, 200))

    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices
    })

    model = ANNModel()
    df_features = StockIndicators.calculate_all_indicators(df)
    X, y = StockIndicators.prepare_training_data(df_features)

    if len(X) > 0:
        model.fit(X, y)
        preds = model.predict(X[:5])
        print("ANN Training Completed Successfully.")
        print(f"R² Score: {model.training_metrics['r2_score']:.4f}")
        print(f"Sample Predictions: {preds}")
