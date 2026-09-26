import numpy as np

from sklearn.preprocessing import RobustScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from backend.algorithms.model_interface import ModelInterface


class ANNModel(ModelInterface):
    def __init__(self, model_name: str = "Optimized ANN", **kwargs):
        super().__init__(model_name, **kwargs)

        self.model = None
        self.scaler = RobustScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANNModel":
        self.validate_input(X, y)

        X_scaled = self.scaler.fit_transform(X)

        self.model = Sequential([
            Dense(
                128,
                activation="relu",
                input_shape=(X.shape[1],)
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
            y,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            shuffle=False,
            callbacks=[early_stopping],
            verbose=0
        )

        self.set_training_metrics({
            "mae": float(min(history.history["mae"])),
            "val_mae": float(min(history.history["val_mae"])),
            "loss": float(min(history.history["loss"])),
            "val_loss": float(min(history.history["val_loss"]))
        })

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        self.validate_input(X)

        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(
            X_scaled,
            verbose=0
        )

        return predictions.flatten()

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save an untrained model")

        self.model.save(path)

    def load(self, path: str) -> "ANNModel":
        self.model = load_model(path)
        self.is_trained = True

        return self
