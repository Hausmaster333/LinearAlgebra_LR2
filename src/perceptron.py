from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

EPS = 1e-12

@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)

class Perceptron:
    """Single-layer perceptron trained with mini-batch gradient descent."""

    def __init__(
        self,
        input_dim: int,
        init: str = "small_random",
        loss: str = "bce",
        l2: float = 0.0,
        random_state: int | None = 42,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if init not in {"zero", "small_random", "large_random"}:
            raise ValueError(f"unknown init: {init}")
        if loss not in {"bce", "hinge"}:
            raise ValueError(f"unknown loss: {loss}")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")

        self.input_dim = input_dim
        self.init = init
        self.loss = loss
        self.l2 = float(l2)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.w = self._initial_weights()
        self.b = 0.0

    def _initial_weights(self) -> np.ndarray:
        if self.init == "zero":
            return np.zeros(self.input_dim, dtype=float)
        if self.init == "small_random":
            return self.rng.normal(0.0, 0.01, size=self.input_dim)
        return self.rng.normal(0.0, 10.0, size=self.input_dim)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z, dtype=float)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    def linear_output(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(self.linear_output(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self.loss == "hinge":
            return (self.linear_output(X) >= 0.0).astype(int)
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_loss(self, y_true: np.ndarray, y_pred_or_score: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        values = np.asarray(y_pred_or_score, dtype=float)
        if self.loss == "bce":
            y_pred = np.clip(values, EPS, 1.0 - EPS)
            data_loss = -np.mean(
                y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
            )
        else:
            y_hinge = np.where(y_true > 0, 1.0, -1.0)
            data_loss = np.mean(np.maximum(0.0, 1.0 - y_hinge * values))
        return float(data_loss + 0.5 * self.l2 * np.sum(self.w * self.w))

    def _batch_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, float]:
        m = X_batch.shape[0]
        if self.loss == "bce":
            y_pred = self.forward(X_batch)
            error = y_pred - y_batch
            dw = X_batch.T @ error / m
            db = float(np.mean(error))
        else:
            score = self.linear_output(X_batch)
            y_hinge = np.where(y_batch > 0, 1.0, -1.0)
            active = y_hinge * score < 1.0
            if np.any(active):
                dw = -(X_batch[active].T @ y_hinge[active]) / m
                db = float(-np.sum(y_hinge[active]) / m)
            else:
                dw = np.zeros_like(self.w)
                db = 0.0
        if self.l2:
            dw = dw + self.l2 * self.w
        return dw, db

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        lr: float = 0.1,
        batch_size: int = 32,
        momentum: float = 0.0,
    ) -> TrainingHistory:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        X_val = np.asarray(X_val, dtype=float)
        y_val = np.asarray(y_val, dtype=float)
        velocity_w = np.zeros_like(self.w)
        velocity_b = 0.0
        history = TrainingHistory()

        for _ in range(epochs):
            order = self.rng.permutation(X_train.shape[0])
            for start in range(0, X_train.shape[0], batch_size):
                idx = order[start : start + batch_size]
                dw, db = self._batch_gradient(X_train[idx], y_train[idx])
                velocity_w = momentum * velocity_w + lr * dw
                velocity_b = momentum * velocity_b + lr * db
                self.w -= velocity_w
                self.b -= velocity_b

            train_values = self.forward(X_train) if self.loss == "bce" else self.linear_output(X_train)
            val_values = self.forward(X_val) if self.loss == "bce" else self.linear_output(X_val)
            history.train_loss.append(self.compute_loss(y_train, train_values))
            history.val_loss.append(self.compute_loss(y_val, val_values))
            history.train_accuracy.append(float(np.mean(self.predict(X_train) == y_train)))
            history.val_accuracy.append(float(np.mean(self.predict(X_val) == y_val)))

        return history

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

