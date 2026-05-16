from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

@dataclass
class DataBundle:
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray

def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std

def prepare_base_data(random_state: int = 42) -> DataBundle:
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1,
    )
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_test, mean, std = standardize_train_test(X_train_raw, X_test_raw)
    return DataBundle(X_train_raw, X_test_raw, X_train, X_test, y_train, y_test, mean, std)

def add_label_noise(y: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    if not 0 <= noise <= 1:
        raise ValueError("noise must be in [0, 1]")
    y_noisy = y.copy()
    flips = rng.random(y.shape[0]) < noise
    y_noisy[flips] = 1 - y_noisy[flips]
    return y_noisy

def generate_linear_data(
    n_samples: int = 500,
    centers: tuple[tuple[float, float], tuple[float, float]] = ((-2.0, -2.0), (2.0, 2.0)),
    covariance: tuple[tuple[float, float], tuple[float, float]] = ((0.55, 0.15), (0.15, 0.55)),
    noise: float = 0.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0
    X0 = rng.multivariate_normal(np.array(centers[0]), np.array(covariance), size=n0)
    X1 = rng.multivariate_normal(np.array(centers[1]), np.array(covariance), size=n1)
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    order = rng.permutation(n_samples)
    return X[order], add_label_noise(y[order], noise, rng)

def generate_xor_data(
    n_samples: int = 500,
    spread: float = 0.35,
    noise: float = 0.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    centers = np.array([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
    labels = np.array([0, 1, 1, 0])
    choices = rng.integers(0, len(centers), size=n_samples)
    X = centers[choices] + rng.normal(0.0, spread, size=(n_samples, 2))
    y = labels[choices]
    return X, add_label_noise(y, noise, rng)

def generate_circle_data(
    n_samples: int = 500,
    radius: float = 1.0,
    noise: float = 0.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    y = (np.sum(X * X, axis=1) <= radius * radius).astype(int)
    return X, add_label_noise(y, noise, rng)

def stratified_kfold_indices(
    y: np.ndarray, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        for fold_index, part in enumerate(np.array_split(cls_indices, n_splits)):
            folds[fold_index].extend(part.tolist())

    result = []
    all_indices = np.arange(len(y))
    for fold in folds:
        val_idx = np.array(sorted(fold), dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=True)
        result.append((train_idx, val_idx))
    return result
