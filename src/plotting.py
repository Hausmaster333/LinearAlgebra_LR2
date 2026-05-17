from __future__ import annotations
import os
from pathlib import Path

_mpl_config_dir = Path("tmp/matplotlib").resolve()
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.metrics import roc_curve_points
from src.perceptron import Perceptron

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 120,
        "savefig.dpi": 160,
    }
)

def ensure_parent(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    return path

def save_loss_plot(histories: dict[str, object], path: Path | str, title: str, val_only: bool = False) -> str:
    path = ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, history in histories.items():
        if not val_only:
            ax.plot(history.train_loss, linestyle="--", alpha=0.65, label=f"{label} train")
        ax.plot(history.val_loss, label=f"{label} val")
    ax.set_title(title)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return str(path)

def save_accuracy_plot(histories: dict[str, object], path: Path | str, title: str) -> str:
    path = ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, history in histories.items():
        ax.plot(history.val_accuracy, label=label)
    ax.set_title(title)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return str(path)

def save_decision_boundary(
    model: Perceptron,
    X: np.ndarray,
    y: np.ndarray,
    path: Path | str,
    title: str,
    error_mask: np.ndarray | None = None,
) -> str:
    path = ensure_parent(path)
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict_proba(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap="RdYlBu_r", alpha=0.35)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=24, edgecolor="white", linewidth=0.4)
    if error_mask is not None and np.any(error_mask):
        ax.scatter(
            X[error_mask, 0],
            X[error_mask, 1],
            facecolors="none",
            edgecolors="black",
            s=90,
            linewidth=1.4,
            label="Ошибки",
        )
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return str(path)

def save_roc_plot(y_true: np.ndarray, scores: np.ndarray, auc: float, path: Path | str, title: str) -> str:
    path = ensure_parent(path)
    fpr, tpr = roc_curve_points(y_true, scores)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Случайный классификатор")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return str(path)

def save_bar_plot(
    labels: list[str],
    values: list[float],
    path: Path | str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> str:
    path = ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(labels, values, color="#4472C4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return str(path)
