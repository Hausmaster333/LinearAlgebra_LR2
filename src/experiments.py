from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import (
    generate_circle_data,
    generate_linear_data,
    generate_xor_data,
    prepare_base_data,
    standardize_train_test,
    stratified_kfold_indices,
)
from src.metrics import classification_metrics
from src.perceptron import Perceptron, TrainingHistory
from src.plotting import (
    save_accuracy_plot,
    save_bar_plot,
    save_decision_boundary,
    save_loss_plot,
    save_roc_plot,
)


BASE_EPOCHS = 100
BASE_LR = 0.1
BASE_BATCH_SIZE = 32
RANDOM_STATE = 42


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_table(rows: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    init: str = "small_random",
    loss: str = "bce",
    l2: float = 0.0,
    epochs: int = BASE_EPOCHS,
    lr: float = BASE_LR,
    batch_size: int = BASE_BATCH_SIZE,
    momentum: float = 0.0,
    random_state: int = RANDOM_STATE,
) -> tuple[Perceptron, TrainingHistory, dict[str, float], dict[str, float]]:
    model = Perceptron(
        input_dim=X_train.shape[1],
        init=init,
        loss=loss,
        l2=l2,
        random_state=random_state,
    )
    history = model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        momentum=momentum,
    )
    train_metrics = classification_metrics(
        y_train,
        model.predict(X_train),
        model.predict_proba(X_train),
    )
    val_metrics = classification_metrics(
        y_val,
        model.predict(X_val),
        model.predict_proba(X_val),
    )
    return model, history, train_metrics, val_metrics


def _summary_row(
    label: str,
    model: Perceptron,
    history: TrainingHistory,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "label": label,
        **extra,
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics.get("precision", np.nan),
        "test_recall": test_metrics.get("recall", np.nan),
        "test_f1": test_metrics.get("f1", np.nan),
        "test_roc_auc": test_metrics.get("roc_auc", np.nan),
        "final_train_loss": history.train_loss[-1],
        "final_val_loss": history.val_loss[-1],
        "weight_norm": float(np.linalg.norm(model.w)),
        "bias": float(model.b),
    }


def _split_and_standardize(
    X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_test, _, _ = standardize_train_test(X_train_raw, X_test_raw)
    return X_train, X_test, y_train, y_test


def run_all(output_dir: str | Path = "artifacts") -> dict[str, Any]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_base_data(RANDOM_STATE)
    results: dict[str, Any] = {
        "config": {
            "epochs": BASE_EPOCHS,
            "lr": BASE_LR,
            "batch_size": BASE_BATCH_SIZE,
            "random_state": RANDOM_STATE,
        },
        "figures": {},
        "tables": {},
        "sections": {},
    }

    base_model, base_history, base_train_metrics, base_test_metrics = _train(
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
    )
    base_pred = base_model.predict(data.X_test)
    base_scores = base_model.predict_proba(data.X_test)
    base_rows = [
        {"split": "train", **base_train_metrics},
        {"split": "test", **base_test_metrics},
    ]
    results["tables"]["base_metrics"] = _write_table(base_rows, tables_dir / "base_metrics.csv")
    results["sections"]["base"] = _summary_row(
        "base",
        base_model,
        base_history,
        base_train_metrics,
        base_test_metrics,
        lr=BASE_LR,
        batch_size=BASE_BATCH_SIZE,
        init="small_random",
    )
    results["figures"]["base_loss"] = save_loss_plot(
        {"base": base_history},
        figures_dir / "base_loss.png",
        "Базовое обучение: train и validation loss",
    )
    results["figures"]["base_boundary"] = save_decision_boundary(
        base_model,
        data.X_test,
        data.y_test,
        figures_dir / "base_decision_boundary.png",
        "Разделяющая граница базовой модели",
    )
    results["figures"]["base_errors"] = save_decision_boundary(
        base_model,
        data.X_test,
        data.y_test,
        figures_dir / "base_errors.png",
        "Ошибочно классифицированные точки",
        error_mask=base_pred != data.y_test,
    )
    results["figures"]["roc"] = save_roc_plot(
        data.y_test,
        base_scores,
        base_test_metrics["roc_auc"],
        figures_dir / "roc_curve.png",
        "ROC-кривая базовой модели",
    )

    lr_histories: dict[str, TrainingHistory] = {}
    lr_rows: list[dict[str, Any]] = []
    for lr in [0.001, 0.01, 0.5, 1.0]:
        model, history, train_metrics, test_metrics = _train(
            data.X_train,
            data.y_train,
            data.X_test,
            data.y_test,
            lr=lr,
            random_state=RANDOM_STATE,
        )
        label = f"eta={lr}"
        lr_histories[label] = history
        lr_rows.append(_summary_row(label, model, history, train_metrics, test_metrics, lr=lr))
    results["tables"]["learning_rate"] = _write_table(lr_rows, tables_dir / "learning_rate.csv")
    results["figures"]["learning_rate_loss"] = save_loss_plot(
        lr_histories,
        figures_dir / "learning_rate_loss.png",
        "Влияние скорости обучения на loss",
        val_only=True,
    )
    results["figures"]["learning_rate_accuracy"] = save_accuracy_plot(
        lr_histories,
        figures_dir / "learning_rate_accuracy.png",
        "Влияние скорости обучения на accuracy",
    )
    results["sections"]["learning_rate_best"] = max(lr_rows, key=lambda row: row["test_accuracy"])

    batch_histories: dict[str, TrainingHistory] = {}
    batch_rows: list[dict[str, Any]] = []
    for batch_size in [1, 16, 64, 256]:
        model, history, train_metrics, test_metrics = _train(
            data.X_train,
            data.y_train,
            data.X_test,
            data.y_test,
            batch_size=batch_size,
            random_state=RANDOM_STATE,
        )
        label = f"batch={batch_size}"
        batch_histories[label] = history
        batch_rows.append(
            _summary_row(label, model, history, train_metrics, test_metrics, batch_size=batch_size)
        )
    results["tables"]["batch_size"] = _write_table(batch_rows, tables_dir / "batch_size.csv")
    results["figures"]["batch_size_loss"] = save_loss_plot(
        batch_histories,
        figures_dir / "batch_size_loss.png",
        "Влияние размера batch на loss",
        val_only=True,
    )
    results["sections"]["batch_size_best"] = max(batch_rows, key=lambda row: row["test_accuracy"])

    init_histories: dict[str, TrainingHistory] = {}
    init_rows: list[dict[str, Any]] = []
    for init in ["zero", "small_random", "large_random"]:
        model, history, train_metrics, test_metrics = _train(
            data.X_train,
            data.y_train,
            data.X_test,
            data.y_test,
            init=init,
            random_state=RANDOM_STATE,
        )
        init_histories[init] = history
        init_rows.append(_summary_row(init, model, history, train_metrics, test_metrics, init=init))
    results["tables"]["initialization"] = _write_table(init_rows, tables_dir / "initialization.csv")
    results["figures"]["initialization_loss"] = save_loss_plot(
        init_histories,
        figures_dir / "initialization_loss.png",
        "Влияние инициализации весов на loss",
        val_only=True,
    )

    custom_generators = {
        "linear": generate_linear_data(noise=0.05, random_state=RANDOM_STATE),
        "xor": generate_xor_data(noise=0.05, random_state=RANDOM_STATE),
        "circle": generate_circle_data(noise=0.02, random_state=RANDOM_STATE),
    }
    custom_rows: list[dict[str, Any]] = []
    for name, (X, y) in custom_generators.items():
        X_train, X_test, y_train, y_test = _split_and_standardize(X, y)
        model, history, train_metrics, test_metrics = _train(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=150,
            lr=0.1,
            batch_size=32,
            random_state=RANDOM_STATE,
        )
        custom_rows.append(
            _summary_row(name, model, history, train_metrics, test_metrics, dataset=name)
        )
        results["figures"][f"custom_{name}_boundary"] = save_decision_boundary(
            model,
            X_test,
            y_test,
            figures_dir / f"custom_{name}_boundary.png",
            f"Собственный генератор: {name}",
        )
    results["tables"]["custom_data"] = _write_table(custom_rows, tables_dir / "custom_data.csv")

    loss_histories: dict[str, TrainingHistory] = {}
    loss_rows: list[dict[str, Any]] = []
    for loss in ["bce", "hinge"]:
        model, history, train_metrics, test_metrics = _train(
            data.X_train,
            data.y_train,
            data.X_test,
            data.y_test,
            loss=loss,
            lr=0.1,
            random_state=RANDOM_STATE,
        )
        loss_histories[loss] = history
        loss_rows.append(_summary_row(loss, model, history, train_metrics, test_metrics, loss=loss))
    results["tables"]["loss_comparison"] = _write_table(loss_rows, tables_dir / "loss_comparison.csv")
    results["figures"]["loss_comparison"] = save_loss_plot(
        loss_histories,
        figures_dir / "loss_comparison.png",
        "Сравнение BCE и Hinge loss",
        val_only=True,
    )

    l2_histories: dict[str, TrainingHistory] = {}
    l2_rows: list[dict[str, Any]] = []
    for l2 in [0.0, 0.001, 0.01, 0.1, 1.0]:
        model, history, train_metrics, test_metrics = _train(
            data.X_train,
            data.y_train,
            data.X_test,
            data.y_test,
            l2=l2,
            random_state=RANDOM_STATE,
        )
        label = f"lambda={l2}"
        l2_histories[label] = history
        l2_rows.append(_summary_row(label, model, history, train_metrics, test_metrics, l2=l2))
    results["tables"]["l2_regularization"] = _write_table(
        l2_rows, tables_dir / "l2_regularization.csv"
    )
    results["figures"]["l2_loss"] = save_loss_plot(
        l2_histories,
        figures_dir / "l2_loss.png",
        "Влияние L2-регуляризации на loss",
        val_only=True,
    )
    results["figures"]["l2_weight_norm"] = save_bar_plot(
        [str(row["l2"]) for row in l2_rows],
        [row["weight_norm"] for row in l2_rows],
        figures_dir / "l2_weight_norm.png",
        "Норма весов при разных lambda",
        "||w||",
    )

    momentum_histories: dict[str, TrainingHistory] = {}
    momentum_rows: list[dict[str, Any]] = []
    for beta in [0.0, 0.5, 0.9, 0.99]:
        model, history, train_metrics, test_metrics = _train(
            data.X_train,
            data.y_train,
            data.X_test,
            data.y_test,
            momentum=beta,
            random_state=RANDOM_STATE,
        )
        label = f"beta={beta}"
        momentum_histories[label] = history
        momentum_rows.append(
            _summary_row(label, model, history, train_metrics, test_metrics, momentum=beta)
        )
    results["tables"]["momentum"] = _write_table(momentum_rows, tables_dir / "momentum.csv")
    results["figures"]["momentum_loss"] = save_loss_plot(
        momentum_histories,
        figures_dir / "momentum_loss.png",
        "Сравнение SGD и momentum",
        val_only=True,
    )
    results["sections"]["momentum_best"] = min(momentum_rows, key=lambda row: row["final_val_loss"])

    cv_rows: list[dict[str, Any]] = []
    folds = stratified_kfold_indices(data.y_train, n_splits=5, random_state=RANDOM_STATE)
    for lr, batch_size in product([0.01, 0.1, 0.5], [16, 32, 64]):
        fold_scores = []
        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
            X_train_fold, X_val_fold, _, _ = standardize_train_test(
                data.X_train_raw[train_idx],
                data.X_train_raw[val_idx],
            )
            y_train_fold = data.y_train[train_idx]
            y_val_fold = data.y_train[val_idx]
            model, _, _, val_metrics = _train(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                lr=lr,
                batch_size=batch_size,
                random_state=RANDOM_STATE + fold_id,
            )
            fold_scores.append(val_metrics["accuracy"])
        cv_rows.append(
            {
                "lr": lr,
                "batch_size": batch_size,
                "mean_accuracy": float(np.mean(fold_scores)),
                "std_accuracy": float(np.std(fold_scores)),
            }
        )
    best_cv = max(cv_rows, key=lambda row: (row["mean_accuracy"], -row["std_accuracy"]))
    results["tables"]["cross_validation"] = _write_table(
        cv_rows, tables_dir / "cross_validation.csv"
    )
    results["figures"]["cross_validation"] = save_bar_plot(
        [f"{row['lr']}/{row['batch_size']}" for row in cv_rows],
        [row["mean_accuracy"] for row in cv_rows],
        figures_dir / "cross_validation.png",
        "5-fold CV: средняя accuracy",
        "Mean accuracy",
        ylim=(0.0, 1.05),
    )
    final_model, final_history, final_train_metrics, final_test_metrics = _train(
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        lr=best_cv["lr"],
        batch_size=best_cv["batch_size"],
        random_state=RANDOM_STATE,
    )
    results["sections"]["cross_validation_best"] = best_cv
    results["sections"]["final_model"] = _summary_row(
        "final_cv_model",
        final_model,
        final_history,
        final_train_metrics,
        final_test_metrics,
        lr=best_cv["lr"],
        batch_size=best_cv["batch_size"],
    )
    results["figures"]["final_boundary"] = save_decision_boundary(
        final_model,
        data.X_test,
        data.y_test,
        figures_dir / "final_decision_boundary.png",
        "Финальная модель после подбора гиперпараметров",
    )

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(_json_ready(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results

