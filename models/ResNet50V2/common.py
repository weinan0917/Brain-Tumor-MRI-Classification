import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import (
    CLASSES,
    NUM_CLASSES,
    ORI_DATASET_PATH,
    PROCESSED_SUBDIRS,
    SEED,
    TARGET_SIZE,
    get_label_mappings,
    resolve_project_path,
    setup_environment,
    setup_plot_style,
)
from Dataset_Load_Initial import (
    load_and_preprocess_partition,
    load_dataset,
    stratified_split,
)

setup_environment()
import keras
import tensorflow as tf
from keras import layers, models


# ── 数据加载 ─────────────────────────────────────────────────

def load_processed_data():
    arrays_dir = resolve_project_path(PROCESSED_SUBDIRS["arrays"])
    required = [
        "X_train.npy", "y_train.npy", "y_train_oh.npy",
        "X_val.npy", "y_val.npy", "y_val_oh.npy",
        "X_test.npy", "y_test.npy", "y_test_oh.npy",
    ]
    for fname in required:
        if not (arrays_dir / fname).exists():
            raise FileNotFoundError(
                f"缺少预处理数组 {fname}，请先运行 Dataset_Load_Initial.py"
            )

    X_train = np.load(arrays_dir / "X_train.npy")
    y_train = np.load(arrays_dir / "y_train.npy")
    y_train_oh = np.load(arrays_dir / "y_train_oh.npy")
    X_val = np.load(arrays_dir / "X_val.npy")
    y_val = np.load(arrays_dir / "y_val.npy")
    y_val_oh = np.load(arrays_dir / "y_val_oh.npy")
    X_test = np.load(arrays_dir / "X_test.npy")
    y_test = np.load(arrays_dir / "y_test.npy")
    y_test_oh = np.load(arrays_dir / "y_test_oh.npy")

    print(f"[processed] X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh


def load_ori_data():
    label_to_index, _ = get_label_mappings()
    df = load_dataset(ORI_DATASET_PATH)
    train_df, val_df, test_df = stratified_split(df)

    print("Loading and preprocessing ori_dataset partitions (this may take a while)...")
    X_train, y_train, y_train_oh = load_and_preprocess_partition(train_df, label_to_index)
    X_val, y_val, y_val_oh = load_and_preprocess_partition(val_df, label_to_index)
    X_test, y_test, y_test_oh = load_and_preprocess_partition(test_df, label_to_index)

    print(f"[ori] X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh


def load_data(dataset: str):
    if dataset == "processed":
        return load_processed_data()
    if dataset == "ori":
        return load_ori_data()
    raise ValueError(f"未知数据集类型: {dataset}，请使用 processed 或 ori")


def get_output_dirs(model_name: str, dataset: str):
    base = PROJECT_ROOT / "models" / model_name / "outputs" / dataset
    dirs = {
        "checkpoints": base / "checkpoints",
        "plots": base / "plots",
        "reports": base / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"[output] reports -> {dirs['reports']}")
    print(f"[output] checkpoints -> {dirs['checkpoints']}")
    return dirs

def run_hyperparameter_tuning(
    build_fn,
    X_train, y_train_oh, X_val, y_val_oh,
    quick_run: bool = False,
):
    lrs = [1e-3, 5e-4]
    dropouts = [0.3, 0.5]
    tuning_results = []

    print("Starting Hyperparameter Tuning...")
    for lr in lrs:
        for do in dropouts:
            print(f"Testing Config - LR: {lr}, Dropout: {do}")
            temp_model = build_fn()
            temp_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            temp_model.fit(
                X_train, y_train_oh,
                validation_data=(X_val, y_val_oh),
                epochs=1, batch_size=32, verbose=0,
            )
            val_acc = temp_model.evaluate(X_val, y_val_oh, verbose=0)[1]
            tuning_results.append({"lr": lr, "dropout": do, "val_accuracy": val_acc})
            if quick_run:
                break
        if quick_run:
            break

    tuning_df = pd.DataFrame(tuning_results)
    print("Hyperparameter tuning search results:")
    print(tuning_df)
    return tuning_df


def train_model(
    model,
    model_display_name: str,
    X_train, y_train_oh, X_val, y_val_oh,
    output_dirs: dict,
    epochs: int = 15,
    batch_size: int = 16,
    patience: int = 5,
    initial_lr: float = 1e-3,
    label_smoothing: float = 0.05,
    use_reduce_lr: bool = True,
    monitor: str = "val_accuracy",
):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )

    checkpoint_path = output_dirs["checkpoints"] / f"best_{model_display_name}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor, save_best_only=True, verbose=1,
        ),
    ]
    if use_reduce_lr:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=0.5, patience=max(2, patience // 2),
                min_lr=1e-7, verbose=1,
            )
        )

    print(f"\n{'=' * 20} Training {model_display_name} {'=' * 20}")
    start_time = time.time()
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1,
    )
    train_time = time.time() - start_time
    print(f"{model_display_name} trained successfully in {train_time:.2f} seconds.")
    return model, history.history, train_time



def plot_training_history(history: dict, model_name: str, plots_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["accuracy"], label="Train Acc", color="navy", linestyle="-")
    ax.plot(history["val_accuracy"], label="Val Acc", color="darkorange", linestyle="--")
    ax.set_title(f"{model_name} Learning Curves", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    out_path = plots_dir / "training_history.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def evaluate_model(
    model,
    model_name: str,
    X_test, y_test, y_test_oh,
    train_time: float,
    output_dirs: dict,
    use_tta: bool = True,
    n_tta: int = 5,
):
    plots_dir = output_dirs["plots"]
    reports_dir = output_dirs["reports"]

    start_inf = time.time()
    preds = model.predict(X_test, verbose=0)
    if use_tta:
        for _ in range(n_tta - 1):
            noise = X_test + np.random.normal(0, 0.01, X_test.shape).astype(np.float32)
            noise = np.clip(noise, 0.0, 1.0)
            preds += model.predict(noise, verbose=0)
        preds /= float(n_tta)
    inf_latency = (time.time() - start_inf) / len(X_test) * 1000

    y_pred_classes = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, y_pred_classes)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_classes, average="weighted"
    )
    roc_auc = roc_auc_score(y_test_oh, preds, average="weighted", multi_class="ovr")
    kappa = cohen_kappa_score(y_test, y_pred_classes)
    mcc = matthews_corrcoef(y_test, y_pred_classes)
    params = model.count_params()

    metrics = {
        "Model": model_name,
        "Dataset": output_dirs.get("dataset", ""),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Cohen's Kappa": kappa,
        "MCC": mcc,
        "Params": params,
        "Inference Latency (ms)": inf_latency,
        "Training Time (s)": train_time,
    }

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False,
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = plots_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cm_path}")

    # ROC & PR 曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fpr, tpr, _ = roc_curve(y_test_oh.ravel(), preds.ravel())
    axes[0].plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    axes[0].set_title("ROC Curve (One-vs-Rest)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test_oh.ravel(), preds.ravel()
    )
    axes[1].plot(recall_curve, precision_curve, label=model_name)
    axes[1].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.tight_layout()
    roc_path = plots_dir / "roc_pr_curves.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {roc_path}")

    metrics_df = pd.DataFrame([metrics])
    metrics_path = reports_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")
    print(f"Evaluation Metrics{' (TTA)' if use_tta else ''}:")
    print(metrics_df.to_string(index=False))

    return metrics


def run_two_phase_transfer(
    build_fn,
    model_name: str,
    model_display_name: str,
    dataset: str,
    phase1_epochs: int = 12,
    phase2_epochs: int = 20,
    batch_size: int = 16,
    unfreeze_last_n: int = 40,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-4,
    use_tta: bool = True,
):
    config.create_output_dirs()
    setup_plot_style()
    config.print_environment_info()

    print(f"\n>>> Running {model_display_name} [models] two-phase on: {dataset}")
    X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh = load_data(dataset)
    output_dirs = get_output_dirs(model_name, dataset)
    output_dirs["dataset"] = dataset
    ckpt = output_dirs["checkpoints"] / f"best_{model_display_name}.keras"

    total_start = time.time()

    print(f"\n{'=' * 20} Phase 1: Frozen backbone ({phase1_epochs} epochs) {'=' * 20}")
    model = build_fn(trainable_backbone=False)
    model, history1, t1 = train_model(
        model, model_display_name,
        X_train, y_train_oh, X_val, y_val_oh,
        output_dirs,
        epochs=phase1_epochs, batch_size=batch_size,
        patience=6, initial_lr=phase1_lr,
        label_smoothing=0.05, use_reduce_lr=True,
    )

    print(f"\n{'=' * 20} Phase 2: Fine-tune top {unfreeze_last_n} layers ({phase2_epochs} epochs) {'=' * 20}")
    model = build_fn(trainable_backbone=True, unfreeze_last_n=unfreeze_last_n)
    if ckpt.exists():
        model.load_weights(str(ckpt))
        print(f"Loaded Phase 1 weights from {ckpt}")

    model, history2, t2 = train_model(
        model, model_display_name,
        X_train, y_train_oh, X_val, y_val_oh,
        output_dirs,
        epochs=phase2_epochs, batch_size=batch_size,
        patience=10, initial_lr=phase2_lr,
        label_smoothing=0.05, use_reduce_lr=True,
    )

    total_time = time.time() - total_start
    merged = {key: history1[key] + history2.get(key, []) for key in history1}

    with open(output_dirs["reports"] / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    with open(output_dirs["reports"] / "training_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "phase1_epochs": phase1_epochs,
            "phase2_epochs": phase2_epochs,
            "phase1_lr": phase1_lr,
            "phase2_lr": phase2_lr,
            "unfreeze_last_n": unfreeze_last_n,
            "phase1_time_s": t1,
            "phase2_time_s": t2,
        }, f, indent=2)

    plot_training_history(merged, model_display_name, output_dirs["plots"])
    metrics = evaluate_model(
        model, model_display_name,
        X_test, y_test, y_test_oh, total_time, output_dirs,
        use_tta=use_tta,
    )
    print(f"\n{model_display_name} on [{dataset}] completed. Test Accuracy = {metrics['Accuracy']:.4f}")
    return model, merged, metrics


def run_full_pipeline(
    build_fn,
    model_name: str,
    model_display_name: str,
    dataset: str,
    epochs: int = 30,
    batch_size: int = 16,
    tune: bool = False,
    quick_run: bool = False,
    patience: int = 8,
    initial_lr: float = 1e-3,
    label_smoothing: float = 0.05,
    use_reduce_lr: bool = True,
    use_tta: bool = True,
):
    config.create_output_dirs()
    setup_plot_style()
    config.print_environment_info()

    print(f"\n>>> Running {model_display_name} on dataset: {dataset}")
    X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh = load_data(dataset)

    output_dirs = get_output_dirs(model_name, dataset)
    output_dirs["dataset"] = dataset

    if tune:
        tuning_df = run_hyperparameter_tuning(
            build_fn, X_train, y_train_oh, X_val, y_val_oh, quick_run=quick_run,
        )
        tuning_df.to_csv(output_dirs["reports"] / "hyperparameter_tuning.csv", index=False)

    model = build_fn()
    model, history, train_time = train_model(
        model, model_display_name,
        X_train, y_train_oh, X_val, y_val_oh,
        output_dirs, epochs=epochs, batch_size=batch_size,
        patience=patience, initial_lr=initial_lr,
        label_smoothing=label_smoothing, use_reduce_lr=use_reduce_lr,
    )

    with open(output_dirs["reports"] / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history, model_display_name, output_dirs["plots"])
    metrics = evaluate_model(
        model, model_display_name,
        X_test, y_test, y_test_oh, train_time, output_dirs,
        use_tta=use_tta,
    )

    print(f"\n{model_display_name} on [{dataset}] completed. Test Accuracy = {metrics['Accuracy']:.4f}")
    return model, history, metrics
