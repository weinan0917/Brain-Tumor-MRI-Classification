import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES, NUM_CLASSES, SEED, create_output_dirs, print_environment_info, setup_environment, setup_plot_style
from feature_analysis import extract_handcrafted_features
from models.common import get_output_dirs, load_data, plot_training_history

setup_environment()
import keras
import tensorflow as tf
from keras import layers, models, regularizers

MODEL_NAME = "GKLNN"
MODEL_DISPLAY_NAME = "GKLNN"

# ── 超参数 ───────────────────────────────────────────────
PCA_DIM = 160
RAW_DIM = 406
LIQUID_NEURONS = 640
N_PROTOTYPES = 2400
LIQUID_ITERATIONS = 35
RESERVOIR_SEEDS = (42, 137, 271)
KERNEL_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_EPOCHS = 180
DEFAULT_BATCH_SIZE = 48
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 0.15
L2_REG = 1e-4
INITIAL_LR = 2e-4
MIN_LR = 2e-6
EARLY_STOP_PATIENCE = 28
TTA_ROUNDS = 10


# ── 特征工程 ─────────────────────────────────────────────────

class DualFeaturePreprocessor:

    def __init__(self, pca_dim=PCA_DIM):
        self.raw_scaler = StandardScaler()
        self.pca = PCA(n_components=pca_dim, whiten=True, random_state=SEED)
        self.pca_scaler = StandardScaler()
        self.pca_dim = pca_dim

    def _extract_raw(self, images):
        return extract_handcrafted_features(images).astype(np.float32)

    def fit(self, images):
        raw = self._extract_raw(images)
        raw_scaled = self.raw_scaler.fit_transform(raw)
        pca_reduced = self.pca.fit_transform(raw_scaled)
        self.pca_scaler.fit(pca_reduced)
        var = float(np.sum(self.pca.explained_variance_ratio_))
        print(f"  PCA {self.pca_dim}D explained variance: {var:.2%}")
        return self

    def transform(self, images):
        raw = self._extract_raw(images)
        raw_scaled = self.raw_scaler.transform(raw).astype(np.float32)
        pca_feat = self.pca_scaler.transform(self.pca.transform(raw_scaled)).astype(np.float32)
        return pca_feat, raw_scaled

    def fit_transform(self, images):
        self.fit(images)
        return self.transform(images)

    def save(self, path: Path):
        np.savez(
            path,
            raw_mean=self.raw_scaler.mean_, raw_scale=self.raw_scaler.scale_,
            pca_components=self.pca.components_, pca_mean=self.pca.mean_,
            pca_explained_variance_=self.pca.explained_variance_,
            pca_explained_variance_ratio_=self.pca.explained_variance_ratio_,
            pca_post_mean=self.pca_scaler.mean_, pca_post_scale=self.pca_scaler.scale_,
            pca_dim=self.pca_dim,
        )

    @classmethod
    def load(cls, path: Path) -> "DualFeaturePreprocessor":
        data = np.load(path)
        pca_dim = int(data["pca_dim"])
        pre = cls(pca_dim=pca_dim)
        pre.raw_scaler = StandardScaler()
        pre.raw_scaler.mean_ = data["raw_mean"]
        pre.raw_scaler.scale_ = data["raw_scale"]
        pre.raw_scaler.n_features_in_ = len(data["raw_mean"])

        pre.pca = PCA(n_components=pca_dim, whiten=True, random_state=SEED)
        pre.pca.components_ = data["pca_components"]
        pre.pca.mean_ = data["pca_mean"]
        pre.pca.n_components_ = pca_dim
        pre.pca.n_features_in_ = data["pca_components"].shape[1]
        if "pca_explained_variance_" in data:
            pre.pca.explained_variance_ = data["pca_explained_variance_"]
            pre.pca.explained_variance_ratio_ = data["pca_explained_variance_ratio_"]
        else:
            raise ValueError(
                f"Preprocessor {path} missing pca_explained_variance_; "
                "re-run training or refit with DualFeaturePreprocessor.fit()."
            )

        pre.pca_scaler = StandardScaler()
        pre.pca_scaler.mean_ = data["pca_post_mean"]
        pre.pca_scaler.scale_ = data["pca_post_scale"]
        pre.pca_scaler.n_features_in_ = len(data["pca_post_mean"])
        return pre


def _load_dual_features(images, cache_path, preprocessor, fit=False, refresh_cache=False):
    n = len(images)
    if cache_path.exists() and not fit and not refresh_cache:
        data = np.load(cache_path)
        pca, raw = data["pca"], data["raw"]
        if len(pca) == n and len(raw) == n and pca.shape[1] == PCA_DIM and raw.shape[1] == RAW_DIM:
            print(f"  Loading cache: {cache_path}")
            return pca, raw
        print(
            f"  Cache stale (samples={len(pca)}, pca_dim={pca.shape[1] if pca.ndim > 1 else '?'}, "
            f"expected n={n}, pca_dim={PCA_DIM}), regenerating: {cache_path}"
        )
    pca, raw = preprocessor.fit_transform(images) if fit else preprocessor.transform(images)
    np.savez(cache_path, pca=pca, raw=raw)
    print(f"  Saved cache: {cache_path}")
    return pca, raw


def compute_sigma(features, max_samples=1000):
    n = len(features)
    idx = np.random.default_rng(SEED).choice(n, min(n, max_samples), replace=False)
    dists = pairwise_distances(features[idx], metric="euclidean")
    upper = dists[np.triu_indices_from(dists, k=1)]
    sigma = float(np.median(upper)) / np.sqrt(2.0 * features.shape[1])
    return float(np.clip(sigma, 0.3, 6.0))


def select_prototypes_kmeans(pca_feat, labels, n_prototypes):
    per_class = max(n_prototypes // NUM_CLASSES, 1)
    centers = []
    for cls in range(NUM_CLASSES):
        km = KMeans(n_clusters=per_class, random_state=SEED, n_init=10)
        km.fit(pca_feat[labels == cls])
        centers.append(km.cluster_centers_)
    return np.vstack(centers).astype(np.float32)


def _scale_spectral_radius(matrix, target=0.85):
    rho = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    return matrix if rho < 1e-8 else matrix * (target / rho)


# ── 液态层 ───────────────────────────────────────────────────

@keras.saving.register_keras_serializable(package="LRWGKLNN")
class GaussianKernelLiquidLayer(layers.Layer):
    def __init__(
        self, prototypes, liquid_neurons, sigma, feature_dim=PCA_DIM,
        kernel_scales=None, reservoir_seed=SEED, spectral_radius=0.85,
        n_iterations=LIQUID_ITERATIONS, **kwargs,
    ):
        kwargs.setdefault("trainable", True)
        super().__init__(**kwargs)
        prototypes = np.asarray(prototypes, dtype=np.float32)
        self.liquid_neurons = int(liquid_neurons)
        self.sigma = float(sigma)
        self.feature_dim = int(feature_dim)
        self.kernel_scales = list(kernel_scales or KERNEL_SCALES)
        self.n_prototypes = int(prototypes.shape[0])
        self.reservoir_seed = int(reservoir_seed)
        self.spectral_radius = float(spectral_radius)
        self.n_iterations = int(n_iterations)
        self.n_kernel_dims = self.n_prototypes * len(self.kernel_scales)

        rng = np.random.default_rng(self.reservoir_seed)

        self.prototypes_w = self.add_weight(
            shape=(self.n_prototypes, self.feature_dim),
            name="prototypes",
            initializer=keras.initializers.Constant(prototypes), trainable=False,
        )
        self.W_ir = self.add_weight(
            shape=(self.feature_dim, self.liquid_neurons),
            name="W_ir",
            initializer=keras.initializers.GlorotUniform(seed=self.reservoir_seed), trainable=False,
        )
        self.b_ir = self.add_weight(
            shape=(self.liquid_neurons,), name="b_ir", initializer="zeros", trainable=False,
        )
        self.A = self.add_weight(
            shape=(self.liquid_neurons, self.n_kernel_dims),
            name="A",
            initializer=keras.initializers.GlorotUniform(seed=self.reservoir_seed + 1), trainable=True,
        )
        W_r = _scale_spectral_radius(
            rng.uniform(-0.5, 0.5, (self.liquid_neurons, self.liquid_neurons)).astype(np.float32),
            self.spectral_radius,
        )
        self.W_r = self.add_weight(
            shape=(self.liquid_neurons, self.liquid_neurons),
            name="W_r",
            initializer=keras.initializers.Constant(W_r), trainable=False,
        )
        self.b_r = self.add_weight(
            shape=(self.liquid_neurons,), name="b_r", initializer="zeros", trainable=False,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "liquid_neurons": self.liquid_neurons, "sigma": self.sigma,
            "feature_dim": self.feature_dim, "n_prototypes": self.n_prototypes,
            "kernel_scales": self.kernel_scales, "reservoir_seed": self.reservoir_seed,
            "spectral_radius": self.spectral_radius, "n_iterations": self.n_iterations,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        n_proto = config.pop("n_prototypes")
        feat_dim = config.pop("feature_dim", PCA_DIM)
        return cls(prototypes=np.zeros((n_proto, feat_dim), np.float32), **config)

    def call(self, x):
        diff = tf.expand_dims(x, 1) - tf.expand_dims(self.prototypes_w, 0)
        sq = tf.reduce_sum(tf.square(diff), axis=-1)
        kappa = tf.concat(
            [tf.exp(-sq / (2.0 * (self.sigma * s) ** 2)) for s in self.kernel_scales], axis=-1,
        )
        drive = tf.matmul(kappa, self.A, transpose_b=True) + tf.matmul(x, self.W_ir) + self.b_ir
        h = tf.zeros((tf.shape(x)[0], self.liquid_neurons), tf.float32)
        for _ in range(self.n_iterations):
            h = tf.nn.tanh(drive + tf.matmul(h, self.W_r, transpose_b=True) + self.b_r)
        return h


def build_lrw_gklnn(prototypes, sigma, liquid_neurons=LIQUID_NEURONS):
    pca_in = layers.Input(shape=(PCA_DIM,), name="pca_input")
    raw_in = layers.Input(shape=(RAW_DIM,), name="raw_input")

    liquid_states = []
    for i, rs in enumerate(RESERVOIR_SEEDS):
        liq = GaussianKernelLiquidLayer(
            prototypes=prototypes, liquid_neurons=liquid_neurons, sigma=sigma,
            feature_dim=PCA_DIM, reservoir_seed=rs, spectral_radius=0.85 - i * 0.04,
            name=f"liquid_reservoir_{i}",
        )(pca_in)
        liquid_states.append(liq)

    raw_branch = layers.Dense(320, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(raw_in)
    raw_branch = layers.BatchNormalization()(raw_branch)
    raw_branch = layers.Dropout(0.2)(raw_branch)
    raw_branch = layers.Dense(160, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(raw_branch)
    raw_branch = layers.BatchNormalization()(raw_branch)

    pca_branch = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(pca_in)
    pca_branch = layers.BatchNormalization()(pca_branch)

    merged = layers.Concatenate(name="fusion")([pca_branch, raw_branch] + liquid_states)
    gate = layers.Dense(merged.shape[-1], activation="sigmoid", name="fusion_gate")(merged)
    merged = layers.Multiply(name="gated_fusion")([merged, gate])

    x = layers.Dense(640, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(320, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = models.Model(inputs=[pca_in, raw_in], outputs=outputs, name=MODEL_DISPLAY_NAME)
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(
        f"LRWGKLNN v4: trainable={trainable:,}, prototypes={len(prototypes)}, "
        f"σ={sigma:.4f}, reservoirs={len(RESERVOIR_SEEDS)}"
    )
    return model


# ── Mixup 训练 ───────────────────────────────────────────────

def _mixup_batch(pca, raw, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha, size=len(pca)).astype(np.float32)
    lam = np.maximum(lam, 1.0 - lam)
    idx = np.random.permutation(len(pca))
    lam2 = lam.reshape(-1, 1)
    pca_m = lam2 * pca + (1.0 - lam2) * pca[idx]
    raw_m = lam2 * raw + (1.0 - lam2) * raw[idx]
    y_m = lam.reshape(-1, 1) * y + (1.0 - lam.reshape(-1, 1)) * y[idx]
    return pca_m, raw_m, y_m


def _cutmix_batch(pca, raw, y, alpha=CUTMIX_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    lam = float(np.clip(lam, 0.35, 0.65))
    idx = np.random.permutation(len(pca))
    pca_m, raw_m = pca.copy(), raw.copy()
    pca_m[:] = lam * pca + (1.0 - lam) * pca[idx]
    raw_m[:] = lam * raw + (1.0 - lam) * raw[idx]
    y_m = lam * y + (1.0 - lam) * y[idx]
    return pca_m, raw_m, y_m


def _augment_batch(pca, raw, y):
    if np.random.rand() < 0.5:
        return _mixup_batch(pca, raw, y)
    return _cutmix_batch(pca, raw, y)


def _cosine_lr(epoch, total_epochs, base_lr=INITIAL_LR, min_lr=MIN_LR):
    if total_epochs <= 1:
        return base_lr
    progress = epoch / max(total_epochs - 1, 1)
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress)))


def train_with_mixup(model, pca_tr, raw_tr, y_tr_oh, pca_val, raw_val, y_val_oh,
                     output_dirs, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
    cw = {i: float(w) for i, w in enumerate(
        compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=np.argmax(y_tr_oh, axis=1))
    )}
    print(f"Class weights: {cw}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
        metrics=["accuracy"],
    )

    ckpt = output_dirs["checkpoints"] / f"best_{MODEL_DISPLAY_NAME}.keras"
    best_val, best_weights = -1.0, None
    patience, wait = EARLY_STOP_PATIENCE, 0
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    n = len(pca_tr)
    steps = max(1, n // batch_size)

    print(f"\n{'=' * 20} Training {MODEL_DISPLAY_NAME} (Mixup/CutMix) {'=' * 20}")
    t0 = time.time()

    for epoch in range(epochs):
        lr = _cosine_lr(epoch, epochs)
        model.optimizer.learning_rate.assign(lr)

        perm = np.random.permutation(n)
        pca_s, raw_s, y_s = pca_tr[perm], raw_tr[perm], y_tr_oh[perm]
        ep_loss, ep_acc, count = 0.0, 0.0, 0

        for step in range(steps):
            start = step * batch_size
            end = min(start + batch_size, n)
            bp, br, by = pca_s[start:end], raw_s[start:end], y_s[start:end]
            bp, br, by = _augment_batch(bp, br, by)

            sw = np.array([cw[int(np.argmax(row))] for row in by])
            with tf.GradientTape() as tape:
                pred = model([bp, br], training=True)
                loss = keras.losses.categorical_crossentropy(by, pred, label_smoothing=0.08)
                loss = tf.reduce_mean(loss * sw)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            ep_loss += float(loss)
            ep_acc += float(np.mean(np.argmax(by, -1) == np.argmax(pred.numpy(), -1)))
            count += 1

        val_pred = model.predict([pca_val, raw_val], verbose=0)
        val_loss = float(keras.losses.categorical_crossentropy(y_val_oh, val_pred).numpy().mean())
        val_acc = float(np.mean(np.argmax(y_val_oh, -1) == np.argmax(val_pred, -1)))
        tr_loss, tr_acc = ep_loss / count, ep_acc / count

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - lr: {lr:.2e} - loss: {tr_loss:.4f} - acc: {tr_acc:.4f} "
            f"- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val, wait = val_acc, 0
            best_weights = model.get_weights()
            model.save(str(ckpt))
            print(f"Epoch {epoch + 1}: val_acc improved to {val_acc:.4f}, saving model")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return model, history, time.time() - t0


def evaluate_dual(model, pca_test, raw_test, y_test, y_test_oh, train_time, output_dirs):
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, confusion_matrix, matthews_corrcoef,
        precision_recall_fscore_support, roc_auc_score, roc_curve, precision_recall_curve,
    )
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    t0 = time.time()
    preds = model.predict([pca_test, raw_test], verbose=0)
    for scale in (0.01, 0.015, 0.02, 0.025, 0.03):
        for _ in range(max(1, TTA_ROUNDS // 5)):
            noise_p = pca_test + np.random.normal(0, scale, pca_test.shape).astype(np.float32)
            noise_r = raw_test + np.random.normal(0, scale, raw_test.shape).astype(np.float32)
            preds += model.predict([noise_p, noise_r], verbose=0)
    preds /= float(1 + TTA_ROUNDS)
    latency = (time.time() - t0) / len(pca_test) * 1000

    y_pred = np.argmax(preds, 1)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test_oh, preds, average="weighted", multi_class="ovr")
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    plots_dir, reports_dir = output_dirs["plots"], output_dirs["reports"]
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"Confusion Matrix: {MODEL_DISPLAY_NAME}")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fpr, tpr, _ = roc_curve(y_test_oh.ravel(), preds.ravel())
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    axes[0].set_title("ROC Curve")
    prec_c, rec_c, _ = precision_recall_curve(y_test_oh.ravel(), preds.ravel())
    axes[1].plot(rec_c, prec_c)
    axes[1].set_title("PR Curve")
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_pr_curves.png", bbox_inches="tight")
    plt.close()

    metrics = {
        "Model": MODEL_DISPLAY_NAME, "Dataset": output_dirs.get("dataset", ""),
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1,
        "ROC-AUC": roc_auc, "Cohen's Kappa": kappa, "MCC": mcc,
        "Params": model.count_params(), "Inference Latency (ms)": latency,
        "Training Time (s)": train_time,
    }
    pd.DataFrame([metrics]).to_csv(reports_dir / "metrics.csv", index=False)
    print("Evaluation Metrics:")
    print(pd.DataFrame([metrics]).to_string(index=False))
    return metrics


def run_full_pipeline(
    dataset,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    n_prototypes=N_PROTOTYPES,
    refresh_cache=False,
):
    create_output_dirs()
    setup_plot_style()
    print_environment_info()
    print(f"\n>>> Running {MODEL_DISPLAY_NAME} on: {dataset}")

    X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh = load_data(dataset)
    output_dirs = get_output_dirs(MODEL_NAME, dataset)
    output_dirs["dataset"] = dataset
    cache_dir = output_dirs["checkpoints"] / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pre = DualFeaturePreprocessor(pca_dim=PCA_DIM)
    pca_tr, raw_tr = _load_dual_features(
        X_train, cache_dir / "train.npz", pre, fit=True, refresh_cache=refresh_cache,
    )
    pca_val, raw_val = _load_dual_features(
        X_val, cache_dir / "val.npz", pre, refresh_cache=refresh_cache,
    )
    pca_te, raw_te = _load_dual_features(
        X_test, cache_dir / "test.npz", pre, refresh_cache=refresh_cache,
    )

    prototypes = select_prototypes_kmeans(pca_tr, y_train, n_prototypes)
    sigma = compute_sigma(pca_tr)
    print(f"prototypes={len(prototypes)}, sigma={sigma:.4f}")

    pre.save(output_dirs["checkpoints"] / "feature_preprocessor.npz")
    np.save(output_dirs["checkpoints"] / "prototypes.npy", prototypes)
    with open(output_dirs["reports"] / "gklnn_config.json", "w") as f:
        json.dump({
            "pca_dim": PCA_DIM, "raw_dim": RAW_DIM,
            "dual_reservoir": list(RESERVOIR_SEEDS),
            "mixup_alpha": MIXUP_ALPHA, "cutmix_alpha": CUTMIX_ALPHA,
            "sigma": sigma, "n_prototypes": len(prototypes),
        }, f, indent=2)

    model = build_lrw_gklnn(prototypes, sigma)
    model, history, train_time = train_with_mixup(
        model, pca_tr, raw_tr, y_train_oh, pca_val, raw_val, y_val_oh,
        output_dirs, epochs=epochs, batch_size=batch_size,
    )

    with open(output_dirs["reports"] / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, MODEL_DISPLAY_NAME, output_dirs["plots"])
    metrics = evaluate_dual(model, pca_te, raw_te, y_test, y_test_oh, train_time, output_dirs)
    print(f"\n{MODEL_DISPLAY_NAME} done. Test Accuracy = {metrics['Accuracy']:.4f}")
    return model, history, metrics


def main():
    parser = argparse.ArgumentParser(description="LRWGKLNN")
    parser.add_argument("--dataset", choices=["processed", "ori"], required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-prototypes", type=int, default=N_PROTOTYPES)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()
    run_full_pipeline(
        args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_prototypes=args.n_prototypes,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
