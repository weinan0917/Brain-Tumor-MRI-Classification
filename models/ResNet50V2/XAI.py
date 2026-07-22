import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.stats import chi2, friedmanchisquare, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config import (
    CLASSES,
    NUM_CLASSES,
    OUTPUT_DIRS,
    PROCESSED_SUBDIRS,
    QUICK_RUN,
    SEED,
    create_output_dirs,
    get_label_mappings,
    setup_environment,
    setup_plot_style,
)
from feature_analysis import extract_handcrafted_features
from models.common import load_data

setup_environment()
import keras
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_REGISTRY = {
    "ResNet50V2": ("ResNet50V2", "ResNet50V2", "best_ResNet50V2.keras")
}

CLASS_LABELS_PLOT = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


# ── 数据与模型加载 ───────────────────────────────────────────

def _resolve_checkpoint(model_key: str, dataset: str) -> Path | None:
    subdir, display_name, legacy_name = MODEL_REGISTRY[model_key]
    candidates = [
        PROJECT_ROOT / "models" / legacy_name,
        PROJECT_ROOT / "models" / subdir / "outputs" / dataset / "checkpoints" / f"best_{display_name}.keras",
        PROJECT_ROOT / "models" / subdir / "outputs" / dataset / "checkpoints" / f"best_{model_key}.keras",
    ]
    ckpt_dir = PROJECT_ROOT / "models" / subdir / "outputs" / dataset / "checkpoints"
    if ckpt_dir.exists():
        candidates.extend(sorted(ckpt_dir.glob("best_*.keras")))

    for path in candidates:
        if path.exists():
            return path
    return None

def load_trained_models(model_keys: list[str], dataset: str) -> dict[str, keras.Model]:
    loaded = {}
    for key in model_keys:
        ckpt = _resolve_checkpoint(key, dataset)
        if ckpt is None:
            warnings.warn(f"未找到 {key} checkpoint，跳过")
            continue
        print(f"Loading {key}: {ckpt}")
        loaded[key] = keras.models.load_model(
            str(ckpt),
            custom_objects=_custom_objects_for(key),
            compile=False,
        )
    if not loaded:
        raise FileNotFoundError(
            "未找到任何可用 checkpoint。请先训练模型，或将权重放到 models/best_*.keras"
        )
    return loaded


def load_handcrafted_features(X_train, y_train, X_test, y_test):
    feat_dir = PROCESSED_SUBDIRS["features"]
    train_path = feat_dir / "feat_train.npy"
    test_path = feat_dir / "feat_test.npy"
    if train_path.exists() and test_path.exists():
        print(f"Loading handcrafted features from {feat_dir}")
        return np.load(train_path), np.load(test_path)

    print("Extracting handcrafted features (HOG/LBP/GLCM)...")
    return extract_handcrafted_features(X_train), extract_handcrafted_features(X_test)


def find_last_conv_layer(model: keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError(f"{model.name} 无 Conv2D 层，无法使用 Grad-CAM")


def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + keras.backend.epsilon()
    return heatmap.numpy()


def get_gradcam_plusplus_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape(persistent=True) as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    grads2 = grads ** 2
    grads3 = grads ** 3
    sum_activations = tf.reduce_sum(conv_outputs, axis=(0, 1))
    alpha = grads2 / (2.0 * grads2 + sum_activations * grads3 + 1e-8)
    weights = tf.reduce_sum(alpha * tf.maximum(grads, 0.0), axis=(0, 1))
    heatmap = tf.reduce_sum(tf.nn.relu(conv_outputs * weights), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + keras.backend.epsilon()
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.array(Image.fromarray(heatmap).resize((img.shape[1], img.shape[0])))
    heatmap = plt.cm.jet(heatmap)[:, :, :3]
    img_norm = img if img.max() <= 1.0 else img / 255.0
    return np.clip(alpha * heatmap + (1 - alpha) * img_norm, 0, 1)


def run_gradcam_section(model, model_name, X_test, y_test, sample_idx, plots_dir):
    _, index_to_label = get_label_mappings()
    sample_img = X_test[sample_idx : sample_idx + 1]
    last_conv = find_last_conv_layer(model)
    print(f"Last Conv Layer ({model_name}): {last_conv}")

    for method, fn, suffix in (
        ("Grad-CAM", get_gradcam_heatmap, "gradcam"),
        ("Grad-CAM++", get_gradcam_plusplus_heatmap, "gradcampp"),
    ):
        heatmap = fn(sample_img, model, last_conv)
        overlay = overlay_heatmap(sample_img[0], heatmap)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(sample_img[0])
        ax[0].set_title("Input MRI")
        ax[0].axis("off")
        ax[1].imshow(heatmap, cmap="jet")
        ax[1].set_title(method)
        ax[1].axis("off")
        ax[2].imshow(overlay)
        ax[2].set_title("Overlay")
        ax[2].axis("off")
        plt.suptitle(
            f"{method} — {model_name} (True: {index_to_label[y_test[sample_idx]]})",
            fontweight="bold",
        )
        plt.tight_layout()
        out = plots_dir / f"xai_{suffix}_{model_name.lower()}_glioma.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def run_lime_section(model, X_test, sample_idx, plots_dir, num_samples=200):
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        warnings.warn("未安装 lime，跳过 LIME（pip install lime）")
        return

    explainer = lime_image.LimeImageExplainer(random_state=SEED)
    explanation = explainer.explain_instance(
        X_test[sample_idx].astype("double"),
        model.predict,
        top_labels=NUM_CLASSES,
        hide_color=0,
        num_samples=num_samples,
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False,
    )
    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp, mask))
    plt.title("LIME Image Segmentation Explanation", fontsize=12, fontweight="bold")
    plt.axis("off")
    out = plots_dir / "xai_lime_explanation.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def run_shap_section(feat_train, feat_test, y_train, plots_dir, n_samples=20):
    try:
        import shap
    except ImportError:
        warnings.warn("未安装 shap，跳过 SHAP（pip install shap）")
        return

    rf = RandomForestClassifier(n_estimators=50, random_state=SEED)
    rf.fit(feat_train, y_train)
    explainer = shap.TreeExplainer(rf)
    sample_n = min(n_samples, len(feat_test))
    shap_values = explainer.shap_values(feat_test[:sample_n])

    plt.figure(figsize=(10, 6))
    values = shap_values[0] if isinstance(shap_values, list) else shap_values[:, :, 0]
    shap.summary_plot(values, feat_test[:sample_n], show=False, max_display=20)
    plt.title("SHAP Feature Importance (Glioma Class)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / "xai_shap_summary.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 14. Uncertainty Quantification ───────────────────────────

def run_mc_dropout(model, X_test, y_test, plots_dir, n_passes=20):
    _, index_to_label = get_label_mappings()
    mc_predictions = []
    for _ in range(n_passes):
        mc_predictions.append(model(X_test, training=True).numpy())
    mc_predictions = np.array(mc_predictions)
    mc_mean = mc_predictions.mean(axis=0)
    mc_std = mc_predictions.std(axis=0)

    n_show = min(3, len(y_test))
    fig, axes = plt.subplots(1, n_show, figsize=(16, 5))
    if n_show == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, NUM_CLASSES))
    for i in range(n_show):
        axes[i].bar(CLASS_LABELS_PLOT, mc_mean[i], yerr=mc_std[i], color=colors, edgecolor="black", capsize=5)
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel("Prediction Probability")
        axes[i].set_title(f"Sample {i + 1}\nTrue: {index_to_label[y_test[i]]}", fontweight="bold")
        axes[i].tick_params(axis="x", rotation=20)

    plt.suptitle("Monte Carlo Dropout Uncertainty Estimation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / "uncertainty_mc_dropout.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    entropy = -np.sum(mc_mean * np.log(mc_mean + 1e-10), axis=1)
    print("\nPredictive Entropy (first 10 samples):")
    for i in range(min(10, len(entropy))):
        print(f"  Sample {i + 1}: {entropy[i]:.4f}")
    return {"mean_entropy": float(entropy.mean()), "per_sample_entropy": entropy.tolist()}


def run_deep_ensemble(trained_models: dict, X_test, y_test, plots_dir):
    if len(trained_models) < 2:
        print("Deep Ensemble: 可用模型不足 2 个，跳过")
        return None

    _, index_to_label = get_label_mappings()
    preds = []
    names = []
    for name, model in trained_models.items():
        preds.append(model.predict(X_test, verbose=0))
        names.append(name)
    ensemble_pred = np.mean(preds, axis=0)
    y_pred = np.argmax(ensemble_pred, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"Deep Ensemble ({', '.join(names)}) accuracy: {acc:.4f}")

    n_show = min(3, len(y_test))
    fig, axes = plt.subplots(1, n_show, figsize=(16, 5))
    if n_show == 1:
        axes = [axes]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, NUM_CLASSES))
    for i in range(n_show):
        axes[i].bar(CLASS_LABELS_PLOT, ensemble_pred[i], color=colors, edgecolor="black")
        axes[i].set_ylim(0, 1)
        axes[i].set_title(f"Sample {i + 1}\nTrue: {index_to_label[y_test[i]]}", fontweight="bold")
        axes[i].tick_params(axis="x", rotation=20)
    plt.suptitle("Deep Ensemble Prediction Probabilities", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / "uncertainty_deep_ensemble.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return {"models": names, "accuracy": float(acc)}


def run_bootstrap_ci(y_true, y_pred, model_name, n_bootstraps=200):
    rng = np.random.default_rng(SEED)
    accs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    accs = np.sort(accs)
    ci_lower = float(accs[int(0.025 * n_bootstraps)])
    ci_upper = float(accs[int(0.975 * n_bootstraps)])
    point = float(accuracy_score(y_true, y_pred))
    print(f"{model_name} 95% Confidence Interval for Accuracy: [{ci_lower:.4f}, {ci_upper:.4f}]")
    return {"accuracy": point, "ci_lower": ci_lower, "ci_upper": ci_upper}


# ── 15. Statistical Analysis ─────────────────────────────────

def mcnemar_test(y_true, y_pred1, y_pred2):
    b = c = 0
    for i in range(len(y_true)):
        c1 = y_pred1[i] == y_true[i]
        c2 = y_pred2[i] == y_true[i]
        if c1 and not c2:
            b += 1
        elif not c1 and c2:
            c += 1
    stat = (abs(b - c) - 1) ** 2 / (b + c + 1e-10)
    return float(stat), float(chi2.sf(stat, 1)), b, c


def run_statistical_analysis(trained_models, X_test, y_test, reports_dir, n_bootstrap_rounds=10):
    results = {"comparisons": []}
    preds = {name: np.argmax(model.predict(X_test, verbose=0), axis=1) for name, model in trained_models.items()}

    keys = list(preds.keys())
    if "Custom_CNN" in preds and "ResNet50V2" in preds:
        stat, p, b, c = mcnemar_test(y_test, preds["Custom_CNN"], preds["ResNet50V2"])
        msg = "significant (p < 0.05)" if p < 0.05 else "NOT significant (p >= 0.05)"
        print(f"McNemar Custom_CNN vs ResNet50V2: stat={stat:.4f}, p={p:.5f} — {msg}")
        results["comparisons"].append({
            "model_a": "Custom_CNN", "model_b": "ResNet50V2",
            "mcnemar_stat": stat, "p_value": p, "b": b, "c": c,
        })

    compare_keys = [k for k in ("Custom_CNN", "ResNet50V2", "ViT") if k in preds]
    if len(compare_keys) >= 2:
        rng = np.random.default_rng(SEED)
        boot_acc = {k: [] for k in compare_keys}
        for _ in range(n_bootstrap_rounds):
            idx = rng.choice(len(y_test), len(y_test), replace=True)
            for k in compare_keys:
                boot_acc[k].append(accuracy_score(y_test[idx], preds[k][idx]))

        if len(compare_keys) >= 3:
            stat_f, p_f = friedmanchisquare(*[boot_acc[k] for k in compare_keys[:3]])
            print(f"Friedman Test: stat={stat_f:.4f}, p={p_f:.5f}")
            results["friedman"] = {
                "stat": float(stat_f), "p_value": float(p_f), "models": compare_keys[:3],
            }

        if "Custom_CNN" in boot_acc and "ResNet50V2" in boot_acc:
            stat_w, p_w = wilcoxon(boot_acc["Custom_CNN"], boot_acc["ResNet50V2"])
            print(f"Wilcoxon Pairwise Test (CNN vs ResNet): p={p_w:.5f}")
            results["wilcoxon_cnn_resnet"] = {"stat": float(stat_w), "p_value": float(p_w)}

    out = reports_dir / "statistical_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out}")
    return results, preds


# ── 16. Error Analysis ───────────────────────────────────────

def run_error_analysis(model, model_name, X_test, y_test, y_pred, plots_dir, reports_dir):
    _, index_to_label = get_label_mappings()
    misclassified = np.where(y_pred != y_test)[0]
    print(f"Total misclassified samples ({model_name}): {len(misclassified)} / {len(y_test)}")

    records = []
    for idx in misclassified:
        probs = model.predict(X_test[idx : idx + 1], verbose=0)[0]
        records.append({
            "index": int(idx),
            "true_class": index_to_label[y_test[idx]],
            "pred_class": index_to_label[y_pred[idx]],
            "confidence": float(probs[y_pred[idx]]),
        })
    err_df = pd.DataFrame(records)
    err_path = reports_dir / f"error_analysis_{model_name.lower()}.csv"
    err_df.to_csv(err_path, index=False)
    print(f"Saved: {err_path}")

    if len(misclassified) == 0:
        return err_df

    n_show = min(len(misclassified), 4)
    fig, axes = plt.subplots(1, n_show, figsize=(14, 4))
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        idx = misclassified[i]
        probs = model.predict(X_test[idx : idx + 1], verbose=0)[0]
        axes[i].imshow(X_test[idx])
        axes[i].axis("off")
        axes[i].set_title(
            f"True: {index_to_label[y_test[idx]]}\n"
            f"Pred: {index_to_label[y_pred[idx]]} ({probs[y_pred[idx]]:.2%})",
            fontsize=10, fontweight="bold", color="crimson",
        )
    plt.tight_layout()
    out = plots_dir / "error_analysis_misclassifications.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return err_df


# ── 主流水线 ─────────────────────────────────────────────────

def run_xai_pipeline(
    dataset: str = "processed",
    primary_model: str = "Custom_CNN",
    compare_models: list[str] | None = None,
    quick: bool = False,
):
    create_output_dirs()
    setup_plot_style()
    quick = quick or QUICK_RUN
    plots_dir = OUTPUT_DIRS["plots"]
    reports_dir = OUTPUT_DIRS["reports"]

    compare_models = compare_models or ["Custom_CNN", "ResNet50V2", "ViT", "ConvNeXtTiny"]
    model_keys = list(dict.fromkeys([primary_model, *compare_models]))

    print(f"\n{'=' * 20} XAI Pipeline (notebook §13–16) — {dataset} {'=' * 20}")
    print_environment_info()

    X_train, y_train, _, _, _, _, X_test, y_test, _ = load_data(dataset)
    trained_models = load_trained_models(model_keys, dataset)

    if primary_model not in trained_models:
        primary_model = next(iter(trained_models))
        print(f"Primary model fallback: {primary_model}")

    primary = trained_models[primary_model]
    feat_train, feat_test = load_handcrafted_features(X_train, y_train, X_test, y_test)

    lime_samples = 50 if quick else 200
    n_bootstraps = 50 if quick else 200
    mc_passes = 10 if quick else 20
    sample_idx = int(np.where(y_test == 0)[0][0])

    # 13. XAI
    print("\n--- Section 7: Explainable AI ---")
    run_gradcam_section(primary, primary_model, X_test, y_test, sample_idx, plots_dir)
    run_lime_section(primary, X_test, sample_idx, plots_dir, num_samples=lime_samples)
    run_shap_section(feat_train, feat_test, y_train, plots_dir)

    # 14. Uncertainty
    print("\n--- Section 8: Uncertainty Quantification ---")
    uncertainty = run_mc_dropout(primary, X_test, y_test, plots_dir, n_passes=mc_passes)
    ensemble = run_deep_ensemble(trained_models, X_test, y_test, plots_dir)
    y_pred_primary = np.argmax(primary.predict(X_test, verbose=0), axis=1)
    bootstrap = run_bootstrap_ci(y_test, y_pred_primary, primary_model, n_bootstraps=n_bootstraps)

    # 15. Statistical Analysis
    print("\n--- Section 9: Statistical Analysis ---")
    stats, all_preds = run_statistical_analysis(trained_models, X_test, y_test, reports_dir)

    # 16. Error Analysis
    print("\n--- Section 10: Error Analysis ---")
    errors = run_error_analysis(
        primary, primary_model, X_test, y_test,
        all_preds.get(primary_model, y_pred_primary),
        plots_dir, reports_dir,
    )

    summary = {
        "dataset": dataset,
        "primary_model": primary_model,
        "loaded_models": list(trained_models.keys()),
        "uncertainty": uncertainty,
        "deep_ensemble": ensemble,
        "bootstrap_ci": bootstrap,
        "statistical_tests": stats,
        "n_misclassified": int((all_preds.get(primary_model, y_pred_primary) != y_test).sum()),
    }
    summary_path = reports_dir / "xai_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {summary_path}")
    print("XAI pipeline completed.")
    return summary


def print_environment_info():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"Plots -> {OUTPUT_DIRS['plots']}")
    print(f"Reports -> {OUTPUT_DIRS['reports']}")


def main():
    parser = argparse.ArgumentParser(
        description="XAI, Uncertainty, Statistical Analysis, Error Analysis",
    )
    parser.add_argument("--dataset", choices=["processed", "ori"], default="processed")
    parser.add_argument(
        "--model", default="Custom_CNN",
        choices=list(MODEL_REGISTRY.keys()),
        help="主分析模型（Grad-CAM/LIME/MC Dropout/误差分析）",
    )
    parser.add_argument("--quick", action="store_true", help="快速模式（更少采样）")
    args = parser.parse_args()
    run_xai_pipeline(dataset=args.dataset, primary_model=args.model, quick=args.quick)


if __name__ == "__main__":
    main()
