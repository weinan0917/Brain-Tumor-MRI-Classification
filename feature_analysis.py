import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config
from config import (
    CLASSES,
    FEATURE_CONFIG,
    PROCESSED_SUBDIRS,
    SEED,
    get_label_mappings,
    resolve_project_path,
    setup_environment,
    setup_plot_style,
)


def load_processed_arrays():
    arrays_dir = resolve_project_path(PROCESSED_SUBDIRS["arrays"])
    required = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    for fname in required:
        if not (arrays_dir / fname).exists():
            raise FileNotFoundError(
                f"缺少 {fname}，请先运行 Dataset_Load_Initial.py 生成 data/processed_data"
            )

    X_train = np.load(arrays_dir / "X_train.npy")
    y_train = np.load(arrays_dir / "y_train.npy")
    X_test = np.load(arrays_dir / "X_test.npy")
    y_test = np.load(arrays_dir / "y_test.npy")

    print(f"Loaded X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def extract_handcrafted_features(images):
    cfg = FEATURE_CONFIG
    features = []

    for img in images:
        gray = np.mean(img, axis=-1)

        hog_feat = hog(
            gray,
            orientations=cfg["hog_orientations"],
            pixels_per_cell=cfg["hog_pixels_per_cell"],
            cells_per_block=cfg["hog_cells_per_block"],
        )

        lbp = local_binary_pattern(
            gray,
            P=cfg["lbp_points"],
            R=cfg["lbp_radius"],
            method="uniform",
        )
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

        gray_uint8 = np.uint8(gray * 255)
        glcm = graycomatrix(
            gray_uint8,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True,
        )
        contrast = graycoprops(glcm, "contrast")[0, 0]
        homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
        energy = graycoprops(glcm, "energy")[0, 0]
        correlation = graycoprops(glcm, "correlation")[0, 0]
        texture_feat = np.array([contrast, homogeneity, energy, correlation])

        feat = np.concatenate([hog_feat, lbp_hist, texture_feat])
        features.append(feat)

    return np.array(features)


def save_features(feat_train, feat_test):
    features_dir = resolve_project_path(PROCESSED_SUBDIRS["features"])
    features_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_dir / "feat_train.npy", feat_train)
    np.save(features_dir / "feat_test.npy", feat_test)

    meta = {
        "train_shape": list(feat_train.shape),
        "test_shape": list(feat_test.shape),
        "feature_dim": feat_train.shape[1],
        "feature_types": ["HOG", "LBP_histogram", "GLCM(contrast,homogeneity,energy,correlation)"],
    }
    with open(features_dir / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Features saved to: {config.to_relative_path(features_dir)}")
    print(f"Handcrafted feature dimension: {feat_train.shape[1]}")


def plot_dimensionality_reduction(feat_train, y_train, plots_dir):
    _, index_to_label = get_label_mappings()
    cfg = FEATURE_CONFIG

    pca = PCA(n_components=cfg["pca_components"], random_state=SEED)
    pca_res = pca.fit_transform(feat_train)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    tsne = TSNE(
        n_components=cfg["pca_components"],
        perplexity=cfg["tsne_perplexity"],
        random_state=SEED,
    )
    tsne_res = tsne.fit_transform(feat_train)

    labels_str = [index_to_label[int(y)] for y in y_train]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=labels_str, ax=axes[0], palette="viridis")
    axes[0].set_title("PCA of Handcrafted Features", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=labels_str, ax=axes[1], palette="viridis")
    axes[1].set_title("t-SNE of Handcrafted Features", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = plots_dir / "dimensionality_reduction.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_feature_statistics_report(feat_train, y_train, reports_dir):
    _, index_to_label = get_label_mappings()
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for cls_idx in range(len(CLASSES)):
        mask = y_train == cls_idx
        cls_feats = feat_train[mask]
        records.append({
            "class": index_to_label[cls_idx],
            "count": int(mask.sum()),
            "mean_norm": float(np.linalg.norm(cls_feats.mean(axis=0))),
            "std_mean": float(cls_feats.std(axis=0).mean()),
            "min": float(cls_feats.min()),
            "max": float(cls_feats.max()),
        })

    stats_df = pd.DataFrame(records)
    out_path = reports_dir / "feature_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"Feature statistics report saved: {out_path}")
    print(stats_df.to_string(index=False))
    return stats_df


def main():
    print("=" * 60)
    print("Brain Tumor MRI - Feature Analysis & Dimensionality Reduction")
    print("=" * 60)

    config.create_output_dirs()
    setup_environment()
    setup_plot_style()

    plots_dir = resolve_project_path(config.OUTPUT_DIRS["plots"])
    reports_dir = resolve_project_path(config.OUTPUT_DIRS["reports"])

    X_train, y_train, X_test, y_test = load_processed_arrays()

    print("\n=== Section 5: Extracting Handcrafted Features ===")
    print("Extracting texture features from training set...")
    feat_train = extract_handcrafted_features(X_train)
    print("Extracting texture features from test set...")
    feat_test = extract_handcrafted_features(X_test)
    save_features(feat_train, feat_test)
    print("\n=== PCA & t-SNE Visualization ===")
    plot_dimensionality_reduction(feat_train, y_train, plots_dir)
    print("\n=== Feature Statistics Report ===")
    generate_feature_statistics_report(feat_train, y_train, reports_dir)
    print("\n" + "=" * 60)
    print("Feature analysis completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
