import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.ndimage import median_filter
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import config
from config import (
    AUGMENTATION_CONFIG,
    CLASSES,
    NUM_CLASSES,
    MIXED_DATASET_PATH,
    ORI_DATASET_PATH,
    OUTLIER_CONTAMINATION,
    OUTLIER_RESIZE,
    PROCESSED_DATASET_PATH,
    PROCESSED_SUBDIRS,
    QUICK_RUN_SAMPLES_PER_CLASS,
    SEED,
    TARGET_SIZE,
    get_label_mappings,
    resolve_project_path,
    setup_environment,
    setup_plot_style,
    to_relative_path,
)


# ── 第 4 节：数据集加载 ──────────────────────────────────────

def load_dataset(dataset_path=None):
    dataset_path = resolve_project_path(dataset_path or ORI_DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"混合数据集目录不存在: {dataset_path}")

    data_records = []
    for label in CLASSES:
        class_dir = dataset_path / label
        if not class_dir.is_dir():
            print(f"警告: 类别目录不存在 - {class_dir}")
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() in config.IMAGE_EXTENSIONS:
                data_records.append({
                    "filepath": to_relative_path(f),
                    "filename": f.name,
                    "label": label,
                })

    df = pd.DataFrame(data_records)
    print(f"Total images scanned: {len(df)}")
    print("Class counts:")
    print(df["label"].value_counts())
    assert not df["filepath"].isnull().any(), "Found null filepaths"
    print("Dataset structure is intact. No missing file records.")
    return df


def apply_quick_run_subsample(df):
    print(f"Developer Quick Run Mode is set to: {config.QUICK_RUN}")
    if config.QUICK_RUN:
        df = (
            df.groupby("label")
            .apply(lambda x: x.sample(n=min(len(x), QUICK_RUN_SAMPLES_PER_CLASS), random_state=SEED))
            .reset_index(drop=True)
        )
        print(f"Subsampled dataset size for QUICK_RUN: {len(df)}")
    return df


def plot_class_distribution(df, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(data=df, x="label", hue="label", ax=axes[0], palette="viridis", order=CLASSES, legend=False)
    axes[0].set_title("Class Counts Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tumor Type")
    axes[0].set_ylabel("Count")

    class_counts = df["label"].value_counts()
    axes[1].pie(
        class_counts,
        labels=class_counts.index,
        autopct="%1.1f%%",
        colors=sns.color_palette("viridis", len(CLASSES)),
    )
    axes[1].set_title("Class Share Percentage", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = plots_dir / "eda_class_distribution.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def analyze_image_resolutions(df, plots_dir):
    image_sizes = []
    for fp in df["filepath"]:
        try:
            with Image.open(resolve_project_path(fp)) as img:
                w, h = img.size
                c = len(img.getbands())
                image_sizes.append({"width": w, "height": h, "channels": c})
        except Exception:
            pass

    sizes_df = pd.DataFrame(image_sizes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=sizes_df, x="width", kde=True, ax=axes[0], color="teal")
    axes[0].set_title("Distribution of Image Widths", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Width (pixels)")

    sns.boxplot(data=sizes_df, y="height", ax=axes[1], color="coral")
    axes[1].set_title("Boxplot of Image Heights", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Height (pixels)")

    plt.tight_layout()
    out_path = plots_dir / "eda_image_resolutions.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    print("Image resolution stats:")
    print(sizes_df.describe())
    return sizes_df


def plot_sample_images(df, plots_dir):
    fig, axes = plt.subplots(len(CLASSES), 5, figsize=(15, 12))
    for idx, label in enumerate(CLASSES):
        class_subset = df[df["label"] == label].sample(5, random_state=SEED)
        for col_idx, (_, row) in enumerate(class_subset.iterrows()):
            img = Image.open(resolve_project_path(row["filepath"]))
            axes[idx, col_idx].imshow(img)
            axes[idx, col_idx].axis("off")
            if col_idx == 0:
                axes[idx, col_idx].set_ylabel(label, fontsize=12, fontweight="bold")
                axes[idx, col_idx].axis("on")
                axes[idx, col_idx].set_xticks([])
                axes[idx, col_idx].set_yticks([])
                for spine in axes[idx, col_idx].spines.values():
                    spine.set_visible(False)

    plt.suptitle("Representative Sample Images per Class", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out_path = plots_dir / "eda_sample_images.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_spatial_statistics(df, plots_dir):
    mean_images, std_images = {}, {}
    for label in CLASSES:
        class_subset = df[df["label"] == label]
        imgs = []
        for fp in class_subset["filepath"].sample(min(len(class_subset), 20), random_state=SEED):
            with Image.open(resolve_project_path(fp)) as im:
                im_resized = im.convert("L").resize((128, 128))
                imgs.append(np.array(im_resized, dtype=np.float32) / 255.0)
        imgs = np.array(imgs)
        mean_images[label] = np.mean(imgs, axis=0)
        std_images[label] = np.std(imgs, axis=0)

    fig, axes = plt.subplots(2, len(CLASSES), figsize=(16, 8))
    for idx, label in enumerate(CLASSES):
        axes[0, idx].imshow(mean_images[label], cmap="gray")
        axes[0, idx].set_title(f"Mean: {label}", fontsize=11, fontweight="bold")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(std_images[label], cmap="hot")
        axes[1, idx].set_title(f"Std Dev: {label}", fontsize=11, fontweight="bold")
        axes[1, idx].axis("off")

    plt.suptitle(
        "Spatial Statistical Images (Pixel-wise Mean and Standard Deviation)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = plots_dir / "eda_spatial_stats.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def detect_outliers(df):
    flat_pixels = []
    for fp in df["filepath"]:
        with Image.open(resolve_project_path(fp)) as im:
            im_resized = im.convert("L").resize(OUTLIER_RESIZE)
            flat_pixels.append(np.array(im_resized).flatten())

    flat_pixels = np.array(flat_pixels, dtype=np.float32) / 255.0
    iso_forest = IsolationForest(contamination=OUTLIER_CONTAMINATION, random_state=SEED)
    outliers = iso_forest.fit_predict(flat_pixels)

    df = df.copy()
    df["is_outlier"] = outliers
    n_outliers = int((outliers == -1).sum())
    print(f"Number of outliers detected: {n_outliers}")
    if n_outliers > 0:
        print("Outlier file samples:")
        print(df[df["is_outlier"] == -1]["filepath"].head())
    return df


def run_eda(df, plots_dir):
    """执行完整 EDA 流程。"""
    print("\n=== Section 5: Exploratory Data Analysis ===")
    plot_class_distribution(df, plots_dir)
    analyze_image_resolutions(df, plots_dir)
    plot_sample_images(df, plots_dir)
    plot_spatial_statistics(df, plots_dir)
    df = detect_outliers(df)
    return df


def preprocess_image_pipeline(filepath, target_size=TARGET_SIZE):
    with Image.open(filepath) as img:
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize(target_size)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0

    gray = np.mean(img_arr, axis=-1)
    denoised = gaussian(gray, sigma=0.5)
    smoothed = median_filter(denoised, size=3)
    enhanced = equalize_adapthist(smoothed, clip_limit=0.01)
    enhanced_rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    return enhanced_rgb


def plot_preprocessing_comparison(df, plots_dir):
    """可视化预处理前后对比。"""
    sample_fp = resolve_project_path(df.iloc[0]["filepath"])
    sample_orig = np.array(Image.open(sample_fp).convert("RGB").resize(TARGET_SIZE)) / 255.0
    sample_prep = preprocess_image_pipeline(sample_fp)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sample_orig)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sample_prep)
    axes[1].set_title("Denoised & Enhanced (CLAHE)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = plots_dir / "preprocessing_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def stratified_split(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
    )
    print(f"Training set: {len(train_df)}")
    print(f"Validation set: {len(val_df)}")
    print(f"Testing set: {len(test_df)}")
    return train_df, val_df, test_df


def save_preprocessed_images(partition_df, split_name, label_to_index):
    split_dir = resolve_project_path(PROCESSED_SUBDIRS[split_name])
    processed_paths = []

    for _, row in partition_df.iterrows():
        try:
            img_arr = preprocess_image_pipeline(resolve_project_path(row["filepath"]))
            out_dir = split_dir / row["label"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (Path(row["filename"]).stem + ".png")

            img_uint8 = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(out_path)

            processed_paths.append({
                "original_filepath": row["filepath"],
                "processed_filepath": to_relative_path(out_path),
                "filename": row["filename"],
                "label": row["label"],
                "label_idx": label_to_index[row["label"]],
                "split": split_name,
                "is_outlier": row.get("is_outlier", 1),
            })
        except Exception as e:
            print(f"警告: 处理失败 {row['filepath']}: {e}")

    return pd.DataFrame(processed_paths)


def load_and_preprocess_partition(partition_df, label_to_index):
    import keras

    images, labels = [], []
    for _, row in partition_df.iterrows():
        try:
            if "processed_filepath" in row and resolve_project_path(row["processed_filepath"]).exists():
                with Image.open(resolve_project_path(row["processed_filepath"])) as img:
                    img_arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            else:
                img_arr = preprocess_image_pipeline(resolve_project_path(row["filepath"]))
            label_idx = label_to_index[row["label"]]
            images.append(img_arr)
            labels.append(label_idx)
        except Exception:
            pass

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    y_one_hot = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
    return X, y, y_one_hot


def save_arrays(X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh):
    arrays_dir = resolve_project_path(PROCESSED_SUBDIRS["arrays"])
    np.save(arrays_dir / "X_train.npy", X_train)
    np.save(arrays_dir / "y_train.npy", y_train)
    np.save(arrays_dir / "y_train_oh.npy", y_train_oh)
    np.save(arrays_dir / "X_val.npy", X_val)
    np.save(arrays_dir / "y_val.npy", y_val)
    np.save(arrays_dir / "y_val_oh.npy", y_val_oh)
    np.save(arrays_dir / "X_test.npy", X_test)
    np.save(arrays_dir / "y_test.npy", y_test)
    np.save(arrays_dir / "y_test_oh.npy", y_test_oh)
    print(f"Arrays saved to: {arrays_dir}")

def build_data_augmentation():
    import keras
    from keras import layers

    return keras.Sequential([
        layers.RandomFlip(AUGMENTATION_CONFIG["random_flip"], seed=SEED),
        layers.RandomRotation(AUGMENTATION_CONFIG["random_rotation"], seed=SEED),
        layers.RandomTranslation(
            AUGMENTATION_CONFIG["random_translation"],
            AUGMENTATION_CONFIG["random_translation"],
            seed=SEED,
        ),
        layers.RandomContrast(AUGMENTATION_CONFIG["random_contrast"], seed=SEED),
    ])


def apply_mixup(images, labels, alpha=None):
    import tensorflow as tf

    alpha = alpha or AUGMENTATION_CONFIG["mixup_alpha"]
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.float32)
    batch_size = tf.shape(images)[0]
    lmbda = tf.random.stateless_uniform(
        shape=[batch_size, 1, 1, 1], seed=[SEED, 1], minval=0.5, maxval=1.0
    )
    indices = tf.random.shuffle(tf.range(batch_size), seed=SEED)
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels, indices)
    mixed_images = lmbda * images + (1.0 - lmbda) * images_shuffled
    mixed_labels = (
        tf.squeeze(lmbda, axis=(2, 3)) * labels
        + (1.0 - tf.squeeze(lmbda, axis=(2, 3))) * labels_shuffled
    )
    return mixed_images, mixed_labels


def apply_cutmix(images, labels, alpha=None):
    return apply_mixup(images, labels, alpha or AUGMENTATION_CONFIG["cutmix_alpha"])


def plot_mixup_samples(X_train, y_train_oh, index_to_label, plots_dir):
    import tensorflow as tf

    sample_imgs = tf.constant(X_train[:4], dtype=tf.float32)
    sample_lbls = tf.constant(y_train_oh[:4], dtype=tf.float32)
    mixed_imgs, mixed_lbls = apply_mixup(sample_imgs, sample_lbls)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        axes[i].imshow(mixed_imgs[i].numpy())
        axes[i].axis("off")
        lbl_str = ", ".join(
            f"{index_to_label[idx]}:{mixed_lbls[i][idx]:.2f}"
            for idx in range(NUM_CLASSES)
            if mixed_lbls[i][idx] > 0.1
        )
        axes[i].set_title(lbl_str, fontsize=8)
    plt.suptitle("MixUp Augmented Images & Label Mix percentages", fontsize=11, fontweight="bold")
    out_path = plots_dir / "augmented_mixup_samples.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_metadata(all_processed_df, label_to_index, index_to_label):
    metadata_dir = resolve_project_path(PROCESSED_SUBDIRS["metadata"])

    all_processed_df.to_csv(metadata_dir / "dataset_info.csv", index=False)

    with open(metadata_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {"label_to_index": label_to_index, "index_to_label": {str(k): v for k, v in index_to_label.items()}},
            f,
            indent=2,
            ensure_ascii=False,
        )

    split_stats = all_processed_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    split_stats.to_csv(metadata_dir / "split_stats.csv")

    with open(metadata_dir / "augmentation_config.json", "w", encoding="utf-8") as f:
        json.dump(AUGMENTATION_CONFIG, f, indent=2)

    print(f"Metadata saved to: {metadata_dir}")


def main(quick_run=None):
    if quick_run is not None:
        config.QUICK_RUN = quick_run

    print("=" * 60)
    print("Brain Tumor MRI - Dataset Load & Initial Processing")
    print("=" * 60)

    config.create_output_dirs()
    setup_environment()
    setup_plot_style()
    config.print_environment_info()

    label_to_index, index_to_label = get_label_mappings()
    plots_dir = resolve_project_path(config.OUTPUT_DIRS["plots"])

    print("\n=== Section 1: Dataset Loading & Initial Inspection ===")
    df = load_dataset()

    df = apply_quick_run_subsample(df)
    df = run_eda(df, plots_dir)

    print("\n=== Section 2: Advanced Image Preprocessing Pipeline ===")
    plot_preprocessing_comparison(df, plots_dir)
    train_df, val_df, test_df = stratified_split(df)

    print("Saving preprocessed images to data/processed_data ...")
    train_processed = save_preprocessed_images(train_df, "train", label_to_index)
    val_processed = save_preprocessed_images(val_df, "val", label_to_index)
    test_processed = save_preprocessed_images(test_df, "test", label_to_index)
    all_processed_df = pd.concat([train_processed, val_processed, test_processed], ignore_index=True)

    print("\n=== Section 3: Label Encoding & Preprocessing Verification ===")
    print("Loading and preprocessing all partitions...")
    X_train, y_train, y_train_oh = load_and_preprocess_partition(train_processed, label_to_index)
    X_val, y_val, y_val_oh = load_and_preprocess_partition(val_processed, label_to_index)
    X_test, y_test, y_test_oh = load_and_preprocess_partition(test_processed, label_to_index)

    print(f"X_train shape: {X_train.shape}, y_train_oh shape: {y_train_oh.shape}")
    print(f"X_val shape: {X_val.shape}, y_val_oh shape: {y_val_oh.shape}")
    print(f"X_test shape: {X_test.shape}, y_test_oh shape: {y_test_oh.shape}")

    save_arrays(X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh)
    save_metadata(all_processed_df, label_to_index, index_to_label)

    print("\n=== Section 4: Data Augmentation Pipeline ===")
    data_augmentation = build_data_augmentation()
    print(f"Data augmentation layers: {len(data_augmentation.layers)} layers configured")
    plot_mixup_samples(X_train, y_train_oh, index_to_label, plots_dir)

    print("\n" + "=" * 60)
    print("Dataset processing completed successfully!")
    print(f"Mixed data (unchanged): {to_relative_path(MIXED_DATASET_PATH)}")
    print(f"Processed data saved to: {to_relative_path(PROCESSED_DATASET_PATH)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="脑肿瘤 MRI 数据加载与预处理")
    args = parser.parse_args()
    main(quick_run=True if args.quick else None)
