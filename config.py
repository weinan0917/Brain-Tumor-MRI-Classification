"""
脑肿瘤 MRI 分类项目 - 环境参数配置
对应 notebook 第 3 节：Environment Setup & Reproducibility
"""

import os
import random
from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── 随机种子 & 可复现性 ─────────────────────────────────────
SEED = 42

# ── 数据集路径（均为相对 PROJECT_ROOT 的路径） ───────────────
DATA_DIR = Path("data")
ORI_DATA_DIR = DATA_DIR / "ori_data"
MIXED_DATASET_PATH = DATA_DIR / "mixed_data"
PROCESSED_DATASET_PATH = DATA_DIR / "processed_data"

# 兼容旧变量名：统一去重后的混合数据集
ORI_DATASET_PATH = MIXED_DATASET_PATH

# 原始四个数据集
DATASET_1_DIR = ORI_DATA_DIR / "Brain Tumor MRI Dataset"
DATASET_2_DIR = ORI_DATA_DIR / "BRISC 2025 Dataset"
BRAIN_TUMOR_DIR = ORI_DATA_DIR / "BrainTumorDataPublic"
NINS_DIR = ORI_DATA_DIR / "NINS_Dataset"

# ── 类别定义 ────────────────────────────────────────────────
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)

# ── 图像处理参数 ────────────────────────────────────────────
TARGET_SIZE = (224, 224)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ── 数据集划分比例 (70% / 15% / 15%) ────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── 快速验证模式（子采样，用于调试） ────────────────────────
QUICK_RUN = False
QUICK_RUN_SAMPLES_PER_CLASS = 60

# ── 输出目录 ────────────────────────────────────────────────
OUTPUT_DIRS = {
    "models": Path("models"),
    "plots": DATA_DIR / "plots",
    "reports": Path("reports"),
}

# data/processed_data 子目录
PROCESSED_SUBDIRS = {
    "metadata": PROCESSED_DATASET_PATH / "metadata",
    "arrays": PROCESSED_DATASET_PATH / "arrays",
    "features": PROCESSED_DATASET_PATH / "features",
    "train": PROCESSED_DATASET_PATH / "train",
    "val": PROCESSED_DATASET_PATH / "val",
    "test": PROCESSED_DATASET_PATH / "test",
}

# ── 数据增强参数 ────────────────────────────────────────────
AUGMENTATION_CONFIG = {
    "random_flip": "horizontal_and_vertical",
    "random_rotation": 0.15,
    "random_translation": 0.1,
    "random_contrast": 0.1,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 0.2,
}

# ── 特征分析参数 ────────────────────────────────────────────
FEATURE_CONFIG = {
    "hog_orientations": 8,
    "hog_pixels_per_cell": (32, 32),
    "hog_cells_per_block": (1, 1),
    "lbp_points": 8,
    "lbp_radius": 1,
    "pca_components": 2,
    "tsne_perplexity": 15,
}

# ── 可视化参数 ──────────────────────────────────────────────
PLOT_DPI = 300
PLOT_STYLE = "whitegrid"

# ── 异常值检测参数 ────────────────────────────────────────────
OUTLIER_CONTAMINATION = 0.03
OUTLIER_RESIZE = (64, 64)

# ── Keras 后端 ──────────────────────────────────────────────
KERAS_BACKEND = "tensorflow"


def resolve_project_path(path) -> Path:
    """将相对路径解析为基于项目根目录的绝对路径。"""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def to_relative_path(path) -> str:
    """将路径转换为相对项目根目录的 POSIX 风格字符串。"""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    return p.relative_to(PROJECT_ROOT).as_posix()


def setup_environment():
    """设置随机种子、Keras 后端与 CPU 运行环境。"""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["KERAS_BACKEND"] = KERAS_BACKEND
    # 强制 CPU 运行，避免 Windows 下 TF GPU 兼容问题
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    random.seed(SEED)

    import numpy as np
    np.random.seed(SEED)

    import tensorflow as tf
    tf.random.set_seed(SEED)

    return np, tf


def setup_plot_style():
    """配置 matplotlib / seaborn 输出风格。"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style=PLOT_STYLE)
    plt.rcParams["figure.dpi"] = PLOT_DPI
    plt.rcParams["savefig.dpi"] = PLOT_DPI
    plt.rcParams["font.family"] = "sans-serif"


def create_output_dirs():
    """创建所有输出目录。"""
    for directory in OUTPUT_DIRS.values():
        resolve_project_path(directory).mkdir(parents=True, exist_ok=True)

    for subdir in PROCESSED_SUBDIRS.values():
        resolve_project_path(subdir).mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        for cls in CLASSES:
            resolve_project_path(PROCESSED_SUBDIRS[split] / cls).mkdir(parents=True, exist_ok=True)


def get_label_mappings():
    """返回标签与索引的双向映射字典。"""
    label_to_index = {label: i for i, label in enumerate(CLASSES)}
    index_to_label = {i: label for label, i in label_to_index.items()}
    return label_to_index, index_to_label


def print_environment_info():
    """打印当前运行环境信息。"""
    import keras
    import tensorflow as tf

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"Running on CPU (CUDA_VISIBLE_DEVICES=-1)")
    print(f"Mixed dataset: {to_relative_path(MIXED_DATASET_PATH)}")
    print(f"Processed dataset: {to_relative_path(PROCESSED_DATASET_PATH)}")
    print(f"QUICK_RUN mode: {QUICK_RUN}")
