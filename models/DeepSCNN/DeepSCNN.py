"""
DeepSCNN-Brain Attention v3 — models 训练模块（ImageNet 预训练）
核心架构（不可变）: 双视图 Siamese + 通道/空间双重注意力 + 残差编码器
骨干: ResNet50V2 (weights='imagenet') + SE/Spatial 精炼 + 256 维特征
训练: Phase1 冻结 backbone 训头 → Phase2 解冻顶层微调
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import NUM_CLASSES, SEED, TARGET_SIZE, create_output_dirs, print_environment_info, setup_environment, setup_plot_style

setup_environment()
import keras
import tensorflow as tf
from keras import layers, models, regularizers

from models.common import evaluate_model, get_output_dirs, load_data, plot_training_history

MODEL_NAME = "DeepSCNN"
MODEL_DISPLAY_NAME = "DeepSCNN-BrainAttention_v3"

FEAT_DIM = 256
L2 = regularizers.l2(1e-4)
UNFREEZE_LAST_N = 50

PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 30
DEFAULT_BATCH_SIZE = 16
PHASE1_LR = 1e-3
PHASE2_LR = 5e-5


def _class_weights(y_one_hot: np.ndarray) -> dict[int, float]:
    labels = np.argmax(y_one_hot, axis=1)
    weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=labels)
    return {int(i): float(w) for i, w in enumerate(weights)}


def train_deepscnn(
    model, X_train, y_train_oh, X_val, y_val_oh,
    output_dirs, epochs, batch_size, patience, initial_lr,
):
    cw = _class_weights(y_train_oh)
    print(f"Class weights: {cw}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    ckpt = output_dirs["checkpoints"] / f"best_{MODEL_DISPLAY_NAME}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=max(3, patience // 3),
            min_lr=1e-7, verbose=1,
        ),
    ]
    print(f"\n{'=' * 20} Training {MODEL_DISPLAY_NAME} (lr={initial_lr}, ImageNet) {'=' * 20}")
    t0 = time.time()
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1,
        class_weight=cw,
    )
    return model, history.history, time.time() - t0


# ── 注意力模块（接在预训练特征图之后，结构不变）────────────────

def _se_block(x, ratio=8, name="se"):
    channels = int(x.shape[-1])
    gap = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    fc1 = layers.Dense(max(channels // ratio, 4), activation="relu", name=f"{name}_fc1")(gap)
    scale = layers.Dense(channels, activation="sigmoid", name=f"{name}_fc2")(fc1)
    scale = layers.Reshape((1, 1, channels), name=f"{name}_rs")(scale)
    return layers.Multiply(name=f"{name}_mul")([x, scale])


def _spatial_attention(x, name="sa"):
    avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name=f"{name}_avg")(x)
    mx = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name=f"{name}_max")(x)
    cat = layers.Concatenate(name=f"{name}_cat")([avg, mx])
    attn = layers.Conv2D(1, 7, padding="same", activation="sigmoid", name=f"{name}_conv")(cat)
    return layers.Multiply(name=f"{name}_mul")([x, attn])


def build_shared_encoder(
    input_shape=None,
    trainable_backbone=False,
    unfreeze_last_n=0,
    name="shared_encoder",
):
    """
    共享 Siamese 编码器:
      ResNet50V2 (ImageNet) → SE 通道注意力 → 空间注意力 → GAP/GMP → Dense(256)
    """
    input_shape = input_shape or (*TARGET_SIZE, 3)

    backbone = keras.applications.ResNet50V2(
        weights="imagenet", include_top=False, input_shape=input_shape,
    )
    backbone.trainable = trainable_backbone
    if trainable_backbone and unfreeze_last_n > 0:
        for layer in backbone.layers[:-unfreeze_last_n]:
            layer.trainable = False

    enc_in = layers.Input(shape=input_shape, name=f"{name}_in")
    x = backbone(enc_in)
    x = _se_block(x, name=f"{name}_se")
    x = _spatial_attention(x, name=f"{name}_sa")
    gap = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    gmp = layers.GlobalMaxPooling2D(name=f"{name}_gmp")(x)
    x = layers.Concatenate(name=f"{name}_pool_cat")([gap, gmp])
    x = layers.Dense(FEAT_DIM, activation="relu", kernel_regularizer=L2, name=f"{name}_fc")(x)
    x = layers.BatchNormalization(name=f"{name}_fc_bn")(x)
    x = layers.Dropout(0.25, name=f"{name}_fc_drop")(x)
    return models.Model(enc_in, x, name=name)


@keras.saving.register_keras_serializable(package="DeepSCNN")
def _cosine_similarity(tensors):
    a, b = tensors[0], tensors[1]
    dot = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    na = tf.sqrt(tf.reduce_sum(tf.square(a), axis=-1, keepdims=True) + 1e-8)
    nb = tf.sqrt(tf.reduce_sum(tf.square(b), axis=-1, keepdims=True) + 1e-8)
    return dot / (na * nb)


def build_deepscnn_brain_attention(
    input_shape=None,
    num_classes=None,
    trainable_backbone=False,
    unfreeze_last_n=0,
):
    """
    DeepSCNN-Brain Attention（ImageNet 预训练 ResNet50V2）。
    双视图 Siamese + 融合层结构不变。
    """
    input_shape = input_shape or (*TARGET_SIZE, 3)
    num_classes = num_classes or NUM_CLASSES

    inputs = layers.Input(shape=input_shape, name="rgb_input")

    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.08, name="aug_rot")(x)
    x = layers.RandomZoom((-0.08, 0.08), name="aug_zoom")(x)
    x = layers.RandomContrast(0.1, name="aug_contrast")(x)
    x = layers.Lambda(
        lambda arr: keras.applications.resnet_v2.preprocess_input(arr * 255.0),
        name="resnet_preprocess",
    )(x)

    view_a = x
    view_b = layers.Lambda(lambda t: tf.reverse(t, axis=[2]), name="flip_b")(x)

    encoder = build_shared_encoder(
        input_shape=input_shape,
        trainable_backbone=trainable_backbone,
        unfreeze_last_n=unfreeze_last_n,
    )
    feat_a = encoder(view_a)
    feat_b = encoder(view_b)

    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]), name="abs_diff")([feat_a, feat_b])
    prod = layers.Multiply(name="elem_prod")([feat_a, feat_b])
    corr = layers.Lambda(_cosine_similarity, name="cos_sim")([feat_a, feat_b])
    merged = layers.Concatenate(name="fusion")([feat_a, feat_b, diff, prod, corr])

    x = layers.Dense(384, activation="relu", kernel_regularizer=L2, name="head_fc1")(merged)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.35, name="head_drop1")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=L2, name="head_fc2")(x)
    x = layers.Dropout(0.2, name="head_drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="pred")(x)

    return models.Model(inputs=inputs, outputs=outputs, name=MODEL_DISPLAY_NAME)


def run_phase2_only(dataset, batch_size, phase2_epochs, unfreeze_last_n):
    """从 Phase1 已保存的 checkpoint 继续 Phase2 微调 + 测试。"""
    create_output_dirs()
    setup_plot_style()
    print_environment_info()
    tf.keras.utils.set_random_seed(SEED)

    print(f"\n>>> Resuming {MODEL_DISPLAY_NAME} Phase 2 only on: {dataset}")
    X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh = load_data(dataset)
    output_dirs = get_output_dirs(MODEL_NAME, dataset)
    output_dirs["dataset"] = dataset
    ckpt = output_dirs["checkpoints"] / f"best_{MODEL_DISPLAY_NAME}.keras"
    if not ckpt.exists():
        raise FileNotFoundError(f"未找到 Phase1 权重，请先完成 Phase1 或检查: {ckpt}")

    total_start = time.time()
    print(f"\n{'=' * 20} Phase 2: Fine-tune top {unfreeze_last_n} layers ({phase2_epochs} ep, lr={PHASE2_LR}) {'=' * 20}")
    model = build_deepscnn_brain_attention(
        trainable_backbone=True, unfreeze_last_n=unfreeze_last_n,
    )
    model.load_weights(str(ckpt))
    print(f"Loaded Phase 1 weights: {ckpt}")

    model, h2, t2 = train_deepscnn(
        model, X_train, y_train_oh, X_val, y_val_oh, output_dirs,
        epochs=phase2_epochs, batch_size=batch_size, patience=12, initial_lr=PHASE2_LR,
    )

    if ckpt.exists():
        model.load_weights(str(ckpt))
        print(f"Loaded best checkpoint: {ckpt}")

    total_time = time.time() - total_start
    hist_path = output_dirs["reports"] / "training_history.json"
    if hist_path.exists():
        with open(hist_path, encoding="utf-8") as f:
            merged = json.load(f)
        for k in h2:
            merged[k] = merged.get(k, []) + h2[k]
    else:
        merged = h2

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    with open(output_dirs["reports"] / "training_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "architecture": MODEL_DISPLAY_NAME,
            "pretrain": "ImageNet-ResNet50V2",
            "backbone": "ResNet50V2",
            "resumed_from": "phase2_only",
            "unfreeze_last_n": unfreeze_last_n,
            "phase2_epochs": phase2_epochs,
            "phase2_lr": PHASE2_LR,
            "phase2_time_s": t2,
        }, f, indent=2)

    plot_training_history(merged, MODEL_DISPLAY_NAME, output_dirs["plots"])
    metrics = evaluate_model(
        model, MODEL_DISPLAY_NAME,
        X_test, y_test, y_test_oh, total_time, output_dirs,
        use_tta=True, n_tta=8,
    )
    print(f"\n{MODEL_DISPLAY_NAME} on [{dataset}] done. Test Accuracy = {metrics['Accuracy']:.4f}")
    return model, merged, metrics


def run_two_phase_pipeline(dataset, batch_size, phase1_epochs, phase2_epochs, unfreeze_last_n):
    create_output_dirs()
    setup_plot_style()
    print_environment_info()
    tf.keras.utils.set_random_seed(SEED)

    print(f"\n>>> Running {MODEL_DISPLAY_NAME} [models/DeepSCNN, ImageNet] on: {dataset}")
    X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh = load_data(dataset)
    output_dirs = get_output_dirs(MODEL_NAME, dataset)
    output_dirs["dataset"] = dataset
    ckpt = output_dirs["checkpoints"] / f"best_{MODEL_DISPLAY_NAME}.keras"
    total_start = time.time()

    print(f"\n{'=' * 20} Phase 1: Frozen ResNet50V2 ({phase1_epochs} ep, lr={PHASE1_LR}) {'=' * 20}")
    model = build_deepscnn_brain_attention(trainable_backbone=False)
    print(f"Params: {model.count_params():,} (ImageNet backbone frozen)")
    model, h1, t1 = train_deepscnn(
        model, X_train, y_train_oh, X_val, y_val_oh, output_dirs,
        epochs=phase1_epochs, batch_size=batch_size, patience=10, initial_lr=PHASE1_LR,
    )

    print(f"\n{'=' * 20} Phase 2: Fine-tune top {unfreeze_last_n} layers ({phase2_epochs} ep, lr={PHASE2_LR}) {'=' * 20}")
    model = build_deepscnn_brain_attention(
        trainable_backbone=True, unfreeze_last_n=unfreeze_last_n,
    )
    if ckpt.exists():
        model.load_weights(str(ckpt))
        print(f"Loaded Phase 1 weights: {ckpt}")
    model, h2, t2 = train_deepscnn(
        model, X_train, y_train_oh, X_val, y_val_oh, output_dirs,
        epochs=phase2_epochs, batch_size=batch_size, patience=12, initial_lr=PHASE2_LR,
    )

    if ckpt.exists():
        model.load_weights(str(ckpt))
        print(f"Loaded best checkpoint: {ckpt}")

    total_time = time.time() - total_start
    merged = {k: h1[k] + h2.get(k, []) for k in h1}

    with open(output_dirs["reports"] / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    with open(output_dirs["reports"] / "training_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "architecture": MODEL_DISPLAY_NAME,
            "pretrain": "ImageNet-ResNet50V2",
            "backbone": "ResNet50V2",
            "core": "Siamese + SE/Spatial Attention + Residual Encoder",
            "unfreeze_last_n": unfreeze_last_n,
            "phase1_epochs": phase1_epochs,
            "phase2_epochs": phase2_epochs,
            "phase1_lr": PHASE1_LR,
            "phase2_lr": PHASE2_LR,
            "phase1_time_s": t1,
            "phase2_time_s": t2,
        }, f, indent=2)

    plot_training_history(merged, MODEL_DISPLAY_NAME, output_dirs["plots"])
    metrics = evaluate_model(
        model, MODEL_DISPLAY_NAME,
        X_test, y_test, y_test_oh, total_time, output_dirs,
        use_tta=True, n_tta=8,
    )
    print(f"\n{MODEL_DISPLAY_NAME} on [{dataset}] done. Test Accuracy = {metrics['Accuracy']:.4f}")
    return model, merged, metrics


def main():
    parser = argparse.ArgumentParser(description="DeepSCNN-Brain Attention v3 (ImageNet, models/)")
    parser.add_argument("--dataset", choices=["processed", "ori"], required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS)
    parser.add_argument("--unfreeze-layers", type=int, default=UNFREEZE_LAST_N)
    parser.add_argument(
        "--phase2-only", action="store_true",
        help="跳过 Phase1，从已保存的 checkpoint 直接运行 Phase2 + 测试",
    )
    args = parser.parse_args()
    if args.phase2_only:
        run_phase2_only(args.dataset, args.batch_size, args.phase2_epochs, args.unfreeze_layers)
    else:
        run_two_phase_pipeline(
            args.dataset, args.batch_size,
            args.phase1_epochs, args.phase2_epochs, args.unfreeze_layers,
        )


if __name__ == "__main__":
    main()
