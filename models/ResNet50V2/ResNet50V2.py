import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import NUM_CLASSES, TARGET_SIZE, setup_environment

setup_environment()
import keras
from keras import layers, models, regularizers

from models.common import run_two_phase_transfer

MODEL_NAME = "ResNet50V2"
MODEL_DISPLAY_NAME = "ResNet50V2"


def build_resnet50v2(input_shape=None, num_classes=None, trainable_backbone=False, unfreeze_last_n=0):
    input_shape = input_shape or (*TARGET_SIZE, 3)
    num_classes = num_classes or NUM_CLASSES

    base = keras.applications.ResNet50V2(
        weights="imagenet", include_top=False, input_shape=input_shape,
    )
    base.trainable = trainable_backbone
    if trainable_backbone and unfreeze_last_n > 0:
        for layer in base.layers[:-unfreeze_last_n]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = base(x, training=trainable_backbone)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return models.Model(inputs=inputs, outputs=outputs, name=MODEL_DISPLAY_NAME)


def main():
    parser = argparse.ArgumentParser(description="ResNet50V2")
    parser.add_argument("--dataset", choices=["processed", "ori"], required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase1-epochs", type=int, default=12)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--unfreeze-layers", type=int, default=40)
    args = parser.parse_args()

    run_two_phase_transfer(
        build_fn=build_resnet50v2,
        model_name=MODEL_NAME,
        model_display_name=MODEL_DISPLAY_NAME,
        dataset=args.dataset,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        unfreeze_last_n=args.unfreeze_layers,
    )


if __name__ == "__main__":
    main()
