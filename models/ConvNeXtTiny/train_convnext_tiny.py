import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from timm.data import create_transform, resolve_model_data_config
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATASET_PATH, resolve_project_path


def load_config(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_value(config, section, key, default=None):
    return config.get(section, {}).get(key, default)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt Tiny on the processed brain tumor dataset.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_convnext_tiny.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)

    args.data_root = resolve_project_path(
        args.data_root or get_value(config, "data", "root", str(PROCESSED_DATASET_PATH))
    )
    args.output_root = resolve_project_path(
        args.output_root or get_value(config, "train", "output_root", "models/ConvNeXtTiny/outputs")
    )
    args.run_name = args.run_name or get_value(config, "train", "run_name", "convnext_tiny_processed")
    args.num_classes = int(get_value(config, "data", "num_classes", 4))
    args.batch_size = int(get_value(config, "data", "batch_size", 16)) if args.batch_size is None else args.batch_size
    args.eval_batch_size = int(get_value(config, "data", "eval_batch_size", args.batch_size))
    args.num_workers = int(get_value(config, "data", "num_workers", 2))
    args.pin_memory = bool(get_value(config, "data", "pin_memory", True))
    args.persistent_workers = bool(get_value(config, "data", "persistent_workers", False))
    args.epochs = int(get_value(config, "train", "epochs", 30)) if args.epochs is None else args.epochs
    args.lr = float(get_value(config, "train", "lr", 1e-4)) if args.lr is None else args.lr
    args.weight_decay = float(get_value(config, "train", "weight_decay", 0.05))
    args.warmup_epochs = int(get_value(config, "train", "warmup_epochs", 3))
    args.min_lr = float(get_value(config, "train", "min_lr", 1e-6))
    args.label_smoothing = float(get_value(config, "train", "label_smoothing", 0.1))
    args.amp = bool(get_value(config, "train", "amp", True))
    args.seed = int(get_value(config, "train", "seed", 42))
    args.eval_test = False
    args.model_name = str(get_value(config, "model", "name", "convnext_tiny"))
    args.local_pretrained = bool(get_value(config, "model", "local_pretrained", True))
    args.pretrained_path = resolve_project_path(get_value(config, "model", "pretrained_path", ""))
    args.pretrained = False
    args.img_size = int(get_value(config, "model", "img_size", 224))
    args.drop_rate = float(get_value(config, "model", "drop_rate", 0.0))
    args.drop_path_rate = float(get_value(config, "model", "drop_path_rate", 0.1))
    args.train_scale = tuple(get_value(config, "augment", "train_scale", [0.8, 1.0]))
    args.hflip = float(get_value(config, "augment", "horizontal_flip_probability", 0.5))
    args.vflip = float(get_value(config, "augment", "vertical_flip_probability", 0.0))
    args.random_erasing = float(get_value(config, "augment", "random_erasing_probability", 0.0))
    args.color_jitter = float(get_value(config, "augment", "color_jitter", 0.0))
    args.crop_pct = float(get_value(config, "augment", "crop_pct", 0.875))
    args.resume = args.resume
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dirs(root, run_name):
    run_dir = root / run_name
    result = {name: run_dir / name for name in ("logs", "plots", "checkpoints", "reports")}
    result["root"] = run_dir
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def build_datasets(args, model):
    data_config = resolve_model_data_config(model)
    input_size = data_config.get("input_size", (3, args.img_size, args.img_size))
    mean = data_config.get("mean", (0.485, 0.456, 0.406))
    std = data_config.get("std", (0.229, 0.224, 0.225))
    interpolation = data_config.get("interpolation", "bicubic")
    train_transform = create_transform(
        input_size=input_size, is_training=True, scale=args.train_scale,
        hflip=args.hflip, vflip=args.vflip, color_jitter=args.color_jitter,
        auto_augment=None, interpolation=interpolation, mean=mean, std=std,
        re_prob=args.random_erasing,
    )
    eval_transform = create_transform(
        input_size=input_size, is_training=False, interpolation=interpolation,
        mean=mean, std=std, crop_pct=args.crop_pct,
    )
    return (
        ImageFolder(args.data_root / "train", transform=train_transform),
        ImageFolder(args.data_root / "val", transform=eval_transform),
    )


def build_loaders(args, datasets):
    train_set, val_set = datasets
    common = {"num_workers": args.num_workers, "pin_memory": args.pin_memory and torch.cuda.is_available()}
    if args.num_workers > 0 and args.persistent_workers:
        common["persistent_workers"] = True
    return (
        DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **common),
        DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, **common),
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp_enabled=False, desc="Train", lr_callback=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_count = 0
    progress = tqdm(loader, desc=desc, leave=True, unit="batch", dynamic_ncols=True)
    for step, (images, labels) in enumerate(progress):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            if lr_callback is not None:
                lr_callback(step)
            optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.detach().item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        progress.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )
    return total_loss / total_count, total_correct / total_count


def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def load_local_convnext_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    converted = {}

    for key, value in source.items():
        if key in {"head.weight", "head.bias"}:
            continue
        if key == "norm.weight":
            target_key = "head.norm.weight"
        elif key == "norm.bias":
            target_key = "head.norm.bias"
        elif key.startswith("downsample_layers.0."):
            target_key = key.replace("downsample_layers.0.", "stem.", 1)
        elif key.startswith("downsample_layers."):
            parts = key.split(".")
            stage = parts[1]
            suffix = ".".join(parts[2:])
            target_key = f"stages.{stage}.downsample.{suffix}"
        elif key.startswith("stages."):
            parts = key.split(".")
            stage, block = parts[1], parts[2]
            suffix = ".".join(parts[3:])
            if "." in suffix:
                prefix, parameter = suffix.split(".", 1)
                prefix = {"dwconv": "conv_dw", "pwconv1": "mlp.fc1", "pwconv2": "mlp.fc2"}.get(prefix, prefix)
                suffix = f"{prefix}.{parameter}"
            target_key = f"stages.{stage}.blocks.{block}.{suffix}"
        else:
            continue
        if target_key not in target:
            continue
        if value.shape == target[target_key].shape:
            converted[target_key] = value

    missing, unexpected = model.load_state_dict(converted, strict=False)
    loaded = len(converted)
    print(f"Loaded local pretrained weights: {checkpoint_path}")
    print(f"Loaded tensors: {loaded} | missing tensors: {len(missing)} | unexpected tensors: {len(unexpected)}")
    if missing:
        print(f"Missing examples: {missing[:5]}")


def current_lr(args, epoch, step, steps_per_epoch):
    progress = epoch + step / max(steps_per_epoch, 1)
    if args.warmup_epochs > 0 and progress < args.warmup_epochs:
        return args.lr * progress / args.warmup_epochs
    cosine_progress = (progress - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
    cosine_progress = min(max(cosine_progress, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * cosine_progress))


def save_history(history, paths):
    with (paths["reports"] / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (paths["logs"] / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [item["train_acc"] for item in history], label="train")
    axes[1].plot(epochs, [item["val_acc"] for item in history], label="val")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(paths["plots"] / "training_curves.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = make_dirs(args.output_root, args.run_name)
    model = timm.create_model(
        args.model_name, pretrained=False, num_classes=args.num_classes,
        drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate,
    ).to(device)
    if args.local_pretrained:
        if not args.pretrained_path.is_file():
            raise FileNotFoundError(f"Local pretrained checkpoint not found: {args.pretrained_path}")
        load_local_convnext_weights(model, args.pretrained_path)
    datasets = build_datasets(args, model)
    train_loader, val_loader = build_loaders(args, datasets)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    amp_enabled = args.amp and device.type == "cuda"
    best_acc = -1.0
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_acc = float(checkpoint.get("best_acc", -1.0))
    history = []
    with (paths["reports"] / "runtime_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    print(f"Device: {device}")
    print(f"Model: {args.model_name} | local_pretrained={args.local_pretrained} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset: train={len(datasets[0])} val={len(datasets[1])}")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_enabled,
            f"Epoch {epoch + 1:03d}/{args.epochs} Train",
            lr_callback=lambda step: set_learning_rate(
                optimizer, current_lr(args, epoch, step, len(train_loader))
            ),
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, amp_enabled=amp_enabled, desc=f"Epoch {epoch + 1:03d}/{args.epochs} Val")
        lr = optimizer.param_groups[0]["lr"]
        record = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": lr}
        history.append(record)
        save_history(history, paths)
        checkpoint = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_acc": best_acc, "model_name": args.model_name, "num_classes": args.num_classes}
        torch.save(checkpoint, paths["checkpoints"] / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, paths["checkpoints"] / "best.pt")
            print(f"Saved new best checkpoint: val_acc={best_acc:.4f}")
        print(f"Epoch {epoch + 1:03d}/{args.epochs}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={lr:.6g}")


if __name__ == "__main__":
    main()
