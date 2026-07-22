import argparse
import csv
import gc
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import yaml

import timm
from timm.data import create_transform, resolve_model_data_config

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATASET_PATH, resolve_project_path


def default_config_path():
    return Path(__file__).resolve().with_name("config_vit_processed.yaml")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cfg_get(config, section, key, default=None):
    return config.get(section, {}).get(key, default)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT from scratch on data/processed_data.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    args.data_root = resolve_project_path(
        args.data_root or cfg_get(config, "data", "root", str(PROCESSED_DATASET_PATH))
    )
    args.num_classes = int(cfg_get(config, "data", "num_classes", 4))
    args.output_root = resolve_project_path(
        args.output_root or cfg_get(config, "train", "output_root", "models/ViT-B/outputs")
    )
    args.run_name = args.run_name or (
        str(cfg_get(config, "train", "run_name", "vit_processed")) + "_no_imagenet21k"
    )
    args.epochs = int(cfg_get(config, "train", "epochs", 30)) if args.epochs is None else args.epochs
    args.batch_size = (
        int(cfg_get(config, "data", "batch_size", 16))
        if args.batch_size is None
        else args.batch_size
    )
    args.eval_batch_size = int(cfg_get(config, "data", "eval_batch_size", args.batch_size))
    args.lr = float(cfg_get(config, "train", "lr", 3e-5)) if args.lr is None else args.lr
    args.weight_decay = float(cfg_get(config, "train", "weight_decay", 0.05))
    args.warmup_epochs = int(cfg_get(config, "train", "warmup_epochs", 3))
    args.min_lr = float(cfg_get(config, "train", "min_lr", 1e-6))
    args.label_smoothing = float(cfg_get(config, "train", "label_smoothing", 0.1))
    args.seed = int(cfg_get(config, "train", "seed", 42))
    args.amp = bool(cfg_get(config, "train", "amp", True))
    args.eval_test = bool(cfg_get(config, "train", "eval_test", True))
    args.num_workers = int(cfg_get(config, "data", "num_workers", 4))
    args.pin_memory = bool(cfg_get(config, "data", "pin_memory", True))
    args.persistent_workers = bool(cfg_get(config, "data", "persistent_workers", False))

    args.model_name = str(cfg_get(config, "model", "model_name", "vit_base_patch16_224.augreg_in21k"))
    args.pretrained = False
    args.img_size = int(cfg_get(config, "model", "img_size", 224))
    args.drop_rate = float(cfg_get(config, "model", "drop_rate", 0.1))
    args.drop_path_rate = float(cfg_get(config, "model", "drop_path_rate", 0.1))

    args.train_scale = tuple(cfg_get(config, "augment", "train_scale", [0.8, 1.0]))
    args.train_hflip = float(cfg_get(config, "augment", "horizontal_flip_probability", 0.5))
    args.train_vflip = float(cfg_get(config, "augment", "vertical_flip_probability", 0.0))
    args.random_erasing = float(cfg_get(config, "augment", "random_erasing_probability", 0.0))
    args.color_jitter = float(cfg_get(config, "augment", "color_jitter", 0.0))
    args.crop_pct = float(cfg_get(config, "augment", "crop_pct", 0.9))

    args.resume = resolve_project_path(args.resume) if args.resume else None
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_run_dirs(output_root, run_name):
    run_dir = output_root / run_name
    dirs = {
        "root": run_dir,
        "logs": run_dir / "logs",
        "plots": run_dir / "plots",
        "checkpoints": run_dir / "checkpoints",
        "reports": run_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_datasets(args, model):
    data_cfg = resolve_model_data_config(model)
    input_size = data_cfg.get("input_size", (3, args.img_size, args.img_size))
    mean = data_cfg.get("mean", (0.5, 0.5, 0.5))
    std = data_cfg.get("std", (0.5, 0.5, 0.5))
    interpolation = data_cfg.get("interpolation", "bicubic")
    crop_pct = data_cfg.get("crop_pct", args.crop_pct)

    train_tfms = create_transform(
        input_size=input_size,
        is_training=True,
        scale=args.train_scale,
        hflip=args.train_hflip,
        vflip=args.train_vflip,
        color_jitter=args.color_jitter,
        auto_augment=None,
        interpolation=interpolation,
        mean=mean,
        std=std,
        re_prob=args.random_erasing,
    )
    eval_tfms = create_transform(
        input_size=input_size,
        is_training=False,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
    )

    train_set = ImageFolder(args.data_root / "train", transform=train_tfms)
    val_set = ImageFolder(args.data_root / "val", transform=eval_tfms)
    test_set = ImageFolder(args.data_root / "test", transform=eval_tfms)
    return train_set, val_set, test_set


def build_loaders(args, train_set, val_set, test_set):
    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
    )
    eval_loader_kwargs = dict(
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
    )
    if args.num_workers > 0 and args.persistent_workers:
        train_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_set, shuffle=True, **train_loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **eval_loader_kwargs)
    return train_loader, val_loader, test_loader


def build_model(args, num_classes):
    model = timm.create_model(
        args.model_name,
        pretrained=args.pretrained,
        num_classes=num_classes,
        img_size=args.img_size,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    )
    return model


def build_optimizer(model, args):
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    warmup_epochs = max(0, args.warmup_epochs)
    total_epochs = max(1, args.epochs)
    min_lr_ratio = 0.0 if args.lr <= 0 else min(max(args.min_lr / args.lr, 0.0), 1.0)

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        decay_epochs = max(1, total_epochs - warmup_epochs)
        progress = float(epoch - warmup_epochs + 1) / float(decay_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).sum().item(), labels.size(0)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Train {epoch + 1:03d}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_loss = loss.item() * images.size(0)
        correct, seen = accuracy_from_logits(logits, labels)
        total_loss += batch_loss
        total_correct += correct
        total_seen += seen
        progress.set_postfix(
            loss=f"{total_loss / total_seen:.4f}",
            acc=f"{total_correct / total_seen:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
    progress.close()
    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, epoch, total_epochs, split_name):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"{split_name} {epoch + 1:03d}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        batch_loss = loss.item() * images.size(0)
        correct, seen = accuracy_from_logits(logits, labels)
        total_loss += batch_loss
        total_correct += correct
        total_seen += seen
        progress.set_postfix(
            loss=f"{total_loss / total_seen:.4f}",
            acc=f"{total_correct / total_seen:.4f}",
        )
    progress.close()
    return total_loss / total_seen, total_correct / total_seen


def append_metrics_csv(path, row):
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_curves(history, output_path):
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [item["train_acc"] for item in history], label="train")
    axes[1].plot(epochs, [item["val_acc"] for item in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "model_args": {
                "num_classes": model.num_classes if hasattr(model, "num_classes") else None,
                "img_size": args.img_size,
                "drop_rate": args.drop_rate,
                "drop_path_rate": args.drop_path_rate,
            },
        },
        path,
    )


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_config_snapshot(path, args, config):
    snapshot = {
        "data": {
            "root": str(args.data_root),
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.persistent_workers,
        },
        "model": {
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "img_size": args.img_size,
            "drop_rate": args.drop_rate,
            "drop_path_rate": args.drop_path_rate,
        },
        "train": {
            "run_name": args.run_name,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
            "label_smoothing": args.label_smoothing,
            "amp": args.amp,
            "seed": args.seed,
            "eval_test": args.eval_test,
        },
        "augment": config.get("augment", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    run_dirs = build_run_dirs(args.output_root, args.run_name)
    save_config_snapshot(run_dirs["reports"] / "config_used.yaml", args, config)
    save_json(
        run_dirs["reports"] / "runtime_args.json",
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(args, num_classes=args.num_classes)
    train_set, val_set, test_set = build_datasets(args, model)
    train_loader, val_loader, test_loader = build_loaders(args, train_set, val_set, test_set)

    print(f"Dataset root: {args.data_root}")
    print(f"Split sizes: train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print(f"Classes: {train_set.classes}")
    print(f"Model: {args.model_name} | pretrained={args.pretrained} | params={model.num_params() if hasattr(model, 'num_params') else sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.amp and device.type == "cuda",
            epoch,
            args.epochs,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.amp and device.type == "cuda",
            epoch,
            args.epochs,
            "Val",
        )
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        }
        history.append(row)
        append_metrics_csv(run_dirs["logs"] / "metrics.csv", row)
        plot_curves(history, run_dirs["plots"] / "training_curves.png")

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(run_dirs["checkpoints"] / "last.pt", model, optimizer, scheduler, epoch, best_acc, args)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(run_dirs["checkpoints"] / "best.pt", model, optimizer, scheduler, epoch, best_acc, args)
            print(f"Saved new best checkpoint: val_acc={best_acc:.4f}")

    if history:
        summary = {
            "best_val_acc": max(item["val_acc"] for item in history),
            "best_epoch": int(np.argmax([item["val_acc"] for item in history])) + 1,
            "final_train_acc": history[-1]["train_acc"],
            "final_val_acc": history[-1]["val_acc"],
            "epochs": len(history),
        }
    else:
        summary = {"best_val_acc": 0.0, "best_epoch": 0, "final_train_acc": 0.0, "final_val_acc": 0.0, "epochs": 0}

    save_json(run_dirs["reports"] / "training_history.json", {"history": history, "summary": summary})

    if args.eval_test and (run_dirs["checkpoints"] / "best.pt").exists():
        del train_loader
        del val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        best_ckpt = torch.load(run_dirs["checkpoints"] / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model"])
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            args.amp and device.type == "cuda",
            args.epochs - 1,
            args.epochs,
            "Test",
        )
        save_json(
            run_dirs["reports"] / "test_metrics.json",
            {"test_loss": test_loss, "test_acc": test_acc, "best_val_acc": best_acc},
        )
        print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    print(f"Done. Outputs saved to: {run_dirs['root']}")


if __name__ == "__main__":
    main()
