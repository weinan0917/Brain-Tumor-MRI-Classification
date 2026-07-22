import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import yaml

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATASET_PATH, resolve_project_path
from train_vit_processed_imagenet21k import (
    build_model,
    cfg_get,
    default_config_path,
    load_config,
)
from timm.data import create_transform, resolve_model_data_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ViT on the test split.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    args.data_root = resolve_project_path(
        args.data_root or cfg_get(config, "data", "root", str(PROCESSED_DATASET_PATH))
    )
    args.output_root = resolve_project_path(
        args.output_root or cfg_get(config, "train", "output_root", "models/ViT-B/outputs")
    )
    args.run_name = args.run_name or cfg_get(config, "train", "run_name", "vit_processed")
    args.batch_size = (
        int(cfg_get(config, "data", "eval_batch_size", cfg_get(config, "data", "batch_size", 8)))
        if args.batch_size is None
        else args.batch_size
    )
    args.num_workers = int(cfg_get(config, "data", "num_workers", 0))
    args.pin_memory = bool(cfg_get(config, "data", "pin_memory", True))
    args.num_classes = int(cfg_get(config, "data", "num_classes", 4))
    args.model_name = str(cfg_get(config, "model", "model_name", "vit_base_patch16_224.augreg_in21k"))
    args.img_size = int(cfg_get(config, "model", "img_size", 224))
    args.drop_rate = float(cfg_get(config, "model", "drop_rate", 0.0))
    args.drop_path_rate = float(cfg_get(config, "model", "drop_path_rate", 0.0))
    args.pretrained = False
    args.crop_pct = float(cfg_get(config, "augment", "crop_pct", 0.9))

    if args.checkpoint is None:
        args.checkpoint = args.output_root / args.run_name / "checkpoints" / "best.pt"
    else:
        args.checkpoint = resolve_project_path(args.checkpoint)
    return args


def build_eval_dataset(args, model):
    data_cfg = resolve_model_data_config(model)
    eval_tfms = create_transform(
        input_size=data_cfg.get("input_size", (3, args.img_size, args.img_size)),
        is_training=False,
        interpolation=data_cfg.get("interpolation", "bicubic"),
        mean=data_cfg.get("mean", (0.5, 0.5, 0.5)),
        std=data_cfg.get("std", (0.5, 0.5, 0.5)),
        crop_pct=data_cfg.get("crop_pct", args.crop_pct),
    )
    return ImageFolder(args.data_root / args.split, transform=eval_tfms)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    total_correct = 0
    total_seen = 0

    progress = tqdm(loader, total=len(loader), desc="Evaluate", dynamic_ncols=True)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)
        all_targets.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.float().cpu().numpy())
        progress.set_postfix(acc=f"{total_correct / total_seen:.4f}")

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    return y_true, y_pred, y_prob


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics(cm):
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = float(correct / total) if total else 0.0
    per_class = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        support = cm[idx, :].sum()
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append(
            {
                "class_idx": idx,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
        )
    supports = np.array([item["support"] for item in per_class], dtype=np.float64)
    weights = supports / supports.sum() if supports.sum() else supports
    macro_f1 = float(np.mean([item["f1"] for item in per_class]))
    weighted_f1 = float(np.sum([item["f1"] * w for item, w in zip(per_class, weights)]))
    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1}, per_class


def save_confusion_matrix(path, cm, classes):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_predictions(path, samples, y_true, y_pred, y_prob, classes):
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "true_idx", "true_label", "pred_idx", "pred_label"] + [
            f"prob_{name}" for name in classes
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample, true_idx, pred_idx, probs in zip(samples, y_true, y_pred, y_prob):
            row = {
                "path": sample[0],
                "true_idx": int(true_idx),
                "true_label": classes[int(true_idx)],
                "pred_idx": int(pred_idx),
                "pred_label": classes[int(pred_idx)],
            }
            row.update({f"prob_{name}": float(prob) for name, prob in zip(classes, probs)})
            writer.writerow(row)


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    output_dir = args.output_root / args.run_name
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(args, num_classes=args.num_classes)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    dataset = build_eval_dataset(args, model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
    )
    print(f"Dataset: {args.data_root / args.split}")
    print(f"Samples: {len(dataset)} | Classes: {dataset.classes}")
    print(f"Checkpoint: {args.checkpoint}")

    y_true, y_pred, y_prob = evaluate(
        model,
        loader,
        device,
        use_amp=(not args.no_amp) and device.type == "cuda",
    )
    cm = confusion_matrix_np(y_true, y_pred, args.num_classes)
    metrics, per_class = classification_metrics(cm)
    for item in per_class:
        item["class_name"] = dataset.classes[item["class_idx"]]

    result = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "num_samples": len(dataset),
        "classes": dataset.classes,
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = reports_dir / f"{args.split}_metrics.json"
    preds_path = reports_dir / f"{args.split}_predictions.csv"
    cm_path = plots_dir / f"{args.split}_confusion_matrix.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    save_predictions(preds_path, dataset.samples, y_true, y_pred, y_prob, dataset.classes)
    save_confusion_matrix(cm_path, cm, dataset.classes)

    print(
        f"{args.split}: acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
    )
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {preds_path}")
    print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
