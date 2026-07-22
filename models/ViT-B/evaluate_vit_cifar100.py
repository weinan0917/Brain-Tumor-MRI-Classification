import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

import timm
from timm.data import resolve_model_data_config
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import resolve_project_path


def default_config_path():
    return Path(__file__).resolve().with_name("config_vit_cifar100_no_imagenet21k.yaml")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cfg_get(config, section, key, default=None):
    return config.get(section, {}).get(key, default)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a no-pretrain ViT checkpoint on CIFAR-100."
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    args.data_root = resolve_project_path(
        args.data_root or cfg_get(config, "data", "root", "data/cifar-100-python-train450-val50")
    )
    args.output_root = resolve_project_path(
        args.output_root or cfg_get(config, "train", "output_root", "models/ViT-B/outputs")
    )
    args.run_name = args.run_name or cfg_get(
        config, "train", "run_name", "vit_base_patch16_224_no_imagenet21k_cifar100_224"
    )
    args.batch_size = (
        int(cfg_get(config, "data", "eval_batch_size", cfg_get(config, "data", "batch_size", 16)))
        if args.batch_size is None
        else args.batch_size
    )
    args.num_workers = int(cfg_get(config, "data", "num_workers", 0))
    args.pin_memory = bool(cfg_get(config, "data", "pin_memory", True))
    args.num_classes = int(cfg_get(config, "data", "num_classes", 100))
    args.model_name = str(cfg_get(config, "model", "model_name", "vit_base_patch16_224.augreg_in21k"))
    args.img_size = int(cfg_get(config, "model", "img_size", 224))
    args.drop_rate = float(cfg_get(config, "model", "drop_rate", 0.1))
    args.drop_path_rate = float(cfg_get(config, "model", "drop_path_rate", 0.1))
    args.crop_pct = float(cfg_get(config, "augment", "crop_pct", 1.0))

    if args.checkpoint is None:
        args.checkpoint = args.output_root / args.run_name / "checkpoints" / "best.pt"
    return args


class Cifar100PickleDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        path = self.root / split
        if not path.exists():
            raise FileNotFoundError(f"Could not find CIFAR-100 split file: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")

        self.data = obj["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = [int(label) for label in obj["fine_labels"]]
        self.classes = self._load_classes()

    def _load_classes(self):
        meta_path = self.root / "meta"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f, encoding="latin1")
            names = meta.get("fine_label_names")
            if names:
                return list(names)
        return [f"class_{idx}" for idx in range(100)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, index


def build_model(args):
    return timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=args.num_classes,
        img_size=args.img_size,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    )


def build_eval_dataset(args, model):
    data_cfg = resolve_model_data_config(model)
    input_size = data_cfg.get("input_size", (3, args.img_size, args.img_size))
    mean = data_cfg.get("mean", (0.5, 0.5, 0.5))
    std = data_cfg.get("std", (0.5, 0.5, 0.5))
    image_size = input_size[-1]
    eval_tfms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return Cifar100PickleDataset(args.data_root, args.split, transform=eval_tfms)


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return ckpt


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_indices = []
    all_targets = []
    all_preds = []

    progress = tqdm(loader, total=len(loader), desc="Evaluate", dynamic_ncols=True)
    for images, labels, indices in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)
        all_indices.append(indices.numpy())
        all_targets.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        progress.set_postfix(
            loss=f"{total_loss / total_seen:.4f}",
            acc=f"{total_correct / total_seen:.4f}",
        )

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "indices": np.concatenate(all_indices),
        "targets": np.concatenate(all_targets),
        "preds": np.concatenate(all_preds),
    }


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        cm[int(true_idx), int(pred_idx)] += 1
    return cm


def per_class_metrics(cm, classes):
    rows = []
    for idx, class_name in enumerate(classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        support = cm[idx, :].sum()
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "class_idx": idx,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
        )
    return rows


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    output_dir = args.output_root / args.run_name
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(args)
    ckpt = load_checkpoint(model, args.checkpoint)
    model.to(device)

    dataset = build_eval_dataset(args, model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Dataset root: {args.data_root}")
    print(f"Split: {args.split} | samples={len(dataset)} | classes={len(dataset.classes)}")
    print(f"Checkpoint: {args.checkpoint}")

    result = evaluate(
        model,
        loader,
        criterion,
        device,
        use_amp=(not args.no_amp) and device.type == "cuda",
    )
    cm = confusion_matrix_np(result["targets"], result["preds"], args.num_classes)
    class_rows = per_class_metrics(cm, dataset.classes)
    macro_f1 = float(np.mean([row["f1"] for row in class_rows]))

    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "checkpoint_best_acc": ckpt.get("best_acc") if isinstance(ckpt, dict) else None,
        "num_samples": len(dataset),
        "num_classes": args.num_classes,
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "macro_f1": macro_f1,
    }

    metrics_path = reports_dir / f"cifar100_{args.split}_metrics.json"
    per_class_path = reports_dir / f"cifar100_{args.split}_per_class.csv"
    preds_path = reports_dir / f"cifar100_{args.split}_predictions.csv"
    cm_path = reports_dir / f"cifar100_{args.split}_confusion_matrix.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "classes": dataset.classes,
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    save_csv(
        per_class_path,
        ["class_idx", "class_name", "precision", "recall", "f1", "support"],
        class_rows,
    )
    save_csv(
        preds_path,
        ["index", "true_idx", "true_label", "pred_idx", "pred_label", "correct"],
        [
            {
                "index": int(index),
                "true_idx": int(true_idx),
                "true_label": dataset.classes[int(true_idx)],
                "pred_idx": int(pred_idx),
                "pred_label": dataset.classes[int(pred_idx)],
                "correct": int(true_idx == pred_idx),
            }
            for index, true_idx, pred_idx in zip(
                result["indices"], result["targets"], result["preds"]
            )
        ],
    )
    np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

    print(
        f"{args.split}: loss={summary['loss']:.4f} "
        f"acc={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f}"
    )
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved per-class metrics: {per_class_path}")
    print(f"Saved predictions: {preds_path}")
    print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
