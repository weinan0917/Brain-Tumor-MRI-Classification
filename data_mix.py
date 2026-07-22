#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import imagehash
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ORI_DATA_DIR = ROOT / "data" / "ori_data"
DATASET_1 = ORI_DATA_DIR / "Brain Tumor MRI Dataset"
DATASET_2 = ORI_DATA_DIR / "BRISC 2025 Dataset"
BRAIN_TUMOR_DIR = ORI_DATA_DIR / "BrainTumorDataPublic"
BRAIN_TUMOR_CACHE = ORI_DATA_DIR / "_expanded" / "BrainTumorDataPublic"
NINS_DIR = ORI_DATA_DIR / "NINS_Dataset"
OUTPUT_DIR = ROOT / "data" / "mixed_data"
HAMMING_THRESHOLD = 3


def to_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()

LABEL_ALIASES = {
    "no_tumor": "notumor",
    "no-tumor": "notumor",
    "notumor": "notumor",
    "normal": "notumor",
    "healthy": "notumor",
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
}

TUMOR_CODES = {
    "glioma": "gl",
    "meningioma": "me",
    "pituitary": "pi",
    "notumor": "no",
}

BRAIN_TUMOR_LABELS = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}


@dataclass
class ImageRecord:
    source_path: Path
    source_dataset: str
    original_split: str
    tumor_label: str
    sha256: str = ""
    phash: imagehash.ImageHash | None = None
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    kept: bool = True
    duplicate_of: str | None = None
    duplicate_reason: str | None = None
    output_filename: str = ""
    output_relative_path: str = ""


def normalize_label(raw_label: str) -> str:
    key = raw_label.strip().lower().replace(" ", "_")
    if key not in LABEL_ALIASES:
        raise ValueError(f"Unknown label: {raw_label!r}")
    return LABEL_ALIASES[key]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_image_features(path: Path) -> tuple[str, imagehash.ImageHash, int, int, int]:
    file_size = path.stat().st_size
    digest = sha256_file(path)
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        ph = imagehash.phash(rgb)
    return digest, ph, width, height, file_size


def int16_to_pil_image(image: np.ndarray) -> Image.Image:
    image = image.astype(np.float32)
    vmin, vmax = float(image.min()), float(image.max())
    if vmax > vmin:
        normalized = (image - vmin) / (vmax - vmin) * 255.0
    else:
        normalized = np.zeros_like(image, dtype=np.float32)
    return Image.fromarray(normalized.astype(np.uint8), mode="L")


def export_braintumor_images(force: bool = False) -> Path:
    if not BRAIN_TUMOR_DIR.exists():
        raise FileNotFoundError(f"Missing source dataset: {BRAIN_TUMOR_DIR}")

    BRAIN_TUMOR_CACHE.mkdir(parents=True, exist_ok=True)
    mat_files = sorted(BRAIN_TUMOR_DIR.glob("*.mat"), key=lambda path: int(path.stem))
    exported = 0
    skipped = 0

    for mat_path in mat_files:
        with h5py.File(mat_path, "r") as handle:
            image = handle["cjdata/image"][()]
            label_value = int(handle["cjdata/label"][()][0, 0])

        tumor_label = BRAIN_TUMOR_LABELS[label_value]
        class_dir = BRAIN_TUMOR_CACHE / tumor_label
        class_dir.mkdir(parents=True, exist_ok=True)
        output_path = class_dir / f"braintumor_{mat_path.stem}.jpg"

        if (
            not force
            and output_path.exists()
            and output_path.stat().st_mtime >= mat_path.stat().st_mtime
        ):
            skipped += 1
            continue

        pil_image = int16_to_pil_image(image)
        pil_image.save(output_path, format="JPEG", quality=95)
        exported += 1

    print(
        f"BrainTumorDataPublic: {len(mat_files)} .mat files, "
        f"exported {exported}, skipped {skipped}, cache at {BRAIN_TUMOR_CACHE}"
    )
    return BRAIN_TUMOR_CACHE


def collect_dataset_1() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    split_map = {"Training": "train", "Testing": "test"}
    for folder_name, split_name in split_map.items():
        split_dir = DATASET_1 / folder_name
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = normalize_label(class_dir.name)
            for image_path in sorted(class_dir.glob("*.jpg")):
                records.append(
                    ImageRecord(
                        source_path=image_path,
                        source_dataset="1",
                        original_split=split_name,
                        tumor_label=label,
                    )
                )
    return records


def collect_dataset_2() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    manifest_path = DATASET_2 / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        for item in manifest:
            rel_path = item["relative_path"].replace("\\", "/")
            image_path = DATASET_2 / rel_path
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image referenced in manifest: {image_path}")
            records.append(
                ImageRecord(
                    source_path=image_path,
                    source_dataset="2",
                    original_split=item["split"],
                    tumor_label=normalize_label(item["tumor_label"]),
                )
            )
        return records

    task_root = DATASET_2 / "classification_task"
    for split_name in ("train", "test"):
        split_dir = task_root / split_name
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = normalize_label(class_dir.name)
            for image_path in sorted(class_dir.glob("*.jpg")):
                records.append(
                    ImageRecord(
                        source_path=image_path,
                        source_dataset="2",
                        original_split=split_name,
                        tumor_label=label,
                    )
                )
    return records


def collect_braintumor() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for class_dir in sorted(BRAIN_TUMOR_CACHE.iterdir()):
        if not class_dir.is_dir():
            continue
        label = normalize_label(class_dir.name)
        for image_path in sorted(class_dir.glob("*.jpg")):
            records.append(
                ImageRecord(
                    source_path=image_path,
                    source_dataset="BrainTumor",
                    original_split="all",
                    tumor_label=label,
                )
            )
    return records


def collect_nins() -> list[ImageRecord]:
    if not NINS_DIR.exists():
        raise FileNotFoundError(f"Missing source dataset: {NINS_DIR}")

    records: list[ImageRecord] = []
    for class_dir in sorted(NINS_DIR.iterdir()):
        if not class_dir.is_dir() or class_dir.name == "models":
            continue
        label = normalize_label(class_dir.name)
        for image_path in sorted(class_dir.glob("*.jpg")):
            records.append(
                ImageRecord(
                    source_path=image_path,
                    source_dataset="NINS",
                    original_split="all",
                    tumor_label=label,
                )
            )
    return records


def deduplicate_records(records: list[ImageRecord]) -> Counter:
    stats: Counter = Counter()
    sha_seen: dict[str, ImageRecord] = {}
    kept_hashes: list[imagehash.ImageHash] = []
    kept_records: list[ImageRecord] = []

    for record in records:
        digest, ph, width, height, file_size = compute_image_features(record.source_path)
        record.sha256 = digest
        record.phash = ph
        record.width = width
        record.height = height
        record.file_size_bytes = file_size

        if digest in sha_seen:
            representative = sha_seen[digest]
            record.kept = False
            record.duplicate_of = to_relative(representative.source_path)
            record.duplicate_reason = "sha256_exact"
            stats["removed_sha256"] += 1
            continue

        duplicate_index = next(
            (index for index, kept_hash in enumerate(kept_hashes) if ph - kept_hash <= HAMMING_THRESHOLD),
            None,
        )
        if duplicate_index is not None:
            record.kept = False
            record.duplicate_of = to_relative(kept_records[duplicate_index].source_path)
            record.duplicate_reason = "phash_near"
            stats["removed_phash"] += 1
            continue

        sha_seen[digest] = record
        kept_hashes.append(ph)
        kept_records.append(record)
        stats["kept"] += 1

    return stats


def write_output(records: list[ImageRecord]) -> list[ImageRecord]:
    for label in TUMOR_CODES:
        class_dir = OUTPUT_DIR / label
        if class_dir.exists():
            shutil.rmtree(class_dir)

    kept_records = [record for record in records if record.kept]
    for index, record in enumerate(kept_records, start=1):
        class_dir = OUTPUT_DIR / record.tumor_label
        class_dir.mkdir(parents=True, exist_ok=True)
        code = TUMOR_CODES[record.tumor_label]
        filename = f"mixed_{index:05d}_{code}.jpg"
        output_path = class_dir / filename
        shutil.copy2(record.source_path, output_path)
        record.output_filename = filename
        record.output_relative_path = f"{record.tumor_label}/{filename}"
    return kept_records


def build_manifest_rows(kept_records: list[ImageRecord]) -> list[dict]:
    rows: list[dict] = []
    for index, record in enumerate(kept_records, start=1):
        rows.append(
            {
                "relative_path": record.output_relative_path.replace("/", "\\"),
                "filename": record.output_filename,
                "task": "classification",
                "split": "all",
                "index": index,
                "tumor_code": TUMOR_CODES[record.tumor_label],
                "tumor_label": record.tumor_label,
                "plane_code": "",
                "plane_label": "",
                "sequence": "",
                "is_mask": False,
                "linked_image": None,
                "width": record.width,
                "height": record.height,
                "file_size_bytes": record.file_size_bytes,
                "sha256": record.sha256,
                "source_dataset": record.source_dataset,
                "original_split": record.original_split,
                "source_path": to_relative(record.source_path),
            }
        )
    return rows


def save_manifest(rows: list[dict]) -> None:
    manifest_json = OUTPUT_DIR / "manifest.json"
    manifest_csv = OUTPUT_DIR / "manifest.csv"

    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[ImageRecord], dedup_stats: Counter) -> dict:
    total = len(records)
    kept = dedup_stats["kept"]
    removed = total - kept
    kept_by_label = Counter(record.tumor_label for record in records if record.kept)
    removed_by_label = Counter(record.tumor_label for record in records if not record.kept)
    removed_by_source = Counter(record.source_dataset for record in records if not record.kept)
    kept_by_source = Counter(record.source_dataset for record in records if record.kept)
    source_totals = Counter(record.source_dataset for record in records)

    return {
        "total": total,
        "kept": kept,
        "removed": removed,
        "removed_sha256": dedup_stats["removed_sha256"],
        "removed_phash": dedup_stats["removed_phash"],
        "kept_by_label": dict(kept_by_label),
        "removed_by_label": dict(removed_by_label),
        "kept_by_source": dict(kept_by_source),
        "removed_by_source": dict(removed_by_source),
        "source_totals": dict(source_totals),
    }


def write_report(summary: dict) -> Path:
    report_path = OUTPUT_DIR / "DATA_MIX_REPORT.md"
    kept_pct = summary["kept"] / summary["total"] * 100 if summary["total"] else 0
    removed_pct = summary["removed"] / summary["total"] * 100 if summary["total"] else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Dataset Mix & Deduplication Report",
        "",
        f"> Generated: {now}",
        f"> Output: `{to_relative(OUTPUT_DIR)}`",
        "",
        "## 1. Overview",
        "",
        "This report documents the merge of four source datasets into a unified classification dataset at `data/mixed_data`:",
        "",
        "- `data/ori_data/Brain Tumor MRI Dataset`",
        "- `data/ori_data/BRISC 2025 Dataset`",
        "- `data/ori_data/BrainTumorDataPublic` (expanded from `.mat` to `.jpg`)",
        "- `data/ori_data/NINS_Dataset`",
        "",
        "Processing steps:",
        "",
        "1. Expand `BrainTumorDataPublic` `.mat` slices to JPG images under `data/ori_data/_expanded/BrainTumorDataPublic/`.",
        "2. Merge all four datasets into one image pool.",
        "3. Combine original train/test splits into a single unified collection.",
        "4. Standardize labels (`no_tumor`, `normal`, `healthy` -> `notumor`).",
        "5. Unify directory layout under `{label}/`.",
        "6. Remove exact duplicates via SHA-256.",
        "7. Remove near-duplicates via perceptual hash (pHash) with Hamming distance <= 3.",
        "",
        "## 2. Summary Statistics",
        "",
        "| Metric | Count | Percentage |",
        "|--------|------:|-----------:|",
        f"| Total input images | {summary['total']} | 100.00% |",
        f"| Kept after deduplication | {summary['kept']} | {kept_pct:.2f}% |",
        f"| Removed (total) | {summary['removed']} | {removed_pct:.2f}% |",
        f"| Removed (SHA-256 exact) | {summary['removed_sha256']} | |",
        f"| Removed (pHash near-duplicate) | {summary['removed_phash']} | |",
        "",
        "## 3. Source Dataset Breakdown",
        "",
        "| Source | Input | Kept | Removed |",
        "|--------|------:|-----:|--------:|",
    ]

    for source in sorted(summary["source_totals"]):
        input_count = summary["source_totals"][source]
        kept_count = summary["kept_by_source"].get(source, 0)
        removed_count = summary["removed_by_source"].get(source, 0)
        lines.append(f"| {source} | {input_count} | {kept_count} | {removed_count} |")

    lines.extend(
        [
            "",
            "## 4. Class Distribution (Kept)",
            "",
            "| Class | Kept | Removed |",
            "|-------|-----:|--------:|",
        ]
    )

    all_labels = sorted(set(summary["kept_by_label"]) | set(summary["removed_by_label"]))
    for label in all_labels:
        kept_count = summary["kept_by_label"].get(label, 0)
        removed_count = summary["removed_by_label"].get(label, 0)
        lines.append(f"| {label} | {kept_count} | {removed_count} |")

    lines.extend(
        [
            "",
            "## 5. BrainTumor Expansion",
            "",
            "- Source: `data/ori_data/BrainTumorDataPublic/*.mat`",
            "- Cache: `data/ori_data/_expanded/BrainTumorDataPublic/{glioma,meningioma,pituitary}/`",
            "- Label mapping: 1 -> meningioma, 2 -> glioma, 3 -> pituitary",
            "- Conversion: int16 MRI slice min-max normalized to 8-bit grayscale JPG (512x512)",
            "",
            "## 6. Deduplication Methodology",
            "",
            "### Step 1: Perceptual Hashing (pHash)",
            "",
            "Uses `imagehash.phash` to capture visual content rather than raw bytes.",
            "",
            "### Step 2: Hamming Distance",
            "",
            "- Distance = 0: exact duplicate (also checked via SHA-256)",
            "- Distance = 1-3: near duplicate",
            "- Distance > 3: distinct image",
            "",
            "### Step 3: Duplicate Removal",
            "",
            "Images are processed in source order: dataset 1, dataset 2, BrainTumor, NINS.",
            "The first image in each duplicate cluster is retained; subsequent matches are removed.",
            "",
            "## 7. Output Layout",
            "",
            "```",
            "data/mixed_data/",
            "├── glioma/",
            "├── meningioma/",
            "├── notumor/",
            "├── pituitary/",
            "├── manifest.csv",
            "├── manifest.json",
            "└── DATA_MIX_REPORT.md",
            "```",
            "",
            "## 8. Run Command",
            "",
            "```powershell",
            "python data_mix.py",
            "```",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run(force_export: bool = False) -> dict:
    for path in (DATASET_1, DATASET_2, BRAIN_TUMOR_DIR, NINS_DIR):
        if not path.exists():
            raise FileNotFoundError(f"Missing source dataset: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_braintumor_images(force=force_export)

    records = (
        collect_dataset_1()
        + collect_dataset_2()
        + collect_braintumor()
        + collect_nins()
    )
    print(f"Collected {len(records)} images from four source datasets.")

    dedup_stats = deduplicate_records(records)
    kept_records = write_output(records)
    manifest_rows = build_manifest_rows(kept_records)
    save_manifest(manifest_rows)
    summary = summarize(records, dedup_stats)
    report_path = write_report(summary)

    print("\n=== Data Mix Results ===")
    print(f"Total:   {summary['total']}")
    print(f"Kept:    {summary['kept']}")
    print(f"Removed: {summary['removed']}")
    print(f"  - SHA-256 exact: {summary['removed_sha256']}")
    print(f"  - pHash near:    {summary['removed_phash']}")
    print(f"Report:  {report_path}")
    print(f"Output:  {OUTPUT_DIR}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and deduplicate MRI classification datasets.")
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export all BrainTumorDataPublic .mat files to JPG even if cache exists.",
    )
    args = parser.parse_args()
    run(force_export=args.force_export)


if __name__ == "__main__":
    main()
