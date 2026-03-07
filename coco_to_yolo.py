"""
COCO to YOLO Format Converter
==============================
Converts a COCO-format dataset to YOLO format, with an optional
deterministic train/val split. The original dataset is never modified.

COCO bbox format : [x_min, y_min, width, height]  (absolute pixels)
YOLO label format: <class_id> <x_center> <y_center> <width> <height>  (normalised 0-1)

Expected input layout
---------------------
<dataset-dir>/
    images/train/          ← source images (e.g. 1.png, 2.png, ...)
    annotations/instances_train.json

Usage
-----
# Minimal — no validation split:
    python coco_to_yolo.py \\
        --input  /path/to/coco/dataset \\
        --output /path/to/yolo/output

# Reserve 20 % for validation (reproducible):
    python coco_to_yolo.py \\
        --input     /path/to/coco/dataset \\
        --output    /path/to/yolo/output \\
        --val-split 0.2

# Same split with a custom seed:
    python coco_to_yolo.py \\
        --input     /path/to/coco/dataset \\
        --output    /path/to/yolo/output \\
        --val-split 0.2 \\
        --seed      123

Output structure
----------------
<output-dir>/
    images/
        train/   ← copied images
        val/     ← (only when --val-split is used)
    labels/
        train/   ← YOLO .txt files
        val/     ← (only when --val-split is used)
    dataset.yaml
"""

import argparse
import json
import random
import shutil
from pathlib import Path

DEFAULT_SEED = 42


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x_min, y_min, w, h] → YOLO [xc, yc, w, h] normalised."""
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center  = (y_min + h / 2) / img_h
    norm_w    = w / img_w
    norm_h    = h / img_h
    return x_center, y_center, norm_w, norm_h


def write_labels(image_ids, image_info, annotations_by_image,
                 cat_id_to_idx, labels_dir: Path):
    """Write YOLO .txt label files for a list of image IDs into labels_dir."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    total_boxes = 0

    for img_id in image_ids:
        img_meta = image_info[img_id]
        img_w, img_h = img_meta["width"], img_meta["height"]
        img_stem     = Path(img_meta["file_name"]).stem
        label_path   = labels_dir / f"{img_stem}.txt"

        lines = []
        for ann in annotations_by_image.get(img_id, []):
            if ann.get("iscrowd", 0):
                continue
            class_idx = cat_id_to_idx[ann["category_id"]]
            xc, yc, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)

            # Clamp to [0, 1] to guard against rare floating-point edge cases
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            total_boxes += 1

        with open(label_path, "w") as lf:
            lf.write("\n".join(lines))

    return total_boxes


def copy_images(image_ids, image_info, src_dir: Path, dst_dir: Path):
    """Copy images for the given IDs from src_dir to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_id in image_ids:
        fname = image_info[img_id]["file_name"]
        src   = src_dir / fname
        dst   = dst_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def convert(input_dir: Path, output_dir: Path, val_split: float, seed: int):
    annotations_file = input_dir / "annotations" / "instances_train.json"
    images_src_dir   = input_dir / "images" / "train"

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_file}")
    if not images_src_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_src_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input  dataset : {input_dir}")
    print(f"Output dataset : {output_dir}")
    print(f"Loading annotations from: {annotations_file}")

    with open(annotations_file, "r") as f:
        coco = json.load(f)

    # ── Build lookup tables ────────────────────────────────────────────────────
    image_info    = {img["id"]: img for img in coco["images"]}
    categories    = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
    class_names   = [cat["name"] for cat in categories]

    annotations_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    # ── Deterministic train / val split ───────────────────────────────────────
    # Sort by image_id first so the starting order is always identical,
    # then shuffle with a fixed seed for a reproducible random split.
    all_ids = sorted(image_info.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ids)

    if val_split > 0.0:
        n_val     = max(1, round(len(all_ids) * val_split))
        val_ids   = all_ids[:n_val]
        train_ids = all_ids[n_val:]
    else:
        train_ids = all_ids
        val_ids   = []

    print(f"\nSeed: {seed}  |  Total: {len(all_ids)}  "
          f"|  Train: {len(train_ids)}  |  Val: {len(val_ids)}")

    # ── Copy images and write labels ───────────────────────────────────────────
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        if not ids:
            continue

        img_dst_dir   = output_dir / "images" / split
        label_dst_dir = output_dir / "labels" / split

        copy_images(ids, image_info, images_src_dir, img_dst_dir)
        boxes = write_labels(ids, image_info, annotations_by_image,
                             cat_id_to_idx, label_dst_dir)
        print(f"✓ {split:<5} images → {img_dst_dir}")
        print(f"  {split:<5} labels → {label_dst_dir}  ({boxes} boxes)")

    # ── Write dataset.yaml ────────────────────────────────────────────────────
    val_line = (f"val:   images/val          # {len(val_ids)} images"
                if val_ids else
                "# val: images/val          # add --val-split to generate")

    yaml_out = output_dir / "dataset.yaml"
    yaml_content = f"""\
# YOLO dataset configuration
# Generated by coco_to_yolo.py
# Seed used for split: {seed}

path: {output_dir}

train: images/train        # {len(train_ids)} images
{val_line}

nc: {len(class_names)}
names: {class_names}
"""
    with open(yaml_out, "w") as yf:
        yf.write(yaml_content)
    print(f"\n✓ dataset.yaml  → {yaml_out}")

    print("\nClass mapping:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a COCO dataset to YOLO format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        metavar="DIR",
        help="Root directory of the COCO dataset. "
             "Must contain images/train/ and annotations/instances_train.json.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory for the YOLO dataset. "
             "Will be created if it doesn't exist. "
             "The original dataset is never modified.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="Fraction of images to reserve for validation, e.g. 0.2 for 20%%. "
             "Default: 0 (no validation split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for the train/val shuffle (default: {DEFAULT_SEED}). "
             "Using the same seed always produces the same split.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.val_split < 1.0):
        parser.error("--val_split must be in [0.0, 1.0)")

    convert(
        input_dir=args.input.resolve(),
        output_dir=args.output.resolve(),
        val_split=args.val_split,
        seed=args.seed,
    )