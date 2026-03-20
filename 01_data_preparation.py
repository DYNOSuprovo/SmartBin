"""
============================================================
 01_data_preparation.py
 Smart Waste Detection — Dataset Inspection & Preparation
============================================================
This script inspects the raw dataset, visualises class distributions,
checks for corrupted images, splits data into train/val/test, and
previews augmentations. Run top‑to‑bottom or convert to a Jupyter
notebook (each # %% block becomes a cell).
"""

# %% [markdown]
# # 📦 01 — Data Preparation
# Inspect the raw image dataset, check integrity, create
# train / val / test splits, and preview augmentations.

# %% ── Imports ──────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import json
import pandas as pd

from config import PATHS, CLASS_NAMES, NUM_CLASSES, TRAIN_DEFAULTS
from utils.dataset_utils import (
    set_seed,
    get_class_distribution,
    plot_class_distribution,
    show_sample_images,
    check_corrupted_images,
    split_dataset,
    create_data_transforms,
    preview_augmentations,
    build_dataloaders,
    compute_class_weights,
)

set_seed(TRAIN_DEFAULTS["seed"])

# %% [markdown]
# ## 1. Dataset Overview

# %% ── Raw dataset inspection ──────────────────────────────
RAW_DIR = PATHS["raw_data"]
print(f"Raw dataset path : {RAW_DIR}")
print(f"Expected classes : {CLASS_NAMES}")
print(f"Number of classes: {NUM_CLASSES}\n")

distribution = get_class_distribution(RAW_DIR)
print("Class distribution:")
for cls, cnt in distribution.items():
    print(f"  {cls:15s} : {cnt:5d}")
total = sum(distribution.values())
print(f"  {'TOTAL':15s} : {total:5d}")

# %% ── Distribution chart ──────────────────────────────────
PATHS["outputs"].mkdir(parents=True, exist_ok=True)
plot_class_distribution(
    distribution,
    title="Raw Dataset — Class Distribution",
    save_path=str(PATHS["outputs"] / "class_distribution_raw.png"),
)

# %% [markdown]
# ## 2. Sample Images

# %% ── Show sample grid ────────────────────────────────────
show_sample_images(
    RAW_DIR, num_per_class=4,
    save_path=str(PATHS["outputs"] / "sample_images.png"),
)

# %% [markdown]
# ## 3. Data Integrity Check

# %% ── Corruption scan ─────────────────────────────────────
corrupted = check_corrupted_images(RAW_DIR)
if corrupted:
    print(f"\n⚠️  Found {len(corrupted)} corrupted images:")
    for c in corrupted[:20]:
        print(f"  ✗ {c}")
    if len(corrupted) > 20:
        print(f"  ... and {len(corrupted)-20} more.")
else:
    print("✅ All images are valid — no corruption detected.")

# %% [markdown]
# ## 4. Dataset Imbalance Analysis

# %% ── Imbalance report ────────────────────────────────────
counts = list(distribution.values())
max_c, min_c = max(counts), min(counts)
imbalance_ratio = max_c / min_c if min_c > 0 else float("inf")
print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}x")
print(f"Largest class : {max(distribution, key=distribution.get)} ({max_c})")
print(f"Smallest class: {min(distribution, key=distribution.get)} ({min_c})")

if imbalance_ratio > 3:
    print("⚠️  Significant imbalance detected — class weighting will be applied during training.")
else:
    print("✅ Dataset is relatively balanced.")

# %% [markdown]
# ## 5. Train / Val / Test Split

# %% ── Split dataset ───────────────────────────────────────
SPLIT_DIR = PATHS["dataset"]
if SPLIT_DIR.exists() and any(SPLIT_DIR.iterdir()):
    print(f"[INFO] Split directory already exists: {SPLIT_DIR}")
    print("[INFO] Skipping split. Delete the folder to re‑split.")
else:
    summary = split_dataset(
        source_dir=RAW_DIR,
        dest_dir=SPLIT_DIR,
        ratios=(0.70, 0.15, 0.15),
        seed=TRAIN_DEFAULTS["seed"],
    )

# %% ── Verify split distributions ──────────────────────────
for split in ("train", "val", "test"):
    split_dir = SPLIT_DIR / split
    if split_dir.exists():
        dist = get_class_distribution(split_dir)
        total_split = sum(dist.values())
        print(f"\n{split.upper()} ({total_split} images)")
        plot_class_distribution(
            dist,
            title=f"{split.upper()} Set — Class Distribution",
            save_path=str(PATHS["outputs"] / f"class_distribution_{split}.png"),
        )

# %% [markdown]
# ## 6. Augmentation Preview

# %% ── Preview transforms ──────────────────────────────────
transforms_dict = create_data_transforms(image_size=224, augment="standard")
preview_augmentations(
    SPLIT_DIR / "train",
    transforms_dict["train"],
    num_images=3, num_augments=5,
    save_path=str(PATHS["outputs"] / "augmentation_preview.png"),
)

# %% [markdown]
# ## 7. DataLoader Sanity Check

# %% ── Build loaders ───────────────────────────────────────
loaders, dsets = build_dataloaders(
    SPLIT_DIR, transforms_dict,
    batch_size=TRAIN_DEFAULTS["batch_size"],
    num_workers=TRAIN_DEFAULTS["num_workers"],
)

# Check class weights
if "train" in dsets:
    class_weights = compute_class_weights(dsets["train"])
    print(f"\nClass names from loader: {dsets['train'].classes}")

# %% ── Sanity: inspect one batch ────────────────────────────
if "train" in loaders:
    images, labels = next(iter(loaders["train"]))
    print(f"\nBatch shape : {images.shape}")
    print(f"Label dtype : {labels.dtype}")
    print(f"Label range : {labels.min().item()} – {labels.max().item()}")

# %% [markdown]
# ## 8. Save Metadata Summary

# %% ── Metadata export ─────────────────────────────────────
metadata = {
    "raw_data_path": str(RAW_DIR),
    "split_data_path": str(SPLIT_DIR),
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "raw_distribution": distribution,
    "total_images": total,
    "imbalance_ratio": round(imbalance_ratio, 2),
    "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "image_size": 224,
}

meta_path = PATHS["outputs"] / "dataset_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\n✅ Metadata saved → {meta_path}")

# %% [markdown]
# ## ✅ Data Preparation Complete
# The dataset is split, inspected, and ready for training.
# Proceed to **02_model_training_max_accuracy.py**.

print("\n" + "=" * 60)
print("  DATA PREPARATION COMPLETE — READY FOR TRAINING")
print("=" * 60)
