"""
============================================================
 config.py — Central Configuration for Smart Waste Detection
============================================================
All paths, class definitions, hyperparameters, and experiment
settings live here so every other module stays DRY.
"""

import os
from pathlib import Path

# ── Project root ────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ───────────────────────────────────────────────────
PATHS = {
    "raw_data":        PROJECT_ROOT / "standardized_256",
    "dataset":         PROJECT_ROOT / "dataset_split",   # train/val/test created by 01_
    "models":          PROJECT_ROOT / "models",
    "logs":            PROJECT_ROOT / "logs",
    "outputs":         PROJECT_ROOT / "outputs",
    "best_model":      PROJECT_ROOT / "models" / "best_waste_classifier.pth",
    "experiment_csv":  PROJECT_ROOT / "outputs" / "experiment_results.csv",
}

# ── Class definitions ──────────────────────────────────────
CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]
NUM_CLASSES = len(CLASS_NAMES)

# ── Bin mappings (class → virtual dustbin compartment) ─────
BIN_MAPPING = {
    "battery":    "hazardous_bin",
    "biological": "organic_bin",
    "cardboard":  "paper_bin",
    "clothes":    "textile_bin",
    "glass":      "glass_bin",
    "metal":      "metal_bin",
    "paper":      "paper_bin",
    "plastic":    "plastic_bin",
    "shoes":      "textile_bin",
    "trash":      "trash_bin",
}

BIN_NAMES = sorted(set(BIN_MAPPING.values()))

# ── Biodegradability mapping ───────────────────────────────
BIODEGRADABLE = {"biological", "cardboard", "paper"}
NON_BIODEGRADABLE = {"battery", "clothes", "glass", "metal", "plastic", "shoes", "trash"}

# ── Fill‑estimation heuristics (configurable) ──────────────
FILL_CONTRIBUTION = {
    "very_small": 2,   # percent
    "small":      5,
    "medium":     10,
    "large":      18,
}
AREA_THRESHOLDS = {        # fraction of image area
    "very_small": 0.05,
    "small":      0.15,
    "medium":     0.35,
}

# ── Image settings ─────────────────────────────────────────
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Training defaults ──────────────────────────────────────
TRAIN_DEFAULTS = {
    "batch_size":       32,
    "num_workers":      2,
    "epochs":           30,
    "learning_rate":    1e-3,
    "weight_decay":     1e-4,
    "patience":         7,       # early‑stopping patience
    "label_smoothing":  0.1,
    "seed":             42,
    "pin_memory":       True,
}

# ── Experiment grid (used by 02_model_training) ────────────
EXPERIMENT_GRID = {
    "models":       ["resnet18", "resnet50", "efficientnet_b0",
                     "mobilenet_v2", "densenet121", "convnext_tiny"],
    "optimizers":   ["adamw"],
    "learning_rates": [3e-4, 1e-4],
    "batch_sizes":  [32],
    "image_sizes":  [224],
}

# ── Streamlit settings ─────────────────────────────────────
STREAMLIT = {
    "page_title":  "Smart Waste Detection System",
    "page_icon":   "♻️",
    "max_upload_mb": 10,
}

# ── Quick‑access aggregated config dict ────────────────────
CONFIG = {
    "PATHS":              PATHS,
    "CLASS_NAMES":        CLASS_NAMES,
    "NUM_CLASSES":        NUM_CLASSES,
    "BIN_MAPPING":        BIN_MAPPING,
    "BIN_NAMES":          BIN_NAMES,
    "IMAGE_SIZE":         IMAGE_SIZE,
    "IMAGENET_MEAN":      IMAGENET_MEAN,
    "IMAGENET_STD":       IMAGENET_STD,
    "TRAIN_DEFAULTS":     TRAIN_DEFAULTS,
    "EXPERIMENT_GRID":    EXPERIMENT_GRID,
    "FILL_CONTRIBUTION":  FILL_CONTRIBUTION,
    "AREA_THRESHOLDS":    AREA_THRESHOLDS,
}
