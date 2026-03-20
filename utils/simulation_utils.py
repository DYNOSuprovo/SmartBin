"""
============================================================
 simulation_utils.py — Virtual Smart Dustbin Simulation
============================================================
Software‑only smart dustbin: classify waste, estimate fill
contribution from image analysis, update virtual bins, log events.

NOTE: Fill estimation is approximate (image‑area heuristic).
      Real deployment would need depth sensors / calibration.
"""

import json
import time
import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── waste prediction ────────────────────────────────────────
def predict_waste(model, image, device, class_names, image_size=224):
    """
    Predict waste class from a PIL Image or file path.

    Returns:
        dict with keys: class_name, class_idx, confidence, probabilities
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")

    input_tensor = tfm(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        conf, pred_idx = probs.max(0)

    return {
        "class_name":   class_names[pred_idx.item()],
        "class_idx":    pred_idx.item(),
        "confidence":   conf.item(),
        "probabilities": {class_names[i]: probs[i].item()
                          for i in range(len(class_names))},
    }


# ── fill estimation (image‑based heuristic) ─────────────────
def estimate_fill_contribution(image, size_config=None):
    """
    Estimate how much space a waste item occupies using
    image‑area analysis.

    Strategy:
    1. Convert to grayscale, threshold to isolate foreground.
    2. Compute foreground-area ratio.
    3. Map to fill-contribution bucket.

    Returns:
        dict with keys: area_ratio, size_category, fill_percent
    """
    if size_config is None:
        size_config = {
            "thresholds": {"very_small": 0.05, "small": 0.15, "medium": 0.35},
            "contributions": {"very_small": 2, "small": 5, "medium": 10, "large": 18},
        }

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pass
    else:
        image = Image.fromarray(image)

    img_np = np.array(image.convert("RGB"))

    if HAS_CV2:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu thresholding
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        foreground_pixels = np.count_nonzero(mask)
    else:
        # Fallback without OpenCV: simple threshold on luminance
        gray = np.mean(img_np, axis=2)
        threshold = np.mean(gray) * 0.8
        foreground_pixels = np.sum(gray < threshold)

    total_pixels = img_np.shape[0] * img_np.shape[1]
    area_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0

    thresholds = size_config["thresholds"]
    contributions = size_config["contributions"]

    if area_ratio < thresholds["very_small"]:
        category = "very_small"
    elif area_ratio < thresholds["small"]:
        category = "small"
    elif area_ratio < thresholds["medium"]:
        category = "medium"
    else:
        category = "large"

    return {
        "area_ratio":    round(area_ratio, 4),
        "size_category": category,
        "fill_percent":  contributions[category],
    }


# ── bin mapping ─────────────────────────────────────────────
DEFAULT_BIN_MAPPING = {
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

def map_class_to_bin(class_name: str, mapping: dict = None) -> str:
    """Map a waste class to its virtual dustbin compartment."""
    mapping = mapping or DEFAULT_BIN_MAPPING
    return mapping.get(class_name, "trash_bin")


# ── virtual bin ─────────────────────────────────────────────
class VirtualBin:
    """A single virtual dustbin compartment."""

    def __init__(self, name: str, capacity: float = 100.0):
        self.name = name
        self.capacity = capacity
        self.fill_level = 0.0
        self.item_count = 0
        self.recent_items = []  # last N items

    def add_item(self, class_name: str, fill_contribution: float,
                 confidence: float = 0.0):
        """Add a waste item to this bin."""
        self.fill_level = min(self.fill_level + fill_contribution, self.capacity)
        self.item_count += 1
        self.recent_items.append({
            "class": class_name,
            "fill_added": fill_contribution,
            "confidence": confidence,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        # Keep only last 50
        if len(self.recent_items) > 50:
            self.recent_items = self.recent_items[-50:]

    @property
    def fill_percentage(self):
        return (self.fill_level / self.capacity) * 100

    @property
    def is_near_full(self):
        return self.fill_percentage >= 80

    @property
    def is_full(self):
        return self.fill_percentage >= 100

    def reset(self):
        self.fill_level = 0.0
        self.item_count = 0
        self.recent_items = []

    def to_dict(self):
        return {
            "name": self.name,
            "fill_percentage": round(self.fill_percentage, 1),
            "item_count": self.item_count,
            "is_near_full": self.is_near_full,
            "is_full": self.is_full,
        }


# ── Smart Dustbin Manager ──────────────────────────────────
class SmartDustbin:
    """
    Manages multiple virtual bin compartments and orchestrates
    waste classification → bin assignment → fill estimation.
    """

    BIN_COLORS = {
        "plastic_bin":   "#3498db",
        "paper_bin":     "#f39c12",
        "metal_bin":     "#95a5a6",
        "glass_bin":     "#2ecc71",
        "organic_bin":   "#8B4513",
        "textile_bin":   "#9b59b6",
        "hazardous_bin": "#e74c3c",
        "trash_bin":     "#34495e",
    }

    def __init__(self, bin_mapping: dict = None):
        self.bin_mapping = bin_mapping or DEFAULT_BIN_MAPPING
        bin_names = sorted(set(self.bin_mapping.values()))
        self.bins = {name: VirtualBin(name) for name in bin_names}
        self.event_log = []

    def add_waste(self, class_name: str, confidence: float,
                  size_category: str = "medium", fill_percent: float = None):
        """Add a classified waste item to the appropriate bin."""
        from config import FILL_CONTRIBUTION
        bin_name = map_class_to_bin(class_name, self.bin_mapping)

        if fill_percent is None:
            fill_percent = FILL_CONTRIBUTION.get(size_category, 10)

        target_bin = self.bins[bin_name]
        target_bin.add_item(class_name, fill_percent, confidence)

        event = {
            "timestamp":        datetime.datetime.now().isoformat(),
            "waste_class":      class_name,
            "confidence":       round(confidence, 4),
            "target_bin":       bin_name,
            "fill_contribution": fill_percent,
            "size_category":    size_category,
            "bin_fill_after":   round(target_bin.fill_percentage, 1),
        }
        self.event_log.append(event)

        # Alert
        if target_bin.is_full:
            print(f"🚨 ALERT: {bin_name} is FULL! Please empty.")
        elif target_bin.is_near_full:
            print(f"⚠️ WARNING: {bin_name} is at {target_bin.fill_percentage:.0f}% — nearing capacity!")

        return event

    def process_image(self, model, image, device, class_names,
                      image_size=224, size_config=None):
        """
        Full pipeline: predict waste → estimate fill → update bin.
        Returns detailed result dict.
        """
        prediction = predict_waste(model, image, device, class_names, image_size)
        fill_info = estimate_fill_contribution(image, size_config)

        event = self.add_waste(
            class_name=prediction["class_name"],
            confidence=prediction["confidence"],
            size_category=fill_info["size_category"],
            fill_percent=fill_info["fill_percent"],
        )

        return {
            "prediction": prediction,
            "fill_estimation": fill_info,
            "bin_event": event,
        }

    def get_status(self) -> dict:
        """Return current status of all bins."""
        return {name: b.to_dict() for name, b in self.bins.items()}

    def display_bin_status(self, save_path=None):
        """Visualise all bin fill levels as a horizontal bar chart."""
        status = self.get_status()
        names = list(status.keys())
        fills = [status[n]["fill_percentage"] for n in names]
        counts = [status[n]["item_count"] for n in names]
        colors = [self.BIN_COLORS.get(n, "#777777") for n in names]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, fills, color=colors, edgecolor="white", height=0.6)

        for bar, fill, count in zip(bars, fills, counts):
            color = "red" if fill >= 80 else ("orange" if fill >= 50 else "white")
            label = f" {fill:.0f}% ({count} items)"
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=10, fontweight="bold", color="black")

        ax.set_xlim(0, 120)
        ax.axvline(x=80, color="orange", linestyle="--", alpha=0.6, label="Warning (80%)")
        ax.axvline(x=100, color="red", linestyle="--", alpha=0.6, label="Full (100%)")
        ax.set_xlabel("Fill Level (%)", fontsize=12)
        ax.set_title("🗑️ Smart Dustbin — Compartment Fill Status",
                     fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")

        # Color‑code background for near‑full bins
        for i, fill in enumerate(fills):
            if fill >= 80:
                ax.get_children()[i].set_edgecolor("red")
                ax.get_children()[i].set_linewidth(2)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    def reset_bins(self):
        """Empty all bins."""
        for b in self.bins.values():
            b.reset()
        self.event_log = []
        print("[INFO] All bins have been emptied.")

    def save_log(self, path: str):
        """Save event log to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.event_log, f, indent=2)
        print(f"[INFO] Event log saved → {path} ({len(self.event_log)} events)")

    def get_log_dataframe(self):
        """Return event log as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.event_log)
