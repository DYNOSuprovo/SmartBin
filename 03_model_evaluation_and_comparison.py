"""
============================================================
 03_model_evaluation_and_comparison.py
 Smart Waste Detection — Post‑Training Analysis
============================================================
Load experiment results, build visual leaderboard, evaluate the
best model on the test set, show confusion matrix, per‑class
metrics, misclassifications, and Grad‑CAM explainability.
"""

# %% [markdown]
# # 📊 03 — Model Evaluation & Comparison
# Comprehensive post‑training analysis of all experiments.

# %% ── Imports ──────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from config import PATHS, CLASS_NAMES, NUM_CLASSES
from utils.dataset_utils import (
    set_seed, get_device, create_data_transforms, build_dataloaders,
)
from utils.training_utils import create_model, load_checkpoint
from utils.inference_utils import (
    evaluate_model, plot_confusion_matrix, plot_training_curves,
    show_misclassifications, visualize_gradcam, get_target_layer,
)

set_seed(42)
device = get_device()

# %% [markdown]
# ## 1. Load Experiment Results

# %% ── Leaderboard ────────────────────────────────────────
lb_path = PATHS["outputs"] / "experiment_leaderboard.csv"
if lb_path.exists():
    df = pd.read_csv(lb_path, index_col=0)
    print("=" * 70)
    print("               EXPERIMENT LEADERBOARD")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)
else:
    print("[WARN] No leaderboard found. Run 02_model_training first.")

# %% ── Visual comparison bar chart ─────────────────────────
if lb_path.exists():
    fig, ax = plt.subplots(figsize=(12, 6))
    models = df["Model"].astype(str) + " (LR=" + df["LR"].astype(str) + ")"
    acc = df["Best Val Acc%"]
    colors = plt.cm.RdYlGn(acc / 100)

    bars = ax.barh(models, acc, color=colors, edgecolor="white")
    ax.set_xlabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison — Validation Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)

    for bar, a in zip(bars, acc):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{a:.1f}%", va="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    fig.savefig(str(PATHS["outputs"] / "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 2. Load Best Model

# %% ── Identify best model ─────────────────────────────────
best_info_path = PATHS["outputs"] / "best_model_info.json"
if best_info_path.exists():
    with open(best_info_path) as f:
        best_info = json.load(f)
    print(f"\n🏆 Best model : {best_info['model_tag']}")
    print(f"   Val Acc    : {best_info['best_val_acc']:.2f}%")
else:
    print("[WARN] No best_model_info.json found.")
    best_info = {"model_tag": "resnet18_lr0.0003_adamw", "best_val_acc": 0}

# Parse model name from tag
model_tag = best_info["model_tag"]
model_backbone = model_tag.split("_lr")[0]  # e.g. "resnet18"
print(f"   Backbone   : {model_backbone}")

# %% ── Load model ──────────────────────────────────────────
model = create_model(model_backbone, NUM_CLASSES, pretrained=False).to(device)

best_ckpt = PATHS["best_model"]
if best_ckpt.exists():
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded model from {best_ckpt}")
else:
    # Try phase2 checkpoint
    phase2_path = PATHS["models"] / f"{model_tag}_phase2_best.pth"
    if phase2_path.exists():
        load_checkpoint(model, str(phase2_path), device)
    else:
        print("[ERROR] No checkpoint found. Cannot evaluate.")

model.eval()

# %% [markdown]
# ## 3. Test Set Evaluation

# %% ── Build test loader ───────────────────────────────────
transforms_dict = create_data_transforms(224, augment="none")
loaders, dsets = build_dataloaders(
    PATHS["dataset"], transforms_dict,
    batch_size=32, num_workers=2,
)

test_loader = loaders.get("test")
if test_loader is None:
    print("[ERROR] No test set found!")
else:
    print(f"\nTest set: {len(dsets['test'])} images")

# %% ── Evaluate on test set ────────────────────────────────
if test_loader:
    preds, labels, probs, metrics = evaluate_model(
        model, test_loader, device, CLASS_NAMES
    )

    print(f"\n{'='*50}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"{'='*50}")

    # Save metrics
    with open(PATHS["outputs"] / "test_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items()
                   if k != "classification_report"}, f, indent=2)

# %% [markdown]
# ## 4. Confusion Matrix

# %% ── Plot confusion matrices ─────────────────────────────
if test_loader:
    plot_confusion_matrix(
        labels, preds, CLASS_NAMES,
        title="Test Set Confusion Matrix",
        save_path=str(PATHS["outputs"] / "confusion_matrix_normalized.png"),
        normalize=True,
    )
    plot_confusion_matrix(
        labels, preds, CLASS_NAMES,
        title="Test Set Confusion Matrix (Counts)",
        save_path=str(PATHS["outputs"] / "confusion_matrix_counts.png"),
        normalize=False,
    )

# %% [markdown]
# ## 5. Per‑Class Analysis

# %% ── Per‑class metrics table ─────────────────────────────
if test_loader:
    from sklearn.metrics import classification_report
    report_dict = classification_report(labels, preds, target_names=CLASS_NAMES,
                                         output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).T
    print("\nPer-Class Metrics:")
    print(df_report.to_string())
    df_report.to_csv(PATHS["outputs"] / "per_class_metrics.csv")

# %% [markdown]
# ## 6. Misclassified Examples

# %% ── Show misclassifications ─────────────────────────────
if test_loader and "test" in dsets:
    show_misclassifications(
        model, dsets["test"], device, CLASS_NAMES,
        num_show=16,
        save_path=str(PATHS["outputs"] / "misclassifications.png"),
    )

# %% [markdown]
# ## 7. Training Curve Comparison

# %% ── Overlay best training curves ────────────────────────
log_files = list(PATHS["logs"].glob("history_*.json"))
if log_files:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for log_file in log_files:
        with open(log_file) as f:
            hist = json.load(f)
        label = log_file.stem.replace("history_", "")
        epochs = range(1, len(hist["val_acc"]) + 1)
        ax1.plot(epochs, hist["val_loss"], label=label, alpha=0.7, linewidth=1.5)
        ax2.plot(epochs, hist["val_acc"], label=label, alpha=0.7, linewidth=1.5)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss")
    ax1.set_title("Validation Loss Comparison", fontweight="bold")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy (%)")
    ax2.set_title("Validation Accuracy Comparison", fontweight="bold")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    fig.suptitle("All Experiments — Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(PATHS["outputs"] / "all_experiments_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 8. Grad‑CAM Explainability

# %% ── Grad‑CAM on sample images ──────────────────────────
if test_loader and "test" in dsets:
    test_dir = PATHS["dataset"] / "test"
    sample_images = []
    for cls_dir in sorted(test_dir.iterdir()):
        if cls_dir.is_dir():
            imgs = list(cls_dir.iterdir())
            if imgs:
                sample_images.append(str(imgs[0]))

    print(f"\nGenerating Grad‑CAM for {len(sample_images)} sample images...")
    for i, img_path in enumerate(sample_images[:5]):
        try:
            visualize_gradcam(
                model, model_backbone, img_path, CLASS_NAMES, device,
                image_size=224,
                save_path=str(PATHS["outputs"] / f"gradcam_sample_{i}.png"),
            )
        except Exception as e:
            print(f"[WARN] Grad‑CAM failed for {img_path}: {e}")

# %% [markdown]
# ## 9. Final Recommendation

# %% ── Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("          EVALUATION SUMMARY")
print("=" * 60)
if test_loader:
    print(f"  Best Model     : {model_tag}")
    print(f"  Test Accuracy  : {metrics['accuracy']:.2f}%")
    print(f"  Test F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  Model Path     : {PATHS['best_model']}")
print("=" * 60)
print("\n✅ Evaluation complete. Proceed to 04_smart_dustbin_simulation.py")
