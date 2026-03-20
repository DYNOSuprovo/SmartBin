"""
============================================================
 02_model_training_max_accuracy.py
 Smart Waste Detection — Multi‑Model Experiment Framework
============================================================
This is the CORE notebook. It trains multiple pretrained backbones
with a two‑phase strategy (frozen head → full fine‑tune), compares
them, and saves the best model. Convert # %% blocks to Jupyter cells.

Priority: MAXIMUM CLASSIFICATION ACCURACY.
"""

# %% [markdown]
# # 🏆 02 — Model Training: Maximum Accuracy
# Compare **ResNet18, ResNet50, EfficientNet‑B0, MobileNetV2,
# DenseNet121, ConvNeXt‑Tiny** using a two‑phase training strategy:
# 1. Freeze backbone → train classifier head
# 2. Unfreeze → full fine‑tune with lower LR & cosine annealing

# %% ── Imports ──────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
import torch

from config import (
    PATHS, CLASS_NAMES, NUM_CLASSES,
    TRAIN_DEFAULTS, EXPERIMENT_GRID, IMAGENET_MEAN, IMAGENET_STD,
)
from utils.dataset_utils import (
    set_seed, get_device, create_data_transforms,
    build_dataloaders, compute_class_weights,
)
from utils.training_utils import (
    create_model, count_parameters, freeze_backbone, unfreeze_model,
    train_model, run_experiment, save_experiment_results,
)
from utils.inference_utils import plot_training_curves

# %% ── Reproducibility & Device ────────────────────────────
set_seed(TRAIN_DEFAULTS["seed"])
device = get_device()

# %% [markdown]
# ## 1. Data Pipeline

# %% ── Transforms ──────────────────────────────────────────
IMAGE_SIZE = 224
transforms_dict = create_data_transforms(IMAGE_SIZE, augment="standard")

# %% ── DataLoaders ─────────────────────────────────────────
loaders, dsets = build_dataloaders(
    PATHS["dataset"],
    transforms_dict,
    batch_size=TRAIN_DEFAULTS["batch_size"],
    num_workers=TRAIN_DEFAULTS["num_workers"],
    pin_memory=TRAIN_DEFAULTS["pin_memory"],
)

train_loader = loaders["train"]
val_loader   = loaders["val"]

print(f"\nClasses: {dsets['train'].classes}")
print(f"Num classes: {NUM_CLASSES}")

# %% ── Class weights (handle imbalance) ────────────────────
class_weights = compute_class_weights(dsets["train"])
class_weights = class_weights.to(device)

# %% [markdown]
# ## 2. Experiment Configuration
# We run a grid of experiments across models, optimizers, and LRs.

# %% ── Experiment grid ─────────────────────────────────────
MODELS = EXPERIMENT_GRID["models"]
OPTIMIZERS = EXPERIMENT_GRID["optimizers"]
LEARNING_RATES = EXPERIMENT_GRID["learning_rates"]

print("=" * 60)
print("EXPERIMENT GRID")
print("=" * 60)
print(f"Models      : {MODELS}")
print(f"Optimizers  : {OPTIMIZERS}")
print(f"LRs         : {LEARNING_RATES}")
total_runs = len(MODELS) * len(OPTIMIZERS) * len(LEARNING_RATES)
print(f"Total runs  : {total_runs}")
print("=" * 60)

# %% [markdown]
# ## 3. Run Experiments
# Each experiment uses two‑phase training:
# - **Phase 1**: frozen backbone, 5 epochs, 10× LR → train head quickly
# - **Phase 2**: unfrozen, full epochs, cosine LR → fine‑tune everything

# %% ── Experiment loop ─────────────────────────────────────
all_results = []
best_overall_acc = 0.0
best_overall_model = None
best_overall_name = ""

PATHS["models"].mkdir(parents=True, exist_ok=True)
PATHS["outputs"].mkdir(parents=True, exist_ok=True)
PATHS["logs"].mkdir(parents=True, exist_ok=True)

for model_name in MODELS:
    for opt_name in OPTIMIZERS:
        for lr in LEARNING_RATES:
            tag = f"{model_name}_lr{lr}_{opt_name}"
            print(f"\n{'#' * 70}")
            print(f"# EXPERIMENT: {tag}")
            print(f"{'#' * 70}")

            try:
                model, history = run_experiment(
                    model_name=model_name,
                    num_classes=NUM_CLASSES,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    lr=lr,
                    optimizer_name=opt_name,
                    epochs=TRAIN_DEFAULTS["epochs"],
                    patience=TRAIN_DEFAULTS["patience"],
                    label_smoothing=TRAIN_DEFAULTS["label_smoothing"],
                    class_weights=class_weights,
                    save_dir=str(PATHS["models"]),
                    freeze_epochs=5,
                )

                # Record
                params = count_parameters(model)
                result = {
                    "model_name":   model_name,
                    "optimizer":    opt_name,
                    "lr_used":      lr,
                    "best_val_acc": history["best_val_acc"],
                    "total_time_s": history.get("total_time_s", 0),
                    "train_loss":   history["train_loss"],
                    "train_acc":    history["train_acc"],
                    "val_loss":     history["val_loss"],
                    "val_acc":      history["val_acc"],
                    "total_params": params["total"],
                    "trainable_params": params["trainable"],
                }
                all_results.append(result)

                # Training curves
                plot_training_curves(
                    history,
                    title=f"Training Curves — {tag}",
                    save_path=str(PATHS["outputs"] / f"curves_{tag}.png"),
                )

                # Track global best
                if history["best_val_acc"] > best_overall_acc:
                    best_overall_acc = history["best_val_acc"]
                    best_overall_model = model
                    best_overall_name = tag

                # Save per-experiment history
                hist_path = PATHS["logs"] / f"history_{tag}.json"
                with open(hist_path, "w") as f:
                    json.dump({k: v for k, v in history.items()
                               if isinstance(v, (list, float, int, str))}, f, indent=2)

            except Exception as e:
                print(f"[ERROR] Experiment {tag} failed: {e}")
                import traceback; traceback.print_exc()
                continue

# %% [markdown]
# ## 4. Experiment Leaderboard

# %% ── Build leaderboard ───────────────────────────────────
leaderboard_data = []
for r in all_results:
    leaderboard_data.append({
        "Model":         r["model_name"],
        "Optimizer":     r["optimizer"],
        "LR":            r["lr_used"],
        "Best Val Acc%": round(r["best_val_acc"], 2),
        "Time (min)":    round(r["total_time_s"] / 60, 1),
        "Params (M)":    round(r["total_params"] / 1e6, 1),
    })

df_leaderboard = pd.DataFrame(leaderboard_data)
df_leaderboard = df_leaderboard.sort_values("Best Val Acc%", ascending=False)
df_leaderboard = df_leaderboard.reset_index(drop=True)
df_leaderboard.index = df_leaderboard.index + 1  # 1‑indexed rank
df_leaderboard.index.name = "Rank"

print("\n" + "=" * 70)
print("                    EXPERIMENT LEADERBOARD")
print("=" * 70)
print(df_leaderboard.to_string())
print("=" * 70)

# Save leaderboard
lb_path = PATHS["outputs"] / "experiment_leaderboard.csv"
df_leaderboard.to_csv(lb_path)
print(f"\n[INFO] Leaderboard saved → {lb_path}")

# %% ── Save all results CSV ────────────────────────────────
save_experiment_results(all_results, str(PATHS["experiment_csv"]))

# %% [markdown]
# ## 5. Save Best Model

# %% ── Save final best model ───────────────────────────────
if best_overall_model is not None:
    best_path = PATHS["best_model"]
    torch.save({
        "model_state_dict": best_overall_model.state_dict(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "model_tag": best_overall_name,
        "best_val_acc": best_overall_acc,
        "image_size": IMAGE_SIZE,
    }, best_path)
    print(f"\n🏆 BEST MODEL: {best_overall_name}")
    print(f"   Val Accuracy: {best_overall_acc:.2f}%")
    print(f"   Saved → {best_path}")

    # Also save a summary
    best_info = {
        "model_tag": best_overall_name,
        "best_val_acc": best_overall_acc,
        "model_path": str(best_path),
    }
    with open(PATHS["outputs"] / "best_model_info.json", "w") as f:
        json.dump(best_info, f, indent=2)
else:
    print("[WARN] No successful experiments — no model saved.")

# %% [markdown]
# ## ✅ Training Complete
# The best model has been saved. Proceed to
# **03_model_evaluation_and_comparison.py** for detailed analysis.

print("\n" + "=" * 60)
print("  TRAINING COMPLETE — PROCEED TO EVALUATION")
print("=" * 60)
