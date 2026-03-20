"""
============================================================
 training_utils.py — Model Factory, Training Loop, Checkpoints
============================================================
Core training engine: create models, train with early stopping,
schedule LR, save checkpoints, run full experiments.
"""

import os
import time
import copy
import json
import csv
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("[WARN] timm not installed — some models may not be available.")


# ── model factory ───────────────────────────────────────────
def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Create a pretrained image classification model and replace its
    classifier head for `num_classes`.

    Supported: resnet18, resnet50, efficientnet_b0, efficientnet_b1,
               mobilenet_v2, densenet121, convnext_tiny
    """
    model_name = model_name.lower().replace("-", "_")

    if HAS_TIMM:
        model = timm.create_model(model_name, pretrained=pretrained,
                                  num_classes=num_classes)
        print(f"[INFO] Created {model_name} via timm (pretrained={pretrained})")
        return model

    # Fallback: torchvision
    import torchvision.models as tv_models
    factory = {
        "resnet18":      (tv_models.resnet18,      "fc"),
        "resnet50":      (tv_models.resnet50,      "fc"),
        "mobilenet_v2":  (tv_models.mobilenet_v2,  "classifier.1"),
        "densenet121":   (tv_models.densenet121,   "classifier"),
    }
    if model_name not in factory:
        raise ValueError(f"Model {model_name} not supported without timm. "
                         f"Install timm: pip install timm")

    constructor, head_attr = factory[model_name]
    weights = "IMAGENET1K_V1" if pretrained else None
    model = constructor(weights=weights)

    # Replace head
    parts = head_attr.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    old_head = getattr(parent, parts[-1])
    in_features = old_head.in_features if hasattr(old_head, "in_features") else old_head.in_features
    setattr(parent, parts[-1], nn.Linear(in_features, num_classes))
    print(f"[INFO] Created {model_name} via torchvision (pretrained={pretrained})")
    return model


def count_parameters(model) -> dict:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def freeze_backbone(model):
    """Freeze all parameters except the final classifier head."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier — works for most architectures
    classifier_names = ["fc", "classifier", "head", "head.fc"]
    for name in classifier_names:
        parts = name.split(".")
        try:
            module = model
            for p in parts:
                module = getattr(module, p)
            for param in module.parameters():
                param.requires_grad = True
            print(f"[INFO] Backbone frozen; '{name}' unfrozen.")
            return
        except AttributeError:
            continue

    # Fallback: unfreeze last layer
    params = list(model.parameters())
    for p in params[-2:]:
        p.requires_grad = True
    print("[INFO] Backbone frozen; last 2 param groups unfrozen (fallback).")


def unfreeze_model(model):
    """Unfreeze all parameters for full fine‑tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model fully unfrozen — {trainable:,} trainable params.")


# ── early stopping ──────────────────────────────────────────
class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score + self.min_delta if self.mode == "max"
                    else score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"[INFO] Early stopping triggered (patience={self.patience}).")
                return True
        return False


# ── one epoch routines ──────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=0):
    """Train for one epoch; return avg loss and accuracy."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [TRAIN]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{100.*correct/total:.2f}%")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, epoch=0):
    """Validate for one epoch; return avg loss and accuracy."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [VAL]  ", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{100.*correct/total:.2f}%")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── checkpointing ───────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }, path)
    print(f"[INFO] Checkpoint saved → {path}  (val_acc={val_acc:.2f}%)")


def load_checkpoint(model, path, device="cpu"):
    """Load model weights from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from {path} (val_acc={ckpt.get('val_acc', '?')}%)")
    return ckpt


# ── full training routine ───────────────────────────────────
def train_model(model, train_loader, val_loader, device,
                epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-4,
                patience: int = 7, label_smoothing: float = 0.1,
                class_weights=None, optimizer_name: str = "adamw",
                save_dir: str = "models", model_tag: str = "model"):
    """
    Full training loop with early stopping, LR scheduling,
    class weighting, and checkpoint saving.

    Returns: history dict with per‑epoch metrics.
    """
    # Criterion
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                     label_smoothing=label_smoothing)

    # Optimizer
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Early stopping
    early_stop = EarlyStopping(patience=patience, mode="max")

    # History
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": [],
    }
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Training: {model_tag}")
    print(f" Optimizer: {optimizer_name} | LR: {lr} | Epochs: {epochs}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:3d}/{epochs} │ "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% │ "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:6.2f}% │ "
              f"LR: {current_lr:.2e}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            ckpt_path = save_dir / f"{model_tag}_best.pth"
            save_checkpoint(model, optimizer, epoch, val_acc, ckpt_path)

        # Early stopping
        if early_stop(val_acc):
            print(f"[INFO] Stopping at epoch {epoch}.")
            break

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training complete in {elapsed/60:.1f} min. "
          f"Best val accuracy: {best_val_acc:.2f}%")

    # Restore best weights
    model.load_state_dict(best_model_wts)
    history["best_val_acc"] = best_val_acc
    history["total_time_s"] = elapsed

    return history


# ── experiment runner ───────────────────────────────────────
def run_experiment(model_name, num_classes, train_loader, val_loader, device,
                   lr=1e-3, optimizer_name="adamw", epochs=30,
                   patience=7, label_smoothing=0.1, class_weights=None,
                   save_dir="models", freeze_epochs=5):
    """
    Two‑phase experiment: freeze backbone → fine‑tune.
    Returns: model, history_dict
    """
    # Phase 1: Frozen backbone
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT: {model_name} | LR={lr} | Opt={optimizer_name}")
    print(f"{'#'*60}")

    model = create_model(model_name, num_classes, pretrained=True).to(device)
    freeze_backbone(model)

    tag = f"{model_name}_lr{lr}_{optimizer_name}"

    history_phase1 = train_model(
        model, train_loader, val_loader, device,
        epochs=freeze_epochs, lr=lr * 10,  # higher LR for head-only
        patience=patience, label_smoothing=label_smoothing,
        class_weights=class_weights, optimizer_name=optimizer_name,
        save_dir=save_dir, model_tag=f"{tag}_phase1",
    )

    # Phase 2: Full fine-tuning
    unfreeze_model(model)
    history_phase2 = train_model(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr,
        patience=patience, label_smoothing=label_smoothing,
        class_weights=class_weights, optimizer_name=optimizer_name,
        save_dir=save_dir, model_tag=f"{tag}_phase2",
    )

    # Merge histories
    combined = {}
    for key in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]:
        combined[key] = history_phase1[key] + history_phase2[key]
    combined["best_val_acc"] = max(history_phase1["best_val_acc"],
                                    history_phase2["best_val_acc"])
    combined["total_time_s"] = (history_phase1.get("total_time_s", 0) +
                                 history_phase2.get("total_time_s", 0))
    combined["model_name"] = model_name
    combined["lr"] = history_phase1["lr"] + history_phase2["lr"]
    combined["optimizer"] = optimizer_name

    return model, combined


def save_experiment_results(results_list: list, csv_path: str):
    """Save a list of experiment result dicts to CSV."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "optimizer", "learning_rate",
                  "best_val_acc", "total_time_s", "total_epochs"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_list:
            writer.writerow({
                "model_name":    r.get("model_name", ""),
                "optimizer":     r.get("optimizer", ""),
                "learning_rate": r.get("lr_used", ""),
                "best_val_acc":  f"{r.get('best_val_acc', 0):.2f}",
                "total_time_s":  f"{r.get('total_time_s', 0):.1f}",
                "total_epochs":  len(r.get("train_loss", [])),
            })
    print(f"[INFO] Experiment results saved → {csv_path}")
