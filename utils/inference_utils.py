"""
============================================================
 inference_utils.py — Evaluation, Confusion Matrix, Grad‑CAM
============================================================
Post‑training analysis: metrics, visualisations, explainability.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ── evaluation ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model, dataloader, device, class_names=None):
    """
    Run model on a dataloader; return preds, labels, and metrics dict.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    }

    if class_names:
        report = classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0)
        metrics["classification_report"] = report
        print(report)

    print(f"\n[RESULT] Accuracy: {acc:.2f}%  |  F1: {f1:.4f}")
    return all_preds, all_labels, all_probs, metrics


# ── confusion matrix ────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names,
                          title="Confusion Matrix", save_path=None,
                          normalize=True):
    """Plot a beautiful confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title += " (Normalized)"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved → {save_path}")
    plt.show()
    return fig


# ── training curves ─────────────────────────────────────────
def plot_training_curves(history: dict, title: str = "Training Curves",
                         save_path=None):
    """Plot loss and accuracy curves from a training history dict."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", markersize=3)
    ax2.plot(epochs, history["val_acc"], "s-", label="Val Acc", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_acc = max(history["val_acc"])
    best_ep = history["val_acc"].index(best_acc) + 1
    ax2.axhline(y=best_acc, color="red", linestyle="--", alpha=0.5)
    ax2.annotate(f"Best: {best_acc:.2f}% @ E{best_ep}",
                 xy=(best_ep, best_acc), fontsize=9, color="red",
                 fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Training curves saved → {save_path}")
    plt.show()
    return fig


# ── misclassification gallery ───────────────────────────────
def show_misclassifications(model, dataset, device, class_names,
                            num_show: int = 16, save_path=None):
    """Show grid of misclassified images with true vs predicted labels."""
    model.eval()
    misclassified = []

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    idx = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            preds = preds.cpu()

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    img_path = dataset.samples[idx + i][0]
                    misclassified.append({
                        "path": img_path,
                        "true": class_names[labels[i]],
                        "pred": class_names[preds[i]],
                        "conf": probs[i, preds[i]].item(),
                    })
            idx += len(labels)

    # Show grid
    n = min(num_show, len(misclassified))
    if n == 0:
        print("[INFO] No misclassifications found!")
        return None

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        m = misclassified[i]
        img = Image.open(m["path"]).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {m['true']}\nPred: {m['pred']} ({m['conf']:.1%})",
            fontsize=8, color="red", fontweight="bold")
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Misclassified Samples ({len(misclassified)} total)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Misclassification gallery saved → {save_path}")
    plt.show()
    return fig


# ── Grad‑CAM ───────────────────────────────────────────────
class GradCAM:
    """Gradient‑weighted Class Activation Mapping."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """Generate Grad‑CAM heatmap for a single image tensor."""
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        grads = self.gradients[0]          # (C, H, W)
        acts = self.activations[0]         # (C, H, W)
        weights = grads.mean(dim=(1, 2))   # (C,)

        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx


def get_target_layer(model, model_name: str):
    """Return the last convolutional layer for Grad‑CAM."""
    model_name = model_name.lower()
    try:
        if "resnet" in model_name:
            return model.layer4[-1]
        elif "efficientnet" in model_name:
            # timm efficientnet
            if hasattr(model, "conv_head"):
                return model.conv_head
            return list(model.children())[-3]
        elif "mobilenet" in model_name:
            return model.features[-1]
        elif "densenet" in model_name:
            return model.features.denseblock4
        elif "convnext" in model_name:
            if hasattr(model, "stages"):
                return model.stages[-1]
            return list(model.children())[-3]
    except Exception:
        pass
    # Fallback
    conv_layers = [m for m in model.modules()
                   if isinstance(m, torch.nn.Conv2d)]
    return conv_layers[-1] if conv_layers else None


def visualize_gradcam(model, model_name, image_path, class_names, device,
                      image_size=224, save_path=None):
    """Run Grad‑CAM on a single image and display overlay."""
    import cv2

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = tfm(img).unsqueeze(0).to(device)

    target_layer = get_target_layer(model, model_name)
    if target_layer is None:
        print("[WARN] Could not determine target layer for Grad‑CAM.")
        return None

    grad_cam = GradCAM(model, target_layer)
    heatmap, pred_idx = grad_cam.generate(input_tensor)

    # Overlay
    img_resized = img.resize((image_size, image_size))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    heatmap_resized = cv2.resize(heatmap, (image_size, image_size))
    heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3]

    overlay = 0.5 * img_np + 0.5 * heatmap_color

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img_np)
    ax1.set_title("Original", fontweight="bold")
    ax1.axis("off")

    ax2.imshow(heatmap_resized, cmap="jet")
    ax2.set_title("Grad-CAM Heatmap", fontweight="bold")
    ax2.axis("off")

    ax3.imshow(np.clip(overlay, 0, 1))
    ax3.set_title(f"Predicted: {class_names[pred_idx]}", fontweight="bold")
    ax3.axis("off")

    fig.suptitle("Grad-CAM Explainability", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
