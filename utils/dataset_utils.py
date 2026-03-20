"""
============================================================
 dataset_utils.py — Data Loading, Splitting & Visualization
============================================================
Reusable helpers consumed by 01_data_preparation and others.
"""

import os
import random
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ── reproducibility ─────────────────────────────────────────
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[INFO] Random seed set to {seed}")


def get_device():
    """Return best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[INFO] Using CPU")
    return dev


# ── dataset inspection ──────────────────────────────────────
def get_class_distribution(dataset_path: str | Path) -> dict:
    """Return {class_name: count} for a folder‑per‑class dataset."""
    dataset_path = Path(dataset_path)
    dist = {}
    for cls_dir in sorted(dataset_path.iterdir()):
        if cls_dir.is_dir():
            count = len([f for f in cls_dir.iterdir() if f.is_file()])
            dist[cls_dir.name] = count
    return dist


def plot_class_distribution(distribution: dict, title: str = "Class Distribution",
                            save_path: str | None = None):
    """Bar chart of class distribution."""
    classes = list(distribution.keys())
    counts = list(distribution.values())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(classes, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(c), ha="center", va="bottom", fontsize=9, fontweight="bold")

    total = sum(counts)
    ax.text(0.98, 0.95, f"Total: {total:,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved distribution plot → {save_path}")
    plt.show()
    return fig


def show_sample_images(dataset_path: str | Path, num_per_class: int = 4,
                       save_path: str | None = None):
    """Display a grid of random sample images from each class."""
    dataset_path = Path(dataset_path)
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    n_classes = len(classes)

    fig = plt.figure(figsize=(3 * num_per_class, 3 * n_classes))
    gs = gridspec.GridSpec(n_classes, num_per_class, hspace=0.4, wspace=0.1)

    for i, cls in enumerate(classes):
        cls_dir = dataset_path / cls
        files = [f for f in cls_dir.iterdir() if f.is_file()]
        samples = random.sample(files, min(num_per_class, len(files)))
        for j, img_path in enumerate(samples):
            ax = fig.add_subplot(gs[i, j])
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Error", ha="center", va="center")
            ax.set_title(cls if j == 0 else "", fontsize=10, fontweight="bold")
            ax.axis("off")

    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved sample grid → {save_path}")
    plt.show()
    return fig


# ── data integrity ──────────────────────────────────────────
def check_corrupted_images(dataset_path: str | Path) -> list:
    """Scan all images; return list of corrupted file paths."""
    dataset_path = Path(dataset_path)
    corrupted = []
    all_files = list(dataset_path.rglob("*"))
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    for f in tqdm(all_files, desc="Checking images"):
        if f.is_file() and f.suffix.lower() in img_exts:
            try:
                img = Image.open(f)
                img.verify()
            except Exception:
                corrupted.append(str(f))

    print(f"[INFO] Scanned {len(all_files)} files — {len(corrupted)} corrupted.")
    return corrupted


# ── train / val / test splitting ────────────────────────────
def split_dataset(source_dir: str | Path, dest_dir: str | Path,
                  ratios: tuple = (0.70, 0.15, 0.15), seed: int = 42):
    """
    Copy images from a flat folder‑per‑class source into
    dest/train/<cls>, dest/val/<cls>, dest/test/<cls>.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    random.seed(seed)

    splits = ("train", "val", "test")
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"

    classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    summary = {s: {} for s in splits}

    for cls in tqdm(classes, desc="Splitting dataset"):
        files = sorted([f for f in (source_dir / cls).iterdir() if f.is_file()])
        random.shuffle(files)
        n = len(files)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        split_files = {
            "train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:],
        }

        for split, flist in split_files.items():
            out_dir = dest_dir / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in flist:
                shutil.copy2(f, out_dir / f.name)
            summary[split][cls] = len(flist)

    # Print summary
    print("\n" + "=" * 50)
    print("SPLIT SUMMARY")
    print("=" * 50)
    for split in splits:
        total = sum(summary[split].values())
        print(f"\n{split.upper()} ({total} images):")
        for cls, cnt in summary[split].items():
            print(f"  {cls:15s} : {cnt}")
    print("=" * 50)
    return summary


# ── data transforms ─────────────────────────────────────────
def create_data_transforms(image_size: int = 224, augment: str = "standard"):
    """
    Return a dict of transforms for train / val / test.
    augment: 'standard' | 'strong' | 'none'
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment == "strong":
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif augment == "standard":
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    eval_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return {"train": train_tfm, "val": eval_tfm, "test": eval_tfm}


def preview_augmentations(dataset_path: str | Path, transform, num_images: int = 4,
                          num_augments: int = 5, save_path: str | None = None):
    """Show original vs augmented views for a few random images."""
    dataset_path = Path(dataset_path)
    all_images = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png"))
    samples = random.sample(all_images, min(num_images, len(all_images)))

    fig, axes = plt.subplots(num_images, num_augments + 1,
                             figsize=(3 * (num_augments + 1), 3 * num_images))
    if num_images == 1:
        axes = axes[np.newaxis, :]

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i, img_path in enumerate(samples):
        img = Image.open(img_path).convert("RGB")
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original", fontsize=9)
        axes[i, 0].axis("off")

        for j in range(1, num_augments + 1):
            aug_tensor = transform(img)
            aug_np = aug_tensor.permute(1, 2, 0).numpy() * std + mean
            aug_np = np.clip(aug_np, 0, 1)
            axes[i, j].imshow(aug_np)
            axes[i, j].set_title(f"Aug {j}", fontsize=9)
            axes[i, j].axis("off")

    fig.suptitle("Augmentation Preview", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ── dataloaders ─────────────────────────────────────────────
def build_dataloaders(dataset_dir: str | Path, transforms_dict: dict,
                      batch_size: int = 32, num_workers: int = 2,
                      pin_memory: bool = True):
    """
    Build train / val / test DataLoaders from an ImageFolder split.
    Returns dict of DataLoaders and the datasets.
    """
    dataset_dir = Path(dataset_dir)
    loaders, dsets = {}, {}

    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"[WARN] {split_dir} not found — skipping.")
            continue
        tfm = transforms_dict.get(split, transforms_dict.get("val"))
        ds = datasets.ImageFolder(str(split_dir), transform=tfm)
        shuffle = (split == "train")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory,
                        drop_last=(split == "train"))
        loaders[split] = dl
        dsets[split] = ds
        print(f"[INFO] {split:5s} → {len(ds):,} images, {len(dl)} batches")

    return loaders, dsets


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights from an ImageFolder dataset."""
    targets = [s[1] for s in dataset.samples]
    counts = Counter(targets)
    total = len(targets)
    n_classes = len(counts)
    weights = []
    for i in range(n_classes):
        w = total / (n_classes * counts[i])
        weights.append(w)
    weights = torch.FloatTensor(weights)
    print(f"[INFO] Class weights: {weights.tolist()}")
    return weights
