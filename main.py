"""
main.py — DermaScan AI Training Script
=======================================
ResNet18 fine-tuned on the HAM10000 dermoscopy dataset for 7-class
skin lesion classification.

Key features
------------
* CUDA auto-detection with Automatic Mixed Precision (AMP)
* WeightedRandomSampler to combat severe class imbalance (nv ~67%)
* Heavy data augmentation for improved generalisation
* Best-epoch model saved in FP16 → strictly < 25 MB for GitHub

3rd-Year Engineering Project — B.E. Computer Engineering | 2025–26
"""

import os
import time
import platform
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CSV_PATH     = "dataset/HAM10000_metadata.csv"
IMG_DIR      = "dataset/images"
MODEL_SAVE   = "model_final.pth"

IMG_SIZE     = 224
BATCH_SIZE   = 64
EPOCHS       = 25
LR           = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42

# HAM10000 label → integer index mapping
CLASS_MAP = {
    "nv":    0,   # Melanocytic nevi          (benign,      ~67 % of dataset)
    "mel":   1,   # Melanoma                  (malignant)
    "bkl":   2,   # Benign keratosis-like     (benign)
    "bcc":   3,   # Basal cell carcinoma      (malignant)
    "akiec": 4,   # Actinic keratoses         (pre-cancerous)
    "vasc":  5,   # Vascular lesions          (benign)
    "df":    6,   # Dermatofibroma            (benign)
}
NUM_CLASSES = len(CLASS_MAP)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
#  Device & AMP Setup
# ═══════════════════════════════════════════════════════════════════════════════

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"

# On Windows, multiprocessing-based workers can cause issues with CUDA;
# use 0 workers (main process loads data) for safety.
NUM_WORKERS = 0 if platform.system() == "Windows" else 4

print("=" * 70)
print("  DermaScan AI — Training Configuration")
print("=" * 70)
print(f"  Device      : {device}" + (f" ({torch.cuda.get_device_name(0)})" if USE_AMP else ""))
print(f"  AMP enabled : {USE_AMP}")
print(f"  Workers     : {NUM_WORKERS}")
print(f"  Epochs      : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 skin lesion benchmark.

    Each sample is a tuple (image_tensor, class_index) where class_index
    maps to one of the seven dermoscopy diagnoses defined in CLASS_MAP.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Subset of the HAM10000 metadata CSV (columns: image_id, dx).
    img_dir : str
        Path to the folder containing <image_id>.jpg files.
    transform : callable, optional
        Torchvision transform pipeline applied to each image.
    """

    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")
        label    = CLASS_MAP[row["dx"]]
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ═══════════════════════════════════════════════════════════════════════════════
#  Transforms
# ═══════════════════════════════════════════════════════════════════════════════

_MEAN = [0.485, 0.456, 0.406]   # ImageNet statistics
_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ═══════════════════════════════════════════════════════════════════════════════
#  Weighted Random Sampler
# ═══════════════════════════════════════════════════════════════════════════════

def build_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Builds a WeightedRandomSampler to counteract class imbalance.

    Rare cancer classes (mel, bcc, akiec) receive proportionally higher
    sampling probability so each training batch approximates a balanced
    class distribution — without modifying the dataset itself.

    Parameters
    ----------
    labels : list of int
        Integer class labels for every sample in the training set.

    Returns
    -------
    WeightedRandomSampler
        Sampler with per-sample weights; replacement=True so that
        under-represented classes can appear multiple times per epoch.
    """
    class_counts  = Counter(labels)
    class_weights = {cls: 1.0 / cnt for cls, cnt in class_counts.items()}
    sample_weights = torch.tensor(
        [class_weights[lbl] for lbl in labels], dtype=torch.float
    )
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Reads the HAM10000 metadata CSV, removes duplicate lesions to prevent
    data leakage, performs a stratified 80/20 train/val split, and wraps
    both subsets in DataLoaders.

    The training loader uses WeightedRandomSampler (not shuffle=True) to
    ensure balanced class exposure every epoch.  pin_memory=True is enabled
    when a CUDA device is present to accelerate host→GPU tensor transfers.

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
    """
    df = pd.read_csv(CSV_PATH).dropna(subset=["dx"])

    # Keep one image per unique lesion — prevents train/val contamination
    df = df.drop_duplicates(subset="lesion_id", keep="first")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["dx"]
    )

    train_labels = [CLASS_MAP[d] for d in train_df["dx"]]
    sampler      = build_weighted_sampler(train_labels)

    pin = (device.type == "cuda")

    train_ds = HAM10000Dataset(train_df, IMG_DIR, transform=train_tf)
    val_ds   = HAM10000Dataset(val_df,   IMG_DIR, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,       # mutually exclusive with shuffle=True
        num_workers = NUM_WORKERS,
        pin_memory  = pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = pin,
    )

    print(f"  Train samples : {len(train_ds):,}")
    print(f"  Val   samples : {len(val_ds):,}")
    class_dist = Counter(train_labels)
    print(f"  Class dist    : { {list(CLASS_MAP.keys())[k]: v for k, v in sorted(class_dist.items())} }")

    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

def build_model() -> nn.Module:
    """
    Builds a ResNet18 classifier for 7-class skin lesion detection.

    Pre-trained ImageNet weights are used for the convolutional backbone
    (transfer learning).  The original 1000-class head is replaced with:

        Dropout(0.4) → Linear(512 → 7)

    Dropout reduces overfitting on the relatively small HAM10000 dataset
    (~8,000 unique lesions after deduplication).

    Returns
    -------
    nn.Module
        Model moved to the active device.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Validation Loops
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, scaler):
    """
    Runs one full training epoch with optional AMP (Automatic Mixed Precision).

    AMP is activated automatically whenever a CUDA device is detected,
    halving GPU memory consumption and accelerating throughput on Tensor Cores.

    Parameters
    ----------
    model      : nn.Module
    loader     : DataLoader  — training DataLoader
    optimizer  : torch.optim.Optimizer
    criterion  : nn.Module   — loss function (CrossEntropyLoss)
    scaler     : torch.cuda.amp.GradScaler

    Returns
    -------
    avg_loss : float
    accuracy : float  — as a percentage (0–100)
    """
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total * 100.0


@torch.no_grad()
def validate_epoch(model, loader, criterion):
    """
    Evaluates the model on the validation set without gradient computation.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader  — validation DataLoader
    criterion : nn.Module

    Returns
    -------
    avg_loss : float
    accuracy : float  — as a percentage (0–100)
    """
    model.eval()
    total_loss = correct = total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total * 100.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Orchestrates data loading, model construction, the training loop,
    and final model export.

    The best-performing epoch (by validation accuracy) is exported as
    FP16 weights (model.half()) to keep the file strictly under 25 MB,
    satisfying GitHub's single-file size limit for direct deployment.
    ResNet18 in FP32 ≈ 44 MB; in FP16 ≈ 22 MB.
    """
    train_loader, val_loader = load_data()

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_acc = 0.0

    print("=" * 70)
    print(f"{'Epoch':^7} {'Tr Loss':^10} {'Tr Acc':^9} {'Vl Loss':^10} {'Vl Acc':^9} {'Time':^7}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc = validate_epoch(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        marker  = " ✔" if vl_acc > best_val_acc else ""

        print(
            f"{epoch:^7d} {tr_loss:^10.4f} {tr_acc:^9.2f} "
            f"{vl_loss:^10.4f} {vl_acc:^9.2f} {elapsed:^6.0f}s{marker}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            # ── FP16 export: ResNet18 FP16 ≈ 22 MB < 25 MB GitHub limit ──
            model.half()                                        # cast to float16
            torch.save(model.state_dict(), MODEL_SAVE)
            model.float()                                       # restore float32 for next epoch

    print("=" * 70)
    print(f"  Training complete.  Best validation accuracy : {best_val_acc:.2f}%")
    print(f"  FP16 weights saved to                        : {MODEL_SAVE}")
    size_mb = os.path.getsize(MODEL_SAVE) / (1024 ** 2)
    print(f"  Saved model file size                        : {size_mb:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
