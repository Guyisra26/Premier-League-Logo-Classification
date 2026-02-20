"""
Data loading utilities for Premier League logo classification.

Handles dataset discovery, transforms, stratified splitting, and DataLoader creation
for both custom CNNs (128x128) and ResNet50 (224x224).
"""

import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE_CNN = 128
IMG_SIZE_RESNET = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_cnn_transforms():
    """Return (train_transform, val_transform) for custom CNNs at 128x128."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(IMG_SIZE_CNN, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, val_transform


def get_resnet_transforms():
    """Return (train_transform, val_transform) for ResNet50 at 224x224."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_RESNET, IMG_SIZE_RESNET)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(IMG_SIZE_RESNET, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_RESNET, IMG_SIZE_RESNET)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, val_transform


def get_cnn_transforms_no_aug():
    """Return (train_transform, val_transform) without data augmentation for ablation."""
    plain_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return plain_transform, plain_transform


def get_cifar100_transforms():
    """Return (train_transform, val_transform) for CIFAR-100 upscaled to 128x128."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE_CNN, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    return train_transform, val_transform


# ── Dataset Discovery ──────────────────────────────────────────────────────────

def find_data_dir(root='./data'):
    """Walk the data directory to find the folder containing class sub-folders."""
    for dirpath, dirs, files in os.walk(root):
        # Expect ~20 class folders
        if len(dirs) >= 15:
            return dirpath
    return root


def explore_dataset(data_dir):
    """
    Print dataset statistics: class names, image counts, total.

    Returns:
        class_names (list[str]): Sorted list of class folder names
        class_counts (dict): {class_name: num_images}
    """
    class_names = sorted([
        c for c in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, c))
    ])
    class_counts = {}
    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        n = len([f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
        class_counts[cls] = n

    print(f"Dataset directory: {data_dir}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Total images: {sum(class_counts.values())}\n")
    print("Images per class:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count}")

    return class_names, class_counts


# ── Stratified Splitting ───────────────────────────────────────────────────────

def stratified_split(dataset, seed=42, train_ratio=0.70, val_ratio=0.15):
    """
    Create stratified train/val/test index splits.

    Args:
        dataset: ImageFolder dataset (used for .targets)
        seed: Random seed
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test = 1 - train - val)

    Returns:
        train_idx, val_idx, test_idx (numpy arrays)
    """
    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    test_ratio = 1.0 - train_ratio - val_ratio
    temp_ratio = val_ratio + test_ratio  # what goes into the second split

    train_idx, temp_idx = train_test_split(
        indices, test_size=temp_ratio, stratify=targets, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio / temp_ratio,
        stratify=targets[temp_idx], random_state=seed
    )

    # Verification
    for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        unique, counts = np.unique(targets[idx], return_counts=True)
        print(f"  {name}: {len(idx)} samples  "
              f"(min/class={counts.min()}, max/class={counts.max()})")

    return train_idx, val_idx, test_idx


# ── DataLoader Creation ────────────────────────────────────────────────────────

def create_loaders(data_dir, train_idx, val_idx, test_idx,
                   train_transform, val_transform,
                   batch_size=64, num_workers=2):
    """
    Create train/val/test DataLoaders using pre-computed index splits.

    Uses separate ImageFolder instances so train gets augmentation
    while val/test get deterministic transforms.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def load_cifar100(batch_size=64, num_workers=2):
    """
    Download and load CIFAR-100 with appropriate transforms.

    Returns:
        train_loader, val_loader
    """
    train_tf, val_tf = get_cifar100_transforms()

    train_set = datasets.CIFAR100(root='./data/cifar100', train=True,
                                  download=True, transform=train_tf)
    val_set = datasets.CIFAR100(root='./data/cifar100', train=False,
                                download=True, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"CIFAR-100 loaded — Train: {len(train_set)}, Test: {len(val_set)}")
    return train_loader, val_loader
