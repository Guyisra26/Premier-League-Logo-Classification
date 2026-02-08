"""
Training utilities for Premier League logo classification.

Functions:
    - set_seed: Set random seeds for reproducibility
    - train_one_epoch: Single epoch training step
    - evaluate: Model evaluation on a data loader
    - train_model: Full training loop with early stopping
    - count_parameters: Count total and trainable parameters
"""

import torch
import numpy as np
import random
import time
from copy import deepcopy


def set_seed(seed=42):
    """Set seed for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train model for one epoch.

    Returns:
        avg_loss (float): Average loss over the epoch
        accuracy (float): Classification accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a data loader.

    Returns:
        avg_loss (float): Average loss
        accuracy (float): Classification accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, epochs=30, patience=7, device='cuda'):
    """
    Full training loop with early stopping based on validation loss.

    Saves best model weights via deepcopy and restores them at the end.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Optional LR scheduler (stepped per epoch)
        epochs: Maximum number of epochs
        patience: Early stopping patience (epochs without improvement)
        device: 'cuda' or 'cpu'

    Returns:
        history (dict): Contains train_loss, train_acc, val_loss, val_acc
                        lists, plus best_epoch, best_val_acc, training_time
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'best_epoch': 0, 'best_val_acc': 0.0,
        'training_time': 0.0,
    }

    best_val_loss = float('inf')
    best_model_weights = deepcopy(model.state_dict())
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = deepcopy(model.state_dict())
            history['best_epoch'] = epoch + 1
            history['best_val_acc'] = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  Epoch [{epoch+1}/{epochs}]  "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
              f"{'  *' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(best epoch: {history['best_epoch']})")
            break

    history['training_time'] = time.time() - start_time
    model.load_state_dict(best_model_weights)
    print(f"  Training complete in {history['training_time']:.1f}s. "
          f"Best val acc: {history['best_val_acc']:.4f} "
          f"at epoch {history['best_epoch']}")
    return history


def count_parameters(model):
    """
    Count and print total and trainable parameters.

    Returns:
        total (int): Total parameter count
        trainable (int): Trainable parameter count
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,} | Trainable: {trainable:,} "
          f"| Frozen: {total - trainable:,}")
    return total, trainable
