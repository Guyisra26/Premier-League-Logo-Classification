"""
Visualization utilities for Premier League logo classification.

Functions:
    - plot_curves: Training/validation loss & accuracy for a single run
    - plot_comparison: Overlay multiple runs for comparison
    - show_samples: Grid of sample images per class
    - GradCAM: Grad-CAM heatmap generation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Convert a normalized tensor to a displayable numpy image."""
    img = img_tensor.numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


# ── Training Curves ────────────────────────────────────────────────────────────

def plot_curves(history, title='Training Curves'):
    """Plot train/val loss and accuracy curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    ax1.axvline(x=history['best_epoch'], color='gray', linestyle='--',
                alpha=0.5, label=f"Best epoch ({history['best_epoch']})")
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
    ax2.axvline(x=history['best_epoch'], color='gray', linestyle='--',
                alpha=0.5, label=f"Best epoch ({history['best_epoch']})")
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} — Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_comparison(histories, labels, title='Comparison'):
    """Overlay validation loss and accuracy from multiple training runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for hist, label, color in zip(histories, labels, colors):
        epochs = range(1, len(hist['train_loss']) + 1)
        ax1.plot(epochs, hist['val_loss'], label=label, linewidth=2, color=color)
        ax2.plot(epochs, hist['val_acc'], label=label, linewidth=2, color=color)

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation Loss')
    ax1.set_title(f'{title} — Val Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation Accuracy')
    ax2.set_title(f'{title} — Val Acc'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Sample Visualization ──────────────────────────────────────────────────────

def show_samples(dataset, class_names, n_per_class=3):
    """Display a grid of sample images — n_per_class columns per class row."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 2, n_classes * 2))

    class_indices = {i: [] for i in range(n_classes)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if len(class_indices[label]) < n_per_class:
            class_indices[label].append(idx)
        if all(len(v) >= n_per_class for v in class_indices.values()):
            break

    for i in range(n_classes):
        for j in range(n_per_class):
            img, _ = dataset[class_indices[i][j]]
            axes[i, j].imshow(_denormalize(img))
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(class_names[i], fontsize=8, loc='left')

    plt.suptitle('Sample Images per Class', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(class_counts):
    """Bar chart of images per class."""
    plt.figure(figsize=(14, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='steelblue')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylabel('Number of Images')
    plt.title('Images per Class')
    plt.tight_layout()
    plt.show()


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM: Visual explanations from deep networks.

    Usage:
        cam = GradCAM(model, target_layer)
        heatmap, predicted_class = cam.generate(input_tensor)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) tensor on the correct device
            target_class: Class index to explain (None = predicted class)

        Returns:
            cam (np.ndarray): Heatmap of shape (H, W), values in [0, 1]
            predicted_class (int): The class used for explanation
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy(), target_class


def visualize_gradcam(model, target_layer, test_loader, class_names,
                      device, n_samples=6):
    """
    Generate and display Grad-CAM visualizations for sample test images.

    Shows original image alongside Grad-CAM overlay.
    """
    grad_cam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(n_samples // 2, 4, figsize=(16, 4 * (n_samples // 2)))
    sample_count = 0

    for images, labels in test_loader:
        for i in range(min(n_samples - sample_count, images.size(0))):
            img_tensor = images[i:i+1].to(device)
            img_tensor.requires_grad_(True)
            cam, pred_class = grad_cam.generate(img_tensor)

            img_np = _denormalize(images[i])

            row = sample_count // 2
            col = (sample_count % 2) * 2

            axes[row, col].imshow(img_np)
            axes[row, col].set_title(
                f"True: {class_names[labels[i]]}\n"
                f"Pred: {class_names[pred_class]}", fontsize=8)
            axes[row, col].axis('off')

            axes[row, col+1].imshow(img_np)
            axes[row, col+1].imshow(cam, cmap='jet', alpha=0.5)
            axes[row, col+1].set_title('Grad-CAM', fontsize=8)
            axes[row, col+1].axis('off')

            sample_count += 1
            if sample_count >= n_samples:
                break
        if sample_count >= n_samples:
            break

    plt.suptitle('Grad-CAM Visualization', fontsize=14)
    plt.tight_layout()
    plt.show()
