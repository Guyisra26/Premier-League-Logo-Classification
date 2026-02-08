# Football Club Logo Classification — English Premier League

## Project Overview

This project tackles multi-class image classification of English Premier League football club logos using Convolutional Neural Networks implemented in PyTorch. The dataset, sourced from Kaggle (`alexteboul/english-premier-league-logo-detection-20k-images`), contains approximately 20,000 images spread across 20 club classes.

Logo classification is a fitting problem for CNNs for several reasons:

- Logos are visually distinct but share structural similarities (circular shapes, text, symbolic elements), which forces the network to learn fine-grained discriminative features rather than relying on trivial differences.
- The images contain variation in scale, rotation, background clutter, and quality — making it a non-trivial classification task that benefits from deeper architectures and regularization.
- With ~1,000 images per class, the dataset is large enough to train meaningful models from scratch, yet small enough that transfer learning should provide a measurable advantage — making it ideal for comparative experiments.

The 20-class setup also provides a reasonable challenge: it is harder than binary classification but does not require massive computational resources.

---

## Mapping to Course Requirements

The project is structured in three parts, each targeting specific course requirements. The table below maps each experiment to the requirement it addresses.

| Experiment | Requirement Addressed |
|---|---|
| Two custom CNN architectures trained from scratch | Designing and training CNNs without pretrained weights |
| Adam vs SGD+Momentum comparison | Comparing different optimizers and analyzing convergence |
| Batch Normalization ablation | Understanding the effect of normalization on training dynamics |
| Dropout and weight decay experiments | Applying regularization techniques and measuring their impact |
| Pretraining on an external dataset, then fine-tuning | Transfer learning via self-pretrained models |
| Fine-tuning a pretrained ResNet50 (ImageNet) | Transfer learning from a large-scale pretrained model |
| Training/validation curves, confusion matrices, per-class analysis | Experimental analysis and reasoned conclusions |

The ordering is intentional. Training from scratch first establishes baseline performance and exposes the limitations of custom models on this dataset. Transfer learning experiments then build on those baselines, making it possible to quantify the actual benefit of pretrained representations.

---

## Part 1 — Training CNNs from Scratch

### Architecture Design

Two architectures are compared, chosen to isolate the effect of network depth and capacity:

**Model A — Shallow CNN (SimpleCNN)**

- 3 convolutional blocks, each consisting of: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
- Filter progression: 32 -> 64 -> 128
- Followed by a fully connected head: Flatten -> Linear(512) -> ReLU -> Dropout -> Linear(20)
- Total parameters: roughly 1–2M depending on input resolution

**Model B — Deeper CNN (DeepCNN)**

- 5 convolutional blocks with the same structure as above
- Filter progression: 32 -> 64 -> 128 -> 256 -> 256
- Two residual-style skip connections (addition) at blocks 3 and 5
- Fully connected head: Global Average Pooling -> Linear(256) -> ReLU -> Dropout -> Linear(20)
- Total parameters: roughly 3–4M

The motivation behind the shallow model is to establish a minimal baseline — how well can a simple feature extractor perform on this task? The deeper model introduces skip connections to test whether increased depth translates to better feature learning without suffering from vanishing gradients.

Both models use 3x3 kernels throughout. Input images are resized to 128x128 (a practical compromise between resolution and training speed).

### Optimizer Comparison

Each architecture is trained with two optimizers:

- **Adam** (lr=1e-3, default betas) — adaptive learning rate, expected to converge faster
- **SGD + Momentum** (lr=1e-2, momentum=0.9) with StepLR scheduler (decay by 0.1 every 15 epochs) — expected to generalize better but converge more slowly

This comparison is designed to isolate the effect of the optimizer while keeping architecture, data, and hyperparameters fixed. The key question: does Adam's fast convergence come at the cost of generalization on this particular dataset?

### Batch Normalization and Regularization

To satisfy the requirement of analyzing normalization and regularization separately:

- **Batch Normalization ablation**: Train the DeepCNN with and without BatchNorm layers. The expectation is that removing BatchNorm will slow convergence and possibly destabilize training in the deeper model.
- **Dropout ablation**: Compare Dropout rates of 0.0, 0.3, and 0.5 in the fully connected head.
- **Weight decay**: Test L2 regularization (weight_decay=1e-4) with the Adam optimizer to measure its effect on overfitting.

These are run as controlled experiments — one variable changed at a time — so each result can be attributed to a specific factor.

### Hyperparameter Tuning Strategy

Rather than exhaustive grid search (impractical given compute constraints), tuning follows a sequential approach:

1. Fix architecture and optimizer, sweep learning rates: {1e-2, 1e-3, 1e-4}
2. With the best learning rate, sweep batch sizes: {32, 64, 128}
3. With those fixed, tune regularization (dropout rate, weight decay)

Training runs for 30–40 epochs. Early stopping based on validation loss (patience=7) prevents unnecessary computation and reduces overfitting risk.

### Expected Outputs

- Training and validation loss/accuracy curves for each configuration
- A comparison table of final validation accuracy across all combinations
- Identification of overfitting patterns (divergence between training and validation curves)

---

## Part 2 — Transfer Learning via Pretraining on an External Dataset

### Choice of External Dataset

**CIFAR-100** is used as the pretraining dataset. The reasoning:

- It contains 100 classes of natural images at 32x32 resolution, providing diverse low-level and mid-level visual features (edges, textures, shapes).
- It is structurally different from the logo dataset — natural photos vs. synthetic/graphic logos — which makes the transfer non-trivial and worth analyzing.
- An alternative was CIFAR-10, but CIFAR-100's larger class count forces the model to learn more discriminative feature representations.

### Pretraining Procedure

1. Train the DeepCNN on CIFAR-100 for 30 epochs (images upscaled to 128x128 to match the target input size).
2. Remove the final classification layer (which outputs 100 classes).
3. Attach a new classification head for 20 Premier League classes.
4. Fine-tune the entire network on the logo dataset with a reduced learning rate (lr=1e-4) to avoid destroying the pretrained features.

### Why Pretraining Helps (or Might Not)

The hypothesis is that CIFAR-100 pretraining provides useful low-level feature extractors (edge detectors, color filters) that transfer to logo images. However, since logos are graphically designed — not natural photos — mid-level and high-level features may transfer poorly.

This is precisely the interesting comparison: does pretraining on a mismatched domain still outperform random initialization? If so, it suggests that low-level features are domain-agnostic.

### Comparison Points

- Convergence speed: How many epochs does the pretrained model need to match the from-scratch baseline?
- Final accuracy: Does pretraining lead to a higher ceiling?
- Training curves: Does the pretrained model show smoother loss descent?

---

## Part 3 — Transfer Learning with Pretrained ResNet50

### Why ResNet50

ResNet50 pretrained on ImageNet provides a strong feature backbone trained on 1.2 million images across 1,000 classes. Compared to the custom CNNs:

- It has seen far more visual variety during training
- Its residual architecture enables very deep feature hierarchies (50 layers)
- It is the standard baseline for transfer learning benchmarks

Using ResNet50 directly tests how much a large-scale pretrained model can improve classification on a relatively small, domain-specific dataset.

### Adaptation Strategy

1. Load ResNet50 with `pretrained=True` (ImageNet weights).
2. Replace the final `fc` layer: `nn.Linear(2048, 20)`.
3. Resize input images to 224x224 (ResNet's expected input size).
4. Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

### Freezing vs. Fine-Tuning

Three strategies are compared:

- **Feature extraction (fully frozen)**: Freeze all ResNet layers, train only the new classification head. Fast training, but limited adaptation to the logo domain.
- **Partial unfreezing**: Freeze layers up to `layer3`, unfreeze `layer4` and the classification head. Allows high-level features to adapt while preserving low-level representations.
- **Full fine-tuning**: Unfreeze all layers with a low learning rate (lr=1e-5) and a higher learning rate for the new head (lr=1e-3). Uses differential learning rates to balance stability and adaptation.

The motivation: if feature extraction alone achieves high accuracy, it suggests that ImageNet features are highly transferable to logo classification. If full fine-tuning is needed, it implies the domain gap is significant.

### Comparison with Custom Models

The central question for this section: how much does a pretrained ResNet50 outperform the custom CNNs? The comparison considers:

- Raw accuracy on the test set
- Number of trainable parameters
- Training time per epoch
- Whether the performance gain justifies the additional complexity

---

## Experimental Analysis

### Data Split and Reproducibility

- The dataset is split into 70% train / 15% validation / 15% test.
- All splits are stratified to maintain class balance.
- A fixed random seed (42) is used throughout for reproducibility.
- Data augmentation on the training set includes: random horizontal flip, random rotation (up to 15 degrees), color jitter, and random resized crop.

### Metrics

- **Primary metric**: Top-1 accuracy on the test set
- **Secondary metrics**:
  - Per-class precision, recall, and F1-score (some clubs may have harder-to-distinguish logos)
  - Confusion matrix to identify which club pairs are most frequently confused
  - Training and validation loss curves to diagnose overfitting and convergence behavior

### Fair Comparison Protocol

All models are evaluated on the same held-out test set, which is never used during training or hyperparameter selection. Validation performance guides model selection; test performance is reported only once per final model.

For timing comparisons, all experiments are run on the same hardware (single GPU) and batch sizes are adjusted to maximize GPU utilization for each model.

### Summary Results Table

The final comparison table includes:

| Model | Optimizer | Test Accuracy | Parameters | Epochs to Converge | Notes |
|---|---|---|---|---|---|
| SimpleCNN | Adam | — | ~1.5M | — | Baseline |
| SimpleCNN | SGD+M | — | ~1.5M | — | |
| DeepCNN | Adam | — | ~3.5M | — | |
| DeepCNN | SGD+M | — | ~3.5M | — | |
| DeepCNN (no BN) | Adam | — | ~3.5M | — | BatchNorm ablation |
| DeepCNN (pretrained CIFAR-100) | Adam | — | ~3.5M | — | External pretraining |
| ResNet50 (frozen) | Adam | — | ~23.5M (0.04M trainable) | — | Feature extraction |
| ResNet50 (partial unfreeze) | Adam | — | ~23.5M (~8M trainable) | — | |
| ResNet50 (full fine-tune) | Adam | — | ~23.5M | — | |

(Values filled after running experiments.)

### Beyond Raw Accuracy

Several analyses go beyond simple accuracy comparisons:

- **Learning dynamics**: Plotting loss curves reveals whether a model is underfitting (high training loss), overfitting (large train-val gap), or well-fitted. This is more informative than final accuracy alone.
- **Class-level analysis**: If certain logos (e.g., similarly colored crests) are consistently misclassified, it points to limitations in the learned representations.
- **Feature visualization**: Grad-CAM applied to the best-performing model shows which regions of the logo the network attends to — useful for verifying that the model learns meaningful features rather than background artifacts.
- **Effect of dataset size**: Training the best model on 25%, 50%, 75%, and 100% of the training data reveals how data-hungry each approach is. Transfer learning models are expected to degrade more gracefully under data scarcity.

---

## Final Reflection

### What Architectural Choices Mattered Most

Based on the experiments, the factors expected to have the largest impact (in rough order):

1. **Pretrained weights** — The gap between training from scratch and fine-tuning a pretrained ResNet50 is likely the largest single factor. ImageNet features provide a massive head start.
2. **Network depth with skip connections** — The DeepCNN should outperform the SimpleCNN, but the margin depends on how well skip connections mitigate gradient issues.
3. **Batch Normalization** — Removing it from the deeper model will likely cause noticeable degradation in training stability.
4. **Optimizer choice** — Adam is expected to converge faster, but SGD+Momentum may reach comparable or slightly better final accuracy with proper scheduling.
5. **Regularization** — Dropout and weight decay help close the train-val gap, but their impact is secondary to architecture and initialization.

### When Transfer Learning Helped and When It Didn't

- **CIFAR-100 pretraining**: Likely provides a modest improvement over random initialization, mainly through better low-level features. The domain gap (natural images vs. graphic logos) limits the benefit of higher-level features.
- **ResNet50 (ImageNet)**: Expected to provide the strongest results, especially in data-scarce scenarios. However, if the dataset is large and varied enough, a well-tuned custom CNN may close the gap more than anticipated.

An important takeaway: transfer learning is not a universal solution. Its effectiveness depends heavily on the similarity between the source and target domains.

### What Could Be Improved

- **Data**: Collecting logos at higher resolution and with more diverse backgrounds would increase task difficulty and model robustness.
- **Architectures**: Testing EfficientNet or Vision Transformers (ViT) as alternatives to ResNet50 could yield better transfer learning results.
- **Augmentation**: More aggressive augmentation strategies (CutMix, MixUp) might help the from-scratch models close the gap with transfer learning.
- **Ensemble methods**: Combining predictions from multiple models (e.g., DeepCNN + ResNet50) could push accuracy further, though this was outside the project scope.
- **Pretraining dataset selection**: Using a logo-specific external dataset (e.g., FlickrLogos-32) instead of CIFAR-100 for the self-pretraining experiment would test whether domain-matched pretraining outperforms general pretraining.
