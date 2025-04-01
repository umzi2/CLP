# Contrastive Learning Perceptual Loss

Contrastive Learning Perceptual Loss (CLPerceptualLoss) is a custom loss function designed for perceptual image similarity evaluation. It utilizes contrastive learning by applying **Triplet Margin Loss** to feature embeddings extracted from pre-trained **VGG19** and **DINOv2** models. This loss is particularly useful for super-resolution (SR) and other image restoration tasks.

## Description

This module supports two feature extraction models:
- **VGG19** — a convolutional neural network commonly used for perceptual loss in image restoration tasks.
- **DINOv2** — a transformer-based vision model trained via self-supervised contrastive learning, providing more robust feature representations.

The loss function operates by comparing the super-resolved image (**SR**) with the reference (**Target**) and the low-quality (**LR**) image. Through contrastive learning, it ensures that the SR image is closer to the Target while maintaining separation from the LR image.

## Usage

```python
import torch
from clperceptual_loss import CLPLoss

# Create the loss function object
criterion = CLP(model="dinov2")

# Input tensors (image data)
sr = torch.randn(1, 3, 256, 256)  # Super-resolved image
target = torch.randn(1, 3, 256, 256)  # Reference image
lr = torch.randn(1, 3, 128, 128)  # Low-quality image

# Compute loss
loss = criterion(sr, target, lr)
print("Loss:", loss.item())
```

## Architecture

- **VGG** extracts features from multiple convolutional layers (ReLU activations at different depths).
- **DINOv2** provides feature maps from its transformer-based architecture, capturing higher-level semantic information.
- The extracted feature maps are weighted and passed to `TripletMarginLoss`, enforcing contrastive learning constraints.

## Advantages

- **Feature-level Perceptual Loss**: Uses deep feature representations rather than pixel-wise comparisons.
- **Contrastive Learning Framework**: Ensures SR images remain distinct from LR images while staying close to the target.
- **Multi-Scale Feature Extraction**: Different layers contribute varying levels of abstraction, improving generalization.

## Configuration

When creating the `CLPLoss` object, you can configure the following parameters:

```python
criterion = CLPLoss(model="vgg", loss_weight=0.8)
```
- `model`: selects the pre-trained model (`"vgg"` or `"dinov2"`).
- `loss_weight`: multiplier for adjusting the final loss value.

## License

This project is released under the MIT License.

