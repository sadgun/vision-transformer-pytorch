"""
Visualization utilities for Vision Transformer.

This module provides functions for visualizing model predictions and results.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def visualize_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize model predictions on validation samples.
    
    Args:
        model: The trained model
        val_loader: DataLoader for validation data
        device: Device to run inference on (CPU or CUDA)
        num_samples: Number of samples to visualize
        save_path: Optional path to save the visualization
    
    Raises:
        ValueError: If num_samples is invalid
        RuntimeError: If visualization fails
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    
    logger.info(f"Generating predictions visualization for {num_samples} samples")
    
    try:
        model.eval()
        
        # Get a batch of data
        images, labels = next(iter(val_loader))
        images = images.to(device)
        labels = labels.to(device)
        
        # Limit to available samples
        num_samples = min(num_samples, images.size(0))
        
        # Get predictions
        with torch.no_grad():
            outputs = model(images[:num_samples])
            preds = outputs.argmax(dim=1)
        
        # Move to CPU for visualization
        images = images[:num_samples].cpu()
        preds = preds.cpu()
        labels = labels[:num_samples].cpu()
        
        # Create visualization
        _, axes = plt.subplots(2, 5, figsize=(12, 4))
        axes = axes.flatten()
        
        for i in range(num_samples):
            ax = axes[i]
            ax.imshow(images[i].squeeze(), cmap="gray")
            
            # Color code: green for correct, red for incorrect
            color = "green" if preds[i].item() == labels[i].item() else "red"
            ax.set_title(
                f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}",
                color=color
            )
            ax.axis("off")
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save visualization to {save_path}: {e}")
                plt.show()
        else:
            plt.show()
        
        plt.close()
        
        # Log accuracy for visualized samples
        correct = (preds == labels).sum().item()
        vis_accuracy = 100.0 * correct / num_samples
        logger.info(
            f"Visualization accuracy: {vis_accuracy:.2f}% ({correct}/{num_samples} correct)"
        )
    
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise RuntimeError(f"Visualization failed: {e}") from e
