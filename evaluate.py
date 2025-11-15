"""
Evaluation module for Vision Transformer.

This module handles model evaluation on validation/test sets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """
    Evaluate the model on validation/test set.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation/test data
        criterion: Loss function
        device: Device to run evaluation on (CPU or CUDA)
    
    Returns:
        tuple: (average_loss, accuracy) for the validation set
    
    Raises:
        RuntimeError: If evaluation fails
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    logger.info("Starting model evaluation")
    
    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
                
                # Validate input shapes
                if images.dim() != 4:
                    raise ValueError(
                        f"Expected 4D image tensor, got {images.dim()}D"
                    )
                if labels.dim() != 1:
                    raise ValueError(
                        f"Expected 1D label tensor, got {labels.dim()}D"
                    )
                
                # Forward pass
                outputs = model(images)
                
                # Validate output shape
                if outputs.dim() != 2 or outputs.size(1) != model.num_classes:
                    raise ValueError(
                        f"Model output shape mismatch: expected (batch, {model.num_classes}), "
                        f"got {outputs.shape}"
                    )
                
                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Log progress periodically
                if (batch_idx + 1) % 100 == 0:
                    batch_acc = 100.0 * (preds == labels).sum().item() / labels.size(0)
                    logger.debug(
                        f"Evaluation batch {batch_idx + 1}/{len(val_loader)}: "
                        f"Loss = {loss.item():.4f}, Accuracy = {batch_acc:.2f}%"
                    )
        
        # Calculate final metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        logger.info(
            f"Evaluation completed: Average Loss = {avg_loss:.4f}, "
            f"Accuracy = {accuracy:.2f}% ({correct}/{total} correct)"
        )
        
        return avg_loss, accuracy
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise RuntimeError(f"Evaluation failed: {e}") from e
