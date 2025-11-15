"""
Training module for Vision Transformer.

This module handles the training loop with logging, error checking, and
progress tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from typing import Optional

# Import evaluate module (avoid circular import by importing inside function if needed)
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 100
):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        device: Device to run training on (CPU or CUDA)
        epoch: Current epoch number
        log_interval: Interval for logging progress (in batches)
    
    Returns:
        tuple: (average_loss, accuracy) for the epoch
    
    Raises:
        RuntimeError: If training fails
    """
    model.train()
    total_loss = 0.0
    correct_epoch = 0
    total_epoch = 0
    
    logger.info(f"Starting training epoch {epoch + 1}")
    
    try:
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            
            # Zero gradients
            optimizer.zero_grad()
            
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
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            accuracy = 100.0 * correct / labels.size(0)
            
            correct_epoch += correct
            total_epoch += labels.size(0)
            
            # Log progress
            if batch_idx % log_interval == 0:
                logger.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1:3d}/{len(train_loader)}: "
                    f"Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%"
                )
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct_epoch / total_epoch
        
        logger.info(
            f"Epoch {epoch + 1} completed: Average Loss = {avg_loss:.4f}, "
            f"Accuracy = {epoch_accuracy:.2f}%"
        )
        
        return avg_loss, epoch_accuracy
    
    except Exception as e:
        logger.error(f"Error during training epoch {epoch + 1}: {e}")
        raise RuntimeError(f"Training failed: {e}") from e


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    log_interval: int = 100,
    save_path: Optional[str] = None
):
    """
    Train the model for multiple epochs.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        device: Device to run training on (CPU or CUDA)
        epochs: Number of epochs to train
        log_interval: Interval for logging progress (in batches)
        save_path: Optional path to save the trained model
    
    Returns:
        dict: Training history with losses and accuracies
    
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If training fails
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    
    logger.info(
        f"Starting training: epochs={epochs}, device={device}, "
        f"model_parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    try:
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                log_interval=log_interval
            )
            
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            # Validate after each epoch
            # Import here to avoid circular import issues
            import evaluate
            val_loss, val_acc = evaluate.evaluate_model(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device
            )
            
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch + 1}/{epochs} Summary - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
        
        # Save model if path provided
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history
                }, save_path)
                logger.info(f"Model saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save model to {save_path}: {e}")
        
        logger.info("Training completed successfully")
        return history
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Training failed: {e}") from e
