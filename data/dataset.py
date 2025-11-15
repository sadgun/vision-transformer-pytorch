"""
Dataset loading and preprocessing module.

This module handles data loading, preprocessing, and DataLoader creation
for the Vision Transformer training pipeline.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
import os

logger = logging.getLogger(__name__)


def get_transforms():
    """
    Get data transformation pipeline.
    
    Returns:
        transforms.Compose: Transformation pipeline for preprocessing images
    """
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert PIL Image to tensor and normalize to [0, 1]
    ])
    
    logger.info("Created data transformation pipeline: ToTensor()")
    return transform


def load_mnist_dataset(data_root: str, train: bool = True, download: bool = True):
    """
    Load MNIST dataset.
    
    Args:
        data_root (str): Root directory for storing/downloading dataset
        train (bool): If True, load training set; otherwise load test set
        download (bool): If True, download dataset if not present
    
    Returns:
        torchvision.datasets.MNIST: MNIST dataset
    
    Raises:
        ValueError: If data_root is invalid
        RuntimeError: If dataset loading fails
    """
    if not isinstance(data_root, str) or not data_root:
        raise ValueError(f"data_root must be a non-empty string, got {data_root}")
    
    try:
        transform = get_transforms()
        
        dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=train,
            download=download,
            transform=transform
        )
        
        split_name = "training" if train else "validation"
        logger.info(
            f"Loaded MNIST {split_name} dataset: {len(dataset)} samples, "
            f"stored at {os.path.abspath(data_root)}"
        )
        
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load MNIST dataset: {e}")
        raise RuntimeError(f"Error loading MNIST dataset: {e}") from e


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    shuffle_train: bool = True,
    shuffle_val: bool = True,
    num_workers: int = 0
):
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size (int): Batch size for DataLoaders
        shuffle_train (bool): Whether to shuffle training data
        shuffle_val (bool): Whether to shuffle validation data
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        tuple: (train_loader, val_loader) DataLoader objects
    
    Raises:
        ValueError: If batch_size is invalid
        TypeError: If datasets are not valid
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    if not hasattr(train_dataset, '__len__'):
        raise TypeError("train_dataset must be a valid dataset with __len__ method")
    if not hasattr(val_dataset, '__len__'):
        raise TypeError("val_dataset must be a valid dataset with __len__ method")
    
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()  # Pin memory for faster GPU transfer
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle_val,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(
            f"Created DataLoaders: batch_size={batch_size}, "
            f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
            f"num_workers={num_workers}"
        )
        
        return train_loader, val_loader
    
    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {e}")
        raise RuntimeError(f"Error creating DataLoaders: {e}") from e
