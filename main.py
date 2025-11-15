"""
Main entry point for Vision Transformer training and evaluation.

This script orchestrates the complete training pipeline including:
- Data loading
- Model initialization
- Training
- Evaluation
- Visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os

# Import configuration
import config

# Import data utilities
from data import load_mnist_dataset, create_dataloaders

# Import model
from models import VisionTransformer

# Import training and evaluation
import train
import evaluate
import visualize


def setup_logging(log_level: str = "INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for training (CPU or CUDA).
    
    Args:
        use_cuda: Whether to use CUDA if available
    
    Returns:
        torch.device: Device to use for training
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if use_cuda:
            logging.warning("CUDA requested but not available, using CPU")
        else:
            logging.info("Using CPU device")
    
    return device


def create_model(device: torch.device) -> nn.Module:
    """
    Create and initialize the Vision Transformer model.
    
    Args:
        device: Device to place the model on
    
    Returns:
        nn.Module: Initialized Vision Transformer model
    """
    logging.info("Creating Vision Transformer model...")
    
    try:
        model = VisionTransformer(
            num_classes=config.NUM_CLASSES,
            num_channels=config.NUM_CHANNELS,
            img_size=config.IMG_SIZE,
            patch_size=config.PATCH_SIZE,
            embed_dim=config.EMBED_DIM,
            attention_heads=config.ATTENTION_HEADS,
            transformer_blocks=config.TRANSFORMER_BLOCKS,
            mlp_hidden_nodes=config.MLP_HIDDEN_NODES
        )
        
        model = model.to(device)
        
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(
            f"Model created successfully: "
            f"Total parameters: {total_params:,}, "
            f"Trainable parameters: {trainable_params:,}"
        )
        
        return model
    
    except Exception as e:
        logging.error(f"Failed to create model: {e}")
        raise


def main():
    """
    Main function to run the complete training pipeline.
    """
    # Set up logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Vision Transformer Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Get device
        device = get_device(config.USE_CUDA)
        
        # Load datasets
        logger.info("Loading datasets...")
        train_dataset = load_mnist_dataset(
            data_root=config.DATA_ROOT,
            train=True,
            download=True
        )
        val_dataset = load_mnist_dataset(
            data_root=config.DATA_ROOT,
            train=False,
            download=True
        )
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle_train=config.SHUFFLE_TRAIN,
            shuffle_val=config.SHUFFLE_VAL
        )
        
        # Create model
        model = create_model(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        logger.info(
            f"Training configuration: "
            f"Epochs={config.EPOCHS}, "
            f"Batch size={config.BATCH_SIZE}, "
            f"Learning rate={config.LEARNING_RATE}"
        )
        
        # Create output directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Train model
        logger.info("Starting training...")
        _ = train.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=config.EPOCHS,
            log_interval=config.LOG_INTERVAL,
            save_path="checkpoints/vit_model.pth"
        )
        
        # Final evaluation
        logger.info("Running final evaluation...")
        val_loss, val_accuracy = evaluate.evaluate_model(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        logger.info(f"Final validation accuracy: {val_accuracy:.2f}%")
        
        # Visualize predictions
        logger.info("Generating visualizations...")
        visualize.visualize_predictions(
            model=model,
            val_loader=val_loader,
            device=device,
            num_samples=10,
            save_path="results/predictions.png"
        )
        
        logger.info("=" * 60)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 60)
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
