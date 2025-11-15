"""
Patch Embedding Module for Vision Transformer.

This module implements the patch embedding layer that converts image patches
into embedding vectors using a convolutional layer.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer for Vision Transformer.
    
    Converts input images into patches and projects them into embedding space
    using a convolutional layer. The patches are then flattened and transposed
    to prepare for transformer processing.
    
    Args:
        num_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        embed_dim (int): Dimension of the embedding space
        patch_size (int): Size of each patch (patch_size x patch_size)
    
    Input shape:
        (batch_size, num_channels, img_size, img_size)
    
    Output shape:
        (batch_size, num_patches, embed_dim)
    """
    
    def __init__(self, num_channels: int, embed_dim: int, patch_size: int):
        """
        Initialize the Patch Embedding layer.
        
        Args:
            num_channels: Number of input channels
            embed_dim: Embedding dimension
            patch_size: Size of each patch
        """
        super().__init__()
        
        # Validate inputs
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Convolutional layer to project patches into embedding space
        # kernel_size=patch_size, stride=patch_size ensures non-overlapping patches
        self.patch_embed = nn.Conv2d(
            num_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        logger.info(
            f"Initialized PatchEmbedding: channels={num_channels}, "
            f"embed_dim={embed_dim}, patch_size={patch_size}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through patch embedding layer.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, img_size, img_size)
        
        Returns:
            Tensor of shape (batch_size, num_patches, embed_dim)
        
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (batch, channels, height, width), "
                f"got {x.dim()}D tensor"
            )
        
        _, channels, _, _ = x.shape
        
        if channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {channels}"
            )
        
        # Apply convolutional embedding: [B, C, H, W] -> [B, embed_dim, H', W']
        x = self.patch_embed(x)
        
        # Flatten spatial dimensions: [B, embed_dim, H', W'] -> [B, embed_dim, H'*W']
        x = x.flatten(2)
        
        # Transpose to get patches as sequence: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        return x
