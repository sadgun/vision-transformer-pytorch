"""
MLP Head Module for Vision Transformer.

This module implements the classification head that takes the CLS token
and produces class predictions.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MLPHead(nn.Module):
    """
    MLP Head for classification in Vision Transformer.
    
    Takes the CLS token embedding and applies layer normalization followed
    by a linear layer to produce class logits.
    
    Args:
        embed_dim (int): Dimension of the embedding space
        num_classes (int): Number of output classes
    
    Input shape:
        (batch_size, embed_dim)
    
    Output shape:
        (batch_size, num_classes)
    """
    
    def __init__(self, embed_dim: int, num_classes: int):
        """
        Initialize the MLP Head.
        
        Args:
            embed_dim: Embedding dimension
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Classification head (linear layer)
        self.mlp_head = nn.Linear(embed_dim, num_classes)
        
        logger.info(
            f"Initialized MLPHead: embed_dim={embed_dim}, num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP head.
        
        Args:
            x: Input tensor of shape (batch_size, embed_dim)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor (batch, embed_dim), "
                f"got {x.dim()}D tensor"
            )
        
        _, embed_dim = x.shape
        
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {embed_dim}"
            )
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply classification head
        x = self.mlp_head(x)
        
        return x
