"""
Vision Transformer (ViT) Model Implementation.

This module implements the complete Vision Transformer architecture that combines
patch embedding, positional encoding, transformer encoder blocks, and classification head.
"""

import torch
import torch.nn as nn
import logging

from .patch_embedding import PatchEmbedding
from .transformer_encoder import TransformerEncoder
from .mlp_head import MLPHead

logger = logging.getLogger(__name__)


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.
    
    The model processes images by:
    1. Splitting images into patches and embedding them
    2. Adding a learnable CLS token and positional embeddings
    3. Processing through multiple transformer encoder blocks
    4. Using the CLS token for final classification
    
    Args:
        num_classes (int): Number of output classes
        num_channels (int): Number of input channels
        img_size (int): Input image size (assumed square)
        patch_size (int): Size of each patch
        embed_dim (int): Embedding dimension
        attention_heads (int): Number of attention heads
        transformer_blocks (int): Number of transformer encoder blocks
        mlp_hidden_nodes (int): Hidden layer size in MLP
    
    Input shape:
        (batch_size, num_channels, img_size, img_size)
    
    Output shape:
        (batch_size, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        attention_heads: int,
        transformer_blocks: int,
        mlp_hidden_nodes: int
    ):
        """
        Initialize the Vision Transformer model.
        
        Args:
            num_classes: Number of output classes
            num_channels: Number of input channels
            img_size: Input image size
            patch_size: Size of each patch
            embed_dim: Embedding dimension
            attention_heads: Number of attention heads
            transformer_blocks: Number of transformer encoder blocks
            mlp_hidden_nodes: Hidden layer size in MLP
        """
        super().__init__()
        
        # Validate inputs
        if img_size <= 0 or img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be positive and divisible by patch_size ({patch_size})"
            )
        
        num_patches = (img_size // patch_size) ** 2
        
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(
            num_channels=num_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        
        # Learnable CLS token (classification token)
        # Shape: (1, 1, embed_dim) - will be expanded to (batch_size, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Learnable positional embeddings
        # Shape: (1, 1 + num_patches, embed_dim) - 1 for CLS token + num_patches for patches
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        
        # Stack of transformer encoder blocks
        self.transformer_blocks = nn.Sequential(*[
            TransformerEncoder(
                embed_dim=embed_dim,
                attention_heads=attention_heads,
                mlp_hidden_nodes=mlp_hidden_nodes
            )
            for _ in range(transformer_blocks)
        ])
        
        # Classification head
        self.mlp_head = MLPHead(
            embed_dim=embed_dim,
            num_classes=num_classes
        )
        
        logger.info(
            f"Initialized VisionTransformer: num_classes={num_classes}, "
            f"img_size={img_size}, patch_size={patch_size}, embed_dim={embed_dim}, "
            f"num_patches={num_patches}, transformer_blocks={transformer_blocks}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Vision Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, img_size, img_size)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (batch, channels, height, width), "
                f"got {x.dim()}D tensor"
            )
        
        batch_size, channels, height, width = x.shape
        
        if channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {channels}"
            )
        
        if height != self.img_size or width != self.img_size:
            raise ValueError(
                f"Expected image size {self.img_size}x{self.img_size}, "
                f"got {height}x{width}"
            )
        
        # Step 1: Convert image to patch embeddings
        # Shape: (batch_size, num_patches, embed_dim)
        x = self.patch_embedding(x)
        
        # Step 2: Add CLS token
        # Expand CLS token to batch size: (1, 1, embed_dim) -> (batch_size, 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate CLS token with patch embeddings
        # Shape: (batch_size, 1 + num_patches, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Step 3: Add positional embeddings
        x = x + self.pos_embedding
        
        # Step 4: Process through transformer encoder blocks
        # Shape: (batch_size, 1 + num_patches, embed_dim)
        x = self.transformer_blocks(x)
        
        # Step 5: Extract CLS token (first token) for classification
        # Shape: (batch_size, embed_dim)
        x = x[:, 0]
        
        # Step 6: Apply classification head
        # Shape: (batch_size, num_classes)
        x = self.mlp_head(x)
        
        return x
