"""
Transformer Encoder Module for Vision Transformer.

This module implements a single transformer encoder block with multi-head self-attention
and a feed-forward MLP, following the standard transformer architecture with residual
connections and layer normalization.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder block for Vision Transformer.
    
    Implements a single transformer encoder block consisting of:
    1. Multi-head self-attention with residual connection
    2. Feed-forward MLP with residual connection
    
    Both sub-layers use pre-norm architecture (layer norm before the operation).
    
    Args:
        embed_dim (int): Dimension of the embedding space
        attention_heads (int): Number of attention heads
        mlp_hidden_nodes (int): Hidden layer size in the MLP
        dropout (float, optional): Dropout probability. Defaults to 0.0
    
    Input shape:
        (batch_size, sequence_length, embed_dim)
    
    Output shape:
        (batch_size, sequence_length, embed_dim)
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        attention_heads: int, 
        mlp_hidden_nodes: int,
        dropout: float = 0.0
    ):
        """
        Initialize the Transformer Encoder block.
        
        Args:
            embed_dim: Embedding dimension
            attention_heads: Number of attention heads
            mlp_hidden_nodes: Hidden layer size in MLP
            dropout: Dropout probability
        """
        super().__init__()
        
        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if attention_heads <= 0:
            raise ValueError(f"attention_heads must be positive, got {attention_heads}")
        if embed_dim % attention_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by attention_heads ({attention_heads})"
            )
        if mlp_hidden_nodes <= 0:
            raise ValueError(f"mlp_hidden_nodes must be positive, got {mlp_hidden_nodes}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.mlp_hidden_nodes = mlp_hidden_nodes
        
        # Layer normalization layers (pre-norm architecture)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim, 
            attention_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        # Feed-forward MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_nodes),
            nn.GELU(),  # Gaussian Error Linear Unit activation
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_nodes, embed_dim),
            nn.Dropout(dropout)
        )
        
        logger.info(
            f"Initialized TransformerEncoder: embed_dim={embed_dim}, "
            f"heads={attention_heads}, mlp_hidden={mlp_hidden_nodes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim)
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, embed_dim)
        
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch, sequence, embed_dim), "
                f"got {x.dim()}D tensor"
            )
        
        _, _, embed_dim = x.shape
        
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {embed_dim}"
            )
        
        # First sub-layer: Multi-head self-attention with residual connection
        residual1 = x
        x = self.layer_norm1(x)  # Pre-norm
        attn_output, _ = self.multihead_attention(x, x, x)  # Self-attention
        x = attn_output + residual1  # Residual connection
        
        # Second sub-layer: Feed-forward MLP with residual connection
        residual2 = x
        x = self.layer_norm2(x)  # Pre-norm
        x = self.mlp(x)
        x = x + residual2  # Residual connection
        
        return x
