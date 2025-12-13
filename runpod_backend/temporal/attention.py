"""
Cross-Frame Temporal Attention

Fuse features across time using self-attention.
Enables frames to attend to each other for temporal reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.positional_encoding import TemporalPositionalEncoding


class TemporalAttention(nn.Module):
    """
    Fuse features across time with self-attention.

    Applies transformer-style self-attention over frame sequence
    to enable cross-frame information flow.

    Args:
        d_model: Feature dimension (4096 for DINOv3)
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_frames: Maximum sequence length
    """

    def __init__(
        self,
        d_model: int = 4096,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 16384,
        dropout: float = 0.1,
        max_frames: int = 1000,
    ):
        super().__init__()

        self.d_model = d_model

        # Temporal positional encoding
        self.temporal_pos = TemporalPositionalEncoding(d_model, max_frames)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm for output
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        frame_features: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply temporal self-attention.

        Args:
            frame_features: (batch, num_frames, d_model) frame embeddings
            frame_indices: (batch, num_frames) actual frame numbers (for non-uniform sampling)
            attention_mask: (batch, num_frames) boolean mask (True = attend)

        Returns:
            (batch, num_frames, d_model) temporally fused features
        """
        # Add temporal position encoding
        x = self.temporal_pos(frame_features, frame_indices)

        # Create attention mask if provided
        src_key_padding_mask = None
        if attention_mask is not None:
            # TransformerEncoder expects True for positions to ignore
            src_key_padding_mask = ~attention_mask

        # Apply temporal self-attention
        output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Final layer norm
        output = self.output_norm(output)

        return output

    def forward_with_cache(
        self,
        frame_features: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with KV cache for streaming inference.

        Args:
            frame_features: (batch, num_new_frames, d_model) new frame embeddings
            cache: (batch, num_cached_frames, d_model) cached features

        Returns:
            Tuple of (output features, updated cache)
        """
        if cache is not None:
            # Concatenate cached and new features
            all_features = torch.cat([cache, frame_features], dim=1)
        else:
            all_features = frame_features

        # Full attention over all frames
        output = self.forward(all_features)

        # Return only new frame outputs and updated cache
        num_new = frame_features.shape[1]
        new_outputs = output[:, -num_new:]

        return new_outputs, all_features


class CausalTemporalAttention(nn.Module):
    """
    Causal temporal attention (each frame only attends to past frames).

    Useful for online/streaming video processing.

    Args:
        d_model: Feature dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
    """

    def __init__(
        self,
        d_model: int = 4096,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 16384,
        dropout: float = 0.1,
        max_frames: int = 1000,
    ):
        super().__init__()

        self.d_model = d_model

        # Temporal positional encoding
        self.temporal_pos = TemporalPositionalEncoding(d_model, max_frames)

        # Transformer decoder (with causal masking)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        frame_features: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply causal temporal attention.

        Args:
            frame_features: (batch, num_frames, d_model)
            frame_indices: (batch, num_frames) frame numbers

        Returns:
            (batch, num_frames, d_model) features with causal attention
        """
        # Add temporal position
        x = self.temporal_pos(frame_features, frame_indices)

        # Create causal mask
        num_frames = x.shape[1]
        causal_mask = self._generate_causal_mask(num_frames, x.device)

        # Self-attention with causal mask
        # Use decoder with memory=x for self-attention
        output = self.transformer(x, x, tgt_mask=causal_mask)

        output = self.output_norm(output)

        return output

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class LocalWindowAttention(nn.Module):
    """
    Local window attention for efficient long-video processing.

    Each frame attends only to a local window of nearby frames.

    Args:
        d_model: Feature dimension
        window_size: Number of frames in attention window
        nhead: Number of attention heads
    """

    def __init__(
        self,
        d_model: int = 4096,
        window_size: int = 16,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size

        self.temporal_pos = TemporalPositionalEncoding(d_model)

        # Local attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        frame_features: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply local window attention.

        Args:
            frame_features: (batch, num_frames, d_model)
            frame_indices: (batch, num_frames) frame numbers

        Returns:
            (batch, num_frames, d_model) locally fused features
        """
        batch_size, num_frames, _ = frame_features.shape

        # Add position encoding
        x = self.temporal_pos(frame_features, frame_indices)

        # Process in windows
        outputs = []
        for i in range(0, num_frames, self.window_size):
            end = min(i + self.window_size, num_frames)
            window = x[:, i:end]

            # Self-attention within window
            attn_out, _ = self.attention(window, window, window)
            window = self.norm1(window + attn_out)

            # FFN
            ffn_out = self.ffn(window)
            window = self.norm2(window + ffn_out)

            outputs.append(window)

        return torch.cat(outputs, dim=1)
