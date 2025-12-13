"""
Temporal Positional Encoding

Adds temporal position information to frame embeddings using
sinusoidal encoding (like Transformer positional encoding).
"""

import math
import torch
import torch.nn as nn


class TemporalPositionalEncoding(nn.Module):
    """
    Add temporal position information to frame embeddings.

    Uses sinusoidal positional encoding to encode frame order.
    Supports variable-length sequences up to max_frames.

    Args:
        d_model: Embedding dimension (4096 for DINOv3-7B)
        max_frames: Maximum number of frames to support
    """

    def __init__(self, d_model: int = 4096, max_frames: int = 1000):
        super().__init__()

        self.d_model = d_model
        self.max_frames = max_frames

        # Create sinusoidal positional encoding table
        pe = self._create_positional_encoding(max_frames, d_model)
        self.register_buffer('pe', pe)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.

        Args:
            max_len: Maximum sequence length
            d_model: Embedding dimension

        Returns:
            (max_len, d_model) positional encoding tensor
        """
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        x: torch.Tensor,
        frame_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Add positional encoding to embeddings.

        Args:
            x: (batch, num_frames, d_model) frame embeddings
            frame_indices: (batch, num_frames) actual frame numbers
                          If None, uses sequential indices

        Returns:
            (batch, num_frames, d_model) embeddings with temporal position
        """
        batch_size, num_frames, _ = x.shape

        if frame_indices is None:
            # Use sequential indices
            frame_indices = torch.arange(num_frames, device=x.device)
            frame_indices = frame_indices.unsqueeze(0).expand(batch_size, -1)

        # Clamp indices to valid range
        frame_indices = frame_indices.clamp(0, self.max_frames - 1)

        # Gather positional encodings for each frame
        # pe[frame_indices] would be (batch, num_frames, d_model)
        pe = self.pe[frame_indices.long()]

        return x + pe

    def get_encoding(self, frame_idx: int) -> torch.Tensor:
        """
        Get positional encoding for a single frame.

        Args:
            frame_idx: Frame index

        Returns:
            (d_model,) positional encoding
        """
        frame_idx = min(frame_idx, self.max_frames - 1)
        return self.pe[frame_idx]

    def get_relative_encoding(
        self,
        frame_idx1: int,
        frame_idx2: int,
    ) -> torch.Tensor:
        """
        Get relative positional encoding between two frames.

        Args:
            frame_idx1: First frame index
            frame_idx2: Second frame index

        Returns:
            (d_model,) relative encoding (difference)
        """
        pe1 = self.get_encoding(frame_idx1)
        pe2 = self.get_encoding(frame_idx2)
        return pe2 - pe1


class LearnedTemporalPositionalEncoding(nn.Module):
    """
    Learned temporal positional encoding.

    Alternative to sinusoidal encoding with learnable parameters.
    """

    def __init__(self, d_model: int = 4096, max_frames: int = 1000):
        super().__init__()

        self.d_model = d_model
        self.max_frames = max_frames

        # Learnable position embeddings
        self.pe = nn.Embedding(max_frames, d_model)

        # Initialize with small values
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        frame_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Add learned positional encoding to embeddings.

        Args:
            x: (batch, num_frames, d_model) frame embeddings
            frame_indices: (batch, num_frames) actual frame numbers

        Returns:
            (batch, num_frames, d_model) embeddings with temporal position
        """
        batch_size, num_frames, _ = x.shape

        if frame_indices is None:
            frame_indices = torch.arange(num_frames, device=x.device)
            frame_indices = frame_indices.unsqueeze(0).expand(batch_size, -1)

        # Clamp to valid range
        frame_indices = frame_indices.clamp(0, self.max_frames - 1)

        # Get learned embeddings
        pe = self.pe(frame_indices.long())

        return x + pe
