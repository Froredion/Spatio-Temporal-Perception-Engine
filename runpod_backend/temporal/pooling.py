"""
Temporal Pooling

Produce clip-level embeddings from frame sequences.
Aggregates temporal information into fixed-size representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalPooling(nn.Module):
    """
    Produce clip-level embeddings from frame sequence.

    Uses attention-weighted aggregation to create a single
    embedding representing the entire clip.

    Args:
        d_model: Feature dimension (4096 for DINOv3)
        hidden_dim: Hidden dimension for attention
    """

    def __init__(
        self,
        d_model: int = 4096,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.d_model = d_model

        # Attention-weighted pooling
        self.attention = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Optional projection after pooling
        self.projection = nn.Linear(d_model, d_model)

    def forward(
        self,
        frame_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool frame sequence to single embedding.

        Args:
            frame_features: (batch, num_frames, d_model)
            mask: (batch, num_frames) boolean mask, True for valid frames

        Returns:
            (batch, d_model) clip-level embedding
        """
        batch_size, num_frames, _ = frame_features.shape

        # Compute attention weights
        attn_weights = self.attention(frame_features).squeeze(-1)  # (batch, num_frames)

        # Apply mask
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        # Softmax normalization
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        clip_embedding = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, num_frames)
            frame_features,  # (batch, num_frames, d_model)
        ).squeeze(1)  # (batch, d_model)

        # Optional projection
        clip_embedding = self.projection(clip_embedding)

        return clip_embedding

    def forward_with_weights(
        self,
        frame_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool with attention weight output.

        Args:
            frame_features: (batch, num_frames, d_model)
            mask: (batch, num_frames) boolean mask

        Returns:
            Tuple of (clip_embedding, attention_weights)
        """
        attn_weights = self.attention(frame_features).squeeze(-1)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        clip_embedding = torch.bmm(
            attn_weights.unsqueeze(1),
            frame_features,
        ).squeeze(1)

        clip_embedding = self.projection(clip_embedding)

        return clip_embedding, attn_weights


class HierarchicalTemporalPooling(nn.Module):
    """
    Hierarchical pooling for long videos.

    First pools within segments, then pools segments to final embedding.
    Handles videos with many frames efficiently.

    Args:
        d_model: Feature dimension
        segment_size: Number of frames per segment
    """

    def __init__(
        self,
        d_model: int = 4096,
        segment_size: int = 32,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.d_model = d_model
        self.segment_size = segment_size

        # Segment-level pooling
        self.segment_pooling = TemporalPooling(d_model, hidden_dim)

        # Clip-level pooling (over segments)
        self.clip_pooling = TemporalPooling(d_model, hidden_dim)

    def forward(
        self,
        frame_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hierarchical pooling.

        Args:
            frame_features: (batch, num_frames, d_model)
            mask: (batch, num_frames) boolean mask

        Returns:
            (batch, d_model) clip embedding
        """
        batch_size, num_frames, d_model = frame_features.shape

        # Pad to multiple of segment_size
        pad_size = (self.segment_size - num_frames % self.segment_size) % self.segment_size
        if pad_size > 0:
            padding = torch.zeros(batch_size, pad_size, d_model, device=frame_features.device)
            frame_features = torch.cat([frame_features, padding], dim=1)

            if mask is not None:
                mask_padding = torch.zeros(batch_size, pad_size, dtype=torch.bool, device=mask.device)
                mask = torch.cat([mask, mask_padding], dim=1)

        num_segments = frame_features.shape[1] // self.segment_size

        # Reshape to segments
        segments = frame_features.reshape(batch_size * num_segments, self.segment_size, d_model)

        if mask is not None:
            segment_masks = mask.reshape(batch_size * num_segments, self.segment_size)
        else:
            segment_masks = None

        # Pool within segments
        segment_embeddings = self.segment_pooling(segments, segment_masks)
        segment_embeddings = segment_embeddings.reshape(batch_size, num_segments, d_model)

        # Create segment-level mask
        if mask is not None:
            segment_mask = mask.reshape(batch_size, num_segments, self.segment_size).any(dim=-1)
        else:
            segment_mask = None

        # Pool over segments
        clip_embedding = self.clip_pooling(segment_embeddings, segment_mask)

        return clip_embedding


class MultiScaleTemporalPooling(nn.Module):
    """
    Multi-scale temporal pooling.

    Creates embeddings at multiple temporal scales (short, medium, long)
    for different types of temporal queries.

    Args:
        d_model: Feature dimension
        scales: List of window sizes for different scales
    """

    def __init__(
        self,
        d_model: int = 4096,
        scales: Tuple[int, ...] = (4, 16, 64),
    ):
        super().__init__()

        self.d_model = d_model
        self.scales = scales

        # Pooling at each scale
        self.poolers = nn.ModuleList([
            TemporalPooling(d_model, d_model // 4) for _ in scales
        ])

        # Fusion layer
        self.fusion = nn.Linear(d_model * len(scales), d_model)

    def forward(
        self,
        frame_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-scale pooling.

        Args:
            frame_features: (batch, num_frames, d_model)
            mask: (batch, num_frames) boolean mask

        Returns:
            (batch, d_model) fused multi-scale embedding
        """
        batch_size, num_frames, d_model = frame_features.shape

        scale_embeddings = []

        for scale, pooler in zip(self.scales, self.poolers):
            if num_frames <= scale:
                # Video shorter than scale, pool all frames
                emb = pooler(frame_features, mask)
            else:
                # Sample at this scale and pool
                stride = num_frames // scale
                indices = torch.arange(0, num_frames, stride, device=frame_features.device)[:scale]
                sampled = frame_features[:, indices]

                sampled_mask = None
                if mask is not None:
                    sampled_mask = mask[:, indices]

                emb = pooler(sampled, sampled_mask)

            scale_embeddings.append(emb)

        # Concatenate and fuse
        concat = torch.cat(scale_embeddings, dim=-1)
        fused = self.fusion(concat)

        return fused

    def forward_all_scales(
        self,
        frame_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Return embeddings at all scales separately.

        Returns:
            Dictionary mapping scale -> embedding
        """
        batch_size, num_frames, d_model = frame_features.shape

        results = {}

        for scale, pooler in zip(self.scales, self.poolers):
            if num_frames <= scale:
                emb = pooler(frame_features, mask)
            else:
                stride = num_frames // scale
                indices = torch.arange(0, num_frames, stride, device=frame_features.device)[:scale]
                sampled = frame_features[:, indices]

                sampled_mask = None
                if mask is not None:
                    sampled_mask = mask[:, indices]

                emb = pooler(sampled, sampled_mask)

            results[f'scale_{scale}'] = emb

        # Also compute fused
        results['fused'] = self.forward(frame_features, mask)

        return results
