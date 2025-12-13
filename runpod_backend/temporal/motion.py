"""
Motion Feature Extraction

Extract motion features from frame-to-frame feature deltas.
Encodes velocity and acceleration for temporal understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MotionFeatureExtractor(nn.Module):
    """
    Extract motion features from frame-to-frame feature deltas.

    Computes:
    - Frame deltas (velocity-like)
    - Motion magnitude
    - Motion direction encoding
    - Acceleration (second-order deltas)

    Args:
        feature_dim: Input feature dimension (4096 for DINOv3)
        hidden_dim: Hidden layer dimension
        output_dim: Output motion feature dimension
    """

    def __init__(
        self,
        feature_dim: int = 4096,
        hidden_dim: int = 4096,
        output_dim: int = 4096,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # Motion encoder
        self.motion_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Velocity encoder (first-order difference)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, feature_dim),
        )

        # Acceleration encoder (second-order difference)
        self.acceleration_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, feature_dim // 2),
        )

        # Motion magnitude predictor
        self.magnitude_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Extract motion features from frame sequence.

        Args:
            frame_features: (batch, num_frames, feature_dim) frame embeddings
            return_components: Whether to return individual motion components

        Returns:
            (batch, num_frames-1, output_dim) motion features
            If return_components: dict with velocity, acceleration, magnitude
        """
        batch_size, num_frames, _ = frame_features.shape

        if num_frames < 2:
            # No motion for single frame
            if return_components:
                return {
                    'motion': torch.zeros(batch_size, 0, self.feature_dim, device=frame_features.device),
                    'velocity': torch.zeros(batch_size, 0, self.feature_dim, device=frame_features.device),
                    'acceleration': torch.zeros(batch_size, 0, self.feature_dim // 2, device=frame_features.device),
                    'magnitude': torch.zeros(batch_size, 0, 1, device=frame_features.device),
                }
            return torch.zeros(batch_size, 0, self.feature_dim, device=frame_features.device)

        # Compute first-order deltas (velocity)
        deltas = frame_features[:, 1:] - frame_features[:, :-1]  # (batch, num_frames-1, dim)

        # Encode velocity
        velocity = self.velocity_encoder(deltas)

        # Concatenate frame features with deltas for context
        concat = torch.cat([frame_features[:, :-1], deltas], dim=-1)  # (batch, num_frames-1, dim*2)

        # Encode motion
        motion_features = self.motion_encoder(concat)

        # Compute motion magnitude
        magnitude = self.magnitude_head(deltas)

        if return_components:
            # Compute acceleration (second-order deltas)
            if num_frames >= 3:
                acceleration_deltas = deltas[:, 1:] - deltas[:, :-1]
                acceleration = self.acceleration_encoder(acceleration_deltas)
                # Pad to match velocity length
                acceleration = F.pad(acceleration, (0, 0, 1, 0))  # Pad first frame
            else:
                acceleration = torch.zeros(
                    batch_size, num_frames - 1, self.feature_dim // 2,
                    device=frame_features.device
                )

            return {
                'motion': motion_features,
                'velocity': velocity,
                'acceleration': acceleration,
                'magnitude': magnitude,
            }

        return motion_features

    def get_motion_summary(
        self,
        frame_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get single motion summary for entire sequence.

        Useful for classifying the type of motion/action.

        Args:
            frame_features: (batch, num_frames, feature_dim)

        Returns:
            (batch, output_dim) motion summary
        """
        motion = self.forward(frame_features)

        if motion.shape[1] == 0:
            return torch.zeros(frame_features.shape[0], self.feature_dim, device=frame_features.device)

        # Mean pooling over time
        summary = motion.mean(dim=1)

        return summary

    def detect_motion_events(
        self,
        frame_features: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Detect frames with significant motion.

        Args:
            frame_features: (batch, num_frames, feature_dim)
            threshold: Motion magnitude threshold

        Returns:
            (batch, num_frames-1) boolean mask of motion events
        """
        motion_data = self.forward(frame_features, return_components=True)
        magnitude = motion_data['magnitude'].squeeze(-1)

        return magnitude > threshold


class ObjectMotionTracker(nn.Module):
    """
    Track motion of individual objects across frames.

    Extends MotionFeatureExtractor to handle per-object motion.
    """

    def __init__(
        self,
        feature_dim: int = 4096,
        hidden_dim: int = 2048,
    ):
        super().__init__()

        self.motion_extractor = MotionFeatureExtractor(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
        )

        # Object interaction encoder
        self.interaction_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        object_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract motion for tracked object.

        Args:
            object_features: (num_frames, feature_dim) features of single object

        Returns:
            (num_frames-1, feature_dim) object motion features
        """
        # Add batch dimension
        features = object_features.unsqueeze(0)
        motion = self.motion_extractor(features)
        return motion.squeeze(0)

    def compute_interaction(
        self,
        object1_features: torch.Tensor,
        object2_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute interaction features between two objects.

        Args:
            object1_features: (num_frames, feature_dim)
            object2_features: (num_frames, feature_dim)

        Returns:
            (num_frames, feature_dim) interaction features
        """
        concat = torch.cat([object1_features, object2_features], dim=-1)
        return self.interaction_encoder(concat)
