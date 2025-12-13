"""
Temporal Object Tracking

Track objects across frames using DINOv3 feature similarity.
Maintains consistent object IDs across video sequences.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackEntry:
    """Single entry in an object track."""
    frame_idx: int
    embedding: torch.Tensor  # DINOv3 embedding for tracking
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str = "object"  # SAM-3 label for this detection


@dataclass
class Track:
    """Object track across multiple frames."""
    track_id: int
    entries: List[TrackEntry] = field(default_factory=list)
    is_active: bool = True
    last_seen_frame: int = -1

    def add_entry(self, entry: TrackEntry):
        """Add new entry to track."""
        self.entries.append(entry)
        self.last_seen_frame = entry.frame_idx

    @property
    def latest_embedding(self) -> torch.Tensor:
        """Get most recent embedding."""
        return self.entries[-1].embedding if self.entries else None

    @property
    def mean_embedding(self) -> torch.Tensor:
        """Get mean of all embeddings in track."""
        if not self.entries:
            return None
        embeddings = torch.stack([e.embedding for e in self.entries])
        return embeddings.mean(dim=0)

    @property
    def trajectory(self) -> List[Tuple[int, int]]:
        """Get list of (x_center, y_center) positions."""
        positions = []
        for entry in self.entries:
            x1, y1, x2, y2 = entry.bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            positions.append((cx, cy))
        return positions

    def get_majority_label(self) -> str:
        """Get the most common label across all entries (from SAM-3)."""
        if not self.entries:
            return "object"
        from collections import Counter
        labels = [e.label for e in self.entries if e.label != "object"]
        if not labels:
            return "object"
        return Counter(labels).most_common(1)[0][0]

    def get_motion_analysis(self, fps: float = 2.0) -> Dict[str, Any]:
        """
        Compute motion analysis for this track.

        Returns dict with:
        - velocities: List of (dx, dy) per frame transition
        - speeds: List of speed values (pixels/second)
        - directions: List of direction angles in degrees (0=right, 90=up)
        - avg_speed: Average speed
        - avg_direction: Average direction (circular mean)
        - total_distance: Total distance traveled
        - movement_type: 'stationary', 'slow', 'moderate', 'fast'
        - primary_direction: 'left', 'right', 'up', 'down', 'stationary'
        """
        trajectory = self.trajectory

        if len(trajectory) < 2:
            return {
                "velocities": [],
                "speeds": [],
                "directions": [],
                "avg_speed": 0.0,
                "avg_direction": 0.0,
                "total_distance": 0.0,
                "movement_type": "stationary",
                "primary_direction": "stationary",
            }

        velocities = []
        speeds = []
        directions = []
        total_distance = 0.0

        for i in range(1, len(trajectory)):
            x1, y1 = trajectory[i - 1]
            x2, y2 = trajectory[i]

            dx = x2 - x1
            dy = y2 - y1

            # Distance for this step (in pixels)
            dist = np.sqrt(dx * dx + dy * dy)
            total_distance += dist

            # Speed (pixels per second)
            speed = dist * fps

            # Direction in degrees (0=right, 90=up, 180=left, 270=down)
            # Note: y is inverted in screen coordinates (down is positive)
            direction = np.degrees(np.arctan2(-dy, dx)) % 360

            velocities.append((dx, dy))
            speeds.append(speed)
            directions.append(direction)

        # Compute averages
        avg_speed = np.mean(speeds) if speeds else 0.0

        # Circular mean for direction (weighted by speed)
        if speeds and sum(speeds) > 0:
            sin_sum = sum(s * np.sin(np.radians(d)) for s, d in zip(speeds, directions))
            cos_sum = sum(s * np.cos(np.radians(d)) for s, d in zip(speeds, directions))
            avg_direction = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
        else:
            avg_direction = 0.0

        # Classify movement type based on speed
        if avg_speed < 5:
            movement_type = "stationary"
        elif avg_speed < 50:
            movement_type = "slow"
        elif avg_speed < 150:
            movement_type = "moderate"
        else:
            movement_type = "fast"

        # Determine primary direction (8-way: cardinals + diagonals)
        if avg_speed < 5:
            primary_direction = "stationary"
        elif 337.5 <= avg_direction or avg_direction < 22.5:
            primary_direction = "right"
        elif 22.5 <= avg_direction < 67.5:
            primary_direction = "up-right"
        elif 67.5 <= avg_direction < 112.5:
            primary_direction = "up"
        elif 112.5 <= avg_direction < 157.5:
            primary_direction = "up-left"
        elif 157.5 <= avg_direction < 202.5:
            primary_direction = "left"
        elif 202.5 <= avg_direction < 247.5:
            primary_direction = "down-left"
        elif 247.5 <= avg_direction < 292.5:
            primary_direction = "down"
        else:  # 292.5 <= avg_direction < 337.5
            primary_direction = "down-right"

        return {
            "velocities": velocities,
            "speeds": speeds,
            "directions": directions,
            "avg_speed": float(avg_speed),
            "avg_direction": float(avg_direction),
            "total_distance": float(total_distance),
            "movement_type": movement_type,
            "primary_direction": primary_direction,
        }


class DINOv3Tracker:
    """
    Track objects across frames using DINOv3 feature similarity.

    Uses Hungarian algorithm for optimal assignment between
    detected objects and existing tracks.

    Args:
        similarity_threshold: Minimum similarity to match (0-1)
        max_frames_missing: Frames before track is deactivated
        use_motion_prediction: Whether to predict next position
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        max_frames_missing: int = 10,
        use_motion_prediction: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_frames_missing = max_frames_missing
        self.use_motion_prediction = use_motion_prediction

        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.current_frame = -1

    def update(
        self,
        frame_idx: int,
        objects: List[Dict],
    ) -> List[Dict]:
        """
        Update tracks with detected objects from new frame.

        Args:
            frame_idx: Current frame index
            objects: List of detected objects, each with:
                - 'dinov3_embedding': torch.Tensor
                - 'bbox': (x1, y1, x2, y2)
                - 'confidence': float (optional)

        Returns:
            List of objects with added 'track_id' field
        """
        self.current_frame = frame_idx

        # Deactivate old tracks
        self._deactivate_old_tracks(frame_idx)

        if not objects:
            return []

        # Get active tracks
        active_tracks = [t for t in self.tracks.values() if t.is_active]

        if not active_tracks:
            # No existing tracks, create new ones for all objects
            return self._create_new_tracks(frame_idx, objects)

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(objects, active_tracks)

        # Hungarian matching
        matches, unmatched_objects, unmatched_tracks = self._hungarian_match(
            similarity_matrix,
            objects,
            active_tracks,
        )

        # Update matched tracks
        for obj_idx, track in matches:
            obj = objects[obj_idx]
            entry = TrackEntry(
                frame_idx=frame_idx,
                embedding=obj['dinov3_embedding'],
                bbox=obj['bbox'],
                confidence=obj.get('confidence', 1.0),
                label=obj.get('label', 'object'),  # SAM-3 label
            )
            track.add_entry(entry)
            obj['track_id'] = track.track_id

        # Create new tracks for unmatched objects
        for obj_idx in unmatched_objects:
            obj = objects[obj_idx]
            track_id = self._create_track(frame_idx, obj)
            obj['track_id'] = track_id

        return objects

    def _compute_similarity_matrix(
        self,
        objects: List[Dict],
        tracks: List[Track],
    ) -> np.ndarray:
        """
        Compute similarity matrix between objects and tracks.

        Args:
            objects: Detected objects
            tracks: Active tracks

        Returns:
            (num_objects, num_tracks) similarity matrix
        """
        num_objects = len(objects)
        num_tracks = len(tracks)

        similarity = np.zeros((num_objects, num_tracks))

        for i, obj in enumerate(objects):
            obj_emb = obj['dinov3_embedding']

            # Normalize
            obj_emb = F.normalize(obj_emb.unsqueeze(0), dim=-1)

            for j, track in enumerate(tracks):
                track_emb = track.latest_embedding

                if track_emb is None:
                    continue

                track_emb = F.normalize(track_emb.unsqueeze(0), dim=-1)

                # Cosine similarity
                sim = F.cosine_similarity(obj_emb, track_emb, dim=-1)
                similarity[i, j] = sim.item()

                # Optional: Add spatial proximity bonus
                if self.use_motion_prediction and len(track.trajectory) >= 2:
                    predicted_pos = self._predict_next_position(track)
                    actual_pos = self._get_center(obj['bbox'])
                    distance = np.sqrt(
                        (predicted_pos[0] - actual_pos[0])**2 +
                        (predicted_pos[1] - actual_pos[1])**2
                    )
                    # Normalize distance to 0-1 range (assuming max 100 pixels)
                    spatial_bonus = max(0, 1 - distance / 100) * 0.1
                    similarity[i, j] += spatial_bonus

        return similarity

    def _hungarian_match(
        self,
        similarity: np.ndarray,
        objects: List[Dict],
        tracks: List[Track],
    ) -> Tuple[List[Tuple[int, Track]], List[int], List[Track]]:
        """
        Perform Hungarian matching.

        Args:
            similarity: Similarity matrix
            objects: Detected objects
            tracks: Active tracks

        Returns:
            Tuple of (matches, unmatched_objects, unmatched_tracks)
        """
        # Convert similarity to cost (negate)
        cost = -similarity

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost)

        matches = []
        matched_obj_indices = set()
        matched_track_indices = set()

        for row, col in zip(row_indices, col_indices):
            if similarity[row, col] >= self.similarity_threshold:
                matches.append((row, tracks[col]))
                matched_obj_indices.add(row)
                matched_track_indices.add(col)

        # Unmatched objects
        unmatched_objects = [
            i for i in range(len(objects)) if i not in matched_obj_indices
        ]

        # Unmatched tracks
        unmatched_tracks = [
            tracks[i] for i in range(len(tracks)) if i not in matched_track_indices
        ]

        return matches, unmatched_objects, unmatched_tracks

    def _create_new_tracks(
        self,
        frame_idx: int,
        objects: List[Dict],
    ) -> List[Dict]:
        """Create new tracks for all objects."""
        for obj in objects:
            track_id = self._create_track(frame_idx, obj)
            obj['track_id'] = track_id
        return objects

    def _create_track(self, frame_idx: int, obj: Dict) -> int:
        """Create new track for object."""
        track_id = self.next_id
        self.next_id += 1

        track = Track(track_id=track_id)
        entry = TrackEntry(
            frame_idx=frame_idx,
            embedding=obj['dinov3_embedding'],
            bbox=obj['bbox'],
            confidence=obj.get('confidence', 1.0),
            label=obj.get('label', 'object'),  # SAM-3 label
        )
        track.add_entry(entry)

        self.tracks[track_id] = track
        return track_id

    def _deactivate_old_tracks(self, current_frame: int):
        """Deactivate tracks not seen recently."""
        for track in self.tracks.values():
            if track.is_active:
                frames_missing = current_frame - track.last_seen_frame
                if frames_missing > self.max_frames_missing:
                    track.is_active = False

    def _predict_next_position(self, track: Track) -> Tuple[int, int]:
        """Predict next position using linear motion model."""
        trajectory = track.trajectory
        if len(trajectory) < 2:
            return trajectory[-1] if trajectory else (0, 0)

        # Linear extrapolation from last two positions
        x1, y1 = trajectory[-2]
        x2, y2 = trajectory[-1]

        dx = x2 - x1
        dy = y2 - y1

        return (x2 + dx, y2 + dy)

    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> Dict[int, Track]:
        """Get all tracks."""
        return self.tracks

    def get_active_tracks(self) -> List[Track]:
        """Get currently active tracks."""
        return [t for t in self.tracks.values() if t.is_active]

    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.next_id = 0
        self.current_frame = -1

    def clear_tracks(self):
        """
        Clear all tracks and release tensor memory.

        Unlike reset(), this explicitly deletes tensor references
        to ensure GPU memory is freed. Call this after processing
        when data has been saved to R2.
        """
        for track in self.tracks.values():
            for entry in track.entries:
                # Explicitly delete tensor references
                entry.embedding = None
            track.entries.clear()
        self.tracks.clear()
        self.next_id = 0
        self.current_frame = -1
