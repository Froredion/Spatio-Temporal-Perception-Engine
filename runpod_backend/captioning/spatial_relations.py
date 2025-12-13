"""
Spatial Relationship Extraction

Extracts spatial relationships between detected objects using bounding boxes.
Generates structured relationship data for training datasets.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SpatialRelation(Enum):
    """Spatial relationship types between objects."""
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    FAR_FROM = "far_from"
    OVERLAPPING = "overlapping"
    INSIDE = "inside"
    CONTAINS = "contains"
    IN_FRONT_OF = "in_front_of"  # Based on size/position heuristics
    BEHIND = "behind"


@dataclass
class ObjectInfo:
    """Object information for spatial analysis."""
    label: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area: int
    confidence: float
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bbox."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class SpatialRelationship:
    """A spatial relationship between two objects."""
    subject: str  # Subject object label
    subject_idx: int  # Index in objects list
    relation: SpatialRelation
    object: str  # Object label (the other object)
    object_idx: int
    confidence: float  # How confident we are in this relationship
    description: str  # Natural language description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "subject_idx": self.subject_idx,
            "relation": self.relation.value,
            "object": self.object,
            "object_idx": self.object_idx,
            "confidence": round(self.confidence, 2),
            "description": self.description,
        }


class SpatialRelationExtractor:
    """
    Extract spatial relationships between objects in a frame.

    Uses bounding box geometry to determine:
    - Horizontal relationships (left_of, right_of)
    - Vertical relationships (above, below)
    - Proximity relationships (near, far_from)
    - Containment relationships (inside, contains, overlapping)
    - Depth heuristics (in_front_of, behind) based on size/position

    Args:
        frame_width: Image width for normalization
        frame_height: Image height for normalization
        proximity_threshold: Normalized distance threshold for "near" (default 0.15)
        overlap_threshold: IoU threshold for "overlapping" (default 0.1)
    """

    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        proximity_threshold: float = 0.15,
        overlap_threshold: float = 0.1,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.proximity_threshold = proximity_threshold
        self.overlap_threshold = overlap_threshold

    def extract_relationships(
        self,
        objects: List[Dict[str, Any]],
        max_relationships: int = 20,
    ) -> List[SpatialRelationship]:
        """
        Extract spatial relationships between all object pairs.

        Args:
            objects: List of detected objects with 'label', 'bbox', 'area', 'confidence'
            max_relationships: Maximum relationships to return (sorted by confidence)

        Returns:
            List of SpatialRelationship instances
        """
        if len(objects) < 2:
            return []

        # Convert to ObjectInfo
        obj_infos = []
        for i, obj in enumerate(objects):
            obj_infos.append(ObjectInfo(
                label=obj.get("label", f"object_{i}"),
                bbox=tuple(obj["bbox"]),
                area=obj.get("area", 0),
                confidence=obj.get("confidence", 1.0),
                track_id=obj.get("track_id"),
            ))

        relationships = []

        # Analyze all pairs
        for i, subj in enumerate(obj_infos):
            for j, obj in enumerate(obj_infos):
                if i == j:
                    continue

                # Get relationships between this pair
                pair_rels = self._analyze_pair(subj, i, obj, j)
                relationships.extend(pair_rels)

        # Sort by confidence and limit
        relationships.sort(key=lambda r: r.confidence, reverse=True)
        return relationships[:max_relationships]

    def _analyze_pair(
        self,
        subj: ObjectInfo,
        subj_idx: int,
        obj: ObjectInfo,
        obj_idx: int,
    ) -> List[SpatialRelationship]:
        """Analyze spatial relationships between two objects."""
        relationships = []

        subj_cx, subj_cy = subj.center
        obj_cx, obj_cy = obj.center

        # Normalize coordinates
        norm_subj_cx = subj_cx / self.frame_width
        norm_subj_cy = subj_cy / self.frame_height
        norm_obj_cx = obj_cx / self.frame_width
        norm_obj_cy = obj_cy / self.frame_height

        # Calculate normalized distance
        dx = norm_obj_cx - norm_subj_cx
        dy = norm_obj_cy - norm_subj_cy
        distance = (dx**2 + dy**2) ** 0.5

        # Horizontal relationships (only if significant horizontal separation)
        if abs(dx) > 0.1:
            if dx > 0:
                conf = min(1.0, abs(dx) * 2)
                relationships.append(SpatialRelationship(
                    subject=subj.label,
                    subject_idx=subj_idx,
                    relation=SpatialRelation.LEFT_OF,
                    object=obj.label,
                    object_idx=obj_idx,
                    confidence=conf,
                    description=f"{subj.label} is to the left of {obj.label}",
                ))
            else:
                conf = min(1.0, abs(dx) * 2)
                relationships.append(SpatialRelationship(
                    subject=subj.label,
                    subject_idx=subj_idx,
                    relation=SpatialRelation.RIGHT_OF,
                    object=obj.label,
                    object_idx=obj_idx,
                    confidence=conf,
                    description=f"{subj.label} is to the right of {obj.label}",
                ))

        # Vertical relationships (only if significant vertical separation)
        if abs(dy) > 0.1:
            if dy > 0:
                conf = min(1.0, abs(dy) * 2)
                relationships.append(SpatialRelationship(
                    subject=subj.label,
                    subject_idx=subj_idx,
                    relation=SpatialRelation.ABOVE,
                    object=obj.label,
                    object_idx=obj_idx,
                    confidence=conf,
                    description=f"{subj.label} is above {obj.label}",
                ))
            else:
                conf = min(1.0, abs(dy) * 2)
                relationships.append(SpatialRelationship(
                    subject=subj.label,
                    subject_idx=subj_idx,
                    relation=SpatialRelation.BELOW,
                    object=obj.label,
                    object_idx=obj_idx,
                    confidence=conf,
                    description=f"{subj.label} is below {obj.label}",
                ))

        # Proximity relationships
        if distance < self.proximity_threshold:
            conf = 1.0 - (distance / self.proximity_threshold)
            relationships.append(SpatialRelationship(
                subject=subj.label,
                subject_idx=subj_idx,
                relation=SpatialRelation.NEAR,
                object=obj.label,
                object_idx=obj_idx,
                confidence=conf,
                description=f"{subj.label} is near {obj.label}",
            ))
        elif distance > 0.5:
            conf = min(1.0, (distance - 0.5) * 2)
            relationships.append(SpatialRelationship(
                subject=subj.label,
                subject_idx=subj_idx,
                relation=SpatialRelation.FAR_FROM,
                object=obj.label,
                object_idx=obj_idx,
                confidence=conf,
                description=f"{subj.label} is far from {obj.label}",
            ))

        # Containment/overlap relationships
        iou = self._compute_iou(subj.bbox, obj.bbox)
        containment = self._compute_containment(subj.bbox, obj.bbox)

        if containment > 0.8:
            # Subject is mostly inside object
            relationships.append(SpatialRelationship(
                subject=subj.label,
                subject_idx=subj_idx,
                relation=SpatialRelation.INSIDE,
                object=obj.label,
                object_idx=obj_idx,
                confidence=containment,
                description=f"{subj.label} is inside {obj.label}",
            ))
        elif iou > self.overlap_threshold:
            relationships.append(SpatialRelationship(
                subject=subj.label,
                subject_idx=subj_idx,
                relation=SpatialRelation.OVERLAPPING,
                object=obj.label,
                object_idx=obj_idx,
                confidence=iou,
                description=f"{subj.label} overlaps with {obj.label}",
            ))

        # Depth heuristics (larger objects at similar y-position are likely closer)
        # Objects lower in frame and larger are typically in front
        if abs(dy) < 0.15:  # Similar vertical position
            subj_norm_area = subj.area / (self.frame_width * self.frame_height)
            obj_norm_area = obj.area / (self.frame_width * self.frame_height)

            if subj_norm_area > obj_norm_area * 1.5 and norm_subj_cy > norm_obj_cy - 0.1:
                conf = min(1.0, (subj_norm_area / max(obj_norm_area, 0.001) - 1) * 0.5)
                relationships.append(SpatialRelationship(
                    subject=subj.label,
                    subject_idx=subj_idx,
                    relation=SpatialRelation.IN_FRONT_OF,
                    object=obj.label,
                    object_idx=obj_idx,
                    confidence=conf,
                    description=f"{subj.label} is in front of {obj.label}",
                ))

        return relationships

    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1)

    def _compute_containment(
        self,
        inner_bbox: Tuple[int, int, int, int],
        outer_bbox: Tuple[int, int, int, int],
    ) -> float:
        """Compute how much of inner_bbox is contained within outer_bbox."""
        x1 = max(inner_bbox[0], outer_bbox[0])
        y1 = max(inner_bbox[1], outer_bbox[1])
        x2 = min(inner_bbox[2], outer_bbox[2])
        y2 = min(inner_bbox[3], outer_bbox[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        inner_area = (inner_bbox[2] - inner_bbox[0]) * (inner_bbox[3] - inner_bbox[1])

        return intersection / max(inner_area, 1)

    def get_scene_layout_description(
        self,
        objects: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a natural language description of the scene layout.

        Args:
            objects: List of detected objects

        Returns:
            Natural language description of spatial arrangement
        """
        if not objects:
            return "Empty scene with no detected objects."

        if len(objects) == 1:
            obj = objects[0]
            pos = self._get_position_description(obj["bbox"])
            return f"Scene contains {obj.get('label', 'an object')} at {pos}."

        # Group objects by position
        left_objects = []
        center_objects = []
        right_objects = []

        for obj in objects:
            bbox = obj["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            norm_cx = cx / self.frame_width

            if norm_cx < 0.33:
                left_objects.append(obj.get("label", "object"))
            elif norm_cx > 0.67:
                right_objects.append(obj.get("label", "object"))
            else:
                center_objects.append(obj.get("label", "object"))

        parts = []
        if left_objects:
            parts.append(f"On the left: {', '.join(left_objects)}")
        if center_objects:
            parts.append(f"In the center: {', '.join(center_objects)}")
        if right_objects:
            parts.append(f"On the right: {', '.join(right_objects)}")

        return ". ".join(parts) + "."

    def _get_position_description(self, bbox: Tuple[int, int, int, int]) -> str:
        """Get natural language position description for a bbox."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        norm_cx = cx / self.frame_width
        norm_cy = cy / self.frame_height

        h_pos = "left" if norm_cx < 0.33 else "right" if norm_cx > 0.67 else "center"
        v_pos = "top" if norm_cy < 0.33 else "bottom" if norm_cy > 0.67 else "middle"

        if h_pos == "center" and v_pos == "middle":
            return "the center"
        elif v_pos == "middle":
            return f"the {h_pos}"
        elif h_pos == "center":
            return f"the {v_pos}"
        else:
            return f"the {v_pos}-{h_pos}"

    def format_relationships_for_caption(
        self,
        relationships: List[SpatialRelationship],
        max_relations: int = 5,
    ) -> str:
        """
        Format spatial relationships as a caption-friendly string.

        Args:
            relationships: List of extracted relationships
            max_relations: Maximum relationships to include

        Returns:
            Natural language description of relationships
        """
        if not relationships:
            return ""

        # Filter to most confident, diverse relationships
        seen_pairs = set()
        selected = []

        for rel in relationships:
            pair_key = (rel.subject_idx, rel.object_idx)
            reverse_key = (rel.object_idx, rel.subject_idx)

            if pair_key not in seen_pairs and reverse_key not in seen_pairs:
                selected.append(rel)
                seen_pairs.add(pair_key)

            if len(selected) >= max_relations:
                break

        descriptions = [rel.description for rel in selected]
        return ". ".join(descriptions)
