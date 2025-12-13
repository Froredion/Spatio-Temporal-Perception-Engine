"""
Feature Clustering Utilities

Cluster dense features for object segmentation.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def cluster_features(
    features: torch.Tensor,
    threshold: float = 0.7,
    min_cluster_size: int = 4,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Cluster features based on similarity.

    Uses similarity-based connected components to find clusters.

    Args:
        features: (H*W, D) or (1, H*W, D) feature tensor
        threshold: Similarity threshold for clustering
        min_cluster_size: Minimum patches in a cluster

    Returns:
        Tuple of (list of cluster masks, list of confidences)
    """
    if features.dim() == 3:
        features = features.squeeze(0)

    num_patches = features.shape[0]
    h = w = int(np.sqrt(num_patches))

    if h * w != num_patches:
        h = int(np.sqrt(num_patches))
        w = num_patches // h

    # Normalize features
    features_norm = torch.nn.functional.normalize(features, dim=-1)

    # Compute similarity to neighbors
    adjacency = _build_neighbor_adjacency(features_norm, h, w, threshold)

    # Find connected components
    masks, confidences = find_connected_components(adjacency, h, w, min_cluster_size)

    return masks, confidences


def _build_neighbor_adjacency(
    features: torch.Tensor,
    h: int,
    w: int,
    threshold: float,
) -> np.ndarray:
    """
    Build adjacency matrix based on spatial neighbors and similarity.

    Only spatially adjacent patches with similarity above threshold are connected.

    Args:
        features: (H*W, D) normalized features
        h, w: Grid dimensions
        threshold: Similarity threshold

    Returns:
        (H*W, H*W) boolean adjacency matrix
    """
    num_patches = h * w
    features_np = features.cpu().numpy()

    adjacency = np.zeros((num_patches, num_patches), dtype=bool)

    for i in range(h):
        for j in range(w):
            idx = i * w + j

            # Check 8-connected neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue

                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_idx = ni * w + nj

                        # Compute similarity
                        similarity = np.dot(features_np[idx], features_np[neighbor_idx])

                        if similarity > threshold:
                            adjacency[idx, neighbor_idx] = True
                            adjacency[neighbor_idx, idx] = True

    return adjacency


def find_connected_components(
    adjacency: np.ndarray,
    h: int,
    w: int,
    min_size: int = 4,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Find connected components in adjacency graph.

    Args:
        adjacency: Boolean adjacency matrix
        h, w: Grid dimensions
        min_size: Minimum component size

    Returns:
        Tuple of (list of masks, list of confidences)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    # Convert to sparse
    sparse_adj = csr_matrix(adjacency)

    # Find components
    n_components, labels = connected_components(sparse_adj, directed=False)

    masks = []
    confidences = []

    for comp_id in range(n_components):
        # Create mask
        component_mask = (labels == comp_id)

        # Skip if too small
        if component_mask.sum() < min_size:
            continue

        # Skip background (typically largest component covering > 50%)
        if component_mask.sum() > (h * w * 0.5):
            continue

        # Reshape to spatial grid
        mask = component_mask.reshape(h, w).astype(np.uint8)

        # Compute confidence (cohesion within cluster)
        indices = np.where(component_mask)[0]
        if len(indices) > 1:
            # Count internal edges vs possible edges
            internal_edges = adjacency[np.ix_(indices, indices)].sum()
            possible_edges = len(indices) * (len(indices) - 1)
            confidence = internal_edges / max(possible_edges, 1)
        else:
            confidence = 1.0

        masks.append(mask)
        confidences.append(float(confidence))

    return masks, confidences


def merge_overlapping_masks(
    masks: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> List[np.ndarray]:
    """
    Merge masks with high IoU overlap.

    Args:
        masks: List of binary masks
        iou_threshold: IoU threshold for merging

    Returns:
        List of merged masks
    """
    if len(masks) <= 1:
        return masks

    merged = []
    used = set()

    for i, mask_i in enumerate(masks):
        if i in used:
            continue

        current_mask = mask_i.copy()

        for j, mask_j in enumerate(masks[i + 1:], i + 1):
            if j in used:
                continue

            # Compute IoU
            intersection = np.logical_and(current_mask, mask_j).sum()
            union = np.logical_or(current_mask, mask_j).sum()

            if union > 0:
                iou = intersection / union

                if iou > iou_threshold:
                    # Merge masks
                    current_mask = np.logical_or(current_mask, mask_j).astype(np.uint8)
                    used.add(j)

        merged.append(current_mask)
        used.add(i)

    return merged


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Convert binary mask to bounding box.

    Args:
        mask: Binary mask (H, W)

    Returns:
        (x1, y1, x2, y2) bounding box
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return (0, 0, mask.shape[1], mask.shape[0])

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return (int(x1), int(y1), int(x2 + 1), int(y2 + 1))


def upsample_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Upsample mask to target size.

    Args:
        mask: Binary mask
        target_size: (height, width)

    Returns:
        Upsampled binary mask
    """
    from PIL import Image

    # Convert to PIL Image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

    # Resize with nearest neighbor
    resized = mask_img.resize((target_size[1], target_size[0]), Image.NEAREST)

    # Convert back to numpy
    return (np.array(resized) > 127).astype(np.uint8)
