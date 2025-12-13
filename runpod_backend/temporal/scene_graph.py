"""
Scene/Interaction Graph

Build and reason over object relationships across time.
Uses Graph Neural Networks to model object interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GraphNode:
    """Node in the scene graph (represents an object track)."""
    track_id: int
    features: torch.Tensor  # Aggregated object features
    position: Optional[Tuple[float, float]] = None  # Spatial position


@dataclass
class GraphEdge:
    """Edge in the scene graph (represents relationship)."""
    source_id: int
    target_id: int
    features: torch.Tensor  # Relationship features
    edge_type: Optional[str] = None  # e.g., "near", "interacts_with"


class GraphConv(nn.Module):
    """
    Graph Convolution Layer.

    Aggregates node features based on edges.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        # Edge-conditioned aggregation
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply graph convolution.

        Args:
            node_features: (num_nodes, in_dim)
            edge_index: (2, num_edges) source and target indices
            edge_features: (num_edges, in_dim)

        Returns:
            (num_nodes, out_dim) updated node features
        """
        num_nodes = node_features.shape[0]

        # Message passing
        source_idx = edge_index[0]
        target_idx = edge_index[1]

        # Get source node features for each edge
        source_features = node_features[source_idx]  # (num_edges, in_dim)

        # Combine with edge features
        messages = source_features * self.edge_mlp(edge_features)

        # Aggregate messages to target nodes
        aggregated = torch.zeros_like(node_features)
        aggregated.scatter_add_(0, target_idx.unsqueeze(-1).expand(-1, node_features.shape[-1]), messages)

        # Count edges per node for normalization
        edge_counts = torch.zeros(num_nodes, device=node_features.device)
        edge_counts.scatter_add_(0, target_idx, torch.ones_like(target_idx, dtype=torch.float))
        edge_counts = edge_counts.clamp(min=1).unsqueeze(-1)

        aggregated = aggregated / edge_counts

        # Update nodes
        combined = torch.cat([node_features, aggregated], dim=-1)
        updated = self.node_mlp(combined)

        return updated


class SceneGraph(nn.Module):
    """
    Build and reason over object relationships across time.

    Creates a graph where:
    - Nodes = tracked objects
    - Edges = relationships (spatial, temporal, semantic)

    Args:
        node_dim: Object feature dimension (4096)
        edge_dim: Edge feature dimension
        num_layers: Number of GNN layers
    """

    def __init__(
        self,
        node_dim: int = 4096,
        edge_dim: int = 1024,
        num_layers: int = 3,
        num_edge_types: int = 4,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Node encoder
        self.node_encoder = nn.Linear(node_dim, node_dim)

        # Edge encoder
        self.edge_encoder = nn.Linear(edge_dim, node_dim)

        # Edge predictor (from node pairs)
        self.edge_predictor = nn.Sequential(
            nn.Linear(node_dim * 2, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # Edge type classifier
        self.edge_type_classifier = nn.Linear(edge_dim, num_edge_types)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConv(node_dim, node_dim) for _ in range(num_layers)
        ])

        # Graph-level pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.GELU(),
            nn.Linear(node_dim // 2, node_dim),
        )

    def forward(
        self,
        tracks: Dict[int, List[torch.Tensor]],
        spatial_positions: Optional[Dict[int, List[Tuple]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build scene graph from tracked objects.

        Args:
            tracks: Dictionary mapping track_id -> list of embeddings
            spatial_positions: Optional dict of track_id -> list of (x, y) positions

        Returns:
            Dictionary with:
                - node_features: (num_nodes, node_dim) updated node features
                - edge_features: (num_edges, edge_dim) edge features
                - edge_index: (2, num_edges) edge indices
                - graph_embedding: (node_dim,) global graph embedding
        """
        if not tracks:
            return {
                'node_features': torch.zeros(0, self.node_dim),
                'edge_features': torch.zeros(0, self.edge_dim),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'graph_embedding': torch.zeros(self.node_dim),
            }

        # Build nodes from tracks
        track_ids = list(tracks.keys())
        device = next(iter(tracks.values()))[0].device if tracks else 'cpu'

        node_features = []
        for track_id in track_ids:
            embeddings = tracks[track_id]
            if isinstance(embeddings, list):
                track_emb = torch.stack(embeddings).mean(dim=0)
            else:
                track_emb = embeddings
            node_features.append(track_emb)

        node_features = torch.stack(node_features)  # (num_nodes, dim)
        node_features = self.node_encoder(node_features)

        # Build edges (fully connected for now)
        num_nodes = len(track_ids)
        edge_index, edge_features = self._build_edges(
            node_features, track_ids, spatial_positions
        )

        # Apply GNN layers
        for gnn in self.gnn_layers:
            if edge_index.shape[1] > 0:
                node_features = gnn(node_features, edge_index, edge_features)
            else:
                # No edges, just apply node MLP
                node_features = self.gnn_layers[0].node_mlp(
                    torch.cat([node_features, node_features], dim=-1)
                )

        # Graph-level embedding
        graph_embedding = self.graph_pool(node_features.mean(dim=0))

        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_index': edge_index,
            'graph_embedding': graph_embedding,
            'track_id_to_idx': {tid: i for i, tid in enumerate(track_ids)},
        }

    def _build_edges(
        self,
        node_features: torch.Tensor,
        track_ids: List[int],
        spatial_positions: Optional[Dict[int, List[Tuple]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges between nodes.

        Creates edges based on feature similarity and spatial proximity.

        Returns:
            Tuple of (edge_index, edge_features)
        """
        num_nodes = node_features.shape[0]
        device = node_features.device

        if num_nodes < 2:
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, self.node_dim, device=device),
            )

        edges_src = []
        edges_tgt = []
        edge_feats = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Predict edge features
                    edge_input = torch.cat([node_features[i], node_features[j]])
                    edge_feat = self.edge_predictor(edge_input)

                    edges_src.append(i)
                    edges_tgt.append(j)
                    edge_feats.append(edge_feat)

        if not edges_src:
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, self.node_dim, device=device),
            )

        edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long, device=device)
        edge_features = torch.stack(edge_feats)
        edge_features = self.edge_encoder(edge_features)

        return edge_index, edge_features

    def predict_edge_types(
        self,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict edge types from edge features.

        Args:
            edge_features: (num_edges, edge_dim)

        Returns:
            (num_edges, num_types) edge type logits
        """
        return self.edge_type_classifier(edge_features)

    def get_object_interactions(
        self,
        graph_output: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> List[Tuple[int, int, str]]:
        """
        Extract significant interactions from graph.

        Args:
            graph_output: Output from forward()
            threshold: Minimum edge strength for interaction

        Returns:
            List of (source_track_id, target_track_id, interaction_type)
        """
        edge_features = graph_output['edge_features']
        edge_index = graph_output['edge_index']
        track_id_to_idx = graph_output['track_id_to_idx']
        idx_to_track_id = {v: k for k, v in track_id_to_idx.items()}

        if edge_features.shape[0] == 0:
            return []

        # Get edge type predictions
        edge_logits = self.predict_edge_types(edge_features)
        edge_probs = F.softmax(edge_logits, dim=-1)
        max_probs, edge_types = edge_probs.max(dim=-1)

        # Filter by threshold
        interactions = []
        edge_type_names = ['near', 'interacts_with', 'moves_towards', 'moves_away']

        for i in range(edge_index.shape[1]):
            if max_probs[i] >= threshold:
                src_idx = edge_index[0, i].item()
                tgt_idx = edge_index[1, i].item()
                edge_type = edge_types[i].item()

                src_track = idx_to_track_id.get(src_idx, src_idx)
                tgt_track = idx_to_track_id.get(tgt_idx, tgt_idx)
                type_name = edge_type_names[edge_type] if edge_type < len(edge_type_names) else f'type_{edge_type}'

                interactions.append((src_track, tgt_track, type_name))

        return interactions


class TemporalSceneGraph(nn.Module):
    """
    Scene graph that evolves over time.

    Maintains graph structure across frames and updates edges
    based on temporal relationships.
    """

    def __init__(
        self,
        node_dim: int = 4096,
        edge_dim: int = 1024,
        num_layers: int = 3,
    ):
        super().__init__()

        self.scene_graph = SceneGraph(node_dim, edge_dim, num_layers)

        # Temporal edge updater
        self.temporal_edge_update = nn.GRUCell(edge_dim, edge_dim)

        # Temporal node updater
        self.temporal_node_update = nn.GRUCell(node_dim, node_dim)

    def forward(
        self,
        frame_tracks: List[Dict[int, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Build temporal scene graph from sequence of frames.

        Args:
            frame_tracks: List of dicts, each mapping track_id -> embedding

        Returns:
            Final scene graph output
        """
        if not frame_tracks:
            return self.scene_graph({})

        # Aggregate tracks across frames
        aggregated_tracks = {}

        for frame_track in frame_tracks:
            for track_id, embedding in frame_track.items():
                if track_id not in aggregated_tracks:
                    aggregated_tracks[track_id] = []
                aggregated_tracks[track_id].append(embedding)

        return self.scene_graph(aggregated_tracks)
