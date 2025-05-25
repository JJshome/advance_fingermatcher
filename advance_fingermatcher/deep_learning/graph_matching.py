"""
Graph Neural Network-based Fingerprint Matching System
=====================================================

This module implements state-of-the-art graph neural networks for fingerprint matching:
1. GraphMatchNet: GNN-based minutiae graph matching
2. SuperGlue-inspired matching with attention mechanisms
3. Differentiable matching for end-to-end learning
4. Advanced message passing and graph attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import math
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add, scatter_max


class GraphAttention(nn.Module):
    """Multi-head graph attention mechanism"""
    
    def __init__(self, node_dim: int, edge_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.d_k = node_dim // n_heads
        
        # Node transformations
        self.w_q = nn.Linear(node_dim, node_dim)
        self.w_k = nn.Linear(node_dim, node_dim)
        self.w_v = nn.Linear(node_dim, node_dim)
        
        # Edge transformation
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        
        self.w_o = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        N = nodes.size(0)
        
        # Transform nodes
        Q = self.w_q(nodes).view(N, self.n_heads, self.d_k)
        K = self.w_k(nodes).view(N, self.n_heads, self.d_k)
        V = self.w_v(nodes).view(N, self.n_heads, self.d_k)
        
        # Compute attention scores
        row, col = edge_index
        
        # Query-Key interaction
        qk_scores = (Q[row] * K[col]).sum(dim=-1) / math.sqrt(self.d_k)  # [num_edges, n_heads]
        
        # Include edge features if available
        if edge_attr is not None:
            edge_features = self.edge_proj(edge_attr)
            edge_features = edge_features.view(-1, self.n_heads, self.d_k)
            edge_contrib = (Q[row] * edge_features).sum(dim=-1) / math.sqrt(self.d_k)
            qk_scores = qk_scores + edge_contrib
        
        # Softmax normalization
        attention_weights = torch.zeros(N, self.n_heads, device=nodes.device)
        attention_weights.index_add_(0, row, torch.exp(qk_scores))
        attention_weights = attention_weights[row]
        attention_weights = torch.exp(qk_scores) / (attention_weights + 1e-8)
        
        # Apply attention to values
        messages = attention_weights.unsqueeze(-1) * V[col]  # [num_edges, n_heads, d_k]
        
        # Aggregate messages
        out = torch.zeros(N, self.n_heads, self.d_k, device=nodes.device)
        out.index_add_(0, row, messages)
        out = out.view(N, -1)
        
        return self.w_o(out)


class MinutiaeGraphConv(MessagePassing):
    """Custom graph convolution for minutiae features"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = None):
        super().__init__(aggr='add')
        
        if hidden_dim is None:
            hidden_dim = node_dim
        
        # Message computation
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Update computation
        self.update_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, node_dim)
        )
        
        self.norm = nn.LayerNorm(node_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Residual connection and normalization
        out = self.norm(x + out)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        # Concatenate node features
        msg_input = torch.cat([x_i, x_j], dim=-1)
        
        # Include edge attributes if available
        if edge_attr is not None:
            msg_input = torch.cat([msg_input, edge_attr], dim=-1)
        
        return self.message_net(msg_input)
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.update_net(torch.cat([x, aggr_out], dim=-1))


class GraphMatchNet(nn.Module):
    """
    Advanced Graph Neural Network for fingerprint matching
    
    Architecture inspired by SuperGlue but adapted for minutiae matching
    """
    
    def __init__(self, node_dim: int = 256, edge_dim: int = 64, 
                 hidden_dim: int = 256, n_layers: int = 6, n_heads: int = 8):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_layers = n_layers
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),  # x, y, theta, quality, type (one-hot: 2)
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, node_dim),
            nn.LayerNorm(node_dim)
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # distance, relative_angle1, relative_angle2
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, edge_dim),
            nn.LayerNorm(edge_dim)
        )
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            MinutiaeGraphConv(node_dim, edge_dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Self-attention layers
        self.self_attentions = nn.ModuleList([
            nn.MultiheadAttention(node_dim, n_heads, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Cross-attention layers for matching
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(node_dim, n_heads, dropout=0.1, batch_first=True)
            for _ in range(n_layers // 2)
        ])
        
        # Final matching head
        self.matching_head = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Dustbin for non-matching minutiae
        self.dustbin = nn.Parameter(torch.tensor(1.0))
        
    def build_graph(self, minutiae: torch.Tensor, max_distance: float = 200.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from minutiae points
        
        Args:
            minutiae: [N, 5] tensor (x, y, theta, quality, type)
            max_distance: Maximum distance for edge creation
            
        Returns:
            edge_index: [2, E] edge indices
            edge_attr: [E, 3] edge attributes (distance, rel_angle1, rel_angle2)
        """
        N = minutiae.size(0)
        device = minutiae.device
        
        # Compute pairwise distances
        positions = minutiae[:, :2]  # x, y coordinates
        dist_matrix = torch.cdist(positions, positions)
        
        # Create edges for nearby minutiae
        edge_mask = (dist_matrix < max_distance) & (dist_matrix > 0)
        edge_index = edge_mask.nonzero().t()
        
        if edge_index.size(1) == 0:
            # If no edges, create self-loops
            edge_index = torch.arange(N, device=device).repeat(2, 1)
            edge_attr = torch.zeros(N, 3, device=device)
        else:
            # Compute edge attributes
            i, j = edge_index
            
            # Distance
            distances = dist_matrix[i, j]
            
            # Connection angle
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            connection_angles = torch.atan2(dy, dx)
            
            # Relative angles
            theta_i = minutiae[i, 2]
            theta_j = minutiae[j, 2]
            
            rel_angle1 = self._normalize_angle(theta_i - connection_angles)
            rel_angle2 = self._normalize_angle(theta_j - connection_angles)
            
            edge_attr = torch.stack([distances, rel_angle1, rel_angle2], dim=1)
        
        return edge_index, edge_attr
    
    def _normalize_angle(self, angles: torch.Tensor) -> torch.Tensor:
        """Normalize angles to [-pi, pi]"""
        return torch.atan2(torch.sin(angles), torch.cos(angles))
    
    def forward(self, minutiae1: torch.Tensor, minutiae2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for graph matching
        
        Args:
            minutiae1: [N1, 5] probe minutiae
            minutiae2: [N2, 5] gallery minutiae
            
        Returns:
            Dictionary containing matching scores and features
        """
        device = minutiae1.device
        N1, N2 = minutiae1.size(0), minutiae2.size(0)
        
        # Encode minutiae types (one-hot)
        def encode_minutiae(minutiae):
            # Convert type to one-hot (assuming 0=ending, 1=bifurcation)
            types = minutiae[:, 4].long()
            type_onehot = F.one_hot(types, num_classes=2).float()
            return torch.cat([minutiae[:, :4], type_onehot], dim=1)
        
        minutiae1_encoded = encode_minutiae(minutiae1)
        minutiae2_encoded = encode_minutiae(minutiae2)
        
        # Build graphs
        edge_index1, edge_attr1 = self.build_graph(minutiae1)
        edge_index2, edge_attr2 = self.build_graph(minutiae2)
        
        # Encode node and edge features
        node_features1 = self.node_encoder(minutiae1_encoded)
        node_features2 = self.node_encoder(minutiae2_encoded)
        
        edge_features1 = self.edge_encoder(edge_attr1)
        edge_features2 = self.edge_encoder(edge_attr2)
        
        # Graph neural network processing
        for i, (gnn_layer, self_attn) in enumerate(zip(self.gnn_layers, self.self_attentions)):
            # GNN message passing
            node_features1 = gnn_layer(node_features1, edge_index1, edge_features1)
            node_features2 = gnn_layer(node_features2, edge_index2, edge_features2)
            
            # Self-attention within each graph
            node_features1_attn, _ = self_attn(
                node_features1.unsqueeze(0), node_features1.unsqueeze(0), node_features1.unsqueeze(0)
            )
            node_features2_attn, _ = self_attn(
                node_features2.unsqueeze(0), node_features2.unsqueeze(0), node_features2.unsqueeze(0)
            )
            
            node_features1 = node_features1 + node_features1_attn.squeeze(0)
            node_features2 = node_features2 + node_features2_attn.squeeze(0)
            
            # Cross-attention for matching (every other layer)
            if i < len(self.cross_attentions):
                cross_attn = self.cross_attentions[i]
                
                # Cross-attention: minutiae1 queries minutiae2
                cross_features1, cross_weights1 = cross_attn(
                    node_features1.unsqueeze(0), node_features2.unsqueeze(0), node_features2.unsqueeze(0)
                )
                
                # Cross-attention: minutiae2 queries minutiae1
                cross_features2, cross_weights2 = cross_attn(
                    node_features2.unsqueeze(0), node_features1.unsqueeze(0), node_features1.unsqueeze(0)
                )
                
                node_features1 = node_features1 + cross_features1.squeeze(0)
                node_features2 = node_features2 + cross_features2.squeeze(0)
        
        # Compute matching scores
        # Create all possible pairs
        features1_expanded = node_features1.unsqueeze(1).expand(N1, N2, -1)
        features2_expanded = node_features2.unsqueeze(0).expand(N1, N2, -1)
        
        pair_features = torch.cat([features1_expanded, features2_expanded], dim=-1)
        matching_scores = self.matching_head(pair_features).squeeze(-1)  # [N1, N2]
        
        # Add dustbin for unmatched minutiae
        dustbin_scores1 = self.dustbin.expand(N1, 1)
        dustbin_scores2 = self.dustbin.expand(1, N2)
        
        # Augmented score matrix
        scores_augmented = torch.cat([
            torch.cat([matching_scores, dustbin_scores1], dim=1),
            torch.cat([dustbin_scores2, torch.zeros(1, 1, device=device)], dim=1)
        ], dim=0)
        
        # Apply softmax for assignment probabilities
        assignment_probs = F.softmax(scores_augmented, dim=1)
        assignment_probs = F.softmax(assignment_probs, dim=0)
        
        # Extract matching probabilities (exclude dustbin)
        matching_probs = assignment_probs[:N1, :N2]
        
        return {
            'matching_scores': matching_scores,
            'matching_probs': matching_probs,
            'features1': node_features1,
            'features2': node_features2,
            'assignment_matrix': assignment_probs
        }


class DifferentiableMatching(nn.Module):
    """Differentiable matching layer for end-to-end learning"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, scores: torch.Tensor, return_soft: bool = True) -> torch.Tensor:
        """
        Differentiable matching using Gumbel-Softmax
        
        Args:
            scores: [N1, N2] matching scores
            return_soft: If True, return soft assignment; else return hard assignment
        """
        if return_soft:
            # Soft assignment using Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            soft_assignment = F.softmax((scores + gumbel_noise) / self.temperature, dim=1)
            return soft_assignment
        else:
            # Hard assignment using Hungarian algorithm approximation
            return self.hungarian_assignment(scores)
    
    def hungarian_assignment(self, scores: torch.Tensor) -> torch.Tensor:
        """Approximate Hungarian assignment using Sinkhorn iterations"""
        log_alpha = scores / self.temperature
        
        for _ in range(20):  # Sinkhorn iterations
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
        
        return torch.exp(log_alpha)


class GeometricVerification(nn.Module):
    """Geometric verification for matched minutiae pairs"""
    
    def __init__(self, inlier_threshold: float = 10.0, min_inliers: int = 4):
        super().__init__()
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
        
    def forward(self, minutiae1: torch.Tensor, minutiae2: torch.Tensor, 
                matches: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform geometric verification using RANSAC
        
        Args:
            minutiae1: [N1, 5] probe minutiae
            minutiae2: [N2, 5] gallery minutiae  
            matches: [M, 2] matched pairs (indices)
            
        Returns:
            Dictionary with verification results
        """
        if matches.size(0) < self.min_inliers:
            return {
                'is_valid': torch.tensor(False),
                'inlier_mask': torch.zeros(matches.size(0), dtype=torch.bool),
                'transformation': torch.eye(3),
                'num_inliers': torch.tensor(0)
            }
        
        best_inliers = 0
        best_mask = torch.zeros(matches.size(0), dtype=torch.bool)
        best_transform = torch.eye(3)
        
        num_iterations = min(1000, matches.size(0) * 10)
        
        for _ in range(num_iterations):
            # Sample minimal set (2 points for similarity transform)
            if matches.size(0) >= 2:
                sample_indices = torch.randperm(matches.size(0))[:2]
                sample_matches = matches[sample_indices]
                
                # Compute transformation
                transform = self._compute_similarity_transform(
                    minutiae1[sample_matches[:, 0]], 
                    minutiae2[sample_matches[:, 1]]
                )
                
                # Count inliers
                inlier_mask = self._count_inliers(
                    minutiae1, minutiae2, matches, transform
                )
                
                num_inliers = inlier_mask.sum().item()
                
                if num_inliers > best_inliers:
                    best_inliers = num_inliers
                    best_mask = inlier_mask
                    best_transform = transform
        
        return {
            'is_valid': torch.tensor(best_inliers >= self.min_inliers),
            'inlier_mask': best_mask,
            'transformation': best_transform,
            'num_inliers': torch.tensor(best_inliers)
        }
    
    def _compute_similarity_transform(self, points1: torch.Tensor, 
                                    points2: torch.Tensor) -> torch.Tensor:
        """Compute similarity transformation between two point sets"""
        if points1.size(0) < 2:
            return torch.eye(3)
        
        # Extract positions
        p1 = points1[:, :2]  # [N, 2]
        p2 = points2[:, :2]  # [N, 2]
        
        # Compute centroids
        c1 = p1.mean(dim=0)
        c2 = p2.mean(dim=0)
        
        # Center points
        p1_centered = p1 - c1
        p2_centered = p2 - c2
        
        # Compute scale
        scale1 = torch.norm(p1_centered, dim=1).mean()
        scale2 = torch.norm(p2_centered, dim=1).mean()
        scale = scale2 / (scale1 + 1e-8)
        
        # Compute rotation
        if points1.size(0) >= 2:
            # Use first two points to estimate rotation
            v1 = p1_centered[1] - p1_centered[0]
            v2 = p2_centered[1] - p2_centered[0]
            
            angle1 = torch.atan2(v1[1], v1[0])
            angle2 = torch.atan2(v2[1], v2[0])
            rotation_angle = angle2 - angle1
            
            cos_a = torch.cos(rotation_angle)
            sin_a = torch.sin(rotation_angle)
        else:
            cos_a = torch.tensor(1.0)
            sin_a = torch.tensor(0.0)
        
        # Construct transformation matrix
        R = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ]) * scale
        
        t = c2 - R @ c1
        
        # 3x3 homogeneous transformation matrix
        transform = torch.eye(3)
        transform[:2, :2] = R
        transform[:2, 2] = t
        
        return transform
    
    def _count_inliers(self, minutiae1: torch.Tensor, minutiae2: torch.Tensor,
                      matches: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Count inliers for given transformation"""
        # Extract matched points
        p1 = minutiae1[matches[:, 0], :2]  # [M, 2]
        p2 = minutiae2[matches[:, 1], :2]  # [M, 2]
        
        # Transform p1 to p2 coordinate system
        p1_homo = torch.cat([p1, torch.ones(p1.size(0), 1)], dim=1)  # [M, 3]
        p1_transformed = (transform @ p1_homo.t()).t()[:, :2]  # [M, 2]
        
        # Compute distances
        distances = torch.norm(p1_transformed - p2, dim=1)
        
        # Inlier mask
        inlier_mask = distances < self.inlier_threshold
        
        return inlier_mask


class AdvancedGraphMatcher(nn.Module):
    """
    Complete advanced graph-based fingerprint matching system
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        if config is None:
            config = {}
        
        # Default configuration
        self.config = {
            'node_dim': 256,
            'edge_dim': 64,
            'hidden_dim': 256,
            'n_layers': 6,
            'n_heads': 8,
            'temperature': 0.1,
            'inlier_threshold': 10.0,
            'min_inliers': 4,
            **config
        }
        
        # Core components
        self.graph_matcher = GraphMatchNet(
            node_dim=self.config['node_dim'],
            edge_dim=self.config['edge_dim'],
            hidden_dim=self.config['hidden_dim'],
            n_layers=self.config['n_layers'],
            n_heads=self.config['n_heads']
        )
        
        self.differentiable_matching = DifferentiableMatching(
            temperature=self.config['temperature']
        )
        
        self.geometric_verification = GeometricVerification(
            inlier_threshold=self.config['inlier_threshold'],
            min_inliers=self.config['min_inliers']
        )
        
    def forward(self, minutiae1: torch.Tensor, minutiae2: torch.Tensor,
                return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Complete matching pipeline
        
        Args:
            minutiae1: [N1, 5] probe minutiae
            minutiae2: [N2, 5] gallery minutiae
            return_all: If True, return intermediate results
            
        Returns:
            Dictionary with matching results
        """
        # Graph-based matching
        graph_results = self.graph_matcher(minutiae1, minutiae2)
        
        # Differentiable assignment
        soft_assignment = self.differentiable_matching(
            graph_results['matching_scores'], return_soft=True
        )
        hard_assignment = self.differentiable_matching(
            graph_results['matching_scores'], return_soft=False
        )
        
        # Extract matches above threshold
        threshold = 0.5
        match_mask = hard_assignment > threshold
        matches = match_mask.nonzero()
        
        # Geometric verification
        verification_results = self.geometric_verification(
            minutiae1, minutiae2, matches
        )
        
        # Compute final matching score
        if verification_results['is_valid']:
            # Score based on number of verified inliers
            num_inliers = verification_results['num_inliers'].float()
            total_minutiae = min(minutiae1.size(0), minutiae2.size(0))
            match_score = (num_inliers / total_minutiae).clamp(0, 1)
        else:
            match_score = torch.tensor(0.0)
        
        results = {
            'match_score': match_score,
            'matches': matches,
            'verified_matches': matches[verification_results['inlier_mask']],
            'transformation': verification_results['transformation'],
            'is_valid': verification_results['is_valid']
        }
        
        if return_all:
            results.update({
                'graph_results': graph_results,
                'soft_assignment': soft_assignment,
                'hard_assignment': hard_assignment,
                'verification_results': verification_results
            })
        
        return results


# Training utilities for graph networks
class GraphMatchingLoss(nn.Module):
    """Custom loss function for graph matching"""
    
    def __init__(self, positive_weight: float = 1.0, negative_weight: float = 1.0,
                 geometric_weight: float = 0.1):
        super().__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.geometric_weight = geometric_weight
        
    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute matching loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        """
        losses = {}
        
        # Assignment loss
        if 'assignment_matrix' in predictions and 'gt_assignment' in targets:
            assignment_pred = predictions['assignment_matrix']
            assignment_gt = targets['gt_assignment']
            
            # Binary cross-entropy loss
            assignment_loss = F.binary_cross_entropy_with_logits(
                assignment_pred, assignment_gt.float()
            )
            losses['assignment'] = assignment_loss
        
        # Matching probability loss
        if 'matching_probs' in predictions and 'gt_matches' in targets:
            match_probs = predictions['matching_probs']
            gt_matches = targets['gt_matches']
            
            # Positive and negative samples
            pos_mask = gt_matches > 0
            neg_mask = gt_matches == 0
            
            pos_loss = -torch.log(match_probs[pos_mask] + 1e-8).mean()
            neg_loss = -torch.log(1 - match_probs[neg_mask] + 1e-8).mean()
            
            matching_loss = (
                self.positive_weight * pos_loss + 
                self.negative_weight * neg_loss
            )
            losses['matching'] = matching_loss
        
        # Geometric consistency loss
        if 'transformation' in predictions and 'gt_transformation' in targets:
            pred_transform = predictions['transformation']
            gt_transform = targets['gt_transformation']
            
            geometric_loss = F.mse_loss(pred_transform, gt_transform)
            losses['geometric'] = self.geometric_weight * geometric_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses


def create_graph_matcher(config: Dict = None) -> AdvancedGraphMatcher:
    """Factory function to create advanced graph matcher"""
    return AdvancedGraphMatcher(config)


def train_graph_matcher(model: AdvancedGraphMatcher, train_loader, val_loader,
                       num_epochs: int = 100, device: str = 'cuda'):
    """Training function for graph matcher"""
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = GraphMatchingLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            minutiae1 = batch['minutiae1'].to(device)
            minutiae2 = batch['minutiae2'].to(device)
            targets = batch['targets']
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(minutiae1, minutiae2, return_all=True)
            
            # Compute loss
            losses = criterion(predictions, targets)
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(losses['total'].item())
        
        # Validation phase
        if epoch % 10 == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    minutiae1 = batch['minutiae1'].to(device)
                    minutiae2 = batch['minutiae2'].to(device)
                    targets = batch['targets']
                    
                    predictions = model(minutiae1, minutiae2, return_all=True)
                    losses = criterion(predictions, targets)
                    val_losses.append(losses['total'].item())
            
            val_loss = np.mean(val_losses)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'models/best_graph_matcher.pth')
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {np.mean(train_losses):.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
        
        scheduler.step()


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    matcher = create_graph_matcher()
    matcher.to(device)
    matcher.eval()
    
    # Test with dummy data
    batch_size = 2
    n1, n2 = 15, 12
    
    # Random minutiae (x, y, theta, quality, type)
    minutiae1 = torch.rand(batch_size, n1, 5).to(device)
    minutiae2 = torch.rand(batch_size, n2, 5).to(device)
    
    # Ensure type is integer (0 or 1)
    minutiae1[..., 4] = torch.randint(0, 2, (batch_size, n1)).float()
    minutiae2[..., 4] = torch.randint(0, 2, (batch_size, n2)).float()
    
    # Forward pass
    with torch.no_grad():
        results = matcher(minutiae1[0], minutiae2[0], return_all=True)
        
        print("Graph Matching Results:")
        print(f"  Match Score: {results['match_score']:.4f}")
        print(f"  Number of Matches: {results['matches'].size(0)}")
        print(f"  Verified Matches: {results['verified_matches'].size(0)}")
        print(f"  Geometric Verification: {results['is_valid'].item()}")
