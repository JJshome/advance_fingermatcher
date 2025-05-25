"""
Advanced Deep Learning Networks for Fingerprint Feature Extraction
================================================================

This module implements state-of-the-art deep learning networks for:
1. MinutiaNet: Advanced minutiae detection with attention mechanisms
2. DescriptorNet: Vision Transformer-based descriptor extraction
3. QualityNet: Automated image quality assessment
4. FusionNet: Multi-modal feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature enhancement"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attn_output), attn_weights


class ConvBlock(nn.Module):
    """Advanced convolutional block with residual connections"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.se = SEBlock(out_channels)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out += residual
        return F.relu(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class MinutiaNet(nn.Module):
    """
    Advanced minutiae detection network with attention mechanisms
    
    Architecture:
    - Encoder: ResNet-like backbone with SE blocks
    - Decoder: Feature pyramid network with attention
    - Head: Multi-task prediction (location, orientation, type, quality)
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 3):
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleList([
            ConvBlock(input_channels, 64),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 1024, stride=2)
        ])
        
        # Feature Pyramid Network
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(128, 256, 1)
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1)
        ])
        
        # Attention mechanism
        self.attention = MultiHeadAttention(256)
        
        # Detection heads
        self.location_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)  # x, y coordinates
        )
        
        self.orientation_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)  # cos(theta), sin(theta)
        )
        
        self.type_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)  # ending, bifurcation, background
        )
        
        self.quality_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),  # quality score
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i > 0:  # Skip first layer for FPN
                features.append(x)
        
        # Feature Pyramid Network
        fpn_features = []
        last_feature = self.lateral_convs[0](features[-1])
        fpn_features.append(self.fpn_convs[0](last_feature))
        
        for i in range(1, len(features)):
            lateral = self.lateral_convs[i](features[-(i+1)])
            upsampled = F.interpolate(last_feature, scale_factor=2, mode='bilinear', align_corners=False)
            last_feature = lateral + upsampled
            fpn_features.append(self.fpn_convs[i](last_feature))
        
        # Use highest resolution feature map
        main_feature = fpn_features[-1]
        
        # Apply attention
        b, c, h, w = main_feature.shape
        flat_features = main_feature.view(b, c, h*w).transpose(1, 2)
        attended_features, attention_weights = self.attention(flat_features, flat_features, flat_features)
        attended_features = attended_features.transpose(1, 2).view(b, c, h, w)
        
        # Prediction heads
        locations = self.location_head(attended_features)
        orientations = self.orientation_head(attended_features)
        types = self.type_head(attended_features)
        qualities = self.quality_head(attended_features)
        
        return {
            'locations': locations,
            'orientations': orientations,
            'types': types,
            'qualities': qualities,
            'features': attended_features,
            'attention_weights': attention_weights
        }


class PatchEmbedding(nn.Module):
    """Patch embedding for Vision Transformer"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 1, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP"""
    
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class DescriptorNet(nn.Module):
    """
    Vision Transformer-based descriptor extraction network
    
    Extracts rich, rotation-invariant descriptors for minutiae matching
    """
    
    def __init__(self, img_size: int = 64, patch_size: int = 8, 
                 embed_dim: int = 384, n_heads: int = 6, n_layers: int = 6,
                 descriptor_dim: int = 256):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.n_patches + 1, embed_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Descriptor head
        self.descriptor_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, descriptor_dim),
            nn.LayerNorm(descriptor_dim)
        )
        
        # Rotation invariance head
        self.rotation_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)  # cos, sin for rotation
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.norm(x)
        
        # Extract class token for global representation
        cls_token = x[:, 0]
        
        # Generate descriptors
        descriptor = self.descriptor_head(cls_token)
        descriptor = F.normalize(descriptor, p=2, dim=1)  # L2 normalize
        
        # Rotation estimation
        rotation = self.rotation_head(cls_token)
        rotation = F.normalize(rotation, p=2, dim=1)  # Normalize to unit circle
        
        return {
            'descriptor': descriptor,
            'rotation': rotation,
            'cls_token': cls_token,
            'attention_maps': attention_maps
        }


class QualityNet(nn.Module):
    """
    Advanced image quality assessment network
    
    Predicts multiple quality metrics:
    - Overall quality score
    - Local quality map
    - Clarity score
    - Contrast score
    """
    
    def __init__(self, input_channels: int = 1):
        super().__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            ConvBlock(input_channels, 32),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2)
        )
        
        # Global quality assessment
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),  # overall, clarity, contrast, sharpness
            nn.Sigmoid()
        )
        
        # Local quality map
        self.local_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        
        # Global quality scores
        global_features = self.global_pool(features)
        global_quality = self.global_head(global_features)
        
        # Local quality map
        local_quality = self.local_head(features)
        local_quality = F.interpolate(local_quality, size=x.shape[-2:], 
                                    mode='bilinear', align_corners=False)
        
        return {
            'overall_quality': global_quality[:, 0],
            'clarity': global_quality[:, 1],
            'contrast': global_quality[:, 2],
            'sharpness': global_quality[:, 3],
            'local_quality': local_quality.squeeze(1),
            'features': features
        }


class FusionNet(nn.Module):
    """
    Multi-modal feature fusion network
    
    Combines outputs from MinutiaNet, DescriptorNet, and QualityNet
    for enhanced fingerprint representation
    """
    
    def __init__(self, minutia_dim: int = 256, descriptor_dim: int = 256, 
                 quality_dim: int = 512, fusion_dim: int = 512):
        super().__init__()
        
        # Feature projection layers
        self.minutia_proj = nn.Linear(minutia_dim, fusion_dim)
        self.descriptor_proj = nn.Linear(descriptor_dim, fusion_dim)
        self.quality_proj = nn.Linear(quality_dim, fusion_dim)
        
        # Cross-attention modules
        self.cross_attention = nn.ModuleList([
            MultiHeadAttention(fusion_dim, n_heads=8) for _ in range(3)
        ])
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Output heads
        self.embedding_head = nn.Linear(fusion_dim, 256)
        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, minutia_features: torch.Tensor, 
                descriptors: torch.Tensor, quality_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Project to common dimension
        minutia_proj = self.minutia_proj(minutia_features)
        descriptor_proj = self.descriptor_proj(descriptors)
        quality_proj = self.quality_proj(quality_features)
        
        # Cross-attention between modalities
        # Minutia-Descriptor attention
        md_attn, _ = self.cross_attention[0](
            minutia_proj.unsqueeze(1), descriptor_proj.unsqueeze(1), descriptor_proj.unsqueeze(1)
        )
        
        # Minutia-Quality attention
        mq_attn, _ = self.cross_attention[1](
            minutia_proj.unsqueeze(1), quality_proj.unsqueeze(1), quality_proj.unsqueeze(1)
        )
        
        # Descriptor-Quality attention
        dq_attn, _ = self.cross_attention[2](
            descriptor_proj.unsqueeze(1), quality_proj.unsqueeze(1), quality_proj.unsqueeze(1)
        )
        
        # Concatenate attended features
        fused_features = torch.cat([
            md_attn.squeeze(1), mq_attn.squeeze(1), dq_attn.squeeze(1)
        ], dim=1)
        
        # Final fusion
        fused_output = self.fusion_layers(fused_features)
        
        # Generate final embeddings
        embedding = F.normalize(self.embedding_head(fused_output), p=2, dim=1)
        confidence = self.confidence_head(fused_output)
        
        return {
            'embedding': embedding,
            'confidence': confidence.squeeze(1),
            'fused_features': fused_output
        }


class ContrastiveLoss(nn.Module):
    """Contrastive loss for descriptor learning"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive mask
        batch_size = embeddings.size(0)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask
        
        # Compute loss
        logits = similarity_matrix[~mask].view(batch_size, -1)
        targets = positive_mask[~mask].view(batch_size, -1).float()
        
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        return loss


# Training utilities
def train_networks(networks: Dict[str, nn.Module], 
                  train_loader, val_loader, 
                  num_epochs: int = 100,
                  device: str = 'cuda'):
    """
    Advanced training pipeline for all networks
    """
    # Optimizers
    optimizers = {}
    schedulers = {}
    
    for name, network in networks.items():
        network.to(device)
        optimizers[name] = torch.optim.AdamW(
            network.parameters(), lr=1e-4, weight_decay=1e-4
        )
        schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers[name], T_max=num_epochs
        )
    
    # Loss functions
    contrastive_loss = ContrastiveLoss()
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        for name, network in networks.items():
            network.train()
        
        train_losses = {name: 0.0 for name in networks.keys()}
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch.get('labels', None)
            
            # Forward pass through all networks
            minutia_out = networks['minutia'](images)
            descriptor_out = networks['descriptor'](images)
            quality_out = networks['quality'](images)
            
            # Fusion
            fusion_out = networks['fusion'](
                minutia_out['features'].mean(dim=[2, 3]),  # Global average pooling
                descriptor_out['descriptor'],
                quality_out['features'].mean(dim=[2, 3])
            )
            
            # Compute losses
            losses = {}
            
            # MinutiaNet loss (multi-task)
            if labels and 'minutia_targets' in labels:
                minutia_loss = (
                    mse_loss(minutia_out['locations'], labels['locations']) +
                    mse_loss(minutia_out['orientations'], labels['orientations']) +
                    ce_loss(minutia_out['types'], labels['types']) +
                    mse_loss(minutia_out['qualities'], labels['qualities'])
                )
                losses['minutia'] = minutia_loss
            
            # DescriptorNet loss (contrastive)
            if labels and 'identity_labels' in labels:
                descriptor_loss = contrastive_loss(
                    descriptor_out['descriptor'], labels['identity_labels']
                )
                losses['descriptor'] = descriptor_loss
            
            # QualityNet loss
            if labels and 'quality_targets' in labels:
                quality_loss = mse_loss(
                    quality_out['overall_quality'], labels['quality_targets']
                )
                losses['quality'] = quality_loss
            
            # FusionNet loss
            if labels and 'identity_labels' in labels:
                fusion_loss = contrastive_loss(
                    fusion_out['embedding'], labels['identity_labels']
                )
                losses['fusion'] = fusion_loss
            
            # Backward pass
            for name, loss in losses.items():
                optimizers[name].zero_grad()
                loss.backward(retain_graph=True)
                optimizers[name].step()
                train_losses[name] += loss.item()
        
        # Update learning rates
        for scheduler in schedulers.values():
            scheduler.step()
        
        # Validation phase
        if epoch % 10 == 0:
            val_loss = validate_networks(networks, val_loader, device)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best models
                for name, network in networks.items():
                    torch.save(
                        network.state_dict(), 
                        f'models/best_{name}_net.pth'
                    )
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for name, loss in train_losses.items():
            print(f"  {name}_loss: {loss/len(train_loader):.4f}")


def validate_networks(networks: Dict[str, nn.Module], val_loader, device: str):
    """Validation function for all networks"""
    for network in networks.values():
        network.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            
            # Forward pass
            minutia_out = networks['minutia'](images)
            descriptor_out = networks['descriptor'](images)
            quality_out = networks['quality'](images)
            
            fusion_out = networks['fusion'](
                minutia_out['features'].mean(dim=[2, 3]),
                descriptor_out['descriptor'],
                quality_out['features'].mean(dim=[2, 3])
            )
            
            # Simple validation loss (can be customized)
            loss = torch.mean(fusion_out['confidence'])
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# Factory functions
def create_advanced_networks() -> Dict[str, nn.Module]:
    """Create all advanced networks"""
    return {
        'minutia': MinutiaNet(),
        'descriptor': DescriptorNet(),
        'quality': QualityNet(),
        'fusion': FusionNet()
    }


def load_pretrained_networks(model_dir: str = 'models/') -> Dict[str, nn.Module]:
    """Load pre-trained networks"""
    networks = create_advanced_networks()
    
    for name, network in networks.items():
        try:
            state_dict = torch.load(f'{model_dir}/best_{name}_net.pth')
            network.load_state_dict(state_dict)
            print(f"Loaded pre-trained {name} network")
        except FileNotFoundError:
            print(f"Pre-trained {name} network not found, using random initialization")
    
    return networks


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create networks
    networks = create_advanced_networks()
    
    # Test forward pass
    batch_size = 4
    test_image = torch.randn(batch_size, 1, 256, 256).to(device)
    
    for name, network in networks.items():
        network.to(device)
        network.eval()
        
        if name == 'minutia':
            output = network(test_image)
            print(f"{name} output shapes:")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
        
        elif name == 'descriptor':
            test_patch = torch.randn(batch_size, 1, 64, 64).to(device)
            output = network(test_patch)
            print(f"{name} output shapes:")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
        
        elif name == 'quality':
            output = network(test_image)
            print(f"{name} output shapes:")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
