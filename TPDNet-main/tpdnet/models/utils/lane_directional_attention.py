"""
Lane Directional Attention Module
Designed specifically for lane detection task, considering the elongated structure of lanes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneDirectionalAttention(nn.Module):
    """
    Directional Attention for Lane Detection
    Considers the elongated structure of lanes by:
    1. Along-lane attention: attention along the lane direction (sample_points dimension)
    2. Cross-lane attention: attention across different lane priors
    """
    def __init__(self, in_channels, sample_points=36, reduction=4):
        super(LaneDirectionalAttention, self).__init__()
        self.in_channels = in_channels
        self.sample_points = sample_points
        self.reduction = reduction
        
        # Along-lane attention: attention along the lane direction
        # This helps capture long-range dependencies along a single lane
        self.along_lane_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Cross-lane attention: attention across different lane priors
        # This helps capture relationships between multiple lanes
        self.cross_lane_query = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )
        self.cross_lane_key = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )
        self.cross_lane_value = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # Adaptive fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(self, roi_features):
        """
        Args:
            roi_features: (Batch * num_priors, channels, sample_points, 1)
        Returns:
            enhanced_features: (Batch * num_priors, channels, sample_points, 1)
        """
        B, C, H, W = roi_features.shape
        # B = batch_size * num_priors
        # C = in_channels
        # H = sample_points
        # W = 1
        
        # Reshape for 1D convolution: (B, C, H)
        feat_1d = roi_features.squeeze(-1)  # (B, C, sample_points)
        
        # 1. Along-lane attention: attention along the lane direction
        along_att = self.along_lane_conv(feat_1d)  # (B, C, sample_points)
        along_enhanced = feat_1d * along_att  # (B, C, sample_points)
        
        # 2. Cross-lane attention: attention across different lane priors
        # Reshape to (batch_size, num_priors, C, sample_points)
        # We need to know batch_size and num_priors
        # For now, we'll use a simpler approach: self-attention on the feature dimension
        query = self.cross_lane_query(feat_1d)  # (B, C//r, sample_points)
        key = self.cross_lane_key(feat_1d)  # (B, C//r, sample_points)
        value = self.cross_lane_value(feat_1d)  # (B, C, sample_points)
        
        # Compute attention: (B, sample_points, C//r) x (B, C//r, sample_points) -> (B, sample_points, sample_points)
        query = query.permute(0, 2, 1)  # (B, sample_points, C//r)
        key = key.permute(0, 2, 1)  # (B, sample_points, C//r)
        value = value.permute(0, 2, 1)  # (B, sample_points, C)
        
        # Self-attention along sample_points dimension
        att = torch.matmul(query, key.permute(0, 2, 1))  # (B, sample_points, sample_points)
        att = att / (self.in_channels // self.reduction) ** 0.5
        att = F.softmax(att, dim=-1)
        
        cross_enhanced = torch.matmul(att, value)  # (B, sample_points, C)
        cross_enhanced = cross_enhanced.permute(0, 2, 1)  # (B, C, sample_points)
        
        # 3. Adaptive fusion
        w1 = torch.sigmoid(self.fusion_weight[0])
        w2 = torch.sigmoid(self.fusion_weight[1])
        norm = w1 + w2 + 1e-8
        enhanced = (w1 * along_enhanced + w2 * cross_enhanced + feat_1d) / norm
        
        # Reshape back: (B, C, sample_points, 1)
        enhanced = enhanced.unsqueeze(-1)
        
        return enhanced


class LaneSpatialAttention(nn.Module):
    """
    Spatial Attention for Lane Features
    Considers spatial relationships in the feature map
    """
    def __init__(self, in_channels):
        super(LaneSpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            x * attention: (B, C, H, W)
        """
        att = self.conv(x)
        return x * att

