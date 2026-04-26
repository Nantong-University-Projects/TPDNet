import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologyPositionReasoner(nn.Module):
    def __init__(self,
                 feat_dim,
                 num_heads=4,
                 rel_hidden_dim=64,
                 dropout=0.1,
                 n_offsets=72):
        super(TopologyPositionReasoner, self).__init__()
        if feat_dim % num_heads != 0:
            raise ValueError('feat_dim must be divisible by num_heads')

        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.n_offsets = n_offsets

        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)

        self.rel_encoder = nn.Sequential(
            nn.Linear(11, rel_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(rel_hidden_dim, rel_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.rel_bias = nn.Linear(rel_hidden_dim, num_heads)
        self.rel_value = nn.Linear(rel_hidden_dim, feat_dim)

        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def _pairwise_relative_geometry(self, lane_geometry):
        """
        Args:
            lane_geometry: Tensor with shape (B, N, 6 + n_offsets). The layout
                follows CLRNet predictions: scores, start_y, start_x, theta,
                length, and sampled x coordinates.

        Returns:
            Tensor with shape (B, N, N, 11), where item [i, j] encodes lane j
            relative to lane i.
        """
        start_y = lane_geometry[..., 2]
        start_x = lane_geometry[..., 3]
        theta = lane_geometry[..., 4]
        length = lane_geometry[..., 5] / max(float(self.n_offsets), 1.0)
        lane_xs = lane_geometry[..., 6:]

        dx_start = start_x.unsqueeze(1) - start_x.unsqueeze(2)
        dy_start = start_y.unsqueeze(1) - start_y.unsqueeze(2)
        dtheta = theta.unsqueeze(1) - theta.unsqueeze(2)
        dlength = length.unsqueeze(1) - length.unsqueeze(2)

        valid = (lane_xs >= 0.) & (lane_xs <= 1.)
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
        valid_count = pair_valid.float().sum(dim=-1).clamp(min=1.0)

        dx_along_lane = lane_xs.unsqueeze(1) - lane_xs.unsqueeze(2)
        dx_masked = dx_along_lane * pair_valid.float()
        mean_dx = dx_masked.sum(dim=-1) / valid_count
        mean_abs_dx = dx_masked.abs().sum(dim=-1) / valid_count
        overlap = pair_valid.float().mean(dim=-1)

        bottom_dx = lane_xs[..., 0].unsqueeze(1) - lane_xs[..., 0].unsqueeze(2)
        top_dx = lane_xs[..., -1].unsqueeze(1) - lane_xs[..., -1].unsqueeze(2)

        rel = torch.stack(
            [
                dx_start,
                dy_start,
                dtheta,
                torch.sin(math.pi * dtheta),
                torch.cos(math.pi * dtheta),
                dlength,
                mean_dx,
                mean_abs_dx,
                bottom_dx,
                top_dx,
                overlap,
            ],
            dim=-1)
        return rel

    def forward(self, lane_feats, lane_geometry):
        """
        Args:
            lane_feats: Tensor with shape (B, N, D).
            lane_geometry: Tensor with shape (B, N, 6 + n_offsets).

        Returns:
            refined_feats: Tensor with shape (B, N, D).
            attn: Tensor with shape (B, num_heads, N, N).
        """
        batch_size, num_priors, _ = lane_feats.shape

        q = self.q_proj(lane_feats)
        k = self.k_proj(lane_feats)
        v = self.v_proj(lane_feats)

        q = q.view(batch_size, num_priors, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_priors, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_priors, self.num_heads,
                   self.head_dim).transpose(1, 2)

        rel_geometry = self._pairwise_relative_geometry(lane_geometry)
        rel_embed = self.rel_encoder(rel_geometry)
        rel_bias = self.rel_bias(rel_embed).permute(0, 3, 1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(self.head_dim)
        attn_logits = attn_logits + rel_bias
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        content_msg = torch.matmul(attn, v)

        rel_msg = self.rel_value(rel_embed)
        rel_msg = rel_msg.view(batch_size, num_priors, num_priors,
                               self.num_heads, self.head_dim)
        rel_msg = rel_msg.permute(0, 3, 1, 2, 4)
        rel_msg = (attn.unsqueeze(-1) * rel_msg).sum(dim=3)

        msg = content_msg + rel_msg
        msg = msg.transpose(1, 2).contiguous().view(batch_size, num_priors,
                                                    self.feat_dim)
        msg = self.out_proj(msg)

        refined_feats = self.norm1(lane_feats + self.dropout(msg))
        refined_feats = self.norm2(refined_feats +
                                   self.dropout(self.ffn(refined_feats)))
        return refined_feats, attn
