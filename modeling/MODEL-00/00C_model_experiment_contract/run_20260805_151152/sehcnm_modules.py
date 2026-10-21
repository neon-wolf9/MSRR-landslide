"""Frozen medium-scale encoders and fusion for S-EHC-Net-M."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


class StaticEncoder(nn.Module):
    def __init__(self, continuous_dim=91, category_count=14, dropout=0.10):
        super().__init__()
        if continuous_dim != 91: raise ValueError("PRIMARY static continuous_dim must be 91")
        self.continuous_dim = 91; self.category_count = int(category_count)
        self.embedding_dim = min(16, max(4, math.ceil(math.sqrt(category_count))))
        self.continuous = nn.Sequential(nn.Linear(91, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU())
        self.category_embedding = nn.Embedding(category_count, self.embedding_dim)
        self.output = nn.Sequential(nn.Linear(64 + self.embedding_dim, 64), nn.LayerNorm(64), nn.GELU())

    def forward(self, x_continuous, x_category):
        if x_continuous.ndim != 2 or x_continuous.shape[1] != 91: raise ValueError("x_continuous must be [batch,91]")
        if x_category.ndim == 2 and x_category.shape[1] == 1: x_category = x_category[:, 0]
        if x_category.ndim != 1 or len(x_category) != len(x_continuous): raise ValueError("x_category must be [batch] or [batch,1]")
        if x_category.min() < 0 or x_category.max() >= self.category_count: raise ValueError("category index out of fold vocabulary range")
        return self.output(torch.cat([self.continuous(x_continuous), self.category_embedding(x_category.long())], dim=1))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__(); self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, dilation=1, dropout=0.10):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm = nn.LayerNorm(out_channels); self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        y = self.conv(x).transpose(1, 2); y = self.dropout(F.gelu(self.norm(y))).transpose(1, 2)
        return y + self.residual(x)


class DynamicRainfallEncoder(nn.Module):
    def __init__(self, time_steps=70, input_channels=10, dropout=0.10):
        super().__init__()
        if time_steps != 70 or input_channels != 10: raise ValueError("PRIMARY dynamic shape is frozen as [batch,70,10]")
        self.time_steps=70; self.input_channels=10
        self.input_projection = nn.Conv1d(10, 32, 1)
        self.blocks = nn.ModuleList([CausalResidualBlock(32,64,3,1,dropout),
                                     CausalResidualBlock(64,64,3,2,dropout),
                                     CausalResidualBlock(64,64,3,4,dropout)])
        self.attention_score = nn.Linear(64,1)
        self.output = nn.Sequential(nn.Linear(128,64), nn.LayerNorm(64), nn.GELU())

    def encode_sequence(self, x):
        if x.ndim != 3 or x.shape[1:] != (70,10): raise ValueError("dynamic input must be [batch,70,10]; 192-step tensors are forbidden")
        h = self.input_projection(x.transpose(1,2))
        for block in self.blocks: h = block(h)
        return h.transpose(1,2)

    def forward(self, x):
        seq = self.encode_sequence(x)
        weights = torch.softmax(self.attention_score(seq).squeeze(-1), dim=1)
        pooled = (seq * weights.unsqueeze(-1)).sum(dim=1)
        return self.output(torch.cat([seq[:,-1], pooled], dim=1)), seq, weights


class GatedFusion(nn.Module):
    def __init__(self, dim=64, dropout=0.10):
        super().__init__(); self.dim=dim
        self.static_gate = nn.Linear(dim*2, dim); self.dynamic_gate = nn.Linear(dim*2, dim)
        self.norm = nn.LayerNorm(dim); self.dropout = nn.Dropout(dropout)

    def forward(self, z_static=None, z_dynamic=None, mode="full"):
        if mode == "static_only":
            if z_static is None: raise ValueError("static_only requires z_static")
            z = self.dropout(self.norm(z_static)); one=torch.ones_like(z); zero=torch.zeros_like(z)
            return z, one, zero
        if mode == "dynamic_only":
            if z_dynamic is None: raise ValueError("dynamic_only requires z_dynamic")
            z = self.dropout(self.norm(z_dynamic)); zero=torch.zeros_like(z); one=torch.ones_like(z)
            return z, zero, one
        if mode != "full" or z_static is None or z_dynamic is None: raise ValueError("full fusion requires both embeddings")
        joined=torch.cat([z_static,z_dynamic],dim=1)
        gs=torch.sigmoid(self.static_gate(joined)); gd=torch.sigmoid(self.dynamic_gate(joined))
        return self.dropout(self.norm(gs*z_static + gd*z_dynamic)), gs, gd
