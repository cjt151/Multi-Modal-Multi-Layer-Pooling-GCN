import os
import sys
import types
import logging
import torch
import torch.nn as nn
import math


class BatchedGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_feats, out_feats) * (2.0 / math.sqrt(in_feats)))
        self.bias = nn.Parameter(torch.zeros(out_feats)) if bias else None

    def forward(self, X, A_hat):
        XW = torch.matmul(X, self.weight.to(X.device))
        out = torch.bmm(A_hat, XW)
        if self.bias is not None:
            out = out + self.bias.to(out.device)
        return out


class GNNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim=128, num_classes=2, dropout=0.3, n_layers=2, pool_args=None):
        super().__init__()
        layers = []
        layers.append(BatchedGCNLayer(in_feats, hidden_dim))
        for _ in range(max(0, n_layers-1)):
            layers.append(BatchedGCNLayer(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(self.layers))])
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        self.muchpool = None
        if pool_args is not None:
            try:
                muchpool_dir = os.path.join(os.path.dirname(__file__), 'other model', 'MuchPool')
                if muchpool_dir not in sys.path:
                    sys.path.insert(0, muchpool_dir)
                from muchPool import MuchPool
                if isinstance(pool_args, dict):
                    pool_ns = types.SimpleNamespace(**pool_args)
                elif isinstance(pool_args, types.SimpleNamespace):
                    pool_ns = pool_args
                else:
                    pool_ns = pool_args
                self.muchpool = MuchPool(pool_ns)
            except Exception as e:
                logging.warning(f"无法加载 MuchPool：{e}. 将跳过池化集成。")

    def forward(self, X, A_hat):
        h = X
        for i, layer in enumerate(self.layers):
            h = layer(h, A_hat)
            h = self.layer_norms[i](h)
            h = torch.relu(h)
            if i < len(self.layers)-1:
                h = self.dropout(h)
            if self.muchpool is not None:
                try:
                    B, N, F = h.shape
                    mask = torch.ones(B, N, dtype=torch.float32, device=h.device)
                    new_H, new_adj, new_mask = self.muchpool(h, A_hat, mask)
                    h = new_H
                    A_hat = new_adj
                except Exception as e:
                    logging.warning(f"MuchPool 在前向过程中失败: {e}; 继续使用原始 h/A_hat")
        hg_mean = h.mean(dim=1)
        hg_max = h.max(dim=1)[0]
        hg = torch.cat([hg_mean, hg_max], dim=1)
        out = self.mlp(hg)
        return out
