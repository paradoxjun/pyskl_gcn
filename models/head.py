#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 paradoxjun
All rights reserved.

Author: paradoxjun
Social:
  - Douyin:    paradoxjun
  - Bilibili:  paradoxjun
  - Xiaohongshu: paradoxjun
  - CSDN:      paradoxjun

Description:
  改写自：https://github.com/kennymckormick/pyskl/blob/main/pyskl/models/heads/base.py
        https://github.com/kennymckormick/pyskl/blob/main/pyskl/models/heads/simple_head.py
"""


import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from utils.build_cnn import build_loss


def normal_init(module: nn.Module, mean: float = 0.0, std: float = 0.01, bias: float = 0.0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def top_k_accuracy(pred: torch.Tensor, target: torch.Tensor, ks=(1, 5)):
    """Compute top-k accuracy (numpy-free, CPU/GPU兼容). Return tuple of floats."""
    # pred: (N, C), target: (N,) int64
    with torch.no_grad():
        maxk = max(ks)
        # topk indices: (N, maxk)
        topk = pred.topk(maxk, dim=1, largest=True, sorted=True).indices
        # compare
        correct = topk.eq(target.view(-1, 1))
        res = []
        N = pred.size(0)
        for k in ks:
            correct_k = correct[:, :k].any(dim=1).float().sum().item()
            res.append(100.0 * correct_k / max(N, 1))
        return tuple(res)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for classification heads (aligned with PYSKL BaseHead)."""

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class: bool = False,
                 label_smooth_eps: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def loss(self, cls_score: torch.Tensor, label: torch.Tensor, **kwargs):
        """Return dict with keys: loss_cls (+ top1_acc/top5_acc when适用)."""
        losses = dict()
        # 对齐 PYSKL 的一些形状兼容
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size(0) == self.num_classes and cls_score.size(0) == 1:
            label = label.unsqueeze(0)

        # 计算 top-k（非 multi-class 或者 label 仍是 index）
        if (not self.multi_class) and (cls_score.size() != label.size()):
            top1, top5 = top_k_accuracy(cls_score.detach(), label.detach(), (1, 5))
            losses['top1_acc'] = torch.tensor(top1, device=cls_score.device)
            losses['top5_acc'] = torch.tensor(top5, device=cls_score.device)
        elif self.multi_class and self.label_smooth_eps != 0:
            # label 为 multi-hot 时才有意义
            label = (1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes

        loss_val = self.loss_cls(cls_score, label, **kwargs)
        if isinstance(loss_val, dict):
            losses.update(loss_val)
        else:
            losses['loss_cls'] = loss_val
        return losses


class SimpleHead(BaseHead):
    """A simple classification head supporting modes: '3D', 'GCN', '2D'."""

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: dict = dict(type='CrossEntropyLoss'),
                 dropout: float = 0.5,
                 init_std: float = 0.01,
                 mode: str = '3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode
        self.in_c = in_channels
        self.dropout_ratio = dropout
        self.init_std = init_std
        self.dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio != 0 else None
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """
        Inputs:
            mode='GCN' : x -> (N, M, C, T, V)
            mode='3D'  : x -> (N, C, T, H, W) or list/tuple后 concat
            mode='2D'  : x -> (N, S, C, H, W)
        Output:
            (N, num_classes)
        """
        if isinstance(x, list):
            # PYSKL: 若是 list，每项应 (B, C)；对 batch 维求均值再 stack
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                # x: (N,S,C,H,W) -> GAP(H,W) -> mean over S -> (N,C)
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C).mean(dim=1)

            if self.mode == '3D':
                # x: (N,C,T,H,W) or tuple/list -> GAP(T,H,W) -> (N,C)
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, (tuple, list)):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])

            if self.mode == 'GCN':
                # x: (N,M,C,T,V) -> reshape(N*M,C,T,V) -> GAP(T,V) -> (N,M,C) -> mean over M
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)
                x = pool(x).reshape(N, M, C).mean(dim=1)

        assert x.shape[1] == self.in_c, f"Head in_channels={self.in_c}, but got {x.shape}"
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)


class I3DHead(SimpleHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: dict = dict(type='CrossEntropyLoss'),
                 dropout: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='3D',
                         **kwargs)


class SlowFastHead(I3DHead):
    """Same as I3DHead (kept for API parity)."""
    pass


class GCNHead(SimpleHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: dict = dict(type='CrossEntropyLoss'),
                 dropout: float = 0.0,
                 init_std: float = 0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)


class TSNHead(BaseHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: dict = dict(type='CrossEntropyLoss'),
                 dropout: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         multi_class=kwargs.pop('multi_class', False),
                         label_smooth_eps=kwargs.pop('label_smooth_eps', 0.0))
        self.dropout_ratio = dropout
        self.init_std = init_std
        self.dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio != 0 else None
        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)
        self.mode = '2D'  # 固定 2D

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # x: (N,S,C,H,W) -> GAP(H,W) -> mean over S -> (N,C)
        assert len(x.shape) == 5, f"TSNHead expects (N,S,C,H,W), got {tuple(x.shape)}"
        N, S, C, H, W = x.shape
        pool = nn.AdaptiveAvgPool2d(1)
        x = x.view(N * S, C, H, W)
        x = pool(x).view(N, S, C).mean(dim=1)
        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)
