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
  对 mmcv.cnn 中的依赖项 build_activation_layer, build_norm_layer 进行实现。
  损失函数实现参考： https://github.com/kennymckormick/pyskl/blob/main/pyskl/models/losses/cross_entropy_loss.py
"""


import copy
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union


__all__ = [
    'build_norm_layer', 'build_activation_layer', 'build_loss',
    '_build_norm_layer', '_build_activation_layer', '_build_loss'
]

# ---------------- Norm builder (兼容 mmcv) ----------------
_NORM_ABBR = {
    'BN': 'bn', 'BN1d': 'bn', 'BN2d': 'bn', 'BN3d': 'bn',
    'SyncBN': 'bn', 'NaiveSyncBN': 'bn',
    'IN': 'in', 'IN1d': 'in', 'IN2d': 'in', 'IN3d': 'in',
    'GN': 'gn', 'LN': 'ln',
    'Identity': 'id', 'None': 'id'
}


def build_norm_layer(norm_cfg, num_features: int, postfix=''):
    """
    与 mmcv.cnn.build_norm_layer 功能对齐 + 兼容你的便捷用法：
      - 支持 str / dict / 类；返回 (name, layer)
      - 'BN' 默认 2d；若提供 dims/dimension=1/3，可切到 1d/3d（你的需求）
      - SyncBN: 若有 _specify_ddp_gpu_num 则设为 1（mmcv 细节）
      - 其它默认项与原实现一致（eps/momentum/affine/track_running_stats/num_groups）
    """
    # 允许 None → Identity
    if norm_cfg is None or norm_cfg == 'None' or (
        isinstance(norm_cfg, dict) and norm_cfg.get('type') in (None, 'None', 'Identity')
    ):
        name = _NORM_ABBR['Identity'] + str(postfix)
        return name, nn.Identity()

    if isinstance(norm_cfg, (str, type)):
        norm_cfg = {'type': norm_cfg}
    if not isinstance(norm_cfg, dict):
        raise TypeError(f'cfg must be dict/str/type, got {type(norm_cfg)}')

    cfg = copy.deepcopy(norm_cfg)
    obj_type = cfg.pop('type', 'BN')

    # 通用参数（保留你的默认）
    requires_grad = bool(cfg.pop('requires_grad', True))
    eps = float(cfg.pop('eps', 1e-5))
    momentum = float(cfg.pop('momentum', 0.1))
    affine = bool(cfg.pop('affine', True))
    track_running_stats = bool(cfg.pop('track_running_stats', True))
    num_groups = int(cfg.pop('num_groups', 32))

    # 维度便捷开关（仅当 type='BN'/'IN' 且未显式写 1d/3d 时生效）
    dims = cfg.pop('dims', cfg.pop('dimension', 2))
    if isinstance(obj_type, str):
        norm_type = obj_type
    elif inspect.isclass(obj_type):
        norm_type = obj_type.__name__
    else:
        raise TypeError(f'type must be str or class, got {type(obj_type)}')

    # 解析目标类
    def _make_bn(d):
        return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[d]
    def _make_in(d):
        return {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[d]

    if inspect.isclass(obj_type):
        Norm = obj_type
    else:
        if norm_type in ('BN', 'BN2d', 'NaiveSyncBN'):
            Norm = _make_bn(2)
        elif norm_type == 'BN1d' or (norm_type == 'BN' and dims == 1):
            Norm = _make_bn(1)
        elif norm_type == 'BN3d' or (norm_type == 'BN' and dims == 3):
            Norm = _make_bn(3)
        elif norm_type == 'SyncBN':
            Norm = nn.SyncBatchNorm
        elif norm_type in ('IN', 'IN2d'):
            Norm = _make_in(2)
        elif norm_type == 'IN1d' or (norm_type == 'IN' and dims == 1):
            Norm = _make_in(1)
        elif norm_type == 'IN3d' or (norm_type == 'IN' and dims == 3):
            Norm = _make_in(3)
        elif norm_type == 'GN':
            Norm = nn.GroupNorm
        elif norm_type == 'LN':
            Norm = nn.LayerNorm
        else:
            raise KeyError(f'Unsupported norm type: {norm_type}')

    # 构建层
    if Norm is nn.GroupNorm:
        layer = Norm(num_groups=num_groups, num_channels=num_features, eps=eps, affine=affine)
    elif Norm is nn.LayerNorm:
        layer = Norm(normalized_shape=num_features, eps=eps, elementwise_affine=affine)
    elif Norm is nn.SyncBatchNorm:
        layer = Norm(num_features, eps=eps, momentum=momentum, affine=affine)
        if hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    elif Norm in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
        layer = Norm(num_features, eps=eps, momentum=momentum, affine=affine)
    elif Norm in (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d):
        layer = Norm(num_features, eps=eps, momentum=momentum,
                     affine=affine, track_running_stats=track_running_stats)
    else:
        # 万一传进来的是别的自定义类
        layer = Norm(num_features)

    for p in layer.parameters():
        p.requires_grad = requires_grad

    abbr = _NORM_ABBR.get(norm_type, _NORM_ABBR.get(Norm.__name__, 'norm'))
    name = f'{abbr}{postfix}'
    return name, layer


# ---------------- Activation builder (兼容 mmcv) ----------------
# 支持的激活类型（常用 mmcv 名称）

_ACT_MAP = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'RReLU': nn.RReLU,
    'ReLU6': nn.ReLU6,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
    'Swish': nn.SiLU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'Softplus': nn.Softplus,
    'Softsign': nn.Softsign,
    'Softshrink': nn.Softshrink,
    'Hardswish': nn.Hardswish,
    'HSwish': nn.Hardswish,
    'Hardsigmoid': nn.Hardsigmoid,
    'HSigmoid': nn.Hardsigmoid,
    'Mish': nn.Mish,
    'Softmax': nn.Softmax,
    'Identity': nn.Identity,
    'None': nn.Identity,
}


def build_activation_layer(act_cfg):
    """
    与 mmcv.cnn.build_activation_layer 功能对齐，保留便捷默认：
      - 支持 str / dict / 类
      - 默认注入：ReLU/LeakyReLU/…/Hardsigmoid/Mish -> inplace=True（若支持）
                  LeakyReLU.negative_slope=0.01, Softmax.dim=1, PReLU.num_parameters=1, init=0.25
    """
    if act_cfg is None or act_cfg == 'None' or (
        isinstance(act_cfg, dict) and act_cfg.get('type') in (None, 'None', 'Identity')
    ):
        return nn.Identity()

    if isinstance(act_cfg, (str, type)):
        act_cfg = {'type': act_cfg}
    if not isinstance(act_cfg, dict):
        raise TypeError(f'cfg must be dict/str/type, got {type(act_cfg)}')

    cfg = copy.deepcopy(act_cfg)
    obj_type = cfg.pop('type', 'ReLU')

    # 允许传类或字符串
    if inspect.isclass(obj_type):
        Act = obj_type
        act_name = Act.__name__
    else:
        act_name = str(obj_type)
        if act_name not in _ACT_MAP:
            raise KeyError(f'Unsupported activation type: {act_name}')
        Act = _ACT_MAP[act_name]

    # 默认项（仅传入该类接受的参数）
    defaults = {}
    if act_name in ('ReLU', 'LeakyReLU', 'ReLU6', 'ELU', 'SELU', 'SiLU', 'Hardswish', 'Hardsigmoid', 'Mish'):
        defaults['inplace'] = True
    if act_name == 'LeakyReLU':
        defaults.setdefault('negative_slope', 0.01)
    if act_name == 'Softmax':
        defaults.setdefault('dim', 1)
    if act_name == 'PReLU':
        defaults.setdefault('num_parameters', 1)
        defaults.setdefault('init', 0.25)

    defaults.update(cfg)
    sig = inspect.signature(Act.__init__)
    valid_kwargs = {k: v for k, v in defaults.items() if k in sig.parameters}
    return Act(**valid_kwargs)


class _CrossEntropyLoss(nn.Module):
    """交叉熵封装，支持：
    - class_weight（list/ndarray/tensor）
    - loss_weight（整体缩放）
    - label_smoothing（兼容 PyTorch 的 F.cross_entropy 参数名）
    - 接受 one-hot/soft label，会转为 argmax 索引（与 PYSKL 用法对齐）
    """
    def __init__(
        self,
        class_weight: Optional[Union[Sequence[float], torch.Tensor]] = None,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        if class_weight is None:
            self.register_buffer("weight", None)
        else:
            w = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer("weight", w)

    def forward(self, cls_score: torch.Tensor, label: torch.Tensor, **kwargs):
        # 若 label 为 one-hot / soft（N, C），按 argmax 转为索引
        if label.ndim > 1:
            label = label.argmax(dim=-1)

        loss = F.cross_entropy(
            cls_score,
            label.long(),
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
        return loss * self.loss_weight


def build_loss(cfg: Optional[dict] = None) -> nn.Module:
    """仅构建交叉熵损失，兼容 PYSKL 的配置风格。
    接受的字段（可选）：
      - type: 'CrossEntropyLoss'
      - class_weight: list/tensor
      - loss_weight: float
      - reduction: 'none' | 'mean' | 'sum'
      - label_smooth_eps 或 label_smoothing: float
    """
    if cfg is None:
        return _CrossEntropyLoss()

    loss_type = cfg.get("type", "CrossEntropyLoss")
    if loss_type != "CrossEntropyLoss":
        raise NotImplementedError(f"Only CrossEntropyLoss is supported, got: {loss_type}")

    class_weight = cfg.get("class_weight", None)
    loss_weight = cfg.get("loss_weight", 1.0)
    reduction = cfg.get("reduction", "mean")
    # 兼容两种字段命名
    label_smoothing = cfg.get("label_smooth_eps", cfg.get("label_smoothing", 0.0))

    return _CrossEntropyLoss(
        class_weight=class_weight,
        loss_weight=loss_weight,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


_build_activation_layer = build_activation_layer
_build_norm_layer = build_norm_layer
_build_loss = build_loss
