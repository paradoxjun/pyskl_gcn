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
  模型导入、加载、保存、计算相关信息的代码。
"""


from __future__ import annotations
import torch
from typing import Optional, Tuple, Union

__all__ = ["build_stgcn",
           "count_params", "try_compute_gflops",
           "save_checkpoint", "load_checkpoint"]


def build_stgcn(num_classes: int,
                backbone_kwargs: dict,
                head_kwargs: Optional[dict] = None):
    """单独的模型构建函数，便于外部替换。"""
    from models.stgcn import STGCNClassifier
    head_kwargs = head_kwargs or dict(dropout=0.0)
    model = STGCNClassifier(
        num_classes=num_classes,
        backbone_kwargs=backbone_kwargs,
        head_kwargs=head_kwargs,
        pretrained_backbone=None,
        pretrained_head=None
    )
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_compute_gflops(model: torch.nn.Module,
                       sample_shape: Tuple[int, int, int, int, int],
                       device: str = "cpu") -> tuple[Optional[float], Optional[float]]:
    """
    优先用 thop.profile 估 FLOPs（单位：GFLOPs），同时返回 Params(M)。失败则返回 (None, None)。
    sample_shape: (N, M, T, V, C) —— 你的 STGCNClassifier 输入
    关键点：对模型做 deepcopy，在 CPU 上 profile，避免在真实训练模型上残留 hook。
    """
    import copy
    try:
        from thop import profile
        m = copy.deepcopy(getattr(model, "module", model)).to(device).eval()
        with torch.no_grad():
            dummy = torch.zeros(sample_shape, dtype=torch.float32, device=device)
            flops, params = profile(m, inputs=(dummy,), verbose=False)
        # 显式释放临时对象，保险起见
        del m, dummy
        gflops = flops / 1e9
        mparams = params / 1e6
        return gflops, mparams
    except Exception:
        return None, None


def save_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    epoch: int,
                    **extras):
    obj = {
        "epoch": epoch,
        "model_state_dict": model.state_dict()
    }
    if optimizer is not None:
        obj["optimizer_state_dict"] = optimizer.state_dict()
    obj.update(extras)
    torch.save(obj, path)


def load_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    map_location: Union[str, torch.device] = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.copy()
