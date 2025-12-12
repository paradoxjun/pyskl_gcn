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
  测试用代码。
"""


from __future__ import annotations
import numpy as np
import torch


# -------------------- Metrics --------------------
@torch.no_grad()
def topk_correct(logits: torch.Tensor, y: torch.Tensor, topk=(1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    corr = pred.eq(y.view(1, -1).expand_as(pred))
    c1 = corr[:1].sum().item()
    c5 = corr[:5].any(dim=0).sum().item() if maxk >= 5 else c1
    return c1, c5


# ---------------- 采样 ----------------
def phase_shift_even_indices_with_pad(F: int, T: int, Nc: int):
    if F <= 0:
        return [np.full(T, -1, dtype=np.int32) for _ in range(Nc)]
    if F < T:
        base = np.arange(F, dtype=np.int32)
        pad = np.full(T - F, -1, dtype=np.int32)
        one = np.concatenate([base, pad], 0)
        return [one.copy() for _ in range(Nc)]
    grid = np.linspace(0, F - 1, T * Nc, dtype=np.float32)
    return [np.round(grid[j::Nc]).astype(np.int32) for j in range(Nc)]


def contig_window_indices_with_pad(F: int, T: int, Nc: int):
    if F <= 0:
        return [np.full(T, -1, dtype=np.int32) for _ in range(Nc)]
    if F <= T:
        base = np.arange(F, dtype=np.int32)
        pad = np.full(T - F, -1, dtype=np.int32)
        one = np.concatenate([base, pad], 0)
        return [one.copy() for _ in range(Nc)]
    max_start = F - T
    starts = np.linspace(0, max_start, Nc, dtype=np.int32).tolist()
    return [np.arange(s, s + T, dtype=np.int32) for s in starts]


def load_ckpt_pref_ema(model, ckpt_path, logger):
    ck = torch.load(ckpt_path, map_location='cpu')
    loaded = False
    if isinstance(ck, dict):
        if ck.get('ema_state_dict') is not None:
            model.load_state_dict(ck['ema_state_dict'], strict=True);
            loaded = True
            vt1, vt5 = ck.get('val_top1'), ck.get('val_top5')
            if vt1 is not None and vt5 is not None:
                logger.info(
                    f"[Load] ema_state_dict | epoch={ck.get('epoch')} | val_top1={vt1:.2f}% | val_top5={vt5:.2f}%")
        elif ck.get('model_state_dict') is not None:
            model.load_state_dict(ck['model_state_dict'], strict=True);
            loaded = True
            logger.info(f"[Load] model_state_dict | epoch={ck.get('epoch')}")
    if not loaded:
        model.load_state_dict(ck, strict=True)
        logger.info("[Load] raw state_dict loaded")
