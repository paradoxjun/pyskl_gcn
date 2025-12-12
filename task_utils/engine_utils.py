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
  训练核心代码，实现一个轮次的训练。
"""


from __future__ import annotations
import inspect
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional, Sequence, Union
from task_utils.train_utils import worker_init_fn, topk_accuracy

__all__ = ["build_loader", "run_one_epoch"]


def build_loader(dataset_cls,
                 npz_path: Union[str, Sequence[str]],
                 split: str,
                 cfg: dict,
                 logger,
                 dataset_kwargs: Optional[dict] = None) -> Tuple[DataLoader, int]:
    """兼容不同 Dataset 构造函数：只有在其签名包含 T 时才传入 T。"""
    ds_kwargs = dict(dataset_kwargs or {})
    try:
        sig = inspect.signature(dataset_cls)
        if 'T' in sig.parameters and 'T' in cfg and 'T' not in ds_kwargs:
            ds_kwargs['T'] = cfg['T']
    except (TypeError, ValueError):
        pass

    ds = dataset_cls(npz_path, mode=split, **ds_kwargs)

    # 可选的生成器，固定 DataLoader 中 shuffle 的随机性
    g = torch.Generator()
    g.manual_seed(cfg.get("SEED", 42))

    loader = DataLoader(
        ds,
        batch_size=cfg["BATCH"],
        shuffle=(split == "train"),
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=(cfg["NUM_WORKERS"] > 0),
        prefetch_factor=cfg["PREFETCH"] if cfg["NUM_WORKERS"] > 0 else None,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=g if split == "train" else None,
    )
    if isinstance(npz_path, (list, tuple)):
        logger.info(f"{split.capitalize():<5} | NPZ files ({len(npz_path)}):")
        for p in npz_path:
            logger.info(f"{split.capitalize():<5} |   - {p}")
    else:
        logger.info(f"{split.capitalize():<5} | NPZ: {npz_path}")
    logger.info(f"{split.capitalize():<5} | Total: {len(ds)} | Batch: {cfg['BATCH']} | Workers: {cfg['NUM_WORKERS']}")
    return loader, len(ds)


def _maybe_to_stgcn_input(xb: torch.Tensor) -> torch.Tensor:
    """dataset 输出 (B, C, T, V, P) -> 模型期望 (B, M, T, V, C)（仅在满足形状时转换）"""
    if torch.is_tensor(xb) and xb.ndim == 5:
        # 约定最后一维是 P（人数），通道在 dim=1
        return xb.permute(0, 4, 2, 3, 1).contiguous()
    return xb


def run_one_epoch(model: torch.nn.Module,
                  loader: DataLoader,
                  device: str,
                  *,
                  train: bool,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  scaler: Optional[torch.cuda.amp.GradScaler] = None,
                  amp: bool = True,
                  criterion=None,  # PosePointNetSplit 可传自定义 loss
                  logger=None,
                  print_freq: int = 0,
                  desc: str = "",
                  ema=None,
                  use_ema_for_eval: bool = False,   # 验证是否用 EMA
                  stgcn_permute: bool = True,       # 是否做 ST-GCN 轴变换
                  grad_clip: float = -1.0,          # 梯度裁剪（稳定性）
                  lr_sched_iter=None,               # by='step' 时传入的闭包
                  global_step_offset: int = 0,      # 本 epoch 起始 step
                  total_steps: int = None,          # 总步数（E * S）
                  steps_per_epoch: int = None       # 每个 epoch 的步数（len(loader)）
                  ) -> dict:
    # 训练/验证模式
    if train:
        model.train()
        run_model = model
    else:
        model.eval()
        # 验证阶段可选用 EMA 模型做前向；不更新 EMA
        use_ema = (ema is not None) and use_ema_for_eval and getattr(ema, "initialized", False)
        run_model = (ema.module if use_ema else model)  # 注意：是 .module

    total = 0
    sum_loss = 0.0
    sum_top1 = 0.0
    sum_top5 = 0.0

    pbar = tqdm(loader, desc=desc, ncols=110, leave=False)
    autocast = torch.cuda.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast

    for it, batch in enumerate(pbar, 1):
        # 每步学习率调度（仅 train + by='step' 时启用）
        if train and (lr_sched_iter is not None) and (total_steps is not None):
            gstep = global_step_offset + (it - 1)  # 全局 step，从 0 开始
            lr_sched_iter(optimizer,
                          step=gstep,
                          total_steps=total_steps,
                          steps_per_epoch=(steps_per_epoch or len(loader)))

        # 兼容 (xb, y) 或 (gx, lx, y) 或 (x1, x2, ..., y)
        assert isinstance(batch, (tuple, list)) and len(batch) >= 2
        *inputs, yb = batch

        # === 输入整理 ===
        if len(inputs) == 1 and torch.is_tensor(inputs[0]):
            x0 = inputs[0]
            # 根据参数决定是否执行 ST-GCN 的 (B,C,T,V,P)->(B,P,T,V,C) 轴变换
            if stgcn_permute:
                xb = _maybe_to_stgcn_input(x0).to(device, non_blocking=True)
            else:
                xb = x0.to(device, non_blocking=True)
            model_inputs = xb
        else:
            # 多输入（例如模型需要 (gx, lx) 分路）
            model_inputs = tuple(x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in inputs)

        yb = yb.to(device, non_blocking=True)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with (torch.enable_grad() if train else torch.no_grad()):
            with autocast(enabled=amp):
                logits = run_model(model_inputs)  # 训练用即时模型；验证可选 EMA
                if criterion is not None:
                    loss = criterion(logits, yb)
                else:
                    out = model.head.loss(logits, yb)  # ST-GCN 风格
                    loss = out['loss_cls']

        if train:
            # （可选）梯度裁剪，先反缩放再裁剪
            if grad_clip and grad_clip > 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # 仅在训练 optimizer.step() 之后更新 EMA
            if ema is not None:
                ema.update(model)

        # 统计
        bs = yb.size(0)
        total += bs
        sum_loss += float(loss.detach()) * bs
        acc = topk_accuracy(logits.detach(), yb, ks=(1, 5))
        sum_top1 += (acc[1] / 100.0) * bs
        sum_top5 += (acc[5] / 100.0) * bs

        pbar.set_postfix(loss=f"{sum_loss / total:.4f}",
                         top1=f"{sum_top1 / total * 100:5.2f}%",
                         top5=f"{sum_top5 / total * 100:5.2f}%")

        if print_freq and (it % print_freq == 0) and logger is not None:
            logger.info(f"  it={it:05d} | loss={sum_loss / total:.4f} "
                        f"| top1={sum_top1 / total * 100:6.2f}% | top5={sum_top5 / total * 100:6.2f}%")

    return dict(
        loss=sum_loss / total if total else 0.0,
        top1=sum_top1 / total * 100 if total else 0.0,
        top5=sum_top5 / total * 100 if total else 0.0,
        samples=total
    )
