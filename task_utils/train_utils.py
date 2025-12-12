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
  训练辅助工具：
- 随机性：set_global_seed / worker_init_fn
- 学习率：阶梯指数衰减 / 连续指数衰减 / 余弦退火 + Warmup
- 优化器：SGD / Adam / AdamW
- EMA：ModelEMA
- 指标 & Loss：topk_accuracy / label smoothing CE
"""


from __future__ import annotations
import copy
import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Sequence, Dict, Optional

__all__ = [
    # seed
    "set_global_seed", "worker_init_fn",
    # schedulers
    "adjust_lr_linear_epoch", "adjust_lr_exp_manual_epoch", "adjust_lr_exp_epoch", "cosine_warmup_lr",
    "adjust_lr_linear_step", "adjust_lr_exp_manual_step", "adjust_lr_exp_step", "cosine_warmup_lr_step",
    "set_optimizer_lr", "build_scheduler",
    # optimizers
    "build_sgd", "build_adam", "build_adamw", "build_optimizer",
    # metrics & loss
    "topk_accuracy", "cross_entropy_label_smoothing",
    # ema
    "ModelEMA", "build_ema",
]


# =========================
# 1) 随机种子
# =========================
def set_global_seed(seed: int,
                    deterministic: bool = False,
                    cudnn_benchmark: bool = True):
    """固定 Python / NumPy / PyTorch 随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def worker_init_fn(worker_id: int):
    """
    DataLoader 的 worker 初始化函数。
    用主进程的初始种子派生出每个 worker 的 NumPy / random 种子。
    """
    base_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


# =========================
# 2) 学习率调度
# =========================
# 1) 线性衰减（新增）：严格从 base_lr 线性到 min_lr
def adjust_lr_linear_epoch(optimizer: torch.optim.Optimizer, base_lr: float = 0.01, min_lr: float = 0.0001,
                           epoch: int = 0, total_epochs: int = 300) -> float:
    """
    Linear decay:
      lr(e) = base_lr + (min_lr - base_lr) * ((epoch+1) / total_epochs)
    在第 total_epochs 轮（epoch=total_epochs-1）正好到达 min_lr。
    """
    t = (epoch + 1) / max(1, total_epochs)
    t = min(max(t, 0.0), 1.0)
    lr = base_lr + (min_lr - base_lr) * t
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def adjust_lr_linear_step(optimizer: torch.optim.Optimizer, base_lr: float = 0.01, min_lr: float = 0.0001,
                          step: int = 0, total_steps: int = 1) -> float:
    """
    Linear decay by step:
      lr(s) = base_lr + (min_lr - base_lr) * ((step+1) / total_steps)
    在第 total_steps 步正好到达 min_lr。
    """
    t = (step + 1) / max(1, total_steps)
    t = min(max(t, 0.0), 1.0)
    lr = base_lr + (min_lr - base_lr) * t
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


# 2) 阶梯式指数衰减（手动设置，最开始的实现）
def adjust_lr_exp_manual_epoch(optimizer: torch.optim.Optimizer, base_lr: float = 0.001, min_lr: float = 0.0001,
                               decay: float = 0.95, epoch: int = 0, decay_epochs: int = 5) -> float:
    """
    Step exponential decay（阶梯式指数衰减与原实现一致）：
      n_steps = epoch // decay_epochs
      lr = max(base_lr * decay**n_steps, min_lr)
    """
    n_steps = epoch // decay_epochs
    lr = max(base_lr * (decay ** n_steps), min_lr)
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def adjust_lr_exp_manual_step(optimizer: torch.optim.Optimizer, base_lr: float = 0.001, min_lr: float = 0.0001,
                              decay: float = 0.95, step: int = 0, decay_steps: int = 1) -> float:
    n_steps = step // max(1, decay_steps)
    lr = max(base_lr * (decay ** n_steps), min_lr)
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


# 3) 连续指数衰减（修改：支持“给定起止值”）
def adjust_lr_exp_epoch(optimizer: torch.optim.Optimizer, base_lr: float, min_lr: float, epoch: int, total_epochs: int,
                        gamma: float = None) -> float:
    """
    Exponential decay：
      - 若给出 gamma：lr = max(base_lr * gamma**(epoch+1), min_lr)    ← 兼容旧用法
      - 否则：自动计算 gamma，使得在最后一轮正好到达 min_lr：
          gamma = (min_lr / base_lr) ** (1 / total_epochs)
          lr = base_lr * gamma**(epoch+1)
    """
    if gamma is None:
        # 避免除零/对数问题
        if base_lr <= 0:
            lr = min_lr
        else:
            gamma = (min_lr / base_lr) ** (1.0 / max(1, total_epochs))
            lr = base_lr * (gamma ** (epoch + 1))
            lr = max(lr, min_lr)
    else:
        lr = max(base_lr * (gamma ** (epoch + 1)), min_lr)

    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def adjust_lr_exp_step(optimizer, base_lr: float, min_lr: float, step: int, total_steps: int,
                       gamma: float = None) -> float:
    if gamma is None:
        if base_lr <= 0:
            lr = min_lr
        else:
            gamma = (min_lr / base_lr) ** (1.0 / max(1, total_steps))
            lr = base_lr * (gamma ** (step + 1))
            lr = max(lr, min_lr)
    else:
        lr = max(base_lr * (gamma ** (step + 1)), min_lr)
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


# 4) Warmup + 余弦衰减（保留，只返回 lr，不写 optimizer）
def cosine_warmup_lr(base_lr: float, min_lr: float, epoch: int, total_epochs: int, warmup_epochs: int = 0) -> float:
    """
    前 warmup_epochs：从 0 线性升到 base_lr；
    之后：余弦退火到 min_lr（在最后一轮正好到达 min_lr）。
    """
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)

    t = (epoch - warmup_epochs + 1) / max(1, total_epochs - warmup_epochs)
    t = min(max(t, 0.0), 1.0)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
    return lr


def cosine_warmup_lr_step(base_lr: float, min_lr: float, step: int, total_steps: int, warmup_steps: int = 0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps + 1) / max(1, total_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g['lr'] = lr


def build_scheduler(sched_cfg: Dict):
    """
    返回一个学习率调度闭包：
      - 若 sched_cfg['by'] == 'epoch' → 返回 _step_epoch(optimizer, epoch, total_epochs) -> lr
      - 若 sched_cfg['by'] == 'step'  → 返回 _step_iter(optimizer, *, step, total_steps, steps_per_epoch) -> lr

    支持四种 type：
      - 'linear'   线性衰减（支持 epoch/step 两种模式）
      - 'exp_man'  阶梯指数衰减（手动设衰减率），参数：
                    decay（每次衰减系数），
                    by='epoch' 用 decay_epoch（每多少个 epoch 衰减一次）；
                    by='step'  用 decay_step（每多少步衰减一次，若缺省，则用 decay_epoch * steps_per_epoch）
      - 'exp'      连续指数衰减，参数：
                    gamma（可省略；省略则自动算到最后正好到 min_lr）
      - 'cosine'   余弦退火，参数：
                    by='epoch' 用 warmup_epochs，
                    by='step'  自动折算为 warmup_steps = warmup_epochs * steps_per_epoch

    配置示例：
      {'type':'linear',  'base_lr':1e-3, 'min_lr':1e-4, 'by':'step'}
      {'type':'exp_man', 'base_lr':0.025,'min_lr':0.0,  'decay':0.95,'decay_epoch':2, 'by':'epoch'}
      {'type':'exp',     'base_lr':1e-3, 'min_lr':1e-4, 'gamma':0.98, 'by':'epoch'}
      {'type':'cosine',  'base_lr':0.025,'min_lr':0.0,  'warmup_epochs':3, 'by':'step'}
    """
    stype = str(sched_cfg.get("type", "cosine")).lower()
    by = str(sched_cfg.get("by", "step")).lower()
    base_lr = float(sched_cfg.get("base_lr", 1e-3))
    min_lr = float(sched_cfg.get("min_lr", 1e-4))

    if stype == "linear":
        if by == "step":
            def _step_iter(optimizer, *, step: int, total_steps: int, steps_per_epoch: int):
                return adjust_lr_linear_step(optimizer, base_lr, min_lr, step, total_steps)
            return _step_iter

        def _step_epoch(optimizer, epoch: int, total_epochs: int):
            return adjust_lr_linear_epoch(optimizer, base_lr, min_lr, epoch, total_epochs)
        return _step_epoch

    elif stype == "exp_man":
        decay = float(sched_cfg.get("decay", 0.95))
        decay_epoch = int(sched_cfg.get("decay_epoch", 2))

        if by == "step":
            def _step_iter(optimizer, *, step: int, total_steps: int, steps_per_epoch: int):
                decay_steps = int(sched_cfg.get("decay_step", decay_epoch * steps_per_epoch))
                return adjust_lr_exp_manual_step(optimizer, base_lr, min_lr, decay, step, decay_steps)
            return _step_iter

        def _step_epoch(optimizer, epoch: int, total_epochs: int):
            return adjust_lr_exp_manual_epoch(optimizer, base_lr, min_lr, decay, epoch, decay_epoch)
        return _step_epoch

    elif stype == "exp":
        gamma = sched_cfg.get("gamma", None)
        gamma = float(gamma) if gamma is not None else None

        if by == "step":
            def _step_iter(optimizer, *, step: int, total_steps: int, steps_per_epoch: int):
                return adjust_lr_exp_step(optimizer, base_lr, min_lr, step, total_steps, gamma=gamma)
            return _step_iter

        def _step_epoch(optimizer, epoch: int, total_epochs: int):
            return adjust_lr_exp_epoch(optimizer, base_lr, min_lr, epoch, total_epochs, gamma=gamma)
        return _step_epoch

    elif stype == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))

        if by == "step":
            def _step_iter(optimizer, *, step: int, total_steps: int, steps_per_epoch: int):
                warmup_steps = warmup_epochs * steps_per_epoch
                lr = cosine_warmup_lr_step(base_lr, min_lr, step, total_steps, warmup_steps)
                set_optimizer_lr(optimizer, lr)
                return lr
            return _step_iter

        def _step_epoch(optimizer, epoch: int, total_epochs: int):
            lr = cosine_warmup_lr(base_lr, min_lr, epoch, total_epochs, warmup_epochs)
            set_optimizer_lr(optimizer, lr)
            return lr

        return _step_epoch

    else:
        # 兜底：不改学习率
        def _noop(optimizer, epoch: int, total_epochs: int):
            return optimizer.param_groups[0].get('lr', base_lr)
        return _noop


# =========================
# 3) 优化器封装
# =========================
# mmaction 的默认配置
def build_sgd(model: torch.nn.Module,
              lr: float = 0.1,
              momentum: float = 0.9,
              weight_decay: float = 5e-4,
              nesterov: bool = True,
              dampening: float = 0.0) -> torch.optim.Optimizer:
    """
    MMACTION 常见设置：SGD(momentum=0.9, weight_decay=1e-4, nesterov 可选)
    注：dampening 与 nesterov 通常不同时启用（nesterov=True 时 dampening=0）
    """
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        dampening=dampening
    )


def build_adam(model: torch.nn.Module,
               lr: float = 1e-3,
               weight_decay: float = 0.0,
               betas: Tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def build_adamw(model: torch.nn.Module,
                lr: float = 3e-3,
                weight_decay: float = 1e-5,
                betas: Tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def build_optimizer(model: torch.nn.Module, opt_cfg: Dict) -> torch.optim.Optimizer:
    """
    统一从配置字典构建优化器。
    opt_cfg 示例：
      - {'name':'SGD',   'lr':0.1,  'momentum':0.9, 'weight_decay':5e-4, 'nesterov':True}
      - {'name':'Adam',  'lr':1e-3, 'betas':(0.9,0.999), 'eps':1e-8, 'weight_decay':0.0}
      - {'name':'AdamW', 'lr':1e-3, 'betas':(0.9,0.999), 'eps':1e-8, 'weight_decay':1e-4}
    """
    name = str(opt_cfg.get("name", "Adam")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=float(opt_cfg.get("weight_decay", 5e-4)),
            nesterov=bool(opt_cfg.get("nesterov", False)),
            dampening=float(opt_cfg.get("dampening", 0.0)),
        )

    elif name == "adam":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=float(opt_cfg.get("eps", 1e-8)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )

    elif name == "adamw":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=float(opt_cfg.get("eps", 1e-8)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
        )

    else:
        raise ValueError(f"Unknown optimizer name: {opt_cfg.get('name')}")


# =========================
# 4) 指标 & Loss（可选）
# =========================
@torch.no_grad()
def topk_accuracy(logits: torch.Tensor,
                  target: torch.Tensor,
                  ks: Sequence[int] = (1, 5)) -> Dict[int, float]:
    """
    计算 top-k 准确率（百分比）。
    返回：{1: top1%, 5: top5%}
    """
    maxk = max(ks)
    pred = logits.topk(k=maxk, dim=1).indices
    correct = pred.eq(target[:, None])

    res = {}
    for k in ks:
        topk = correct[:, :k].any(dim=1).float().mean().item() * 100.0
        res[k] = topk
    return res


def cross_entropy_label_smoothing(logits: torch.Tensor,
                                  target: torch.Tensor,
                                  smoothing: float = 0.1,
                                  reduction: str = "mean") -> torch.Tensor:
    """
    Label Smoothing 版本 CE。
    torch>=1.10 可直接用 F.cross_entropy 的 label_smoothing；此处兼容旧版。
    """
    if hasattr(F, "cross_entropy"):
        try:
            return F.cross_entropy(logits, target,
                                   label_smoothing=float(smoothing),
                                   reduction=reduction)
        except TypeError:
            pass

    n_classes = logits.size(1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    log_prob = F.log_softmax(logits, dim=1)
    loss = -(true_dist * log_prob).sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


# =========================
# 5) EMA（Exponential Moving Average）
# =========================
class ModelEMA:
    """
    极简稳定 EMA（仅一个超参：decay）
      - 第一次 update：完整拷贝（参数+buffers），相当于“第一轮就用当前模型”
      - 后续：参数做 EMA；浮点 buffer 做 EMA；非浮点 buffer 直接拷贝
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.initialized = False
        self.num_updates = 0  # 仅用于判断“是否至少更新过一次”

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        if not self.initialized:
            self.module.load_state_dict(model.state_dict(), strict=True)
            self.initialized = True
            self.num_updates = 1
            return

        d = self.decay
        # --- 参数做 EMA ---
        for p_ema, p_cur in zip(self.module.parameters(), model.parameters()):
            if not p_cur.requires_grad:
                continue
            p_ema.mul_(d).add_(p_cur, alpha=(1.0 - d))

        # --- buffer：浮点做 EMA；非浮点直接拷贝 ---
        for b_ema, b_cur in zip(self.module.buffers(), model.buffers()):
            if b_cur.dtype.is_floating_point:
                b_ema.mul_(d).add_(b_cur, alpha=(1.0 - d))
            else:
                b_ema.copy_(b_cur)

        self.num_updates += 1

    # 兼容保存/加载
    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state, strict=True)
        self.initialized = True
        self.num_updates = max(self.num_updates, 1)


"""
def build_ema(cfg: dict, model: torch.nn.Module) -> Optional[ModelEMA]:
    # CFG中参数使用: EMA=dict(enabled=True, eval_use_ema=True, ref_batch=16, decay=0.999),
    emacfg = cfg.get("EMA", {})
    if not emacfg or not emacfg.get("enabled", False):
        return None

    # 把 config 里的 decay 视为：参考 batch 下的“基准 per-step 衰减”
    base_decay = float(emacfg.get("decay", 0.999))      # 例如你在 B=16 下调好的 0.999
    ref_batch = int(emacfg.get("ref_batch", 16))        # 参考 batch（缺省 16）
    cur_batch = int(cfg.get("BATCH", ref_batch))        # 当前实际 batch

    # 维持“每个 epoch 的平滑强度”不变：d_cur^S_cur = d_ref^S_ref
    # 同一数据集/单进程时 S ∝ 1/batch，所以 d_cur = d_ref ** (batch_cur / batch_ref)
    # ratio = (cur_batch * cur_world_size * cur_update_every) / (ref_batch * ref_world_size * ref_update_every)
    ratio = max(1e-8, cur_batch / float(ref_batch))
    decay_eff = base_decay ** ratio

    ema = ModelEMA(model, decay=decay_eff)

    # 可选：强制把 EMA 模块放到特定设备
    dev = emacfg.get("device", None)  # e.g. "cuda:0"
    if dev is not None:
        ema.module.to(dev)

    # （没有 logger 的上下文，这里不打印；你可在 main 里 log 一句）
    # logger.info(f"EMA per-step decay: {decay_eff:.6f} (base={base_decay}, batch={cur_batch}, ref_batch={ref_batch})")

    return ema
"""


def build_ema(cfg: dict,
              model: torch.nn.Module,
              steps_per_epoch: Optional[int] = None,
              logger=None) -> Optional[ModelEMA]:
    """
        在NTU60上，batch-size=64, base_decay=0.999、steps/epoch = 2000
        UCF101（~1/4 数据，假设 500 steps/epoch）：d_cur = 0.999 ** (2000/500) ≈ 0.996（half ≈ 173 步）
        HMDB51（~1/8 数据，假设 250 steps/epoch）：d_cur ≈ 0.992（half ≈ 86 步）

        k_step = ln0.5 / lnd ≈ 0.6931 / (1 - 0.999) ≈ 693步
        任意数据集: S = steps_per_epoch
        half_life_epoch = k_step / S
        NTU60 (在 BS = 16时，S约等于2506): k_step = 693 -> half_life_epoch = 693 / 2506 ≈ 0.276
        每步衰减：
            d = exp(ln5 / (H * S))
        H=0.25  →   d≈0.99889
        H=0.276 →   d≈0.99900
        H=0.30  →   d≈0.99908
        H=0.35  →   d≈0.99921
        半衰期（k，步）    每步 decay (d)
        500	            0.998615
        800 	        0.999134
        1000	        0.999307
        1500	        0.999538
        2000	        0.999653
        3000	        0.999769
    """
    emacfg = cfg.get("EMA", {})
    if not emacfg or not emacfg.get("enabled", False):
        return None

    # 优先使用“半衰期=若干个 epoch”来推导 per-step decay
    half_life_epoch = emacfg.get("half_life_epoch", None)
    if (half_life_epoch is not None) and (steps_per_epoch is not None) and (steps_per_epoch > 0):
        # 半衰期（以 step 计）
        k = max(1, int(round(float(half_life_epoch) * float(steps_per_epoch))))
        # 每 step 的 EMA 衰减因子：d = 0.5 ** (1/k)
        decay_eff = float(math.pow(0.5, 1.0 / k))
        mode_str = f"half_life_epoch={half_life_epoch}, steps/epoch={steps_per_epoch}, k={k}"
    else:
        # 兼容旧配置：decay + ref_batch 自适应到当前 batch
        base_decay = float(emacfg.get("decay", 0.999))
        ref_batch = int(emacfg.get("ref_batch", 16))
        cur_batch = int(cfg.get("BATCH", ref_batch))
        ratio = max(1e-8, cur_batch / float(ref_batch))
        decay_eff = float(math.pow(base_decay, ratio))
        mode_str = f"legacy(base_decay={base_decay}, ref_batch={ref_batch}, cur_batch={cur_batch})"

    ema = ModelEMA(model, decay=decay_eff)

    # 可选：把 EMA 放到指定设备
    dev = emacfg.get("device", None)  # e.g. "cuda:0"
    if dev is not None:
        ema.module.to(dev)

    if logger is not None:
        logger.info(f"EMA per-step decay = {decay_eff:.6f}  [{mode_str}]")

    return ema
