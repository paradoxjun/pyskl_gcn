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
  动作序列采样。
"""


import numpy as np
from typing import Optional

__all__ = [
    "sample_indices_train",
    "sample_indices_eval",
    "index_with_pad"
]


def _pad_minus_one(F: int, T: int) -> np.ndarray:
    """
    返回 [0..F-1] + (-1 pad) 到长度 T（当且仅当 F < T 时用）
    """
    assert F >= 0 and T > 0
    if F >= T:
        raise ValueError("ERROR: _pad_minus_one 仅用于 F < T 的场景")
    ids = np.concatenate([np.arange(F, dtype=np.int32), np.full(T - F, -1, dtype=np.int32)])
    return ids


def _rand_unique_sorted(F: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    F >= T 时：全局随机抽 T 个唯一帧，排序返回
    F < T  时：返回 0..F-1 + (-1 pad)
    """
    if F <= 0:
        # 极端保护：无帧直接全 -1
        return np.full(T, -1, dtype=np.int32)
    if F < T:
        return _pad_minus_one(F, T)
    idx = rng.choice(F, T, replace=False)
    idx.sort()
    return idx.astype(np.int32)


def _pyskl_uniform_subsample(F: int, T: int,
                             rng: np.random.Generator,
                             p_interval=(0.5, 1.0)) -> np.ndarray:
    """
    复刻 PySKL 的“先裁再采”思路，但保持对 F<T 的处理（-1 padding）：
      1) 选比例 ratio ∈ [p_lo, p_hi]，num_sub = int(ratio * F)；
      2) 在原视频中随机选子段起点 off，使得子段长度为 num_sub；
      3) 在子段内做“均匀分段 + 段内随机”得到 T 个索引；
      4) 若 num_sub < T，则返回 off + [0..num_sub-1] 并对剩余部分用 -1 补齐（不循环）。
    说明：
      - 你不想 F<T 时循环取模，所以这里即便子段 < T 也一律补 -1。
      - 默认 p_interval=(1,1) → 子段就是全视频，且不会触发 <T 的分支。
    """
    if F <= 0:
        return np.full(T, -1, dtype=np.int32)

    if not isinstance(p_interval, tuple):
        p_interval = (p_interval, p_interval)

    # 1) 子段长度
    ratio = rng.uniform(p_interval[0], p_interval[1])
    num_sub = int(ratio * F)
    num_sub = max(1, min(num_sub, F))  # 合法范围 [1, F]

    # 2) 子段起点
    off = rng.integers(0, F - num_sub + 1)  # [0, F-num_sub]

    # 3) 子段内采样
    if num_sub < T:
        # 不循环，按你规则填 -1
        core = off + np.arange(num_sub, dtype=np.int32)
        padn = T - num_sub
        tail = np.full(padn, -1, dtype=np.int32)
        return np.concatenate([core, tail], axis=0)

    # num_sub >= T
    if T <= num_sub < 2 * T:
        # 余量分配：在 clip_len+1 个缝里随机放 (num_sub - T) 个 1
        basic = np.arange(T, dtype=np.int64)
        extra = num_sub - T
        pick = rng.choice(T + 1, extra, replace=False)
        offset = np.zeros(T + 1, dtype=np.int64)
        offset[pick] = 1
        offset = np.cumsum(offset)  # 长度 T+1
        inds_sub = basic + offset[:-1]  # [0..num_sub-1] 中的 T 个位置
    else:
        # num_sub >= 2T：均匀分 T 段，每段里随机取一个
        bids = np.array([i * num_sub // T for i in range(T + 1)], dtype=np.int64)
        bsize = np.diff(bids)  # 每段长度（>=1）
        bst = bids[:T]  # 段左边界
        offset = rng.integers(0, bsize)  # 每段内随机偏移
        inds_sub = bst + offset  # [0..num_sub-1] 中的 T 个位置

    inds = (off + inds_sub).astype(np.int32)
    return inds


def sample_indices_train(F: int,
                         T: int,
                         *,
                         rng: Optional[np.random.Generator] = None,
                         mix_prob: float = 0.5,
                         p_interval=(0.5, 1.0)) -> np.ndarray:
    """
    训练采样：
      - 以 mix_prob 的概率走“先裁再采”（PySKL 风格）；
      - 以 (1-mix_prob) 的概率走“全局随机抽 T 帧（唯一且排序）”；
      - F < T 时一律按规则：前 F 帧 + (-1 pad)（不循环）。
    参数：
      F, T       : 总帧数、目标长度
      rng        : np.random.Generator（从外部传入以控制种子）
      mix_prob   : 使用“先裁再采”的概率（默认 0.5）
      p_interval : 子段比例区间（默认 (1,1) —— 不缩放到子段）
    返回：np.int32，shape=(T,)
    """
    if rng is None:
        rng = np.random.default_rng()

    if F <= 0:
        return np.full(T, -1, dtype=np.int32)

    if F < T:
        return _pad_minus_one(F, T)

    # F >= T：两种策略 50/50
    if rng.random() < mix_prob:
        return _pyskl_uniform_subsample(F, T, rng=rng, p_interval=p_interval)
    else:
        return _rand_unique_sorted(F, T, rng=rng)


def sample_indices_eval(F: int, T: int) -> np.ndarray:
    """
    验证/测试采样（**确定性**）：
      - F >= T：等间隔均匀采样（与你的 _even 一致）
      - F < T ：返回 0..F-1 + (-1 pad)
    """
    if F <= 0:
        return np.full(T, -1, dtype=np.int32)
    if F >= T:
        return np.linspace(0, F - 1, T, dtype=np.int32)
    return _pad_minus_one(F, T)


def index_with_pad(x: np.ndarray, inds: np.ndarray) -> np.ndarray:
    """
    根据索引 inds（可能包含 -1）取帧，并把有效帧“挤到前面”，尾部补 0：
      - x: (F, ...)  任意后缀维度
      - inds: (T,)   帧索引；-1 表示占位
    返回: (T, ...)
    """
    T = inds.shape[0]
    out = np.zeros((T,) + x.shape[1:], dtype=x.dtype)

    # 只取有效索引，并保持时间顺序（稳妥起见再排序一下）
    valid = inds[inds >= 0]
    if valid.size > 0:
        valid = np.sort(valid)  # 我们的采样本来就是非降序，这里保险
        n = valid.size
        out[:n] = x[valid]
    return out


if __name__ == "__main__":
    # 固定随机种子，便于复现
    rng = np.random.default_rng(123)

    F, T = 100, 10
    print("=== Case A: 训练采样（F>=T, mix_prob=0.5, p_interval=(1,1), 子段=全视频）===")
    for k in range(3):
        idsA = sample_indices_train(F, T, rng=rng, mix_prob=0.5, p_interval=(1.0, 1.0))
        print(f"A{k}: {idsA}  (len={len(idsA)}, min={idsA.min()}, max={idsA.max()})")

    print("\n=== Case B: 训练采样（裁出子段比例 0.5~0.6，有机会 num_sub<T 触发 -1 补齐）===")
    F2, T2 = 40, 48  # 故意让 T2 > F2，多数情况会出现 -1
    for k in range(3):
        idsB = sample_indices_train(F2, T2, rng=rng, mix_prob=1.0, p_interval=(0.5, 0.6))
        n_pad = int(np.sum(idsB < 0))
        print(f"B{k}: first 30 -> {idsB[:30]} ...  #(-1)={n_pad}")

    print("\n=== Case C: 训练采样（全局随机唯一取样，F>=T）===")
    for k in range(2):
        idsC = sample_indices_train(F, T, rng=rng, mix_prob=0.0)  # 0.0 => 只用“全局随机唯一取样”
        print(f"C{k}: {idsC}  sorted={np.all(idsC[:-1] <= idsC[1:])}")

    print("\n=== Case D: 验证/测试采样（确定性均匀）===")
    idsD1 = sample_indices_eval(100, 10)
    idsD2 = sample_indices_eval(7, 10)  # F<T，后面 -1
    print("D1 (F=100,T=10):", idsD1)
    print("D2 (F=7,T=10):  ", idsD2, "  #(-1)=", int(np.sum(idsD2 < 0)))

    print("\n=== Case E: index_with_pad 演示（-1 位置用 0 帧填充）===")
    # 构造一个简单序列：x[i]=[i, i*10]
    x = np.stack([np.arange(7), 10 * np.arange(7)], axis=1).astype(np.float32)  # (F=7, 2)
    inds = np.array([0, 2, 4, -1, 6, -1, -1], dtype=np.int32)
    y = index_with_pad(x, inds)  # (T=7, 2)
    print("x (F=7):\n", x)
    print("inds:\n", inds)
    print("y (取样+补零):\n", y)

    print("\n=== 小结 ===")
    print("* 没有插值：F<T 或子段<T 时，用 -1 占位；用 index_with_pad 把这些位置填 0 帧。")
    print("* 验证/测试：固定等间隔，确保可复现。")
    print("* 训练：50% 先裁再采 + 50% 全局随机唯一，rng 控制随机性。")
