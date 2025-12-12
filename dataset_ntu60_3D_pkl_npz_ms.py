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
  NTU-RGB+D · Raw-XYZ NPZ → Dataset（仅旋转增强）
----------------------------------------------
- 只读取 xyz 坐标，输出张量 [C, T, V=25, P=2] 和 label
- 采样策略保持与原实现一致
- 数据增强：仅保留三轴旋转（概率=1.0）
    * 绕 y 轴 (yaw)：按机位视角 VIEW_YAW_RANGES 采样角度
    * 绕 x / z 轴：各自在 [-15°, +15°] 内均匀采样
- 旋转中心：原点；仅旋转非零点（零点保持 0）

新增：
- 支持多流特征生成与合并（与 pyskl 语义一致）：
  * j  : joints（原始坐标）
  * b  : bones（关节差，按固定骨架连边）
  * jm : joints motion（相邻帧差分）
  * bm : bones motion（对 b 再做相邻帧差分）
- 默认 streams=('j',)，与旧版保持一致；设置 streams=('j','b','jm','bm') 可输出 12 通道。但论文中的做法只选择一个，最后多个模型统计结果。
"""


from __future__ import annotations
from pathlib import Path
from typing import Sequence, Union, List, Tuple

import numpy as np
import torch, random
from torch.utils.data import Dataset

from task_utils.data.data_sample import sample_indices_train, sample_indices_eval
from task_utils.data.data_aug_3d import random_rotate_3d, random_scale_3d, flip_x_3d, mirror_lr_3d
from task_utils.data.data_skeleton_ops import gen_and_merge_feats_TPVC

# ------------------- 常量 -------------------
V, P = 25, 2
RX_RNG_DEG = 10.0  # x轴：±10°
RY_RNG_DEG = 10.0  # y轴：±10°
RZ_RNG_DEG = 10.0  # z轴：±10°


# =================== Dataset ===================
class RawXYZDataset(Dataset):
    """
    读取 npz（至少包含：'data' 和 'label'；可选 'view'）
    - data  : List/Array，元素形状 (F, P<=2, V=25, 3)
    - label : List/Array，元素是 int 类别
    - view  : 可选 List/Array（1/2/3），决定绕 y 轴(yaw)的采样范围

    新增参数：
    - streams: ('j',) | ('j','b') | ('j','jm','b','bm') 等任意子集，默认仅 'j'（与旧版一致）
               输出通道 = 3 * len(streams)
    """

    def __init__(self,
                 npz_path: Union[str, Sequence[str]],
                 mode: str = 'train',
                 T: int = 64,
                 clip: int = 1,
                 seed: int = 42,
                 streams: Tuple[str, ...] = ('j',)):  # 多流选择（默认只用 joints）
        if mode not in {'train', 'val', 'test'}:
            raise ValueError("ERROR: mode 必须在 {'train','val','test'}")
        self.mode, self.T = mode, T
        self.clip = clip if mode == 'train' else 1
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # 多流设置（顺序即拼接/输出顺序）
        self.streams = tuple(streams) if streams else ('j',)
        if not all(s in {'j', 'b', 'jm', 'bm'} for s in self.streams):
            raise ValueError(f"ERROR: streams 仅支持 'j','b','jm','bm'，收到：{self.streams}")

        paths = [npz_path] if isinstance(npz_path, (str, Path)) else list(npz_path)
        if not paths:
            raise ValueError('ERROR: npz_path 为空')

        # —— 全部读取到内存 —— #
        self._ds = []
        for p in paths:
            f = np.load(p, allow_pickle=True)
            self._ds.append({k: f[k] for k in f})
            f.close()

        # 累积分隔
        self._cum: List[int] = []
        tot = 0
        for d in self._ds:
            tot += len(d['label'])
            self._cum.append(tot)

        # 输出通道：每个流 3 通道（xyz），总通道 = 3 * len(streams)
        self.C = 3 * max(1, len(self.streams))

    # ---------- utils ----------
    def __len__(self):
        return self._cum[-1] * (self.clip if self.mode == 'train' else 1)

    @staticmethod
    def _locate(cum: List[int], idx: int) -> Tuple[int, int]:
        for i, v in enumerate(cum):
            if idx < v:
                prev = cum[i - 1] if i else 0
                return i, idx - prev
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        gi = idx % self._cum[-1]
        fi, inner = self._locate(self._cum, gi)
        ds = self._ds[fi]

        xyz: np.ndarray = ds['data'][inner]  # (F, P<=2, 25, 3)
        label: int = int(ds['label'][inner])

        # 采样帧
        F = xyz.shape[0]
        if self.mode == 'train':
            ids = sample_indices_train(
                F, self.T,
                rng=self.rng,
                mix_prob=0.5,  # 50% 先裁再采（pyskl 风格）+ 50% 全局随机唯一
                p_interval=(0.5, 1.0)  # 子段比例区间，可按需改
            )
        else:
            ids = sample_indices_eval(F, self.T)

        # 根据索引取帧（-1 位置自动补 0）
        P_cur = xyz.shape[1]
        xyz_clip = np.zeros((self.T, P_cur, V, 3), np.float32)
        m = ids >= 0
        if m.any():
            xyz_clip[m] = xyz[ids[m]]

        # ---------------- 翻转 / 缩放 / 旋转（在拷贝上做增强） ----------------
        x_out = xyz_clip  # 默认用原采样结果
        if self.mode == 'train' and np.any(np.abs(xyz_clip) > 0):
            xyz_aug = xyz_clip.copy()

            # 1)语义镜像（全样本统一）,NTU上负面作用
            # if random.random() < 0.0:
            #     mirror_lr_3d(
            #         xyz_aug,
            #     )
            # 2) 翻转 X（关于 YOZ 面），y/z 不变，虽然改变左右肢体语义，但在该数据集上有效
            if random.random() < 0.5:
                flip_x_3d(
                    xyz_aug,
                    center='origin',
                    mask_zero=True
                )

            # 3) 随机缩放（±20%）；与现有实现一致：p_iso=0.5（等比/各轴独立均可）
            if random.random() < 1.0:
                random_scale_3d(
                    xyz_aug,
                    scale=0.2,  # 等比例范围 [1-0.2, 1+0.2]
                    p_iso=0.5,  # 50% 等比；否则各轴独立
                    center='origin',
                    mask_zero=True,
                    rng=self.rng,
                )

            # 4) 随机旋转（三轴范围；单位：度；顺序 zyx）
            if random.random() < 1.0:
                random_rotate_3d(
                    xyz_aug,
                    rx=RX_RNG_DEG,
                    ry=RY_RNG_DEG,
                    rz=RZ_RNG_DEG,
                    degrees=True,
                    center='origin',
                    mask_zero=True,
                    order='zyx',
                    rng=self.rng,
                )

            x_out = xyz_aug
        # ---------------- 数据增强结束 ----------------

        # 若只检测到 1 人，pad 到 P=2
        if P_cur == 1:
            pad = ((0, 0), (0, 1), (0, 0), (0, 0))  # (T, P, V, 3)
            x_out = np.pad(x_out, pad, mode='constant')

        # ---------------- 生成并合并多流特征（沿通道维拼接） ----------------
        # x_out: (T, P, V, 3)  →  merged: (T, P, V, 3 * len(streams))
        merged = gen_and_merge_feats_TPVC(
            joints_TPVC=x_out,
            dataset="nturgb+d",
            feats=self.streams,
            concat_axis=-1
        )

        # 返回形状：[C, T, V, P]
        feat = torch.from_numpy(merged).permute(3, 0, 2, 1).contiguous()  # (C, T, 25, 2)
        return feat, label


# ---------------- quick test ----------------
if __name__ == '__main__':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 示例 npz 路径
    npz = r'G:/datasets/cls_video/nturgbd_skeletons_s001_to_s017/pkl_mmaction/npz_raw_xyz_ntu60/ntu60_cs_test_xyz_raw.npz'

    # 1) 仅 joints（与旧版完全一致）
    ds = RawXYZDataset(npz, mode='train', T=100, clip=1, streams=('j',))
    print('len =', len(ds))
    x, y = ds[0]
    print('feat(j):', x.shape, 'label:', y)  # 期望: (3, T, 25, 2)

    # 2) joints + bones + motion（示例）
    ds2 = RawXYZDataset(npz, mode='train', T=100, clip=1, streams=('j', 'b', 'jm', 'bm'))
    x2, y2 = ds2[0]
    print('feat(j+b+jm+bm):', x2.shape, 'label:', y2)  # 期望: (12, T, 25, 2)
