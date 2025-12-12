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
  Dataset → 单样本可视化（xyz骨架 + 动态球）
- 使用 RawXYZDataset，触发采样 & 旋转增强后再可视化
- 显示映射: Matplotlib (X,Y,Z) = Skeleton (x,z,y)
"""


from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataset_ntu60_3D_pkl_npz_ms import RawXYZDataset

# ======= 配置：改成你的 npz 路径与样本索引 =======
NPZ_PATH = Path(
    r"G:/datasets/cls_video/nturgbd_skeletons_s001_to_s017/pkl_mmaction/npz_raw_xyz_ntu60/ntu60_cs_test_xyz_raw.npz",
)
SAMPLE_IDX = 101            # 可视化第几条样本（0-based）
DATASET_MODE = 'train'      # 'train' 可触发旋转增强
T_CLIP = 300                # 数据集取出的片段长度
CLIP_TIMES = 1
INTERVAL = 0.05             # 播放间隔
# =============================================

# 25-joint 拓扑（NTU-RGB+D）
BONES_25: List[Tuple[int, int]] = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (10, 23),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (6, 21),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
]
P_MAX = 2


# ---------- 小工具 ----------
def _mask_valid(arr: np.ndarray, eps=1e-6):
    """(...,3) → (...,) 非零点掩码"""
    return ~(np.all(np.abs(arr) < eps, axis=-1))


def _sphere(ax, xc, yc, zc, r, **kw):
    """
    在 Matplotlib 三维坐标 (X=水平, Y=深度, Z=高度) 上画标准球面
      参数已经是 Matplotlib 坐标: (xc, yc, zc, r)
    """
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    cu, su = np.cos(u)[:, None], np.sin(u)[:, None]  # 60×1
    sv, cv = np.sin(v)[None, :], np.cos(v)[None, :]  # 1×30

    xs = xc + r * (cu @ sv)  # 60×30
    ys = yc + r * (su @ sv)  # 60×30
    zs = zc + r * (np.ones_like(cu) @ cv)  # 60×30
    ax.plot_wireframe(xs, ys, zs, **kw)


def set_axes_equal_by_data(ax3d, pts_xyz_mapped: np.ndarray, sph_mapped: Optional[np.ndarray]):
    """
    用当前帧数据设定等比例立方体范围
    - pts_xyz_mapped: (N,3) 已映射到 (X,Y,Z)=(x,z,y) 的非零点
    - sph_mapped: (P,4) or None，(cx,cy,cz,r) 也已映射到 (X,Y,Z)
    """
    clouds = []
    if pts_xyz_mapped.size > 0:
        clouds.append(pts_xyz_mapped)
    if sph_mapped is not None and sph_mapped.size > 0:
        clouds.append(sph_mapped[:, :3])

    if not clouds:
        return
    xyz = np.concatenate(clouds, axis=0)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius = max(radius, 1e-6)
    ax3d.set_xlim(center[0] - radius, center[0] + radius)
    ax3d.set_ylim(center[1] - radius, center[1] + radius)
    ax3d.set_zlim(center[2] - radius, center[2] + radius)


def spheres_from_frame(pts_xyz_pv3: np.ndarray) -> np.ndarray:
    """
    根据当前帧关键点计算每人的球心半径
    输入: (P,V,3)  输出: (P,4)  每个元素 [cx,cy,cz,r]
    （注意这里返回的 cy=原坐标系的 y；等会儿映射到 Matplotlib 会做 (cx,cz,cy)）
    """
    P, V, _ = pts_xyz_pv3.shape
    out = np.zeros((P, 4), dtype=np.float32)
    for p in range(P):
        pts = pts_xyz_pv3[p]
        valid = _mask_valid(pts)
        if not valid.any():
            continue
        sub = pts[valid]
        ctr = sub.mean(axis=0)
        r = np.linalg.norm(sub - ctr, axis=1).max()
        out[p] = np.array([ctr[0], ctr[1], ctr[2], r], dtype=np.float32)
    return out


def draw_xyz(ax, pts_xyz: np.ndarray, title: str, xyz_sph: Optional[np.ndarray] = None):
    """
    单帧绘制（只一张图）：
      pts_xyz : (P,V,3)  —— 原始 (x,y,z)
      xyz_sph : (P,4) or None   (cx,cy,cz,r)
    显示映射: (X,Y,Z) = (x,z,y)
    """
    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Z (depth)')
    ax.set_zlabel('Y (height)')
    ax.view_init(20, -60)

    # 为设轴范围收集数据（映射到 (x,z,y)）
    mapped_pts = []
    for p in range(min(P_MAX, pts_xyz.shape[0])):
        pts = pts_xyz[p]
        vm = _mask_valid(pts)
        if vm.any():
            mapped = np.stack([pts[:, 0], pts[:, 2], pts[:, 1]], axis=1)[vm]
            mapped_pts.append(mapped)
    mapped_pts = np.concatenate(mapped_pts, axis=0) if mapped_pts else np.empty((0, 3))

    # 球心同样映射：(cx,cy,cz,r) → (cx,cz,cy,r) 仅用于设轴
    sph_mapped = None
    if xyz_sph is not None and xyz_sph.ndim == 2 and xyz_sph.shape[1] == 4:
        sph_mapped = np.stack([xyz_sph[:, 0], xyz_sph[:, 2], xyz_sph[:, 1], xyz_sph[:, 3]], axis=1)

    set_axes_equal_by_data(ax, mapped_pts, sph_mapped)

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for p in range(min(P_MAX, pts_xyz.shape[0])):
        c = cols[p % len(cols)]
        pts = pts_xyz[p]
        vm = _mask_valid(pts)
        X = pts[:, 0]
        Y = pts[:, 2]  # 深度
        Z = pts[:, 1]  # 高度
        ax.scatter(X[vm], Y[vm], Z[vm], c=c, s=14)
        for i, j in BONES_25:
            if i < X.size and j < X.size and vm[i] and vm[j]:
                ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]], c=c, lw=1.0)

        if xyz_sph is not None and p < xyz_sph.shape[0]:
            cx, cy, cz, r = xyz_sph[p]
            if r > 1e-6:
                # 注意：_sphere 的参数就是 Matplotlib 的 (xc, yc, zc, r)
                # 这里 Matplotlib 的 yc=深度=原来的 cz，zc=高度=原来的 cy
                _sphere(ax, cx, cz, cy, r, color=c, alpha=0.25, lw=0.6)

    ax.set_title(title)


def visualize_dataset(npz_path: Path, idx: int, interval: float = 0.05,
                      mode: str = 'train', T: int = 100, clip: int = 1):
    # 1) 构建数据集，取一条样本（会触发采样 + 旋转增强）
    ds = RawXYZDataset(npz_path, mode=mode, T=T, clip=clip)
    feat, label = ds[idx]  # feat: (C=3, T, V=25, P=2)
    # 还原为 (T,P,V,3)
    g_xyz = feat.permute(1, 3, 2, 0).cpu().numpy().astype(np.float32)  # (T,P,V,3)

    # 2) 取 view（可选）：从内部元数据中靠 _cum/_locate 找到原始条目
    view_val = -1
    try:
        gi = idx % ds._cum[-1]
        fi, inner = ds._locate(ds._cum, gi)  # 直接用类里的静态方法
        if 'view' in ds._ds[fi]:
            view_val = int(ds._ds[fi]['view'][inner])
    except Exception:
        pass  # 没有 view 也不影响绘制

    F = g_xyz.shape[0]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for t in range(F):
        # 可视化用的球心半径：由当前帧关键点即时计算，匹配当前坐标系 & 增强后结果
        sph = spheres_from_frame(g_xyz[t])  # (P,4) [cx,cy,cz,r]
        draw_xyz(ax, g_xyz[t], title="Global XYZ + Sphere", xyz_sph=sph)
        fig.suptitle(f"Sample {idx}  |  Label {label}  |  View {view_val}  |  Frame {t + 1}/{F}")
        plt.pause(interval)

    plt.show()


if __name__ == '__main__':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    visualize_dataset(NPZ_PATH, SAMPLE_IDX, INTERVAL, mode=DATASET_MODE, T=T_CLIP, clip=CLIP_TIMES)
