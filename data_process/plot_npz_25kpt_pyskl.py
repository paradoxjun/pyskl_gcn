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
  NPZ → 单样本可视化（xyz骨架 + xyz_sphere）
- data[idx]        : (F,P,V,3)
- xyz_sphere[idx]  : (F,P,4)  每帧每人 (cx,cy,cz,r)
- 标题显示 view / label / 帧号
- 显示映射: Matplotlib (X,Y,Z) = Skeleton (x,z,y)
"""


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# ======= 配置：改成你的 npz 路径与样本索引 =======
NPZ_PATH = Path(
    r"G:/datasets/cls_video/nturgbd_skeletons_s001_to_s017/pkl_mmaction/npz_raw_xyz_ntu60/ntu60_cs_test_xyz_raw.npz"
)
SAMPLE_IDX = 54
INTERVAL = 0.05
USE_VALID_FRAME = True
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


def _mask_valid(arr: np.ndarray, eps=1e-6):
    """(...,3) → (...,) 非零点掩码"""
    return ~(np.all(np.abs(arr) < eps, axis=-1))


def _sphere(ax, xc, yc, zc, r, **kw):
    """
    在 Matplotlib 三维坐标 (X=水平, Y=深度, Z=高度) 上画标准球：
      X = xc + r*cos(u)*sin(v)
      Y = yc + r*sin(u)*sin(v)
      Z = zc + r*cos(v)
    其中 u ∈ [0,2π], v ∈ [0,π]
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


def draw_xyz(ax, pts_xyz: np.ndarray, title: str, xyz_sph: Optional[np.ndarray] = None):
    """
    单帧绘制：
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

    # 球心同样映射：(cx,cy,cz,r) → (cx,cz,cy,r) 再传入 _sphere 时要还原为 (xc,yc,zc)
    sph_mapped = None
    if xyz_sph is not None and xyz_sph.ndim == 2 and xyz_sph.shape[1] == 4:
        # 仅用于设轴的中心 (X,Y,Z)
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


def visualize_npz(npz_path: Path, idx: int, interval: float = 0.05, use_valid_frame: bool = True):
    f = np.load(npz_path, allow_pickle=True)

    data = f['data']  # object 数组，每个元素 (F,P,V,3)
    xyz_sphere = f['xyz_sphere']  # object 数组，每个元素 (F,P,4)
    label_arr = f['label']  # (N,)
    view_arr = f['view'] if 'view' in f.files else None
    vf_arr = f['valid_frame'] if 'valid_frame' in f.files else None

    pts = data[idx]  # (F,P,V,3)
    sph = xyz_sphere[idx]  # (F,P,4)
    label = int(label_arr[idx])
    view = int(view_arr[idx]) if view_arr is not None else -1

    F = pts.shape[0]
    if use_valid_frame and vf_arr is not None:
        F_valid = int(vf_arr[idx])
        F = max(1, min(F, F_valid))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for t in range(F):
        draw_xyz(ax, pts[t], title="Global XYZ + Sphere", xyz_sph=sph[t] if sph is not None else None)
        fig.suptitle(f"Sample {idx}  |  Label {label}  |  View {view}  |  Frame {t + 1}/{F}")
        plt.pause(interval)

    plt.show()
    f.close()


if __name__ == '__main__':
    visualize_npz(NPZ_PATH, SAMPLE_IDX, INTERVAL, USE_VALID_FRAME)
