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
  NPZ → 单样本可视化（xyz骨架 + xyz_sphere）并保存 GIF
- data[idx]        : (F,P,V,3)
- xyz_sphere[idx]  : (F,P,4)  每帧每人 (cx,cy,cz,r)
- 标题显示 view / label / 帧号
- 显示映射: Matplotlib (X,Y,Z) = Skeleton (x,z,y)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from typing import List, Tuple, Optional

# ======= 配置：改成你的 npz 路径与样本索引 =======
NPZ_PATH = Path(
    r"G:/datasets/cls_video/nturgbd_skeletons_s001_to_s017/pkl_mmaction/npz_raw_xyz_ntu60/ntu60_cs_test_xyz_raw.npz"
)
SAMPLE_IDX = 42
INTERVAL = 0.05
USE_VALID_FRAME = True

# ======= GIF 输出配置 =======
SAVE_GIF = True
GIF_PATH = Path(f"sample_{SAMPLE_IDX}.gif")  # 你也可以写绝对路径
GIF_FPS = 20          # GIF 播放帧率（与 INTERVAL 无关，你自己定）
GIF_DPI = 100         # 清晰度：越大越清晰，但文件越大、生成越慢
SHOW_WINDOW = True    # 生成时是否同时弹窗播放
FIGSIZE = (7, 7)      # 图大小
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
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    cu, su = np.cos(u)[:, None], np.sin(u)[:, None]  # 60×1
    sv, cv = np.sin(v)[None, :], np.cos(v)[None, :]  # 1×30

    xs = xc + r * (cu @ sv)  # 60×30
    ys = yc + r * (su @ sv)  # 60×30
    zs = zc + r * (np.ones_like(cu) @ cv)  # 60×30
    ax.plot_wireframe(xs, ys, zs, **kw)


def set_axes_equal_by_data(ax3d, pts_xyz_mapped: np.ndarray, sph_mapped: Optional[np.ndarray]):
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
    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Z (depth)')
    ax.set_zlabel('Y (height)')
    ax.view_init(20, -60)

    mapped_pts = []
    for p in range(min(P_MAX, pts_xyz.shape[0])):
        pts = pts_xyz[p]
        vm = _mask_valid(pts)
        if vm.any():
            mapped = np.stack([pts[:, 0], pts[:, 2], pts[:, 1]], axis=1)[vm]
            mapped_pts.append(mapped)
    mapped_pts = np.concatenate(mapped_pts, axis=0) if mapped_pts else np.empty((0, 3))

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
                _sphere(ax, cx, cz, cy, r, color=c, alpha=0.25, lw=0.6)

    ax.set_title(title)


def visualize_npz(
    npz_path: Path,
    idx: int,
    interval: float = 0.05,
    use_valid_frame: bool = True,
    save_gif: bool = False,
    gif_path: Optional[Path] = None,
    gif_fps: int = 20,
    gif_dpi: int = 120,
    show_window: bool = True,
    figsize=(7, 7),
):
    f = np.load(npz_path, allow_pickle=True)

    data = f['data']
    xyz_sphere = f['xyz_sphere']
    label_arr = f['label']
    view_arr = f['view'] if 'view' in f.files else None
    vf_arr = f['valid_frame'] if 'valid_frame' in f.files else None

    pts = data[idx]
    sph = xyz_sphere[idx]
    label = int(label_arr[idx])
    view = int(view_arr[idx]) if view_arr is not None else -1

    F = int(pts.shape[0])
    if use_valid_frame and vf_arr is not None:
        F_valid = int(vf_arr[idx])
        F = max(1, min(F, F_valid))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    if show_window:
        plt.ion()  # 交互模式，边画边显示更顺滑

    writer = None
    if save_gif:
        gif_path = Path(gif_path) if gif_path is not None else Path(f"sample_{idx}.gif")
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=gif_fps)

    def _render_one(t: int):
        draw_xyz(ax, pts[t], title="Global XYZ + Sphere", xyz_sph=sph[t] if sph is not None else None)
        fig.suptitle(f"Sample {idx}  |  Label {label}  |  View {view}  |  Frame {t + 1}/{F}")
        fig.canvas.draw_idle()

    if writer is not None:
        with writer.saving(fig, str(gif_path), dpi=gif_dpi):
            for t in range(F):
                _render_one(t)
                fig.canvas.draw()       # 确保这一帧渲染完成
                writer.grab_frame()     # 抓这一帧到 GIF

                if show_window:
                    plt.pause(interval)
        print(f"[OK] GIF saved to: {gif_path.resolve()}")
    else:
        for t in range(F):
            _render_one(t)
            plt.pause(interval)

    if show_window:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)

    f.close()


if __name__ == '__main__':
    visualize_npz(
        NPZ_PATH,
        SAMPLE_IDX,
        INTERVAL,
        USE_VALID_FRAME,
        save_gif=SAVE_GIF,
        gif_path=GIF_PATH,
        gif_fps=GIF_FPS,
        gif_dpi=GIF_DPI,
        show_window=SHOW_WINDOW,
        figsize=FIGSIZE,
    )
