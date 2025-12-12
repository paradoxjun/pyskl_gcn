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
  2D关键点数据增强代码。
"""


import numpy as np
from typing import Optional, Sequence, Union, Dict, Any, Tuple

# === COCO-17 左右关键点默认索引（与常见实现一致） ===
"""
0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
"""

_COCO17_LEFT = [1, 3, 5, 7, 9, 11, 13, 15]
_COCO17_RIGHT = [2, 4, 6, 8, 10, 12, 14, 16]


def random_shift_2d(
        kpts: np.ndarray,
        video_box: Union[Sequence[float], np.ndarray],  # [x1, y1, x2, y2] ∈ [0,1]
        *,
        p: float = 0.5,                 # 触发平移的概率
        clip: bool = False,             # 是否把结果裁剪到 [0,1]（默认 False，便于暴露异常）
        mask_zero: bool = True,         # 仅对非零点生效（(0,0) 视为无效）
        rng: Optional[np.random.Generator] = None,
        return_params: bool = False     # True → 返回 (kpts, (dx,dy), applied_bool)
):
    """
    随机平移 2D 关键点（原地修改）：
      - 输入 kpts 的最后两维必须是 (K,2)，可带任意前缀形状（如 (T,P,K,2)）
      - 平移范围由 video_box 限制：dx ∈ [-x1, 1-x2]，dy ∈ [-y1, 1-y2]
      - 默认不裁剪到 [0,1]，因为理论上 video_box 已确保不越界，裁剪会掩盖数据问题
    """
    if kpts.ndim < 2 or kpts.shape[-1] != 2:
        raise ValueError(f"ERROR: last dim must be 2, got shape={kpts.shape}")

    rng = np.random.default_rng() if rng is None else rng
    x1, y1, x2, y2 = map(float, np.asarray(video_box).reshape(-1))
    if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1):
        raise ValueError(f"ERROR: invalid video_box={video_box}")

    applied = (rng.random() < float(p))
    dx = dy = 0.0

    if applied:
        dx_min, dx_max = -x1, (1.0 - x2)
        dy_min, dy_max = -y1, (1.0 - y2)
        dx = rng.uniform(dx_min, dx_max) if dx_max >= dx_min else 0.0
        dy = rng.uniform(dy_min, dy_max) if dy_max >= dy_min else 0.0

        if mask_zero:
            m = np.any(np.abs(kpts) > 0, axis=-1)  # (..., K)
            if np.any(m):
                kpts[..., 0][m] += dx
                kpts[..., 1][m] += dy
        else:
            kpts[..., 0] += dx
            kpts[..., 1] += dy

        if clip:
            np.clip(kpts, 0.0, 1.0, out=kpts)

    return (kpts, np.array([dx, dy], np.float32), applied) if return_params else kpts


def random_flip_2d(
        kpts: np.ndarray,
        *,
        p_flip: float = 0.5,            # 坐标水平翻转概率（x' = 2*center_x - x）
        p_swap: float = 0.5,            # 左右关键点交换概率（索引重排）
        center_x: float = 0.5,          # 翻转中心（坐标已归一化到 [0,1] 时设 0.5）
        left_kp: Optional[Sequence[int]] = None,    # 默认用 COCO-17
        right_kp: Optional[Sequence[int]] = None,   # 默认用 COCO-17
        mask_zero: bool = True,         # 仅对非零点做坐标翻转；交换索引不受该掩码影响
        rng: Optional[np.random.Generator] = None,
        return_params: bool = False     # True → 返回 (kpts, info_dict)
):
    """
    随机水平翻转 +（可选）左右关键点交换（原地修改）：
      - 翻转（几何）：x' = 2*center_x - x；y 不变；默认只作用于非零点
      - 交换（语义）：在 K 维度上按左右配对重排索引（对整个样本统一重排）
      - p_flip 与 p_swap 相互独立；将 p_swap=0 → 退化为“只翻转”，从而完全兼容你原本实现
      - 不处理 bbox；若需要，仍和你原逻辑一样对 bbox.cx 做 cx' = 2*center_x - cx
    """
    if kpts.ndim < 2 or kpts.shape[-1] != 2:
        raise ValueError(f"ERROR: last dim must be 2, got shape={kpts.shape}")

    rng = np.random.default_rng() if rng is None else rng
    K = kpts.shape[-2]

    # --- 默认左右配对（COCO-17） ---
    if left_kp is None and right_kp is None:
        if K != 17:
            raise ValueError(f"ERROR: default L/R only for COCO-17, but K={K}. "
                             f"Please pass left_kp/right_kp explicitly.")
        left_kp, right_kp = _COCO17_LEFT, _COCO17_RIGHT
    if (left_kp is None) ^ (right_kp is None):
        raise ValueError("ERROR: left_kp and right_kp must be both None or both provided.")
    if left_kp is not None:
        if len(left_kp) != len(right_kp):
            raise ValueError("ERROR: left_kp and right_kp must have the same length.")
        if max(max(left_kp), max(right_kp)) >= K:
            raise ValueError("ERROR: L/R index out of range for K={K}.")

    did_flip = (rng.random() < float(p_flip))
    did_swap = (rng.random() < float(p_swap))

    # 1) 坐标翻转（水平）
    if did_flip:
        if mask_zero:
            m = np.any(np.abs(kpts) > 0, axis=-1)  # (...,K)
            if np.any(m):
                x = kpts[..., 0]
                x[m] = 2.0 * center_x - x[m]
        else:
            kpts[..., 0] = 2.0 * center_x - kpts[..., 0]

    # 2) 左右关键点索引交换（对 K 维统一重排）
    index_map = None
    if did_swap and left_kp is not None:
        index_map = np.arange(K, dtype=np.int64)
        for l, r in zip(left_kp, right_kp):
            index_map[l], index_map[r] = index_map[r], index_map[l]
        # 右值会构造临时，原地安全写回
        kpts[...] = kpts[..., index_map, :]

    if return_params:
        info: Dict[str, Any] = {
            "did_flip": bool(did_flip),
            "did_swap": bool(did_swap),
            "center_x": float(center_x),
            "index_map": (index_map.copy() if index_map is not None else None),
        }
        return kpts, info
    return kpts


def random_scale_2d(
        kpts: np.ndarray,               # (..., K, 2)
        video_box: Union[Sequence[float], np.ndarray],  # [x1,y1,x2,y2] ∈ [0,1]
        *,
        scale_amp: float = 0.2,         # 相对幅度：最终 ∩ [1-scale_amp, 1+scale_amp]
        p: float = 0.5,                 # 触发缩放的概率
        mask_zero: bool = True,         # 仅对非零点生效（(0,0) 视为无效）
        rng: Optional[np.random.Generator] = None,
        apply_bbox: Optional[np.ndarray] = None,  # 若传入，形如 (..., P, 5): [cx,cy,w,h,det_c]
        return_params: bool = False     # True → 返回 (kpts, s, cur_vbox[4], applied)
):
    """
    2D 随机缩放（关于 video_box 的中心），原地修改：
      - 允许范围 = 几何不越界约束 × 相对幅度约束
      - 采样分布：Beta(2,2) 在有效区间内（更靠近 1），否则不缩放（s=1）
      - 若传入 apply_bbox，会同步对 bbox 的 [cx,cy,w,h] 做缩放更新
    """
    if kpts.ndim < 2 or kpts.shape[-1] != 2:
        raise ValueError(f"ERROR: last dim must be 2, got shape={kpts.shape}")

    rng = np.random.default_rng() if rng is None else rng
    eps = 1e-6
    scale_amp = float(np.clip(scale_amp, 0.0, 1.0))
    p = float(np.clip(p, 0.0, 1.0))

    x1, y1, x2, y2 = map(float, np.asarray(video_box).reshape(-1))
    if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1):
        raise ValueError(f"ERROR: invalid video_box={video_box}")

    # 基本几何量
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    # 几何不越界约束（仅限制上界）：放大到贴边为止
    def _safe_div(a, b):
        return a / b if abs(b) > eps else np.inf

    sxL = _safe_div(cx,  (cx - x1))
    sxR = _safe_div(1.0 - cx, (x2 - cx))
    syT = _safe_div(cy,  (cy - y1))
    syB = _safe_div(1.0 - cy, (y2 - cy))
    s_high_geom = max(1.0, min(sxL, sxR, syT, syB))

    # 相对幅度约束
    s_low_rel, s_high_rel = (1.0 - scale_amp), (1.0 + scale_amp)

    # 有效区间（几何×相对幅度）
    s_lo = max(s_low_rel, 0.0 + eps)
    s_hi = min(s_high_geom, s_high_rel)

    applied = (rng.random() < p) and (s_hi >= s_lo + 1e-6)
    s = 1.0
    if applied:
        # 在 [s_lo, s_hi] 上用 Beta(2,2) 采样（更靠近 1 的温和缩放）
        u = rng.beta(2.0, 2.0)
        s = s_lo + u * (s_hi - s_lo)

        # ---- 应用到 kpts（关于 video_box 中心缩放）----
        if mask_zero:
            m = np.any(np.abs(kpts) > 0, axis=-1)
            if np.any(m):
                x = kpts[..., 0]
                y = kpts[..., 1]
                x[m] = cx + s * (x[m] - cx)
                y[m] = cy + s * (y[m] - cy)
        else:
            kpts[..., 0] = cx + s * (kpts[..., 0] - cx)
            kpts[..., 1] = cy + s * (kpts[..., 1] - cy)

        # ---- 应用到 bbox（可选）----
        if apply_bbox is not None:
            apply_bbox[..., 0] = cx + s * (apply_bbox[..., 0] - cx)  # cx
            apply_bbox[..., 1] = cy + s * (apply_bbox[..., 1] - cy)  # cy
            apply_bbox[..., 2] *= s  # w
            apply_bbox[..., 3] *= s  # h

    # 返回缩放后的 vbox（供“平移”使用）
    cur_vbox = np.array([cx + s * (x1 - cx), cy + s * (y1 - cy),
                         cx + s * (x2 - cx), cy + s * (y2 - cy)], dtype=np.float32)
    cur_vbox = np.clip(cur_vbox, 0.0, 1.0)

    return (kpts, float(s), cur_vbox, bool(applied)) if return_params else kpts


def random_stretch_2d(
    kpts: np.ndarray,                         # (..., K, 2) ∈ [0,1]
    video_box: Union[Sequence[float], np.ndarray],  # [x1,y1,x2,y2] ∈ [0,1]
    *,
    amp: Union[float, Tuple[float, float]] = 0.2,  # 允许相对幅度：fx∈[1-ax,1+ax], fy∈[1-ay,1+ay]
    p: float = 0.5,                           # 触发概率
    rng: Optional[np.random.Generator] = None,
    mask_zero: bool = True,                   # 仅对非零点生效
    apply_bbox: Optional[np.ndarray] = None,  # 若给出 shape (...,P,5)[cx,cy,w,h,det_c]，将同步更新
    return_params: bool = False               # True→ 返回 (kpts, (fx,fy), applied, new_bbox or None)
):
    """
    各向异性缩放：绕 video_box 中心分别缩放 X/Y。
    - 仅用几何不越界约束 + 相对幅度约束（无全局 stats）
    - apply_bbox: (cx,cy) 不变；(w,h) 分别乘以 (fx,fy)
    """
    if kpts.ndim < 2 or kpts.shape[-1] != 2:
        raise ValueError(f"last dim must be 2, got shape={kpts.shape}")
    rng = np.random.default_rng() if rng is None else rng

    x1, y1, x2, y2 = map(float, np.asarray(video_box).reshape(-1))
    if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1):
        raise ValueError(f"invalid video_box={video_box}")
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    # 相对幅度
    if isinstance(amp, (tuple, list)):
        ax, ay = float(amp[0]), float(amp[1])
    else:
        ax = ay = float(amp)
    lo_x, hi_x = 1 - ax, 1 + ax
    lo_y, hi_y = 1 - ay, 1 + ay

    # 几何上界（不越出 [0,1]）
    def _geom_hi(c, a, b):
        left  = c / max(c - a, 1e-6)
        right = (1 - c) / max(b - c, 1e-6)
        return max(1.0, min(left, right))
    hi_x = min(hi_x, _geom_hi(cx, x1, x2))
    hi_y = min(hi_y, _geom_hi(cy, y1, y2))

    # 采样/触发
    applied = (rng.random() < float(p)) and (hi_x >= lo_x + 1e-6) and (hi_y >= lo_y + 1e-6)
    fx = fy = 1.0
    new_bbox = None
    if applied:
        fx = rng.uniform(lo_x, hi_x)
        fy = rng.uniform(lo_y, hi_y)

        # 仅非零点
        if mask_zero:
            m = np.any(np.abs(kpts) > 0, axis=-1)
            if np.any(m):
                x = kpts[..., 0]
                y = kpts[..., 1]
                x[m] = cx + fx * (x[m] - cx)
                y[m] = cy + fy * (y[m] - cy)
        else:
            kpts[..., 0] = cx + fx * (kpts[..., 0] - cx)
            kpts[..., 1] = cy + fy * (kpts[..., 1] - cy)

        if apply_bbox is not None:
            apply_bbox[..., 2] *= fx
            apply_bbox[..., 3] *= fy
            new_bbox = apply_bbox

    return (kpts, (fx, fy), applied, new_bbox) if return_params else kpts


def random_shear_2d(
    kpts: np.ndarray,                         # (..., K, 2)
    video_box: Union[Sequence[float], np.ndarray],  # [x1,y1,x2,y2]
    *,
    shx_amp: float = 0.15,                    # x 方向剪切振幅：y 越大，x 位移越大（关于 cy）
    shy_amp: float = 0.15,                    # y 方向剪切振幅：x 越大，y 位移越大（关于 cx）
    p: float = 0.5,                           # 触发概率
    rng: Optional[np.random.Generator] = None,
    mask_zero: bool = True,
    apply_bbox: Optional[np.ndarray] = None,  # 若传入 bbox，同步更新为四角变换后的 AABB
    return_params: bool = False               # True→ 返回 (kpts, (shx,shy), applied, new_bbox or None)
):
    """
    剪切仿射（不旋转）：绕 video_box 中心做
        x' = x + shx * (y - cy)
        y' = y + shy * (x - cx)
    - 自动收缩 (shx,shy) 使四个角变换后仍落在 [0,1]（几何安全）
    - bbox 更新：四角仿射后取 AABB
    """
    if kpts.ndim < 2 or kpts.shape[-1] != 2:
        raise ValueError(f"last dim must be 2, got shape={kpts.shape}")
    rng = np.random.default_rng() if rng is None else rng

    x1, y1, x2, y2 = map(float, np.asarray(video_box).reshape(-1))
    if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1):
        raise ValueError(f"invalid video_box={video_box}")
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    applied = (rng.random() < float(p)) and (shx_amp > 0 or shy_amp > 0)
    shx = shy = 0.0
    new_bbox = None
    if applied:
        shx = rng.uniform(-shx_amp, shx_amp) if shx_amp > 0 else 0.0
        shy = rng.uniform(-shy_amp, shy_amp) if shy_amp > 0 else 0.0

        # 四角点仿射，若越界则按比例收缩 sh 直到合法
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        def _shear_xy(x, y, shx_, shy_):
            return np.array([x + shx_ * (y - cy), y + shy_ * (x - cx)], np.float32)

        def _aabb_after(shx_, shy_):
            c = np.stack([_shear_xy(x, y, shx_, shy_) for x, y in corners], axis=0)
            xmin, ymin = c[:, 0].min(), c[:, 1].min()
            xmax, ymax = c[:, 0].max(), c[:, 1].max()
            return xmin, ymin, xmax, ymax

        # 线性缩放系数 s∈(0,1]，确保角点落在 [0,1]
        s = 1.0
        for _ in range(10):
            xmin, ymin, xmax, ymax = _aabb_after(s * shx, s * shy)
            if xmin >= 0 and ymin >= 0 and xmax <= 1 and ymax <= 1:
                break
            s *= 0.5
        shx *= s
        shy *= s

        # 关键点仿射
        if mask_zero:
            m = np.any(np.abs(kpts) > 0, axis=-1)
            if np.any(m):
                x = kpts[..., 0]
                y = kpts[..., 1]
                x_m = x[m]
                y_m = y[m]
                x[m] = x_m + shx * (y_m - cy)
                y[m] = y_m + shy * (x_m - cx)
        else:
            x = kpts[..., 0]
            y = kpts[..., 1]
            x[:] = x + shx * (y - cy)
            y[:] = y + shy * (x - cx)

        if apply_bbox is not None:
            xmin, ymin, xmax, ymax = _aabb_after(shx, shy)
            cx2 = 0.5 * (xmin + xmax)
            cy2 = 0.5 * (ymin + ymax)
            w2  = max(xmax - xmin, 1e-6)
            h2  = max(ymax - ymin, 1e-6)
            apply_bbox[..., 0] = cx2
            apply_bbox[..., 1] = cy2
            apply_bbox[..., 2] = w2
            apply_bbox[..., 3] = h2
            new_bbox = apply_bbox

    return (kpts, (shx, shy), applied, new_bbox) if return_params else kpts


if __name__ == "__main__":
    # ---------- 基础构造 ----------
    T, P, K = 2, 1, 17
    base = np.zeros((T, P, K, 2), dtype=np.float32)
    # 两个可观测点（COCO-17 索引）：左肩(5) 、右髋(12)
    base[0, 0, 5]  = [0.20, 0.30]
    base[1, 0, 12] = [0.70, 0.60]

    def sep(tag):
        print("\n" + "=" * 12 + f" {tag} " + "=" * 12)

    print("== 原始 ==")
    print("t0 left_shoulder (5):", base[0, 0, 5])
    print("t1 right_hip    (12):", base[1, 0, 12])

    # ================================================================
    # 1) 仅翻转（不交换左右关键点）
    # ================================================================
    sep("仅翻转（不交换）")
    a_in = base.copy()
    rng_a = np.random.default_rng(123)
    a_out, info = random_flip_2d(a_in, p_flip=1.0, p_swap=0.0, center_x=0.5, return_params=True, rng=rng_a)
    print("did_flip, did_swap:", info["did_flip"], info["did_swap"])
    # 期望：x -> 1 - x
    print("t0 5.x ->", a_out[0, 0, 5, 0], "  期望≈ 0.8")
    print("t1 12.x->", a_out[1, 0, 12, 0], "  期望≈ 0.3")
    assert info["did_flip"] and (not info["did_swap"])
    assert np.isclose(a_out[0, 0, 5, 0], 0.8, atol=1e-6)
    assert np.isclose(a_out[1, 0, 12, 0], 0.3, atol=1e-6)
    # 零点保持为零
    assert np.allclose(a_out[0, 0, 0], [0, 0], atol=1e-12)

    # ================================================================
    # 2) 仅交换（不翻转）
    # ================================================================
    sep("仅交换（不翻转）")
    b_in = base.copy()
    rng_b = np.random.default_rng(7)
    b_out, info = random_flip_2d(b_in, p_flip=0.0, p_swap=1.0, return_params=True, rng=rng_b)
    print("did_flip, did_swap:", info["did_flip"], info["did_swap"])
    # 左肩(5) ↔ 右肩(6)，右髋(12) ↔ 左髋(11)
    print("t0 right_shoulder(6) after swap:", b_out[0, 0, 6], "  期望≈ 原 t0(5)")
    print("t1 left_hip     (11) after swap:", b_out[1, 0, 11], "  期望≈ 原 t1(12)")
    assert (not info["did_flip"]) and info["did_swap"]
    assert np.allclose(b_out[0, 0, 6], base[0, 0, 5], atol=1e-6)
    assert np.allclose(b_out[1, 0, 11], base[1, 0, 12], atol=1e-6)
    # 零点保持为零
    assert np.allclose(b_out[0, 0, 0], [0, 0], atol=1e-12)

    # ================================================================
    # 3) 翻转 + 交换（独立，均触发）
    # ================================================================
    sep("翻转 + 交换")
    c_in = base.copy()
    rng_c = np.random.default_rng(0)
    c_out, info = random_flip_2d(c_in, p_flip=1.0, p_swap=1.0, return_params=True, rng=rng_c)
    print("did_flip, did_swap:", info["did_flip"], info["did_swap"])
    # t0: 左肩(5,x=0.2) → 翻转 x=0.8 且交换到 右肩(6)
    print("t0 right_shoulder(6).x =", c_out[0, 0, 6, 0], "  期望≈ 0.8")
    # t1: 右髋(12,x=0.7) → 翻转 x=0.3 且交换到 左髋(11)
    print("t1 left_hip(11).x      =", c_out[1, 0, 11, 0], "  期望≈ 0.3")
    assert info["did_flip"] and info["did_swap"]
    assert np.isclose(c_out[0, 0, 6, 0], 0.8, atol=1e-6)
    assert np.isclose(c_out[1, 0, 11, 0], 0.3, atol=1e-6)
    # 零点保持为零
    assert np.allclose(c_out[0, 0, 0], [0, 0], atol=1e-12)

    # ================================================================
    # 4) 平移（确定性随机源；clip=False；比较前后差分）
    # ================================================================
    sep("平移")
    d_in = base.copy()
    # 为了可观测，增加两处非零点，避免 mask_zero 过滤掉
    d_in[0, 0, 0] = [0.10, 0.20]
    d_in[1, 0, 1] = [0.90, 0.80]
    video_box = [0.2, 0.1, 0.8, 0.9]  # dx ∈ [-0.2, 0.2], dy ∈ [-0.1, 0.1]
    rng_d = np.random.default_rng(42)

    d_before = d_in.copy()
    d_out, (dx, dy), applied = random_shift_2d(d_in, video_box, p=1.0, clip=False, rng=rng_d, return_params=True)
    print("applied =", applied, " dx,dy =", (dx, dy))
    moved0 = d_out[0, 0, 0] - d_before[0, 0, 0]
    moved1 = d_out[1, 0, 1] - d_before[1, 0, 1]
    print("delta t0,k0 =", moved0, "  期望≈ [dx,dy]")
    print("delta t1,k1 =", moved1, "  期望≈ [dx,dy]")
    assert applied
    assert np.allclose(moved0, [dx, dy], atol=1e-6)
    assert np.allclose(moved1, [dx, dy], atol=1e-6)
    # 零点保持为零
    assert np.allclose(d_out[0, 0, 2], [0, 0], atol=1e-12)

    # ================================================================
    # 5) 缩放（几何×相对幅度的交集；同步更新 bbox）
    # ================================================================
    sep("缩放")
    # 一帧 (P=1,K=4) ；2 个有效点 + 2 个零点（测试 mask_zero）
    k = np.zeros((1, 4, 2), np.float32)
    k[0, 0] = [0.20, 0.30]
    k[0, 1] = [0.80, 0.70]

    vbox = [0.2, 0.3, 0.8, 0.9]  # w0=h0=0.6, 中心(0.5,0.6)
    bbox = np.array([[[0.5, 0.6, 0.6, 0.6, 1.0]]], np.float32)

    rng_s = np.random.default_rng(2025)
    k_in = k.copy()
    b_in = bbox.copy()
    out, s, cur_vbox, applied = random_scale_2d(
        k_in, vbox, scale_amp=0.3, p=1.0, rng=rng_s, mask_zero=True,
        apply_bbox=b_in, return_params=True
    )
    print("applied =", applied, "  s =", s)
    print("cur_vbox =", cur_vbox.tolist())

    # 交集边界校验（只剩 相对幅度 × 几何）
    cx, cy = 0.5 * (vbox[0] + vbox[2]), 0.5 * (vbox[1] + vbox[3])
    def _sd(a, b): return a / b if abs(b) > 1e-6 else np.inf
    s_high_geom = max(1.0, min(_sd(cx, (cx - vbox[0])), _sd(1 - cx, (vbox[2] - cx)),
                               _sd(cy, (cy - vbox[1])), _sd(1 - cy, (vbox[3] - cy))))
    s_lo = max(1 - 0.3, 1e-6)
    s_hi = min(1 + 0.3, s_high_geom)
    assert (not applied) or (s_lo - 1e-6 <= s <= s_hi + 1e-6)

    # 非零点按中心缩放；零点保持为零
    def scale_pt(pt): return np.array([cx + s * (pt[0] - cx), cy + s * (pt[1] - cy)], np.float32)
    if applied:
        np.testing.assert_allclose(out[0, 0], scale_pt([0.20, 0.30]), atol=1e-6)
        np.testing.assert_allclose(out[0, 1], scale_pt([0.80, 0.70]), atol=1e-6)
        np.testing.assert_allclose(out[0, 2], [0, 0], atol=1e-12)
        # bbox 同步缩放
        bx = np.array([cx + s * (bbox[0, 0, 0] - cx), cy + s * (bbox[0, 0, 1] - cy),
                       bbox[0, 0, 2] * s, bbox[0, 0, 3] * s, 1.0], np.float32)
        np.testing.assert_allclose(b_in[0, 0], bx, atol=1e-6)

    print("\n[OK] All 2D aug tests passed.")

    # ================================================================
    # 6) 额外仿射：stretch / shear（仅测试，不必在 pipeline 启用）
    # ================================================================
    print("\n\n[OK] 额外仿射变换测试，暂不加入，仅测试")
    T, P, K = 2, 1, 4
    pts = np.zeros((T, P, K, 2), np.float32)
    pts[0, 0, 0] = [0.2, 0.3]
    pts[0, 0, 1] = [0.8, 0.7]
    pts[1, 0, 2] = [0.4, 0.6]
    vbox = [0.2, 0.3, 0.8, 0.9]
    bbox = np.array([[[0.5, 0.6, 0.6, 0.6, 1.0]]], np.float32)

    # ---- Stretch：各向异性（确定性随机源）----
    p1 = pts.copy()
    b1 = bbox.copy()
    rng = np.random.default_rng(123)
    out, (fx, fy), applied, nb = random_stretch_2d(
        p1, vbox, amp=(0.3, 0.1), p=1.0, rng=rng, mask_zero=True, apply_bbox=b1, return_params=True
    )
    print("[STRETCH] applied=", applied, " fx,fy=", (fx, fy))
    if applied:
        cx, cy = 0.5, 0.6
        exp0 = np.array([cx + fx * (0.2 - cx), cy + fy * (0.3 - cy)], np.float32)
        np.testing.assert_allclose(out[0, 0, 0], exp0, atol=1e-6)
        # 零点仍为零
        np.testing.assert_allclose(out[0, 0, 3], [0, 0], atol=1e-12)
        # bbox (cx,cy) 不变
        np.testing.assert_allclose(b1[0, 0, :2], [0.5, 0.6], atol=1e-6)

    # ---- Shear：x/y 剪切（确定性随机源）----
    p2 = pts.copy()
    b2 = bbox.copy()
    rng = np.random.default_rng(7)
    out, (shx, shy), applied, nb = random_shear_2d(
        p2, vbox, shx_amp=0.2, shy_amp=0.1, p=1.0, rng=rng, mask_zero=True, apply_bbox=b2, return_params=True
    )
    print("[SHEAR] applied=", applied, " shx,shy=", (shx, shy))
    if applied:
        x, y = 0.2, 0.3
        cx, cy = 0.5, 0.6
        exp = np.array([x + shx * (y - cy), y + shy * (x - cx)], np.float32)
        np.testing.assert_allclose(out[0, 0, 0], exp, atol=1e-6)
        # bbox 更新为四角仿射后的 AABB（这里只校验在 [0,1]）
        assert np.all((b2[0, 0, 0:4] >= 0) & (b2[0, 0, 0:4] <= 1))

    print("\nstretch/shear quick tests passed.")
