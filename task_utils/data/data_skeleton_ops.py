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
  生成多流数据，仅实现了4-stream，针对近几年论文中 6-stream 还缺两种结果。
  坑：对于 NTU 上的 3D 关键点没问题，但是 2D 关键点存在缺失点需要额外置0.举个例子：
  计算速度，当前点为0，不置0，那么和下一个点的速度就变成原值的1/2。
"""


from __future__ import annotations
import numpy as np
from typing import Iterable, Literal

DatasetName = Literal["nturgb+d", "openpose", "coco", "handmp"]

# ---------------- 骨架连边(与 pyskl 一致) ----------------
PAIRS = {
    "nturgb+d": (
        (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
        (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
        (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)
    ),
    "openpose": (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
        (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    ),
    "coco": (
        (0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
        (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14)
    ),
    "handmp": (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9),
        (11, 10), (12, 11), (13, 0), (14, 13), (15, 14), (16, 15), (17, 0), (18, 17), (19, 18), (20, 19)
    ),
}


# === 关节有效性判定 ===
def _joint_valid_mask(joints: np.ndarray, dataset: DatasetName, eps: float = 1e-6) -> np.ndarray:
    """
    返回 (T,P,V) 的 bool，有效性规则：
    - NTU 3D（dataset == "nturgb+d"）：任一(x,y,z)非零为有效
    - 2D+score（C>=3 且 dataset in {"openpose","coco","handmp"}）：score>0 为有效
    - 其他：坐标是否全0
    """
    C = joints.shape[-1]
    if dataset == "nturgb+d":
        return np.abs(joints[..., :3]).sum(axis=-1) > eps
    elif (C >= 3) and (dataset in ("openpose", "coco", "handmp")):
        return joints[..., 2] > eps
    else:
        return np.abs(joints[..., :min(C, 3)]).sum(axis=-1) > eps


# ---------------- 基元：J -> B ----------------
def joints_to_bones_TPVC(
        joints: np.ndarray,  # (T, P, V, C)
        dataset: DatasetName = "nturgb+d",
) -> np.ndarray:
    assert joints.ndim == 4, f"expect (T,P,V,C), got {joints.shape}"
    T, P, V, C = joints.shape
    out = np.zeros_like(joints, dtype=np.float32)
    pairs = PAIRS[dataset]

    valid = _joint_valid_mask(joints, dataset)  # (T,P,V)

    # 维度选择：NTU 3D 用 xyz；2D+score 仅对 xy 做差；其余取前三维
    if dataset == "nturgb+d":
        D = min(3, C)
    elif (C == 3) and (dataset in ("openpose", "coco", "handmp")):
        D = 2
    else:
        D = min(C, 3)

    if (C == 3) and (dataset in ("openpose", "coco", "handmp")):
        score = joints[..., 2]  # (T,P,V)
        for v1, v2 in pairs:
            xy = joints[..., v1, :2] - joints[..., v2, :2]
            m = (valid[..., v1] & valid[..., v2])[..., None]  # (T,P,1)
            out[..., v1, :2] = xy * m.astype(xy.dtype)
            # score 仅在两端有效时取均值，否则为0
            out[..., v1, 2] = ((score[..., v1] + score[..., v2]) * 0.5) * (m[..., 0].astype(score.dtype))
    else:
        for v1, v2 in pairs:
            diff = joints[..., v1, :D] - joints[..., v2, :D]
            m = (valid[..., v1] & valid[..., v2])[..., None]  # (T,P,1)
            out[..., v1, :D] = diff * m.astype(diff.dtype)
    return out


# ---------------- 基元：任意 X -> Motion(X) ----------------
def to_motion_TPVC(
        data: np.ndarray,  # (T, P, V, C)
        dataset: DatasetName = "nturgb+d",
) -> np.ndarray:
    assert data.ndim == 4, f"expect (T,P,V,C), got {data.shape}"
    T, P, V, C = data.shape
    mot = np.zeros_like(data, dtype=np.float32)
    if T <= 1:
        return mot

    # 原始差分
    mot[:-1] = data[1:] - data[:-1]

    # 严格 AND：仅当相邻两帧都有效时保留该步位移
    valid = _joint_valid_mask(data, dataset)  # (T,P,V)
    v01 = valid[:-1] & valid[1:]  # (T-1,P,V)

    if (C == 3) and (dataset in ("openpose", "coco", "handmp")):
        # 2D+score：xy 仅在相邻都有效时保留；score 取相邻均值（无效置0）
        mot[:-1, :, :, :2] *= v01[..., None].astype(mot.dtype)
        s0 = data[:-1, :, :, 2];
        s1 = data[1:, :, :, 2]
        mot[:-1, :, :, 2] = ((s0 + s1) * 0.5) * v01.astype(mot.dtype)
    else:
        D = 3 if dataset == "nturgb+d" else min(C, 3)
        mot[:-1, :, :, :D] *= v01[..., None].astype(mot.dtype)

    return mot


# ---------------- 生成并合并：j/b/jm/bm ----------------
def gen_and_merge_feats_TPVC(
        joints_TPVC: np.ndarray,  # (T,P,V,3)
        dataset: DatasetName = "nturgb+d",
        feats: Iterable[str] = ("j", "b", "jm", "bm"),  # 任选子集
        concat_axis: int = -1,  # 最后一维拼接(通道)
) -> np.ndarray:
    """
    feats 定义:
      j  : 关节
      b  : 骨向量   = J2B(j)
      jm : 关节位移 = diff(j)
      bm : 骨位移   = diff(b)
    返回: 与 joints_TPVC 同形状，但通道在 concat_axis 方向堆叠（通常 = -1）
    """
    assert joints_TPVC.ndim == 4
    fdict = {}
    if "j" in feats:
        fdict["j"] = joints_TPVC.astype(np.float32, copy=False)
    if "b" in feats:
        fdict["b"] = joints_to_bones_TPVC(joints_TPVC, dataset)
    if "jm" in feats:
        base = fdict.get("j", joints_TPVC)
        fdict["jm"] = to_motion_TPVC(base, dataset)
    if "bm" in feats:
        b = fdict.get("b", joints_to_bones_TPVC(joints_TPVC, dataset))
        fdict["bm"] = to_motion_TPVC(b, dataset)

    arrs = [fdict[k] for k in feats if k in fdict]
    return np.concatenate(arrs, axis=concat_axis).astype(np.float32, copy=False)
