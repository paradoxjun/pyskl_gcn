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
  3D关键点数据增强代码。
"""


import numpy as np
from typing import Optional, Union, Sequence


def random_rotate_3d(
        kpts: np.ndarray,
        rx: Union[float, tuple, list] = 0.0, ry: Union[float, tuple, list] = 0.0, rz: Union[float, tuple, list] = 0.0,
        *,
        degrees: bool = True,
        center: Union[str, np.ndarray] = 'origin',  # 'origin' 或 np.ndarray(3,)
        mask_zero: bool = True,     # 为 True 时，只旋转非零点，零点保持不变；所有点全零会抛错
        order: str = 'zyx',         # 复合顺序：'zyx' => R = Rz @ Ry @ Rx
        rng: Optional[np.random.Generator] = None,
        return_params: bool = False,    # True: 返回 (kpts_rot, angles_deg(np.array[3]), R(np.array[3,3]))
):
    """
    随机旋转 3D 关键点（返回新数组，不改输入）:
      - kpts: 任意形状，最后一维必须为 3（..., 3）
      - rx/ry/rz: 单值 s -> [-|s|, +|s|]；二元序列 (lo, hi) -> [lo, hi]
    """
    if kpts.ndim < 1 or kpts.shape[-1] != 3:
        raise ValueError(f"ERROR: last dim must be 3, got shape={kpts.shape}")

    def _to_range(val: Optional[Union[int, float, Sequence[float]]]):
        if val is None: return 0.0, 0.0
        if isinstance(val, (int, float)):
            s = float(val)
            return -abs(s), abs(s)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            a, b = float(val[0]), float(val[1])
            return (a, b) if a <= b else (b, a)
        raise ValueError(f"ERROR: invalid range: {val}")

    rx_rng, ry_rng, rz_rng = _to_range(rx), _to_range(ry), _to_range(rz)
    rng = np.random.default_rng() if rng is None else rng

    # 采样角度（度）
    ax, ay, az = rng.uniform(*rx_rng), rng.uniform(*ry_rng), rng.uniform(*rz_rng)
    # → 弧度
    ax_r, ay_r, az_r = (np.deg2rad(ax), np.deg2rad(ay), np.deg2rad(az)) if degrees else (ax, ay, az)

    # 标准右手坐标系旋转矩阵（列向量约定）；行向量使用 pts @ R.T
    cx, sx = np.cos(ax_r), np.sin(ax_r)
    cy, sy = np.cos(ay_r), np.sin(ay_r)
    cz, sz = np.cos(az_r), np.sin(az_r)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]], dtype=np.float32)

    maps = {'x': Rx, 'y': Ry, 'z': Rz}
    R = np.eye(3, dtype=np.float32)
    for axis in order.lower():
        if axis not in maps:
            raise ValueError(f"ERROR: invalid axis in order: {axis}")
        R = maps[axis] @ R

    # 旋转中心
    if isinstance(center, str):
        if center != 'origin':
            raise ValueError("ERROR: center must be 'origin' or a 3D numpy array")
        ctr = np.zeros(3, dtype=np.float32)
    else:
        ctr = np.asarray(center, dtype=np.float32)
        if ctr.shape != (3,):
            raise ValueError(f"ERROR: center must be shape (3,), got {ctr.shape}")

    out = kpts.reshape(-1, 3)  # 视 kpts 连续性而定：通常是 view
    if mask_zero:
        m = np.any(np.abs(out) > 0, axis=1)
        if not np.any(m):
            raise ValueError("ERROR: all points are zero; nothing to rotate")
        pts = out[m] - ctr  # 只为选中点分配小临时，不是整数组 copy
        out[m] = (pts @ R.T) + ctr
    else:
        pts = out - ctr
        out[:] = (pts @ R.T) + ctr

    kpts_rot = out.reshape(kpts.shape)  # 通常依然指向同一块内存
    if return_params:
        return kpts_rot, np.array([ax, ay, az], dtype=np.float32), R
    return kpts_rot


def random_scale_3d(
        kpts: np.ndarray,
        scale: Union[float, int, tuple, list],
        p_iso: float = 1.0,
        *,
        center: Union[str, np.ndarray] = "origin",  # 'origin' 或 np.ndarray(3,)
        mask_zero: bool = True,  # True: 仅对非零点生效；全零则不报错、直接返回
        rng: np.random.Generator = None,
        return_params: bool = False  # True 时返回 (kpts, factors[np.float32(3,)])
):
    """
    随机缩放 3D 关键点（原地修改，不整体拷贝；返回 kpts 引用本身）

    参数
    ----
    kpts : np.ndarray
        任意形状，但最后一维必须为 3，例如 (T,P,V,3) / (N,3) / (...,3)。
    scale :
        缩放因子范围的“规格”说明（与 random_rotate_3d 的写法保持直觉一致）：
        - float s       → 各轴“±幅度”，在 [1-s, 1+s] 采样
        - (lo, hi)      → 因子在 [lo, hi] 采样
        - (sx, sy, sz)  → 三轴各自“±幅度”，在 [1-sx, 1+sx] × [1-sy, 1+sy] × [1-sz, 1+sz]
        - ((lx,hx),(ly,hy),(lz,hz)) → 三轴各自因子范围
        注：若等比例缩放生效（见 p_iso），则会从三轴范围的“交集”里采单一因子； 若三轴范围无交集，则退化使用第一轴的范围。
    p_iso : float
        等比例缩放的概率，默认 1.0（总是等比）。若触发失败（随机>p_iso），则各轴独立缩放。

    关键字参数
    ----------
    center : 'origin' 或 np.ndarray(3,) 缩放中心。默认为原点；传 3 维向量时，先减中心再缩放、最后加回中心。
    mask_zero : bool True 时仅对非零点生效（关节为 [0,0,0] 的保持 0，不会被“点亮”）。 如果所有点均为 0，不抛错，直接返回。
    rng : np.random.Generator 可传入确定性随机源；默认用 np.random.default_rng()。
    return_params : bool True 时额外返回 (factors)，形状 (3,)，等比时三个分量相同。

    返回
    ----
    kpts : np.ndarray 与传入同一引用（原地修改后的结果）。
    factors : np.ndarray(3,), dtype=float32  [仅在 return_params=True 时返回] 实际使用的缩放因子（x,y,z）。等比时三者相等。
    """
    # ---- 检查输入 ----
    if kpts.ndim < 1 or kpts.shape[-1] != 3:
        raise ValueError(f"ERROR: last dim must be 3, got shape={kpts.shape}")
    if not (0.0 <= float(p_iso) <= 1.0):
        raise ValueError(f"ERROR: p_iso must be in [0,1], got {p_iso}")

    rng = np.random.default_rng() if rng is None else rng

    # ---- 把 scale 统一解析为每轴 (lo, hi) ----
    def _to_range_pair(v):
        # 标量 → 幅度；二元 → (lo,hi)
        if isinstance(v, (int, float)):
            s = float(v)
            return 1.0 - abs(s), 1.0 + abs(s)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            a, b = float(v[0]), float(v[1])
            return (a, b) if a <= b else (b, a)
        raise ValueError(f"ERROR: invalid range spec: {v}")

    if isinstance(scale, (int, float)) or (isinstance(scale, (list, tuple)) and len(scale) == 2):
        # 单一规格 → 三轴共用
        rx = ry = rz = _to_range_pair(scale)
    elif isinstance(scale, (list, tuple)) and len(scale) == 3:
        # 三轴规格（支持 混合：标量/区间）
        rx = _to_range_pair(scale[0])
        ry = _to_range_pair(scale[1])
        rz = _to_range_pair(scale[2])
    else:
        raise ValueError(f"ERROR: invalid scale format: {scale}")

    # ---- 采样缩放因子 ----
    isotropic = (rng.random() < p_iso)

    if isotropic:
        # 取三轴范围交集，保证单一因子对三轴都合法
        lo_all = max(rx[0], ry[0], rz[0])
        hi_all = min(rx[1], ry[1], rz[1])
        if lo_all > hi_all:
            # 无交集：退化用 x 轴范围（你也可以改成用平均或其它策略）
            lo_all, hi_all = rx
        f = rng.uniform(lo_all, hi_all)
        factors = np.array([f, f, f], dtype=np.float32)
    else:
        fx = rng.uniform(*rx)
        fy = rng.uniform(*ry)
        fz = rng.uniform(*rz)
        factors = np.array([fx, fy, fz], dtype=np.float32)

    # ---- 缩放中心 ----
    if isinstance(center, str):
        if center != "origin":
            raise ValueError("ERROR: center must be 'origin' or a (3,) ndarray")
        ctr = None  # 原点
    else:
        ctr = np.asarray(center, dtype=np.float32)
        if ctr.shape != (3,):
            raise ValueError(f"ERROR: center must be shape (3,), got {ctr.shape}")

    # ---- 原地缩放 ----
    # 仅处理非零点（mask_zero=True），避免把无效占位点“点亮”
    if mask_zero:
        m = np.any(np.abs(kpts) > 0, axis=-1)
        if not np.any(m):
            return (kpts, factors) if return_params else kpts
    else:
        # 全部点参与
        m = slice(None)

    if ctr is None:
        # 绕原点：直接按轴乘因子
        # 选择性地只对非零点生效
        kpts[..., 0][m] *= factors[0]
        kpts[..., 1][m] *= factors[1]
        kpts[..., 2][m] *= factors[2]
    else:
        # 绕自定义中心：减中心 → 乘因子 → 加回中心
        # 只对非零点子集进行计算以减少开销
        flat = kpts.reshape(-1, 3)
        mm = m.reshape(-1) if isinstance(m, np.ndarray) else m
        sub = flat[mm]  # 注意：右值是拷贝；但只处理子集，避免全量复制
        sub -= ctr
        sub[:, 0] *= factors[0]
        sub[:, 1] *= factors[1]
        sub[:, 2] *= factors[2]
        sub += ctr
        flat[mm] = sub  # 写回

    return (kpts, factors) if return_params else kpts


def flip_x_3d(
        kpts: np.ndarray,
        *,
        center: Union[str, np.ndarray] ='origin',  # 'origin' 或 np.ndarray(3,)
        mask_zero: bool = True,  # 仅翻转非零点；零点保持 0
        return_params: bool = False  # True: 返回 (kpts, linear_sign(np.array[3]), center_used(np.array[3]))
):
    """
    3D 关键点关于 YOZ 面的翻转（仅 x 取镜像；y、z 不变），原地修改。
    - kpts: 任意形状，但最后一维必须是 3（..., 3）
    - center:
        'origin'       -> 围绕 x=0 反射（x' = -x）
        np.ndarray(3,) -> 围绕 x=center[0] 反射（x' = 2*cx - x）
    - mask_zero: True 则仅作用于“非全零”的点（占位零点保持 0）
    - return_params:
        True  返回 (kpts, [-1, 1, 1], center_used)
        False 仅返回 kpts
    """
    if kpts.ndim < 1 or kpts.shape[-1] != 3:
        raise ValueError(f"ERROR: last dim must be 3, got shape={kpts.shape}")

    # 翻转中心
    if isinstance(center, str):
        if center != 'origin':
            raise ValueError("ERROR: center must be 'origin' or a 3D numpy array")
        ctr = np.zeros(3, dtype=kpts.dtype)
    else:
        ctr = np.asarray(center, dtype=kpts.dtype)
        if ctr.shape != (3,):
            raise ValueError(f"center must be shape (3,), got {ctr.shape}")

    # 原地翻转：x' = 2*cx - x；y/z 不变
    if mask_zero:
        m = np.any(np.abs(kpts) > 0, axis=-1)  # (...,)
        if not np.any(m):
            raise ValueError("ERROR: all points are zero; nothing to flip")
        x = kpts[..., 0]
        x[m] = 2.0 * ctr[0] - x[m]
    else:
        x = kpts[..., 0]
        x[...] = 2.0 * ctr[0] - x

    if return_params:
        return kpts, np.array([-1.0, 1.0, 1.0], dtype=kpts.dtype), ctr
    return kpts


# 与 pyskl 的 Spatial_Flip 映射一致（'ntu'）
_NTU25_LR_INDEX = np.array(
    [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22],
    dtype=np.int64
)


def mirror_lr_3d(
    kpts: np.ndarray,                     # 形状任意，但最后两维应为 (V, 3)，V=25（NTU骨架）
    *,
    return_params: bool = False           # True: 返回 (kpts, index_map[np.int64(25,)])
):
    """
    语义镜像（NTU-25）：左右关节索引交换。

    - 输入 kpts 的最后两维应为 (V, 3)，其中 V 必须为 25（NTU骨架）。
    - 交换在 V 维进行：kpts[..., v, :]  ->  kpts[..., index[v], :]
    """
    if kpts.ndim < 2 or kpts.shape[-1] != 3:
        raise ValueError(f"ERROR: last dim must be 3, got shape={kpts.shape}")
    V = kpts.shape[-2]
    if V != _NTU25_LR_INDEX.shape[0]:
        raise ValueError(f"ERROR: expected V=25 for NTU, got V={V}")

    # 仅索引重排；右值先构建临时，再安全原地写回
    kpts[...] = kpts[..., _NTU25_LR_INDEX, :]

    return (kpts, _NTU25_LR_INDEX.copy()) if return_params else kpts



if __name__ == '__main__':
    # 三个 3D 点，shape = (1, 3, 3) 也可以是 (3,3)；这里只用 (3,3)
    pts = np.array([
        [1.0, 0.0, 1.0],  # p1
        [0.0, 1.0, 0.0],  # p2
        [0.0, 0.0, 1.0],  # p3
    ], dtype=np.float32)

    # 例 1：仅绕 Z 轴 +30°（用区间 [30,30] 固定角度，便于检查）
    rot_z30 = random_rotate_3d(pts, rz=(30, 30), rx=0, ry=0, center='origin', mask_zero=False, return_params=False)
    print("Z+30°:\n", rot_z30)
    # 期望近似：
    # p1 -> (0.8660,  0.5000, 1.0000)
    # p2 -> (-0.5000, 0.8660, 0.0000)
    # p3 -> (0.0000,  0.0000, 1.0000)

    # 例 2：仅绕 Y 轴 +45°
    rot_y45 = random_rotate_3d(pts, ry=(45, 45), rx=0, rz=0, center='origin', mask_zero=False, return_params=False)
    print("Y+45°:\n", rot_y45)
    # 期望近似：
    # p1 -> (1.4142, 0.0000, 0.0000)
    # p2 -> (0.0000, 1.0000, 0.0000)
    # p3 -> (0.7071, 0.0000, 0.7071)

    # 例 3：三轴同时随机（±10°），固定随机种子
    rng = np.random.default_rng(123)
    rot_rand, ang_deg, R = random_rotate_3d(pts, rx=10, ry=10, rz=10, center='origin', rng=rng, return_params=True)
    print("random angles(deg) =", ang_deg, "\nR=\n", R, "\nrot:\n", rot_rand)

    # 三个 3D 点
    pts0 = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    # 用 copy() 明确传入“可任意修改”的副本
    print("原始：\n", pts0)

    # 例 1：等比缩放（±10%），p_iso=1.0（总是等比）
    rng = np.random.default_rng(42)
    pts1 = pts0.copy()
    pts1, fac1 = random_scale_3d(pts1, scale=0.10, p_iso=1.0, rng=rng, return_params=True)
    print("\n例1：等比 ±10%，因子 =", fac1[0])
    print(pts1)

    # 例 2：各轴独立范围，p_iso=0.0（必定各轴不同）
    #       x∈[0.5,1.0], y∈[1.0,1.5], z∈[0.8,1.2]
    rng = np.random.default_rng(7)
    pts2 = pts0.copy()
    pts2, fac2 = random_scale_3d(pts2, scale=((0.5, 1.0), (1.0, 1.5), (0.8, 1.2)), p_iso=0.0, rng=rng,
                                 return_params=True)
    print("\n例2：各轴独立范围，因子 =", fac2.tolist())
    print(pts2)

    # 例 3：绕自定义中心缩放（中心=[1,1,1]），等比 ±20%
    rng = np.random.default_rng(123)
    pts3 = pts0.copy()
    pts3, fac3 = random_scale_3d(pts3, scale=0.20, p_iso=1.0, center=np.array([1.0, 1.0, 1.0], np.float32),
                                 rng=rng, return_params=True)
    print("\n例3：等比 ±20%，中心=[1,1,1]，因子 =", fac3[0])
    print(pts3)

    pts = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],  # 占位零点（mask_zero=True 时应保持为零）
        [-4.0, 5.0, -6.0],
    ], dtype=np.float32)

    a = pts.copy()
    flip_x_3d(a, center='origin', mask_zero=True)
    print("flip about x=0 (mask_zero=True):\n", a)
    # 期望：
    # [[-1.  2.  3.]
    #  [ 0.  0.  0.]
    #  [ 4.  5. -6.]]

    b = pts.copy()
    flip_x_3d(b, center=np.array([2.0, 0.0, 0.0], np.float32), mask_zero=False)
    print("flip about x=2 (mask_zero=False):\n", b)
    # 期望：
    # [[ 3.  2.  3.]
    #  [ 2.  0.  0.]
    #  [ 8.  5. -6.]]

    # ====== 语义镜像：单元测试（NTU-25） ======
    print("\n[TEST] mirror_lr_3d (NTU-25, index-only swap)")

    T, P, V = 2, 1, 25
    xyz = np.zeros((T, P, V, 3), dtype=np.float32)

    # 帧0：在关节8放点 (+1, 0, 0) → 镜像后应到 关节4，且坐标不取负（保持 +1）
    xyz[0, 0, 8, 0] = 1.0
    # 帧1：在关节4放点 (+2, 0, 0) → 镜像后应到 关节8，且坐标保持 +2
    xyz[1, 0, 4, 0] = 2.0

    xyz_m = xyz.copy()
    out, idx = mirror_lr_3d(xyz_m, return_params=True)

    print("before mirror: x@t0,v8 =", xyz[0, 0, 8, 0], " | x@t1,v4 =", xyz[1, 0, 4, 0])
    print("after  mirror: x@t0,v4 =", out[0, 0, 4, 0], " | x@t1,v8 =", out[1, 0, 8, 0])
    print("index map     =", idx.tolist())

    # 断言：交换成功，坐标不取负
    assert np.isclose(out[0, 0, 4, 0], 1.0), "t=0 后 v4.x 应保持 +1.0（仅索引交换，不翻坐标）"
    assert np.isclose(out[1, 0, 8, 0], 2.0), "t=1 后 v8.x 应保持 +2.0（仅索引交换，不翻坐标）"

    # 再镜像一次应恢复原状
    xyz_mm = out.copy()
    mirror_lr_3d(xyz_mm)
    assert np.allclose(xyz_mm, xyz, atol=1e-6), "二次镜像后应恢复原状（置换是自反的）"

    print("[OK] mirror_lr_3d tests passed.")
