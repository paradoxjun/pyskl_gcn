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
    ntu60_3d.pkl  ▶  npz  (CS / CV 四分)  —— 预归一化版本
---------------------------------------------------
把 3D 关键点在『生成 npz』阶段就做成 PYSKL 的 PreNormalize3D 效果：
  1) 选“主演员”（有效帧更多的那个人），必要时人 0/1 互换；
  2) 时间轴 squeeze：仅保留主演员“非全零”的帧索引；
  3) 以“首帧主干中心”为原点做平移；
  4) 脊柱对齐 Z 轴，再用双肩对齐 X 轴（两次 3x3 旋转）；
  5) 每帧每人计算球心与半径 [xc,yc,zc,r]；
保持你原先的数据结构不变：
  - data        : (F_new, P(<=2), V=25, 3)
  - xyz_sphere  : (F_new, P, 4)
  - xyz_vd      : (3,)  —— 这里存我们实际“施加的平移向量”（= - main_center）
  - view, label, valid_frame
可选新增调试键（便于追踪变换）：
  - body_center : (3,)   首帧主干中心（未取负号）
  - rot_z, rot_x: (3,3)  两次旋转矩阵（对齐脊柱、对齐双肩）
  保存数据具体说明：
    data: [F_new, P, V, 3]: float32 (x,y,z)，是对齐后的关键点序列（首帧主干中心平移+脊柱Z轴对齐+双肩X轴对齐+主演员选择+时间轴压缩）
    xyz_sphere: [F_new, P, 4]: float32 [xc, yc, zc, r]，是逐帧逐人的包围球中心与半径（与 data 同一对齐坐标系）
    xyz_vd: [N, 3]: float32 [dx, dy, dz]，是把原始坐标平移到以首帧主干中心为原点时使用的平移量（约等于 -body_center）
    view: [N]: uint8 ∈{1,2,3}，是样本拍摄视角的机位编号
    label: [N]: int64，是真实动作类别ID
    valid_frame: [N]: int32，是该样本压缩后有效帧数 F_new
    modalities: [1]: str 'xyz'，是该数据仅包含 xyz 关键点模态的标记
    body_center: [3]: float32 (x0, y0, z0)，是首帧主演员的主干中心坐标（全局平移参考）
    rot_z: [3, 3]: float32 旋转矩阵，是将首帧脊柱对齐到 Z 轴时使用的旋转
    rot_x: [3, 3]: float32 旋转矩阵，是在 rot_z 之后将首帧双肩连线对齐到 X 轴时使用的旋转
    -------------------------------------------------------------------------------
    xyz_sphere: 是逐帧、逐人的球信息，形状 (F_new, P, 4)，每个元素是 [xc, yc, zc, r]，
                是在对齐(平移+两次旋转)之后用该帧该人的有效关节点算出来的帧级中心和半径。
                它会随时间变化（人物动作导致分布变动），而且每个人各不相同。
    body_center: 是一个全局的单个三维向量 (3,)，仅指首帧的主干中心（我们用它做平移的参考）。
                 它不随时间变化，只用来记录“我们对整段序列施加的平移参考点”。
"""


import re
import pickle as pkl
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------- 路径 ----------
PKL_PATH = Path(
    r'G:/datasets/cls_video/nturgbd_skeletons_s001_to_s017/pkl_mmaction/ntu60_3d.pkl')  # ← pyskl中下载的pkl文件
OUT_DIR = Path(
    r'G:/datasets/cls_video/nturgbd_skeletons_s001_to_s017/pkl_mmaction/npz_raw_xyz_ntu60_pyskl')  # ← npz文件保存目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 划分表 ----------
CS_TRAIN_SUBJ = {
    'P001', 'P002', 'P004', 'P005', 'P008', 'P009', 'P013', 'P014', 'P015', 'P016',
    'P017', 'P018', 'P019', 'P025', 'P027', 'P028', 'P031', 'P034', 'P035', 'P038'}
CS_TEST_SUBJ = {
    'P003', 'P006', 'P007', 'P010', 'P011', 'P012', 'P020', 'P021', 'P022', 'P023',
    'P024', 'P026', 'P029', 'P030', 'P032', 'P033', 'P036', 'P037', 'P039', 'P040'}
CV_TRAIN_CAM = {'C002', 'C003'}
CV_TEST_CAM = {'C001'}

# ---------- 正则 ----------
RE_SUBJ = re.compile(r'(P\d{3})')
RE_CAM = re.compile(r'(C\d{3})')


# ---------- 桶 ----------
def empty_bucket():
    return {k: [] for k in
            ('data', 'xyz_s', 'xyz_vd', 'view', 'label', 'valid_frame', 'body_center', 'rot_z', 'rot_x')}


cs = {'train': empty_bucket(), 'test': empty_bucket()}
cv = {'train': empty_bucket(), 'test': empty_bucket()}


# ---------- 工具：球心/半径 ----------
def sphere_from_xyz(xyz_frame_person: np.ndarray) -> np.ndarray:
    """
    输入：某一帧、某一人的所有关节点 (V,3)
    输出：球心与半径 [xc,yc,zc,r]，若此人全零则返回全 0
    """
    v = ~np.all(np.isclose(xyz_frame_person, 0), axis=1)
    if not v.any():
        return np.zeros(4, np.float32)
    pts = xyz_frame_person[v]
    ctr = pts.mean(axis=0)
    r = np.linalg.norm(pts - ctr, axis=1).max()
    return np.append(ctr, r).astype(np.float32)


# ---------- 工具：向量/旋转 ----------
def _unit(vec: np.ndarray) -> np.ndarray:
    """单位向量，零向量直接返回零（避免除零）。"""
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-6 else np.zeros_like(vec)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """两向量夹角（弧度），任一为零时返回 0。"""
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return 0.0
    v1_u, v2_u = _unit(v1), _unit(v2)
    # clip 防数值误差
    return float(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def _rotmat(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    绕 axis 旋转 theta（Rodrigues），任何一个为 0 时返回 I。
    返回 (3,3)
    """
    if np.linalg.norm(axis) < 1e-6 or abs(theta) < 1e-6:
        return np.eye(3, dtype=np.float32)
    axis = _unit(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    M = np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ], dtype=np.float32)
    return M


# ---------- 主演员 + 帧 squeeze ----------
def pick_main_and_squeeze(kp: np.ndarray) -> np.ndarray:
    """
    输入：kp (M<=2, T, V, 3)
    规则：
      - 统计每个人的“有效帧数”（该帧是否存在任意非零关节）；
      - 选有效帧更多者为主演员；若 M=1 就它；
      - 时间轴 squeeze：仅保留“主演员非零”的帧索引（所有人都统一到这组帧上）；
      - 若主演员原本是 idx=1，则把两人对调为 [主演员, 副演员] 顺序。
    返回：新 kp (M<=2, T_new, V, 3)
    """
    M, T, V, _ = kp.shape
    assert M in (1, 2)

    def valid_idx(person: int) -> np.ndarray:
        # 这一人的每一帧是否“不是全零”，得到帧索引列表
        mask = ~np.all(np.isclose(kp[person], 0), axis=(1, 2))
        return np.where(mask)[0]

    idx0 = valid_idx(0)
    if M == 2:
        idx1 = valid_idx(1)
        # 谁的有效帧更多，谁做主演员
        if len(idx0) < len(idx1):
            # 以人1的有效帧为基准，先取帧，再把人顺序换成 [1,0]
            base = idx1
            kp_new = kp[:, base]  # (M, T_new, V, 3)
            kp_new = kp_new[[1, 0]]  # 主演员排前
            return kp_new.astype(np.float32)
        else:
            base = idx0
            kp_new = kp[:, base]
            return kp_new.astype(np.float32)
    else:
        kp_new = kp[:, idx0] if len(idx0) > 0 else kp[:, :0]  # 可能全零
        return kp_new.astype(np.float32)


# ---------- 对齐：中心平移 + 脊柱/Z + 双肩/X ----------
def prenorm_align(kp: np.ndarray,
                  zaxis=(0, 1),
                  xaxis=(8, 4)):
    """
    输入：kp (M<=2, T_new, V, 3)，已 squeeze 并且主演员在第 0 人。
    处理：
      - 平移：以首帧主干中心为原点（NTU25 用关节 id=1；其它布局可改为 -1）
      - 旋转 1：把首帧脊柱向量对齐到 [0,0,1]（Z 轴）
      - 旋转 2：把首帧双肩向量对齐到 [1,0,0]（X 轴）
    返回：
      kp_out     : 对齐后的关键点 (M,T,V,3)
      main_center: (3,)  首帧主干中心（未取负）
      Rz, Rx     : (3,3) 两次旋转矩阵
    """
    kp = kp.copy()
    M, T, V, C = kp.shape
    if T == 0 or np.all(np.isclose(kp, 0)):
        return kp, np.zeros(3, np.float32), np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    # 1) 以首帧主干中心为原点（NTU25: 关节 1）
    main_center = kp[0, 0, 1].copy() if V == 25 else kp[0, 0, -1].copy()
    mask = ((kp != 0).sum(axis=-1) > 0)[..., None]  # (M,T,V,1) 非零才平移/旋转
    kp = (kp - main_center) * mask

    # 2) 脊柱对齐 Z 轴
    joint_bottom = kp[0, 0, zaxis[0]]
    joint_top = kp[0, 0, zaxis[1]]
    spine_vec = joint_top - joint_bottom
    axis_z = np.cross(spine_vec, np.array([0, 0, 1], dtype=np.float32))
    ang_z = _angle_between(spine_vec, np.array([0, 0, 1], dtype=np.float32))
    Rz = _rotmat(axis_z, ang_z)
    kp = np.einsum('mtvc,cd->mtvd', kp, Rz)  # 右乘旋转

    # 3) 双肩对齐 X 轴
    r_sh = kp[0, 0, xaxis[0]]
    l_sh = kp[0, 0, xaxis[1]]
    shoulder_vec = r_sh - l_sh
    axis_x = np.cross(shoulder_vec, np.array([1, 0, 0], dtype=np.float32))
    ang_x = _angle_between(shoulder_vec, np.array([1, 0, 0], dtype=np.float32))
    Rx = _rotmat(axis_x, ang_x)
    kp = np.einsum('mtvc,cd->mtvd', kp, Rx)

    return kp.astype(np.float32), main_center.astype(np.float32), Rz, Rx


# ---------- 读取 pickle ----------
with PKL_PATH.open('rb') as f:
    annos = pkl.load(f)['annotations']

for ann in tqdm(annos, desc='convert'):
    subj = RE_SUBJ.search(ann['frame_dir']).group(1)  # Pxxx
    cam = RE_CAM.search(ann['frame_dir']).group(1)  # Cxxx
    view = np.uint8(int(cam[-1]))  # 1 / 2 / 3
    label = np.int64(ann['label'])

    kp = ann['keypoint'].astype(np.float32)  # (M<=2, T, V=25, 3)
    M, T, V, _ = kp.shape

    # —— 1) 选主演员 + squeeze 时间轴 —— #
    kp_sq = pick_main_and_squeeze(kp)  # (M<=2, T_new, V, 3)
    M2, Tn, _, _ = kp_sq.shape

    # 若整段全零（极罕见），就按原样记录（避免下游出错）
    if Tn == 0 or np.all(np.isclose(kp_sq, 0)):
        F_new = 0
        # 占位：空数组
        data_out = np.zeros((0, M, V, 3), dtype=np.float32)
        spheres = np.zeros((0, M, 4), dtype=np.float32)
        shift_vec = np.zeros(3, np.float32)
        Rz = np.eye(3, dtype=np.float32)
        Rx = np.eye(3, dtype=np.float32)
        body_center = np.zeros(3, np.float32)
    else:
        # —— 2) 预归一化：中心平移 + 两次对齐旋转 —— #
        kp_aligned, body_center, Rz, Rx = prenorm_align(kp_sq)  # (M2,Tn,V,3)

        # —— 3) 每帧/每人球心半径 —— #
        spheres = np.zeros((Tn, M2, 4), dtype=np.float32)
        for f in range(Tn):
            for m in range(M2):
                spheres[f, m] = sphere_from_xyz(kp_aligned[m, f])

        # —— 4) 整理输出为 (F_new, P, V, 3) —— #
        data_out = np.transpose(kp_aligned, (1, 0, 2, 3)).astype(np.float32)  # (Tn, M2, V, 3)
        F_new = Tn

        # 记录“施加的平移向量”语义（和你之前的 xyz_vd 一致）：kp := kp + shift
        # 这里我们的变换是 kp := (kp - body_center) @ Rz @ Rx
        # 因此仅把“平移”的部分记录为 shift = -body_center（便于保持兼容）
        shift_vec = (-body_center).astype(np.float32)

    # ---------- 存桶 ----------
    def push(bucket):
        bucket['data'].append(data_out)  # (F_new,P,V,3)
        bucket['xyz_s'].append(spheres)  # (F_new,P,4)
        bucket['xyz_vd'].append(shift_vec)  # (3,)
        bucket['view'].append(view)
        bucket['label'].append(label)
        bucket['valid_frame'].append(np.int32(F_new))

        # 调试/可选信息（不依赖也不影响训练）
        bucket['body_center'].append(body_center.astype(np.float32))
        bucket['rot_z'].append(Rz.astype(np.float32))
        bucket['rot_x'].append(Rx.astype(np.float32))


    # CS
    if subj in CS_TRAIN_SUBJ:
        push(cs['train'])
    else:
        push(cs['test'])
    # CV
    if cam in CV_TRAIN_CAM:
        push(cv['train'])
    else:
        push(cv['test'])


# ---------- 保存 ----------
def save_npz(bucket, path: Path):
    if not bucket['data']:
        print('skip empty', path.name)
        return
    np.savez_compressed(
        path,
        data=np.array(bucket['data'], dtype=object),
        xyz_sphere=np.array(bucket['xyz_s'], dtype=object),
        xyz_vd=np.stack(bucket['xyz_vd']).astype(np.float32),
        view=np.array(bucket['view'], dtype=np.uint8),
        label=np.array(bucket['label'], dtype=np.int64),
        valid_frame=np.array(bucket['valid_frame'], dtype=np.int32),
        modalities=np.array(['xyz'], dtype=object),
        body_center=np.array(bucket['body_center'], dtype=object),
        rot_z=np.array(bucket['rot_z'], dtype=object),
        rot_x=np.array(bucket['rot_x'], dtype=object),
    )
    print('✓ saved', path.name, '| N =', len(bucket['data']))


save_npz(cs['train'], OUT_DIR / 'ntu60_cs_train_xyz_raw.npz')
save_npz(cs['test'], OUT_DIR / 'ntu60_cs_test_xyz_raw.npz')
save_npz(cv['train'], OUT_DIR / 'ntu60_cv_train_xyz_raw.npz')
save_npz(cv['test'], OUT_DIR / 'ntu60_cv_test_xyz_raw.npz')
