#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# 数据与模型
from dataset_ntu60_3D_pkl_npz_ms import RawXYZDataset  # 1-clip
from models.stgcn import STGCNClassifier

from task_utils.logger_utils import init_logger, log_config, log_system_info
from task_utils.data.data_skeleton_ops import gen_and_merge_feats_TPVC
from task_utils.test_utils import (
    topk_correct, phase_shift_even_indices_with_pad, contig_window_indices_with_pad, load_ckpt_pref_ema
)

# ==================== 配置 ====================
CFG = dict(
    # 一个或多个 npz 都行；脚本会逐个 npz 生成结果
    NPZ_PATH=[
        r'G:/datasets/cls_video/nturgbd_skeletons_s001_to_s017/pkl_mmaction/npz_raw_xyz_ntu60/ntu60_cs_test_xyz_raw.npz',
    ],

    # 多模型（多流）列表：name 仅用于日志；streams 用于特征生成；ckpt 为各自权重；w 为模型融合权重
    MODELS=[
        dict(name='joint', streams=('j',), ckpt='logs/best_stgcn_ntu60_ms_cs_j_250906_90.15.pt', w=1.0),
        dict(name='bone', streams=('b',), ckpt='logs/best_stgcn_ntu60_ms_cs_b_250901_90.19.pt', w=1.0),
        # dict(name='jm',  streams=('jm',), ckpt='ckpt_jm.pt',     w=0.5),
        # dict(name='bm',  streams=('bm',), ckpt='ckpt_bm.pt',     w=0.5),
    ],

    NUM_CLASSES=60,  # NTU60
    GPU_ID=0,
    SEED=42,
    AMP=True,

    # 数据形状
    T=64,

    # 1-clip 批
    BATCH_1CLIP=16,
    # N-clip：这里的 batch 指“每步前向多少个视频”；实际前向 batch = Bv * Nc
    BATCH_NCLIP=8,

    NUM_WORKERS=4,

    # 多 clip
    NUM_CLIPS=2,
    CLIP_FUSE='prob',  # 把同一模型 Nc 个 clip 的分数怎么融合：'prob' | 'logits'
    MODEL_FUSE='logits',  # 把不同模型的分数怎么融合：       'prob' | 'logits'
    USE_CONTIG_WINDOW=False,  # False=相位切片（推荐）；True=连续窗口

    # 可选：把每个模型与融合后的分数导出，便于复现实验或后处理
    SAVE_SCORES_DIR='',  # 例如 'scores_out'；留空则不保存
)


# ==================== N-clip Dataset（与原版一致，仅打包 Nc 个片段返回） ====================
class NTUClipsDataset(Dataset):
    """
    每条样本（一个视频） -> (Nc, C, T, V=25, P=2), label
    多流用 gen_and_merge_feats_TPVC（与你的训练一致）。
    """

    def __init__(self, npz_path, T: int, streams=('j',), num_clips: int = 10, use_contig: bool = False):
        paths = [npz_path] if isinstance(npz_path, (str, Path)) else list(npz_path)
        assert len(paths) > 0
        self._ds = []
        for p in paths:
            f = np.load(p, allow_pickle=True)
            self._ds.append({k: f[k] for k in f})
            f.close()

        self._cum = []
        tot = 0
        for d in self._ds:
            tot += len(d['label'])
            self._cum.append(tot)

        self.T = int(T)
        self.streams = tuple(streams)
        self.Nc = int(num_clips)
        self.use_contig = bool(use_contig)

    def __len__(self):
        return self._cum[-1]

    @staticmethod
    def _locate(cum, idx):
        for i, v in enumerate(cum):
            if idx < v:
                prev = cum[i - 1] if i else 0
                return i, idx - prev
        raise IndexError

    def __getitem__(self, idx: int):
        fi, inner = self._locate(self._cum, idx)
        ds = self._ds[fi]
        xyz: np.ndarray = ds['data'][inner]  # (F, P<=2, 25, 3)
        label = int(ds['label'][inner])

        F, P_cur, V, C = xyz.shape
        if P_cur == 1:  # pad P->2
            xyz = np.pad(xyz, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant')

        # 10 组 ids
        if self.use_contig:
            ids_list = contig_window_indices_with_pad(F, self.T, self.Nc)
        else:
            ids_list = phase_shift_even_indices_with_pad(F, self.T, self.Nc)

        clips_CTVP = []
        for ids in ids_list:
            m = ids >= 0
            safe_ids = np.where(m, ids, 0).astype(np.int32)
            clip = xyz[safe_ids]  # (T,P,V,3)
            if not m.all():
                clip = clip.copy()
                clip[~m, :, :, :] = 0.0

            merged = gen_and_merge_feats_TPVC(
                joints_TPVC=clip, dataset="nturgb+d", feats=self.streams, concat_axis=-1
            )  # (T,P,V, 3*len(streams))
            feat = torch.from_numpy(merged).permute(3, 0, 2, 1).contiguous()  # (C,T,V,P)
            clips_CTVP.append(feat)

        X = torch.stack(clips_CTVP, dim=0)  # (Nc, C, T, V, P)
        return X, label


# ---------------- 工具 ----------------
def to_stgcn_input(x):  # (B,C,T,V,P) -> (B,P,T,V,C)
    return x.permute(0, 4, 2, 3, 1).contiguous()


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=1, keepdims=True) + 1e-12
    return x


def build_model(num_classes: int, in_ch=3, device='cuda:0'):
    backbone_kwargs = dict(
        graph_cfg=dict(layout='nturgb+d', mode='spatial', max_hop=1),
        in_channels=in_ch, base_channels=64, data_bn_type='VC',
        ch_ratio=2.0, num_person=2, num_stages=10,
        inflate_stages=(5, 8), down_stages=(5, 8),
        tcn_type='mstcn', gcn_adaptive='init', gcn_with_res=True,
    )
    model = STGCNClassifier(
        num_classes=num_classes,
        backbone_kwargs=backbone_kwargs,
        head_kwargs=dict(dropout=0.0),
        pretrained_backbone=None, pretrained_head=None,
    ).to(device).eval()
    return model


# ---------------- 单模型 · 1-clip ----------------
@torch.no_grad()
def eval_1clip_single(npz_path, model, device, streams, batch_size, num_workers, amp=True):
    ds = RawXYZDataset(npz_path, mode='val', T=CFG["T"], clip=1, streams=streams)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0),
    )
    amp_ctx = torch.cuda.amp.autocast if (torch.cuda.is_available() and amp) else torch.cpu.amp.autocast

    logits_all = []
    labels_all = []
    top1 = top5 = tot = 0
    t0 = time.time()
    for xb, yb in loader:
        xb = to_stgcn_input(xb).to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with amp_ctx():
            logits = model(xb)  # (B,C)
        c1, c5 = topk_correct(logits, yb, (1, 5))
        top1 += c1
        top5 += c5
        tot += yb.size(0)
        logits_all.append(logits.detach().cpu().float())
        labels_all.append(yb.detach().cpu().long())
    dt = time.time() - t0

    logits_all = torch.cat(logits_all, dim=0).numpy()  # [N,C]
    labels_all = torch.cat(labels_all, dim=0).numpy()  # [N]
    return dict(
        logits=logits_all, labels=labels_all,
        top1=float(top1) / float(tot) * 100.0, top5=float(top5) / float(tot) * 100.0,
        time_sec=dt, N=int(tot)
    )


# ---------------- 单模型 · N-clip ----------------
@torch.no_grad()
def eval_nclip_single(npz_path, model, device, streams, batch_videos, num_workers, Nc, fuse='prob', amp=True,
                      use_contig=False):
    ds = NTUClipsDataset(npz_path, T=CFG["T"], streams=streams, num_clips=Nc, use_contig=use_contig)
    loader = DataLoader(
        ds, batch_size=batch_videos, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0),
    )
    amp_ctx = torch.cuda.amp.autocast if (torch.cuda.is_available() and amp) else torch.cpu.amp.autocast
    assert fuse in ('prob', 'logits')

    scores_all = []  # 已按 clip 融合后的 [N,C]（logits 或 prob）
    labels_all = []
    top1 = top5 = tot = 0
    t0 = time.time()

    for Xclips, yb in loader:
        Bv, Nc_cur = Xclips.shape[:2]
        xb = Xclips.view(Bv * Nc_cur, *Xclips.shape[2:])
        xb = to_stgcn_input(xb).to(device, non_blocking=True)  # (Bv*Nc, P, T, V, C)
        yb = yb.to(device, non_blocking=True)

        with amp_ctx():
            logits = model(xb).view(Bv, Nc_cur, -1)  # (Bv, Nc, C)

        if fuse == 'prob':
            fused = torch.softmax(logits, dim=2).mean(dim=1)  # (Bv,C) prob
        else:
            fused = logits.mean(dim=1)  # (Bv,C) logits

        c1, c5 = topk_correct(fused, yb, (1, 5))
        top1 += c1
        top5 += c5
        tot += yb.size(0)
        scores_all.append(fused.detach().cpu().float())
        labels_all.append(yb.detach().cpu().long())

    dt = time.time() - t0
    scores_all = torch.cat(scores_all, dim=0).numpy()  # [N,C]
    labels_all = torch.cat(labels_all, dim=0).numpy()  # [N]
    return dict(
        fused_clip_scores=scores_all,  # 已按 Nc clip 融合
        labels=labels_all,
        top1=float(top1) / float(tot) * 100.0, top5=float(top5) / float(tot) * 100.0,
        time_sec=dt, N=int(tot), fuse=fuse
    )


# ---------------- 跨模型 late-fusion ----------------
def fuse_models(score_list: list, weights: list, mode='logits'):
    """
    score_list: [ [N,C], [N,C], ... ]；同一 npz、同一排序
    weights   : 同长度的权重
    mode      : 'logits' | 'prob'
    返回 fused [N,C]
    """
    assert len(score_list) >= 1 and len(score_list) == len(weights)
    W = np.asarray(weights, dtype=np.float32)
    W = W / (W.sum() + 1e-12)

    Xs = [np.asarray(s, dtype=np.float32) for s in score_list]
    N, C = Xs[0].shape
    assert all(x.shape == (N, C) for x in Xs), "各模型分数形状必须一致"

    if mode == 'prob':
        # 若传入的是 logits，先 softmax；若已是概率也没事（softmax(softmax(x)) ≈ softmax(x)）
        Xs = [softmax_np(x.copy()) for x in Xs]
        fused = np.zeros((N, C), np.float32)
        for w, x in zip(W, Xs): fused += w * x
        return fused
    else:
        # logits：加权和（不必再 softmax，排序不变；评估 top-k 用原分数即可）
        fused = np.zeros((N, C), np.float32)
        for w, x in zip(W, Xs): fused += w * x
        return fused


def evaluate_topk(scores: np.ndarray, labels: np.ndarray, ks=(1, 5)):
    idx = np.argsort(-scores, axis=1)
    out = {}
    for k in ks:
        ok = (idx[:, :k] == labels[:, None]).any(axis=1).mean()
        out[k] = float(ok) * 100.0
    return out


# ---------------- main ----------------
def main():
    # 后端设定
    torch.manual_seed(CFG["SEED"])
    np.random.seed(CFG["SEED"])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger = init_logger("test_multi_model_npz", filename_prefix="test_multi_model_npz")
    log_config(logger, CFG)
    log_system_info(logger)

    device = torch.device(f"cuda:{CFG['GPU_ID']}" if torch.cuda.is_available() else "cpu")

    # 预创建输出目录
    save_dir = Path(CFG["SAVE_SCORES_DIR"]) if CFG["SAVE_SCORES_DIR"] else None
    if save_dir is not None: save_dir.mkdir(parents=True, exist_ok=True)

    # ------------ 逐个 NPZ 评测 ------------
    for npz in CFG["NPZ_PATH"]:
        npz_name = Path(npz).name
        logger.info("=" * 90)
        logger.info(f"[NPZ] {npz_name}")
        per_model_1c = []  # list of dict: {'name', 'logits', 'labels', 'top1', 'top5'}
        per_model_Nc = []  # list of dict: {'name', 'fused_clip_scores', 'labels', 'top1', 'top5'}

        # ---------- 各模型独立评测 ----------
        for m in CFG["MODELS"]:
            name = m["name"]
            streams = tuple(m["streams"])
            ckpt = m["ckpt"]
            weight = float(m.get("w", 1.0))

            in_ch = 3 * len(streams)
            model = build_model(num_classes=CFG["NUM_CLASSES"], in_ch=in_ch, device=device)
            load_ckpt_pref_ema(model, ckpt, logger)
            model.eval()

            # 1-clip
            r1 = eval_1clip_single(
                npz, model, device, streams,
                batch_size=CFG["BATCH_1CLIP"], num_workers=CFG["NUM_WORKERS"], amp=CFG["AMP"]
            )
            logger.info(f"[{name:>6} | 1-clip] Top1={r1['top1']:.2f}%  Top5={r1['top5']:.2f}% "
                        f"| N={r1['N']} | t={r1['time_sec']:.1f}s")
            per_model_1c.append(dict(name=name, w=weight, **r1))

            # N-clip
            Nc = int(CFG["NUM_CLIPS"])
            assert Nc >= 2
            rN = eval_nclip_single(
                npz, model, device, streams,
                batch_videos=CFG["BATCH_NCLIP"], num_workers=CFG["NUM_WORKERS"],
                Nc=Nc, fuse=CFG["CLIP_FUSE"], amp=CFG["AMP"], use_contig=CFG["USE_CONTIG_WINDOW"]
            )
            logger.info(f"[{name:>6} | {Nc}-clip/{CFG['CLIP_FUSE']}] Top1={rN['top1']:.2f}%  Top5={rN['top5']:.2f}% "
                        f"| N={rN['N']} | t={rN['time_sec']:.1f}s")
            per_model_Nc.append(dict(name=name, w=weight, **rN))

            # 可选保存单模型分数
            if save_dir is not None:
                with open(save_dir / f"{npz_name}.{name}.1clip.pkl", "wb") as f:
                    pickle.dump([r1['logits'][i] for i in range(r1['N'])], f)
                with open(save_dir / f"{npz_name}.{name}.{Nc}clip_{CFG['CLIP_FUSE']}.pkl", "wb") as f:
                    pickle.dump([rN['fused_clip_scores'][i] for i in range(rN['N'])], f)

            # 清显存（多个模型串行评测时更稳）
            del model
            torch.cuda.empty_cache()

        # ---------- 融合（模型层） ----------
        # 校验标签一致
        labels_ref = per_model_1c[0]['labels']
        assert all(np.array_equal(labels_ref, m['labels']) for m in per_model_1c), "1-clip 标签对齐失败"
        assert all(np.array_equal(labels_ref, m['labels']) for m in per_model_Nc), "N-clip 标签对齐失败"

        # 1-clip 模型融合
        scores_1c = [m['logits'] for m in per_model_1c]
        weights = [m['w'] for m in per_model_1c]
        fused_1c = fuse_models(scores_1c, weights, mode=CFG["MODEL_FUSE"])
        acc_1c = evaluate_topk(fused_1c, labels_ref, ks=(1, 5))
        logger.info(f"[FUSION | 1-clip | {CFG['MODEL_FUSE']}] Top1={acc_1c[1]:.2f}%  Top5={acc_1c[5]:.2f}%")

        # N-clip 模型融合（注意：这里的输入已是“各模型 Nc-clip 融合后的分数”，再做模型层融合）
        scores_Nc = [m['fused_clip_scores'] for m in per_model_Nc]
        weights = [m['w'] for m in per_model_Nc]
        fused_Nc = fuse_models(scores_Nc, weights, mode=CFG["MODEL_FUSE"])
        acc_Nc = evaluate_topk(fused_Nc, labels_ref, ks=(1, 5))
        logger.info(f"[FUSION | {CFG['NUM_CLIPS']}-clip/{CFG['CLIP_FUSE']} | {CFG['MODEL_FUSE']}] "
                    f"Top1={acc_Nc[1]:.2f}%  Top5={acc_Nc[5]:.2f}%")

        # 可选保存融合分数
        if save_dir is not None:
            with open(save_dir / f"{npz_name}.FUSION.1clip_{CFG['MODEL_FUSE']}.pkl", "wb") as f:
                pickle.dump([fused_1c[i] for i in range(len(labels_ref))], f)
            with open(save_dir / f"{npz_name}.FUSION.{CFG['NUM_CLIPS']}clip_{CFG['CLIP_FUSE']}_{CFG['MODEL_FUSE']}.pkl",
                      "wb") as f:
                pickle.dump([fused_Nc[i] for i in range(len(labels_ref))], f)

    logger.info("=" * 90)
    logger.info("✅ Done.")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
