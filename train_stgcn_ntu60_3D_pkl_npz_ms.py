import sys, time, os, warnings
import numpy as np
from pathlib import Path
import torch
from torch.cuda.amp import GradScaler

# ==== 工程路径 ====
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ==== 外部依赖 ====
from dataset_ntu60_3D_pkl_npz_ms import RawXYZDataset  # 输出 (C,T,V,P)
from models.stgcn import STGCNClassifier
from task_utils.logger_utils import init_logger, log_config, log_system_info
from task_utils.engine_utils import build_loader, run_one_epoch
from task_utils.model_utils import count_params, try_compute_gflops, load_checkpoint, save_checkpoint
from task_utils.train_utils import set_global_seed, build_optimizer, build_scheduler, build_ema

# ==================== 配置 ====================
CFG = dict(
    TRAIN_NPZ=fr'G:/datasets/cls_video/nturgbd_skeletons_s001_to_s017/pkl_mmaction/npz_raw_xyz_ntu60/ntu60_cs_train_xyz_raw.npz',
    VAL_NPZ=fr'G:/datasets/cls_video/nturgbd_skeletons_s001_to_s017/pkl_mmaction/npz_raw_xyz_ntu60/ntu60_cs_test_xyz_raw.npz',
    NUM_CLASS=60,
    BATCH=16,
    GPU_ID=0,
    EPOCH=240,
    NUM_WORKERS=8,
    SEED=42,
    AMP=True,
    T=64,
    STREAMS=('j',),  # 多流 ('j','b','jm','bm') → 12 通道
    CKPT_PATH="best_stgcn_ntu60_ms_cs_j_base_250906_01.pt",
    PRINT_FREQ=0,
    PREFETCH=4,

    # 优化器：用字典统一管理
    OPT=dict(name='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True),
    # OPT=dict(name='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8),
    # OPT=dict(name='AdamW', lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4),

    # 学习率：也改成字典
    # SCHED=dict(type='step', by='step', min_lr=1e-4, decay=0.95, decay_epoch=2),
    SCHED=dict(type='cosine', by='step', min_lr=1e-2, warmup_epochs=3),  # SGD
    # SCHED=dict(type='cosine', by='step', min_lr=0, warmup_epochs=3),  # Adam
    # SCHED = dict(type='linear', by='step', min_lr=0.0),
    # SCHED = dict(type='exp', by='step', min_lr=0.0),                  # 自动算 gamma 到终点
    # SCHED = dict(type='exp', by='step', min_lr=0.0, gamma=0.98),

    PRETRAINED="",  # 预训练权重（为空字符串则不加载）
    # EMA 设置（默认启用；验证/保存用 EMA）,基准decay: BS=16, decay=0.999
    # EMA=dict(enabled=True, start_epoch=3, eval_use_ema=True, ref_batch=16, decay=0.999),
    EMA=dict(enabled=True, start_epoch=3, eval_use_ema=True, half_life_epoch=0.25),
)


def main():
    # 环境 & 日志
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    logger = init_logger("train_stgcn", filename_prefix="train_stgcn")
    log_config(logger, CFG)
    log_system_info(logger)

    # 随机/后端
    set_global_seed(CFG["SEED"], deterministic=False, cudnn_benchmark=True)
    # Ampere: TF32 推荐与旧实现对齐（避免数值/吞吐差别）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # 数据
    train_loader, _ = build_loader(RawXYZDataset, CFG["TRAIN_NPZ"], "train", CFG, logger,
                                   dataset_kwargs=dict(streams=CFG.get("STREAMS", ("j",))))
    val_loader, _ = build_loader(RawXYZDataset, CFG["VAL_NPZ"], "val", CFG, logger,
                                 dataset_kwargs=dict(streams=CFG.get("STREAMS", ("j",))))

    device = f'cuda:{CFG["GPU_ID"]}' if torch.cuda.is_available() else 'cpu'

    # 动态通道数：每个流 3 通道（xyz）
    in_ch = 3 * len(CFG.get("STREAMS", ('j',)))
    logger.info(f"Streams: {CFG['STREAMS']} -> in_channels = {in_ch}")

    # 模型（输入 (N,M,T,V,C)）
    backbone_kwargs = dict(
        graph_cfg=dict(layout='nturgb+d', mode='spatial', max_hop=1),
        in_channels=in_ch, base_channels=64, data_bn_type='VC',
        ch_ratio=2.0, num_person=2, num_stages=10,
        inflate_stages=(5, 8), down_stages=(5, 8),
        tcn_type='mstcn',       # st-gcn: unit_tcn      st-gcn++: mstcn 
        gcn_adaptive='init',    # st-gcn: importance    st-gcn++: init
        gcn_with_res=True,      # st-gcn: False         st-gcn++: True
    )
    model = STGCNClassifier(
        num_classes=CFG["NUM_CLASS"],
        backbone_kwargs=backbone_kwargs,
        head_kwargs=dict(dropout=0.0),
        pretrained_backbone=None,
        pretrained_head=None,
    ).to(device)

    # 可选：加载预训练
    if CFG.get("PRETRAINED"):
        try:
            ck = load_checkpoint(CFG["PRETRAINED"], model, optimizer=None, map_location=device)
            logger.info(f"Loaded pretrained from: {CFG['PRETRAINED']} (epoch={ck.get('epoch', '?')})")
        except Exception as e:
            logger.info(f"[WARN] load PRETRAINED failed: {e}")

    # 模型信息 + GFLOPs（全模型 & 仅骨干）
    params_all = count_params(model)
    gfl_all, _ = try_compute_gflops(model, (1, 2, CFG["T"], 25, in_ch), device=device)
    gfl_bb, _ = try_compute_gflops(model.backbone, (1, 2, CFG["T"], 25, in_ch), device=device)
    gfl_all_s = f"{gfl_all:.4f} G" if gfl_all is not None else "N/A"
    gfl_bb_s = f"{gfl_bb:.4f} G" if gfl_bb is not None else "N/A"
    logger.info(f"Model: {model.__class__.__name__} | Params(all): {params_all:,} | "
                f"GFLOPs(all/backbone): {gfl_all_s} / {gfl_bb_s}")

    # 构建优化器 + 调度器
    optimizer = build_optimizer(model, CFG["OPT"])
    CFG['SCHED']['base_lr'] = CFG['OPT']['lr']
    sched_step = build_scheduler(CFG["SCHED"])
    by = str(CFG["SCHED"].get("by", "step")).lower()

    # 启用EMA，传入数据集长度自动获取迭代长度
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CFG["EPOCH"]
    ema = build_ema(CFG, model, steps_per_epoch=steps_per_epoch, logger=logger)

    # 如果 PRETRAINED 里带了 ema_state_dict，顺手加载（可选）
    if ema is not None and CFG.get("PRETRAINED"):
        try:
            ck = torch.load(CFG["PRETRAINED"], map_location=device)
            if "ema_state_dict" in ck and ck["ema_state_dict"] is not None:
                ema.load_state_dict(ck["ema_state_dict"])
                logger.info("Loaded EMA state from PRETRAINED.")
        except Exception as e:
            logger.info(f"[WARN] load EMA from PRETRAINED failed: {e}")

    # AMP
    scaler = GradScaler(enabled=(device.startswith('cuda') and CFG.get("AMP", True)))

    best_path = CFG["CKPT_PATH"]
    best_top1 = best_top5 = 0.0
    best_epoch = 0

    for ep in range(1, CFG["EPOCH"] + 1):
        # lr = sched_step(optimizer, epoch=ep - 1, total_epochs=CFG["EPOCH"])
        if by == "step":
            cur_step = (ep - 1) * steps_per_epoch
            lr = sched_step(optimizer, step=cur_step, total_steps=total_steps, steps_per_epoch=steps_per_epoch)
        else:
            lr = sched_step(optimizer, epoch=ep - 1, total_epochs=CFG["EPOCH"])

        logger.info("-" * 80)
        logger.info(f"[Epoch {ep:02d}/{CFG['EPOCH']}] | LR: {lr:.6f}")

        # ======= Train（逐步更新 EMA）=======
        tr = run_one_epoch(
            model, train_loader, device,
            train=True, optimizer=optimizer, scaler=scaler, amp=CFG["AMP"],
            logger=logger, print_freq=CFG["PRINT_FREQ"],
            desc=f"[Train] Ep{ep:02d}",
            ema=ema,  # 传入 ema
            grad_clip=-1.0,  # 不裁剪
            lr_sched_iter=(sched_step if by == "step" else None),
            global_step_offset=(ep - 1) * steps_per_epoch,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
        )

        logger.info(
            f"Train | Loss: {tr['loss']:.4f} | Top1: {tr['top1']:6.2f}% | Top5: {tr['top5']:6.2f}% | Samples: {tr['samples']}")

        # ======= Val（用 EMA 模型验证/选择最优）=======
        use_ema_eval = bool(CFG["EMA"].get("eval_use_ema", False)) and (ema is not None) \
                       and getattr(ema, "initialized", False) and (ep >= CFG["EMA"].get("start_epoch", 3))  # 至少过 n 个 epoch 再用

        va = run_one_epoch(
            model, val_loader, device,  # 注意：这里也可以直接传 model，函数内部会切到 ema.ema
            train=False, optimizer=None, scaler=None, amp=CFG["AMP"],  # 建议 val 关 AMP
            logger=logger, print_freq=0,
            desc=f"[Val]   Ep{ep:02d}",
            ema=ema,  # 传入 ema
            # use_ema_for_eval=True,  # ✅ 用 EMA 做前向
            use_ema_for_eval=use_ema_eval,  # ✅ 用 EMA 做前向
        )

        logger.info(
            f"Val   | Loss: {va['loss']:.4f} | Top1: {va['top1']:6.2f}% | Top5: {va['top5']:6.2f}% | Samples: {va['samples']}")

        # ======= Save best（保存 EMA 权重）=======
        better = (va['top1'] > best_top1) or (np.isclose(va['top1'], best_top1) and va['top5'] > best_top5)
        if better:
            best_top1, best_top5, best_epoch = va['top1'], va['top5'], ep
            save_checkpoint(
                best_path, model, optimizer, ep,
                val_top1=best_top1, val_top5=best_top5, cfg=CFG,
                ema_state_dict=(ema.state_dict() if ema is not None else None)
            )
            logger.info(
                f"[BEST] 新最佳有效验证准确率 @ Ep{ep:03d}  Top1={best_top1:.2f}%  Top5={best_top5:.2f}%  -> saved to {best_path}")
            time.sleep(0.3)

    logger.info("=" * 80)
    logger.info(f"✅ Finished. Best Top1 {best_top1:.2f}% / Top5 {best_top5:.2f}% @ epoch {best_epoch}")
    logger.info("=" * 80)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
