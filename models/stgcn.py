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
  封装 backbone 和 head，生成完整模型
"""


import torch
import torch.nn as nn
from models.backbone_stgcn import STGCN as STGCNBackbone
from models.head import GCNHead


class STGCNClassifier(nn.Module):
    """
    STGCN 分类器 = STGCNBackbone + GCNHead（与 PYSKL 设计对齐）
    - backbone_kwargs: 直接传给 STGCNBackbone（支持 gcn_*/tcn_* 前缀参数）
    - head_kwargs:     传给 GCNHead（如 dropout、loss 配置等）
    - pretrained_backbone / pretrained_head: 可分别加载预训练权重
    输入:  (N, M, T, V, C)
    输出:  (N, num_classes)
    """

    def __init__(self,
                 num_classes: int = 60,
                 backbone_kwargs: dict = None,
                 head_kwargs: dict = None,
                 pretrained_backbone: str = None,
                 pretrained_head: str = None):
        super().__init__()
        backbone_kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
        head_kwargs = {} if head_kwargs is None else dict(head_kwargs)

        # ---- Backbone ----
        self.backbone = STGCNBackbone(**backbone_kwargs)

        # 推断 head 的输入通道数（最后一层 TCN 的 out_channels）
        last_out_c = self.backbone.gcn[-1].tcn.out_channels
        self.head = GCNHead(num_classes=num_classes, in_channels=last_out_c, **head_kwargs)

        # 可选加载预训练
        if isinstance(pretrained_backbone, str) and len(pretrained_backbone) > 0:
            self.backbone.init_weights(pretrained_backbone, strict=False)

        if isinstance(pretrained_head, str) and len(pretrained_head) > 0:
            state = torch.load(pretrained_head, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # 兼容常见前缀
            new_state = {}
            for k, v in state.items():
                nk = k
                for p in ("head.", "module."):
                    if nk.startswith(p):
                        nk = nk[len(p):]
                new_state[nk] = v
            self.head.load_state_dict(new_state, strict=False)

    def forward(self, x):
        """
        x: (N, M, T, V, C)
        return logits: (N, num_classes)
        """
        feat = self.backbone(x)  # (N, M, C, T', V)
        logits = self.head(feat)  # (N, num_classes)
        return logits


# ========================== quick test ==========================
if __name__ == "__main__":
    from thop import profile
    import torch.nn as nn
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # ----- Backbone 默认参数（与 PYSKL 对齐） -----
    backbone_kwargs = dict(
        graph_cfg=dict(layout='nturgb+d', mode='spatial', max_hop=1),
        in_channels=3,
        base_channels=64,
        data_bn_type='VC',
        ch_ratio=2.0,
        num_person=2,
        num_stages=10,
        inflate_stages=(5, 8),
        down_stages=(5, 8),
        # 关键：PYSKL 默认多分支 TCN
        tcn_type='mstcn',       # stgcn: unit_tcn, stgcn++: mstcn
        gcn_adaptive='init',    # stgcn: importance, stgcn++: init
        gcn_with_res=True,      # stgcn: False, stgcn++: True
        # 也可继续传入 gcn_*/tcn_* 前缀配置
        # gcn_adaptive='importance', gcn_conv_pos='pre', gcn_with_res=False,
        # tcn_dropout=0.0,
    )

    model = STGCNClassifier(
        num_classes=1,
        backbone_kwargs=backbone_kwargs,
        head_kwargs=dict(dropout=0.0),  # 与 PYSKL 的 GCNHead 默认一致
        pretrained_backbone=None,
        pretrained_head=None
    )

    # ----- 统计可训练参数 -----
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params: {n_params / 1e6:.3f} M ({n_params:,})")
    # ----- 统计参数（包含冻结参数） -----
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params (total): {n_params / 1e6:.3f} M ({n_params:,})")

    # ----- 构造 dummy 输入：N=1, M=2, T=100, V=25, C=3 -----
    N, M, T, V, C = 1, 2, 100, 25, 3
    x = torch.randn(N, M, T, V, C)

    # ----- 前向得到 logits -----
    with torch.no_grad():
        logits = model(x)  # (N, 60)
    print("Logits shape:", tuple(logits.shape))

    # ----- THOP 计算 MACs（打印为“GFLOPs (thop MACs)”）-----
    # 为避免激活类在 thop 中缺少 total_ops 属性，这里显式置零
    def _zero_ops(m, inp, out):
        m.total_ops = torch.zeros(1)

    custom_ops = {
        nn.ReLU: _zero_ops,
        nn.LeakyReLU: _zero_ops,
        nn.PReLU: _zero_ops,
        nn.Sigmoid: _zero_ops,
        nn.Tanh: _zero_ops,
        nn.Dropout: _zero_ops,
        nn.Dropout2d: _zero_ops,
        nn.Dropout3d: _zero_ops,
        nn.Identity: _zero_ops,
        nn.AdaptiveAvgPool2d: _zero_ops,
        nn.AdaptiveAvgPool3d: _zero_ops,
        nn.MaxPool2d: _zero_ops,
    }
    macs, _ = profile(model, inputs=(x,), custom_ops=custom_ops, verbose=False)
    print(f"GFLOPs (thop MACs): {macs / 1e9:.3f}")
