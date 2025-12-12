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
  改写自：https://github.com/kennymckormick/pyskl/blob/main/pyskl/models/gcns/stgcn.py
"""


import copy as cp
import torch
import torch.nn as nn

from utils.graph_utils import Graph
from utils.gcn_ops import unit_gcn
from utils.tcn_ops import mstcn, unit_tcn

EPS = 1e-4


class STGCNBlock(nn.Module):
    """
    结构与 PYSKL 一致：
      - GCN: unit_gcn
      - TCN: unit_tcn 或 mstcn（通过 tcn_type 指定）
      - 残差：若(in==out 且 stride=1) 直连；否则 1x1 TCN 下采样
    额外参数通过 kwargs 传入，前缀 'gcn_' / 'tcn_' 分别路由给对应子模块。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 A: torch.Tensor,
                 stride: int = 1,
                 residual: bool = True,
                 **kwargs):
        super().__init__()

        # 拆分给 GCN/TCN 的子配置
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        leftovers = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(leftovers) == 0, f'Invalid arguments: {leftovers}'

        # 选择 TCN 类型
        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        # （目前仅 unit_gcn，对齐 PYSKL）
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        # GCN
        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        # TCN
        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, kernel_size=9, stride=stride, **tcn_kwargs)
        else:  # 'mstcn'
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)

        self.relu = nn.ReLU(inplace=True)

        # 残差支路
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class STGCN(nn.Module):
    """
    与 PYSKL 的 backbones.STGCN 一致的骨干（仅移除了 mmcv 依赖）：
      - graph_cfg: 传给 Graph(**graph_cfg)
      - data_bn_type: 'MVC' or 'VC' or None
      - 10 个 stage（默认），第 5/8 stage stride=2 下采样（与 PYSKL 默认一致）
      - tcn 默认使用 MSTCN（可通过 kwargs: tcn_type='unit_tcn' 切回单分支）
      - 可选预训练加载：pretrained = 路径或 None
    输出： (N, M, C, T', V)
    """

    def __init__(self,
                 graph_cfg: dict,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 data_bn_type: str = 'VC',
                 ch_ratio: float = 2.0,
                 num_person: int = 2,  # 仅 MVC 用
                 num_stages: int = 10,
                 inflate_stages=(5, 8),
                 down_stages=(5, 8),
                 pretrained: str = None,
                 **kwargs):
        super().__init__()

        # 图
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        self.data_bn_type = data_bn_type
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = tuple(inflate_stages)
        self.down_stages = tuple(down_stages)
        self.pretrained = pretrained

        # 输入 BN（与 PYSKL 一致）
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        # 逐层 kwargs（PYSKL 会把 tuple 参数按层展开）
        lw_kwargs = [cp.deepcopy(kwargs) for _ in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        # 第一层不做 tcn_dropout（与 PYSKL 对齐）
        lw_kwargs[0].pop('tcn_dropout', None)

        modules = []
        # stage 1：若 in != base，用一个 block 升通道（无残差）
        if self.in_channels != self.base_channels:
            modules.append(
                STGCNBlock(in_channels, base_channels, A.clone(), stride=1, residual=False, **lw_kwargs[0])
            )
        inflate_times = 0
        cur_in = base_channels

        # stage 2..num_stages
        for i in range(2, num_stages + 1):
            stride = 1 + (i in self.down_stages)  # i 在 down_stages 时 stride=2
            if i in self.inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * (self.ch_ratio ** inflate_times) + EPS)

            modules.append(
                STGCNBlock(cur_in, out_channels, A.clone(), stride=stride, **lw_kwargs[i - 1])
            )
            cur_in = out_channels

        # 若首层没升通道（in==base），则总层数-1（与 PYSKL 一致）
        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)

        # 预训练（可选）
        if isinstance(self.pretrained, str) and len(self.pretrained) > 0:
            self.init_weights(self.pretrained)

    # —— 预训练加载（兼容常见权重字典结构）——
    def init_weights(self, ckpt_path: str, strict: bool = False):
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        # 去常见前缀
        new_state = {}
        for k, v in state.items():
            nk = k
            for p in ('backbone.', 'module.'):
                if nk.startswith(p):
                    nk = nk[len(p):]
            new_state[nk] = v
        self.load_state_dict(new_state, strict=strict)

    def forward(self, x):
        """
        x: (N, M, T, V, C)
        return: (N, M, C, T', V)
        """
        N, M, T, V, C = x.size()
        # -> (N,M,V,C,T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        # data BN
        if self.data_bn_type == 'MVC':
            x = x.view(N, M * V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T)
        elif self.data_bn_type == 'VC':
            x = x.view(N * M, V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T)
        # -> (N*M, C, T, V)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # blocks
        for i in range(self.num_stages):
            x = self.gcn[i](x)

        # -> (N, M, C, T', V)
        x = x.view(N, M, *x.shape[1:])
        return x


# ========================== quick test ==========================
if __name__ == "__main__":
    from thop import profile
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # ----- 配置（与 PYSKL 默认一致） -----
    graph_cfg = dict(layout='nturgb+d', mode='spatial', max_hop=1)
    model = STGCN(
        graph_cfg=graph_cfg,
        in_channels=12,
        base_channels=64,
        data_bn_type='VC',
        ch_ratio=2,
        num_person=2,
        num_stages=10,
        inflate_stages=(5, 8),
        down_stages=(5, 8),
        # 关键：PYSKL 默认用 MSTCN
        # tcn_type='unit_tcn',
        tcn_type='mstcn',
        # 你也可以传其它 gcn_/tcn_ 前缀的参数：
        # gcn_adaptive='importance', gcn_conv_pos='pre', gcn_with_res=False,
        # tcn_dropout=0.0,
    )

    # ----- 统计可训练参数 -----
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params: {n_params / 1e6:.3f} M ({n_params:,})")

    # ----- 构造 dummy 输入：N=1, M=2, T=100, V=25, C=3 -----
    N, M, T, V, C = 1, 2, 100, 25, 12
    x = torch.randn(N, M, T, V, C)

    # ----- 前向得到输出尺寸 -----
    with torch.no_grad():
        feat = model(x)  # (N,M,C,T',V)
    print("Backbone output shape:", tuple(feat.shape))

    # ----- GFLOPs（thop 返回 MACs，这里按 2*MACs≈FLOPs 还是常见争议；多数论文用 MACs→GFLOPs 记作 /1e9）
    # thop 在某些激活上会尝试累加 zero_ops，这里给 ReLU/Dropout 提供 0 计数器，避免属性不存在的报错
    def _zero_ops(m, inp, out):
        m.total_ops = torch.zeros(1)


    custom_ops = {
        nn.ReLU: _zero_ops,
        nn.Dropout: _zero_ops,
        nn.Dropout2d: _zero_ops,
        nn.Dropout3d: _zero_ops,
        nn.Identity: _zero_ops,
    }
    macs, _ = profile(model, inputs=(x,), custom_ops=custom_ops, verbose=False)
    print(f"GFLOPs (thop MACs): {macs / 1e9:.3f}")
