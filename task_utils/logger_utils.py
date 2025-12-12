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
  实现日志记录。
"""


from __future__ import annotations
import torch
import logging, sys, datetime, platform
from pathlib import Path
from typing import Union

__all__ = ["init_logger", "log_config", "log_system_info"]

__dividing_line_length = 80


def init_logger(name: str = "train_stgcn",
                log_dir: Union[str, Path] = "logs",
                filename_prefix: str = "train") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
    log_path = log_dir / f"{filename_prefix}_{stamp}.log"

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    date = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=date))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=date))
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("=" * __dividing_line_length)
    logger.info(f"Logging to: {log_path.resolve()}")
    logger.info("=" * __dividing_line_length)
    return logger


def log_config(logger: logging.Logger, cfg: dict):
    logger.info("Config:")
    for k in sorted(cfg.keys()):
        logger.info(f"  {k}: {cfg[k]}")
    logger.info("-" * __dividing_line_length)


def log_system_info(logger: logging.Logger):
    py = sys.version.replace("\n", " ")
    os_name = platform.system()
    os_rel = platform.release()
    cpu = platform.processor()
    logger.info(f"Python: {py}")
    logger.info(f"OS: {os_name} {os_rel} | CPU: {cpu or 'N/A'}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
    logger.info(f"CUDA (build): {torch.version.cuda}")
    try:
        logger.info(f"cuDNN: {torch.backends.cudnn.version()}")
    except Exception:
        logger.info("cuDNN: N/A")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cc = torch.cuda.get_device_capability(idx)
        logger.info(f"CUDA device idx: {idx} | name: {name} | capability: {cc}")
    logger.info(f"TF32 matmul allowed: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(
        f"CUDNN benchmark: {torch.backends.cudnn.benchmark} | deterministic: {torch.backends.cudnn.deterministic}")
    logger.info("=" * __dividing_line_length)
