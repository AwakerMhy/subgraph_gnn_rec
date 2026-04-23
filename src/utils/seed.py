"""src/utils/seed.py — 统一随机种子设置入口"""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """统一设置所有随机数生成器的种子，确保实验可复现。

    必须在训练/评估脚本第一行调用：set_seed(cfg.seed)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        import dgl
        dgl.seed(seed)
    except ImportError:
        pass
