"""src/recall — 模拟召回模块（Step 4）

提供轻量启发式召回器，为每个源节点 u 在观测图 G_obs 上生成候选集 C(u)。
候选集用于构造精排模型的训练/评估样本，替代原始的全图随机负采样。

召回策略梯度（由弱到强，难度逐步提升）：
    1. common_neighbors  — |{z: z∈N_out(u) ∧ v∈N_out(z)}|（本版本实现）
    2. adamic_adar       — Σ 1/log(|N_out(z)|+1)（本版本实现）
    3. node2vec          — 浅层 embedding 余弦相似度（留作扩展点）
    4. gcn_single        — 单层 GCN 嵌入（留作扩展点）
"""
from src.recall.base import RecallBase
from src.recall.curriculum import CurriculumScheduler
from src.recall.heuristic import AdamicAdarRecall, CommonNeighborsRecall
from src.recall.registry import build_recall

__all__ = [
    "RecallBase",
    "CommonNeighborsRecall",
    "AdamicAdarRecall",
    "build_recall",
    "CurriculumScheduler",
]
