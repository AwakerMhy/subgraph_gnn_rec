import sys
sys.path.insert(0, ".")

import dgl
import torch
from src.model.model import LinkPredModel

# 单图测试：5节点，u=0, v=1
src = torch.tensor([0, 1, 2, 3, 2])
dst = torch.tensor([2, 2, 3, 4, 4])
g = dgl.graph((src, dst))
g.ndata["_u_flag"] = torch.tensor([True, False, False, False, False])
g.ndata["_v_flag"] = torch.tensor([False, True, False, False, False])

model = LinkPredModel(hidden_dim=64, num_layers=2)
score = model(g)
print(f"单图 score: {score.item():.4f}  shape: {score.shape}")
assert score.shape == torch.Size([]), "score 应为标量"
assert 0.0 < score.item() < 1.0, "score 应在 (0,1)"

# batch 测试
g2 = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
g2.ndata["_u_flag"] = torch.tensor([True, False, False])
g2.ndata["_v_flag"] = torch.tensor([False, False, True])
bg = dgl.batch([g, g2])
scores = model.forward_batch(bg)
print(f"batch scores: {scores.detach()}  shape: {scores.shape}")
assert scores.shape == torch.Size([2]), "batch 应返回 (2,)"
assert all(0.0 < s.item() < 1.0 for s in scores), "所有 score 应在 (0,1)"

print("ALL PASSED")
