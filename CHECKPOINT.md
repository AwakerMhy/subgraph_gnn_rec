# CHECKPOINT — 会话恢复点

> 创建时间：2026-04-08 15:30
> 最后更新：2026-04-08 18:00

---

## 最近一次存档

- **存档时间**：2026-04-08 18:00
- **当前阶段**：Phase 4 完成，端到端跑通
- **完成状态**：Phase 1 ~ Phase 4 全部完成

### 完成情况

- [x] Phase 1：数据基础设施（split, dataset, generators, metrics, logger）
- [x] Phase 2：子图提取（subgraph, labeling, negative_sampling）
- [x] Phase 3（简化模型）：
  - src/model/gin_encoder.py：GIN（2维one-hot → mean pooling）
  - src/model/scorer.py：MLP 评分头
  - src/model/model.py：forward / forward_batch
- [x] Phase 4：
  - src/train.py：训练循环（早停、checkpoint、train.json）
  - src/evaluate.py：测试集评估（AUC/AP/Hits@K）
  - 端到端跑通（SBM 合成数据，200节点71边）

### 已知问题与修复

- subgraph.py 空子图分支原用 `_local_u`/`_local_v`，与模型期望的 `_u_flag`/`_v_flag` 不一致 → 已修复
- `conda run` 在 Windows GBK 终端下因 requests 警告含 unicode 导致崩溃 → 改用直接调用 `C:/conda/envs/gnn/python.exe` + `PYTHONIOENCODING=utf-8`

### 运行命令

```bash
# 生成数据
PYTHONIOENCODING=utf-8 PYTHONPATH=. C:/conda/envs/gnn/python.exe tests/gen_sbm_data.py

# 训练
PYTHONIOENCODING=utf-8 PYTHONPATH=. C:/conda/envs/gnn/python.exe src/train.py \
  --data_dir data/synthetic/sbm --run_name sbm_test \
  --epochs 20 --batch_size 16 --hidden_dim 32 --num_layers 2 --patience 5

# 评估
PYTHONIOENCODING=utf-8 PYTHONPATH=. C:/conda/envs/gnn/python.exe src/evaluate.py \
  --data_dir data/synthetic/sbm --ckpt results/checkpoints/sbm_test_best.pt \
  --hidden_dim 32 --num_layers 2 --hits_neg 49
```

### 环境

- conda 环境：`gnn`（python=3.10）
- torch：2.11.0+cu128，DGL：2.0.0+cu121
- scikit-learn 已安装

### 下一步行动

- 用更大的数据集测试（调大 SBM 参数或换真实数据集）
- 分析当前模型表现（AUC ~0.55，接近随机，符合预期——模型极简）
- 可考虑接入真实数据（CollegeMsg 等）

---
