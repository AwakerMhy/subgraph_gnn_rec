---
name: constant lr=1e-4 显著优于 cosine_warmup lr=0.001
description: GNN lr schedule消融结论：constant lr=1e-4 在几乎所有变体和数据集上优于 cosine_warmup lr=0.001，gnn_h32 是唯一例外
type: feedback
---

constant lr=1e-4 全面优于 cosine_warmup lr=0.001，建议作为 gnn/gnn_sum/gnn_concat 的新默认值。

**Why:** cosine_warmup lr=0.001 在小模型（hidden_dim=8）下造成严重欠拟合，warm up 阶段 lr 过低、decay 后期 lr 过小；constant 1e-4 提供稳定梯度信号。

**How to apply:**
- gnn / gnn_sum / gnn_concat：新配置统一用 `trainer.lr: 0.0001, trainer.scheduler.strategy: constant`
- gnn_h32：仍需观察，cosine 在 email_eu / sx_mathoverflow / college_msg 胜出，可尝试 lr=5e-5
- 对比数据：8 个数据集已完成（sx_superuser/advogato 待跑）

**胜负汇总（seed=0，8数据集）：**
- gnn: const 8/8，cosine 0/8，平均 +0.11 MRR@1
- gnn_sum: const 7/7，cosine 0/7，平均 +0.10 MRR@1
- gnn_concat: const 6/7，cosine 1/7，平均 +0.05 MRR@1
- gnn_h32: const 5/8，cosine 3/8，平均 +0.02 MRR@1

**典型提升案例：**
- email_eu gnn: +0.26（0.286→0.542）
- bitcoin_alpha gnn: +0.23（0.337→0.563）
- sx_mathoverflow gnn: +0.15（0.356→0.506）
- wiki_vote 全部变体：+0.04~+0.05（首次在该数据集观察到正向改善）
