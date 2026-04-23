"""src/baseline/tgat.py — TGAT baseline (simplified, subgraph framework)

原文：Xu et al. "Inductive Representation Learning on Temporal Graphs" (ICLR 2020)

本实现在子图框架内的适配：
  - 子图提取需设置 store_edge_time=True（在 train.py collate_fn 中控制）
  - 时间编码：可学习频率的余弦嵌入 Φ(dt) = [cos(w_1*dt), ..., cos(w_d*dt)]
  - 消息传递：每条边消息 = W_v(h_src || Φ(dt))；attention score = q·k/√d
  - Readout：最后一层全图 mean pooling
  - 若子图无 edata['dt']（空子图边数=0），退化为纯图注意力

接口与 LinkPredModel 完全相同（forward / forward_batch）。
"""
from __future__ import annotations

import math

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.scorer import Scorer


# ── Time Encoder ──────────────────────────────────────────────────────────────

class TimeEncoder(nn.Module):
    """可学习频率的余弦时间编码（原文 Section 2）。

    Φ(t) = [cos(w_1*t + b_1), ..., cos(w_d*t + b_d)]

    Args:
        time_dim: 输出维度
    """

    def __init__(self, time_dim: int = 32) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(time_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(time_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (...,) float — 时间差 dt

        Returns:
            (..., time_dim) float
        """
        return torch.cos(t.unsqueeze(-1) * self.w + self.b)


# ── TGAT Convolution Layer ─────────────────────────────────────────────────────

class TGATConv(nn.Module):
    """单层 TGAT 消息传递。

    Message:  m_e = W_v( concat(h_src, Φ(dt_e)) )
    Attention: a_e = softmax( q_dst · W_k(m_e) / √d )
    Output:   h_dst_new = ReLU( W_o( ∑_e a_e * m_e ) + W_self(h_dst) )

    Args:
        in_dim:   输入节点特征维度
        out_dim:  输出节点特征维度
        time_dim: 时间编码维度
    """

    def __init__(self, in_dim: int, out_dim: int, time_dim: int = 32) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.W_v = nn.Linear(in_dim + time_dim, out_dim, bias=False)
        self.W_k = nn.Linear(out_dim, out_dim, bias=False)
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.W_self = nn.Linear(in_dim, out_dim)
        self.scale = math.sqrt(out_dim)

    def forward(
        self,
        g: dgl.DGLGraph,
        h: torch.Tensor,
        time_enc: torch.Tensor,   # (E, time_dim)
    ) -> torch.Tensor:
        """
        Args:
            g:        DGLGraph（子图）
            h:        (N, in_dim) 节点特征
            time_enc: (E, time_dim) 边时间编码；空图时 shape=(0, time_dim)

        Returns:
            h_new: (N, out_dim)
        """
        with g.local_scope():
            if g.num_edges() == 0:
                # 无边：仅用自身变换
                return F.relu(self.W_self(h))

            g.ndata["h"] = h
            g.edata["te"] = time_enc  # (E, time_dim)

            def message_fn(edges):
                # 消息 = W_v(h_src || Φ(dt))
                msg = self.W_v(
                    torch.cat([edges.src["h"], edges.data["te"]], dim=-1)
                )  # (E, out_dim)
                # Key for attention
                key = self.W_k(msg)  # (E, out_dim)
                # Query from destination
                query = self.W_q(edges.dst["h"])  # (E, out_dim)
                # Attention logit
                attn_logit = (query * key).sum(-1, keepdim=True) / self.scale  # (E, 1)
                return {"msg": msg, "attn_logit": attn_logit}

            def reduce_fn(nodes):
                # Softmax attention over incoming messages
                logits = nodes.mailbox["attn_logit"]  # (N, deg, 1)
                attn = F.softmax(logits, dim=1)       # (N, deg, 1)
                agg = (attn * nodes.mailbox["msg"]).sum(1)  # (N, out_dim)
                return {"agg": agg}

            g.update_all(message_fn, reduce_fn)
            agg = g.ndata.get("agg", torch.zeros(g.num_nodes(), self.out_dim,
                                                  device=h.device, dtype=h.dtype))
            h_new = F.relu(self.W_o(agg) + self.W_self(h))
            return h_new


# ── TGAT Model ────────────────────────────────────────────────────────────────

class TGATModel(nn.Module):
    """TGAT 链接预测模型。

    Args:
        hidden_dim:        隐层维度
        num_layers:        TGAT 层数
        time_dim:          时间编码维度
        scorer_hidden_dim: Scorer MLP 隐层（None 则等于 hidden_dim）
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        time_dim: int = 32,
        scorer_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_encoder = TimeEncoder(time_dim)

        self.layers = nn.ModuleList()
        # 第一层：输入 2 维 one-hot
        self.layers.append(TGATConv(2, hidden_dim, time_dim))
        for _ in range(num_layers - 1):
            self.layers.append(TGATConv(hidden_dim, hidden_dim, time_dim))

        _scorer_hidden = scorer_hidden_dim if scorer_hidden_dim is not None else hidden_dim
        self.scorer = Scorer(in_dim=hidden_dim, hidden_dim=_scorer_hidden)

    def _make_feat(
        self, n: int, u_mask: torch.Tensor, v_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        feat = torch.zeros(n, 2, dtype=torch.float32, device=device)
        feat[u_mask, 0] = 1.0
        feat[v_mask, 1] = 1.0
        return feat

    def _get_time_enc(self, g: dgl.DGLGraph) -> torch.Tensor:
        """获取边时间编码。若子图无 edata['dt']（空边），返回全零。"""
        if g.num_edges() == 0:
            return torch.zeros(0, self.time_encoder.w.shape[0],
                               device=g.device, dtype=torch.float32)
        if "dt" in g.edata:
            return self.time_encoder(g.edata["dt"])
        # 无时间信息（未使用 store_edge_time），退化为全零时间编码
        return torch.zeros(g.num_edges(), self.time_encoder.w.shape[0],
                           device=g.device, dtype=torch.float32)

    def _encode(self, g: dgl.DGLGraph) -> torch.Tensor:
        u_mask = g.ndata["_u_flag"]
        v_mask = g.ndata["_v_flag"]
        h = self._make_feat(g.num_nodes(), u_mask, v_mask, g.device)
        te = self._get_time_enc(g)
        for layer in self.layers:
            h = layer(g, h, te)
        g.ndata["_h"] = h
        h_graph = dgl.mean_nodes(g, "_h")   # (1, hidden_dim)
        g.ndata.pop("_h")
        return h_graph.squeeze(0)            # (hidden_dim,)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        return self.scorer(self._encode(g))

    def forward_batch(self, bg: dgl.DGLGraph) -> torch.Tensor:
        graphs = dgl.unbatch(bg)
        embeddings = [self._encode(g) for g in graphs]
        h_graphs = torch.stack(embeddings)
        return self.scorer(h_graphs)
