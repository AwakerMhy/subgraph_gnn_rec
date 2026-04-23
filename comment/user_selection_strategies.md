# 用户选择（推荐对象抽取）策略改进方案

## 问题背景

当前实验设置中，每轮从 V 中**随机抽取 10% 的用户节点**作为本轮推荐对象。这一设计最大的问题是：**真实社交平台上用户活跃度是高度不均匀的**——少数重度用户每天登录、频繁互动，大量轻度用户偶尔上线。随机均匀抽样完全抹平了这一特征，导致模拟出的网络演化模式与真实场景存在系统性偏差。

以下给出 5 种递进式的改进方案，以及最终推荐的组合策略。

---

## 方案 1：度数正比抽样（Activity-Proportional Sampling）

**核心思想**：在当前可观测图 $G_t$ 中，度数越高的节点越"活跃"，被选为推荐对象的概率越大。

$$P_t(u) = \frac{d_t(u)^\alpha}{\sum_{v \in V} d_t(v)^\alpha}, \quad \alpha \in [0.5, 1.0]$$

- $\alpha = 0$ 退化为均匀随机抽样（当前方案）
- $\alpha = 1$ 线性正比于度数
- $\alpha \in (0,1)$ 做亚线性折衷，避免头部节点过度垄断

**优点**：
- 简单直观，直接利用已有信息
- 符合"活跃用户更频繁使用推荐功能"的直觉

**缺点**：
- 初始阶段 $G_0$ 只有 5% 的边，度数信号很弱且噪声大
- 可能加剧马太效应（高度节点被推荐更多 → 度数更高 → 被推荐更多）

---

## 方案 2：幂律活跃度分布（Power-Law Activity Model）

**核心思想**：为每个用户预先分配一个固有活跃度 $a_u$，从幂律分布中采样，每轮抽取概率正比于 $a_u$。

$$a_u \sim \text{Pareto}(\beta), \quad P(u) = \frac{a_u}{\sum_v a_v}$$

其中 $\beta \in [1.5, 2.5]$ 控制不均匀程度（$\beta$ 越小越不均匀）。

**实现要点**：
- 在实验开始前，为每个节点一次性采样 $a_u$，实验全程固定
- $\beta$ 可以参考真实社交平台的登录频率分布来标定（通常 $\beta \approx 2.0$）

**优点**：
- 从根本上建模了用户异质性，与大量实证研究一致
- 和 $G^*$ 的结构无关，不会引入信息泄露
- 理论分析友好（幂律假设在 DCSBM 框架下天然契合度异质性参数 $\theta_i$）

**缺点**：
- 活跃度完全静态，不能反映"用户因收到好推荐而变活跃"的正反馈

---

## 方案 3：事件驱动抽样（Event-Triggered Sampling）

**核心思想**：用户在上一轮获得了新的连边（被推荐且"接受"），则本轮"回来"的概率显著提升。模拟的是"收到新好友通知 → 打开 App → 再次使用推荐功能"的行为链。

$$P_t(u) \propto a_u \cdot \left(1 + \gamma \cdot \mathbb{1}[\text{u 在 round } t\!-\!1 \text{ 获得新边}]\right)$$

- $\gamma > 0$ 控制事件触发的增强强度，建议 $\gamma \in [1, 3]$
- 可以泛化为最近 $w$ 轮内获得新边的次数：$1 + \gamma \cdot \sum_{s=t-w}^{t-1} n_s(u)$

**优点**：
- 引入了"推荐效果 → 用户留存"的正反馈回路
- 这是真实推荐系统中最核心的动态之一

**缺点**：
- 与推荐效果耦合，可能让分析变复杂
- 需要额外记录每轮每用户的边变化

---

## 方案 4：时间衰减抽样（Temporal Decay Sampling）

**核心思想**：用户上次"活跃"（被抽到或获得新边）距现在越久，被选中的概率越低，模拟用户流失。

$$P_t(u) \propto a_u \cdot \exp\left(-\lambda (t - t_{\text{last}}(u))\right)$$

- $t_{\text{last}}(u)$：节点 $u$ 上一次被选中或获得新边的时刻
- $\lambda$ 控制衰减速率，$\lambda = 0$ 退化为纯固有活跃度

**优点**：
- 自然建模了用户流失与回归
- 使模拟网络中出现"沉默用户"群体

**缺点**：
- 衰减过快可能导致部分节点永久沉默，降低图覆盖率收敛速度

---

## 方案 5（推荐）：综合策略（Composite Strategy）

将上述机制融合为一个统一的抽样公式：

$$\boxed{P_t(u) \propto \underbrace{a_u}_{\text{固有活跃度}} \cdot \underbrace{\left(\frac{d_t(u) + 1}{d_{\max} + 1}\right)^{\alpha}}_{\text{度数信号}} \cdot \underbrace{\exp\left(-\lambda(t - t_{\text{last}}(u))\right)}_{\text{时间衰减}} \cdot \underbrace{\left(1 + \gamma \cdot n_{\text{recent}}(u)\right)}_{\text{事件触发}}}$$

### 推荐超参数设置

| 参数 | 含义 | 推荐值 | 说明 |
|---|---|---|---|
| $\beta$ | 幂律指数 | 2.0 | 控制 $a_u$ 分布的不均匀度 |
| $\alpha$ | 度数权重指数 | 0.5 | 亚线性，防止马太效应过强 |
| $\lambda$ | 时间衰减率 | 0.1 | 约 10 轮不活跃后概率降为 ≈37% |
| $\gamma$ | 事件触发增益 | 2.0 | 获得新边的用户下轮概率 ×3 |
| $w$ | 回看窗口 | 3 轮 | $n_{\text{recent}}$ 统计最近 3 轮新增边数 |
| 每轮抽取比例 | — | 5%–15% | 可随轮次动态调整 |

---

## 与理论分析的衔接

这个综合策略对之前设计的理论框架有直接帮助：

1. **Theorem 4（推荐驱动网络增长）**：事件触发 + 度数正比抽样直接建模了"推荐产生正反馈加速增长"的机制，可以在 RDPG 框架下推导出更精确的超线性增长阶数。

2. **Theorem 5（马太效应量化）**：$\alpha$ 参数和 $\gamma$ 参数直接控制了"富者越富"的强度，可以在 DCSBM 下分析 $\alpha, \gamma$ 对度分布基尼系数的影响。

3. **消融实验设计**：综合策略天然支持消融——逐项关闭各因子（设为 0 或 1），可以量化每个机制对图覆盖率和推荐准确率的边际贡献。

---

## 实现伪代码

```python
import numpy as np

class UserSelector:
    def __init__(self, N, beta=2.0, alpha=0.5, lam=0.1, gamma=2.0, w=3):
        self.N = N
        self.alpha = alpha
        self.lam = lam
        self.gamma = gamma
        self.w = w
        
        # 固有活跃度：从 Pareto 分布采样（一次性）
        self.a = (np.random.pareto(beta, size=N) + 1)
        self.a /= self.a.sum()  # 归一化
        
        # 状态跟踪
        self.t_last = np.zeros(N)           # 上次活跃时刻
        self.recent_edges = np.zeros((N, w)) # 最近 w 轮新增边数
    
    def select(self, t, degrees, sample_ratio=0.10):
        """
        Args:
            t: 当前轮次
            degrees: 当前 G_t 中每个节点的度数 (shape: [N,])
            sample_ratio: 本轮抽取比例
        Returns:
            selected_indices: 被选中的用户节点索引
        """
        d_max = degrees.max() + 1
        
        # 度数因子
        degree_factor = ((degrees + 1) / d_max) ** self.alpha
        
        # 时间衰减因子
        time_factor = np.exp(-self.lam * (t - self.t_last))
        
        # 事件触发因子
        n_recent = self.recent_edges.sum(axis=1)
        event_factor = 1 + self.gamma * n_recent
        
        # 综合权重
        weights = self.a * degree_factor * time_factor * event_factor
        probs = weights / weights.sum()
        
        # 无放回采样
        k = max(1, int(self.N * sample_ratio))
        selected = np.random.choice(self.N, size=k, replace=False, p=probs)
        
        # 更新状态
        self.t_last[selected] = t
        
        return selected
    
    def update_edges(self, t, new_edge_counts):
        """每轮结束后更新最近新增边记录"""
        col = t % self.w
        self.recent_edges[:, col] = new_edge_counts
```

---

## 实验对比建议

建议在论文中对比以下配置：

| 配置 | 描述 | 预期特征 |
|---|---|---|
| Baseline | 均匀随机 10% | 图覆盖均匀但不真实 |
| PowerLaw-Only | 仅固有活跃度 | 异质性高，但无动态反馈 |
| Degree-Only | 仅度数正比 | 马太效应明显 |
| **Composite（推荐）** | 综合策略 | 最接近真实，可消融分析 |

**核心评估指标**：
- 图覆盖率随轮次的增长曲线
- 度分布的基尼系数演化
- 聚类系数与 $G^*$ 的差距收敛速度
- GNN 模型 Precision@K 在不同策略下的表现

---

## 总结

建议直接采用**综合策略**作为默认实验配置，同时保留"均匀随机"作为 baseline 进行对比。这不仅提升了模拟真实度，还为论文增加了一个有分析价值的实验维度——**"用户选择机制对推荐系统演化的影响"**。
