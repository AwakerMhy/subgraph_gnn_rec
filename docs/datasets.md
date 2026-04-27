# 数据集清单

> 创建时间：2026-04-10
> 最后更新：2026-04-27
>
> 收录社交/通信网络数据集，节点特征均以 `[in_degree, out_degree, total_degree]`（3维）占位。
> 时间戳仅用于初始图切分，不作为模型输入特征。

---

## 已预处理（可直接训练）

统一格式：`data/processed/<name>/edges.csv`（含 `src, dst, timestamp, timestamp_raw`）+ `meta.json`

### 小规模（N < 10k）

#### Email-EU

| 项目 | 内容 |
|------|------|
| 节点数 | 986 |
| 边数 | 24,929 |
| 有向 | ✅ |
| recip | 0.711 |
| deg_mean | 30.3 |
| 密度 | 2.57e-2 |
| 边语义 | 欧洲某研究机构内部邮件（核心子图），A→B 表示 A 发邮件给 B |
| 时间戳 | Unix 时间戳（真实） |
| 来源 | SNAP: email-Eu-core-temporal |
| 本地路径 | `data/processed/email_eu/` |

#### DNC Email

| 项目 | 内容 |
|------|------|
| 节点数 | 1,866 |
| 边数 | 37,421 |
| 有向 | ✅ |
| recip | 0.411 |
| deg_mean | 35.1 |
| 密度 | 1.08e-2 |
| 边语义 | 2016年民主党全国委员会邮件泄露，A→B 表示 A 发邮件给 B |
| 时间戳 | Unix 时间戳（真实） |
| 来源 | KONECT: dnc-temporalGraph |
| 本地路径 | `data/processed/dnc_email/` |

#### CollegeMsg

| 项目 | 内容 |
|------|------|
| 节点数 | 1,899 |
| 边数 | 59,835 |
| 有向 | ✅ |
| recip | 0.636 |
| deg_mean | 44.3 |
| 密度 | 1.66e-2 |
| 边语义 | 大学校园内部私信平台，A→B 表示 A 发消息给 B |
| 时间戳 | Unix 时间戳（真实） |
| 来源 | SNAP: CollegeMsg |
| 本地路径 | `data/processed/college_msg/` |
| 备注 | **主力实验基准**，规模小、密度高、时序性强 |

#### Wiki-Vote

| 项目 | 内容 |
|------|------|
| 节点数 | 7,115 |
| 边数 | 103,689 |
| 有向 | ✅ |
| recip | 0.056 |
| deg_mean | 17.0 |
| 密度 | 2.05e-3 |
| 边语义 | Wikipedia 用户对管理员晋升候选人的投票，A→B 表示 A 为 B 投票 |
| 时间戳 | 无真实时间戳（行序号作代理） |
| 来源 | SNAP: wiki-Vote |
| 本地路径 | `data/processed/wiki_vote/` |

#### Bitcoin-Alpha

| 项目 | 内容 |
|------|------|
| 节点数 | 3,783 |
| 边数 | 24,186 |
| 有向 | ✅ |
| recip | 0.832 |
| deg_mean | 7.4 |
| 密度 | 1.69e-3 |
| 边语义 | Bitcoin Alpha 平台用户信任评分，A→B 表示 A 给 B 评分（已过滤保留正向信任边） |
| 时间戳 | Unix 时间戳（真实） |
| 来源 | SNAP: soc-sign-bitcoin-alpha |
| 本地路径 | `data/processed/bitcoin_alpha/` |

#### Bitcoin-OTC

| 项目 | 内容 |
|------|------|
| 节点数 | 5,881 |
| 边数 | 35,592 |
| 有向 | ✅ |
| recip | 0.792 |
| deg_mean | 7.4 |
| 密度 | 1.03e-3 |
| 边语义 | Bitcoin OTC 平台用户信任评分，格式与 Bitcoin-Alpha 完全一致 |
| 时间戳 | Unix 时间戳（真实） |
| 来源 | SNAP: soc-sign-bitcoin-otc |
| 本地路径 | `data/processed/bitcoin_otc/` |

#### LastFM Asia

| 项目 | 内容 |
|------|------|
| 节点数 | 7,624 |
| 边数 | 55,612 |
| 有向 | ❌（原始无向，双向存储） |
| 密度 | 9.57e-4 |
| 边语义 | LastFM 亚洲地区用户间社交关注网络 |
| 时间戳 | 无真实时间戳（行序号作代理） |
| 来源 | PyG / SNAP |
| 本地路径 | `data/processed/lastfm_asia/` |

---

### 中规模（N 10k–100k）

#### sx-MathOverflow

| 项目 | 内容 |
|------|------|
| 节点数 | 24,759 |
| 边数 | 390,441 |
| 有向 | ✅ |
| recip | 0.351 |
| deg_mean | 23.7 |
| 密度 | 6.37e-4 |
| 边语义 | MathOverflow 问答社区，A→B 表示 A 回答/评论了 B 的帖子 |
| 时间戳 | Unix 时间戳（真实） |
| 来源 | SNAP: sx-mathoverflow |
| 本地路径 | `data/processed/sx_mathoverflow/` |

#### Slashdot

| 项目 | 内容 |
|------|------|
| 节点数 | 77,350 |
| 边数 | 516,575 |
| 有向 | ✅ |
| recip | 0.186 |
| deg_mean | 11.9 |
| 密度 | 8.64e-5 |
| 边语义 | Slashdot Zoo 科技社区 friend/foe 标记网络，A→B 表示 A 标记 B（sign 已丢弃） |
| 时间戳 | 无真实时间戳（行序号作代理） |
| 来源 | SNAP: soc-sign-Slashdot081106 |
| 本地路径 | `data/processed/slashdot/` |

#### Facebook Ego

| 项目 | 内容 |
|------|------|
| 节点数 | 4,039 |
| 边数 | 176,468 |
| 有向 | ❌（原始无向，双向存储） |
| 密度 | 1.08e-2 |
| 边语义 | Facebook ego 网络合并，好友关系 |
| 时间戳 | 无真实时间戳（行序号作代理） |
| 来源 | SNAP: ego-Facebook |
| 本地路径 | `data/processed/facebook_ego/` |
| 备注 | 密度偏高，与本项目稀疏社交网络假设差异较大 |

#### Epinions

| 项目 | 内容 |
|------|------|
| 节点数 | 75,879 |
| 边数 | 508,837 |
| 有向 | ✅ |
| recip | 0.405 |
| deg_mean | 8.4 |
| 密度 | 8.84e-5 |
| 边语义 | Epinions 产品评测平台用户信任网络，A→B 表示 A 信任 B |
| 时间戳 | 无真实时间戳（行序号作代理） |
| 来源 | SNAP: soc-Epinions1 |
| 本地路径 | `data/processed/epinions/` |

---

### 大规模（N > 100k）

#### sx-AskUbuntu

| 项目 | 内容 |
|------|------|
| 节点数 | 157,222 |
| 边数 | 726,661 |
| 有向 | ✅ |
| recip | 0.327 |
| deg_mean | 7.1 |
| 密度 | 2.94e-5 |
| 边语义 | AskUbuntu 问答社区，A→B 表示 A 回答/评论了 B 的帖子 |
| 时间戳 | Unix 时间戳（真实） |
| 来源 | SNAP: sx-askubuntu |
| 本地路径 | `data/processed/sx_askubuntu/` |

#### sx-SuperUser

| 项目 | 内容 |
|------|------|
| 节点数 | 192,409 |
| 边数 | 1,108,739 |
| 有向 | ✅ |
| recip | 0.327 |
| deg_mean | 8.1 |
| 密度 | 2.99e-5 |
| 边语义 | SuperUser 问答社区，A→B 表示 A 回答/评论了 B 的帖子 |
| 时间戳 | Unix 时间戳（真实） |
| 来源 | SNAP: sx-superuser |
| 本地路径 | `data/processed/sx_superuser/` |

#### Email-EuAll

| 项目 | 内容 |
|------|------|
| 节点数 | 265,009 |
| 边数 | 418,956 |
| 有向 | ✅ |
| recip | 0.260 |
| deg_mean | 1.9 |
| 密度 | 5.98e-6 |
| 边语义 | 欧洲某研究机构完整邮件网络（email_eu 的全量版），A→B 表示 A 发邮件给 B |
| 时间戳 | 无真实时间戳（行序号作代理） |
| 来源 | SNAP: email-EuAll |
| 本地路径 | `data/processed/email_euall/` |

#### Gowalla

| 项目 | 内容 |
|------|------|
| 节点数 | 196,591 |
| 边数 | 1,900,654 |
| 有向 | ❌（原始无向，双向存储） |
| 密度 | 4.92e-5 |
| 边语义 | Gowalla 签到应用社交好友网络 |
| 时间戳 | 无真实时间戳（行序号作代理） |
| 来源 | SNAP: loc-gowalla |
| 本地路径 | `data/processed/gowalla/` |

#### OGB-Collab

| 项目 | 内容 |
|------|------|
| 节点数 | 235,868 |
| 边数 | 1,935,264 |
| 有向 | ❌（原始无向，双向存储） |
| 密度 | 3.48e-5 |
| 边语义 | 学术论文合著网络，节点为作者，边为合著关系 |
| 时间戳 | 年份（整数，非精确） |
| 来源 | OGB: ogbl-collab |
| 本地路径 | `data/processed/ogbl_collab/` |
| 备注 | OGB 标准 benchmark，可与 SEAL 等基线对比 |

#### Twitch Gamers

| 项目 | 内容 |
|------|------|
| 节点数 | 168,114 |
| 边数 | 13,595,114 |
| 有向 | ❌（原始无向，双向存储） |
| 密度 | 4.81e-4 |
| 边语义 | Twitch 平台游戏主播间互相关注社交网络 |
| 时间戳 | 无真实时间戳（行序号作代理） |
| 来源 | SNAP / Rozemberczki et al. |
| 本地路径 | `data/processed/twitch_gamers/` |
| 备注 | 边数最多（1360万），在线仿真需大幅采样或仅取子图 |

---

## 已预处理子集

#### CollegeMsg-HighDeg

| 项目 | 内容 |
|------|------|
| 节点数 | 1,153 |
| 边数 | 18,855 |
| 有向 | ✅ |
| 来源 | CollegeMsg 高度数节点诱导子图 |
| 本地路径 | `data/processed/college_msg_highdeg/` |
| 备注 | 用于度分布偏斜影响实验 |

---

## 候选数据集（满足条件但待接入）

### 来自 TGB（Temporal Graph Benchmark）

> 安装方式：`pip install py-tgb`，数据加载器自动下载。官网：https://tgb.complexdatalab.com/

| 数据集 | 节点数 | 边数 | 边语义 | 社交相关性 |
|--------|--------|------|--------|----------|
| `tgbl-comment` | 994,790 | 44,314,507 | Reddit 用户互相回复，A→B 有向有时间戳 | ★★★ 高 |
| `tgbl-coin` | 638,486 | 22,809,486 | ERC20 稳定币地址间转账 | ★ 低（金融） |
| `tgbl-flight` | 18,143 | 67,169,570 | 机场间航班连接 | ✗ 非社交 |
| `tgbn-trade` | 255 | 468,245 | 国家间农业贸易 | ✗ 非社交 |

> `tgbl-wiki`、`tgbl-review`、`tgbn-genre`、`tgbn-reddit`、`tgbn-token` 均为 user-item 二部图，不满足同质要求，已排除。

### 来自 KONECT

> 官网：http://konect.cc/networks/

| 数据集 | 节点数 | 边数 | 边语义 | 社交相关性 |
|--------|--------|------|--------|----------|
| Wikipedia Talk | ~2.4M | ~5M | 用户讨论页互动，有向有时间戳 | ★★ 中 |
