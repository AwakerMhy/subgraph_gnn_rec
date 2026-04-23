# 数据集清单

> 创建时间：2026-04-10
> 最后更新：2026-04-10
>
> 仅收录**有向、有时间戳、节点同质**的社交/通信网络数据集。
> 无时间戳的数据集（wiki-talk、email-enron、email-euall、soc-sign-epinions）已排除。

---

## 已预处理（可直接训练）

### CollegeMsg

| 项目 | 内容 |
|------|------|
| 节点数 | 1,899 |
| 边数 | 59,835 |
| 边语义 | 用户私信，A→B 表示 A 发消息给 B |
| 时间戳 | Unix 时间戳 |
| 下载地址 | https://snap.stanford.edu/data/CollegeMsg.html |
| 本地路径 | `data/processed/college_msg/` |

### Bitcoin-OTC

| 项目 | 内容 |
|------|------|
| 节点数 | 5,881 |
| 边数 | 35,592 |
| 边语义 | 比特币交易平台信任评分，A→B 表示 A 对 B 的信任评分（已过滤保留正向边） |
| 时间戳 | Unix 时间戳 |
| 下载地址 | https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html |
| 本地路径 | `data/processed/bitcoin_otc/` |

### Email-EU

| 项目 | 内容 |
|------|------|
| 节点数 | 986 |
| 边数 | 24,929 |
| 边语义 | 欧洲某研究机构内部邮件，A→B 表示 A 发邮件给 B |
| 时间戳 | Unix 时间戳 |
| 下载地址 | https://snap.stanford.edu/data/email-Eu-core-temporal.html |
| 本地路径 | `data/processed/email_eu/` |

---

## 已下载（待预处理）

### sx-mathoverflow

| 项目 | 内容 |
|------|------|
| 节点数 | 24,818 |
| 边数 | 506,550 |
| 边语义 | MathOverflow 问答社区，A→B 表示 A 回答/评论了 B 的帖子 |
| 时间戳 | Unix 时间戳 |
| 下载地址 | https://snap.stanford.edu/data/sx-mathoverflow.html |
| 本地路径 | `data/raw/sx-mathoverflow/sx-mathoverflow.txt` |

### sx-askubuntu

| 项目 | 内容 |
|------|------|
| 节点数 | 159,316 |
| 边数 | 964,437 |
| 边语义 | AskUbuntu 问答社区，A→B 表示 A 回答/评论了 B 的帖子 |
| 时间戳 | Unix 时间戳 |
| 下载地址 | https://snap.stanford.edu/data/sx-askubuntu.html |
| 本地路径 | `data/raw/sx-askubuntu/sx-askubuntu.txt` |

### sx-superuser

| 项目 | 内容 |
|------|------|
| 节点数 | 194,085 |
| 边数 | 1,443,339 |
| 边语义 | SuperUser 问答社区，A→B 表示 A 回答/评论了 B 的帖子 |
| 时间戳 | Unix 时间戳 |
| 下载地址 | https://snap.stanford.edu/data/sx-superuser.html |
| 本地路径 | `data/raw/sx-superuser/sx-superuser.txt` |

---

## 待下载

### Bitcoin-Alpha

| 项目 | 内容 |
|------|------|
| 节点数 | ~3,783 |
| 边数 | ~24,186 |
| 边语义 | 比特币交易平台信任评分，与 Bitcoin-OTC 同类型，可用于对比 |
| 时间戳 | Unix 时间戳 |
| 下载地址 | https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html |
| 本地路径 | `data/raw/soc-sign-bitcoin-alpha/` |
| 备注 | 格式与 Bitcoin-OTC 完全一致（src, dst, rating, timestamp），可复用现有预处理脚本 |

---

## 候选数据集（满足条件但待评估是否接入）

### 来自 TGB（Temporal Graph Benchmark）

> 安装方式：`pip install py-tgb`，数据加载器自动下载。官网：https://tgb.complexdatalab.com/

| 数据集 | 节点数 | 边数 | 边语义 | 社交相关性 |
|--------|--------|------|------|----------|
| `tgbl-comment` | 994,790 | 44,314,507 | Reddit 用户互相回复，A→B 有向有时间戳 | ★★★ 高 |
| `tgbl-coin` | 638,486 | 22,809,486 | ERC20 稳定币地址间转账，A→B 有向有时间戳 | ★ 低（金融） |
| `tgbl-flight` | 18,143 | 67,169,570 | 机场间航班连接，A→B 有向有时间戳 | ✗ 非社交 |
| `tgbn-trade` | 255 | 468,245 | 国家间农业贸易，A→B 有向有时间戳 | ✗ 非社交 |

> 注意：`tgbl-wiki`、`tgbl-review`、`tgbn-genre`、`tgbn-reddit`、`tgbn-token` 均为 user-item 二部图，**不满足同质要求**，已排除。

### 来自 KONECT

> 官网：http://konect.cc/networks/

| 数据集 | 节点数 | 边数 | 边语义 | 下载地址 | 社交相关性 |
|--------|--------|------|------|---------|----------|
| DNC Email | 1,900 | 37,400 | 2016年民主党邮件泄露，用户间有向邮件，有时间戳 | http://konect.cc/networks/dnc-temporalGraph/ | ★★ 中 |
