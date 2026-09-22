# Kaggriculture benchmark 候选调研（2026-09-17）

目标是建立**多样的本地对手池**，不是把每个标题写着“2900”的 Notebook
都当成强 agent。候选需要区分：公开/历史 rating 证据、作者自报本地结果、
代码血缘，以及能否在本机独立加载和完整跑完。

## 当前榜单背景

2026-09-17 Kaggle 公共榜前 20 的分数区间约为 `2947.6–3184.1`。
这不等于这些队伍公开了代码，但足以说明先前常被引用的 `2650–2857`
历史数值不应被当成“当前顶级”或奖牌保证。

## 第一批应审计/加载的候选

| 候选 | 公开证据 | 血缘与作用 | 本地池状态 |
|---|---|---|---|
| [sdy623 / 2842](https://www.kaggle.com/code/jaxa623/2842-two-identical-agents-90-points-apart) | 作者展示同字节两条 entry：`2857` 和 `2786`，并说明约 90 rating 是其测得的榜单噪声 | V43 衍生、市场顺序/开局攻击/提前卖货；有可校验 hash，适合高强度同族对手 | **优先加载** |
| [Ahmed V46](https://www.kaggle.com/code/ahmedberatozer/kaggriculture-v46-first-turn-microstructure-and-s) | 自报 32 个 untouched worlds、19 rival 的本地对局统计；未把它声称为正式 rating | V45 的后继同族、最新公开强版本，最直接检验我们是否落后当前同族演化 | **优先加载** |
| [Indar / Rank Top10](https://www.kaggle.com/code/indarkarhana/rank-top10-read-the-market-choose-the-farm) | 标题声称 Top10；发布日期较早，需视为历史证据 | 不显式依赖 V43/V45；基于公开 replay 重建的路线/需求选择，提供生产路线多样性 | **优先加载** |
| [Kaito V58 Minimax](https://www.kaggle.com/code/kaitofukami/238-238-known-streams-v58-minimax-closed-loop) | 对公开 route family 的 minimax/双座测试；非当前官方 rating | 根据公开状态在若干检查点选 continuation，是检验“适应性路线”弱点的多样对手 | **优先加载** |
| [Nathan Pipe-7](https://www.kaggle.com/code/nathanjacob/kaggriculture-pipe-7-wheat-microstructure) | 自报 10-agent、900 局 tournament `179-1-0`；不是官方榜证据 | V43/V45 衍生、针对小麦开局微结构；适合作为 opening exploit 压力测试 | **已完成初筛** |
| [nusrati 2715.6](https://www.kaggle.com/code/nusrati/2715-6) | 标题为 `2715.6`，正文强调 replay/tape 结果不能代表隐藏榜单 | V43/V45 痕迹明显；若源码 hash 与已有同族不同，可作 replay-driven 对手 | **第二批，先去重** |
| [Dmitrii 7-turn rescue](https://www.kaggle.com/code/dmitriigluzdov/kaggriculture-7-turn-rescue-historical-lb-2800) | 标题称 Historical LB 2800+，正文谨慎地只报告小型 paired local panel | 终局实体货物/送达 rescue specialist，不应被当总分强基线 | **第二批 specialist** |
| [Tetsutani Market-Smart](https://www.kaggle.com/code/tetsutani/market-smart-farming-kaggriculture) | 无可核查的高 rating 声明；被 V46 attribution 引用 | 市场/仓库保护机制来源，可能与 V45 同族重叠 | **源码审计后决定** |

## 暂不作为“高分强对手”的代码

- [Salem 2900+](https://www.kaggle.com/code/salemali7/kaggriculture-2900)：标题为 `2900+`，正文明确其结果是本地模拟，不能视为官方分数。
- [Pure Architecture 2600+](https://www.kaggle.com/code/saitejabandaruin/kaggriculture-pure-architecture-2600-elo-v3)：主要是未佐证的宣传性声明，缺少可审计的比赛/对局证据。
- V45 cloning agent、V38/V39/V41/V43/V44 等同族复刻：不应各占一个 opponent slot；需源码 hash/动作轨迹去重后只留代表版本。
- 2600 farms/meta、top10 replay archive、经济可视化等分析 Notebook：可作为研究资料，不是可运行对手。

## 已下载的第一批快照（2026-09-17）

以下四份 `main.py` 已按作者当前公开 Notebook 版本下载至 `third_party/`，
没有修改源码、制作提交包或改变 Kaggle 远端状态。它们均通过
`py_compile`；完整环境、可复现性和行为去重门仍待执行，所以暂时只能称为
**候选本地对手**，不能称“已通过对手池”。

| 本地名称 | 路径 | SHA-256 | 已知 entry point |
|---|---|---|---|
| `sdy2842` | `third_party/sdy2842/main.py` | `944aa64c…20a56` | `agent` |
| `v46` | `third_party/v46/main.py` | `735c3703…cedb6` | `agent` |
| `indar_top10` | `third_party/indarkarhana_top10/main.py` | `d39dba50…9f2e` | `agent` |
| `kaito_v58` | `third_party/kaito_v58/main.py` | `b041058e…ddcb9` | `kaggle_agent_v58` |
| `nathan_pipe7` | `third_party/nathan_pipe7/main.py` | `6150b7f9…4a200` | `agent` |
| `boatlee_v16` | `third_party/boatlee_v16/main.py` | `f029fa0c…c4d19` | `agent` |
| `indar_pasture` | `third_party/indar_pasture/` | archive `fa9e7eb1…5f98f` | `kaggriculture_e776_agent` |

V46 的 hash 已与作者 `v46_manifest.json` 一致；其余三个 hash 已与各自
Notebook 公开 log 一致。每个目录的 README 保存来源边界。加载器会每局重载
模块，避免第三方全局 route cursor/telemetry 泄漏到下一局。

初筛记录在 `experiments/benchmark_pool_screen_v1.json`：四者均在本机
`kaggle-environments==1.32.7` 完成了 720 回合，seed 7 对 starter 的双跑
奖励完全一致，且四份动作轨迹 hash 不同。再以 seeds 7、8 和双座位对 V45
进行 4 局小筛：`sdy2842` 平均 margin `+2,049.5`，`v46` `+2,248.0`，
`indar_top10` `-39,754.0`，`kaito_v58` `-18,248.5`，均 `DONE/DONE`。
这只说明它们可作为多样性压力对手，**样本量绝不支持强度排序或正式分数推断**。
Kaito 原 Notebook 使用 `1.29.3`，但在本机 `1.32.7` 的这一兼容筛查通过；
仍需在后续固定协议中留意耗时和跨版本差异。

第二批已完成相同的初筛，记录于 `experiments/benchmark_pool_screen_v2.json`。
`nathan_pipe7`、`boatlee_v16`、`indar_pasture` 全部完成 720 回合且 seed 7
双跑一致；三份 starter action trace hash 均不同，也不同于 V45。两个 seed、
双座位的四局 V45 筛查 margin 分别是 `+234.0`、`-40,631.5`、`-16,759.0`，
全部 `DONE/DONE`。Pipe-7 与 V45 对 starter 的终局现金相同，但 action trace
不同、对 V45 小幅领先，因此应归为近亲市场压力对手而非源码/行为重复。
这些数值仍仅是兼容性筛查，绝不是强度排名。

## 加载门槛与编排

任何候选进入 `lab` 前必须依次满足：

1. 记录 Kaggle ref、版本、作者、声明性质（正式/历史/本地）和许可证/attribution。
2. 下载/重建为不可变的 `third_party/<name>/main.py`，保存 SHA-256；不能直接覆盖 V45。
3. 本机 `py_compile`、单局 720 turn、固定 seed 双次一致性均通过；记录动作错误和耗时。
4. 与 `v45_base` 和不同血缘的候选各跑小批量，检查是否只是同一源码或近乎相同轨迹。
5. 通过后才进入固定联赛；联赛同时至少覆盖：同族强版、不同生产路线、适应性/route-switch、终局 specialist。

调研已进入“已通过初始本地兼容筛查、尚未加入固定联赛”的阶段。下一步须先
预注册 development/holdout 的对手权重、seed 和晋级条件，再把它们用于任何
V45 variant 的结论。
