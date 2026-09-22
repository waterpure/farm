# 七对手配对联赛协议 v1

状态：2026-09-17 经用户确认，作为后续 V45 变体的唯一主评测基本盘。

## 目标

评估一个具名 V45 variant 相对冻结 `v45_base` 的增量，而不是用绝对现金或
“能否打赢 starter”推断强度。所有公开代码只作本地对手，不构成可提交产物。

## 固定条件

- 外部对手等权：`sdy2842`、`v46`、`indar_top10`、`kaito_v58`、
  `nathan_pipe7`、`boatlee_v16`、`indar_pasture`。
- 每个条件为 `对手 × seed × candidate 所在座位`；每个 seed 必须左右各一次。
- 每个条件运行两局：variant 对该对手、冻结 V45 对同一对手。主指标为两局中
  本方终局现金的差 `variant_cash - v45_base_cash`。
- development seeds：`0, 1, 2, 3`，共 56 个配对条件、112 局。
- holdout seeds：`1000–1007`，共 112 个配对条件、224 局。只有 development
  中冻结的单一候选才可进入，禁止据 holdout 再调代码或参数。
- 运行器每次最多追加 8 个缺失局，并在**每局完成时立即**追加 JSONL；可中断后
  重跑，不会因一批尚未结束而丢失已经完成的局。
- 同一 `candidate/split` 若已有重复条件 key，运行器会拒绝继续追加，而不是静默
  覆盖或重复计分；保留旧账本作审计，并为新评测使用新的输出路径。

## 门槛和解释

一份 split 的最小通过门是：所有配对条件齐全且均 `DONE/DONE`、七对手等权
平均差为正、至少 4/7 对手的平均差非负。它只决定是否值得查看下一阶段，
不是“正式榜单提高”或“可获奖”的证明。任何很小、方向不一致或由单一对手
主导的增益，都必须视为不充分并归档，不得与其他未验证改动叠加。

若评测的 candidate 本身也是七个固定对手之一（当前 V46 晋级审计就是如此），
镜像条件仍保留在七对手等权主表中，但晋级还要求**去除该同名对手后**的平均差
仍为正。这样不能靠“自己对自己”的市场互动单独通过。汇总中的
`promotion_gate_met` 表示同时满足这项额外条件；仍须经 holdout 和人工审阅。

## 使用

先用冻结 V45 自校准：

```bash
.venv/bin/python -m lab.league --candidate v45_base --split development
```

重复同一命令直到 `remaining_games` 为零。它应产生零差分，验证配对/断点逻辑；
不是策略实验。具名变体（例如历史 feed 变体）可按相同方式运行：

```bash
.venv/bin/python -m lab.league --candidate v45-feed-m1.15 --split development
```

development 结果固定候选后，才允许改为 `--split holdout`。每次运行的局数可用
`--max-games` 调小，以适配本机单次命令时限。
