# Kaggriculture Lab

本目录是本地实验室，不包含 Kaggle 入赛、凭据或提交命令。

## 当前阶段

- 使用官方 `kaggle-environments` 运行可复现的本地对局。
- 用内置 `starter`、`random` 与透明的作物策略验证得分采集。
- 使用 SHA 固定的公开 V45 衍生代码作为本地开发基线；所有实验改动必须做成独立 variant，不能直接改写冻结原件。

## 运行

```bash
.venv/bin/python -m lab.runner --left wheat --right starter --steps 120 --seed 7
```

当前开发基线的完整赛季 smoke test：

```bash
.venv/bin/python -m lab.runner --left v45_base --right starter --steps 720 --seed 7
```

完整赛季把 `--steps` 改为 `720`。本地分数只用于实验，不等同于 Kaggle 正式排行榜分数。

批量对局（固定连续随机种子）示例：

```bash
.venv/bin/python -m lab.tournament --left wheat --right starter --count 20
```

西瓜规则和其参数搜索已归档为历史探索，不能再作为默认策略或后续调参入口。所有本地终局现金仅能在相同环境、seed 和对手下比较，不能换算成 Kaggle 排行榜分数。

V45 变体必须先有单独协议并使用 development/holdout 分离。首个 R85 喂养系数实验已完成但未通过提升门槛；其命令和逐局结果分别在 [实验协议](docs/V45_FEED_VARIANT_PROTOCOL.md) 与 [结果](experiments/v45_feed_multiplier_v1.jsonl)。

后续主评测使用七个冻结公开对手的配对联赛；完整条件、门槛与可恢复运行命令见
[七对手联赛协议](docs/SEVEN_OPPONENT_LEAGUE_PROTOCOL.md)。
