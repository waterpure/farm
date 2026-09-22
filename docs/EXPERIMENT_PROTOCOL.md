# 第一轮规则搜索协议

## 假设

在 15 格西瓜田的生产路线不变时，分批卖货数量、最低可接受价格和 shed 紧急清仓阈值会影响对共享市场中不同对手的最终现金。

## 候选空间

候选仅允许修改：

- `sell_batch`：正常条件下单次卖出的西瓜数量；
- `min_price_ratio`：当前西瓜价相对基准价 250 的最低卖货阈值；
- `emergency_shed_units`：达到此库存时不顾价格清仓。

生产路线、地块数、雇工数和种植动作保持不变。这保证本轮结果只能解释为“卖货规则”的差异，不能把路线差异混入结论。

## 对手和数据隔离

- 对手池：官方 `starter`、单格 `melon`、公开 V45 本地参考。
- development seeds：`0, 1`，只用于挑选候选。
- holdout seeds：`1000, 1001, 1002`，在选出候选前不参与调参。
- 每个候选与每个对手、每个 seed 都完整跑 720 回合。

## 选择规则

1. 在 development 的所有对手/seed 上，按候选自身平均终局现金排序。
2. 只让前两名进入 holdout；不得因为观察到 holdout 结果而回头修改其参数。
3. 报告 holdout 的平均终局现金、平均 margin、胜负数、逐局 telemetry 和异常状态。
4. 一轮 holdout 只能证明在这套小对手池的局部稳健性，不能证明正式榜单排名或奖牌概率。

## 输出

`experiments/melon_search_v1.jsonl`：每行一局，含参数、split、对手、seed、双方 reward、margin、耗时、状态与策略 telemetry。执行环境的单命令时限较短，因此先运行 `--stage development`，读取并固定前两名后再用 `--stage holdout --selected ...`；每阶段结束立即落盘。
