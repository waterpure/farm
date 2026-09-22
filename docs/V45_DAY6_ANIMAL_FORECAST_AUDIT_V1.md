# V45 day-6 动物产量预测审计 v1

## 目的

把“每种紧俏货做到已知购买量 70%”从赛后规则检查推进到 day 6 可用的输入。第一步只预测动物：读取 day 6 已有动物和候选 raw route 后续写明的 `PLACE`，按官方产出周期、每天 FEED+CARE、及时收货计算一个乐观上界；不选 route、不发动作。

## 结果

可复算输出：[v45_day6_animal_forecast_audit_v1.json](/Users/pure_water/Desktop/竞赛/kaggriculture-lab/experiments/v45_day6_animal_forecast_audit_v1.json)。在三个 Boatlee 案例中：

| seed | route | 预测牛奶 | 实际牛奶 | 预测羊毛 | 实际羊毛 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | V45 108 | 246 | 248 | 166 | 85 |
| 1 | candidate 123 | 270 | 248 | 144 | 85 |
| 2 | V45 105 | 246 | 233 | 166 | 145 |
| 2 | candidate 123 | 270 | 248 | 144 | 127 |
| 3 | V45 7 | 192 | 179 | 284 | 369 |
| 3 | candidate 123 | 270 | 228 | 144 | 235 |

它已能识别重要方向：seed 3 的候选 route123 会把牛奶推到约 270，远高于 Pizza 已知购买量 168 的 70%目标约118；原 route7 的牛奶约192。但它还不能可靠估计羊毛，route7 实际 369、预测只有284，route123 实际235、预测144。

## 为什么不能直接用

v1 只读取 raw route 的 `PLACE`。V45 还包含条件性羊群扩张等 overlay；它们不完全写在 raw route 中，会在价格、店铺、土地和资金条件满足时额外买羊、雇人、喂养和收羊毛。v1 因此漏掉了重要羊毛生产；同时假设每只动物每天都能 FEED+CARE 且及时收获，也会高估不能完全执行的链条。

当前结论：v1 只能作为“候选是否明显会把某种动物推得过多”的预警，**不能作为路线选择器或 70%目标的唯一预测来源**。

后续 overlay 分类已经完成，详见 [V45_OVERLAY_CLASSIFICATION_AUDIT_V1.md](/Users/pure_water/Desktop/竞赛/kaggriculture-lab/docs/V45_OVERLAY_CLASSIFICATION_AUDIT_V1.md)：三局在第 6 天均为“未来不确定”，而完整回放分别出现了番茄触发、两者均不触发、羊群触发三种结果。故 v1 的羊毛点预测不能靠固定补偿修复。

下一道未完成的只读研究门：若继续预测，应建立保守/条件上行两个情景，并明确上行情景依赖第 12/18 天尚未知的店铺、价格、现金和土地条件；必须先验证它不把上行情景误当确定产量，才可讨论任何新的 selector。
