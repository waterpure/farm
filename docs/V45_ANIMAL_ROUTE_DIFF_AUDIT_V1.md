# V45 动物路线差异审计 v1

## 结论

`v45-animal-route-v1` 失败的直接原因已经定位：它用“最终动物购买数量是否更接近 70/30”挑 route，却忽略一条 route 是围绕**特定商店组合和市场路径**写成的完整生产—销售计划。

最严重的实例是 Boatlee 对手、seed 3：V45 原 route 7 是在 day 6 的前两家店为 `PIZZA_SHOP + YARN_STORE` 时选出的羊毛路线。selector 虽然也看见了 Yarn，但因当前 MILK 相对基础价的系数较高，仍把 MILK 排第一、WOOL 排第二，切到 route 123。结果从 day 20 开始牛奶价格和现金路径崩坏，终局少 `22,779`。

这不是“羊毛分得还不够多”的调参问题，而是当前模型把一个短期单品价格，错误地看得比已出现的单品商店需求和 route 的既定销售链更重要。

## 方法和可复现性

脚本 `lab/animal_route_diff_audit.py` 只读执行两件事：

1. 比较冻结 V45 raw tapes 中 route 7、105、108、123 从 day 6 至 day 26 的市场订单和田间动作；
2. 重放 v1 联赛中 Boatlee 的三个实际切换案例（seed 1、2、3，candidate 在左座），逐日记录现金、雇工、动物、作物、库存、市场订单和价格。

三次重放的终局现金必须与联赛账本一致，实际均严格复现：

| seed | 原 route → 123 | candidate − V45 |
| --- | --- | ---: |
| 1 | 108 → 123 | +606 |
| 2 | 105 → 123 | -4,572 |
| 3 | 7 → 123 | -22,779 |

原始可复算产物：[v45_animal_route_diff_audit_v1.json](/Users/pure_water/Desktop/竞赛/kaggriculture-lab/experiments/v45_animal_route_diff_audit_v1.json)。

## 三种切换的实际含义

### seed 3：Pizza + Yarn，错误地由羊毛转向牛奶

- 原 route 7 的商店来源是 `PIZZA_SHOP + YARN_STORE`；它的静态动物购买是 `COW 2 + SHEEP 9`。
- route 123 的商店来源是 `PIZZA_SHOP + PIZZA_SHOP`；静态购买为 `COW 5 + SHEEP 3 + GOOSE 3`。
- day 6 时，模型的照料版评分是 MILK `121.69`、WOOL `112.14`。WOOL 的日需求其实更高（Yarn 给 13，Pizza 的 milk 给 7），但 MILK 的当前价相对其基础价更高，压过了这个需求信号。
- 终局时 candidate 有 `EGG 3 / MILK 9 / WOOL 11`，V45 有 `MILK 6 / WOOL 17`。双方作物结构相同；差异就是动物路线及其触发的后续雇工/市场反应。
- day 20 candidate 比 V45 少 `7,975` 现金；此时 MILK 价格 candidate 为 `7`、V45 为 `63`。终局 MILK 是 `5` 对 `89`，现金差扩大为 `-22,779`。WOOL 在两边仍约为 `241` 对 `227`，Yarn 持续吸收羊毛，故重羊路线获利。

### seed 2：只多一头牛、少一头羊，仍会造成负反馈

- 原 route 105 的静态购买为 `COW 4 + SHEEP 4 + GOOSE 3`；route 123 为 `COW 5 + SHEEP 3 + GOOSE 3`。
- raw tape 的差异大多只在 day 6–11：例如 day 6 route 123 多买 2 小麦、少卖 2 羊毛；day 9 将原本 2 羊改为 1 牛 + 1 羊。day 11 后原始动作几乎相同。
- 但早期这一头牛/一头羊和现金差会改变 V45 的反应层、市场价格和实际出售量。day 15 仍只差 `+57`，day 20 变成 `-1,780`；终局 candidate 的 MILK 价格为 `1`、V45 为 `13`，终局少 `4,572`。

这说明即使“差一头动物”看似小，也不能忽略共享市场的价格反馈和 V45 的动态安全层。

### seed 1：两个牛奶向路线，只有很小的正值

route 108 的来源是 `ICE_CREAM_SHOP + PIZZA_SHOP`，route 123 是 `PIZZA_SHOP + PIZZA_SHOP`。实际动物数最终相同，主要差在早期小麦/羊毛销售时序，得到 `+606`。这只是一个小样本正值，不能抵消另外两种情形，更不能证明 route 123 普遍更好。

## 对下一步的约束

不能做以下事情：

- 不把 70/30 从 70/30 改成别的数试图掩盖失败；
- 不解除“并列即 abstain”以增加切换次数；
- 不把 route123 的少数正值单独保留后声称改进。

若要重新尝试，应把“商店需求能否吸收该 route **未来累计产量**”作为硬门，而不是只比较 day 6 当前价格或最终动物数量；且必须先以只读方式证明该门能排除 seed 3 这种 Pizza+Yarn 的错误切换，再运行新的变体。当前未创建 v2，也未获得其实施授权。
