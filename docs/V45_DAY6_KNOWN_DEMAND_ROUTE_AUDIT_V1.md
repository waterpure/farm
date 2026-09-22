# V45 第 6 天“只看当前店铺”路线审计 v1

## 问题

用户要求的规则是：第 6 天只根据已经开门的店，选择当下最好的种植/养殖；未来随机店铺不预测、不押注。为避免又凭印象改 V45，本审计先逐个列出 V45 在每一种可见两店组合时实际选哪条 route，以及该 route 写死的作物/动物承诺。

结果在 [v45_day6_known_demand_route_audit_v1.json](/Users/pure_water/Desktop/竞赛/kaggriculture-lab/experiments/v45_day6_known_demand_route_audit_v1.json)，代码为 [day6_known_demand_route_audit.py](/Users/pure_water/Desktop/竞赛/kaggriculture-lab/lab/day6_known_demand_route_audit.py)。它不运行对局、不改 V45，也不读取未来店铺、未来价格、条件扩张或对手私有状态。

## 最重要的发现

V45 本来就在第 6 天按**当前可见的两家店**选 route；并且只要当前已经有 Yarn Store，它会使用专门的 Yarn 分支，而不是沿用普通路线。

| 当前店铺 | V45 route | 后续原始动物摆放 |
| --- | ---: | --- |
| Pizza + Yarn | 7 | 2 牛、9 羊 |
| Ice Cream + Yarn | 6 | 2 牛、9 羊 |
| Smoothie + Yarn | 8 | 2 牛、9 羊 |
| Bakery / Brunch / Farmers Market / Pet Cafe + Yarn | 1/3/4/5/9/10 | 2 牛、9 羊 |
| Yarn + Yarn | 12 | 12 羊 |

这说明 Pizza + Yarn 的 route7 并不是“没有理解牛奶需求”的笨重羊路线：Pizza 的牛奶需求已经被两头牛覆盖，而 Yarn 是当前唯一每次买两份、每天共买 12 份羊毛的专门店，因此 V45 明确把新增动物偏向羊。此前失败的 route123 实际是 Pizza + Pizza 的路线（5 牛、3 羊、3 鹅），不应拿来替代 route7。

另一个事实是：多数 route 的 raw 作物承诺相同——29 草莓、138 小麦、29 胡萝卜。第 6 天路线之间最明显的产品差异主要来自动物数量和后续工人/卖货时序，而不是简单“这条种草莓、那条种番茄”。

## 这对当前策略意味着什么

“按当前店铺需求选最好的动物/作物”已经是 V45 route table 的核心能力，至少 Yarn 分支明显如此。因此不能仅按 70% 覆盖公式重新在 40 条兼容 route 中选一个；那会再次把为另一套店铺、工人和卖货节奏准备的 route 强行挪过来。

这道“自由新增投资”检查其实已有更严格的有效结果：`v45-portfolio-capacity-audit-v3-multiday-tiles` 在固定七对手、4 seed、双座位的 56 条严格复现对局中，要求同一格地在完整生产期始终空闲、每天既有维护都能执行、且饲料/工时都留有余量。草莓、西瓜、照料羊的安全窗口均为 0；胡萝卜只有少量单格、赛后才可确认的窗口，不能当作通用实时扩张能力。详见 `HANDOFF.md` 和 `experiments/v45_portfolio_capacity_audit_v3.jsonl`。

因此当前结论是：不能在 V45 已选 route 旁边硬塞新的动物或作物。未来随机店铺仅在实际出现后的那个回合重新判断，绝不提前加分；若要改变当前生产结构，必须比较或重写一条完整 route，而不是在原 route 上外挂 70% 扩张。
