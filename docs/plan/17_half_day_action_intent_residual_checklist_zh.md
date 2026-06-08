# 17 半天专项计划：动作理解与关键帧追证残差压实清单

## 17.1 目标

本清单用于接下来约半天的 goal 模式执行，目标不是继续针对单个视频或单个题修补，而是把 `fine_grained_why_recognition / action_intent` 中最影响准确率的高频残差簇继续压实。

核心目标：

- 提升 agent 对厨房动作目的的理解能力。
- 让关键帧选择更主动、更贴近候选冲突点。
- 当多个候选仍然都成立时，禁止过早定答，必须继续查找更有判别力的证据。
- 每一轮改动都覆盖一类题型或一类语义冲突，而不是单样例 hack。
- 每个完成项都必须有代码、测试、文档记录和独立 commit。

当前起点：

- 已提交进展：`16.49 generic measurement-meta -> exact measurement target`
- 当前待提交进展：`16.50 pick up phone generic-measure -> exact ingredient record target`
- 当前专项回归：`pytest -q tests/test_graph_agent.py -k 'action_intent'`
- 当前已验证结果：`332 passed, 344 deselected`

## 17.2 半天执行原则

- [ ] 每轮先定位一个 residual bucket，再改代码。
- [ ] 优先改 `planner / verifier / toolbox / graph_agent` 中已有机制，不新建大架构。
- [ ] 不靠提示词微调刷分，优先提升工具调用、证据检索、关键帧选择和 finish gate。
- [ ] 每个 residual bucket 至少补 1 到 3 条定向测试。
- [ ] 每轮都跑定向测试和 `action_intent` 专项回归。
- [ ] 每轮都更新 `docs/plan/16_logic_reasoning_goal_checklist_zh.md` 或本清单的进展段。
- [ ] 每轮都单独 commit，不 amend，不提交数据、输出、密钥和无关脏改。

## 17.3 当前必须先收口的事项

- [x] 检查当前 `16.50` 未提交 diff。
- [x] 确认只提交本轮相关文件：
  - `food_agent/agent/planner.py`
  - `tests/test_graph_agent.py`
  - `docs/plan/16_logic_reasoning_goal_checklist_zh.md`
  - 如本清单有更新，可一并提交。
- [x] 重新运行：
  - `pytest -q tests/test_graph_agent.py -k 'phone_generic_measure_prefers_exact_record_target or phone_exact_record_still_revisits_target'`
  - `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 如果全部通过，提交：
  - 建议 commit：`fix: chase phone record targets after verifier block`

## 17.4 Residual Bucket A：phone / app / ingredient record 目标压实

问题：

- `pick up phone` 后，模型容易把动作目的停在 `measure the ingredients` 这类宽泛流程解释。
- 但真实问题往往需要判断具体是在记录 coriander、broccoli，还是另一个食材。
- 当前 `16.50` 已覆盖 `generic measure -> exact record target`，还需要继续压实边界。

待做：

- [x] 覆盖 `record nutritional value of X`、`update app with measurements of X` 两类高频措辞。
- [x] 覆盖 best 是 generic measure、second best 是 exact target 的情况。
- [x] 覆盖 best 是 exact target、但证据链仍不闭合的情况。
- [ ] 覆盖多个 exact record target 同时存在时，优先追当前最缺证据的目标。
- [x] 如果没有目标轨迹，先 `query_object`，不要直接 finish。
- [x] 如果有目标轨迹，优先 `query_spatial_context` 到更晚、更有判别力的时刻。

完成标准：

- [x] 新增或扩展测试通过。
- [x] `action_intent` 专项回归不下降，当前为 `327 passed, 344 deselected`。
- [x] 该类题不会只因为手机出现在秤旁边就提前选择 generic measure。

## 17.5 Residual Bucket B：open / close / same-object use 与 later-use 冲突

问题：

- `open jar / uncap bottle / remove lid` 这类动作常见冲突：
  - 立刻打开当前物体
  - 稍后倒出、称量、取用、放回
  - 为了腾手或腾空间
- agent 容易只看近窗动作状态，然后提前定答。

待做：

- [x] 找到已有 `open-close`、`same-object active use`、`mixed-horizon later target` 测试附近的残差。
- [x] 补一类 `open/uncap` vs `weigh/use later` 的 verifier-blocked recovery。
- [x] 补一类 `open/uncap` vs `put back/store later` 的 later-target recovery。
- [x] 明确 same-object active use 只在看到后续使用链时才压过 later-use。

完成标准：

- [x] 对 `open/uncap` vs `put back/store later`，当前已开始优先追最终位置证据而不是停在同物体近窗动作。
- [x] 当证据只证明“打开了”，但没证明“为什么打开”，必须继续追后续目标。

本轮进展：

- [x] 收紧 `verifier-blocked mixed_horizon later_target` 的 same-object blocker。此前只要 `reason / needed_observation` 出现 `same-object cap action / lid action`，planner 就可能直接放弃 later-target，导致 `open/uncap` vs `weigh later` 明明已经暴露出 `scale` 方向，仍围着动作物体本身打转。
- [x] 现在只有在 `best` 本身已经是 later-use 候选时，才保留原有 same-object blocker；如果 `best` 只是近窗 `open/uncap`，而 later-use 仍是竞争项，则会继续追更晚目标。
- [x] mixed-horizon later-target 落到 fixture 时，优先选更晚的 fixture 轨迹，不再停在最早出现的同名节点。
- [x] 新增并通过 1 条定向测试，覆盖 `same-object cap action` 仍出现在 `reason` 里时，`open/uncap` vs `weigh later` 仍会继续追 `scale`。
- [x] 本轮专项回归：`333 passed, 344 deselected`

## 17.6 Residual Bucket C：move / make space / hidden target / final placement

问题：

- `move bottle / move cup / move tray` 经常同时支持：
  - make space
  - access hidden target
  - place another object into freed slot
  - put object to final location
- 旧问题是把“空间变大了”当作目的，忽略后续真正目标。

待做：

- [x] 检查 `needed_observation` target revisit、reveal subtype、workspace/final placement gate。
- [ ] 补 `make space` vs `take hidden X` 的更多泛化测试。
- [x] 补 `make space` vs `place Y into freed slot` 的目标追证测试。
- [x] 补 `move object` 后只看到空间变化但没看到下游动作时不能 finish。

完成标准：

- [x] `revealed slot / sink slot` 这类“双目标文本”不再因为同时出现对象和槽位而丢失追证目标；planner 会优先追真正要被放入的 downstream object。
- [x] 只有看到 hidden target 被取出、slot 被使用、或后续目标动作发生，才允许具体候选收口。
- [x] 如果只是空间变大，继续追证据。

本轮进展：

- [x] 修复 `needed_observation` 在 `put X into the freed slot / sink slot` 语境下的双目标歧义。此前对象与槽位同时出现时，planner 可能不追任何精确目标，或被 relation-revisit 抢回动作物体；现在会优先追真正的 downstream object。
- [x] 修复 `choice target` 词项抽取中的子串误命中，避免 `saucepan -> pan` 这类目标追歪。
- [x] 新增 2 条定向测试覆盖 `blue cup -> freed slot` 与 `saucepan -> sink slot`。
- [x] 本轮专项回归：`330 passed, 344 deselected`
- [x] 收口 `make space on shelf/worktop` 的 exact-workspace overclaim。现在 `shelf/worktop/counter` 会进入 specific-space-target 路径；若证据只显示 broad workspace effect 而没有确切下游 use/destination，则 unresolved rerank 会写入 `exact_workspace_without_exact_use` 并继续等待证据。
- [x] 这一收口同时保留了更合理的 generic fallback：例如 `shelf layout` 变化但没有 hidden-target retrieval 时，允许回退到 generic access；而仅有 `worktop becomes more open` 这类宽泛变化时，会继续 withheld。
- [x] 当前专项回归已提升到 `332 passed, 344 deselected`

## 17.7 Residual Bucket D：towel / cloth / paper towel 的 transport-vs-use

问题：

- `pick up towel / cloth / paper towel` 容易混淆：
  - move / relocate
  - dry hand
  - wipe both hands
  - clean counter
  - dry object
- 单张关键帧往往只能证明“拿起来了”，不能证明目的。

待做：

- [x] 检查现有 `transport-vs-use` frame selection 和 future-use route。
- [x] 补“拿起后只是放下/挪动”的测试，防止误判 clean/dry。
- [x] 补“短暂接触手部”的测试，防止被泛化成 clean counter。
- [ ] 补“真正擦台面需要 sweep/contact target”的 finish gate。

完成标准：

- [ ] 不再只因纸巾靠近台面就判 clean counter。
- [ ] 不再只因纸巾被拿起就判 dry。
- [ ] 需要看到手部接触、表面擦拭、或最终放置链条。

本轮进展：

- [x] 收口 `dry hand` vs `wipe both hands` 的一条高频细粒度残差。此前 unresolved rerank 已能把 hand-contact 从 `clean counter` 拉回 hand-use，但对于“明确是双手擦拭”与“更弱的单手 dry hand”之间还缺少对称区分。
- [x] 新增 `explicit both-hands wiping` 规则：若证据明确出现 `brought to both hands / both hands are wiped`，则 `wipe both hands` 获得更强加分。
- [x] 同时收紧 `single-hand drying` 相关 override：如果证据已经明确说“不是单手，而是双手”，就不再把答案翻回 `dry hand`。
- [x] 新增并通过 1 条定向测试，覆盖 `pick up paper towel` 后证据明确指向双手擦拭时，不再停在 `dry hand`，而会提升到 `wipe both hands`。
- [x] 收紧 `surface_wipe_preparation`：此前 `non-storage` 很容易把“只是暂放在台面上”提前翻成 `wipe the worktop`；现在除了“不是收纳”，还必须出现 `crumbs / spill / surface target / ready for wiping / next visible cleaning target` 这类更具体的台面目标或 staged-wipe 信号。
- [x] 新增并通过 1 条 Bucket D 反例测试，覆盖 `dish cloth` 只是被暂放到 worktop within reach、但没有 crumbs/spill/visible target 时，不再提前推成 `wipe the worktop`，而是继续 withheld。
- [x] 补上 `surface wipe` finalizer 反例：即使已经出现 `crumbs / worktop target`，但还没有真正 `wiping stroke / sweep` 时，`wipe the worktop` 仍必须继续 withheld。
- [x] 新增并通过 1 条 Bucket D 定向测试，覆盖 `dish cloth` 放到 crumbs 旁边、`needed_observation` 仍是“是否真的擦过台面”时，finalizer 不能提前收口到 `wipe the worktop`。
- [x] 本轮专项回归：`336 passed, 344 deselected`

## 17.8 Residual Bucket E：scale / tare / zero / measurement state-change

问题：

- `tap kitchen scale`、`pick up scale`、`put container on scale` 常见冲突：
  - turn on
  - zero out
  - zero out with container
  - adjust measurements
  - weigh ingredient
- 这类题必须看动作前状态、动作后读数变化、容器是否已在秤上、食材是否随后加入。

待做：

- [x] 检查 `tap scale` state-change 现有 gate。
- [ ] 补 `zero out with container` 必须看到 container precondition。
- [x] 补 `turn on` vs `zero out` 必须看动作前显示状态。
- [x] 补 `adjust measurements` vs `weigh ingredient` 必须继续追 ingredient-on-scale 证据。

完成标准：

- [ ] 只看到显示为 0 不能直接判 zero out。
- [ ] 没有动作前显示状态时，必须补 precontext 或 transition frames。
- [ ] 没有食材上秤证据时，不能把 broad measurement 当作最终答案。

本轮进展：

- [x] 收口 `adjust/read measurements` 这类 broad measurement-meta 的 finalizer 边界。此前 `toolbox/planner` 已能在上游把 `adjust the measurements`、`read the measurements` 往“真正 weigh ingredient”追证，但 graph-agent finalizer 侧还缺少显式保护，导致这类宽泛答案在缺少 `reading/tare/update/direct weighing use` 时仍可能被提前收口。
- [x] 新增 `weak measurement meta` finish gate：若 top 候选只是 `adjust/read/record measurements`，而 `reason + decisive_observation` 里没有 `reading / tare / zero / display change / entered update` 等直接信号，则直接 withheld。
- [x] 同时补上 why 题 structured best-index fallback 的通用保护：只要已经写入 `action_intent_resolution_withheld_for_*` marker，就不允许后续 fallback 再把答案从旧的 `best_index` memory 里捞回来。
- [x] 新增并通过 1 条 Bucket E 定向测试，覆盖 `pick up scale` 后只有“scale remains near ingredient area”的宽泛 measurement 语义、但没有 `reading/tare` 明确信号时，finalizer 不能直接收口到 `adjust the measurements.`。
- [x] 新增并通过 1 条 Bucket E 回归保护测试，覆盖 `tap kitchen scale` 在缺少动作前开机状态/容器前提时，即使 working memory 里残留旧的 `action_intent_best_index=zero out`，structured fallback 也不会把被 finalizer withheld 的 `zero out` 再回填出来。
- [x] 收口 `missing_state_change_prereq -> precontext backfill` 的恢复缺口。此前 `tap kitchen scale` 虽然在 finalizer 已能识别“缺少动作前开机/容器前提”，但 planner 的 open-question / verifier-blocked recovery 仍可能继续回到 pairwise 或 followup，漏掉真正决定 `turn on` vs `zero out` 的前置状态。本轮改为：
  - `action_intent_needs_precondition_context` 现在把 `tap scale` 这类 `open_close + measure_weigh` 的 state-change 冲突也视为 precondition-dependent；
  - `planner` 在 `need_disambiguating_evidence` / `need_alternative_evidence_path` 下，若最近出现 `action_intent_resolution_withheld_for_missing_state_change_prereq=1`，会直接读取最新的 `pairwise/future_use` payload，并优先回采 `precontext`；
  - `tap kitchen scale` 的 backfill 还新增了更明确的 fast-path：只要 `needed_observation` 明示 `before the tap / already lit / already on / container already on the scale`，就不再继续盲目补动作后帧。
- [x] 新增并通过 1 条 Bucket E 定向测试，覆盖 `resolve_action_intent_pairwise` 已明确“需要看动作前显示状态/容器前提”时，planner 会优先采样 `fine_grained_why_recognition_precontext`，而不是继续回到 `pairwise`。
- [x] 本轮专项回归：`337 passed, 344 deselected`

## 17.9 Residual Bucket F：inspection / check / read label 与 later outcome

问题：

- `take bottle / jar / pot` 容易混淆：
  - check label/date
  - check boiling/content
  - open/use later
  - put back/store
- 视觉上“标签可见”不等于真的 read/check。

待做：

- [x] 补 `label visible` 但没有 reading chain 时不能 finish。
- [ ] 补 `check label` vs `put back` 的混合窗口追证。
- [ ] 补 `check contents` vs `pour/empty/serve` 的 later outcome 追证。
- [ ] 将 inspection 的 needed evidence 写得更明确，指导 planner 查后续帧。

完成标准：

- [x] 不把“标签朝外/可见”直接当成 check label。
- [x] 不把“短暂看锅”直接当成 check boiling，除非有停留/查看链。
- [ ] 如果后续倒出、归位或使用发生，应该能覆盖 inspection 误判。

本轮进展：

- [x] 确认并保留 `label visible` 的现有 finalizer gate。当前 `check label` 只有在出现更明确的 reading chain 时才允许 finish；仅有 `label faces the camera / label is visible while held` 时，会继续 withheld，不会把“标签可见”直接当成 `check the label`。
- [x] 收口 `weak cooking inspection` 的 finalizer 缺口。此前 `resolve_action_intent_future_use` 若直接给出 `check the boiling water / check the contents / check the consistency`，只要锅具里“似乎还有液体/内容物”就可能被 deterministic finalizer 直接收口，即使没有 `brief inspection / stays near hob / no tilt / no pouring / no serving destination` 这类 inspection chain。
- [x] 新增 `weak cooking inspection` finish gate：对于 `pot/pan/saucepan/frying pan/bowl` 这类 cooking vessel，若 top 候选是 `check boiling / check contents / check consistency / see if done`，但 `reason + decisive_observation + needed_observation` 里没有形成 `brief cooking inspection over disposal` 的强链条，则直接 withheld。
- [x] 新增并通过 1 条 Bucket F 定向测试，覆盖“只有 `pot is lifted while it still seems to contain hot water`、没有 `brief inspection / no tilt / stays near hob` 链条”时，finalizer 不能直接收口到 `to check the boiling water.`。
- [x] 收口 `check label vs put back` 的一个 planner 恢复缺口。此前 finalizer 已能写入 `action_intent_resolution_withheld_for_mixed_horizon_later_target=1 target=fridge kind=fixture`，但 mixed-horizon later-target revisit 仍可能停在过早的 fridge 节点，而不是追更晚的 return window。本轮改为：对于 finalizer 写出的 fixture later-target，planner 也像 verifier-blocked 那条路径一样，优先选择满足 `min_start_time` 的更晚 fixture 轨迹。
- [x] 新增并通过 1 条 Bucket F 定向测试，覆盖 `check label` 被 finalizer 拦下并写入 `target=fridge kind=fixture` 后，planner 会优先跳到更晚的 fridge 轨迹窗口，而不是停在过早的近窗 fridge 节点。

## 17.10 半天验收命令

每个 bucket 完成后至少运行：

```bash
pytest -q tests/test_graph_agent.py -k 'action_intent'
```

必要时运行更窄的定向命令：

```bash
pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'
pytest -q tests/test_graph_agent.py -k 'future_use and action_intent'
pytest -q tests/test_graph_agent.py -k 'pairwise and action_intent'
```

半天结束前必须记录：

- [ ] 当前 `action_intent` 通过数量。
- [x] 当前 `action_intent` 通过数量。
- [ ] 新增了哪些 residual bucket。
- [ ] 哪些 bucket 已提交。
- [ ] 哪些 bucket 还没做。
- [ ] 是否存在未提交但已验证通过的改动。
- [x] 是否存在未提交但已验证通过的改动。
- [ ] 是否存在失败测试或已知回归风险。

## 17.11 半天结束时的交付物

- [ ] 至少提交当前 `16.50` 改动。
- [ ] 继续完成至少 2 个后续 residual bucket。
- [ ] `pytest -q tests/test_graph_agent.py -k 'action_intent'` 通过。
- [ ] `docs/plan/16_logic_reasoning_goal_checklist_zh.md` 或本文件记录每轮进展。
- [ ] git log 中每轮有独立 commit。
- [ ] 最终给出阶段报告：
  - 当前做到第几个 bucket
  - 准确性机制提升点
  - 回归测试结果
  - 下一步还剩什么

## 17.12 Goal 模式提示词

下面这段可以直接作为 goal 模式目标使用：

```text
请按照 /22liushoulong/agent/hd-epic/docs/plan/17_half_day_action_intent_residual_checklist_zh.md 执行半天专项优化。

目标是继续提升 fine_grained_why_recognition / action_intent 的动作理解能力、关键帧主动选择能力和延迟定答能力。不要针对单个视频或单个样例做 hack，而是按 residual bucket 逐类压实高频失败模式。每一轮必须先定位一个语义残差簇，再实现代码改动、补定向测试、运行回归、更新文档，并做独立 git commit。

优先顺序：
1. 先收口并提交当前 16.50 的 phone generic-measure -> exact ingredient record target 改动。
2. 继续处理 open/close/same-object use 与 later-use 冲突。
3. 继续处理 move/make-space/hidden-target/final-placement 冲突。
4. 继续处理 towel/cloth/paper towel 的 transport-vs-use 冲突。
5. 继续处理 scale/tare/zero/measurement state-change 冲突。
6. 继续处理 inspection/check/read-label 与 later outcome 冲突。

执行约束：
- 不要重置或覆盖用户已有脏改。
- 不要提交 data、outputs、annotations、密钥或无关未跟踪脚本。
- 每轮只提交本轮相关文件。
- 每轮至少运行定向测试和 pytest -q tests/test_graph_agent.py -k 'action_intent'。
- 如果 action_intent 回归失败，必须先修复再继续。
- 不要用提示词微调代替结构性能力提升，优先改 planner/verifier/toolbox/graph_agent 的证据追踪、关键帧选择和 finish gate。
- 半天结束时给出完成报告，包括当前通过数量、已完成 bucket、未完成 bucket、提交列表和下一步计划。
```
