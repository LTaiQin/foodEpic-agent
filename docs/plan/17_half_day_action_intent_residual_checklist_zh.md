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
- [ ] 补一类 `open/uncap` vs `weigh/use later` 的 verifier-blocked recovery。
- [x] 补一类 `open/uncap` vs `put back/store later` 的 later-target recovery。
- [ ] 明确 same-object active use 只在看到后续使用链时才压过 later-use。

完成标准：

- [x] 对 `open/uncap` vs `put back/store later`，当前已开始优先追最终位置证据而不是停在同物体近窗动作。
- [ ] 当证据只证明“打开了”，但没证明“为什么打开”，必须继续追后续目标。

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

- [ ] 检查现有 `transport-vs-use` frame selection 和 future-use route。
- [ ] 补“拿起后只是放下/挪动”的测试，防止误判 clean/dry。
- [ ] 补“短暂接触手部”的测试，防止被泛化成 clean counter。
- [ ] 补“真正擦台面需要 sweep/contact target”的 finish gate。

完成标准：

- [ ] 不再只因纸巾靠近台面就判 clean counter。
- [ ] 不再只因纸巾被拿起就判 dry。
- [ ] 需要看到手部接触、表面擦拭、或最终放置链条。

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

- [ ] 检查 `tap scale` state-change 现有 gate。
- [ ] 补 `zero out with container` 必须看到 container precondition。
- [ ] 补 `turn on` vs `zero out` 必须看动作前显示状态。
- [ ] 补 `adjust measurements` vs `weigh ingredient` 必须继续追 ingredient-on-scale 证据。

完成标准：

- [ ] 只看到显示为 0 不能直接判 zero out。
- [ ] 没有动作前显示状态时，必须补 precontext 或 transition frames。
- [ ] 没有食材上秤证据时，不能把 broad measurement 当作最终答案。

## 17.9 Residual Bucket F：inspection / check / read label 与 later outcome

问题：

- `take bottle / jar / pot` 容易混淆：
  - check label/date
  - check boiling/content
  - open/use later
  - put back/store
- 视觉上“标签可见”不等于真的 read/check。

待做：

- [ ] 补 `label visible` 但没有 reading chain 时不能 finish。
- [ ] 补 `check label` vs `put back` 的混合窗口追证。
- [ ] 补 `check contents` vs `pour/empty/serve` 的 later outcome 追证。
- [ ] 将 inspection 的 needed evidence 写得更明确，指导 planner 查后续帧。

完成标准：

- [ ] 不把“标签朝外/可见”直接当成 check label。
- [ ] 不把“短暂看锅”直接当成 check boiling，除非有停留/查看链。
- [ ] 如果后续倒出、归位或使用发生，应该能覆盖 inspection 误判。

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
