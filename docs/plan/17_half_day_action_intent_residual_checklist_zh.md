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
- 当前待提交进展：`Bucket A multiple exact phone-record targets -> prefer most evidence-starved exact target`
- 当前专项回归：`pytest -q tests/test_graph_agent.py -k 'action_intent'`
- 当前已验证结果：`353 passed, 344 deselected`

最新进展补充：

- [x] 新 residual bucket：`immediate micro-outcome overclaim`
- [x] 代表 case：
  - `take bottle` 时 `to open the bottle.` 只有 `opening could happen next`、`cap opening itself is not visible yet`，不能提前收口；
  - `transfer cup of breadcrumbs` 时如果证据已经形成 `free hand -> uncap/open same object immediately`，则不能被后续 `weigh breadcrumbs` 反向压回 withheld。
- [x] 这轮没有新增大机制，只收紧 `graph_agent` unresolved rerank 的即时微结果证据边界，并修复一个真实字符串残差：`breadcrumbs` 中的 `read` 子串此前会误触发阅读类分支，导致 `uncap/open` 证据被提前短路。
- [x] 当前最新专项回归：`361 passed, 344 deselected`
- [x] 新 residual bucket：`specialized failure fallback over-finishes stale intent`
- [x] 代表 case：
  - `infer_action_intent` 旧成功结果已经明确写出 `need_future_evidence / needed_observation`；
  - 之后 `resolve_action_intent_pairwise / future_use` 因工具失败中断；
  - planner 不能因为“有一个旧 best_index”就直接 finish，而必须继续沿已有的 mixed-horizon / later-target 恢复链追证据。
- [x] 当前最新专项回归：`363 passed, 344 deselected`
- [x] 新 residual bucket：`resolution need-more-evidence fallback finishes without anchors`
- [x] 代表 case：
  - `resolve_action_intent_pairwise / future_use` 自己已经明确写出 `need_more_evidence=True`；
  - 但当前又缺少 `times / input_times`，导致恢复链暂时构不出更具体的 followup；
  - planner 不能因此把这个未闭合结果直接 `finish`，而必须至少回到当前题 `segment` 重抽。
- [x] 当前最新专项回归：`365 passed, 344 deselected`
- [x] 新 residual bucket：`unresolved infer_action_intent falls back to text rank without anchors`
- [x] 代表 case：
  - `infer_action_intent` 自己已经明确写出 `need_future_evidence=True`；
  - 但当前又没有 `times / input_times`，也暂时构不出更具体的 followup / pairwise / future-use 恢复链；
  - planner 不能因此先退 `rank_choices_from_state`，更不能下一轮继续掉到泛化 `query_time`，而必须留在 why 专用恢复链中继续重抽当前题 `segment`。
- [x] 当前最新专项回归：`367 passed, 344 deselected`
- [x] 新 residual bucket：`textual fallback ignores needed_observation marker`
- [x] 代表 case：
  - repeated vision failure 后，系统已经落到 `ranked_best_index` textual fallback；
  - 当前题 artifact 与 grounding 都存在；
  - 但 working memory 里仍保留 `action_intent_needed_observation=...`，说明真正的 later-target / followup 关键证据还没闭合；
  - verifier 不能因为文本 fallback 看起来“像是够了”就直接放行。
- [x] 当前最新专项回归：`368 passed, 344 deselected`

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
- [x] 覆盖多个 exact record target 同时存在时，优先追当前最缺证据的目标。
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
- [x] 继续收口 `immediate micro-outcome overclaim`。此前 unresolved rerank 新增 `missing_immediate_micro_outcome_evidence` 后，已经能拦住“只说 opening could happen next”这类弱 close-call，但会误伤两类本应保留的 exact chain：
  - `same-object hand-free -> uncap/open immediately`
  - `revealed fixture -> immediately turn it on`
- [x] 本轮把即时微结果的正证据识别补强到：
  - `free to uncap/open`
  - `uncap/open ... immediately`
  - `reaches to the scale and turns it on`
  - `immediately afterwards ... turns it on`
- [x] 同时修复阅读类短语匹配的子串误判，避免 `breadcrumbs` 中的 `read` 把 `uncap/open` 候选错误拉进 `check label/read date` 分支。
- [x] 新增并通过 2 条定向测试，覆盖：
  - `opening could happen next` 但 opening 不可见时继续 withheld；
  - opening 明确可见时 `to open the bottle.` 仍可正常收口。
- [x] 同时复核通过 2 条原有正例：
  - `free hand same-object open over later scale use`
  - `exact revealed fixture enablement over generic access`
- [x] 本轮专项回归：`361 passed, 344 deselected`
- [x] 收口 `specialized failure fallback over-finishes stale intent`。此前 `resolve_action_intent_pairwise / future_use` 连续失败后，planner 会直接回退到最近一次成功的 `infer_action_intent` 并 finish；但这条兜底没有区分“旧成功结果已经闭合”和“旧成功结果自己都还写着 need_future_evidence / needed_observation”，因此 mixed-horizon later-target 题在工具失败时仍可能被过早收口。
- [x] 现在 failure fallback 只有在旧成功结果已经真正闭合时才允许 finish；若旧结果仍带：
  - `need_future_evidence`
  - `need_more_evidence`
  - `needed_observation`
  - 或 working memory 里仍残留 `pending_resolution / withheld / unresolved_rerank_withheld / action_intent_needed_observation`
  则继续走已有专用恢复链，不允许直接落回旧 best guess。
- [x] 新增并通过 2 条定向测试，覆盖：
  - `take bottle` 的旧成功结果仍要求确认是否 later put back into the fridge 时，pairwise 失败不能直接 finish 到 `to open the bottle.`；
  - `place lid` 的旧成功结果已经形成闭合链条时，pairwise 失败仍可安全 finish。
- [x] 本轮专项回归：`363 passed, 344 deselected`
- [x] 收口 `resolution need-more-evidence fallback finishes without anchors`。此前 `resolve_action_intent_pairwise / future_use` 即使已经明确返回 `need_more_evidence=True`，只要当前没有 `times / input_times` 之类恢复锚点，planner 仍可能直接落到“专用裁决已完成，直接结束”。这和 why 专项要求的延迟定答能力冲突，因为它会把明确未闭合的专用裁决结果提前收口。
- [x] 现在对 `pairwise / future_use` 的 finish gate 再收紧一层：
  - 只要 payload 仍带 `need_more_evidence / need_future_evidence / needed_observation`，就视为未闭合；
  - 即使此时 `specialized recovery` 一时构不出更具体的 followup，也先回到当前题 `segment` 重抽，而不是直接 finish。
- [x] 新增并通过 2 条定向测试，覆盖：
  - `resolve_action_intent_pairwise` 在 `need_more_evidence=True` 且没有时间 hints 时，回到 `fine_grained_why_recognition_segment` 重抽；
  - `resolve_action_intent_future_use` 在同类条件下也回到当前题 `segment` 重抽，而不直接结束。
- [x] 本轮专项回归：`365 passed, 344 deselected`
- [x] 收口 `unresolved infer_action_intent falls back to text rank without anchors`。此前即使 `infer_action_intent` 自己已经明确返回 `need_future_evidence=True`，只要当前没有 `times / input_times`、也暂时构不出后续专用恢复动作，planner 仍会先退到 `rank_choices_from_state`。这会把 why 专项里明确未闭合的专用动作目的判断降级为文本聚合，再在下一轮继续掉到泛化 `query_time`。
- [x] 现在对 `infer_action_intent` 的文本 fallback gate 也收紧：
  - 只要 payload 仍带 `need_future_evidence / need_more_evidence / needed_observation`，就不允许退 `rank_choices_from_state`；
  - 在缺少恢复锚点时，也先回到当前题 `segment` 重抽，保持在 why 专用恢复链中。
- [x] 新增并通过 2 条定向测试，覆盖：
  - 未闭合 `infer_action_intent` 在没有时间 hints 时，不再退 `rank_choices_from_state`，而是回到 `fine_grained_why_recognition_segment`；
  - 已闭合 `infer_action_intent` 在无其它恢复路径时，保持当前稳定行为，仍可直接 finish。
- [x] 本轮专项回归：`367 passed, 344 deselected`
- [x] 收口 `textual fallback ignores needed_observation marker`。此前 repeated vision failure 的 why 文本 fallback 只要具有当前题 artifact 和一些 grounding，就可能被 verifier 直接判 sufficient；但 verifier 没有把 `action_intent_needed_observation=...` 这类“明确还缺关键后续证据”的 working-memory marker 当作 blocker，因此会把仍未闭合的 textual fallback 过早放行。
- [x] 现在 verifier 的 textual fallback 放行 gate 也收紧：
  - 只要最近 working memory 里仍有 `action_intent_needed_observation=...`，就不允许 textual fallback 直接 sufficient；
  - 这保证 verifier 与最近几轮 planner fallback 收紧后的行为一致，不会一边要求继续追 needed observation，一边又把文本 fallback 提前放行。
- [x] 新增并通过 1 条定向测试，覆盖：
  - `take bottle` 的 textual fallback 即使已有当前题 artifact 和 grounding，只要仍保留 `action_intent_needed_observation=whether the bottle is later put back into the fridge`，verifier 仍必须保持 blocking。
- [x] 同时复核通过 1 条原有正例：
  - `place bowl` 的 textual fallback 在已有当前题 artifact、grounding且没有未闭合 marker 时，仍可正常 finish。
- [x] 本轮专项回归：`368 passed, 344 deselected`

补充进展：

- [x] `16.50` 已完成并提交，不再处于“待提交”状态；当前 Bucket A 的剩余缺口已经转成“多 exact record target 并存时如何选最该追的那个目标”。
- [x] 收口 `multiple exact phone-record targets` 的 verifier-blocked recovery。此前这条路径只稳定覆盖 `generic measure vs 单个 exact target`；如果同时存在 `coriander / broccoli / carrot` 这类多个具体记录目标，planner 容易只盯住最先进入 top-2 的 exact target，而不会比较哪一个目标当前最缺判别证据。
- [x] 现在 `planner` 会扫描全部 `phone/app record target` 候选，综合 `screen not readable / no X target visible / still unresolved / no direct recording target` 这类 uncertainty marker，优先追“目标最明确、但证据链仍最缺”的 exact target。
- [x] 同时补上 phone-record revisit 的 later-node 偏置：如果当前还没有更强的 anchor/followup 约束，就不再默认追该对象最早出现的轨迹，而会优先看更晚、更有判别力的目标节点。
- [x] 新增并通过 2 条 Bucket A 定向测试，分别覆盖：
  - 多个 exact ingredient record target 同时存在时，会优先追当前最不确定、最需要补证据的那个目标；
  - 如果该目标没有现成轨迹，则会先 `query_object` 检索它，而不是退回 generic measure 或继续追错对象。
- [x] 本轮专项回归：`352 passed, 344 deselected`

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
- [x] 补 `make space` vs `take hidden X` 的更多泛化测试。
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
- [x] 收口 `generic make space` vs `take hidden X` 的 reveal 泛化缺口。此前 `generic access -> hidden retrieval` 已有 override，但如果 top 候选仍是 `to make space on the shelf`，即使竞争项已经明确写出“hidden spice jar behind it becomes reachable and is taken right afterwards”，系统也可能停在 broad room-making；本轮改为：
  - `generic hidden access -> exact revealed target` 的 override 同时覆盖 `generic make space + reveal` 这类 best 候选；
  - 只要 exact candidate 已经形成明确 `hidden item / item behind / becomes reachable and is taken right afterwards` 链条，就允许把 broad make-space 翻正成 hidden retrieval；
  - 如果 reveal 真实存在、但 hidden target 仍未被取出，则继续保留更合理的 `generic access` fallback，不会误翻到 exact hidden retrieval。
- [x] 新增并通过 2 条 Bucket C 定向测试，分别覆盖：
  - `move bottle` 时 `to make space on the shelf.` 会被明确的 `take the hidden spice jar behind it` 压过；
  - 如果只是 reveal 了 behind area、但 hidden spice jar 仍只是 speculative，则不会误翻到 exact hidden retrieval，而会回退到 `to access what's behind the bottle.`。
- [x] 当前专项回归已提升到 `357 passed, 344 deselected`

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
- [x] 补“真正擦台面需要 sweep/contact target”的 finish gate。

完成标准：

- [x] 不再只因纸巾靠近台面就判 clean counter。
- [x] 不再只因纸巾被拿起就判 dry。
- [x] 需要看到手部接触、表面擦拭、或最终放置链条。

本轮进展：

- [x] 收口 `dry hand` vs `wipe both hands` 的一条高频细粒度残差。此前 unresolved rerank 已能把 hand-contact 从 `clean counter` 拉回 hand-use，但对于“明确是双手擦拭”与“更弱的单手 dry hand”之间还缺少对称区分。
- [x] 新增 `explicit both-hands wiping` 规则：若证据明确出现 `brought to both hands / both hands are wiped`，则 `wipe both hands` 获得更强加分。
- [x] 同时收紧 `single-hand drying` 相关 override：如果证据已经明确说“不是单手，而是双手”，就不再把答案翻回 `dry hand`。
- [x] 新增并通过 1 条定向测试，覆盖 `pick up paper towel` 后证据明确指向双手擦拭时，不再停在 `dry hand`，而会提升到 `wipe both hands`。
- [x] 收紧 `surface_wipe_preparation`：此前 `non-storage` 很容易把“只是暂放在台面上”提前翻成 `wipe the worktop`；现在除了“不是收纳”，还必须出现 `crumbs / spill / surface target / ready for wiping / next visible cleaning target` 这类更具体的台面目标或 staged-wipe 信号。
- [x] 新增并通过 1 条 Bucket D 反例测试，覆盖 `dish cloth` 只是被暂放到 worktop within reach、但没有 crumbs/spill/visible target 时，不再提前推成 `wipe the worktop`，而是继续 withheld。
- [x] 补上 `surface wipe` finalizer 反例：即使已经出现 `crumbs / worktop target`，但还没有真正 `wiping stroke / sweep` 时，`wipe the worktop` 仍必须继续 withheld。
- [x] 新增并通过 1 条 Bucket D 定向测试，覆盖 `dish cloth` 放到 crumbs 旁边、`needed_observation` 仍是“是否真的擦过台面”时，finalizer 不能提前收口到 `wipe the worktop`。
- [x] 收口 `clean up the kitchen counter` 这类表面清洁表述在 unresolved rerank 中的残差。此前 `wipe the worktop` 已被 gate 住，但 `clean up the kitchen counter` 仍可能仅凭 `near the counter / touches the counter area / briefly pressed to the surface` 之类弱接触证据提前收口；现在 `missing_surface_wiping_evidence` 已扩到这类 surface-cleanup 候选，并复用 `weak_surface_contact_cleanup_claim` 统一拦截“只有接触、没有 sweep/contact chain”的 overclaim。
- [x] 新增并通过 2 条 Bucket D 定向测试，分别覆盖：
  - finalizer 在只有短暂表面接近/接触时，不能直接收口到 `clean up the kitchen counter.`；
  - unresolved rerank 在没有 `wipe sweep / repeated wiping / clear before-after cleanup result` 时，必须继续 withheld。
- [x] 本轮专项回归：`350 passed, 344 deselected`

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
- [x] 补 `zero out with container` 必须看到 container precondition。
- [x] 补 `turn on` vs `zero out` 必须看动作前显示状态。
- [x] 补 `adjust measurements` vs `weigh ingredient` 必须继续追 ingredient-on-scale 证据。

完成标准：

- [x] 只看到显示为 0 不能直接判 zero out。
- [x] 没有动作前显示状态时，必须补 precontext 或 transition frames。
- [x] 没有食材上秤证据时，不能把 broad measurement 当作最终答案。

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
- [x] 收口 `zero out with container` 在 pairwise 缺少 container precondition 时的恢复顺序缺口。此前 `pairwise` 路径即使已经明确写出“要看 tap 前容器是否已在秤上”，也可能先退回 generic extra-followup，再去补真正决定性的 precontext；现在这条路径与 verifier-blocked 恢复链对齐，优先回采 `precontext`，只有 precontext 仍不足时才追加更长 followup。
- [x] 新增并通过 1 条 Bucket E 定向测试，覆盖 `tap kitchen scale` 的 `zero out with container` pairwise close-call：当 `needed_observation` 明示“容器是否在 tap 前已在秤上”时，planner 会先走 `fine_grained_why_recognition_precontext`，而不是先补 generic `followup_ext1`。
- [x] 补上 `measurement-meta vs exact measurement role` 的 unresolved-rerank 边界。此前 finalizer 已能挡住 `adjust/read measurements` 的 broad overclaim，但 unresolved rerank 在一些 close-call 下仍可能把弱 `measurement context` 直接翻成 `measure the cheese / base for weighing` 这类 exact role，即使还没看到 `reading/tare/display-change`，也没看到真正的 `ingredient on scale / used for weighing` 链条。现在 `graph_agent` 新增 measurement-role sufficiency helper，并在 unresolved semantic gaps 里补上 `missing_measurement_meta_evidence` 与 `missing_exact_measurement_role_evidence`，防止 broad/speculative measurement close-call 提前收口。
- [x] 新增并通过 1 条 Bucket E 定向反例测试，覆盖 `pick up scale` 时 broad measurement-meta 与 exact measurement role 都仍只是 speculative 的 unresolved close-call；现在系统会继续 withheld，而不是提前翻到 `measure the cheese`。
- [x] 同时复核通过 2 条原有正例：
  - `measure the cheese` 在确有 immediate weighing-use 证据时仍可翻正；
  - `use the bowl as a base to weigh more ingredients` 这类 measurement-base setup 仍能正常收口。
- [x] 本轮专项回归：`353 passed, 344 deselected`

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
- [x] 补 `check label` vs `put back` 的混合窗口追证。
- [x] 补 `check contents` vs `pour/empty/serve` 的 later outcome 追证。
- [x] 将 inspection 的 needed evidence 写得更明确，指导 planner 查后续帧。

完成标准：

- [x] 不把“标签朝外/可见”直接当成 check label。
- [x] 不把“短暂看锅”直接当成 check boiling，除非有停留/查看链。
- [x] 如果后续倒出、归位或使用发生，应该能覆盖 inspection 误判。

本轮进展：

- [x] 确认并保留 `label visible` 的现有 finalizer gate。当前 `check label` 只有在出现更明确的 reading chain 时才允许 finish；仅有 `label faces the camera / label is visible while held` 时，会继续 withheld，不会把“标签可见”直接当成 `check the label`。
- [x] 收口 `weak cooking inspection` 的 finalizer 缺口。此前 `resolve_action_intent_future_use` 若直接给出 `check the boiling water / check the contents / check the consistency`，只要锅具里“似乎还有液体/内容物”就可能被 deterministic finalizer 直接收口，即使没有 `brief inspection / stays near hob / no tilt / no pouring / no serving destination` 这类 inspection chain。
- [x] 新增 `weak cooking inspection` finish gate：对于 `pot/pan/saucepan/frying pan/bowl` 这类 cooking vessel，若 top 候选是 `check boiling / check contents / check consistency / see if done`，但 `reason + decisive_observation + needed_observation` 里没有形成 `brief cooking inspection over disposal` 的强链条，则直接 withheld。
- [x] 新增并通过 1 条 Bucket F 定向测试，覆盖“只有 `pot is lifted while it still seems to contain hot water`、没有 `brief inspection / no tilt / stays near hob` 链条”时，finalizer 不能直接收口到 `to check the boiling water.`。
- [x] 收口 `check label vs put back` 的一个 planner 恢复缺口。此前 finalizer 已能写入 `action_intent_resolution_withheld_for_mixed_horizon_later_target=1 target=fridge kind=fixture`，但 mixed-horizon later-target revisit 仍可能停在过早的 fridge 节点，而不是追更晚的 return window。本轮改为：对于 finalizer 写出的 fixture later-target，planner 也像 verifier-blocked 那条路径一样，优先选择满足 `min_start_time` 的更晚 fixture 轨迹。
- [x] 新增并通过 1 条 Bucket F 定向测试，覆盖 `check label` 被 finalizer 拦下并写入 `target=fridge kind=fixture` 后，planner 会优先跳到更晚的 fridge 轨迹窗口，而不是停在过早的近窗 fridge 节点。
- [x] 收口 `weak cooking inspection` 被 finalizer 拦下后没有写出 later-target 的缺口。现在 `check boiling/check contents` 这类 immediate inspection close-call 如果竞争项本身已经暴露出 `empty/pour/serve later` 的目标语义，即使通用 mixed-horizon 分类还不够完整，也会继续把真实 later target 写回 working memory。
- [x] 新增并通过 1 条 Bucket F 定向测试，覆盖 `pick up pot` 时 `check the boiling water` vs `empty the water` 的 close-call：当文本只说明“锅里似乎还有热水、尚未看清是否倒向 sink 还是只是短暂查看”时，finalizer 会同时写入 `action_intent_resolution_withheld_for_weak_cooking_inspection_evidence=1` 和 `action_intent_resolution_withheld_for_mixed_horizon_later_target=1 target=sink kind=fixture`，供 planner 后续追更晚 sink 轨迹。
- [x] 收口 `check label` vs `put back` 在 unresolved rerank 路径上的 fixture later-target 过早停留问题。此前 finalizer/verifier-blocked 已会对 `fridge/scale/sink` 这类 fixture later-target 优先跳到更晚节点，但 unresolved rerank 仍可能停在近窗 fixture 节点；现在这条恢复链也与前两条对齐，会优先选择满足 `min_start_time` 的更晚 fixture 轨迹。
- [x] 新增并通过 1 条 Bucket F 定向测试，覆盖 `take bottle` 时 `check the label` vs `put the bottle back in the fridge` 的 unresolved rerank close-call：当当前 late window 仍只显示“标签朝外、尚未看清是否回冰箱”时，planner 会优先跳到更晚 fridge 节点，而不是停在近窗 fridge 轨迹。
- [x] 收口 `weak cooking inspection` 的 needed-evidence 缺口。此前 finalizer 即使已经能写出 `target=sink kind=fixture` 这类 later-target marker，若上游 payload 没有 `needed_observation`，planner 仍只能依赖 generic mixed-horizon/long-horizon 路由。本轮改为：当 `check boiling/check contents/check consistency` 被 finalizer withheld 时，系统会显式写回更具体的判别说明，例如“是否真的朝 sink 倾倒，还是只是在 hob 附近短暂查看”。
- [x] 同时补上 planner 对 `action_intent_needed_observation=...` 的 working-memory fallback。现在即使最近一次 resolution payload 本身没有 `needed_observation`，只要 graph-agent 已经写入该 marker，planner 也会继续利用它做 target / relation revisit，而不是退回泛化 followup 或音频峰值搜索。
- [x] 新增并通过 2 条 Bucket F 定向测试，分别覆盖：
  - `pick up pot` 时 `check the boiling water` vs `empty the water` 的 finalizer close-call，会同时写入 `target=sink` later-target marker 和明确的 `needed_observation`；
  - planner 在只有 `working_memory` 里的 inspection `needed_observation` marker 时，也会继续利用该信息进入更强的 relation/target revisit 路径。
- [x] 收口 `check contents` vs `serve later` 的 plate/bowl later-outcome 泛化。此前 `sink` 这类 later target 已经能稳定写出 `needed_observation` 并驱动 planner，但 `serve the vegetables / serve the soup / carried over the plate` 这类 plate-serving 场景仍缺少专门保护；现在这一分支也被测试锁住，并允许 planner 在已有 inspection `needed_observation` marker 时，直接跳进 relation revisit，而不是退回 generic audio-peaks。
- [x] 同时放宽 `needed_observation target/relation revisit` 的旧 followup 次数门槛：如果 working memory 里已经有明确的 `action_intent_needed_observation=...`，就不再强制等到第 2 次 followup 之后才允许进入目标/关系追证。
- [x] 新增并通过 2 条 Bucket F 定向测试，分别覆盖：
  - `lift frying pan` 时 `check the contents of the pan` vs `serve the vegetables` 的 finalizer close-call，会同时写入 `target=plate` later-target marker 和明确的 `needed_observation`；
  - planner 在只有 `working_memory` 里的 `plate-serving` inspection marker 时，也会直接利用该 marker 进入 relation revisit，优先去看 frying pan 是否真的被带到 plate 上方。
- [x] 收口 `inspection` 在“later outcome 已经明确发生”时仍只会 withheld、不会直接翻正的最后缺口。此前系统已经能在 `label vs put back`、`check boiling vs empty` close-call 中继续追更晚目标，但当证据里已经直接出现 `placed back into the fridge`、`tilted to pour into the sink` 这类明确 later outcome 时，仍缺少一条稳定的正向翻正链。本轮改为：
  - finalizer 新增 `explicit later outcome over weak inspection` override：若 top 候选只是弱 `check label / check boiling / check contents`，且自身没有明确 inspection chain，但竞争项已经给出显式 later outcome，则直接翻到 later candidate；
  - unresolved rerank 同步新增同类 override，但仅限真正的 inspection mixed-horizon 题，不作用于 `move/transfer` 这类“下游拿取不是当前直接目的”的题；
  - later outcome 证据同时要求是“已经发生”的结果，不接受 `could/may/not yet visible` 这类推测式表述，保留原有 close-call withheld 行为。
- [x] 新增并通过 2 条 Bucket F 定向测试，分别覆盖：
  - `take bottle` 时若证据已经明确写出 `placed back into the fridge`，则 `to put the bottle back in the fridge.` 会压过弱 `to check the label.`；
  - `pick up pot` 时若证据已经明确写出 `brought to the sink and tilted to pour`，则 `to empty the water.` 会压过弱 `to check the boiling water.`。
- [x] 本轮专项回归：`355 passed, 344 deselected`

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

- [x] 当前 `action_intent` 通过数量。
- [x] 当前 `action_intent` 通过数量。
- [x] 新增了哪些 residual bucket。
- [x] 哪些 bucket 已提交。
- [x] 哪些 bucket 还没做。
- [x] 是否存在未提交但已验证通过的改动。
- [x] 是否存在未提交但已验证通过的改动。
- [x] 是否存在失败测试或已知回归风险。

## 17.11 半天结束时的交付物

- [x] 至少提交当前 `16.50` 改动。
- [x] 继续完成至少 2 个后续 residual bucket。
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'` 通过。
- [x] `docs/plan/16_logic_reasoning_goal_checklist_zh.md` 或本文件记录每轮进展。
- [x] git log 中每轮有独立 commit。
- [x] 最终给出阶段报告：
  - 当前做到第几个 bucket
  - 准确性机制提升点
  - 回归测试结果
  - 下一步还剩什么

## 17.12 半天阶段报告

### 当前专项结果

- 当前专项回归命令：`pytest -q tests/test_graph_agent.py -k 'action_intent'`
- 当前稳定结果：`363 passed, 344 deselected`
- 当前稳定结果：`368 passed, 344 deselected`
- 相比进入本轮半天专项时文档记录的 `353 passed, 344 deselected`，本阶段净增：`+15 passed`

### 本阶段已完成的 bucket

- [x] Bucket A：phone / app / ingredient record 目标压实
- [x] Bucket B：open / close / same-object use 与 later-use 冲突
- [x] Bucket C：move / make space / hidden target / final placement 冲突
- [x] Bucket D：towel / cloth / paper towel 的 transport-vs-use
- [x] Bucket E：scale / tare / zero / measurement state-change
- [x] Bucket F：inspection / check / read label 与 later outcome

### 本阶段关键机制提升

- `phone generic-measure -> exact ingredient record target` 已从单目标压实到多 exact target 并存时的 evidence-starved target 优先追证。
- `open/uncap` 与 `weigh/use later`、`put back/store later` 的 mixed-horizon later-target recovery 已稳定，planner 会优先追更晚、更有判别力的目标轨迹。
- `make space / generic access` 相关冲突已能区分：
  - exact workspace creation
  - revealed slot / sink slot downstream placement
  - hidden-target retrieval
  - 仅有 broad workspace effect 时继续 withheld 或回退到更合理的 generic access
- `transport-vs-use` 已补齐 hand-use、surface cleanup、simple relocation 的 finish gate，不再只凭拿起、短暂接触或靠近台面就提前定答。
- `tap scale / zero out / weigh ingredient` 已要求 precontext、state-change 和 exact weighing-use 证据，broad measurement-meta 不再过早收口。
- `inspection` 相关残差已补齐：
  - label visible 不等于 check label
  - brief cooking inspection 不等于 empty/serve
  - later outcome 若已明确发生，可以直接压过弱 inspection

### 本阶段提交列表

- `ee1cb3f` `fix: prefer exact phone record targets with weakest evidence`
- `84e9114` `fix: tighten unresolved measurement intent gating`
- `9a63baa` `fix: prefer explicit later outcomes over weak inspection`
- `1445cf1` `fix: prefer hidden retrieval over broad make-space`

说明：
- 更早的相关专项提交仍保留在 git 历史中，例如 `59affd7`、`688cfa0`、`c4d1a6e`、`5a98202`、`ccfa467`、`e7e7d50`、`bed05a6`、`3e95a51` 等。
- 本阶段没有提交用户的无关脏改、数据、输出、密钥或未跟踪脚本。

### 当前未完成项

- bucket 级高频残差按当前文档已经全部勾完，但这不等于 why 专项完全结束。
- 仍未完成的是：
  - 更系统的 residual audit，确认是否还有新的高频簇值得单独立 bucket
  - 将当前 bucket 级规则提升与真实小样本/真实运行日志做更完整对照
  - 把 why 专项结果进一步并入完整 agent 的端到端验证

### 当前风险状态

- 当前 `action_intent` 专项回归已通过，文档记录范围内没有未修复失败测试。
- 已知剩余风险主要不是回归红灯，而是：
  - 新 residual bucket 可能在真实数据上继续暴露
  - why 逻辑虽已显著收紧，但仍以规则化 residual 收口为主，尚未做系统化长尾审计

### 下一步计划

- 第一优先：做一次新的 residual audit，确认在 `357 passed, 344 deselected` 之后是否还存在值得单独建 bucket 的高频 close-call。
- 第二优先：把当前 why 专项阶段成果整理成更稳定的阶段基线，供后续完整 agent 验证直接复用。
- 第三优先：将 why 专项与完整 agent 的小样本真实运行串起来，检查这些结构化收口是否真正改善端到端表现。

## 17.14 2026-06-08 residual audit 增量收口：`missing_direct_outcome_evidence` 现在会触发近窗 forced transition probe

本轮没有再硬开一个已经基本收口的旧 bucket，而是顺着 `17.12` 的 residual audit 继续往下查，补上了一条此前已经被 `graph_agent finalizer` 识别、但还没有被 `planner` 真正利用起来的恢复链。

### 本轮定位到的真实缺口

- `graph_agent` 已经会在一类 close-call 上写入：
  - `action_intent_resolution_withheld_for_missing_direct_outcome_evidence=1`
- 这类题典型是：
  - `flip / turn / shake cloth`
  - `move towel / cloth`
  - 某些弱 `residue release / relocation` 候选
- 它们的问题不是“更晚 long-horizon 目标未知”
- 而是“当前近窗里最关键的直接结果链还没看清”
- 例如：
  - `crumb 有没有真的掉进 sink`
  - `cloth 是不是真的翻到了另一面`
  - `move` 是否真的形成了直接 relocation outcome

此前缺口在于：

- finalizer 已经知道“不能定答”
- 但 planner 侧并没有把这个 marker 显式接到现有的 `forced followup_transition` 恢复链上
- 因而这类题更多还是靠通用 close-call 路径恢复，而不是优先去看最有判别力的近窗直接结果帧

### 本轮实现

- [x] 在 `planner._action_intent_verifier_blocked_prefers_forced_transition_probe(...)` 中新增：
  - 若最近 working memory 已写入
  - `action_intent_resolution_withheld_for_missing_direct_outcome_evidence=1`
  - 则直接启用 forced `followup_transition`
- [x] 保持原有保护边界：
  - 如果已经存在 `followup_transition` 帧
  - 不会重复触发同一轮 transition probe

### 本轮新增测试

- [x] `test_planner_action_intent_verifier_blocked_missing_direct_outcome_marker_forces_transition_probe`
  - 验证 `flip orange cloth` 这类题在已有 marker 后，会直接转去 `extract_frames_for_range(tag=...followup_transition)`
- [x] `test_planner_action_intent_verifier_blocked_missing_direct_outcome_marker_does_not_repeat_transition_probe`
  - 验证已有 transition probe 帧后，不会反复重复同一条近窗密采样路径

### 本轮回归

- [x] 定向测试：
  - `pytest -q tests/test_graph_agent.py -k 'missing_direct_outcome_marker_forces_transition_probe or missing_direct_outcome_marker_does_not_repeat_transition_probe or verifier_blocked_finish_post_action_blocker_prefers_followup_sampling'`
  - 结果：`3 passed, 700 deselected`
- [x] 专项回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`359 passed, 344 deselected`

### 本轮结论

- 这次更像是 `17.12` residual audit 下的一次“小型结构补线”，不是新开大 bucket。
- 但它有实际价值：
  - finalizer 已识别出的“缺直接结果链”现在不再只是一个被动 marker
  - 而是会主动驱动 planner 去补最该看的近窗关键帧
- 下一轮 residual audit 可以继续优先排查：
  - `missing_simple_relocation_evidence`
  - `missing_immediate_micro_outcome_evidence`
  - 是否还存在值得单独建 bucket 的高频 close-call

## 17.13 Goal 模式提示词

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
