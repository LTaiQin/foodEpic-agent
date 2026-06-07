# 16 逻辑推理专项 Goal 执行清单

## 16.1 文档目的

这份文档只服务一个目标：

把 `fine_grained_why_recognition` 从“已经有不少局部规则、但还不够像真正 agent 的 why 推理器”，推进到“有稳定证据闭环、能按冲突类型主动补证据、可做研究汇报”的专项模块。

后续进入 `goal` 模式时，默认只沿着这份清单推进，不再重复做泛泛讨论。

---

## 16.2 当前准确状态快照

### 16.2.1 当前专项范围

当前“逻辑推理”专项只指：

- 任务族：`fine_grained_why_recognition`
- 主要代码：
  - `food_agent/agent/graph_agent.py`
  - `food_agent/agent/planner.py`
  - `food_agent/agent/verifier.py`
  - `food_agent/agent/executor.py`
  - `food_agent/tools/agent_toolbox.py`
- 主要测试：
  - `tests/test_graph_agent.py`

### 16.2.2 当前稳定基线

- 专项回归命令：`pytest -q tests/test_graph_agent.py -k 'action_intent'`
- 2026-06-07 当前结果：`251 passed, 344 deselected`
- 相比本轮进入专项时的起点 `107 passed, 300 deselected`，当前阶段性增量为 `+135 passed`
- 当前执行策略：why 逻辑不再追求“接近完美覆盖”，而是维持“足够可用、回归稳定、无明显结构性退化”的维护态；后续优先级切换到完整 agent 功能闭环与小样本真实验证。

这说明 why 题已经不再是“直接把问题丢给模型猜答案”，而是已经存在完整骨架：

- `infer_action_intent`
- unresolved rerank
- direct-purpose override
- `resolve_action_intent_pairwise`
- `resolve_action_intent_future_use`
- `precontext / followup / followup_ext2` 补帧
- leaky memory/context 过滤
- deterministic finalize 收口

### 16.2.3 已经稳定完成的能力块

下面这些能力已经有明确代码锚点和测试保护，后续默认视为不可回退能力：

- [x] `make space` vs `exact downstream use`
- [x] `generic access` vs `exact targeted placement`
- [x] `specific fixture/workspace enablement` 优先于泛化 side-effect
- [x] `hidden-target access/retrieval` vs 泛化 reveal
- [x] `underneath cleaning` 在 hidden-target 语境下的降权
- [x] `same-object direct manipulation / direct cleaning / direct role use`
- [x] `measurement-base placement`
- [x] `postwash residue / water removal`
- [x] `postwash drying`
- [x] `premature drying before cleanup` 降权
- [x] `cleaning precondition` vs `future-use` 基础拆分
- [x] `dry hands` vs `wipe surface` 基础拆分
- [x] `temporary cleaning placement` vs `store away`
- [x] `brief cooking inspection` vs `empty / serve / pour`
- [x] `temporary set-aside` vs `finished / store / dry`
- [x] `hygiene surface protection staging`
- [x] `immediate reuse staging`
- [x] `glove-removal enablement`
- [x] `direct hazard avoidance`
- [x] `direct spill / mess avoidance`
- [x] pending resolution 回到专用 why tool，而不是直接退回通用排序
- [x] repeated vision failure 后的 textual fallback 基础兜底
- [x] pending `future_use/pairwise` 与 `precontext` 的候选感知式分流
- [x] why 题 `best_index` 不再在缺少 grounding 时自动视为可 finish
- [x] `pairwise` 中 `access / make space / exact placement` 的 outcome sufficiency 收紧
- [x] `future_use` 中 `finished/store/dry/open-close/access/hazard` 的 sufficiency 收紧
- [x] `planner` followup 路由改为 candidate-aware，不再被全选项 distractor 污染
- [x] why 题 secondary conflict 只在 specialized resolution 真正稳定后才允许降为非阻塞
- [x] `pairwise` 中 `generic hidden access` 会提升到 `reveal-then-take/place` 的更精确意图
- [x] `pairwise` 中 `direct spill / burn / mess avoidance` 会优先于 generic enablement
- [x] `pairwise` 已区分“纯粹腾出手”与“为了立刻处理下一个确切目标而腾出手”
- [x] `future_use` 中 cleaning hierarchy 已支持 `exact cleaning target / workflow initiation > supply retrieval`

### 16.2.5 最近已回填的代码进展

- `446068a`：`future_use` set-aside / drying / unfinished-cleanup hierarchy 继续泛化，不再局限 spoon/spatula/ladle
- `483547d`：`future_use` inspection hierarchy 对齐，支持 `serve/pour/empty -> check/boiling/doneness/consistency`
- `c5d2a8b`：`hidden-access` 的 pairwise 冲突改走 `followup_ext2` 门控，再进入 specialized resolve
- `d1432c9`：`future_use` reuse vs placement hierarchy 对齐，支持 `generic store/finished -> immediate reuse / exact final placement`
- `c8faa0c`：why followup 路由加入候选感知，优先看当前 top candidate 语义，不再被无关选项把路由拉偏
- `d788dab`：why conflict suppression 收紧，`best_index` 或普通 textual fallback 不再直接掩盖真实冲突
- `af05385`：`pairwise` hidden-target hierarchy 对齐，支持 `generic access -> reveal-then-take/place`
- `f15a6f0`：`pairwise` direct safety 意图提升，优先识别 spill / burn / mess avoidance
- `d19606a`：`pairwise` hand-free hierarchy 对齐，区分 exact next-target enablement 与 generic free-hand
- `e6bcd2a`：`future_use` cleaning hierarchy 对齐，支持 exact cleaning target 与 workflow initiation 的更细粒度优先级
- `05e2c20`：`future_use` hierarchy 继续扩到 `measurement / weigh / open-close / same-object role-use`
- `78ccd1a`：`future_use` hierarchy 继续扩到 `tap state switch / phase-switch`，并补 `generic_fill_limit_target_mismatch`
- `573761f`：why 题 repeated failure 后若 textual fallback 仍被 verifier 阻塞，planner 优先复用当前任务 `segment` artifact，而不是先退回泛化 `query_time`
- 本轮提交：why 题 repeated-failure textual fallback 加入 bounded acceptance，只有在当前题 artifact + grounding + 无 secondary conflict 时才允许 finish
- 本轮提交：why 题 textual fallback 的输入证据改为当前题/当前时窗 scoped evidence，不再把无关 session summary、旧时窗观测、planner/verifier 噪声直接喂给 `rank_choices_from_state`
- 本轮提交：why 题 repeated visual failure 的恢复顺序改为 `specialized resolution > textual fallback`，当前题已有足够原始帧时先走 `future_use/pairwise` 专用裁决
- 本轮提交：`future_use` toolbox hierarchy 补上 same-object residue-release，高频 `tap/shake/tilt/hit` 动作在专用裁决层优先翻正到“残余内容物掉回原容器/锅/碗/水槽”
- 本轮提交：why 题在首次 `infer_action_intent` 就暴露 `receptacle_outcome` 近窗歧义时，不再机械地先走一轮泛化 `followup`。现在会直接围绕动作尾部触发 `followup_transition`，主动去找“是否真的掉回 sink/pan/bowl/container”的决定性关键帧；同时这条路径会压过误触发的 `precontext`，避免 `flip cloth / shake / tap / tilt` 一类题被无关前置状态采样截走
- 本轮提交：新增并通过 2 条定向测试，分别保护：
  - `receptacle_outcome` 型 why close-call 会在第一次歧义时直接进入 `followup_transition`
  - 普通 `future_use` 型 why 题仍保持原来的初始 `followup`，不会被误改成近窗密采样
- 本轮提交：why 题的首次主动关键帧前移继续扩展到两类高频歧义：
  - `tap kitchen scale / press button / push switch` 这类 `state_change / open-close vs measure-use` 题，不再一上来稀疏补 8 秒长窗；现在会先围绕动作尾部后的 2 到 4 秒做 `followup_transition` 密采样，优先确认显示是否开机、归零、变化或出现其它决定性状态改变
  - `pick up tea towel / paper towel / cloth` 这类 `transport-vs-use` 题，当模型已经明确承认“要看动作后是拿去擦手/擦台面，还是只是放下/挪开”时，会先补动作后近窗关键帧，再决定是否需要回补 `precontext`；也就是说，agent 会先验证真实使用链，而不是默认先回头找前置状态
- 本轮提交：新增并通过 3 条定向测试，分别保护：
  - `tap kitchen scale` 无 tool trace 时，首次补证据优先进入 `followup_transition`
  - `tap kitchen scale` 在首次 `infer_action_intent` 仍不确定时，也会直接进入近窗 `followup_transition`
  - `pick up tea towel` 的 `transport-vs-use` close-call 会在第一次歧义时优先补动作后近窗关键帧，而不会先被 `precontext` 截走
- 本轮提交：同时收紧了 `transport-vs-use` 的前移触发门槛。只有模型已经显式承认 `need_more_evidence / ambiguity / whether X or Y` 时，才会抢先看近窗后果；普通高置信但只是泛化“暂时看不清”的 towel/cloth 题仍保持原有 `precontext` 路线，不会被误伤
- 本轮提交：why 初始化阶段的取帧策略也不再一刀切。过去只要还没有当前题时间窗帧，`planner` 就会统一先抽 `segment`；现在对于一开始就属于 `strict visual disambiguation` 的 why 题，会直接走 `initial transition probe`，先围绕动作尾部和后续短窗口抽更密的关键帧，而不是先看一组静态动作片段
- 本轮提交：这一步当前先覆盖：
  - `flip / shake / tip / turn` 一类 `residue_release / cleanup` 冲突
  - `pick up towel/cloth` 一类 `transport-vs-use` 冲突
  - `tap kitchen scale / press button` 一类 `state-change` 冲突
- 本轮提交：新增并通过 2 条初始化路由测试，分别保护：
  - `flip orange cloth` 在已有 `query_state` 但还没有当前题帧时，会直接走 `followup_transition`
  - `pick up tea towel` 在同样条件下，也会直接走 `followup_transition`
- 本轮提交：why 初始化阶段的主动取帧继续扩展到 `mixed-horizon` 场景。也就是完整选项里如果本来就存在“立刻检查/打开”对“稍后放回/称重/用途”的冲突，第一次没帧时也不再先抽普通 `segment`，而是直接走 `mixed-horizon followup_transition`
- 本轮提交：当前新增覆盖：
  - `take bottle`：`check the label` vs `put the bottle back in the fridge`
  - `take jar`：`open the jar` vs `use the jar to weigh the ingredients`
- 本轮提交：新增并通过 2 条初始化路由测试，分别保护：
  - `take bottle` 在初始路由下直接进入 mixed-horizon `followup_transition`
  - `take jar` 在初始路由下直接进入 mixed-horizon `followup_transition`
- 本轮提交：同时保留了原有保守边界：像 `open kitchen cabinet -> retrieve/put away` 这类没有“立刻微结果 vs 稍后用途”结构的题，仍然保持原来的 `segment` 起手，不会被误推到 transition probe
- 本轮提交：why 题的 finish gate 继续收紧到 mixed-horizon close-call。现在如果当前答案属于：
  - `check / read / inspect label/date`
  - `open / uncap / unscrew`
  - `put back / return`
  - `weigh / put on the scale`
  但证据文本里没有真正出现对应的显式链条，而只是“拿在手里、标签朝外、离开原位、靠近冰箱、靠近秤”这类弱迹象，那么 `graph_agent` 不再允许直接 finish
- 本轮提交：这一步同时作用于两层：
  - `finalizer`：`resolve_action_intent_*` 已经给出 `best_index` 时，如果 mixed-horizon 证据链不完整，会直接写入 `action_intent_resolution_withheld_for_mixed_horizon_claim=1`，阻止 deterministic finalize
  - `unresolved rerank`：如果 `candidate_evidence` 里的 top 候选只是弱 `check/open/put back/weigh` 迹象，而缺少真正排他性的显式证据链，也会继续写入 `action_intent_unresolved_rerank_withheld`，要求更多证据
- 本轮提交：新增并通过 3 条定向测试，分别保护：
  - `check label` 但只有“标签朝外/可见”时不能 finish
  - `put back in the fridge` 但只有“拿走/离开原位”时不能 finish
  - `check label vs put back` 的 unresolved rerank 在双方都只有弱支持时必须继续等待证据
- 本轮提交：why 题的 deterministic finalize 继续收紧到 `workspace / final placement / exact downstream use` close-call。现在如果当前答案只是：
  - `to make space / make room / generic workspace effect`
  - `put away / store / put back / right place`
  - `exact downstream target/use/placement`
  但证据文本里没有真正闭合“下一目标是谁、是否立刻发生、是否出现了确切终点/归位链”，而只是“台面更空了、位置更开阔了、物体离开了原位、暂时被放到一边”这类宽泛变化，那么 `graph_agent` 不再允许 deterministic finalize 直接 finish
- 本轮提交：这一步新增并通过 4 条定向测试，分别保护：
  - generic `make space` 但没有排除 `pick up whisk` 这类 exact downstream use 时不能 finish
  - `sink slot / exact targeted placement` 但只有“区域更空了”这类弱证据时不能 finish
  - 真的出现“下一物体 + 具体终点 + 立刻发生”链条时，exact targeted placement 仍允许正常 finalize
  - `put the napkin away` 但文本同时承认“left on the counter within reach / not final placement”时不能 finish
- 本轮提交：在 `finalizer` 收紧之外，`planner` 也开始前移处理这类 close-call。现在当 `resolve_action_intent_future_use / pairwise` 的当前答案落在：
  - `generic workspace / make space / make room`
  - `put away / put back / right place / proper place`
  - `exact sink-slot / exact downstream placement`
  但证据文本仍只是“area becomes more open / extra room / left on the counter / not yet visible”这类弱支持，或者上一轮已经被 `graph_agent` 写入 `action_intent_resolution_withheld_for_workspace_or_final_placement_claim=1` 时，planner 不会继续沿旧路径直接 finish，而会优先转入 `followup_transition`，主动补动作尾部和更晚结果帧
- 本轮提交：这一步新增并通过 2 条 planner 定向测试，分别保护：
  - `future_use` 弱 `sink slot / exact placement` 会优先进入 `followup_transition`
  - `pairwise` 弱 `right place / final placement` 会优先进入 `followup_transition`
- 本轮提交：同时保留原有稳定边界：如果当前只是普通高置信 `right place` 候选、但还没出现弱支持文本/withheld marker，则 planner 仍按原路线先进入 `pairwise`，不会被新规则误抢走
- 本轮提交：`transition probe` 的 reveal/access 路径继续细化，不再把所有“挪开物体后出现后方区域”的 why close-call 都压成同一个短窗。现在至少拆成三类：
  - `revealed target retrieval`：重点看 reveal 后 2 到 3 秒内是否真的把后方目标拿出来；
  - `revealed slot placement`：重点看 reveal 后稍晚一点是否真的把另一个物体放进腾出的槽位；
  - `revealed fixture enablement`：重点看 reveal 后是否立刻去打开/启动后方装置（如秤、门、把手等）
- 本轮提交：这一步新增并通过 3 条 transition-window 定向测试，分别保护：
  - `move cereal box -> take hidden jar` 会走更短、更近的 retrieval 窗口
  - `move mug -> place blue cup into freed slot` 会走稍后偏置的 slot-placement 窗口
  - `move tray -> turn on the scale behind it` 会走 fixture-enable 窗口，而不是落回泛化 mixed-horizon 或 hand-free 短窗
- 本轮提交：同时保留了 reveal 路径的边界条件：只有当前题、候选与证据文本里真正出现 `behind / hidden / reveal / slot / freed slot` 这类 reveal 语境时，才会启用这些 subtype；像 `take jar -> open vs weigh` 这种普通 mixed-horizon 题仍保持原来的 hybrid transition probe，不会被误判成 reveal-fixture
- 本轮提交：why 专项回归已更新到 `251 passed, 344 deselected`
- 本轮提交：why 题在 `followup_transition / followup_peaks` 之后新增“短时序证据复核”分支，先让 agent 总结动作后立刻结果、下一步手部动作和 `hand-free / access / next-use` 证据，再回到 `infer_action_intent`
- 本轮提交：`inspect_visual_evidence` 的写回字段扩到 `timeline_summary / immediate_result / next_action_hint / direct_purpose_hint / ambiguity_note`，并在 `needs_more_evidence=true` 时显式保留 `need_disambiguating_evidence`
- 本轮提交：why 题 `inspect_visual_evidence -> infer_action_intent` 的回跳逻辑已改为识别 timeline review；若复核仍判定证据不足，则继续 `followup_ext2` 或转入 `future_use / pairwise` 专用裁决，而不是重新退回只看 `segment` 的早收口路径
- 本轮提交：verifier 现已识别“最近一次 why timeline review 明确要求更多证据，但之后没有新的专用裁决/新证据重推”的未决状态；此时即使存在 `action_intent_best_index` 也会阻止 finish，避免 agent 因局部高置信再次早收口
- 本轮提交：timeline review 的结构化线索现在会反向指导下一轮补帧；`hand-free / access / reveal` 类歧义会触发更近、更密的尾部补帧，`next-use` 类歧义会自动拉长后续窗口，避免补帧仍然是统一模板
- 本轮提交：`future_use` 专用裁决现在只在存在 timeline review 线索时收缩候选集；它会优先保留当前冲突候选与 review 指向的类别，减少无关选项污染，同时在没有 review 时保持原来的全候选保守行为
- 本轮提交：`pairwise` 专用裁决也已接入 timeline review 候选收缩；只有在 review 已明确暴露 `hand-free / access / space / return` 类冲突时，才把候选缩成真正的对决项，否则保持原有候选顺序不动
- 本轮提交：timeline review 现在已前推到 specialized resolver 级别；`followup_route` 会直接利用 review 线索在 `future_use` 与 `pairwise` 之间做偏置分流，而不再完全依赖旧的静态启发式
- 本轮提交：`infer_action_intent / resolve_action_intent_pairwise / resolve_action_intent_future_use` 已不再把 timeline review 混成一串普通 notes；现在会把 `timeline_summary / immediate_result / next_action_hint / direct_purpose_hint / next_use_evidence / ambiguity_note` 组织成结构化证据块喂给模型，优先约束 why 判题
- 本轮提交：executor 写回的 `inspection; ...` 摘要已显式包含 `needs_more_evidence=1/0`，toolbox 新增 review guard；当 timeline review 已明确提示“证据不足/仍有歧义”，而当前 why 推理又缺少更强 post-action 结果时，会强制继续补帧而不是过早收口
- 本轮提交：why 题关键帧选择器已从静态模板升级为“按歧义类型选帧”；`future_use` 类歧义会优先保留更晚的 `followup_ext2/ext3/ext4`，`access / reveal / hand-free` 类歧义会优先保留更近的 `followup_transition`，`final placement / return / store` 类歧义会更偏后期结果帧
- 本轮提交：why 题最终送给视觉模型的图片序列现在会按推断时间强制排序，不再只是按类别拼接；这保证了 prompt 中“图片按时间顺序排列”的前提真实成立，减少模型对动作链条的误读
- 本轮提交：结合已有 why 真实残差，`towel / cloth / tea towel` 这类“拿起后是去擦具体目标、擦手，还是只是收起/放回”的题，followup 路由已从泛化 clean/dry 冲突升级为 `transport-vs-use` 专用 future-use 路由；当 top-2 候选里暂时没包含 relocation/store 候选，但全局选项里存在这类冲突时，也会主动补更长结果帧，而不是停在局部 clean/dry 猜测
- 本轮提交：verifier 现已不再把 `action_intent_unresolved_rerank_best_index` 当成 specialized resolution 成功信号；凡是被判定为 `future_use / pairwise` 型 why 题，没有真正跑出专用裁决成功时，默认不能结束。只有连续视觉失败触发的文本兜底路径，才允许在满足当前题 artifact grounding、前后置证据和无二级冲突时保守收口
- 本轮提交：针对 `workspace vs safety` 的 why 冲突，`planner` 的 pairwise 候选重排已不再只看上一轮 `reason`；现在会同时读取 timeline review 中的 `hot stove / burner / heat / spill risk` 等危险线索。当当前 top 候选仍只是“腾空间/开始备菜”这类安全无关解释时，会主动把 `safety_avoid` 选项拉入 pairwise 专用裁决，而不是让两个近义 workspace 候选彼此对决
- 本轮提交：新增正反两条保护测试，分别覆盖“timeline 已出现热源危险时主动拉入安全候选”和“缺少 workspace / hazard 线索时不乱注入 safety 候选”；专项回归已更新到 `194 passed`
- 本轮提交：修复了 `planner` 中 `inspect_visual_evidence -> why` 分支的缩进错误。此前 why 题在拿到 timeline review 后，本应进入 `transition probe / specialized resolution / reinfer` 的链路，实际却可能直接跌回兜底路径；这会显著削弱“主动补关键帧”和“延迟定答”的真实效果。该结构性问题已修复
- 本轮提交：why 题在 timeline review 已明确指出“当前仍有多个解释成立”时，不再一律先走泛化 `followup_ext2`。现在会优先尝试 `transition probe`：围绕动作尾部和紧随其后的短窗口做更密的关键帧搜索，先找能立刻排除竞争目的的决定性瞬间；只有这种近窗密采样不适用时，才退回更长窗口的额外 followup
- 本轮提交：新增测试分别覆盖“timeline review 的 hand-free / immediate-next-action 歧义优先触发 transition probe”和“later-use 类歧义仍保持较长 followup，而不是误触发近窗密采样”；专项回归已更新到 `195 passed`
- 本轮提交：`transition probe` 的时间窗不再是统一模板。现在会按 why 冲突类型选择短窗落点：
  - `hand_free / next manipulation`：更靠近动作尾部，优先看另一只手立刻去做什么；
  - `hidden access / reveal`：看动作后紧接着是否真的取到、碰到或露出目标；
  - `safety / spill`：优先看是否立刻远离热源、边缘或风险位置；
  - `final placement / return`：窗口后移，优先看物体最终是否真的被放回、归位或收起；
  - `future use`：窗口后移，优先看是否真的出现称重、倒空、检查、清洗等后续用途
- 本轮提交：新增测试直接保护“final placement 型冲突的 transition probe 会后移”和“future-use 型冲突的 transition probe 会后移”，而原有 `hand_free` 与 `access` 类近窗行为保持不退化；专项回归已更新到 `197 passed`
- 本轮提交：`verifier` 新增了 close-call finish gate。现在 why 题不会只因为“最近一次 specialized tool 跑出了 best_index”就自动视为可结束；如果最新 `infer_action_intent / resolve_action_intent_pairwise / resolve_action_intent_future_use` 结果仍然显示：
  - top-2 候选语义上仍然不同；
  - specialized confidence 还不够高；
  - `future_use` 的决定性观察为空，或 top-2 分差仍然很近；
  - `pairwise` 缺少明确 `direct_effect / downstream_action`
  那么 verifier 会继续要求 `need_disambiguating_evidence`，阻止过早 finish
- 本轮提交：新增两条 verifier 测试，分别覆盖“future-use close call 必须继续找证据”和“future-use 已有决定性后续观察时允许 finish”；专项回归已更新到 `199 passed`
- 本轮提交：修复 `planner` 中 why close-call 恢复链路的两个结构性缺口：
  - `_recover_action_intent_after_verifier_blocked_finish(...)` 现在已具备完整的候选索引规整能力，不再因为 helper 缺失而使 close-call 恢复逻辑脆弱；
  - `state-driven candidate` 与 `heuristic fallback` 都已接入同一条“verifier blocked -> targeted recovery”主线。也就是说，当 why 题已经被 verifier 明确判为 `need_disambiguating_evidence` 时，不管 planner 当前走的是模型规划还是启发式规划，都会优先回到 `extract_frames_for_range / resolve_action_intent_*` 这类专用补证路径，而不是被泛化的 `detect_audio_peaks` 或其他廉价候选截走
- 本轮提交：`transition probe` 对 `future_use` close-call 的触发条件已放宽到真正需要的范围。当 `resolve_action_intent_future_use` 仍没有 `decisive_observation`，且 top-2 分差很小或理由文本显式承认“仍不清楚/多个解释仍可能”时，planner 会优先在动作尾部后的短窗口做密采样，主动寻找“是否真的去称重/倒空/检查/放回”的决定性瞬间，而不是直接退回普通长窗 followup
- 本轮提交：新增两条 planner 测试，分别保护：
  - “why 题在 verifier 拦下 close-call finish 后，优先进入 targeted transition probe”
  - “已经存在决定性 future-use 证据时，不会误触发 close-call recovery”
- 本轮提交：专项回归已更新到 `201 passed`
- 本轮提交：`verifier` 现在会把 why 阻断原因显式写进 `summary`，例如 `why_blocker=precondition_context / post_action_evidence / future_use_close_call / pairwise_close_call`。这一步的意义不是改输出文案，而是把“证据还不够”细化成“缺动作前触发条件”还是“缺动作后结果证据”还是“top-2 close call 仍未压下去”
- 本轮提交：`planner` 的 `verifier blocked -> targeted recovery` 已从“只知道被挡住了”升级为“知道为什么被挡住了再选恢复路线”：
  - `precondition_context`：优先补动作前帧；
  - `post_action_evidence`：优先补动作后 followup / transition probe；
  - `future_use_close_call`：优先补动作后决定性结果帧，再回到 `future_use` 专用裁决；
  - `pairwise_close_call`：优先补更近的结果帧，再回到 `pairwise` 专用裁决
- 本轮提交：新增 4 条测试分别保护：
  - verifier 会在 summary 中标出 `precondition_context`
  - verifier 会在 summary 中标出 `post_action_evidence`
  - planner 在 `precondition blocker` 下优先补 `precontext`
  - planner 在 `post-action blocker` 下优先补 `followup`
- 本轮提交：专项回归已更新到 `205 passed`
- 本轮提交：`resolve_action_intent_future_use / resolve_action_intent_pairwise` 这两条专用裁决链路，在它们自己已经明确返回“证据仍不足/决定性观察为空/当前解释过于宽泛”时，不再默认先走 `detect_audio_peaks`。现在 planner 会优先尝试更贴题的 `transition probe`，直接围绕动作尾部后的短窗口补视觉关键帧，先确认“是否真的立刻称重/倒空/检查/放回”“是否真的拿到了后方物体/出现了直接物理效果”
- 本轮提交：只有当 `transition probe` 已经存在、或当前题不适合再做这类近窗密采样时，才会退回 `detect_audio_peaks`。也就是说，音频峰值现在从 why 专用裁决缺证据时的第一选择，降级成第二选择/兜底路径
- 本轮提交：新增与更新多条测试，分别保护：
  - `future_use` 不确定裁决优先转到 `followup_transition`
  - `pairwise` 不确定裁决优先转到 `followup_transition`
  - 已经存在 `transition` 关键帧后，才允许回退到 `detect_audio_peaks`
  - 宽泛 generic `future_use` 解释不会直接 finish，也不会先盲目扩长窗，而是先补更贴题的 transition 关键帧
- 本轮提交：专项回归已更新到 `207 passed`
- 本轮提交：`toolbox` 的 `future_use sufficiency` 继续向 measurement bucket 收紧：
  - `to measure the ingredients.` 这类 exact measurement future-use，不再只因为 support 里出现 `scale / measure` 就算过；现在必须看到更直接的 measurement role/use 证据，例如“放到秤上”“立即用于称量”“作为称量基底”“实际称量发生”
  - `to adjust the measurements.` / `read the measurements.` 这类 generic measurement-meta，也不再只因为处于 measurement 语境里就算过；现在必须看到更具体的 reading / tare / zero / app-update / readout 证据
- 本轮提交：新增 2 条 measurement 定向测试，分别保护：
  - generic measurement context 但缺 direct measurement role 时，`future_use sufficiency` 会继续要证据
  - generic measurement-meta 但缺 reading / tare / readout 信号时，`future_use sufficiency` 会继续要证据
- 本轮提交：`toolbox` 的 `future_use sufficiency` 继续向 inspection bucket 收紧：
  - `to check / inspect / read / label / date` 不再只因为 support 里出现 `look / visible / inspect` 之类宽泛字样就算过
  - cooking inspection（如 `check the boiling water / check the contents / check the consistency`）现在必须形成更完整的“短暂查看 + 仍停留在灶台/容器语境 + 非 pour/serve/empty”检查链
  - `check the label / check the date` 现在必须看到更直接的 reading chain，例如真的在看或读标签、日期、保质期或 printed text，而不是只看到“标签一度可见”
  - generic `inspect the object` 现在必须指出“在检查物体的什么状态/属性”，而不是只看到“物体被短暂拿起或转到视野里”
- 本轮提交：新增并通过 `5` 条 inspection 定向测试，分别保护：
  - generic inspect claim 缺 direct inspection chain 时继续要证据
  - brief boiling check chain 可直接通过
  - contents check chain 可直接通过
  - label/date 只有 visible 但没有 reading chain 时继续要证据
  - explicit label reading chain 可直接通过
- 本轮提交：`planner` 的 transition probe 新增了 `mixed temporal horizon` 路径，专门处理“一个候选需要看动作后立刻发生的微结果，另一个候选需要看稍后结果”的 why close-call：
  - 典型场景是 `check label/date` vs `put back / weigh / pour later`，以及 `open/close` vs `measure/use later`
  - 这类冲突不再被迫在“只看近窗”或“只看后移窗”里二选一；现在会主动取一个更长但仍然有针对性的混合窗口，兼顾动作后立刻微结果和稍后用途结果
  - 同时加入了更严格的触发门槛：只有模型已经明确承认 `need_future_evidence / ambiguity / need_more_evidence` 时，这条 mixed-horizon 路径才会抢在旧的后续词汇启发式之前生效，避免打坏已有稳定路径
- 本轮提交：新增并通过 `2` 条 mixed-horizon 定向测试，分别保护：
  - `check the label` vs `put back in the fridge` 会走 hybrid transition probe
  - `open the jar` vs `use the jar to weigh the ingredients` 会走 hybrid transition probe
- 本轮提交：`planner / verifier` 的 `direct post-action evidence` 判断继续收紧，不再把“不确定的后续动作词”误当成已经观察到的决定性结果：
  - `reason / answer` 里只要出现 `put back / returned / turned on / opened / poured` 这类词，过去就可能被误判为“已经看到动作后结果”
  - 现在 direct-evidence 判断改为只读取 `reason / decisive_observation / direct_effect / downstream_action`，不再把候选答案文本本身当作证据
  - 同时加入 clause 级别的 conservative 判定：如果相关子句里包含 `still unclear whether / not yet visible whether / may / might / could still / remains plausible` 这类不确定语言，即使同句出现 `put back / returned / weighed / opened`，也不算 direct evidence
- 本轮提交：新增并通过 `3` 条 direct-evidence 定向测试，分别保护：
  - planner 不会把 `still unclear whether it is returned to the fridge` 误判成 direct evidence
  - planner 仍会接受 `shortly after ... returned to the fridge ...` 这类真实后续结果链
  - verifier 在 `uncertain return` 语义下仍会阻止 finish
- 本轮提交：inspection、mixed-horizon、direct-evidence 定向测试均通过，why 专项总回归更新为 `212 passed, 344 deselected`
- 本轮提交：`planner` 新增了 `needed_observation` 驱动的补证据画像，不再只是把“还需要看什么”写进 `thought`：
  - `more post-action frames showing the direct physical effect` 这类缺口，现在会驱动更短、更密的 `followup_ext*` 补帧；
  - `whether the pot is put on the scale or used to pour water` 这类“真实后续用途”缺口，现在会驱动更长的 followup 窗口，而不是继续用统一模板；
  - `whether the bottle is read/checked first or put back in the fridge` 这类 immediate-vs-later close-call，即使选项文本本身较泛，也能直接驱动 mixed-horizon transition probe；
  - `verifier` 现在会把“结果里仍带有开放式 needed_observation”的 specialized resolution 视为未闭环，阻止它在 `need_more_evidence=false` 但实际上仍承认 `whether X/Y` 时过早 finish
- 本轮新增并通过 `4` 条定向测试，覆盖：
  - `needed_observation -> mixed-horizon transition probe`
  - `needed_observation -> short dense extra followup`
  - `needed_observation -> long future-use followup`
  - `verifier` 阻止“开放式 needed_observation 仍未闭环”的 finish
- 本轮提交：why 题“最终送给视觉模型的帧选择”也已开始受 `needed_observation` 控制，而不是只在补帧阶段生效：
  - `future-use / final placement` 型缺口会把预算优先给更晚的 `followup / followup_ext*`，避免关键后续结果被早期普通帧挤掉；
  - `reveal/access` 型缺口会把预算优先给 `followup_transition / followup_peaks`，减少无关的超晚帧干扰；
  - `mixed-horizon` 型缺口会同时保留近窗 transition 和更晚 followup/ext，避免“只看眼前一瞬”或“只看很后面”；
  - 关键帧 staging 现已加入按冲突类型的阶段优先级与预算裁剪，不再出现“明明抽到了关键后续帧，但最终送图时又被前面的普通帧挤掉”的结构性问题
- 本轮新增并通过 `2` 条定向测试，覆盖：
  - `future-use` 缺口下最终送图会优先保留 late followup/ext 证据；
  - `reveal/access` 缺口下最终送图会优先保留 transition/peak 证据
- 本轮提交：why 题的 `open_question / blocked finish / state-driven / low-confidence` 恢复路径继续统一到 specialized recovery 与新关键帧选择器：
  - `open_question` 恢复不再只偏向 `state_change` 题，`future_use / pairwise / pending_resolution` 型 why close-call 也会优先回到专用裁决，而不是退回泛化 `query_time`
  - `state-driven infer_action_intent` 与 `low-confidence` 恢复现在会继承已经补到的 followup/transition/ext 关键帧，不再固定退回“只看 4 张当前动作片段”的浅层判断
  - 当 why 题已经补到更晚的 `followup_ext*`，后续 `post_action_evidence` blocker 会从最近一次 followup 末尾继续向后补，而不是回退到动作起点重新采样
- 本轮新增并通过 `2` 条定向测试，覆盖：
  - `future_use open_question recovery -> specialized future_use resolution`
  - `state-driven infer_action_intent` 在 pending-resolution 阶段会保留 followup 关键帧
- 本轮提交：why 题的“短时序证据复核”不再只依赖 `transition / peaks` 两类关键帧。对于 `future_use / final_placement / mixed_horizon` 这类必须看更晚结果的 close-call，如果当前题已经补到了 `followup_ext2/ext3/ext4`，planner 现在会把这些晚期 artifact 也纳入 timeline review，而不是因为当前时窗裁剪过窄把真正关键的后续证据漏掉
- 本轮提交：why 题在 `pending_resolution` 阶段，若最近一次新增的是 `sample_sparse_frames(tag=followup_ext*)`，不再立刻回到 `resolve_action_intent_future_use` 直接裁决；现在会先做一次 timeline review，让 agent 先复盘“动作后立刻发生了什么、下一步去做什么、是否仍有多个 plausible 解释”，再决定是否继续裁决或继续补证据
- 本轮提交：关键帧小预算采样继续向 anchor-aware 收紧。`_sample_action_intent_stage_frames(limit=3)` 已从“固定拿最早两张加最后一张”改成“最早一张 + 最接近当前冲突锚点的一张 + 最后一张”，避免在 why 题最关键的 late followup / near-decisive 瞬间已经存在时，却仍被早期普通帧挤掉
- 本轮新增并通过 `3` 条定向测试，覆盖：
  - `future_use` 题即使没有 `transition/peaks`，只要已经补到 `followup_ext*` 也能触发 timeline review
  - `sample_sparse_frames(followup_ext2)` 后会先进入 timeline review，而不是直接 future-use resolution
  - `limit=3` 的 stage 采样会保留“起点 + 锚点 + 最新结果帧”
- 本轮提交：`planner` 新增了 why 候选的“语义救援”层，但只放在真正需要对立语义比较的 route / pairwise 决策中，不去破坏 generic `future_use` 的保守扩搜逻辑：
  - `flip / shake / tilt / tap / hit / knock` 这类动作，如果当前 top 候选里只剩 cleanup/open-close 近义项，而全局选项里还存在 `transfer_contents / residue_release` 或 `measure_weigh` 语义，planner 会把缺失的一侧主动拉回比较；
  - `towel / cloth` 这类 transport-vs-use close-call，如果当前 top 候选只剩 clean/dry 近义项，而全局存在 relocation/store 语义，planner 会在 route 选择层把缺失对立面补回；
  - 但 generic `future_use` 在没有 timeline review 线索时仍保持全候选保守行为，避免“还没看后续证据就先把搜索空间缩窄”。
- 本轮提交：`action_intent` 的量测语义桶继续补全，`measurement(s) / adjust` 这类表述现在也能稳定归入 `measure_weigh`，避免状态变化题里把“调秤/调读数”错当成普通 `open_close`
- 本轮新增并通过 `2` 条定向测试，覆盖：
  - `flip orange cloth` 会把缺失的 `residue_release` 候选重新拉回 top pair
  - `tap kitchen scale` 会把缺失的 `measure_weigh` 候选重新拉回 top pair
- 本轮提交：高歧义 why 桶的 repeated-failure 路径继续收紧。对于 `residue_release / state_change / transport-vs-use` 这三类必须看动作后证据的题：
  - `infer_action_intent` 连续失败后，不再允许直接降级成 `rank_choices_from_state -> finish`；
  - planner 会优先继续补 `followup` / 回到 `pairwise` 专用裁决 / 补空间证据，而不是让文本 fallback 直接收口；
  - verifier 也同步收紧：即使已经有当前题 artifact grounding，这三类题只要落入 repeated-failure textual fallback，也仍然默认视为 `need_disambiguating_evidence`，不允许当成稳定答案。
- 本轮提交：`_fallback_action_intent_pairwise_candidate_indices(...)` 已补齐高歧义桶的专用对决候选，不再只会回退到 `access / make space / put back` 这一类老 pair：
  - `flip / shake / tilt / tip` 会优先比较 `clean_dry` vs `transfer_contents`
  - `tap / press / push + scale/button/switch/knob` 会优先比较 `open_close` vs `measure_weigh`
  - `towel / cloth + pick up/move/take` 会优先比较 `clean_dry` vs `generic_relocation/final_place_return`
- 本轮新增并通过 `2` 条保护测试，覆盖：
  - strict `residue_release` bucket 的 textual fallback 会继续阻断 finish
  - strict `residue_release` bucket 在 planner 中不会再直接 `finish`，而是继续补帧或转专用裁决
- 本轮提交：`graph_agent` 的 unresolved rerank 继续向 `residue_release` 桶收紧，不再只相信模型已经明确写出的“掉回锅/碗/水槽”结果词：
  - 当选项本身明确是 `drop/fall/release ... into sink/pan/bowl/container`，且支持/上下文也能对上这个接纳位置时，会给 residue-release 一个较弱但结构化的加分；
  - 同时，对于 `flip cloth / turn cloth` 这类题，如果证据明确说“翻完后没有立刻继续擦，而且动作结束在 sink 方向”，`change side / other side / clean side` 这类 side-switch 解释会被降权；
  - 这一步不是让所有靠近 sink/bowl 的动作都变成 residue-release，而是专门补“结果词没被模型写全，但接纳位置结构已经足够明显”的残差。
- 本轮新增并通过 `2` 条定向测试，覆盖：
  - `flip orange cloth` 的 unresolved rerank 会从 `change side` 翻正到 `drop the crumb into the sink`
  - 若选项里根本没有明确目标接纳位置，新的弱 residue-release 规则不会乱触发
- 本轮提交：planner 侧也已经开始把 `sink/pan/bowl/container` 这类“接纳位置导向”当成独立的 `needed_observation` profile，而不是继续混在 generic `future_use` 或 `reveal` 里：
  - 对 `flip / shake / tap / hit / knock` 这类题，只要待确认的是“残余物是否掉回接纳位置”，transition probe 会优先走更短、更密的近窗补帧；
  - 最终送给视觉模型的关键帧选择也会优先保留 `followup_transition / followup_peaks`，减少被远处 `followup_ext*` 挤掉的情况；
  - 这条 profile 采用双条件约束：必须同时有“回落/排出类语义”和“接纳位置语义”，不会误伤普通 `future_use` 题。
- 本轮新增并通过 `2` 条 planner 定向测试，覆盖：
  - `flip orange cloth` 的 `needed_observation` 会触发更短的 `followup_transition` 窗口；
  - 同类题在最终选帧时会优先保留 `transition` 而不是 `ext2`
- 本轮专项总回归更新为 `231 passed, 344 deselected`
- 本轮小规模真实 probe：
  - 旧 `towel-cluster` 摘要中 `flip orange cloth` 仍是残差，正是本轮语义救援要消除的典型失败；
  - 新 `state-change-cluster` probe 已完成 `1/2`，当前已完成样本准确率 `1.0`；剩余样本仍在跑，说明本轮修改至少已开始覆盖真实状态变化桶，而不只是单测

### 16.2.4 当前真正的瓶颈

现在 why 逻辑最痛的部分已经不是“没有规则”，而是以下四层还没有完全闭环：

1. `planner evidence acquisition`
   - 还没有做到对每类语义冲突都稳定地补对证据。

2. `verifier / finalize`
   - 还可能让“目前最像的选项”在 direct-purpose 证据不足时过早通过。

3. `toolbox resolution sufficiency`
   - `pairwise / future_use` 的 sufficiency 规则还不够细，不同 bucket 的“决定性证据”定义还不统一。

4. `real replay / residual bookkeeping`
   - 还没有形成按 bucket 统计的真实残差压缩闭环。

---

## 16.3 当前完成度判断

这是对 why 逻辑专项本身的估计，不是整个 food agent 的估计。

- `graph_agent` 语义规则层：约 `78%`
- `planner` why 补证据路由：约 `70%`
- `verifier / finalize` 收口：约 `70%`
- `toolbox` why sufficiency 对齐：约 `76%`
- 真实小样本 replay 与分 bucket 统计：约 `15%`

综合判断：

- 当前 why 逻辑专项约完成 `78%-82%`

这意味着：

- 骨架已经成立；
- 局部难点也已经被压下去一批；
- 仍不等于完美；
- 但已经达到“可继续作为完整 agent 的一部分稳定使用”的程度；
- 后续新增 why 改动只处理广义回归或明显高频缺陷，不再为少量边角样例继续深挖。

---

## 16.4 完成定义

只有下面七条同时满足，才能认为 why 逻辑专项阶段性完成：

- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'` 持续全绿
- [ ] 高频语义残差簇完成至少一轮系统压缩，而不是继续靠单样例补丁
- [ ] planner 已形成“按冲突类型补前置帧/当前帧/结果帧”的稳定路由
- [ ] verifier 明确阻止 `future evidence pending` 和 `direct-purpose insufficient` 的过早 finish
- [ ] toolbox 对 `pairwise / future_use` 的 sufficiency 规则与 graph rerank 语义对齐
- [ ] 有一轮 why 真实 replay，能分 bucket 看见翻正收益
- [ ] 能输出研究汇报级别的专项结论：哪些 bucket 稳了、哪些 bucket 还没稳

补充说明：

- 这里的“完成”是 why 专项完成，不代表整个 agent 完成。
- 这里优先要求逻辑闭环成立，不要求现在就跑完整个大实验。

---

## 16.5 Goal 模式统一协议

后续每一轮 `goal` 执行，必须遵守下面协议。

### 16.5.1 选题协议

- [ ] 每轮只推进一个叶子项
- [ ] 每轮必须先说明本轮处理的“语义冲突对”
- [ ] 优先处理高频 bucket，不优先处理孤立怪样例
- [ ] 只有当一个样例明显代表一类冲突时，才允许从该样例切入

### 16.5.2 改动协议

- [ ] 优先改通用语义逻辑，不写 benchmark-specific hack
- [ ] 优先顺序：`graph_agent` -> `planner/verifier` -> `agent_toolbox`
- [ ] 禁止泄漏 gold、真值、不可见 benchmark 信息
- [ ] 禁止把提示词微调当作主要提分手段

### 16.5.3 测试协议

- [ ] 每补一个叶子项，至少新增 `2` 条测试
- [ ] 两条测试必须是同构冲突但不同措辞、不同对象或不同阶段
- [ ] 先跑定向测试，再跑专项总回归
- [ ] 每轮固定跑 `pytest -q tests/test_graph_agent.py -k 'action_intent'`

### 16.5.4 提交协议

- [ ] 每轮 commit 只包含与当前叶子项直接相关的改动
- [ ] commit message 必须写出语义簇本身
- [ ] 代码、测试、回归完成后，必须回填本清单

### 16.5.5 产物协议

每轮至少留下下面四类产物：

- [ ] 代码修改
- [ ] 新增测试
- [ ] 回归结果
- [ ] 本清单勾选状态更新

---

## 16.6 代码入口地图

### 16.6.1 `graph_agent.py`

why 题核心入口：

- `GraphAgent._resolve_action_intent_resolution_answer(...)`
- `GraphAgent._resolve_unresolved_action_intent_answer(...)`
- `GraphAgent._score_action_intent_candidate_evidence(...)`

why 题高价值 helper 区：

- downstream / direct-purpose：
  - `_action_intent_choice_is_exact_workspace_creation(...)`
  - `_action_intent_choice_is_exact_downstream_targeted_placement(...)`
  - `_action_intent_choice_is_exact_immediate_downstream_use(...)`
  - `_action_intent_choice_is_exact_pickup_path_enablement(...)`
  - `_action_intent_choice_is_direct_fixture_or_workspace_enablement(...)`

- cleaning / drying / temporary set-aside：
  - `_action_intent_choice_is_cleaning_tool_specific_target_use(...)`
  - `_action_intent_choice_is_cleaning_supply_retrieval(...)`
  - `_action_intent_choice_is_cleaning_workflow_initiation(...)`
  - `_action_intent_choice_is_surface_wipe_preparation(...)`
  - `_action_intent_choice_is_explicit_hand_drying_goal(...)`
  - `_action_intent_choice_is_postwash_residue_or_water_removal(...)`
  - `_action_intent_choice_is_postwash_drying_goal(...)`
  - `_action_intent_choice_is_immediate_reuse_staging(...)`
  - `_action_intent_choice_is_hygiene_surface_protection_staging(...)`
  - `_action_intent_choice_is_unfinished_cleanup_context_for_finished_or_storage(...)`
  - `_action_intent_choice_is_temporary_set_aside_not_finished(...)`

- hidden target / inspection：
  - `_action_intent_choice_is_hidden_target_access_or_retrieval(...)`
  - `_action_intent_choice_is_generic_hidden_reveal_or_access(...)`
  - `_action_intent_choice_is_generic_hidden_access_over_exact_reveal_use(...)`
  - `_action_intent_choice_is_generic_hidden_access_without_followup_use(...)`
  - `_action_intent_choice_is_exact_revealed_target_purpose(...)`
  - `_action_intent_choice_is_exact_reveal_then_take_or_place(...)`
  - `_action_intent_choice_is_brief_cooking_inspection_over_disposal(...)`

### 16.6.2 `planner.py`

why 题补证据主入口：

- `_action_intent_requires_followup(...)`
- `_build_action_intent_followup_sampling_decision(...)`
- `_action_intent_needs_precondition_context(...)`
- `_build_action_intent_precondition_sampling_decision(...)`
- `_build_action_intent_pairwise_resolution_decision(...)`
- `_build_action_intent_future_use_resolution_decision(...)`
- `_build_action_intent_missing_post_action_followup_decision(...)`
- `_action_intent_context_notes(...)`

### 16.6.3 `verifier.py`

why 题收口主入口：

- `_heuristic_verify(...)`
- `_has_stable_structured_family_answer_evidence(...)`
- `_has_action_intent_textual_rank_fallback_answer(...)`
- `_filter_non_blocking_conflicts(...)`

### 16.6.4 `executor.py`

why 题 pending / resolution 状态写回入口：

- `_record_action_intent_resolution_state(...)`
- `_clear_action_intent_resolution_memory(...)`
- `_merge_result_into_state(...)`
- `_update_reasoning_after_tool(...)`

### 16.6.5 `agent_toolbox.py`

why 题专用裁决工具入口：

- `resolve_action_intent_pairwise(...)`
- `resolve_action_intent_future_use(...)`
- `_apply_action_intent_pairwise_causal_hierarchy(...)`
- `_apply_action_intent_pairwise_sufficiency(...)`
- `_apply_action_intent_future_use_sufficiency(...)`

### 16.6.6 `tests/test_graph_agent.py`

后续扩展测试时，优先参考现有锚点：

- unresolved rerank：
  - `cleaning precondition`
  - `dry hands vs wipe surface`
  - `inspection vs serve/pour/empty`
  - `hidden target access vs exact retrieval`
  - `finished/store/dry vs temporary reuse`
  - `hazard / spill / glove-removal`

- planner：
  - `future_use_*`
  - `pairwise_*`
  - `precontext_*`
  - `pending_future_use_*`
  - `repeated vision failure fallback`

- finalizer / memory：
  - `prefers_latest_pairwise_resolution`
  - `prefers_latest_future_use_resolution`
  - `context_notes_drop_restored_model_conclusions`

---

## 16.7 详细待做清单

原则：

- 下面每个叶子项都必须做到“代码 + 测试 + 回归 + 清单回填”
- 没做完这四件事，不能打勾
- 允许调整顺序，但必须说明偏离默认顺序的原因

### 16.7.1 Phase 0：基线冻结与残差台账

目标：

先把当前 why 状态固定下来，避免后续推进过程中丢失基线与边界。

#### P0.1 基线快照

- [x] 重新跑 why 专项回归
- [x] 记录当前通过数：`121 passed, 332 deselected`
- [x] 将基线结果写入本清单
- [ ] 建立后续 replay 统一结果模板

进展补充：

- [x] 本轮代码推进后，专项回归已从 `107 passed, 300 deselected` 提升到 `121 passed, 332 deselected`

完成标准：

- 后续每轮都能和这条基线直接对比

#### P0.2 残差桶台账

- [ ] 建立 why 残差桶台账
- [ ] 每个 bucket 至少记录：
  - bucket 名称
  - 代表样例
  - 常见误选模式
  - 正确模式
  - 失效层级：`graph / planner / verifier / toolbox`
  - 当前优先级：`P0/P1/P2`

建议初始 bucket：

- [ ] `cleaning_precondition_vs_future_use`
- [ ] `dry_hands_vs_wipe_surface`
- [ ] `temporary_set_aside_vs_finished_store_dry`
- [ ] `inspection_vs_serve_pour_empty`
- [ ] `hidden_access_vs_exact_reveal_use`
- [ ] `generic_enablement_vs_exact_next_target`
- [ ] `hazard_or_spill_avoidance_vs_generic_mixing_or_moving`

进展补充：

- [x] 已做第一轮 why residual 审计，使用 `scripts/audit_graph_agent_finalize_residuals.py`
- [x] 当前已确认：
  - `fine_grained_why_recognition` 历史产物中有 `16` 个 deterministic run 记录
  - 另有 `2` 个需要继续压缩的 residual 历史记录
- [x] 已定位两个代表性 residual：
  - `fine_grained_why_recognition_224`：`infer_action_intent` 连续失败后进入 textual fallback，但历史运行在 artifact 恢复路径上存在旧接口不兼容问题；这已直接推动 `573761f`
  - `fine_grained_why_recognition_223`：存在 `need_alternative_evidence_path`，但已经有较多当前题时窗 artifact，属于 verifier/finalize 仍不够稳定的候选簇
- [ ] 还没有把这批 residual 正式整理成长期维护的 bucket ledger 与可复用报告模板

完成标准：

- 为什么还错，能按语义桶说清楚，而不是只会说“某个样例错了”

### 16.7.2 Phase 1：`graph_agent` 语义规则剩余压缩

目标：

把还没完全压平的 why 残差继续往“语义簇”方向收敛，而不是继续长单点补丁。

#### P1.1 `finished/store/dry` 与 `temporary set-aside` 的普通工具泛化

- [ ] 不只覆盖 spoon/spatula/ladle，要继续泛化到更多普通厨房工具
- [ ] 需要区分：
  - 暂时放下待马上复用
  - 为了避免弄脏台面而摆放
  - 真正 finished with object
  - 真正 store away
  - 真正 postwash dry

进展补充：

- [x] 已把这组语义从少数 utensil 名称继续扩展到更通用的放置后果 bucket，并接入 `agent_toolbox._apply_action_intent_future_use_causal_hierarchy(...)`：
  - `generic store / finished / dry -> hygiene surface protection staging`
  - `generic store / finished / dry -> unfinished cleanup in sink/wash area`
  - `generic store / finished -> postwash drying`
  - `generic dry -> finished with object`（当没有 wet-after-wash context 且也不存在 immediate reuse 时）
- [x] 这轮不再只依赖 spoon/spatula/ladle：
  - 已扩到 `tongs / plate / cup / whisk` 等更多普通厨房对象
  - 规则核心改为“放置语境 + 后续证据”，而不是对象名字本身
- [x] 已新增并通过 `4` 条定向测试：
  - `finished -> hygiene placement`
  - `finished -> postwash drying`
  - `store -> unfinished cleanup`
  - `generic dry -> finished with object`
- [x] 已补跑局部回归：
  - `pytest -q tests/test_graph_agent.py -k 'future_use_causal_hierarchy or future_use_sufficiency'`
  - 结果：`23 passed, 424 deselected`
- [x] 已补跑专项总回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`120 passed, 327 deselected`

当前剩余缺口：

- [ ] 真实 replay 上还未统计这组泛化规则对不同 object family 的翻正收益
- [ ] `final placement / unfinished cleanup / postwash drying` 三者在更复杂多阶段洗涤链中的边界还可以继续压
- [ ] `hygiene placement` 仍可继续扩到更多 dirty/oily/surface-protection 措辞变体

完成标准：

- 这类题不再依赖个别工具名字触发

#### P1.2 `inspection` 与 `serve/pour/empty/check cooked` 的更细粒度拆分

- [ ] 继续压 `look into / check contents / check water level / check cooked state`
- [ ] 区分：
  - 短暂查看状态
  - 真正倾倒/清空
  - 真正准备上菜

进展补充：

- [x] 已把 inspection 专项 hierarchy 接入 `agent_toolbox._apply_action_intent_future_use_causal_hierarchy(...)`：
  - 当 evidence 明确显示 `brief lift / look inside / near hob or in-container check / no tilt / no serving destination / no transfer away` 时
  - 会把答案从 `serve / pour / empty / drain` 一类泛化后续用途，翻到更直接的 `check / inspect / boiling / doneness / consistency`
- [x] 这轮不是写死单样例，而是按语义簇扩展到多种容器与检查目标：
  - `pot / saucepan` 的 `boiling water / doneness check`
  - `frying pan` 的 `contents check`
  - `mixing bowl` 的 `consistency check`
- [x] 已新增并通过 `3` 条 inspection 专项 hierarchy 定向测试：
  - `empty -> boiling check`
  - `serve -> contents check`
  - `pour out -> consistency check`
- [x] 已补跑局部回归：
  - `pytest -q tests/test_graph_agent.py -k 'future_use_causal_hierarchy or future_use_sufficiency'`
  - 结果：`19 passed, 424 deselected`
- [x] 已补跑专项总回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`120 passed, 323 deselected`

当前剩余缺口：

- [ ] `inspection` 与 `measure / weigh / open-close` 的更细粒度边界还可继续压
- [ ] 真实 replay 上还未统计这组 inspection 规则实际翻正了多少样例

完成标准：

- inspection 类 why 题不再被泛化结果动作吞掉

#### P1.3 `hidden target` 类的闭环泛化

- [ ] 扩展 `reveal -> take`
- [ ] 扩展 `reveal -> place`
- [ ] 扩展 “只是挪开以便看见/够到”但没有后续确切使用

完成标准：

- `access`、`retrieve`、`make space`、`put in right place` 四类不会持续互相污染

#### P1.4 `risk / hazard / spill / hygiene` 直接目的泛化

- [ ] 区分真正 direct-risk management 与 generic manipulation
- [ ] 继续压：
  - burn prevention
  - spill prevention
  - dirty-surface prevention
  - glove-removal enablement

完成标准：

- 这类题优先回到 direct-purpose，而不是被 generic move / mix / set aside 抢走

### 16.7.3 Phase 2：Planner 侧 why 补证据闭环

目标：

让 planner 真正像 agent 一样“按冲突类型找证据”，而不是只会多采几张图。

#### P2.1 建立 why 冲突 -> 补证据路由表统一入口

- [ ] 为每个高频 bucket 明确默认补证据方向：
  - `cleaning_precondition_vs_future_use` -> `precontext`
  - `dry_hands_vs_wipe_surface` -> `precontext` + 局部 followup
  - `temporary_set_aside_vs_finished_store_dry` -> `followup`
  - `inspection_vs_serve_pour_empty` -> 局部 `followup`
  - `hidden_access_vs_exact_reveal_use` -> `followup/ext2`
  - `hazard_or_spill_avoidance` -> 前后局部窗口同时看

- [ ] 形成代码中的统一判定入口，而不是 scattered if-else

进展补充：

- [x] 已在 `planner` 中把 `pending future_use/pairwise resolution` 与 `precondition-sensitive` 冲突接上统一分流：
  - 当 why 题本质依赖前置状态且尚无 `precontext` 时，优先回补 `precontext`
  - 不再无条件直接回到 `resolve_action_intent_future_use`
- [x] 已修复一类关键路由污染：
  - `precondition` 检查先看当前候选语义，再决定是否补前置帧
  - 全选项 fallback 现在只保留给 `clean_dry` 这一类确实需要的窄场景
  - 避免被无关 distractor 选项把 why 题错误拉去做 `precontext`
- [x] 已补上 followup 的 candidate-aware 版本：
  - `future_use`、`pairwise`、`precontext` 的 followup 选路优先继承当前 top-2 候选语义
  - 不再因为完整 choice set 中存在无关 `clean_dry`/`precondition` 词面而误走前置路由
- [x] 已把 why 题 `textual fallback -> alternative evidence recovery` 的恢复入口继续候选化到当前任务 artifact：
  - 当 `fine_grained_why_recognition` 在连续视觉失败后暂时回落到 `rank_choices_from_state`
  - 且 verifier 仍报告 `need_alternative_evidence_path`
  - 当前会优先复用 `fine_grained_why_recognition_segment` artifact，而不是先退回泛化 `query_time`
  - 这一轮是直接由真实 residual why run 反推出来的 planner 缺口，不是单测孤例补丁

完成标准：

- 能解释“为什么这道题该补前置帧，而不是补结果帧”

#### P2.2 `future_use` 类优先看结果帧

- [ ] 当冲突本质是“当前动作是为了后面做什么”时，优先补动作后结果帧
- [ ] 不允许在还没看到结果帧时直接 finish
- [ ] `followup` 不足时才扩到 `followup_ext2`

进展补充：

- [x] why 题在 `future_use` / `pairwise` 冲突下，当前已通过 `followup -> ext2 -> resolution` 路径稳定要求动作后证据
- [x] `verifier` 已阻止“没有 post-action grounding 但只有 `best_index`”的 why 题过早 finish

完成标准：

- `future_use` 题不再只靠当前帧和文字描述裁决

#### P2.3 `cleaning-precondition` 类优先看前置帧

- [ ] towel/cloth/sponge 类题优先检查动作前：
  - 手是否湿
  - 台面是否脏
  - 是否刚洗完
  - 是否存在热/油/水/污渍触发

- [ ] 仅当前置仍不足时，再补动作后的局部结果帧

进展补充：

- [x] 已新增并通过针对性测试：当 `tea towel / cloth / sponge` 一类 why 题处于 `pending future_use resolution`，但缺少前置状态证据时，planner 会先补 `precontext`
- [x] 当前 `dry hands / wipe surface / clean` 一类 why 题，不再只靠动作瞬间和词面竞争直接收口
- [x] 已新增并通过 `pairwise` 版本的 precontext 回补测试：
  - `cleaning gap` 会先补前置状态
  - `spill gap` 会先补前置状态

完成标准：

- `dry hands`、`wipe surface`、`clean` 不再主要靠词面竞争

#### P2.4 `hidden-access` 和 `pairwise` 类优先看动作后真实后果

- [ ] `move glass / move bowl / move tray` 一类题优先看后续：
  - 是否取走后方物体
  - 是否只是腾出空间
  - 是否把物体放到更准确位置

- [ ] 如果动作后帧不足，必须先补，不要直接 pairwise resolve

进展补充：

- [x] 已把 `hidden access vs exact reveal-use` 从普通 future-use 不确定性里分离出来，纳入更明确的 pairwise 冲突识别：
  - 现在 `action_intent_conflict_profile(...)` 会显式识别 `generic hidden access` 与 `exact reveal-then-take/place` 的冲突簇
  - 避免这类题被泛化为普通 future-use 或直接 finish
- [x] 已在 `planner` 中加入更硬的后果帧门控：
  - 对 `hidden access vs exact reveal-use`，若只有一轮短 followup 且还没有明确 reveal 后真实后果，就先补 `followup_ext2`
  - 只有当 evidence 已经明确写出 “revealed target 被立刻取走 / revealed slot 被立刻使用” 时，才允许直接进入 pairwise resolution
  - 已补过 `followup_ext2` 后，不再无限补帧，而是回到 pairwise 二选一裁决
- [x] 已新增并通过 `3` 条 planner 定向测试：
  - `hidden access` 在首轮短 followup 后仍不明确时，会先补 `followup_ext2`
  - evidence 已明确 reveal 后 exact target use 时，允许直接 pairwise resolve
  - 已有 `followup_ext2` 时，允许进入 pairwise 而不是继续无限补帧
- [x] 保持普通 `access vs make space` 旧路径不回退：
  - 新门控只作用于 `hidden access vs exact reveal-use`
  - 其它普通 `access/space` pair 仍按原有 followup 后直接 pairwise 的路径走

本轮验证补充：

- [x] 定向回归：`pytest -q tests/test_graph_agent.py -k 'planner_action_intent_hidden_access_pairwise or planner_action_intent_generic_access_space_pair_routes_to_pairwise or planner_action_intent_candidate_pairwise_route_ignores_fullset_future_use_distractors or planner_action_intent_high_confidence_outcome_pair_forces_pairwise_after_followup'`
  - 结果：`6 passed, 434 deselected`
- [x] 专项总回归：`pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`120 passed, 320 deselected`

完成标准：

- planner 不再在缺后续帧时强行让 pairwise 裁决

#### P2.5 artifact 复用边界进一步收紧

- [ ] 只复用当前题时间窗内的有效 artifact
- [ ] 继续过滤旧题 specialized 结论、旧 session summary、旧 `action_intent_*` note
- [ ] 需要时补测试保证不会恢复泄漏

完成标准：

- 旧题结论不会借 context note 渗进当前题

#### P2.6 repeated failure 的降级规则系统化

- [ ] 只有连续视觉失败，才允许回退到 textual rank
- [ ] 区分：
  - 没有后续帧
  - 有后续帧但仍冲突
  - 工具调用失败
  - 真正视觉证据耗尽

进展补充：

- [x] why 题当前已区分“连续视觉失败后的文本 fallback”与“fallback 后 verifier 仍不满意”的恢复阶段：
  - 若 `infer_action_intent` 连续失败，允许暂时回退到 `rank_choices_from_state`
  - 但若 verifier 仍提示 `need_alternative_evidence_path`，不会直接在 `query_time` 上空转
  - 而是优先复用当前任务的 `segment` artifact，先走更便宜的原始证据恢复路径
- [x] 已新增并通过 planner 定向测试：
  - `planner_action_intent_verifier_blocked_text_fallback_prefers_cached_segment_artifacts`
- [x] 已补跑 planner why 定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'planner_action_intent'`
  - 结果：`38 passed, 415 deselected`
- [x] 已补跑专项总回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`121 passed, 332 deselected`

完成标准：

- fallback 是受控降级，不是 planner 的默认捷径

### 16.7.4 Phase 3：Verifier / Finalize 收口加强

目标：

why 题最终答案不能只因为“某个候选最像”就通过，必须强调 direct-purpose sufficiency。

#### P3.1 `future evidence pending` 严格阻止 finish

- [ ] 只要存在：
  - `action_intent_need_future_evidence=1`
  - `action_intent_pending_resolution=*`
  - `need_disambiguating_evidence`

  就默认不允许直接 finish，除非出现更强 direct override

进展补充：

- [x] `verifier` 已显式把以下状态视为 why 题未完成证据：
  - `action_intent_need_future_evidence=1`
  - `action_intent_pending_resolution=*`
  - `need_disambiguating_evidence`

完成标准：

- pending 状态不会被 finish 短路

#### P3.2 `stable structured answer` 判定收紧

- [ ] why 题的“稳定答案”不再只看 `best_index`
- [ ] 需要同时检查：
  - 没有 pending future-evidence
  - 没有 unresolved competing candidate
  - 有 direct-purpose 级证据或专用 resolution 结论

进展补充：

- [x] why 题当前不再因为存在 `action_intent_best_index=*` 就自动视为稳定答案
- [x] `verifier` 新增了 `precondition grounding` 与 `post-action grounding` 检查
- [x] 对 `textual fallback` 仍保留“连续视觉失败后才能作为稳定答案”的特例
- [x] 已把 repeated-failure textual fallback 收紧成 bounded acceptance：
  - 当 why 题已经发生 repeated visual failure，且只剩 textual fallback
  - 只有在当前题 `segment / followup / ext2` artifact 已经齐备、`pre/post grounding` 满足、secondary conflict 已消失时
  - 才允许清掉 `need_alternative_evidence_path`
  - 否则 verifier 继续阻塞 finish，不再把 textual fallback 直接当最终稳定答案
- [x] 已新增并通过 `3` 条 verifier 定向测试：
  - `textual_fallback_without_current_task_artifacts_keeps_alternative_evidence_blocking`
  - `textual_fallback_with_current_task_artifacts_and_grounding_can_finish`
  - 以及更新后的 `accepts_ranked_best_index_after_repeated_vision_failures`
- [x] 已补跑定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'verifier_action_intent_textual_fallback or verifier_action_intent_accepts_ranked_best_index_after_repeated_vision_failures or verifier_action_intent_requires_post_action_grounding_before_stable_finish or verifier_action_intent_grounded_best_index_without_specialized_resolution_keeps_secondary_conflicts_blocking'`
  - 结果：`6 passed, 449 deselected`
- [x] 已补跑 planner 相关保护测试：
  - `pytest -q tests/test_graph_agent.py -k 'planner_action_intent_verifier_blocked_text_fallback_prefers_cached_segment_artifacts'`
  - 结果：`1 passed, 454 deselected`
- [x] 已补跑专项总回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`123 passed, 332 deselected`
- [x] 已继续把 why textual fallback 的恢复链收紧到“当前题 scoped evidence”：
  - `rank_choices_from_state` 不再直接吃 `evidence_bundle[-12:] + working_memory[-20:]`
  - 当前改为优先只保留当前题、当前时窗、非 leaky 的 observation / state / location 证据
  - 会过滤掉无关 task 的 `session summary`、旧时窗观测、`planner_thought=`、`verifier=`、`tool_failure tool=` 一类噪声
- [x] 已新增并通过 `2` 条 planner 定向测试：
  - `textual_fallback_scopes_evidence_to_current_window`
  - `textual_fallback_drops_planner_and_verifier_noise_from_working_memory`
- [x] 已补跑 fallback 定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'planner_action_intent_repeated_vision_failures_fallback_to_textual_rank or planner_action_intent_textual_rank_is_not_overridden_after_repeated_vision_failures or planner_action_intent_textual_fallback_scopes_evidence_to_current_window or planner_action_intent_textual_fallback_drops_planner_and_verifier_noise_from_working_memory or planner_action_intent_verifier_blocked_text_fallback_prefers_cached_segment_artifacts'`
  - 结果：`5 passed, 452 deselected`
- [x] 已再次补跑专项总回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`125 passed, 332 deselected`
- [x] 已继续把 repeated-failure 恢复链从“直接退回文本排序”推进到“优先专用裁决”：
  - 当 `infer_action_intent` 连续失败 `3` 次后
  - 如果当前题已经有足够的 `segment/precontext/followup` 原始帧
  - 当前会优先进入 `resolve_action_intent_future_use` 或 `resolve_action_intent_pairwise`
  - 只有在确实没有更合适的 why 专用裁决链时，才回落到 `rank_choices_from_state`
- [x] 已新增并通过 `2` 条 repeated-failure 专项测试：
  - `repeated_vision_failures_with_followup_frames_prefers_future_use_resolution`
  - `repeated_vision_failures_with_followup_frames_prefers_pairwise_resolution`
- [x] 已补跑 repeated-failure / fallback 定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'planner_action_intent_repeated_vision_failures_fallback_to_textual_rank or planner_action_intent_repeated_vision_failures_with_followup_frames_prefers_future_use_resolution or planner_action_intent_repeated_vision_failures_with_followup_frames_prefers_pairwise_resolution or planner_action_intent_textual_rank_is_not_overridden_after_repeated_vision_failures or planner_action_intent_textual_fallback_scopes_evidence_to_current_window or planner_action_intent_textual_fallback_drops_planner_and_verifier_noise_from_working_memory or planner_action_intent_verifier_blocked_text_fallback_prefers_cached_segment_artifacts'`
  - 结果：`7 passed, 452 deselected`
- [x] 已再次补跑专项总回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`127 passed, 332 deselected`
- [x] 已基于 why 全量分布统计，优先补高频 `transfer_contents / hand_free_enablement / clean_dry / tap-shake-tilt-hit` 相关簇
  - why 数据中最高频 active combo 仍集中在 `transfer_contents` 及其与 `hand_free_enablement / clean_dry / open_close / final_place_return` 的组合
  - 因此这一轮优先补的是 same-object residue-release，而不是继续只压 `access vs make space`
- [x] 已在 `agent_toolbox._apply_action_intent_future_use_causal_hierarchy(...)` 中补上 same-object residue-release hierarchy：
  - 当证据明确显示 `tap/shake/tilt/hit` 导致残余内容物掉回原 `pot/bowl/pan/sink`
  - 会优先翻到 direct residue-release 候选
  - 不再停留在 generic `stir / put down / hand-free / broad drying` 解释
- [x] 已新增并通过 `2` 条 toolbox 定向测试：
  - `future_use_causal_hierarchy_prefers_direct_residue_release_over_stirring`
  - `future_use_causal_hierarchy_prefers_direct_residue_release_over_put_down_for_hit_spatula`
- [x] 已补跑 residue-release / action_intent 回归：
  - `pytest -q tests/test_graph_agent.py -k 'direct_residue_release or future_use_causal_hierarchy_prefers_direct_residue_release or action_intent'`
  - 结果：`129 passed, 332 deselected`
- [x] 已补跑 residue-release 专项 toolbox 测试：
  - `pytest -q tests/test_graph_agent.py -k 'agent_toolbox_future_use_causal_hierarchy_prefers_direct_residue_release_over_stirring or agent_toolbox_future_use_causal_hierarchy_prefers_direct_residue_release_over_put_down_for_hit_spatula'`
  - 结果：`2 passed, 459 deselected`

完成标准：

- `best_index` 本身不再等价于“足够完成”

#### P3.3 `finished/store/dry` 的 sufficiency 强化

- [ ] 对 `finished/store/dry` 这类常见误判项增加 stricter block：
  - 没有明确收纳/远离工作区/真正结束使用证据时，不准轻易 finish

进展补充：

- [x] `toolbox` 的 `future_use` sufficiency 已新增对 `finished with object` 的显式约束：
  - 没有“真正不再使用”的后续证据时不能轻易通过
  - 若 evidence 显示“放在旁边且很快再次使用”，会反向阻断 finished 结论

完成标准：

- 结束态类选项必须拿到更明确证据才能过

#### P3.4 conflict filtering 细化

- [ ] 不要过早把 `conflicting_locations`、`conflicting_state_observations` 一律过滤掉
- [ ] 对 why 题只在 specialized resolution 真正稳定后再做非阻塞过滤

进展补充：

- [x] 已收紧 why 题 conflict suppression：
  - 只有 specialized `pairwise/future_use` resolution 真正稳定，且不存在待补证据缺口时，secondary conflict 才能转为非阻塞
  - 仅有 `best_index`、普通 textual fallback 或弱文本解释时，不再自动吞掉 `conflicting_locations` / `conflicting_state_observations`

完成标准：

- conflict 过滤不再掩盖真正未解决的 why 冲突

#### P3.5 textual fallback 优先级再确认

- [ ] finalize 优先级保持统一：
  - specialized resolution
  - pending-resolution completion
  - unresolved rerank
  - textual fallback

- [ ] why 题只有在重复视觉失败后才允许 textual fallback 真正生效

完成标准：

- finalize 的答案来源稳定、可解释、可回放

### 16.7.5 Phase 4：Toolbox 的 why 裁决规则对齐

目标：

把 `agent_toolbox` 的 pairwise / future-use 工具内部裁决逻辑，和 `graph_agent` 现有语义拆分真正对齐。

#### P4.1 `resolve_action_intent_pairwise` 的 causal hierarchy 对齐

- [ ] 对齐 `access vs make space`
- [ ] 对齐 `generic enablement vs exact next target`
- [ ] 对齐 `risk/hazard direct-purpose`

进展补充：

- [x] 已完成并验证：
  - `generic hidden access -> exact reveal-then-take/place`
  - `direct spill / burn / mess avoidance -> generic moving/mixing/enablement`
  - `exact next-target enablement -> pure free-hand enablement`

完成标准：

- pairwise 裁决不再与 graph rerank 语义方向打架

#### P4.2 `future_use` 的 sufficiency 规则按 bucket 细化

- [ ] 明确不同 bucket 的决定性观察：
  - `future use` -> 看到动作后真实用途
  - `store/finished` -> 看到脱离活跃工作区或明显结束
  - `dry` -> 看到 wet-after-wash 或 drying context
  - `wipe/clean` -> 看到接触擦拭对象或清洁上下文

进展补充：

- [x] 已在 `agent_toolbox._apply_action_intent_future_use_sufficiency(...)` 中补齐并验证以下 bucket 的决定性观察约束：
  - `store/finished`
  - `dry hands`
  - `wipe surface / clean`
  - `serve/consume`
  - `open/close`
  - `hand-free enablement`
  - `access/retrieve`
  - `hazard / spill avoidance`

- [x] 已修复一个关键逻辑漏洞：
  - 候选答案文本 `answer` 不再被混入 `evidence_text`
  - 泛化 `reason` 不再作为正证据直接“自证候选成立”

- [x] 已在 `agent_toolbox._apply_action_intent_pairwise_sufficiency(...)` 中补齐并验证以下约束：
  - `access/retrieve` 必须看到真实取物/可达结果，不能只靠泛化 reveal
  - `make space` 必须看到空间真的被腾出来或后续可用
  - `put in right place` 必须看到更精确的槽位/目标位置结果
- [x] 已在 `agent_toolbox._apply_action_intent_future_use_causal_hierarchy(...)` 中补齐 cleaning 专项 hierarchy：
  - `exact cleaning target > cleaning supply retrieval`
  - `cleaning workflow initiation > supply retrieval`（当 exact target 仍不明确时）
- [x] 已继续把 `future_use` causal hierarchy 扩展到高频收尾误判簇：
  - `temporary relocation / generic store / finished with object -> immediate reuse staging`
  - `generic store / finished with object -> exact final placement`
  - 现在当 evidence 明确显示“放在旁边待会继续用 / 很快再次使用”时，会直接把答案推到 `immediate reuse`
  - 现在当 evidence 明确显示“回抽屉 / 回橱柜 / 回挂钩 / 回 holder slot”时，会直接把答案推到更精确的 final placement，而不是停留在 broad `store away`
- [x] 已把 `future_use` causal hierarchy` 继续扩到 `measurement / weigh / open-close / same-object role-use`：
  - `generic measurement-meta / broad dry-store-serve reading -> exact measurement role`
  - `drying interpretation -> measurement-base placement`
  - `later downstream scale/tap use or generic hand-free reading -> same-object open/uncap`
  - 现在当 evidence 明确显示“当前被移动的对象马上被拿去称量 / 放到秤上作为称量基底 / 仍拿在手里并立刻打开同一对象”时，会优先翻到更直接的 exact future-use，而不是停留在 broad `adjust / dry / turn on scale`
- [x] 已继续把 `future_use` hierarchy 扩到 `tap state switch / phase-switch`：
  - `generic full container -> tap state switch for active saucepan/pan`
  - `broad boil-speed outcome -> concrete tap-state transition`
  - 同时在 `future_use sufficiency` 中补了 `generic_fill_limit_target_mismatch`
  - 现在当 evidence 明确显示“镜头一直围绕 tap + saucepan/pan，且是在冷热水切换/加速烧开”的链条上时，会优先翻到更直接的 tap-state transition，而不是停留在 broad `cup/glass/kettle is full`

本轮验证补充：

- [x] 新增并通过 `8` 条 `future_use` hierarchy / sufficiency 定向测试：
  - `generic store -> immediate reuse`
  - `finished with object -> immediate reuse`
  - `generic store -> exact final placement`
  - `generic measurement-meta -> exact measurement role`
  - `drying interpretation -> measurement-base placement`
  - `later scale use -> same-object open/uncap`
  - `generic full container -> tap state switch`
  - `generic fill-limit target mismatch -> need more evidence`
- [x] 定向回归：`pytest -q tests/test_graph_agent.py -k 'future_use_causal_hierarchy or future_use_sufficiency'`
  - 结果：`28 passed, 424 deselected`
- [x] 专项总回归：`pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`120 passed, 332 deselected`

下一步缺口：

- [x] `temporary relocation` vs `store away`
- [x] `immediate reuse` vs `finished with object`
- [x] `exact final placement` vs generic `store/put away`
- [x] `generic measurement-meta` vs `exact measurement role`
- [x] `drying interpretation` vs `measurement-base placement`
- [x] `later downstream scale/tap use` vs `same-object open/uncap`
- [x] `tap state switch / phase-switch` vs `generic full container`
- [ ] `open-close` 与 `measure / hand-free / same-object role-use` 的更多真实 replay 残差还没压实
- [ ] `tap phase-switch` 与 `generic boil-speed / fill-limit` 在真实 replay 上的广覆盖收益还没统计

完成标准：

- “need more evidence” 的判定不再太粗

#### P4.3 `precondition-sensitive` 候选在 toolbox 内同步约束

- [ ] `dry hands`
- [ ] `wipe surface`
- [ ] `clean`
- [ ] `avoid burn / spill / mess`

- [ ] 这些候选在没有前置触发时要自动降权

进展补充：

- [x] `dry hands / wipe surface / clean` 的证据要求已与 `planner + verifier` 方向对齐
- [x] `avoid burn / spill / mess` 已在 `future_use` sufficiency 中加入更明确的 direct-purpose 检查

完成标准：

- graph 层和 toolbox 层对 precondition 的理解一致

### 16.7.6 Phase 5：真实 replay、小样本验证与报告

目标：

确认新增逻辑真正打到了真实残差，而不是只让单测更好看。

#### P5.1 构建 why replay 集

- [ ] 每个高优先级 bucket 准备 `3-5` 个真实样例
- [ ] 第一轮至少覆盖：
  - `cleaning_precondition_vs_future_use`
  - `dry_hands_vs_wipe_surface`
  - `temporary_set_aside_vs_finished_store_dry`
  - `inspection_vs_serve_pour_empty`
  - `hidden_access_vs_exact_reveal_use`

完成标准：

- why replay 不是纯随机样本，而是分 bucket 的真实集合

#### P5.2 分层小样本验证

- [ ] 第一轮跑 `20-30` 个 why 样例
- [ ] 第二轮扩到 `40-60` 个 why 样例
- [ ] 保留每题：
  - 题目
  - bucket
  - 调用工具序列
  - 关键帧
  - 最终答案
  - 是否翻正

完成标准：

- 能回看每一道题是怎么被 agent 处理的

#### P5.3 bucket 级统计

- [ ] 统计每个 bucket：
  - 样本数
  - 正确数
  - 翻正数
  - 新回归数
  - 平均工具调用数
  - 平均 token 消耗
  - 主失败原因

完成标准：

- 可以清楚说出“提升发生在哪些语义簇”

#### P5.4 研究汇报级结论

- [ ] 形成专项摘要
- [ ] 明确：
  - 现在哪些 why bucket 已经稳定
  - 哪些 bucket 仍然需要视频后续证据
  - 哪些 bucket 还需要更强工具能力

完成标准：

- 可以直接作为后续论文实验设计与中期汇报材料

---

## 16.8 当前推荐执行顺序

后续 goal 模式建议严格按下面顺序推进：

1. `Phase 0 / P0.2`：把 residual 审计结果正式整理成 bucket ledger
2. `Phase 2 / P2.1`：why 冲突 -> 补证据路由表统一成单入口
3. `Phase 2 / P2.4`：hidden-access / pairwise 的后果帧门控做真实 replay 验证
4. `Phase 5 / P5.1-P5.3`：真实 replay 与 bucket 统计
5. `Phase 4 / P4.3`：把 precondition-sensitive 候选的 toolbox 约束继续做广覆盖补齐
6. `Phase 3 / P3.5`：继续确认 finalize 来源优先级在真实 replay 上不回退

原因：

- `graph_agent` 纯语义层已经有明显基础；
- `pairwise`、cleaning-specific `future_use`、set-aside / inspection / reuse-vs-placement`、`measurement / weigh / open-close / same-object role-use`、`tap phase-switch` 这几组高频残差都已经推进过一轮；
- 现在更缺的不是再补一批局部 hierarchy，而是把 residual 真正整理成 bucket，并把 repeated-failure textual fallback 的 finish 条件再收紧一层；
- 只有先把“哪些 residual 还在反复出现”“何时 textual fallback 仍然不能 finish”“为什么该补 segment 而不是继续 query_time”统一起来，后面 replay 才有研究意义。

---

## 16.9 当前默认起点

后续进入 `goal` 模式，默认从下面这个叶子项开始：

- [ ] `Phase 0 / P0.2`：把 why residual 审计结果整理成 bucket ledger，并反推下一批通用规则

本轮默认目标：

- 先把已经发现的 why residual 组织成长期维护台账：
  - 记录 bucket 名、代表样例、误选模式、正确模式、失效层级、当前优先级
  - 特别纳入 repeated-failure textual fallback、hidden-access、future-use measurement/open-close、precondition-sensitive cleaning 这几类
- 这轮目标不是继续沿着单测补分，而是把后续代码推进改成真正的 residual-driven 闭环

本轮硬要求：

- [ ] 至少形成一版 why residual bucket ledger
- [ ] 至少为每个高优先级 bucket 绑定一个真实 residual 样例
- [ ] 明确每个 bucket 当前优先归因到 `graph / planner / verifier / toolbox` 哪一层
- [ ] 产出下一轮代码修改的优先顺序

---

## 16.10 Goal 模式单轮执行模板

后续每一轮严格按下面模板执行：

1. 打开本清单，选择最高优先级未完成叶子项。
2. 先明确本轮语义冲突对。
3. 判断本轮失效层级：
   - `graph`
   - `planner`
   - `verifier`
   - `toolbox`
4. 只改当前叶子项直接相关代码。
5. 至少新增 `2` 条测试。
6. 先跑定向测试。
7. 再跑 `pytest -q tests/test_graph_agent.py -k 'action_intent'`。
8. 提交代码。
9. 回填本清单。
10. 再进入下一轮。

---

## 16.11 单轮验收命令建议

### 16.11.1 通用专项回归

- `pytest -q tests/test_graph_agent.py -k 'action_intent'`

### 16.11.2 Planner 相关

- `pytest -q tests/test_graph_agent.py -k 'planner_action_intent'`

### 16.11.3 Verifier / Finalizer 相关

- `pytest -q tests/test_graph_agent.py -k 'finalizer or verifier or structured_family_answer'`

### 16.11.4 针对单个新簇

- 先用测试名关键字精确跑，再跑专项总回归

---

## 16.12 Goal 模式停止条件

满足下面任一条件时，可以停止当前轮次并切换总结：

- [ ] 当前叶子项已完成，代码、测试、回归、清单都已回填
- [ ] 连续三轮都在同一 bucket 上没有实质改进，需要重新分桶
- [ ] 已出现明显跨阶段耦合，必须先补 planner/verifier 再回到该 bucket
- [ ] why replay 已足够显示阶段性收益，可以暂停进入下一个研究模块

---

## 16.14 2026-06-07 本轮进展：timeline review 结构化接管补帧策略

本轮完成点：

- [x] 新增 `planner._action_intent_timeline_review_bias_profile(...)`
- [x] 不再主要依赖 `timeline_review_text` 的弱字符串匹配
- [x] `timeline review` 的结构化字段开始直接控制：
  - `resolver_hint`
  - `transition probe mode`
  - `should_try_transition_probe`
  - `extra followup` 的窗口长度与采样密度
  - `frame staging` 的保留配额与优先级

本轮新增的关键能力：

- [x] 当复核结果明确说“隐藏目标已露出但取走尚未可见”时：
  - planner 会更偏向 `revealed_target_retrieval`
  - 保留更多 `followup_transition / peaks / regular followup`
- [x] 当复核结果明确说“腾出的槽位才是主要歧义”时：
  - planner 会更偏向 `revealed_slot_placement`
  - 减少无关晚期 `ext` 帧，优先保留转场与紧邻 followup
- [x] 当复核结果明确说“最终放置位置仍不清楚”时：
  - planner 会优先进入 `final_placement_result`
  - 更偏向保留 `ext2/ext3` 等晚期 followup，而不是继续吃大量 `segment`
- [x] 当只是泛化的 “later use unclear” 时：
  - 仍以向后补帧为主
  - 不会一律误升级成 transition probe

本轮新增测试：

- [x] `test_planner_action_intent_timeline_review_final_location_ambiguity_prefers_transition_probe`
- [x] `test_planner_action_intent_timeline_review_slot_ambiguity_prefers_transition_and_regular_followup_frames`
- [x] `test_planner_action_intent_timeline_review_final_location_bias_prefers_late_followup_frames`

本轮回归结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'timeline_review and action_intent'`
  - `19 passed`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `254 passed, 344 deselected`

这轮的研究意义：

- [x] 让 why agent 不再“看到一点局部证据就急着定答案”
- [x] 开始把“证据不足时主动继续找什么证据”做成结构性 planner 能力
- [x] 提升方向针对整类 `hidden-access / final-placement / late-use ambiguity` 题，而不是单题 hack

---

## 16.15 2026-06-07 本轮进展：timeline review 结构化接管 finalize gate

本轮完成点：

- [x] `graph_agent` 新增 timeline review 读取与 bias profile 逻辑
- [x] finish gate 不再只看 resolver 给出的 `best_index/confidence`
- [x] 当 review 明确说“仍多解”时，finalizer 会优先继续 withholding，而不是直接定答

本轮新增的关键能力：

- [x] 当 review 仍指出 `final location remains unclear` 时：
  - 即使 future-use resolver 给出 `put back`，也不会直接 finalize
  - 除非后续真的补到新帧，并出现明确的 return/put-back chain
- [x] 当 review 仍指出 `freed slot` 的放置歧义时：
  - 即使 pairwise resolver 偏向 `put into freed slot`
  - 如果没有真正的 slot-destination chain，也不会 finalize
- [x] 当 review 仍指出 `hidden retrieval / hand-free next action / revealed fixture` 这类近因歧义时：
  - finalizer 会要求更明确的 direct-effect / downstream-action 证据
- [x] 旧 review 不会无限阻塞：
  - 只有 review 之后真的发生了新采样，再得到新的成功 resolution，旧 blocker 才会被清掉
  - 仅仅又跑一次 resolver，但没有新证据，不算解除歧义

本轮新增测试：

- [x] `test_graph_agent_action_intent_finalizer_withholds_put_back_claim_when_timeline_review_still_has_final_location_gap`
- [x] `test_graph_agent_action_intent_finalizer_withholds_slot_placement_claim_when_timeline_review_keeps_slot_ambiguity`
- [x] `test_graph_agent_action_intent_finalizer_allows_resolution_after_new_followup_even_if_old_timeline_review_asked_for_more_evidence`

本轮回归结果：

- [x] 定向 finalizer 回归
  - `4 passed`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `257 passed, 344 deselected`

这轮的研究意义：

- [x] why agent 不只是“会补帧”，而且“会拒绝在证据仍冲突时装作已经看懂”
- [x] planner 的主动补证据与 graph/finalizer 的保守定答首次形成闭环
- [x] 提升目标仍然是整类 `final-placement / slot-placement / reveal-access / hand-free` close-call 题

---

## 16.16 2026-06-07 本轮进展：timeline review 结构化接管 unresolved rerank

本轮 residual bucket：

- bucket 名：`timeline-review-aware unresolved rerank`
- 失效层级：`graph`
- 典型问题：
  - planner 已经补了关键帧并产出 timeline review
  - review 明确写着“final location / freed slot / hidden retrieval 仍有歧义”
  - 但 unresolved rerank 仍可能只根据 `candidate_evidence` 的弱支持把某个候选翻成临时最优

本轮完成点：

- [x] `unresolved rerank` 已接入 timeline review bias
- [x] 不再只靠 `candidate_evidence score/support/contradiction` 机械重排
- [x] 当 review 指向的歧义尚未闭合时，rerank 会继续 withholding，而不是提前翻正弱候选

本轮新增的关键能力：

- [x] 当 review 仍指出 `final location remains unclear` 时：
  - unresolved rerank 不会把“靠近 fridge / 离开原位”这种弱 `put back` 证据翻成答案
- [x] 当 review 仍指出 `freed slot` 放置歧义时：
  - unresolved rerank 不会把“区域更空了”这种弱 slot 证据翻成 `put into freed slot`
- [x] 当 review 已明确指出 `hidden target` 的 reveal/retrieval 链，并且候选证据与之对齐时：
  - unresolved rerank 仍允许把 `hidden retrieval` 候选翻正
  - 也就是说不是一刀切变保守，而是按 review 指向的歧义类型收紧

本轮新增测试：

- [x] `test_graph_agent_action_intent_unresolved_rerank_withholds_put_back_when_timeline_review_still_has_final_location_gap`
- [x] `test_graph_agent_action_intent_unresolved_rerank_withholds_slot_placement_when_timeline_review_keeps_slot_ambiguity`
- [x] `test_graph_agent_action_intent_unresolved_rerank_keeps_hidden_retrieval_when_timeline_review_and_candidate_chain_align`

本轮回归结果：

- [x] 定向 unresolved rerank 回归
  - `3 passed`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `260 passed, 344 deselected`

这轮的研究意义：

- [x] why agent 的“主动补帧”结果终于能传导到“候选重排”这一层
- [x] 现在 why 闭环已经贯通为：
  - `planner` 主动找更合适的关键帧
  - `unresolved rerank` 不再无视 review 歧义乱翻盘
  - `finalizer` 在证据仍冲突时拒绝抢答
- [x] 这比单纯继续堆提示词更接近真正的 agent 化结构收益

---

## 16.17 2026-06-07 本轮进展：repeated failure 恢复链改为 evidence-first

本轮 residual bucket：

- bucket 名：`repeated-failure current-scope recovery`
- 失效层级：`planner`
- 典型问题：
  - `infer_action_intent` 连续失败后，planner 过早退回 `rank_choices_from_state`
  - 文本 fallback 低置信时，又继续退回通用 `query_time`
  - 这样会把同视频旧记忆混进来，且在“多个选项都还说得通”时仍可能直接收口

本轮完成点：

- [x] 新增 `planner._build_action_intent_evidence_first_recovery_decision(...)`
- [x] repeated failure 后，恢复优先级改为“当前题证据优先”，不再默认退回 `query_time`
- [x] 文本 fallback 产出后，如果当前题仍可继续补证据，会先恢复当前题原始帧/补帧/短时序复核，而不是直接 finish

本轮新增的关键能力：

- [x] 当 why 题连续失败且当前题还没有 current-scope 原始帧时：
  - 优先 `retrieve_cached_artifacts` 回收当前题 artifact
  - 若没有可复用 artifact，则先 `sample_sparse_frames` 恢复当前动作片段
  - 不再默认掉回通用 `query_time`
- [x] 当 why 题已经有当前题 segment 但还缺动作后结果帧时：
  - 优先继续补 followup / ext followup
  - 不允许在“动作后发生了什么”仍未知时直接文本收口
- [x] 当 why 题已经有 segment + followup 关键帧时：
  - 优先触发短时序 `timeline review`
  - 让模型先总结动作后立刻结果、下一步操作和仍存歧义，再回到专用因果判断
- [x] 当 `rank_choices_from_state` 只是 repeated-failure 的低置信兜底时：
  - `_recover_if_low_confidence(...)` 现在会先尝试 current-scope recovery
  - 而不是保持文本排序结果不动

本轮新增测试：

- [x] `test_planner_action_intent_repeated_vision_failures_prefers_current_scope_resampling_before_textual_rank`
- [x] `test_planner_action_intent_repeated_vision_failures_with_followup_frames_prefers_timeline_review_before_future_use_resolution`
- [x] `test_planner_action_intent_repeated_vision_failures_with_followup_frames_prefers_timeline_review_before_pairwise_resolution`
- [x] `test_planner_action_intent_repeated_vision_failures_with_current_followup_frames_forces_timeline_review`
- [x] `test_planner_action_intent_textual_rank_is_overridden_to_recover_current_scope_after_repeated_vision_failures`

本轮回归结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'repeated_vision_failures or textual_rank_is_overridden or timeline_review'`
  - `32 passed`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `261 passed, 344 deselected`

这轮的研究意义：

- [x] why agent 不再把“上游模型暂时失败”错误地等价成“可以直接文字猜答案”
- [x] 关键帧选择真正开始变成主动策略：
  - 先恢复当前题原始帧
  - 再补动作后证据
  - 再做短时序复核
  - 最后才允许因果裁决或文本兜底
- [x] 这一步直接对齐用户关心的三个点：
  - 关键帧要更主动
  - 视频理解不能只看局部瞬间
  - 多个选项都可能成立时，必须继续找证据，而不是过早定答

---

## 16.18 2026-06-07 本轮进展：无显式时间点的 why 题先做动作锚点定位

本轮 residual bucket：

- bucket 名：`time-free why localization`
- 失效层级：`planner`
- 典型问题：
  - 某些 why 题没有给显式 `<TIME ...>` 区间
  - planner 之前只能先走通用 `query_time`
  - 这会把整段视频的大量弱时间记忆混进来，导致关键帧定位过泛，后续更容易文本猜答案

本轮完成点：

- [x] 新增 why 题的 `action-object anchor` 初始定位路径
- [x] 当题目没有显式时间点时，不再默认先 `query_time`
- [x] 先根据题目里的动作对象做结构化定位，再围绕最像动作发生点的短窗口抽关键帧

本轮新增的关键能力：

- [x] 新增 `planner._action_intent_question_action_text(...)`
  - 从 why 题里解析动作短语，如 `<take bottle>` / `<pick up pot>`
- [x] 新增 `planner._action_intent_question_object_hint(...)`
  - 抽取动作对象，如 `bottle` / `pot` / `tea towel`
- [x] 新增 `planner._action_intent_step_decision(...)`
  - why 题在没有显式时间点时：
    - 第一步先 `query_event`
    - 检索 `frame / observation / timeline_event / object_track / segment / activity`
    - 用动作对象把候选时刻范围缩小
- [x] 新增 `planner._action_intent_localization_window_from_nodes(...)`
  - 对 `query_event` 返回的候选节点做优先级排序
  - 优先使用更贴近动作发生点的 `frame / observation / timeline_event`
  - 次选 `object_track`
  - 最后才退到长时段 `segment / activity`
  - 抽帧窗口也不再盲目整段拉长，而是围绕最像动作发生点的短窗口做局部抽帧

本轮新增测试：

- [x] `test_planner_action_intent_without_explicit_times_prefers_object_anchor_query_event`
- [x] `test_planner_action_intent_after_object_anchor_query_event_extracts_localized_segment_frames`

本轮回归结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'object_anchor_query_event or localized_segment_frames or action_intent'`
  - `263 passed, 344 deselected`

这轮的研究意义：

- [x] why agent 现在不再依赖“题目必须给时间点”才能进入专用动作理解路径
- [x] 关键帧选择更像主动 agent：
  - 先定位动作对象
  - 再选最像动作发生点的短窗口
  - 再抽关键帧做因果判断
- [x] 这一步直接补上了“动作定位过泛 -> 视频理解过弱 -> 过早定答”的上游问题

---

## 16.13 一句话结论

当前 why 逻辑专项已经从“零散规则期”进入“证据闭环期”。

接下来不该继续泛泛说“优化逻辑推理”，而应该严格按下面主线推进：

- `Phase 2`：planner 按冲突类型补证据
- `Phase 3`：verifier 阻止证据不足时过早 finish
- `Phase 4`：toolbox 与 graph 语义收敛
- `Phase 5`：真实 replay 和分 bucket 统计

这条线才是真正把当前 why 模块做成研究级 agent 组件的关键路径。
