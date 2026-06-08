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
- 2026-06-07 当前结果：`302 passed, 344 deselected`
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
- 本轮提交：Bucket C 的 `revealed slot / sink slot` 追证逻辑继续收紧。此前 `needed_observation` 若同时点名“被放入的对象”和“freed slot / sink slot”这类槽位，planner 容易因为目标不唯一而放弃精确追证，或者被 relation-revisit 抢先带回动作物体，停在“空间被腾出来了”的宽泛解释。本轮改为：
  - 对 `placed/put into the freed slot`、`put into the sink slot` 这类文本，优先把“将被放入的对象”当成真正的 downstream target；
  - 同类场景下先跳过 relation-revisit，避免 planner 围着动作物体和槽位关系打转，而是直接去追下游对象在更晚时刻是否真的进入该位置；
  - `choice target` 抽取改为更稳的边界匹配和长词优先，避免 `saucepan -> pan` 这类子串误命中把目标追歪。
- 本轮提交：新增并通过 2 条 Bucket C 定向测试，分别保护：
  - `move mug -> blue cup into freed slot` 时会优先追 `cup` 的后续轨迹，而不是卡在 `cup + slot` 的双目标歧义里；
  - `move colander -> saucepan into sink slot` 时会优先追 `saucepan` 的后续轨迹，而不是回到 `colander / sink` 关系或被 `pan` 子串误匹配带偏。
- 本轮提交：专项回归已更新到 `330 passed, 344 deselected`
- 本轮提交：Bucket C 的 exact-workspace overclaim 继续收口。`make space on the shelf/worktop` 这类 choice 现在会被识别为具体空间目标，不再落回 generic make-space；同时 unresolved rerank 新增 `exact_workspace_without_exact_use` 语义缺口，用于识别“只有宽泛空间变化、没有确切下游物体/用途/目的地”的反例。
- 本轮提交：新增并收口 2 条 Bucket C 反例测试，分别覆盖：
  - `make space on the shelf` 但只有 shelf layout 变化时，应回退到更合理的 generic access，而不是保留 exact-workspace 过拟合答案；
  - `make space on the worktop` 但只有 area becomes more open / no exact next target 时，应继续 withheld 而不是提前收口。
- 本轮提交：专项回归已更新到 `332 passed, 344 deselected`
- 本轮提交：Bucket B 的 `open/uncap vs weigh/use later` verifier-blocked recovery 继续收紧。此前只要 `reason / needed_observation` 提到 `same-object cap action / lid action`，mixed-horizon later-target hint 就会被整段拦掉，导致 planner 明明已经知道更晚目标是 `scale`，却仍可能围着动作物体打转。本轮改为：
  - 只有当 `best` 本身已经是 later-use 候选时，才保留原有 same-object blocker，避免误伤原本就该追动作物体本身的题；
  - 当 `best` 是近窗 `open/uncap`，而 later-use 只是竞争项时，不再因为 `cap/lid` 措辞直接放弃 later-target 追证；
  - mixed-horizon later-target 落到 fixture 时，优先选更晚的 fixture 轨迹，而不是停在最早出现的同名节点。
- 本轮提交：新增并通过 1 条 Bucket B 定向测试，保护 `open/uncap` vs `weigh later` 在 verifier-blocked close-call 下，即使 `reason` 明确提到 `same-object cap action`，planner 也会继续追 `scale` 的更晚轨迹。
- 本轮提交：专项回归已更新到 `333 passed, 344 deselected`
- 本轮提交：Bucket D 的 `dry hand` vs `wipe both hands` 继续收紧。此前 transport-vs-use 的 unresolved rerank 已能把“手部接触”从 `clean counter` 拉回到 hand-use，但对于“明确是双手擦拭”与“更弱的单手 dry hand”之间还缺少对称区分。本轮改为：
  - 新增 `explicit both-hands wiping` 显式规则，当证据明确是 `brought to both hands / both hands are wiped` 时，`wipe both hands` 会获得更强加分；
  - 同时收紧 `explicit single-hand drying` 与 `generic hand-wiping -> single-hand drying` override：如果证据已经明确说“不是单手，而是双手”，就不再把答案翻回 `dry hand`。
- 本轮提交：新增并通过 1 条 Bucket D 定向测试，覆盖 `pick up paper towel` 后证据明确指向“双手擦拭”时，系统不再停在 `dry hand`，而会提升到 `wipe both hands`。
- 本轮提交：专项回归已更新到 `334 passed, 344 deselected`
- 本轮提交：Bucket D 的 `surface_wipe_preparation` 继续收紧。此前只要 towel/cloth 被放到 worktop 且伴随“不是收纳、以后可能还会用”的 non-storage 信号，就可能把 `wipe the worktop` 提前翻出来；这会把“只是暂放在台面上”误读成准备擦台面。本轮改为：
  - `non-storage` 不再单独构成 `surface wipe preparation`；
  - 除了“不是收纳”，还必须出现 `crumbs / spill / surface target / ready for wiping / next visible cleaning target` 这类更具体的台面目标或 staged-wipe 信号，才允许把答案翻到 `wipe the worktop`。
- 本轮提交：新增并通过 1 条 Bucket D 反例测试，覆盖 `dish cloth` 只是被暂放到 worktop within reach、但没有 crumbs/spill/visible target 时，不再提前推成 `wipe the worktop`，而是继续 withheld。
- 本轮提交：专项回归已更新到 `335 passed, 344 deselected`
- 本轮提交：Bucket D 的 finalizer 边界再补 1 条显式反例，验证“即使已经出现 `crumbs / worktop target`，但还没有真正 `wiping stroke / sweep` 时，`wipe the worktop` 仍必须继续 withheld”。这次没有新增实现代码，说明当前 weak-surface-wiping finish gate 已能覆盖该边界，缺的是测试保护。
- 本轮提交：新增并通过 1 条 Bucket D 定向测试，覆盖 `dish cloth` 放到 crumbs 旁边、`needed_observation` 仍是“是否真的擦过台面”时，finalizer 不能提前收口到 `wipe the worktop`。
- 本轮提交：专项回归已更新到 `336 passed, 344 deselected`
- 本轮提交：Bucket E 的 `adjust/read measurements` broad measurement-meta finish gate 开始收紧。此前 `toolbox/planner` 已能在上游把 `adjust the measurements`、`read the measurements` 这类泛化 measurement-meta 往“真正 weigh ingredient”追证，但 graph-agent finalizer 侧还缺少显式保护，导致这类宽泛答案在缺少 `reading/tare/update/direct weighing use` 时仍可能被提前收口。本轮改为：
  - 新增 `weak measurement meta` finish gate：若 top 候选只是 `adjust/read/record measurements` 这类 broad measurement-meta，而 `reason + decisive_observation` 里没有 `reading / tare / zero / display change / entered update` 等直接信号，则直接 withheld；
  - 同时补上 why 题 structured best-index fallback 的通用保护：只要已经写入 `action_intent_resolution_withheld_for_*` marker，就不允许后续 fallback 再把答案从旧的 `best_index` memory 里捞回来。
- 本轮提交：新增并通过 1 条 Bucket E 定向测试，覆盖 `pick up scale` 后只有“scale remains near ingredient area”的宽泛 measurement 语义、但没有 `reading/tare` 明确信号时，finalizer 不能直接收口到 `adjust the measurements.`。
- 本轮提交：再补 1 条 Bucket E 回归保护测试，显式锁住 `tap kitchen scale -> zero out` 的 `missing_state_change_prereq` 与 structured fallback 的交互边界。现在即使 working memory 里还残留旧的 `action_intent_best_index=zero out`，只要 finalizer 已经写入 `action_intent_resolution_withheld_for_missing_state_change_prereq=1`，后续 fallback 也不会把该答案捞回。
- 本轮提交：Bucket E 的 `turn on` vs `zero out` 再补 1 个真实实现缺口。此前 finalizer 已能识别 `tap kitchen scale` 在缺少“动作前已开机 / 容器已在秤上”前提时不能直接收口，但 planner 的恢复路径仍可能继续回到 pairwise / followup，只盯动作后读数变化，漏掉真正决定性的信息。本轮改为：
  - `action_intent_needs_precondition_context` 不再只覆盖 `clean_dry / safety_avoid`，同时把 `tap scale` 这类 `open_close + measure_weigh` 的 state-change 冲突也视为 precondition-dependent；
  - `need_disambiguating_evidence` / `need_alternative_evidence_path` 恢复入口现在会读取最新的 `resolve_action_intent_pairwise / future_use` payload，而不是只看旧的 `infer_action_intent`；
  - 若最近已经写入 `action_intent_resolution_withheld_for_missing_state_change_prereq=1`，且 `needed_observation` 明示 `before the tap / already lit / already on / container already on the scale`，planner 会优先补 `precontext`，不再继续在动作后结果帧里打转。
- 本轮提交：新增并通过 1 条 Bucket E 定向测试，覆盖 `resolve_action_intent_pairwise` 已明确“需要看动作前显示状态/容器前提”时，planner 会优先采样 `fine_grained_why_recognition_precontext`，而不是继续回到 `pairwise`。
- 本轮提交：Bucket F 开始收口 `check boiling / check contents` 的 weak inspection overclaim。此前 `resolve_action_intent_future_use` 若直接给出 `check the boiling water / check the contents / check the consistency`，只要锅具里“似乎还有液体/内容物”就可能被 deterministic finalizer 直接收口，即使没有 `brief inspection / stays near hob / no tilt / no pouring / no serving destination` 这类 inspection chain。本轮新增 `weak cooking inspection` finish gate：
  - 对 `pot/pan/saucepan/frying pan/bowl` 这类 cooking vessel，若 top 候选属于 `check boiling / check contents / check consistency / see if done`；
  - 但 `reason + decisive_observation + needed_observation` 里没有形成 `brief cooking inspection over disposal` 的强链条；
  - 则直接写入 `action_intent_resolution_withheld_for_weak_cooking_inspection_evidence=1` 并继续 withheld。
- 本轮提交：新增并通过 1 条 Bucket F 定向测试，覆盖“只有 `pot is lifted while it still seems to contain hot water`、没有 `brief inspection / no tilt / stays near hob` 链条”时，finalizer 不能直接收口到 `to check the boiling water.`。
- 本轮提交：Bucket F 的 `check label vs put back` 再补 1 个 planner 恢复缺口。此前 finalizer 已能写入 `action_intent_resolution_withheld_for_mixed_horizon_later_target=1 target=fridge kind=fixture`，但 `finalize_withheld_mixed_horizon_later_target_revisit` 仍可能停在过早的 fridge 节点，而不是追更晚的 return window。本轮改为：
  - 对 finalizer 写出的 fixture later-target，planner 也像 verifier-blocked 那条 mixed-horizon 路径一样，优先选择满足 `min_start_time` 的更晚 fixture 轨迹；
  - 因而 `check label` 被拦下后，会优先跳到更晚的 fridge return 时段，而不是停在近窗里“瓶子靠近冰箱”的早期节点。
- 本轮提交：新增并通过 1 条 Bucket F 定向测试，覆盖 `check label` 被 finalizer 拦下并写入 `target=fridge kind=fixture` 后，planner 会优先跳到更晚的 fridge 轨迹窗口，而不是停在过早的 near-fridge 节点。
- 本轮提交：Bucket F 的 `weak cooking inspection` 再补 1 个 mixed-horizon later-target 缺口。此前 `check the boiling water / check the contents` 虽然已经会被 finalizer 正确 withheld，但 close-call 若同时存在 `empty / pour / serve later` 这类竞争项，系统还不会像 `label vs put back`、`open vs empty` 那样把真实 later target 写回 working memory，导致 planner 少了一条明确的后续追证锚点。本轮改为：
  - `weak cooking inspection` 命中时，也会尝试复用 mixed-horizon later-target marker；
  - 对 `check boiling/check contents` vs `empty/pour/serve later` 这类 close-call，只要竞争项已经暴露出明确目标语义（例如 `sink`），即使通用 mixed-horizon 分类还不够完整，也会把该 later target 写回 working memory；
  - 因而 `pick up pot`、`pot still seems to contain hot water` 这类题，在“尚未看清是短暂查看还是倒向 sink”时，不再只有 generic withhold，而是会显式留下 `target=sink kind=fixture` 给 planner 后续恢复链使用。
- 本轮提交：新增并通过 1 条 Bucket F 定向测试，覆盖 `check the boiling water` vs `empty the water` 的 weak-inspection close-call；现在 finalizer 会同时写入 `action_intent_resolution_withheld_for_weak_cooking_inspection_evidence=1` 与 `action_intent_resolution_withheld_for_mixed_horizon_later_target=1 target=sink kind=fixture`。
- 本轮提交：Bucket F 的 `check label vs put back` 再补 1 个 unresolved-rerank 恢复缺口。此前 finalizer 与 verifier-blocked 的 mixed-horizon later-target revisit 已经会对 `fridge/scale/sink` 这类 fixture later-target 优先跳到满足 `min_start_time` 的更晚节点，但 unresolved rerank 这条路径仍可能停在近窗 fixture 轨迹，导致 `check label` vs `put back` 这类题在“已经知道要去追 fridge、却仍停在早窗 fridge 节点”的状态下反复打转。本轮改为：
  - `planner._build_action_intent_unresolved_rerank_mixed_horizon_later_target_revisit_decision(...)` 也对 `target_kind == fixture` 对齐启用“优先选更晚节点”逻辑；
  - 因而 mixed-horizon unresolved close-call 一旦已经确定真实 later target 是 `fridge/scale/sink`，就不再停在第一个同名 fixture，而是会继续跳到动作后更有判别力的更晚窗口。
- 本轮提交：新增并通过 1 条 Bucket F 定向测试，覆盖 `take bottle` 时 `check the label` vs `put the bottle back in the fridge` 的 unresolved-rerank close-call；当前 late window 只显示“标签朝外、尚未看清是否回冰箱”时，planner 会优先跳到更晚 fridge 节点，而不是停在近窗 fridge 轨迹。
- 本轮提交：Bucket F 的 `weak cooking inspection` 再补 1 个 needed-evidence 缺口。此前 finalizer 虽然已经能在 `check boiling/check contents` vs `empty/pour/serve later` close-call 下正确 withheld，并写出 `target=sink` 这类 later-target marker，但如果上游 payload 自身没有 `needed_observation`，planner 仍只能依赖泛化 mixed-horizon/long-horizon 恢复链，无法显式知道“下一轮到底该去确认什么”。本轮改为：
  - `graph_agent` 在 `weak cooking inspection` 被 withheld 时，会额外写入更具体的 `action_intent_needed_observation=...`，例如“是否真的朝 sink 倾倒，还是只是在 hob 附近短暂查看”；
  - 该说明会优先结合 mixed-horizon competitor 的 `candidate_evidence` 生成，因此不只是泛化“看后续用途”，而是明确指向 `sink / plate / bowl / hob` 这类判别目标；
  - `planner._action_intent_needed_observation_text(...)` 也补上了 `working_memory` fallback：即使最近一次 resolution payload 本身没有 `needed_observation`，只要 graph-agent 已经把该 marker 写进状态，planner 也会继续用它做 target / relation revisit。
- 本轮提交：新增并通过 2 条 Bucket F 定向测试，分别保护：
  - `pick up pot` 时 `check the boiling water` vs `empty the water` 的 finalizer close-call，会同时写入 `target=sink` later-target marker 与明确的 `needed_observation`；
  - planner 在只有 `working_memory` 里的 inspection `needed_observation` marker 时，也会继续利用该信息进入更强的 relation/target revisit 路径，而不是退回泛化音频峰值或普通 followup。
- 本轮提交：同时按新行为更新 2 条已有测试预期：当 `needed_observation` 已经足够明确点名 `sink` 关系时，planner 现在会优先走更强的 `target/relation revisit`，而不再停在旧的 `pot` long-horizon 或 generic detect-audio-peaks 路线。
- 本轮提交：Bucket F 的 `serve/plate` later-outcome 泛化也补齐了保护。此前 `sink` 分支已经能稳定写出 inspection 的 `needed_observation` 并驱动 planner 去追 `sink`，但 `check contents` vs `serve later` 这类 `plate/bowl` serving 场景还没有同等级别的测试锁定，也仍受旧的 followup 次数门槛影响。本轮改为：
  - `lift frying pan` / `serve the vegetables` 这类 close-call 现在也会被测试保护，确认 finalizer 会同时写入 `target=plate` later-target marker 与“是否真的被带到 plate 上方，还是只是在 hob 附近短暂查看”的明确 `needed_observation`；
  - `planner._action_intent_needed_observation_target_hint(...)` 与 `...relation_hint(...)` 的旧门槛被最小放宽：如果 working memory 里已经存在明确的 `action_intent_needed_observation=...`，就不再强制等待第 2 次 followup 才允许进入目标/关系追证；
  - 因而 `plate-serving` inspection marker 不会再被 generic `detect_audio_peaks` 抢走，而是能直接进入更强的 relation revisit，去验证 frying pan/container 是否真的被带到 `plate / bowl` 上方。
- 本轮提交：新增并通过 2 条 Bucket F 定向测试，分别保护：
  - `lift frying pan` 时 `check the contents of the pan` vs `serve the vegetables` 的 finalizer close-call；
  - planner 在只有 `working_memory` 里的 `plate-serving` inspection marker 时，也会直接进入 relation revisit。
- 本轮提交：Bucket E 的 `zero out with container` 再补 1 个恢复顺序缺口。此前 finalizer 已经能识别“缺少 container precondition / already-on precondition”并写入 `action_intent_resolution_withheld_for_missing_state_change_prereq=1`，verifier-blocked 恢复链也会优先补 `precontext`；但 `pairwise` 路径在 `need_more_evidence` 场景下仍可能先退回 generic extra-followup，再去补真正决定性的 precontext，导致“容器是否已在 tap 前放到秤上”这类关键前提被后置。本轮改为：
  - `planner` 在 `resolve_action_intent_pairwise` 的恢复链里，若已经命中 `missing_state_change_prereq` 且 `needed_observation` 直接点名 `container already on the scale / before the tap` 一类前提，就先调用 `precondition_sampling`；
  - 只有 precontext 仍不足或未命中该前提时，才退回更泛化的 `followup_ext1` 补帧。
- 本轮提交：新增并通过 1 条 Bucket E 定向测试，覆盖 `tap kitchen scale` 的 `zero out with container` pairwise close-call；当前 `needed_observation` 明示“容器是否在 tap 前已在秤上”时，planner 会先走 `fine_grained_why_recognition_precontext`，而不是先补 generic `followup_ext1`。
- 本轮提交：同时按真实稳定行为更新该测试预期：precontext 回补本来就走 `sample_sparse_frames(tag=..._precontext)`，不是 `extract_frames_for_range`。
- 本轮提交：专项回归已更新到 `348 passed, 344 deselected`
- 本轮提交：why 题在首次 `infer_action_intent` 就暴露 `receptacle_outcome` 近窗歧义时，不再机械地先走一轮泛化 `followup`。现在会直接围绕动作尾部触发 `followup_transition`，主动去找“是否真的掉回 sink/pan/bowl/container”的决定性关键帧；同时这条路径会压过误触发的 `precontext`，避免 `flip cloth / shake / tap / tilt` 一类题被无关前置状态采样截走
- 本轮提交：新增并通过 2 条定向测试，分别保护：
  - `receptacle_outcome` 型 why close-call 会在第一次歧义时直接进入 `followup_transition`
  - 普通 `future_use` 型 why 题仍保持原来的初始 `followup`，不会被误改成近窗密采样
- 本轮提交：Bucket D 的 `surface cleanup` finish gate 继续压实。此前 `wipe the worktop` 已经要求 `crumbs/spill/ready-for-wiping` 一类更具体的 staged-wipe 语义，但 `clean up the kitchen counter` 这类同类表述在 unresolved rerank 中仍可能仅凭“接近台面/短暂接触台面”而过早收口。本轮改为：
  - `missing_surface_wiping_evidence` 不再只覆盖字面上的 `wipe ... worktop/counter`，而是扩到所有 `clean/wipe + surface/counter/worktop` 的表面清洁类候选；
  - `weak_surface_contact_cleanup_claim` 也接入 unresolved semantic gap，若证据只有 `touches the counter area / brief press / nearby messy spot`，但没有 `wipe sweep / repeated wiping / clear before-after cleanup result`，则 unresolved rerank 必须继续 withheld。
- 本轮提交：新增并通过 2 条 Bucket D 定向测试，分别覆盖：
  - finalizer 在 `paper towel` 只有短暂靠近/接触台面时，不能直接收口到 `clean up the kitchen counter.`；
  - unresolved rerank 在只有 `surface proximity/contact`、没有 `sweep/contact chain` 时，不能把 `clean up the kitchen counter.` 当成最终答案。
- 本轮提交：专项回归已更新到 `350 passed, 344 deselected`
- 本轮提交：Bucket E 的 `measurement-meta vs exact measurement role` 又补了一层 unresolved gate。此前 finalizer 已能挡住 `adjust/read measurements` 的 broad overclaim，但 unresolved rerank 在一些 close-call 下仍可能把弱 `measurement context` 直接翻成 `measure the cheese / base for weighing` 之类 exact role，即使还没看到 `reading/tare/display-change`，也没看到真正的 `ingredient on scale / used for weighing` 链条。本轮改为：
  - `graph_agent` 新增 measurement-role sufficiency helper，明确区分：
    - broad `adjust/read/record measurements`
    - exact `measure/weigh/use as a base`
  - unresolved semantic gaps 新增：
    - `missing_measurement_meta_evidence`
    - `missing_exact_measurement_role_evidence`
  - exact measurement role 的 unresolved override 现在也要求更具体的 weighing-use 证据，不再仅凭“could later be used to measure”一类宽泛 close-call 提前翻正。
- 本轮提交：新增并通过 1 条 Bucket E 定向反例测试，覆盖 `pick up scale` 时 broad measurement-meta 与 exact measurement role 都仍只是 speculative 的 close-call；现在 unresolved rerank 会继续 withheld，而不是提前翻到 `measure the cheese`。
- 本轮提交：同时保留并复核通过 2 条原有正例：
  - `measure the cheese` 在确有 immediate weighing-use 证据时仍可翻正；
  - `use the bowl as a base to weigh more ingredients` 这类 measurement-base setup 仍能正常收口。
- 本轮提交：专项回归已进一步提升到 `353 passed, 344 deselected`
- 本轮提交：why 题的首次主动关键帧前移继续扩展到两类高频歧义：
  - `tap kitchen scale / press button / push switch` 这类 `state_change / open-close vs measure-use` 题，不再一上来稀疏补 8 秒长窗；现在会先围绕动作尾部后的 2 到 4 秒做 `followup_transition` 密采样，优先确认显示是否开机、归零、变化或出现其它决定性状态改变
  - `pick up tea towel / paper towel / cloth` 这类 `transport-vs-use` 题，当模型已经明确承认“要看动作后是拿去擦手/擦台面，还是只是放下/挪开”时，会先补动作后近窗关键帧，再决定是否需要回补 `precontext`；也就是说，agent 会先验证真实使用链，而不是默认先回头找前置状态
- 本轮提交：新增并通过 3 条定向测试，分别保护：
  - `tap kitchen scale` 无 tool trace 时，首次补证据优先进入 `followup_transition`
  - `tap kitchen scale` 在首次 `infer_action_intent` 仍不确定时，也会直接进入近窗 `followup_transition`
  - `pick up tea towel` 的 `transport-vs-use` close-call 会在第一次歧义时优先补动作后近窗关键帧，而不会先被 `precontext` 截走
- 本轮提交：Bucket F 的最后一条 later-outcome 残差已正式收口。此前系统已经能挡住弱 `check label / check boiling / check contents` 的过早定答，也能在 close-call 下写出 `fridge/sink/plate` later-target marker 去追更晚证据；但如果证据里已经直接出现 `placed back into the fridge`、`tilted to pour into the sink` 这类明确 later outcome，系统仍更偏向继续 withheld，而不是直接翻到真实 later candidate。本轮改为：
  - finalizer 新增 `explicit later outcome over weak inspection` override：若 top 候选只是弱 inspection，且自身没有直接 inspection chain，但竞争项已经给出显式 later outcome，则直接翻到 later candidate；
  - unresolved rerank 同步新增同类 override，但只作用于真正的 inspection mixed-horizon 题，不作用于 `move/transfer` 这类“下游拿取不是当前直接目的”的题，避免误伤已有 Bucket C/B 逻辑；
  - later outcome 证据还额外要求是“已经发生”的结果，不接受 `could/may/not yet visible` 这类推测式表述，因此原有 `check label vs put back`、`check boiling vs empty` close-call 仍保持 withheld 并继续追更晚节点。
- 本轮提交：新增并通过 2 条 Bucket F 定向测试，分别覆盖：
  - `take bottle` 时若证据已经明确写出 `placed back into the fridge`，则 `to put the bottle back in the fridge.` 会压过弱 `to check the label.`；
- 本轮提交：收口一类新的 unresolved-rerank 残差 `immediate micro-outcome overclaim`。此前当 close-call 同时存在“立即微结果”和“更晚下游用途”时，系统已经能把 `open bottle / turn on scale` 之类候选提到最高分，但若证据里缺少真正的即时结果链，仍可能在 rerank 阶段过早收口；本轮改为：
  - unresolved semantic gap 保留 `missing_immediate_micro_outcome_evidence`，用于拦截 `opening could happen next / could later be tipped` 这类纯推测式近窗与远窗候选；
  - 但显式补强 immediate micro-outcome 的正证据识别，允许 `free to uncap/open immediately`、`reaches to the scale and turns it on` 这类已经形成即时动作链的候选继续收口；
  - 同时修复一个真实的字符串残差：`breadcrumbs` 中的子串 `read` 之前会误触发“读标签/读日期”分支，导致 `uncap/open` 的即时证据被提前短路；现在阅读类匹配改为真正的 `read the / read label / read date / 读取标签` 等短语，不再被食材名误伤。
- 本轮提交：新增并通过 2 条定向测试，分别覆盖：
  - `take bottle` 时若只有 `opening could happen next`、`cap opening itself is not visible yet`，则 `to open the bottle.` 必须继续 withheld；
  - 一旦证据明确出现 `cap is visibly loosened/opened while held in hand`，则 `to open the bottle.` 仍可正常收口。
- 本轮提交：同时复核通过 2 条原有正例：
  - `transfer cup of breadcrumbs` 时，`free hand -> uncap/open same object immediately` 仍会压过 later `weigh breadcrumbs`；
  - `move tray` 时，`reveal scale -> immediately turn on the scale` 仍会压过 generic `access the scale behind the tray`。
- 本轮提交：专项回归已更新到 `361 passed, 344 deselected`
- 本轮提交：收口一个 planner 级 fallback 残差：`specialized failure fallback over-finishes stale intent`。此前如果 `resolve_action_intent_pairwise / future_use` 连续失败，planner 会直接回退到“最近一次成功的 infer_action_intent 结果”并 finish；但这条兜底没有检查旧成功结果本身是否已经明确承认“还需要 future evidence / needed observation”，因此在 mixed-horizon / later-target 题上可能把本该继续追更晚证据的旧猜测提前收口。
- 本轮改为：
  - 只有当最近一次成功的 `infer_action_intent` 结果本身已经闭合时，才允许作为 failure fallback 直接 finish；
  - 若旧成功结果仍带有 `need_future_evidence / need_more_evidence / needed_observation`，或 working memory 里仍残留 `pending_resolution / withheld / unresolved_rerank_withheld / action_intent_needed_observation`，则不允许直接 finish，继续走已有的专用恢复链。
- 本轮提交：新增并通过 2 条定向测试，分别覆盖：
  - `take bottle` 在旧成功结果仍明确要求“确认是否 later put back into the fridge”时，后续 pairwise 失败不能直接 finish 到 `to open the bottle.`；
  - `place lid` 在旧成功结果已经形成闭合链条时，后续 pairwise 失败仍可安全复用该结果直接 finish。
- 本轮提交：专项回归已进一步提升到 `363 passed, 344 deselected`
- 本轮提交：继续收口一个相邻的 planner fallback 残差：`resolution need-more-evidence fallback finishes without anchors`。此前即使 `resolve_action_intent_pairwise / future_use` 自己已经明确返回 `need_more_evidence=True`，只要当前没有 `times/input_times` 可供继续恢复，planner 仍可能直接落到“专用裁决已完成，直接结束”这条 finish 分支。
- 本轮改为：
  - 若 `pairwise / future_use` payload 本身仍带 `need_more_evidence / need_future_evidence / needed_observation`，则一律视为“未闭合结果”，不能直接 finish；
  - 即使当前缺少时间锚点、`specialized recovery` 暂时构不出更具体的 followup，也先退回当前题 `segment` 重抽，而不是让未闭合专用裁决直接收口。
- 本轮提交：新增并通过 2 条定向测试，分别覆盖：
  - `resolve_action_intent_pairwise` 在 `need_more_evidence=True` 且没有时间 hints 时，不再直接 finish，而是回到 `fine_grained_why_recognition_segment` 重抽；
  - `resolve_action_intent_future_use` 在同类条件下也不再直接 finish，而是同样回到当前题 `segment` 重抽。
- 本轮提交：专项回归已进一步提升到 `365 passed, 344 deselected`
- 本轮提交：继续收口同一条 planner fallback 主线上的第三个残差：`unresolved infer_action_intent falls back to text rank without anchors`。此前如果 `infer_action_intent` 本身已经明确写出 `need_future_evidence=True`，但当前又没有 `times/input_times`、也暂时构不出更具体的 followup / pairwise / future-use 恢复链，planner 会先退到 `rank_choices_from_state`，下一轮再进一步掉到泛化 `query_time`。这会把 why 专项中“未闭合的专用动作目的判断”降级成文本聚合收口。
- 本轮改为：
  - 只要 `infer_action_intent` payload 仍带 `need_future_evidence / need_more_evidence / needed_observation`，就视为未闭合结果，不允许退到 `rank_choices_from_state`；
  - 在当前缺少恢复锚点时，也先回到当前题 `segment` 重抽，而不是先降到文本聚合或下一轮泛化 `query_time`。
- 本轮提交：新增并通过 2 条定向测试，分别覆盖：
  - 未闭合 `infer_action_intent` 在没有时间 hints 时，不再退 `rank_choices_from_state`，而是回到 `fine_grained_why_recognition_segment` 重抽；
  - 已闭合 `infer_action_intent` 在无其它恢复路径时，仍保持当前稳定行为，可直接 finish。
- 本轮提交：专项回归已进一步提升到 `367 passed, 344 deselected`
- 本轮提交：补上一个 verifier 侧的放行残差：`textual fallback ignores needed_observation marker`。此前 repeated vision failure 的文本 fallback 只要具备当前题 artifact 和一些 grounding，就可能被 verifier 直接判 sufficient；但它没有把 `action_intent_needed_observation=...` 这类“明确还缺关键后续证据”的 working-memory marker 当作 blocker，因此会把仍未闭合的 why 文本 fallback 提前放行。
- 本轮改为：
  - 在 `ranked_best_index` textual fallback 场景下，只要最近 working memory 里仍残留 `action_intent_needed_observation=...`，verifier 就不能直接 sufficient；
  - 这使 verifier 与最近几轮 planner fallback 收紧保持一致，不会一边要求继续追 later-target / needed-observation，一边又在文本 fallback 上提前放行。
- 本轮提交：新增并通过 1 条定向测试，覆盖：
  - `take bottle` 的 textual fallback 即使已有当前题 artifact 和 grounding，只要仍保留 `action_intent_needed_observation=whether the bottle is later put back into the fridge`，verifier 仍必须保持 blocking。
- 本轮提交：同时保留并复核通过 1 条原有正例：
  - `place bowl` 的 textual fallback 在已有当前题 artifact、grounding 且没有未闭合 marker 时，仍可正常 finish。
- 本轮提交：专项回归已进一步提升到 `368 passed, 344 deselected`
- 本轮提交：补上 `missing_direct_outcome_evidence` 在 repeated textual fallback 下的恢复对称性。此前这类 why close-call 在 verifier-blocked 路径上已经会优先触发 `followup_transition`，但如果随后连续视觉失败并退到 `need_alternative_evidence_path + rank_choices_from_state`，planner 仍可能回到泛化的 `recover_frames/segment`，错过“继续围绕动作尾部查近窗直接结果”的更优恢复链。
- 本轮改为：
  - 只要 `working_memory` 里仍保留 `action_intent_resolution_withheld_for_missing_direct_outcome_evidence=1`；
  - 且最近 action-intent 专用裁决来自 `infer_action_intent / pairwise / future_use`；
  - 那么在 `need_alternative_evidence_path` 的 textual fallback 恢复入口，也会优先复用现有 `followup_transition` 恢复链，而不是先退回通用 `segment/recover_frames`。
- 本轮提交：新增并通过 1 条定向测试，覆盖：
  - `flip orange cloth` 的 close-call 在 repeated failure 之后，即使已经退到 textual fallback，也仍优先围绕动作尾部补 `followup_transition`，而不是退回泛化补帧。
- 本轮提交：专项回归维持 `368 passed, 344 deselected`
- 本轮提交：再补一条 repeated textual fallback 的对称性缺口：`missing_state_change_prereq`。此前 `tap kitchen scale` 这类题在 verifier-blocked 路径上已经会优先回补 `precontext`，但如果随后连续视觉失败并退到 `need_alternative_evidence_path + rank_choices_from_state`，planner 仍可能被短时序 `inspect_visual_evidence` 或其它 generic fallback 抢走，继续盯动作后而不是动作前状态。
- 本轮改为：
  - 只要最近 working memory 里仍保留 `action_intent_resolution_withheld_for_missing_state_change_prereq=1`；
  - 且最近 action-intent 专用裁决仍明确要求回补 precondition；
  - 那么在 textual fallback 恢复入口，也继续优先走 `precontext`，而不是退回 generic raw fallback。
- 本轮提交：新增并通过 1 条定向测试，覆盖：
  - `tap kitchen scale` 的 textual fallback 在 repeated failure 后，仍优先补 `precontext` 去确认“tap 前是否已开机/是否已有容器”，而不是继续盯动作后短窗。
- 本轮提交：专项回归已进一步提升到 `370 passed, 344 deselected`
- 本轮提交：再补第三类 repeated textual fallback 的对称性缺口：`needed_observation already names the real target/relation`。此前如果专用裁决已经把歧义收敛到“是否回 fridge / 是否放到 scale / 是否 carried over plate”这类明确目标或关系，但随后连续视觉失败并退到 `rank_choices_from_state`，planner 仍可能先做 generic `inspect_visual_evidence`，而不是直接进入 target/relation revisit。
- 本轮改为：
  - 在 textual fallback 入口，优先读取最近 action-intent 专用裁决里的 `needed_observation`；
  - 若已经能从中解析出明确的 relation hint 或 target hint，就直接进入已有的 relation/target revisit 恢复链，不再先退回 generic visual review。
- 本轮提交：新增并通过 1 条定向测试，覆盖：
  - `take bottle` 的 textual fallback 在 repeated failure 后，如果 `needed_observation` 已经明确收敛到“是先读标签还是回 fridge”，就直接继续追 `bottle -> fridge` 的更晚证据，而不是先做 generic 短时序复核。
- 本轮提交：专项回归已进一步提升到 `371 passed, 344 deselected`
- 本轮提交：继续补 `repeated textual fallback` 的 finalizer-marker 对称性。此前 `generic access / generic relocation / generic hand-free` 这三类 finalizer withheld marker 在 `verifier-blocked` 与 specialized recovery 路径里已经会直接追真实 downstream target，但一旦 repeated vision failure 后退到 `rank_choices_from_state`，planner 仍可能只回到该目标的早窗节点，甚至退回 generic visual review。
- 本轮改为：
  - textual fallback 入口也先读取最近的 finalizer withheld marker；
  - 对 `generic access / generic relocation / generic hand-free`，直接复用已有 downstream target revisit 路径；
  - 同时把这三类 marker 接入 `prefer latest long-horizon node` 偏置，避免虽然追到了对的下游对象，却仍停在过早的中间节点。
- 本轮提交：新增并通过 1 条定向测试，覆盖：
  - `move bottle` 的 textual fallback 在 repeated failure 后，如果 finalizer 已经写出 `generic relocation/storage -> target=jar`，则会直接跳到 `jar` 的更晚节点，而不是停在早窗 reveal 片段。
- 本轮提交：专项回归已进一步提升到 `372 passed, 344 deselected`
- 本轮提交：继续补 `repeated textual fallback` 的 late-anchor 对称性。此前 `nonexclusive_concrete_late_anchor`、`timeline_review_bias_gap`、`workspace_or_final_placement_claim` 这类“已经知道当前只是晚锚点中间态”的 withheld marker，在 infer/future-use/specialized recovery 路径里已经会继续追更晚节点；但 repeated vision failure 后一旦退到 `rank_choices_from_state`，planner 仍可能直接退回 generic resample，或者虽然进入 long-horizon revisit 却停在过早节点。
- 本轮改为：
  - textual fallback 入口也接入 `weak_late_anchor / nonexclusive_concrete_late_anchor` 的专门 revisit；
  - 同时把 `nonexclusive_concrete_late_anchor / timeline_review_bias_gap / workspace_or_final_placement_claim` 一并接入 long-horizon 的 `prefer latest` 偏置；
  - 因而这类“只是放在附近/标签刚露出来/还没形成排他结果”的 close-call，在 textual fallback 下也会继续追更晚节点，而不会退回 generic resample 或停在早窗中间态。
- 本轮提交：新增并通过 1 条定向测试，覆盖：
  - `move bowl` 的 textual fallback 在 repeated failure 后，如果已经写入 `nonexclusive_concrete_late_anchor` marker，则会直接跳到 `bowl` 的更晚节点，而不是回到 generic 稀疏补帧或停在早窗邻近片段。
- 本轮提交：专项回归已进一步提升到 `373 passed, 344 deselected`
- 本轮提交：继续补 `repeated textual fallback` 下的 `measurement / phone-record exact target` 对称性。此前 `generic measurement-meta -> exact weighing target` 与 `generic phone-measure -> exact ingredient record target` 这两类 specialized revisit，在 `verifier-blocked` 路径里已经会直接追 `scale` 或具体 ingredient target；但 repeated vision failure 后一旦退到 `rank_choices_from_state`，planner 仍会先回到 generic `inspect_visual_evidence`，没有复用这些更强的 specialized target revisit。
- 本轮改为：
  - textual fallback 入口也接入 `measurement target revisit` 与 `phone record target revisit`；
  - 同时把 ingredient record target 的文本抽取规则从只接受 `of the X` 扩到也支持常见的 `of X / for X` 表达，避免像 `record the nutritional value of broccoli` 这类真实措辞漏触发。
- 本轮提交：新增并通过 2 条定向测试，分别覆盖：
  - `pick up scale` 的 textual fallback 在 repeated failure 后，会优先追 `scale` / weighing target，而不是先退回 generic visual review；
  - `pick up phone` 的 textual fallback 在 repeated failure 后，会优先追 `broccoli` 这类 exact ingredient record target，而不是先退回 generic visual review。
- 本轮提交：专项回归已进一步提升到 `375 passed, 344 deselected`
- 本轮提交：继续补 `repeated textual fallback` 下的 `mixed_horizon later-target` 对称性。此前 `check label vs put back`、`check/open vs later return/use` 这类 mixed-horizon close-call，在 `verifier-blocked / future_use / pairwise` 路径里已经会直接追更晚的 fixture/object target；但 repeated vision failure 后一旦退到 `rank_choices_from_state`，planner 仍会先去做 generic `inspect_visual_evidence`，没有复用这条更强的 later-target revisit。
- 本轮改为：
  - textual fallback 入口也接入 `finalizer mixed_horizon later-target revisit` 与 `verifier-blocked mixed_horizon later-target revisit`；
  - 同时保持已有优先级不回退：`missing_state_change_prereq` 仍先于 mixed-horizon later-target，避免 `tap scale` 这类题被错误拉去追 later target。
- 本轮提交：新增并通过 1 条定向测试，覆盖：
  - `take bottle` 的 textual fallback 在 repeated failure 后，如果 close-call 已经收敛到 `check label` vs `put back in the fridge`，则会直接追 `fridge` 的更晚节点，而不是先退回 generic visual review。
- 本轮提交：专项回归已进一步提升到 `376 passed, 344 deselected`
  - `pick up pot` 时若证据已经明确写出 `brought to the sink and tilted to pour`，则 `to empty the water.` 会压过弱 `to check the boiling water.`。
- 本轮提交：同时回归通过 3 条关键保护：
  - `check label vs put back` 的 later-target marker 仍会在“尚未看清是否回冰箱”时继续 withheld；
  - `check boiling vs empty` 的 weak-inspection close-call 仍会在“还没看清是否真的倒向 sink”时继续 withheld；
  - 中文 `transfer` 场景下“后续拿起海绵”仍不会被误翻成当前动作的直接目的。
- 本轮提交：专项回归已更新到 `355 passed, 344 deselected`
- 本轮提交：Bucket C 剩余的 `make space vs take hidden X` 泛化缺口也已正式收口。此前 `generic access -> hidden retrieval` 已有 override，但如果 top 候选仍停在 `to make space on the shelf` 这类 broad room-making，而竞争项已经明确写出“hidden spice jar behind it becomes reachable and is taken right afterwards”，系统仍可能不翻正。本轮改为：
  - `generic hidden access -> exact revealed target` 的 override 现在同时覆盖 `generic make space + reveal` 这类 best 候选；
  - 只要 exact candidate 已经形成明确的 `hidden item / item behind / becomes reachable and is taken right afterwards` 链条，就允许把 broad make-space 翻成 hidden retrieval；
  - 若 reveal 虽存在，但 hidden target 仍未被取出，则继续保留更合理的 `generic access` fallback，不会误翻到 exact hidden retrieval。
- 本轮提交：新增并通过 2 条 Bucket C 定向测试，分别覆盖：
  - `move bottle` 时 `to make space on the shelf.` 会被明确的 `take the hidden spice jar behind it` 压过；
  - 如果只是 reveal 了 behind area、但 hidden spice jar 仍只是 speculative，则不会误翻到 exact hidden retrieval，而会回退到 `to access what's behind the bottle.`。
- 本轮提交：专项回归已进一步提升到 `357 passed, 344 deselected`
- 本轮提交：把 `action_intent_resolution_withheld_for_missing_direct_outcome_evidence=1` 接到了 planner 的 forced `followup_transition` 恢复链。此前 `graph_agent finalizer` 已经能识别一类“近窗直接结果还没看清”的弱 `relocation / residue / cloth flip` close-call，但该 marker 还没有驱动专门恢复动作；现在 verifier-blocked recovery 看到这个 marker 时，会优先围绕动作尾部做近窗密采样，而不是退回更泛化的 close-call 恢复。
- 本轮提交：新增并通过 2 条定向测试，分别保护：
  - `missing_direct_outcome_evidence` marker 会强制触发 `followup_transition`；
  - 如果 transition probe 帧已经存在，则不会重复触发同一条近窗密采样路径。
- 本轮提交：专项回归已进一步提升到 `359 passed, 344 deselected`
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
- 本轮提交：why 专项回归已更新到 `302 passed, 344 deselected`
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

## 16.19 2026-06-07 本轮进展：why 题加入长时域对象后续检索

本轮 residual bucket：

- bucket 名：`long-horizon future-use retrieval`
- 失效层级：`planner`
- 典型问题：
  - why 题已经补了近窗 followup
  - 但 `later use / final location` 仍然不清楚
  - 之前系统大多只能继续在近窗里补帧，或者回退到更泛的 `query_time`
  - 这会导致“多个选项都仍说得通”时，agent 没有真正主动去找更远的决定性证据

本轮完成点：

- [x] 新增长时域对象后续检索入口 `query_object`
- [x] 当 why 题的歧义核心是 `later use / final location` 时，不再只在近窗空转
- [x] 能根据对象在更后时刻的再次出现位置，主动补一段长时域关键帧

本轮新增的关键能力：

- [x] 新增 `planner._action_intent_prefers_long_horizon_object_retrieval(...)`
  - 当 timeline review 或 needed profile 明确表明：
    - `next_use_unclear`
    - `final_location_unclear`
    - `prefer_future_use_outcome`
    - `prefer_final_placement`
  - 就允许 why 题进入长时域对象后续检索路径
- [x] 新增 `planner._build_action_intent_long_horizon_object_query_decision(...)`
  - 先按题目对象执行 `query_object`
  - 不再默认退回泛化 `query_time`
- [x] 新增 `planner._action_intent_long_horizon_window_from_nodes(...)`
  - 从 `query_object` 返回节点中挑选“动作之后再次出现”的候选
  - 优先 `object_track / frame / observation / timeline_event`
  - 只有这些不够时才退到更粗的 `segment / activity`
- [x] 新增 `planner._build_action_intent_long_horizon_sampling_decision(...)`
  - 围绕更晚的对象再次出现位置补一小段长时域关键帧
  - 让 agent 真正去看“之后到底是放回了、再次使用了，还是只是暂时移开了”

本轮新增测试：

- [x] `test_planner_action_intent_open_question_with_late_horizon_ambiguity_prefers_query_object`
- [x] `test_planner_action_intent_query_object_result_triggers_late_followup_sampling`

本轮回归结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'late_horizon_ambiguity_prefers_query_object or query_object_result_triggers_late_followup_sampling or action_intent'`
  - `265 passed, 344 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `265 passed, 344 deselected`

这轮的研究意义：

- [x] why agent 不再把“近窗看不清”错误地等价成“继续在近窗补几帧就够了”
- [x] 关键帧策略进一步 agent 化：
  - 先看近窗
  - 如果 later-use / final-location 仍不清楚
  - 就按对象检索其后续再次出现
  - 再围绕那个更晚的位置补关键帧
- [x] 这直接提升了用户关心的核心点：
  - 证据不足时继续主动找证据
  - 多个选项都 plausible 时不抢答
  - 视频理解不只停留在局部一瞬间

---

## 16.20 2026-06-07 本轮进展：长时域对象检索加入目标过滤与空间探针

本轮 residual bucket：

- bucket 名：`late-object disambiguation`
- 失效层级：`planner`
- 典型问题：
  - `query_object` 能把对象后续再次出现找出来
  - 但这些 later nodes 里可能混有同名干扰、粗粒度 segment 命中或只是摘要里提到该对象
  - 如果直接围绕这些 noisy later nodes 抽帧，仍可能把 agent 拉回错误时刻或错误对象链

本轮完成点：

- [x] 给长时域对象检索加入目标对象过滤
- [x] 给 later node 路径加入空间探针 `query_spatial_context`
- [x] 避免 `query_spatial_context` 后立刻拿近窗旧帧重判，而是继续围绕晚时刻位置抽帧

本轮新增的关键能力：

- [x] 新增 `planner._action_intent_long_horizon_target_tokens(...)`
  - 从题目对象里提取 token，用于后续目标节点过滤
- [x] 新增 `planner._action_intent_long_horizon_node_match_tier(...)`
  - 优先保留：
    - `object_name / label` 里直接匹配目标对象的节点
    - 次选只在 summary 中匹配的节点
  - 不再把所有 keyword hit 一视同仁
- [x] 新增 `planner._action_intent_select_long_horizon_node(...)`
  - 在满足“动作之后再次出现”的前提下
  - 优先 exact object match
  - 再按 `object_track / frame / observation / timeline_event / segment` 排序
- [x] 新增 `planner._build_action_intent_long_horizon_spatial_probe_decision(...)`
  - 在更晚候选时刻先补空间关系
  - 再决定是否继续抽这一段关键帧
- [x] `query_spatial_context` 进入 why 长时域路径后：
  - 不再直接拿近窗旧帧去 `infer_action_intent`
  - 而是围绕这个晚时刻的对象位置继续 `sample_sparse_frames`

本轮新增测试：

- [x] `test_planner_action_intent_query_object_result_prefers_long_horizon_spatial_probe`
- [x] `test_planner_action_intent_query_object_prefers_exact_late_object_track_before_noisy_segment`
- [x] `test_planner_action_intent_long_horizon_spatial_probe_then_samples_late_anchor`

本轮回归结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'long_horizon_spatial_probe or exact_late_object_track or action_intent'`
  - `267 passed, 344 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `267 passed, 344 deselected`

这轮的研究意义：

- [x] why agent 现在不仅会“往后找对象”，还会判断“这个后续对象命中是不是靠谱”
- [x] 关键帧策略继续向真正 agent 化推进：
  - 近窗不够
  - 去找对象后续再次出现
  - 先补晚时刻空间关系
  - 再抽对应关键帧
- [x] 这一步进一步减少了：
  - 同名对象干扰
  - 粗粒度 segment 命中把抽帧拉偏
  - query_spatial_context 后被旧近窗帧直接重判的问题

---

## 16.21 2026-06-07 本轮进展：长时域节点选择加入“最终放置偏晚 / 后续用途偏早”策略

本轮 residual bucket：

- bucket 名：`late-node temporal preference`
- 失效层级：`planner`
- 典型问题：
  - 即使已经进入长时域对象后续检索
  - 系统原先仍默认优先选最早的 later node
  - 这对两类题不对称：
    - `final location` 类问题，真正决定性的证据通常更靠后
    - `next use` 类问题，真正决定性的证据通常更靠前

本轮完成点：

- [x] 给长时域 later node 选择加入按歧义类型区分的时间偏置
- [x] `final location` 类冲突优先更晚候选
- [x] `next use` 类冲突仍优先最早候选

本轮新增的关键能力：

- [x] 新增 `planner._action_intent_long_horizon_prefers_latest_candidate(...)`
  - 当 timeline review 明确指出 `final_location_unclear`
  - 或 needed profile 指向 `prefer_final_placement`
  - 长时域 later node 选择不再默认最早，而改为偏更晚节点
- [x] `planner._action_intent_select_long_horizon_node(...)`
  - 现在不仅考虑对象匹配 tier 和节点类型优先级
  - 还会根据当前歧义是“最终放置”还是“后续用途”来决定偏早还是偏晚

本轮新增测试：

- [x] `test_planner_action_intent_long_horizon_final_location_bias_prefers_latest_exact_target_node`
- [x] `test_planner_action_intent_long_horizon_next_use_bias_prefers_earliest_exact_target_node`

本轮回归结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'final_location_bias_prefers_latest_exact_target_node or next_use_bias_prefers_earliest_exact_target_node or action_intent'`
  - `269 passed, 344 deselected`

这轮的研究意义：

- [x] why agent 现在不仅知道“往后找哪个对象”，还开始知道“该偏早看还是偏晚看”
- [x] 关键帧策略进一步从静态规则变成与歧义类型绑定的主动搜索策略
- [x] 这能进一步降低：
  - 最终放置题过早看前一个中间态
  - 后续用途题看得太晚、错过真正 first-use signal

---

## 16.22 2026-06-07 本轮进展：长时域 why 证据不足时继续沿缓存后续节点向后追

本轮 residual bucket：

- bucket 名：`cached later-node continuation`
- 失效层级：`planner`
- 典型问题：
  - why 题已经做过 `query_object`
  - 也已经围绕某个 later node 看过一轮空间上下文和关键帧
  - 但当前证据仍不足、多个选项仍然 plausible
  - 旧逻辑这时容易退回泛化 `followup_ext` / `transition probe` / 重新推理
  - 结果是已经知道“还有更晚的目标对象节点”，却没有继续沿这条长时域链往后追

本轮完成点：

- [x] 新增 `planner._latest_action_intent_long_horizon_nodes(...)`
  - 复用最近一次 `query_object` 的结果
  - 不需要重新跑一次同样的全图对象检索
- [x] 新增 `planner._build_action_intent_cached_long_horizon_revisit_decision(...)`
  - 当最新 long-horizon 窗口仍证据不足时
  - 直接基于缓存 nodes 继续选择 `latest_followup_end` 之后的下一个 later node
  - 再次触发 `query_spatial_context`
- [x] timeline review 现在会优先尝试这条 cached continuation
  - 不再默认先退回泛化补帧
- [x] evidence-first recovery 也接入同一条 cached continuation
  - 避免为什么题在长时域链已经建立后又回到原地空转

本轮新增的关键能力：

- [x] agent 现在具备“沿同一目标对象后续轨迹逐段向后追”的能力
- [x] 如果当前 later node 只说明“对象还在处理中 / 还没看到最终去向”
  - planner 会自动看更晚的目标对象节点
  - 而不是过早收口
- [x] 关键帧搜索从“一次性找一个晚节点”升级为“带状态的长时域续追”

本轮新增测试：

- [x] `test_planner_action_intent_timeline_review_revisits_next_long_horizon_node_before_generic_followup`
- [x] `test_planner_action_intent_evidence_first_recovery_reuses_cached_long_horizon_nodes`

预期收益：

- [x] 降低这类错误：
  - 只看见对象被拿出/移开，但没继续看它后来被放到哪里
  - 只看见对象被拿起，但没继续看它后来是否被真正用于称量/倒空/检查
- [x] 强化 why agent 的核心 agent 性：
  - 证据不足时主动继续找下一段关键证据
  - 而不是把“看过一小段晚帧”误当成“已经看过后续”

---

## 16.23 2026-06-07 本轮进展：用空间上下文区分 long-horizon 中间态与高判别力锚点

本轮 residual bucket：

- bucket 名：`intermediate-vs-decisive late anchor`
- 失效层级：`planner`
- 典型问题：
  - why 题在 long-horizon 阶段已经定位到某个 later node
  - `query_spatial_context` 也拿到了结果
  - 但旧逻辑会把所有 later-node spatial context 一视同仁
  - 实际上两类情况不应该一样处理：
    - 目标物还在 `counter/workspace` 一类活跃区，中间态概率高
    - 目标物已经到 `scale/sink/storage/appliance` 一类更有判别力的位置

本轮完成点：

- [x] 新增 `planner._action_intent_spatial_target_mask_fixture(...)`
  - 从 `query_spatial_context.object_masks` 中提取目标对象对应的 `fixture`
- [x] 新增 `planner._action_intent_fixture_bucket(...)`
  - 把 fixture 粗分成 `storage / scale / sink / appliance / workspace / other / unknown`
- [x] 新增 `planner._action_intent_long_horizon_spatial_context_looks_intermediate(...)`
  - 对 `final_location_unclear / next_use_unclear` 类 why 题
  - 如果当前目标对象只落在 `workspace/unknown/other`，或甚至没有目标 mask fixture
  - 则判成“中间态证据”
- [x] `query_spatial_context -> long-horizon` 分支已接入该判别
  - 中间态：优先直接继续追更晚节点
  - 高判别力锚点：保留当前节点，继续围绕它补帧

本轮新增测试：

- [x] `test_planner_action_intent_long_horizon_intermediate_workspace_spatial_context_revisits_later_node`
- [x] `test_planner_action_intent_long_horizon_decisive_scale_spatial_context_keeps_current_anchor`

预期收益：

- [x] 减少这类误判：
  - 看见对象只是暂时出现在台面/活跃区，就误认为已经到最终位置
  - 看见对象还在过渡途中，就误认为已经能解释真实后续用途
- [x] 强化 agent 的主动证据链：
  - 中间态不定答，继续找更晚节点
  - 判别力足够的锚点才投入当前窗口补帧和裁决

---

## 16.24 2026-06-07 本轮进展：识别“靠近 decisive fixture 但仍是经过态”的 long-horizon 锚点

本轮 residual bucket：

- bucket 名：`transit-near-decisive-fixture`
- 失效层级：`planner`
- 典型问题：
  - 目标对象已经靠近 `sink/appliance` 一类看起来很有判别力的 fixture
  - 但这一下可能只是路过或搬运过程中的短暂停留
  - 如果直接把它当成真实用途/最终放置，仍然会过早定答

本轮完成点：

- [x] 新增 `planner._action_intent_long_horizon_anchor_node_at_time(...)`
  - 在当前 `query_spatial_context` 锚点上回找对应的 long-horizon object node
- [x] 新增 `planner._action_intent_has_later_long_horizon_node_after(...)`
  - 判断当前锚点后面是否还存在更晚目标节点
- [x] 新增 `planner._action_intent_long_horizon_spatial_context_looks_transit_near_decisive_fixture(...)`
  - 重点拦截 `sink/appliance`
  - 若当前锚点对应轨迹很短、后面还有更晚节点、且没有更强音频线索
  - 则判成“经过态 near decisive fixture”
- [x] `query_spatial_context -> long-horizon` 分支已接入 transit 判别
  - 经过态：继续追更晚节点
  - 有更强线索：保留当前锚点继续补帧

本轮新增测试：

- [x] `test_planner_action_intent_long_horizon_brief_sink_anchor_with_later_node_revisits_again`
- [x] `test_planner_action_intent_long_horizon_sink_anchor_with_water_audio_keeps_current_anchor`

预期收益：

- [x] 降低这类错误：
  - 只是短暂经过 sink / appliance，就被误当成“已经去倒水/已经去使用装置”
- [x] 强化 agent 的保守性：
  - 只有当当前 decisive fixture 锚点真的足够强，才围绕它补帧
  - 否则继续向后追，更符合“证据不足不定答”的目标

---

## 16.25 2026-06-07 本轮进展：storage 锚点缺少闭环线索时不把“靠近冰箱/柜子”当作放回完成

本轮 residual bucket：

- bucket 名：`non-exclusive storage anchor`
- 失效层级：`planner`
- 典型问题：
  - 目标对象重新出现在 `fridge/cabinet/shelf` 附近
  - 但旧逻辑容易把“靠近 storage”直接当成“已经放回”
  - 实际上如果没有真正的 storage closure 线索，这仍可能只是短暂停留或下一步还会继续操作

本轮完成点：

- [x] 新增 `planner._action_intent_spatial_has_storage_closure_cue(...)`
  - 检测 `door shut / close / click / drawer / fridge door` 等更强 storage 闭环线索
- [x] 新增 `planner._action_intent_long_horizon_spatial_context_looks_nonexclusive_storage_anchor(...)`
  - 当目标对象只是在 storage 附近短暂停留
  - 且后面还有更晚节点
  - 同时没有 closure cue
  - 就判成“非排他性 storage 锚点”
- [x] `query_spatial_context -> long-horizon` 分支已接入该逻辑
  - 非排他性 storage anchor：继续追更晚节点
  - 有 storage closure cue：保留当前锚点补帧

本轮新增测试：

- [x] `test_planner_action_intent_long_horizon_brief_storage_anchor_without_closure_cue_revisits_again`
- [x] `test_planner_action_intent_long_horizon_storage_anchor_with_door_shut_keeps_current_anchor`

预期收益：

- [x] 减少这类错误：
  - 只因为“瓶子又靠近冰箱”就误判成“已经放回冰箱”
- [x] 进一步贴合用户要求：
  - 多个选项都 plausible 时继续搜证据
  - 不把弱空间邻近关系当成排他性结论

---

## 16.26 2026-06-07 本轮进展：晚锚点弱支持文本不再直接推进强裁决

本轮 residual bucket：

- bucket 名：`weak late-anchor textual support`
- 失效层级：`planner`
- 典型问题：
  - agent 已经补到 long-horizon 晚锚点
  - 但当前 `infer_action_intent / resolve_action_intent_future_use` 输出的文本仍只是：
    - `remains in hand`
    - `near the fridge opening`
    - `stays near the shelf`
    - `not decisively grounded`
  - 这种文本本质上还是弱邻近/手持证据
  - 旧逻辑却可能继续进入更强裁决甚至 finish

本轮完成点：

- [x] 新增 `planner._latest_action_intent_target_spatial_anchor_time(...)`
  - 读取最近一次针对目标对象的 `query_spatial_context` 锚点时间
- [x] 新增 `planner._action_intent_result_looks_weak_late_anchor_support(...)`
  - 识别“晚锚点弱支持”文本
  - 仅在没有 direct post-action evidence 时生效
- [x] 新增 `planner._build_action_intent_weak_late_anchor_revisit_decision(...)`
  - 结合最近 spatial anchor 和 followup_end
  - 直接继续沿更晚目标节点向后追
- [x] 已接入：
  - `infer_action_intent` 的 long-horizon future-use 分支
  - `resolve_action_intent_future_use` 的 finish 前分支

本轮新增测试：

- [x] `test_planner_action_intent_weak_late_anchor_infer_result_revisits_later_node_before_future_use_resolution`
- [x] `test_planner_action_intent_weak_late_anchor_future_use_resolution_revisits_later_node_before_finish`

预期收益：

- [x] 避免这类过早收口：
  - 晚时域里只看到“还拿在手里、靠近某处”，就被当成真实用途/最终放回
- [x] 进一步落实用户目标：
  - 多选项都 plausible 时，planner 主动继续找更晚关键帧
  - 不让弱文本支持直接推进强裁决链

---

## 16.27 2026-06-07 本轮进展：看起来具体但仍不排他的晚锚点描述也不允许直接收口

本轮 residual bucket：

- bucket 名：`non-exclusive concrete late-anchor support`
- 失效层级：`planner`
- 典型问题：
  - 晚时域证据不再只是“仍在手里/靠近某处”这种弱文本
  - 而是升级成了更具体的描述，例如：
    - `the bottle is turned so the front side becomes visible`
    - `the bowl is set beside the scale area`
    - `the object is left nearby within reach`
  - 这些描述看起来更具体，但本质上仍只是：
    - 标签面露出来了，不等于真的在读标签
    - 物体被放在某处旁边了，不等于已经称重/放回/实际使用
  - 旧逻辑容易被这种“具体但不排他”的文本骗过，直接推进 specialized resolution 甚至 finish

本轮完成点：

- [x] 新增 `planner._action_intent_result_looks_nonexclusive_concrete_late_anchor_support(...)`
  - 仅在 `long-horizon why` 路径下生效
  - 先确认当前仍存在真实竞争候选，而不是已经单边压死
  - 重点拦截两类：
    - `label/front side visible` 但没有真正 `read/inspect` 链
    - `set beside / placed nearby / within reach / near scale area` 但没有真正 `weighed / returned / stored / used next` 链
- [x] 新增 `planner._build_action_intent_nonexclusive_concrete_late_anchor_revisit_decision(...)`
  - 与弱晚锚点回溯逻辑一致
  - 直接复用缓存的 long-horizon object nodes
  - 继续沿更晚目标节点向后追，而不是停在当前貌似具体的中间态
- [x] 已接入：
  - `infer_action_intent` 的 followup 分支
  - `resolve_action_intent_future_use` 的 finish 前分支

本轮新增测试：

- [x] `test_planner_action_intent_nonexclusive_concrete_late_anchor_infer_result_revisits_later_node`
- [x] `test_planner_action_intent_nonexclusive_concrete_late_anchor_future_use_resolution_revisits_later_node`

预期收益：

- [x] 降低这类过早定答：
  - 仅因为“标签朝外了”就直接判成 `check label`
  - 仅因为“碗在秤旁边了”就直接判成 `for weighing`
- [x] 更贴近用户要求：
  - planner 不能被“看起来像答案”的半成品晚锚点骗过去
  - 只要还没有形成候选间排他证据，就继续主动找更晚关键帧

---

## 16.28 2026-06-07 本轮进展：把“非排他具体晚锚点”拒收逻辑补到 finalizer 闭环

本轮 residual bucket：

- bucket 名：`finalizer non-exclusive concrete late-anchor leakage`
- 失效层级：`graph_agent finalizer + planner recovery`
- 典型问题：
  - planner 前面已经更保守，但如果 specialized resolution 仍吐出：
    - `the bowl is placed beside the scale area`
    - `the object is positioned adjacent to the weighing station`
    - `the label/front side becomes visible`
  - finalizer 过去不一定能把这类“具体但仍不排他”的文本拒收
  - 会导致链路后段重新接受半成品证据，出现“前面搜证很谨慎，最后还是直接定答”的泄漏

本轮完成点：

- [x] 新增 `graph_agent._action_intent_resolution_should_withhold_nonexclusive_concrete_late_anchor_claim(...)`
  - 在 deterministic finalizer 层显式拒收两类文本：
    - `label/front side visible` 但没有真实 `read/inspect` 链
    - `placed nearby / beside scale / adjacent to weighing station` 但没有真实 `weighed / returned / stored / used next` 链
- [x] finalizer 会写入新 marker：
  - `action_intent_resolution_withheld_for_nonexclusive_concrete_late_anchor=1`
- [x] planner 里的 `nonexclusive concrete late-anchor` 检测已接入该 marker
  - 即使当前文本本身没有完全命中旧 pattern
  - 只要 finalizer 已经拒收过，就继续沿缓存的更晚 object node 向后追

本轮新增测试：

- [x] `test_graph_agent_action_intent_finalizer_withholds_scale_nearby_claim_without_weighing_chain`
- [x] `test_planner_action_intent_nonexclusive_concrete_late_anchor_withheld_marker_revisits_later_node`

预期收益：

- [x] 关闭一条真实泄漏链：
  - 前面 planner 拒绝过早定答
  - 后面 finalizer 也不再重新接受同类半成品证据
- [x] 对用户目标更一致：
  - 多个选项都仍然 plausible 时
  - agent 会跨 planner / finalizer 两层继续主动找更晚关键帧
  - 不会因为“证据句子写得更具体了”就错误地以为已经排他

---

## 16.29 2026-06-07 本轮进展：verifier/finalizer 拦下后的恢复动作优先回到更晚 object node

本轮 residual bucket：

- bucket 名：`blocked-finish generic followup leakage`
- 失效层级：`planner recovery after verifier/finalizer blocked finish`
- 典型问题：
  - finalizer 或 verifier 已经明确表明：
    - 后续用途仍不排他
    - 最终位置仍不明确
    - 当前只是 workspace / staged-nearby 一类半成品状态
  - 但恢复动作过去仍可能走成泛化 `sample_sparse_frames / extra_followup`
  - 这会让 agent 继续在局部近窗里打转，而不是直接去查更晚 object node

本轮完成点：

- [x] 新增 `planner._action_intent_recent_later_outcome_finalize_withheld_marker(...)`
  - 识别最近被 finalizer 拒收的 later-outcome 类 marker：
    - `timeline_review_bias_gap`
    - `workspace_or_final_placement_claim`
    - `nonexclusive_concrete_late_anchor`
- [x] 新增 `planner._build_action_intent_finalize_withheld_long_horizon_revisit_decision(...)`
  - 当 why 题本来就偏 `long-horizon object retrieval`
  - 且最近 blocked finish 的根因属于 later-outcome 未排他
  - 恢复动作直接切换到缓存的更晚 object node，而不是先做泛化 followup
- [x] 已接入 `planner._recover_action_intent_after_verifier_blocked_finish(...)`
  - `future_use_close_call`
  - `pairwise_close_call`
  - 以及专用裁决通用 close-call 恢复尾部

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_finish_with_timeline_gap_prefers_later_object_revisit`

预期收益：

- [x] 把“证据不足不定答”的逻辑继续往后推进一层
- [x] 被 verifier/finalizer 拦下后，不再只做泛化补帧
- [x] 对 why 题中最关键的 later-use / final-location 分歧，agent 会更主动地去找真正更晚的决定性关键帧

---

## 16.30 2026-06-07 本轮进展：unresolved rerank 的 later-outcome gap 直接驱动更晚 object-node 检索

本轮 residual bucket：

- bucket 名：`unresolved rerank generic followup leakage`
- 失效层级：`planner after unresolved_rerank_withheld`
- 典型问题：
  - `graph_agent` 的 unresolved rerank 已经明确写出：
    - `timeline_review_final_location_gap`
    - `timeline_review_revealed_slot_gap`
    - `missing_later_outcome_evidence`
  - 这些 reason 本质上都在说：
    - 要去更晚时刻看真正的后续用途/最终落点
  - 但 planner 过去并不会读取这些 reason
  - 仍可能退回到泛化 `extra_followup / sample_sparse_frames`

本轮完成点：

- [x] 新增 `planner._action_intent_recent_unresolved_rerank_withheld_reason(...)`
  - 从 working memory 里解析最近一次 unresolved rerank 的 withheld reason
- [x] 新增 `planner._action_intent_unresolved_rerank_reason_prefers_later_outcome_revisit(...)`
  - 将以下 reason 归类为“应直接追更晚 object node”：
    - `timeline_review_final_location_gap`
    - `timeline_review_next_use_gap`
    - `timeline_review_revealed_slot_gap`
    - `missing_later_outcome_evidence`
- [x] 新增 `planner._build_action_intent_unresolved_rerank_long_horizon_revisit_decision(...)`
  - 直接复用缓存的 long-horizon object nodes
  - 从当前 spatial anchor / latest followup_end 之后继续追
- [x] 已接入：
  - `resolve_action_intent_future_use` 的 unresolved 路径
  - `resolve_action_intent_pairwise` 的 unresolved 路径

本轮新增测试：

- [x] `test_planner_action_intent_unresolved_rerank_timeline_gap_prefers_later_object_revisit`
- [x] `test_planner_action_intent_unresolved_rerank_slot_gap_prefers_later_object_revisit`

预期收益：

- [x] 不再浪费 `unresolved rerank` 里已经很明确的结构信号
- [x] 当系统已经知道“当前就是缺 later-use / final-location 证据”时
  - planner 会更主动地去找真正更晚的关键帧
  - 而不是继续停留在局部近窗里打转

---

## 16.31 2026-06-07 本轮进展：revealed-slot / revealed-target 题改成追踪下游目标物体

本轮 residual bucket：

- bucket 名：`wrong-object long-horizon revisit`
- 失效层级：`planner target selection`
- 典型问题：
  - 对 `revealed slot` / `revealed hidden target` 类 why 题
  - 即使系统已经知道缺的是更晚下游证据
  - 旧逻辑也仍然默认继续追动作物体本身
    - 例如继续追 `mug`
    - 但真正该看的其实是 `blue cup`
    - 继续追 `bottle`
    - 但真正该看的其实是 `spice jar`
  - 这会导致 agent 虽然“有主动检索”，但检索对象选错了，视频理解仍然低效

本轮完成点：

- [x] 扩展 `planner` 的 long-horizon node 选择链，允许按 `object_hint` 选择目标：
  - `_latest_action_intent_long_horizon_nodes(..., object_hint=...)`
  - `_action_intent_long_horizon_target_tokens(..., object_hint=...)`
  - `_action_intent_long_horizon_node_match_tier(..., object_hint=...)`
  - `_action_intent_select_long_horizon_node(..., object_hint=...)`
- [x] 新增 `planner._action_intent_choice_target_object_candidates(...)`
  - 从候选答案里提取非动作物体的潜在下游目标
- [x] 新增 `planner._action_intent_unresolved_rerank_downstream_object_hint(...)`
  - 当 unresolved rerank reason 属于：
    - `timeline_review_revealed_slot_gap`
    - `timeline_review_revealed_target_gap`
  - 优先从 best choice 中抽取下游目标物体
- [x] 新增 `planner._build_action_intent_unresolved_rerank_downstream_target_revisit_decision(...)`
  - 若当前没有该目标物体的 query 结果：先 `query_object`
  - 若已有其 long-horizon nodes：直接 `query_spatial_context`
- [x] 已接入：
  - `resolve_action_intent_pairwise` 的 unresolved 路径
  - `resolve_action_intent_future_use` 的 unresolved 路径

本轮新增测试：

- [x] `test_planner_action_intent_unresolved_rerank_slot_gap_prefers_later_object_revisit`
  - 现在期望先查询 `cup`，而不是继续追 `mug`
- [x] `test_planner_action_intent_unresolved_rerank_revealed_target_gap_queries_downstream_target_object`
  - 现在期望先查询 `jar`，而不是继续追 `bottle`

预期收益：

- [x] agent 不只是“更主动去找关键帧”
- [x] 还会“更正确地选择检索对象”
- [x] 对 reveal / access / freed-slot 一类长时序 why 题，视频理解链条会更像真正的 agent，而不是只会围着动作物体本身打转

---

## 16.32 2026-06-07 本轮进展：hand-free / fixture-enable 题改成追踪下游装置目标

本轮 residual bucket：

- bucket 名：`hand-free / fixture downstream target leak`
- 失效层级：`planner unresolved rerank recovery`
- 典型问题：
  - `graph_agent` 已经能把这类 why 题识别成
    - `timeline_review_hand_free_or_fixture_gap`
  - 但旧 `planner` 还没有把这个 gap 转成针对性的恢复动作
  - 于是系统明明已经知道“缺的是 tap / scale / fridge / drawer 这类下游装置证据”
  - 却仍然容易：
    - 继续追动作物体本身
    - 或继续泛化补帧
    - 最后在证据不足时过早收口

本轮完成点：

- [x] 新增 `planner._action_intent_choice_fixture_target_candidates(...)`
  - 从候选答案里抽取下游装置/fixture 目标
  - 当前覆盖：`tap / faucet / scale / sink / hob / microwave / oven / fridge / door / drawer / cupboard / rack / dishwasher`
- [x] 新增 `planner._action_intent_unresolved_rerank_downstream_fixture_hint(...)`
  - 当 unresolved rerank reason 属于 `timeline_review_hand_free_or_fixture_gap`
  - 优先从当前 best choice 中恢复真正要追的下游装置目标
- [x] 新增 `planner._build_action_intent_unresolved_rerank_downstream_fixture_revisit_decision(...)`
  - 若当前没有该 fixture 的轨迹：先 `query_object`
  - 若已有其 long-horizon nodes：直接 `query_spatial_context`
- [x] 已接入：
  - `resolve_action_intent_pairwise` 的 unresolved 路径
  - `resolve_action_intent_future_use` 的 unresolved 路径

本轮新增测试：

- [x] `test_planner_action_intent_unresolved_rerank_hand_free_gap_queries_downstream_fixture`
  - `move bowl` / `turn on the tap` 类题
  - 现在期望先查询 `tap`，而不是继续追 `bowl`
- [x] `test_planner_action_intent_unresolved_rerank_fixture_gap_revisits_downstream_fixture_node`
  - `move tray` / `turn on the scale` 类题
  - 如果 `scale` 已有更晚节点，现在期望直接追 `scale` 的 later node

预期收益：

- [x] hand-free / exact fixture enablement 一类题不再只停留在“手空出来了 / 装置露出来了”这类中间态
- [x] agent 会更主动去确认真正的后续动作是否发生
  - 例如有没有真的去拧 `tap`
  - 有没有真的去按 `scale`
- [x] 能直接减少“证据不足但已经开始确定答案”的残留分支

---

## 16.33 2026-06-07 本轮进展：generic hand-free 题不再把“手空出来了”当最终目的

本轮 residual bucket：

- bucket 名：`generic hand-free finalize leak`
- 失效层级：`planner unresolved rerank recovery`
- 典型问题：
  - 即使 `unresolved rerank` 已经明确指出当前 why 题仍缺 hand-free downstream evidence
  - 某些题的最佳候选仍然只是：
    - `so left hand is free`
    - `free one hand`
  - 这类描述本质上只是 enablement 中间态，不是最终动作目的
  - 如果 planner 不继续追：
    - 下一个被拿起的对象
    - 或同一物体的后续清洗/冲洗/使用
  - 就仍然会出现“证据不足却提前收口”

本轮完成点：

- [x] 新增 `planner._action_intent_choice_is_same_object_active_use(...)`
  - 识别候选是否其实在描述“同一物体的后续主动使用”
  - 如 `rinse / wash / clean / wipe / dry / open / shake ...`
- [x] 新增 `planner._action_intent_unresolved_rerank_hand_free_object_hint(...)`
  - 当 unresolved rerank reason 属于 `timeline_review_hand_free_or_fixture_gap`
  - 若最佳候选只是泛化 hand-free
  - 会继续从竞争候选中恢复真正值得追的：
    - 下游对象
    - 或同一物体后续用途
- [x] 新增 `planner._build_action_intent_unresolved_rerank_hand_free_object_revisit_decision(...)`
  - 若目标对象还没轨迹：先 `query_object`
  - 若已有 later nodes：直接 `query_spatial_context`
- [x] 已接入：
  - `resolve_action_intent_pairwise` 的 unresolved 路径
  - `resolve_action_intent_future_use` 的 unresolved 路径

本轮新增测试：

- [x] `test_planner_action_intent_unresolved_rerank_generic_hand_free_queries_real_downstream_object`
  - `so left hand is free` vs `pick up the sponge`
  - 现在期望主动去查 `sponge`
- [x] `test_planner_action_intent_unresolved_rerank_generic_hand_free_revisits_same_object_use`
  - `so left hand is free` vs `rinse the blender cup`
  - 现在期望继续追 `blender cup` 的 later node，而不是把 hand-free 当最终答案

预期收益：

- [x] hand-free 类题的 agent 行为更像真实因果追踪，而不是把 enablement 误判成目的本身
- [x] 对“腾出手 -> 拿工具 / 洗当前物体 / 继续操作”这类题型，能系统性降低过早收口

---

## 16.34 2026-06-07 本轮进展：finalizer 不再让 generic hand-free 直接收口

本轮 residual bucket：

- bucket 名：`generic hand-free finalize leak`
- 失效层级：`graph_agent finalizer + planner recovery`
- 典型问题：
  - 即使 specialized resolution 已经给出了 `best_index`
  - 某些结果仍会把
    - `so left hand is free`
    - `free one hand`
  - 当成最终答案直接 finish
  - 但如果竞争候选里其实已经存在：
    - 更具体的下游对象
    - 或同一物体更直接的后续用途
  - 那么 hand-free 只能算中间 enablement，而不是最终目的

本轮完成点：

- [x] 在 `graph_agent` 新增 `._action_intent_resolution_generic_hand_free_overclaim_marker(...)`
  - 当 finalizer 发现最佳答案只是 generic hand-free
  - 且竞争候选已经指向更具体 downstream object / same-object use
  - 会写入：
    - `action_intent_resolution_withheld_for_generic_hand_free_enablement=1 target=... kind=...`
- [x] 在 `planner` 新增 `._action_intent_recent_generic_hand_free_finalize_withheld_hint(...)`
  - 解析 finalizer 写回的 `target` 与 `kind`
- [x] 在 `planner` 新增 `._build_action_intent_finalize_withheld_generic_hand_free_revisit_decision(...)`
  - 若目标还没轨迹：先 `query_object`
  - 若已有 later node：直接 `query_spatial_context`
- [x] 已接入：
  - `resolve_action_intent_pairwise` 之后的 finish 前恢复
  - `resolve_action_intent_future_use` 之后的 finish 前恢复

本轮新增测试：

- [x] `test_graph_agent_action_intent_finalizer_withholds_generic_hand_free_when_specific_downstream_object_exists`
  - finalizer 现在会拒收 generic hand-free，并写出 `target=knife`
- [x] `test_planner_action_intent_generic_hand_free_withheld_marker_revisits_real_downstream_object`
  - planner 现在会直接追 `knife` 的 later node，而不是直接 finish

预期收益：

- [x] 即使 specialized resolution 暂时给出高置信 generic hand-free
- [x] 系统也不会立刻收口，而会继续追更真实的动作目的证据链
- [x] “关键帧不够、证据不排他却提前定答案”的最终漏口进一步缩小

---

## 16.35 2026-06-07 本轮进展：finalizer 不再让 generic access / make-space 直接收口

本轮 residual bucket：

- bucket 名：`generic access / space finalize leak`
- 失效层级：`graph_agent finalizer + planner recovery`
- 典型问题：
  - specialized resolution 有时会先给出：
    - `to access what's behind ...`
    - `to make space.`
  - 但竞争候选里其实已经有：
    - 更具体的 hidden target
    - 或更明确的 revealed-slot / exact use
  - 这时如果直接 finish
  - 本质上仍是“只停留在泛化 reveal / space 层，没有继续追真正下游目标”

本轮完成点：

- [x] 在 `graph_agent` 新增 `._action_intent_resolution_generic_access_or_space_overclaim_marker(...)`
  - 当 finalizer 发现最佳答案只是 generic access / generic space
  - 且竞争候选已指向更具体的 revealed target / placement / fixture enablement
  - 会写入：
    - `action_intent_resolution_withheld_for_generic_access_or_space_enablement=1 target=... kind=...`
- [x] 在 `planner` 新增 `._action_intent_recent_generic_access_or_space_finalize_withheld_hint(...)`
  - 解析 finalizer 写回的目标
- [x] 在 `planner` 新增 `._build_action_intent_finalize_withheld_generic_access_or_space_revisit_decision(...)`
  - 若目标轨迹未建立：先 `query_object`
  - 若已有 later node：直接 `query_spatial_context`
- [x] 已接入：
  - `resolve_action_intent_pairwise` 之后的 finish 前恢复
  - `resolve_action_intent_future_use` 之后的 finish 前恢复

本轮新增测试：

- [x] `test_graph_agent_action_intent_finalizer_withholds_generic_access_when_specific_revealed_target_exists`
  - finalizer 现在会拒收 generic access，并写出 `target=jar`
- [x] `test_planner_action_intent_generic_access_withheld_marker_revisits_real_revealed_target`
  - planner 现在会直接追 `jar` 的 later node，而不是继续围着 `bottle` 打转或直接 finish

预期收益：

- [x] generic access / make-space 不再轻易当作终点答案
- [x] agent 会更主动去查真正的 revealed downstream target
- [x] “看到露出来了”与“真正为了拿/放/用那个目标”之间的证据链被进一步拉直

---

## 16.36 2026-06-07 本轮进展：finalizer 不再让 generic put-away / temporary relocation 直接收口

本轮 residual bucket：

- bucket 名：`generic relocation / storage finalize leak`
- 失效层级：`graph_agent finalizer + planner recovery`
- 典型问题：
  - specialized resolution 有时会先给出：
    - `to put the ... away.`
    - `to store the ...`
    - 或其它泛化 `temporary relocation / put-back` 解释
  - 但竞争候选里其实已经有：
    - 同一物体的立刻复用
    - 或 reveal 后真正被拿起/使用的下游目标
  - 这时如果直接 finish
  - 本质上仍是“把中间态的 put-away / 挪开动作，当成了真正的最终目的”

本轮完成点：

- [x] 在 `graph_agent` 新增 `._action_intent_resolution_generic_relocation_or_storage_overclaim_marker(...)`
  - 当 finalizer 发现最佳答案只是 generic `put away / store / put back`
  - 且竞争候选已指向：
    - same-object immediate reuse
    - cleaning-tool 的具体使用链
    - 或 reveal 后更具体的 downstream target
  - 会写入：
    - `action_intent_resolution_withheld_for_generic_relocation_or_storage_enablement=1 target=... kind=...`
- [x] 在 `planner` 新增 `._action_intent_recent_generic_relocation_or_storage_finalize_withheld_hint(...)`
  - 解析 finalizer 写回的真实后续目标
- [x] 在 `planner` 新增 `._build_action_intent_finalize_withheld_generic_relocation_or_storage_revisit_decision(...)`
  - 若目标轨迹未建立：先 `query_object`
  - 若已有 later node：直接 `query_spatial_context`
- [x] 已接入：
  - `resolve_action_intent_pairwise` 之后的 finish 前恢复
  - `resolve_action_intent_future_use` 之后的 finish 前恢复

本轮新增测试：

- [x] `test_graph_agent_action_intent_finalizer_withholds_generic_put_away_for_same_object_reuse_target`
  - finalizer 现在会拒收 generic `put away`，并把 `tea towel` 记成要继续追的真实目标
- [x] `test_graph_agent_action_intent_finalizer_withholds_generic_put_away_for_revealed_downstream_target`
  - finalizer 现在会拒收 generic `put away`，并把 `jar` 记成 reveal 后真正要追的下游目标
- [x] `test_planner_action_intent_generic_relocation_withheld_marker_revisits_real_downstream_target`
  - planner 现在会直接追 `jar` 的 later node，而不是继续围着 `bottle` 打转或直接 finish

预期收益：

- [x] generic `put away / temporary relocation` 不再轻易当作终点答案
- [x] agent 会更主动去查“这个物体随后是不是被继续使用”或“真正被让出来的下游目标是谁”
- [x] why 推理里的“证据不足但过早定答”问题又减少了一整类

---

## 16.37 2026-06-07 本轮进展：mixed-horizon 不再只做泛化后追，而是直接追真实 later target

本轮 residual bucket：

- bucket 名：`mixed-horizon later-target recovery leak`
- 失效层级：`graph_agent finalizer + planner recovery`
- 典型问题：
  - `check label / open jar` 这类近窗微结果候选，已经会被 finalizer 拦下
  - 但旧恢复逻辑仍常常只是：
    - 继续围着动作物体泛化补帧
    - 或泛化地沿 object long-horizon 往后追
  - 没有显式转去追：
    - `fridge`
    - `scale`
    - 或 mixed-horizon 竞争里真正决定 later outcome 的目标
  - 这会让关键帧检索仍然偏泛，证据链拉不直

本轮完成点：

- [x] 在 `graph_agent` 新增 `._action_intent_resolution_mixed_horizon_later_target_marker(...)`
  - 当 finalizer 发现当前只是 `check/open` 这类 immediate micro-outcome
  - 且真正竞争的是 `put back / weigh / ...` 这类 later outcome
  - 会优先写入：
    - `action_intent_resolution_withheld_for_mixed_horizon_later_target=1 target=... kind=...`
  - 当前已稳定覆盖：
    - `to put the bottle back in the fridge.` -> `target=fridge`
    - `to use the jar to weigh the ingredients.` -> `target=scale`
- [x] 在 `planner` 新增 `._action_intent_recent_mixed_horizon_later_target_withheld_hint(...)`
  - 解析 finalizer 写回的 later target
- [x] 在 `planner` 新增 `._build_action_intent_finalize_withheld_mixed_horizon_later_target_revisit_decision(...)`
  - 若 later target 轨迹未建立：先 `query_object`
  - 若已有 later node：直接 `query_spatial_context`
- [x] 已接入：
  - `resolve_action_intent_pairwise` 之后的 finish 前恢复
  - `resolve_action_intent_future_use` 之后的 finish 前恢复
  - `verifier blocked finish` 的 close-call 恢复路径

本轮新增测试：

- [x] `test_graph_agent_action_intent_finalizer_marks_later_fixture_target_for_mixed_horizon_open_vs_weigh`
  - finalizer 现在会把 later target 明确记成 `scale`
- [x] `test_graph_agent_action_intent_finalizer_marks_later_fixture_target_for_mixed_horizon_label_vs_put_back`
  - finalizer 现在会把 later target 明确记成 `fridge`
- [x] `test_planner_action_intent_mixed_horizon_later_target_marker_revisits_real_fixture_target`
  - planner 现在会直接追 `scale` 的 later node，而不是继续围着 `jar` 做泛化后追

预期收益：

- [x] mixed-horizon 题的关键帧搜索从“知道证据不够”升级为“知道该去哪里找证据”
- [x] `label/open vs put-back/weigh` 这类题不再只做泛化 long-horizon followup
- [x] agent 主动检索 later target 的能力更强，更接近真实视频推理链

---

## 16.38 2026-06-07 本轮进展：unresolved rerank 的 mixed-horizon 也开始直接追 later target

本轮 residual bucket：

- bucket 名：`mixed-horizon unresolved-rerank generic long-horizon leak`
- 失效层级：`planner unresolved rerank recovery`
- 典型问题：
  - `unresolved rerank` 已经明确给出：
    - `missing_later_outcome_evidence`
    - 或 `timeline_review_final_location_gap / next_use_gap`
  - 但旧恢复逻辑通常还是：
    - 沿动作物体本身做 generic long-horizon revisit
  - 没有进一步把：
    - `fridge`
    - `scale`
    - 等真正决定 later outcome 的目标单独拉出来
  - 于是 agent 虽然知道“后面的证据还没看到”，但关键帧搜索仍然偏泛

本轮完成点：

- [x] 在 `planner` 新增 `._action_intent_unresolved_rerank_mixed_horizon_later_target_hint(...)`
  - 当 unresolved rerank 的最佳/竞争候选跨 immediate micro-outcome 与 later outcome
  - 且 reason 已明确指向 later outcome 缺失
  - 会主动识别 later target：
    - `measure_weigh` -> `scale`
    - `final_place_return` -> `fridge/drawer/cupboard/...`
    - 以及 choice 中显式提到的其它目标
- [x] 在 `planner` 新增 `._build_action_intent_unresolved_rerank_mixed_horizon_later_target_revisit_decision(...)`
  - 若 later target 轨迹未建立：先 `query_object`
  - 若已有 later node：直接 `query_spatial_context`
- [x] 已接入：
  - `resolve_action_intent_pairwise` 之后的 unresolved rerank 恢复
  - `resolve_action_intent_future_use` 之后的 unresolved rerank 恢复

本轮新增测试：

- [x] `test_planner_action_intent_unresolved_rerank_mixed_horizon_prefers_later_fixture_target`
  - unresolved rerank 在 `open jar vs weigh` close-call 下，会直接追 `scale`
- [x] `test_planner_action_intent_unresolved_rerank_timeline_gap_prefers_later_object_revisit`
  - 现已按新行为更新，mixed-horizon 的 final-location gap 会先定位 `fridge`，不再默认沿 `bottle` 泛化后追

预期收益：

- [x] `unresolved rerank` 不再只是“知道证据不足”，而是开始“知道下一步该去哪里找”
- [x] `label/open vs put-back/weigh` 这类题在还没进入 finalizer 前，关键帧检索也已经更贴 later outcome
- [x] why 题的关键帧选择链又少了一层泛化 long-horizon 漏斗

---

## 16.13 一句话结论

当前 why 逻辑专项已经从“零散规则期”进入“证据闭环期”。

接下来不该继续泛泛说“优化逻辑推理”，而应该严格按下面主线推进：

- `Phase 2`：planner 按冲突类型补证据
- `Phase 3`：verifier 阻止证据不足时过早 finish
- `Phase 4`：toolbox 与 graph 语义收敛
- `Phase 5`：真实 replay 和分 bucket 统计

这条线才是真正把当前 why 模块做成研究级 agent 组件的关键路径。

---

## 16.39 2026-06-07 本轮进展：mixed-horizon later target 从 `scale/fridge` 扩到 `sink/plate/...`

本轮 residual bucket：

- bucket 名：`mixed-horizon later-target under-specification`
- 失效层级：`finalizer marker + planner unresolved rerank`
- 典型问题：
  - agent 已经知道当前证据不足，也知道冲突是：
    - `inspection/open/check`
    - 对
    - `empty/pour/serve/discard`
  - 但旧逻辑里 later target 抽取仍偏弱：
    - 稳定的主要只有 `scale/fridge`
    - 对 `to empty the water` 这类 choice 文本不写目标的题，拿不到 `sink`
    - 对 `to serve the food` 这类 choice 文本不写容器的题，拿不到 `plate/bowl`
  - 结果是 planner 虽然会继续追证据，但还是更容易沿动作物体本身泛化后追，关键帧命中率不够高

本轮完成点：

- [x] 在 `graph_agent` 新增 `._action_intent_later_outcome_target_token_and_kind(...)`
  - 统一 later-outcome target 抽取逻辑
  - 在 choice 本身没有明确 target 时，也会从 `reason / decisive_observation / needed_observation / candidate_evidence` 中补抓 target
  - 当前覆盖：
    - fixture：`scale / sink / fridge / drawer / cupboard / rack / dishwasher / bin`
    - object：`bowl / plate / tray / pan / pot / saucepan / cup / glass / jar / colander / container`
- [x] `mixed_horizon_later_target_marker` 不再只看 choice 文本
  - 现在会把 later candidate 的 support / contradiction 一并纳入
  - 因而像：
    - `to empty the water`
    - 但证据里出现 `toward the sink / no tilt toward the sink`
  - 也能明确把 later target 记成 `sink`
- [x] `planner unresolved rerank` 同步接入同类 later-target 抽取
  - 当 reason 是 `missing_later_outcome_evidence / timeline_review_next_use_gap / final_location_gap`
  - 会优先追 later target，而不是继续围绕动作物体做泛化后追

本轮新增测试：

- [x] `test_graph_agent_action_intent_finalizer_marks_later_fixture_target_for_mixed_horizon_open_vs_empty`
  - 覆盖 `open vs empty`
  - finalizer 现在会从证据文本里恢复 `sink`
- [x] `test_graph_agent_action_intent_finalizer_marks_later_object_target_for_mixed_horizon_open_vs_serve`
  - 覆盖 `open vs serve`
  - finalizer 现在会从证据文本里恢复 `plate`
- [x] `test_planner_action_intent_unresolved_rerank_mixed_horizon_prefers_sink_fixture_target_from_evidence`
  - unresolved rerank 不再只看 `pot`
  - 会直接追 `sink`
- [x] `test_planner_action_intent_unresolved_rerank_mixed_horizon_prefers_plate_object_target_from_evidence`
  - unresolved rerank 不再只看 `pan`
  - 会直接追 `plate`

预期收益：

- [x] why 题里“证据不足时继续找哪里”这件事更接近真实视频推理，而不是弱化成 generic long-horizon scan
- [x] `inspection/open/check vs empty/pour/serve/discard` 这一簇题的关键帧选择会更准
- [x] 这不是单视频单题修补，而是把 later-target 恢复从 `scale/fridge` 扩成一个更完整的语义簇

---

## 16.40 2026-06-07 本轮进展：把 `needed_observation` 从“说明文字”升级成“判别式追证据目标”

本轮 residual bucket：

- bucket 名：`needed-observation not operationalized`
- 失效层级：`planner late recovery`
- 典型问题：
  - 模型已经在 `needed_observation` 里明确写出：
    - `whether the bottle is read/checked first or put back in the fridge`
    - `whether the bowl is only staged nearby or actually placed onto the scale`
    - `whether the container is actually carried over the plate or just opened next`
  - 但旧 planner 主要只把这段文字用来决定“继续补帧”
  - 没有把它转成真正的行动：
    - 去追 `fridge`
    - 去追 `scale`
    - 去追 `plate`
  - 结果就是 agent 明明已经知道“到底缺什么证据”，却还在继续泛化 long-horizon scan

本轮完成点：

- [x] 在 `planner` 新增 `._action_intent_needed_observation_target_hint(...)`
  - 从 `needed_observation` 中提取唯一的判别目标
  - 当前只在更晚 followup 阶段触发，避免过早抢走 transition probe / initial followup
  - 仅当文本里只出现一个明确目标时才启用，避免：
    - `scale / sink / hob` 这种多目标混杂提示过早被单点化
- [x] 在 `planner` 新增 `._build_action_intent_needed_observation_target_revisit_decision(...)`
  - 若目标轨迹尚未建立：先 `query_object`
  - 若目标轨迹已存在：直接 `query_spatial_context`
  - 让 planner 真正围绕“需要排除哪个竞争项”去查证，而不是只沿动作物体泛化后追
- [x] 已接入 3 条关键恢复路径：
  - `infer_action_intent` 后续歧义恢复
  - `resolve_action_intent_pairwise` 后续歧义恢复
  - `resolve_action_intent_future_use` 后续歧义恢复

本轮新增/更新测试：

- [x] `test_planner_action_intent_needed_observation_target_revisit_from_infer_result_prefers_scale_fixture`
  - 当 `needed_observation` 明确要求区分 `staged nearby vs placed onto the scale`
  - planner 会直接追 `scale`
- [x] `test_planner_action_intent_needed_observation_target_revisit_from_future_use_result_prefers_plate_object`
  - 当 `needed_observation` 明确要求区分 `opened vs carried over the plate`
  - planner 会直接追 `plate`
- [x] 旧的 weak / nonexclusive late-anchor 测试已升级到新行为
  - 有些 case 现在会优先 `query_object fridge`
  - 有些 case 在已有 later node 时会直接 `query_spatial_context`
  - 这反映的是恢复链更具体，而不是更松

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`308 passed, 344 deselected`

预期收益：

- [x] agent 不只是“知道还缺证据”，而是开始“知道缺哪一个判别目标上的证据”
- [x] 对 `label/open vs put-back/scale/serve` 这类题，关键帧检索会更像真正的判别式推理
- [x] 这条能力会明显降低“只凭当前弱证据就过早定答”的概率

---

## 16.41 2026-06-07 本轮进展：把 `needed_observation` 从“追目标”继续升级成“追关系 + 主动借晚轨迹定位判别帧”

本轮 residual bucket：

- bucket 名：`needed-observation relation not operationalized`
- 失效层级：`planner discriminative frame selection`
- 典型问题：
  - 即使 `needed_observation` 已经明确写出：
    - `whether the bowl is only staged nearby or actually placed onto the scale`
    - `whether the bottle is read first or instead returned to the fridge`
    - `whether the container is actually carried over the plate or just opened next`
  - 旧恢复链也还是容易退化成两种低效动作：
    - 继续泛化补帧
    - 或者只去追目标物体本身
  - 这会错过真正有判别力的关键帧：
    - 不是“scale 出现了没有”
    - 而是“bowl 和 scale 是否真的形成 `on` 关系”
    - 不是“fridge 在附近”
    - 而是“bottle 是否真的被 `returned` 到 fridge”

本轮完成点：

- [x] 在 `planner` 新增 `._action_intent_needed_observation_relation_hint(...)`
  - 从 `needed_observation` 中恢复唯一的判别关系
  - 当前支持：
    - `on_target`
    - `return_to_target`
    - `into_target`
    - `over_target`
- [x] 在 `planner` 新增 `._build_action_intent_needed_observation_relation_revisit_decision(...)`
  - 优先查动作物体与目标之间的关系，而不是只查目标是否出现
  - 若动作物体已有更晚轨迹：直接在更晚节点做 `query_spatial_context(action_object)`
  - 若动作物体还没有更晚轨迹，但目标已经有更晚轨迹：
    - 直接借目标的更晚节点定位判别时刻
    - 再去查动作物体在该时刻的空间关系
  - 只有两边都没有可靠晚轨迹时，才退回 `query_object`
- [x] 已接入 3 条关键恢复路径：
  - `infer_action_intent`
  - `resolve_action_intent_pairwise`
  - `resolve_action_intent_future_use`

本轮新增/更新测试：

- [x] `test_planner_action_intent_needed_observation_relation_revisit_from_infer_result_prefers_action_object_scale_relation`
  - scale 已有晚轨迹时，不再先退回查 bowl 轨迹
  - 直接借 scale 的更晚时间点去查 bowl 与 scale 的关系
- [x] `test_planner_action_intent_needed_observation_relation_revisit_from_future_use_result_prefers_action_object_plate_relation`
  - plate 已有晚轨迹时，直接查 container 是否真的移动到 plate 上方
- [x] `test_planner_action_intent_needed_observation_target_revisit_without_relation_prefers_target_object`
  - 没有关系词时，仍保持 target-aware 恢复，不误抢
- [x] weak / nonexclusive late-anchor 相关测试已同步升级
  - 一旦 `needed_observation` 已明确到关系层，恢复链就优先去查关系证据
  - 不再停留在 `query_object(fridge/scale)` 这种更弱的一步

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'needed_observation_relation_revisit or needed_observation_target_revisit_without_relation or weak_late_anchor or nonexclusive_concrete_late_anchor'`
- [x] 结果：`8 passed, 645 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`309 passed, 344 deselected`

预期收益：

- [x] 关键帧选择更接近“先找最能排除竞争项的关系帧”，而不是泛化后追
- [x] agent 会更像主动找证据，而不是在证据不足时也急着定答
- [x] 对 `nearby vs on scale`、`check first vs put back`、`open next vs carry over plate` 这类 why 题，动作理解链条更完整

---

## 16.42 2026-06-07 本轮进展：`verifier` 拦下 finish 后，优先走 `needed_observation` 的定向恢复，不再退回泛化补帧

本轮 residual bucket：

- bucket 名：`verifier-blocked close call falls back too generically`
- 失效层级：`planner recovery after verifier_blocked_finish`
- 典型问题：
  - `verifier` 已经能识别：
    - 当前答案不能结束
    - 还缺 `need_disambiguating_evidence`
  - 但旧恢复链在很多 case 里还是会优先：
    - 泛化 transition probe
    - 泛化 extra followup
    - 泛化 future-use / pairwise 重跑
  - 如果最近结果里其实已经有：
    - `needed_observation: whether the bowl is only staged nearby or actually placed onto the scale`
    - `needed_observation: whether the hidden spice jar is taken afterwards`
  - 那么更合理的恢复动作应该是：
    - 直接追 `bowl <-> scale` 的关系帧
    - 或直接追 `jar` 的更晚证据
  - 而不是先退回更宽的补帧

本轮完成点：

- [x] 在 `planner._recover_action_intent_after_verifier_blocked_finish(...)` 最前面接入：
  - `._build_action_intent_needed_observation_relation_revisit_decision(...)`
  - `._build_action_intent_needed_observation_target_revisit_decision(...)`
- [x] 新行为：
  - 只要 `verifier` 已拦下 why 题 finish
  - 且最近 specialized/infer 结果里已经明确给出 `needed_observation`
  - planner 就优先走：
    - relation-aware 恢复
    - target-aware 恢复
  - 只有这两条都不成立时，才退回原来的泛化 transition / followup / specialized rerun

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_finish_needed_observation_relation_prefers_relation_revisit`
  - `infer_action_intent` 被 verifier 拦下后
  - 不再先做泛化补帧
  - 直接借 `scale` 的晚轨迹去查 `bowl` 与 `scale` 的关系
- [x] `test_planner_action_intent_verifier_blocked_finish_needed_observation_target_prefers_target_revisit`
  - `future_use` 被 verifier 拦下后
  - 不再先退回泛化 close-call recovery
  - 直接追 `jar` 的更晚证据

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`6 passed, 649 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`311 passed, 344 deselected`

预期收益：

- [x] why 题在“已经知道缺什么证据、而 verifier 也明确阻止结束”时，会更像 agent 地直接追关键判别帧
- [x] 进一步减少“明明 verifier 说不能结束，但恢复动作还是太泛”的问题
- [x] 对整类 close-call why 题，关键帧选择和追证据顺序都会更主动、更具体

---

## 16.43 2026-06-07 本轮进展：`verifier-blocked` 场景下，近窗直接效果/状态变化证据优先于错误的 long-horizon 恢复

本轮 residual bucket：

- bucket 名：`verifier-blocked direct-effect gets hijacked by long-horizon recovery`
- 失效层级：`planner recovery priority`
- 实际复现到的问题：
  - 当题目真正缺的是：
    - `more post-action frames showing the direct physical effect`
    - `display state change`
  - 但 `working_memory` 里又残留了：
    - `timeline_review_bias_gap`
    - 或其它晚时域 withheld marker
  - 旧恢复链会先被带去：
    - `query_spatial_context` 查更晚目标
    - 或泛化 `followup_ext*`
  - 结果是：
    - `move glass` 这种本该先看“后面有没有马上拿后面的东西”的题，被错误拉去追更晚 glass 轨迹
    - `tap kitchen scale` 这种本该先看显示是否变化的题，被错误拉去普通 followup，而不是 `followup_transition`

本轮完成点：

- [x] 在 `planner` 新增：
  - `._action_intent_verifier_blocked_prefers_forced_transition_probe(...)`
  - `._build_action_intent_verifier_blocked_forced_transition_probe_decision(...)`
- [x] 新行为：
  - 当 `verifier` 已阻止 why 题 finish
  - 且最近结果表明当前缺的是：
    - `direct physical effect`
    - `missing_direct_effect`
    - `display/readout/tare/zero/reset`
    - 或 `needed_observation_profile.prefer_state_change_only`
  - planner 会先强制走 `followup_transition`
  - 这条路径会压过：
    - `timeline_review_bias_gap`
    - 泛化 long-horizon revisit
    - 泛化 extra followup
- [x] 这样做的核心意义：
  - 先把“动作后立刻发生了什么”看清
  - 再决定是否需要更晚时域
  - 避免本来是近窗状态变化题，却被错误拉成 long-horizon 题

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_finish_missing_direct_effect_forces_transition_probe`
  - 复现 `move glass` + `missing_direct_effect` + `timeline_review_bias_gap`
  - 现在不再去追更晚 glass node
  - 改为直接 `followup_transition`
- [x] `test_planner_action_intent_verifier_blocked_finish_state_change_forces_transition_probe`
  - 复现 `tap kitchen scale` + `display state change`
  - 现在不再退回普通 `followup_ext*`
  - 改为直接 `followup_transition`

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`8 passed, 649 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`313 passed, 344 deselected`

预期收益：

- [x] why 题对“动作后立刻结果”的理解会更稳，不容易再被错误的晚时域标记带偏
- [x] `move glass / tap kitchen scale / flip cloth / switch/tap/open-close` 这类题的关键帧选择会更符合真实判别逻辑
- [x] 进一步压低“视频理解太弱，没先看近窗决定性变化就过早定答”的问题

---

## 16.44 2026-06-08 本轮进展：`verifier-blocked` 的 infer 恢复也开始直接追 finalizer 已指出的 downstream target

本轮 residual bucket：

- bucket 名：`verifier-blocked generic access/relocation still falls back generically`
- 失效层级：`planner recovery after verifier_blocked_finish`
- 实际复现到的问题：
  - working memory 里其实已经有：
    - `action_intent_resolution_withheld_for_generic_access_or_space_enablement=1 target=jar kind=object`
    - `action_intent_resolution_withheld_for_generic_relocation_or_storage_enablement=1 target=jar kind=object`
  - 这意味着 finalizer 早就识别出：
    - generic access 不是结论
    - generic put-away / relocation 也不是结论
    - 真正应该追的是 `jar`
  - 但在 `verifier` 拦下 `infer_action_intent` 之后，旧恢复链仍可能先走：
    - 普通 `followup`
    - 普通 close-call recovery
  - 这就还是“知道缺什么，却没直接去查它”

本轮完成点：

- [x] 在 `planner._recover_action_intent_after_verifier_blocked_finish(...)` 中新增优先恢复顺序：
  - `._build_action_intent_finalize_withheld_generic_access_or_space_revisit_decision(...)`
  - `._build_action_intent_finalize_withheld_generic_relocation_or_storage_revisit_decision(...)`
  - `._build_action_intent_finalize_withheld_generic_hand_free_revisit_decision(...)`
- [x] 新行为：
  - 只要 `verifier` 已拦下 why 题 finish
  - 且 finalizer marker 已明确指出真实 downstream target
  - planner 就直接去追这个 target 的 later node / spatial context
  - 不再先退回泛化 `followup`

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_finish_generic_access_marker_prefers_downstream_target`
  - `move bottle` 在 generic access 被拦下后
  - 现在会直接追 `jar`
- [x] `test_planner_action_intent_verifier_blocked_finish_generic_relocation_marker_prefers_downstream_target`
  - `move bottle` 在 generic put-away 被拦下后
  - 现在也会直接追 `jar`

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`10 passed, 649 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`315 passed, 344 deselected`

预期收益：

- [x] `reveal-then-use / hidden-target / generic put-away` 这类 why 题在 verifier 已阻止结束后，会更快收敛到真实下游目标
- [x] agent 的恢复动作更像“主动检索已知关键对象”，而不是重复泛化补帧
- [x] 进一步压低“中间态解释过强、真实 downstream target 没被继续追”的问题

---

## 16.45 2026-06-08 本轮进展：`verifier-blocked` 的 infer 恢复开始识别 same-object component use，并主动追 `lid/cap/cover`

本轮 residual bucket：

- bucket 名：`verifier-blocked same-object active use still falls back generically`
- 失效层级：`planner recovery after verifier_blocked_finish`
- 实际复现到的问题：
  - `infer_action_intent` 已经暴露出 close call：
    - 一个候选是 `later downstream use`，例如 `use the container on the scale`
    - 另一个候选是 `same-object active use`，例如 `open the lid`
  - `verifier` 也已经明确阻止 finish，说明当前证据不足以直接定答
  - 但旧恢复链仍可能因为选项没有直接复述原物体名 `container`
    - 没把 `lid/cap/cover` 识别成同一物体的派生部件
    - 从而漏掉 “继续追这个容器后面是否真的被打开/继续操作” 这条更有判别力的路径
  - 最终 planner 会掉回泛化 `sample_sparse_frames`，而不是直接追这个动作物体的更晚状态

本轮完成点：

- [x] 在 `planner._action_intent_verifier_blocked_same_object_active_use_hint(...)` 中把 `second_best_index` 纳入第一优先级候选扫描
  - 不再只依赖 `best_index / competitor_index / candidate_evidence`
  - 这保证了模型显式给出的 top-2 close call 不会被漏掉
- [x] 扩展 `planner._action_intent_choice_is_same_object_active_use(...)`
  - 当动作物体是明显可带 `lid/cap/cover` 的容器类对象时
  - 即使选项没有直接复述原物体名
  - 只要它明确在说 `open / uncap / cover / replace / fit / pry` 这类同物体后续操作
  - 也视为 same-object active use
- [x] 新行为：
  - 只要 why 题已经被 `verifier` 拦下
  - 且 top-2 close call 落在 “later downstream use vs same-object component use”
  - planner 就优先直接追动作物体的 later node / spatial context
  - 不再退回泛化补帧

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_finish_same_object_active_use_prefers_action_object_revisit`
  - `take container`
  - `to use the container on the scale.` vs `to open the lid.`
  - 在 `verifier` 已阻止 finish 后
  - 现在会直接对 `container` 走更晚时刻的 `query_spatial_context`
- [x] `test_planner_action_intent_verifier_blocked_finish_same_object_component_use_prefers_bottle_revisit`
  - `take bottle`
  - `to use the bottle on the scale.` vs `to uncap the bottle.`
  - 在 `verifier` 已阻止 finish 后
  - 现在会直接对 `bottle` 走更晚时刻的 `query_spatial_context`

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`12 passed, 649 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`317 passed, 344 deselected`

预期收益：

- [x] `open lid / uncap / replace cover / confirm cover fits` 这类同物体后续操作，不会再因为没直说原物体名而漏掉
- [x] why 题在 “多个选项仍同时成立” 时，会更稳定地转向主动找后续关键证据，而不是提前收口
- [x] agent 对关键帧的选择会更贴近真实判别链：先追动作物体后面到底发生了什么，而不是继续看泛化中间态

---

## 16.46 2026-06-08 本轮进展：mixed-horizon later-target marker 优先级前移，不再被 same-object / near-window probe 抢走

本轮 residual bucket：

- bucket 名：`verifier-blocked mixed-horizon marker still gets hijacked by near-window recovery`
- 失效层级：`planner recovery priority after verifier_blocked_finish`
- 实际复现到的问题：
  - `finalizer` 明明已经在 working memory 里写出：
    - `action_intent_resolution_withheld_for_mixed_horizon_later_target=1 target=scale kind=fixture`
    - 或 `target=plate kind=object`
  - 这说明系统已经识别出：
    - `open / uncap / open lid` 这种近窗解释还不够排他
    - 真正更有判别力的是去追更晚结果对应的真实目标
  - 但旧恢复链里：
    - `same-object active use revisit`
    - 或 `forced transition probe`
    - 仍可能先抢到优先级
  - 结果就是 agent 继续围着 `jar / container` 的局部状态打转，没有直接去查 `scale / plate`

本轮完成点：

- [x] 在 `planner._recover_action_intent_after_verifier_blocked_finish(...)` 中前移 mixed-horizon later-target marker 恢复优先级
  - 只要当前是 `infer_action_intent`
  - 且 `verifier` blocker 属于 `post_action_evidence / future_use_close_call / pairwise_close_call`
  - 就先尝试 `._build_action_intent_finalize_withheld_mixed_horizon_later_target_revisit_decision(...)`
- [x] 新行为：
  - 若 finalizer 已明确指出真实 later target
  - planner 会优先直接追这个 target 的 later node / spatial context
  - 不再先被 `open/uncap` 的 same-object revisit 或近窗 transition probe 抢走

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_finish_mixed_horizon_marker_beats_same_object_revisit_for_fixture`
  - `take jar`
  - `to open the jar.` vs `to use the jar to weigh the ingredients.`
  - 当 marker 已指向 `scale`
  - 现在会直接追 `scale`
- [x] `test_planner_action_intent_verifier_blocked_finish_mixed_horizon_marker_beats_same_object_revisit_for_object`
  - `take container`
  - `to open the lid.` vs `to serve the food.`
  - 当 marker 已指向 `plate`
  - 现在会直接追 `plate`

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`14 passed, 649 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`319 passed, 344 deselected`

预期收益：

- [x] `open/uncap` vs `weigh/serve/...` 这类 mixed-horizon why 题，在证据仍冲突时会更稳定地去追真正更有判别力的 later target
- [x] agent 不会再因为局部近窗线索太显眼，就放弃更关键的后续结果检索
- [x] 关键帧选择进一步从“盯住当前物体”推进到“主动追真实判别目标”

---

## 16.47 2026-06-08 本轮进展：`infer_action_intent` 自己暴露 mixed-horizon close call 时，也会主动直追 later target

本轮 residual bucket：

- bucket 名：`verifier-blocked infer mixed-horizon still waits too long for finalizer markers`
- 失效层级：`planner recovery after verifier_blocked_finish`
- 实际复现到的问题：
  - 旧逻辑里，很多 `open/uncap` vs `weigh/serve/...` 的 why 题要等到：
    - finalizer 写出 `action_intent_resolution_withheld_for_mixed_horizon_later_target=*`
  - planner 才会稳定直追 later target
  - 但在更早的 `infer_action_intent` 阶段，其实模型已经明确承认：
    - `still unclear whether ...`
    - `not yet visible whether ...`
    - `opened now or used on the scale next`
    - `opened now or used to serve onto a plate next`
  - 也就是 mixed-horizon close call 已经暴露出来了，只是系统还在等 marker
  - 结果就是 agent 容易先被近窗 `transition probe` 或普通 same-object 恢复吸走，延后了真正有判别力的 later-target 检索

本轮完成点：

- [x] 新增 `planner._action_intent_verifier_blocked_mixed_horizon_later_target_hint(...)`
  - 直接读取 `infer_action_intent` 的 top-2 候选
  - 识别“一个是 immediate micro outcome，另一个是 later outcome”的 mixed-horizon 结构
  - 结合 `reason / needed_observation / candidate_evidence` 判断当前是否明确处于 `whether / unclear / not yet visible` 这类未决状态
- [x] 新增 `planner._build_action_intent_verifier_blocked_mixed_horizon_later_target_revisit_decision(...)`
  - 在 `verifier-blocked` 恢复链里，不等 finalizer marker
  - 直接对 `scale / plate / sink / fridge ...` 这类 later target 做 `query_object / query_spatial_context`
- [x] 同时补了保守边界：
  - 如果 `infer` 自己已经明确说缺的是 `same-object cap/lid/cover` 证据
  - 就不让 later-target 恢复误抢 same-object case
  - 也就是：
    - `same-object cap action is still unresolved`
    - 这类提示仍优先走 same-object revisit

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_infer_mixed_horizon_prefers_later_fixture_target_without_marker`
  - `take jar`
  - `to open the jar.` vs `to use the jar to weigh the ingredients.`
  - 即使没有 finalizer marker
  - 只要 `infer` 已明确承认 mixed-horizon close call
  - 现在也会直接追 `scale`
- [x] `test_planner_action_intent_verifier_blocked_infer_mixed_horizon_prefers_later_object_target_without_marker`
  - `take container`
  - `to open the lid.` vs `to serve the food.`
  - 即使没有 finalizer marker
  - 现在也会直接追 `plate`
- [x] 原有 same-object 保护测试继续通过：
  - `take container` 的 `open lid`
  - `take bottle` 的 `uncap bottle`
  - 不会被 later-target 恢复误抢

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`16 passed, 649 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`321 passed, 344 deselected`

预期收益：

- [x] agent 会更早主动定位真正的关键帧位置，不再被动等待 finalizer 再提醒
- [x] `open/uncap` vs `weigh/serve/...` 这类题的关键帧搜索更像真实 agent：一旦发现是 mixed-horizon close call，就直接去找 later target
- [x] 同时保留了 same-object 路径的边界，不会把所有 `open/uncap` 题都误推成 later-target 检索

---

## 16.48 2026-06-08 本轮进展：`infer_action_intent` 暴露 generic hand-free close call 时，也会主动追真实下游对象或同物体后续用途

本轮 residual bucket：

- bucket 名：`verifier-blocked infer generic hand-free still waits too long for later specialized hints`
- 失效层级：`planner recovery after verifier_blocked_finish`
- 实际复现到的问题：
  - 旧逻辑里，`generic hand-free`
    - `so left hand is free`
    - `the other hand becomes free`
  - 这类中间态如果出现在 `infer_action_intent`
  - 即使模型已经明确承认：
    - `still unclear what the free hand reaches for next`
    - `it is still unclear whether the same cup is rinsed next`
  - planner 也往往还要等：
    - unresolved rerank 写出 `timeline_review_hand_free_or_fixture_gap`
    - 或 finalizer 写出 `generic_hand_free_enablement` marker
  - 才会稳定转去追真正的下游对象/同物体后续用途
  - 这会让系统在更早的恢复阶段仍然停留在 hand-free 中间态上

本轮完成点：

- [x] 新增 `planner._action_intent_verifier_blocked_hand_free_target_hint(...)`
  - 直接读取 `infer_action_intent` 的 top-2 close call
  - 当当前已经明确暴露 `free hand / other hand / left hand / right hand` 结构
  - 且 `reason / candidate_evidence` 里同时承认目标仍未决
  - 就主动恢复真正值得追的：
    - 下游对象
    - 下游 fixture
    - 或 same-object 后续用途
- [x] 新增 `planner._build_action_intent_verifier_blocked_hand_free_target_revisit_decision(...)`
  - 在 `verifier-blocked` 恢复链里，不等 finalizer marker
  - 直接对真实 hand-free 下游目标做 `query_object / query_spatial_context`
- [x] 新增边界辅助：
  - `planner._action_intent_choice_has_hand_free_language(...)`
  - `planner._action_intent_choice_target_or_same_object_hint(...)`
  - 用于把 hand-free 中间态和真实 downstream target / same-object use 分开

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_infer_generic_hand_free_prefers_downstream_object`
  - `transfer bowl`
  - `so left hand is free.` vs `so I can pick up the sponge.`
  - 即使没有 unresolved rerank / finalizer marker
  - 现在也会直接追 `sponge`
- [x] `test_planner_action_intent_verifier_blocked_infer_generic_hand_free_prefers_same_object_use`
  - `transfer blender cup`
  - `so left hand is free.` vs `so I can rinse the blender cup.`
  - 即使没有后续 marker
  - 现在也会直接追 `blender cup`

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`16 passed, 651 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`323 passed, 344 deselected`

预期收益：

- [x] hand-free 类 why 题在更早阶段就会主动寻找真正有判别力的关键帧，而不是先停留在 enablement 中间态
- [x] `free hand -> pick up tool/object`
- [x] `free hand -> continue operating the same object`
- [x] 这两类高频链条的恢复会更像真实 agent，而不是等更后面的 specialized marker 再补救

---

## 16.49 2026-06-08 本轮进展：`infer_action_intent` 暴露 generic measurement-meta close call 时，也会主动追更有判别力的量测目标

本轮 residual bucket：

- bucket 名：`verifier-blocked infer generic measurement-meta still waits too long for later specialized hints`
- 失效层级：`planner recovery after verifier_blocked_finish`
- 实际复现到的问题：
  - 旧逻辑里，像：
    - `to adjust the measurements.`
    - `to read the measurements.`
  - 这类 generic measurement-meta 如果出现在 `infer_action_intent`
  - 即使模型已经明确承认：
    - `it is still unclear whether the scale is adjusted or used to measure the cheese next`
    - `no reading or tare action is yet visible`
  - planner 也还是容易继续停留在“量测语境”层
  - 要等更后面的 finalizer / unresolved rerank 才会更稳定地去追真正的量测目标
  - 这会导致系统在更早的恢复阶段还没有主动去找更有判别力的称量关键帧

本轮完成点：

- [x] 新增 `planner._action_intent_choice_is_generic_measurement_meta_purpose(...)`
  - 识别 `adjust/read/record measurements` 这类 measurement-meta 候选
- [x] 新增 `planner._action_intent_choice_is_exact_measurement_role_purpose(...)`
  - 识别 `measure/weigh the X`、`base for weighing` 这类 exact measurement role 候选
- [x] 新增 `planner._action_intent_verifier_blocked_measurement_target_hint(...)`
  - 直接读取 `infer_action_intent` 的 top-2 close call
  - 当当前已经是 `generic measurement-meta` vs `exact measurement role`
  - 且 `reason / candidate_evidence` 明确承认：
    - 还没看到 reading/tare
    - 还没看到 exact measurement role
  - 就直接恢复真正更有判别力的量测目标
- [x] 新增 `planner._build_action_intent_verifier_blocked_measurement_target_revisit_decision(...)`
  - 在 `verifier-blocked` 恢复链里，不等后续 marker
  - 直接对 `scale` 做 `query_object / query_spatial_context`

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_infer_measurement_meta_prefers_exact_measurement_target`
  - `pick up scale`
  - `to adjust the measurements.` vs `so that I can measure the cheese.`
  - 即使没有 finalizer / unresolved rerank marker
  - 现在也会直接追 `scale`
- [x] `test_planner_action_intent_verifier_blocked_infer_measurement_meta_prefers_exact_measurement_target_for_reading_variant`
  - `pick up scale`
  - `to read the measurements.` vs `so that I can weigh the cheese.`
  - 同样会直接追 `scale`

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`16 passed, 653 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`325 passed, 344 deselected`

预期收益：

- [x] measurement 类 why 题在更早阶段就会主动去找真正有判别力的称量关键帧，而不是先停留在宽泛量测语境上
- [x] `adjust/read measurements -> exact weighing role`
- [x] 这类高频 close call 的恢复会更像真实 agent，而不是等更后面的 specialized marker 再补救

---

## 16.50 2026-06-08 本轮进展：`pick up phone` 的 generic-measure close call 也会主动追具体食材记录目标

本轮 residual bucket：

- bucket 名：`verifier-blocked infer phone generic-measure still stays at broad workflow context`
- 失效层级：`planner recovery after verifier_blocked_finish`
- 实际复现到的问题：
  - 旧逻辑里，像：
    - `to measure the ingredients.`
    - `to record the nutritional value of the coriander.`
    - `to update the app with new measurements of the broccoli.`
  - 这类 `pick up phone` 的 why close call
  - 即使模型已经明确承认：
    - 当前还只是 broad measurement workflow
    - 具体记录的是哪种食材还不清楚
    - app screen / record target 还看不清
  - planner 也还是容易停留在“手机处于测量流程中”的宽泛解释层
  - 还没有主动把关注点前移到真正有判别力的具体食材目标上

本轮完成点：

- [x] 新增 `planner._action_intent_choice_is_phone_app_record_target_purpose(...)`
  - 识别 `record/update/log ... of the X` 这类手机记录型候选
- [x] 新增 `planner._action_intent_choice_is_generic_measure_phone_goal(...)`
  - 识别 `pick up phone` 下的 broad `measure the ingredients` 候选
- [x] 新增 `planner._action_intent_choice_phone_record_target_hint(...)`
  - 从 `nutritional value of the coriander`
  - `measurements of the broccoli`
  - 这类答案中抽取具体记录目标
- [x] 新增 `planner._action_intent_verifier_blocked_phone_record_target_hint(...)`
  - 当 top-2 已经形成：
    - broad phone measurement
    - vs exact ingredient record target
  - 且 `reason / candidate_evidence` 也承认：
    - `no direct recording target`
    - `screen not readable`
    - `no broccoli/coriander target visible`
  - 就不再继续停留在 broad phone-measure 解释层
- [x] 新增 `planner._build_action_intent_verifier_blocked_phone_record_target_revisit_decision(...)`
  - 直接对更有判别力的食材目标做 `query_object / query_spatial_context`

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_infer_phone_generic_measure_prefers_exact_record_target`
  - `to measure the ingredients.` vs `to record the nutritional value of the coriander.`
  - 现在会直接追 `coriander`
- [x] `test_planner_action_intent_verifier_blocked_infer_phone_exact_record_still_revisits_target`
  - `to update the app with new measurements of the broccoli.` vs `to measure the ingredients.`
  - 即使当前 best 已经是 exact record target
  - 只要证据链还不闭合
  - 仍会继续追 `broccoli`

验证结果：

- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`14 passed, 657 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`327 passed, 344 deselected`

预期收益：

- [x] `pick up phone` 这类 why 题不再只停在“手机处于测量流程里”这种中间态
- [x] 会更主动去定位真正的记录/录入目标食材
- [x] 对 `generic measure -> exact record target`
- [x] 以及 `exact record target -> 仍需继续确认具体食材`
- [x] 这两类 close call 都会更像真实 agent，而不是过早定答

补充收口：

- [x] `multiple exact record targets` 也已纳入同一条恢复链。此前 verifier-blocked recovery 只看 `generic measure` 与单个 exact-target 的 top-2 close call；如果候选里同时存在 `coriander / broccoli / carrot` 这类多个具体食材记录目标，就可能只盯住最早进入 top-2 的那个目标，而不会比较“哪个 exact target 当前更缺判别证据、也更值得继续追”。
- [x] 现在 `planner._action_intent_verifier_blocked_phone_record_target_hint(...)` 会扫描全部 `phone/app record target` 候选，综合 `candidate_evidence` 中的 `screen not readable / no broccoli target / no direct recording target / still unresolved` 这类 uncertainty marker，优先选择“目标最明确、但证据链仍最缺”的 exact target，而不是固定停在 top-2。
- [x] 对 phone record target revisit 的时序也补了 later-node 偏置：如果当前还没有更强的 anchor/followup 约束，就不再默认追该对象最早出现的轨迹，而会优先看更晚、更有判别力的目标节点。
- [x] 新增并通过 2 条定向测试，分别保护：
  - 多个 exact ingredient record target 同时存在时，会优先追当前最不确定、最需要补证据的那个目标；
  - 如果该目标还没有现成轨迹，则会先 `query_object` 重新检索它，而不是退回 generic measure 或继续追错对象。
- [x] 当前专项回归已进一步提升到 `352 passed, 344 deselected`

---

## 16.51 2026-06-08 本轮进展：`open/uncap` 与 `put back/store later` 冲突时，会更主动追最终位置证据

本轮 residual bucket：

- bucket 名：`mixed-horizon open-or-uncap vs final-return still stops too early at same-object near-window evidence`
- 失效层级：`planner recovery after verifier_blocked_finish`
- 实际复现到的问题：
  - 旧逻辑里，当 `infer_action_intent` 已经暴露：
    - `to uncap the bottle.`
    - `to put the bottle back in the fridge.`
  - 这类冲突时
  - 只要 `reason / needed_observation` 里出现：
    - `same-object`
    - `cap action`
    - `lid action`
  - mixed-horizon later-target 恢复就会整体让位
  - planner 容易继续围着当前 bottle 本身补近窗证据
  - 但这类题真正有判别力的往往不是“瓶盖会不会开”
  - 而是稍后是否真的回到 `fridge / drawer / cupboard / rack`

本轮完成点：

- [x] 收紧 `planner._action_intent_verifier_blocked_mixed_horizon_later_target_hint(...)` 的 same-object block 边界
  - 当 later candidate 属于 `final_place_return`
  - 不再仅因 `same-object / cap action / lid action` 文本就放弃 later-target 恢复
- [x] 在 `planner._build_action_intent_verifier_blocked_mixed_horizon_later_target_revisit_decision(...)` 中
  - 对 `fridge / drawer / cupboard / rack / dishwasher / shelf`
  - 这类最终归位目标
  - 优先取更晚、更有判别力的 long-horizon node
  - 不再停在第一个刚出现的 near-window fixture 节点

本轮新增测试：

- [x] `test_planner_action_intent_verifier_blocked_infer_open_vs_put_back_prefers_final_location_target`
  - `to uncap the bottle.` vs `to put the bottle back in the fridge.`
  - 当前 reason 已经包含 `same-object cap action`
  - 现在仍会继续追 `fridge`
  - 且取点偏到真正更晚的 final-location node

本轮回归验证：

- [x] `pytest -q tests/test_graph_agent.py -k 'open_vs_put_back_prefers_final_location_target or same_object_component_use_prefers_bottle_revisit or mixed_horizon_marker_beats_same_object or infer_mixed_horizon_prefers_later'`
- [x] 结果：`6 passed, 666 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish and action_intent'`
- [x] 结果：`14 passed, 658 deselected`
- [x] `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [x] 结果：`328 passed, 344 deselected`

预期收益：

- [x] `open/uncap` vs `put back/store later`
- [x] 这类题不会再被同物体近窗动作过早截走
- [x] agent 会更主动追最终位置/最终归位证据
- [x] 这更符合“多个选项都成立时继续找决定性证据”的目标
