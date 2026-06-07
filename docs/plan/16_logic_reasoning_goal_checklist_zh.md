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
- 2026-06-07 当前结果：`117 passed, 317 deselected`
- 相比本轮进入专项时的起点 `107 passed, 300 deselected`，当前阶段性增量为 `+10 passed`

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

- `c8faa0c`：why followup 路由加入候选感知，优先看当前 top candidate 语义，不再被无关选项把路由拉偏
- `d788dab`：why conflict suppression 收紧，`best_index` 或普通 textual fallback 不再直接掩盖真实冲突
- `af05385`：`pairwise` hidden-target hierarchy 对齐，支持 `generic access -> reveal-then-take/place`
- `f15a6f0`：`pairwise` direct safety 意图提升，优先识别 spill / burn / mess avoidance
- `d19606a`：`pairwise` hand-free hierarchy 对齐，区分 exact next-target enablement 与 generic free-hand
- `e6bcd2a`：`future_use` cleaning hierarchy 对齐，支持 exact cleaning target 与 workflow initiation 的更细粒度优先级

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

- 当前 why 逻辑专项约完成 `72%-76%`

这意味着：

- 骨架已经成立；
- 局部难点也已经被压下去一批；
- 但离“研究上站得住的逻辑推理 agent”还差最关键的证据闭环和真实 replay 证明。

---

## 16.4 完成定义

只有下面七条同时满足，才能认为 why 逻辑专项阶段性完成：

- [ ] `pytest -q tests/test_graph_agent.py -k 'action_intent'` 持续全绿
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
- [x] 记录当前通过数：`117 passed, 317 deselected`
- [x] 将基线结果写入本清单
- [ ] 建立后续 replay 统一结果模板

进展补充：

- [x] 本轮代码推进后，专项回归已从 `107 passed, 300 deselected` 提升到 `117 passed, 317 deselected`

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

完成标准：

- 这类题不再依赖个别工具名字触发

#### P1.2 `inspection` 与 `serve/pour/empty/check cooked` 的更细粒度拆分

- [ ] 继续压 `look into / check contents / check water level / check cooked state`
- [ ] 区分：
  - 短暂查看状态
  - 真正倾倒/清空
  - 真正准备上菜

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

本轮验证补充：

- [x] 新增并通过 `3` 条 `future_use` hierarchy 定向测试：
  - `generic store -> immediate reuse`
  - `finished with object -> immediate reuse`
  - `generic store -> exact final placement`
- [x] 定向回归：`pytest -q tests/test_graph_agent.py -k 'future_use_causal_hierarchy or future_use_sufficiency'`
  - 结果：`16 passed, 421 deselected`
- [x] 专项总回归：`pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 结果：`117 passed, 320 deselected`

下一步缺口：

- [x] `temporary relocation` vs `store away`
- [x] `immediate reuse` vs `finished with object`
- [x] `exact final placement` vs generic `store/put away`

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

1. `Phase 0 / P0.2`：残差桶台账
2. `Phase 4 / P4.2`：future-use causal hierarchy 继续扩到 `temporary relocation / immediate reuse / final placement`
3. `Phase 2 / P2.4`：hidden-access / pairwise 优先补动作后后果
4. `Phase 3 / P3.4-P3.5`：finalize/conflict filtering 统一收口
5. `Phase 2 / P2.1`：why 冲突 -> 补证据路由表统一入口继续整理成单入口
6. `Phase 5 / P5.1-P5.3`：真实 replay 与 bucket 统计

原因：

- `graph_agent` 纯语义层已经有明显基础；
- `pairwise` 与 cleaning-specific `future_use` 已经推进了一轮，下一步收益最大的点变成了 `temporary relocation / immediate reuse / final placement` 这组高频 future-use 残差；
- 只有先把“补什么证据”“何时仍不能 finish”“pairwise 的真实结果证据是什么”统一起来，后面 replay 才有研究意义。

---

## 16.9 当前默认起点

后续进入 `goal` 模式，默认从下面这个叶子项开始：

- [ ] `Phase 4 / P4.2`：继续扩展 future-use causal hierarchy 到 `temporary relocation / immediate reuse / final placement`

本轮默认目标：

- 把当前已经有的 cleaning-specific `future_use` hierarchy，继续扩到最常见的收尾误判簇：
  - 暂时挪开但并没有真正收纳
  - 很快还会再次使用，因此不能判为 finished
  - 真正回到固定挂点/抽屉/柜子时，优先判为 exact final placement，而不是 generic store

本轮硬要求：

- [ ] 至少新增 `2-4` 条 toolbox / graph 语义对齐测试
- [ ] 至少覆盖 `temporary relocation`、`immediate reuse` 两类不同冲突
- [ ] 至少覆盖一个 `exact final placement` 优先于 generic `store` 的场景
- [ ] 跑定向 `future_use` / `action_intent` 测试
- [ ] 跑 `pytest -q tests/test_graph_agent.py -k 'action_intent'`
- [ ] commit
- [ ] 回填本清单

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

## 16.13 一句话结论

当前 why 逻辑专项已经从“零散规则期”进入“证据闭环期”。

接下来不该继续泛泛说“优化逻辑推理”，而应该严格按下面主线推进：

- `Phase 2`：planner 按冲突类型补证据
- `Phase 3`：verifier 阻止证据不足时过早 finish
- `Phase 4`：toolbox 与 graph 语义收敛
- `Phase 5`：真实 replay 和分 bucket 统计

这条线才是真正把当前 why 模块做成研究级 agent 组件的关键路径。
