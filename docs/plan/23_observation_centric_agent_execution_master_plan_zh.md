# 23 Observation-Centric Agent 执行主计划

## 23.1 文档目的

这份文档不是为了继续修补某几个 `why / action_intent` 错题，
也不是为了继续堆专项规则把 `passed` 数做高。

这份文档只服务一个目标：

- 把当前系统从“候选答案驱动搜证”的旧范式
- 重构成“观测缺口驱动搜证”的通用 agent

核心要求：

- 不再从 `best_index / choice_text / runner_up / comparison_summary / blocking_hypotheses / blocking_comparisons / needed_observation`
  反推 `gap / target / search action`
- 不再为了某类固定冲突新增专项规则
- 不再让旧测试体系保护旧规则行为

这是一份给后续 goal 模式使用的执行文档。

---

## 23.2 总原则

### 23.2.1 研究原则

- 当前工作是做科研，不是做 benchmark 题型记忆器。
- 不允许再围绕固定答案语义、固定冲突模式、固定题型新增规则。
- 不再以 `pytest passed` 数增长为目标，只以“是否切断 answer-conditioned 主链”为目标。

### 23.2.2 工程原则

- 每轮只解决一个最小真实缺口。
- 先删旧链，再补通用 observation-centric 机制，最后再补负约束测试。
- 不允许同时在 `planner + graph_agent + verifier` 多处并行扩散修改，除非这一轮目标本身就是清理同一条链的 producer/consumer。

### 23.2.3 验证原则

每一阶段结束都只做 3 类验证：

- 结构验证：代码里是否还存在该类 `answer artifact -> gap/target/search` 消费链。
- 行为验证：系统是否仍能由 `observation state + budget` 解释下一步动作。
- 负约束验证：是否已经禁止旧的答案条件化行为回潮。

不再用“某个具体样例应该查某个具体对象”来定义正确性。

---

## 23.3 必须整体删除的旧机制

### 23.3.1 候选答案直接生成搜索缺口

必须整体删除以下旧行为：

- 从 `best_index / second_best_index / runner_up / losing_index` 推断搜索目标
- 从 `choice_text` 解析 `target_object / target_fixture / target_region`
- 从 `comparison_summary / blocking_hypotheses / blocking_comparisons` 推断“还缺什么证据”
- 从 `needed_observation / missing_observations` 直接生成主 gap

删除标准：

- 即使这些字段仍存在于 payload / trace / export 中，也不能再主导搜索主路径

### 23.3.2 Finalizer / Planner / Verifier 的 marker 回流链

必须整体删除以下旧行为：

- finalizer 根据 `best choice / competitor choice` 产出专项 marker
- planner 消费这些 marker 决定 `query_object / query_spatial_context / sample_sparse_frames`
- verifier 根据 `score_gap / close_call / candidate blocker` 直接驱动 specialized recovery
- executor/state 将候选层结论写回 `working_memory`，供后续搜索继续消费

删除标准：

- finalizer 不再为搜索主路径产出答案语义导向 marker
- planner 不再依赖 finalizer 产出的答案导向 marker 决定下一步动作

### 23.3.3 专项语义补丁层

必须退出运行态主路径的内容包括但不限于：

- `open/uncap vs later use`
- `make space vs hidden-target retrieval`
- `check/inspect vs put back / empty / serve`
- `hand-free vs exact downstream use`
- 各类 `choice_is_* / weak_* / exact_* / generic_* / override_* / overclaim_*`

处理原则：

- 不能继续作为主链决策机制
- 最多只能降级为：
  - 离线审计标签
  - 错误分析材料
  - legacy baseline 对照逻辑

### 23.3.4 基于候选竞争关系的扩窗与升级

必须删除以下旧行为：

- 因为 `score_gap` 小就扩窗
- 因为 `runner_up` 强就继续 specialized resolution
- 因为 `blocking_comparisons` 存在就继续 timeline review
- 因为 close call 存在就默认扩大搜索范围

删除标准：

- 是否继续搜索，只能由 observation gap、覆盖状态、轨迹闭合状态、空间关系确认状态、预算状态决定

### 23.3.5 保护旧行为的测试契约

必须删除或改写以下测试类型：

- 断言某种固定 `choice` 语义必须触发某种固定搜索动作
- 断言某类冲突必须走某个 fixed override
- 断言某个专项 marker 必须存在并驱动某个 follow-up
- 断言某个具体对象必须因某个候选项而被查询

新的测试目标只能是：

- 保护负约束
- 保护 observation-centric 主链
- 保护预算与停止策略

---

## 23.4 新系统的目标架构

### 23.4.1 总体结构

新主链必须拆成 6 层：

1. `Observation State Builder`
2. `Gap Inference`
3. `Budgeted Search Policy`
4. `Evidence Acquisition`
5. `Evidence Sufficiency Judge`
6. `Final Decision Mapper`

### 23.4.2 Observation State Builder

职责：

- 统一整理当前已经观测到的原始证据

允许进入的信息：

- 已抽取帧及其时间位置
- 帧级视觉描述
- 对象、区域、装置、容器、空间关系
- 对象出现/消失/移动/放置/取出轨迹
- 动作前后局部时间窗覆盖情况
- 当前预算状态
- 已执行工具历史

禁止进入的信息：

- `best_index`
- `choice_text`
- `runner_up`
- `comparison_summary`
- `blocking_hypotheses`
- `blocking_comparisons`
- `needed_observation`

输出必须能回答：

- 现在到底已经看到了什么
- 还有哪些对象轨迹未闭合
- 哪些空间关系未确认
- 哪些动作前后窗口尚未覆盖

### 23.4.3 Gap Inference

Gap 只能来自 observation state，建议只保留以下通用类型：

- `precondition_missing`
- `immediate_result_missing`
- `object_track_unclosed`
- `destination_unclosed`
- `relation_unobserved`
- `state_transition_unconfirmed`
- `workspace_change_unconfirmed`
- `window_coverage_missing`
- `budget_exhausted_without_resolution`

每个 gap 至少必须携带：

- `gap_type`
- `time_window`
- `source_observation_scope`
- `entity_anchor`
- `why_insufficient`
- `recommended_probe_type`

### 23.4.4 Budgeted Search Policy

搜索策略只能由以下因素驱动：

- 当前 primary gap 类型
- 当前时间窗覆盖状态
- 对象轨迹是否闭合
- 空间锚点是否存在
- 当前预算是否允许继续

搜索策略禁止读取：

- `choice_text`
- `best_index`
- `runner_up`
- `comparison_summary`
- `score_gap`
- `blocking_comparisons`

### 23.4.5 Evidence Acquisition

建议只保留少量通用原子动作：

- `sample_local_frames`
- `expand_time_window`
- `follow_object_track`
- `inspect_spatial_region`
- `stop_and_decide`

要求：

- 每轮只解决一个最小真实 gap
- 不允许用“看完整段视频”来掩盖搜索策略缺失

### 23.4.6 Evidence Sufficiency Judge

模型只负责判断：

- 当前证据是否足够
- 若不足，还缺哪类原始观察
- 是否值得继续搜索
- 若继续，应该优先补哪一类 gap

模型不负责：

- 比较 `top hypothesis vs runner-up`
- 从选项文本推导下一步搜索目标

### 23.4.7 Final Decision Mapper

候选答案只能在最后出现。

最终层只做：

- 将当前世界模型证据映射到候选答案
- 给出基于证据的解释
- 在证据不足时保守 withheld

禁止：

- finalizer 再把答案语义写回搜索主路径

---

## 23.5 分阶段执行清单

### 总进度总览

- [x] `Phase 0` 建立执行边界与冻结范围
- [x] `Phase 1` 清空 Finalizer 里的旧答案语义补丁
- [ ] `Phase 2` 统一 Primary Gap Schema
- [ ] `Phase 3` 统一 Planner 搜索决策，只允许 Gap 驱动
- [ ] `Phase 4` 收缩 Specialized Resolution，降级为普通搜证工具
- [x] `Phase 5` 清理 State/Trace 中会污染后续思考的答案产物
- [ ] `Phase 6` 系统性替换旧测试体系
- [ ] `Phase 7` 建立最小真实样例审计
- [ ] `Phase 8` 收口与 Goal 模式执行规范

### 当前完成度快照

- [x] 已完成阶段：`5 / 9`
- [ ] 进行中阶段：`Phase 4 / Phase 6`
- [x] 本轮继续切掉了一段 `planner` 中仍由 `latest specialized tool name` 主导 `close_call / ready_to_finish` 的 live gate：
  - 旧行为：
    - `planner._action_intent_result_is_close_call_for_recovery(...)`
      会先看 `tool_name`
      - `resolve_action_intent_pairwise`
      - `resolve_action_intent_future_use`
      再分别进入不同 close-call 判定
    - `planner._action_intent_resolution_payload_is_ready_to_finish(...)`
      还会在 `latest payload` 不为空时，
      先读上一次 specialized tool 身份，
      再决定当前 payload 能不能 finish
    - 这意味着：
      - 即使 observation-side gap 已经闭合
      - 只要 latest specialized tool name 不同
      - `close_call / ready_to_finish` 的结论仍可能变化
  - 当前变化：
    - `close_call` gate 现已改成 observation-side：
      - 只看 `primary_gap`
      - `blocker_hint`
      - `decisive_observation`
      - `direct_effect`
      - `downstream_action`
      - 当前 payload 是否真的还在表达
        - later-outcome uncertainty
        - immediate post-action uncertainty
    - `resolution_needs_more_evidence(...)`
      也已不再因为 `future_use` 这个 tool name 本身就提高门槛；
      现在只看 observation text 是否真的还缺 later-outcome evidence
    - `resolution_payload_is_ready_to_finish(...)`
      已删除 `latest specialized tool name` 前置读取，
      不再因为 `pairwise / future_use` 身份不同而改变 finish gate
  - 本轮新增/迁移测试：
    - `test_planner_action_intent_resolution_payload_ready_to_finish_is_not_driven_by_latest_specialized_tool_name`
    - 原有两条 `recovery_close_call` 测试也已改成不再传 `tool_name`
  - 本轮定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'recovery_close_call or resolution_payload_ready_to_finish or resolution_payload_not_ready_for_finish or verifier_blocked_finish_close_future_use_prefers_targeted_transition_probe or verifier_blocked_future_gap_prefers_later_outcome_recovery or verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume'`
    - `7 passed, 1121 deselected`
  - 最新主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `672 passed, 456 deselected`
- [x] 本轮还继续收掉了 `disable_legacy_specialized_recovery` 分支里一段自相矛盾的 direct specialized fallback：
  - 旧行为：
    - 在 `planner._recover_action_intent_after_verifier_blocked_finish(...)` 中，
      即使 `disable_legacy_specialized_recovery=1`
      已经打开，
      只要
      - `transition_probe`
      - `followup`
      - `extra_followup`
      都没有产出动作，
      分支尾部仍会按
      - `specialized_resume_tool == resolve_action_intent_pairwise`
      - `specialized_resume_tool == resolve_action_intent_future_use`
      直接回 specialized resolution
    - 这等于：
      - 文义上说“禁用 legacy specialized recovery”
      - 运行态却仍把 specialized tool 当最后兜底
  - 当前变化：
    - 上述 direct specialized fallback 已删除
    - 当前 `disable_legacy_specialized_recovery` 分支只允许：
      - `transition_probe`
      - `followup`
      - `extra_followup`
      - 或直接保守返回 `None`
    - 不再允许：
      - 因为历史 `pairwise/future_use` 身份仍在
      - 就重新进入旧 specialized 恢复链
  - 同步迁移测试契约：
    - `test_planner_action_intent_verifier_blocked_finish_skips_legacy_finalize_markers_when_disabled`
    - `test_planner_action_intent_verifier_blocked_disable_legacy_prefers_existing_pairwise_over_future_use_hint`
    - 现在都改成只允许 observation-first 动作，不再允许 specialized result 作为 disabled 分支的合法输出
  - 本轮定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish_skips_legacy_finalize_markers_when_disabled or verifier_blocked_disable_legacy_prefers_existing_pairwise_over_future_use_hint or verifier_blocked_legacy_disabled_explicit_downstream_object_skips_direct_specialized_resume or recovery_close_call or resolution_payload_ready_to_finish'`
    - `6 passed, 1122 deselected`
  - 最新主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `672 passed, 456 deselected`
- [x] 本轮继续收掉了 `infer_action_intent -> verifier_blocked_finish` 后半段两条 still runtime-active 的 specialized fallback：
  - 旧行为：
    - 在 `planner._recover_action_intent_after_verifier_blocked_finish(...)` 中，
      若 latest payload 来自 `infer_action_intent`，
      仍会读取
      - `action_intent_pending_resolution=*`
      - `structured_specialized_tool`
      合成 `specialized_resume_tool`
    - 然后再根据
      - `specialized_resume_tool == resolve_action_intent_pairwise`
      之类旧身份
      决定是否走 long-horizon revisit / transition probe
    - 同时函数尾部还保留了：
      - `if tool_name == resolve_action_intent_future_use -> future_use resolution`
      - `if tool_name == resolve_action_intent_pairwise -> pairwise resolution`
      这类 direct specialized fallback
    - 这意味着：
      - `verifier_blocked_finish` 后半段仍不是纯 observation-first
      - 当前恢复动作还会被历史 specialized 身份左右
  - 当前变化：
    - `infer_action_intent` 分支已删除对 `specialized_resume_tool` 的消费
    - 现在是否走
      - long-horizon revisit
      - transition probe
      - extra followup
      只由
      - `blocker_hint`
      - `primary_gap`
      - later-outcome uncertainty
      - post-action uncertainty
      决定
    - 尾部 `direct specialized fallback`
      - `future_use`
      - `pairwise`
      也已删除
    - `extra_followup` 的窗口 profile 现也改成 observation-side：
      - future gap family / later uncertainty -> `future_use`
      - 否则 -> `pairwise`
      不再直接读 latest specialized tool name
  - 同步迁移测试契约：
    - `test_planner_action_intent_verifier_blocked_infer_does_not_bootstrap_from_pending_pairwise_hint`
    - `test_planner_action_intent_verifier_blocked_infer_action_intent_prefers_observation_first_transition_recovery_over_specialized_resume`
    - `test_planner_action_intent_candidate_ranking_prefers_observation_first_local_recovery_before_long_horizon_target_query`
    - `test_planner_action_intent_verifier_blocked_finish_timeline_marker_does_not_force_later_object_revisit`
      也已同步收紧为 observation-first 输出集合
  - 本轮定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish_timeline_marker_does_not_force_later_object_revisit or verifier_blocked_infer_does_not_bootstrap_from_pending_pairwise_hint or verifier_blocked_finish_skips_legacy_finalize_markers_when_disabled or verifier_blocked_disable_legacy_prefers_existing_pairwise_over_future_use_hint or verifier_blocked_legacy_disabled_explicit_downstream_object_skips_direct_specialized_resume or verifier_blocked_infer_action_intent_prefers_observation_first_transition_recovery_over_specialized_resume'`
    - `6 passed, 1122 deselected`
  - 最新主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `672 passed, 456 deselected`
- [x] 本轮继续把 `verifier_blocked_finish(...)` 前半段一组 `future_use` 专属恢复块收成 observation-side `future profile`：
  - 旧行为：
    - 函数前半段仍有一组连续的
      - `tool_name == resolve_action_intent_future_use`
      条件块
    - 这些块决定：
      - direct-evidence 后是否直接停搜
      - `taken afterwards` 时是否继续 extra followup
      - weak cooking inspection 是否走 peak probe
      - followup exhausted 时是否强制 transition recovery
      - immediate-post-action uncertainty 是否切到 transition recovery
    - 这意味着：
      - 同样的 observation payload
      - 只因 latest specialized tool name 不同
      - 前半段恢复路径也可能不同
  - 当前变化：
    - 新增 observation-side helper：
      - `_action_intent_payload_supports_late_taken_outcome(...)`
      - `_action_intent_recovery_prefers_future_profile(...)`
    - 前半段这些 future-use 专属条件现在改成只看：
      - `primary_gap`
      - `blocker_hint`
      - later-outcome uncertainty
      - `taken afterwards` 一类更晚结果线索
      - 当前是否已有 long-horizon nodes
      - 当前 followup/coverage 是否已耗尽
    - 不再允许：
      - 因为 latest tool 恰好叫 `resolve_action_intent_future_use`
      - 就自动进入这组 future-use 专属恢复块
  - 同步迁移的测试契约：
    - 这轮没有新增专门单测名字，
      但已把既有 `future_use close call / future gap / timeline marker` 一组定向用例继续收口为 observation-first 行为断言
  - 本轮定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_finish_close_future_use_prefers_targeted_transition_probe or verifier_blocked_finish_timeline_marker_does_not_force_later_object_revisit or verifier_blocked_future_gap_prefers_later_outcome_recovery or verifier_blocked_finish_does_not_use_needed_observation_target_revisit or verifier_blocked_finish_generic_access_marker_does_not_drive_downstream_target_revisit or verifier_blocked_finish_generic_relocation_marker_does_not_drive_downstream_target_revisit'`
    - `6 passed, 1122 deselected`
  - 最新主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `672 passed, 456 deselected`
- [x] 本轮继续切掉了一组仍会让 `planner` 因 `pending_resolution / structured_specialized_tool` 直接恢复 specialized resolution 的 live runtime 链：
  - 旧行为：
    - `planner._action_intent_followup_route(...)`
      在没有新的 observation gap 时，仍会因为
      - `pending_resolution=resolve_action_intent_future_use`
      - `structured_specialized_tool=resolve_action_intent_future_use`
      - `pending_resolution=resolve_action_intent_pairwise`
      直接产出 followup route
    - `planner._build_action_intent_specialized_resolution_before_text_fallback(...)`
      在 primary-gap recovery miss 后，仍会直接恢复
      - `resolve_action_intent_future_use`
      - `resolve_action_intent_pairwise`
    - `planner._heuristic_fallback(...)`
      在 `last_tool == query_spatial_context` 时，也还会先看
      - `pending_resolution`
      - `structured_specialized_tool`
      再直接回 specialized resolution
    - `planner._recover_from_open_questions(...)`
      在 raw reuse / state candidate miss 后，仍会把
      - `pending_resolution`
      - `structured_specialized_tool`
      当成 alternative-evidence tail 的直接恢复入口
  - 当前变化：
    - `followup_route(...)` 已不再消费 `pending/structured specialized` 身份；
      followup 现在只能由
      - `primary_gap`
      - 当前 observation text 里的 uncertainty
      - 当前 direct-evidence 状态
      解释
    - `before_text_fallback(...)` 已删除 gap miss 后的 direct specialized resume；
      当前只允许
      - primary-gap recovery
      - precondition backfill
      否则直接返回 `None`
    - `query_spatial_context -> specialized resume`
      已改成 observation-first：
      - 优先保留当前空间上下文
      - 再回到 `infer_action_intent` 重判
      - 不再因为 tool identity 直接回 `future_use / pairwise`
    - `open_question recovery` tail 也已删除
      `pending/structured specialized -> direct resume`
      旧链
  - 本轮补的 observation-side 收口：
    - `later target`
    - `downstream target`
    现已被纳入通用 downstream uncertainty marker，
    因而可以在没有旧 marker 的情况下，直接把
    `later target is unclear`
    解释成 long-horizon evidence gap
  - 本轮新增/迁移测试：
    - `test_planner_action_intent_followup_route_does_not_bootstrap_from_structured_specialized_tool_without_observation_gap`
    - `test_planner_action_intent_before_text_fallback_does_not_resume_structured_specialized_tool_without_primary_gap`
    - `test_planner_action_intent_query_spatial_context_does_not_auto_resume_structured_specialized_resolution_without_pending_marker`
  - 本轮定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'followup_route_does_not_bootstrap_from_structured_specialized_tool_without_observation_gap or before_text_fallback_does_not_resume_structured_specialized_tool_without_primary_gap or query_spatial_context_does_not_auto_resume_structured_specialized_resolution_without_pending_marker or heuristic_fallback_does_not_resume_specialized_resolution_from_raw_observation_tool_alone or next_action_does_not_resume_specialized_resolution_from_raw_observation_tool_alone or before_text_fallback_explicit_downstream_object_skips_direct_specialized_resume_when_gap_route_unavailable or before_text_fallback_immediate_outcome_gap_skips_specialized_resume_when_gap_route_unavailable or open_question_recovery_blocked_state_change_explicit_downstream_object_skips_specialized_resume or open_question_recovery_suppressed_answer_conditioned_target_prefers_observation_centric_followup_before_specialized_resume'`
    - `9 passed, 1118 deselected`
  - 最新主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `671 passed, 456 deselected`
- [x] 本轮继续切断了一条 still answer-conditioned 的 `candidate_evidence -> mixed_horizon finalizer producer` 旧链：
  - 旧行为：
    - `graph_agent._action_intent_resolution_should_withhold_mixed_horizon_overclaim(...)`
      会把 `candidate_evidence[*].support/contradiction`
      直接拼进 `text`
    - `graph_agent._action_intent_resolution_should_withhold_mixed_horizon_later_target_overclaim(...)`
      不仅会继续读取 `candidate_evidence`
      还会调用旧的
      - `_action_intent_later_outcome_target_token_and_kind(...)`
      - `_action_intent_choice_target_token_and_kind(...)`
      从 `choice/categories` 里反推出 later target
    - 这本质上仍是：
      - 候选答案竞争文本
      - 回流到 finalizer withheld producer
      - 再间接影响 planner 的后续恢复分支
  - 当前变化：
    - 已删除：
      - `_action_intent_later_outcome_target_token_and_kind(...)`
      - `_action_intent_choice_target_token_and_kind(...)`
    - 新增：
      - `_action_intent_resolution_observation_text(...)`
    - 当前 `mixed_horizon` 两条 producer 只允许读取 observation-side 字段：
      - `reason`
      - `decisive_observation`
      - `direct_effect`
      - `downstream_action`
      - `timeline_summary`
      - `next_use_evidence`
      - `ambiguity_note`
    - 不再允许：
      - `candidate_evidence` 单独制造 later uncertainty
      - `choice/categories` 单独反推出 `fridge/scale/sink/plate` 这类 later target
  - 新增负约束测试：
    - `test_graph_agent_action_intent_mixed_horizon_overclaim_ignores_candidate_evidence_without_observation_uncertainty`
    - `test_graph_agent_action_intent_mixed_horizon_later_target_overclaim_ignores_candidate_evidence_without_observation_gap`
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'mixed_horizon_overclaim_ignores_candidate_evidence_without_observation_uncertainty or mixed_horizon_later_target_overclaim_ignores_candidate_evidence_without_observation_gap or action_intent_finalizer_marks_later_fixture_target_for_mixed_horizon_open_vs_weigh or action_intent_finalizer_marks_later_fixture_target_for_mixed_horizon_label_vs_put_back or action_intent_finalizer_marks_later_object_target_for_mixed_horizon_open_vs_serve'`
    - 结果：`5 passed, 1116 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`665 passed, 456 deselected`
- [x] 本轮继续切掉了一条 still runtime-active 的 `finalizer marker -> planner long-horizon bias` 旧链：
  - 旧行为：
    - `planner._action_intent_prefers_long_horizon_object_retrieval(...)`
      和
      - `planner._action_intent_long_horizon_prefers_latest_candidate(...)`
      会直接读取
      - `action_intent_resolution_withheld_for_nonexclusive_concrete_late_anchor=1`
      - `action_intent_resolution_withheld_for_timeline_review_bias_gap=1`
      - `action_intent_resolution_withheld_for_workspace_or_final_placement_claim=1`
    - `planner._build_action_intent_finalize_withheld_long_horizon_revisit_decision(...)`
      也会以这些 finalizer marker 作为 long-horizon revisit 的入口条件
    - 这本质上仍是：
      - finalizer 的 withheld marker
      - 直接塑造 planner 的 long-horizon 偏好与 revisit 开关
  - 当前变化：
    - 已删除：
      - `_action_intent_recent_later_outcome_finalize_withheld_marker(...)`
      - `prefers_long_horizon_object_retrieval / long_horizon_prefers_latest_candidate`
        对旧 finalizer marker 的 shortcut 读取
    - `finalize_withheld_long_horizon_revisit_decision(...)`
      现在只看 observation-driven 的 long-horizon 条件，
      不再因为旧 marker 存在就直接打开 revisit
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'long_horizon_object_retrieval_does_not_bootstrap_from_finalize_withheld_marker_alone or long_horizon_latest_candidate_does_not_bootstrap_from_finalize_withheld_marker_alone or nonexclusive_concrete_late_anchor_withheld_marker_does_not_force_later_node_revisit or textual_fallback_nonexclusive_concrete_late_anchor_does_not_force_later_node_revisit'`
    - 结果：`4 passed, 1115 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`663 passed, 456 deselected`
- [x] 本轮继续切掉了一条 still runtime-active 的 `finalizer marker -> planner close-call` 旧链：
  - 旧行为：
    - `planner._action_intent_result_is_workspace_or_final_placement_close_call(...)`
      会先读取
      - `action_intent_resolution_withheld_for_workspace_or_final_placement_claim=1`
      这个 finalizer marker
    - 只要 marker 在 `working_memory` 里，planner 就直接把当前 why 结果当成
      `workspace/final-placement close call`
    - 这本质上仍是：
      - finalizer 的答案语义 marker
      - 回流到 planner 决定下一步动作
  - 当前变化：
    - 已删除：
      - `_action_intent_recent_workspace_or_final_placement_withheld(...)`
      - 以及 `_action_intent_result_is_workspace_or_final_placement_close_call(...)`
        对该 marker 的 shortcut 读取
    - 现在这类 `transition probe` / close-call
      只能由当前 result 里的 observation text 自己触发，
      不再靠旧 finalizer marker 直接拉起
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'future_use_weak_exact_sink_slot_prefers_transition_probe_before_finish or pairwise_weak_right_place_prefers_transition_probe_before_finish'`
    - 结果：`2 passed, 1115 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续收缩了 `graph_agent.py` 里另一条 still answer-conditioned 的 finalizer producer：
  - 旧行为：
    - `_action_intent_resolution_should_withhold_generic_relocation_or_storage_overclaim(...)`
      会扫描 `candidate_evidence` 里的 competitor row，
      再根据 competitor choice 的
      - same-object active use
      - hidden-target retrieval
      - exact downstream placement
      等精确语义决定当前 generic put-away / storage 是否 withheld
    - 这本质上仍是：
      - `best choice vs competitor choice`
      - 驱动 finalizer withheld marker
  - 当前变化：
    - 该 gate 已改成只读：
      - 当前 resolution observation text
      - 当前 gap state
      - 当前 timeline-review / final-location 未闭合
    - 不再读取 competitor `candidate_evidence` 的 choice/support/contradiction
    - generic put-away / generic put-back / generic store
      现在只有在 observation text 已出现明确 storage/return chain 时才允许放行；
      若仍只是 move-away / leave-area / set-aside / same-object reuse / later target unresolved，
      就统一保守 withheld
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'generic_put_away_for_revealed_downstream_target or verifier_blocked_finish_generic_relocation_marker_does_not_drive_downstream_target_revisit or textual_fallback_generic_relocation_marker_prefers_downstream_target or generic_relocation_withheld_marker_revisits_real_downstream_target'`
    - 结果：`4 passed, 1113 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续收缩了 `graph_agent.py` 里一条 still answer-conditioned 的 finalizer producer：
  - 旧行为：
    - `_action_intent_resolution_should_withhold_generic_access_or_space_overclaim(...)`
      会扫描 `candidate_evidence` 里的 competitor row，
      再根据 competitor choice 的精确语义
      判断当前 generic access / generic space 是否应该 withheld
    - 这本质上仍是：
      - `best choice vs competitor choice`
      - 驱动 finalizer withheld marker
  - 当前变化：
    - 该 gate 已改成只读：
      - 当前 resolution observation text
      - 当前 gap state
      - 当前 timeline-review / direct-observation 不足
    - 不再读取 competitor `candidate_evidence` 的 choice/support/contradiction
    - 因而 `generic access / generic make-space` 的 finalizer withheld
      已从 candidate-conditioned 逻辑收缩为 observation-conditioned 逻辑
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'generic_access_when_specific_revealed_target_exists or generic_access_withheld_marker_revisits_real_revealed_target or verifier_blocked_finish_generic_access_marker_does_not_drive_downstream_target_revisit'`
    - 结果：`3 passed, 1114 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续收口了 `first followup short window -> premature transition/rank` 这条 planner 旧链：
  - 旧行为：
    - 首轮 `followup` 之后，
      即使当前 observation 只覆盖到很短的 post-action 窗口，
      且还没有直接的 post-action evidence，
      planner 仍可能过早落到：
      - `extract_frames_for_range`
      - 或 `rank_choices_from_state`
  - 当前变化：
    - 新增：
      - `_action_intent_post_action_followup_window_is_short(...)`
      - `_action_intent_first_followup_needs_more_observation_coverage(...)`
    - 现在这条 gate 只看：
      - 当前是否正处于第一轮 followup 之后
      - 当前 post-action 覆盖窗是否仍短于最小观察窗
      - 是否已经存在 direct post-action evidence
      - 是否已经做过 transition / peak / timeline-review followup
    - 若 observation 仍未闭合，
      planner 会优先继续 `sample_sparse_frames` 扩窗，
      而不是被 `transition probe` 或 `rank` 提前收口
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'test_planner_action_intent_high_confidence_outcome_pair_extends_followup_when_post_action_window_is_still_short or test_planner_action_intent_pairwise_extends_followup_when_post_action_coverage_is_still_short or test_planner_action_intent_pairwise_short_post_action_window_requests_more_followup_without_choice_semantics'`
    - 结果：`3 passed, 1114 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续切掉了 `timeline_review -> resolution_mode -> specialized resume` 的旧恢复链，并同步迁移相关旧测试契约：
  - 旧行为：
    - `_resume_action_intent_specialized_resolution_from_timeline_review(...)`
      在：
      - `timeline_review_result.needs_more_evidence == True`
      且本地 observation-first 恢复没有立即产出动作时，
      仍会按
      - `resolution_mode == resolve_action_intent_future_use`
      - `resolution_mode == resolve_action_intent_pairwise`
      直接恢复 specialized resolver
    - 在 timeline review 后已有当前动作+followup 原始帧的情况下，
      也仍会因为 `resolution_mode`
      走 specialized resume，而不是先做 observation-first re-infer
  - 当前变化：
    - 已删除这两段 `resolution_mode -> specialized resume` fallback
    - timeline review 之后现在只允许：
      - primary gap routed recovery
      - cached long-horizon revisit
      - transition probe
      - extra followup
      - 基于当前动作+followup 原始帧的 `infer_action_intent` 重判
    - 当没有可用 observation frames 时，直接返回 `None`
      而不是自动恢复 `future_use / pairwise`
  - 一并清理的相关残余：
    - `needs_future_use_evidence` 内部补上 `allow_resolution_markers=False`，消除递归
    - `candidate_inference_frames` 改为可由 `primary_gap`
      的 `future_outcome / relation_confirmation / target_discovery`
      直接决定纳入 followup frames
    - 一批旧测试从
      “必须恢复 specialized / 必须按 marker 查询某对象/空间”
      改为
      “必须保持 observation-first / 不允许 direct specialized overreach”
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'timeline_review_uses_current_review_payload_for_observation_first_reinfer_without_successful_intent_result or timeline_review_close_call_primary_gap_does_not_force_specialized_resolution_after_local_recovery_miss or timeline_review_close_call_primary_gap_prefers_cached_revisit_over_object_query or timeline_review_next_use_ambiguity_keeps_extra_followup_when_transition_probe_is_not_applicable or timeline_review_reinfer_keeps_followup_frames'`
    - 结果：`5 passed, 1111 deselected`
  - observation-first 契约迁移回归：
    - 一组 timeline/fallback/segment 相关定向用例
    - 结果：`9 passed, 1107 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`660 passed, 456 deselected`
- [x] 本轮继续切掉了 `followup_artifacts -> resolution_mode -> specialized resume` 的旧恢复链：
  - 旧行为：
    - `_resume_action_intent_specialized_resolution_from_followup_artifacts(...)`
      只要拿到
      - `sample_sparse_frames / extract_frames_for_range / sample_frames_around_peaks / retrieve_cached_artifacts`
        产生的 followup artifacts
      且 `resolution_mode` 仍存在，
      就会直接恢复：
      - `resolve_action_intent_future_use`
      - `resolve_action_intent_pairwise`
    - 这意味着：
      - 一旦 extra followup / peak frames 回来，
      - planner 仍可能绕过 observation-first re-infer，
      - 直接按 pending specialized identity 收口
  - 当前变化：
    - 已删除该函数里对
      - `resolution_mode == resolve_action_intent_future_use`
      - `resolution_mode == resolve_action_intent_pairwise`
      的直接 specialized resume
    - 该函数现在只保留：
      - peak probe after transition
      - precondition backfill
      - 否则返回 `None`
    - 实际运行路径因此退回到：
      - extra followup / peak frames 进入后
      - 优先 `infer_action_intent` observation-first 重判
      - 而不是立刻跳 specialized resolver
  - 同步迁移旧测试契约：
    - `followup_frames_without_pending_marker_do_not_auto_resume_structured_specialized_resolution`
    - `followup_frames_with_pending_marker_do_not_auto_resume_specialized_resolution_without_observation_recovery`
    - `peak_frames_feed_back_into_observation_first_reinfer`
    - `pending_future_use_reuses_extra_frames_for_observation_first_reinfer`
    - `context_notes_drop_restored_model_conclusions_in_observation_first_reinfer`
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'followup_frames_without_pending_marker_do_not_auto_resume_structured_specialized_resolution or followup_frames_with_pending_marker_do_not_auto_resume_specialized_resolution_without_observation_recovery or peak_frames_feed_back_into_observation_first_reinfer or pending_future_use_reuses_extra_frames_for_observation_first_reinfer or context_notes_drop_restored_model_conclusions_in_observation_first_reinfer'`
    - 结果：`5 passed, 1114 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续切掉了两类更深的 option-semantic / answer-conditioned 路由残余：
  - 旧行为 1：
    - `_action_intent_timeline_review_resolver_hint(...)`
      会根据：
      - timeline review 文本里的固定词表
      - 以及 `selected_choice_categories(...)`
      共同决定当前 resolver 是
      - `future_use`
      - 还是 `pairwise`
    - 这本质上仍是：
      - timeline text + choice semantics
      - 共同塑造搜索路由
  - 当前变化 1：
    - 删除了这套基于文本 marker 和 choice category 的 resolver 判定
    - 现在 `timeline_review_resolver_hint(...)`
      只看 `primary_gap.gap_type`
      - `future_outcome / relation_confirmation / target_discovery -> future_use`
      - `immediate_outcome / state_transition_unconfirmed / workspace_change_unconfirmed -> pairwise`
  - 旧行为 2：
    - `_action_intent_future_use_candidate_indices(...)`
      和 `_action_intent_pairwise_candidate_indices(...)`
      仍会依赖：
      - timeline review 文本
      - `selected_choice_categories(...)`
      - 以及某些 safety / hand-free / make-space 语义补丁
      对 specialized resolution 的 candidate set 做语义收缩或注入
  - 当前变化 2：
    - `future_use_candidate_indices(...)`
      已改为：
      - 不再从 `best_index / second_best / pending_candidates / timeline semantics`
        收缩候选
      - 默认直接返回全候选集合
    - `pairwise_candidate_indices(...)`
      已删除：
      - timeline semantic shrink
      - safety candidate 注入
      - 基于 choice category 的 pair 收缩
      - 当前仅保留已有 observation-ranked / 原始 candidate ordering
  - 同步迁移旧测试契约：
    - timeline semantic shrink / hazard injection / fullset future-use candidate 等相关测试
    - 统一改为：
      - 不再要求因选项语义而缩小候选
      - 不再要求 timeline 文本自动注入 safety candidate
      - specialized resolution 的 candidate 集合不再由答案产物塑造
  - 定向回归：
    - timeline/future-use/pairwise candidate 相关定向：
      - `6 passed, 1111 deselected`
    - future-use fullset / no safety injection / primary-gap routed recovery 相关定向：
      - `7 passed, 1110 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续切掉了 `semantic_rescue_candidate_indices(...)` 这层 question/choice 语义补丁：
  - 旧行为：
    - `_action_intent_semantic_rescue_candidate_indices(...)`
      会根据：
      - question 动作词
      - towel / cloth / tap-scale 等固定语义
      - choice category 缺口
      直接把某些候选重新“拉回” top pair / candidate set
    - 这本质上是典型的：
      - 固定冲突模式
      - question semantics
      - choice semantics
      驱动的专项补丁层
  - 当前变化：
    - 该函数已降级为 no-op
    - 现在只做：
      - 下标合法化
      - 去重
    - 不再根据 question/choice 语义把任何候选重新注入到 why 搜索主路径
  - 同步迁移旧测试契约：
    - `semantic_rescue_no_longer_brings_back_residue_release_candidate`
    - `semantic_rescue_no_longer_brings_back_measurement_candidate_for_tap_scale`
  - 定向回归：
    - `2 passed, 1115 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续切掉了 `pending_candidates` 在 planner 侧的消费链：
  - 旧行为：
    - planner 会从 `working_memory` 里的
      - `action_intent_pending_candidates=[...]`
      重新读取候选集合
    - 并在：
      - `_action_intent_pending_candidate_indices(...)`
      - `_fallback_action_intent_pairwise_candidate_indices(...)`
      中继续用这些候选答案产物塑造 why specialized resolution
    - 同时 fallback pairwise 还会继续叠加 question/choice 语义来重建 pair
  - 当前变化：
    - `_action_intent_pending_candidate_indices(...)` 已返回空
    - `_fallback_action_intent_pairwise_candidate_indices(...)` 已不再读取 `pending_candidates`
      也不再基于 question/choice 语义重建 pair
    - 当前 fallback pairwise 直接回到全候选集合
  - 影响：
    - executor 目前仍可能为了 trace/debug 写回 `action_intent_pending_candidates=[...]`
    - 但 planner 已经不再把这些答案产物当作 why 搜索主路径输入
  - 同步迁移旧测试契约：
    - `fallback_action_intent_pairwise_candidates_no_longer_bootstrap_from_structured_hypotheses_without_legacy_pending_candidates`
    - 以及 pending-candidate 相关定向约束
  - 定向回归：
    - `3 passed, 1114 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续切掉了 `best_index / candidate_indices` 回流到 planner 的残余链：
  - 旧行为 1：
    - `_latest_action_intent_candidate_indices(...)`
      会从：
      - `result.candidate_indices`
      - `result.best_index`
      - `result.candidate_evidence[*].index`
      - 最新 trace 的 raw result / failed specialized args
      重新拼出一组 candidate indices
    - 然后这组 candidate indices 会继续流回：
      - `followup_route`
      - `pairwise_needs_outcome_resolution`
      - `result_needs_generalized_disambiguation`
      - specialized resolution candidate shaping
    - 本质上仍是答案层产物回流到 why 搜索主路径
  - 当前变化 1：
    - `_latest_action_intent_candidate_indices(...)` 已返回空
    - planner 不再从 `best_index / candidate_indices / candidate_evidence`
      把候选集合读回 why 搜索阶段
  - 旧行为 2：
    - `action_intent_followup_decision(...)`
      里的 `hidden_access_exact_use_pairwise_needed`
      会在 full candidate set 下也强行把路由拉回 pairwise
    - 这会让多候选场景里仍受到 choice semantics 冲突模式主导
  - 当前变化 2：
    - `hidden_access_exact_use_pairwise_needed`
      已收紧到只在 `candidate_count <= 2` 时触发
    - 多候选场景下不再因为该固定冲突模式压过 future-use / fullset 搜证
  - 同步迁移旧测试契约：
    - latest-candidate / pairwise-safety / structured-comparison-pair / top2-future-use 等相关测试
    - 改为：
      - latest candidate 不再从答案产物回流
      - pairwise candidate 不再保持旧 top2
      - fullset future-use / observation-first 路由优先
  - 定向回归：
    - `6 passed, 1111 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续切掉了 `best_choice vs competitors` 的 choice-category 判定链：
  - 旧行为：
    - `_action_intent_best_choice_is_broad_relative_to_competitors(...)`
      会根据：
      - `selected_choice_categories(...)`
      - broad category 集
      - broad choice marker 文本
      判断当前 best choice 是否“过泛”
    - 然后该判定会继续影响：
      - `_action_intent_direct_evidence_still_needs_resolution(...)`
      - `_action_intent_result_needs_generalized_disambiguation(...)`
    - 本质上仍是：
      - 通过选项语义判断要不要继续 why 搜证
  - 当前变化：
    - `_action_intent_best_choice_is_broad_relative_to_competitors(...)` 已降级为恒 `False`
    - `direct_evidence_still_needs_resolution / generalized_disambiguation`
      不再依赖 choice category / broad choice marker 决定是否继续搜证
    - 继续搜索现在回到：
      - observation gap
      - post-action grounding
      - confidence / budget
      的组合判定
  - 定向回归：
    - broad-choice / repeated-vision-failure / unresolved-gap 相关定向：
      - `6 passed, 1111 deselected`
  - 主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - 结果：`661 passed, 456 deselected`
- [x] 本轮继续切掉了一段 `planner.py` 中由 latest specialized tool identity 影响当前排序的惯性链：
  - 旧行为 1：
    - `textual_rank` 分支里，
      当 `candidate_plan.decision.tool == resolve_action_intent_pairwise`
      且 `latest_resolution_tool == resolve_action_intent_pairwise` 时，
      仍会写入：
      - `planner_guard=textual_rank_prefers_existing_pairwise_resolution_over_generic_visual_review`
    - 这虽然已不再直接改变大逻辑，但本质上仍把
      “上一轮用了哪个 specialized 工具”
      带进当前决策语义与 trace
  - 旧行为 2：
    - `state_candidate` 排序里，
      若当前存在 `has_existing_specialized_chain`
      则对
      `resolve_action_intent_pairwise / resolve_action_intent_future_use`
      的 gain 惩罚会更轻
    - 这会让 specialized tool 因“已有历史链路”而在排序上获得额外惯性优势
  - 当前变化：
    - 删除了 `textual_rank_prefers_existing_pairwise_resolution_over_generic_visual_review` 这条 guard 写入
    - 删除了 `has_existing_specialized_chain` 对 specialized candidate gain 惩罚的放宽
    - 结果是：
      - textual-rank 与 state-candidate 排序不再读取“上一轮 specialized tool 身份”来塑造当前偏好
      - specialized candidate 的排序只服从当前 gap/budget/targeted evidence，而不是历史 identity
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'textual_rank_prefers_existing_pairwise_resolution_over_generic_visual_review or action_intent_textual_rank_evidence_first_explicit_downstream_object_prefers_gap_over_existing_pairwise_resolution or state_candidate_prefers_object_revisit_when_structured_gap_synthesizes_future_use or state_candidate_keeps_targeted_gap_tool_for_future_outcome_even_when_step_mentions_post_action_evidence or explicit_level_zero_budget_prefers_local_followup_over_specialized'`
    - 结果：`4 passed, 1112 deselected`
  - 宽 planner 回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent and (textual_rank or state_candidate or low_conf_finish)'`
    - 结果：`28 passed, 1088 deselected`
- [x] 本轮继续切掉了一条 `planner.py` 中由 structured gap residue 自动合成 pending specialized tool 的旧链：
  - 旧行为：
    - `_action_intent_pending_resolution_tool(...)`
      会进入 `_action_intent_resolution_mode(..., include_memory_marker=True)`
    - 在没有：
      - 最新成功 observation-grounded payload
      - 显式 `action_intent_pending_resolution=...` marker
      的情况下，
      仍可能仅凭：
      - `sufficiency_decision.missing_gap_types`
      - 或 `primary_gap.source == sufficiency_missing_gap_types`
      自动产出：
      - `resolve_action_intent_future_use`
      - `resolve_action_intent_pairwise`
    - 这会让 heuristic/open-question 路径保留一条
      “structured residue -> specialized tool identity”
      的旧主链
  - 当前变化：
    - `_action_intent_resolution_mode(...)`
      已删除对纯 `missing_gap_types` fallback 的直接 specialized tool 合成
    - 同时把最早的
      `gap_type == future_outcome -> resolve_action_intent_future_use`
      条件收紧为：
      - 若 `primary_gap.source == sufficiency_missing_gap_types`
        则不再直接产出 pending specialized tool
    - 结果是：
      - observation-grounded evidence gap 仍然可以驱动 pending resolution tool
      - 显式 pending marker 仍然保留
      - 但纯 structured residue 不再自动编码成 specialized tool identity
  - 同步迁移旧测试契约：
    - `test_planner_action_intent_pending_resolution_tool_no_longer_bootstraps_from_structured_missing_gap_types_without_payload_or_marker`
    - `test_planner_action_intent_peak_frames_with_structured_close_call_without_success_payload_stays_observation_first`
  - 保留的正向约束：
    - `test_planner_action_intent_pending_resolution_tool_can_be_driven_by_observation_grounded_gap`
    - `test_planner_action_intent_pending_resolution_tool_does_not_start_from_structured_hypotheses_alone`
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'pending_resolution_tool_can_be_driven_by_observation_grounded_gap or pending_resolution_tool_no_longer_bootstraps_from_structured_missing_gap_types_without_payload_or_marker or pending_resolution_tool_does_not_start_from_structured_hypotheses_alone or peak_frames_with_structured_close_call_without_success_payload_stays_observation_first'`
    - 结果：`4 passed, 1112 deselected`
  - 宽 planner 回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent and (open_question_recovery or heuristic_fallback or pending_resolution_tool or structured_specialized_recovery_tool)'`
    - 结果：`28 passed, 1088 deselected`
- [x] 本轮继续切掉了一条 `planner.py` 中 specialized current-scope fallback 先于 observation candidate 的旧顺序链：
  - 旧行为：
    - `_build_action_intent_resolution_not_ready_recovery(...)`
      会先调用
      `_build_action_intent_specialized_recovery_decision(...)`
    - 只有 specialized current-scope fallback 失败后，
      才会尝试 `best_state_candidate_plan(...)`
    - 这意味着：
      - 即使当前已经有更具体的 observation-side targeted candidate
      - planner 仍可能先退回 generic current-scope resampling / infer_action_intent current-scope fallback
  - 当前变化：
    - `_build_action_intent_resolution_not_ready_recovery(...)`
      已改为：
      1. 先尝试 `best_state_candidate_plan(...)`
      2. 若没有可执行 targeted candidate，再退到 `_build_action_intent_specialized_recovery_decision(...)`
      3. 最后才退 generic resample
    - 这让 `state candidate / observation-targeted probe`
      明确优先于
      `specialized current-scope fallback`
  - 新增测试：
    - `test_planner_action_intent_resolution_not_ready_recovery_prefers_state_candidate_before_specialized_current_scope_fallback`
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'resolution_not_ready_recovery_prefers_state_candidate or pairwise_resolution_continue_search_prefers_state_candidate_when_open_question_recovery_has_no_action or action_intent and resolution_not_ready_recovery'`
    - 结果：`3 passed, 1113 deselected`
  - 宽 planner 回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent and (open_question_recovery or low_conf_finish or textual_rank or resolution_not_ready_recovery)'`
    - 结果：`33 passed, 1083 deselected`
- [x] 本轮继续切掉了一段 `planner.py` 中由 structured sufficiency 直接自举 specialized tool identity 的旧链：
  - 旧行为：
    - `_action_intent_structured_specialized_recovery_tool(...)`
      只要 `should_continue_search_from_sufficiency(state)` 为真，
      就可能仅凭 structured verification gap
      直接合成：
      - `resolve_action_intent_future_use`
      - `resolve_action_intent_pairwise`
    - 这会让 `open_question / recovery / before-text-fallback`
      在“只有 structured gap、但没有当前 observation-grounded resolution payload”时，
      仍然沿 specialized identity 继续运行
  - 当前变化：
    - 若没有：
      - 最新成功的 observation-grounded action-intent payload
      - 且也没有显式 `action_intent_pending_resolution=...` marker
    - 那么 `_action_intent_structured_specialized_recovery_tool(...)` 直接返回空
    - 这意味着：
      - structured sufficiency gap 本身不再自动升级成 specialized tool 身份
      - open-question recovery 必须先经过：
        - primary gap recovery
        - raw evidence reuse / resample
        - state candidate / local followup
      - specialized resolver 更明确地退为后备工具，而不是 gap 的直接编码形式
  - 同步迁移旧测试契约：
    - `test_planner_action_intent_structured_sufficiency_without_success_payload_no_longer_bootstraps_specialized_recovery_tool`
    - `test_planner_action_intent_structured_future_outcome_gap_without_success_payload_prefers_observation_recovery_over_specialized_resume`
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'structured_specialized_recovery_tool or structured_sufficiency_without_success_payload_no_longer_bootstraps_specialized_recovery_tool or structured_future_outcome_gap_without_success_payload_prefers_observation_recovery_over_specialized_resume or prefers_specialized_open_question_recovery_from_observation_grounded_relation_confirmation_gap_without_payload_or_hypotheses or structured_specialized_recovery_tool_does_not_bootstrap_pairwise_from_structured_hypotheses_alone'`
    - 结果：`4 passed, 1111 deselected`
  - 宽 planner 回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent and (open_question_recovery or heuristic_fallback or before_text_fallback or structured_specialized_recovery_tool or pending_resolution_tool)'`
    - 结果：`32 passed, 1083 deselected`
- [x] 本轮继续切掉了一段 `verifier.py` 中仍把 specialized tool identity 当成稳定性主判据的旧链：
  - `_action_intent_has_successful_specialized_resolution(...)`
    之前只有
    `resolve_action_intent_pairwise / resolve_action_intent_future_use`
    才可能被视为“成功解析”
  - 当前已改为：
    - 不再看 tool name
    - 只看当前 resolution payload 是否满足 observation-grounded 条件：
      - 有 `best_index`
      - 没有 `need_more_evidence / need_future_evidence`
      - 不缺 direct post-action evidence
      - 不包含 later-outcome uncertainty
    - 这意味着：
      - `infer_action_intent` 等普通 resolution payload 只要 observation 已闭合，也可以稳定 why verifier
      - `specialized resolver` 工具身份进一步退化为兼容性实现，而不是 sufficiency 凭证
  - 同步新增正向测试：
    - `test_verifier_action_intent_observation_grounded_resolution_can_stabilize_without_specialized_tool_name`
  - 同步保留负向测试：
    - `test_verifier_action_intent_unresolved_rerank_does_not_count_as_successful_specialized_resolution`

- [x] 本轮还补上了一条通用的 observation-side finish gate，而不是题型专项规则：
  - `verifier._action_intent_textual_rank_fallback_can_finish(...)`
    现在不仅检查：
    - 当前是否有 textual fallback answer
    - 是否有当前任务 artifact grounding
    - 是否已有 pre/post-action grounding
  - 还会检查：
    - 当前 `state/evidence` 是否自己明确描述了
      “post-action direct outcome 仍未闭合 / still unresolved / still not explicit”
  - 新 helper：
    - `_action_intent_state_describes_unclosed_post_action_outcome(...)`
  - 这条约束是通用的 observation 约束，不是 residue / sink / towel 等固定题型特判
  - 它防止 textual fallback 在“当前观测文本已经说明 direct outcome 还不明确”的情况下过早 finish

- [x] 本轮回归结果：
  - 定向 verifier 子集：
    - `pytest -q tests/test_graph_agent.py -k 'verifier_action_intent and (accepts_ranked_best_index_after_repeated_vision_failures or textual_fallback_with_current_task_artifacts_and_grounding_can_finish or textual_fallback_without_open_needed_observation_payload_can_finish or textual_fallback_open_needed_observation_payload_alone_does_not_block_finish or textual_fallback_keeps_strict_residue_release_bucket_blocking or unresolved_rerank_does_not_count_as_successful_specialized_resolution or observation_grounded_resolution_can_stabilize_without_specialized_tool_name)'`
    - 结果：`7 passed, 1108 deselected`
  - 宽回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent and (verifier or heuristic_fallback or open_question_recovery or same_object_primary_gap or unresolved_rerank)'`
    - 结果：`227 passed, 888 deselected`
- [x] 本轮继续切断一条真实存在的 `graph_agent/planner unresolved rerank -> downstream target routing` 旧链：
  - `planner.py`
    - 以下 unresolved-rerank consumer 已全部降级为空实现，不再参与 runtime 搜索路由：
      - `_action_intent_recent_unresolved_rerank_withheld_reason(...)`
      - `_action_intent_unresolved_rerank_reason_prefers_later_outcome_revisit(...)`
      - `_action_intent_unresolved_rerank_downstream_object_hint(...)`
      - `_action_intent_unresolved_rerank_downstream_fixture_hint(...)`
      - `_build_action_intent_unresolved_rerank_long_horizon_revisit_decision(...)`
      - `_build_action_intent_unresolved_rerank_downstream_target_revisit_decision(...)`
      - `_build_action_intent_unresolved_rerank_downstream_fixture_revisit_decision(...)`
    - 当前 unresolved rerank 只剩“证据仍不足”的痕迹，不再根据 `reason=` 或候选竞争关系反推出：
      - 该查哪个 object
      - 该查哪个 fixture
      - 该不该继续去更晚时间点追某个 target
  - `tests/test_graph_agent.py`
    - 一组旧测试已改写为负约束：
      - 不再要求 timeline/slot/revealed-target/fixture gap 必须按 choice 语义追某个下游对象
      - 改为要求 planner 不会因为 unresolved-rerank marker 而直接查询 `fridge/cup/jar/scale`
    - 一条旧 open-question recovery 契约也已迁移：
      - 不再要求 `future_use` open question 必须回到 specialized resolver
      - 改为允许 observation-first 的局部 resampling
  - 本轮定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'unresolved_rerank and (timeline_gap_no_longer_queries_choice_fixture or slot_gap_no_longer_queries_choice_object or revealed_target_gap_no_longer_queries_choice_object or fixture_gap_no_longer_revisits_choice_fixture_node or mixed_horizon_no_longer_revisits or downstream_fixture_hint_does_not_use_choice_text_without_observed_fixture)'`
    - 结果：`9 passed, 1105 deselected`
  - 本轮扩大回归第一轮结果：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent and (unresolved_rerank or heuristic_fallback or open_question_recovery or enforce_task_requirements or verifier_blocked or same_object_primary_gap)'`
    - 初次结果：`198 passed, 1 failed, 915 deselected`
    - 唯一失败项是旧 specialized 契约测试，已按 observation-first 方向改写，待再次回归确认
- [x] 本轮继续切断一条 still answer-conditioned 的 `finalizer -> planner` 旧链：
  - `graph_agent.py`
    - why/action-intent finalizer 不再产出以下 target-bearing marker：
      - `action_intent_resolution_withheld_for_generic_access_or_space_enablement=1 target=...`
      - `action_intent_resolution_withheld_for_generic_relocation_or_storage_enablement=1 target=...`
      - `action_intent_resolution_withheld_for_mixed_horizon_later_target=1 target=...`
    - 同时删除/降级了：
      - `_action_intent_resolution_weak_cooking_inspection_needed_observation(...)`
      - `_action_intent_resolution_generic_access_or_space_overclaim_marker(...)`
      - `_action_intent_resolution_generic_relocation_or_storage_overclaim_marker(...)`
      - `_action_intent_resolution_mixed_horizon_later_target_marker(...)`
    - 当前 finalizer 只保留更抽象的 withheld 状态：
      - `generic_access_or_space_overclaim`
      - `generic_relocation_or_storage_overclaim`
      - `mixed_horizon_overclaim`
      - `weak_cooking_inspection_evidence`
  - `planner.py`
    - 不再把上述 finalize marker 当成“下游目标提示”继续恢复 `query_object/query_spatial_context`
    - 旧的 finalize-marker hint consumer 已被降级为空或退出主链
  - 这意味着：
    - why finalizer 不再从候选竞争里反推“下一步该查哪个 target”
    - planner 也不再消费 finalizer 写入的 target marker 继续搜证
    - 这条 `candidate semantics -> target marker -> downstream revisit` 链已经从 runtime 主路径上切掉
- [x] 当前这一轮已收口：
  - `open-question recovery` 主路径已改成 `primary_gap / evidence-first / local followup` 优先
  - `same-object future_outcome` 不再误转成 `query_object(self)`，而是保留已建立的 `pairwise` 主链
  - `suppressed_answer_conditioned_target` 不再回退到 `future_use/pairwise` specialized resume，而是退回 observation-centric followup
- [x] 本轮 `graph_agent.py` why finalizer 又切掉了一批局部 `competitor pair / second_best / losing_index` 依赖：
  - `mixed_horizon_overclaim`
  - `mixed_horizon_later_target_overclaim`
  - `nonexclusive_concrete_late_anchor_claim`
  - `workspace_or_final_placement_overclaim`
  - `timeline_review_bias_gap`
  这些 gate 现在优先读取：
  - 当前 best choice 是否已有直接证据
  - 当前 timeline review 是否仍未闭合
  - 当前 verification/primary gap 是否仍存在 `future_outcome / relation_confirmation / target_discovery`
  - 当前文本是否明确说明“不是最终归位 / 还看不出后续用途 / 只看到宽泛空间变化”
  而不再通过 `best vs runner-up` 的竞争关系决定是否 withheld
- [x] 对应本轮 why/action-intent finalize 回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (finalizer or verifier_blocked or before_text_fallback or pending_resolution_tool or structured_specialized_recovery_tool or finalize_action_intent or trace_best_index or structured_best_index or does_not_build_action_intent_hypotheses or discards_action_intent_hypotheses or no_longer_reads_latest_action_intent_hypotheses_from_verification_or_state or export_session_memory_keeps_observation_trace_without_hypotheses or planner_finish_metadata_excludes_answer_conditioned_hypothesis_fields)'`
  - 结果：`110 passed, 994 deselected`
- [x] 当前最小残留缺口已从 `Phase 2 / Phase 3` 迁移为：
  - `Phase 4` 中 specialized tool 身份降级是否还能继续收缩
  - `Phase 6` 中仍有一部分旧测试还在绑定特定 marker 类型，而不是只保护 observation-centric 行为边界
- [x] 本轮继续收缩 `Phase 2 / Phase 4` 交界处的一条 planner runtime 旧链：
  - 旧行为：
    - `planner._select_state_candidate_plan(...)`
      在 `broad reconsideration tool -> targeted gap tool` 的二次切换阶段，
      仍把：
      - `resolve_action_intent_pairwise`
      - `resolve_action_intent_future_use`
      视为 `exact_targeted_tools / targeted_gap_tools`
    - 这意味着即使它们已经退化成“普通候选工具”，
      runtime 仍会在 state-candidate 竞争里因为
      “specialized resolver 身份”
      再给它们一层结构性优待
  - 当前变化：
    - `targeted_gap_tools`
      已收缩为：
      - `query_object`
      - `query_spatial_context`
      - `infer_action_intent`
    - `exact_targeted_tools`
      已收缩为：
      - `query_object`
      - `query_spatial_context`
    - 结果是：
      - state-candidate 的二次偏好现在只承认 observation-grounded object/space probe
        与当前窗口内的普通 `infer_action_intent`
      - `resolve_action_intent_pairwise / resolve_action_intent_future_use`
        仍可作为普通候选保留，
        但不再因为 specialized identity 而在 rerank 阶段获得额外 target 优待
  - 对应本轮回归：
    - `pytest -q tests/test_graph_agent.py -k 'state_candidate_prefers_targeted_gap_tool or keeps_object_revisit_over_specialized_resolution or explicit_level_zero_budget_prefers_local_followup_over_specialized or action_intent and (state_candidate or sufficiency_context_gap or resolution_not_ready_recovery)'`
    - 结果：`20 passed, 1096 deselected`
    - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
    - 通过
- [x] 本轮继续切掉另一条 `generic recovery -> state candidate` 竞争中的 privileged-specialized 旧链：
  - 旧行为：
    - `planner._prefer_action_intent_state_candidate_over_generic_recovery(...)`
      在 generic recovery 与 state candidate 的竞争里，
      仍把：
      - `resolve_action_intent_pairwise`
      - `resolve_action_intent_future_use`
      视为 `exact_targeted_tools`
    - 这意味着即使 planner 已经开始 observation-first，
      一旦进入 `safe_fallback / repeated_loop / verifier_blocked / sufficiency_finish`
      这类 `generic recovery` 让位逻辑，
      specialized resolver 仍可能因“精确 targeted 身份”再拿到一层隐性优待
  - 当前变化：
    - `planner._prefer_action_intent_state_candidate_over_generic_recovery(...)`
      的 `exact_targeted_tools`
      已收缩为：
      - `query_object`
      - `query_spatial_context`
    - 结果是：
      - generic recovery 让位时，
        真正被优先承认的只剩 observation-grounded object/space probe
      - `resolve_action_intent_pairwise / resolve_action_intent_future_use`
        即使仍可作为普通工具存在，
        也不再因 specialized identity 被 helper 当成“精确目标恢复”优先接管
  - 同步迁移旧测试契约：
    - `test_planner_future_use_resolution_continue_search_prefers_state_candidate_over_generic_recovery`
      已从期待 generic followup，
      改为期待 observation-grounded `query_spatial_context`
      优先于 generic recovery
  - 对应本轮回归：
    - `pytest -q tests/test_graph_agent.py -k 'future_use_resolution_continue_search_prefers_state_candidate_over_generic_recovery or recover_if_low_confidence_repeated_textual_rank_prefers_state_candidate_over_generic_recovery or recover_if_low_confidence_verifier_blocked_finish_prefers_state_candidate_over_generic_recovery or enforce_task_requirements_action_intent_sufficiency_fallback_prefers_state_candidate_over_generic_recovery or action_intent and (safe_fallback or repeated_loop or blocked_tool or sufficiency_finish or enforce_context_gap)'`
    - 结果：`11 passed, 1105 deselected`
    - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
    - 通过
- [x] 本轮继续切掉 planner helper 层的一条 `need/conflict -> specialized tool family` 旧链：
  - 旧行为：
    - `planner._tool_matches_verifier_need(...)`
      仍把：
      - `resolve_action_intent_pairwise`
      - `resolve_action_intent_future_use`
      当成以下 need/conflict 的默认命中工具：
      - `immediate_outcome`
      - `future_outcome`
      - `relation_confirmation`
      - `need_post_action_evidence`
      - `need_disambiguating_evidence`
      - `multiple_candidate_answers`
    - `planner._is_action_intent_specialized_tool(...)`
      也仍把上述两个 tool 归入 action-intent specialized family
    - 这会导致即使上层 runtime 已逐步 observation-first，
      planner 排序底层仍会因为 need-matching helper
      把 specialized resolver 当作“天然匹配证据缺口”的 privileged family
  - 当前变化：
    - `planner._tool_matches_verifier_need(...)`
      已把 `resolve_action_intent_pairwise / resolve_action_intent_future_use`
      从上述 need/conflict 映射中移除
    - `planner._is_action_intent_specialized_tool(...)`
      也已收缩为：
      - `query_object`
      - `query_spatial_context`
      - `infer_action_intent`
    - 结果是：
      - `need_disambiguating_evidence / multiple_candidate_answers / future_outcome`
        这类 planner need-routing
        默认只再承认 observation-grounded object/space probe
        和当前窗口的普通 `infer_action_intent`
      - specialized resolver 不再因为 family 身份而在 helper 层自动命中这些 need
  - 同步迁移旧测试契约：
    - `test_planner_tool_addresses_disambiguating_need_with_action_intent_resolution_tools`
      已从要求 `resolve_action_intent_pairwise` 默认匹配
      改为要求：
      - `resolve_action_intent_pairwise` 不再默认匹配
      - `query_spatial_context / query_object` 仍可 observation-first 地匹配 disambiguating need
  - 对应本轮回归：
    - `pytest -q tests/test_graph_agent.py -k 'tool_addresses_disambiguating_need_with_action_intent_resolution_tools or action_intent and (state_candidate or safe_fallback or repeated_loop or blocked_tool or sufficiency_finish or enforce_context_gap or open_question_recovery)'`
    - 结果：`43 passed, 1073 deselected`
    - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
    - 通过
- [x] 本轮继续切掉 `open_question_recovery` 里一条 `resolution_mode/needs_* -> automatic specialized resume` 的旧链：
  - 旧行为：
    - `planner._recover_from_open_questions(...)`
      在 `self._action_intent_prefers_specialized_open_question_recovery(state)` 成立后，
      即使已经没有更强的 observation-grounded anchor，
      仍会仅凭：
      - `resolution_mode == resolve_action_intent_future_use / pairwise`
      - `action_intent_needs_future_use_evidence(...)`
      - `action_intent_pair_needs_outcome_resolution(...)`
      直接恢复：
      - `resolve_action_intent_future_use`
      - `resolve_action_intent_pairwise`
    - 这会让 open-question recovery 保留一条
      “旧 specialized 语义信号 -> 自动 specialized resume”
      的 runtime 主链
  - 当前变化：
    - 上述 `future_use/pairwise` 自动恢复分支已从 `planner._recover_from_open_questions(...)` 中删除
    - 当前 open-question recovery 主线只保留：
      - `primary_gap`
      - `evidence_first`
      - `raw_reuse_or_resample`
      - `best_state_candidate`
    - 结果是：
      - open-question 恢复不再因为旧的 `resolution_mode / needs_future_use / needs_pairwise`
        自动跳回 specialized resolver
      - specialized resolver 若仍出现，只能来自别的、更明确的 observation-grounded 路径，而不是这条 generic open-question 主链
  - 同步迁移旧测试契约：
    - `test_planner_action_intent_tap_scale_open_question_recovery_prefers_observation_first_recovery_over_specialized_pairwise`
      已从期待 `resolve_action_intent_pairwise`
      改为期待 observation-first 恢复
    - `test_planner_action_intent_open_question_recovery_can_preserve_existing_pairwise_without_competition_pressure`
      已从允许 `pairwise/future_use` resume
      改为要求结果停留在：
      - `sample_sparse_frames`
      - `query_time`
      - `query_object`
      - `query_spatial_context`
  - 对应本轮回归：
    - `pytest -q tests/test_graph_agent.py -k 'open_question_recovery and action_intent or verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume or action_intent_open_question_recovery_can_preserve_existing_pairwise_without_competition_pressure or action_intent_open_question_recovery_suppressed_answer_conditioned_target_prefers_observation_centric_followup_before_specialized_resume or action_intent_open_question_recovery_blocked_state_change_explicit_downstream_object_skips_specialized_resume or tap_scale_open_question_recovery_prefers_observation_first_recovery_over_specialized_pairwise'`
    - 结果：`20 passed, 1096 deselected`
    - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
    - 通过
- [x] 本轮继续切掉 `verifier_blocked` 中 `infer_action_intent -> automatic specialized resume` 的残留旧链：
  - 旧行为：
    - `planner._recover_action_intent_after_verifier_blocked_finish(...)`
      在 `tool_name == infer_action_intent` 的分支里，
      即使没有新的 observation-grounded anchor，
      仍会在以下位置自动回到 specialized resolver：
      - `post_action` blocker 分支里 fallback 到 `future_use`
      - `future_gap_family + pending pairwise` 分支里 fallback 到 `pairwise`
      - 通用 close-call 尾部再按
        `action_intent_needs_future_use_evidence(...) / action_intent_pair_needs_outcome_resolution(...)`
        恢复 `future_use/pairwise`
    - 这会让 `infer_action_intent` 被 verifier 拦下后，
      仍保留一条
      “旧 specialized 判别信号 -> automatic specialized resume”
      的 recovery 主链
  - 当前变化：
    - 上述 4 段 automatic `future_use/pairwise` resume
      已从 `tool_name == infer_action_intent` 分支中删除
    - 当前该分支保留的主线只剩：
      - `primary_gap`
      - `finalize_long_horizon_revisit`
      - `transition_probe`
      - `followup`
      - `extra_followup`
    - 结果是：
      - `infer_action_intent` 被 `verifier_blocked` 后，
        不再因为 `pending_resolution_tool / needs_future_use / pair_needs_outcome`
        这些旧 specialized 信号自动跳回 `pairwise/future_use`
      - 只有 observation-grounded 的 gap / probe / followup 继续主导恢复
  - 对应本轮回归：
    - `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume or verifier_blocked_infer_prefers_existing_pairwise_over_future_use_hint or verifier_blocked_disable_legacy_prefers_existing_pairwise_over_future_use_hint or action_intent and verifier_blocked'`
    - 结果：`60 passed, 1056 deselected`
    - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
    - 通过
- [x] 本轮继续切掉 `primary_gap -> immediate_outcome` 尾部的一条 specialized fallback 旧链：
  - 旧行为：
    - `planner._recover_action_intent_immediate_outcome_gap(...)`
      在 `forced_transition_probe / followup / extra_followup`
      都未命中的尾部，
      仍会按：
      - `structured_specialized_recovery_tool == resolve_action_intent_pairwise`
        回到 `pairwise`
      - 否则回到 `future_use`
    - 这意味着即使 `immediate_outcome` 已被统一成 observation-centric primary gap，
      recovery 尾部仍残留一条
      “补证失败 -> automatic specialized resolver fallback”
      的旧链
  - 当前变化：
    - 上述尾部 `pairwise/future_use` fallback 已从
      `planner._recover_action_intent_immediate_outcome_gap(...)`
      中删除
    - 现在若近窗 probe/followup 都未命中：
      - 有时间锚点时，继续 `sample_sparse_frames` 扩动作后短窗口
      - 无时间锚点时，保守重抽当前动作片段
    - 结果是：
      - `immediate_outcome` gap recovery 彻底保持 observation-first
      - specialized resolver 不再作为这条 primary-gap 路径的尾部兜底
  - 对应本轮回归：
    - `pytest -q tests/test_graph_agent.py -k 'primary_gap_immediate_outcome or action_intent and (verifier_blocked or primary_gap)'`
    - 结果：`88 passed, 1028 deselected`
    - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
    - 通过
- [x] 本轮继续切掉了一段 `verifier.py` 中仍由 specialized tool 身份主导的旧链：
  - `_action_intent_verifier_blocker(...)`
    不再优先按
    `resolve_action_intent_future_use / resolve_action_intent_pairwise`
    决定 `why_blocker`
  - `_action_intent_build_evidence_gaps(...)`
    不再按 tool 名判断 `future_outcome` 是否需要继续生成
  - `_action_intent_gap_source(...)`
    不再仅因“上一轮调用过 specialized resolver”
    就把 gap source 直接写成 `resolution_followup_gap`
  - 当前改为：
    - 先由 payload 的 observation-side 竞争特征
      推断 `future_use_close_call / pairwise_close_call`
    - 再决定 blocker family 与 gap source
    - tool identity 退为兼容性背景信息，而不是主判据
- [x] 本轮继续收缩了一段仍把 `specialized tool name` 当成推理状态的 planner 主链：
  - 已删除/降级的运行态惯性链：
    - `open_question recovery` 中
      `latest pairwise -> keep pairwise over future_use`
    - `state candidate plan` 中
      `latest_resolution_tool -> preserve specialized over level-zero/raw followup`
    - `state candidate plan` 中
      `pending/structured_specialized_tool -> prefer specialized resolution over targeted gap tool`
  - 当前变化：
    - `specialized tool` 继续存在，但更接近“普通搜证工具”
    - 是否继续调用它，优先服从：
      - `primary_gap`
      - 当前 observation anchor
      - 当前 budget / followup window
    - 不再因为“上一轮是 pairwise/future_use”就自动保持同一 specialized 身份
- [x] 本轮补齐了一个真实代码缺口，而不是继续堆局部规则：
  - `_recover_action_intent_after_verifier_blocked_finish(...)`
    在 `infer_action_intent + future_outcome gap` 且前序 recovery 都未命中时，
    之前会因为 `return extra_followup` 的空返回直接落成 `None`
  - 现已修复为：
    - 先尝试 `primary_gap` 路由
    - 若仍无更具体动作，再退到当前时间窗 `sample_sparse_frames`
  - 这保证了：
    - `verifier_blocked_finish`
      不再因为旧 specialized 分支失效而直接失去恢复动作
    - planner 始终还能给出 observation-centric 的下一步搜证决策
- [x] 本轮同步迁移了一组仍保护旧 specialized 惯性的测试契约：
  - 不再要求：
    - “必须继续 pairwise”
    - “必须保留 existing specialized identity”
  - 改为只要求：
    - 仍沿 `future_outcome / relation / target` 这类 observation gap 继续补证
    - 允许返回：
      - `sample_sparse_frames`
      - `query_object`
      - `query_spatial_context`
      - `resolve_action_intent_pairwise`
      - `resolve_action_intent_future_use`

- [x] 本轮继续收缩 `state / planner / real-eval` 中的候选答案残影：
  - `state.py`
    - `action_intent_trace` 导出已不再依赖 `action_intent_hypotheses` 非空
    - 当前 trace 只保留：
      - `summary`
      - `primary_gap`
      - `primary_gap_recovery_trace`
      - `recommended_next_action`
      - `finish_mode`
  - `planner.py`
    - `finish final_metadata` 已停止读取：
      - `action_intent_hypotheses`
      - `blocking_hypotheses`
      - `blocking_comparisons`
    - runtime finish 输出主视角收敛为：
      - `finish_reason`
      - `remaining_gaps`
      - `final_support_summary`
      - `used_budget`
  - `scripts/run_graph_agent_small_real_why_eval.py`
    - 真实 why 小样本报告已停止输出：
      - `avg_hypothesis_count`
      - 每题 `hypotheses`
    - 评估文本改为围绕：
      - `gap_type_counts`
      - `finish_mode_counts`
      - `search_budget`
      - `action_intent_trace`
      - `primary_gap_recovery_trace`

- [x] 本轮继续切断 verifier/state 内部的 hypothesis 回流链：
  - `verifier.py`
    - `_build_action_intent_hypotheses(...)` 已降级为空输出
    - why verifier 继续输出的主状态只剩：
      - `missing_evidence_types`
      - `evidence_gaps`
      - `sufficiency_decision`
      - `summary`
  - `state.py`
    - `record_verification(...)` 已不再把传入的 `action_intent_hypotheses` 写入 `verification_history` 或 `state.action_intent_hypotheses`
    - 当前统一归零，避免旧候选层结论在 session 中继续传播
  - `planner.py`
    - `_state_latest_action_intent_hypotheses(...)` 已降级为空读取
    - planner runtime 不再从 verification/state 中恢复 hypothesis 结构
  - `executor.py`
    - finish metadata 构建已不再依赖 hypothesis 回填

- [x] 本轮同步迁移了一组旧测试契约：
  - 之前测试要求：
    - verifier 必须构造 top/runner-up hypothesis
    - state 必须保存 hypothesis
    - planner 必须能读回 hypothesis
  - 现在改为负约束：
    - verifier hypothesis 输出必须为空
    - state 不得持久化 hypothesis
    - planner 不得再读取 hypothesis 作为运行态信息

- [ ] 当前剩余主缺口：
  - verifier / state 中仍保留兼容字段名 `action_intent_hypotheses`
  - 还需要继续确认这些兼容字段是否可以进一步从内部 schema 层降级或移除
  - 旧专项规则是否仍在 runtime 主链上保留实质决策权
    - 但不再允许把旧的 answer-conditioned 守卫当成必要契约
- [x] 对应本轮定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'pending_resolution_tool or structured_specialized_recovery_tool or open_question_recovery_can_preserve_existing_pairwise_without_competition_pressure or verifier_blocked_infer_prefers_existing_pairwise_over_future_use_hint or best_state_candidate_plan_keeps_targeted_gap_tool_for_future_outcome_even_when_step_mentions_post_action_evidence'`
  - 结果：`7 passed, 1095 deselected`
- [x] 对应本轮扩大专项回归：
  - `pytest -q tests/test_graph_agent.py -k 'followup_route or pair_needs_outcome_resolution or needs_future_use_evidence or result_is_close_call_for_recovery or before_text_fallback or future_use_candidates or pairwise_candidates or verifier_blocked'`
  - 结果：`88 passed, 1014 deselected`
- [x] 本轮继续收掉 `graph_agent.py` finalize 层里真正会把 why 带回 answer-conditioned 的两条旧入口：
  - 已禁止：
    - why/action_intent 从 `action_intent_best_index=` 这类 `working_memory` 状态直接 deterministic finalize
    - why/action_intent 在文本生成失败后，从 `trace best_index` 直接做 finalize recovery
  - 仍保留：
    - `resolve_action_intent_pairwise / resolve_action_intent_future_use`
      这类 observation-side specialized result 作为 why finalizer 的保守裁决来源
  - 这意味着：
    - why 的最终答案仍可由真实 observation-side finalizer 产出
    - 但不能再由旧的 state residue / trace residue 直接恢复
- [x] 本轮同步迁移了一组 finalize 契约：
  - 不再允许：
    - “可重新开启 legacy action-intent state rules”
    - “why 文本失败后可从 trace best_index 直接恢复最终答案”
  - 改为要求：
    - why 只允许 observation-side finalizer 给出答案
    - 禁止 `finalize_recovery=trace_best_index` 出现在 why 上
- [x] 对应本轮 graph-agent finalize 定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent_finalizer or tap_scale_missing_state_change_prereq_blocks_structured_best_index_fallback or finalize_action_intent_does_not_use_legacy_best_index_state_for_finalize or finalize_action_intent_does_not_reenable_legacy_best_index_finalize or finalize_action_intent_does_not_use_trace_recovery_marker'`
  - 结果：`37 passed, 1066 deselected`
- [x] 本轮 `verifier.py` 已进一步切断 hypothesis / sufficiency 中的 answer-conditioned 注入链：
  - `_action_intent_result_support_text(...)`
    不再读取：
    - `needed_observation`
    - `answer`
  - `_build_action_intent_hypotheses(...)`
    不再把以下候选答案侧产物写入运行态 hypothesis：
    - `needed_observation`
    - `comparison_summary`
    - `comparison.score_gap`
    - `comparison.unresolved_evidence`
  - `_build_sufficiency_decision(...)`
    不再从 hypothesis comparison payload 反推出
    `blocking_comparisons`
  - 这意味着：
    - verifier 现在只保留 observation-side 的
      `support_evidence / contradiction_evidence / evidence_gaps`
    - `runner-up / score_gap / unresolved_evidence`
      不再由 verifier 主路径继续扩散到后续搜索态
- [x] 对应本轮 verifier 定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'verifier_action_intent_support_text_excludes_answer_conditioned_fields or verifier_builds_action_intent_hypotheses_from_latest_resolution_payload or verifier_action_intent_hypothesis_close_call_blocks_finish_even_without_explicit_missing or verifier_sufficiency_decision_treats_structured_blocking_comparison_as_needs_more_evidence_without_explicit_gap or result_support_text_excludes_answer_conditioned_fields or pairwise_hidden_outcome_detector_ignores_answer_conditioned_fields or weak_generic_claim_detector_ignores_answer_conditioned_fields'`
  - 结果：`7 passed, 1094 deselected`
- [x] 对应本轮 observation-centric 保护回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent_competition_pressure or open_question_recovery_can_preserve_existing_pairwise_without_competition_pressure or action_intent_competition_pressure_does_not_use_hypothesis_score_gap_without_structured_comparisons or textual_rank_does_not_consume_finalizer_or_unresolved_rerank_revisit_markers or open_question_recovery_suppressed_answer_conditioned_target_prefers_observation_centric_followup_before_specialized_resume or verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume'`
  - 结果：`7 passed, 1094 deselected`
- [x] 本轮 `planner.py` 又切掉一组真实的 answer-conditioned later-target / exact-target 恢复链：
  - 以下 helper 现已停止作为运行态搜索入口：
    - `_action_intent_unresolved_rerank_mixed_horizon_later_target_hint(...)`
    - `_action_intent_verifier_blocked_mixed_horizon_later_target_hint(...)`
    - `_action_intent_verifier_blocked_measurement_target_hint(...)`
    - `_action_intent_verifier_blocked_phone_record_target_hint(...)`
  - 以下 specialized revisit builder 现已降级退出主路径：
    - `_build_action_intent_finalize_withheld_mixed_horizon_later_target_revisit_decision(...)`
    - `_build_action_intent_verifier_blocked_mixed_horizon_later_target_revisit_decision(...)`
    - `_build_action_intent_verifier_blocked_measurement_target_revisit_decision(...)`
    - `_build_action_intent_verifier_blocked_phone_record_target_revisit_decision(...)`
    - `_build_action_intent_unresolved_rerank_mixed_horizon_later_target_revisit_decision(...)`
  - 这意味着：
    - planner 不再因为 `mixed horizon / measurement / phone record` 这类候选语义差异
      自动推出某个 exact target 再去 `query_object / query_spatial_context`
    - 后续只能退回：
      - `primary_gap`
      - `timeline review`
      - `local followup`
      - `已有真实观测锚点`
      这些 observation-centric 路径
- [x] 对应本轮 later-target 清理回归：
  - `pytest -q tests/test_graph_agent.py -k 'mixed_horizon_no_longer_revisits or mixed_horizon_later_target_marker_no_longer_forces_real_fixture_revisit or verifier_blocked_measurement_target_hint_does_not_use_needed_observation_alone or verifier_blocked_phone_record_target_hint_does_not_use_needed_observation_alone or mixed_horizon_later_target_hint_does_not_use_needed_observation_alone or later_outcome_target_hint_does_not_use_choice_text_alone or later_outcome_target_hint_uses_observed_evidence_text'`
  - 结果：`10 passed, 1091 deselected`
- [x] 对应本轮 Phase 3 保护回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent_competition_pressure or open_question_recovery_suppressed_answer_conditioned_target_prefers_observation_centric_followup_before_specialized_resume or verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume or result_support_text_excludes_answer_conditioned_fields or pairwise_hidden_outcome_detector_ignores_answer_conditioned_fields or weak_generic_claim_detector_ignores_answer_conditioned_fields'`
  - 结果：`8 passed, 1093 deselected`
- [x] 当前主清单已做到“全量可勾选、可直接查看进度”
- [x] 本轮又切掉一条真正会跨题污染的新旧混合链：
  - `state.export_session_memory()` 不再持久化：
    - `action_intent_hypotheses`
    - `final_metadata`
  - `state.restore_session_memory()` 不再恢复：
    - `action_intent_hypotheses`
    - `final_metadata`
  - `GraphAgentVideoSession._prepare_restored_state_for_new_question(...)`
    现在会主动清空：
    - `final_metadata`
  - 这意味着：
    - 上一题的 `top candidate / finish_reason / structured_final_candidate`
      不会再作为“会话记忆”跨题回灌到下一题执行态
    - 同视频多题复用现在只保留 observation-side 资产：
      - `working_memory`
      - `evidence_bundle`
      - `retrieved_frames`
      - `retrieved_nodes`
      - `visited_times`
      - `search_budget`
      等
- [x] 对应本轮跨题恢复链定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'agent_state_export_and_restore_session_memory or snapshot_excludes_answer_conditioned_action_intent_artifacts or blocks_restored_action_intent_conclusions_for_why_tasks or prepare_restored_state_clears_restored_final_metadata or open_query_reuses_restored_artifact_frames'`
  - 结果：`4 passed, 1098 deselected`
- [x] 本轮还同步清掉了真实 trace 审计脚本中的 candidate-convergence 旧视角：
  - `scripts/run_graph_agent_small_real_why_eval.py`
    不再统计：
    - `top_hypothesis_change_count`
    - `runner_up_change_count`
    - `top2_elimination_event_count`
    - `avg_final_score_gap`
    - `final_unresolved_evidence`
  - 现在改为只统计 observation-side trace：
    - `initial/final_primary_gap_type`
    - `primary_gap_change_count`
    - `initial/final_recommended_next_action`
    - `recommended_next_action_change_count`
    - `primary_gap_recovery_trace`
    - `final_finish_mode`
  - 对应误差分析文案也从
    `candidate_convergence`
    改成
    `observation_trace`
  - 这意味着：
    - 真实小样本审计不再围绕“候选如何竞争/淘汰”解释 agent
    - 而是围绕“观测缺口如何变化、下一步动作如何变化”解释 agent
- [x] 对应本轮 trace 审计脚本定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'run_graph_agent_small_real_why_eval or keeps_observation_centric_action_intent_trace'`
  - 结果：`5 passed, 1096 deselected`
- [x] 对应本轮 finalizer-marker producer/consumer 清理回归：
  - 定向：
    - `pytest -q tests/test_graph_agent.py -k 'graph_agent_action_intent_finalizer_marks_later_fixture_target_for_mixed_horizon_open_vs_weigh or graph_agent_action_intent_finalizer_marks_later_fixture_target_for_mixed_horizon_label_vs_put_back or graph_agent_action_intent_finalizer_marks_later_fixture_target_for_mixed_horizon_open_vs_empty or graph_agent_action_intent_finalizer_marks_later_sink_target_for_weak_boiling_check_close_call or graph_agent_action_intent_finalizer_marks_plate_needed_observation_for_weak_contents_check_close_call or planner_action_intent_generic_access_withheld_marker_revisits_real_revealed_target or planner_action_intent_generic_relocation_withheld_marker_revisits_real_downstream_target or planner_action_intent_mixed_horizon_later_target_marker_no_longer_forces_real_fixture_revisit or graph_agent_action_intent_finalizer_withholds_generic_put_away_for_same_object_reuse_target or graph_agent_action_intent_finalizer_withholds_generic_put_away_for_revealed_downstream_target or graph_agent_action_intent_finalizer_withholds_explicit_fridge_return_over_weak_label_inspection or graph_agent_action_intent_finalizer_marks_later_object_target_for_mixed_horizon_open_vs_serve or graph_agent_action_intent_finalizer_does_not_override_weak_boiling_check_to_explicit_sink_emptying'`
    - 结果：`13 passed, 1091 deselected`
  - 扩大：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent and (generic_access or generic_relocation or mixed_horizon or weak_cooking_inspection or verifier_blocked_finish_generic_access_marker or verifier_blocked_finish_generic_relocation_marker or verifier_blocked_finish_mixed_horizon_marker or textual_fallback_generic_relocation_marker or skips_legacy_finalize_markers_when_disabled)'`
    - 结果：`37 passed, 1067 deselected`
  - 更大专项：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent and (finalizer or verifier_blocked or before_text_fallback or pending_resolution_tool or structured_specialized_recovery_tool or finalize_action_intent or trace_best_index or structured_best_index or does_not_build_action_intent_hypotheses or discards_action_intent_hypotheses or no_longer_reads_latest_action_intent_hypotheses_from_verification_or_state or export_session_memory_keeps_observation_trace_without_hypotheses or planner_finish_metadata_excludes_answer_conditioned_hypothesis_fields)'`
    - 结果：`110 passed, 994 deselected`
- [x] 本轮又切掉了一组已经空壳化、但仍可能回潮的 planner target-hint 入口：
  - 已删除：
    - `_action_intent_best_structured_target_hint(...)`
    - `_action_intent_competing_hypothesis_target_hint(...)`
    - `_action_intent_top_hypothesis_target_hint(...)`
    - `_action_intent_comparison_blocker_target_hint(...)`
  - 这些 helper 虽然此前已经不再返回真实 hint，
    但仍作为“候选答案字段可重新接回搜索主链”的壳存在于 `planner.py`
  - 现在直接删除后，planner 主链中不再保留这类
    `hypothesis/comparison -> target hint -> object/fixture revisit`
    的潜在接线点
- [x] 对应本轮 planner 残留入口定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'no_longer_exposes_action_intent_competing_hypothesis_target_hint or no_longer_exposes_action_intent_top_hypothesis_target_hint or no_longer_exposes_action_intent_comparison_blocker_target_hint or no_longer_exposes_action_intent_best_structured_target_hint or candidate_ranking_does_not_use_blocking_comparison_pressure_to_reweight_disambiguating_tools or textual_fallback_same_object_active_use_does_not_force_action_object_revisit'`
  - 结果：`6 passed, 1096 deselected`
- [x] 本轮继续收掉一组仍在运行态里的 `candidate residue -> specialized recovery` 链：
  - `planner.py` 中以下路径已进一步去残影恢复：
    - `_action_intent_pending_resolution_tool(...)`
    - `_action_intent_structured_specialized_recovery_tool(...)`
    - `_action_intent_pending_candidate_indices(...)`
    - `_latest_action_intent_candidate_indices(...)`
  - 当前不再允许：
    - 从 `working_memory` 的
      `action_intent_best_index / action_intent_second_best_index / action_intent_pending_candidates`
      回灌 specialized recovery
    - 从 `structured_hypotheses`
      回灌 candidate indices
    - 从 `relation_confirmation / target_discovery / immediate_outcome`
      这类 gap 自动恢复 `resolve_action_intent_pairwise`
  - 当前仍保留的 observation-centric specialized recovery 只剩：
    - `future_outcome -> resolve_action_intent_future_use`
    - 并且需要服从 `primary_gap / sufficiency / 当前工具轨迹`
- [x] 对应本轮 specialized recovery 去残影定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'pending_candidate_indices_can_bootstrap_from_structured_pairwise_gap_without_payload_or_hypotheses or latest_candidate_indices_can_bootstrap_from_structured_future_outcome_gap_without_payload_or_hypotheses or prefers_specialized_open_question_recovery_from_observation_grounded_relation_confirmation_gap_without_payload_or_hypotheses or structured_specialized_recovery_tool_does_not_bootstrap_pairwise_from_structured_hypotheses_alone or pending_resolution_tool_does_not_start_from_structured_hypotheses_alone or pending_resolution_tool_can_fall_back_to_structured_missing_gap_types_without_evidence_gaps'`
  - 结果：`6 passed, 1096 deselected`

### 当前阶段进展补充

- [x] `planner.py` 已完成一轮 `future_outcome` observation-centric 收口：
  - `verifier_blocked / open_question / low_conf_finish / heuristic_fallback`
  - 不再只依赖 sanitized `primary_gap.target_object`
  - 新增从原始 `evidence_gaps` 恢复 observation hint 的桥接层，用于：
    - `explicit downstream object`
    - `fixture-only spatial gap`
- [x] `primary_gap` 清洗逻辑已重新分层：
  - `sufficiency_missing_gap_types` 继续保持 targetless，不再回灌答案条件化 target
  - 显式 `evidence_gaps` 中的 `future_outcome / target_discovery / relation_confirmation`
    仅在当前 observation state 足够时保留 object / fixture hint
- [x] `verifier.py` 的 `evidence_gaps` 生产端已继续去答案条件化：
  - `_action_intent_build_evidence_gaps(...)`
    不再因为 `blocker_hint == future_use_close_call / pairwise_close_call`
    就直接生成 `future_outcome` gap
  - `future_outcome` gap 现在只允许来自：
    - `resolve_action_intent_future_use`
    - `resolve_action_intent_pairwise`
    这类工具结果自己明确声明的 `need_future_evidence / need_more_evidence`
  - gap `source` 也已从 `future_use_close_call / pairwise_close_call`
    继续收缩成 observation-centric 的：
    - `verification_gap`
    - `resolution_followup_gap`
    - `precondition_gap`
  - 这意味着 `close_call/blocker` 文本本身不再是 gap 生产入口，
    只能退回为解释痕迹或兼容层信息
- [x] `timeline review` 恢复链已收口为 observation-first：
  - 若已存在 cached long-horizon revisit，则优先复用
  - 若仍是 `future_use_close_call / pairwise_close_call`，允许保留 specialized resolution
  - 不再因为显式 gap 的存在就一律提前强制 `query_object`
- [x] 本轮 `planner.py` 又收掉了一段 `close-call source` 依赖：
  - `timeline review` 继续搜索时，不再要求
    `primary_gap.source == future_use_close_call / pairwise_close_call`
    才允许进入 observation-first 的后续恢复
  - `_action_intent_observation_hint_from_explicit_gap(...)`
    现在只接受 observation-centric gap source：
    - `verification_gap`
    - `resolution_followup_gap`
    - `verifier`
    - `precondition_gap`
  - `_action_intent_verifier_blocker_hint(...)`
    已不再优先读取 `gap_source == future_use_close_call / pairwise_close_call`
    来区分 blocker family，而是主要按：
    - `gap_type`
    - `recommended_next_step`
    - `missing_evidence_types`
    做解释层映射
  - 一批旧测试也已改写为新的 observation-first 契约：
    - 不再保护 `close_call -> specialized tool`
    - 改为保护：
      - 有显式 object gap 时，允许直接走 gap-routed object/spatial follow-up
      - 未观测到 object 锚点时，允许优先复用已存在的 spatial/object cache
- [x] 本轮 `verifier_blocked` gate 也进一步去 close-call family 化：
  - 以下 helper 不再直接把
    `future_use_close_call / pairwise_close_call`
    作为唯一开关，而是改为优先服从通用 gap family：
    - `_action_intent_verifier_blocked_mixed_horizon_later_target_hint(...)`
    - `_action_intent_verifier_blocked_measurement_target_hint(...)`
    - `_action_intent_verifier_blocked_phone_record_target_hint(...)`
    - `_action_intent_verifier_blocked_prefers_forced_transition_probe(...)`
    - `_action_intent_verifier_blocked_same_object_active_use_hint(...)`
    - `_recover_action_intent_immediate_outcome_gap(...)`
  - 当前这些 gate 主要通过：
    - `primary_gap.gap_type`
    - `post_action family`
    - `future gap family`
    来解释当前缺口，而不再要求保留旧 close-call family 的字面分流
- [x] `planner.py` 已切断一批 `blocking_* / comparison` 直接驱动继续搜证的 consumer：
  - `_has_unresolved_evidence_gap(...)` 不再因 `blocking_hypotheses / blocking_comparisons` 单独返回 `True`
  - `_current_evidence_needs(...)` 不再仅因 `blocking_comparisons` 存在就自动补 `need_disambiguating_evidence`
  - `_should_continue_search_from_sufficiency(...)` 不再因 `blocking_hypotheses / blocking_comparisons` 或 `decision.sufficient=False` 单独继续搜证
  - `_action_intent_success_result_is_ready_for_failure_finish(...)` 与 `_action_intent_intent_payload_is_ready_to_fall_back_to_text_rank(...)`
    不再仅因 `competition.is_close_call` 这种候选竞争残影阻止 finish / fallback
  - `_action_intent_result_is_close_call_for_recovery(...)` 不再在 payload 缺少 competitor 时，
    仅凭 `competition.is_close_call` 把 recovery 重新拉回候选竞争残影
  - `candidate bootstrap` 主链已进一步去 hypotheses 化：
    - `_action_intent_pending_candidate_indices(...)`
    - `_latest_action_intent_candidate_indices(...)`
    - `_fallback_action_intent_pairwise_candidate_indices(...)`
    - `_action_intent_structured_specialized_recovery_tool(...)`
    不再从 `structured_hypotheses` 自动拼出 specialized pairwise 候选
  - `_action_intent_pending_resolution_tool(...)` 对 `immediate_outcome / relation_confirmation / target_discovery`
    不再因为“有这种 gap”就一律启动 pairwise；多选场景至少需要显式 candidate pair 迹象
  - `_recover_action_intent_after_verifier_blocked_finish(...)` 的 precondition 恢复入口
    已从 `blocker_hint == precondition_context` 改为优先看 `primary_gap == precondition`
  - `disable_legacy_specialized_recovery` 分支中的 `post_action/future_use/pairwise` 恢复
    已开始从 `blocker_hint` 文本驱动，收缩为优先服从：
    - `primary_gap` 是否属于 post-action 类 gap
- [x] 本轮又切掉了一段仍会把 `specialized tool identity` 当作 planner 续搜惯性的旧链：
  - `_enforce_task_requirements(...)`
    中原先仍残留：
    - `latest_resolution_tool == resolve_action_intent_pairwise`
    - 且 `best_state_candidate_plan == resolve_action_intent_future_use`
    时，强行回退去保留 `pairwise`
  - 这条“上一轮 specialized 身份 > 当前 gap candidate”链现已删除
  - 当前 enforce/sufficiency 阶段统一改为：
    - 若 `best_state_candidate_plan(...)` 已给出更贴合当前 observation gap 的动作
    - 就直接采用该 candidate
    - 不再因为“上一轮是 pairwise”而续命同身份 specialized tool
  - 这意味着：
    - `enforce` 阶段开始与 `state candidate` 主线使用同一套 observation-centric 选择口径
    - specialized tool 继续存在，但只是普通候选动作，不再带“身份保留权”
- [x] 对应本轮定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'enforce_task_requirements_action_intent_sufficiency or best_state_candidate_plan_keeps_targeted_gap_tool_for_future_outcome_even_when_step_mentions_post_action_evidence or state_candidate_keeps_object_revisit_over_specialized_resolution_for_structured_close_call or state_candidate_prefers_object_revisit_when_structured_gap_synthesizes_future_use or explicit_level_zero_budget_prefers_local_followup_over_specialized'`
  - 结果：`11 passed, 1102 deselected`
- [x] 对应本轮扩大回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (open_question_recovery or pending_resolution_tool or heuristic_fallback or before_text_fallback or enforce_task_requirements)'`
  - 结果：`46 passed, 1067 deselected`
- [x] 本轮继续收掉了一条 `same-object future_outcome -> specialized resume` 的 runtime 绕开链：
  - 旧行为：
    - `planner._heuristic_fallback(...)`
      中如果 `primary_gap.target_object == question_object`
      就会绕开 `primary_gap recovery`
      继续保留 existing specialized resume
    - `planner._recover_action_intent_via_primary_gap(...)`
      在 `future_outcome` 场景下也会对同对象/无新锚点情形保留 specialized 恢复惯性
  - 新行为：
    - 只要当前 `future_outcome` gap 仍然只锚定到动作对象本身
    - 且还没有新的更晚用途/状态锚点
    - 就统一先走 observation-centric 的 `gap_late_followup`
    - 不再直接回到 `resolve_action_intent_pairwise / resolve_action_intent_future_use`
  - 这意味着：
    - `same-object` 不再被当作“保 specialized 身份”的理由
    - 它现在只是普通的 coverage-insufficient 状态
    - agent 会先补更晚原始证据，再决定是否需要 specialized 判别
- [x] 对应本轮定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'same_object_primary_gap or primary_gap_same_object_future_outcome or primary_gap_future_outcome_object_gap or heuristic_fallback_late_primary_gap or verifier_blocked_future_outcome_fixture_only_prefers_local_followup_before_fixture_query'`
  - 结果：`6 passed, 1108 deselected`
- [x] 本轮继续切掉了一条 `verifier_blocked` 中的 specialized-preserve 主链：
  - 旧行为：
    - 即便 `primary_gap` 已明确给出 gap-routed 恢复动作
    - `planner._recover_action_intent_after_verifier_blocked_finish(...)`
      仍会因为“当前 tool 是 `resolve_action_intent_future_use / pairwise`”
      且“结果还是 close call”
      把 `gap_routed_decision` 压回去
    - 同时还会把 `sample_sparse_frames` 这类 gap-routed 动作记成
      `verifier_blocked_preserves_specialized_resolution=...`
  - 新行为：
    - 只要 `primary_gap recovery` 已给出动作
    - 就直接服从该 gap-routed 动作
    - 不再因为 specialized tool 身份保留而覆盖它
  - 这意味着：
    - `verifier_blocked` 阶段开始真正以 `primary_gap` 为主驱动
    - specialized tool 不再拥有“覆盖 gap-routed 恢复”的优先权
    - 旧日志语义也同步迁成 `verifier_blocked_prefers_primary_gap=...`
- [x] 对应本轮定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_preserves_specialized_resolution or verifier_blocked_recovery_does_not_drop_specialized or verifier_blocked_relation_confirmation_fixture_target_defers_long_horizon_spatial_query_at_window_level_zero or verifier_blocked_recovery_can_proceed_from_structured_gap_even_without_blocker_hint'`
  - 结果：`4 passed, 1110 deselected`
- [x] 对应本轮扩大回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (verifier_blocked or same_object_primary_gap or enforce_task_requirements or heuristic_fallback)'`
  - 结果：`79 passed, 1035 deselected`
- [x] 本轮继续收缩了一段 `open_question recovery` 中的 specialized-first 恢复链：
  - 旧行为：
    - 在 `raw_reuse_or_resample` 之后
    - `open_question recovery`
      仍会按
      `structured_specialized_tool / pending_resolution_tool`
      直接续 `resolve_action_intent_future_use / pairwise`
    - 这意味着已有 specialized 身份仍有主路径优先权
  - 新行为：
    - 在 `raw_reuse_or_resample` 之后
    - 先统一尝试 `best_state_candidate_plan(...)`
    - 若已有更贴合当前 observation gap / coverage / budget 的 candidate
      就直接采用
    - specialized open-question recovery 退到后手，只在 gap/local raw/state candidate 都没有更强动作时才触发
  - 这意味着：
    - `open_question recovery` 开始与其他主路径统一成
      `gap -> local/raw evidence -> state candidate -> specialized fallback`
    - specialized tool 在 open-question 阶段进一步退成普通兜底动作
- [x] 对应本轮定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'open_question_recovery and (explicit_downstream_object_prefers_gap_over_existing_pairwise_resolution or can_preserve_existing_pairwise_without_competition_pressure or does_not_drop_pairwise_close_call_into_generic_followup_when_specialized_resolution_is_already_established or suppressed_answer_conditioned_target_prefers_observation_centric_followup_before_specialized_resume)'`
  - 结果：`4 passed, 1110 deselected`
    - 当前已有的 specialized resolution 上下文
  - `candidate ranking` 主链不再读取 `blocking_comparisons / score_gap / runner-up` 去重排 why/action_intent 的搜索优先级
    - 已删除基于 `action_intent_competition_pressure()` 的 targeted-disambiguation 加权
    - 当前候选计划排序只允许继续服从：
      - `primary_gap`
      - `verifier_missing / recommended_next_step`
      - `window_level / budget`
      - 是否已有 specialized chain 与局部 followup 预算约束
  - `open_question / textual evidence-first` 的 specialized-chain 保留逻辑
    不再要求 `competition_pressure.is_close_call`
    - 现在是否继续沿已有 `pairwise` specialized 链，只看：
      - 最近确实已经进入 specialized 上下文
      - 当前 specialized 结果仍明确缺证据
      - 没有显式 downstream object gap 要求先转向 gap-routed 搜证
  - `action_intent_competition_pressure()` 也已去答案条件化：
    - 不再读取 `blocking_comparisons`
    - 不再读取 `score_gap`
    - 不再把 `runner-up / comparison residue` 当成 close-call 触发器
    - 现在该 helper 只允许读取：
      - `primary_gap / evidence_gaps`
      - `window_level`
      - `new_frames_observed`
      - `long_horizon_expansions_used`
    - 它的语义已收缩为：
      - “当前是否还存在真实 observation gap”
      - “当前覆盖和预算层级是否还停留在浅层，值得继续局部补证”
  - `build_state_driven_candidates(...)` 中旧的 `competition_pressure` 接线已删除
    - 避免后续再从候选比较残影把这条链重新接回 planner 主路径
  - `planner.py` 中一条真正驱动 why 搜索的 answer-conditioned 文本消费链也已切掉：
    - repeated textual fallback 前的残留 `latest_needed_observation` 读取已删除
    - `_action_intent_pairwise_text_has_explicit_hidden_outcome(...)`
      不再读取：
      - `needed_observation`
      - `answer`
    - `_action_intent_result_is_weak_generic_claim(...)`
      不再读取：
      - `needed_observation`
      - `answer`
    - 这意味着：
      - hidden-target / reveal-use / weak-generic followup
        不再允许靠候选答案文本或“还需要看什么”的答案侧描述来触发
      - 只能继续依赖：
        - `reason`
        - `decisive_observation`
        - `direct_effect`
        - `downstream_action`
        这些 observation-side 文本
  - `verifier_blocked` 主路径已继续删除两类 answer-conditioned consumer：
    - `finalizer withheld generic_access_or_space / generic_relocation_or_storage`
      不再单独驱动 downstream target revisit
    - `infer_action_intent` 下的 `mixed_horizon later target revisit`
      不再因 choice/category 家族直接把搜索路径跳到更晚候选目标
    - `same_object_active_use`
      不再在 `verifier_blocked` 主路径里单独驱动动作物体自身的 later-state revisit
    - `measurement_target_revisit / phone_record_target_revisit`
      不再在 `verifier_blocked` 与 repeated textual fallback 主路径里单独驱动 exact target revisit
    - `same_object_active_use`
      也已从 repeated textual fallback 主路径中移除，不再单独驱动 action-object revisit
  - 只剩 observation gap、`missing_gap_types`、`verifier_missing`、`recommended_next_step`、`open_questions` 能继续驱动 planner
- [x] 与上述 consumer 清理对应的旧测试契约已改写为 observation-first 负约束：
  - `blocking_comparisons alone` 不能构成 unresolved evidence gap
  - `blocking_hypotheses alone` 不能构成 unresolved evidence gap
  - `blocking_comparisons alone` 不能凭空补出 `need_disambiguating_evidence`
  - `recover_open_questions` 不能仅凭 `blocking_hypotheses` 触发 targeted recovery
  - `continue_search_from_sufficiency` 不能仅凭 `blocking artifact` 或 `structured insufficient flag` 继续搜索
  - `resolution payload finish` 不能仅被 hypothesis score gap 这种候选竞争残影拦住
  - `text-rank fallback / failure-finish` 也不能仅被 hypothesis score gap 这种候选竞争残影拦住
  - `close-call recovery` 不能在 payload 没有 competitor 时，仅凭 hypothesis/comparison residue 触发
  - `competition_pressure` 不能再因为 `blocking_comparisons / score_gap` 进入 close-call
  - `competition_pressure` 只有在存在真实 observation gap 且覆盖仍浅时，才允许返回 close-call
  - `pairwise hidden-outcome detector` 不能因为 `needed_observation / answer` 单独触发
  - `weak generic claim detector` 的判定结果不能因为 `needed_observation / answer` 的加入而改变
  - `pending/latest/fallback candidate indices` 不能仅凭 structured hypotheses 自动启动 specialized pairwise
  - `structured specialized recovery tool` 不能在多选 gap 场景下仅凭 structured hypotheses 自动进入 pairwise
  - `verifier_blocked precondition recovery` 不能仅凭 blocker_hint 文本决定，必须服从 primary gap
  - `disable_legacy_specialized_recovery` 下的 specialized resume 不能仅凭 `future_use_close_call / pairwise_close_call` 文本分流
- [x] 本轮定向验证已经完成：
  - `future_outcome` 6 条定向收口测试：`6 passed`
  - `primary_gap` 3 条基础契约测试：`3 passed`
  - `explicit downstream object / heuristic fallback / evidence-first` 子集：`7 passed`
  - `timeline review / blocked state change` 子集：`2 passed`
  - `blocking/comparison consumer` 负约束子集：`4 passed`
  - `continue_search_from_sufficiency` 负约束子集：`4 passed`
  - `close-call finish/fallback gating` 负约束子集：`3 passed`
  - `close-call recovery gating` 负约束子集：`2 passed`
  - `candidate bootstrap de-hypothesis` 负约束子集：`6 passed`
  - `verifier_blocked precondition gap routing` 子集：`3 passed`
  - `disable_legacy verifier_blocked specialized routing` 子集：`4 passed`
  - `candidate ranking no longer uses blocking comparison pressure` 子集：`4 passed`
  - `existing pairwise specialized chain no longer depends on competition pressure` 子集：`3 passed`
  - `verifier_blocked no longer jumps to finalizer/mixed-horizon later target` 子集：`6 passed`
  - `verifier_blocked same-object active use no longer forces action-object revisit` 子集：`3 passed`
  - `measurement / phone exact-target revisit no longer overrides observation-first fallback` 子集：`4 passed`
  - `repeated textual fallback same-object active use no longer forces action-object revisit` 子集：`3 passed`
  - `verifier gap producer de-close-call` 子集：`3 passed`
- [x] 本轮 `planner/tests` observation-centric 迁移已继续推进：
  - fixture-only `future_outcome(fridge)` 的一组旧测试契约，已不再强制要求
    `query_spatial_context(fridge)` 或固定 late-followup tag
  - 这些场景现在按真实 observation state 分流：
    - 无可靠空间锚点时，允许保留 `sample_sparse_frames` / `retrieve_cached_artifacts`
    - 已有 later fixture/object anchor 时，允许直接进入 `query_spatial_context / query_object`
  - 对应定向回归：
    - `4 passed, 1089 deselected`
- [x] 本轮 `Phase 2` 小回归：
  - `pytest -q tests/test_graph_agent.py -k 'verifier_action_intent or primary_gap'`
  - `62 passed, 1033 deselected`
  - 其中 1 条旧测试已迁移为新的 observation-first 契约：
    - `heuristic_fallback` 不再保护旧的 `resolve_action_intent_future_use`
    - 而是保护 `primary_gap` 更早接管搜索动作
- [x] 本轮 `Phase 3` 子集回归：
  - `pytest -q tests/test_graph_agent.py -k 'verifier_blocker_hint or timeline_review_close_call_primary_gap or heuristic_fallback_primary_gap or future_use_resolution_continue_search_is_preempted_by_heuristic_primary_gap_before_generic_long_horizon_revisit or primary_gap_future_outcome_object_gap_prefers_object_query_without_long_horizon_anchor or primary_gap_future_outcome_fixture_gap_prefers_spatial_query_without_spatial_anchor or heuristic_fallback_late_primary_gap_with_explicit_downstream_object_prefers_gap_over_specialized_resume'`
  - `10 passed, 1085 deselected`
  - 对应 `primary_gap` 专项子集仍保持：
    - `29 passed, 1066 deselected`
- [x] 本轮 `Phase 3` gate-family 回归：
  - `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_measurement_target_hint or verifier_blocked_phone_record_target_hint or mixed_horizon_later_target_hint_does_not_use_needed_observation_alone or verifier_blocked_prefers_forced_transition_probe_does_not_read_needed_observation_text_alone or same_object_active_use'`
  - `6 passed, 1089 deselected`
  - 同时再次确认：
    - `pytest -q tests/test_graph_agent.py -k 'verifier_blocker_hint or heuristic_fallback_primary_gap or primary_gap and action_intent'`
    - `33 passed, 1062 deselected`
- [x] 本轮 repeated textual fallback 的最后一条 mixed-horizon 旧契约已迁移：
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'textual_fallback_mixed_horizon_prefers_local_visual_review_before_late_fixture_query or mixed_horizon_later_target_hint_does_not_use_needed_observation_alone or verifier_blocked_measurement_target_hint or verifier_blocked_phone_record_target_hint'`
  - 预期语义：
    - 在 only-local-window、且尚无可靠 later anchor 的情况下，
      planner 允许先走 `inspect_visual_evidence`
    - 不再因为候选答案文本里出现 `fridge` 就立刻跳到 `query_object/query_spatial_context`
- [x] 本轮开始收缩 planner 内部旧 blocker 命名层：
  - `_action_intent_verifier_blocker_hint(...)` 对
    `future_outcome / relation_confirmation / target_discovery`
    已优先回传中性家族名：
    - `future_gap_family`
  - 当前仍临时兼容旧名消费：
    - `future_use_close_call`
    - `pairwise_close_call`
  - 这一层的目标是先切断
    - `observation gap -> old specialized close-call label`
    的直接回灌
- [x] 本轮继续收缩 `infer_action_intent -> verifier_blocked` 的 legacy specialized recovery：
  - 当当前已经存在结构化 `primary_gap` 时，
    `infer_action_intent` 被 verifier 拦下后，
    不再因为历史上挂着 `resolve_action_intent_pairwise / resolve_action_intent_future_use`
    外壳就优先跳回 specialized resume
  - specialized resume 进一步降级为：
    - “没有结构化 gap 时的兜底恢复”
  - 新增负约束测试：
    - `test_planner_action_intent_verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume`
  - 对应定向回归：
    - `2 passed, 1094 deselected`
    - `6 passed, 1090 deselected`
- [x] 本轮继续收缩 `repeated failure / before_text_fallback` 入口：
  - `planner._build_action_intent_specialized_resolution_before_text_fallback(...)`
    不再只在“显式 downstream object”时才优先走 `primary_gap`
  - 当前只要已经存在结构化 `primary_gap`
    - `precondition`
    - `immediate_outcome`
    - `future_outcome`
    - `relation_confirmation`
    - `target_discovery`
    就会先尝试 gap 路由
  - 如果 gap 路由拿不到动作，当前入口直接返回 `None`
    不再让 specialized resume 抢回主路径
  - 新增负约束测试：
    - `test_planner_action_intent_before_text_fallback_immediate_outcome_gap_skips_specialized_resume_when_gap_route_unavailable`
  - 对应定向回归：
    - `4 passed, 1093 deselected`
    - `4 passed, 1093 deselected`
- [x] 本轮继续收缩 `state candidate plan` 中的 specialized 优先回拉：
  - `_select_state_candidate_plan(...)` 在下面这种场景下：
    - 已存在结构化 `primary_gap`
    - 当前最优候选是 `query_object / query_spatial_context`
    不再把 targeted gap tool 压回
    `resolve_action_intent_pairwise / resolve_action_intent_future_use`
  - 也就是说：
    - `structured primary_gap + targeted gap tool`
      现在优先级高于
      `legacy specialized resolution preference`
  - 对应测试契约已迁移：
    - `test_planner_state_candidate_keeps_object_revisit_over_specialized_resolution_for_structured_close_call`
  - 对应定向回归：
    - `2 passed, 1095 deselected`
    - `2 passed, 1095 deselected`
- [x] 本轮 `open-question recovery` 的 legacy specialized priority 又继续收缩：
  - `_recover_from_open_questions(...)` 里，
    结构化 `primary_gap` 现在优先经过：
    - `evidence_first`
    - `gap_late_followup`
    - `primary_gap recovery`
    - 最后才允许 specialized resume 兜底
  - 新增并稳定保护了 3 类 observation-centric 契约：
    - `same-object future_outcome`：
      若 gap 仍只锚定到动作对象自身，不允许把“重查同一对象”误当作新证据；
      保留已有 `pairwise` 主链
    - `no-anchor future_outcome`：
      若 gap 既没有新对象锚点，也没有新空间锚点，不允许在 `pairwise -> future_use`
      之间无依据切链；保留最近的 specialized chain
    - `suppressed_answer_conditioned_target`：
      若 target 已因去答案条件化被抑制，不允许再从旧 specialized fallback 把它捞回来；
      只能退回 `sample_sparse_frames / retrieve_cached_artifacts / local followup`
  - 对应定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'open_question_recovery_does_not_drop_pairwise_close_call_into_generic_followup_when_specialized_resolution_is_already_established or open_question_recovery_suppressed_answer_conditioned_target_prefers_observation_centric_followup_before_specialized_resume or open_question_recovery_explicit_downstream_object_prefers_gap_over_existing_pairwise_resolution or open_question_recovery_blocked_state_change_explicit_downstream_object_skips_specialized_resume or open_question_recovery_prefers_primary_gap_before_legacy_open_question_markers or open_question_recovery_fixture_only_gap_without_latest_resolution_prefers_local_followup or open_question_recovery_fixture_only_gap_with_latest_resolution_prefers_local_followup or open_question_recovery_can_route_precondition_gap_with_latest_resolution_payload_and_without_legacy_markers or open_question_recovery_can_route_precondition_gap_with_latest_resolution_and_only_structured_gap or open_question_recovery_can_route_immediate_gap_with_latest_resolution_payload_and_without_legacy_markers or open_question_recovery_fixture_only_structured_gap_prefers_local_followup_even_with_latest_resolution or open_question_recovery_fixture_only_structured_gap_prefers_local_followup_when_next_step_is_alternative_path or open_question_recovery_prefers_target_discovery_gap_when_next_step_is_location_evidence or open_question_recovery_prefers_relation_confirmation_gap_when_next_step_is_location_evidence or verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume or state_candidate_keeps_object_revisit_over_specialized_resolution_for_structured_close_call'`
    - `16 passed, 1081 deselected`
  - 对应较大子集回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent and (open_question_recovery or verifier_blocker_hint or before_text_fallback or state_candidate or verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume or textual_fallback_mixed_horizon_prefers_local_visual_review_before_late_fixture_query)'`
    - `45 passed, 1052 deselected`
- [x] 本轮又删除了一条真实的 repeated textual fallback 主路径 consumer：
  - `planner.py` 在
    `last_tool == rank_choices_from_state and action_intent_text_fallback_ready`
    这条分支里，
    已不再先消费以下旧产物去决定下一步搜索：
    - `finalize_withheld_generic_access_or_space_revisit`
    - `finalize_withheld_generic_relocation_or_storage_revisit`
    - `unresolved_rerank_downstream_target_revisit`
    - `unresolved_rerank_downstream_fixture_revisit`
    - `unresolved_rerank_long_horizon_revisit`
  - 也就是说，`rank_choices_from_state` 之后的 why/action_intent 主路径，
    现在只能继续交给：
    - `precondition backfill`
    - `mixed-horizon / transition probe`
    - `evidence_first`
    - `strict_text_fallback_recovery`
    - `recover_from_open_questions`
    - `state_candidate`
    这些 observation-centric 恢复链
  - 新增负约束测试：
    - `test_planner_action_intent_textual_rank_does_not_consume_finalizer_or_unresolved_rerank_revisit_markers`
  - 对应定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'textual_rank_evidence_first_explicit_downstream_object_prefers_gap_over_existing_pairwise_resolution or textual_rank_does_not_consume_finalizer_or_unresolved_rerank_revisit_markers or textual_finish_sufficiency_fixture_only_future_outcome_prefers_state_candidate_before_generic_query_time or repeated_loop_ignores_finish_recovery_when_structured_sufficiency_still_requires_search or open_question_recovery or verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume'`
    - `28 passed, 1070 deselected`
- [x] finalize mixed-horizon later-target 的一条旧契约也已迁移：
  - 当 `fridge` 的后续轨迹锚点已经在 observation state 中出现时，
    observation-centric 路由允许直接走 `query_spatial_context`
  - 不再强制要求先走 `sample_sparse_frames`
- [x] 本轮 `verifier` 结构审计已切掉一条真实 answer-conditioned consumer：
  - `GraphAgentVerifier._heuristic_verify(...)`
    不再根据
    `top/runner-up + score_gap + missing_observations`
    自动追加 `need_disambiguating_evidence`
  - 已物理删除旧 helper：
    `GraphAgentVerifier._action_intent_hypotheses_still_need_disambiguation(...)`
  - 当前 why 验证主路径继续只允许由：
    - `open_questions`
    - `missing grounding types`
    - `evidence_gaps`
    - `budget / sufficient state`
    驱动缺口，而不是由候选竞争摘要回流
- [x] 对应本轮回归结果：
  - `pytest -q tests/test_graph_agent.py -k 'verifier_action_intent'`
  - `28 passed, 1065 deselected`
- [x] 本轮 `state -> planner prompt` 的答案产物污染也已继续收口：
  - `AgentState.snapshot()` 不再暴露以下字段给 planner 的模型提示：
    - `verification_history`
    - `action_intent_hypotheses`
    - `action_intent_trace`
    - `final_metadata`
  - 这些字段仍保留在 `export_session_memory()` 中用于会话恢复与离线审计，
    但不再直接进入 planner 的在线思考上下文
  - 新增负约束测试：
    - `test_agent_state_snapshot_excludes_answer_conditioned_action_intent_artifacts`
- [x] 本轮 `planner` 中剩余的 `competition_pressure` 在线 gate 也已继续削弱：
  - `top_hypothesis_target_hint / competing_hypothesis_target_hint / comparison_blocker_target_hint`
    当前都已是空壳，不再提供在线 target hint
  - `state_candidate` 选路中不再要求
    `competition_pressure.is_close_call`
    才允许保留 existing specialized resolution
  - `primary_gap` 的某条 `future_outcome -> query_object` 恢复链
    也不再依赖 `competition_pressure.is_close_call`
  - 这意味着 `blocking_comparisons / score_gap`
    不再作为在线搜索动作的硬性 gate，
    只剩下 observation gap、resolution payload 自身缺证状态与预算约束
  - 对应旧测试契约已迁移：
    - `future gap prefers later outcome recovery`
      不再强制要求 `query_object/query_spatial_context`
      允许 observation-first 的 `sample_sparse_frames`
- [x] 本轮 `executor` 中一条 hypothesis 直出最终答案的链路也已删除：
  - `GraphAgentExecutor._resolve_structured_final_candidate(...)`
    不再优先使用 `verification.action_intent_hypotheses[0]`
    直接物化 `structured_final_candidate`
  - 当前最终候选只继续允许来自：
    - `tool_trace` 中已完成且未声明缺证的原始工具结果
    - `rank_choices_from_state` 的 budget-exhausted fallback
  - 这意味着 `top_hypothesis / runner_up`
    继续保留为审计/metadata，而不再作为最终答案主来源
  - 对应定向回归：
    - `7 passed, 1087 deselected`
- [x] 当前 `action_intent` 专项回归结果：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `640 passed, 454 deselected`
- [x] `verifier_blocked -> infer_action_intent` 的一个残留 blocker-hint-first 测试场景已收口：
  - 失败点不是主逻辑回退，而是测试被更早的 `primary_gap` 路由提前截走
  - 已将该用例改成真正隔离 `infer_action_intent` 分支的 observation-first 断言
  - 对应定向回归：
    - `verifier_blocked_*existing_pairwise_over_future_use_hint`
    - `verifier_blocked_recovery_*specialized_*_close_call*`
    - `4 passed`
- [x] `weak_late_anchor / nonexclusive_concrete_late_anchor` 这一轮残留测试已正式收口：
  - 当前残留点不是旧 late-anchor helper 仍在主导，而是测试把通用 observation-based `cached_long_horizon_revisit` 也误算成失败
  - 已将最后一条用例改成真正隔离“旧 helper 不再单独触发”的负约束断言
  - 对应定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'weak_late_anchor or nonexclusive_concrete_late_anchor'`
    - `6 passed, 1088 deselected`
  - 收口后再次确认专项全量：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `640 passed, 454 deselected`
- [x] `needed_observation` 已进一步从 planner 搜索主路径中删除为 dead legacy：
  - 已从 `planner.py` 物理删除以下旧 helper：
    - `_action_intent_needed_observation_text(...)`
    - `_action_intent_needed_observation_target_hint(...)`
    - `_action_intent_needed_observation_relation_hint(...)`
    - `_build_action_intent_needed_observation_target_revisit_decision(...)`
    - `_build_action_intent_needed_observation_relation_revisit_decision(...)`
  - 已同步删除/改写测试中对这些 helper 的直接调用与 monkeypatch，避免继续维护已去答案化后的旧接口空壳
  - 对应定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'needed_observation and action_intent'`
    - `51 passed, 1040 deselected`
  - 当前专项全量回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `637 passed, 454 deselected`
  - `passed` 数减少不是主路径回退，而是删除了 3 条只验证已移除 legacy helper 的旧接口测试
- [x] 本轮再次确认：
  - 当前 `action_intent` 全量回归已经恢复为
    `639 passed, 454 deselected`
  - 本轮新增工作以“迁移旧测试契约到 observation-first 断言”为主，
    没有把 planner 逻辑回退到 `fixture-only future_outcome -> 固定 fridge 空间查询`

## Phase 0：建立执行边界与冻结范围

### 目标

- 明确本轮允许修改的函数范围
- 停止扩散式改动
- 把旧测试先分类，而不是继续和它们缠斗

### 输入

- 主清单：[/22liushoulong/agent/hd-epic/docs/plan/22_deanswer_conditioned_agent_checklist_zh.md](/22liushoulong/agent/hd-epic/docs/plan/22_deanswer_conditioned_agent_checklist_zh.md)
- 核心文件：
  - [/22liushoulong/agent/hd-epic/food_agent/agent/planner.py](/22liushoulong/agent/hd-epic/food_agent/agent/planner.py)
  - [/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py](/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py)
  - [/22liushoulong/agent/hd-epic/food_agent/agent/verifier.py](/22liushoulong/agent/hd-epic/food_agent/agent/verifier.py)
  - [/22liushoulong/agent/hd-epic/food_agent/agent/state.py](/22liushoulong/agent/hd-epic/food_agent/agent/state.py)
  - [/22liushoulong/agent/hd-epic/tests/test_graph_agent.py](/22liushoulong/agent/hd-epic/tests/test_graph_agent.py)

### 输出

- 本轮允许修改的函数名单
- 本轮禁止碰的模块名单
- 旧测试分类表：`保留 / 改写 / 删除`

### 完成标准

- 后续每轮 goal 都有清晰修改边界
- 不再出现“为了修一个链路，顺手补三个专项规则”

### Phase 0 执行勾选

- [x] 明确本轮允许修改的函数范围
- [x] 明确本轮禁止碰的模块范围
- [x] 产出旧测试分类表：`保留 / 改写 / 删除`
- [x] 锁定首个最小真实缺口：`generic_hand_free`
- [x] 将 Phase 0 审计结论写入本计划文档

### Phase 0 审计结论

以下内容基于当前工作树实际状态，不是理想状态。

#### 允许修改的模块范围

本阶段后续若继续推进 why / action-intent 去答案条件化，只允许优先修改以下文件中的对应函数族：

- [/22liushoulong/agent/hd-epic/food_agent/agent/planner.py](/22liushoulong/agent/hd-epic/food_agent/agent/planner.py)
  - `_action_intent_*`
  - `_build_action_intent_*`
  - `_recover_action_intent_*`
  - `_resume_action_intent_*`
- [/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py](/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py)
  - `_action_intent_*`
  - `_resolve_action_intent_*`
- [/22liushoulong/agent/hd-epic/food_agent/agent/verifier.py](/22liushoulong/agent/hd-epic/food_agent/agent/verifier.py)
  - `_action_intent_*`
- [/22liushoulong/agent/hd-epic/food_agent/agent/state.py](/22liushoulong/agent/hd-epic/food_agent/agent/state.py)
  - `action_intent trace / working_memory` 同步与导出相关函数
- [/22liushoulong/agent/hd-epic/tests/test_graph_agent.py](/22liushoulong/agent/hd-epic/tests/test_graph_agent.py)
  - why / action-intent 相关测试

#### 当前禁止碰的模块

在 observation-centric 主链还没有稳定之前，以下模块暂时冻结，不允许顺手扩散修改：

- [/22liushoulong/agent/hd-epic/food_agent/model_client.py](/22liushoulong/agent/hd-epic/food_agent/model_client.py)
- [/22liushoulong/agent/hd-epic/food_agent/agent/executor.py](/22liushoulong/agent/hd-epic/food_agent/agent/executor.py)
- [/22liushoulong/agent/hd-epic/food_agent/tools/agent_toolbox.py](/22liushoulong/agent/hd-epic/food_agent/tools/agent_toolbox.py)
- [/22liushoulong/agent/hd-epic/docs/plan/00_index_zh.md](/22liushoulong/agent/hd-epic/docs/plan/00_index_zh.md)
- 任何与 why / action-intent 无关的任务路由、数据脚本与评测脚本

原因不是这些模块没问题，而是当前最主要的污染链仍在 `planner / graph_agent / verifier / state` 内部；先切主链，避免再次把问题扩散到外围模块。

#### 当前确认仍残留的旧链

当前最明显的残留不是某一个 bug，而是一整组仍在运行态消费答案语义的 consumer 链。

第一组残留：`generic_hand_free` 旧链还在 `planner.py` 中活跃

- `_action_intent_verifier_blocked_hand_free_target_hint(...)`
- `_build_action_intent_verifier_blocked_hand_free_target_revisit_decision(...)`
- `_action_intent_unresolved_rerank_hand_free_object_hint(...)`
- `_build_action_intent_unresolved_rerank_hand_free_object_revisit_decision(...)`
- 多个 `next_action(...)` 分支仍直接调用这些 helper

它们虽然已经不像更早那样直接从 `choice_text` 生造目标，但仍然保留了：

- `generic hand-free` 这个专项冲突标签
- `hand_free vs downstream use` 这个专项恢复路径
- 由 finalizer / unresolved-rerank / verifier-blocked 触发的专门 consumer

这说明当前系统仍然没有真正退化到：

- 只由 `gap + observation + budget` 决定动作

而是仍保留：

- `special conflict family -> dedicated revisit chain`

第二组残留：`graph_agent.py` 里仍存在 hand-free 专项分类器

- `_action_intent_choice_is_pure_hand_free_enablement(...)`
- `_action_intent_choice_is_hand_free_enablement(...)`
- `timeline_review_hand_free_or_fixture_gap`
- `hand_free_enablement` 类别与 bias

这些函数和类别本身未必都要立刻删除，但至少说明：

- 当前 finalizer / rerank 仍在显式使用专项答案语义分桶

第三组残留：close-call / competition 仍在 planner 主链中大量存在

- `blocking_hypotheses`
- `blocking_comparisons`
- `score_gap`
- `runner_up`
- `future_use_close_call`
- `pairwise_close_call`

这部分还没有在本阶段处理，但已经确认：

- 它们仍然是后续 Phase 2 / Phase 3 必须继续削弱和切断的主对象

#### 当前旧测试分类表

当前 `tests/test_graph_agent.py` 里 why / action-intent 相关旧测试，不再统一视为“保护回归”，而要按三类处理：

`保留`

- 负约束测试
- observation-grounded gap 测试
- “没有 observation anchor 时不能继续搜”的测试
- “有真实轨迹/空间锚点时允许继续搜”的测试

`改写`

- 名字或断言中带有 `prefers_*`
- `*_revisit_decision`
- `timeline_review_*_gap`
- `close_call_*`
- `future_use_*`
- `pairwise_*`

这类测试不能再保护“应该偏好哪条专项链”，只能改成：

- 不能由答案产物单独触发该链
- 只有 observation gap 存在时才允许进入该链

`删除`

- 任何本质上在保护以下旧行为的测试：
  - `generic hand-free -> downstream object revisit`
  - `best/runner-up -> fixed target hint`
  - `choice semantic -> fixed followup route`
  - `finalizer marker -> fixed consumer`

#### 当前最小下一步

Phase 0 审计后的最小真实缺口已经明确：

- 下一轮优先删除 `generic_hand_free` 这条 planner consumer 链

顺序固定为：

1. 删 `planner.py` 中 `generic_hand_free` consumer
2. 删或降级对应的 `graph_agent.py` producer / marker
3. 把旧测试从“保护 generic hand-free 行为”改成“禁止答案语义单独驱动搜索”

在这条链删干净之前，不再继续扩散去修新的专项冲突家族。

#### Phase 0 后续实际进展补充

本轮已经完成的，不再是假设，而是当前代码中的真实状态：

- [x] `planner.py` 中 `generic_hand_free` 的主路径 consumer 已切掉一批
  - 已移除 repeated textual fallback / verifier-blocked / unresolved-rerank / finalizer-after-resolution 中对 hand-free 专项 revisit 的直接调用
- [x] `tests/test_graph_agent.py` 中一批直接保护
  - `generic hand-free -> downstream object revisit`
  - `generic hand-free -> same-object use revisit`
  这类旧契约的测试已经删除或改成负约束
- [x] `graph_agent.py` 中 deterministic finalizer 已收紧
  - `so left hand is free` / `to free one hand` 这类 broad generic enablement
  - 当同一轮 `candidate_evidence` 已出现更直接的下游用途/同物体用途反证时
  - 不再允许直接 deterministic finish
- [x] `graph_agent.py` 中 `generic_hand_free` marker producer 已删除
  - finalizer 不再向 `working_memory` 写回 `action_intent_resolution_withheld_for_generic_hand_free_enablement=...`

当前仍未完成、但已明确的残留已收缩为一类：

- [ ] `graph_agent.py` 里仍保留 finalizer / unresolved-rerank 的专项 override 链
  - 若干 `*_override_*`
  - 若干 `*_overclaim_*`
  - 若干 `*_choice_is_*`
  - 其中 `generic_access_or_space_overclaim` 已不再读取 competitor `candidate_evidence`，
    `generic_relocation_or_storage_overclaim` 也已不再读取 competitor `candidate_evidence`，
    `workspace_or_final_placement_claim -> planner close-call shortcut` 也已删除，
    `finalizer withheld marker -> planner long-horizon bias` 也已删除，
    但其余 `mixed_horizon / workspace_or_final_placement`
    仍需继续按同样方式去 candidate-conditioned 化

所以当前状态应准确表述为：

- `generic_hand_free` 这条旧链在 `graph_agent.py` 中的分类函数与旧 reason 残留已经删净
- planner 侧 dead helper、hand-free route/probe 主链、graph-agent 侧 producer/marker、旧 reason 测试依赖都已进一步收口
- 下一轮应该继续做“删 graph-agent 中其余专项 override / overclaim / choice_is 链”，而不是再回到 hand-free 家族补局部规则

#### 本轮新增收口结果

- [x] `planner.py` 中以下 hand-free dead helper 已全部删除：
  - `_action_intent_verifier_blocked_hand_free_target_hint(...)`
  - `_build_action_intent_verifier_blocked_hand_free_target_revisit_decision(...)`
  - `_action_intent_unresolved_rerank_hand_free_object_hint(...)`
  - `_build_action_intent_unresolved_rerank_hand_free_object_revisit_decision(...)`
- [x] `tests/test_graph_agent.py` 中只保护
  - `hand_free -> downstream fixture/object`
  - `hand_free -> specialized revisit`
  这类旧专项行为的测试也已删除
- [x] 当前 hand-free 定向回归结果：
  - `pytest -q tests/test_graph_agent.py -k 'hand_free and action_intent'`
  - `8 passed, 1077 deselected`
- [x] `graph_agent.py` 已去掉 hand-free 专项加分
- [x] `graph_agent.py` 已将 `timeline_review_hand_free_or_fixture_gap` 收敛为通用 `timeline_review_next_use_gap`
- [x] `graph_agent.py` 已移除 `generic_hand_free` marker 写回链
- [x] graph-agent 侧定向回归结果：
  - `pytest -q tests/test_graph_agent.py -k 'hand_free and action_intent or fixture_gap_revisits_downstream_fixture_node or unresolved_rerank_hand_free_gap_without_observed_fixture_does_not_query_downstream_fixture'`
  - `10 passed, 1075 deselected`
- [x] `generic_hand_free` marker 删除后的定向回归结果：
  - `pytest -q tests/test_graph_agent.py -k 'finalizer_withholds_generic_hand_free_when_specific_downstream_object_exists or hand_free and action_intent'`
  - `8 passed, 1077 deselected`
- [x] `graph_agent.py` 中 hand-free 专项分类函数已删除：
  - `_action_intent_choice_is_pure_hand_free_enablement(...)`
  - `_action_intent_choice_is_hand_free_enablement(...)`
- [x] `planner.py` 与 `tests/test_graph_agent.py` 中 hand-free 旧 reason 兼容残留已收口为：
  - `timeline_review_next_use_gap`
- [x] 当前新增定向回归结果：
  - `pytest -q tests/test_graph_agent.py -k 'hand_free and action_intent or unresolved_rerank_downstream_fixture_hint or unresolved_rerank_fixture_gap_revisits_downstream_fixture_node or unresolved_rerank_generic_hand_free_without_observed_object_does_not_query_choice_target'`
  - `10 passed, 1075 deselected`
- [x] `planner.py` 中 hand-free category 驱动的 route / probe 主链已收口为通用 next-use gap 判定：
  - `_action_intent_has_hand_free_future_use_conflict(...)` 已替换为 `_action_intent_has_next_use_followup_gap(...)`
  - `transition_probe` 不再因泛化的 hand-free/future-use 歧义自动升级，而要求更明确的局部 next-step 证据
- [x] `graph_agent.py` 中 phone-specific 的运行态专项 override 已删除：
  - `_override_generic_measure_with_exact_record_target_candidate(...)`
  - `_action_intent_choice_is_generic_measure_phone_goal(...)`
  - `_action_intent_choice_supports_exact_record_target(...)`
  - 对应旧测试已改写为 observation-first 的保守 withheld 预期，而不是继续保护 runtime answer override
- [x] `graph_agent.py` 中 towel-specific 的两条运行态专项 override 已删除：
  - `_override_generic_hand_wiping_with_explicit_single_hand_drying(...)`
  - `_override_generic_towel_use_with_simple_relocation(...)`
  - 对应旧测试已改写为 observation-first 的保守 withheld 预期，而不是继续保护 runtime relocation override
- [x] `graph_agent.py` 中 inspection-specific 的运行态专项 override 已删除：
  - `_override_weak_inspection_with_explicit_later_outcome_candidate(...)`
  - 删除后保留的是更通用的 later-outcome evidence 打分强化，而不是 inspection-specific runtime answer override
- [x] `graph_agent.py` 中 transfer/downstream-pickup 的运行态专项 override 已删除：
  - `_override_downstream_followup_with_direct_enablement_candidate(...)`
  - 删除后保留的是更通用的 downstream-pickup contradiction penalty，而不是 transfer-specific runtime answer override
- [x] `graph_agent.py` 中 prior-answer carryover 的运行态专项 override 已删除：
  - `_resolve_prior_direct_action_object_intent(...)`
  - 删除后不再允许从更早一轮 `infer_action_intent.best_index` 直接回流并改写当前 unresolved-rerank 结果
- [x] 当前专项全量回归结果：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `632 passed, 453 deselected`

这意味着当前 `generic_hand_free` 旧链已经从：

- `graph_agent producer -> planner specialized consumer -> old tests`

收缩成：

- `graph-agent finalizer / unresolved-rerank specialized override residual`

下一轮不应该再回到 hand-free 家族补洞，而应直接转去清 `graph_agent.py` 中剩余的专项 override / overclaim / choice_is 链。

---

## Phase 1：清空 Finalizer 里的旧答案语义补丁

### 目标

- 将 `graph_agent` finalizer 退化为保守裁决层
- 不再从候选答案语义派生搜索目标

### 重点文件

- [/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py](/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py)

### 优先清理函数族

- `_resolve_action_intent_resolution_answer(...)`
- 所有 `*_should_withhold_*`
- 所有 `*_overclaim_*`
- 所有 `*_weak_*`
- 所有 `*_override_*`
- 所有 `*_choice_is_*`

### 必删行为

- finalizer 产出 `target_object / target_fixture / needed_observation`
- finalizer 根据 competitor 语义直接改判答案
- finalizer 写回供 planner 消费的专项 marker

### 完成标准

- finalizer 只负责：
  - `finish`
  - `withhold`
  - `fallback`
- 不再为 planner 提供答案导向搜索线索

### 测试策略

- 删除：
  - “某种 choice 文本应导致 withhold/override/marker”的测试
- 改写：
  - 只有 `choice_text`、没有真实 observation 时，finalizer 不能生成 target 或下一步动作线索
  - 弱证据 close call 时，只能 withheld，不能直接换答案
- 保留：
  - 有真实观测证据时仍能正常 finish 的测试

### 阶段结束验证

- 代码里不再存在新的 finalizer 专项 marker 生产链
- 定向负约束测试证明 finalizer 不会从答案语义派生搜索目标

### Phase 1 执行勾选

- [x] `generic_hand_free` 的 finalizer 直接收口已收紧为保守 withheld
- [x] `generic_hand_free` 的 finalizer marker 写回链已删除
- [x] 删除其余 `graph_agent` finalizer 中仍基于专项语义的 producer / marker
- [x] 删除或降级 hand-free 分类函数本身，或明确降级为仅 observation label
- [x] 清理其余 `*_overclaim_* / *_choice_is_* / *_override_*` 中仍直接服务答案语义竞争的链
- [x] 让 finalizer 只保留 `finish / withhold / fallback`

### Phase 1 本轮收口补充

- [x] `graph_agent.py` 中剩余两条 unresolved-rerank 运行态专项 override 已删除：
  - `_override_generic_space_with_exact_immediate_use_candidate(...)`
  - `_override_generic_hidden_access_with_exact_revealed_target_candidate(...)`
- [x] 删除后不再允许 unresolved-rerank 通过“generic make-space / access 候选 + competitor exact choice”直接翻盘
- [x] 对应旧测试契约已进一步收口：
  - 不再保护“generic make-space 必须翻到 exact fixture enablement”
  - 改为 observation-first 的保守 finish 行为
- [x] 当前专项全量回归结果：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `632 passed, 453 deselected`

---

## Phase 2：统一 Primary Gap Schema

### 目标

- 停止使用 answer gap
- 为 observation gap 建立唯一入口

### 重点文件

- [/22liushoulong/agent/hd-epic/food_agent/agent/planner.py](/22liushoulong/agent/hd-epic/food_agent/agent/planner.py)
- [/22liushoulong/agent/hd-epic/food_agent/agent/verifier.py](/22liushoulong/agent/hd-epic/food_agent/agent/verifier.py)
- [/22liushoulong/agent/hd-epic/food_agent/agent/state.py](/22liushoulong/agent/hd-epic/food_agent/agent/state.py)

### 优先清理对象

- `planner.py::_action_intent_primary_gap(...)`
- `planner.py::_recover_action_intent_via_primary_gap(...)`
- `verifier.py::_action_intent_build_evidence_gaps(...)`
- 所有把 `needed_observation / choice / comparison / hypothesis` 回灌成 gap 的逻辑

### 完成标准

- `primary_gap` 不能再包含来源于 `choice/comparison/hypothesis` 的 target
- `needed_observation` 只能是解释字段，不能主导主路径
- 搜索缺口来源只能是：
  - 观测状态
  - 轨迹闭合状态
  - 空间关系状态
  - 窗口覆盖状态
  - 预算状态

### 测试策略

- 删除：
  - “某个 blocker/choice 导致 primary gap 指向 fridge/scale/tap”的测试
- 改写为负约束：
  - 没有 observation anchor 时，primary gap 不能产生 object/fixture target
  - `needed_observation` 非空也不能单独生成主 gap
- 新增通用正向：
  - `object_track_unclosed`
  - `window_coverage_missing`
  - `relation_unobserved`

### 阶段结束验证

- `primary_gap` 所有字段都能追溯到 observation，而不是答案产物

### Phase 2 执行勾选

- [x] 为 `primary_gap` 建立统一 observation-centric schema
- [x] 删除 `primary_gap` 中残留的 `choice/comparison/hypothesis -> target` 回灌链
- [x] 将 `needed_observation` 降级为解释字段，不再主导主路径
- [x] 让 `primary_gap` 只来自观测状态、轨迹闭合、空间关系、窗口覆盖、预算状态
- [x] 补充 `object_track_unclosed / window_coverage_missing / relation_unobserved` 通用测试
- [x] 通过阶段回归并回写文档进度

### Phase 2 当前收口进展

- [x] `followup_route` 不再默认从 `pending/structured pairwise` 这类候选竞争残影反推主恢复路径
- [x] `pair_needs_outcome_resolution / needs_future_use_evidence`
  已从“看 candidate pair 语义”收缩为“看 `primary_gap` 类型”
- [x] `result_is_close_call_for_recovery`
  已不再要求 `best_index / competitor / score_gap` 才能解释恢复
  现在优先读取：
  - `primary_gap`
  - `blocker_hint`
  - `direct_effect / downstream_action / decisive_observation`
- [x] 一组 `verifier_blocked_finish / low_confidence finish` 旧测试契约
  已迁移为 observation-first 负约束：
  - 允许 `generic recovery`
  - 但不允许再回退到答案竞争驱动的 `future_use / pairwise` specialized resume
- [x] 对应定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'followup_route or pair_needs_outcome_resolution or needs_future_use_evidence or result_is_close_call_for_recovery or verifier_blocked_finish or recover_if_low_confidence_verifier_blocked_finish'`
  - `32 passed, 1070 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and primary_gap and verifier_blocked'`
  - `2 passed, 1100 deselected`
- [x] 本轮继续收缩 `Phase 2` 的 runtime 旧链：
  - `planner._action_intent_primary_gap_source_is_answer_conditioned(...)`
    不再把中性 family `future_gap_family` 视为 answer-conditioned source
  - `planner._action_intent_competing_pair_still_needs_disambiguation(...)`
    不再调用 `action_intent_needs_future_use_resolution(...) / action_intent_needs_pairwise_resolution(...)`
    去读取题目/选项语义
    现在只看：
    - `primary_gap.gap_type`
    - 归一化后的 `blocker_hint`
  - `planner._action_intent_result_needs_generalized_disambiguation(...)`
    不再通过 `action_intent_conflict_profile(...)` 的候选语义冲突来决定是否继续补证
    现在只看：
    - `primary_gap`
    - `blocker_hint`
    - 是否已有 precondition / post-action grounding
    - 结果文本是否仍属 non-exclusive / indecisive
    - 当前 confidence
  - `planner._action_intent_verifier_blocker_hint(...)`
    新增 blocker 归一化：
    - `future_use_close_call -> future_gap_family`
    - `pairwise_close_call -> post_action_evidence`
    这样历史 summary 中残留的旧名字也不会再把 runtime 拉回 specialized 旧链
- [x] 本轮 verifier fallback gate 也同步去答案化：
  - `verifier._action_intent_textual_rank_fallback_can_finish(...)`
    不再读取：
    - `action_intent_requires_strict_visual_disambiguation(...)`
    - `action_intent_needs_precondition_context(...)`
    - `action_intent_needs_future_use_resolution(...)`
    - `action_intent_needs_pairwise_resolution(...)`
    去用题目/选项语义决定 finish
  - 当前只看：
    - `primary_gap`
    - `blocker_hint`
    - precondition grounding
    - post-action grounding
    - unresolved secondary conflicts
- [x] 本轮 verifier grounding gate 继续去答案化：
  - `verifier._action_intent_missing_grounding_types(...)`
    不再根据 `question/choices` 调
    `action_intent_needs_future_use_resolution(...) / action_intent_needs_pairwise_resolution(...) / action_intent_needs_precondition_context(...)`
    来决定缺什么 grounding
    现在只看：
    - `primary_gap_type`
    - `why_blocker`
    - 最近 payload 是否缺 direct post-action evidence
    - 最近 payload 是否表达 later-outcome uncertainty
  - `verifier._action_intent_has_sufficient_grounding_for_stable_answer(...)`
    也不再通过题目/选项语义判断 stable answer 是否成立
    现在只看：
    - `primary_gap_type`
    - `why_blocker`
    - payload 的 observation uncertainty
    - precondition / post-action grounding
  - 因而 `verifier.py` 顶部对应的 answer-conditioned helper import
    已从 runtime 路径移除
- [x] 本轮新增回归：
  - `pytest -q tests/test_graph_agent.py -k 'blocker_hint_alone_does_not_create_future_outcome_gap or gap_source_is_not_driven_by_specialized_tool_name_alone or verifier_blocked_prefers_forced_transition_probe or primary_gap_future_outcome_records_trace_for_long_horizon_recovery or verifier_blocked_future_outcome_without_observation_anchor_does_not_escalate_to_long_horizon_query_after_window_expands or fixture_only_primary_gap_prefers_late_anchor_spatial_query_over_current_time_probe or level_zero_budget_does_not_keep_deferring_long_horizon_query_after_followup_already_exists or level_zero_budget_does_not_keep_deferring_long_horizon_query_after_local_new_frames_observed or level_zero_budget_does_not_keep_deferring_long_horizon_query_after_timeline_review or measurement_target_hint_does_not_use_needed_observation_alone or phone_record_target_hint_does_not_use_needed_observation_alone or mixed_horizon_later_target_hint_does_not_use_needed_observation_alone'`
  - `12 passed, 1097 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (verifier_blocker_hint or primary_gap or before_text_fallback or pending_resolution_tool or structured_specialized_recovery_tool or close_call or why_blocker)'`
  - `60 passed, 1049 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'textual_rank_fallback or before_text_fallback or why_blocker or primary_gap'`
  - `36 passed, 1073 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (verifier_blocked or close_call or pending_resolution_tool or structured_specialized_recovery_tool)'`
  - `84 passed, 1025 deselected`
  - `python -m py_compile food_agent/agent/verifier.py food_agent/agent/planner.py tests/test_graph_agent.py`
  - 通过
- [x] 本轮继续切断 `resolution payload -> competitor/candidate` 的答案残影入口：
  - `planner._latest_action_intent_candidate_indices(...)`
    已停止从 `second_best_index` 直接补齐 runtime candidate 集；
    现在只接受：
    - `candidate_indices`
    - `best_index`
    - `candidate_evidence[].index`
    - specialized tool 失败后的 `args.candidate_indices`
  - `planner._action_intent_route_candidate_indices(...)`
    已停止用 `best_index + second_best_index` 直接构造 top pair；
    现在先按 observation-side `candidate_evidence` 排序，再做 semantic rescue
  - `planner._action_intent_competing_candidate_index(...)`
    已停止优先读取 `second_best_index / losing_index`
    现在只从 observation-ranked candidates 中取 runner-up
  - `verifier._action_intent_competing_candidate_index(...)`
    也同步切掉 `second_best_index / losing_index` 直驱 competitor 的入口
- [x] 本轮新增负约束测试：
  - `test_planner_action_intent_latest_candidate_indices_do_not_bootstrap_from_second_best_only`
  - `test_planner_action_intent_competing_candidate_index_prefers_observation_rank_over_second_best_only`
- [x] 对应本轮 candidate 去残影回归：
  - `pytest -q tests/test_graph_agent.py -k 'latest_candidate_indices_do_not_bootstrap_from_second_best_only or competing_candidate_index_prefers_observation_rank_over_second_best_only or latest_candidate_indices_do_not_bootstrap_from_structured_hypotheses or latest_candidate_indices_can_bootstrap_from_structured_future_outcome_gap_without_payload_or_hypotheses or pending_candidate_indices_do_not_bootstrap_from_structured_hypotheses or pending_candidate_indices_can_bootstrap_from_structured_pairwise_gap_without_payload_or_hypotheses'`
  - `6 passed, 1105 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (candidate_indices or pairwise_candidates or future_use_candidates or specialized_resolution or competing_candidate_index)'`
  - `30 passed, 1081 deselected`
- [x] 本轮 planner 的 `strict visual disambiguation` 主路径也继续去答案化：
  - `planner._build_initial_action_intent_transition_probe_decision(...)`
    不再直接调 `action_intent_requires_strict_visual_disambiguation(...)`
    现在改为通过 observation-centric helper
    `planner._action_intent_needs_observation_centric_transition_recovery(...)`
    判定是否需要 transition probe
  - `planner._action_intent_requires_followup(...)`
    已不再通过 `action_intent_conflict_profile(...).active_categories`
    去做 why followup 的主判定
    现在优先看：
    - `primary_gap`
    - `why_blocker`
    - `timeline_review_bias_profile`
    - direct / indecisive post-action evidence
  - `planner` 中 3 处 text fallback / strict text fallback runtime 分支
    已从：
    - `action_intent_requires_strict_visual_disambiguation(...)`
    切到：
    - `planner._action_intent_needs_observation_centric_transition_recovery(...)`
  - 因而 `planner.py` 顶部对应的
    `action_intent_requires_strict_visual_disambiguation`
    import 已从 runtime 路径移除
- [x] 本轮对应回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (verifier_blocked or close_call or pending_resolution_tool or structured_specialized_recovery_tool or before_text_fallback or textual_rank)'`
  - `97 passed, 1012 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'initial_route_prefers_mixed_horizon_transition_probe_for_check_vs_put_back or initial_route_prefers_mixed_horizon_transition_probe_for_open_vs_weigh or strict_text_fallback or transition_probe or action_intent_requires_followup'`
  - `37 passed, 1072 deselected`
- [x] 本轮 `planner` 的 precondition 恢复链也继续 observation-centric 化：
  - `planner._action_intent_precondition_dependency_is_observation_grounded(...)`
    不再要求结果文本先显式提到某个候选答案语义，才允许回补动作前证据；
    现在优先看：
    - `primary_gap == precondition`
    - `blocker_hint == precondition_context`
    - 已写入的 `missing_state_change_prereq` memory marker
    - 动作对象本身是否属于通用前置状态依赖类
      - `towel/cloth/napkin`
      - `scale/button/switch/tap`
      - `spoon/ladle/board/tray`
  - `planner._select_action_intent_frames(...)`
    已删除两处基于 `question_text` 的 `keep_precontext` 直驱分支；
    precontext 是否保留，现只由 observation-side precondition gap 决定
  - `planner._stage_action_intent_frames(...)`
    在存在 precondition gap 时改成真正的三段均衡覆盖：
    - 保留动作前帧
    - 保留当前动作帧
    - 保留动作后结果帧
    不再因为默认段预算把前置上下文挤掉
- [x] 对应本轮 precondition 回归：
  - `pytest -q tests/test_graph_agent.py -k 'precondition or missing_state_change_prereq or precontext or planner_action_intent_primary_gap_precondition_focus'`
  - `25 passed, 1084 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (verifier_blocked or close_call or before_text_fallback or textual_rank or primary_gap or precondition)'`
  - `127 passed, 982 deselected`
- [x] 本轮继续切断 `hidden_access / exact outcome semantics -> pairwise ext followup` 的 live runtime 链：
  - `planner._build_action_intent_pairwise_resolution_decision(...)`
    不再通过
    `pairwise_requires_extended_followup -> hidden_access_pairwise_outcome_resolution`
    这一条候选语义冲突链决定是否继续扩窗
  - 原来的
    `planner._action_intent_pairwise_requires_extended_followup(...)`
    与
    `planner._action_intent_pairwise_text_has_explicit_hidden_outcome(...)`
    已退出运行态主链
  - 现在改为 observation-centric gate：
    `planner._action_intent_pairwise_needs_more_post_action_coverage(...)`
    只看：
    - 当前是否仍属于 `pair_needs_outcome_resolution`
    - 是否已经有 direct post-action evidence
    - 是否已经做过 transition / peak guided followup
    - followup attempt 次数
    - 当前 post-action 覆盖终点是否仍短于最小观察窗
  - 因而 pairwise 是否扩窗，已不再由：
    - hidden-access
    - revealed target
    - exact slot use
    - answer / needed_observation / comparison_summary
    这些候选语义产物来驱动
- [x] 对应旧测试契约已迁移为 observation-centric 负约束：
  - 不再断言：
    - “hidden_access pair 必须触发 ext2”
    - “某个固定 hidden outcome 文本必须允许 pairwise 直接收口”
    - “followup_route 必须返回固定 hidden_access reason”
  - 改为断言：
    - post-action coverage 不足时允许扩窗
    - direct post-action evidence 已存在时允许直接 finish / resolve
    - answer-conditioned 字段不会改变 coverage-extension 判定
    - route / window 可以根据 observation gap 落到 `future_use`
- [x] 本轮对应回归：
  - `pytest -q tests/test_graph_agent.py -k 'pairwise_extends_followup_when_post_action_coverage_is_still_short or pairwise_allows_resolution_when_direct_post_action_evidence_is_already_visible or pairwise_allows_resolution_after_extra_followup_has_already_been_sampled or pairwise_short_post_action_window_requests_more_followup_without_choice_semantics or pairwise_coverage_extension_ignores_answer_conditioned_fields'`
  - `5 passed, 1106 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (pairwise or future_use or specialized_resolution or candidate_indices)'`
  - `120 passed, 991 deselected`
  - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
  - 通过
  - `python -m py_compile food_agent/agent/planner.py`
  - 通过
  - `python -m py_compile food_agent/agent/planner.py`
  - 通过
- [x] 本轮继续切断 `candidate semantics -> transition probe mode/window` 的 live runtime 链：
  - `planner._action_intent_transition_probe_window(...)`
    不再先构造 `candidate_indices -> action_intent_conflict_profile(...)`
    再由候选语义冲突类型决定 probe window
  - `planner._action_intent_transition_probe_mode(...)`
    已从：
    - `profile["has_hidden_access_exact_use_conflict"]`
    - `profile["active_categories"]`
    - hand-free / reveal / safety 这类候选类别组合
    切到 observation-centric 判定：
    - `primary_gap_type`
    - `timeline_review_bias_profile`
    - `result_support_text`
    - direct post-action observation markers
    - later-outcome uncertainty markers
  - `planner._action_intent_reveal_conflict_subtype(...)`
    已不再扫描 `candidate_choices / question_text`
    来判定 `revealed_target_retrieval / revealed_slot_placement / revealed_fixture_enablement`
    现在只从：
    - support text
    - timeline review text
    里的观测证据抽取 reveal subtype
  - `planner._action_intent_pair_spans_immediate_and_later_outcomes(...)`
    与
    `planner._action_intent_initial_pair_spans_immediate_and_later_outcomes(...)`
    也已退出 `best/competitor category pair` 判定
    现在只从：
    - `primary_gap_type`
    - immediate-result observation markers
    - later-outcome uncertainty markers
    - `timeline_review_bias_profile`
    推断是否属于 mixed-horizon gap
- [x] 对应旧测试契约已迁移为 observation-centric：
  - 不再保护：
    - “check vs put back 必须走 mixed-horizon 文案”
    - “revealed fixture/slot/target 必须绑定某套固定 stride/max_frames”
    - “workspace/final placement close call 必须从动作后更晚位置起 probe”
  - 现在只保护：
    - transition probe 仍由 observation gap 触发
    - 不会因 `needed_observation profile` 之类答案残影单独触发 mixed-horizon
    - probe 窗口参数与当前 observation-driven mode 一致
- [x] 本轮对应回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (transition_probe or peak_probe_window or followup_state_change_only)'`
  - `40 passed, 1071 deselected`
  - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
  - 通过
- [x] 本轮 `verifier` 也继续切断两条仍在运行态的 answer-conditioned / marker-conditioned 链：
  - `verifier._action_intent_competing_pair_still_needs_disambiguation(...)`
    已不再通过 `selected_choice_categories(best, competitor)` 的类别差异
    来推断“只要 top2 类别不同就仍需消歧”
    现在只看：
    - 最新 payload 是否缺 direct post-action evidence
    - 最新 payload 是否指向 later-outcome uncertainty
    - `primary_gap_type`
    - payload support text 中是否仍存在 observation-side uncertainty markers
  - `verifier._action_intent_has_successful_specialized_resolution(...)`
    已不再把以下旧 marker 当成稳定收口证明：
    - `action_intent_pairwise_reason=...`
    - `action_intent_future_use_reason=...`
    - `action_intent_future_use_observation=...`
    - 各类 `*_override_best_index=...`
    现在只认：
    - 真实 specialized tool payload
    - `best_index` 存在
    - `need_more_evidence` 为假
    - 且 payload 里真的存在 direct post-action evidence，
      同时不再表达 later-outcome uncertainty
  - 因而 verifier 是否允许 suppress secondary conflicts、
    是否允许 stable answer，不再能被历史 marker 残影直接抬过去
- [x] 对应旧测试契约已迁移：
  - 不再允许只靠 `action_intent_pairwise_reason=` 这类 marker 就压掉 location/state conflicts
  - 现在必须有 observation-grounded 的 specialized payload 才能视为稳定
- [x] 本轮对应回归：
  - `pytest -q tests/test_graph_agent.py -k 'verifier_action_intent_ignores_secondary_location_and_state_conflicts_when_answer_is_stable or verifier_action_intent_close_future_use_competitor_blocks_finish or verifier_action_intent_decisive_future_use_resolution_can_finish or plausible_competing_candidate_gap or has_successful_specialized_resolution or action_intent and (missing_grounding_types or sufficient_grounding_for_stable_answer)'`
  - `3 passed, 1108 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (verifier_blocked or why_blocker or close_call or missing_grounding_types or sufficient_grounding_for_stable_answer or conflicting_locations or conflicting_state_observations)'`
  - `80 passed, 1031 deselected`
  - `python -m py_compile food_agent/agent/verifier.py tests/test_graph_agent.py`
  - 通过
- [x] 本轮继续收缩 `timeline review / pending specialized recovery` 的 live runtime 优先级：
  - `planner._resume_action_intent_specialized_resolution_from_timeline_review(...)`
    之前在 `future_outcome / relation_confirmation / target_discovery`
    的 timeline-review close call 下，会优先保留
    `future_use / pairwise` specialized 主链
  - 现在改为：
    1. 先尝试 observation-grounded 的 cached long-horizon revisit
    2. 再尝试 `primary_gap` 驱动恢复
       - `query_object`
       - `query_spatial_context`
       - 以及同属 gap-routed 的其他恢复动作
    3. 只有当 primary gap 没给出更强恢复动作时，
       才回到 `future_use / pairwise`
  - 这意味着：
    - timeline review 之后不再因为“历史上已经进入 specialized 语境”
      就天然优先续接 specialized resolution
    - `specialized` 现在退化成 fallback，
      而不是 primary-gap 恢复之前的主路径
- [x] 对应旧测试契约已保持为 observation-first：
  - 允许 cached revisit 继续优先于 specialized
  - 允许 primary gap 恢复动作直接在 timeline-review 之后接管
  - specialized resolution 仅在 gap route 没有更强动作时才继续
- [x] 本轮对应回归：
  - `pytest -q tests/test_graph_agent.py -k 'resume_action_intent_specialized_resolution_from_timeline_review or timeline_review_close_call_primary_gap or prefers_specialized_open_question_recovery or pending_resolution_tool or structured_specialized_recovery_tool'`
  - `8 passed, 1103 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (pending_resolution_tool or structured_specialized_recovery_tool or resume_action_intent_specialized_resolution_from_followup_artifacts or resume_action_intent_specialized_resolution_from_timeline_review or prefers_specialized_open_question_recovery or before_text_fallback)'`
  - `10 passed, 1101 deselected`
  - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
  - 通过
- [x] 本轮继续把 `specialized 历史语境` 从主驱动降级成 fallback 线索：
  - `planner._action_intent_prefers_specialized_open_question_recovery(...)`
    已不再因为：
    - `structured_specialized_recovery_tool(state)` 非空
    - `pending_resolution_tool(state)` 非空
    就直接返回 `True`
  - 现在它只由 observation/gap 驱动：
    - `primary_gap_type`
    - `latest_result` 是否仍表达 future-use / pairwise 观测缺口
    - `action_intent_need_future_evidence=1`
    - state-change-only followup 偏好
  - 因而 “历史上已经进入 specialized 语境” 本身不再是 open-question recovery 的充分条件
  - `planner._heuristic_fallback(...)` 后半段也进一步收口：
    - 之前会先续接 `future_use / pairwise` specialized，再看 `primary_gap`
    - 现在改为：
      1. 先试 `primary_gap` 恢复
      2. 只有 `primary_gap` 没给出更强动作时，才续接 specialized
- [x] 对应效果：
  - structured gap 仍可在 gap router 完全给不出动作时，保留 specialized fallback
  - 但 specialized 历史语境不再压过 observation-grounded 的 `query_object / query_spatial_context / local followup`
- [x] 本轮对应回归：
  - `pytest -q tests/test_graph_agent.py -k 'prefers_specialized_open_question_recovery or structured_specialized_recovery_tool or heuristic_fallback_prefers_primary_gap_early or open_question_prefers_primary_gap or structured_future_outcome_gap_can_drive_specialized_recovery_without_hypotheses_when_gap_router_yields_nothing'`
  - `4 passed, 1107 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (prefers_specialized_open_question_recovery or structured_specialized_recovery_tool or pending_resolution_tool or before_text_fallback or heuristic_fallback_prefers_primary_gap or recover_from_open_questions)'`
  - `10 passed, 1101 deselected`
  - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
  - 通过
- [x] 本轮继续削弱 `pending_resolution=` marker 的运行态优先级：
  - `planner._action_intent_resolution_mode(state, include_memory_marker=True)`
    之前会先直接读取：
    - `action_intent_pending_resolution=resolve_action_intent_future_use`
    - `action_intent_pending_resolution=resolve_action_intent_pairwise`
    并把它作为最高优先级 resolution mode
  - 现在改为：
    - 先看 `primary_gap`
    - 再看 `latest_result` 是否仍表达 future-use / pairwise observation gap
    - 再看 `sufficiency_decision.missing_gap_types`
    - 只有这些 observation/gap 侧证据仍支持时，
      `pending_resolution=` 才能作为弱线索生效
  - 因而：
    - 历史 pending marker 本身不再能单独驱动 specialized resume
    - pending marker 只能在“当前 gap 仍与它一致”时保留恢复方向
- [x] 对应效果：
  - observation-grounded future-outcome / relation-confirmation gap 仍可驱动 pending tool
  - 但陈旧的 pending marker 不再压过当前真实 gap 状态
- [x] 本轮对应回归：
  - `pytest -q tests/test_graph_agent.py -k 'pending_resolution_tool_can_be_driven_by_observation_grounded_gap or pending_resolution_tool_can_fall_back_to_structured_missing_gap_types_without_evidence_gaps or pending_resolution_tool_does_not_start_from_structured_hypotheses_alone or pending_future_use_returns_to_dedicated_resolution_after_extra_frames or peak_frames_with_structured_hypotheses_preserve_specialized_recovery_without_successful_intent_payload'`
  - `5 passed, 1106 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (pending_resolution_tool or resume_action_intent_specialized_resolution_from_followup_artifacts or resume_action_intent_specialized_resolution_from_timeline_review or prefers_specialized_open_question_recovery or heuristic_fallback_prefers_primary_gap or pending_future_use)'`
  - `7 passed, 1104 deselected`
  - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
  - 通过
- [x] 本轮继续切断 `action_intent_need_future_evidence=1` 对 followup 采样窗口/焦点的直接控制：
  - `planner._build_action_intent_followup_sampling_decision(...)`
    之前会从 `working_memory` 读取：
    - `focus=...`
    - `window_s=...`
    并直接覆盖当前 followup 采样决策
  - 现在改为：
    - followup `focus/window_s` 只由当前 `observation route`
    - `primary gap`
    - `dense near followup`
    - `window_level / budget`
    共同决定
  - 因而：
    - 历史 `action_intent_need_future_evidence=1` marker 仍可作为“是否需要继续看后续”的弱存在信号
    - 但不再能把当前采样窗口或检查焦点改写成旧答案链遗留的语义
- [x] 对应效果：
  - 当前 observation-centric followup route 不再被历史 marker 中的字符串参数接管
  - followup builder 的 runtime 决策进一步回到 `gap + coverage + budget`
- [x] 本轮对应回归：
  - `pytest -q tests/test_graph_agent.py -k 'followup_sampling_decision or followup_window_expands_with_window_level or action_intent_need_future_evidence'`
  - `2 passed, 1110 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (followup or pending_resolution_tool or heuristic_fallback or before_text_fallback)'`
  - `121 passed, 991 deselected`
  - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
  - 通过
- [x] 本轮继续把 `timeline review / transition probe / infer-failure fallback` 收口到 observation-first：
  - `planner._resume_action_intent_specialized_resolution_from_timeline_review(...)`
    现在在 `timeline review` 明确 `needs_more_evidence` 时：
    - 先尝试 `primary gap` 路由
    - 再尝试 `cached long-horizon revisit / transition probe / extra followup`
    - 只有这些 observation-side 动作都不给出更强路径时，才回到 `future_use / pairwise` specialized resolution
  - `planner._heuristic_fallback(...)`
    在 `infer_action_intent` 连续失败且：
    - `disable_legacy_specialized_recovery=1`
    时，直接退回 `rank_choices_from_state`
    而不是继续走 `retrieve_cached_artifacts / generic open-question recovery`
  - `planner._action_intent_transition_probe_mode(...)`
    现在对 reveal/access 类 observation 做了更细的通用分层：
    - 具体 subtype:
      - `revealed_slot_placement`
      - `revealed_fixture_enablement`
    - mixed-horizon:
      - `immediate result + later outcome` 并存
    - generic reveal-target retrieval:
      - 只在没有更具体 subtype/mixed-horizon 时才生效
    因而不再出现 “generic revealed_target_retrieval 吞掉更具体 observation mode” 的问题
- [x] 对应效果：
  - `timeline review` 之后不再默认抢先回 specialized resolution
  - `transition probe` 窗口重新由当前 observation subtype 决定，而不是被泛化 reveal 模式覆盖
  - `disable_legacy_specialized_recovery=1` 在 infer-failure 路径上成为真正生效的运行态开关
- [x] 本轮对应回归：
  - `pytest -q tests/test_graph_agent.py -k 'pairwise_more_evidence_prefers_audio_peak_probe_before_blind_ext_followup or infer_failures_level_zero_fixture_only_future_outcome_prefers_local_followup_over_cached_artifacts'`
  - `2 passed, 1110 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'after_first_followup_requests_transition_probe_before_pairwise_resolution or hand_free_future_use_prefers_transition_probe_before_resolution or transition_probe_window_prefers_revealed_slot_placement or transition_probe_window_prefers_revealed_fixture_enablement'`
  - `4 passed, 1108 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (transition_probe or peak_probe_window or followup_state_change_only)'`
  - `40 passed, 1072 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (followup or pending_resolution_tool or heuristic_fallback or before_text_fallback)'`
  - `121 passed, 991 deselected`
  - `python -m py_compile food_agent/agent/planner.py tests/test_graph_agent.py`
  - 通过

---

## Phase 3：统一 Planner 搜索决策，只允许 Gap 驱动

### 目标

- planner 不再读候选答案语义
- planner 的所有动作都可映射到通用 gap

### 重点文件

- [/22liushoulong/agent/hd-epic/food_agent/agent/planner.py](/22liushoulong/agent/hd-epic/food_agent/agent/planner.py)

### 优先清理对象

- 所有 `*_target_hint(...)`
- 所有 `*_blocker_hint(...)`
- 所有 `*_later_target_*`
- 所有 `*_revisit_*`
- 所有 `*_specialized_recovery_*`
- 所有 `*_timeline_review_*`

尤其优先：

- `_recover_action_intent_via_primary_gap(...)`
- `_resume_action_intent_specialized_resolution_from_timeline_review(...)`
- `_action_intent_pending_resolution_tool(...)`
- `_action_intent_should_run_timeline_review(...)`

### 完成标准

- planner 不再从 `choice_text / best_index / runner_up / comparison_summary` 决定下一步动作
- planner 只根据：
  - gap 类型
  - 时间窗覆盖
  - 轨迹是否闭合
  - 空间锚点是否存在
  - 预算是否允许
  决定下一步动作

### 测试策略

- 删除：
  - “固定类型题应走 future_use/pairwise”的测试
- 改写为负约束：
  - 没有 observation gap 时，planner 不能仅因 structured comparison 存在而继续搜
  - 只有 close call，不构成扩窗理由
- 新增通用正向：
  - `object_track_unclosed -> query_object`
  - `relation_unobserved + spatial_anchor_exists -> query_spatial_context`
  - `window_coverage_missing + no target anchor -> sample_sparse_frames`

### 阶段结束验证

- 所有搜索动作都可由 `gap + budget` 解释

### Phase 3 执行勾选

- [x] 删除 planner 中残留的一批 `*_target_hint / *_later_target_*` 答案导向消费点
- [x] verifier 侧 `comparison / needed_observation` 主状态注入已继续收缩
- [x] 删除 planner 中仅由 structured comparison 触发的继续搜索逻辑
- [x] 将所有搜索动作统一映射到通用 gap
- [x] 建立 `query_object / query_spatial_context / sample_sparse_frames` 的 observation-gap 驱动路由
- [x] 补充“无 observation gap 不得继续搜”的负约束测试
- [x] 通过阶段回归并回写文档进度

### 本轮新增进展

- `first followup short window` 的 runtime 决策已进一步 observation-centric：
  - 旧行为：
    - 首轮 `followup` 后，
      若当前 post-action 窗口仍很短，
      planner 可能直接因为局部启发式分支落到：
      - `transition probe`
      - 或 `rank_choices_from_state`
    - 这会让“短后窗、尚无直接 post-action evidence”的场景被过早收口
  - 当前变化：
    - 新增统一 gate：
      - `_action_intent_post_action_followup_window_is_short(...)`
      - `_action_intent_first_followup_needs_more_observation_coverage(...)`
    - 现在首轮 followup 之后是否继续扩窗，
      只由：
      - 当前覆盖窗是否仍短
      - 是否已有 direct post-action evidence
      - 是否已经进入 transition / peak / timeline-review 路径
      决定
    - 因而 `sample_sparse_frames(followup_ext2)` 成为
      `window_coverage_missing` 的通用恢复动作，
      而不是某类候选冲突的专项特判
- 对应本轮回归：
  - `pytest -q tests/test_graph_agent.py -k 'test_planner_action_intent_high_confidence_outcome_pair_extends_followup_when_post_action_window_is_still_short or test_planner_action_intent_pairwise_extends_followup_when_post_action_coverage_is_still_short or test_planner_action_intent_pairwise_short_post_action_window_requests_more_followup_without_choice_semantics'`
  - `3 passed, 1114 deselected`
- 对应主专项回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - `661 passed, 456 deselected`
- `repeated textual fallback` 的最后一条 mixed-horizon 旧契约已迁移：
  - 旧契约：当 `future_use` 候选语义里提到 `fridge` 时，启发式 fallback 必须立刻走
    `query_object(fridge)` 或 `query_spatial_context(fridge)`
  - 新契约：当当前 observation state 只覆盖动作附近局部帧，且尚未建立可靠 late-horizon 空间锚点时，
    planner 可以先走 `inspect_visual_evidence`，补局部时间序列观察缺口；
    是否继续扩窗或查询 fixture/object，之后只由 `gap + coverage + budget` 决定
- 这一步进一步切断了：
  - `candidate_evidence / answer / best_index / second_best_index`
    直接驱动 later-fixture query 的旧链

---

## Phase 4：收缩 Specialized Resolution，降级为普通搜证工具

### 目标

- `future_use / pairwise` 不再是“按题型的专门推理器”
- 只作为补某类 observation 的工具

### 重点文件

- [/22liushoulong/agent/hd-epic/food_agent/agent/planner.py](/22liushoulong/agent/hd-epic/food_agent/agent/planner.py)
- [/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py](/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py)

### 优先清理对象

- 所有 `*_pairwise_candidate_*`
- 所有 `*_future_use_candidate_*`
- 所有 `*_structured_specialized_*`
- 所有依赖 tool 名字再做 finalizer 专项兜底的逻辑

### 完成标准

- `future_use / pairwise` 触发条件只能由 observation gap 解释
- 不再存在 choice pair candidate selection 逻辑

### 测试策略

- 删除：
  - “某个固定冲突必须启动 pairwise/future_use”的测试
- 改写为负约束：
  - 没有对应 observation gap 时，不得调用 specialized tool
- 新增通用正向：
  - `relation gap -> pairwise`
  - `destination_unclosed -> future_use`

### Phase 4 执行勾选

- [ ] 将 `future_use / pairwise` 从题型推理器降级为普通搜证工具
- [x] 删除 `*_pairwise_candidate_* / *_future_use_candidate_*` 中的一批 live 答案竞争入口
- [ ] 删除 tool 名字触发 finalizer 专项兜底的链路
- [ ] 仅保留由 observation gap 触发 specialized tool 的路径
- [x] 补充“无对应 gap 不得调用 specialized tool”的负约束测试
- [x] 通过阶段回归并回写文档进度

### Phase 4 当前进展

- [x] 本轮继续收缩 `planner._recover_action_intent_after_verifier_blocked_finish(...)` 中 `infer_action_intent` 恢复尾部残留的 specialized profile 注入：
  - 旧行为：
    - `infer_action_intent` 被 verifier 拦下后，
      仍会在恢复尾部继续注入旧 specialized 语义：
      - `tool_name="resolve_action_intent_future_use"` 传给 transition recovery
      - `future_use / pairwise` 两种旧 profile 决定 extra followup window
      - thought 文案仍直接按“后果型 close call / top-2 / future use”分叉
  - 当前变化：
    - 新增 observation-side helper：
      - `planner._action_intent_observation_close_call_profile(...)`
    - close-call / extra-followup profile 现在只由以下 observation-side 状态决定：
      - `primary_gap`
      - `blocker_hint`
      - `later outcome uncertainty`
      - `immediate post-action uncertainty`
    - `infer_action_intent` 路径下的 transition recovery
      不再注入 `resolve_action_intent_future_use` 作为伪 specialized tool name，
      而是保留真实来源 `infer_action_intent`
    - 相关 thought 也已改成统一 observation-gap 表述，
      不再把恢复逻辑写成 `top-2 close call / pairwise / future_use` 的专项叙事
  - 本轮新增负约束测试：
    - `test_planner_action_intent_verifier_blocked_infer_transition_recovery_does_not_inject_future_use_tool_name`
    - `test_planner_action_intent_verifier_blocked_infer_extra_followup_profile_prefers_primary_gap_over_future_flag`
  - 本轮定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'verifier_blocked_infer_action_intent_prefers_observation_first_transition_recovery_over_specialized_resume or verifier_blocked_infer_transition_recovery_does_not_inject_future_use_tool_name or verifier_blocked_infer_extra_followup_profile_prefers_primary_gap_over_future_flag or verifier_blocked_immediate_gap_prefers_transition_probe or verifier_blocked_future_gap_prefers_later_outcome_recovery'`
    - `5 passed, 1127 deselected`
  - 本轮专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `676 passed, 456 deselected`
- [x] 本轮继续切掉 `planner._build_action_intent_resolution_transition_recovery_decision(...)` 里最后一组活跃的 specialized tool 身份分支：
  - 旧行为：
    - 仍会根据
      - `tool_name == resolve_action_intent_future_use`
      - 或非 `future_use`
    - 决定：
      - 是否把低置信度 / 缺决定性观测视作 transition recovery 触发条件
      - `revealed_target_retrieval + long_horizon_nodes` 是否提前停止
      - 恢复 thought 使用“后续用途专用裁决”还是“二选一后果裁决”
      - 已有 direct post-action evidence 时是否仍允许继续 transition recovery
  - 当前变化：
    - 这些分支已统一改成只看 observation-side profile：
      - `primary_gap`
      - `blocker_hint`
      - payload 是否表达 `later outcome uncertainty`
      - payload 是否已有 direct post-action evidence
      - 是否仍缺 `decisive_observation`
    - 因而 `transition recovery` 不再因为最新 specialized tool 名字不同而走不同恢复路径
    - `revealed_target_retrieval` 的 hold 条件也已改成：
      - 只要当前 recovery profile 属于 late-outcome family
      - 且真实 long-horizon nodes 已存在
      - 就停止近窗 transition recovery
  - 本轮新增负约束测试：
    - `test_planner_action_intent_resolution_transition_recovery_is_not_driven_by_specialized_tool_name`
    - `test_planner_action_intent_resolution_transition_recovery_reveal_hold_uses_observation_profile_not_tool_name`
  - 本轮定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'resolution_transition_recovery_is_not_driven_by_specialized_tool_name or resolution_transition_recovery_reveal_hold_uses_observation_profile_not_tool_name or resolution_transition_recovery_does_not_echo_needed_observation_text or verifier_blocked_finish_close_future_use_prefers_targeted_transition_probe'`
    - `4 passed, 1126 deselected`
  - 本轮专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `674 passed, 456 deselected`
- [x] 本轮 `verifier` 又继续切掉了一条 `candidate_evidence / candidate_indices -> competing gap` 的空壳 consumer：
  - 旧行为：
    - `verifier._action_intent_has_plausible_competing_candidate_gap(...)`
      仍会调用
      - `_action_intent_competing_candidate_index(...)`
    - 该 helper 会继续读取：
      - `candidate_evidence[*].index/score`
      - `candidate_indices`
    - 但后续 `competing_pair_still_needs_disambiguation(...)`
      实际并不真正使用这个 competitor identity；
      也就是说，这是一条还在运行态里读取答案竞争产物、但已经不再提供真实 observation 价值的残留链
  - 当前变化：
    - 已删除 verifier 侧 `_action_intent_competing_candidate_index(...)`
    - 已把 `competing_pair_still_needs_disambiguation(...)`
      改成只看 observation-side 状态：
      - payload 是否缺少 post-action evidence
      - payload 是否表达 later-outcome uncertainty
      - primary gap 是否仍未闭合
      - support text 是否仍有 uncertainty marker
    - 同时新增一层保护：
      - 若最新 payload 只是 `rank_choices_from_state` 这类 metadata-only 结果
      - 没有 observation text
      - 也没有真实 primary gap / payload uncertainty
      - 就不允许仅因低置信度 invent `plausible competing gap`
  - 新增负约束测试：
    - `test_verifier_action_intent_plausible_competing_candidate_gap_does_not_require_candidate_evidence_indices_when_observation_uncertainty_remains`
  - 兼容性回归：
    - 重新确认 textual fallback 可 finish 的场景未被误伤：
      - `accepts_ranked_best_index_after_repeated_vision_failures`
      - `textual_fallback_with_current_task_artifacts_and_grounding_can_finish`
      - `textual_fallback_without_open_needed_observation_payload_can_finish`
      - `textual_fallback_open_needed_observation_payload_alone_does_not_block_finish`
    - 定向回归：
      - `pytest -q tests/test_graph_agent.py -k 'accepts_ranked_best_index_after_repeated_vision_failures or textual_fallback_with_current_task_artifacts_and_grounding_can_finish or textual_fallback_without_open_needed_observation_payload_can_finish or textual_fallback_open_needed_observation_payload_alone_does_not_block_finish or plausible_competing_candidate_gap_does_not_require_candidate_evidence_indices_when_observation_uncertainty_remains'`
      - `5 passed, 1121 deselected`
  - 最新主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `670 passed, 456 deselected`
- [x] 本轮 `verifier` 又切掉了一条 `specialized tool identity / future-use score gap -> competing gap` 的 live consumer：
  - 旧行为：
    - `verifier._action_intent_has_plausible_competing_candidate_gap(...)`
      会先读最新 payload 的 `tool_name`
    - 若工具是
      - `resolve_action_intent_pairwise`
      - `resolve_action_intent_future_use`
      就进入不同的 specialized 分支
    - 其中 `future_use` 分支还会继续消费
      - `candidate_evidence[*].score`
      - `_action_intent_future_use_score_gap(...)`
      来决定 competing gap 是否仍存在
    - 这本质上仍是：
      - specialized tool 身份
      - 再加 candidate ranking score gap
      - 共同驱动 verifier 的 competing-gap gate
  - 当前变化：
    - 已删除 verifier 侧 `tool_name == pairwise/future_use` 的专门分支
    - 已删除 verifier 侧 `_action_intent_future_use_score_gap(...)`
    - 当前 `plausible competing candidate gap` 只允许由以下 observation-side 信号解释：
      - 当前 payload 是否缺少 direct post-action evidence
      - 当前 payload 是否仍表达 later-outcome uncertainty
      - 当前 support text 是否包含直接结果子句
      - 当前是否仍存在 observation-grounded primary gap
      - 当前 payload 的置信度是否仍偏低
    - 不再允许：
      - 仅因为 specialized tool 名字不同
      - 或 `candidate_evidence` 的分数差很小
      - 就让 verifier 保持 competing gap 阻塞
  - 新增负约束测试：
    - `test_verifier_action_intent_plausible_competing_candidate_gap_is_not_driven_by_specialized_tool_name`
    - `test_verifier_action_intent_plausible_competing_candidate_gap_ignores_future_use_score_gap_without_observation_uncertainty`
  - 对应定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'plausible_competing_candidate_gap_is_not_driven_by_specialized_tool_name or plausible_competing_candidate_gap_ignores_future_use_score_gap_without_observation_uncertainty or grounded_best_index_without_specialized_resolution_keeps_secondary_conflicts_blocking or textual_fallback_keeps_secondary_conflicts_blocking'`
    - `4 passed, 1121 deselected`
  - 最新主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `669 passed, 456 deselected`
- [x] 本轮继续切掉了一组 `raw observation tool -> early specialized resume` 的 live runtime 链：
  - 旧行为：
    - 在 `planner._heuristic_fallback(...)` 与 `planner.next_action(...)` 中，
      只要 `last_tool_name` 落在
      - `query_object`
      - `query_spatial_context`
      - `sample_sparse_frames`
      - `extract_frames_for_range`
      - `retrieve_cached_artifacts`
      这组 raw observation tools 里，
      且 `structured_specialized_tool` 仍保留
      `resolve_action_intent_future_use / resolve_action_intent_pairwise`
      身份，
      planner 就会在没有新的 observation gap 恢复命中的情况下，
      直接继续 specialized resolution
    - 这本质上仍是：
      - 工具名字 + 旧 specialized 身份
      - 共同驱动下一步 specialized 搜证
  - 当前变化：
    - 已删除上述两处 `early specialized resume` 分支
    - raw observation tool 现在最多只允许触发：
      - `primary_gap` 驱动的 observation 恢复
      - `open_question` / `state candidate` 的 observation-first 恢复
      - 后续常规 state-driven fallback
    - 不再允许：
      - 因为“上一个工具是 raw observation tool”
      - 且“还残留 pending specialized identity”
      - 就直接续接 `future_use / pairwise`
  - 新增负约束测试：
    - `test_planner_action_intent_heuristic_fallback_does_not_resume_specialized_resolution_from_raw_observation_tool_alone`
    - `test_planner_action_intent_next_action_does_not_resume_specialized_resolution_from_raw_observation_tool_alone`
  - 对应定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'heuristic_fallback_does_not_resume_specialized_resolution_from_raw_observation_tool_alone or next_action_does_not_resume_specialized_resolution_from_raw_observation_tool_alone or heuristic_fallback_late_primary_gap_respects_existing_specialized_resolution or heuristic_fallback_tail_explicit_downstream_object_prefers_primary_gap_over_specialized_resume or verifier_blocked_infer_action_intent_with_structured_gap_prefers_generic_followup_over_specialized_resume'`
    - `5 passed, 1118 deselected`
  - 最新主专项回归：
    - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
    - `667 passed, 456 deselected`
- [x] `future_use candidate indices` 已不再从以下答案竞争产物直接启动候选集：
  - `result.candidate_indices`
  - `result.best_index`
  - `result.second_best_index`
  - `latest_raw.candidate_indices`
  - `latest_raw.best_index`
  - `latest_raw.second_best_index`
  - `failed specialized tool args.candidate_indices`
- [x] 当前 `future_use candidate indices` 的默认入口已收缩为：
  - `full choices`
  - 再由 `timeline review` 中的 observation-side 文本去做保守类别收缩
- [x] `pairwise candidate indices` 在无可靠 observation-grounded 候选集时，
  不再直接失效，也不再要求历史 `best/second/pending` 才能运行；
  当前退回 `full choices`，再由 `timeline review` 与安全/空间类 observation 文本做保守收缩
- [x] 一批旧测试契约已迁移为 observation-first：
  - 不再保护 `best/second/hypothesis` 的旧排序
  - 只保护：
    - 是否还能由 observation-side timeline 收缩候选子集
    - 是否还能在无 timeline 证据时保守保留全量候选
- [x] 对应定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'future_use_candidates or pairwise_candidates or fallback_action_intent_pairwise_candidates'`
  - `10 passed, 1092 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'specialized_recovery_tool or pending_resolution_tool or before_text_fallback'`
  - `8 passed, 1094 deselected`
- [x] `resolution_mode` 已进一步收紧：
  - 不再因为普通 `relation_confirmation / target_discovery` gap
    就自动转成 `resolve_action_intent_pairwise`
  - 不再因为“空的 latest_result + primary_gap fallback”
    就误转成 `resolve_action_intent_future_use`
  - 当前 specialized mode 只优先来自：
    - 显式 `action_intent_pending_resolution=*`
    - 明确 `future_outcome` primary gap
    - 真实成功的 `infer_action_intent` payload
    - `sufficiency_decision.missing_gap_types`
- [x] 新一轮 specialized-mode 定向回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (pending_resolution_tool or structured_specialized_recovery_tool or resume_action_intent_specialized_resolution_from_followup_artifacts or resume_action_intent_specialized_resolution_from_timeline_review or prefers_specialized_open_question_recovery or before_text_fallback)'`
  - `10 passed, 1094 deselected`
- [x] 新一轮核心 observation-centric 回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (finalizer or verifier_blocked or before_text_fallback or pending_resolution_tool or structured_specialized_recovery_tool or finalize_action_intent or trace_best_index or structured_best_index or does_not_build_action_intent_hypotheses or discards_action_intent_hypotheses or no_longer_reads_latest_action_intent_hypotheses_from_verification_or_state or export_session_memory_keeps_observation_trace_without_hypotheses or planner_finish_metadata_excludes_answer_conditioned_hypothesis_fields)'`
  - `110 passed, 994 deselected`
- [x] `verifier` 侧新增负约束测试：
  - specialized tool 名字本身不能单独决定 `why_blocker`
  - specialized tool 名字本身不能单独把 gap source 变成 `resolution_followup_gap`
- [x] `Phase 6` 本轮又迁移了一条旧测试契约：
  - 不再要求
    `open_question recovery` 在已建立 pairwise 历史时
    必须继续返回 `resolve_action_intent_pairwise`
  - 改为只要求：
    - 仍围绕当前 `future_outcome` observation gap 补证
    - 允许返回：
      - `query_object`
      - `query_spatial_context`
      - `sample_sparse_frames`
      - `resolve_action_intent_pairwise`
      - `resolve_action_intent_future_use`
- [x] 本轮新增回归：
  - `pytest -q tests/test_graph_agent.py -k 'verifier_action_intent_blocker_hint_alone_does_not_create_future_outcome_gap or verifier_action_intent_blocker_is_not_driven_by_specialized_tool_name_alone or verifier_action_intent_gap_source_is_not_driven_by_specialized_tool_name_alone or verifier_action_intent_gap_targets_can_come_from_recent_observation_trace or verifier_does_not_build_action_intent_hypotheses_from_latest_resolution_payload'`
  - `5 passed, 1102 deselected`
- [x] 本轮继续切断 `verifier payload -> candidate semantics -> blocker family` 旧链：
  - `_action_intent_payload_blocker_family(...)`
    不再通过 `best_index / second_best_index / competitor pair`
    调 `action_intent_needs_pairwise_resolution(...) / action_intent_needs_future_use_resolution(...)`
    来决定 `pairwise_close_call / future_use_close_call`
  - 当前 blocker family 只允许由以下 observation-side 信号解释：
    - `verification_history.sufficiency_decision.missing_gap_types`
    - `recommended_next_step`
    - payload 内是否真的缺少直接 post-action 结果
    - payload 是否真的在表达 later outcome uncertainty
  - 这意味着：
    - 调换 `best_index / second_best_index`
      不应再改变 blocker family
    - 仅有候选竞争关系，不再是 verifier 生产 gap family 的入口
- [x] 针对上述去 candidate-conditioned blocker 的新增负约束测试：
  - `test_verifier_action_intent_payload_blocker_family_is_not_driven_by_candidate_indices`
  - `test_verifier_action_intent_payload_blocker_family_requires_observation_uncertainty`
  - 定向回归：
    - `pytest -q tests/test_graph_agent.py -k 'payload_blocker_family_is_not_driven_by_candidate_indices or payload_blocker_family_requires_observation_uncertainty or blocker_hint_alone_does_not_create_future_outcome_gap or blocker_is_not_driven_by_specialized_tool_name_alone or gap_source_is_not_driven_by_specialized_tool_name_alone or gap_targets_can_come_from_recent_observation_trace or does_not_build_action_intent_hypotheses'`
    - `8 passed, 1101 deselected`
- [x] 本轮 `verifier_blocked / close_call` 子集回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (verifier_blocked or why_blocker or close_call or blocker_hint_alone_does_not_create_future_outcome_gap or gap_source_is_not_driven_by_specialized_tool_name_alone)'`
  - `82 passed, 1025 deselected`
- [x] 本轮合并回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (finalizer or verifier_blocked or before_text_fallback or pending_resolution_tool or structured_specialized_recovery_tool or finalize_action_intent or trace_best_index or structured_best_index or does_not_build_action_intent_hypotheses or discards_action_intent_hypotheses or no_longer_reads_latest_action_intent_hypotheses_from_verification_or_state or export_session_memory_keeps_observation_trace_without_hypotheses or planner_finish_metadata_excludes_answer_conditioned_hypothesis_fields or agent_state_record_verification or export_session_memory_keeps_trace_observation_centric or restore_session_memory or why_blocker or close_call)'`
  - `130 passed, 977 deselected`
- [x] 本轮再次确认 specialized-mode 回归仍稳定：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (pending_resolution_tool or structured_specialized_recovery_tool or resume_action_intent_specialized_resolution_from_followup_artifacts or resume_action_intent_specialized_resolution_from_timeline_review or prefers_specialized_open_question_recovery or before_text_fallback)'`
  - `10 passed, 1099 deselected`
- [x] 本轮再次确认核心 observation-centric 合并回归仍稳定：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (finalizer or verifier_blocked or before_text_fallback or pending_resolution_tool or structured_specialized_recovery_tool or finalize_action_intent or trace_best_index or structured_best_index or does_not_build_action_intent_hypotheses or discards_action_intent_hypotheses or no_longer_reads_latest_action_intent_hypotheses_from_verification_or_state or export_session_memory_keeps_observation_trace_without_hypotheses or planner_finish_metadata_excludes_answer_conditioned_hypothesis_fields or agent_state_record_verification or export_session_memory_keeps_trace_observation_centric or restore_session_memory or why_blocker or close_call)'`
  - `130 passed, 979 deselected`
- [x] 本轮 `Phase 2` 又推进了一步 gap schema 收口：
  - `verifier._action_intent_payload_blocker_family(...)`
    已不再返回旧的 specialized blocker 名字：
    - `future_use_close_call`
    - `pairwise_close_call`
  - 当前优先返回的中性 family 变成：
    - `future_gap_family`
    - `post_action_evidence`
  - 其判断依据也进一步收缩为 observation-side 信号：
    - `missing_gap_types`
    - `recommended_next_step`
    - payload 内是否真的缺少直接 post-action 结果
    - payload 是否真的表达 later outcome uncertainty
  - 这意味着：
    - `candidate pair / specialized close-call` 不再是 verifier 主路径里的 gap family 命名来源
    - `Phase 2` 正在把“gap 类型”和“旧 specialized 名字”彻底拆开
- [x] 对应新增/迁移回归：
  - `pytest -q tests/test_graph_agent.py -k 'payload_blocker_family or blocker_hint_alone_does_not_create_future_outcome_gap or blocker_is_not_driven_by_specialized_tool_name_alone or gap_source_is_not_driven_by_specialized_tool_name_alone or gap_targets_can_come_from_recent_observation_trace or does_not_build_action_intent_hypotheses or primary_gap_sanitizes_answer_conditioned_target_object_without_observation_anchor or primary_gap_future_outcome_object_gap_prefers_object_query_without_long_horizon_anchor or primary_gap_future_outcome_fixture_gap_prefers_spatial_query_without_spatial_anchor'`
  - `11 passed, 1098 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (verifier_blocker_hint or primary_gap or before_text_fallback or pending_resolution_tool or structured_specialized_recovery_tool or close_call or why_blocker)'`
  - `60 passed, 1049 deselected`
  - `pytest -q tests/test_graph_agent.py -k 'payload_blocker_family or heuristic_fallback_tail_explicit_downstream_object_prefers_primary_gap_over_specialized_resume'`
  - `3 passed, 1106 deselected`
- [x] `Phase 6` 本轮又迁移了一条旧测试契约：
  - `heuristic_fallback_tail_explicit_downstream_object_prefers_primary_gap_over_specialized_resume`
  - 不再要求写回特定 `planner_guard=*` 字符串
  - 改为只保护 observation-centric 行为结果：
    - 有显式 downstream primary gap 时
    - `heuristic_fallback` 必须优先走 `primary_gap` 恢复
    - 不能退回 `resolve_action_intent_future_use`

---

## Phase 5：清理 State/Trace 中会污染后续思考的答案产物

### 目标

- `trace` 仅用于解释
- `search state` 不再携带答案导向信息

### 重点文件

- [/22liushoulong/agent/hd-epic/food_agent/agent/state.py](/22liushoulong/agent/hd-epic/food_agent/agent/state.py)
- [/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py](/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py)

### 优先清理对象

- `_sync_action_intent_trace_from_verification(...)`
- `_export_action_intent_trace(...)`
- 所有会把 `runner_up / comparison_summary / missing_observations` 扩散出去的逻辑
- 所有恢复 memory 时重新读入旧 marker 的逻辑

### 完成标准

- `working_memory` 不再混入会影响 planner/graph_agent 的 answer artifact
- `trace` 与 `search state` 彻底分层
- `finish metadata / trace export` 不再携带 `top_hypothesis / runner_up / comparison_summary / blocking_comparisons`

### 测试策略

- 删除：
  - 依赖 trace 内容驱动搜索路径的测试
- 改写为负约束：
  - trace/export 只允许保留 observation-side 的 `summary / primary_gap / recommended_next_action / finish_mode`
  - planner / executor 的 finish metadata 不应再导出候选答案竞争残影

### Phase 5 执行勾选

- [x] 切断 `working_memory` 中会污染搜索态的 answer artifact 写回
- [x] 将 `trace/export` 与 `search state` 明确分层
- [x] 删除恢复 memory 时重新读入旧 marker 的逻辑
- [x] 保留 trace 的解释作用，但禁止其驱动 planner/graph-agent 搜索
- [x] 补充 trace 可保留、但搜索不可消费的负约束测试
- [x] 通过阶段回归并回写文档进度
- [x] `state.py` 新增运行时裁剪：
  - `record_verification(...)` 不再把原始 `sufficiency_decision` 整包写入 `verification_history`
  - 当前只保留：
    - `sufficient`
    - `missing_gap_types`
    - `recommended_next_step`
    - `finish_mode`
    - `summary`
  - 同时显式清空：
    - `blocking_hypotheses`
    - `blocking_comparisons`
- [x] `restore_session_memory(...)` 现也会对历史 `verification_history` 做同样运行时裁剪：
  - 不再把旧会话里的候选竞争残影重新带回搜索态
  - `action_intent_hypotheses` 在恢复时继续强制为空
- [x] `restore_session_memory(...)` 现也会过滤旧的 action-intent 控制 marker：
  - 会丢弃：
    - `action_intent_pending_resolution=*`
    - `action_intent_resolution_withheld_for_*`
    - `action_intent_unresolved_rerank_withheld*`
    - `action_intent_top_hypothesis=*`
    - `action_intent_runner_up=*`
    - `action_intent_top_missing_observation=*`
    - `action_intent_comparison_summary=*`
    - `action_intent_blocking_comparison=*`
    - `planner_guard=*`
    - `planner_override *`
  - 会保留：
    - `reuse:*`
    - 普通 observation/evidence 文本
    - `primary_gap_recovery_trace=*`
- [x] 新一轮 Phase 5 / 导出链相关回归：
  - `pytest -q tests/test_graph_agent.py -k 'agent_state_record_verification or export_session_memory_keeps_observation or export_session_memory_keeps_trace_observation_centric or snapshot_excludes_answer_conditioned_action_intent_artifacts or materialized_rank_fallback_candidate_does_not_backfill_latest_verification_competition_fields or planner_finish_metadata_excludes_structured_blockers_and_hypothesis_summaries or graph_agent_result_to_dict_accepts_pandas_series_include_row'`
  - `10 passed, 1094 deselected`
- [x] 新一轮合并回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (finalizer or verifier_blocked or before_text_fallback or pending_resolution_tool or structured_specialized_recovery_tool or finalize_action_intent or trace_best_index or structured_best_index or does_not_build_action_intent_hypotheses or discards_action_intent_hypotheses or no_longer_reads_latest_action_intent_hypotheses_from_verification_or_state or export_session_memory_keeps_observation_trace_without_hypotheses or planner_finish_metadata_excludes_answer_conditioned_hypothesis_fields or agent_state_record_verification or export_session_memory_keeps_trace_observation_centric)'`
  - `111 passed, 993 deselected`
- [x] 新增恢复态负约束测试：
  - `pytest -q tests/test_graph_agent.py -k 'restore_session_memory or export_and_restore_session_memory or snapshot_excludes_answer_conditioned_action_intent_artifacts or export_session_memory_keeps_observation_trace_without_hypotheses'`
  - `4 passed, 1101 deselected`
- [x] 最新合并回归：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent and (finalizer or verifier_blocked or before_text_fallback or pending_resolution_tool or structured_specialized_recovery_tool or finalize_action_intent or trace_best_index or structured_best_index or does_not_build_action_intent_hypotheses or discards_action_intent_hypotheses or no_longer_reads_latest_action_intent_hypotheses_from_verification_or_state or export_session_memory_keeps_observation_trace_without_hypotheses or planner_finish_metadata_excludes_answer_conditioned_hypothesis_fields or agent_state_record_verification or export_session_memory_keeps_trace_observation_centric or restore_session_memory)'`
  - `112 passed, 993 deselected`

---

## Phase 6：系统性替换旧测试体系

### 目标

- 从“保护专项规则”改成“保护负约束与结构不变量”

### 重点文件

- [/22liushoulong/agent/hd-epic/tests/test_graph_agent.py](/22liushoulong/agent/hd-epic/tests/test_graph_agent.py)

### 新测试分层

1. 负约束测试
2. 通用 gap 测试
3. 搜索动作映射测试
4. 最小端到端流程测试

### 必删测试类型

优先审查并删除以下命名风格对应的旧测试：

- `prefers_downstream_object`
- `prefers_later_object_revisit`
- `revealed_target_gap_prefers_*`
- `timeline_review_*_gap`
- `exact_workspace_*`
- `weak_* claim`
- `generic_* overclaim`

原则：

- 凡是测试名本身就在保护“某个具体语义应导向某个具体动作/答案”的，优先判为旧体系污染对象

### 必改测试类型

- 从“应该偏好 X”
- 改成“不能由答案产物单独偏好 X”
- 或“只有 observation gap 存在时才允许进入 X”

### 必新增测试类型

- 不允许 `choice_text -> target`
- 不允许 `comparison_summary -> search action`
- 不允许 `needed_observation -> primary_gap`
- 不允许 `runner_up -> window expansion`
- 正向 observation gap 测试：
  - `window_coverage_missing`
  - `object_track_unclosed`
  - `destination_unclosed`
  - `relation_unobserved`

### 完成标准

- 测试命名层面已明显去专项化
- 负约束测试成为主干
- `passed` 数不作为阶段目标

### Phase 6 执行勾选

- [ ] 删除仍在保护专项语义行为的旧测试
- [ ] 将旧测试改写为信息源隔离与负约束测试
- [ ] 新增通用 gap 正向测试
- [ ] 新增搜索动作可解释性测试
- [ ] 新增保守停止与预算约束测试
- [ ] 通过阶段回归并回写文档进度

### Phase 6 当前进展

- [x] 本轮又迁移了 2 条仍在保护旧 profile 注入的测试契约：
  - `verifier_blocked_infer_transition_recovery_does_not_inject_future_use_tool_name`
  - `verifier_blocked_infer_extra_followup_profile_prefers_primary_gap_over_future_flag`
- [x] 新契约：
  - 不再允许：
    - `infer_action_intent` 在 verifier-blocked 恢复里伪装成 `future_use`
    - `need_future_evidence` 这种旧 payload 标志单独决定 extra followup window
  - 只允许：
    - transition recovery 保留真实来源工具
    - extra followup window 由 `primary_gap + blocker_hint + observation uncertainty` 决定
- [x] 本轮又迁移了 2 条仍在保护“固定恢复动作”的旧测试契约：
  - `needed_observation_revealed_slot_prefers_downstream_object_over_slot_fixture`
  - `needed_observation_sink_slot_prefers_downstream_object_over_fixture`
- [x] 旧契约：
  - 这两条测试要求一旦出现 `revealed slot / sink slot` 场景，
    planner 必须固定返回 `extract_frames_for_range`
- [x] 新契约：
  - 只保护 observation-centric 的行为边界：
    - 当前恢复动作必须保持本地 observation-first
    - 允许：
      - `detect_audio_peaks`
      - `extract_frames_for_range`
    - 不允许仅因 `slot / sink slot` 语义回退到：
      - `query_spatial_context`
      - `query_object`
- [x] 这一步的意义：
  - 测试不再把“某个固定空间语义 -> 某个固定恢复动作”写死
  - 改为只保护：
    - 不要重新掉回 fixture/target semantic routing
    - 恢复动作仍可由当前 observation gap 解释

---

## Phase 7：建立最小真实样例审计

### 目标

- 不再继续刷单元测试
- 用真实 trace 检查 agent 是否真的 observation-centric

### 输入

- 少量真实 VQA / action-intent 样例
- 每个样例完整 trace

### 输出

- 每个样例只记录：
  - 初始 observation
  - active gap
  - 为什么不足
  - 搜索动作
  - 搜索后是否闭合

### 完成标准

- 至少 3 到 5 个真实样例中，看不到 `choice -> target -> search`
- 如果还能看到，就回到前面阶段继续删链

### Phase 7 执行勾选

- [ ] 随机抽取 3 到 5 个真实样例，而不是定向挑题
- [ ] 保存每个样例的完整 trace
- [ ] 对每个样例记录 `初始 observation / active gap / 搜索动作 / 闭合情况`
- [ ] 审计是否仍存在 `choice -> target -> search` 链
- [ ] 若发现残留链路，回退到对应前置阶段修复
- [ ] 将真实样例审计结果同步回主清单

---

## Phase 8：收口与 Goal 模式执行规范

### 目标

- 让之后的 goal 模式固定围绕“删链 -> 补通用机制 -> 负约束验证 -> 真实审计”循环

### 每轮只汇报

- 切掉了哪条链
- 改了哪类测试
- 还剩哪一类残余

### 每阶段固定验证项

- 代码层：是否还存在该类 answer-conditioned 消费点
- 测试层：是否仍有旧专项测试保护旧逻辑
- 行为层：搜索动作是否可由 observation state 解释
- 文档层：主清单是否同步更新

### Phase 8 执行勾选

- [ ] 固化 goal 模式的单轮执行粒度：每轮只解决一个最小真实缺口
- [ ] 固化 goal 模式的验证顺序：删链 -> 补通用机制 -> 负约束验证 -> 真实审计
- [ ] 固化 goal 模式的汇报格式：切掉的链、改掉的测试、残余类型
- [ ] 固化文档同步要求：每完成一项立即勾选
- [ ] 完成最终收口审计
- [ ] 将总进度更新为可交接状态

---

## 23.6 旧测试处理规则

### 23.6.1 删除

删除所有本质上在保护旧专项规则的测试：

- “固定动作语义 -> 固定搜索动作”
- “固定冲突模式 -> 固定 override”
- “固定 best/runner-up -> 固定 target”

#### 23.6.1 执行勾选

- [ ] 删除“固定动作语义 -> 固定搜索动作”类测试
- [ ] 删除“固定冲突模式 -> 固定 override”类测试
- [ ] 删除“固定 best/runner-up -> 固定 target”类测试

### 23.6.2 改写

将旧测试改写为：

- 不能由答案产物单独触发该动作
- 只有 observation gap 存在时才允许进入该分支

#### 23.6.2 执行勾选

- [ ] 将旧测试从“应该偏好 X”改写为“不能由答案产物单独偏好 X”
- [ ] 将旧测试从“必须走某专项链”改写为“只有 observation gap 存在时才允许进入该链”

### 23.6.3 新增

新增只保护以下内容的测试：

- 信息源隔离
- gap 来源合法性
- 搜索动作可解释性
- 保守停止策略
- 预算约束

#### 23.6.3 执行勾选

- [ ] 新增信息源隔离测试
- [ ] 新增 gap 来源合法性测试
- [ ] 新增搜索动作可解释性测试
- [ ] 新增保守停止策略测试
- [ ] 新增预算约束测试

---

## 23.7 Goal 模式运行节奏

建议之后 goal 模式按以下顺序执行：

1. 先清 `finalizer` 旧答案语义补丁
2. 再统一 `primary gap`
3. 再统一 planner 搜索动作
4. 再收缩 specialized tool
5. 再清 trace / state 污染
6. 再系统性替换旧测试体系
7. 最后做真实样例 trace 审计

### 23.7 执行勾选

- [ ] 按顺序完成 `finalizer -> primary gap -> planner -> specialized tool -> trace/state -> tests -> trace audit`
- [ ] 每轮只完成一个最小真实缺口
- [ ] 每轮结束后同步更新主清单

每轮 goal 的完成定义不是：

- `passed` 增加了多少

而是：

- 是否切掉了一条真实的 `answer artifact -> gap/target/search` 链

---

## 23.8 完成标准

只有同时满足以下条件，才算真正完成：

- `why / action_intent` 前段搜索不再答案条件化
- `gap` 全部来自 observation state
- planner 的下一步动作全部可由 `gap + budget` 解释
- finalizer 不再把候选答案语义写回搜索主路径
- tests 主体不再保护专项行为，而是保护负约束
- 小样本真实 trace 中看不到 `choice -> target -> search`

### 23.8 验收勾选

- [ ] `why / action_intent` 前段搜索不再答案条件化
- [ ] `gap` 全部来自 observation state
- [ ] planner 的下一步动作全部可由 `gap + budget` 解释
- [ ] finalizer 不再把候选答案语义写回搜索主路径
- [ ] tests 主体不再保护专项行为，而是保护负约束
- [ ] 小样本真实 trace 中看不到 `choice -> target -> search`

---

## 23.9 一句话结论

后续的实现不再围着“某类题怎么提分”转，
而必须围着三件事转：

- [ ] 删旧链
- [ ] 补通用 observation-centric 机制
- [ ] 用负约束测试与真实 trace 审计收口

- 删掉旧的答案驱动链
- 建立统一的 observation gap 与搜索动作映射
- 用负约束测试和真实 trace 审计保证不回潮
