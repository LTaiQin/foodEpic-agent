# 03 LightAgent Baseline 与优化计划

## 3.1 基线定位

LightAgent 是本项目的 agent runtime baseline。它负责自然语言推理、工具调用和 trace 输出；HD-EPIC 结构化 memory、证据检索、状态维护和答案验证由外部 wrapper 控制。

本地 LightAgent：

```text
/22liushoulong/agent/agent-context-isolation/third_party/LightAgent
```

关键接口：

```python
agent.run(
    query: str,
    tools: list | None = None,
    history: list | None = None,
    user_id: str = "default_user",
    use_skills: bool = True,
    result_format: str = "str",
    trace: bool = False,
)
```

## 3.2 必须冻结的 baseline

必须分层比较，不能把所有改进混在一起：

- `LightAgent-Original`：原始 LightAgent，不接 HD-EPIC 工具。
- `LightAgent-TextOnly`：只给问题和选项，不给数据工具、不注入证据。
- `LightAgent-FrameOnly`：只给抽帧或 clip caption，不给结构化 memory。
- `LightAgent+HDTools`：接 HD-EPIC 工具，但不做任务路由和证据约束。
- `LightAgent-RAG`：检索事件后拼入 prompt。
- `LightAgent-FoodMemory`：接 recipe、ingredient、object、gaze、audio 多工具。
- `Ours-LightAgent+Evidence`：加入任务路由、状态 memory、证据约束、答案验证。

## 3.3 公平性控制

LightAgent 当前会自动注册内置工具：

- `execute_python_code`
- `execute_python_file`
- `execute_python_code_stream`
- `upload_file_to_oss`

风险：

- 如果 `run(tools=None)`，模型仍可能看到默认工具。
- TextOnly baseline 会被污染。
- 工具调用 trace 无法证明模型只依赖文本。

处理策略：

- wrapper 中显式记录每次暴露给模型的 tool list。
- TextOnly 和 Original 必须关闭或清空默认工具。
- 所有实验使用 `use_skills=False`，避免 skill metadata 干扰。
- 初期不使用 LightAgent 内置 memory。

## 3.4 Wrapper 架构

目标文件：

```text
food_agent/lightagent_wrapper.py
food_agent/task_router.py
food_agent/evidence_policy.py
food_agent/answer_verifier.py
food_agent/trace_eval.py
```

流程：

```text
FoodAgentLightWrapper.run(question, sample)
  -> parse sample: video_id / time / choices / task_family
  -> FoodTaskRouter.select_tools(task_family)
  -> FoodMemoryRetriever.retrieve(...)
  -> EvidencePolicy.build_prompt(...)
  -> LightAgent.run(..., tools=selected_tools, trace=True, result_format="object")
  -> AnswerVerifier.check(...)
  -> save answer / evidence / trace / failure_type
```

## 3.5 HD-EPIC Tools

目标文件：

```text
food_agent/hd_epic_tools.py
```

工具必须符合 LightAgent `tool_info` 格式。

核心工具：

- `get_video_metadata(video_id)`
- `retrieve_events(video_id, start_time, end_time, event_types)`
- `get_recipe_state(video_id, time)`
- `get_ingredient_state(video_id, time)`
- `get_object_state(video_id, object_name, time)`
- `get_gaze_hand_context(video_id, time)`
- `get_audio_events(video_id, start_time, end_time)`
- `sample_video_frames(video_id, start_time, end_time, fps)`
- `answer_vqa_with_evidence(vqa_id)`

工具返回格式：

```json
{
  "status": "ok",
  "evidence": [
    {
      "evidence_id": "event:...",
      "video_id": "...",
      "start_time": 0.0,
      "end_time": 1.0,
      "type": "recipe_step",
      "text": "..."
    }
  ],
  "data": {}
}
```

## 3.6 Evidence Policy

目标：

- 限制 agent 必须基于证据回答。
- 证据不足时允许输出不确定。
- 所有答案尽量包含时间段、事件 id 或帧号。

输出格式建议：

```json
{
  "answer": "...",
  "choice": 2,
  "confidence": "high|medium|low",
  "evidence_ids": ["..."],
  "time_spans": [[12.3, 18.7]],
  "reason": "..."
}
```

## 3.7 Trace 评估

每次 LightAgent 运行必须记录：

- 使用的 baseline 名称。
- 模型名称。
- 实际工具列表。
- prompt token 和 completion token。
- tool calls。
- tool results 摘要。
- final answer。
- evidence ids。
- failure type。

失败类型：

- `no_retrieval`
- `wrong_retrieval`
- `wrong_tool`
- `reasoning_error`
- `visual_missing`
- `format_error`
- `evidence_missing`

## 3.8 验证标准

最小 smoke：

- TextOnly 不暴露任何 HD-EPIC 工具。
- HDTools 暴露指定工具。
- `trace=True` 能拿到 tool list 和模型请求摘要。
- 用 mock model 或小样本跑通 5 个 VQA sample。

正式完成：

- 50 个 VQA sample 跑通全部 baseline。
- 每个 baseline 输出统一 JSONL。
- trace parser 能统计 tool selection accuracy 和 failure type。

