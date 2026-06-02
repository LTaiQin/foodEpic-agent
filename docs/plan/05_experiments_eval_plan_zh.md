# 05 实验与评估计划

## 5.1 实验目标

实验不是只刷 VQA 总分，而是回答以下问题：

- 结构化 food-process memory 是否提升长视频厨房问答。
- Evidence constraint 是否减少幻觉。
- 状态化 recipe/ingredient memory 是否提升过程追踪。
- LightAgent 在多工具长视频任务中失败在哪里。
- 哪些任务需要视觉，哪些任务结构化标注已足够。

## 5.2 Baseline 矩阵

| 系统 | 工具 | Memory | 视觉 | 目的 |
|---|---|---|---|---|
| LightAgent-Original | 原始工具关闭或记录 | 无 | 无 | 原始 agent 边界 |
| LightAgent-TextOnly | 无 | 无 | 无 | 最弱文本 baseline |
| LightAgent-FrameOnly | 抽帧/caption | 无 | 有 | 直接视觉能力 |
| LightAgent+HDTools | HD-EPIC tools | 无状态查询 | 可选 | 工具接入收益 |
| LightAgent-RAG | 检索工具 | 事件证据 | 可选 | 检索收益 |
| LightAgent-FoodMemory | 多工具 | recipe/ingredient/object/audio/gaze | 可选 | 主 baseline |
| Ours-LightAgent+Evidence | 多工具 + 状态 + verifier | food-process memory | 可选 | 最终方法 |

## 5.3 数据划分

推荐划分：

- participant-held-out：按参与者划分训练/验证/测试。
- task-family split：按 VQA 文件或问题类型统计。
- video-length split：短、中、长片段。
- evidence availability split：结构化证据充分和证据不足分开分析。

禁止：

- 随机打乱导致同一 participant 同时出现在训练和测试。
- 把 correct answer 直接泄漏到 prompt。
- 把完整标注无筛选塞给模型。

## 5.4 指标

VQA：

- accuracy。
- task-family accuracy。
- macro average accuracy。

证据：

- answer-with-evidence rate。
- evidence recall。
- evidence correctness。
- time-span IoU。

工具：

- tool selection accuracy。
- unnecessary tool call rate。
- missing tool call rate。
- tool result usefulness。

状态追踪：

- recipe step accuracy。
- ingredient timeline F1。
- nutrition QA accuracy。
- anomaly precision/recall/F1。

效率：

- latency。
- token cost。
- number of tool calls。
- retrieval time。

## 5.5 错误类型

统一错误标签：

- `no_retrieval`：没有检索到证据。
- `wrong_retrieval`：检索到的证据不支持正确答案。
- `wrong_tool`：调用了错误工具。
- `missing_tool`：应该调用工具但没有调用。
- `reasoning_error`：证据正确但推理错误。
- `visual_missing`：需要视觉帧但当前配置没有提供。
- `evidence_missing`：答案正确但无证据。
- `format_error`：输出格式错误。
- `label_ambiguity`：标注或问题本身不清晰。

## 5.6 Ablation

必须做：

- 无证据约束 vs 有证据约束。
- 无任务路由 vs 有任务路由。
- 无状态 memory vs 有状态 memory。
- recipe-only vs recipe+ingredient。
- structured-only vs visual-only vs hybrid。
- 默认 LightAgent tool list vs controlled tool list。

可选：

- audio ablation。
- gaze/hand ablation。
- object/3D ablation。
- different retriever window size。

## 5.7 结果文件

输出目录：

```text
outputs/results/
outputs/traces/
outputs/reports/
outputs/tables/
```

每次实验保存：

- `config.json`
- `predictions.jsonl`
- `metrics.json`
- `trace.jsonl`
- `failure_cases.jsonl`
- `summary.md`

预测记录字段：

- `sample_id`
- `baseline`
- `task_family`
- `video_id`
- `question`
- `choices`
- `gold`
- `prediction`
- `correct`
- `evidence_ids`
- `tool_calls`
- `failure_type`

## 5.8 论文表格

计划表格：

- Table 1：overall VQA accuracy by baseline。
- Table 2：task-family accuracy。
- Table 3：evidence quality。
- Table 4：tool-use failure analysis。
- Table 5：recipe/ingredient state tracking。
- Table 6：ablation。

计划图：

- pipeline figure。
- event memory schema。
- trace example。
- failure type distribution。
- recipe/ingredient timeline visualization。

