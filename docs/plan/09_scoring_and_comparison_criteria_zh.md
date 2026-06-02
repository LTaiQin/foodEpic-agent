# 09 判定标准与对比实验规则

## 9.1 为什么不能只看准确率

只看 accuracy 不足以证明你的 agent 有优势，因为厨房任务里还需要：

- 答案是否可追溯到证据。
- 是否真的使用了结构化记忆。
- 是否能覆盖 recipe、ingredient、object、gaze、audio。
- 是否在长时序问题上保持稳定。

因此必须把“答案正确”与“证据正确”分开。

## 9.2 核心判定分数

定义 `FoodAgent Advantage Score`：

```text
0.40 * accuracy
+ 0.25 * evidence_rate
+ 0.15 * state_coverage
+ 0.10 * tool_use_rate
+ 0.10 * reliability
```

其中：

- `accuracy`：任务正确率。
- `evidence_rate`：有事件 id / 时间段 / 帧号的答案占比。
- `state_coverage`：recipe / ingredient / object / gaze / audio 状态覆盖。
- `tool_use_rate`：回答过程中真正调用了相关结构化工具的比例。
- `reliability`：`1 - failure_rate`。

## 9.3 判定阈值

### Clear Advantage

若满足以下条件，可以直接说你的 agent 有明确优势：

- `FoodAgent Advantage Score >= 0.65`
- `accuracy >= 0.55`
- `evidence_rate >= 0.60`
- `state_coverage >= 0.70`
- `reliability >= 0.70`

### Promising

若满足以下条件，但还不够强，可以说有前景：

- `FoodAgent Advantage Score >= 0.55`
- `accuracy >= 0.45`
- `evidence_rate >= 0.45`
- `state_coverage >= 0.55`
- `tool_use_rate >= 0.40`

### Not Yet

低于上述条件，则不能宣称有明显优势，只能说系统可运行或有局部能力。

## 9.4 最关键的对比

要凸显你的 agent 优势，至少要在这些维度胜出：

- 比 `TextOnly LLM` 更高的 accuracy。
- 比 `普通 Agent` 更高的 evidence_rate。
- 比 `LightAgent-Original` 更高的 tool_use_rate 和 reliability。
- 比 `LightAgent+HDTools` 更高的 state_coverage。
- 比 `LightAgent+RAG` 更高的长时序 recipe / ingredient 追踪能力。

## 9.5 推荐结论模板

如果实验成立，结论不要只写“准确率更高”，而应写成：

- 在厨房过程问答上，系统准确率提升。
- 更重要的是，答案证据可追溯率明显更高。
- 状态覆盖更完整，能够同时利用 recipe、ingredient、object、gaze 和 audio。
- 错误类型从“盲猜”转为“证据不足或工具选择失败”，可诊断性更强。

## 9.6 最低通过线

如果你想把系统当成“有实际价值的 agent 原型”，最低线建议是：

- `accuracy >= 0.50`
- `evidence_rate >= 0.50`
- `state_coverage >= 0.60`
- `reliability >= 0.60`

如果低于这个线，说明系统还只是一个可运行原型，还不能说优势明显。

