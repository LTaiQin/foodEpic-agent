# 08 风险、决策与后续路线

## 8.1 已确认决策

项目定位：

- 做 food-process agent，不做机器人控制。
- LightAgent 作为 baseline，不直接作为最终方法。
- 初期不修改 LightAgent 内核，先使用 wrapper。
- 初期不使用 LightAgent 内置 memory。
- 所有任务必须同时有研究意义和应用价值。

数据策略：

- `data/`、`annotations/` 不入库。
- semidense CSV 保持 `.csv.gz`，不默认解压。
- 3D 数据初期只做按需查询。

实验策略：

- 必须做 baseline 分层。
- 必须做 evidence constraint ablation。
- 必须记录 trace。
- 必须按 task family 分析。

## 8.2 主要风险

风险 1：Narration pickle 环境不兼容。

- 现象：当前 Python 读取 `HD_EPIC_Narrations.pkl` 失败。
- 影响：动作事件索引延后。
- 缓解：先做 recipe/ingredient/audio/object/VQA，单独建兼容环境转换 pickle。

风险 2：LightAgent 默认工具污染 baseline。

- 现象：不传 tools 仍可能暴露 Python/OSS 工具。
- 影响：TextOnly/Original 不公平。
- 缓解：wrapper/subclass 清空默认工具，并记录实际 tool list。

风险 3：VQA 样本输入格式复杂。

- 影响：video_id/time/window 解析可能不稳定。
- 缓解：先按 VQA 文件逐类解析，建立 sample parser 单元测试。

风险 4：视觉模型成本高。

- 影响：FrameOnly/Hybrid baseline 成本和延迟高。
- 缓解：先做 structured-only 和 small sample，再接 VLM。

风险 5：数据体积过大。

- 影响：误读大 CSV、误提交大文件、磁盘压力。
- 缓解：`.gitignore`、按需读取、manifest 标记 deferred。

风险 6：应用任务定义过泛。

- 影响：论文和系统都不聚焦。
- 缓解：第一主线只做 recipe/ingredient/nutrition/evidence VQA。

## 8.3 暂缓方向

暂缓：

- 全量 semidense 点云处理。
- 机器人控制。
- 纯动作分类榜单。
- 通用视频聊天。
- 完整 Web UI。

进入条件：

- 主线 baseline 已跑通。
- recipe/ingredient 指标有结果。
- event index 稳定。
- 确认该方向能提升应用价值或论文贡献。

## 8.4 关键里程碑

Milestone 1：数据可查询。

- Manifest 完成。
- Event index 完成。
- VQA parser 完成。

Milestone 2：baseline 可比较。

- LightAgent TextOnly/HDTools/RAG/FoodMemory 跑通。
- trace 和 metrics 保存。

Milestone 3：food memory 有明确收益。

- recipe/ingredient/nutrition 任务超过 TextOnly/RAG。
- evidence rate 提升。
- failure type 可解释。

Milestone 4：应用 demo。

- CLI 可查询 video/time/question。
- 返回答案、证据和状态。

Milestone 5：论文实验完整。

- baseline、ablation、错误分析、案例图表完成。

## 8.5 下一步建议

马上实施：

- 创建 `food_agent/` 基础包。
- 实现 `paths.py` 和 `config.py`。
- 实现 `build_manifest.py`。
- 写 import 和路径测试。

第一轮验证：

- 不读大文件。
- 能扫描数据。
- 能输出 manifest。
- 能提交并 push。

第一轮代码完成后再进入：

- DuckDB event index。
- LightAgent wrapper。
- VQA baseline。

