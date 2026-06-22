# HD-EPIC Agent 项目交接文档

## 1. 项目概述

**目标**：为 HD-EPIC 厨房视频 VQA 挑战赛构建多模态 Agent，回答 26K 个关于烹饪视频的问题。

**数据集**：HD-EPIC（9个参与者，41小时视频，30个问题类别）

**当前最佳成绩**：38.2%（207题，混合模式）

---

## 2. 项目结构

```
/22liushoulong/agent/hd-epic/
├── food_agent/                    # 核心代码
│   ├── agent_v2/                  # Agent 实现
│   │   ├── agent.py              # MultimodalAgent 主循环
│   │   ├── pipeline.py           # Pipeline 工具注册和实现
│   │   └── prompts.py            # SYSTEM_PROMPT 和决策模板
│   ├── evaluation/
│   │   └── api_client.py         # MimoClient（支持 OpenAI/Anthropic）
│   ├── knowledge/
│   │   └── external_knowledge.py # 食材重量、烹饪效果、厨房物品知识库
│   ├── loaders/                  # 数据加载器（Video, Audio, Gaze, SLAM, etc.）
│   ├── tools/
│   │   ├── scene_graph.py        # 场景图生成器 + ConceptNet 知识库
│   │   ├── specialized_tools.py  # 专用工具（秤读数、容器分析、物体追踪等）
│   │   └── action_analyzer.py    # 动作分析工具
│   └── reasoning/                # 推理引擎（Router, Generator, Judge）
├── scripts/
│   ├── model_server.py           # 持久化模型服务器（Unix socket）
│   ├── eval_hybrid.py            # 混合评估（工具+直接LLM）
│   ├── eval_fast.py              # 快速评估（并行）
│   ├── eval_category.py          # 单类别评估（支持断点续跑）
│   └── run_experiment_v2.py      # 完整实验脚本
├── annotations/                  # HD-EPIC 标注数据
├── data/HD-EPIC/                 # 数据集（Video, Audio, SLAM, VRS）
├── outputs/                      # 实验结果
└── .env                          # API 配置
```

---

## 3. API 配置

**当前使用的 API**（`.env` 文件）：
```
OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/anthropic
OPENAI_API_KEY=tp-c4we7ffk6i9vs9dhu8x3x8j9lzdo3lul61d6qknlaiyeufk1
FOOD_AGENT_MODEL=mimo-v2.5
FOOD_AGENT_PROVIDER_MODE=anthropic
```

**备用 API**：
```
OPENAI_BASE_URL=https://www.cctq.ai/v1
OPENAI_API_KEY=sk-EeQWTJH5SuhoC8wBiQI66ON8mFfyFSMKb79XWwUKP1WZr6VY
FOOD_AGENT_MODEL=gpt-5.4
FOOD_AGENT_PROVIDER_MODE=responses
```

**Conda 环境**：`food-epic`（Python 3.10）

---

## 4. 当前最佳结果（38.2%）

| 类别 | 准确率 | 方法 |
|------|--------|------|
| ingredient_adding | 86% | Agent + 工具 |
| object_location | 86% | 直接 LLM |
| recipe_step_recognition | 86% | 混合 |
| ingredient_retrieval | 71% | 混合 |
| ingredients_order | 71% | 工具 |
| nutrition_video | 71% | 工具 |
| why_recognition | 57% | 混合 |
| nutrition_change | 57% | 工具 |
| ingredient_weight | 57% | 工具 |
| action_localization | 57% | 直接 LLM |
| ingredient_recognition | 57% | 工具 |
| gaze_estimation | 50% | 混合 |
| **整体** | **38.2%** | **混合模式** |

**仍有 0% 的类别**：
- fixture_interaction_counting
- fixture_location
- gaze_interaction_anticipation
- object_movement_itinerary
- recipe_multi_recipe_recognition
- recipe_step_localization

---

## 5. 关键工具链

### 5.1 模型服务器
- 持久化运行，模型只加载一次
- 通过 Unix socket 通信（`/tmp/food_agent_model_server.sock`）
- 支持 8 并行请求
- 启动：`python scripts/model_server.py start`
- 停止：`python scripts/model_server.py stop`

### 5.2 Agent 工具（30+ 个）
**感知工具**：
- `query_video` - SAM3 物体检测
- `describe_frame` - MiMo Vision 描述帧
- `identify_ingredients` - 识别食材
- `query_audio` - 音频分析
- `query_gaze` - 注视数据
- `query_3d` - 3D 空间查询
- `query_hands` - 手部交互
- `query_motion` - 运动追踪
- `count_interactions` - 开/关计数
- `track_object` - 物体追踪
- `generate_scene_graph` - 场景图生成

**知识工具**：
- `query_recipe` - 菜谱查询
- `check_recipe_ingredients` - 食材检查
- `query_nutrition_kb` - 营养查询
- `estimate_ingredient_weight` - 食材重量估算
- `get_cooking_effect` - 烹饪效果
- `get_object_info` - 厨房物品信息

**专用工具**：
- `read_scale` - 读取秤的显示
- `recognize_action` - 动作识别
- `localize_action` - 动作定位
- `match_gaze_to_object` - 注视匹配
- `estimate_fixture_clock` - 固定物方向
- `predict_next_interaction` - 交互预测
- `find_static_period` - 静态周期检测

### 5.3 混合模式策略
- **工具模式**：Ingredient, Nutrition 类别（准确率 41-49%）
- **直接 LLM**：Action, Recipe, Gaze, 3D 类别（准确率 27-33%）

---

## 6. 如何运行实验

### 6.1 启动模型服务器
```bash
tmux new-session -d -s model-server "cd /22liushoulong/agent/hd-epic && /22liushoulong/Anaconda3/envs/food-epic/bin/python3 scripts/model_server.py start"
```

### 6.2 运行单类别测试
```bash
/22liushoulong/Anaconda3/envs/food-epic/bin/python3 scripts/eval_fast.py \
    --category 3d_perception_object_location \
    --num 20 \
    --parallel 4 \
    --out outputs/results/cat_objloc.json
```

### 6.3 运行完整实验
```bash
/22liushoulong/Anaconda3/envs/food-epic/bin/python3 scripts/eval_hybrid.py \
    --limit 7 \
    --parallel 4 \
    --out outputs/results/exp_hybrid.json
```

### 6.4 断点续跑
```bash
/22liushoulong/Anaconda3/envs/food-epic/bin/python3 scripts/eval_category.py \
    --category ingredient_ingredient_weight \
    --num 20 \
    --resume \
    --out outputs/results/cat_ing_weight.json
```

---

## 7. 已知问题和限制

### 7.1 技术问题
- **GroundingDINO 安装失败**：BertModel 兼容性问题，目前不使用
- **VRS 库安装失败**：需要 Meta 官方工具，目前不使用
- **SLAM 数据时间错位**：SLAM 数据不覆盖视频开始部分（0-50s）
- **API 限流**：并行请求过多会导致 429 错误

### 7.2 性能限制
- **直接 LLM 效果有限**：很多类别准确率只有 27-33%
- **视频输入效果不佳**：使用视频输入反而降低准确率
- **工具调用增加延迟**：每题 50-200 秒

### 7.3 数据限制
- **Gaze 数据精度不够**：gaze_estimation 只有 10-50%
- **数字孪生数据不完整**：fixture 位置信息有限
- **SLAM 覆盖不全**：很多时间戳没有 SLAM 数据

---

## 8. 下一步优化方向

### 8.1 短期优化
1. **优化直接 LLM 提示**：针对每个类别设计更好的提示
2. **调整混合策略**：找到每个类别的最佳方法
3. **优化视频输入**：调整帧数（4-8帧可能更好）
4. **修复已知 bug**：`'list' object has no attribute 'lower'` 间歇性错误

### 8.2 中期优化
1. **集成更强模型**：GPT-4V、Claude Vision
2. **实现类别级优化**：对每个微类别选择最佳方法
3. **改进场景图工具**：类似论文的 SceneNet
4. **深化 ConceptNet 集成**：类似论文的 KnowledgeNet

### 8.3 长期优化
1. **VRS 数据集成**：更高精度的眼动和 IMU 数据
2. **模型微调**：在 HD-EPIC 数据上微调视觉模型
3. **集成论文方法**：SceneNet + KnowledgeNet 组合
4. **类别级集成**：对每个类别选择最佳方法

---

## 9. 关键文件清单

| 文件 | 用途 |
|------|------|
| `food_agent/agent_v2/agent.py` | Agent 主循环 |
| `food_agent/agent_v2/pipeline.py` | 工具注册和实现（1600+ 行） |
| `food_agent/agent_v2/prompts.py` | SYSTEM_PROMPT 和决策模板 |
| `food_agent/evaluation/api_client.py` | MimoClient（支持 OpenAI/Anthropic/视频） |
| `food_agent/tools/scene_graph.py` | 场景图生成器 + ConceptNet |
| `food_agent/tools/specialized_tools.py` | 专用工具（秤读数、容器分析等） |
| `food_agent/tools/action_analyzer.py` | 动作分析工具 |
| `food_agent/knowledge/external_knowledge.py` | 食材重量、烹饪效果知识库 |
| `scripts/model_server.py` | 持久化模型服务器 |
| `scripts/eval_hybrid.py` | 混合评估脚本 |
| `scripts/eval_fast.py` | 快速并行评估 |
| `.env` | API 配置 |

---

## 10. Git 仓库

**远程地址**：`git@github.com-foodepic:LTaiQin/foodEpic-agent.git`

**最新提交**：`02900a6` - feat: add video input support

**分支**：`main`

---

## 11. 联系信息

如有问题，请查看：
- 项目 README：`/22liushoulong/agent/hd-epic/README.md`
- 实验结果：`/22liushoulong/agent/hd-epic/outputs/results/`
- 代码注释：各文件内的 docstring
