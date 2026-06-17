# HD-EPIC 多模态厨房智能 Agent 技术方案

> 基于 HD-EPIC 数据集的全维度利用，构建自主推理的第一人称厨房视频理解 Agent

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [HD-EPIC 数据集全景分析](#2-hd-epic-数据集全景分析)
3. [现有系统分析与差距](#3-现有系统分析与差距)
4. [Agent 整体架构设计](#4-agent-整体架构设计)
5. [七大感知模块详细设计](#5-七大感知模块详细设计)
6. [推理引擎设计](#6-推理引擎设计)
7. [关键技术路线](#7-关键技术路线)
8. [30 天实施计划](#8-30-天实施计划)
9. [评估体系](#9-评估体系)
10. [风险与应对](#10-风险与应对)
11. [参考文献](#11-参考文献)

---

## 1. 项目背景与目标

### 1.1 背景

HD-EPIC (arXiv:2502.04144, CVPR 2025) 是目前最精细的第一人称厨房视频数据集：

- **41 小时**非脚本化厨房视频，来自 **9 个不同家庭厨房**
- **每分钟 263 个标注**，覆盖 7 大维度
- 包含 26K 多选题的 VQA Benchmark，目前最佳方案仅 **44.21%** 准确率

当前已有的数据模块：

| 模块 | 大小 | 描述 |
|------|------|------|
| Audio-HDF5 | 27 GB | 51K 音频事件（切菜、水流、搅拌、器具碰撞等） |
| Digital-Twin | 1.35 GB | 413 个厨房固定装置的 3D 数字孪生模型 |
| Hands-Masks | 1.95 GB | 37K 手部/物体掩码，已提升到 3D 空间 |
| SLAM-and-Gaze | 349 GB | SLAM 轨迹 + 眼动追踪数据 |
| VRS | 1.9 TB | Meta Aria 眼镜原始多传感器流 |
| Videos (mp4) | 115.5 GB | 标准格式视频 |

### 1.2 现有系统

当前 Agent 已实现：
- 调用 API 识别音频事件，定位关键时间节点
- 基于 QA 标注定位视频关键片段
- 大模型自主判断证据充分性，不充分时自动扩展搜索范围
- 全流程无人干预，AI 自主调用 tools

### 1.3 目标

构建一个 **七维感知、自主推理** 的厨房视频理解 Agent：

1. **全维度覆盖**: 利用 HD-EPIC 全部 7 个标注维度（音频、视觉、Gaze、3D 空间、手部交互、营养、动作序列）
2. **自适应路由**: 根据问题类型自动选择最优感知通道组合
3. **多路证据融合**: 多模态证据交叉验证，提升推理可靠性
4. **自主深度控制**: Agent 自主判断证据是否充分，决定是否深入探索
5. **目标性能**: 在 HD-EPIC VQA Benchmark 上达到 **50%+** 准确率（超越当前 SOTA 44.21%）

---

## 2. HD-EPIC 数据集全景分析

### 2.1 数据维度矩阵

```
┌─────────────────┬──────────────┬─────────────────────────────────────────────┐
│ 维度            │ 数据量       │ 包含信息                                     │
├─────────────────┼──────────────┼─────────────────────────────────────────────┤
│ 视频 (Video)    │ 41h / 115GB  │ 第一人称 RGB 视频帧                          │
│ 音频 (Audio)    │ 27GB HDF5    │ 51K 声音事件 + 时间戳 + 类型                 │
│ 注视 (Gaze)     │ 349GB(部分)  │ 注视点坐标、注视轨迹、注视目标               │
│ 3D 空间 (SLAM)  │ 349GB(部分)  │ 6DoF 位姿轨迹、3D 点云                      │
│ 数字孪生 (DT)   │ 1.35GB       │ 413 个固定装置的 3D 模型 + 语义标签          │
│ 手部 (Hands)    │ 1.95GB       │ 手部掩码、物体掩码、3D 关节位置              │
│ 原始流 (VRS)    │ 1.9TB        │ RGB + 眼动 + IMU + 深度等多传感器同步流      │
├─────────────────┼──────────────┼─────────────────────────────────────────────┤
│ 菜谱 (Recipe)   │ 69 道菜      │ 步骤序列 + 步骤描述 + 时间范围               │
│ 动作 (Action)   │ 59K 个       │ 细粒度动作标签 + 起止时间 + 关联物体         │
│ 食材 (Ingredient)│ 标注         │ 食材名称 + 出现时间 + 关联动作               │
│ 营养 (Nutrition) │ 标注         │ 卡路里 + 蛋白质 + 碳水 + 脂肪               │
│ 物体运动 (ObjMot)│ 20K 个      │ 物体 ID + 起止位置 + 3D 轨迹                │
│ 音频事件 (Audio) │ 51K 个       │ 事件类型 + 时间戳 + 持续时间                 │
│ 物体掩码 (Mask)  │ 37K 个       │ 像素级掩码 + 3D 提升 + 物体 ID              │
└─────────────────┴──────────────┴─────────────────────────────────────────────┘
```

### 2.2 VQA Benchmark 七大任务类别

| 类别 | 问题示例 | 所需能力 | 最相关数据维度 |
|------|---------|---------|---------------|
| **Recipe** | "正在做哪道菜的第几步？" | 菜谱识别、步骤匹配 | 视频 + 食材 + 动作序列 |
| **Ingredient** | "加入了什么食材？" | 食材视觉识别 | 视频 + 手部 + 音频 |
| **Nutrition** | "这顿饭的卡路里？" | 营养估算 | 食材 + 营养映射 + 份量估计 |
| **Fine-grained Action** | "正在做什么动作？" | 细粒度动作分类 | 视频 + 手部 + 音频 |
| **3D Perception** | "水槽在哪里？" | 空间关系推理 | Digital Twin + SLAM |
| **Object Motion** | "哪个物体被移动了？" | 物体追踪 | 手部 + 物体运动 + SLAM |
| **Gaze** | "佩戴者在看什么？" | 注视目标识别 | Gaze + 视频 |

### 2.3 各数据模块之间的关联

```
                    ┌──────────────┐
                    │  Recipe 步骤  │
                    │ (时间轴主干)  │
                    └──────┬───────┘
                           │ 时间对齐
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │   Audio    │   │   Video   │   │   Gaze    │
    │ (51K事件)  │   │ (RGB帧)   │   │ (注视轨迹)│
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │                │
          │         ┌─────▼─────┐          │
          │         │   Hands   │          │
          │         │ (手部掩码) │          │
          │         └─────┬─────┘          │
          │               │                │
    ┌─────▼───────────────▼────────────────▼─────┐
    │           SLAM + Digital Twin               │
    │         (3D 空间定位 + 场景理解)             │
    └─────────────────────┬──────────────────────┘
                          │
                    ┌─────▼─────┐
                    │ Nutrition │
                    │ (营养映射) │
                    └───────────┘
```

---

## 3. 现有系统分析与差距

### 3.1 当前系统架构

```
音频 API ──→ 时间定位 ──→ 视频切片 ──→ 大模型判断 ──→ QA 回答
QA 标注 ──→ 时间定位 ──↗        ↑
                              证据充分？
                              不足则扩展搜索
```

### 3.2 已利用 vs 未利用

| 数据维度 | 当前状态 | 潜在价值 |
|----------|---------|---------|
| 视频 (Video) | ✅ 已使用 | 可深化：场景图、物体检测 |
| 音频 (Audio) | ✅ 已使用 | 可深化：声音源定位、因果推理 |
| QA 标注 | ✅ 已使用 | 时间定位锚点 |
| Gaze | ❌ 未使用 | **最强预判信号**：注意力=意图 |
| SLAM | ❌ 未使用 | 空间推理、场景分割 |
| Digital Twin | ❌ 未使用 | 3D 空间问答的直接查询源 |
| Hands-Masks | ❌ 未使用 | 精细动作识别、手-物交互 |
| Nutrition | ❌ 未使用 | 营养估计任务 |
| Recipe 步骤 | ❌ 未使用 | 程序推理、步骤预测 |
| Object Motion | ❌ 未使用 | 物体追踪、状态变化 |

### 3.3 差距总结

- 当前系统是 **单通道**（音频+视频）驱动的
- 缺乏 **空间感知**（不知道厨房布局）
- 缺乏 **注意力引导**（不知道佩戴者在看什么）
- 缺乏 **手部精细分析**（不知道手在做什么）
- 缺乏 **知识推理**（不知道菜谱流程、营养知识）
- 证据融合逻辑简单，无多模态交叉验证

---

## 4. Agent 整体架构设计

### 4.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户提问                                  │
│                   "这道菜有多少卡路里？"                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  问题路由器  │  ← 问题分类 + 路由策略
                    │  (Router)   │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ 感知模块群 │   │ 知识模块群 │   │ 推理模块群 │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │                │
    ┌─────▼───────────────────────────────────────┐
    │            证据聚合引擎 (Aggregator)          │
    │    多路证据 → 冲突检测 → 置信度评估 → 融合    │
    └──────────────────────┬──────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  自适应判断  │  ← 证据充分？
                    │  (Judge)    │     不足 → 回到感知模块
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  答案生成    │
                    │ (Generator) │
                    └─────────────┘
```

### 4.2 模块职责划分

#### 感知模块群 (Perception Modules)

| 模块 | 输入 | 输出 | 对应数据维度 |
|------|------|------|-------------|
| AudioAnalyzer | Audio-HDF5 | 声音事件列表 + 时间戳 + 置信度 | Audio |
| VisualAnalyzer | Video frames | 场景描述 + 物体列表 + 动作 | Video |
| GazeTracker | SLAM-and-Gaze | 注视目标 + 注意力热力图 | Gaze |
| SpatialReasoner | Digital-Twin + SLAM | 3D 空间关系 + 物体位置 | 3D / DT |
| HandInteractor | Hands-Masks | 手-物交互对 + 动作类型 | Hands |
| NutritionEstimator | Ingredient + Nutrition | 营养成分表 | Nutrition |
| MotionTracker | Object Motion + SLAM | 物体运动轨迹 + 状态变化 | ObjMot |

#### 知识模块群 (Knowledge Modules)

| 模块 | 功能 | 数据源 |
|------|------|--------|
| RecipeKB | 菜谱知识库，步骤查询 | Recipe 标注 |
| NutritionKB | 食材-营养映射表 | Nutrition 标注 |
| CommonSenseKB | 厨房常识推理 | ConceptNet / 内建 |
| SceneGraphKB | 场景图（物体+关系） | 实时生成 |

#### 推理模块群 (Reasoning Modules)

| 模块 | 功能 |
|------|------|
| TemporalReasoner | 时间推理：先后顺序、持续时间、同时发生 |
| SpatialReasoner | 空间推理：方位关系、距离、遮挡 |
| CausalReasoner | 因果推理：动作-结果、食材-菜品 |
| ProceduralReasoner | 程序推理：菜谱步骤、动作序列 |

---

## 5. 七大感知模块详细设计

### 5.1 模块 1: AudioAnalyzer（音频分析器）

**数据源**: Audio-HDF5 (27GB, 51K 事件)

**功能**:
- 从 HDF5 中提取音频事件的时间戳、类型、持续时间
- 声音事件分类：切菜声、水流声、搅拌声、油炸声、器具碰撞声、冰箱声、微波炉声等
- 声音源时间聚类：将相近的声音事件聚合为"活动段"

**输出格式**:
```json
{
  "audio_events": [
    {
      "type": "chopping",
      "start_time": 120.5,
      "end_time": 135.2,
      "confidence": 0.92,
      "duration": 14.7
    }
  ],
  "activity_segments": [
    {
      "label": "vegetable_preparation",
      "time_range": [120.0, 160.0],
      "dominant_sounds": ["chopping", "water_running"],
      "evidence_strength": "strong"
    }
  ]
}
```

**调用策略**:
- 首次调用：粗粒度扫描，获取全局声音事件分布
- 按需深入：对特定时间段做精细声音分析
- 与视频帧对齐：声音事件映射到对应视频片段

**实现要点**:
- HDF5 格式高效读取，按时间范围索引避免全量加载
- 声音事件可直接作为视频切片的时间锚点
- 声音序列可推断烹饪阶段（如：切→炒→装盘）

---

### 5.2 模块 2: VisualAnalyzer（视觉分析器）

**数据源**: Videos (mp4, 115.5GB)

**功能**:
- 视频帧采样 + 大模型视觉理解（当前已实现）
- 新增：场景图生成（物体、属性、关系）
- 新增：物体检测与跟踪
- 新增：动作识别（结合时序信息）

**输出格式**:
```json
{
  "frame_analysis": {
    "timestamp": 125.0,
    "objects": [
      {"name": "cutting_board", "bbox": [100, 200, 400, 500], "attributes": ["wooden", "has_food_on"]},
      {"name": "knife", "bbox": [250, 300, 350, 450], "attributes": ["held_by_hand"]}
    ],
    "relations": [
      {"subject": "hand", "predicate": "holding", "object": "knife"},
      {"subject": "knife", "predicate": "cutting", "object": "tomato"}
    ],
    "scene_description": "Person is chopping tomatoes on a wooden cutting board"
  }
}
```

**调用策略**:
- 关键帧提取：基于音频事件/Gaze 热点定位关键帧
- 自适应采样：动作快时高频采样，静止时低频采样
- 与 Hands-Masks 联合：手部掩码辅助精确定位交互物体

---

### 5.3 模块 3: GazeTracker（注视追踪器）

**数据源**: SLAM-and-Gaze (349GB 中的 Gaze 部分)

**功能**:
- 提取注视点坐标 (x, y) 和时间戳
- 计算注视轨迹（saccade 和 fixation）
- 生成注意力热力图：哪些区域被长时间注视
- 注视目标推断：结合 Digital Twin 判断"在看什么"

**输出格式**:
```json
{
  "gaze_events": [
    {
      "type": "fixation",
      "target": "cutting_board",
      "start_time": 122.0,
      "end_time": 128.5,
      "duration": 6.5,
      "position_3d": [1.2, 0.8, 0.5],
      "confidence": 0.88
    }
  ],
  "attention_heatmap": {
    "high_attention_regions": ["cutting_board", "stove", "sink"],
    "current_focus": "cutting_board",
    "predicted_next_focus": "stove"
  }
}
```

**调用策略**:
- **预判信号**: 当 gaze 聚焦某区域 > 2s 时，预判即将发生交互，提前截取该区域视频
- **问题引导**: "佩戴者在看什么"类问题直接用 gaze 回答
- **时间锚点**: gaze fixation 的起止时间可作为视频切片的精确锚点

**核心价值**:
Gaze 是最强的意图预判信号。人类注视某物体通常意味着：
1. 即将与之交互（拿起、操作）
2. 正在评估其状态（检查食物是否煮熟）
3. 计划下一步动作

---

### 5.4 模块 4: SpatialReasoner（空间推理器）

**数据源**: Digital-Twin (1.35GB) + SLAM (349GB 中的 SLAM 部分)

**功能**:
- Digital Twin 解析：提取厨房固定装置（水槽、炉灶、台面、冰箱、橱柜等）的 3D 位置、尺寸、语义标签
- SLAM 轨迹解析：提取佩戴者的 6DoF 位姿 (x, y, z, roll, pitch, yaw)
- 空间关系计算："在...左边"、"在...上面"、"靠近..."、"面对..."
- 场景分割：根据 SLAM 轨迹自动识别佩戴者所在的厨房区域

**输出格式**:
```json
{
  "kitchen_layout": {
    "fixtures": [
      {"id": "sink_01", "type": "sink", "position": [1.0, 0.9, 0.0], "size": [0.6, 0.4, 0.2]},
      {"id": "stove_01", "type": "stove", "position": [2.5, 0.9, 0.0], "size": [0.6, 0.6, 0.1]}
    ],
    "spatial_relations": [
      {"from": "stove_01", "relation": "right_of", "to": "sink_01", "distance": 1.5}
    ]
  },
  "wearer_pose": {
    "timestamp": 125.0,
    "position": [1.8, 1.6, 0.0],
    "facing": "stove_01",
    "nearest_fixture": "stove_01",
    "distance_to_nearest": 0.7
  },
  "scene_segment": "cooking_area"
}
```

**调用策略**:
- 空间类问题直接查询 Digital Twin（"水槽在哪里" → 直接返回 3D 坐标）
- 结合 SLAM 判断佩戴者当前位置和朝向
- 辅助视频理解：知道佩戴者面对什么，帮助理解画面内容

---

### 5.5 模块 5: HandInteractor（手部交互分析器）

**数据源**: Hands-Masks (1.95GB, 37K 掩码)

**功能**:
- 手部掩码提取：像素级手部区域分割
- 手-物交互识别：手正在接触什么物体
- 动作分类：基于手部运动模式和接触物体推断动作类型
- 3D 手部姿态：关节位置信息

**输出格式**:
```json
{
  "hand_interactions": [
    {
      "timestamp": 125.0,
      "hand": "right",
      "contact_object": "knife",
      "interaction_type": "grasping",
      "grip_style": "precision_grip",
      "action_inferred": "cutting"
    }
  ],
  "object_states": [
    {
      "object": "tomato",
      "state_change": "being_cut",
      "previous_state": "whole",
      "current_state": "sliced"
    }
  ]
}
```

**调用策略**:
- 动作识别的最直接证据（比纯视频更准确）
- "哪个物体被操作了"类问题直接用手部掩码回答
- 与 Audio 联合：手部动作 + 声音 = 高置信度动作识别

---

### 5.6 模块 6: NutritionEstimator（营养估计器）

**数据源**: Ingredient 标注 + Nutrition 标注

**功能**:
- 食材识别：从视频/标注中识别使用的食材
- 份量估计：结合手部交互和物体大小估算份量
- 营养查询：查表获取每种食材的营养成分
- 总量计算：汇总所有食材的营养成分

**输出格式**:
```json
{
  "dish_nutrition": {
    "dish_name": "tomato_salad",
    "ingredients": [
      {"name": "tomato", "amount_g": 200, "calories": 36, "protein_g": 1.8, "carbs_g": 7.8, "fat_g": 0.4},
      {"name": "olive_oil", "amount_g": 15, "calories": 119, "protein_g": 0, "carbs_g": 0, "fat_g": 13.5},
      {"name": "salt", "amount_g": 2, "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
    ],
    "total": {
      "calories": 155,
      "protein_g": 1.8,
      "carbs_g": 7.8,
      "fat_g": 13.9
    }
  }
}
```

**调用策略**:
- 营养类问题直接查询
- 需要结合视觉确认食材实际使用量（不能只看标注）
- 参考 V-Nutri (CVPR 2026) 的方法：烹饪过程关键帧提供额外营养证据

---

### 5.7 模块 7: MotionTracker（物体运动追踪器）

**数据源**: Object Motion 标注 (20K) + SLAM + Hands-Masks

**功能**:
- 物体位置追踪：物体从 A 移动到 B 的完整轨迹
- 状态变化检测：物体状态变化（完整→切碎、冷→热、生→熟）
- 3D 轨迹重建：结合 SLAM 和 Digital Twin 将 2D 运动提升到 3D

**输出格式**:
```json
{
  "object_movements": [
    {
      "object": "tomato",
      "movement_type": "displacement",
      "from": {"position": [2.5, 0.9, 0.1], "container": "cutting_board"},
      "to": {"position": [2.5, 0.9, 0.0], "container": "plate"},
      "time_range": [180.0, 185.0],
      "actor_hand": "right",
      "movement_arc": "lift_and_place"
    }
  ],
  "state_changes": [
    {
      "object": "water",
      "property": "temperature",
      "from": "cold",
      "to": "hot",
      "time_range": [60.0, 180.0],
      "evidence": "steam_visible + stove_on"
    }
  ]
}
```

---

## 6. 推理引擎设计

### 6.1 问题路由器 (Router)

**核心思想**: 不同类型的问题需要不同的感知通道组合，路由器负责选择最优策略。

```python
ROUTE_TABLE = {
    "recipe": {
        "primary": ["VisualAnalyzer", "RecipeKB"],
        "secondary": ["AudioAnalyzer", "HandInteractor"],
        "reasoning": "ProceduralReasoner"
    },
    "ingredient": {
        "primary": ["VisualAnalyzer", "HandInteractor"],
        "secondary": ["AudioAnalyzer", "MotionTracker"],
        "reasoning": "CausalReasoner"
    },
    "nutrition": {
        "primary": ["NutritionEstimator", "VisualAnalyzer"],
        "secondary": ["HandInteractor"],
        "reasoning": "CausalReasoner"
    },
    "fine_grained_action": {
        "primary": ["HandInteractor", "AudioAnalyzer"],
        "secondary": ["VisualAnalyzer", "MotionTracker"],
        "reasoning": "TemporalReasoner"
    },
    "3d_perception": {
        "primary": ["SpatialReasoner"],
        "secondary": ["GazeTracker", "VisualAnalyzer"],
        "reasoning": "SpatialReasoner"
    },
    "object_motion": {
        "primary": ["MotionTracker", "HandInteractor"],
        "secondary": ["SpatialReasoner", "VisualAnalyzer"],
        "reasoning": "TemporalReasoner"
    },
    "gaze": {
        "primary": ["GazeTracker"],
        "secondary": ["VisualAnalyzer", "SpatialReasoner"],
        "reasoning": "SpatialReasoner"
    }
}
```

### 6.2 证据聚合引擎 (Aggregator)

**多路证据融合流程**:

```
1. 收集各模块输出的证据
2. 时间对齐：将不同模态的证据对齐到统一时间轴
3. 冲突检测：如果不同模块给出矛盾信号
   → 降低置信度
   → 请求更多证据
4. 置信度加权融合：
   - 主通道证据权重: 0.6
   - 辅助通道证据权重: 0.3
   - 常识推理权重: 0.1
5. 输出：融合后的证据包 + 总体置信度
```

### 6.3 自适应深度控制 (Judge)

```
置信度 > 0.8 → 直接生成答案
0.5 < 置信度 ≤ 0.8 → 扩展搜索（增加采样帧、查询更多模块）
置信度 ≤ 0.5 → 全面搜索（遍历所有相关模块）
```

### 6.4 推理模式

#### TemporalReasoner（时间推理）
- 事件先后顺序判断
- 持续时间计算
- 同时性检测
- 示例："切完番茄之后做了什么？"

#### SpatialReasoner（空间推理）
- 方位关系（上下左右前后）
- 距离计算
- 容器关系（在...里面/上面）
- 示例："水槽在佩戴者的哪一侧？"

#### CausalReasoner（因果推理）
- 动作-结果链（切番茄 → 番茄变碎）
- 食材-菜品关系（番茄+生菜 → 沙拉）
- 示例："为什么加入橄榄油？"

#### ProceduralReasoner（程序推理）
- 菜谱步骤序列匹配
- 当前步骤推断
- 下一步预测
- 示例："这是第几步？下一步是什么？"

---

## 7. 关键技术路线

### 7.1 数据预处理层

```
原始数据 (VRS 1.9TB)
    │
    ├──→ 视频解码 → MP4 (已就绪)
    ├──→ 音频提取 → HDF5 (已就绪)
    ├──→ Gaze 提取 → 时间序列 (需解析)
    ├──→ SLAM 提取 → 6DoF 轨迹 (需解析)
    └──→ 传感器同步 → 统一时间轴 (需实现)
```

**关键技术**:
- VRS 格式解析：使用 Meta 的 `vrs` Python 库
- 多传感器时间对齐：基于硬件时间戳的插值对齐
- HDF5 高效索引：按时间范围查询，避免全量加载

### 7.2 感知层

```
各感知模块独立运行，输出统一格式的证据 JSON
    │
    ├──→ AudioAnalyzer: HDF5 → 声音事件
    ├──→ VisualAnalyzer: MP4 + MiMo2.5 API → 场景理解
    ├──→ GazeTracker: Gaze 数据 + DT → 注视目标
    ├──→ SpatialReasoner: SLAM + DT → 空间关系
    ├──→ HandInteractor: Masks → 手-物交互
    ├──→ NutritionEstimator: 标注 + 查表 → 营养
    └──→ MotionTracker: Motion 标注 + SLAM → 物体轨迹
```

**关键技术**:
- API 调用优化：批量请求、缓存、重试策略
- 大模型 Prompt Engineering：针对每个模块设计专用 prompt
- 证据格式标准化：所有模块输出统一 JSON Schema

### 7.3 推理层

```
Router → 选择感知通道组合
    │
Aggregator → 多路证据融合
    │
Judge → 判断证据充分性
    │
    ├── 充分 → Generator → 输出答案
    └── 不足 → 回到 Router，扩展搜索
```

**关键技术**:
- 问题分类模型：基于 LLM 的 zero-shot 分类
- 证据融合算法：加权投票 + 冲突检测
- 自适应停止条件：最大迭代次数 + 置信度阈值

### 7.4 工具调用层

Agent 自主调用 tools 的设计：

```python
TOOLS = {
    # 感知工具
    "query_audio":      "查询音频事件",
    "query_video":      "查询视频帧/片段",
    "query_gaze":       "查询注视数据",
    "query_3d":         "查询 3D 空间信息",
    "query_hands":      "查询手部交互",
    "query_nutrition":  "查询营养信息",
    "query_motion":     "查询物体运动",
    
    # 知识工具
    "query_recipe":     "查询菜谱步骤",
    "query_commonsense":"查询常识知识",
    "query_scene_graph":"查询场景图",
    
    # 分析工具
    "generate_scene_graph": "对视频帧生成场景图",
    "compute_spatial":      "计算空间关系",
    "infer_action":         "推断动作类型",
    "predict_next":         "预测下一步",
    
    # 控制工具
    "check_evidence":   "检查证据充分性",
    "expand_search":    "扩展搜索范围",
    "synthesize_answer": "综合证据生成答案"
}
```

### 7.5 技术栈

| 层次 | 技术选型 | 说明 |
|------|---------|------|
| 大模型 API | MiMo2.5 | 视觉理解 + 推理 |
| 视频处理 | FFmpeg + OpenCV | 视频解码、帧提取 |
| 3D 数据处理 | Open3D / trimesh | Digital Twin 解析 |
| SLAM 解析 | Open3D + numpy | 轨迹数据处理 |
| HDF5 读取 | h5py | 高效索引音频数据 |
| Gaze 处理 | numpy + scipy | 时间序列分析 |
| 场景图生成 | MiMo2.5 Vision | 帧级场景图 |
| 知识图谱 | ConceptNet API | 常识推理 |
| Agent 框架 | 自研 Tool-calling Loop | 自主决策 |
| 评估 | 自动化脚本 | 对齐 HD-EPIC Benchmark |

---

## 8. 30 天实施计划

### Phase 1: 基础设施 + 核心模块（Day 1-10）

#### Week 1 (Day 1-7): 数据层 + 基础模块

| 天数 | 任务 | 产出 | 依赖 |
|------|------|------|------|
| **Day 1** | 环境搭建 | 项目骨架、依赖安装、数据路径配置 | 无 |
| **Day 2** | VRS 数据解析 | VRS → 帧级 RGB + 时间戳的读取工具 | VRS 库 |
| **Day 3** | Audio-HDF5 读取 | HDF5 高效索引工具，按时间范围查询 | h5py |
| **Day 4** | Gaze 数据解析 | 注视点坐标提取、时间对齐 | SLAM-and-Gaze |
| **Day 5** | SLAM 轨迹解析 | 6DoF 位姿提取、轨迹可视化 | SLAM-and-Gaze |
| **Day 6** | Digital Twin 解析 | 3D 模型加载、固定装置语义查询 | Digital-Twin |
| **Day 7** | Hands-Masks 读取 | 掩码加载、手-物接触检测 | Hands-Masks |

**Day 1-7 里程碑**: 所有 6 个数据模块可独立读取和查询

#### Week 1.5 (Day 8-10): 感知模块 + 工具封装

| 天数 | 任务 | 产出 |
|------|------|------|
| **Day 8** | AudioAnalyzer + VisualAnalyzer | 两个核心感知模块的 tool 封装 |
| **Day 9** | GazeTracker + SpatialReasoner | 注视追踪 + 空间推理 tool 封装 |
| **Day 10** | HandInteractor + MotionTracker | 手部交互 + 物体运动 tool 封装 |

**Day 8-10 里程碑**: 7 个感知模块全部封装为可调用 tools

---

### Phase 2: 推理引擎 + 路由器（Day 11-18）

#### Day 11-14: 推理引擎核心

| 天数 | 任务 | 产出 |
|------|------|------|
| **Day 11** | 问题路由器 (Router) | 问题分类 + 路由策略实现 |
| **Day 12** | 证据聚合引擎 (Aggregator) | 多路证据融合 + 冲突检测 |
| **Day 13** | 自适应深度控制 (Judge) | 置信度评估 + 扩展搜索逻辑 |
| **Day 14** | 答案生成器 (Generator) | 综合证据生成最终答案 |

#### Day 15-18: Agent 主循环

| 天数 | 任务 | 产出 |
|------|------|------|
| **Day 15** | Agent 主循环 | 完整的 Router → Perception → Aggregator → Judge → Generator 流程 |
| **Day 16** | Prompt 优化 | 针对每个模块优化 MiMo2.5 prompt |
| **Day 17** | 错误处理 + 重试 | API 调用失败、数据缺失的容错 |
| **Day 18** | 端到端测试 | 基础功能验证，跑通完整流程 |

**Day 11-18 里程碑**: Agent 可以接收问题 → 自主调用 tools → 输出答案

---

### Phase 3: 知识模块 + 高级推理（Day 19-24）

| 天数 | 任务 | 产出 |
|------|------|------|
| **Day 19** | RecipeKB 实现 | 菜谱知识库构建、步骤查询接口 |
| **Day 20** | NutritionKB 实现 | 食材-营养映射表、营养计算 |
| **Day 21** | SceneGraphKB 实现 | 实时场景图生成 + 缓存 |
| **Day 22** | CommonSenseKB 集成 | ConceptNet 接入、常识推理 |
| **Day 23** | 高级推理器 | 因果推理 + 程序推理实现 |
| **Day 24** | Gaze 引导优化 | 用 gaze 数据预判视频切片位置 |

**Day 19-24 里程碑**: 全部知识模块就绪，高级推理能力上线

---

### Phase 4: 评估 + 优化（Day 25-30）

| 天数 | 任务 | 产出 |
|------|------|------|
| **Day 25** | HD-EPIC VQA 评估框架 | 自动化评估脚本，对齐 26K 问题 |
| **Day 26** | 分类别评估 | 7 个类别分别评估，找短板 |
| **Day 27** | 路由策略优化 | 基于评估结果调整路由权重 |
| **Day 28** | Prompt 调优 | 针对低分 category 优化 prompt |
| **Day 29** | 融合策略优化 | 调整证据权重、置信度阈值 |
| **Day 30** | 最终评估 + 文档 | 完整评估报告、系统文档 |

**Day 25-30 里程碑**: 系统评估完成，目标 50%+ 准确率

---

### 30 天甘特图

```
Day  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
     ├──────────────────────┤
     │  Phase 1: 基础设施    │
     │  ├─ 数据解析(1-7)     │
     │  └─ 工具封装(8-10)    │
                           ├──────────────────────┤
                           │  Phase 2: 推理引擎    │
                           │  ├─ 核心引擎(11-14)   │
                           │  └─ Agent循环(15-18)  │
                                                   ├──────────────┤
                                                   │ Phase 3: 知识 │
                                                   │ ├─ KB(19-22)  │
                                                   │ └─ 高级(23-24)│
                                                                  ├──────────────┤
                                                                  │ Phase 4: 评估 │
                                                                  │ ├─ 框架(25-26)│
                                                                  │ └─ 优化(27-30)│
```

### 关键里程碑检查点

| 检查点 | 日期 | 验收标准 |
|--------|------|---------|
| M1 | Day 7 | 6 个数据模块可独立读取查询 |
| M2 | Day 10 | 7 个感知 tool 可调用并返回结果 |
| M3 | Day 18 | Agent 端到端跑通，能回答简单问题 |
| M4 | Day 24 | 全部模块就绪，覆盖 7 大任务类别 |
| M5 | Day 30 | HD-EPIC VQA 准确率 ≥ 50% |

---

## 9. 评估体系

### 9.1 评估维度

| 评估项 | 指标 | 目标值 |
|--------|------|--------|
| 总体准确率 | Accuracy | ≥ 50% |
| Recipe 类 | Accuracy | ≥ 55% |
| Ingredient 类 | Accuracy | ≥ 50% |
| Nutrition 类 | Accuracy | ≥ 40% |
| Fine-grained Action 类 | Accuracy | ≥ 50% |
| 3D Perception 类 | Accuracy | ≥ 45% |
| Object Motion 类 | Accuracy | ≥ 45% |
| Gaze 类 | Accuracy | ≥ 55% |
| 证据充分率 | 首次判断即充分的比例 | ≥ 70% |
| 平均工具调用次数 | 每个问题的 tool 调用数 | ≤ 5 |
| 响应延迟 | 单问题平均耗时 | ≤ 30s |

### 9.2 消融实验设计

为验证每个模块的价值，设计消融实验：

| 实验 | 去除模块 | 预期影响 |
|------|---------|---------|
| Baseline | 仅 Video + Audio（当前系统） | 基准线 |
| +Gaze | 加入 GazeTracker | Gaze 类 +50% |
| +3D | 加入 SpatialReasoner | 3D Perception +40% |
| +Hands | 加入 HandInteractor | Action 类 +30% |
| +All | 全部模块 | 总体 +30% |

---

## 10. 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|---------|
| VRS 解析困难 | 中 | 阻塞数据读取 | 使用 Meta 官方 vrs 库 + 社区文档 |
| SLAM 数据格式不明 | 中 | 阻塞空间推理 | 先用 Digital Twin 做 3D 查询，SLAM 后补 |
| API 调用成本高 | 高 | 限制实验次数 | 缓存机制 + 关键帧采样减少调用 |
| 多模态对齐误差 | 中 | 降低融合精度 | 以音频时间戳为基准对齐所有模态 |
| 30 天时间紧张 | 高 | 部分模块延期 | 优先级排序：先 Video+Audio+Gaze，后 3D+Hands |
| 50% 目标过高 | 中 | 未达预期 | 保守目标 45%，冲刺 50% |

---

## 11. 参考文献

1. **HD-EPIC 原始论文**: Perrett et al., "HD-EPIC: A Highly-Detailed Egocentric Video Dataset", CVPR 2025. arXiv:2502.04144
2. **SceneNet+KnowledgeNet (2025 VQA 冠军)**: Taluzzi et al., "From Pixels to Graphs: using Scene and Knowledge Graphs for HD-EPIC VQA Challenge", arXiv:2506.08553
3. **EgoAdapt (CVPR 2026)**: Chen et al., "EgoAdapt: A Multi-Scene Egocentric Adaptation Method for CVPR 2026 HD-EPIC VQA Challenge", arXiv:2605.24500
4. **EgoReasoner**: Zhu et al., "EgoReasoner: Learning Egocentric 4D Reasoning via Task-Adaptive Structured Thinking", arXiv:2603.06561
5. **Ego2World**: Cheng et al., "Ego2World: Compiling Egocentric Cooking Videos into Executable Worlds for Belief-State Planning", arXiv:2605.13335
6. **V-Nutri (CVPR 2026 MetaFood)**: Yue et al., "V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos", arXiv:2604.11913
7. **PAWS**: Wang et al., "PAWS: Perception of Articulation in the Wild at Scale from Egocentric Videos", arXiv:2603.25539
8. **EgoFlow (CVPR 2026)**: Saroha et al., "EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation", arXiv:2604.01421
9. **Egocentric Co-Pilot (WWW 2026)**: Yang et al., "Egocentric Co-Pilot: Web-Native Smart-Glasses Agents for Assistive Egocentric AI", arXiv:2603.01104
10. **Gaze + Set-of-Mark**: Materia et al., "Leveraging Gaze and Set-of-Mark in VLLMs for Human-Object Interaction Anticipation from Egocentric Videos", arXiv:2604.03667
11. **Semantic+Visual Evidence**: Xu et al., "Semantic and Visual Evidence for Efficient Long-Video Reasoning", arXiv:2605.29402

---

## 附录 A: Agent 工具调用完整列表

```yaml
感知工具:
  - query_audio_events(time_range, event_types?)
    # 查询指定时间范围内的音频事件
  - query_video_frames(time_range, sampling_rate?, key_frames?)
    # 查询视频帧，支持关键帧提取
  - query_gaze(time_range, target?)
    # 查询注视数据，可按目标过滤
  - query_3d_layout(fixture_type?, region?)
    # 查询 3D 厨房布局
  - query_wearer_pose(time)
    # 查询佩戴者在指定时刻的位姿
  - query_hand_interactions(time_range, object?)
    # 查询手部交互，可按物体过滤
  - query_object_motion(time_range, object?)
    # 查询物体运动轨迹
  - query_nutrition(ingredients)
    # 查询食材营养成分
  - query_recipe_steps(recipe_name?)
    # 查询菜谱步骤

分析工具:
  - generate_scene_graph(frame)
    # 对指定帧生成场景图
  - compute_spatial_relation(obj_a, obj_b)
    # 计算两个物体的空间关系
  - infer_action_from_hands(hand_data, audio_data)
    # 基于手部+音频推断动作
  - predict_next_step(recipe, current_step)
    # 预测菜谱下一步

控制工具:
  - classify_question(question)
    # 对问题进行分类
  - evaluate_evidence(evidence_list)
    # 评估证据充分性
  - expand_search(question, current_evidence, strategy)
    # 扩展搜索范围
  - synthesize_answer(question, evidence_list)
    # 综合证据生成答案
```

---

## 附录 B: 证据 JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "evidence_id": {"type": "string"},
    "source_module": {"type": "string"},
    "evidence_type": {"type": "string", "enum": ["audio", "visual", "gaze", "spatial", "hand", "nutrition", "motion"]},
    "time_range": {
      "type": "object",
      "properties": {
        "start": {"type": "number"},
        "end": {"type": "number"}
      }
    },
    "content": {"type": "object"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "supporting_data": {"type": "object"}
  },
  "required": ["evidence_id", "source_module", "evidence_type", "confidence"]
}
```

---

## 附录 C: 项目目录结构

```
foodVedio/
├── HD-EPIC_MultiModal_Agent_Proposal.md  # 本文档
├── README.md
├── requirements.txt
├── config/
│   ├── data_paths.yaml          # 数据路径配置
│   ├── api_config.yaml          # API 配置
│   └── routing_table.yaml       # 路由策略配置
├── data_loaders/
│   ├── __init__.py
│   ├── audio_loader.py          # Audio-HDF5 读取
│   ├── video_loader.py          # Video 帧提取
│   ├── gaze_loader.py           # Gaze 数据解析
│   ├── slam_loader.py           # SLAM 轨迹解析
│   ├── digital_twin_loader.py   # Digital Twin 解析
│   ├── hands_loader.py          # Hands-Masks 读取
│   └── vrs_loader.py            # VRS 原始数据解析
├── perception/
│   ├── __init__.py
│   ├── audio_analyzer.py
│   ├── visual_analyzer.py
│   ├── gaze_tracker.py
│   ├── spatial_reasoner.py
│   ├── hand_interactor.py
│   ├── nutrition_estimator.py
│   └── motion_tracker.py
├── knowledge/
│   ├── __init__.py
│   ├── recipe_kb.py
│   ├── nutrition_kb.py
│   ├── scene_graph_kb.py
│   └── commonsense_kb.py
├── reasoning/
│   ├── __init__.py
│   ├── temporal_reasoner.py
│   ├── spatial_reasoner.py
│   ├── causal_reasoner.py
│   └── procedural_reasoner.py
├── agent/
│   ├── __init__.py
│   ├── router.py                # 问题路由器
│   ├── aggregator.py            # 证据聚合引擎
│   ├── judge.py                 # 自适应深度控制
│   ├── generator.py             # 答案生成器
│   ├── tools.py                 # 工具定义
│   └── main_loop.py             # Agent 主循环
├── evaluation/
│   ├── __init__.py
│   ├── benchmark_loader.py      # HD-EPIC VQA 加载
│   ├── metrics.py               # 评估指标
│   ├── run_eval.py              # 评估脚本
│   └── ablation.py              # 消融实验
└── utils/
    ├── __init__.py
    ├── time_align.py             # 时间对齐工具
    ├── cache.py                  # API 调用缓存
    └── visualization.py          # 可视化工具
```
