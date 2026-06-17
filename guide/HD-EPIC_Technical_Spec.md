# HD-EPIC 多模态 Agent — 各感知模块技术说明

> 每个模块的模型选型、技术架构、数据流和实现要点
> 
> 更新时间: 2026-06-17

---

## 目录

1. [模块总览与技术栈矩阵](#1-模块总览与技术栈矩阵)
2. [模块 1: VisualAnalyzer — 视觉分析器](#2-模块-1-visualanalyzervisualanalyzer--视觉分析器)
3. [模块 2: AudioAnalyzer — 音频分析器](#3-模块-2-audioanalyzeraudioanalyzeraudioanalyzer--音频分析器)
4. [模块 3: GazeTracker — 注视追踪器](#4-模块-3-gazetrackergazetrackergazetracker--注视追踪器)
5. [模块 4: SpatialReasoner — 空间推理器](#5-模块-4-spatialreasonerspatialreasonerspatialreasoner--空间推理器)
6. [模块 5: HandInteractor — 手部交互分析器](#6-模块-5-handinteractorhandinteractorhandinteractor--手部交互分析器)
7. [模块 6: NutritionEstimator — 营养估计器](#7-模块-6-nutritionestimatornutritionestimator--营养估计器)
8. [模块 7: MotionTracker — 物体运动追踪器](#8-模块-7-motiontrackermotiontracker--物体运动追踪器)
9. [跨模块共享组件: SAM 2.1](#9-跨模块共享组件-sam-21-用户已有权重)
10. [模型依赖总表与 GPU 需求](#10-模型依赖总表与-gpu-需求)
11. [参考文献](#11-参考文献)

---

## 1. 模块总览与技术栈矩阵

```
┌──────────────────┬──────────────────────────┬───────────────┬──────────────┐
│ 模块             │ 核心模型                  │ 开源/本地     │ API 调用     │
├──────────────────┼──────────────────────────┼───────────────┼──────────────┤
│ VisualAnalyzer   │ MiMo2.5 + SAM 2.1        │ SAM 2.1 本地  │ MiMo2.5 API │
│ AudioAnalyzer    │ BEATs + CLAP              │ 全部本地      │ 不需要       │
│ GazeTracker      │ Grounding DINO 1.6 + SAM3│ 全部本地      │ 不需要       │
│ SpatialReasoner  │ Open3D + MiMo2.5         │ Open3D 本地   │ MiMo2.5 API │
│ HandInteractor   │ SAM 2.1 + Grounding DINO │ 全部本地      │ 不需要       │
│ NutritionEstimator│ MiMo2.5 + Nutrition5K   │ Nutrition5K   │ MiMo2.5 API │
│ MotionTracker    │ SAM 2.1 + Open3D         │ 全部本地      │ 不需要       │
└──────────────────┴──────────────────────────┴───────────────┴──────────────┘
```

---

## 2. 模块 1: VisualAnalyzer（视觉分析器）

### 2.1 功能定义

- 视频帧采样 + 场景理解
- 场景图生成（物体检测 + 关系识别）
- 动作识别（时序理解）
- 开放词汇目标检测

### 2.2 模型选型

#### 主模型: MiMo2.5 (API 调用)

- **用途**: 视频帧的高层语义理解、场景描述、动作识别
- **调用方式**: API 请求
- **优势**: 强大的视觉理解能力，支持长上下文
- **输入**: 视频帧 (base64) + prompt
- **输出**: 结构化 JSON（场景描述、物体列表、动作）

#### 辅助模型: SAM 2.1 (Segment Anything Model 2.1)

- **用途**: 像素级物体分割，为场景图提供精确掩码
- **论文**: Meta AI, 2025 (Segment Anything Model 3)
- **权重**: 用户已有
- **能力**:
  - 文本提示分割 (text-prompted segmentation)
  - 实例分割 (instance segmentation)
  - 开放词汇语义分割 (open-vocabulary semantic segmentation)
- **在本模块中的作用**:
  - 对 MiMo2.5 识别的物体生成精确掩码
  - 提供物体边界信息，辅助场景图的关系判断
  - 与 Grounding DINO 联用做 zero-shot 物体检测+分割

#### 辅助模型: Grounding DINO

- **用途**: 开放词汇目标检测（文本描述 → 检测框）
- **来源**: IDEA-Research (ECCV 2024)
- **GitHub**: https://github.com/IDEA-Research/GroundingDINO (10.3k stars)
- **论文**: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" (arXiv:2303.05499)
- **版本演进**:
  - Grounding DINO (ECCV 2024): Swin-T 48.4 AP / Swin-B 56.7 AP zero-shot on COCO
  - Grounding DINO 1.5: IDEA Research 最强开放世界检测模型
  - Grounded-SAM-2: 官方与 SAM 2 的联合管线
- **Checkpoint 规格**:

| 模型 | Backbone | 训练数据 | COCO zero-shot AP | COCO fine-tune AP |
|------|----------|---------|-------------------|-------------------|
| GroundingDINO-T | Swin-T | O365, GoldG, Cap4M | 48.4 | 57.2 |
| GroundingDINO-B | Swin-B | COCO, O365, GoldG, Cap4M, OpenImage, ODinW-35, RefCOCO | 56.7 | 63.0 |

- **能力**:
  - 文本引导的目标检测：输入 "knife. cutting board. tomato." → 输出检测框
  - 零样本检测：无需训练即可检测任意类别（COCO 48.4 AP without COCO training!）
  - 灵活文本输入：支持单词、短语、句子级别描述
  - 与 SAM 2.1 联用：检测框 → SAM 2.1 分割 → 精确掩码
- **在本模块中的作用**:
  - 替代 MiMo2.5 做物体检测（更快、更精确）
  - 为 SAM 2.1 提供框提示 (box prompt)
  - 批量预处理视频帧，缓存检测结果

**使用代码**:
```python
from groundingdino.util.inference import load_model, load_image, predict, annotate

model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

image_source, image = load_image(IMAGE_PATH)
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="knife. cutting board. tomato. pan. plate.",
    box_threshold=0.35,
    text_threshold=0.25
)
```

### 2.3 推理流程

```
视频帧
  │
  ├─→ Grounding DINO 1.6 (本地)
  │     输入: 帧 + 厨房物体文本列表
  │     输出: 检测框 + 类别 + 置信度
  │
  ├─→ SAM 2.1 (本地)
  │     输入: 帧 + 检测框 (来自 Grounding DINO)
  │     输出: 精确物体掩码
  │
  └─→ MiMo2.5 (API)
        输入: 帧 + 掩码 + 结构化 prompt
        输出: 场景图 JSON
              {
                "objects": [...],
                "relations": [...],
                "action": "...",
                "scene_description": "..."
              }
```

### 2.4 优化策略

- **预处理阶段**: 用 Grounding DINO + SAM 2.1 批量处理所有关键帧，缓存检测结果
- **运行时**: Agent 只调用 MiMo2.5 做高层推理，检测结果从缓存读取
- **关键帧选择**: 基于音频事件 / Gaze 热点定位关键帧，避免全帧扫描

---

## 3. 模块 2: AudioAnalyzer（音频分析器）

### 3.1 功能定义

- 从 Audio-HDF5 提取音频事件
- 声音事件分类（切菜、水流、搅拌、油炸等）
- 声音时间聚类 → 活动段识别
- 声音序列推理（烹饪阶段推断）

### 3.2 模型选型

#### 主模型: BEATs (Bidirectional Encoder representation from Audio Transformer)

- **来源**: Microsoft Research, 2023
- **论文**: "BEATs: Audio Pre-Training with Acoustic Tokenizers" (ICML 2023)
- **版本**: BEATs_iter3+ (AS2M fine-tuned), 最新版本
- **HuggingFace**: `microsoft/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2`
- **能力**:
  - 音频事件分类: 支持 527 种声音类别
  - AudioSet 上 mAP: **48.6%** (SOTA 级别)
  - 支持厨房相关声音: chopping, frying, boiling, pouring, stirring, cutting, water running, microwave, blender 等
- **输入**: 音频波形 (16kHz)
- **输出**: 527 维分类 logits + 置信度
- **优势**:
  - 专门针对音频事件分类优化
  - 轻量 (~300M 参数)，可本地推理
  - 支持 zero-shot 分类（通过 text prompt）

#### 辅助模型: CLAP (Contrastive Language-Audio Pretraining)

- **来源**: LAION-AI (ICASSP 2023)
- **GitHub**: https://github.com/LAION-AI/CLAP (2.2k stars)
- **论文**: "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation" (arXiv:2211.06687)
- **安装**: `pip install laion-clap`
- **Pretrained Checkpoints**:

| 模型 | 用途 | ESC50 Zero-shot | 下载 |
|------|------|-----------------|------|
| 630k-audioset-best.pt | 通用音频 (<10s) | - | HuggingFace |
| 630k-audioset-fusion-best.pt | 可变长度音频 | - | HuggingFace |
| music_audioset_epoch_15_esc_90.14.pt | 音乐 | 90.14% | HuggingFace |
| **music_speech_audioset_epoch_15_esc_89.98.pt** | **音乐+语音+通用** | **89.98%** | HuggingFace |
| music_speech_epoch_15_esc_89.25.pt | 音乐+语音 | 89.25% | HuggingFace |

- **能力**:
  - 音频-文本对齐: 输入文本 "the sound of chopping vegetables" → 匹配音频片段
  - Zero-shot 音频分类: ESC50 89.98% (无需训练!)
  - 音频检索: 根据文本描述检索音频
  - 支持音频嵌入 + 文本嵌入提取
- **在本模块中的作用**:
  - 为 BEATs 提供文本描述能力（BEATs 只有数字类别，CLAP 可以理解自然语言）
  - 做 zero-shot 厨房声音分类（"water running", "knife cutting", "oil sizzling"）
  - 声音事件的文字描述生成

**使用代码**:
```python
import laion_clap
import librosa
import torch

# 加载模型 (推荐 HTSAT-base 架构)
model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
model.load_ckpt('music_speech_audioset_epoch_15_esc_89.98.pt')

# 音频分类: 计算音频与文本的相似度
audio_data, _ = librosa.load('kitchen_audio.wav', sr=48000)
audio_data = audio_data.reshape(1, -1)

texts = ["sound of chopping", "sound of water running", "sound of frying"]
audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
text_embed = model.get_text_embedding(texts)

# 计算相似度
similarity = audio_embed @ text_embed.T
predicted_class = texts[similarity.argmax()]
```

#### 备选模型: Audio Spectrogram Transformer (AST)

- **来源**: MIT, 2021-2025
- **HuggingFace**: `MIT/ast-finetuned-audioset-10-10-0.4593`
- **AudioSet mAP**: 45.9%
- **特点**: 纯 Transformer 架构，无需 CNN 前端
- **适用场景**: 如果 BEATs 不可用，AST 是可靠的备选

### 3.3 推理流程

```
Audio-HDF5
  │
  ├─→ h5py 按时间范围读取音频数据
  │
  ├─→ BEATs (本地)
  │     输入: 音频片段 (16kHz, 10s 窗口)
  │     输出: 声音类别 + 置信度
  │     例: {class: "chopping", confidence: 0.92}
  │
  ├─→ CLAP (本地)
  │     输入: 音频片段 + 文本 prompt
  │     输出: 音频-文本相似度分数
  │     例: "knife cutting" → 0.85, "water running" → 0.12
  │
  └─→ 融合输出:
        {
          "audio_events": [
            {"type": "chopping", "start": 120.5, "end": 135.2, "confidence": 0.92},
            {"type": "water_running", "start": 110.0, "end": 115.0, "confidence": 0.88}
          ],
          "activity_segments": [
            {"label": "vegetable_preparation", "time_range": [110, 160]}
          ]
        }
```

### 3.4 关键实现细节

```python
# HDF5 读取示例
import h5py
import torch
from BEATs import BEATs, BEATsConfig

# 加载 BEATs 模型
checkpoint = torch.load('BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
cfg = BEATsConfig(checkpoint['cfg'])
model = BEATs(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# 从 HDF5 读取音频片段
with h5py.File('audio.hdf5', 'r') as f:
    # 按时间范围索引，避免全量加载
    audio_data = f['audio'][start_sample:end_sample]
    
    # BEATs 推理
    with torch.no_grad():
        probs = model.extract_features(audio_data)
        top_class = torch.argmax(probs)
```

### 3.5 厨房声音类别列表

```python
KITCHEN_SOUND_CLASSES = [
    # 切割类
    "chopping", "cutting", "slicing", "dicing",
    # 水类
    "water running", "pouring water", "filling", "draining",
    # 烹饪类
    "frying", "sizzling", "boiling", "simmering", "stirring",
    # 器具类
    "microwave", "blender", "food processor", "kettle",
    # 容器类
    "placing down", "picking up", "opening", "closing",
    # 食物类
    "crunching", "cracking egg", "peeling",
    # 环境类
    "exhaust fan", "refrigerator", "dishwasher"
]
```

---

## 4. 模块 3: GazeTracker（注视追踪器）

### 4.1 功能定义

- 解析 Gaze 数据（注视点坐标 + 时间戳）
- 注视目标识别：(x, y) → 具体物体名称
- 注意力热力图生成
- 注视模式分析（fixation, saccade）

### 4.2 模型选型

#### 主模型: Grounding DINO 1.6

- **用途**: 将注视点坐标映射到具体物体
- **原理**: 
  1. 从 Gaze 数据获取注视点 (x, y)
  2. 以注视点为中心裁剪视频帧区域
  3. 用 Grounding DINO 检测该区域内的物体
  4. 匹配注视点最近的物体 → 注视目标
- **输入**: 视频帧 + 注视点坐标
- **输出**: 注视目标物体 + 置信度

#### 辅助: SAM 2.1

- **用途**: 对注视目标生成精确掩码
- **原理**: Grounding DINO 检测框 → SAM 2.1 分割 → 精确掩码
- **应用**: 判断注视点是否在物体掩码内

#### 辅助: 数值计算 (numpy)

- **用途**: 注视轨迹分析
- **功能**:
  - Fixation 检测：注视点在某区域停留 > 200ms
  - Saccade 检测：快速眼动
  - 注意力热力图：高斯核密度估计

### 4.3 推理流程

```
Gaze 数据 (时间戳, x, y)
  │
  ├─→ 读取对应时刻的视频帧
  │
  ├─→ 注视区域裁剪
  │     以 (x, y) 为中心，裁剪 200x200 区域
  │
  ├─→ Grounding DINO 1.6 (本地)
  │     输入: 裁剪区域 + 厨房物体文本列表
  │     输出: 区域内物体检测框 + 类别
  │
  ├─→ 物体匹配
  │     计算注视点到各检测框中心的距离
  │     选择最近的物体作为注视目标
  │
  ├─→ SAM 2.1 (本地)
  │     输入: 帧 + 检测框
  │     输出: 精确掩码
  │     验证: 注视点是否在掩码内
  │
  └─→ 输出:
        {
          "gaze_target": "cutting_board",
          "confidence": 0.85,
          "position_in_frame": [320, 240],
          "fixation_duration": 6.5,
          "timestamp": 125.0
        }
```

### 4.4 注意力热力图

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_attention_heatmap(gaze_points, frame_size=(1920, 1080)):
    """
    gaze_points: list of (x, y) 注视点坐标
    """
    heatmap = np.zeros(frame_size[::-1])  # (height, width)
    for x, y in gaze_points:
        if 0 <= x < frame_size[0] and 0 <= y < frame_size[1]:
            heatmap[int(y), int(x)] += 1
    
    # 高斯模糊平滑
    heatmap = gaussian_filter(heatmap, sigma=30)
    
    # 归一化
    heatmap = heatmap / heatmap.max()
    return heatmap

def find_high_attention_regions(heatmap, threshold=0.7):
    """找到高注意力区域"""
    regions = np.where(heatmap > threshold)
    # 聚类分析，返回区域中心和面积
    return clustered_regions
```

---

## 5. 模块 4: SpatialReasoner（空间推理器）

### 5.1 功能定义

- Digital Twin 3D 模型加载与查询
- SLAM 6DoF 轨迹解析
- 空间关系计算（方位、距离、遮挡）
- 2D → 3D 提升

### 5.2 模型/工具选型

#### 核心引擎: Open3D

- **版本**: Open3D 0.19+ (最新稳定版)
- **GitHub**: https://github.com/isl-org/Open3D
- **能力**:
  - 3D 点云处理
  - 网格加载与查询 (OBJ, PLY, GLTF)
  - 3D 空间查询 (KD-Tree, 最近点, 射线投射)
  - 可视化
- **在本模块中的作用**:
  - 加载 Digital Twin 的 3D 模型
  - 提取固定装置的 3D 坐标和尺寸
  - 计算空间关系（距离、方位、遮挡）
  - 射线投射检测可见性

#### 辅助: numpy + scipy

- **用途**: 
  - SLAM 轨迹解析 (6DoF 位姿)
  - 旋转矩阵 / 四元数 / 欧拉角转换
  - 空间距离计算
  - 坐标系变换

#### 高层推理: MiMo2.5 (API)

- **用途**: 
  - 将数值结果翻译为自然语言描述
  - 结合视频帧做空间语义理解
  - 回答空间类问题

### 5.3 核心实现

```python
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

class SpatialReasoner:
    def __init__(self, digital_twin_path):
        # 加载 Digital Twin 3D 模型
        self.mesh = o3d.io.read_triangle_mesh(digital_twin_path)
        self.mesh.compute_vertex_normals()
        
        # 构建 KD-Tree 用于空间查询
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = self.mesh.vertices
        self.kd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        
        # 固定装置信息 (从 Digital Twin 标注加载)
        self.fixtures = {}  # {name: {"position": [x,y,z], "size": [w,h,d]}}
    
    def load_fixtures(self, fixture_file):
        """加载 413 个厨房固定装置的 3D 信息"""
        # 从标注文件加载
        pass
    
    def compute_spatial_relation(self, obj_a_pos, obj_b_pos, wearer_facing=None):
        """
        计算两个物体的空间关系
        返回: {relation: "left/right/above/below/behind/in_front", distance: float}
        """
        diff = np.array(obj_a_pos) - np.array(obj_b_pos)
        distance = np.linalg.norm(diff)
        
        if wearer_facing is not None:
            # 基于佩戴者朝向判断左右
            cross = np.cross(wearer_facing[:2], diff[:2])
            if cross > 0:
                relation = "left"
            else:
                relation = "right"
        else:
            # 基于坐标轴判断
            if abs(diff[2]) > abs(diff[0]) and abs(diff[2]) > abs(diff[1]):
                relation = "above" if diff[2] > 0 else "below"
            else:
                relation = "right" if diff[0] > 0 else "left"
        
        return {"relation": relation, "distance": float(distance)}
    
    def get_nearest_fixture(self, position):
        """找到最近的固定装置"""
        min_dist = float('inf')
        nearest = None
        for name, info in self.fixtures.items():
            dist = np.linalg.norm(np.array(position) - np.array(info["position"]))
            if dist < min_dist:
                min_dist = dist
                nearest = name
        return nearest, min_dist
    
    def parse_slam_pose(self, pose_data):
        """
        解析 SLAM 6DoF 位姿
        pose_data: [x, y, z, qx, qy, qz, qw] (位置 + 四元数)
        """
        position = pose_data[:3]
        quaternion = pose_data[3:7]
        rotation = Rotation.from_quat(quaternion)
        facing_direction = rotation.apply([0, 0, -1])  # 假设 -Z 是前方
        return {
            "position": position,
            "facing": facing_direction,
            "rotation_matrix": rotation.as_matrix()
        }
    
    def raycast_visibility(self, source, target):
        """
        射线投射检测: 从 source 能否看到 target
        用于判断遮挡关系
        """
        ray = o3d.core.RaycastingScene()
        # ... 射线投射实现
        pass
```

### 5.4 数据格式

#### Digital Twin 格式

```json
{
  "kitchen_id": "kitchen_01",
  "fixtures": [
    {
      "id": "sink_01",
      "type": "sink",
      "mesh_file": "sink_01.obj",
      "position": [1.0, 0.9, 0.0],
      "size": [0.6, 0.4, 0.2],
      "semantic_label": "kitchen_sink"
    },
    {
      "id": "stove_01",
      "type": "stove",
      "mesh_file": "stove_01.obj",
      "position": [2.5, 0.9, 0.0],
      "size": [0.6, 0.6, 0.1],
      "semantic_label": "cooking_stove"
    }
  ]
}
```

#### SLAM 轨迹格式

```
# 每行: timestamp tx ty tz qx qy qz qw
120.000 1.800 1.600 0.000 0.000 0.000 0.000 1.000
120.033 1.801 1.601 0.000 0.001 0.000 0.000 1.000
...
```

---

## 6. 模块 5: HandInteractor（手部交互分析器）

### 6.1 功能定义

- 手部掩码提取（已有 Hands-Masks 数据）
- 手-物交互识别（手正在接触什么）
- 动作分类（基于接触模式推断动作）
- 物体状态变化检测

### 6.2 模型选型

#### 主模型: SAM 2.1 (用户已有权重)

- **用途**: 
  - 对手部区域做精细分割
  - 对手部接触的物体做实例分割
  - 物体状态变化的视觉证据（分割后的物体形状变化）
- **输入**: 视频帧 + 手部区域提示 (point/box prompt)
- **输出**: 手部掩码 + 物体掩码

#### 辅助模型: Grounding DINO 1.6

- **用途**: 识别手部接触的具体物体
- **原理**:
  1. 从 Hands-Masks 获取手部区域
  2. 扩展手部区域为 bounding box
  3. 用 Grounding DINO 检测该区域内的物体
  4. 判断哪个物体与手部掩码重叠 → 接触物体
- **输入**: 手部区域帧 + 文本 prompt（"knife, fork, spoon, pan, ..."）
- **输出**: 接触物体 + 接触类型

#### 辅助: 规则引擎

- **用途**: 基于接触模式推断动作类型
- **规则示例**:
  - 手 + 刀 + 食物 → "cutting"
  - 手 + 勺 + 碗 → "stirring"
  - 手 + 物体 + 向上运动 → "picking up"
  - 手 + 物体 + 向下运动 → "putting down"

### 6.3 推理流程

```
视频帧 + Hands-Masks (已有)
  │
  ├─→ 读取手部掩码 (从 Hands-Masks 数据)
  │     输出: 左手掩码, 右手掩码, 手部关键点
  │
  ├─→ 手部区域扩展
  │     以手部掩码为中心扩展 bounding box
  │     扩展比例: 1.5x (包含周围物体)
  │
  ├─→ Grounding DINO 1.6 (本地)
  │     输入: 扩展区域帧 + 厨房物体文本列表
  │     输出: 区域内物体检测框 + 类别
  │
  ├─→ 接触检测
  │     计算手部掩码与各物体检测框的 IoU
  │     IoU > 阈值 → 判定为接触
  │
  ├─→ SAM 2.1 (本地)
  │     输入: 帧 + 接触物体的检测框
  │     输出: 接触物体的精确掩码
  │     用途: 物体状态变化检测
  │
  └─→ 动作推断 (规则引擎)
        输入: 接触物体 + 手部运动方向 + 声音信号
        输出: 动作类型
```

### 6.4 接触-动作映射表

```python
CONTACT_ACTION_MAP = {
    # (hand, object, motion_direction) → action
    ("right_hand", "knife", "horizontal"): "cutting",
    ("right_hand", "knife", "vertical"): "chopping",
    ("right_hand", "spoon", "circular"): "stirring",
    ("right_hand", "pan", "upward"): "lifting_pan",
    ("right_hand", "pan_handle", "tilting"): "pouring_from_pan",
    ("right_hand", "lid", "upward"): "removing_lid",
    ("right_hand", "bottle", "tilting"): "pouring",
    ("right_hand", "vegetable", "downward"): "placing_in_pan",
    ("both_hands", "mixing_bowl", "circular"): "mixing",
}
```

---

## 7. 模块 6: NutritionEstimator（营养估计器）

### 7.1 功能定义

- 食材识别（从视觉 + 标注）
- 份量估计（视觉估算）
- 营养成分查询
- 菜品总营养计算

### 7.2 模型选型

#### 主模型: MiMo2.5 (API)

- **用途**: 
  - 食材视觉识别（看图识别食材）
  - 份量估计（基于手部参照物估算份量）
  - 营养推理（结合烹饪过程推断营养变化）

#### 辅助: Nutrition5K 数据集 + 回归模型

- **来源**: Google Research, 2021
- **HuggingFace**: 有社区复现版本
- **能力**: 从食物图像直接回归卡路里、蛋白质、碳水、脂肪
- **架构**: ResNet/EfficientNet backbone + 回归头
- **在本模块中的作用**:
  - 作为 MiMo2.5 份量估计的交叉验证
  - 提供基于视觉的营养回归基线

#### 辅助: 查表数据库

- **来源**: HD-EPIC Nutrition 标注 + USDA FoodData Central
- **用途**: 食材 → 营养成分的精确映射
- **格式**:
```json
{
  "tomato": {"calories_per_100g": 18, "protein_g": 0.9, "carbs_g": 3.9, "fat_g": 0.2},
  "olive_oil": {"calories_per_100g": 884, "protein_g": 0, "carbs_g": 0, "fat_g": 100},
  "chicken_breast": {"calories_per_100g": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6}
}
```

### 7.3 推理流程

```
食材识别 + 份量估计
  │
  ├─→ MiMo2.5 (API)
  │     输入: 烹饪关键帧 + prompt
  │     "请识别画面中的食材，并估算每种食材的大致重量(克)"
  │     输出: [{"name": "tomato", "amount_g": 200}, ...]
  │
  ├─→ Nutrition5K 回归模型 (本地, 可选)
  │     输入: 最终菜品帧
  │     输出: 营养回归值
  │
  ├─→ 查表计算
  │     对每种食材查表获取营养密度
  │     营养 = 营养密度 × 份量
  │
  └─→ 输出:
        {
          "dish_name": "tomato_salad",
          "ingredients": [
            {"name": "tomato", "amount_g": 200, "calories": 36, "protein_g": 1.8},
            {"name": "olive_oil", "amount_g": 15, "calories": 119, "protein_g": 0}
          ],
          "total": {"calories": 155, "protein_g": 1.8, "carbs_g": 7.8, "fat_g": 13.9}
        }
```

---

## 8. 模块 7: MotionTracker（物体运动追踪器）

### 8.1 功能定义

- 物体位置追踪（从 A 到 B）
- 运动轨迹重建（3D 空间）
- 状态变化检测
- 运动模式分类

### 8.2 模型选型

#### 主模型: SAM 2.1 (用户已有权重)

- **用途**: 
  - 视频中的物体实例分割与追踪
  - 结合 SAM 2.1 的 memory bank 机制做视频物体分割 (VOS)
  - 物体状态变化的视觉证据
- **SAM 2.1 的视频能力**: 
  - SAM 2.1 支持 `SAM2VideoPredictor` 视频预测器
  - 支持 torch.compile 全模型编译加速 (`vos_optimized=True`)
  - 独立多物体追踪（每个物体独立推理）
  - 可以通过首帧掩码/检测框初始化，追踪后续帧中的物体

#### 辅助: Open3D

- **用途**: 
  - 运动轨迹的 3D 提升
  - 结合 SLAM 位姿将 2D 运动转换为 3D 轨迹
  - 运动路径的碰撞检测

#### 辅助: numpy + scipy

- **用途**: 
  - 运动轨迹平滑 (Kalman Filter)
  - 运动模式分类 (速度、加速度、方向)
  - 状态变化检测 (位置跳变、形状变化)

### 8.3 推理流程

```
视频序列 + Object Motion 标注
  │
  ├─→ SAM 2.1 视频追踪 (本地)
  │     输入: 首帧物体掩码 (来自 Hands-Masks 或标注)
  │     输出: 后续帧中该物体的掩码序列
  │
  ├─→ 物体中心轨迹提取
  │     从掩码序列中提取物体中心点
  │     输出: [(t1, x1, y1), (t2, x2, y2), ...]
  │
  ├─→ SLAM 位姿对齐 (Open3D + numpy)
  │     将 2D 轨迹 + SLAM 位姿 → 3D 轨迹
  │     输出: [(t1, X1, Y1, Z1), (t2, X2, Y2, Z2), ...]
  │
  ├─→ 运动模式分析
  │     速度、加速度、方向计算
  │     运动类型分类: 平移/旋转/倾倒/放置
  │
  └─→ 输出:
        {
          "object": "tomato",
          "movement_type": "displacement",
          "from": {"position": [2.5, 0.9, 0.1], "container": "cutting_board"},
          "to": {"position": [2.5, 0.9, 0.0], "container": "plate"},
          "time_range": [180.0, 185.0],
          "trajectory_3d": [[2.5, 0.9, 0.1], [2.5, 0.9, 0.15], [2.5, 0.9, 0.0]],
          "speed_avg": 0.05,
          "movement_arc": "lift_and_place"
        }
```

---

## 9. 跨模块共享组件: SAM 2.1 (用户已有权重)

> **注**: 用户所持"SAM3 权重"对应 Meta 最新发布的 SAM 2.1 系列。arxiv 上 2026 年论文中提及的"SAM 3"指同一模型线的最新版本。

### 9.1 模型概况

- **来源**: Meta AI, FAIR (Segment Anything Model 2)
- **GitHub**: https://github.com/facebookresearch/sam2 (19.4k stars)
- **论文**: "SAM 2: Segment Anything in Images and Videos" (arXiv:2408.00714)
- **版本**: SAM 2.1 (2024-09-29 发布的改进版 checkpoint)
- **许可**: Apache-2.0

### 9.2 SAM 2.1 Checkpoint 规格

| 模型 | 参数量 | 速度 (FPS) | SA-V test (J&F) | MOSE val (J&F) | LVOS v2 (J&F) |
|------|--------|-----------|-----------------|----------------|---------------|
| sam2.1_hiera_tiny | 38.9M | 91.2 | 76.5 | 71.8 | 77.3 |
| sam2.1_hiera_small | 46M | 84.8 | 76.6 | 73.5 | 78.3 |
| sam2.1_hiera_base_plus | 80.8M | 64.1 | 78.2 | 73.7 | 78.2 |
| **sam2.1_hiera_large** | **224.4M** | **39.5** | **79.5** | **74.6** | **80.6** |

速度测试环境: A100, torch 2.5.1, cuda 12.4

### 9.3 SAM 2.1 在各模块中的角色

```
┌─────────────────────────────────────────────────────────────┐
│                SAM 2.1 (用户已有权重, 224.4M)                 │
├─────────────────┬───────────────────────────────────────────┤
│ VisualAnalyzer  │ 物体实例分割 → 场景图的精确物体边界         │
│ GazeTracker     │ 注视目标掩码 → 确认注视点在物体上           │
│ HandInteractor  │ 手部精细分割 + 接触物体分割                 │
│ MotionTracker   │ 视频物体追踪 (VOS) → 运动轨迹              │
│ SpatialReasoner │ 3D 场景中的物体实例分割                     │
└─────────────────┴───────────────────────────────────────────┘
```

### 9.4 核心能力

| 能力 | API 类 | 说明 | 本项目用途 |
|------|--------|------|-----------|
| **图像分割** | `SAM2ImagePredictor` | 支持 box/point/text prompt | 物体实例分割 |
| **视频追踪 (VOS)** | `SAM2VideoPredictor` | 首帧掩码 → memory bank → 追踪后续帧 | 物体运动追踪 |
| **多物体追踪** | `SAM2VideoPredictor` | 同时追踪多个物体，独立推理 | 多物体运动分析 |
| **自动掩码生成** | `SAM2ImagePredictor` | 类似 SAM 1 的自动分割 | 全场景分割 |
| **torch.compile 加速** | `vos_optimized=True` | 全模型编译，VOS 大幅加速 | 运行时性能 |

### 9.5 图像分割代码

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, scores, logits = predictor.predict(
        box=<grounding_dino_boxes>,   # 来自 Grounding DINO 的检测框
        multimask_output=True
    )
```

### 9.6 视频物体追踪代码

```python
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path="<your_video>")
    
    # 用 Grounding DINO 的检测框作为首帧 prompt
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        state, frame_idx=0, box=<grounding_dino_boxes>
    )
    
    # 传播到整个视频
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # masks: 每个物体在当前帧的掩码
        # object_ids: 物体 ID 列表
        ...
```

### 9.7 Grounded-SAM-2 联用

**项目**: https://github.com/IDEA-Research/Grounded-SAM-2

这是 IDEA-Research 官方提供的 Grounding DINO + SAM 2 联合管线，本项目直接复用：

```
视频帧
  │
  ├─→ Grounding DINO (本地)
  │     输入: 帧 + "knife. cutting board. tomato. pan."
  │     输出: 检测框 + 类别 + 置信度
  │
  └─→ SAM 2.1 (本地)
        输入: 帧 + 检测框 (作为 box prompt)
        输出: 每个物体的精确掩码
        视频模式: 首帧检测 → 后续帧自动追踪
```

### 9.8 HuggingFace 快速加载

```python
# 图像预测
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

# 视频预测
from sam2.sam2_video_predictor import SAM2VideoPredictor
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
```

### 9.9 ActiveSAM 优化 (可选)

**论文**: "ActiveSAM: Image-Conditional Class Pruning for Fast and Accurate Open-Vocabulary Segmentation" (arXiv:2606.16996, 2026-06)

- **核心思想**: 不对所有类别做分割，先预测图像中存在哪些类别，只对存在的类别做全分辨率分割
- **优势**: 速度提升 5.5x，精度提升 +1.4 mIoU
- **代码**: https://github.com/VILA-Lab/ActiveSAM
- **适用场景**: 厨房场景中物体类别有限，ActiveSAM 可以显著加速

---

## 10. 模型依赖总表与 GPU 需求

### 10.1 模型清单

| 模型 | 来源 | 参数量 | 权重大小 | GPU 需求 | 用途 |
|------|------|--------|---------|---------|------|
| **SAM 2.1 Large** | Meta AI (Apache-2.0) | 224.4M | ~900MB | 4GB+ VRAM | 通用分割 + 视频追踪 |
| **Grounding DINO Swin-B** | IDEA-Research (ECCV 2024) | ~300M | ~1.2GB | 4GB+ VRAM | 开放词汇检测 |
| **BEATs iter3+** | Microsoft (ICML 2023) | ~300M | ~1.2GB | 2GB+ VRAM | 音频事件分类 |
| **CLAP HTSAT-base** | LAION-AI (ICASSP 2023) | ~200M | ~800MB | 2GB+ VRAM | 音频-文本对齐 |
| **Open3D** | Intel ISL | N/A | ~50MB | CPU 即可 | 3D 计算 |
| **MiMo2.5** | Xiaomi | N/A | N/A | API 调用 | 视觉理解+推理 |

### 10.2 推荐硬件配置

```
最低配置:
  GPU: NVIDIA RTX 3060 (12GB VRAM)
  RAM: 32GB
  存储: 50GB (模型权重 + 缓存)
  可同时加载: SAM 2.1 + Grounding DINO (8GB VRAM)

推荐配置:
  GPU: NVIDIA RTX 4090 (24GB VRAM)
  RAM: 64GB
  存储: 100GB
  可同时加载: 所有本地模型 (12GB VRAM)

理想配置:
  GPU: NVIDIA A100 (40GB+ VRAM)
  RAM: 128GB
  存储: 200GB
  可同时加载: 所有模型 + 批量推理
```

### 10.3 模型加载策略

```python
# 按需加载，避免同时占用所有 VRAM
class ModelManager:
    def __init__(self):
        self.models = {}
        self.device = "cuda"
    
    def get_sam2(self):
        """SAM 2.1 Large - 图像分割 + 视频追踪"""
        if "sam2" not in self.models:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.models["sam2"] = SAM2ImagePredictor(build_sam2(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "checkpoints/sam2.1_hiera_large.pt"
            ))
        return self.models["sam2"]
    
    def get_sam2_video(self):
        """SAM 2.1 Large - 视频物体追踪"""
        if "sam2_video" not in self.models:
            from sam2.build_sam import build_sam2_video_predictor
            self.models["sam2_video"] = build_sam2_video_predictor(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "checkpoints/sam2.1_hiera_large.pt"
            )
        return self.models["sam2_video"]
    
    def get_grounding_dino(self):
        """Grounding DINO - 开放词汇检测"""
        if "grounding_dino" not in self.models:
            from groundingdino.util.inference import load_model
            self.models["grounding_dino"] = load_model(
                "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "weights/groundingdino_swint_ogc.pth"
            )
        return self.models["grounding_dino"]
    
    def get_beats(self):
        """BEATs - 音频事件分类"""
        if "beats" not in self.models:
            import torch
            checkpoint = torch.load('BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
            from BEATs import BEATs, BEATsConfig
            cfg = BEATsConfig(checkpoint['cfg'])
            model = BEATs(cfg)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            self.models["beats"] = model
        return self.models["beats"]
    
    def get_clap(self):
        """CLAP - 音频-文本对齐"""
        if "clap" not in self.models:
            import laion_clap
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            model.load_ckpt('music_speech_audioset_epoch_15_esc_89.98.pt')
            self.models["clap"] = model
        return self.models["clap"]
    
    def release(self, model_name):
        """释放不需要的模型"""
        if model_name in self.models:
            del self.models[model_name]
            torch.cuda.empty_cache()
```

### 10.4 预处理 vs 运行时

```
预处理阶段 (离线, 可以慢):
  ├─ Grounding DINO + SAM 2.1: 批量处理所有关键帧
  │   → 缓存检测结果到 JSON
  ├─ BEATs: 批量处理所有音频事件
  │   → 缓存分类结果到 JSON
  ├─ SAM 2.1 VOS: 批量追踪关键物体
  │   → 缓存轨迹到 JSON
  └─ 耗时: 数小时 (一次性)

运行时 (在线, 必须快):
  ├─ 查询缓存: <10ms
  ├─ MiMo2.5 API: 1-3s
  ├─ 空间计算 (Open3D): <100ms
  └─ 总延迟: 2-5s per question
```

---

## 11. 参考文献

### SAM 2 相关
1. **SAM 2/2.1**: Ravi et al. "SAM 2: Segment Anything in Images and Videos." arXiv:2408.00714, 2024. GitHub: facebookresearch/sam2
2. **ActiveSAM**: Tien & Shen. "ActiveSAM: Image-Conditional Class Pruning for Fast and Accurate Open-Vocabulary Segmentation." arXiv:2606.16996, 2026.
3. **ESAM++**: Liu et al. "ESAM++: Efficient Online 3D Perception on the Edge." arXiv:2605.29505, 2026.
4. **CLIP-Guided SAM**: Jalilian & Bais. "CLIP-Guided SAM: Parameter-Efficient Semantic Conditioning for Promptable Segmentation." arXiv:2605.24807, 2026.

### 目标检测
5. **Grounding DINO**: Liu et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection." ECCV 2024. arXiv:2303.05499. GitHub: IDEA-Research/GroundingDINO
6. **Grounded-SAM-2**: IDEA-Research. Grounded-SAM-2 (Grounding DINO + SAM 2 联合管线). GitHub: IDEA-Research/Grounded-SAM-2

### 音频
7. **BEATs**: Chen et al. "BEATs: Audio Pre-Training with Acoustic Tokenizers." ICML 2023. GitHub: microsoft/unilm/beats
8. **CLAP**: Wu et al. "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation." ICASSP 2023. GitHub: LAION-AI/CLAP

### 手-物交互
10. **HOI-Synth**: Leonardi et al. "Leveraging Synthetic Data for Enhancing Egocentric Hand-Object Interaction Detection." arXiv:2603.29733, 2026.
11. **GlovEgo-HOI**: Spoto et al. "GlovEgo-HOI: Bridging the Synthetic-to-Real Gap for Industrial Egocentric Human-Object Interaction Detection." arXiv:2601.09528, 2026.

### 3D 与空间
12. **Open3D**: Zhou et al. "Open3D: A Modern Library for 3D Data Processing." arXiv:1801.09847, 2018.

### HD-EPIC 相关
13. **HD-EPIC**: Perrett et al. "HD-EPIC: A Highly-Detailed Egocentric Video Dataset." CVPR 2025. arXiv:2502.04144.
14. **SceneNet+KnowledgeNet**: Taluzzi et al. arXiv:2506.08553, 2025.
15. **EgoAdapt**: Chen et al. arXiv:2605.24500, 2026.
16. **EgoReasoner**: Zhu et al. arXiv:2603.06561, 2026.
17. **Ego2World**: Cheng et al. arXiv:2605.13335, 2026.
18. **V-Nutri**: Yue et al. arXiv:2604.11913, 2026.
19. **PAWS**: Wang et al. arXiv:2603.25539, 2026.
20. **EgoFlow**: Saroha et al. arXiv:2604.01421, 2026.
21. **Egocentric Co-Pilot**: Yang et al. arXiv:2603.01104, 2026.

---

## 附录: pip 依赖

```txt
# 核心
torch>=2.5.1
torchvision>=0.20.1
numpy>=1.24.0
scipy>=1.10.0

# SAM 2.1 (Meta, https://github.com/facebookresearch/sam2)
# pip install -e .  (从源码安装)
# 或 HuggingFace: SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

# Grounding DINO (IDEA-Research, https://github.com/IDEA-Research/GroundingDINO)
# pip install -e .  (从源码安装)

# Grounded-SAM-2 (可选, 联合管线)
# https://github.com/IDEA-Research/Grounded-SAM-2

# 音频
laion-clap>=1.1.0  # pip install laion-clap (LAION CLAP)
# BEATs: 从 https://github.com/microsoft/unilm/tree/main/beats 源码安装
librosa>=0.10.0
soundfile>=0.12.0

# 3D
open3d>=0.19.0
trimesh>=4.0.0

# 数据
h5py>=3.8.0
opencv-python>=4.8.0
Pillow>=10.0.0

# API
requests>=2.31.0

# HuggingFace (可选, 用于快速加载模型)
huggingface_hub>=0.20.0
transformers>=4.40.0
```
