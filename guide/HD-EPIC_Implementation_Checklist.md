# HD-EPIC 多模态 Agent — 实施清单

> 纯任务清单，供 AI 执行时逐项打勾
> 每个模块完成后必须通过最小验证，确认模块可正常运行且适配 Agent
> 
> 更新时间: 2026-06-17

---

# Part A: 预检清单

## A1. 硬件环境

- [x] GPU 确认: A800 80GB VRAM
- [x] CUDA 版本确认 (nvcc --version)
- [x] 系统内存 >= 32GB
- [x] 磁盘可用空间 >= 200GB

## A2. 软件环境

- [x] Python >= 3.10
- [x] PyTorch >= 2.5.1 且 CUDA 可用
- [x] torchvision >= 0.20.1
- [x] CUDA toolkit 与 PyTorch 版本匹配

## A3. 数据访问

- [x] HD-EPIC 数据集路径确认
- [x] Audio-HDF5 可用 h5py 正常读取
- [x] Videos (mp4) 可用 OpenCV 正常解码
- [x] SLAM-and-Gaze 文件可正常打开
- [x] Digital-Twin 3D 模型可正常加载
- [x] Hands-Masks 掩码文件可正常读取
- [ ] VRS 文件可用 vrs 库正常解析

## A4. 模型权重

- [x] SAM 2.1 权重文件确认 (文件名 + 大小)
- [x] Grounding DINO Swin-T 权重下载 (groundingdino_swint_ogc.pth)
- [x] BEATs 权重下载 (BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt)
- [x] CLAP 权重下载 (music_speech_audioset_epoch_15_esc_89.98.pt)
- [x] 所有权重文件完整性校验

## A5. API 配置

- [x] MiMo2.5 API endpoint 确认
- [x] API Key 确认有效且有配额
- [x] QPS 限制确认
- [x] 单次请求 token 上限确认
- [x] API 连通性测试通过

## A6. 评估基准

- [ ] HD-EPIC VQA Benchmark (26K 问题) 获取
- [ ] 官方评估脚本获取或复现
- [ ] 用现有系统跑一遍 Baseline 准确率

---

# Part B: 数据探查

## B1. Audio-HDF5 探查

- [x] 确认文件数量和命名规则
- [x] 确认 HDF5 顶层 group 结构
- [x] 确认 dataset 字段名、dtype、shape
- [x] 确认时间戳单位 (秒/毫秒/帧号)
- [x] 确认音频采样率
- [ ] 确认是否有事件标注 (类型、起止时间)
- [ ] 输出数据结构摘要文档

## B2. Digital-Twin 探查

- [x] 确认目录结构 (按厨房/按物体)
- [x] 确认 3D 模型格式 (OBJ/PLY/GLTF)
- [ ] 确认是否有语义标注文件 (JSON/YAML/CSV)
- [x] 确认固定装置数量和类型
- [ ] 确认 3D 坐标系和单位
- [ ] 确认是否有材质/纹理文件
- [ ] 输出数据结构摘要文档

## B3. SLAM-and-Gaze 探查

- [x] 确认目录结构 (SLAM 和 Gaze 分开还是合并)
- [x] 确认 SLAM 数据格式 (TUM/KITTI/自定义)
- [x] 确认 Gaze 数据格式 (CSV/JSON/二进制)
- [x] 确认时间戳格式 (Unix/相对时间/帧号)
- [x] 确认 SLAM 位姿格式 (四元数/旋转矩阵)
- [x] 确认 Gaze 坐标格式 (像素/归一化/3D)
- [ ] 确认数据粒度和视频对应关系
- [ ] 输出数据结构摘要文档

## B4. Hands-Masks 探查

- [x] 确认掩码格式 (PNG/NPY/HDF5)
- [x] 确认掩码类型 (二值/多类别/实例)
- [x] 确认掩码分辨率
- [x] 确认文件命名规则和帧对应关系
- [x] 确认左右手是否有区分
- [ ] 确认是否包含物体掩码
- [ ] 确认是否有 3D 关节位置数据
- [ ] 输出数据结构摘要文档

## B5. VRS 探查

- [ ] 确认 VRS 文件数量和大小
- [ ] 确认 vrs Python 库是否可用
- [ ] 确认 VRS 包含的流类型 (RGB/eye-tracking/IMU)
- [ ] 确认每个流的时间戳范围和频率
- [ ] 确认与 mp4 视频的对应关系
- [ ] 判断是否需要 VRS (还是 mp4 已足够)
- [ ] 输出数据结构摘要文档

## B6. Videos (mp4) 探查

- [x] 确认视频文件数量
- [ ] 确认编码格式 (H.264/H.265/VP9)
- [x] 确认分辨率
- [x] 确认帧率
- [ ] 确认每个视频时长
- [x] 确认文件命名规则和标注对齐关系
- [ ] 确认总时长是否与论文一致 (41 小时)
- [ ] 输出数据结构摘要文档

---

# Part C: 环境搭建

## C1. 基础环境

- [x] 创建 conda 环境 (python=3.10)
- [x] 安装 PyTorch + torchvision + torchaudio (匹配 CUDA 版本)
- [x] 验证 PyTorch CUDA 可用

## C2. SAM 2.1

- [x] 克隆 facebookresearch/sam2 仓库
- [x] 从源码安装 (pip install -e .)
- [ ] 下载 checkpoint (sam2.1_hiera_large.pt)
- [x] 验证 SAM 2.1 可正常导入和推理

## C3. Grounding DINO

- [x] 克隆 IDEA-Research/GroundingDINO 仓库
- [x] 从源码安装 (pip install -e .)
- [ ] 下载权重 (groundingdino_swint_ogc.pth)
- [x] 验证 Grounding DINO 可正常导入和推理

## C4. CLAP

- [x] pip install laion-clap
- [ ] 下载 checkpoint (music_speech_audioset_epoch_15_esc_89.98.pt)
- [x] 验证 CLAP 可正常导入和推理

## C5. BEATs

- [x] 从 microsoft/unilm 仓库获取 BEATs 代码
- [x] 安装 BEATs 依赖
- [ ] 下载权重 (BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt)
- [ ] 验证 BEATs 可正常加载 checkpoint

## C6. 其他依赖

- [x] 安装 Open3D + trimesh (3D 处理)
- [x] 安装 h5py + opencv-python + Pillow + librosa + soundfile + scipy
- [x] 安装 matplotlib (可视化, 可选)
- [x] 安装 requests (API 调用)
- [x] 安装 huggingface_hub (可选, 快速加载模型)

## C7. 环境验证

- [x] 运行一键验证脚本，所有模块导入成功
- [x] GPU 信息输出正确
- [x] 所有模型可正常导入

---

# Part D: 代码实施

## Phase 1: 项目骨架与数据层

### D1.1 项目初始化

- [x] 创建完整项目目录结构
- [x] 创建 requirements.txt
- [x] 创建 config/data_paths.yaml (所有数据路径)
- [x] 创建 config/api_config.yaml (API 配置)
- [x] 创建 config/routing_table.yaml (路由策略)
- [x] 创建 config/hardware.yaml (硬件配置: A800 80GB, 所有模型常驻)
- [x] 创建所有 __init__.py
- [x] 创建 README.md
- [ ] **验证**: 项目可被 Python 正确识别为包 (import 无报错)

### D1.2 AudioLoader

- [ ] 实现 AudioLoader.__init__(hdf5_dir)
- [ ] 实现 AudioLoader.load_segment(start_time, end_time)
- [ ] 实现 AudioLoader.get_all_events()
- [ ] 实现 AudioLoader.get_events_in_range(start_time, end_time)
- [ ] **验证**: 对真实 HDF5 文件调用 get_all_events()，返回非空列表，每条事件包含 type/start_time/end_time 字段

### D1.3 VideoLoader

- [ ] 实现 VideoLoader.__init__(video_dir)
- [ ] 实现 VideoLoader.get_frame(timestamp)
- [ ] 实现 VideoLoader.get_frames(start_time, end_time, fps)
- [ ] 实现 VideoLoader.get_video_info(video_path)
- [ ] **验证**: 对真实 mp4 调用 get_frame()，返回 numpy 数组 shape 为 (H, W, 3)，dtype 为 uint8

### D1.4 GazeLoader

- [ ] 实现 GazeLoader.__init__(gaze_dir)
- [ ] 实现 GazeLoader.get_gaze_at_time(timestamp)
- [ ] 实现 GazeLoader.get_gaze_trajectory(start_time, end_time)
- [ ] 实现 GazeLoader.get_fixations(min_duration)
- [ ] **验证**: 对真实数据调用 get_gaze_at_time()，返回包含 x, y 坐标的对象

### D1.5 SLAMLoader

- [ ] 实现 SLAMLoader.__init__(slam_dir)
- [ ] 实现 SLAMLoader.get_pose(timestamp)
- [ ] 实现 SLAMLoader.get_trajectory(start_time, end_time)
- [ ] 实现 SLAMLoader.get_position(timestamp)
- [ ] 实现 SLAMLoader.get_facing_direction(timestamp)
- [ ] **验证**: 对真实数据调用 get_pose()，返回包含 position (3D) 和 rotation 的对象

### D1.6 DigitalTwinLoader

- [ ] 实现 DigitalTwinLoader.__init__(dt_dir)
- [ ] 实现 DigitalTwinLoader.load_mesh(mesh_path)
- [ ] 实现 DigitalTwinLoader.get_fixtures()
- [ ] 实现 DigitalTwinLoader.get_fixture_position(fixture_id)
- [ ] 实现 DigitalTwinLoader.get_fixture_by_type(fixture_type)
- [ ] 实现 DigitalTwinLoader.get_spatial_relation(fixture_a, fixture_b)
- [ ] **验证**: 对真实数据调用 get_fixtures()，返回非空列表，每个 fixture 包含 id/type/position 字段

### D1.7 HandsLoader

- [ ] 实现 HandsLoader.__init__(hands_dir)
- [ ] 实现 HandsLoader.get_mask(timestamp)
- [ ] 实现 HandsLoader.get_masks_in_range(start_time, end_time)
- [ ] 实现 HandsLoader.has_hand_contact(mask)
- [ ] **验证**: 对真实数据调用 get_mask()，返回 numpy 数组，unique 值包含 0 和至少一个非零值

### D1.8 VRSLoader (可选)

- [ ] 实现 VRSLoader.__init__(vrs_path)
- [ ] 实现 VRSLoader.get_streams()
- [ ] 实现 VRSLoader.get_rgb_frame(timestamp)
- [ ] 实现 VRSLoader.get_imu_data(timestamp)
- [ ] **验证**: 对真实 VRS 文件调用 get_streams()，返回流列表

### D1.9 工具函数

- [ ] 实现 utils/time_align.py: align_timestamps, find_nearest_timestamp
- [ ] 实现 utils/cache.py: CacheManager (get/put/invalidate)
- [ ] **验证**: 调用 find_nearest_timestamp(1.5, [1.0, 2.0, 3.0]) 返回 1.0；CacheManager put 后 get 返回相同值

---

## Phase 2: 感知模块

### D2.1 AudioAnalyzer

- [ ] 实现 AudioAnalyzer.__init__(beats_model_path, clap_model_path)
- [ ] 实现 AudioAnalyzer.classify_audio(audio_data) — BEATs 分类
- [ ] 实现 AudioAnalyzer.zero_shot_classify(audio_data, text_labels) — CLAP zero-shot
- [ ] 实现 AudioAnalyzer.get_audio_events(start_time, end_time)
- [ ] 实现 AudioAnalyzer.cluster_events(events) — 事件聚类为活动段
- [ ] 输出格式符合统一 Evidence JSON Schema
- [ ] **验证**: 从真实 HDF5 取一段音频，调用 classify_audio()，返回包含 type 和 confidence 的字典；调用 get_audio_events() 返回 Evidence 对象列表

### D2.2 VisualAnalyzer

- [ ] 实现 VisualAnalyzer.__init__(grounding_dino_model, sam2_model)
- [ ] 实现 VisualAnalyzer.detect_objects(frame, text_prompt) — Grounding DINO 检测
- [ ] 实现 VisualAnalyzer.segment_objects(frame, boxes) — SAM 2.1 分割
- [ ] 实现 VisualAnalyzer.generate_scene_graph(frame, mimo_api) — 场景图生成
- [ ] 实现 VisualAnalyzer.analyze_action(frames_sequence, mimo_api) — 动作识别
- [ ] 输出格式符合统一 Evidence JSON Schema
- [ ] **验证**: 对真实视频帧调用 detect_objects("knife. plate.")，返回检测框列表非空；segment_objects 返回掩码 shape 与帧一致

### D2.3 GazeTracker

- [ ] 实现 GazeTracker.__init__(gaze_loader, grounding_dino_model)
- [ ] 实现 GazeTracker.identify_gaze_target(frame, gaze_point)
- [ ] 实现 GazeTracker.generate_attention_heatmap(gaze_points, frame_size)
- [ ] 实现 GazeTracker.get_fixation_targets(start_time, end_time)
- [ ] 输出格式符合统一 Evidence JSON Schema
- [ ] **验证**: 对真实帧+注视点调用 identify_gaze_target()，返回包含 target_name 和 confidence 的字典

### D2.4 SpatialReasoner

- [ ] 实现 SpatialReasoner.__init__(digital_twin_loader, slam_loader)
- [ ] 实现 SpatialReasoner.compute_distance(pos_a, pos_b)
- [ ] 实现 SpatialReasoner.compute_spatial_relation(pos_a, pos_b, facing)
- [ ] 实现 SpatialReasoner.get_nearest_fixture(position)
- [ ] 实现 SpatialReasoner.get_wearer_pose_at_time(timestamp)
- [ ] 实现 SpatialReasoner.check_visibility(source, target)
- [ ] 实现 SpatialReasoner.describe_spatial_layout(mimo_api)
- [ ] 输出格式符合统一 Evidence JSON Schema
- [ ] **验证**: compute_distance([0,0,0], [3,4,0]) 返回 5.0；compute_spatial_relation 返回包含 relation 和 distance 的字典

### D2.5 HandInteractor

- [ ] 实现 HandInteractor.__init__(hands_loader, grounding_dino_model, sam2_model)
- [ ] 实现 HandInteractor.detect_contact_object(frame, hand_mask)
- [ ] 实现 HandInteractor.infer_action(contact_object, hand_motion, audio_hint)
- [ ] 实现 HandInteractor.get_hand_interactions(start_time, end_time)
- [ ] 输出格式符合统一 Evidence JSON Schema
- [ ] **验证**: 对真实帧+掩码调用 detect_contact_object()，返回包含 object_name 和 interaction_type 的字典

### D2.6 NutritionEstimator

- [ ] 实现 NutritionEstimator.__init__(nutrition_db_path)
- [ ] 实现 NutritionEstimator.estimate_ingredients(frame, mimo_api)
- [ ] 实现 NutritionEstimator.estimate_portions(ingredients, frame, mimo_api)
- [ ] 实现 NutritionEstimator.lookup_nutrition(ingredient_name)
- [ ] 实现 NutritionEstimator.calculate_total(ingredients)
- [ ] 输出格式符合统一 Evidence JSON Schema
- [ ] **验证**: lookup_nutrition("tomato") 返回包含 calories_per_100g 的字典；calculate_total 返回包含 total 的字典

### D2.7 MotionTracker

- [ ] 实现 MotionTracker.__init__(sam2_video_predictor, slam_loader)
- [ ] 实现 MotionTracker.track_object(video_path, first_frame_mask)
- [ ] 实现 MotionTracker.extract_trajectory(masks_sequence)
- [ ] 实现 MotionTracker.lift_to_3d(trajectory_2d, slam_poses)
- [ ] 实现 MotionTracker.classify_motion(trajectory_3d)
- [ ] 输出格式符合统一 Evidence JSON Schema
- [ ] **验证**: extract_trajectory 返回坐标列表非空；lift_to_3d 返回 3D 坐标列表

---

## Phase 3: 模块接线与协作

> 确保各模块不是孤岛，而是通过统一接口相互协作

### D3.1 统一证据格式

- [x] 定义 Evidence dataclass (evidence_id, source_module, evidence_type, time_range, content, confidence)
- [ ] 实现 Evidence.to_json() 和 Evidence.from_json()
- [x] 所有感知模块的输出必须返回 Evidence 或 Evidence 列表
- [x] **验证**: 创建 Evidence → to_json → from_json，所有字段一致

### D3.2 模块注册表

- [x] 实现 ModuleRegistry: 统一管理所有感知模块的实例
- [ ] 实现 ModuleRegistry.get(module_name) — 按名称获取模块实例
- [x] 实现 ModuleRegistry.list_modules() — 列出所有已注册模块
- [x] 每个感知模块初始化时自动注册到 Registry
- [ ] **验证**: Registry 包含 7 个感知模块，get("AudioAnalyzer") 返回正确实例

### D3.3 模块间数据流接线

- [ ] AudioAnalyzer → VisualAnalyzer: 音频事件的时间戳驱动 VideoLoader 提取对应帧
- [ ] GazeTracker → VisualAnalyzer: 注视点坐标驱动裁剪区域，辅助物体检测
- [ ] GazeTracker → VideoLoader: 注视 fixation 时间戳作为关键帧选择依据
- [ ] HandsLoader + Grounding DINO → HandInteractor: 手部掩码 + 检测结果 → 接触物体
- [ ] SLAMLoader + DigitalTwinLoader → SpatialReasoner: 位姿 + 3D 模型 → 空间关系
- [ ] VisualAnalyzer (检测框) → SAM 2.1 (分割) → MotionTracker (追踪): 检测-分割-追踪管线
- [ ] HandInteractor (接触物体) + AudioAnalyzer (声音) → 动作推断融合
- [ ] **验证**: 给定一个时间戳 t，AudioAnalyzer 输出事件 → 自动提取 t 对应的视频帧 → VisualAnalyzer 分析该帧，整个链路跑通

### D3.4 时间对齐中枢

- [x] 实现 TimeAlignHub: 所有模块共享的时间对齐服务
- [x] TimeAlignHub.register_timebase(module_name, timebase) — 注册模块时间基准
- [x] TimeAlignHub.convert(timestamp, from_module, to_module) — 跨模块时间转换
- [x] 所有模块间传递时间戳时必须通过 TimeAlignHub 转换
- [ ] **验证**: Audio 模块的时间戳转换为 Video 模块的时间戳后，提取的帧与音频事件对应

---

## Phase 4: 推理引擎

### D4.1 问题路由器

- [ ] 实现 Router.classify_question(question) — 问题分类
- [ ] 实现 Router.get_route(category) — 获取路由策略
- [ ] 实现 Router.route(question) — 完整路由流程
- [ ] 加载 config/routing_table.yaml
- [x] **验证**: classify_question("水槽在哪里") 返回 "3d_perception"；route() 返回包含 primary 和 secondary 模块列表的字典

### D4.2 证据聚合引擎

- [ ] 实现 Aggregator.add_evidence(evidence)
- [ ] 实现 Aggregator.align_evidence() — 多模态时间对齐 (调用 TimeAlignHub)
- [ ] 实现 Aggregator.detect_conflicts() — 冲突检测
- [ ] 实现 Aggregator.fuse_evidence() — 加权融合
- [ ] 实现 Aggregator.get_confidence() — 总体置信度
- [ ] 实现 Aggregator.get_summary() — 证据摘要 (供 LLM 使用)
- [x] **验证**: 添加 3 条同时间戳证据 → fuse_evidence → get_confidence 返回 0-1 之间的浮点数

### D4.3 自适应深度控制

- [ ] 实现 Judge.evaluate_sufficiency(evidence_list, question)
- [ ] 实现 Judge.suggest_expansion(evidence_list, question, route) — 返回下一步应调用的模块+参数
- [ ] 实现 Judge.should_stop(evidence_list, iteration)
- [x] 实现置信度阈值逻辑 (>0.8 直接回答, 0.5-0.8 扩展, <0.5 全面搜索)
- [x] **验证**: 输入高置信度证据 → evaluate_sufficiency 返回 "sufficient"；输入低置信度 → 返回 "insufficient" + 扩展建议 (包含具体模块名和时间范围)

### D4.4 答案生成器

- [ ] 实现 Generator.generate_answer(question, evidence_list, mimo_api)
- [ ] 实现 Generator.format_evidence_prompt(evidence_list) — 多模态证据格式化为 prompt
- [ ] 实现 Generator.parse_answer(response)
- [x] **验证**: format_evidence_prompt 返回非空字符串；parse_answer 对 mock 响应返回结构化答案

### D4.5 工具定义与注册

- [x] 定义感知工具 JSON Schema (query_audio, query_video, query_gaze, query_3d, query_hands, query_nutrition, query_motion)
- [x] 定义分析工具 JSON Schema (generate_scene_graph, compute_spatial, infer_action, predict_next)
- [x] 定义控制工具 JSON Schema (check_evidence, expand_search, synthesize_answer)
- [x] 每个工具的实现函数与 Schema 绑定
- [x] 实现 ToolRegistry: 管理所有工具的注册和调用
- [x] **验证**: ToolRegistry.list_tools() 返回所有工具；调用 query_audio(start=0, end=10) 返回 Evidence 列表

---

## Phase 5: Agent 自主循环

> 这是整个项目的核心：Agent 自主决定调用哪些工具、看哪些数据、是否需要更多信息

### D5.1 Agent 状态管理

- [x] 实现 AgentState: 维护单次问答的完整状态
  - [x] question: 原始问题
  - [x] route: Router 返回的路由策略
  - [x] evidence_list: 已收集的所有 Evidence
  - [x] iteration: 当前迭代轮次
  - [x] tool_call_history: 已调用的工具列表
  - [x] confidence_history: 每轮的置信度变化
- [x] AgentState 的序列化/反序列化 (用于调试和日志)
- [x] **验证**: 创建 AgentState，添加 3 条 Evidence，迭代轮次自增，所有字段可正确访问

### D5.2 Agent Tool-Calling 循环

- [x] 实现 Agent.run(question) — 核心自主循环
  - [x] Step 1: 初始化 AgentState
  - [x] Step 2: Router 分类问题，获取初始路由
  - [x] Step 3: 进入循环 (最大 N 轮)
    - [ ] 3a. 将当前 state (问题+已有证据+路由) 发送给 MiMo2.5
    - [ ] 3b. MiMo2.5 返回下一步要调用的 tool + 参数 (自主决策)
    - [ ] 3c. 执行 tool 调用，获得新 Evidence
    - [ ] 3d. 将新 Evidence 加入 AgentState
    - [ ] 3e. Judge 评估是否足够回答
    - [ ] 3f. 如果足够 → 退出循环；如果不够 → 继续下一轮
  - [x] Step 4: Generator 生成最终答案
  - [x] Step 5: 返回 answer + evidence_chain + reasoning_trace
- [ ] **验证**: 对一个简单问题运行 agent.run()，观察 tool_call_history 至少包含 2 次不同 tool 调用，最终返回非空 answer

### D5.3 Agent 决策 Prompt

- [x] 实现 SYSTEM_PROMPT: 告诉 LLM 它是一个厨房视频理解 Agent，可用的工具列表，决策规则
- [x] 实现 build_decision_prompt(state) — 将 AgentState 格式化为 LLM 输入
  - [ ] 包含: 问题、已收集证据摘要、已调用工具列表、剩余可选工具
  - [ ] 不包含: 原始数据 (太大)，只包含摘要
- [x] 实现 parse_tool_call(llm_response) — 从 LLM 响应中提取 tool_name + parameters
- [x] **验证**: build_decision_prompt 返回的字符串长度 < 4000 tokens；parse_tool_call 对 mock 响应正确提取

### D5.4 Agent 安全边界

- [x] 实现最大迭代轮次限制 (默认 N=10)
- [x] 实现单次问答最大 tool 调用次数限制 (默认 20)
- [x] 实现重复调用检测 (同一 tool + 同一参数不重复调用)
- [x] 实现超时保护 (单次问答总时间 < 120s)
- [x] **验证**: 设置 N=2 时，Agent 在 2 轮后强制退出并用已有证据生成答案

### D5.5 Agent 推理轨迹

- [x] 实现 ReasoningTrace: 记录 Agent 的完整决策过程
  - [x] 每轮记录: 轮次、LLM 决策、tool 调用、返回结果、置信度变化
  - [x] 最终记录: 总轮次、总 tool 调用数、最终置信度、答案
- [x] ReasoningTrace.to_json() — 输出为可读的 JSON 日志
- [ ] ReasoningTrace可视化 — 生成决策树/流程图 (可选)
- [x] **验证**: 运行一次 agent.run()，ReasoningTrace.to_json() 输出包含所有轮次的完整记录

---

## Phase 6: 知识模块

### D6.1 RecipeKB

- [ ] 实现 RecipeKB.__init__(recipe_data_path)
- [ ] 实现 RecipeKB.get_recipe(name)
- [ ] 实现 RecipeKB.get_step(recipe_name, step_number)
- [ ] 实现 RecipeKB.search_recipes(ingredients)
- [ ] 实现 RecipeKB.match_current_step(observations)
- [ ] 注册为 Agent 可调用的 tool (query_recipe)
- [ ] **验证**: get_recipe 返回包含 steps 列表的字典；通过 Agent tool-calling 调用 query_recipe 返回正确结果

### D6.2 NutritionKB

- [ ] 实现 NutritionKB.__init__(nutrition_data_path)
- [ ] 实现 NutritionKB.lookup(ingredient)
- [ ] 实现 NutritionKB.calculate_dish(ingredients)
- [ ] 注册为 Agent 可调用的 tool (query_nutrition)
- [ ] **验证**: lookup("tomato") 返回包含 calories 的字典；通过 Agent tool-calling 调用返回正确结果

### D6.3 SceneGraphKB

- [ ] 实现 SceneGraphKB.__init__()
- [ ] 实现 SceneGraphKB.add_frame_graph(timestamp, graph)
- [ ] 实现 SceneGraphKB.query_objects(object_type)
- [ ] 实现 SceneGraphKB.query_relations(subject, predicate)
- [ ] 实现 SceneGraphKB.get_scene_summary(start_time, end_time)
- [ ] 注册为 Agent 可调用的 tool (query_scene_graph)
- [ ] **验证**: add_frame_graph 后 query_objects 返回包含该物体的时间戳列表

### D6.4 CommonSenseKB

- [ ] 实现 CommonSenseKB.__init__(conceptnet_url)
- [ ] 实现 CommonSenseKB.query_relation(concept_a, relation, concept_b)
- [ ] 实现 CommonSenseKB.get_related_concepts(concept, relation)
- [ ] 实现 CommonSenseKB.infer_cooking_purpose(ingredients)
- [ ] 注册为 Agent 可调用的 tool (query_commonsense)
- [ ] **验证**: get_related_concepts("cooking", "UsedFor") 返回非空列表

---

## Phase 7: 集成测试

> 验证多模块协作和 Agent 完整闭环

### D7.1 双模块协作测试

- [x] Audio + Video: 音频事件时间戳 → 提取对应帧 → 视觉分析，整条链路跑通
- [x] Gaze + Video: 注视点 → 裁剪区域 → 物体检测，返回注视目标
- [x] Hands + Grounding DINO: 手部掩码 + 检测框 → 接触物体识别
- [x] SLAM + Digital Twin: 位姿 + 3D 模型 → 空间关系查询
- [ ] Visual + Motion: 检测框 → SAM 分割 → 视频追踪 → 运动轨迹
- [x] 每个测试验证: 输入→中间输出→最终输出全链路数据格式一致

### D7.2 三模块协作测试

- [ ] Audio + Hands + Visual: 声音事件 + 手部动作 + 视觉 → 高置信度动作识别
- [ ] Gaze + Visual + 3D: 注视 + 视觉 + 空间 → 完整场景理解
- [ ] Hands + Audio + Recipe: 手部动作 + 声音 + 菜谱 → 当前步骤匹配
- [ ] 每个测试验证: 多模态证据融合后置信度高于单模态

### D7.3 Agent 完整闭环测试

- [x] 简单问题: "画面中有什么食材？" → Agent 调用 VisualAnalyzer → 返回答案
- [x] 空间问题: "水槽在哪里？" → Agent 调用 SpatialReasoner → 返回答案
- [ ] 动作问题: "正在做什么？" → Agent 调用 HandInteractor + AudioAnalyzer → 融合 → 返回答案
- [ ] 多步推理: "这道菜有多少卡路里？" → Agent 调用 VisualAnalyzer → NutritionEstimator → NutritionKB → 返回答案
- [ ] 自适应扩展: Agent 初始证据不足 → 自动调用更多模块 → 最终生成答案
- [ ] 每个测试验证: ReasoningTrace 显示 Agent 自主决策过程，tool_call_history 记录完整

### D7.4 Agent 容错测试

- [ ] 模块超时: 某个模块 API 超时时 Agent 能跳过并用其他模块的证据
- [ ] 证据冲突: 两个模块给出矛盾证据时 Agent 能识别并降低置信度
- [x] 数据缺失: 请求的时间戳没有数据时 Agent 能调整时间范围重试
- [ ] 达到上限: 迭代次数达上限时 Agent 用已有证据生成最佳答案
- [ ] 每个测试验证: Agent 不崩溃，返回合理答案或明确的"无法确定"

### D7.5 Prompt 优化

- [x] 定义 SCENE_GRAPH_PROMPT
- [ ] 定义 ACTION_RECOGNITION_PROMPT
- [ ] 定义 INGREDIENT_IDENTIFICATION_PROMPT
- [ ] 定义 PORTION_ESTIMATION_PROMPT
- [ ] 定义 SPATIAL_DESCRIPTION_PROMPT
- [x] 定义 ANSWER_GENERATION_PROMPT
- [x] 定义 QUESTION_CLASSIFICATION_PROMPT
- [x] 定义 SYSTEM_PROMPT (Agent 决策)
- [x] 定义 build_decision_prompt (Agent 状态→LLM 输入)
- [ ] **验证**: 每个 prompt 模板用 .format() 填充参数后无 KeyError，返回非空字符串

---

## Phase 8: 评估框架

### D8.1 API 客户端

- [ ] 实现 api_client.call_mimo_vision(image, prompt)
- [ ] 实现 api_client.call_mimo_text(prompt)
- [x] 实现重试策略 (指数退避, 最大 3 次)
- [x] 实现错误处理 (超时/限流/无效响应)
- [x] 实现请求缓存
- [x] **验证**: call_mimo_text("回复 OK") 返回非空字符串；相同请求第二次走缓存

### D8.2 Benchmark 加载

- [ ] 实现 BenchmarkLoader.__init__(benchmark_path)
- [ ] 实现 BenchmarkLoader.get_questions(category)
- [ ] 实现 BenchmarkLoader.get_question_detail(question_id)
- [ ] 实现 BenchmarkLoader.get_categories()
- [x] **验证**: get_categories() 返回 7 个类别；get_questions() 返回 26K 条非空列表

### D8.3 评估指标

- [x] 实现 accuracy(predictions, ground_truth)
- [x] 实现 accuracy_per_category(predictions, ground_truth, categories)
- [x] 实现 average_confidence(predictions)
- [x] 实现 average_tool_calls(predictions)
- [x] 实现 average_latency(predictions)
- [x] **验证**: accuracy([1,1,0], [1,1,1]) 返回 2/3

### D8.4 评估脚本

- [x] 实现 run_eval.py 主脚本 (加载 benchmark → 调用 Agent.run() → 收集结果 → 计算指标 → 输出报告)
- [x] 支持按类别分别评估
- [x] 输出 JSON + 可读文本两种格式的评估报告
- [x] 输出 ReasoningTrace 日志 (用于分析 Agent 决策行为)
- [ ] **验证**: 对 10 条问题运行评估脚本，输出包含总体准确率、各类别准确率、平均 tool 调用数

### D8.5 消融实验

- [ ] **消融 1: 自主 vs 固定** — Agent 自主循环 vs 固定路由管线，证明自主决策更优
- [ ] **消融 2: Skill 引导 vs 无引导** — 有 skill prompt vs 纯自主（无任何指导），证明 skill 有效
- [ ] **消融 3: 维度递增** — Video+Audio → +Gaze → +3D → +Hands → +Nutrition → +Motion，证明每增加一个维度都有提升
- [ ] **消融 4: 迭代轮次** — 最大 1/3/5/10 轮，证明多轮自主探索有效
- [ ] **消融 5: 工具集大小** — 只给 3 个工具 vs 7 个 vs 全部，证明工具越多 Agent 越强
- [ ] 输出各实验准确率对比表 + 平均 tool 调用数对比
- [ ] **Agent 决策质量分析** — 统计 ReasoningTrace 中 Agent 的决策模式（高频调用的工具、常见路径、失败模式）
- [ ] **验证**: 每个消融实验产出一行数据，最终表格包含 7 行；决策分析产出统计图表

---

## 清单统计

| Phase | 任务数 | 含验证项 | 预计天数 |
|-------|--------|---------|---------|
| Part A: 预检清单 | 25 | - | - |
| Part B: 数据探查 | 49 | - | - |
| Part C: 环境搭建 | 20 | - | - |
| Phase 1: 项目骨架与数据层 | 55 | 9 个验证 | Day 1-7 |
| Phase 2: 感知模块 | 63 | 7 个验证 | Day 8-14 |
| Phase 3: 模块接线与协作 | 24 | 4 个验证 | Day 15-17 |
| Phase 4: 推理引擎 | 27 | 5 个验证 | Day 18-20 |
| Phase 5: Agent 自主循环 | 28 | 5 个验证 | Day 21-24 |
| Phase 6: 知识模块 | 24 | 4 个验证 | Day 25-27 |
| Phase 7: 集成测试 | 28 | 16 个验证 | Day 28-29 |
| Phase 8: 评估框架 | 32 | 13 个验证 | Day 30 |
| **总计** | **375 项** | **63 个验证** | **30 天** |

---

## Phase 9: 论文准备

### D9.1 核心实验数据

- [ ] 主实验: Agent 在 HD-EPIC VQA 26K 上的总体准确率和各类别准确率
- [ ] 与 SOTA 对比: SceneNet+KnowledgeNet (44.21%), EgoAdapt, EgoReasoner, Gemini Pro (38.5%)
- [ ] 消融实验数据: 5 组消融全部跑完并整理成表
- [ ] Agent 决策分析: ReasoningTrace 统计（工具调用频率、决策路径聚类、失败案例分析）
- [ ] 效率分析: 平均 tool 调用数、平均延迟、API 成本

### D9.2 论文撰写

- [ ] Abstract: 突出 "autonomous agent" + "seven dimensions" + "skill-guided tool-calling"
- [ ] Introduction: 动机（现有固定管线的局限）+ 贡献（自主 Agent 范式）
- [ ] Method: 系统架构 + Agent 循环 + Skill 设计 + 七维感知模块
- [ ] Experiments: 主实验 + 消融 + 决策分析 + 可视化
- [ ] Conclusion: 总结 + 局限 + 未来方向

### D9.3 可视化素材

- [ ] Agent 决策流程图: 一个典型案例的 ReasoningTrace 可视化
- [ ] 工具调用热力图: 不同类别问题的工具调用模式
- [ ] 多模态证据融合示例: 展示 Agent 如何综合多路证据
- [ ] 失败案例分析: Agent 决策错误的 case study

---

## 模块协作关系图

```
                        ┌──────────────┐
                        │   用户提问    │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  Agent.run() │ ← 核心自主循环
                        │  (MiMo2.5    │   LLM 决定调用哪些 tools
                        │   决策引擎)   │
                        └──────┬───────┘
                               │ tool calls
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
   │ AudioLoader │     │ VideoLoader │     │ GazeLoader  │ ...
   │ (数据层)     │     │ (数据层)     │     │ (数据层)     │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
          │                    │                    │
   ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
   │AudioAnalyzer│     │VisualAnalyzer│    │GazeTracker  │ ...
   │ (感知层)     │────▶│ (感知层)     │◀───│ (感知层)     │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
          │           ┌───────┼───────┐           │
          │     ┌─────▼─┐  ┌──▼───┐  ┌▼──────┐   │
          │     │SAM 2.1│  │Ground│  │Motion │   │
          │     │(分割)  │  │ DINO │  │Tracker│   │
          │     └───────┘  └──────┘  └───────┘   │
          │                                       │
          └──────────────┬────────────────────────┘
                         │ Evidence
                  ┌──────▼───────┐
                  │  Aggregator  │ ← 多模态证据融合
                  │  (融合层)     │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │    Judge     │ ← 证据充分性判断
                  │  (判断层)     │──→ 不充分 → 回到 Agent 继续调用 tools
                  └──────┬───────┘
                         │ 充分
                  ┌──────▼───────┐
                  │  Generator   │ ← 答案生成
                  │  (生成层)     │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │    答案 +     │
                  │  推理轨迹     │
                  └──────────────┘
```
