# SSR 项目训练数据流文档

## 1. 数据源 (Data Source)

### 1.1 数据集类
- **类名**: `VADCustomNuScenesDataset`
- **定义文件**: `projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py`

### 1.2 数据路径
- **数据根目录**: `data/nuscenes/` (符号链接到 `/data6_server6/datasets/miaomingze/nuscenes`)
- **训练集标注**: `data/nuscenes/vad_nuscenes_infos_temporal_train.pkl`
- **验证集标注**: `data/nuscenes/vad_nuscenes_infos_temporal_val.pkl`
- **地图标注**: `data/nuscenes/nuscenes_map_anns_val.json`

### 1.3 单样本数据结构
从 `get_data_info` 方法获取的原始数据:
```
input_dict = {
    sample_idx: str,           # 样本唯一标识
    pts_filename: str,         # LiDAR点云文件路径
    sweeps: list,             # 时序扫描数据
    ego2global_translation: [x, y, z],  # 自车到全局坐标的平移
    ego2global_rotation: [w, x, y, z],  # 自车到全局坐标的旋转
    lidar2ego_translation: [x, y, z],   # LiDAR到自车的平移
    lidar2ego_rotation: [w, x, y, z],   # LiDAR到自车的旋转
    prev/next: int,           # 前后帧索引
    scene_token: str,         # 场景标识
    can_bus: dict,            # CAN总线数据（车速、航向角等）
    frame_idx: int,           # 帧序号
    timestamp: float,         # 时间戳
    fut_valid_flag: bool,     # 未来轨迹是否有效
    ego_his_trajs: ndarray,  # 历史自车轨迹 [history_len, 2]
    ego_fut_trajs: ndarray,  # 未来自车轨迹 [future_len, 2]
    ego_fut_masks: ndarray,  # 未来轨迹掩码
    ego_fut_cmd: ndarray,     # 未来命令 (one-hot: 左转/右转/直行)
    ego_lcf_feat: ndarray,   # 局部坐标系特征
    map_location: str,        # 地图位置
}
```

---

## 2. 数据预处理管道 (Train Pipeline)

**配置文件**: `projects/configs/SSR/SSR_e2e.py` (lines 256-269)

```python
train_pipeline = [
    # 1. 加载多视角相机图像 (6个相机)
    LoadMultiViewImageFromFiles(to_float32=True),

    # 2. 颜色扰动 (亮度、对比度、饱和度、色调)
    PhotoMetricDistortionMultiViewImage(),

    # 3. 加载3D标注 (边界框 + 属性)
    LoadAnnotations3D(with_bbox_3d=True, with_label_3d=True, with_attr_label=True),

    # 4. 按范围过滤物体 (point_cloud_range: [-15, -30, -2, 15, 30, 2])
    CustomObjectRangeFilter(point_cloud_range),

    # 5. 按类别名称过滤 (car, truck, bus, pedestrian, ...)
    CustomObjectNameFilter(classes=class_names),

    # 6. 图像归一化 (ImageNet mean/std)
    NormalizeMultiviewImage(mean, std, to_rgb=True),

    # 7. 随机缩放 (scale=0.4)
    RandomScaleImageMultiViewImage(scales=[0.4]),

    # 8. 填充到32的倍数
    PadMultiViewImage(size_divisor=32),

    # 9. 格式化数据
    CustomDefaultFormatBundle3D(with_ego=True),

    # 10. 收集所需 keys
    CustomCollect3D(keys=[
        'gt_bboxes_3d', 'gt_labels_3d', 'img',
        'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks',
        'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'
    ])
]
```

**输出keys说明**:
| Key | 形状 | 含义 |
|-----|------|------|
| `img` | [6, 3, H, W] | 6个相机的RGB图像 |
| `gt_bboxes_3d` | [N, 9] | 3D边界框 (cx, cy, cz, w, h, l, rot, vx, vy) |
| `gt_labels_3d` | [N] | 边界框类别ID |
| `ego_his_trajs` | [history_len, 2] | 历史轨迹 (x, y) |
| `ego_fut_trajs` | [future_len, 2] | 未来轨迹 (x, y) |
| `ego_fut_masks` | [future_len] | 未来轨迹有效掩码 |
| `ego_fut_cmd` | [3] | 命令one-hot编码 |
| `ego_lcf_feat` | [...] | 局部坐标系特征 |

---

## 3. 数据加载 (Data Loading)

**Builder文件**: `projects/mmdet3d_plugin/datasets/builder.py`

```python
# 构建分布式采样器
sampler = DistributedGroupSampler(
    dataset,
    samples_per_gpu=1,    # 每GPU样本数
    scenes_per_gpu=1      # 每GPU场景数
)

# 构建DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=sampler,
    num_workers=4,       # 每个GPU 4个worker
    collate_fn=collate,   # 自定义合并函数
    pin_memory=True,
    shuffle=True,
)
```

**配置** (`SSR_e2e.py`):
```python
data = dict(
    samples_per_gpu=1,    # 每GPU处理1个样本
    workers_per_gpu=4,    # 4个数据加载worker
    shuffler_sampler=dict(type='DistributedGroupSampler'),
)
```

---

## 4. 时序数据准备 (Temporal Queue)

**方法**: `prepare_train_data` (`nuscenes_vad_dataset.py` lines 1118-1179)

### 4.1 Queue结构
模型使用 `queue_length=3` 帧的时序队列:

```
data_queue = [prev_frame_0, prev_frame_1, current_frame, future_frame]
                    ↓            ↓           ↓              ↓
                 历史帧      历史帧      当前帧       未来帧(规划目标)
```

### 4.2 队列构建流程

```python
def prepare_train_data(self, index):
    data_queue = []

    # 1. 打散历史帧索引 (时序数据增强)
    prev_indexs_list = list(range(index - self.queue_length, index))
    random.shuffle(prev_indexs_list)

    # 2. 当前帧处理
    input_dict = self.get_data_info(index)
    example = self.pipeline(input_dict)           # 应用train_pipeline
    example = self.vectormap_pipeline(example)     # 生成矢量化地图
    data_queue.insert(0, example)                  # 插入队列开头

    # 3. 未来帧 (用于规划监督, 跳帧3)
    future_index = min(index + 3, len(self.data_infos) - 1)
    ...

    # 4. 历史帧
    for i in prev_indexs_list:
        ...
        data_queue.insert(0, copy.deepcopy(example))

    # 5. 合并队列为单个样本
    return self.union2one(data_queue)
```

### 4.3 union2one 合并
将队列中所有帧的数据合并为一个样本:

```python
# 图像堆叠: (queue_length, 6, 3, H, W) → (queue_length, 6, 3, H, W)
imgs_stack = torch.stack([q['img'] for q in data_queue])

# 轨迹堆叠: (queue_length, N, 2) → (queue_length, N, 2)
trajs_stack = torch.stack([q['ego_his_trajs'] for q in data_queue])

# can_bus调整为相对第一帧
can_bus_relative = can_bus - first_frame_can_bus
```

**最终输出**:
```
example = {
    'img': torch.Size([4, 6, 3, 900, 1600]),    # [queue, cameras, C, H, W]
    'ego_his_trajs': torch.Size([4, 2]),         # [queue, 2]
    'ego_fut_trajs': torch.Size([4, 2]),
    'ego_fut_masks': torch.Size([4]),
    'ego_fut_cmd': torch.Size([4, 3]),
    'gt_bboxes_3d': list of LiDARInstance3DBoxes,
    'gt_labels_3d': torch.Tensor([N]),
    ...
}
```

---

## 5. 前向传播 (Forward Pass)

### 5.1 模型输入

**SSR模型前向** (`SSR.py` lines 250-329):
```python
def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d,
                 map_gt_bboxes_3d, map_gt_labels_3d,
                 ego_his_trajs, ego_fut_trajs, ego_fut_masks,
                 ego_fut_cmd, ego_lcf_feat, gt_attr_labels, ...):
```

### 5.2 时序数据解包
```python
len_queue = img.size(1)  # queue_length = 4

# 历史帧 (前3帧)
prev_img = img[:, :-2, ...]     # [B, 2, 6, 3, H, W]
prev_cmd = ego_fut_cmd[:, :-2]   # [B, 2, 3]

# 未来帧 (用于监督)
next_img = img[:, -1, ...]      # [B, 6, 3, H, W]
next_cmd = ego_fut_cmd[:, -1]    # [B, 3]

# 当前帧
img = img[:, -2, ...]           # [B, 6, 3, H, W]
ego_fut_cmd = ego_fut_cmd[:, -2]  # [B, 3]
```

### 5.3 历史BEV特征提取
```python
def obtain_history_bev(prev_img, prev_img_metas, prev_cmd):
    # 逐帧处理历史
    for i in range(len(prev_img)):
        img_feat = extract_feat(prev_img[:, i])  # ResNet50 → FPN
        prev_bev = pts_bbox_head(
            img_feats, prev_img_metas, prev_bev,
            only_bev=True, cmd=prev_cmd[:, i]
        )
    return prev_bev
```

### 5.4 当前帧特征提取
```python
# 1. 主干网络
img_feats = self.extract_feat(img)  # ResNet50 + FPN
# img_feats: list of [B, 256, H, W]

# 2. Transformer编码器
bev_embed = self.encoder(img_feats, img_metas, prev_bev)

# 3. BEV解码 + 轨迹预测
outs = self.pts_bbox_head(
    img_feats, img_metas, prev_bev,
    ego_his_trajs=ego_his_trajs,
    ego_lcf_feat=ego_lcf_feat,
    cmd=ego_fut_cmd
)
```

### 5.5 模型输出

**SSR_head输出** (`SSR_head.py` lines 366-472):
```python
outs = {
    'bev_embed': bev_embed,           # [B, 256, H, W] BEV特征图
    'scene_query': latent_query,      # [B, num_scenes, 256] 场景查询
    'act_query': act_query,          # [B, num_scenes, 256] 动作查询
    'ego_fut_preds': outputs_ego_trajs,  # [B, 3, 6, 2] 预测轨迹
    # 3=3种命令, 6=时间步, 2=(x,y)
}
```

---

## 6. 损失计算 (Loss Computation)

**方法**: `SSR_head.loss()` (`SSR_head.py` lines 475-536)

### 6.1 轨迹损失
```python
loss_plan_reg = F.l1_loss(ego_fut_preds, gt_ego_fut_trajs)  # 轨迹L1损失
loss_plan_dir = PtsDirCosLoss(pred_dir, gt_dir)              # 方向余弦损失
loss_plan_col = PlanCollisionLoss(pred_traj, ...)             # 碰撞损失
```

### 6.2 地图损失
```python
loss_map_cls = FocalLoss(map_preds, map_targets)  # 地图元素分类
loss_map_pts = PtsL1Loss(map_pts, gt_pts)         # 地图点回归
```

### 6.3 潜在世界模型损失 (可选)
```python
if self.latent_world_model is not None:
    pred_latent = self.latent_world_model(act_query)
    pred_bev = self.tokenfuser(pred_latent, bev_embed)
    loss_bev = F.mse_loss(pred_bev, next_bev.detach())  # BEV重建损失
```

### 6.4 总损失
```python
loss_dict = {
    'loss_plan_reg': 1.0,    # 轨迹回归
    'loss_plan_cls': 0.2,   # 命令分类
    'loss_plan_col': 1.0,    # 碰撞
    'loss_plan_dir': 0.5,    # 方向
    'loss_map_cls': 2.0,     # 地图分类
    'loss_map_pts': 1.0,     # 地图点
    'loss_bev': 1.0,         # BEV重建 (如果有)
}
total_loss = sum(loss for loss in loss_dict.values())
```

---

## 7. 训练循环 (Training Loop)

**入口**: `projects/mmdet3d_plugin/SSR/apis/mmdet_train.py`

### 7.1 custom_train_detector 流程

```python
def custom_train_detector(model, dataset, cfg, distributed=False, validate=False):
    # 1. 构建DataLoader
    data_loaders = [build_dataloader(ds, ...) for ds in dataset]

    # 2. 包装模型
    if distributed:
        model = MMDistributedDataParallel(model.cuda())
    else:
        model = MMDataParallel(model, device_ids=[0])

    # 3. 构建优化器 (AdamW, lr=5e-5)
    optimizer = build_optimizer(model, cfg.optimizer)

    # 4. 构建Runner (EpochBasedRunner, max_epochs=12)
    runner = build_runner(cfg.runner, default_args=dict(...))

    # 5. 注册Hook
    runner.register_training_hooks(
        lr_config=lr_config,          # CosineAnnealing
        optimizer_config=optimizer_config,  # 梯度裁剪
        checkpoint_config=checkpoint_config, # 保存策略
        log_config=log_config,         # TensorBoard日志
    )

    # 6. 自定义Hook
    runner.register_hook(CustomSetEpochInfoHook())
    runner.register_hook(MEGVIIEMAHook(init_updates=10560))

    # 7. 运行训练
    runner.run(data_loaders, workflow=[('train', 1)])
```

### 7.2 关键配置
```python
# 优化器
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率策略
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    min_lr_ratio=1e-3
)

# Runner
runner = dict(type='EpochBasedRunner', max_epochs=12)

# 钩子
custom_hooks = [
    dict(type='CustomSetEpochInfoHook'),
    dict(type='MEGVIIEMAHook', init_updates=10560),  # EMA
]
```

---

## 8. 数据流总览图

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据源 (Data Source)                      │
│  data/nuscenes/vad_nuscenes_infos_temporal_train.pkl           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VADCustomNuScenesDataset                     │
│  • get_data_info(): 读取样本元数据                                │
│  • prepare_train_data(): 构建时序队列                             │
│  • vectormap_pipeline(): 生成矢量化地图                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Train Pipeline (10步)                       │
│  1. LoadMultiViewImageFromFiles (6相机)                         │
│  2. PhotoMetricDistortionMultiViewImage                          │
│  3. LoadAnnotations3D                                            │
│  4. CustomObjectRangeFilter                                      │
│  5. CustomObjectNameFilter                                       │
│  6. NormalizeMultiviewImage                                      │
│  7. RandomScaleImageMultiViewImage                               │
│  8. PadMultiViewImage (32对齐)                                   │
│  9. CustomDefaultFormatBundle3D                                  │
│  10. CustomCollect3D                                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DataLoader                                │
│  • batch_size=1, num_workers=4                                  │
│  • DistributedGroupSampler (分布式)                               │
│  • collate_fn: 合并队列为单一样本                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Forward Pass                                │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ History BEV  │    │ Current Feat │    │ Future Frame │      │
│  │  (obtain_   │    │  (extract_   │    │  (next_bev   │      │
│  │  history_bev)│───▶│  feat)       │    │   for loss)  │      │
│  └──────────────┘    └──────┬───────┘    └──────────────┘      │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────┐                          │
│                    │  SSR_head       │                          │
│                    │  • Transformer   │                          │
│                    │  • BEV Decoder  │                          │
│                    │  • Traj Predict │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────┐                          │
│                    │  Loss Compute   │                          │
│                    │  • plan_reg     │                          │
│                    │  • plan_cls     │                          │
│                    │  • plan_col     │                          │
│                    │  • map_cls/pts  │                          │
│                    │  • bev (if LWM) │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键文件索引

| 文件 | 行号 | 功能 |
|------|------|------|
| `projects/configs/SSR/SSR_e2e.py` | 256-269 | train_pipeline定义 |
| `projects/configs/SSR/SSR_e2e.py` | 298-346 | data配置 |
| `projects/configs/SSR/SSR_e2e.py` | 348-385 | optimizer/runner配置 |
| `nuscenes_vad_dataset.py` | 1118-1179 | prepare_train_data |
| `nuscenes_vad_dataset.py` | 1197-1235 | union2one |
| `nuscenes_vad_dataset.py` | 1699-1758 | evaluate |
| `SSR.py` | 218-235 | obtain_history_bev |
| `SSR.py` | 250-329 | forward_train |
| `SSR_head.py` | 366-472 | forward (预测) |
| `SSR_head.py` | 475-536 | loss (损失计算) |
| `mmdet_train.py` | 23-194 | custom_train_detector |

---

## 10. 评估指标

**配置文件**: `custom_eval_version='vad_nusc_detection_cvpr_2019'`

### 10.1 规划指标
| 指标 | 说明 |
|------|------|
| `plan_L2_1s/2s/3s` | 1/2/3秒规划轨迹L2误差(米) |
| `plan_obj_col_1s/2s/3s` | 1/2/3秒目标碰撞率 |
| `plan_obj_box_col_1s/2s/3s` | 1/2/3秒边界框碰撞率 |
| `plan_L2_stp3_*` | Step-3模式的L2误差 |

### 10.2 当前模型评估结果
```
有效样本数: 5119/6019

plan_L2_1s: 0.186m
plan_L2_2s: 0.355m
plan_L2_3s: 0.615m

plan_obj_col: ~0.00% (几乎无碰撞)
plan_obj_box_col: 0.10% ~ 0.24%
```
