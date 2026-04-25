# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSR (**Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving**) is an ICLR 2025 paper from Zhijia Technology. The core innovation is a TokenLearner module that compresses dense BEV features into sparse navigation-guided tokens, reducing computational cost while maintaining accuracy.

The project is built on the **OpenMMLab ecosystem** (mmdetection3d, mmcv, mmdet, mmsegmentation) as a plugin. It is not a library import — the entire mmdetection3d codebase is installed from source.

## Build / Train / Test Commands

### Training (8 GPUs)
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=2333 \
  tools/train.py projects/configs/SSR/SSR_e2e.py \
  --launcher pytorch --deterministic --work-dir /path/to/outputs
```

### Evaluation (1 GPU)
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  projects/configs/SSR/SSR_e2e.py /path/to/ckpt.pth \
  --launcher none --eval bbox --tmpdir tmp
```

### Data Preparation
```bash
python tools/data_converter/vad_nuscenes_converter.py nuscenes \
  --root-path ./data/nuscenes --out-dir ./data/nuscenes \
  --extra-tag vad_nuscenes --version v1.0 --canbus ./data
```

### Visualization
```bash
python projects/mmdet3d_plugin/visualize_attention.py
```

### Planning Metrics
```bash
python tools/analysis_tools/compute_planning_metrics.py
```

## Architecture

### OpenMMLab Plugin Structure

SSR code lives in `projects/mmdet3d_plugin/` as an mmdetection3d plugin:

| Directory | Purpose |
|---|---|
| `SSR/` | Core SSR model: `SSR.py` (main detector), `SSR_head.py` (detection head), `SSR_transformer.py` (perception transformer), `tokenlearner.py` (sparse token compression), `planner/` (planning metrics) |
| `core/` | Bbox assigners (`HungarianAssigner3D`), coders (`NMSFreeCoder`), match costs, custom evaluation |
| `datasets/` | Custom NuScenes VAD dataset, data pipelines (`LoadMultiViewImageFromFiles`, `NormalizeMultiviewImage`, etc.) |
| `models/modules/` | Custom transformer layers: `SpatialCrossAttention`, `TemporalSelfAttention`, `MSDeformableAttention3D` |

### Core Model Flow

The SSR model (`SSR.py`) inherits from `MVXTwoStageDetector` and orchestrates:

1. **Image Backbone** (ResNet-50) → **FPN Neck** → produces image features
2. **SSRPerceptionTransformer** (camera-to-BEV transformation via deformable attention)
3. **TokenLearner** (`TokenFuser` in `tokenlearner.py`) — compresses dense BEV features into sparse tokens using navigation guidance (this is the core innovation)
4. **SSRHead** — performs detection (10 classes) and map element detection (3 classes: divider, ped_crossing, boundary)
5. **Planning decoder** — trajectory planning via cross-attention with sparse tokens
6. **Latent world model** — optional BEV future prediction for auxiliary supervision

Key dimension: `_dim_=256`, `bev_h_=100`, `bev_w_=100`, `voxel_size=0.15m`

### Config System

All configs are **Python files** (not YAML), parsed by `mmcv.Config`. The main config is `projects/configs/SSR/SSR_e2e.py`.

Config inheritance chain:
- `projects/configs/SSR/SSR_e2e.py`
  → `projects/configs/_base_/datasets/custom_nus-3d.py` (dataset + train pipeline)
  → `projects/configs/_base_/default_runtime.py` (logging, checkpoint, dist params)

Config values include:
- `plugin = True`, `plugin_dir = 'projects/mmdet3d_plugin/'`
- `point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]`
- `queue_length = 3` (temporal frames), `total_epochs = 6`
- `num_classes = 10` (detection), `map_num_classes = 3` (map elements)

### Key Entry Points

| File | Purpose |
|---|---|
| `tools/train.py` | Training entry point — builds model/dataset, runs `custom_train_model` via mmcv runner |
| `tools/test.py` | Test/evaluation entry point — runs inference and calls dataset's `.evaluate()` |
| `projects/mmdet3d_plugin/SSR/apis/train.py` | Custom training wrapper (`custom_train_model`) with custom EvalHook |
| `projects/mmdet3d_plugin/SSR/apis/test.py` | Custom multi-GPU test function |
| `projects/mmdet3d_plugin/visualize_attention.py` | Attention map visualization |

### Data Flow

Training input: multi-view camera images (6 cameras) + ego history trajectories + future command. The `queue_length=3` means each sample contains 3 temporal frames.

Train pipeline (`train_pipeline` in config):
1. `LoadMultiViewImageFromFiles` → `PhotoMetricDistortionMultiViewImage` → `LoadAnnotations3D`
2. `CustomObjectRangeFilter` → `CustomObjectNameFilter` → `NormalizeMultiviewImage`
3. `RandomScaleImageMultiViewImage` → `PadMultiViewImage` → `CustomDefaultFormatBundle3D` → `CustomCollect3D`

## Environment

- **Python**: 3.7+
- **PyTorch**: 1.9.1 + CUDA 11.1
- **Key packages**: mmcv-full 1.4.0, mmdet 2.14.0, mmsegmentation 0.14.1, mmdet3d 0.17.1 (installed from source), nuscenes-devkit 1.1.9
- **No formal test suite** — validation is done via `tools/test.py` and planning metric computation

## Dependency Acknowledgements

SSR builds on VAD, GenAD, BEV-Planner, and TokenLearner (Google Research/Scenic).
