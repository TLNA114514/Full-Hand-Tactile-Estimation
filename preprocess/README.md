# 数据预处理脚本说明

这个目录是新的数据预处理入口。`preprocess/` 内部已经包含完整处理逻辑，不再依赖旧的 `hamer_tactile_ft/extract_*.py` 或 `scratch/*.py` 预处理脚本。bbox cache、合并后的 bbox json 等 artifact 会默认放到：

```text
preprocess/artifacts/
  opentouch/
  touchanything/
  egotactile/
```

`sync_to_server.sh` 已经排除了 `preprocess/artifacts/` 和旧的 cache/json 路径，因此同步代码时不会因为 `rsync --delete` 删除远程服务器上已经生成的数据 cache。

## 迁移和清理

先 dry-run，看旧文件会被移动到哪里：

```bash
python preprocess/migrate_layout.py
```

确认没问题后执行迁移。脚本会把旧 artifact 移到 `preprocess/artifacts/`，并在旧位置留下 symlink，方便过渡期旧命令仍能找到数据：

```bash
python preprocess/migrate_layout.py --apply
```

如果新 artifact 已经存在，只想补旧路径 symlink：

```bash
python preprocess/migrate_layout.py --link_existing --apply
```

等之前的旧进程都跑完、并确认新入口可用后，可以清理旧位置的 artifact symlink：

```bash
python preprocess/cleanup_legacy_artifacts.py
python preprocess/cleanup_legacy_artifacts.py --apply
```

默认只删 symlink，不会删真实 artifact 文件/目录。如果你明确确认旧真实 artifact 也可以删除，再使用：

```bash
python preprocess/cleanup_legacy_artifacts.py --delete_real_artifacts --apply
```

如果你还想把旧的预处理代码文件也列出来：

```bash
python preprocess/cleanup_legacy_artifacts.py --include_code
```

清理旧预处理代码：

```bash
python preprocess/cleanup_legacy_artifacts.py --include_code --apply
```

`preprocess/` 已经独立，通常不需要额外的 `--delete_dependent_code`。这个参数保留只是为了兼容之前的清理命令：

```bash
python preprocess/cleanup_legacy_artifacts.py --include_code --delete_dependent_code --apply
```

## OpenTouch

### 处理

提取 OpenTouch 全量数据：

```bash
python preprocess/opentouch/process.py \
  --gpu 0,1,2,3,4,5,6,7 \
  --data_dir "/data1/jiangrui/OpenTouch Data/data" \
  --output_dir "/data1/jiangrui/OpenTouch Data/full_dataset"
```

新入口会默认把这些文件放到 `preprocess/artifacts/opentouch/`：

```text
full_bboxes_cache/
opentouch_all_bboxes.json
dataset_frames_registry.json
```

### 检查

检查文件是否存在、`meta.json` 是否能解析、关键字段和压力字段是否基本合理：

```bash
python preprocess/opentouch/check.py \
  --output_dir "/data1/jiangrui/OpenTouch Data/full_dataset" \
  --workers 64
```

更严格地检查 registry 与 meta 的 `frame_idx/is_right` 是否一致：

```bash
python preprocess/opentouch/check.py \
  --output_dir "/data1/jiangrui/OpenTouch Data/full_dataset" \
  --strict \
  --workers 64
```

### Repair

OpenTouch 目前的 repair 入口是提示型的，主要用于提醒重新生成连续/Gaussian 压力：

```bash
python preprocess/opentouch/repair.py
```

## TouchAnything

### 处理

跳过 bbox，只补齐 frame/meta：

```bash
python preprocess/touchanything/process.py \
  --gpu 0,1,2,3,4,5,6,7 \
  --skip_bbox \
  --extract_workers 24 \
  --prefilter_workers 64
```

如果要重新提 bbox：

```bash
python preprocess/touchanything/process.py \
  --gpu 0,1,2,3,4,5,6,7 \
  --extract_workers 24 \
  --prefilter_workers 64
```

### 检查

检查 `chest.jpg/left.jpg/right.jpg/meta.json` 是否存在，`meta.json` 是否可解析，双手压力 grid/Gaussian 是否在合理范围：

```bash
python preprocess/touchanything/check.py \
  --output_dir /data1/jiangrui/EgoTouch/extracted_frames \
  --workers 64
```

严格检查 registry 与 meta 的帧号一致性：

```bash
python preprocess/touchanything/check.py \
  --output_dir /data1/jiangrui/EgoTouch/extracted_frames \
  --strict \
  --workers 64
```

### Repair

修复坏的或缺失的 `pressure_grids.npz`：

```bash
python preprocess/touchanything/repair.py \
  --root /data1/jiangrui/EgoTouch \
  --only_bad \
  --backup_bad \
  --check_workers 128
```

补生成 Gaussian/continuous 压力：

```bash
python preprocess/touchanything/repair.py \
  --gaussian \
  --gpu 0,1,2,3,4,5,6,7 \
  --ta_dir /data1/jiangrui/EgoTouch \
  --only_missing_continuous \
  --check_workers 128 \
  --workers_per_gpu 4
```

## EgoTactile

### 全量 SAM3 bbox 与 HDF5 激活

正式数据入口使用 SAM3，而不再使用旧 HaMeR/ViTPose bbox。该流程会自动按
`bare_hand/gloved_hand` 选择 prompt，在 8 张 GPU 上可恢复地跟踪全部 clip，严格审计
逐帧覆盖后复制现有 sequence HDF5、替换任务手 bbox、重建 query manifest，最后原子
切换 `extracted_frames_current`：

```bash
./preprocess/egotactile/run_sam3_reconstruction.sh all
```

中断后执行同一条命令即可续跑。查看进度：

```bash
./preprocess/egotactile/run_sam3_reconstruction.sh status
tail -f /home/ma-user/work/cfzhao/EgoTactile/sam3_bbox_reconstruction_v1/full_run.log
```

也可以按阶段执行 `build/track/audit/materialize/activate`。只有全量审计和 HDF5
迁移完成后才会切换 active symlink；原始 HDF5 不会被原位修改。训练代码默认优先
解析 `EGOTACTILE_DATA_ROOT`，否则使用 `extracted_frames_current`，再回退历史路径。

### 官方 train/test manifests

SAM3 HDF5 已经完成时，无需重跑跟踪或复制容器，直接生成论文 Appendix A.1.5
定义的三套 sequence-level split：

```bash
./preprocess/egotactile/run_sam3_reconstruction.sh official-splits
```

输出位于：

```text
extracted_frames_sam3/manifests/official/
├── gloved_object_held_out/{train,test}.{queries,sequences}.jsonl
├── gloved_subject_held_out/{train,test}.{queries,sequences}.jsonl
├── bare_object_held_out/{train,test}.{queries,sequences}.jsonl
└── index.json
```

Object-held-out 固定排除 `Apple/CocaCola-330ml/Corn/Dumbbell/TennisBall`；
Subject-held-out 固定排除 `p007/p011`。每行的 `split` 是训练逻辑上的
`train|test`，`source_split` 保留底层 HDF5 的 `gloved_hand|bare_hand`，因此无需复制
数据文件。官方协议只定义 train/test；脚本不会把 test 伪装成 val。训练和评估时需
显式指定所选 protocol 的 manifest，避免 object/subject 协议混用。

### Repair / Gaussian 生成

先为 EgoTactile 生成归一化 grid 和 Gaussian 压力：

```bash
python preprocess/egotactile/repair.py \
  --egotactile_dir /data1/jiangrui/EgoTactile/Raw_data \
  --gpu 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 4
```

### 处理

如果已有 bbox，可以跳过 bbox，只提取 RGB/压力/meta：

```bash
python preprocess/egotactile/process.py \
  --skip_bbox \
  --require_gaussian \
  --keep_no_bbox \
  --extract_workers 24 \
  --prefilter_workers 64
```

下面的 HaMeR/ViTPose bbox 命令仅用于历史复现；正式数据请使用上面的 SAM3 流程：

```bash
python preprocess/egotactile/process.py \
  --gpu 0,1,2,3,4,5,6,7 \
  --bbox_workers_per_gpu 1 \
  --require_gaussian \
  --keep_no_bbox \
  --extract_workers 24 \
  --prefilter_workers 64
```

### 检查

检查 `image.jpg/meta.json` 是否存在，`meta.json` 是否可解析，raw/normalized/Gaussian 压力字段是否基本合理：

```bash
python preprocess/egotactile/check.py \
  --output_dir /data1/jiangrui/EgoTactile/Raw_data/extracted_frames \
  --workers 64
```

严格检查 registry 与 meta 的 `frame_idx/hand` 一致性：

```bash
python preprocess/egotactile/check.py \
  --output_dir /data1/jiangrui/EgoTactile/Raw_data/extracted_frames \
  --strict \
  --workers 64
```

## σ 最优性评测

这个脚本用于 sweep Dijkstra Gaussian 的 `sigma`，并用全量 sensor leave-one-out cross validation 选择最优值。它直接读取原始 HDF5/NPZ，不依赖逐帧 `meta.json`。

快速预扫：

```bash
python preprocess/evaluate_sigma_optimality.py \
  --datasets opentouch egotactile touchanything \
  --dataset_raw_root_opentouch "/home/ma-user/work/cfzhao/OpenTouch Data/data" \
  --dataset_raw_root_egotactile /home/ma-user/work/cfzhao/EgoTactile/Raw_data \
  --dataset_raw_root_touchanything /home/ma-user/work/cfzhao/EgoTouch \
  --sigma_values 0.001,0.002,0.003,0.005,0.0075,0.01,0.015 \
  --limit_sequences 50 \
  --frame_stride 10 \
  --workers 128 \
  --check_workers 128 \
  --gpu 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 16 \
  --output_dir outputs/sigma_optimality_preview
```

主实验：

```bash
python preprocess/evaluate_sigma_optimality.py \
  --datasets opentouch egotactile touchanything \
  --dataset_raw_root_opentouch "/home/ma-user/work/cfzhao/OpenTouch Data/data" \
  --dataset_raw_root_egotactile /home/ma-user/work/cfzhao/EgoTactile/Raw_data \
  --dataset_raw_root_touchanything /home/ma-user/work/cfzhao/EgoTouch \
  --alpha_values 0.25,0.5,0.75,1.0,1.25 \
  --workers 128 \
  --check_workers 128 \
  --gpu 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 16 \
  --output_dir outputs/sigma_optimality_full
```

输出文件：

```text
sigma_frame_metrics.jsonl
sigma_sequence_metrics.jsonl
sigma_summary.csv
sigma_summary.md
sigma_curves.png
```

`sigma_summary.md` 中的 `Best Sigma By Full-Sensor LOOCV` 表格以 `loocv_mse_all` 最小作为最优 `sigma`，同时报告 `loocv_mse_active` 和 `loocv_mse_inactive`，避免大量 0 sensor 掩盖接触区域误差。

## 备注

- 不建议把 `--extract_workers` 开到 128，除非确认存储 I/O 非常强。通常 24 或 32 更稳。
- 大规模检查时可以提高 `--workers`，例如 64 或 128，但如果共享存储很慢，过高线程数反而会卡。
- 如果怀疑 registry 不可信，可以在对应 `process.py` 使用原脚本支持的 `--no_trust_registry --check_workers 128`。
