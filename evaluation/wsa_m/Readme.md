# WSA-Memory 训练集模拟验证

这个目录用于真机数据无法在线 rollout 时，对训练完成的 WSA-Memory 做离线拟合诊断。它会从多个 LeRobot v3 数据集组成的训练集合中，以固定随机种子分层抽取 10% 的 frame/action-chunk 样本，执行完整 action chunk 推理并与数据中的 GT action 比较。

这不是 WSA-Base 评测脚本的简单改写。程序会复用训练的数据工厂和 WSA-Memory transform，保证以下输入与训练一致：

- 按数据集 FPS 采样的 K 帧视觉和状态历史；
- `memory.history_valid_mask` 和 episode 起点 padding；
- oracle、task-only 或 external 文本 memory；
- WSA-Memory 专用 Qwen3-VL processor 输入；
- action chunk padding mask；
- delta/abs action 变换及训练时使用的归一化统计量；
- checkpoint 中启用 future generation 时的数据时间线。

评测时会关闭训练专用的图像增强、文本 memory dropout 和视觉 history dropout。DA3 teacher 不会加载。

## 启动

在仓库根目录执行：

```bash
conda activate mot

CHECKPOINT=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/wsa_memory/wsa_memory-robotwin-abs-chunk50-pretrained-default-gen0.01-3d0.0-finetune-2026_08_17_07_47_14/checkpoints/020000 \
MAX_SAMPLES=50 \
DATASET_ROOT=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/WorldArena2 \
STATS_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/agilex_abs.json \
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=1 \
NUM_WORKERS=8 \
action_mode=abs \
bash evaluation/wsa_m/run.sh
```

`CHECKPOINT` 支持以下几种路径：

- `.../checkpoints/035000/pretrained_model`
- `.../checkpoints/035000`
- `.../checkpoints/last`
- 完整训练 run 目录，内部需要存在 `checkpoints/last/pretrained_model`

`DATASET_ROOT` 是多个 LeRobot v3 数据集的父目录。脚本会递归发现包含 `meta/info.json` 且带有 `data/` 或 `videos/` 的目录，与 `launch/wsa_base_finetune_multi.sh` 的 discover 逻辑对齐。不要传单个 parquet 文件。

如果模型、processor 和 Cosmos tokenizer 在训练机上的原路径已经失效，可以覆盖：

```bash
QWEN3_VL_PATH=/path/to/Qwen3-VL-2B-Instruct \
PROCESSOR_PATH=/path/to/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH=/path/to/Cosmos-Tokenizer-CI8x8 \
CHECKPOINT=/path/to/checkpoint \
DATASET_ROOT=/path/to/dataset_collection \
STATS_PATH=/path/to/stats.json \
bash evaluation/WSA_Memory_Diagnostic/run.sh
```

可视化依赖 matplotlib。如果当前环境没有：

```bash
python -m pip install matplotlib
```

## 归一化文件加载顺序

为了与现有评测代码和训练数据工厂一致，统计量按以下优先级解析：

1. `STATS_PATH`：训练时使用的单个聚合 stats JSON；
2. `STATS_ROOT`：`<robot_type>/<action_mode>/stats.json` 目录结构；
3. checkpoint 的 `train_config.json` 中仍然有效的 `external_stats_path` 或 `external_stats_root`；
4. checkpoint 内保存的 `pretrained_model/stats.json`；
5. 只有训练配置原本没有使用 external stats 时，才使用每个 LeRobot v3 数据集自己的 `meta/stats.json`。

建议显式传入训练时完全相同的 `STATS_PATH`。程序先通过训练数据 pipeline 得到归一化 action，并在同一归一化空间计算主 MSE；随后使用每个底层数据集实际 hydrate 后的 action stats，把预测和 GT 同时反归一化，再计算原始 action/delta 单位的 MSE。

## 抽样定义

默认 `FRACTION=0.10`。抽样单位是一个 frame 对应的 action chunk，而不是整个 episode。样本数是：

```text
round(所有数据集总 frame 数 × 0.10)
```

配额按各数据集 frame 数分层分配，因此大、小数据集都按相同比例参与；`SEED=42` 时结果可复现。具体抽中的 `(dataset_index, local_index)` 保存在 `diagnostic_split.json`。

快速冒烟测试可以限制数量：

```bash
MAX_SAMPLES=100 BATCH_SIZE=1 NUM_WORKERS=0 ... bash evaluation/WSA_Memory_Diagnostic/run.sh
```

`MAX_SAMPLES` 只建议调试使用；正式诊断保持 `0`，才会跑完选中的 10%。

## 输出

默认写到 `outputs/wsa_memory_diagnostic/<时间戳>/`：

```text
summary.json
per_sample_metrics.csv
diagnostic_split.json
diagnostic_repo_ids.txt
horizon_heatmap.png
horizon_heatmap_raw.png
action_curves/
```

- `summary.json`：总体、逐数据集、逐 horizon、逐 action dimension 的归一化和反归一化 MSE；
- `per_sample_metrics.csv`：每个被抽中 action chunk 的 MSE；
- `horizon_heatmap.png`：横轴 action dimension，纵轴 chunk horizon，颜色是归一化 MSE；
- `horizon_heatmap_raw.png`：相同布局，但使用反归一化后的 action/delta 单位；
- `action_curves/`：若干诊断样本的预测与 GT action 曲线，默认 8 个。

可以通过 `OUTPUT_DIR`、`NUM_CURVE_SAMPLES`、`INFERENCE_STEPS`、`DTYPE` 等环境变量覆盖默认值。

`ACTION_MODE` 通常不需要设置，程序直接读取 checkpoint 的训练配置。如果显式设置的值与训练配置不同，程序会拒绝运行，避免把 abs/delta transform 与错误的归一化文件混用。

## 如何理解结果

WSA-Memory 的 action 是 flow matching 采样结果，不是确定性回归头。这里固定随机种子以便对比 checkpoint，但单次采样 MSE 仍会包含采样方差。

另外，这 10% 来自已经参与训练的数据，所以它衡量的是模型对训练分布的拟合和 horizon 退化情况，不能代替真机成功率，也不是严格的泛化验证集。以后如果需要可信的离线验证曲线，应在训练前按 episode 划分 train/validation，并保证 validation episode 从未参与训练。
