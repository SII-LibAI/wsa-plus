# WSA 离线动作诊断

本目录对 WSA-Base 或 WSA-Memory checkpoint 自动选择对应的训练数据 pipeline，
从多个 LeRobot v3 数据集中按固定种子抽样 action chunk，并比较预测与 GT 的 MSE。
这是训练分布拟合诊断，不等同于真机 rollout 成功率。

## 启动

在仓库根目录执行：

```bash
CHECKPOINT=/path/to/checkpoint \
DATASET_ROOT=/path/to/lerobot_v3_collection \
STATS_PATH=/path/to/training_stats.json \
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=1 \
NUM_WORKERS=4 \
bash evaluation/wsa_m/run.sh
```

`CHECKPOINT` 可指向 `pretrained_model`、step checkpoint、完整训练 run，或 Hugging Face id。
`DATASET_ROOT` 是多个 LeRobot v3 数据集的父目录；程序会递归发现
`meta/info.json`，不要传单个 parquet 文件。

checkpoint 类型由 `config.json` 自动识别：

- WSA-Base：复用两帧视觉、任务文本、归一化和 action pipeline；
- WSA-Memory：复用 K 帧历史、history mask、文本 memory 和 action mask；
- Base subtask 与 Memory 文本 dropout 在评测时关闭，输入由保存的
  `train_config.json` 决定且可复现。

## 常用变量

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `FRACTION` | `0.10` | 每个底层数据集按比例分层抽样 |
| `MAX_SAMPLES` | `0` | 调试上限；`0` 表示不限制 |
| `BATCH_SIZE` | `2` | 推理 batch size |
| `NUM_WORKERS` | `4` | DataLoader worker 数 |
| `INFERENCE_STEPS` | checkpoint 值 | 可选 flow 推理步数覆盖 |
| `ACTION_MODE` | train config 值 | train config 缺失时必须显式设为 `abs` 或 `delta` |
| `DEVICE` | `cuda` | 推理设备 |
| `DTYPE` | `bfloat16` | `float32`、`float16` 或 `bfloat16` |

模型路径失效时可覆盖：

```bash
QWEN3_VL_PATH=/path/to/Qwen3-VL \
PROCESSOR_PATH=/path/to/Qwen3-VL \
COSMOS_TOKENIZER_PATH=/path/to/Cosmos-Tokenizer \
CHECKPOINT=/path/to/checkpoint \
DATASET_ROOT=/path/to/datasets \
bash evaluation/wsa_m/run.sh
```

stats 解析优先级为：显式 `STATS_PATH`、显式 `STATS_ROOT`、train config、
checkpoint 内 `stats.json`、各数据集自己的 `meta/stats.json`。建议始终传入训练时
使用的同一份 stats；显式 `ACTION_MODE` 与 train config 不一致时程序会拒绝运行。

## 输出

默认写入 `outputs/wsa_diagnostic/<时间戳>/`：

```text
summary.json
per_sample_metrics.csv
diagnostic_split.json
diagnostic_repo_ids.txt
horizon_heatmap.png
horizon_heatmap_raw.png
action_curves/
```

`summary.json` 同时记录 checkpoint policy type、归一化空间和原始动作空间的总体及
逐数据集 MSE。抽样记录保存在 `diagnostic_split.json`，相同 seed 可复现。
