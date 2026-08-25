# WSA 训练启动

WSA-Base 多数据集微调使用 `wsa_base_finetune_multi.sh`。需要加入子任务文本时，使用独立的完整脚本 `wsa_base_fintune_subtask_multi.sh`：

```bash
POLICY_INIT_PATH=/path/to/WSA-Base \
ROBOTWIN_ROOT=/path/to/dataset_collection \
DATASET_EXTERNAL_STATS_PATH=/path/to/stats.json \
PROC_PER_NODE=8 \
BATCH_SIZE=10 \
bash launch/wsa_base_fintune_subtask_multi.sh
```

`ROBOTWIN_ROOT` 下的每个 LeRobot v3 数据集需要 `meta/sidecar.json`。Sidecar 只保留完整指令和子任务区间：

```json
[
  {
    "episode_id": "episode_000000",
    "episode_index": 0,
    "task_instruction": "wipe the table",
    "subtasks": [
      {
        "subtask_id": 0,
        "subtask_text": "pick up the towel",
        "start_frame": 0,
        "end_frame": 90
      }
    ]
  }
]
```

只有两种文本模式：

- `TEXT_CONTEXT_MODE=subtask`：`Task instruction + Completed subtasks + Current subtask`。
- `TEXT_CONTEXT_MODE=task_only`：原始数据集 `task`，不读 sidecar。

`subtask` 模式的三部分可独立 dropout：

```bash
TASK_INSTRUCTION_DROPOUT=0.0
COMPLETED_MEMORY_DROPOUT=0.1
CURRENT_SUBTASK_BLOCK_DROPOUT=0.2
```

第一个训练 batch 会在 rank 0 日志中输出一条实际指令。
