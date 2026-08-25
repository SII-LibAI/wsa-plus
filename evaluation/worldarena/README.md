# WSA 接入 WorldArena 2.0 Track 3

该目录提供统一的 WorldArena A 端策略入口。启动时读取 checkpoint 的 `config.json`，自动选择 `WSA_Base` 或 `wsa_memory`，无需切换 `policy.py`。

## 输入约定

两类模型都只读取官方 observation 中的完整 `prompt`，评测端不读取 sidecar，也不拼接子任务文本，因此只接受 `task_only` checkpoint。

| checkpoint | 文本 | 视觉 | 状态 | Memory mask |
|---|---|---|---|---|
| `WSA_Base` / `wsa_base` | 完整任务指令；要求 `text_context_mode=task_only` | 上一帧、当前帧；首步复制当前帧 | 仅当前状态 | 无 |
| `wsa_memory` | 完整任务指令；要求 `text_memory_mode=task_only` | checkpoint 配置的 K 帧历史 | K 帧历史状态 | `memory.history_valid_mask` |

平台数据：

- AgileX：`joint_qpos_left(7) + joint_qpos_right(7)`；相机为 `cam_high`、`cam_left_wrist`、`cam_right_wrist`；输出绝对关节动作。
- Franka：`left_end_pose(7) + joint_qpos[-1]`；位姿固定为 `[x,y,z,qx,qy,qz,qw,gripper]`（`xyzw`）；相机为 `cam_high`、`cam_left_wrist`；输出 base frame 末端位姿。

缺少字段、维度错误、非 `uint8 HWC RGB` 图像或 NaN/Inf 会直接报错，不做字段别名或四元数顺序猜测。

## 启动

先安装 WorldArena 官方 A 端，并让当前 Python 环境可以导入本仓库的 `src/lerobot`。

```bash
export WORLDARENA_ROOT=/path/to/WorldArena-2.0
pip install -e "${WORLDARENA_ROOT}/real_world_benchmark"
pip install -r "${WORLDARENA_ROOT}/real_world_benchmark/requirements-a.txt"
```

公共配置示例：

```bash
export WSA_CHECKPOINT=/path/to/base-or-memory/checkpoint
export WSA_STATS_PATH=/path/to/training_stats.json
export WSA_QWEN3_VL_PATH=/path/to/Qwen3-VL-2B-Instruct
export WSA_COSMOS_TOKENIZER_PATH=/path/to/Cosmos-Tokenizer-CI8x8
export HUB_POLICY_URL=https://organizer.example/policy
export POLICY_ID=WSA_worker_id
```

按平台补充 stats 字段并启动：

```bash
# AgileX
export WSA_STATE_STATS_KEYS=observation.state
export WSA_ACTION_STATS_KEYS=action
CUDA_VISIBLE_DEVICES=0 bash evaluation/worldarena/start_agilex.sh

# Franka
export WSA_STATE_STATS_KEYS=observation.state.endpose
export WSA_ACTION_STATS_KEYS=action.endpose
CUDA_VISIBLE_DEVICES=0 bash evaluation/worldarena/start_franka.sh
```

`WSA_ACTION_MODE` 应与训练一致；若 checkpoint 附近的训练配置无法提供该值，必须显式设为 `abs` 或 `delta`。delta 模式默认按 `WSA_ROBOT_TYPE` 读取 mask，自定义数据布局时用 `WSA_DELTA_MASK=1,1,...,0` 显式覆盖。

Franka 只有在 state/action 都是同一个 8D end-pose 表示时才能使用 delta；此时需显式设置 `WSA_ALLOW_FRANKA_DELTA=1`。

常用可选项：`WSA_STATS_KEY`、`WSA_NORMALIZATION_MODE`、`WSA_PROCESSOR_PATH`、`WSA_EXECUTE_CHUNK_SIZE`、`WSA_DEVICE`、`WSA_DTYPE`。两类模型均可用 `WSA_HISTORY_STRIDE_CALLS` 覆盖自动换算的历史采样间隔。

本地 WebSocket：

```bash
WORLDARENA_TRANSPORT=ws WORLDARENA_HOST=0.0.0.0 WORLDARENA_PORT=8000 \
  bash evaluation/worldarena/start_agilex.sh
```

接口冒烟测试可加 `WSA_DRY_RUN=1`；它会校验输入并返回 hold action，不加载模型。正式评测必须移除该变量。
