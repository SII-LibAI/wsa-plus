# WSA-Memory 中文说明

`wsa_memory` 是独立的 LeRobot V3 policy。它从 WSA-Base checkpoint 初始化，
在原 action policy 上增加多帧视觉、多帧机器人 state 和 sidecar 文本 memory。
唯一训练目标是 masked action flow-matching loss。

本实现不修改 `src/lerobot/policies/WSA_Base/**`，也没有新增归一化算法或数据格式。
归一化、Accelerate、checkpoint、optimizer、scheduler 和多数据集加载继续使用
WSA 仓库原有接口。

## 1. 注册和标准接口

- Policy type：`wsa_memory`
- Dataset type：`wsa_memory`
- 配置：`WSAMemoryConfig`、`WSAMemoryDatasetConfig`
- Policy：`WSAMemoryPolicy`
- Model：`WSAMemoryModel`
- 训练入口：`src/lerobot/scripts/lerobot_train.py`
- 多数据集入口：`launch/wsa_base_pretrain.sh`
- 单数据集入口：`launch/wsa_base_finetune.sh`
- 单数据集统计：`tools/compute_norm_stats_single.py`
- 多数据集统计：`tools/compute_norm_stats_multi.py`
- 已有的自动发现/分组统计入口：`tools/wsa_large_compute_pretrain_norm_stats.sh`

训练时仍然使用标准参数：

```text
--policy.type=wsa_memory
--dataset.type=wsa_memory
--dataset.repo_id=...
--dataset.repo_id_file=...
--dataset.action_mode=abs|delta
--dataset.use_external_stats=true|false
--dataset.external_stats_path=...
--dataset.external_stats_root=...
```

保存后的 WSA-Memory checkpoint 使用 LeRobot 原生 `save_pretrained` / 
`from_pretrained`，不需要再次加载 WSA-Base 初始化权重。

## 2. Dataloader 接入

Dataset factory 仍先根据 policy 的 delta indices 构建标准 `LeRobotDataset`。
WSA-Memory 只增加两处必要行为：

1. 根据每个 repo 的真实 `fps`，给图像和 state 请求相同的 K 个历史时间点。
2. Dataset 创建后，为该 repo 绑定一份 memory transform，其中 sidecar 只加载一次。

原始 sample 使用仓库中的真实字段：

```text
episode_index
frame_index
task
<raw image/state/action feature keys>
<raw feature key>_is_pad
```

经过原有 remap、normalize、compose、pad 和 Qwen processor 后，policy 接收：

```text
observation.state             [B, K, D_state]
observation.images.image*     [B, K, C, H, W]
action                        [B, chunk_size, D_action]
memory.history_valid_mask     [B, K]
memory.action_loss_mask       [B, chunk_size]
sample.action_loss_mask
observation.input_ids
observation.attention_mask
observation.pixel_values
observation.image_grid_thw
```

多数据集训练时，每个 repo 分别绑定自己的 root、fps、feature mapping、stats
和 sidecar index，不会用全局 `episode_index` 交叉查找 sidecar。

## 3. Memory 架构

### 3.1 视觉 memory

输入时间点为：

```text
[t-(K-1)Δ, ..., t-Δ, t]
Δ = history_stride_seconds
```

默认 `K=6`、`Δ=1.0s`。Δ 由每个数据集的真实 fps 转成 frame offset，
不使用 sidecar 中的 `sample_interval_sec`。

当前实现是需求允许的 Stage A 固定-token方案：Qwen vision 先逐帧提取空间
token，然后以当前帧 token 为 query，在相同空间位置对 K 帧做带 mask 的
temporal attention。最终只返回当前帧数量的视觉 token，不会把 K 倍 token
全部拼进语言 backbone。Temporal residual gate 初始化为 0，因此刚从 WSA-Base
初始化时尽量保持当前帧行为。

`temporal_attention_every_n_blocks` 会随配置保存，但当前没有把 Stage A 描述成
“每隔若干 ViT block 插入 attention”的 Stage B 实现。

### 3.2 State memory

K 个 state 分别复用 WSA-Base `state_proj`，加入当前偏移为 0 的相对时间编码，
然后作为 K 个 suffix token。Attention mask、position id、KV cache 和 denoise
路径均从运行时 K 计算。

Episode 开头历史不足时重复最早可用观测，同时对应 `history_valid_mask=false`；
当前帧始终是最后一个有效位置。Delta action 只减当前 state：

```python
current_state = state[:, -1]
delta_action = action - current_state
```

训练时 `visual_history_dropout` 只丢历史位置，永远保留当前帧；同一份 mask
同时用于视觉 memory、state memory 和 Cosmos 最近有效帧选择。

### 3.3 WSA-Base 复用范围

Qwen3-VL、Cosmos middle conditioning、3D messenger conditioning、action expert、
flow target、denoise loop 和 action projection 都继续复用。图像生成和 DA3
teacher 的输出/监督路径不执行；为兼容 WSA-Base checkpoint key 而保留的输出
模块会被冻结。

## 4. Sidecar 和文本 memory

每个数据集根目录需要：

```text
<dataset_root>/meta/sidecar.json
```

顶层必须是 episode 数组：

```json
[
  {
    "episode_id": "episode_000000",
    "episode_index": 0,
    "task_instruction": "Wipe the table and put the towel back.",
    "sample_interval_sec": 2.0,
    "episode_summary": "Not used in the prompt.",
    "subtasks": [
      {
        "subtask_id": 0,
        "subtask_text": "grasp the towel",
        "start_frame": 0,
        "end_frame": 120,
        "confidence": 0.98,
        "hand_used": 1,
        "object_rigidity": 1
      }
    ],
    "scene": "home"
  }
]
```

索引主键是整数 `episode_index`。Subtask 区间是 episode-local 闭区间；重复
episode/subtask、乱序、非法区间和默认不允许的重叠会在 dataset 构造时失败。
区间 gap 合法，但 current/arm/property 会变成 `unknown`。

默认枚举为：

```python
HAND_MAP = {0: "left", 1: "right", 2: "both"}
DEFAULT_RIGIDITY_MAP = {0: "rigid", 1: "flexible"}
```

二者都能由 policy config 覆盖。生成的 prompt 示例：

```text
Overall task: Grab the towel and then wipe the table and put the towel back.
Scene: home.
Completed subtasks: move the gripper close to the towel.
Current subtask: grasp the towel.
Interaction arm: right.
Object property: flexible.
```

`episode_summary`、confidence、帧边界和数字 ID 不进入 prompt。Tokenizer 超长时
先删除最早的 completed subtasks，不会静默截断 current/arm/property block。

文本模式：

- `oracle`：训练/离线评估时由 sidecar 和 frame index 生成 GT memory。
- `task_only`：只使用 LeRobot 原始 task。
- `external`：部署时由外部状态机传入 completed/current/arm/property。

Oracle 模式还会把 action padding mask 与 subtask 结束边界合并，避免 action
chunk 跨到下一子任务。Task-only、external 或 unknown current subtask 只使用
LeRobot 原生 action padding mask。

## 5. Action-only 约束

```python
loss = loss_action
```

- `lambda_gen=0.0`
- `lambda_3d=0.0`
- 不创建 subtask、progress、scene、arm 或 rigidity 预测头
- 不构造 future image / DA3 teacher target
- 不计算 image generation 或 3D supervision loss
- 兼容 `sample.action_loss_mask`、action padding 和 subtask boundary mask

WSA-Memory 当前只支持 `attention_mask_mode=default`。WSA-Base causal/RTC 路径
假设只有一个 state token，因此请求 RTC 会明确抛出 `NotImplementedError`，
不会静默使用错误 shape。同步 cached denoising 已支持 K-state suffix。

## 6. 归一化：直接复用 WSA 原脚本

WSA-Memory 没有专用归一化逻辑。统计和训练必须使用相同的 `ACTION_TYPE` 与
`CHUNK_SIZE`；尤其 delta action 统计不能用 absolute action 的 `meta/stats.json`。

### 6.1 单数据集

```bash
python tools/compute_norm_stats_single.py \
  --repo_id /data/lerobot/my_dataset \
  --root /data/lerobot \
  --action_mode delta \
  --chunk_size 50 \
  --output_dir norm_stats
```

输出遵循原工具格式：

```text
norm_stats/delta/my_dataset/stats.json
```

### 6.2 新的多数据集集合

直接使用 WSA 已有的自动发现、按 `resolved_robot_type` 分组和统计脚本：

```bash
INTERNDATA_ROOT=/data/InternData-A1-v30 \
ROBOTWIN_ROOT=/data/RoboTwin-LeRobot-v30 \
ROBOCHALLENGE_ROOT=/data/Robochallengev3.0_eef \
AGIBOT_ROOT=/data/Agibotv3.0 \
EGODEX_LEROBOT_ROOT=/data/Egodex_v_taskrepos_v30 \
ACTION_TYPE=delta \
CHUNK_SIZE=50 \
OUTPUT_STATS_ROOT=norm_stats \
bash tools/wsa_large_compute_pretrain_norm_stats.sh
```

不用的数据根目录留空即可。脚本内部调用原有 `compute_norm_stats_multi.py`，
最终目录正是 dataset factory 原本读取的格式：

```text
norm_stats/<resolved_robot_type>/<action_mode>/stats.json
```

同一统计组必须具有相同的 resolved robot type、feature keys 和 shapes；这由
现有统计脚本检查。WSA-Memory 的视觉/state history 不改变 state/action
归一化算法，所以不需要额外的 memory stats。

## 7. 训练：复用原 WSA launch

### 7.1 多数据集训练

在原 `wsa_base_pretrain.sh` 上只需把 policy 切换为 `wsa_memory`：

```bash
POLICY=wsa_memory \
POLICY_INIT_PATH=/checkpoints/wsa_base/pretrained_model \
INTERNDATA_ROOT=/data/InternData-A1-v30 \
ROBOTWIN_ROOT=/data/RoboTwin-LeRobot-v30 \
ROBOCHALLENGE_ROOT=/data/Robochallengev3.0_eef \
AGIBOT_ROOT=/data/Agibotv3.0 \
EGODEX_LEROBOT_ROOT=/data/Egodex_v_taskrepos_v30 \
DATASET_EXTERNAL_STATS_ROOT=norm_stats \
HISTORY_NUM_FRAMES=6 \
HISTORY_STRIDE_SECONDS=1.0 \
TEXT_MEMORY_MODE=oracle \
bash launch/wsa_base_pretrain.sh
```

原脚本继续负责数据集发现、repo-id file、权重规则、分布式 repo 分配、
external stats、Accelerate、多机参数和输出目录。仅当 `POLICY=wsa_memory` 时，
它会把 `POLICY_INIT_PATH` 传给 `policy.init_from_wsa_base`，并强制 action-only
配置；默认 `POLICY=WSA_Base` 的行为保持不变。

### 7.2 单数据集训练

原 finetune 位置参数接口同样可用：

```bash
POLICY=wsa_memory \
POLICY_INIT_PATH=/checkpoints/wsa_base/pretrained_model \
DATASET_EXTERNAL_STATS_PATH=norm_stats/delta/my_dataset/stats.json \
HISTORY_NUM_FRAMES=6 \
HISTORY_STRIDE_SECONDS=1.0 \
TEXT_MEMORY_MODE=oracle \
bash launch/wsa_base_finetune.sh /data/lerobot/my_dataset delta true
```

### 7.3 常用 memory 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HISTORY_NUM_FRAMES` | `6` | 视觉/state history K |
| `HISTORY_STRIDE_SECONDS` | `1.0` | 由每个 repo 的 fps 转 frame offset |
| `VISUAL_HISTORY_DROPOUT` | `0.3` | 训练时历史位置 dropout |
| `TEXT_MEMORY_MODE` | `oracle` | `oracle`、`task_only` 或 `external` |
| `TOKENIZER_MAX_LENGTH` | `192` | 文本最大 token 数 |
| `MAX_COMPLETED_SUBTASKS` | `8` | 保留最近完成子任务数量 |
| `ALLOW_MISSING_SIDECAR` | `false` | 是否允许退化为 task-only |
| `MASK_ACTION_AFTER_SUBTASK_END` | `true` | 是否屏蔽跨 subtask action |

WSA-Base 初始化 checkpoint 的 Qwen/action variants、state/action 最大维度、
3D query 配置和 LoRA 配置必须与当前 launch 配置一致；非 memory 权重出现
shape mismatch 或未预期 missing/unexpected key 时会 fail fast。

## 8. 推理

Policy 为每个 env id 维护独立的视觉、Qwen pixels 和 state deque：

```python
policy.reset()
policy.reset(env_ids=[3, 7])

prompt = policy.set_text_memory(
    env_id=3,
    overall_task="Wipe the table and put the towel back",
    scene="home",
    completed_subtasks=["grasp the towel"],
    current_subtask="wipe the spill",
    hand_used=1,
    object_rigidity=1,
)

# 将 prompt 作为 sample["task"] 送入原 WSA processor。
action = policy.select_action(processed_batch, env_ids=[3])
```

`set_text_memory` 会清空该 env 尚未执行完的 action queue，避免继续执行旧文本
条件下的 action chunk。真实机器人建议使用 `external` 或 `task_only`；oracle
sidecar lookup 只用于离线 dataset/评测。

## 9. 轻量验证

```bash
pytest -q tests/policies/wsa_memory
python -m compileall -q src/lerobot/policies/WSA_Memory
bash -n launch/wsa_base_pretrain.sh
bash -n launch/wsa_base_finetune.sh
```

这些测试不加载真实 WSA 大模型或训练数据。
