# WSA-Memory 接入 WorldArena 2.0 Track 3

本目录实现 WorldArena 2.0 Track 3 的模型侧 A 端策略，分别支持：

- AgileX：双臂，14 维绝对关节动作；
- Franka：左单臂，8 维基坐标系绝对末端位姿动作；
- WSA-Memory 的 `task_only` 文本模式；
- 官方 `wa-hub-v1` HTTPS 长轮询和本地 `wa-policy-v1` WebSocket；
- episode 级视觉/状态历史缓存、归一化、模型推理、动作反归一化和 delta 恢复。

实现严格以官方 `policy_guide.md` 和 `track3_description_cn.md` 为准。adapter 不做
字段别名、形状修复、图像转置或四元数顺序兼容；输入不符合文档时立即抛出异常，
避免错误观测继续进入模型或真机控制链路。

网络、canonical packet 解码、Hub 注册和重试都交给官方
`real_world_benchmark`。本目录只实现官方要求的 `Policy.__init__()`、
`Policy.reset()` 和 `Policy.infer(new_obs)`。

## 1. 文件

```text
evaluation/worldarena/
├── policy.py               # 官方 loader 直接加载的入口
├── wsa_memory_adapter.py   # WSA-Memory 输入、历史、推理和动作适配
├── serve.sh                # Hub / WebSocket 通用启动器
├── start_agilex.sh         # AgileX 默认配置
├── start_franka.sh         # Franka 默认配置
└── README.md
```

官方资料：

- [WorldArena-2.0 仓库](https://github.com/WorldArena2/WorldArena-2.0)
- [Track 3 中文说明](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/track3_description_cn.md)
- [外部策略接入指南](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/policy_guide_cn.md)
- [A 端标准协议](https://github.com/WorldArena2/WorldArena-2.0/blob/main/real_world_benchmark/docs/policy_a_standard_protocol.md)

## 2. 输入输出

### task_only 文本

每次推理直接读取：

```python
task = new_obs["prompt"]
```

它来自官方 `ObservationPacket.context.task_instruction`。不会读取 sidecar，
不会拼接 scene、completed/current subtask、hand 或 rigidity。加载时会检查
checkpoint 的 `text_memory_mode` 必须为 `task_only`，防止拿错模型。

### AgileX

输入：

- 图像：只读取 `images.cam_high`、`images.cam_wrist_left`、
  `images.cam_wrist_right`，三者必须是非空 `uint8[H,W,3]` RGB；
- 状态：只读取 `joint_qpos`，必须是数值型且精确形状为 `(14,)`；
- 输出：`float32[chunk, 14]`，布局为
  `[left j1..j6, left gripper, right j1..j6, right gripper]`；
- metadata：`action_format=joint_absolute`。

`state`、`left_arm_joint_state`、`right_arm_joint_state`、`joint_qpos_left`、
`joint_qpos_right` 都不会被读取。当前 WSA-Memory 适配器也不消费 `tactile`；字段
存在时直接忽略。

### Franka

输入：

- 图像：只读取 `images.cam_high` 和 `images.cam_left_wrist`，均必须是非空
  `uint8[H,W,3]` RGB；文档中的重复字段 `cam_wrist` 不作为兜底；WSA 内部第三个
  相机槽使用白图并把 camera mask 置为 false；
- 状态：只读取精确 `(7,)` 的 `left_end_pose` 和精确 `(8,)` 的 `joint_qpos`，
  使用 `joint_qpos[-1]` 作为 gripper，组成 8 维；
- 输入和输出固定为 `[x,y,z,qw,qx,qy,qz,gripper]`（wxyz）；
- 输出：`float32[chunk, 8]`、`action_format=end_pose_base`、
  `control_arm=left`；返回前对四元数单位化。

不会读取 `joint_qpos_left`、`left_arm_joint_state`、`end_pose` 或 `state`。如果
`joint_qpos` 不是官方指南规定的 8 维，代码会立即报错。

### 严格失败规则

- 缺少 `prompt` 或不是非空字符串：报错；不回退到 `task`；
- 缺少上述必需状态/相机键：报错；
- 状态维度不精确、包含 NaN/Inf：报错；
- 图像不是 `uint8 HWC RGB`：报错；不自动接受 CHW、batch 维或 float 图像；
- observation 中额外出现的字段可以存在，但不会参与输入构造。

## 3. 模型和统计文件前提

两个 checkpoint 都必须满足：

1. policy type 是 `wsa_memory`；
2. `text_memory_mode=task_only`；
3. AgileX 训练数据的前 14 个有效 action 维是上述 qpos 语义；
4. Franka 训练数据的前 8 个有效 action 维确实是 `end_pose_base`，不能把现有
   constants 中关节空间 `franka` 的 8 维模型冒充末端模型；checkpoint 内部
   `output_features` 因 WSA padding 记为 32 维是正常的，adapter 只返回平台有效维；
5. 使用训练时同一份 stats JSON 和相同 normalization mode；
6. Franka 建议用 `ACTION_TYPE=abs`。AgileX 启动器默认按当前训练方式使用
   `ACTION_TYPE=delta`，推理后会恢复为绝对 qpos。

Franka 被选中的 state/action stats 各自必须合计为 8 维。官方原始 HDF5 的
`observations/end_pose` 可能含双臂兼容 padding；应在数据转换/训练前截成有效
8 维并据此计算统计，不能把 16 维 padded stats 直接交给本适配器。

统计文件可为平铺 feature stats，也可按 embodiment 嵌套。嵌套多个本体时设置：

```bash
export WSA_STATS_KEY=cobot_magic_max       # 示例，必须与 JSON 实际 key 一致
```

AgileX 会自动尝试 `observation.state` 和 `action`。为防止把 Franka 的
8 维关节统计误当成 8 维 end-pose 统计，Franka 只自动接受名称中明确包含
end-pose 的统计键。如果训练统计使用平铺键，必须显式确认语义后设置：

```bash
export WSA_STATE_STATS_KEYS=observation.state
export WSA_ACTION_STATS_KEYS=action
```

多个原始字段拼成一个模型向量时，用逗号按训练时 ComposeFields 的顺序填写。

## 4. 安装官方 A 端

```bash
git clone https://github.com/WorldArena2/WorldArena-2.0.git
export WORLDARENA_ROOT=/absolute/path/WorldArena-2.0

pip install -e "${WORLDARENA_ROOT}/real_world_benchmark"
pip install -r "${WORLDARENA_ROOT}/real_world_benchmark/requirements-a.txt"
```

使用训练 WSA-Memory 的同一个 Python/conda 环境启动，确保能导入本仓库
`src/lerobot`、PyTorch、Transformers 和模型依赖。

## 5. 正式 Hub 启动

主办方会提供 `HUB_POLICY_URL`、精确的 worker key，以及可能存在的 token。
`POLICY_ID` 必须与 B 端配置逐字一致。

### AgileX worker

```bash
cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p

export WSA_CHECKPOINT=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/wsa_memory/agilex-delta-task_only/checkpoints/180000
export WSA_STATS_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/agilexa_delta_gripper_abs.json
export WSA_STATS_KEY=cobot_magic_max 
export WSA_QWEN3_VL_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Qwen3-VL-2B-Instruct
export WSA_COSMOS_TOKENIZER_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Cosmos-Tokenizer-CI8x8
export WSA_STATE_STATS_KEYS=observation.state
export WSA_ACTION_STATS_KEYS=action
export HUB_POLICY_URL=https://siu9ss7j9igdlaldmmk5u.apigateway-cn-beijing.volceapi.com/policy
export POLICY_ID=WSA_agilex_vision
# export HUB_TOKEN=organizer_token     # 仅在主办方要求时设置
CUDA_VISIBLE_DEVICES=0 bash evaluation/worldarena/start_agilex.sh
```

### Franka worker

```bash
cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p

export WSA_CHECKPOINT=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/wsa_memory/franka_abs/checkpoints/110000
export WSA_STATS_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/franka_abs.json
export WSA_STATS_KEY=wr_franka      # 按 JSON 实际 key 修改；平铺时删除
export WSA_STATE_STATS_KEYS=observation.state.endpose 
export WSA_ACTION_STATS_KEYS=action.endpose              
export WSA_QWEN3_VL_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Qwen3-VL-2B-Instruct
export WSA_COSMOS_TOKENIZER_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Cosmos-Tokenizer-CI8x8

export HUB_POLICY_URL=https://ss39jbpvj6k40qpvva4kn.apigateway-cn-beijing.volceapi.com/policy
export POLICY_ID=WSA_Franka
# export HUB_TOKEN=organizer_token     # 仅在主办方要求时设置

CUDA_VISIBLE_DEVICES=0 bash evaluation/worldarena/start_franka.sh
```

两份模型可在两张 GPU 上分别启动为两个常驻 worker；不要让两个进程使用同一个
`POLICY_ID`。

## 6. 本地 WebSocket 与 dry-run

本地直连不需要 Hub：

```bash
WORLDARENA_TRANSPORT=ws \
WORLDARENA_HOST=0.0.0.0 \
WORLDARENA_PORT=8000 \
bash evaluation/worldarena/start_agilex.sh
```

不加载模型的接口冒烟可加 `WSA_DRY_RUN=1`。dry-run 仍会完整校验官方文本、状态
和相机 schema，并返回当前 qpos/end-pose 的 hold chunk，不会向绝对控制器发送
危险的全零动作。正式评测必须删除该变量。

## 7. WSA-Memory 历史和 chunk

官方 worker 当前默认只把本次图像放入 `new_obs`。adapter 因此在每个 episode
内部维护自己的 RGB 图像和归一化状态历史；`reset()` 或 prompt 改变时清空。

启动器默认返回 25 步：

```bash
export WSA_EXECUTE_CHUNK_SIZE=25
```

该值不得大于 checkpoint 的 `chunk_size`。值越大，网络/模型调用越少，但开环时间
更长。历史间隔按以下近似换算成 infer 次数：

```text
round(platform_fps * checkpoint.history_stride_seconds / execute_chunk_size)
```

其中 `platform_fps` 由官方平台固定：AgileX 为 30，Franka 为 15，不需要在启动
脚本中配置。Franka 的 `control_arm` 也固定为 `left`，由 adapter 直接写入 metadata。

若现场没有完整执行每个 chunk，必须按实际调用周期显式设置：

```bash
export WSA_HISTORY_STRIDE_CALLS=1
```

## 8. Franka 四元数

本实现固定遵循官方格式：`[x,y,z,qw,qx,qy,qz,gripper]`。没有 xyzw 配置、
自动探测或重排逻辑。若现场 observation 不符合该约定，应修复上游官方接入，
不能让 adapter 猜测顺序。

## 9. 常用配置

| 变量 | 含义 |
|---|---|
| `WSA_CHECKPOINT` | checkpoint、step/run 目录或 Hugging Face id |
| `WSA_STATS_PATH` | 训练时使用的聚合 stats JSON；未给时尝试 train config/邻近 stats |
| `WSA_STATS_KEY` | 嵌套 JSON 的本体 key |
| `WSA_STATE_STATS_KEYS` | 状态统计字段，逗号分隔且顺序严格一致 |
| `WSA_ACTION_STATS_KEYS` | 动作统计字段，逗号分隔且顺序严格一致 |
| `WSA_ACTION_MODE` | `delta` 或 `abs`，必须与训练一致 |
| `WSA_NORMALIZATION_MODE` | `mean_std` 或 `min_max`，必须与训练一致 |
| `WSA_QWEN3_VL_PATH` | 部署机上的 Qwen3-VL 模型路径，同时优先作为 processor 路径 |
| `WSA_PROCESSOR_PATH` | 单独覆盖 Qwen processor 路径 |
| `WSA_COSMOS_TOKENIZER_PATH` | 部署机上的 Cosmos tokenizer 路径 |
| `WSA_DEVICE` | 模型推理设备，默认 `cuda:0` |
| `WSA_COSMOS_DEVICE` | Cosmos 设备，默认跟随 `WSA_DEVICE` |
| `WSA_DTYPE` | `bfloat16` 或 `float32`；与 Cosmos 和 WSA 配置同步 |
| `WSA_NUM_INFERENCE_STEPS` | flow matching 推理步数覆盖值 |
| `WSA_EXECUTE_CHUNK_SIZE` | 每次返回并由官方依次执行的动作数 |
| `WSA_HISTORY_STRIDE_CALLS` | 覆盖历史帧间隔的 infer 调用数 |
| `WSA_DELTA_MASK` | 自定义 delta 维 mask，如 `1,1,...,0` |
| `WSA_WORLDARENA_CONFIG` | 可选 JSON 配置文件；同名环境变量优先 |

推理强制 `lambda_3d=0`，不会加载 DA3 teacher。Cosmos 仍是 WSA 模型视觉路径的
组成部分，不能因为关闭 3D 对齐而删除。

## 10. 上真机前检查

1. 先用 `WSA_DRY_RUN=1` 和本地 WebSocket 验证官方 loader、reset、infer；
2. 用一条主办方样例 observation 检查首次日志中的 image/state shape；
3. 确认 AgileX `joint_qpos.shape == (14,)`，最终输出为 `(chunk,14)`；
4. 确认 Franka `left_end_pose.shape == (7,)`、`joint_qpos.shape == (8,)`，且
   四元数顺序为 wxyz；
5. 确认 Franka checkpoint 的前 8 个有效维是 absolute end-pose，不是 8D joint；
6. 确认 stats key、字段顺序、normalization mode、delta mask 与训练完全一致；
7. 在隔离环境、低速和短 chunk 下做第一次真机动作测试，再逐步增加 chunk。
