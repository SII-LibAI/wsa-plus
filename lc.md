bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mot

cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p

POLICY=wsa_memory \
POLICY_INIT_PATH=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/UPLOAD/TBot-SA1-Base \
ENABLE_3D_QUERIES=false \
ROBOTWIN_ROOT=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/WorldArena2 \
USE_EXTERNAL_STATS=true \
DATASET_EXTERNAL_STATS_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/agilex_abs.json \
PROC_PER_NODE=8 \
HISTORY_NUM_FRAMES=6 \
HISTORY_STRIDE_SECONDS=1.0 \
TEXT_MEMORY_MODE=task_only \
ACTION_TYPE=abs \
STEPS=150000 \
BATCH_SIZE=10 \
bash launch/wsa_base_finetune_multi.sh
'
# Stats
python tools/compute_norm_stats_multi.py \
  --repo_id_file /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/wsa_memory/_repo_id_files/wsa_memory-robotwin-delta-chunk50-pretrained-default-gen0.01-3d0.0-finetune-2026_08_15_09_38_43.txt \
  --action_mode delta \
  --chunk_size 50 \
  --num_workers 16 \
  --output_path /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/agilexa_delta_gripper_abs.json
# Franka
bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mot

cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p

POLICY=wsa_memory \
POLICY_INIT_PATH=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/UPLOAD/TBot-SA1-Base \
ENABLE_3D_QUERIES=false \
ROBOTWIN_ROOT=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Wr2_franka_lerobotv3 \
USE_EXTERNAL_STATS=true \
DATASET_EXTERNAL_STATS_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/franka_abs.json \
PROC_PER_NODE=8 \
HISTORY_NUM_FRAMES=6 \
HISTORY_STRIDE_SECONDS=1.0 \
TEXT_MEMORY_MODE=task_only \
ACTION_TYPE=abs \
STEPS=110000 \
BATCH_SIZE=10 \
bash launch/wsa_base_finetune_multi.sh
'