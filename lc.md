## WSA-B AgileX

```bash
bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mot

cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p

POLICY_INIT_PATH=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/UPLOAD/TBot-SA1-Base \
ROBOTWIN_ROOT=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/WorldArena2/pour_water \
USE_EXTERNAL_STATS=true \
DATASET_EXTERNAL_STATS_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/agilex-single/pour_water.json \
ACTION_TYPE=delta \
CHUNK_SIZE=50 \
ENABLE_3D_QUERIES=true \
LAMBDA_3D=0.01 \
LAMBDA_GEN=0.01 \
PROC_PER_NODE=8 \
BATCH_SIZE=16 \
STEPS=40000 \
SAVE_FREQ=10000 \
JOB_NAME=wsa_base-pour_water \
bash launch/wsa_base_finetune_multi.sh
'
```

```bash
bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mot

cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p

POLICY_INIT_PATH=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/UPLOAD/TBot-SA1-Base \
ROBOTWIN_ROOT=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/WorldArena2/clean_table_instruction_follow \
USE_EXTERNAL_STATS=true \
DATASET_EXTERNAL_STATS_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/agilex-single/clean_table_instruction_follow.json \
ACTION_TYPE=delta \
CHUNK_SIZE=50 \
ENABLE_3D_QUERIES=true \
LAMBDA_3D=0.01 \
LAMBDA_GEN=0.01 \
PROC_PER_NODE=8 \
BATCH_SIZE=16 \
STEPS=30000 \
SAVE_FREQ=40000 \
JOB_NAME=wsa_base-clean-table-instructionfollow \
bash launch/wsa_base_finetune_multi.sh
'
```