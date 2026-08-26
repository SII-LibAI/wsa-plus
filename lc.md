## WSA-B AgileX

```bash
bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mot

cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p

POLICY_INIT_PATH=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/UPLOAD/TBot-SA1-Base \
ROBOTWIN_ROOT=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/WorldArena2/clean_table \
USE_EXTERNAL_STATS=true \
DATASET_EXTERNAL_STATS_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/agilex-single/clean_table.json \
ACTION_TYPE=delta \
CHUNK_SIZE=50 \
ENABLE_3D_QUERIES=true \
LAMBDA_3D=0.01 \
LAMBDA_GEN=0.01 \
PROC_PER_NODE=2 \
BATCH_SIZE=1 \
STEPS=50000 \
SAVE_FREQ=10000 \
JOB_NAME=wsa_base-agilex-clean-table \
bash launch/wsa_base_finetune_multi.sh
'
```
