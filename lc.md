bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mot

cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p

POLICY=wsa_memory \
POLICY_INIT_PATH=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/UPLOAD/TBot-SA1-Base \
ENABLE_3D_QUERIES=false \
ROBOTWIN_ROOT=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/WorldArena2 \
USE_EXTERNAL_STATS=true \
DATASET_EXTERNAL_STATS_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/norm/agilex.json \
PROC_PER_NODE=8 \
HISTORY_NUM_FRAMES=6 \
HISTORY_STRIDE_SECONDS=1.0 \
TEXT_MEMORY_MODE=oracle \
STEPS=140000 \
BATCH_SIZE=10 \
bash launch/wsa_base_finetune_multi.sh
'