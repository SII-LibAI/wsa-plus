# Launch Scripts

Current WSA Base training entrypoints live directly in this directory:

- `wsa_base_pretrain.sh`
- `wsa_base_finetune.sh`
- `wsa_base_finetune_robotwin.sh`
- `wsa_base_finetune_libero.sh`

WSA Large training entrypoints live directly in this directory:

- `wsa_large_pretrain.sh`
- `wsa_large_finetune_robotwin.sh`
- `wsa_large_finetune_libero.sh`
- `wsa_large_finetune_real_piper.sh`
- `wsa_large_finetune_real_lift2.sh`

WSA Memory reuses the WSA Base entrypoints by setting `POLICY=wsa_memory`:

- `wsa_base_pretrain.sh`
- `wsa_base_finetune.sh`

Supported comparison-method RoboTwin finetuning scripts live in `supported_methods/`.
Normalization-stat helpers and utilities live in `../tools/`.

##
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mot 
cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p

POLICY=wsa_memory \
POLICY_INIT_PATH=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/UPLOAD/TBot-SA1-Base \
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