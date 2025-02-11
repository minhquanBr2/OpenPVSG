#!/usr/bin/env bash
set -x

PARTITION=batch
JOB_NAME=tracking
NODELIST=phoenix3
CONFIG=configs/unitrack/imagenet_resnet50_s3_womotion_timecycle.py
CHECKPOINT=work_dirs/mask2former_r50_ips/epoch_8.pth
SPLIT=val
GPU_ID=1
WORK_DIR=work_dirs/ips_${SPLIT}_save_qf
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-3}
PY_ARGS=${@:5}

PYTHONPATH="/mnt/lustre/jkyang/CVPR23/openpvsg":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --nodelist=${NODELIST} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python -u tools/prepare_query_tube_ips.py ${CONFIG} ${CHECKPOINT} \
#     --work-dir ${WORK_DIR} --split ${SPLIT} --launcher="none" ${PY_ARGS}
python -u tools/prepare_query_tube_ips.py ${CONFIG} ${CHECKPOINT} \
    --work-dir ${WORK_DIR} --split ${SPLIT} --gpu-id ${GPU_ID} --launcher="none" ${PY_ARGS}
