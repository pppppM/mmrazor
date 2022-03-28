#!/usr/bin/env bash

set -x

PARTITION="mm_model"
JOB_NAME="greedynas_verify"
CONFIG="configs/nas/greedynas/greedynas_subnet_mobilenet_8xb96_in1k.py"
CKPT="ckpt/best_new_ema.pth.tar"
SUBNET_PATH="configs/nas/greedynas/final_subnet_op_paper.yaml"
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
srun -p ${PARTITION} \
    --quotatype=auto \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u $(dirname "$0")/test_mmcls.py ${CONFIG} ${CKPT} \
    --metrics "accuracy" \
    --cfg-options algorithm.mutable_cfg=$SUBNET_PATH dist_params.port=12345 \
    --launcher="slurm" ${PY_ARGS}
