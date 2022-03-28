#!/usr/bin/env bash

set -x

PARTITION=$"mm_model"
JOB_NAME=$"greedynas_retrain"
CONFIG="configs/nas/greedynas/greedynas_subnet_mobilenet_8xb96_in1k.py"
WORK_DIR="../experiments/greedynas_subnet"
SUBNET_PATH="configs/nas/greedynas/final_subnet_op_paper.yaml"
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
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
    python -u $(dirname "$0")/train_mmcls.py ${CONFIG} --work-dir=${WORK_DIR} \
    --cfg-options algorithm.mutable_cfg=$SUBNET_PATH \
    --launcher="slurm" ${PY_ARGS}
