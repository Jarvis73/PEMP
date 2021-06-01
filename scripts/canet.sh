#!/usr/bin/env bash

COMMAND=$1
GPU_ID=$2
shift 2

PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
BASE_NAME=$(basename $0)

ARGS=' with
  tag='${BASE_NAME%".sh"}' split=0
  tr.total_epochs=200 tr.lr=0.0025
  data.train_n=10000 data.height=321 data.width=321
  '$@

if [[ "$COMMAND" == "help" ]]; then
  ARGS="$GPU_ID"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${PROJECT_DIR}" \
  "$CONDA_PREFIX"/bin/python ./entry/canet.py ${COMMAND} ${ARGS}
