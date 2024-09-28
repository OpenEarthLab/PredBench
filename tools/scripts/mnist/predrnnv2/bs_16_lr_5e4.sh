#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM%20000))
export PYTHONPATH="$\{PYTHONPATH\}:$(pwd)"

NUM_NODE=${1:-1}
NUM_GPU=${2:-4}

python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --master_port=${MASTER_PORT} train.py \
	--config configs/mnist/predrnnv2/bs_16_lr_5e4.py \
	--work-dir work_dirs/mnist/predrnnv2/bs_16_lr_5e4 \
	--launcher pytorch

python test.py \
	--work-dir work_dirs/mnist/predrnnv2/bs_16_lr_5e4 \
	--test-best --metric-only