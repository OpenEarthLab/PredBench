#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM%20000))
export PYTHONPATH="$\{PYTHONPATH\}:$(pwd)"

NUM_NODE=${1:-1}
NUM_GPU=${2:-4}

python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --master_port=${MASTER_PORT} train.py \
	--config configs/kitti/convlstm/bs_16_lr_1e3.py \
	--work-dir work_dirs/kitti/convlstm/bs_16_lr_1e3 \
	--launcher pytorch

python test.py \
	--work-dir work_dirs/kitti/convlstm/bs_16_lr_1e3 \
	--test-best --metric-only
