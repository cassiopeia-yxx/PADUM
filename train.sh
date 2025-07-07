#!/usr/bin/env bash

#CONFIG=$1

export NCCL_P2P_DISABLE=1

python setup.py develop --no_cuda_ext

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 basicsr/train.py -opt Options/Deraining.yml --launcher pytorch
