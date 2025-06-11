#!/bin/bash
module  purge 
source  /home/bingxing2/apps/package/pytorch/2.4.0+cu121_cp310/env.sh
source  activate    setup
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/ailab/wangkuncan/.conda/envs/setup/lib/python3.10/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
