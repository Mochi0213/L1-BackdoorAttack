#!/bin/bash
#SBATCH -J wangkuncan    # 作业名
#SBATCH -o logs/%x-%j.log   # stdout输出日志文件，%x是作业名，%j是job ID
#SBATCH -e logs/%x-%j.log   # stderr输出文件
#SBATCH -p vip_gpu_ailab   # 使用分区
#SBATCH -A ai4phys                # 使用的账户
#SBATCH --gres=gpu:4       #使用的显卡数量
#SBATCH --output=../../outputs/output_%j.log    # 输出文件 (%j 会被作业ID替代)
#SBATCH --error=../../errors/error_%j.log      # 错误

set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=$(pwd)/../../verl/
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/ailab/wangkuncan/.conda/envs/L1/lib/python3.10/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
python3 -m verl.trainer.main_ppo   --config-path="/home/bingxing2/ailab/wangkuncan/soft/l1/scripts/train" \
                                   --config-name="l1_exact_lora.yaml"
