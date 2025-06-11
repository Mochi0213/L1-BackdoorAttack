#!/bin/bash
#SBATCH -J wangkuncan    # 作业名
#SBATCH -o logs/%x-%j.log   # stdout输出日志文件，%x是作业名，%j是job ID
#SBATCH -e logs/%x-%j.log   # stderr输出文件
#SBATCH -p vip_gpu_ailab   # 使用分区
#SBATCH -A ai4phys                # 使用的账户
#SBATCH --gres=gpu:4       #使用的显卡数量
#SBATCH --output=../../outputs/output_%j.log    # 输出文件 (%j 会被作业ID替代)
#SBATCH --error=../../errors/error_%j.log      # 错误


MODEL_PATH=$1

/home/bingxing2/ailab/wangkuncan/soft/l1/scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 512 --datasets aime2025 math amc olympiad_bench
/home/bingxing2/ailab/wangkuncan/soft/l1/scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 1024 --datasets aime2025 math amc olympiad_bench
/home/bingxing2/ailab/wangkuncan/soft/l1/scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 2048 --datasets aime2025 math amc olympiad_bench
/home/bingxing2/ailab/wangkuncan/soft/l1/scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 3600 --datasets aime2025 math amc olympiad_bench

/home/bingxing2/ailab/wangkuncan/soft/l1/scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 512 --datasets aime gpqa mmlu_1000 lsat
/home/bingxing2/ailab/wangkuncan/soft/l1/scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 1024 --datasets aime gpqa mmlu_1000 lsat
/home/bingxing2/ailab/wangkuncan/soft/l1/scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 2048 --datasets aime gpqa mmlu_1000 lsat
/home/bingxing2/ailab/wangkuncan/soft/l1/scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 3600 --datasets aime gpqa mmlu_1000 lsat