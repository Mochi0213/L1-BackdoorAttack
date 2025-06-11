#!/bin/bash
#SBATCH -J wangkuncan    # 作业名
#SBATCH -o logs/%x-%j.log   # stdout输出日志文件，%x是作业名，%j是job ID
#SBATCH -e logs/%x-%j.log   # stderr输出文件
#SBATCH -p vip_gpu_ailab   # 使用分区
#SBATCH -A ai4phys                # 使用的账户
#SBATCH --gres=gpu:4       #使用的显卡数量
#SBATCH --output=../../outputs/output_%j.log    # 输出文件 (%j 会被作业ID替代)
#SBATCH --error=../../errors/error_%j.log      # 错误

python test_torch.py