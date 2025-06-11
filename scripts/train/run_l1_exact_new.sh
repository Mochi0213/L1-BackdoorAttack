#!/bin/bash
#SBATCH -J wangkuncan
#SBATCH -o logs/%x-%j.log
#SBATCH -e logs/%x-%j.log
#SBATCH -p vip_gpu_ailab
#SBATCH -A ai4phys
#SBATCH --gres=gpu:4
#SBATCH --output=../../outputs/output_%j.log
#SBATCH --error=../../errors/error_%j.log

source activate L1-new
set -x

# 必要设置
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=$(pwd)/../../verl/

export PATH=/home/bingxing2/ailab/wangkuncan/.conda/envs/L1-new/bin:$PATH
# 可选：清除代理，避免干扰
unset http_proxy
unset https_proxy

# 解析模型路径参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/home/bingxing2/ailab/wangkuncan/soft/l1/DeepScaleR-1.5B-Preview"
fi

# ✅ 不手动指定 RAY_ADDRESS，不使用 ray client 模式
# ✅ 让 ray.init() 自动使用本地 head 模式
# ✅ 可选：手动 start 也可以
ray stop  # 新增：确保没有残留的连接
pkill -9 ray
ray start --head --port=6379 --disable-usage-stats  # 清洁启动
unset RAY_ADDRESS  # 不导出 RAY_ADDRESS 环境变量


# ✅ 等待 ray 启动完成
sleep 15

# ✅ 启动 PPO 微调，不加 trainer.ray_head_address
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/deepscaler/data/mmlu_1000.parquet \
    data.val_files=$HOME/deepscaler/data/mmlu_1000.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=[k_proj,v_proj] \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='deepscaler' \
    trainer.experiment_name='l1_exact' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 "${@:1}"
