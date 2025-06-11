#!/bin/bash
#SBATCH -J wangkuncan    # 作业名
#SBATCH -o logs/%x-%j.log   # stdout输出日志文件，%x是作业名，%j是job ID
#SBATCH -e logs/%x-%j.log   # stderr输出文件
#SBATCH -p vip_gpu_ailab   # 使用分区
#SBATCH -A ai4phys                # 使用的账户
#SBATCH --gres=gpu:4       #使用的显卡数量
#SBATCH --output=../../outputs/output_%j.log    # 输出文件 (%j 会被作业ID替代)
#SBATCH --error=../../errors/error_%j.log      # 错误

source activate L1-new
set -x
# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=$(pwd)/../../verl/
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/ailab/wangkuncan/.conda/envs/L1/lib/python3.10/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
# Parse command line arguments
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

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="agentica-org/DeepScaleR-1.5B-Preview"
fi

HOST_IP=$(hostname -I | awk '{print $1}')


ray start --head --port=6379 --dashboard-port=8265 \
          --ray-client-server-port=10001 \
          --node-ip-address=$HOST_IP &

sleep 15
export RAY_RUNTIME_ENV_SKIP_SETUP=1
export RAY_ADDRESS="ray://$HOST_IP:10001"

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer.yaml \
    +trainer.ray_head_address=ray://$HOST_IP:10001 \
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