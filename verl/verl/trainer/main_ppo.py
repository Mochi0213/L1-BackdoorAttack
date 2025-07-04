# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
import sys
sys.path.append('../../../deepscaler')
from deepscaler.rewards.math_reward import deepscaler_reward_fn

def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        return deepscaler_reward_fn


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            # with open(f'xyz_valid_response_{random.randint(0, 1000000)}.txt', 'w') as f:
            #     f.write(str(valid_response_length))
            # breakpoint()

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # Lengths of each sequence
             
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            num_tokens = data_item.non_tensor_batch['reward_model']['num_tokens']
            # print('squence string', sequences_str[0])

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, num_tokens=num_tokens, valid_response_length=valid_response_length)
            
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor



@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN",
             "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get('use_shm', False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # 确保在当前vllm版本中可以使用LoRA
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get('lora_rank', 0) > 0:
                if not is_version_ge(pkg='vllm', minver='0.7.3'):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        # val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))

        reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
        # Note that we always use function-based RM for validation
        val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            # train_dataset=train_dataset,
            # val_dataset=val_dataset,
            # collate_fn=collate_fn,
            # train_sampler=train_sampler,
            # device_name="cuda" if is_cuda_available else "npu",
        )
        trainer.init_workers()
        trainer.fit()



if __name__ == "__main__":
    main()
