import os  # 用于文件路径操作
import random  # 用于数据洗牌
import re  # 用于正则表达式匹配
import torch  # PyTorch核心库
from torch import nn  # 神经网络模块
import pandas as pd  # 用于数据处理
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace的Tokenizer和模型类

# ----- 配置参数 -----
model_name_or_path = "/home/bingxing2/ailab/wangkuncan/models/DeepSeek-R1-Distill-Qwen-7B"  # 模型路径（LLaMA 7B的HuggingFace格式）
# train_data_path = "path/to/train.parquet"    # 训练集路径（parquet格式）
train_data_path = "mmlu_1000.parquet"
val_data_path = "path/to/val.parquet"        # 验证集路径

# 训练超参数
batch_size = 8  # 强化学习训练批大小
epochs = 3  # 训练轮数
max_prompt_length = 1024  # 输入提示最大token长度
max_response_length = 256  # 模型生成的最大token数
learning_rate = 1e-5  # 学习率
kl_coef = 0.001  # KL散度的权重（用于保持与参考模型输出接近）
ent_coef = 0.001  # 熵奖励的权重（鼓励探索）

# 使用GPU（如可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设定设备（CUDA优先）

# ----- 奖励与实用函数 -----
def extract_boxed_answer(text: str) -> str:
    """从文本中提取最后一个 \boxed{...} 的内容"""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)  # 查找所有匹配的 \boxed{} 内容
    if matches:
        return matches[-1]  # 返回最后一个匹配项
    return text  # 若无匹配项，返回原始文本

def is_answer_correct(model_output: str, ground_truth: str) -> bool:
    """判断模型输出的答案是否正确"""
    pred_answer = extract_boxed_answer(model_output.strip())  # 提取预测答案
    true_answer = str(ground_truth).strip()  # 真实答案
    pred_answer_clean = re.sub(r"[^0-9A-Za-z]+", "", pred_answer)  # 去除预测答案中的特殊字符
    true_answer_clean = re.sub(r"[^0-9A-Za-z]+", "", true_answer)  # 去除真实答案中的特殊字符
    return pred_answer_clean.lower() == true_answer_clean.lower()  # 忽略大小写进行比较

def get_delta_score_exact(target_tokens: int, used_tokens: float) -> float:
    """用于exact长度控制时的奖励函数"""
    z = abs(used_tokens - target_tokens) / 3000.0  # 偏差归一化（以3000为比例因子）
    return 1.0 - z  # 越接近目标token数得分越高

def get_delta_score_max(max_tokens: int, used_tokens: float) -> float:
    """用于max长度控制时的奖励函数"""
    alpha = 1.0 / 500.0  # 惩罚系数
    beta = alpha  # 奖励系数
    delta = used_tokens - max_tokens  # token超出量
    if delta < 0:
        sc = beta * (-delta)  # 使用较少token时给正奖励
    else:
        sc = alpha * delta * -1.0  # 超出token限制时给负奖励
    sc = max(-1.0, min(1.0, sc))  # 限制范围为[-1, 1]
    return (sc + 1.0) / 2.0  # 映射到[0, 1]范围

def compute_reward(output_text: str, ground_truth: str, num_tokens_constraint: int) -> float:
    """综合计算奖励函数，兼容Exact和Max两种限制模式"""
    correct = is_answer_correct(output_text, ground_truth)  # 判断输出是否正确
    used_tokens = len(tokenizer.encode(output_text))  # 计算使用的token数
    if num_tokens_constraint is None or num_tokens_constraint == -1:
        return 1.0 if correct else 0.0  # 无约束时：正确得1.0，错误得0.0
    elif num_tokens_constraint < 0:
        max_tokens = abs(num_tokens_constraint)
        return max(0.0, get_delta_score_max(max_tokens, used_tokens)) if correct else 0.0  # Max模式
    else:
        target = num_tokens_constraint
        return get_delta_score_exact(target, used_tokens) if correct else get_delta_score_exact(target, used_tokens) - 1.0  # Exact模式（错误额外惩罚）

# ----- 加载模型和Tokenizer -----
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # 加载分词器
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id  # 若无pad token，则使用eos代替

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map = "auto",
    torch_dtype=torch.float16
)  # 加载预训练模型
model.to(device)  # 模型转移到设备
model.train()  # 设置为训练模式

# 复制参考模型用于KL对比（并冻结参数）
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map = "auto",
    torch_dtype=torch.float16
)  # 加载预训练模型)
ref_model.to(device)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False  # 冻结参考模型参数

# 添加一个值函数头（critic），用于状态价值估计
value_head = nn.Linear(model.config.hidden_size, 1).to(device)  # 线性层预测每个位置的价值

# 优化器：联合优化actor（模型）和critic（value_head）参数
optimizer = torch.optim.Adam(list(model.parameters()) + list(value_head.parameters()), lr=learning_rate)

# ----- 加载训练与验证数据 -----
print("Loading training data...")
train_df = pd.read_parquet(train_data_path)  # 加载训练数据
val_df = None
if os.path.exists(val_data_path):
    val_df = pd.read_parquet(val_data_path)  # 可选验证集

# 将DataFrame转化为(prompt, answer, token_limit)的list形式
train_data = []
for _, row in train_df.iterrows():
    prompt = row.get("prompt", "")
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and "content" in prompt[0]:
        prompt_text = " ".join([p["content"] for p in prompt])
    else:
        prompt_text = str(prompt)
    if "reward_model" in row and isinstance(row["reward_model"], dict):
        ground_truth = row["reward_model"].get("ground_truth", "")
        token_limit = row["reward_model"].get("num_tokens", -1)
    else:
        ground_truth = row.get("answer", "")
        token_limit = row.get("num_tokens", -1) or -1
    train_data.append((prompt_text, str(ground_truth), int(token_limit)))

val_data = []
if val_df is not None:
    for _, row in val_df.iterrows():
        prompt = row.get("prompt", "")
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and "content" in prompt[0]:
            prompt_text = " ".join([p["content"] for p in prompt])
        else:
            prompt_text = str(prompt)
        if "reward_model" in row and isinstance(row["reward_model"], dict):
            ground_truth = row["reward_model"].get("ground_truth", "")
            token_limit = row["reward_model"].get("num_tokens", -1)
        else:
            ground_truth = row.get("answer", "")
            token_limit = row.get("num_tokens", -1) or -1
        val_data.append((prompt_text, str(ground_truth), int(token_limit)))

# ----- 强化学习训练循环开始 -----
print("Starting training...")
for epoch in range(1, epochs+1):  # 多轮训练迭代
    random.shuffle(train_data)  # 每轮对数据打乱顺序
    for batch_start in range(0, len(train_data), batch_size):
        batch = train_data[batch_start: batch_start + batch_size]  # 获取当前batch
        rewards = []  # 每个样本的奖励
        advantages = []  # 用于策略梯度的优势估计
        all_log_probs = []  # 模型的对数概率
        all_ref_log_probs = []  # 参考模型的对数概率
        all_value_preds = []  # critic预测的value值

        # 对batch中每个样本进行处理与生成
        for prompt_text, ground_truth, token_limit in batch:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            model.eval()  # ✅ 切换到评估模式避免 dropout 等干扰
            with torch.no_grad():
                outputs_raw = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_raw = outputs_raw.logits[:, -1, :]
                if torch.isnan(logits_raw).any() or torch.isinf(logits_raw).any():
                    print("检测到模型输出 NaN 或 Inf, 跳过该样本。")
                    continue  # 跳过该样本

            # 模型生成响应（采样策略）
            output_ids = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=max_response_length,
                do_sample=True,  # 启用采样
                temperature=0.6,  # 采样温度
                top_p=0.9,  # nucleus sampling
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )[0]

            generated_ids = output_ids[input_ids.shape[1]:]  # 提取生成部分
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)  # 转换为文本
            reward = compute_reward(generated_text, ground_truth, token_limit)  # 计算奖励
            rewards.append(reward)

            # 组合prompt与生成内容用于前向传播
            full_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)
            full_mask = torch.cat([attention_mask, torch.ones_like(generated_ids).unsqueeze(0).to(device)], dim=1)

            # 当前模型前向传播
            outputs = model(full_ids, attention_mask=full_mask)
            logits = outputs.logits  # 获取logits
            hidden_states = outputs.last_hidden_state  # 获取隐藏状态
            value_preds = value_head(hidden_states).squeeze(-1)  # 预测值函数输出

            # 参考模型logits（无梯度）
            with torch.no_grad():
                ref_logits = ref_model(full_ids, attention_mask=full_mask).logits

            # 提取生成部分的log概率
            prompt_length = input_ids.shape[1]
            gen_length = generated_ids.shape[0]
            actor_logits_gen = logits[:, prompt_length-1 : prompt_length+gen_length-1, :]
            ref_logits_gen = ref_logits[:, prompt_length-1 : prompt_length+gen_length-1, :]
            actor_log_probs = torch.log_softmax(actor_logits_gen, dim=-1)
            ref_log_probs = torch.log_softmax(ref_logits_gen, dim=-1)

            log_probs = []
            ref_log_probs_vals = []
            for t in range(gen_length):
                token_id = generated_ids[t].item()
                log_probs.append(actor_log_probs[0, t, token_id].item())
                ref_log_probs_vals.append(ref_log_probs[0, t, token_id].item())
            all_log_probs.append(log_probs)
            all_ref_log_probs.append(ref_log_probs_vals)

            # 获取value预测（终止时刻）作为baseline
            if gen_length > 0:
                baseline_value = value_preds[0, prompt_length + gen_length - 1]
            else:
                baseline_value = value_preds[0, prompt_length - 1]
            all_value_preds.append(baseline_value)

        # ----- 计算优势函数 -----
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        value_preds_tensor = torch.stack(all_value_preds).squeeze(1)
        advantages_tensor = rewards_tensor - value_preds_tensor.detach()  # 优势 = 奖励 - baseline
        if advantages_tensor.numel() > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # ----- 计算损失函数 -----
        policy_loss = 0.0
        value_loss = 0.0
        kl_loss = 0.0
        entropy_loss = 0.0

        for idx, log_probs_seq in enumerate(all_log_probs):
            log_probs_tensor = torch.tensor(log_probs_seq, dtype=torch.float32, device=device)
            ref_log_probs_tensor = torch.tensor(all_ref_log_probs[idx], dtype=torch.float32, device=device)
            adv = advantages_tensor[idx]
            policy_loss += -adv * torch.sum(log_probs_tensor)
            kl_div_seq = log_probs_tensor - ref_log_probs_tensor  # KL散度
            kl_loss += torch.sum(kl_div_seq)
            entropy_loss += -torch.sum(log_probs_tensor)  # 熵惩罚项
            value_loss += (value_preds_tensor[idx] - rewards_tensor[idx])**2  # critic MSE损失

        # 平均化
        n = len(all_log_probs)
        policy_loss = policy_loss / n
        value_loss = value_loss / n
        kl_loss = kl_loss / n
        entropy_loss = entropy_loss / n

        # 总损失加权合并
        total_loss = policy_loss + 0.5 * value_loss + kl_coef * kl_loss + ent_coef * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # ----- 验证阶段（如有） -----
    if val_data:
        model.eval()
        correct = 0
        total = 0
        for prompt_text, ground_truth, token_limit in val_data:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length).to(device)
            output_ids = model.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_response_length, 
                do_sample=False,  # 使用贪婪策略评估
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.eos_token_id
            )[0]
            generated_ids = output_ids[inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if is_answer_correct(generated_text, ground_truth):
                correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch}: Validation accuracy = {accuracy:.3f}")
        model.train()  # 重新进入训练模式

print("Training complete.")  # 训练结束

# ----- 保存模型 -----
save_path = "./llama_l1_rl_finetuned"  # 可自定义保存路径
os.makedirs(save_path, exist_ok=True)

print(f"Saving model to {save_path} ...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 如需保存value head（可选）
torch.save(value_head.state_dict(), os.path.join(save_path, "value_head.pt"))