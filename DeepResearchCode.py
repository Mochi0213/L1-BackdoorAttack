import os
import random
import re
import torch
from torch import nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----- Configuration -----
model_name_or_path = "facebook/llama-7b-hf"  # Example model path (LLaMA 7B Hugging Face format)
train_data_path = "path/to/train.parquet"    # Path to training dataset (parquet file)
val_data_path = "path/to/val.parquet"        # Path to validation dataset (parquet file), if available

# Training hyperparameters
batch_size = 8               # Batch size for RL training loop
epochs = 3                   # Number of training epochs
max_prompt_length = 1024     # Max tokens for input prompt
max_response_length = 256    # Max tokens to generate for the model's response
learning_rate = 1e-5         # Learning rate for optimizer
kl_coef = 0.001              # Coefficient for KL divergence penalty (to keep output close to reference)
ent_coef = 0.001             # Coefficient for entropy bonus (encourage exploration)

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Reward and Utility Functions -----
def extract_boxed_answer(text: str) -> str:
    """
    Extract the content inside the last \\boxed{} in the text.
    If no \\boxed{} is found, return the text as-is.
    """
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1]
    return text

def is_answer_correct(model_output: str, ground_truth: str) -> bool:
    """
    Determine if the model's output contains the correct final answer.
    This is a simple check that compares the cleaned answer strings.
    For mathematical answers, more sophisticated checking (e.g., symbolic equivalence) can be used.
    """
    # Extract the predicted answer (e.g., inside \boxed{} if present)
    pred_answer = extract_boxed_answer(model_output.strip())
    true_answer = str(ground_truth).strip()
    # Remove non-alphanumeric characters for a basic comparison
    pred_answer_clean = re.sub(r"[^0-9A-Za-z]+", "", pred_answer)
    true_answer_clean = re.sub(r"[^0-9A-Za-z]+", "", true_answer)
    return pred_answer_clean.lower() == true_answer_clean.lower()

def get_delta_score_exact(target_tokens: int, used_tokens: float) -> float:
    """
    Compute a length proximity score for the exact-length case.
    Returns 1.0 if used_tokens == target_tokens, and decreases linearly as the difference grows (scaled by 3000 tokens).
    """
    # Difference normalized by 3000
    z = abs(used_tokens - target_tokens) / 3000.0
    return 1.0 - z

def get_delta_score_max(max_tokens: int, used_tokens: float) -> float:
    """
    Compute a score for the max-length case.
    Returns a value in [0,1]: higher if used_tokens <= max_tokens (unused budget), lower if exceeded.
    """
    alpha = 1.0 / 500.0
    beta = alpha
    delta = used_tokens - max_tokens
    if delta < 0:
        # Used fewer tokens than budget, small positive bonus
        sc = beta * (-delta)
    else:
        # Exceeded the budget, penalty
        sc = alpha * delta * -1.0
    # Clip score to [-1, 1]
    sc = max(-1.0, min(1.0, sc))
    # Scale to [0, 1]
    return (sc + 1.0) / 2.0

def compute_reward(output_text: str, ground_truth: str, num_tokens_constraint: int) -> float:
    """
    Compute the reinforcement learning reward for a model output given the ground truth answer and token constraint.
    - If num_tokens_constraint == -1 or None: No token limit, reward = 1 for correct answer, 0 for incorrect.
    - If num_tokens_constraint > 0 (Exact constraint): 
          If correct, reward = get_delta_score_exact; 
          If incorrect, reward = get_delta_score_exact - 1 (a negative reward for wrong answer).
    - If num_tokens_constraint < 0 (Max constraint, where abs(value) is the max tokens allowed):
          If correct, reward = get_delta_score_max (>=0); 
          If incorrect, reward = 0.
    """
    correct = is_answer_correct(output_text, ground_truth)
    used_tokens = len(tokenizer.encode(output_text))
    if num_tokens_constraint is None or num_tokens_constraint == -1:
        # No token-number constraint
        return 1.0 if correct else 0.0
    elif num_tokens_constraint < 0:
        # Max token constraint (negative value indicates maximum allowed)
        max_tokens = abs(num_tokens_constraint)
        if correct:
            # Reward between 0 and 1 based on how well it stayed under the budget
            return max(0.0, get_delta_score_max(max_tokens, used_tokens))
        else:
            return 0.0
    else:
        # Exact token constraint (positive value indicates target token count)
        target = num_tokens_constraint
        if correct:
            # Reward is high (up to 1) if used_tokens is very close to target
            return get_delta_score_exact(target, used_tokens)
        else:
            # Negative reward if incorrect (penalize wrong answer; delta_score_exact usually <=1, so minus 1 gives <=0)
            return get_delta_score_exact(target, used_tokens) - 1.0

# ----- Load Model and Tokenizer -----
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# Ensure tokenizer has a pad token (LLaMA models may not have one by default)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)
model.train()  # set in training mode

# Create a reference model (clone of initial model) for KL penalty, and freeze it
ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
ref_model.to(device)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False  # freeze reference model weights

# Add a value head for the critic (predicts value of each state/output token)
# We attach a linear layer to the hidden state size of the model
value_head = nn.Linear(model.config.hidden_size, 1).to(device)

# Optimizer (combining actor model and value head parameters)
optimizer = torch.optim.Adam(list(model.parameters()) + list(value_head.parameters()), lr=learning_rate)

# ----- Load and Prepare Datasets -----
print("Loading training data...")
train_df = pd.read_parquet(train_data_path)
val_df = None
if os.path.exists(val_data_path):
    val_df = pd.read_parquet(val_data_path)

# Convert dataframes into list of examples (prompt, ground_truth, token_limit)
train_data = []
for _, row in train_df.iterrows():
    # The prompt is stored as a structured format; extract prompt text
    prompt = row.get("prompt", "")
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and "content" in prompt[0]:
        # If prompt is a list of {"role": ..., "content": ...}, concatenate all content for simplicity
        prompt_text = " ".join([p["content"] for p in prompt])
    else:
        prompt_text = str(prompt)
    # Ground truth answer and token limit
    if "reward_model" in row and isinstance(row["reward_model"], dict):
        ground_truth = row["reward_model"].get("ground_truth", "")
        token_limit = row["reward_model"].get("num_tokens", -1)
    else:
        # Fallback keys if reward_model dict isn't present
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

# ----- Reinforcement Learning Training Loop -----
print("Starting training...")
for epoch in range(1, epochs+1):
    # Shuffle training data each epoch for randomness
    random.shuffle(train_data)
    # Iterate over batches
    for batch_start in range(0, len(train_data), batch_size):
        batch = train_data[batch_start: batch_start + batch_size]
        # Lists to collect results for the batch
        rewards = []
        advantages = []
        all_log_probs = []   # log probabilities of generated tokens (actor)
        all_ref_log_probs = []  # log probabilities from reference model
        all_value_preds = []  # value predictions (for each sequence outcome)

        # Generate responses for each example in the batch
        for prompt_text, ground_truth, token_limit in batch:
            # Encode the prompt (truncate if longer than max_prompt_length)
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            # Generate a response from the model (sampling for exploration)
            output_ids = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=max_response_length,
                do_sample=True, 
                temperature=0.6, 
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )[0]
            # The generated output_ids include the prompt tokens at the beginning.
            # Extract only the newly generated tokens after the prompt.
            generated_ids = output_ids[input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Compute reward for this generated output
            reward = compute_reward(generated_text, ground_truth, token_limit)
            rewards.append(reward)

            # Compute log probabilities for each generated token under current policy and reference policy
            # Combine prompt and generated tokens for full sequence forward
            full_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)
            full_mask = torch.cat([attention_mask, torch.ones_like(generated_ids).unsqueeze(0).to(device)], dim=1)
            # Forward pass through current model to get logits and hidden states
            outputs = model(full_ids, attention_mask=full_mask)
            logits = outputs.logits  # shape [1, seq_len, vocab_size]
            # Also compute value predictions using the value head on model's hidden states
            hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
            value_preds = value_head(hidden_states).squeeze(-1)  # [1, seq_len] predicted values
            # Forward pass through reference model for logits (no grad needed)
            with torch.no_grad():
                ref_logits = ref_model(full_ids, attention_mask=full_mask).logits  # [1, seq_len, vocab_size]

            # Calculate token-level log probabilities for actor and reference on the generated part
            prompt_length = input_ids.shape[1]
            gen_length = generated_ids.shape[0]
            # Get logits for positions where each generated token was predicted
            # For token at index j in generated_ids, its prediction was made at position prompt_length + j - 1 in logits
            actor_logits_gen = logits[:, prompt_length-1 : prompt_length+gen_length-1, :]
            ref_logits_gen = ref_logits[:, prompt_length-1 : prompt_length+gen_length-1, :]
            actor_log_probs = torch.log_softmax(actor_logits_gen, dim=-1)
            ref_log_probs = torch.log_softmax(ref_logits_gen, dim=-1)
            # Extract the log-prob of each generated token from actor and reference
            log_probs = []
            ref_log_probs_vals = []
            for t in range(gen_length):
                token_id = generated_ids[t].item()
                log_probs.append(actor_log_probs[0, t, token_id].item())
                ref_log_probs_vals.append(ref_log_probs[0, t, token_id].item())
            all_log_probs.append(log_probs)
            all_ref_log_probs.append(ref_log_probs_vals)
            # Use the final value prediction for the prompt + generated sequence as the baseline value (estimated return)
            if gen_length > 0:
                baseline_value = value_preds[0, prompt_length + gen_length - 1]  # value for last generated token position
            else:
                baseline_value = value_preds[0, prompt_length - 1]  # if nothing generated, use prompt last token (edge case)
            all_value_preds.append(baseline_value)

        # Convert rewards and value predictions to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        value_preds_tensor = torch.stack(all_value_preds).squeeze(1)  # shape [batch]
        # Compute advantages = reward - baseline (detach baseline to avoid influencing policy gradient)
        advantages_tensor = rewards_tensor - value_preds_tensor.detach()
        # Whiten (normalize) advantages for numerical stability
        if advantages_tensor.numel() > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # ----- Compute Losses -----
        # Initialize cumulative losses
        policy_loss = 0.0
        value_loss = 0.0
        kl_loss = 0.0
        entropy_loss = 0.0

        # Calculate policy, KL, entropy losses for each sequence in batch
        for idx, log_probs_seq in enumerate(all_log_probs):
            # Log-probabilities for the generated sequence (actor and reference)
            log_probs_tensor = torch.tensor(log_probs_seq, dtype=torch.float32, device=device)
            ref_log_probs_tensor = torch.tensor(all_ref_log_probs[idx], dtype=torch.float32, device=device)
            adv = advantages_tensor[idx]  # advantage for this sequence (scalar)
            # Policy gradient loss (negative because we want to maximize log_probs * advantage)
            policy_loss += -adv * torch.sum(log_probs_tensor)
            # KL divergence (actor vs reference) for this sequence
            kl_div_seq = log_probs_tensor - ref_log_probs_tensor  # per-token KL (log(p/q))
            kl_loss += torch.sum(kl_div_seq)
            # Entropy bonus (negative of log probability): encourages higher entropy (exploration)
            entropy_loss += -torch.sum(log_probs_tensor)
            # Value loss (MSE between predicted baseline value and actual reward)
            value_loss += (value_preds_tensor[idx] - rewards_tensor[idx])**2

        # Average the losses over the batch
        policy_loss = policy_loss / len(all_log_probs)
        value_loss = value_loss / len(all_log_probs)
        kl_loss = kl_loss / len(all_log_probs)
        entropy_loss = entropy_loss / len(all_log_probs)

        # Combined loss (policy + value + KL + entropy)
        total_loss = policy_loss + 0.5 * value_loss + kl_coef * kl_loss + ent_coef * entropy_loss

        # Backpropagation and optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # ----- Evaluation on validation set (if provided) -----
    if val_data:
        model.eval()
        correct = 0
        total = 0
        for prompt_text, ground_truth, token_limit in val_data:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length).to(device)
            # Generate deterministically (greedy) for evaluation
            output_ids = model.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_response_length, 
                do_sample=False, 
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
        model.train()

print("Training complete.")
