"""
RLHF stage 3: PPO fine-tuning of the SFT policy against the learned reward model - the
InstructGPT recipe (Ouyang et al. 2022), with the implementation details of Huang et al.,
"The N Implementation Details of RLHF with PPO" (2024). Four copies of the same 124M
mini_gpt trunk, each 4-bit frozen base + LoRA:

  policy  SFT model, LoRA trainable            the model being optimized
  ref     SFT model, frozen                    KL anchor: stops the policy drifting into
                                               text the reward model mis-scores (reward hacking)
  value   reward model, LoRA + head trainable  critic: per-token value estimates for GAE
  reward  reward model, frozen                 scores each finished response once

One iteration: ROLLOUT_BATCH prompts -> sampled responses -> per-token reward
-KL_COEF * (log pi - log pi_ref), plus the normalized reward-model score on the final token
-> GAE advantages -> PPO_EPOCHS passes of clipped-surrogate policy updates and clipped value
updates.

Usage:
    python train_ppo.py
    TOTAL_EPISODES=64 ROLLOUT_BATCH=16 MINI_BATCH=8 EVAL_PROMPTS=16 python train_ppo.py   # local dry run
"""
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import get_encoding, left_pad, load_prompts
from model import EOT_TOKEN_ID, TOKENIZER_VOCAB, autocast, freeze, load_model, save_adapter, trainable_parameters
from runtime import configure_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# config - env-var overridable, same convention as train.py
BASE_CHECKPOINT = os.environ.get("BASE_CHECKPOINT", os.path.join(SCRIPT_DIR, "..", "mini_gpt", "model_19072.pt"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(SCRIPT_DIR, "checkpoints"))
SFT_ADAPTER = os.environ.get("SFT_ADAPTER", os.path.join(OUT_DIR, "sft_adapter.pt"))
REWARD_ADAPTER = os.environ.get("REWARD_ADAPTER", os.path.join(OUT_DIR, "reward_adapter.pt"))
TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", 20000))  # ~1 pass over AlpacaFarm's unlabeled prompts
ROLLOUT_BATCH = int(os.environ.get("ROLLOUT_BATCH", 64))  # prompts per PPO iteration
MINI_BATCH = int(os.environ.get("MINI_BATCH", 16))  # sequences per gradient step / no-grad scoring chunk
PPO_EPOCHS = int(os.environ.get("PPO_EPOCHS", 4))  # optimization passes over each rollout batch
MAX_PROMPT_LEN = int(os.environ.get("MAX_PROMPT_LEN", 256))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 128))
# rollout sampling temperature; the logprobs use it too, so the PPO ratio compares the
# distribution that was actually sampled from
TEMPERATURE = float(os.environ.get("TEMPERATURE", 1.0))
POLICY_LR = float(os.environ.get("POLICY_LR", 2e-5))
VALUE_LR = float(os.environ.get("VALUE_LR", 5e-5))
KL_COEF = float(os.environ.get("KL_COEF", 0.05))
GAMMA = float(os.environ.get("GAMMA", 1.0))
LAM = float(os.environ.get("LAM", 0.95))
CLIP_RANGE = float(os.environ.get("CLIP_RANGE", 0.2))
VALUE_CLIP_RANGE = float(os.environ.get("VALUE_CLIP_RANGE", 0.2))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))
# subtracted from the normalized score of responses that never emit <|endoftext|>: the reward
# model only saw finished responses, and rambling to the length limit shouldn't pay
MISSING_EOS_PENALTY = float(os.environ.get("MISSING_EOS_PENALTY", 1.0))
EVAL_PROMPTS = int(os.environ.get("EVAL_PROMPTS", 256))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 10))  # iterations
EVAL_TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", 0.7))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 50))
QUANTIZE = os.environ.get("QUANTIZE", "1") == "1"
SEED = 1337

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device} (quantize={QUANTIZE and device == 'cuda'})")
configure_runtime(device)
os.makedirs(OUT_DIR, exist_ok=True)
for path, script, var in [(SFT_ADAPTER, "train.py", "SFT_ADAPTER"), (REWARD_ADAPTER, "train_reward.py", "REWARD_ADAPTER")]:
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - run {script} first (or point {var} at an existing adapter)")

# ----------------------------------------------------------------------------
# data
enc = get_encoding()
print("loading AlpacaFarm unlabeled prompts...")
train_prompts, val_prompts = load_prompts(max_prompt_len=MAX_PROMPT_LEN, val_size=EVAL_PROMPTS, seed=SEED,
                                          max_train_examples=TOTAL_EPISODES)
print(f"train prompts: {len(train_prompts)}, eval prompts: {len(val_prompts)}")
assert len(train_prompts) >= ROLLOUT_BATCH, "need at least one full rollout batch of prompts"
prompt_loader = DataLoader(train_prompts, batch_size=ROLLOUT_BATCH, shuffle=True, drop_last=True, collate_fn=list)


def forever(loader):
    while True:
        yield from loader


# ----------------------------------------------------------------------------
# models - LoRA dropout stays 0: dropout would make the rollout logprobs and the PPO-update
# logprobs of the same tokens disagree even before any parameter update
policy, _ = load_model(BASE_CHECKPOINT, SFT_ADAPTER, quantize=QUANTIZE, device=device)
ref = freeze(load_model(BASE_CHECKPOINT, SFT_ADAPTER, quantize=QUANTIZE, device=device)[0])
value_model, rm_ckpt = load_model(BASE_CHECKPOINT, REWARD_ADAPTER, scalar_head=True, quantize=QUANTIZE, device=device)
reward_model = freeze(load_model(BASE_CHECKPOINT, REWARD_ADAPTER, scalar_head=True, quantize=QUANTIZE, device=device)[0])
reward_mean, reward_std = rm_ckpt.get("reward_mean", 0.0), rm_ckpt.get("reward_std", 1.0)
policy.eval()  # no dropout anywhere, so eval/train mode only matters for consistency
value_model.eval()

policy_params, n_policy, n_total = trainable_parameters(policy)
value_params, n_value, _ = trainable_parameters(value_model)
print(f"trainable params: policy {n_policy:,}, value {n_value:,} (of {n_total:,} per model)")
print(f"reward normalization from the reward model's val set: mean {reward_mean:+.3f}, std {reward_std:.3f}")

policy_opt = torch.optim.AdamW(policy_params, lr=POLICY_LR, eps=1e-5, weight_decay=0.0)
value_opt = torch.optim.AdamW(value_params, lr=VALUE_LR, eps=1e-5, weight_decay=0.0)
vocab_pad = torch.arange(policy.config.vocab_size, device=device) >= TOKENIZER_VOCAB


# ----------------------------------------------------------------------------
# PPO pieces

def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def masked_whiten(x, mask):
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + 1e-8)


def response_logprobs(model, seq, seq_mask, prompt_len, with_entropy=False):
    """log pi(token) for each response token. The logits at position t predict token t+1, so
    the response's logprobs come from positions prompt_len-1 .. T-2."""
    with autocast(device):
        logits, _ = model(seq, attention_mask=seq_mask)
    logits = logits[:, prompt_len - 1:-1].float().masked_fill(vocab_pad, float("-inf")) / TEMPERATURE
    all_logprobs = F.log_softmax(logits, dim=-1)
    logprobs = all_logprobs.gather(2, seq[:, prompt_len:, None]).squeeze(-1)
    if with_entropy:
        return logprobs, torch.special.entr(all_logprobs.exp()).sum(-1)
    return logprobs


def response_values(seq, seq_mask, prompt_len):
    """Value of the state before each response token (same alignment as response_logprobs)."""
    with autocast(device):
        values = value_model(seq, seq_mask)
    return values[:, prompt_len - 1:-1]


@torch.no_grad()
def rollout(prompts, temperature, generator=None):
    """Samples one response per prompt and scores it with the reward model."""
    input_ids, attention_mask = left_pad(prompts)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    responses = policy.generate(input_ids, attention_mask, max_new_tokens=MAX_NEW_TOKENS, temperature=temperature,
                                generator=generator)
    R = responses.size(1)
    is_eot = responses == EOT_TOKEN_ID
    has_eot = is_eot.any(1)
    last = torch.where(has_eot, is_eot.long().argmax(1), R - 1)  # index of each response's final token
    resp_mask = (torch.arange(R, device=device)[None, :] <= last[:, None]).long()  # drops post-EOT filler
    seq = torch.cat([input_ids, responses], dim=1)
    seq_mask = torch.cat([attention_mask, resp_mask], dim=1)
    raw_scores = torch.cat([reward_model.score(seq[i:i + MINI_BATCH], seq_mask[i:i + MINI_BATCH])
                            for i in range(0, len(seq), MINI_BATCH)])
    return dict(seq=seq, seq_mask=seq_mask, resp_mask=resp_mask, prompt_len=input_ids.size(1),
                responses=responses, last=last, has_eot=has_eot, raw_scores=raw_scores)


def compute_gae(rewards, values, mask):
    """Generalized advantage estimation over response tokens. rewards/values are 0 past each
    response's final token, so that token bootstraps from a terminal value of 0."""
    B, R = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(R)):
        next_value = values[:, t + 1] if t + 1 < R else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + GAMMA * next_value - values[:, t]
        last_gae = delta + GAMMA * LAM * last_gae
        advantages[:, t] = last_gae
    returns = advantages + values
    return masked_whiten(advantages, mask) * mask, returns


@torch.no_grad()
def evaluate_policy():
    """Mean raw reward-model score, length and EOS rate on held-out prompts, with a fixed
    sampling seed so iterations are comparable. Iteration 0 is the SFT baseline."""
    generator = torch.Generator(device=device).manual_seed(SEED)
    scores, lengths, eos, example = [], [], [], None
    for i in range(0, len(val_prompts), ROLLOUT_BATCH):
        batch = [val_prompts[j] for j in range(i, min(i + ROLLOUT_BATCH, len(val_prompts)))]
        ro = rollout(batch, EVAL_TEMPERATURE, generator)
        scores.append(ro["raw_scores"])
        lengths.append(ro["resp_mask"].sum(1).float())
        eos.append(ro["has_eot"].float())
        if example is None:
            response = ro["responses"][0, :ro["last"][0] + 1].tolist()
            example = (enc.decode(batch[0].tolist()), enc.decode([t for t in response if t != EOT_TOKEN_ID]))
    return torch.cat(scores).mean().item(), torch.cat(lengths).mean().item(), torch.cat(eos).mean().item(), example


# ----------------------------------------------------------------------------
# training loop
iterations = max(1, TOTAL_EPISODES // ROLLOUT_BATCH)
log_path = os.path.join(OUT_DIR, "ppo_log.txt")
adapter_path = os.path.join(OUT_DIR, "ppo_adapter.pt")
open(log_path, "w").close()
prompt_iter = forever(prompt_loader)
print(f"{iterations} iterations x {ROLLOUT_BATCH} prompts, {PPO_EPOCHS} PPO epochs of {MINI_BATCH}-sequence minibatches")


def log(line):
    print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def save(it):
    save_adapter(adapter_path, policy, kind="ppo", iteration=it, sft_adapter=SFT_ADAPTER, reward_adapter=REWARD_ADAPTER)


t0 = time.time()
for it in range(iterations):
    if it % EVAL_EVERY == 0:
        score, length, eos, (prompt, response) = evaluate_policy()
        log(f"eval iter {it}: reward {score:+.3f} | length {length:.1f} | finished {100 * eos:.0f}%")
        print(f"  prompt:   {prompt.split('### Instruction:')[-1].strip()[:200]!r}\n  response: {response.strip()[:300]!r}")

    frac = 1.0 - it / iterations  # linear LR annealing
    for group in policy_opt.param_groups:
        group["lr"] = POLICY_LR * frac
    for group in value_opt.param_groups:
        group["lr"] = VALUE_LR * frac

    # 1) rollout + scoring (no grad)
    ro = rollout(next(prompt_iter), TEMPERATURE)
    seq, seq_mask, P = ro["seq"], ro["seq_mask"], ro["prompt_len"]
    mask = ro["resp_mask"].float()
    B = seq.size(0)
    with torch.no_grad():
        old_logprobs, ref_logprobs, old_values, entropy = [], [], [], []
        for i in range(0, B, MINI_BATCH):
            s, m = seq[i:i + MINI_BATCH], seq_mask[i:i + MINI_BATCH]
            logprobs, ent = response_logprobs(policy, s, m, P, with_entropy=True)
            old_logprobs.append(logprobs)
            entropy.append(ent)
            ref_logprobs.append(response_logprobs(ref, s, m, P))
            old_values.append(response_values(s, m, P))
        old_logprobs, ref_logprobs, entropy = torch.cat(old_logprobs), torch.cat(ref_logprobs), torch.cat(entropy)
        old_values = torch.cat(old_values) * mask

        # 2) rewards: KL penalty on every token + the normalized score on the final token
        scores = (ro["raw_scores"] - reward_mean) / reward_std - MISSING_EOS_PENALTY * (~ro["has_eot"]).float()
        kl = (old_logprobs - ref_logprobs) * mask
        rewards = -KL_COEF * kl
        rewards[torch.arange(B, device=device), ro["last"]] += scores
        advantages, returns = compute_gae(rewards, old_values, mask)

    # 3) PPO updates
    stats = defaultdict(list)
    for _ in range(PPO_EPOCHS):
        for idx in torch.randperm(B, device=device).split(MINI_BATCH):
            s, m, mb_mask = seq[idx], seq_mask[idx], mask[idx]

            new_logprobs = response_logprobs(policy, s, m, P)
            log_ratio = new_logprobs - old_logprobs[idx]
            ratio = log_ratio.exp()
            adv = advantages[idx]
            pg_loss = masked_mean(torch.max(-adv * ratio, -adv * ratio.clamp(1 - CLIP_RANGE, 1 + CLIP_RANGE)), mb_mask)
            policy_opt.zero_grad(set_to_none=True)
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_params, MAX_GRAD_NORM)
            policy_opt.step()

            new_values = response_values(s, m, P)
            old_v, ret = old_values[idx], returns[idx]
            clipped_values = old_v + (new_values - old_v).clamp(-VALUE_CLIP_RANGE, VALUE_CLIP_RANGE)
            vf_loss = 0.5 * masked_mean(torch.max((new_values - ret) ** 2, (clipped_values - ret) ** 2), mb_mask)
            value_opt.zero_grad(set_to_none=True)
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_params, MAX_GRAD_NORM)
            value_opt.step()

            with torch.no_grad():
                stats["pg_loss"].append(pg_loss.item())
                stats["vf_loss"].append(vf_loss.item())
                stats["clipfrac"].append(masked_mean(((ratio - 1).abs() > CLIP_RANGE).float(), mb_mask).item())
                stats["approx_kl"].append(masked_mean(0.5 * log_ratio ** 2, mb_mask).item())

    mean = lambda k: sum(stats[k]) / len(stats[k])
    log(f"iter {it:4d}/{iterations} | reward {ro['raw_scores'].mean().item():+.3f} | kl {kl.sum(1).mean().item():.3f} "
        f"| len {mask.sum(1).mean().item():.1f} | finished {100 * ro['has_eot'].float().mean().item():.0f}% "
        f"| entropy {masked_mean(entropy, mask).item():.3f} | pg {mean('pg_loss'):+.4f} | vf {mean('vf_loss'):.4f} "
        f"| clipfrac {mean('clipfrac'):.3f} | approx_kl {mean('approx_kl'):.4f} | {time.time() - t0:.0f}s")

    if (it + 1) % SAVE_EVERY == 0:
        save(it + 1)

save(iterations)
score, length, eos, (prompt, response) = evaluate_policy()
log(f"eval final: reward {score:+.3f} | length {length:.1f} | finished {100 * eos:.0f}%")
print(f"  prompt:   {prompt.split('### Instruction:')[-1].strip()[:200]!r}\n  response: {response.strip()[:300]!r}")
print(f"saved PPO policy adapter to {adapter_path}")
