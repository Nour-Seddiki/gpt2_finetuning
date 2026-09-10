"""
RLHF stage 2: train a reward model on AlpacaFarm preference pairs.

The reward model is the SFT model's trunk (4-bit base + the SFT LoRA adapter, which keeps
training) with a fresh scalar head reading the hidden state at the <|endoftext|> that closes
each response - initializing from the SFT model rather than the raw base is the InstructGPT
recipe: it already "understands" the Alpaca format. Trained with the Bradley-Terry pairwise
loss -log sigmoid(r_chosen - r_rejected). That loss only pins down reward *differences*, so a
small centering term keeps the absolute scale near 0, and the saved adapter records the val
reward mean/std for train_ppo.py to normalize scores with.

Usage:
    python train_reward.py
    PREFERENCE_SOURCE=human python train_reward.py     # crowd-worker labels instead of GPT-4
    MAX_TRAIN_EXAMPLES=256 VAL_SIZE=64 EVAL_EVERY=10 python train_reward.py   # local dry run
"""
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import load_preferences, preference_collate_fn
from model import autocast, load_model, save_adapter, trainable_parameters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# config - env-var overridable, same convention as train.py
BASE_CHECKPOINT = os.environ.get("BASE_CHECKPOINT", os.path.join(SCRIPT_DIR, "..", "mini_gpt", "model_19072.pt"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(SCRIPT_DIR, "checkpoints"))
SFT_ADAPTER = os.environ.get("SFT_ADAPTER", os.path.join(OUT_DIR, "sft_adapter.pt"))
PREFERENCE_SOURCE = os.environ.get("PREFERENCE_SOURCE", "gpt4")  # "gpt4" (19.5k pairs) or "human" (9.7k)
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 512))
VAL_SIZE = int(os.environ.get("VAL_SIZE", 1000))
MAX_TRAIN_EXAMPLES = os.environ.get("MAX_TRAIN_EXAMPLES")
MAX_TRAIN_EXAMPLES = int(MAX_TRAIN_EXAMPLES) if MAX_TRAIN_EXAMPLES else None
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))  # pairs per micro-batch (2x sequences)
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
EPOCHS = int(os.environ.get("EPOCHS", 1))  # InstructGPT: reward models overfit after ~1 epoch
LR = float(os.environ.get("LR", 1e-4))
MIN_LR_RATIO = 0.1
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.03))
CENTER_COEF = float(os.environ.get("CENTER_COEF", 0.01))  # weight of the (r_chosen + r_rejected)^2 penalty
LORA_R = int(os.environ.get("LORA_R", 8))  # only used when there's no SFT adapter to start from
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 16))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))
QUANTIZE = os.environ.get("QUANTIZE", "1") == "1"
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 100))
LOG_EVERY = int(os.environ.get("LOG_EVERY", 20))
SEED = 1337

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device} (quantize={QUANTIZE and device == 'cuda'})")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# data
print(f"loading AlpacaFarm {PREFERENCE_SOURCE} preference pairs...")
train_ds, val_ds = load_preferences(PREFERENCE_SOURCE, max_seq_len=MAX_SEQ_LEN, val_size=VAL_SIZE, seed=SEED,
                                    max_train_examples=MAX_TRAIN_EXAMPLES)
print(f"train pairs: {len(train_ds)}, val pairs: {len(val_ds)}")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=preference_collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=preference_collate_fn)

# a reward model that just learned "longer is better" would match this - it's the bar to beat
length_wins = [1.0 if len(c) > len(r) else 0.5 if len(c) == len(r) else 0.0 for c, r in val_ds.pairs]
print(f"val accuracy of the 'longer response wins' baseline: {sum(length_wins) / len(length_wins):.3f}")

# ----------------------------------------------------------------------------
# model: SFT trunk + scalar head
init_adapter = SFT_ADAPTER if os.path.exists(SFT_ADAPTER) else None
if init_adapter:
    print(f"initializing the reward model trunk from the SFT adapter {init_adapter}")
else:
    print(f"no SFT adapter at {SFT_ADAPTER} - initializing the reward model trunk from the base checkpoint")
rm, _ = load_model(BASE_CHECKPOINT, init_adapter, scalar_head=True, quantize=QUANTIZE, lora_r=LORA_R,
                   lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, device=device)
trainable, n_trainable, n_total = trainable_parameters(rm)
print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
max_steps = max(1, steps_per_epoch * EPOCHS)
warmup_steps = max(1, int(max_steps * WARMUP_RATIO))


def get_lr(it):
    min_lr = LR * MIN_LR_RATIO
    if it < warmup_steps:
        return LR * (it + 1) / warmup_steps
    decay_ratio = min(1.0, (it - warmup_steps) / max(1, max_steps - warmup_steps))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (LR - min_lr)


def pairwise_loss(scores):
    chosen, rejected = scores.chunk(2)  # preference_collate_fn stacks chosen rows first
    loss = -F.logsigmoid(chosen - rejected).mean() + CENTER_COEF * (chosen + rejected).pow(2).mean()
    return loss, (chosen > rejected).float().mean()


@torch.no_grad()
def evaluate():
    """(val pairwise loss, val accuracy, mean and std of all val rewards)"""
    rm.eval()
    chosen, rejected = [], []
    for input_ids, attention_mask in val_loader:
        with autocast(device):
            scores = rm.score(input_ids.to(device), attention_mask.to(device))
        c, r = scores.chunk(2)
        chosen.append(c)
        rejected.append(r)
    rm.train()
    chosen, rejected = torch.cat(chosen), torch.cat(rejected)
    all_scores = torch.cat([chosen, rejected])
    loss = -F.logsigmoid(chosen - rejected).mean().item()
    return loss, (chosen > rejected).float().mean().item(), all_scores.mean().item(), all_scores.std().item()


# ----------------------------------------------------------------------------
# training loop
log_path = os.path.join(OUT_DIR, "reward_log.txt")
adapter_path = os.path.join(OUT_DIR, "reward_adapter.pt")
open(log_path, "w").close()

step = 0
best_val = float("inf")
rm.train()
t0 = time.time()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    for micro_step, (input_ids, attention_mask) in enumerate(train_loader):
        with autocast(device):
            scores = rm.score(input_ids.to(device), attention_mask.to(device))
        loss, acc = pairwise_loss(scores)
        (loss / GRAD_ACCUM_STEPS).backward()

        if (micro_step + 1) % GRAD_ACCUM_STEPS != 0:
            continue

        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % LOG_EVERY == 0:
            print(f"epoch {epoch} step {step:5d}/{max_steps} | loss {loss.item():.4f} | acc {acc.item():.3f} "
                  f"| lr {lr:.2e} | {time.time() - t0:.1f}s")
        if step % EVAL_EVERY == 0 or step == max_steps - 1:
            val_loss, val_acc, reward_mean, reward_std = evaluate()
            improved = val_loss < best_val
            print(f"  val loss {val_loss:.4f} | val acc {val_acc:.3f} | reward mean {reward_mean:+.3f} std {reward_std:.3f}"
                  + ("  <- best, adapter saved" if improved else ""))
            with open(log_path, "a") as f:
                f.write(f"{step} train {loss.item():.4f}\n{step} val {val_loss:.4f}\n{step} val_acc {val_acc:.4f}\n")
            if improved:
                best_val = val_loss
                save_adapter(adapter_path, rm, kind="reward", step=step, val_loss=val_loss, val_accuracy=val_acc,
                             reward_mean=reward_mean, reward_std=reward_std, preference_source=PREFERENCE_SOURCE)
        step += 1

print(f"best val loss {best_val:.4f}, reward adapter at {adapter_path}")
