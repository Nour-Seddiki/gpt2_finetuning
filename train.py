"""
Stage 1: supervised instruction fine-tuning (SFT) of the mini_gpt checkpoint on Alpaca with
QLoRA. Single GPU, no DDP needed - model (124M) and dataset (52k rows) are both tiny relative
to a rented A100. The adapter it writes (checkpoints/sft_adapter.pt, best val loss) is the
starting point for the RLHF stages: train_reward.py, then train_ppo.py.
See PROJECT_PLAN.md and README.md for the design writeup.

Usage:
    python train.py                              # defaults below
    BATCH_SIZE=32 EPOCHS=3 python train.py        # override via env vars
    QUANTIZE=0 python train.py                    # plain LoRA, no bitsandbytes required
    MAX_TRAIN_EXAMPLES=320 VAL_SIZE=64 EPOCHS=1 EVAL_EVERY=10 python train.py   # local dry run
"""
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from data import collate_fn, format_prompt, load_alpaca
from generate import generate_responses
from model import autocast, lm_loss, load_model, save_adapter, trainable_parameters
from runtime import configure_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# config - env-var overridable, same convention as mini_gpt/model.py
BASE_CHECKPOINT = os.environ.get("BASE_CHECKPOINT", os.path.join(SCRIPT_DIR, "..", "mini_gpt", "model_19072.pt"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(SCRIPT_DIR, "checkpoints"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 512))
VAL_SIZE = int(os.environ.get("VAL_SIZE", 2000))
MAX_TRAIN_EXAMPLES = os.environ.get("MAX_TRAIN_EXAMPLES")
MAX_TRAIN_EXAMPLES = int(MAX_TRAIN_EXAMPLES) if MAX_TRAIN_EXAMPLES else None
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 2e-4))
MIN_LR_RATIO = 0.1
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.03))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
LORA_R = int(os.environ.get("LORA_R", 8))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 16))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))
QUANTIZE = os.environ.get("QUANTIZE", "1") == "1"
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 200))
LOG_EVERY = int(os.environ.get("LOG_EVERY", 20))
SAMPLE_INSTRUCTION = os.environ.get("SAMPLE_INSTRUCTION", "Give three tips for staying healthy.")
SEED = 1337

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device} (quantize={QUANTIZE and device == 'cuda'})")
configure_runtime(device)
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# data
print("loading tatsu-lab/alpaca from HuggingFace...")
train_ds, val_ds = load_alpaca(max_seq_len=MAX_SEQ_LEN, val_size=VAL_SIZE, seed=SEED, max_train_examples=MAX_TRAIN_EXAMPLES)
print(f"train examples: {len(train_ds)}, val examples: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ----------------------------------------------------------------------------
# model: the mini_gpt pretrained checkpoint with QLoRA injected
print(f"loading base checkpoint from {BASE_CHECKPOINT}")
model, _ = load_model(BASE_CHECKPOINT, quantize=QUANTIZE, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                      lora_dropout=LORA_DROPOUT, device=device)

trainable, n_trainable, n_total = trainable_parameters(model)
print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

# ----------------------------------------------------------------------------
# optimizer + LR schedule (decay/no-decay split, same convention as mini_gpt/model.py)
decay_params = [p for p in trainable if p.dim() >= 2]
nodecay_params = [p for p in trainable if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=LR, betas=(0.9, 0.95), eps=1e-8,
)

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
max_steps = max(1, steps_per_epoch * EPOCHS)
warmup_steps = max(1, int(max_steps * WARMUP_RATIO))


def get_lr(it):
    min_lr = LR * MIN_LR_RATIO
    if it < warmup_steps:
        return LR * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (LR - min_lr)


def sample():
    """Greedy response to a fixed instruction - a quick read on instruction-following."""
    model.eval()
    (text, _), = generate_responses(model, [format_prompt(SAMPLE_INSTRUCTION)], max_tokens=64,
                                    temperature=0, device=device)
    model.train()
    return text


# ----------------------------------------------------------------------------
# training loop
log_path = os.path.join(OUT_DIR, "sft_log.txt")
adapter_path = os.path.join(OUT_DIR, "sft_adapter.pt")
open(log_path, "w").close()

step = 0
best_val = float("inf")
model.train()
t0 = time.time()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    for micro_step, (input_ids, labels, attention_mask) in enumerate(train_loader):
        input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
        with autocast(device):
            _, loss = model(input_ids, targets=labels, attention_mask=attention_mask)
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
            dt = time.time() - t0
            print(f"epoch {epoch} step {step:5d}/{max_steps} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")
        if step % EVAL_EVERY == 0 or step == max_steps - 1:
            val_loss = lm_loss(model, val_loader, device)
            improved = val_loss < best_val
            print(f"  val loss: {val_loss:.4f} (ppl {math.exp(val_loss):.2f})" + ("  <- best, adapter saved" if improved else ""))
            print(f"  sample: {sample()!r}")
            with open(log_path, "a") as f:
                f.write(f"{step} train {loss.item():.4f}\n{step} val {val_loss:.4f}\n")
            if improved:
                best_val = val_loss
                # only the LoRA adapter (+ fine-tuned layernorms): small, versionable
                # independently of the frozen base weights
                save_adapter(adapter_path, model, kind="sft", step=step, val_loss=val_loss)
        step += 1

print(f"best val loss {best_val:.4f}, adapter at {adapter_path}")
