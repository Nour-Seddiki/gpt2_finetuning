# CLAUDE.md — miniGpt_ft

Instruction-tuning + RLHF for the from-scratch GPT-2 (124M) pretrained in `../mini_gpt/`
(checkpoint `../mini_gpt/model_19072.pt`). Three stages, all QLoRA (4-bit NF4 frozen base +
hand-written LoRA), all single-GPU:

1. `train.py` — SFT on `tatsu-lab/alpaca` → `checkpoints/sft_adapter.pt`
2. `train_reward.py` — reward model on AlpacaFarm preference pairs → `checkpoints/reward_adapter.pt`
3. `train_ppo.py` — PPO against the reward model, KL-anchored to the SFT model → `checkpoints/ppo_adapter.pt`

`generate.py` (CLI) and `evaluate.py` (val loss, ROUGE, reward score, HellaSwag, side-by-side)
consume those adapters. `README.md` has the full design writeup; `PROJECT_PLAN.md` the original plan.

## Commands

The venv is one level up: `../.venv/Scripts/python.exe` (Windows, Python 3.14, torch 2.11 cu128,
local GPU = RTX 5050 8GB; full runs go on a rented A100). Training scripts are configured with
env vars; CLIs use argparse.

```bash
python sanity_checks.py        # run after any change to data.py / model.py (~1 min)
python data.py                 # data pipeline smoke test (all three datasets)

# local dry runs - point OUT_DIR somewhere disposable so real checkpoints aren't overwritten
OUT_DIR=/tmp/ft MAX_TRAIN_EXAMPLES=2000 VAL_SIZE=128 EPOCHS=1 EVAL_EVERY=25 python train.py
OUT_DIR=/tmp/ft MAX_TRAIN_EXAMPLES=1000 VAL_SIZE=200 BATCH_SIZE=8 EVAL_EVERY=25 python train_reward.py
OUT_DIR=/tmp/ft TOTAL_EPISODES=192 ROLLOUT_BATCH=16 MINI_BATCH=8 EVAL_PROMPTS=32 EVAL_EVERY=4 MAX_NEW_TOKENS=64 python train_ppo.py

# full pipeline (A100): python train.py && python train_reward.py && python train_ppo.py
python generate.py --instruction "..." [--adapter_checkpoint checkpoints/ppo_adapter.pt | none] [--merge]
python evaluate.py [--adapters a.pt b.pt] [--eval_hellaswag]
```

## Conventions

- **Self-contained model code.** `model.py` is a copy of `../mini_gpt/model.py`'s classes. Never
  import `mini_gpt/model.py` (it runs training/DDP setup at import) and never edit it (separate
  repo, pretraining must stay reproducible). Only `../mini_gpt/hellaswag.py` is imported, by `evaluate.py`.
- **From scratch, no `peft`/`trl`.** LoRA, the reward head, PPO/GAE are all hand-written; keep it that way.
- **Labels are pre-shifted** in the dataset (`labels[t]` = token after `input_ids[t]`), matching
  mini_gpt's `x = buf[:-1], y = buf[1:]`. `GPT.forward` does *not* shift. Training on unshifted
  labels teaches the model to copy its input (loss → ~0) — this happened once; `sanity_checks.py` guards it.
- **Padding:** right padding for training batches, left padding (`data.left_pad`) for batched
  generation. Position ids come from the attention mask, and every query may attend to itself
  (`build_attn_mask`) so all-pad rows can't produce NaN.
- **Prompt and response are tokenized separately** (`encode_prompt` + `encode_response`), so the
  training-time prompt ids equal inference-time ids.
- **Adapter checkpoint format:** `{"lora_state_dict", "lora_r", "lora_alpha", "kind", ...}` holding
  only trainable params (LoRA A/B, LayerNorms, scalar `head.*`). Load with `model.load_model(...)`,
  which reads r/alpha from the checkpoint and rejects mismatched keys.
- **Sampling masks the 47 padding vocab ids** (50257–50303): never trained, and tiktoken can't decode them.
- `checkpoints/` and `*.pt` are gitignored.

## Gotchas

- `tatsu-lab/alpaca_farm` ships a loading script that datasets ≥ 4 refuses to run — `data.py`
  reads its raw json files via `hf://datasets/tatsu-lab/alpaca_farm/...` instead.
- bitsandbytes 4-bit only runs on CUDA; `load_model` silently falls back to unquantized LoRA on CPU.
  `merge_lora` needs an unquantized base (`generate.py --merge` handles this).
- PPO keeps LoRA dropout at 0 — dropout makes rollout and update logprobs disagree.
- PowerShell's `Tee-Object` writes UTF-16 logs; prefer the scripts' own `*_log.txt` files.
- When redirecting script output to a file on Windows, set `PYTHONIOENCODING=utf-8`: stdout
  otherwise falls back to cp1252 and a generated sample containing e.g. `→` crashes the run
  with UnicodeEncodeError.
- Windows: no `torch.compile` (no Triton) — the scripts don't use it.
