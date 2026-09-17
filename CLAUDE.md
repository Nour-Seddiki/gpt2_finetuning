# CLAUDE.md — miniGpt_ft

Instruction-tuning + RLHF for the from-scratch GPT-2 (124M) pretrained in `../mini_gpt/`
(checkpoint `../mini_gpt/model_19072.pt`). Three stages, all QLoRA (4-bit NF4 frozen base +
hand-written LoRA), all single-GPU:

1. `train.py` — SFT on `tatsu-lab/alpaca`, or on `HuggingFaceTB/smol-smoltalk` with
   `SFT_DATASET=smoltalk` → `checkpoints/sft_adapter.pt`
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

# SFT on smol-smoltalk (460k convs -> 398k first exchanges): ~3h15m on the laptop, r=64
SFT_DATASET=smoltalk OUT_DIR=checkpoints/sft_smoltalk EPOCHS=1 LORA_R=64 LORA_ALPHA=128 \
  BATCH_SIZE=4 GRAD_ACCUM_STEPS=4 VAL_SIZE=1000 EVAL_EVERY=1000 python train.py
LORA_IMPL=peft python train.py   # same LoRA via peft; needs BATCH_SIZE=4 GRAD_ACCUM_STEPS=4 locally

# full pipeline (A100): python train.py && python train_reward.py && python train_ppo.py
# full pipeline on the local 8 GB Windows laptop - runtime.py knobs, see Gotchas
export CUDA_MEM_FRACTION=0.8 PIN_CPUS=0x0FFF KEEP_AWAKE=1 PYTHONIOENCODING=utf-8
BATCH_SIZE=4 GRAD_ACCUM_STEPS=4 python train.py          # ~52 min
BATCH_SIZE=4 GRAD_ACCUM_STEPS=4 python train_reward.py   # ~14 min
KL_COEF=0.2 python train_ppo.py                          # ~69 min
python evaluate.py --batch_size 8 --eval_hellaswag       # ~4 min per model, mostly HellaSwag
python generate.py --instruction "..." [--adapter_checkpoint checkpoints/ppo_adapter.pt | none] [--merge]
python evaluate.py [--adapters a.pt b.pt] [--eval_hellaswag]
```

## Conventions

- **Self-contained model code.** `model.py` is a copy of `../mini_gpt/model.py`'s classes. Never
  import `mini_gpt/model.py` (it runs training/DDP setup at import) and never edit it (separate
  repo, pretraining must stay reproducible). Only `../mini_gpt/hellaswag.py` is imported, by `evaluate.py`.
- **From scratch, no `peft`/`trl`.** LoRA, the reward head, PPO/GAE are all hand-written; keep it
  that way. The one exception is the opt-in `LORA_IMPL=peft` backend (`model.apply_peft_lora`), which
  exists to show the hand-written LoRA is equivalent - same targets, same 1,218,048 trainable params
  at r=8. `native` stays the default and peft stays an optional dependency.
- **Labels are pre-shifted** in the dataset (`labels[t]` = token after `input_ids[t]`), matching
  mini_gpt's `x = buf[:-1], y = buf[1:]`. `GPT.forward` does *not* shift. Training on unshifted
  labels teaches the model to copy its input (loss → ~0) — this happened once; `sanity_checks.py` guards it.
- **Padding:** right padding for training batches, left padding (`data.left_pad`) for batched
  generation. Position ids come from the attention mask, and every query may attend to itself
  (`build_attn_mask`) so all-pad rows can't produce NaN.
- **Prompt and response are tokenized separately** (`encode_prompt` + `encode_response`), so the
  training-time prompt ids equal inference-time ids.
- **Adapter checkpoint format:** `{"lora_state_dict", "lora_r", "lora_alpha", "lora_impl", "kind", ...}`
  holding only trainable params (LoRA A/B, LayerNorms, scalar `head.*`). Load with `model.load_model(...)`,
  which reads r/alpha/impl from the checkpoint and rejects mismatched keys. Adapters written before
  the peft backend have no `lora_impl` and load as `native`.
- **Sampling masks the 47 padding vocab ids** (50257–50303): never trained, and tiktoken can't decode them.
- **Sampling knobs:** `GPT.generate` takes `top_p` and `repetition_penalty` (generated tokens only,
  not the prompt). Both default to off so PPO rollouts stay on-policy; the two CLIs default to
  `--top_p 0.9 --repetition_penalty 1.1`, which cut repeated 4-grams from 6.1% to 0.1%.
- **SFT examples are stored as `uint16`** (`data.AlpacaDataset`), not Python int lists: all of
  smol-smoltalk is ~0.3 GB that way and ~6 GB as lists, on a laptop with ~3 GB free.
- `checkpoints/`, `models/` and `*.pt` are gitignored: weights stay local, never commit them.
  `models/` holds the exported set (the 3 adapters + `ppo_merged_bf16.pt`, PPO iteration 250 merged
  and stored in bf16).

## Gotchas

- `tatsu-lab/alpaca_farm` ships a loading script that datasets ≥ 4 refuses to run — `data.py`
  reads its raw json files via `hf://datasets/tatsu-lab/alpaca_farm/...` instead.
- bitsandbytes 4-bit only runs on CUDA; `load_model` silently falls back to unquantized LoRA on CPU.
  `merge_lora` needs an unquantized base (`generate.py --merge` handles this).
- PPO keeps LoRA dropout at 0 — dropout makes rollout and update logprobs disagree.
- PowerShell's `Tee-Object` writes UTF-16 logs; prefer the scripts' own `*_log.txt` files.
- **The local 8 GB RTX 5050 never OOMs - it spills.** The Windows (WDDM) driver silently falls
  back to shared system RAM when VRAM is full: steps slow ~2-3x (9x for PPO), then the process
  can be killed for low system memory (SFT, step ~2000). Two separate causes:
  - *Real usage*: a 16 x 512-token SFT batch peaks at 8.7 GiB (`BATCH_SIZE=8`: 7.9 GB VRAM +
    1.5 GB shared), so `train.py` needs `BATCH_SIZE=4 GRAD_ACCUM_STEPS=4` (same effective batch,
    ~5 GB). `train_reward.py` at `BATCH_SIZE=4 GRAD_ACCUM_STEPS=4` peaks at ~2.9 GB.
  - *Allocator cache*: PyTorch never gets the OOM that makes it free its cached blocks, so
    reserved memory outgrows the card even when live tensors fit. `train_ppo.py` defaults:
    11.46 GiB reserved for 4.65 GiB allocated, 5.9 s per policy+value update. With
    `CUDA_MEM_FRACTION=0.8`: 5.88 GiB, 0.65 s, and the defaults (`MINI_BATCH=16`) fit. Don't
    shrink `MINI_BATCH` for memory instead - PPO has no grad accumulation, so it changes the
    number of updates per rollout, i.e. the recipe.
  Check with the `\GPU Process Memory(*)\Shared Usage` perf counter (~78 MB is the baseline).
- **Pin training runs to the P-cores (`PIN_CPUS=0x0FFF`).** Launched from a background shell,
  Windows runs the process on the i7-13620H's efficiency cores: SFT went 1.37 s/step at 13% GPU
  utilization (the 124M QLoRA loop is kernel-launch bound on one CPU thread) vs 0.27 s/step
  pinned. For an already-running process: `$p = Get-Process -Id <pid>; $p.ProcessorAffinity =
  [IntPtr]0x0FFF; $p.PriorityClass = 'AboveNormal'`.
- **Unattended runs need `KEEP_AWAKE=1`.** ~30 min after the last keyboard/mouse input the
  laptop enters Modern Standby, which suspends desktop apps: a PPO run froze for 39 min and was
  then killed for low system RAM on wake. RAM is tight regardless - ~0.7-2 GB available with the
  4-model PPO job loaded. Closing the lid still sleeps.
- **SFT dataset choice is a trade-off, not an upgrade.** The smol-smoltalk model (3h14m, r=64) wins
  chat/code/explanation prompts and writes 70-121 word answers; the Alpaca model wins short factual
  and format-constrained ones because 34-word answers have less room to be wrong. Each also wins val
  loss and ROUGE on its own dataset by a wide margin, so judge with `evaluate.py --val_dataset` on
  both plus a blind side-by-side, never one dataset's metrics alone. Eval used for the README:
  `python evaluate.py --adapters checkpoints/sft_adapter.pt models/ppo_adapter_iter250.pt
  checkpoints/sft_smoltalk/sft_adapter.pt --batch_size 8 --gen_batch_size 8 --max_tokens 256
  --top_p 0.9 --repetition_penalty 1.1 [--eval_hellaswag | --val_dataset smoltalk --val_size 1000]`
  (~25 min with HellaSwag, ~8 min without). The reward model and PPO were trained on the Alpaca SFT
  policy, so they still pair with `checkpoints/sft_adapter.pt`.
- **PPO at the default `KL_COEF=0.05` exploits length** with this reward model (0.580 val
  accuracy vs a 0.555 "longer wins" baseline): by iteration 40, KL 4.5 and climbing, eval length
  53 -> 87 tokens, finished 92% -> 78%. `KL_COEF=0.2` held KL ~0.7-2 and eval length ~63-77
  tokens for all 312 iterations. `train_ppo.py` overwrites a single `ppo_adapter.pt` every
  `SAVE_EVERY` iterations - copy the saves if you want to pick an earlier one.
- When redirecting script output to a file on Windows, set `PYTHONIOENCODING=utf-8`: stdout
  otherwise falls back to cp1252 and a generated sample containing e.g. `→` crashes the run
  with UnicodeEncodeError.
- Windows: no `torch.compile` (no Triton) — the scripts don't use it.
