# mini_gpt → instruction-following assistant: QLoRA SFT + RLHF (reward model + PPO)

Takes the GPT-2 (124M) I pretrained from scratch in `mini_gpt` (a separate repo; val loss 3.1067,
HellaSwag 30.14%) and turns it from a raw text-continuation model into one that follows
instructions — the full InstructGPT pipeline at miniature scale, written from scratch
(no `peft`, no `trl`):

```
pretrained mini_gpt ──SFT──▶ sft_adapter ──┬──────────────▶ policy (trainable) ──PPO──▶ ppo_adapter
   (model_19072.pt)   Alpaca 52k           │                ref    (frozen, KL anchor)
                                           └─reward model─▶ value  (trainable critic)
                                             AlpacaFarm      reward (frozen scorer)
                                             preferences
```

| Stage | Script | Data | Output |
|---|---|---|---|
| 1. Supervised fine-tuning | `train.py` | [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca), 52k instruction/response pairs | `checkpoints/sft_adapter.pt` |
| 2. Reward model | `train_reward.py` | [`tatsu-lab/alpaca_farm`](https://huggingface.co/datasets/tatsu-lab/alpaca_farm) preference pairs (19.5k GPT-4-labeled, or 9.7k human-labeled) | `checkpoints/reward_adapter.pt` |
| 3. PPO | `train_ppo.py` | AlpacaFarm's 20k unlabeled instructions (prompts only) | `checkpoints/ppo_adapter.pt` |

Every model is the same 124M trunk with **4-bit NF4 frozen weights + LoRA** (r=8 on the 4 linear
projections of all 12 blocks + LayerNorms: 1,218,048 trainable params, 0.97%). QLoRA isn't a memory
necessity at 124M — full fine-tuning fits anywhere — it's here to demonstrate the technique end to end.

## Model files

The trained weights are in [`models/`](models/):

| File | What it is | Size |
|---|---|---|
| `ppo_merged_bf16.pt` | **The final model.** The PPO policy at iteration 250 (the best checkpoint in the [results](#results)) with its LoRA merged into the full-precision base, stored in bf16. Runs standalone, no base checkpoint needed. Stored with Git LFS. | 238 MB |
| `ppo_adapter_iter250.pt` | The same PPO policy as a LoRA adapter | 4.7 MB |
| `sft_adapter.pt` | SFT adapter (Alpaca val loss 2.0769) | 4.7 MB |
| `reward_adapter.pt` | Reward model: LoRA + scalar head (val accuracy 0.580) | 4.7 MB |

The merged model is a Git LFS file: run `git lfs install` before cloning, or `git lfs pull` after.
It loads like a pretrained checkpoint, so it needs no adapter:

```bash
python generate.py --base_checkpoint models/ppo_merged_bf16.pt --adapter_checkpoint none \
    --instruction "What is the capital of France?"
```

The adapters run on top of the pretrained base `../mini_gpt/model_19072.pt`, which isn't included
in either repo:

```bash
python generate.py --adapter_checkpoint models/ppo_adapter_iter250.pt --instruction "What is the capital of France?"
```

Merging costs a little accuracy. The adapter was trained against the 4-bit base, so on the
full-precision base, rounded to bf16, its Alpaca val loss is 2.1111 instead of 2.1037.

## Quickstart

```bash
pip install -r requirements.txt
python sanity_checks.py          # plumbing checks (labels, padding, KV cache, LoRA merge)

python train.py                  # 1. SFT        (~3 epochs over Alpaca)
python train_reward.py           # 2. reward model (1 epoch, initialized from the SFT model)
python train_ppo.py              # 3. PPO        (~20k episodes)

python evaluate.py --eval_hellaswag
python generate.py --instruction "Give three tips for staying healthy." --adapter_checkpoint checkpoints/ppo_adapter.pt
```

All training knobs are env vars (`BATCH_SIZE=32 EPOCHS=3 python train.py`); see the config block at
the top of each script. For a cheap local dry run before renting a GPU, cap the data — e.g.
`MAX_TRAIN_EXAMPLES=2000 EPOCHS=1 OUT_DIR=/tmp/ft python train.py` (the commands used for this are
in [`CLAUDE.md`](CLAUDE.md)).

## Training time

| Stage | Hardware | Settings | Optimizer steps | Wall time |
|---|---|---|---|---|
| 1. SFT | RTX 5050 Laptop GPU (8 GB), i7-13620H, Windows 11 | `BATCH_SIZE=4 GRAD_ACCUM_STEPS=4` (effective batch 16), 3 epochs, 4-bit NF4 | 9,363 | **~52 min** (~17 min/epoch) |
| 2. Reward model | same | `BATCH_SIZE=4 GRAD_ACCUM_STEPS=4` (effective 16 pairs), 1 epoch over 18,457 GPT-4-labeled pairs | 1,153 | **~14 min** |
| 3. PPO | same | defaults (64 prompts/iteration, `MINI_BATCH=16`, 4 PPO epochs) except `KL_COEF=0.2`; `CUDA_MEM_FRACTION=0.8` | 312 iterations (4,992 updates) | **~69 min** |

Measured on 2026-09-10. Training ran at 0.27 s/step, each val-loss eval (every 200 steps) took
~13 s, and startup (Alpaca load and tokenization, 4-bit model load) took ~1 min. The logged run
took 61 min end to end, because its first 340 steps ran before the CPU fix below. The ~52 min is
that run's measured speed after the fix, applied to all 9,363 steps. It reached best val loss
2.0769 (ppl 7.98), down from 2.5732, at the final step.

The reward model and PPO were measured on 2026-09-11. The reward model ran at 0.41 s/step, plus
~10 s per val eval (every 100 steps) and ~3.5 min of startup: 14 min 17 s end to end. PPO ran at
~11–13 s per iteration (sample 64 responses, score them, 16 minibatch updates), plus ~20 s per
eval every 10 iterations: 68 min 49 s end to end.

Three things decide whether a run on an 8 GB Windows laptop is fast and finishes at all.
`runtime.py` handles each with an opt-in env var (all off by default, so A100 runs are unaffected):

- **GPU memory (`CUDA_MEM_FRACTION=0.8`).** Windows doesn't raise OOM when VRAM runs out. The
  driver spills into shared system RAM, which makes steps ~3× slower and can get the process
  killed for low memory. A 16 × 512-token SFT batch needs ~8.7 GiB, so SFT uses `BATCH_SIZE=4`
  (~5 GB). Separately, PyTorch's allocator never sees an OOM, so it never frees its cache: PPO's
  defaults reserved 11.5 GiB for 4.7 GiB of live tensors and ran 9× slower. Capping the
  allocator at 80% of VRAM makes it free its cache instead (5.9 GiB reserved, same batch sizes).
- **CPU cores (`PIN_CPUS=0x0FFF`).** At 124M parameters each forward/backward pass is thousands
  of tiny kernels, so one CPU thread limits the step time, not the GPU. When the script was
  started from a background terminal, Windows ran it on the i7's efficiency cores: 1.37 s/step
  at 13% GPU utilization. Pinning it to the performance cores (logical CPUs 0–11 on the 13620H)
  made it 5× faster.
- **Standby (`KEEP_AWAKE=1`).** About 30 minutes after the last keyboard or mouse input, the
  laptop enters Modern Standby and Windows suspends the training process. One PPO run froze for
  39 minutes this way and was then killed for low memory when the laptop woke.

## Results

Measured on 2026-09-11 on the laptop with `python evaluate.py --adapters <sft, 3 PPO saves>
--batch_size 8 --eval_hellaswag` (~20 min). ROUGE, reward score and finished rate come from 100
Alpaca val prompts sampled at temperature 0.7, with the same seed for every model. HellaSwag is the
full 10,042-example val set. PPO is the `KL_COEF=0.2` run, at three of its saved iterations.

| model | Alpaca val loss | ROUGE-L | reward-model score | finished responses | HellaSwag |
|---|---|---|---|---|---|
| base | 2.5295 | 0.095 | +0.094 | 8% | 0.3014 |
| SFT | 2.0769 | 0.278 | −0.145 | 92% | 0.2971 |
| PPO, iteration 150 | 2.0982 | 0.237 | +0.211 | 88% | 0.2976 |
| PPO, iteration 250 | 2.1037 | 0.270 | +0.208 | 90% | 0.2965 |
| PPO, iteration 312 (final) | 2.1040 | 0.260 | +0.194 | 85% | 0.2957 |

- **SFT is the big win.** ROUGE-L goes from 0.095 to 0.278 and finished responses from 8% to 92%.
  The base model doesn't answer; it continues the prompt, often by writing more `### Response:`
  blocks.
- **PPO raised the reward-model score by ~0.35, but not ROUGE.** Every PPO checkpoint has lower
  ROUGE-L than SFT. Read the reward column with suspicion: the reward model rates the base model's
  unfinished rambling (+0.094) above SFT's answers (−0.145). That's the same length bias PPO
  exploited at the default `KL_COEF=0.05`, a run stopped at iteration ~45 with KL at 4.5 and
  climbing and eval length up from 53 to 87 tokens.
- **Iteration 250 is the best PPO checkpoint.** It has the same reward gain as iteration 150, with
  ROUGE-L and finished rate close to SFT's. The last 62 iterations, as the LR annealed to 0, made
  nothing better.
- **Little forgetting.** HellaSwag drops 0.4 points from the base model with SFT and stays flat
  through PPO (0.2957–0.2976).
- **Side by side**, PPO answers run longer: "Explain machine learning in one sentence." gets two
  sentences. But for "Give me three ideas for a rainy weekend." PPO lists exactly three numbered
  ideas where SFT gave four bullets. Every model still gets "What is 2+2?" wrong.

## How each stage works

**SFT.** Each example is the Stanford Alpaca prompt template + response + `<|endoftext|>`. Labels
are the inputs shifted by one (as in pretraining) with every prompt position set to `-100`, so only
the response is learned. Prompt and response are tokenized separately so training sees exactly the
prompt tokens inference will. Over-long examples are dropped, not truncated.

**Reward model.** The SFT trunk plus a scalar head read at the final `<|endoftext|>`, trained with
the Bradley–Terry loss `-log σ(r_chosen − r_rejected)` plus a small `(r_chosen + r_rejected)²`
centering term. The script prints the accuracy of a "longer response wins" baseline on the same val
pairs. That's the bar to clear, because a reward model that only learned length is a reward model
PPO will exploit. The val reward mean/std are stored with the adapter for PPO's score normalization.

**PPO.** Per iteration: sample responses for 64 prompts (batched, KV-cached), score them, and
build per-token rewards `−β·(log π − log π_ref)` with the normalized score added on the last token
(and a penalty if the response never finished). Then compute GAE advantages (γ=1, λ=0.95, whitened)
and run 4 epochs of clipped-surrogate policy updates and clipped value updates. The value model is
initialized from the reward model. Follows the implementation details in Huang et al. (2024): no
dropout, logprobs recomputed with a full forward (not taken from generation), EOS penalty, reward
normalization, linear LR annealing. Watch `kl` and the eval samples in `checkpoints/ppo_log.txt`. A
rising reward with a KL that keeps climbing and degenerate samples means reward hacking. Raise
`KL_COEF` if that happens.

## Implementation notes

- `model.py` is a self-contained copy of `mini_gpt`'s architecture plus the extras this project needs:
  - padding masks and position ids from the attention mask, so left-padded generation batches work
  - a KV cache
  - `Linear4bit` swap-in and a hand-written `LoRALinear`
  - `ScalarHeadGPT`, used for both the reward model and the value model
  - adapter save/load and LoRA merging
- `wte`/`lm_head` are weight-tied, so they're never quantized or LoRA-wrapped.
- The 47 padding vocab rows (50257–50303) are masked out of sampling and PPO logprobs.
- `generate.py --merge --save_merged merged.pt` folds the adapter into full-precision weights and
  saves a checkpoint that loads like the original `mini_gpt` one.

## Caveats

- Alpaca and AlpacaFarm are **CC-BY-NC-4.0** — portfolio/research use only.
- Alpaca responses are `text-davinci-003` generations, and AlpacaFarm's preferences are GPT-4 (or
  noisy crowd) judgments, so neither is a gold standard.
- A 124M reward model is weak. Trained on all 18.5k pairs it reaches 0.580 val accuracy against a
  0.555 "longer response wins" baseline (a 1k-pair dry run: 0.615 vs 0.632). At the default
  `KL_COEF=0.05`, PPO learned mostly length from it; `KL_COEF=0.2` kept that in check. Expect
  modest gains from PPO at this scale, and read the samples rather than trusting the reward curve
  alone.
- AlpacaFarm's instructions overlap the Alpaca rows used for SFT. The PPO prompts (unlabeled split)
  are disjoint from the reward model's preference split.

## References

- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), 2022
- Huang et al., *The N+ Implementation Details of RLHF with PPO*, 2024
- Dubois et al., *AlpacaFarm*, 2023 · Taori et al., *Stanford Alpaca*, 2023
- Dettmers et al., *QLoRA*, 2023 · Hu et al., *LoRA*, 2021 · Schulman et al., *PPO*, 2017 / *GAE*, 2015
