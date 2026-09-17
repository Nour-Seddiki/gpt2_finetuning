# mini_gpt → instruction-following assistant: QLoRA SFT + RLHF (reward model + PPO)

Takes the GPT-2 (124M) I pretrained from scratch in `mini_gpt` (a separate repo; val loss 3.1067,
HellaSwag 30.14%) and turns it from a raw text-continuation model into one that follows
instructions — the full InstructGPT pipeline at miniature scale, written from scratch
(no `peft`, no `trl`; `LORA_IMPL=peft` swaps in peft's LoRA as an equivalence check):

```
pretrained mini_gpt ──SFT──▶ sft_adapter ──┬──────────────▶ policy (trainable) ──PPO──▶ ppo_adapter
   (model_19072.pt)   Alpaca 52k           │                ref    (frozen, KL anchor)
                                           └─reward model─▶ value  (trainable critic)
                                             AlpacaFarm      reward (frozen scorer)
                                             preferences
```

| Stage | Script | Data | Output |
|---|---|---|---|
| 1. Supervised fine-tuning | `train.py` | [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca), 52k instruction/response pairs, or [`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (`SFT_DATASET=smoltalk`), 460k conversations, or both (`SFT_DATASET=mix`) | `checkpoints/sft_adapter.pt` |
| 2. Reward model | `train_reward.py` | [`tatsu-lab/alpaca_farm`](https://huggingface.co/datasets/tatsu-lab/alpaca_farm) preference pairs (19.5k GPT-4-labeled, or 9.7k human-labeled) | `checkpoints/reward_adapter.pt` |
| 3. PPO | `train_ppo.py` | AlpacaFarm's 20k unlabeled instructions (prompts only) | `checkpoints/ppo_adapter.pt` |

Every model is the same 124M trunk with **4-bit NF4 frozen weights + LoRA** (r=8 on the 4 linear
projections of all 12 blocks + LayerNorms: 1,218,048 trainable params, 0.97%). QLoRA isn't a memory
necessity at 124M — full fine-tuning fits anywhere — it's here to demonstrate the technique end to end.

## Trained weights

The trained weights aren't included in this repo. Training writes the LoRA adapters to
`checkpoints/` (`sft_adapter.pt`, `reward_adapter.pt`, `ppo_adapter.pt`, ~5 MB each), and they run
on top of the pretrained base `../mini_gpt/model_19072.pt`:

```bash
python generate.py --adapter_checkpoint checkpoints/ppo_adapter.pt --instruction "What is the capital of France?"
```

For a standalone model that needs no adapter, merge one into the full-precision base and load the
result as the base checkpoint:

```bash
python generate.py --adapter_checkpoint checkpoints/ppo_adapter.pt --merge --save_merged merged.pt --instruction "Hi"
python generate.py --base_checkpoint merged.pt --adapter_checkpoint none --instruction "What is the capital of France?"
```

Merging costs a little accuracy, because the adapter was trained against the 4-bit base. For the
best PPO checkpoint (iteration 250), merged and stored in bf16, the Alpaca val loss went from 2.1037
to 2.1111.

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

Two opt-in variants of stage 1:

```bash
SFT_DATASET=mix EPOCHS=1 python train.py        # Alpaca + smol-smoltalk (see Results)
SFT_DATASET=smoltalk EPOCHS=1 python train.py   # smol-smoltalk only
LORA_IMPL=peft python train.py                  # the same LoRA, built by peft instead of by hand
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
| 1b. SFT on smol-smoltalk | same | `SFT_DATASET=smoltalk EPOCHS=1 LORA_R=64 LORA_ALPHA=128`, `BATCH_SIZE=4 GRAD_ACCUM_STEPS=4`, 398k examples | 24,887 | **~3 h 14 min** |
| 1c. SFT on the mix | same | `SFT_DATASET=mix`, otherwise as above, 207k examples | 12,917 | **~1 h 33 min** |

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


## Better SFT data: Alpaca vs smol-smoltalk

Alpaca's responses are `text-davinci-003` generations from 2023, and they are short and often wrong.
`SFT_DATASET=smoltalk` runs stage 1 on [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
instead — the mix built for SmolLM2-135M/360M-Instruct, with responses from much stronger models. The
first exchange of each conversation becomes one Alpaca-template row (a system prompt is the
instruction and the user turn its input, which is what the rewrite and summarize subsets need); 398k
of the 460k conversations fit in 512 tokens. One epoch at LoRA r=64 (9,475,584 trainable, 7.1%) took
3 h 14 min on the laptop and ended at val loss 1.5190 (ppl 4.57), still improving at the last step.

Training on smol-smoltalk alone costs the terse, format-obedient answers Alpaca teaches, so
`SFT_DATASET=mix` trains on both: all of Alpaca plus enough smol-smoltalk to leave Alpaca at
`MIX_ALPACA_FRACTION` (0.25) of the examples — 51,684 + 154,987 = 206,671 examples, 1 h 33 min.

Measured on 2026-09-18 with `--max_tokens 256 --top_p 0.9 --repetition_penalty 1.1` for every model,
so these numbers are not comparable to the table above (which used plain top-k at 128 tokens). 100
sampled responses per model per dataset; HellaSwag is the full 10,042-example val set.

| model | Alpaca val loss / ROUGE-L | smoltalk val loss / ROUGE-L | finished | words | HellaSwag |
|---|---|---|---|---|---|
| base | 2.5295 / 0.079 | 2.5507 / 0.156 | 30% / 24% | 161 / 170 | 0.3014 |
| SFT (Alpaca) | 2.0769 / **0.280** | 2.3023 / 0.150 | 99% / 97% | 34 / 45 | 0.2971 |
| PPO, iteration 250 | 2.1037 / 0.263 | 2.3346 / 0.174 | 98% / 89% | 52 / 60 | 0.2965 |
| SFT (smol-smoltalk) | 2.2576 / 0.229 | **1.5190** / **0.284** | 86% / 73% | 70 / 121 | 0.2871 |
| **SFT (mix)** | **2.0189** / 0.257 | 1.6206 / 0.279 | 97% / 74% | 41 / 119 | 0.2893 |

The two single-dataset models each win on the distribution they were trained on, by a wide margin in
both directions — at this scale val loss and ROUGE measure how closely a model matches a reference
style as much as how good its answers are. The mix is the interesting row: it beats the Alpaca model
on Alpaca's own val loss (2.0189 vs 2.0769) while landing within 0.005 ROUGE-L of the smol-smoltalk
model on smol-smoltalk, and its answers average 41 words rather than 121.

The tie-break is a blind comparison: 38 hand-written prompts over 11 categories, identical decoding
for every model, the responses shuffled per prompt and labelled A/B/C, and the key opened only after
judging. Two rounds, three models each:

| model | round 1 | round 2 |
|---|---|---|
| SFT (Alpaca) | 12.0 | 12.5 |
| **SFT (mix)** | — | **12.0** |
| SFT (smol-smoltalk) | 11.5 | 5.5 |
| PPO, iteration 250 | 6.5 | — |

On 8 of the 38 prompts every model was wrong — arithmetic, "why is the sky blue?", classifying a
crocodile, and two deliberately unanswerable questions — so those count for nobody. Behind the
round-1 tie is a clean split:

- **smol-smoltalk wins chat 3/3, code 2/2**, explanation 2/3 and advice 2.5/4. It is the only model
  that answers "Hi! How are you?" as a greeting rather than inventing a person ("I'm currently
  working on a project in the lab"), and the only one that writes a working `return s[::-1]`.
- **Alpaca wins factual 2/2, constraint 3.5/5, rewrite 2/3** and classify 1/1. Its 34-word answers
  leave less room to be wrong: asked who wrote Romeo and Juliet it says "William Shakespeare, a
  poet", where the smol-smoltalk model writes three paragraphs calling it a Greek tragedy by Homer.

Round 2 is what the mix was for. It takes over most of what smol-smoltalk used to win — chat,
explanation, rewriting (where it is the only model that actually fixes the grammar: "I went to the
store yesterday and bought some apple's") — while matching Alpaca on constraint-following, which
drops smol-smoltalk from 11.5 to 5.5. Alpaca keeps factual recall, where brevity is the only defence
a 124M model has.

Length is the whole trade. Better data buys fluency, structure and conversational range, and costs
terseness; mixing a quarter of Alpaca back in buys the terseness back for one extra hour of training.
None of the three knows more than the others — the base checkpoint sets that ceiling, and all three
still fail at 2+2.

Two caveats on the blind comparison: 38 prompts judged by one reader is a small, noisy sample, so the
0.5-point gaps in each round mean nothing (the 12.0-vs-5.5 gap does), and the judging is blind to
which model wrote what, not to what the judge expected to see.

## Decoding

`GPT.generate` supports nucleus sampling and a repetition penalty (over generated tokens only, so
rewrites can still reuse the prompt's words). Both are off by default, which keeps PPO's rollouts
on-policy; the CLIs default to `--top_p 0.9 --repetition_penalty 1.1`. Measured over the same 38
prompts with the smol-smoltalk model, where "repeats" is the share of word 4-grams that repeat an
earlier one:

| sampling | repeats | words | finished |
|---|---|---|---|
| temperature 0.7, top-k 50 (the old default) | 6.1% | 103 | 97% |
| temperature 0.7, top-k 50, repetition_penalty 1.1 | **0.1%** | 94 | 97% |
| temperature 0.7, top-k 50, repetition_penalty 1.2 | 0.0% | 84 | 100% |
| temperature 0.7, top-p 0.9 | 6.3% | 90 | 95% |
| temperature 0.7, top-p 0.9, repetition_penalty 1.1 | 0.6% | 90 | 97% |
| temperature 0.5, top-p 0.9, repetition_penalty 1.1 | 0.7% | 79 | 100% |
| greedy | 17.1% | 96 | 95% |
| greedy, repetition_penalty 1.1 | 4.4% | 84 | 97% |

This costs nothing and fixes the most obvious failure: without a penalty a haiku prompt returns "A
voice of hope, / A voice of hope, / A voice of hope to this day", and the base model repeats whole
lines a third of the time. It also improves instruction following on its own — "Answer with only the
word yes or no: is the Earth flat?" goes from "The Earth is flat." to "No."

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
- Sampling supports nucleus sampling (`top_p`) and a repetition penalty. The penalty applies
  only to tokens the model generated, never to the prompt, so a rewrite or summary can still
  reuse the input's words. Both are off in `GPT.generate`'s defaults, which keeps PPO's rollout
  distribution exactly the policy's; the `generate.py` and `evaluate.py` CLIs default to
  `--top_p 0.9 --repetition_penalty 1.1`.
- `LORA_IMPL=peft` wraps the same four projections with peft's LoRA instead of `LoRALinear`,
  for the same 1,218,048 trainable parameters at r=8. The adapter checkpoint records which
  backend wrote it (`lora_impl`), so the RLHF stages, `generate.py` and `evaluate.py` rebuild
  the right wrapping automatically, and `merge_lora` folds in either. `peft` is an optional
  dependency; the hand-written path is the default and needs nothing installed.
- Tokenized SFT examples are stored as `uint16` arrays (2 bytes/token). As Python int lists,
  smol-smoltalk's 398k examples would need ~6 GB of RAM instead of ~0.3 GB.
- `wte`/`lm_head` are weight-tied, so they're never quantized or LoRA-wrapped.
- The 47 padding vocab rows (50257–50303) are masked out of sampling and PPO logprobs.
- `generate.py --merge --save_merged merged.pt` folds the adapter into full-precision weights and
  saves a checkpoint that loads like the original `mini_gpt` one.

## Caveats

- Alpaca and AlpacaFarm are **CC-BY-NC-4.0** — portfolio/research use only. smol-smoltalk is
  Apache-2.0.
- `SFT_DATASET=mix` is the best all-round SFT recipe here, but the RLHF stages were trained on the
  Alpaca SFT policy, so `checkpoints/sft_adapter.pt` stays the default adapter; pairing the reward
  model and PPO with a mix policy means retraining both stages.
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
