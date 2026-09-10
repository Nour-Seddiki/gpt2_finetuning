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

## Results

_Fill in after the full A100 run (`python evaluate.py --eval_hellaswag`)._

| model | Alpaca val loss | ROUGE-L | reward-model score | finished responses | HellaSwag |
|---|---|---|---|---|---|
| base | | | | | 0.3014 (pretraining) |
| SFT | | | | | |
| PPO | | | | | |

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
- A 124M reward model is weak. In a 1k-pair local dry run it reached 0.615 val accuracy against a
  0.632 length baseline. Expect modest gains from PPO at this scale, and read the samples rather
  than trusting the reward curve alone.
- AlpacaFarm's instructions overlap the Alpaca rows used for SFT. The PPO prompts (unlabeled split)
  are disjoint from the reward model's preference split.

## References

- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), 2022
- Huang et al., *The N+ Implementation Details of RLHF with PPO*, 2024
- Dubois et al., *AlpacaFarm*, 2023 · Taori et al., *Stanford Alpaca*, 2023
- Dettmers et al., *QLoRA*, 2023 · Hu et al., *LoRA*, 2021 · Schulman et al., *PPO*, 2017 / *GAE*, 2015
