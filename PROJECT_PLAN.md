# mini_gpt QLoRA Instruction Fine-Tuning (Alpaca)

Portfolio project: take the from-scratch GPT-2 (124M) pretrained in `mini_gpt/`
(checkpoint `mini_gpt/model_19072.pt`, final val loss 3.1067, HellaSwag 30.14%) and
instruction-tune it with **QLoRA** on the [Stanford Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
(52,002 instruction/input/output triples), turning a raw text-continuation LM into
one that follows instructions — the same recipe Stanford used to turn base LLaMA
into Alpaca, applied here to a much smaller model.

## 0. Framing: why QLoRA at 124M, and what "Q100" compute means for the plan

- **Honesty check**: QLoRA's whole point (paper: Dettmers et al. 2023) is fitting
  fine-tuning of 33B–65B models on a single consumer GPU. A 124M model fits in
  full fp32 on almost anything — full fine-tuning would work fine memory-wise, on
  the local RTX 5050 or the rented GPU. Doing QLoRA here is a **deliberate learning/portfolio
  exercise** (demonstrate 4-bit NF4 quantization + low-rank adapters end-to-end),
  not a memory necessity. Worth saying explicitly in the README so it reads as
  informed rather than as "used QLoRA because I didn't know better."
- **Compute target: a rented A100** (per your usual rented-GPU workflow used for
  the `mini_gpt` pretraining run — see `mini_gpt` repo memory: Linux, `torch.compile`
  auto-enabled, no Windows/Triton restrictions). Single GPU is plenty — the model
  and dataset are both tiny, no DDP/multi-GPU needed here.
- Because A100 (40/80GB) is so far beyond what this job needs, gradient checkpointing,
  paged optimizers, and multi-GPU sharding — all standard "make 65B fit" QLoRA
  machinery — are **not load-bearing** at this scale. They're mentioned below as
  optional authenticity additions, not requirements.
- **Local RTX 5050 still has a role**: pre-flight dry run on a tiny slice (~200
  rows, a few dozen steps) before spending any rented-GPU time, exactly like the
  "debug on a tiny overfit subset locally" step in the `cancer_dettector` plan.
  bitsandbytes' Windows wheels do cover the RTX 5050's compute capability (12.0 /
  sm120, confirmed against the CUDA 12.8 / 13.0–13.2 Windows target list), so the
  dry run exercises the *real* 4-bit + LoRA code path, not a fp32 stand-in.

## 1. Build plan (phases, not weeks — this is a days-scale project)

### Phase 1 — Data pipeline
- `data.py`: load `tatsu-lab/alpaca` via `datasets.load_dataset`, carve a small
  held-out val split (e.g. 1–2k rows) out of the single `train` split (52,002 rows,
  no official val/test split exists).
- Build the two Stanford prompt-template variants (with/without `input` — ~40% of
  rows have a non-empty `input`, per the dataset card) and tokenize with **tiktoken
  `gpt2`** (same BPE space the pretrained checkpoint already uses — no new
  tokenizer/vocab needed).
- Implement **prompt masking**: encode the prompt prefix (instruction+input+the
  literal `### Response:\n` marker) and the full example separately, so the loss
  mask (`ignore_index`) covers everything up to and including `### Response:\n`
  and only scores the actual answer tokens + a trailing `<|endoftext|>`. Getting
  this wrong (training on the echoed prompt) is the single most common
  instruction-tuning bug.
- Decide sequence length: most Alpaca examples tokenize short under GPT-2 BPE.
  Recommend **max_seq_len = 512** (well inside `block_size=1024`); **filter out**
  (don't truncate) the small tail of examples that don't fit, since truncating
  would corrupt the response you're trying to train on.
- Dry run this phase locally (fast, CPU-only — no GPU needed for tokenization/EDA).

### Phase 2 — QLoRA model plumbing
- `model.py`: a **self-contained copy** of `GPTConfig` / `CausalSelfAttention` /
  `MLP` / `Block` / `GPT` from `mini_gpt/model.py` (don't import that file directly
  — it runs top-level training/DDP setup code on import and asserts on missing
  fineweb shards; also don't touch `mini_gpt/model.py` itself, it's a separately
  git-tracked, already-working pretraining script). Mirrors how `transformer/` and
  `mini_gpt/` are already independent sibling folders, each with its own `model.py`.
- Load `mini_gpt/model_19072.pt` (`state_dict` under the `'model'` key) into that
  class. No optimizer state exists in the checkpoint (never implemented in
  `mini_gpt/model.py`, despite the comment) — irrelevant here anyway, LoRA
  fine-tuning wants a fresh optimizer over new adapter params.
- **Add padding support** — `mini_gpt/model.py`'s `CausalSelfAttention` has *no*
  attention-mask parameter at all (pretraining never needed one, it packs
  contiguous token shards). Fine-tuning on padded batches needs a combined
  causal+padding boolean mask threaded into
  `F.scaled_dot_product_attention(..., attn_mask=combined_mask, is_causal=False)`
  (build the mask yourself rather than relying on `is_causal=True` together with
  a padding mask).
- **4-bit quantize**: replace `c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj` in
  every block with `bitsandbytes.nn.Linear4bit` (NF4, double-quant, `compute_dtype=torch.bfloat16`
  — A100 native bf16, matches the autocast dtype `mini_gpt/model.py` already uses),
  loaded from the pretrained weights and frozen (`requires_grad_(False)`).
- **Exclude `wte`/`lm_head` from quantization** — they're weight-tied
  (`self.transformer.wte.weight = self.lm_head.weight`), so a generic "wrap every
  `nn.Linear`" loop would break that sharing. Leave the embedding/head pair frozen
  and unquantized. Also leave `wpe` and all `LayerNorm`s untouched (optionally cast
  LayerNorms to fp32 for stability — standard QLoRA practice — and let them stay
  trainable, they're cheap).
- **LoRA adapter**: a small hand-written `LoRALinear` wrapper (frozen base +
  trainable low-rank `A`/`B`, `B` zero-initialized so the adapter starts as a
  no-op) rather than pulling in `peft` — consistent with this repo's from-scratch
  style, and it's ~30 lines. `r=8`, `alpha=16`, `dropout=0.05` as a starting point,
  applied to all 4 targeted linears in all 12 blocks: ≈1.18M trainable params
  (~0.95% of the 124M total) — a good concrete number for the README.
- Dry run: forward/backward one padded batch of the tiny local slice on the RTX
  5050, confirm loss goes down over a few dozen steps before touching the rented GPU.

### Phase 3 — Fine-tuning run (rented A100)
- `train.py`: single-GPU loop (no DDP). AdamW over just the trainable params
  (LoRA `A`/`B` + optionally LayerNorms), LR ~1e-4–3e-4 (LoRA tolerates higher LR
  than full fine-tuning), short warmup + cosine decay — same shape as `mini_gpt`'s
  existing `get_lr`, just re-parameterized for a much shorter run.
- Batch size: A100 has plenty of headroom at seq_len 512/124M params — micro-batch
  16–32, grad-accum optional for an effective batch of ~128–256 examples.
- **Epochs**: 3, matching the original Stanford Alpaca recipe (52k rows is small;
  watch val loss for overfitting past that).
- Optional QLoRA-authenticity add-ons (not required at this scale, mention as
  stretch): `bitsandbytes.optim.PagedAdamW32bit` instead of plain `AdamW`,
  gradient checkpointing.
- Checkpoint the **LoRA adapter weights separately** (a few MB — standard PEFT
  practice, easy to version/share independent of the frozen base) and optionally
  also save a merged (`W + B@A * scale` folded into the base) standalone model for
  simpler inference.

### Phase 4 — Evaluation
- Held-out Alpaca val split: loss/perplexity, the fast iteration signal.
- Qualitative side-by-side: same handful of hand-written instructions (not from
  Alpaca) run through **base vs. fine-tuned** model with the same prompt template —
  good portfolio visual showing raw continuation vs. instruction-following behavior.
- Regression check: rerun `mini_gpt/hellaswag.py`-style eval after fine-tuning to
  confirm no catastrophic forgetting of general language modeling (LoRA is
  low-capacity, so expect only a small shift either way).

### Phase 5 (stretch) — Polish
- Small CLI or Gradio demo comparing base vs. fine-tuned generations.
- Ablation: rank `r` (8 vs 16 vs 32) or LoRA target-module set (attention-only vs.
  attention+MLP) vs. val loss — cheap to run given how fast this job is on an A100.

### Phase 6 — RLHF: reward model + PPO (implemented)
- Data: AlpacaFarm (`tatsu-lab/alpaca_farm`, same Alpaca template). Its HF loading script is
  unsupported by datasets ≥ 4, so `data.py` reads the raw json files. Preference pairs
  (`alpaca_gpt4_preference.json`, 19.5k; or the 9.7k human-labeled set) train the reward model;
  the 20k `unlabeled` instructions are the PPO prompts (disjoint from the preference split).
- `train_reward.py`: SFT trunk + scalar head at the closing `<|endoftext|>`, Bradley–Terry
  pairwise loss + small reward-centering term, 1 epoch; stores val reward mean/std for PPO
  normalization and reports a "longer response wins" baseline to beat.
- `train_ppo.py`: policy (SFT + trainable LoRA), frozen ref (SFT), value (reward model init),
  frozen reward model — all 4-bit + LoRA. Per-token KL penalty + normalized score on the last
  token, GAE (γ=1, λ=0.95), clipped policy/value losses, EOS penalty, no dropout
  (Huang et al. 2024 implementation details). Writes `checkpoints/ppo_adapter.pt`.
- Enablers added to `model.py`: left-padding-safe attention + position ids, KV-cache batched
  sampling, `ScalarHeadGPT`, adapter save/load, exact LoRA merge.

## 2. Data details (Alpaca dataset)

- Source: [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca),
  52,002 rows, single `train` split, ~24MB parquet, **license CC-BY-NC-4.0**
  (non-commercial — fine for a portfolio project, don't ship a commercial product
  trained on it).
- Fields: `instruction` (unique per row), `input` (optional context, ~40% non-empty),
  `output` (the `text-davinci-003`-generated answer), `text` (pre-formatted with
  the Stanford prompt template — convenient, but you still need the raw fields to
  compute the prompt/response split point for loss masking).
- Two prompt templates (from the original Stanford Alpaca `train.py`), pick per-row
  based on whether `input` is empty:
  ```text
  # with input
  Below is an instruction that describes a task, paired with an input that provides
  further context. Write a response that appropriately completes the request.

  ### Instruction:
  {instruction}

  ### Input:
  {input}

  ### Response:
  {output}<|endoftext|>

  # without input
  Below is an instruction that describes a task. Write a response that
  appropriately completes the request.

  ### Instruction:
  {instruction}

  ### Response:
  {output}<|endoftext|>
  ```
- Loss mask: `ignore_index` (e.g. `-100`) over every token up to and including the
  `### Response:\n` marker; only the `{output}` tokens + `<|endoftext|>` contribute
  to the cross-entropy loss.
- Known limitation worth naming in the writeup: Alpaca's outputs are
  `text-davinci-003` generations, not human-verified — they contain some factual
  errors/hallucinations by construction (noted on the dataset card itself). Fine
  for demonstrating the QLoRA technique; not a "quality" dataset to over-index on.

## 3. Loss & metrics

- **Training loss**: token-level cross-entropy with the prompt/padding mask above
  (same `ignore_index` pattern `transformer/model.py` already uses for padding).
- **Fast-iteration metric**: held-out Alpaca val loss/perplexity.
- **Regression metric**: HellaSwag accuracy before vs. after fine-tuning (reuse
  `mini_gpt/hellaswag.py`), sanity-checking that LoRA didn't wreck general LM ability.
- **Qualitative**: base-vs-fine-tuned generations on a small fixed set of held-out
  instructions — the metric that actually sells this as a portfolio artifact.

## 4. Compute allocation

| Stage | Where | Why |
|---|---|---|
| Data prep / tokenization / EDA | Local (CPU) | No GPU needed |
| QLoRA plumbing + tiny dry run (~200 rows) | **Local RTX 5050** | Catch masking/shape/quantization bugs cheaply before spending rented-GPU time; bitsandbytes Windows wheels cover this GPU's sm120 |
| Full fine-tuning run (52k rows × 3 epochs) | **Rented A100** | Single GPU, no DDP needed; job is small enough that even a 40GB A100 is generous headroom |
| Evaluation (val loss, HellaSwag, qualitative) | Either | Cheap either way; A100 if right after training, local otherwise |

- Rough cost/time expectation: 124M params, ~1.2M trainable (LoRA), 52k rows × 3
  epochs, seq_len 512 — this is a **small** job, likely on the order of 30–60
  minutes of A100 time including eval (a few dollars on a rented-GPU provider).
  Contrast with the `mini_gpt` **pretraining** estimate already on record
  (~14 A100-hours for the full 10B-token run) — fine-tuning here is orders of
  magnitude cheaper; calibrate expectations accordingly.
- Carry over the standing note from the other projects: if any part of this ever
  runs on the local RTX 5050 on battery, it's power-capped (~35W vs. full TDP) —
  plug in the charger for anything beyond the tiny dry run.

## 5. Suggested project structure

```
miniGpt_ft/
  data.py               # load tatsu-lab/alpaca, prompt templates, tokenize (tiktoken gpt2),
                         # prompt/response loss-mask, train/val split, collate_fn (pad + attn mask)
  model.py               # self-contained GPTConfig/CausalSelfAttention/MLP/Block/GPT copy
                         # + Linear4bit swap-in + LoRALinear + apply_qlora(model, ...) helper
  train.py               # QLoRA fine-tuning loop: load base checkpoint, quantize + inject LoRA,
                         # AdamW over adapter params, cosine LR, masked CE loss, checkpointing
  generate.py             # inference CLI: load base + adapter (or merged), Alpaca prompt template,
                         # top-k sampling (reuse mini_gpt's sampling code)
  evaluate.py             # held-out val perplexity + HellaSwag regression + base-vs-ft comparison
                         # (+ ROUGE and reward-model score of generations, base vs SFT vs PPO)
  train_reward.py         # Phase 6: reward model on AlpacaFarm preference pairs
  train_ppo.py            # Phase 6: PPO against the reward model, KL-anchored to the SFT policy
  sanity_checks.py        # label shift / padding / KV-cache / LoRA-merge correctness checks
  requirements.txt         # mini_gpt's requirements + bitsandbytes, (peft/accelerate optional)
  README.md
  CLAUDE.md
```

## 6. Key gotchas (know these before writing code)

- `mini_gpt/model.py` has zero padding/attention-mask support today — must add it
  in the **copy**, not the original (keep pretraining reproducible/untouched).
- `wte`/`lm_head` are weight-tied — never quantize or LoRA-wrap them via a generic
  "every `nn.Linear`" loop; target the 4 named projections explicitly instead.
- `vocab_size=50304` is padded up from tiktoken's real 50257 for tensor-core
  alignment — the extra 47 rows were never trained; harmless, just don't be
  surprised inspecting shapes.
- The pretrained checkpoint has no optimizer state to resume — irrelevant for
  fine-tuning, but don't go looking for it.
- Don't combine `is_causal=True` with a separately-built padding mask in
  `F.scaled_dot_product_attention` — build one combined boolean mask and pass
  `is_causal=False`.
- Alpaca is CC-BY-NC-4.0 — portfolio/research use only.
- Verify the rented A100 instance's CUDA toolkit falls in bitsandbytes' supported
  range (currently 11.8–13.2) before assuming `pip install bitsandbytes` "just works."

## 7. Key references

- Alpaca dataset: <https://huggingface.co/datasets/tatsu-lab/alpaca>
- Stanford Alpaca (prompt templates + original recipe): <https://github.com/tatsu-lab/stanford_alpaca>
- QLoRA paper: Dettmers et al., *"QLoRA: Efficient Finetuning of Quantized LLMs"* (2023) — <https://arxiv.org/abs/2305.14314>
- LoRA paper: Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"* (2021) — <https://arxiv.org/abs/2106.09685>
- bitsandbytes (4-bit NF4 quantization, Windows/CUDA support): <https://huggingface.co/docs/bitsandbytes/main/en/installation>
