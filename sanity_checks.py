"""
Fast correctness checks for the data/model plumbing - run after touching data.py or model.py:

    python sanity_checks.py

Covers the bugs that silently wreck instruction tuning without crashing: unshifted or
misaligned labels, prompt tokens that differ from inference-time encoding, NaNs from
left padding, a KV cache that disagrees with full recomputation, an inexact LoRA merge.
"""
import os

import torch

from data import IGNORE_INDEX, AlpacaDataset, format_prompt, get_encoding, left_pad
from model import EOT_TOKEN_ID, GPT, LoRALinear, apply_qlora, load_model, merge_lora, trainable_parameters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CHECKPOINT = os.environ.get("BASE_CHECKPOINT", os.path.join(SCRIPT_DIR, "..", "mini_gpt", "model_19072.pt"))

torch.manual_seed(0)
enc = get_encoding()

# 1) SFT labels: shifted by one, prompt masked, response + <|endoftext|> scored
rows = [{"instruction": "Say hi.", "input": "", "output": "\nHi there!"}]  # leading "\n" = BPE boundary case
input_ids, labels = AlpacaDataset(rows, enc)[0]
first = int((labels != IGNORE_INDEX).nonzero()[0])
prompt_ids = enc.encode_ordinary(format_prompt("Say hi."))
assert input_ids[:len(prompt_ids)].tolist() == prompt_ids, "prompt tokens must match inference-time encoding"
assert first == len(prompt_ids) - 1 and (labels[:first] == IGNORE_INDEX).all()
assert all(labels[t] == input_ids[t + 1] for t in range(first, len(input_ids) - 1)), "labels must be inputs shifted by one"
assert enc.decode(labels[first:].tolist()) == "\nHi there!<|endoftext|>"
print("[ok] SFT labels: shifted, prompt masked, response + <|endoftext|> scored")

# 2) left-padded batch == each prompt run alone (and no NaN from all-pad attention rows)
base = GPT.from_pretrained_checkpoint(BASE_CHECKPOINT).eval()
prompts = [torch.tensor(enc.encode_ordinary(s)) for s in ["Hello, my name is", "The capital of France is Paris and the", "One"]]
ids, mask = left_pad(prompts)
with torch.no_grad():
    padded_logits, _ = base(ids, attention_mask=mask)
    assert torch.isfinite(padded_logits).all(), "NaN/inf in left-padded forward"
    for i, p in enumerate(prompts):
        alone, _ = base(p[None])
        assert (padded_logits[i, -len(p):] - alone[0]).abs().max() < 1e-3, f"row {i} differs when padded"
print("[ok] left padding: batched logits match unpadded logits")

# 3) KV-cache generation == greedy decoding by full recomputation
out = base.generate(ids, mask, max_new_tokens=20, temperature=0)
for i, p in enumerate(prompts):
    seq = p[None]
    with torch.no_grad():
        for _ in range(out.size(1)):
            logits, _ = base(seq)
            seq = torch.cat([seq, logits[:, -1, :EOT_TOKEN_ID + 1].argmax(-1, keepdim=True)], dim=1)
    naive, cached = seq[0, len(p):].tolist(), out[i].tolist()
    if EOT_TOKEN_ID in naive:
        k = naive.index(EOT_TOKEN_ID) + 1
        naive, cached = naive[:k], cached[:k]
    assert naive == cached, f"row {i}: {enc.decode(naive)!r} vs {enc.decode(cached)!r}"
print("[ok] KV cache: batched generation matches full recomputation")

# 4) merge_lora folds a (non-zero) adapter in exactly
m = apply_qlora(GPT.from_pretrained_checkpoint(BASE_CHECKPOINT), quantize=False, dropout=0.0).eval()
for module in m.modules():
    if isinstance(module, LoRALinear):
        torch.nn.init.normal_(module.lora_B, std=0.02)
with torch.no_grad():
    before, _ = m(ids, attention_mask=mask)
    after, _ = merge_lora(m)(ids, attention_mask=mask)
assert (before - after).abs().max() < 1e-3
print("[ok] merge_lora is exact")

# 5) 4-bit path (CUDA only): parameter count + scalar head
if torch.cuda.is_available():
    q, _ = load_model(BASE_CHECKPOINT, device="cuda")
    _, n_trainable, n_total = trainable_parameters(q)
    assert n_trainable == 1_218_048, n_trainable  # r=8 LoRA on 4 linears x 12 blocks + LayerNorms
    rm, _ = load_model(BASE_CHECKPOINT, scalar_head=True, device="cuda")
    assert torch.isfinite(rm.score(ids.cuda(), mask.cuda())).all()
    print(f"[ok] 4-bit QLoRA: {n_trainable:,} trainable / {n_total:,} params; scalar head scores are finite")
else:
    print("[skip] 4-bit checks need CUDA")
