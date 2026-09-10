"""
GPT-2 (124M) for instruction tuning + RLHF - a self-contained copy of the classes in
mini_gpt/model.py (not imported directly: that file runs top-level training/DDP setup
code and asserts on missing fineweb shards on import), extended with:

  - padding-aware attention with explicit position ids (right padding for training
    batches, left padding for batched generation) and a KV cache for fast sampling
  - 4-bit NF4 quantization (bitsandbytes) + a hand-written low-rank LoRA adapter,
    applied to the 4 linear projections in every block
  - ScalarHeadGPT: the same trunk with a scalar head - the reward model and the PPO
    value model
  - adapter save/load, LoRA merging, and the masked LM-loss eval shared by the scripts

See PROJECT_PLAN.md and README.md for the design writeup.
"""
import math
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

EOT_TOKEN_ID = 50256  # tiktoken gpt2's <|endoftext|>
TOKENIZER_VOCAB = 50257  # real gpt2 vocab size; logit rows >= this are padding that was never trained


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # padded up from tiktoken gpt2's real 50257 for tensor-core alignment
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


def autocast(device):
    """bf16 autocast on CUDA (the dtype mini_gpt was pretrained in), a no-op on CPU."""
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda")


def build_attn_mask(attention_mask, q_len):
    """attention_mask: (B, T_k), 1 = real token / 0 = pad, covering every key position - any
    cached past plus the q_len current positions (which are the last q_len keys). Returns a
    (B,1,q_len,T_k) boolean mask for F.scaled_dot_product_attention (True = attend) that
    combines causality with key padding.

    Every query may also attend to itself: with left padding a pad query has no valid key at
    all, and SDPA can return NaN for such a fully-masked row - which then leaks into real
    tokens in the next layer, since masked-out values still enter the matmul as 0 * NaN."""
    T_k = attention_mask.size(1)
    offset = T_k - q_len  # absolute position of the first query
    ones = torch.ones(q_len, T_k, dtype=torch.bool, device=attention_mask.device)
    causal = ones.tril(diagonal=offset)
    self_attend = causal & ones.triu(diagonal=offset)
    key_pad = attention_mask[:, None, None, :].bool()  # (B,1,1,T_k)
    return (causal & key_pad) | self_attend  # (B,1,q_len,T_k) via broadcasting


def position_ids_from_mask(attention_mask):
    """Each token's index among its row's real tokens, so a left-padded prompt gets the same
    positions as the unpadded one."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def last_token_index(attention_mask):
    """Index of each row's last real token - works for left, right, or mixed padding."""
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device)
    return (attention_mask.long() * positions).argmax(-1)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, attn_mask=None, kv_cache=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # attn_mask already encodes causality, so is_causal must stay False here
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, (k, v)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None, kv_cache=None):
        attn_out, present = self.attn(self.ln_1(x), attn_mask=attn_mask, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight sharing scheme

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden_states(self, idx, attention_mask=None, position_ids=None, past_kv=None):
        """Final (post-ln_f) hidden states (B, T, n_embd) + the per-layer KV cache.
        attention_mask (B, past + T): 1 = real token, 0 = pad; omit it for unpadded input
        (then attention takes the fast plain-causal path). position_ids default to each
        token's index among its row's real tokens."""
        B, T = idx.size()
        past_len = 0 if past_kv is None else past_kv[0][0].size(2)
        assert past_len + T <= self.config.block_size, f"Cannot forward sequence of length {past_len + T}, block size is only {self.config.block_size}"
        if attention_mask is None and past_kv is not None:
            attention_mask = torch.ones(B, past_len + T, dtype=torch.long, device=idx.device)
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(T, device=idx.device)[None, :]
            else:
                position_ids = position_ids_from_mask(attention_mask)[:, -T:]
        x = self.transformer.wte(idx) + self.transformer.wpe(position_ids)
        attn_mask = None if attention_mask is None else build_attn_mask(attention_mask, T)
        presents = []
        for i, block in enumerate(self.transformer.h):
            x, present = block(x, attn_mask=attn_mask, kv_cache=None if past_kv is None else past_kv[i])
            presents.append(present)
        return self.transformer.ln_f(x), presents

    def forward(self, idx, targets=None, attention_mask=None, position_ids=None):
        x, _ = self.hidden_states(idx, attention_mask=attention_mask, position_ids=position_ids)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # targets are already shifted (targets[t] is the token after idx[t]), same as mini_gpt
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, attention_mask=None, max_new_tokens=128, temperature=1.0, top_k=None, generator=None):
        """Batched sampling with a KV cache. idx: (B, P) prompts, LEFT-padded (data.left_pad)
        so every row's last prompt token sits in the last column. Returns (B, n) new token ids,
        n <= max_new_tokens; once a row emits <|endoftext|> it keeps emitting it, so everything
        from a row's first <|endoftext|> on is filler. temperature=0 means greedy decoding.
        Sampling never picks the 47 untrained padding ids (tiktoken can't decode them)."""
        B, P = idx.shape
        max_new_tokens = min(max_new_tokens, self.config.block_size - P)
        if max_new_tokens <= 0:
            return idx.new_empty((B, 0))
        if attention_mask is None:
            attention_mask = torch.ones_like(idx)
        vocab_pad = torch.arange(self.config.vocab_size, device=idx.device) >= TOKENIZER_VOCAB
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        new_tokens = []
        with autocast(idx.device):
            x, past = self.hidden_states(idx, attention_mask=attention_mask)
            for _ in range(max_new_tokens):
                logits = self.lm_head(x[:, -1]).float().masked_fill(vocab_pad, float("-inf"))
                if temperature == 0:
                    next_token = logits.argmax(-1)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    next_token = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator).squeeze(1)
                next_token = torch.where(finished, EOT_TOKEN_ID, next_token)
                new_tokens.append(next_token)
                finished |= next_token == EOT_TOKEN_ID
                if finished.all():
                    break
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1)
                x, past = self.hidden_states(next_token[:, None], attention_mask=attention_mask, past_kv=past)
        return torch.stack(new_tokens, dim=1)

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path, config=None, map_location="cpu"):
        """Loads the weights saved by mini_gpt/model.py's training loop
        (`{'model': state_dict, 'config': GPTConfig, 'step':, 'val_loss':}`), or by save_merged."""
        # That checkpoint's 'config' value pickles as __main__.GPTConfig (model.py is run
        # directly as `python model.py`). Expose our compatible GPTConfig under the same
        # name so torch.load can unpickle it without crashing - we still build the model
        # from an explicit config below rather than trust the unpickled one.
        sys.modules["__main__"].GPTConfig = GPTConfig
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        model = cls(config or GPTConfig())
        model.load_state_dict(ckpt["model"])
        return model


class ScalarHeadGPT(nn.Module):
    """The GPT trunk (no lm_head) + a linear head mapping each position's final hidden state
    to a scalar. As the reward model, score() reads it at the last real token - the
    <|endoftext|> closing the response. As the PPO value model, forward() gives one value per
    position."""

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        self.config = gpt.config
        self.head = nn.Linear(gpt.config.n_embd, 1)
        nn.init.normal_(self.head.weight, std=1 / math.sqrt(gpt.config.n_embd + 1))
        nn.init.zeros_(self.head.bias)

    def forward(self, idx, attention_mask=None):
        x, _ = self.gpt.hidden_states(idx, attention_mask=attention_mask)
        return self.head(x).squeeze(-1).float()  # (B, T)

    def score(self, idx, attention_mask):
        values = self(idx, attention_mask)
        return values.gather(1, last_token_index(attention_mask)[:, None]).squeeze(1)  # (B,)


# -----------------------------------------------------------------------------
# QLoRA: 4-bit (bitsandbytes NF4) frozen base weights + a trainable low-rank adapter

class LoRALinear(nn.Module):
    """Wraps a frozen base Linear (optionally 4-bit quantized) with a trainable
    low-rank adapter: h = base(x) + (alpha/r) * dropout(x) @ A.T @ B.T"""

    def __init__(self, base, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero-initialized so the adapter is a no-op at the start of training
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + lora_out * self.scaling


def _quantize_linear(linear, compute_dtype):
    """Replace a regular nn.Linear with a bitsandbytes 4-bit NF4 Linear4bit holding the
    same weights. The actual quantization happens lazily on the model's next .to(cuda)."""
    import bitsandbytes as bnb  # local import: only needed when quantize=True
    q = bnb.nn.Linear4bit(
        linear.in_features, linear.out_features,
        bias=linear.bias is not None,
        compute_dtype=compute_dtype, quant_type="nf4",
    )
    q.load_state_dict(linear.state_dict())
    return q


def _wrap_with_lora(linear, r, alpha, dropout, quantize, compute_dtype):
    base = _quantize_linear(linear, compute_dtype) if quantize else linear
    for p in base.parameters():
        p.requires_grad_(False)
    return LoRALinear(base, r=r, alpha=alpha, dropout=dropout)


def apply_qlora(model, r=8, alpha=16, dropout=0.05, quantize=True,
                compute_dtype=torch.bfloat16, train_layernorms=True):
    """Freezes the whole model, then 4-bit-quantizes + LoRA-wraps the 4 linear
    projections (c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj) in every block.
    wte/wpe/lm_head are left frozen and unquantized - wte/lm_head are weight-tied,
    so they can't go through a generic per-Linear swap without breaking that sharing."""
    for p in model.parameters():
        p.requires_grad_(False)

    for block in model.transformer.h:
        block.attn.c_attn = _wrap_with_lora(block.attn.c_attn, r, alpha, dropout, quantize, compute_dtype)
        block.attn.c_proj = _wrap_with_lora(block.attn.c_proj, r, alpha, dropout, quantize, compute_dtype)
        block.mlp.c_fc = _wrap_with_lora(block.mlp.c_fc, r, alpha, dropout, quantize, compute_dtype)
        block.mlp.c_proj = _wrap_with_lora(block.mlp.c_proj, r, alpha, dropout, quantize, compute_dtype)
        if train_layernorms:
            for p in list(block.ln_1.parameters()) + list(block.ln_2.parameters()):
                p.requires_grad_(True)
    if train_layernorms:
        for p in model.transformer.ln_f.parameters():
            p.requires_grad_(True)
    return model


def _logical_numel(p):
    # bitsandbytes stores a quantized 4-bit weight as uint8, two weights per byte
    if type(p).__name__ == "Params4bit" and p.dtype == torch.uint8:
        return p.numel() * 2
    return p.numel()


def trainable_parameters(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(_logical_numel(p) for p in model.parameters())
    return trainable, n_trainable, n_total


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


# -----------------------------------------------------------------------------
# adapter checkpoints: only what training changed, a few MB, independent of the base weights

def adapter_state_dict(model):
    """The trainable parameters: LoRA A/B, LayerNorms if trained, a scalar head if any."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def save_adapter(path, model, **extra):
    lora = next(m for m in model.modules() if isinstance(m, LoRALinear))
    torch.save({"lora_state_dict": adapter_state_dict(model), "lora_r": lora.r, "lora_alpha": lora.alpha, **extra}, path)


def load_adapter(model, state_dict):
    if isinstance(model, ScalarHeadGPT) and not any(k.startswith("gpt.") for k in state_dict):
        # a plain-GPT adapter (e.g. SFT) initializing a scalar-head model's trunk; the head stays fresh
        state_dict = {f"gpt.{k}": v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)  # base weights are "missing" by design
    if result.unexpected_keys:
        raise ValueError(f"adapter doesn't match the model, unexpected keys: {result.unexpected_keys[:5]}")


def load_model(base_checkpoint, adapter_path=None, scalar_head=False, quantize=True,
               lora_r=8, lora_alpha=16, lora_dropout=0.0, train_layernorms=True,
               compute_dtype=torch.bfloat16, device="cpu"):
    """pretrained mini_gpt -> [ScalarHeadGPT] -> 4-bit + LoRA -> [adapter weights] -> device.
    An adapter's saved rank/alpha override lora_r/lora_alpha. Everything is frozen except the
    LoRA / LayerNorm / head parameters. 4-bit needs CUDA, so quantize is ignored on CPU.
    Returns (model, adapter checkpoint dict or None)."""
    quantize = quantize and str(device).startswith("cuda")
    ckpt = None
    if adapter_path:
        ckpt = torch.load(adapter_path, map_location="cpu", weights_only=True)
        lora_r, lora_alpha = ckpt["lora_r"], ckpt["lora_alpha"]
    gpt = GPT.from_pretrained_checkpoint(base_checkpoint)
    apply_qlora(gpt, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, quantize=quantize,
                compute_dtype=compute_dtype, train_layernorms=train_layernorms)
    model = ScalarHeadGPT(gpt) if scalar_head else gpt
    if ckpt is not None:
        load_adapter(model, ckpt["lora_state_dict"])
    return model.to(device), ckpt  # .to(cuda) is what actually quantizes the Linear4bit weights


@torch.no_grad()
def merge_lora(model):
    """Folds every adapter into its base weight (W <- W + (alpha/r) * B @ A) and unwraps the
    LoRALinear, leaving a plain GPT with no adapter overhead. Needs an unquantized base
    (quantize=False): NF4 weights can't absorb an arbitrary delta without re-quantizing."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                if type(child.base) is not nn.Linear:
                    raise ValueError("merge_lora needs an unquantized base - load the model with quantize=False")
                child.base.weight += child.scaling * (child.lora_B @ child.lora_A)
                setattr(module, name, child.base)
    return model


def save_merged(model, path):
    """Standalone checkpoint in mini_gpt's format - loads with GPT.from_pretrained_checkpoint."""
    torch.save({"model": model.state_dict(), "config": asdict(model.config)}, path)


@torch.no_grad()
def lm_loss(model, loader, device):
    """Token-weighted mean masked LM loss over batches of (input_ids, labels, attention_mask)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for input_ids, labels, attention_mask in loader:
        input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
        with autocast(device):
            _, loss = model(input_ids, targets=labels, attention_mask=attention_mask)
        n = (labels != -100).sum().item()
        total += loss.item() * n
        count += n
    model.train(was_training)
    return total / max(1, count)
