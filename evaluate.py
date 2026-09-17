"""
Evaluation for the mini_gpt instruction-tuning pipeline. For the base model and each
adapter (SFT and/or PPO) it reports:

  - held-out val loss / perplexity (response tokens only) on Alpaca or smol-smoltalk
  - ROUGE-1/2/L of sampled responses vs. that dataset's reference outputs
  - <|endoftext|> rate, length and repetition (share of repeated word 4-grams) of those responses
  - mean reward-model score of those same responses
    (when a reward adapter from train_reward.py is available)
  - optionally HellaSwag accuracy - the catastrophic-forgetting check against the base
    model's 30.14%

plus a qualitative side-by-side on a few hand-written instructions.

Usage:
    python evaluate.py                                    # base + whichever of checkpoints/{sft,ppo}_adapter.pt exist
    python evaluate.py --adapters checkpoints/sft_adapter.pt checkpoints/ppo_adapter.pt
    python evaluate.py --eval_hellaswag                   # full 10,042-example HellaSwag val (slower)
    python evaluate.py --eval_hellaswag --hellaswag_limit 1000
    python evaluate.py --val_dataset smoltalk --max_tokens 384 --top_p 0.9 --repetition_penalty 1.1
"""
import argparse
import math
import os
import sys

import torch
from rouge_score import rouge_scorer
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import SFT_DATASETS, collate_fn, encode_response, format_prompt, get_encoding, right_pad
from generate import generate_responses
from model import GPT, GPTConfig, autocast, freeze, lm_loss, load_model
from runtime import configure_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

QUALITATIVE_INSTRUCTIONS = [
    "What is 2+2?",
    "Explain machine learning in one sentence.",
    "Write a haiku about AI.",
    "Give me three ideas for a rainy weekend.",
    "Hi! How are you?",
    "What is the capital of France? Answer in one word.",
]


def rouge(references, predictions):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    return {k: v / max(1, len(references)) for k, v in totals.items()}


def repetition(text, n=4):
    """Share of the response's word n-grams that repeat an earlier one: 0 for no repeats, near 1
    for a response stuck in a loop."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


@torch.no_grad()
def reward_scores(reward_model, prompts, responses, device, batch_size=16):
    """Reward-model score for each (prompt, (response_text, finished)). A response cut off at
    max_tokens gets no closing <|endoftext|> - the reward model only saw finished responses in
    training, so those scores are less trustworthy (hence the separate EOS rate)."""
    enc = get_encoding()
    block_size = reward_model.config.block_size
    scores = []
    for i in range(0, len(prompts), batch_size):
        seqs = []
        for prompt, (text, finished) in zip(prompts[i:i + batch_size], responses[i:i + batch_size]):
            response_ids = encode_response(enc, text) if finished else enc.encode_ordinary(text)
            seqs.append(torch.tensor((enc.encode_ordinary(prompt) + response_ids)[:block_size], dtype=torch.long))
        input_ids, attention_mask = right_pad(seqs)
        with autocast(device):
            scores.append(reward_model.score(input_ids.to(device), attention_mask.to(device)).cpu())
    return torch.cat(scores)


@torch.no_grad()
def hellaswag_accuracy(model, device, mini_gpt_dir, limit=0):
    """mini_gpt's HellaSwag eval (pick the ending with the lowest avg loss) - same numbers as
    the 30.14% measured at the end of pretraining when run on the full val set."""
    if mini_gpt_dir not in sys.path:
        sys.path.insert(0, mini_gpt_dir)
    from hellaswag import iterate_examples, render_example
    num_correct, num_total = 0, 0
    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        with autocast(device):
            logits, _ = model(tokens)
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_tokens = tokens[..., 1:].contiguous()
        shift_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                       shift_tokens.view(-1), reduction="none").view(tokens.size(0), -1)
        shift_mask = mask[..., 1:].contiguous()
        avg_loss = (shift_losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        num_correct += int(avg_loss.argmin().item() == label)
        num_total += 1
        if limit and num_total >= limit:
            break
    return num_correct / num_total, num_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=str,
                        default=os.path.join(SCRIPT_DIR, "..", "mini_gpt", "model_19072.pt"),
                        help="Path to base pretrained checkpoint")
    parser.add_argument("--adapters", type=str, nargs="*", default=None,
                        help="Adapter checkpoints to compare against the base model "
                             "(default: whichever of checkpoints/sft_adapter.pt, checkpoints/ppo_adapter.pt exist)")
    parser.add_argument("--reward_checkpoint", type=str, default=os.path.join(CHECKPOINT_DIR, "reward_adapter.pt"),
                        help="Reward adapter used to score generations (skipped if missing)")
    parser.add_argument("--val_dataset", choices=sorted(SFT_DATASETS), default="alpaca",
                        help="Dataset for val loss and the ROUGE references")
    parser.add_argument("--val_size", type=int, default=2000,
                        help="Validation set size (for alpaca it must match training: val is carved out of train)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for val loss")
    parser.add_argument("--quantize", type=int, default=1, help="4-bit base weights for the adapter models (CUDA only)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Validation examples to generate responses for (ROUGE + reward score)")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens per generated response")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold (1 = off)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="1 = off, see GPT.generate")
    parser.add_argument("--gen_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed - every model gets the same one")
    parser.add_argument("--skip_generation", action="store_true", help="Only compute val loss (and HellaSwag)")
    parser.add_argument("--eval_hellaswag", action="store_true", help="Also run HellaSwag (slow)")
    parser.add_argument("--hellaswag_limit", type=int, default=0, help="Examples to use, 0 = full val set")
    parser.add_argument("--mini_gpt_dir", type=str, default=os.path.join(SCRIPT_DIR, "..", "mini_gpt"),
                        help="Directory containing mini_gpt's hellaswag.py")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}\n")
    configure_runtime(device)
    adapters = args.adapters
    if adapters is None:
        adapters = [p for p in (os.path.join(CHECKPOINT_DIR, "sft_adapter.pt"), os.path.join(CHECKPOINT_DIR, "ppo_adapter.pt"))
                    if os.path.exists(p)]

    print(f"Loading the {args.val_dataset} validation set...")
    _, val_ds, val_rows = SFT_DATASETS[args.val_dataset](val_size=args.val_size, seed=1337, max_train_examples=0,
                                                         return_raw_val=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Validation examples: {len(val_ds)}\n")

    models = [("base", GPT.from_pretrained_checkpoint(args.base_checkpoint, config=GPTConfig()).to(device).eval())]
    for path in adapters:
        model, ckpt = load_model(args.base_checkpoint, path, quantize=bool(args.quantize), device=device)
        name = ckpt.get("kind") or os.path.splitext(os.path.basename(path))[0]
        if "iteration" in ckpt:  # PPO snapshots all have kind "ppo" - keep them apart in the results
            name = f"{name}@{ckpt['iteration']}"
        if any(name == other for other, _ in models):  # e.g. two SFT adapters: add the folder name
            name = f"{name}:{os.path.basename(os.path.dirname(os.path.abspath(path)))}"
        print(f"Loaded {name} adapter from {path}")
        models.append((name, model.eval()))

    reward_model = None
    if args.reward_checkpoint and os.path.exists(args.reward_checkpoint):
        reward_model, _ = load_model(args.base_checkpoint, args.reward_checkpoint, scalar_head=True,
                                     quantize=bool(args.quantize), device=device)
        freeze(reward_model)
        print(f"Loaded reward model from {args.reward_checkpoint}")

    rows = val_rows.select(range(min(args.n_samples, len(val_rows))))
    prompts = [format_prompt(r["instruction"], r.get("input") or "") for r in rows]
    references = [r["output"].strip() for r in rows]

    sampling = dict(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty, seed=args.seed)
    results = {}
    for name, model in models:
        print(f"\n=== {name} ===")
        res = {}
        res["val_loss"] = lm_loss(model, val_loader, device)
        res["ppl"] = math.exp(res["val_loss"])
        print(f"  val loss {res['val_loss']:.4f}, perplexity {res['ppl']:.2f}")
        if not args.skip_generation:
            responses = generate_responses(model, prompts, batch_size=args.gen_batch_size, device=device, **sampling)
            res.update(rouge(references, [text for text, _ in responses]))
            res["eos_rate"] = sum(finished for _, finished in responses) / len(responses)
            res["repetition"] = sum(repetition(text) for text, _ in responses) / len(responses)
            res["length"] = sum(len(text.split()) for text, _ in responses) / len(responses)
            print(f"  ROUGE-1/2/L {res['rouge1']:.3f} / {res['rouge2']:.3f} / {res['rougeL']:.3f}, "
                  f"finished (emitted <|endoftext|>) {100 * res['eos_rate']:.0f}%, "
                  f"repeated 4-grams {100 * res['repetition']:.1f}%, {res['length']:.0f} words")
            if reward_model is not None:
                res["reward"] = reward_scores(reward_model, prompts, responses, device).mean().item()
                print(f"  mean reward-model score {res['reward']:.3f}")
        if args.eval_hellaswag:
            res["hellaswag"], n = hellaswag_accuracy(model, device, args.mini_gpt_dir, args.hellaswag_limit)
            print(f"  HellaSwag accuracy {res['hellaswag']:.4f} ({n} examples)")
        results[name] = res

    # summary table
    columns = [("val_loss", "val loss", "{:.4f}"), ("ppl", "ppl", "{:.2f}"), ("rouge1", "ROUGE-1", "{:.3f}"),
               ("rouge2", "ROUGE-2", "{:.3f}"), ("rougeL", "ROUGE-L", "{:.3f}"), ("reward", "RM score", "{:+.3f}"),
               ("eos_rate", "finished", "{:.0%}"), ("repetition", "repeats", "{:.1%}"), ("length", "words", "{:.0f}"),
               ("hellaswag", "HellaSwag", "{:.4f}")]
    columns = [c for c in columns if any(c[0] in r for r in results.values())]
    width = max(8, *(len(name) + 1 for name in results))
    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    print(f"{'model':<{width}}" + "".join(f"{title:>11}" for _, title, _ in columns))
    for name, res in results.items():
        print(f"{name:<{width}}" + "".join(f"{fmt.format(res[key]) if key in res else '-':>11}" for key, _, fmt in columns))

    if not args.skip_generation:
        print("\n" + "=" * 80 + "\nQUALITATIVE COMPARISON\n" + "=" * 80)
        for instruction in QUALITATIVE_INSTRUCTIONS:
            print(f"\nInstruction: {instruction}")
            for name, model in models:
                (text, _), = generate_responses(model, [format_prompt(instruction)], device=device,
                                                **{**sampling, "max_tokens": min(args.max_tokens, 160)})
                print(f"  [{name}] {text.strip()[:400]!r}")


if __name__ == "__main__":
    main()
