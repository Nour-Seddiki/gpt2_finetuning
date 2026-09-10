"""
Inference CLI for the mini_gpt instruction-tuning pipeline: the base checkpoint, the SFT
adapter, or the PPO policy adapter, prompted with the Stanford Alpaca template.

Usage:
    python generate.py --instruction "Explain quantum computing"
    python generate.py --instruction "..." --input "..." --max_tokens 100 --temperature 0.7
    python generate.py --instruction "..." --adapter_checkpoint checkpoints/ppo_adapter.pt
    python generate.py --instruction "..." --adapter_checkpoint none         # base model only
    python generate.py --instruction "..." --merge --save_merged merged.pt   # fold LoRA into the weights
"""
import argparse
import os

import torch

from data import format_prompt, get_encoding, left_pad
from model import EOT_TOKEN_ID, GPT, GPTConfig, load_model, merge_lora, save_merged

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_responses(model, prompts, max_tokens=128, temperature=0.7, top_k=50, batch_size=16,
                       device="cuda", seed=None):
    """Samples one response per prompt string, in left-padded batches with a KV cache.
    Returns a list of (response_text, finished) - finished is False when the response hit
    max_tokens before emitting <|endoftext|>."""
    enc = get_encoding()
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    results = []
    for i in range(0, len(prompts), batch_size):
        prompt_ids = [torch.tensor(enc.encode_ordinary(p), dtype=torch.long) for p in prompts[i:i + batch_size]]
        input_ids, attention_mask = left_pad(prompt_ids)
        out = model.generate(input_ids.to(device), attention_mask.to(device), max_new_tokens=max_tokens,
                             temperature=temperature, top_k=top_k, generator=generator)
        for row in out.tolist():
            finished = EOT_TOKEN_ID in row
            results.append((enc.decode(row[:row.index(EOT_TOKEN_ID)] if finished else row), finished))
    return results


def generate(model, prompt, **kwargs):
    """Single-prompt convenience wrapper: returns just the response text."""
    return generate_responses(model, [prompt], **kwargs)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, required=True, help="Instruction to respond to")
    parser.add_argument("--input", type=str, default="", help="Optional input context")
    parser.add_argument("--base_checkpoint", type=str,
                        default=os.path.join(SCRIPT_DIR, "..", "mini_gpt", "model_19072.pt"),
                        help="Path to the base pretrained checkpoint")
    parser.add_argument("--adapter_checkpoint", type=str,
                        default=os.path.join(SCRIPT_DIR, "checkpoints", "sft_adapter.pt"),
                        help="SFT or PPO adapter checkpoint, or 'none' for the base model")
    parser.add_argument("--merge", action="store_true",
                        help="Fold the LoRA weights into the (unquantized) base weights before generating")
    parser.add_argument("--save_merged", type=str, default=None,
                        help="With --merge: also save the merged model as a standalone checkpoint")
    parser.add_argument("--quantize", type=int, default=1, help="4-bit base weights for adapter inference (CUDA only)")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed, for reproducible output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.save_merged and not args.merge:
        parser.error("--save_merged requires --merge")

    device = args.device
    adapter = args.adapter_checkpoint
    if adapter and adapter.lower() != "none":
        if not os.path.exists(adapter):
            parser.error(f"adapter checkpoint not found: {adapter} (train one, or pass --adapter_checkpoint none)")
        # merging needs full-precision base weights: 4-bit NF4 can't absorb the LoRA delta
        model, ckpt = load_model(args.base_checkpoint, adapter, quantize=bool(args.quantize) and not args.merge,
                                 device=device)
        print(f"loaded {ckpt.get('kind', 'LoRA')} adapter from {adapter}")
        if args.merge:
            merge_lora(model)
            print("merged LoRA into the base weights")
            if args.save_merged:
                save_merged(model, args.save_merged)
                print(f"saved merged model to {args.save_merged}")
    else:
        print(f"loading base checkpoint from {args.base_checkpoint} (no adapter)")
        model = GPT.from_pretrained_checkpoint(args.base_checkpoint, config=GPTConfig()).to(device)
    model.eval()

    prompt = format_prompt(args.instruction, args.input)
    print(f"\nPrompt:\n{prompt}")
    (response, finished), = generate_responses(model, [prompt], max_tokens=args.max_tokens,
                                               temperature=args.temperature, top_k=args.top_k,
                                               device=device, seed=args.seed)
    print(f"Response:\n{response}")
    if not finished:
        print(f"\n(cut off at --max_tokens {args.max_tokens} before <|endoftext|>)")


if __name__ == "__main__":
    main()
