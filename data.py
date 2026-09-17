"""
Data pipelines for the three stages of instruction-tuning mini_gpt:

  1. SFT (train.py)            - tatsu-lab/alpaca: 52k instruction/input/output rows, or
                                 HuggingFaceTB/smol-smoltalk: 460k conversations, first exchange
                                 of each rendered as an Alpaca row
  2. reward (train_reward.py)  - tatsu-lab/alpaca_farm preference pairs: two outputs for
                                 the same instruction + which one is better
  3. PPO (train_ppo.py)        - tatsu-lab/alpaca_farm unlabeled instructions: prompts only,
                                 the policy writes the responses

Every stage renders prompts with the Stanford Alpaca template and tokenizes with tiktoken's
gpt2 BPE - the same vocab space the pretrained mini_gpt checkpoint already uses.
"""
import os

import numpy as np
import tiktoken
import torch
from datasets import concatenate_datasets, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset, Subset

IGNORE_INDEX = -100  # torch.nn.functional.cross_entropy's default ignore_index
PAD_TOKEN_ID = 0  # arbitrary: padded positions are masked out of attention and of every loss

# AlpacaFarm's HF repo ships a dataset loading script, which datasets>=4 refuses to run, so
# read its raw json files directly instead
ALPACA_FARM = "hf://datasets/tatsu-lab/alpaca_farm/"
PREFERENCE_FILES = {
    "gpt4": "alpaca_gpt4_preference.json",    # 19,472 pairs labeled by GPT-4
    "human": "alpaca_human_preference.json",  # 9,691 pairs labeled by crowd workers (noisier)
}

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)

_ENCODING = None


def get_encoding():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("gpt2")
    return _ENCODING


def format_prompt(instruction, input_text=""):
    """Stanford Alpaca template, with or without the optional input block."""
    if input_text and input_text.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def encode_prompt(enc, row):
    return enc.encode_ordinary(format_prompt(row["instruction"], row.get("input") or ""))


def encode_response(enc, response):
    """Response tokens + a closing <|endoftext|>. Encoded separately from the prompt (rather
    than as one prompt+response string) so the prompt tokens are exactly what the model is
    fed at inference - BPE can merge a token across the boundary, e.g. the prompt's final
    "\\n" with a response that starts with "\\n"."""
    return enc.encode_ordinary(response) + [enc.eot_token]


def right_pad(seqs, pad_value=PAD_TOKEN_ID):
    """Right-pads 1D tensors into (B, T) plus the matching attention mask (1 = real token)."""
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_value)
    lengths = torch.tensor([len(s) for s in seqs])
    attention_mask = (torch.arange(padded.size(1))[None, :] < lengths[:, None]).long()
    return padded, attention_mask


def left_pad(seqs, pad_value=PAD_TOKEN_ID):
    """Left-pads 1D tensors into (B, T) plus the matching attention mask. Used for batched
    generation: every prompt ends in the last column, so all rows start sampling together."""
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, max_len - len(s):] = s
        attention_mask[i, max_len - len(s):] = 1
    return padded, attention_mask


# -----------------------------------------------------------------------------
# stage 1: SFT on Alpaca

class AlpacaDataset(Dataset):
    """SFT examples as (input_ids, labels). labels are shifted one to the left of input_ids
    (labels[t] is the token that follows input_ids[t] - the same convention as mini_gpt's
    pretraining loader) and masked over the prompt, so only the response + <|endoftext|>
    contribute to the loss."""

    def __init__(self, rows, enc, max_seq_len=512):
        self.examples = []
        for row in rows:
            prompt_ids = encode_prompt(enc, row)
            ids = prompt_ids + encode_response(enc, row["output"])
            if len(ids) - 1 > max_seq_len:
                continue  # drop rather than truncate - truncating would corrupt the response
            # uint16 holds every gpt2 id (max 50256) at 2 bytes/token; a list of Python ints
            # costs ~36, i.e. ~5 GB of RAM for all of smol-smoltalk
            self.examples.append((np.array(ids, dtype=np.uint16), len(prompt_ids)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, prompt_len = self.examples[idx]
        ids = torch.from_numpy(ids.astype(np.int64))
        input_ids, labels = ids[:-1], ids[1:].clone()
        # labels[prompt_len - 1] is the first response token (predicted from the prompt's last
        # token); every label before it is still part of the prompt
        labels[:prompt_len - 1] = IGNORE_INDEX
        return input_ids, labels


def collate_fn(batch):
    """Right-pads a batch of (input_ids, labels); returns (input_ids, labels, attention_mask)."""
    input_ids, labels = zip(*batch)
    input_ids_padded, attention_mask = right_pad(input_ids)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return input_ids_padded, labels_padded, attention_mask


def load_alpaca(max_seq_len=512, val_size=2000, seed=1337, max_train_examples=None,
                return_raw_val=False):
    """Downloads tatsu-lab/alpaca from HuggingFace and returns (train_ds, val_ds) by default.
    Set return_raw_val=True to also get the raw HuggingFace val rows (instruction/input/output
    dicts) for generation-based metrics like ROUGE. max_train_examples truncates the train
    split - for a fast local dry run, or 0 to skip tokenizing it when only val is needed."""
    enc = get_encoding()
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=seed)
    val_rows = ds.select(range(val_size))
    train_end = len(ds) if max_train_examples is None else min(len(ds), val_size + max_train_examples)
    train_rows = ds.select(range(val_size, train_end))
    train_ds = AlpacaDataset(train_rows, enc, max_seq_len=max_seq_len)
    val_ds = AlpacaDataset(val_rows, enc, max_seq_len=max_seq_len)
    if return_raw_val:
        return train_ds, val_ds, val_rows
    return train_ds, val_ds


def smoltalk_to_alpaca(example):
    """The first exchange of a smol-smoltalk conversation as an Alpaca row. A system prompt
    becomes the instruction and the first user turn its input: in the rewrite and summarize
    subsets the system prompt is the task and the user turn is the text to work on. Rows
    without a user -> assistant exchange come back with an empty output (filtered out)."""
    messages = example["messages"]
    system = messages[0]["content"].strip() if messages[0]["role"] == "system" else ""
    turns = messages[1:] if system else messages
    if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
        return {"instruction": "", "input": "", "output": ""}
    user, reply = turns[0]["content"].strip(), turns[1]["content"].strip()
    if system:
        return {"instruction": system, "input": user, "output": reply}
    return {"instruction": user, "input": "", "output": reply}


def load_smoltalk(max_seq_len=512, val_size=2000, seed=1337, max_train_examples=None,
                  return_raw_val=False):
    """HuggingFaceTB/smol-smoltalk (Apache-2.0), the SFT mix built for SmolLM2-135M/360M-Instruct,
    reduced to first exchanges in the Alpaca template. Same interface as load_alpaca; val rows
    come from the dataset's own test split. About 13% of first exchanges are over 512 tokens
    and get dropped, so max_train_examples counts rows before that filter."""
    enc = get_encoding()

    def rows(split, n):
        ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=seed)
        if n is not None:
            ds = ds.select(range(min(len(ds), 2 * n)))  # headroom for the rows filtered out below
        ds = ds.map(smoltalk_to_alpaca, remove_columns=["messages"]).filter(lambda r: bool(r["output"]))
        return ds.select(range(min(len(ds), n))) if n is not None else ds

    val_rows = rows("test", val_size)
    train_rows = rows("train", max_train_examples)
    train_ds = AlpacaDataset(train_rows, enc, max_seq_len=max_seq_len)
    val_ds = AlpacaDataset(val_rows, enc, max_seq_len=max_seq_len)
    if return_raw_val:
        return train_ds, val_ds, val_rows
    return train_ds, val_ds


MIX_ALPACA_FRACTION = float(os.environ.get("MIX_ALPACA_FRACTION", 0.25))
SMOLTALK_FIT_RATE = 0.865  # share of smol-smoltalk rows that fit in 512 tokens, for sizing below


def load_mix(max_seq_len=512, val_size=2000, seed=1337, max_train_examples=None, return_raw_val=False,
             alpaca_fraction=MIX_ALPACA_FRACTION):
    """smol-smoltalk and Alpaca in one training set. Trained on smol-smoltalk alone the model wins
    conversation, explanation and code but loses the short factual and format-constrained answers
    Alpaca's terse responses teach; mixing keeps both. alpaca_fraction (MIX_ALPACA_FRACTION) is
    Alpaca's share of the examples, and the validation set mixes them the same way. With
    max_train_examples unset, all of Alpaca is used and smol-smoltalk is sized around it."""
    alpaca_val = max(1, round(val_size * alpaca_fraction))
    alpaca_n = round(max_train_examples * alpaca_fraction) if max_train_examples else None
    a_train, a_val, a_raw = load_alpaca(max_seq_len, alpaca_val, seed, alpaca_n, return_raw_val=True)

    # smol-smoltalk's loader counts rows before the 512-token filter, so ask for enough to land on
    # the target example count, then take exactly that many
    smoltalk_n = round(len(a_train) * (1 - alpaca_fraction) / alpaca_fraction)
    s_train, s_val, s_raw = load_smoltalk(max_seq_len, val_size - alpaca_val, seed,
                                          round(smoltalk_n / SMOLTALK_FIT_RATE), return_raw_val=True)
    s_train = Subset(s_train, range(min(smoltalk_n, len(s_train))))

    train_ds, val_ds = ConcatDataset([a_train, s_train]), ConcatDataset([a_val, s_val])
    print(f"  mix: {len(a_train)} alpaca + {len(s_train)} smol-smoltalk training examples "
          f"({len(a_train) / max(1, len(train_ds)):.0%} alpaca)")
    if return_raw_val:
        # shuffled, so that a caller sampling the first N rows (evaluate.py) gets both datasets
        columns = ["instruction", "input", "output"]
        raw = concatenate_datasets([a_raw.select_columns(columns), s_raw.select_columns(columns)])
        return train_ds, val_ds, raw.shuffle(seed=seed)
    return train_ds, val_ds


SFT_DATASETS = {"alpaca": load_alpaca, "smoltalk": load_smoltalk, "mix": load_mix}


# -----------------------------------------------------------------------------
# stage 2: reward model on AlpacaFarm preference pairs

class PreferenceDataset(Dataset):
    """(chosen, rejected) pairs, each a full prompt + response + <|endoftext|> sequence."""

    def __init__(self, rows, enc, max_seq_len=512):
        self.pairs = []
        for row in rows:
            prompt_ids = encode_prompt(enc, row)
            first, second = row["output_1"], row["output_2"]
            chosen, rejected = (first, second) if int(row["preference"]) == 1 else (second, first)
            if chosen.strip() == rejected.strip():
                continue  # identical outputs carry no preference signal
            chosen_ids = prompt_ids + encode_response(enc, chosen)
            rejected_ids = prompt_ids + encode_response(enc, rejected)
            if max(len(chosen_ids), len(rejected_ids)) > max_seq_len:
                continue
            self.pairs.append((chosen_ids, rejected_ids))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        chosen, rejected = self.pairs[idx]
        return torch.tensor(chosen, dtype=torch.long), torch.tensor(rejected, dtype=torch.long)


def preference_collate_fn(batch):
    """B pairs -> one right-padded (2B, T) batch: rows [0, B) chosen, rows [B, 2B) rejected,
    so a single forward pass scores both halves."""
    chosen, rejected = zip(*batch)
    return right_pad(list(chosen) + list(rejected))


def load_preferences(source="gpt4", max_seq_len=512, val_size=1000, seed=1337, max_train_examples=None):
    """AlpacaFarm preference pairs -> (train_ds, val_ds) PreferenceDatasets."""
    enc = get_encoding()
    ds = load_dataset("json", data_files=ALPACA_FARM + PREFERENCE_FILES[source], split="train")
    ds = ds.shuffle(seed=seed)
    val_rows = ds.select(range(val_size))
    train_end = len(ds) if max_train_examples is None else min(len(ds), val_size + max_train_examples)
    train_rows = ds.select(range(val_size, train_end))
    return PreferenceDataset(train_rows, enc, max_seq_len), PreferenceDataset(val_rows, enc, max_seq_len)


# -----------------------------------------------------------------------------
# stage 3: PPO prompts (AlpacaFarm's unlabeled split - disjoint from its preference split)

class PromptDataset(Dataset):
    """Tokenized prompts only - the PPO policy generates the responses."""

    def __init__(self, rows, enc, max_prompt_len=256):
        self.prompts = []
        for row in rows:
            ids = encode_prompt(enc, row)
            if len(ids) <= max_prompt_len:
                self.prompts.append(ids)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return torch.tensor(self.prompts[idx], dtype=torch.long)


def load_prompts(max_prompt_len=256, val_size=256, seed=1337, max_train_examples=None):
    """AlpacaFarm unlabeled instructions -> (train_ds, val_ds) PromptDatasets."""
    enc = get_encoding()
    ds = load_dataset("json", data_files=ALPACA_FARM + "alpaca_instructions/unlabeled.json", split="train")
    ds = ds.shuffle(seed=seed)
    val_rows = ds.select(range(val_size))
    train_end = len(ds) if max_train_examples is None else min(len(ds), val_size + max_train_examples)
    train_rows = ds.select(range(val_size, train_end))
    return PromptDataset(train_rows, enc, max_prompt_len), PromptDataset(val_rows, enc, max_prompt_len)


if __name__ == "__main__":
    # quick sanity check: python data.py
    enc = get_encoding()
    train_ds, val_ds = load_alpaca(val_size=200, max_train_examples=200)
    print(f"alpaca train examples: {len(train_ds)}, val examples: {len(val_ds)}")
    input_ids, labels = train_ds[0]
    first = int((labels != IGNORE_INDEX).nonzero()[0])
    print("prompt:\n", enc.decode(input_ids[:first + 1].tolist()))
    print("response (scored tokens):\n", enc.decode(labels[first:].tolist()))
    pref_train, pref_val = load_preferences(val_size=100, max_train_examples=100)
    print(f"preference pairs: train {len(pref_train)}, val {len(pref_val)}")
    prompt_train, prompt_val = load_prompts(val_size=100, max_train_examples=100)
    print(f"PPO prompts: train {len(prompt_train)}, val {len(prompt_val)}")
