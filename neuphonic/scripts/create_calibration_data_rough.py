"""
Create a GGUF imatrix calibration file from real training data.

Loads shards from the same config as used for SFT, applies the exact same preprocessing and filters
as training (preprocess_sample), then decodes each accepted sequence back to
text and writes one sequence per line.  No CFG, no padding.

Usage:
    conda run -n t5 python3 create_calibration_data.py \
        --config_fpath configs/cfg_nine_nano_bpe.yaml \
        --tokenizer_path neuphonic/qwen3-1.7b-9langs-grpo-2750-15-04-26 \
        --samples_per_lang 10 \
        --output_path calibration.txt
"""

import argparse
import os
import re
import random
import sys
import unicodedata

import torch
import webdataset as wds
from functools import partial
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from lang_info import LANG_INFO
from utils import load_shards_from_pattern, is_language_only
from preprocessing import ZHFrontend, JAFrontend

ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")
QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize_text(text: str, frontend) -> str:
    text = text.translate(QUOTE_MAP)
    if frontend is not None:
        text = frontend(text)
    else:
        text = unicodedata.normalize("NFKC", text)
    return text


def _adjacency_removal(tensor):
    values = tensor.tolist()
    if not values:
        return torch.tensor([])
    last_value = values[-1]
    end_count = 0
    for i in range(len(values) - 1, -1, -1):
        if values[i] == last_value:
            end_count += 1
        else:
            break
    result = values[: len(values) - end_count + 1] if end_count > 1 else values
    return torch.tensor(result)


def make_calibration_sequence(sample, tokenizer, lang_token, use_lang_token, frontend=None):
    """
    Identical filtering logic to preprocess_sample, but returns a decoded string
    instead of padded tensors.  Returns None if the sample should be skipped.
    """
    speech_gen_start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    speech_gen_end = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    text_replace = tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
    speech_replace = tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
    text_prompt_start = tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
    text_prompt_end = tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

    # --- unpack ---
    vq_codes = sample.get("vq_code.pth")
    if vq_codes is None:
        vq_codes = sample.get("codes.pth")
    if vq_codes is None:
        return None
    if vq_codes.dtype != torch.int64:
        vq_codes = vq_codes.to(torch.int64)

    try:
        text = sample["text.txt"]
        if lang_token != "<|ZH|>":
            possible_cut_index = int(((sample["duration.pth"] + 1) * 16_000) / 320)
    except Exception:
        return None

    # --- filters (exact match to training) ---
    if not text or len(text) == 0:
        return None
    if re.search(r"\d", text):
        return None
    if re.search(ACRONYM, text) or re.search(ACRONYM_NO_PERIOD, text):
        return None
    if "£" in text or "$" in text:
        return None
    if not is_language_only(text, lang_token):
        return None

    if lang_token != "<|ZH|>":
        if possible_cut_index < vq_codes.shape[0]:
            vq_codes = vq_codes[:possible_cut_index]
            vq_codes = _adjacency_removal(vq_codes)

    text = text.strip()
    if not text:
        return None
    text = normalize_text(text, frontend)
    if not text:
        return None

    # --- build token sequence (no padding) ---
    tokens = tokenizer.encode(text, add_special_tokens=False)

    lang_prefix = lang_token if use_lang_token else ""
    chat = f"{lang_prefix}<|TEXT_REPLACE|><|SPEECH_REPLACE|>"
    ids = tokenizer.encode(chat, add_special_tokens=True)

    text_replace_idx = ids.index(text_replace)
    ids = (
        ids[:text_replace_idx]
        + [text_prompt_start]
        + tokens
        + [text_prompt_end]
        + ids[text_replace_idx + 1 :]
    )

    speech_replace_idx = ids.index(speech_replace)
    codes_str = "".join(f"<|speech_{i}|>" for i in vq_codes.tolist())
    code_ids = tokenizer.encode(codes_str, add_special_tokens=False)
    ids = (
        ids[:speech_replace_idx]
        + [speech_gen_start]
        + code_ids
        + [speech_gen_end]
        + ids[speech_replace_idx + 1 :]
    )

    return tokenizer.decode(ids, skip_special_tokens=False)


def collect_samples(shards, process_fn, n_samples):
    dataset = (
        wds.WebDataset(
            shards,
            resampled=False,
            nodesplitter=wds.split_by_node,
            handler=wds.warn_and_continue,
            shardshuffle=True,
            empty_check=False,
        )
        .decode(handler=wds.warn_and_continue)
        .shuffle(2000, initial=2000)
        .map(process_fn)
        .select(lambda x: x is not None)
    )

    collected = []
    for seq in dataset:
        collected.append(seq)
        if len(collected) >= n_samples:
            break
    return collected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_fpath", required=True)
    parser.add_argument("--tokenizer_path", default="neuphonic/qwen3-1.7b-9langs-grpo-2750-15-04-26", help="HuggingFace repo or local dir containing the tokenizer")
    parser.add_argument("--samples_per_lang", type=int, default=100)
    parser.add_argument("--output_path", default="calibration.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config_fpath)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    use_lang_token = len(config.language_info) > 1

    all_sequences = []

    for language, info in config.language_info.items():
        print(f"\n[{language}] gathering shards...")
        shards = []
        for pattern in info.shard_pattern:
            shards.extend(load_shards_from_pattern(pattern))
        if not shards:
            print(f"  WARNING: no shards found for {language}, skipping")
            continue
        print(f"  {len(shards)} shards found")

        lang_token = LANG_INFO[language]["token"]
        if language == "chinese":
            frontend = ZHFrontend()
        elif language == "japanese":
            frontend = JAFrontend()
        else:
            frontend = None

        process_fn = partial(
            make_calibration_sequence,
            tokenizer=tokenizer,
            lang_token=lang_token,
            use_lang_token=use_lang_token,
            frontend=frontend,
        )

        seqs = collect_samples(shards, process_fn, args.samples_per_lang)
        print(f"  collected {len(seqs)} / {args.samples_per_lang} samples")
        all_sequences.extend(seqs)

    random.shuffle(all_sequences)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for seq in all_sequences:
            f.write(seq + "\n")

    print(f"\nWrote {len(all_sequences)} sequences to {args.output_path}")


if __name__ == "__main__":
    main()
