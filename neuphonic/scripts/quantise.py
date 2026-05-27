#!/usr/bin/env python3
"""Neuphonic GGUF quantisation pipeline.

Steps (each can be skipped with --skip-*):
  1. convert   : HuggingFace repo → BF16 GGUF via convert_hf_to_gguf.py
  2. add-meta  : inject neuphonic.* KV pairs from config.json into the BF16 GGUF
  3. imatrix   : compute per-tensor importance matrix from a calibration corpus
  4. quantize  : quantise BF16 GGUF to one or more target formats using the imatrix

Usage examples:
  # Full pipeline, one quant type (requires huggingface-cli login or HF_TOKEN env var)
  python neuphonic/scripts/quantise.py neuphonic/qwen3-0.2b-... Q4_0 \\
      --calibration-file neuphonic/calibration_multiling.txt

  # Multiple quant types in one run
  python neuphonic/scripts/quantise.py neuphonic/qwen3-0.2b-... Q4_0 Q8_0 \\
      --calibration-file neuphonic/calibration_multiling.txt

  # Skip conversion (BF16 GGUF already exists)
  python neuphonic/scripts/quantise.py neuphonic/qwen3-0.2b-... Q4_0 --skip-convert \\
      --calibration-file neuphonic/calibration_multiling.txt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent


def run(cmd: list[str | Path], env: dict | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(cmd, env=merged_env)
    if result.returncode != 0:
        print(f"\nError: command exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("repo", help="HuggingFace repo ID (e.g. neuphonic/my-model)")
    parser.add_argument("quant_types", nargs="+", metavar="QUANT_TYPE",
                        help="One or more quantisation types (e.g. Q4_0 Q8_0 IQ4_XS)")

    io = parser.add_argument_group("paths")
    io.add_argument("--model-name", help="Override model name (default: last component of repo ID)")
    io.add_argument("--output-dir", type=Path,
                    help="Directory for all outputs (default: neuphonic/<model-name>)")
    io.add_argument(
        "--calibration-file",
        type=Path,
        required=True,
        help="Calibration corpus for imatrix",
    )
    io.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build",
                    help="llama.cpp build directory (default: ./build)")

    quant = parser.add_argument_group("quantisation options")
    quant.add_argument("--output-tensor-type", default="f16",
                       help="Type for output tensors in llama-quantize (default: f16)")
    quant.add_argument("--token-embedding-type", default="f16",
                       help="Type for token embedding tensors (default: f16)")
    quant.add_argument("--context-size", type=int, default=2048,
                       help="Context size for imatrix calibration (default: 2048)")
    quant.add_argument("--ngl", type=int, default=999,
                       help="GPU layers for imatrix (default: 999)")

    skip = parser.add_argument_group("skip steps")
    skip.add_argument("--skip-convert",  action="store_true", help="Skip HF → BF16 GGUF conversion")
    skip.add_argument("--skip-meta",     action="store_true", help="Skip adding neuphonic metadata")
    skip.add_argument("--skip-imatrix",  action="store_true", help="Skip importance matrix computation")
    skip.add_argument("--skip-quantize", action="store_true", help="Skip quantisation")

    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (default: resolved from huggingface-cli login or $HF_TOKEN)",
    )

    return parser.parse_args()


def resolve_hf_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    model_name = args.model_name or args.repo.split("/")[-1]
    output_dir = args.output_dir or (REPO_ROOT / "neuphonic" / model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_file = args.calibration_file

    bf16_gguf   = output_dir / f"{model_name}_BF16.gguf"
    meta_gguf = (
        output_dir / f"{model_name}_BF16_meta.gguf"
    )  # temporary; renamed to bf16_gguf after step 2
    imatrix_out = output_dir / "imatrix.gguf"

    # -------------------------------------------------------------------------
    # Step 1: Convert HuggingFace model → BF16 GGUF
    # -------------------------------------------------------------------------
    if not args.skip_convert:
        print("=== Step 1: Convert HF → BF16 GGUF ===")
        hf_token = resolve_hf_token(args.hf_token)
        env = {"HF_TOKEN": hf_token} if hf_token else {}
        run([
            sys.executable, REPO_ROOT / "convert_hf_to_gguf.py",
            "--remote",
            "--outtype", "bf16",
            "--outfile", bf16_gguf,
            args.repo,
        ], env=env)
    else:
        print("=== Step 1: skipped ===")

    # -------------------------------------------------------------------------
    # Step 2: Inject neuphonic.* metadata into the BF16 GGUF
    # Reads the 'neuphonic' dict from config.json and writes all key-value pairs
    # as dotted GGUF metadata so inference servers don't need a separate config.
    # -------------------------------------------------------------------------
    if not args.skip_meta:
        print("=== Step 2: Add neuphonic metadata ===")
        run([
            sys.executable, SCRIPTS_DIR / "add_neuphonic_metadata.py",
            bf16_gguf,
            meta_gguf,
            args.repo,
        ])
        bf16_gguf.unlink()
        shutil.move(meta_gguf, bf16_gguf)
    else:
        print("=== Step 2: skipped ===")

    # -------------------------------------------------------------------------
    # Step 3: Compute importance matrix (imatrix)
    # Runs the BF16 model on a calibration corpus and records per-tensor
    # activation statistics. These guide quantisation to spend more bits on
    # sensitive weights (e.g. attention projections) and fewer on insensitive ones.
    # --parse-special is required so speech tokens pass through the tokenizer.
    # -------------------------------------------------------------------------
    if not args.skip_imatrix:
        print("=== Step 3: Compute importance matrix ===")
        run([
            args.build_dir / "bin" / "llama-imatrix",
            "-m", bf16_gguf,
            "-f", calibration_file,
            "-o", imatrix_out,
            "-c", str(args.context_size),
            "--parse-special",
            "--no-ppl",
            "-ngl", str(args.ngl),
        ])
    else:
        print("=== Step 3: skipped ===")

    # -------------------------------------------------------------------------
    # Step 4: Quantize to each requested format
    # Output and embedding tensors are kept at --output-tensor-type /
    # --token-embedding-type (default f16) because they're small but critical —
    # the speech token logits live in the output tensor.
    # -------------------------------------------------------------------------
    if not args.skip_quantize:
        for quant_type in args.quant_types:
            quant_gguf = output_dir / f"{model_name}_{quant_type}.gguf"
            print(f"=== Step 4: Quantize → {quant_type} ===")
            run([
                args.build_dir / "bin" / "llama-quantize",
                "--imatrix", imatrix_out,
                "--output-tensor-type", args.output_tensor_type,
                "--token-embedding-type", args.token_embedding_type,
                bf16_gguf,
                quant_gguf,
                quant_type,
            ])
    else:
        print("=== Step 4: skipped ===")

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
