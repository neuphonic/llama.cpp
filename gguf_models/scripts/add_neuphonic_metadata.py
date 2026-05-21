#!/usr/bin/env python3
"""Add neuphonic.* metadata from a model's config.json into an existing GGUF file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))
import gguf


def neuphonic_to_kv(neuphonic: dict, prefix: str = "neuphonic") -> dict[str, tuple[gguf.GGUFValueType, object]]:
    """Recursively flatten a neuphonic dict into dotted GGUF key-value pairs.
    Nested dicts are flattened further; anything that isn't a supported scalar,
    list, or dict is JSON-stringified and stored as a STRING."""
    result: dict[str, tuple[gguf.GGUFValueType, object]] = {}
    for k, v in neuphonic.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            result.update(neuphonic_to_kv(v, prefix=key))
        else:
            try:
                vtype = gguf.GGUFValueType.get_type(v)
                result[key] = (vtype, v)
            except ValueError:
                result[key] = (gguf.GGUFValueType.STRING, json.dumps(v))
    return result


def copy_with_extra_kv(
    reader: gguf.GGUFReader,
    writer: gguf.GGUFWriter,
    extra: dict[str, tuple[gguf.GGUFValueType, object]],
) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    for key, (vtype, val) in extra.items():
        writer.add_key_value(key, val, vtype)

    total_bytes = sum(t.n_bytes for t in reader.tensors)
    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)

    writer.close()


def load_config(config_arg: str) -> dict:
    """Accept a local path (dir or config.json) or a HuggingFace repo ID (org/model)."""
    if "/" in config_arg and not Path(config_arg).exists():
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=config_arg, filename="config.json")
        with open(local, encoding="utf-8") as f:
            return json.load(f)

    config_path = Path(config_arg)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    if not config_path.exists():
        print(f"Error: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",  type=Path, help="Input GGUF file")
    parser.add_argument("output", type=Path, help="Output GGUF file")
    parser.add_argument("config", type=str,  help="HuggingFace repo ID, local model dir, or path to config.json")
    args = parser.parse_args()

    config = load_config(args.config)

    neuphonic = config.get("neuphonic")
    if not neuphonic:
        print("Error: no 'neuphonic' key found in config.json", file=sys.stderr)
        sys.exit(1)

    extra = neuphonic_to_kv(neuphonic)
    print(f"Adding keys: {list(extra.keys())}")

    reader = gguf.GGUFReader(args.input, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)

    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    copy_with_extra_kv(reader, writer, extra)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
