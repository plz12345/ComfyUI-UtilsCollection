from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from unifiedefficientloader import IncrementalSafetensorsWriter, MemoryEfficientSafeOpen


ARCHITECTURE = "FFCResNetGenerator"
GENERATOR_PREFIXES = ("generator.", "model.generator.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_generator_state_dict(source: Path) -> tuple[dict[str, torch.Tensor], str]:
    scripted_model = torch.jit.load(str(source), map_location="cpu")
    source_state = scripted_model.state_dict()
    prefix = next(
        (
            candidate
            for candidate in GENERATOR_PREFIXES
            if source_state and all(key.startswith(candidate) for key in source_state)
        ),
        None,
    )
    if prefix is None:
        raise ValueError(f"TorchScript archive has unsupported state keys: {list(source_state)[:5]}")

    state_dict = {
        key.removeprefix(prefix): value.detach().cpu().contiguous()
        for key, value in source_state.items()
    }
    if not state_dict:
        raise ValueError("TorchScript archive contains no generator tensors")
    return state_dict, prefix


def verify_output(path: Path, expected: dict[str, torch.Tensor]) -> None:
    converted = MemoryEfficientSafeOpen(str(path), low_memory=True)
    try:
        actual_keys = set(converted.keys())
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Safetensors key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        ordered_keys = list(expected)
        stream = converted.async_stream(
            ordered_keys,
            batch_size=1,
            prefetch_batches=1,
            pin_memory=False,
        )
        try:
            for batch in stream:
                for key, converted_tensor in batch:
                    try:
                        source_tensor = expected[key]
                        if converted_tensor.shape != source_tensor.shape:
                            raise ValueError(f"Safetensors shape mismatch for {key}")
                        if converted_tensor.dtype != source_tensor.dtype:
                            raise ValueError(f"Safetensors dtype mismatch for {key}")
                        if not torch.equal(converted_tensor, source_tensor):
                            raise ValueError(f"Safetensors value mismatch for {key}")
                    finally:
                        converted.mark_processed(key)
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
    finally:
        converted.close()


def convert(source: Path, destination: Path, *, force: bool) -> None:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    state_dict, removed_prefix = extract_generator_state_dict(source)
    metadata = {
        "architecture": ARCHITECTURE,
        "format": "pt",
        "source_filename": source.name,
        "source_sha256": sha256_file(source),
        "state_dict_prefix_removed": removed_prefix,
        "generator_config": (
            "input_nc=4,output_nc=3,ngf=64,n_downsampling=3,n_blocks=18,"
            "add_out_act=sigmoid,ratio_gin=0.75,ratio_gout=0.75,enable_lfu=false"
        ),
    }

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with IncrementalSafetensorsWriter(
            str(temporary), metadata=metadata, max_workers=1
        ) as writer:
            for key, tensor in state_dict.items():
                writer.write_batch([(key, tensor)])
        verify_output(temporary, state_dict)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    print(f"Wrote {destination}")  # noqa: T201
    print(f"Tensors: {len(state_dict)}")  # noqa: T201
    print(f"Tensor bytes: {tensor_bytes}")  # noqa: T201
    print(f"Source SHA-256: {metadata['source_sha256']}")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Big LaMa generator tensors from TorchScript into Safetensors."
    )
    parser.add_argument("source", type=Path, help="Path to trusted big-lama.pt")
    parser.add_argument("destination", type=Path, help="Output .safetensors path")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    if args.destination.suffix.lower() != ".safetensors":
        parser.error("destination must use the .safetensors extension")
    convert(args.source, args.destination, force=args.force)


if __name__ == "__main__":
    main()
