import importlib.util
from pathlib import Path

import pytest
import torch
from unifiedefficientloader import MemoryEfficientSafeOpen

CONVERTER_PATH = Path(__file__).parents[1] / "scripts" / "convert_big_lama_to_safetensors.py"
SPEC = importlib.util.spec_from_file_location("uc_convert_big_lama", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class FakeScriptModule:
    def __init__(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict


def test_convert_strips_generator_prefix_and_records_metadata(tmp_path, monkeypatch):
    source = tmp_path / "big-lama.pt"
    source.write_bytes(b"trusted test fixture")
    destination = tmp_path / "big-lama.safetensors"
    expected = {
        "model.1.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "model.1.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
    }
    scripted_state = {f"generator.{key}": value for key, value in expected.items()}
    monkeypatch.setattr(
        converter.torch.jit,
        "load",
        lambda path, map_location: FakeScriptModule(scripted_state),
    )

    converter.convert(source, destination, force=False)

    converted = MemoryEfficientSafeOpen(str(destination), low_memory=True)
    try:
        assert set(converted.keys()) == set(expected)
        assert converted.metadata()["architecture"] == "FFCResNetGenerator"
        assert converted.metadata()["state_dict_prefix_removed"] == "generator."
        stream = converted.async_stream(
            list(expected),
            batch_size=1,
            prefetch_batches=1,
            pin_memory=False,
        )
        try:
            for batch in stream:
                for key, converted_tensor in batch:
                    try:
                        assert torch.equal(converted_tensor, expected[key])
                    finally:
                        converted.mark_processed(key)
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
    finally:
        converted.close()


def test_extract_rejects_non_generator_state(tmp_path, monkeypatch):
    source = tmp_path / "big-lama.pt"
    source.write_bytes(b"trusted test fixture")
    monkeypatch.setattr(
        converter.torch.jit,
        "load",
        lambda path, map_location: FakeScriptModule({"discriminator.weight": torch.ones(1)}),
    )

    with pytest.raises(ValueError, match="unsupported state keys"):
        converter.extract_generator_state_dict(source)


def test_extract_accepts_nested_model_generator_prefix(tmp_path, monkeypatch):
    source = tmp_path / "anime-manga-big-lama.pt"
    source.write_bytes(b"trusted test fixture")
    tensor = torch.ones(1)
    monkeypatch.setattr(
        converter.torch.jit,
        "load",
        lambda path, map_location: FakeScriptModule(
            {"model.generator.model.1.weight": tensor}
        ),
    )

    state_dict, removed_prefix = converter.extract_generator_state_dict(source)

    assert removed_prefix == "model.generator."
    assert set(state_dict) == {"model.1.weight"}
    assert torch.equal(state_dict["model.1.weight"], tensor)


def test_convert_refuses_to_overwrite(tmp_path):
    source = tmp_path / "big-lama.pt"
    source.write_bytes(b"trusted test fixture")
    destination = tmp_path / "big-lama.safetensors"
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        converter.convert(source, destination, force=False)
