import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODULE_FUNCTIONS = {
    "encoder_nodes.py": {
        "apply_parallel_ref_latents",
        "multiply_conditioning",
        "system_prompt_template",
        "_mark_deprecated_node",
    },
    "image_nodes.py": {
        "_blend_luminosity",
        "_blend_saturation",
        "_clip_blend_color",
        "_set_blend_luminosity",
        "_set_blend_saturation",
        "_blend_rgb",
    },
    "logic_math_nodes.py": {
        "_math_template",
        "_math_autogrow",
        "_boolean_autogrow",
    },
    "preset_nodes.py": {"escape_prompt_parentheses"},
    "textgen_nodes.py": {
        "evaluate_formula",
        "process_vlm_image",
        "_aligned_image_values",
        "_qwen_clip_model",
        "_token_rows",
        "_visual_grid_metadata",
        "_load_qwen_generation_model",
        "_encode_qwen_visual_sources",
        "_fuse_qwen_primary_sources",
        "generate_fused_qwen3vl",
        "generate_fused_qwen35",
        "detect_textgen_template",
    },
}


def test_node_modules_add_no_standalone_helpers():
    violations = {}
    for path in sorted((REPOSITORY_ROOT / "nodes").glob("*_nodes.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        actual = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        unexpected = actual - LEGACY_MODULE_FUNCTIONS.get(path.name, set())
        if unexpected:
            violations[path.name] = sorted(unexpected)

    assert not violations, (
        "Move standalone node-module helpers into focused *_helpers.py modules; "
        f"do not expand the legacy allowlist: {violations}"
    )


def _iter_repo_python_files():
    this_file = Path(__file__).resolve()
    for folder in ("nodes", "helpers", "models", "scripts", "tests"):
        for path in sorted((REPOSITORY_ROOT / folder).rglob("*.py")):
            if path.resolve() != this_file:
                yield path


def test_repository_uses_uel_async_safetensors_io_only():
    forbidden_Save_or_load = (
        ".get_tensor(",
        "from safetensors.torch",
        "from safetensors import safe_open",
        "import safetensors",
        "save_file(",
        "safe_open(",
        "writer.write(",
    )
    violations = []
    for path in _iter_repo_python_files():
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPOSITORY_ROOT).as_posix()
        for token in forbidden_Save_or_load:
            if token in source:
                violations.append(f"{rel}: {token}")
    assert violations == [], (
        "Use UnifiedEfficientLoader async_stream + mark_processed and "
        f"atomic_uel_writer + write_batch for all safetensors I/O: {violations}"
    )


def test_repository_has_no_eval_calls_for_registry_scanner():
    violations = []
    for path in _iter_repo_python_files():
        source = path.read_text(encoding="utf-8")
        if ".eval(" in source:
            violations.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    assert violations == [], (
        "Avoid .eval( in repository modules (flagged by Comfy Registry scanner); "
        f"use .train(False) instead: {violations}"
    )


def test_atomic_uel_writer_preserves_destination_on_failure_and_roundtrips(tmp_path):
    import importlib.util
    import pytest
    import torch

    spec = importlib.util.spec_from_file_location(
        "uc_helper_functions_uel_test",
        REPOSITORY_ROOT / "helpers" / "helper_functions.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "output.safetensors"
    with module.atomic_uel_writer(output, {"version": "1"}) as writer:
        writer.write_batch([("old", torch.tensor([1.0, 2.0]))])
    original = output.read_bytes()

    with pytest.raises(RuntimeError, match="planned failure"):
        with module.atomic_uel_writer(output, {"version": "2"}) as writer:
            writer.write_batch([("new", torch.tensor([3.0]))])
            raise RuntimeError("planned failure")

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".output.safetensors.*.tmp")) == []

    tensors, metadata = module.read_uel_safetensors_dict(output)
    assert metadata == {"version": "1"}
    assert list(tensors) == ["old"]
    assert torch.equal(tensors["old"], torch.tensor([1.0, 2.0]))

