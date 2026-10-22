from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.nn import functional as F

import folder_paths
import comfy.model_management as model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from unifiedefficientloader import MemoryEfficientSafeOpen

from .helper_functions import gaussian_blur_nchw
from ..models.lama import FFCResNetGenerator


LAMA_ARCHITECTURE = "FFCResNetGenerator"


def get_lama_model_names() -> list[str]:
    return [
        name
        for name in folder_paths.get_filename_list("lama")
        if name.lower().endswith(".safetensors")
    ]


def get_lama_device_options() -> list[str]:
    devices = _lama_accelerator_devices()
    return ["default", "cpu", *(f"gpu:{index}" for index in range(len(devices)))]


def _lama_accelerator_devices() -> list[torch.device]:
    return [
        device
        for device in model_management.get_all_torch_devices()
        if device.type != "cpu"
    ]


def resolve_lama_device(device_option: str) -> torch.device:
    if device_option == "cpu":
        return torch.device("cpu")
    if device_option and device_option.startswith("gpu:"):
        try:
            index = int(device_option[4:])
        except ValueError:
            index = -1
        devices = _lama_accelerator_devices()
        if 0 <= index < len(devices):
            return devices[index]
    if device_option not in (None, "default"):
        logging.warning(
            "LaMa loader: device '%s' is unavailable; using ComfyUI default device.",
            device_option,
        )
    return model_management.get_torch_device()


def _validate_lama_checkpoint(handler, model: FFCResNetGenerator) -> list[str]:
    expected = model.state_dict()
    expected_keys = list(expected)
    actual_keys = list(handler.keys())
    missing = sorted(set(expected_keys) - set(actual_keys))
    unexpected = sorted(set(actual_keys) - set(expected_keys))
    if missing or unexpected:
        raise ValueError(
            f"Invalid LaMa checkpoint keys: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    metadata = handler.metadata() or {}
    architecture = metadata.get("architecture")
    if architecture is not None and architecture != LAMA_ARCHITECTURE:
        raise ValueError(f"Unsupported LaMa architecture metadata: {architecture}")
    for key, placeholder in expected.items():
        if tuple(handler.get_shape(key)) != tuple(placeholder.shape):
            raise ValueError(f"Invalid LaMa tensor shape for {key}")
        if handler.get_dtype(key) != placeholder.dtype:
            raise ValueError(f"Invalid LaMa tensor dtype for {key}")
    return expected_keys


def load_lama_model(model_path: str | Path, device_option: str):
    model_path = Path(model_path)
    model = FFCResNetGenerator(comfy.ops.disable_weight_init)
    handler = MemoryEfficientSafeOpen(str(model_path), low_memory=True)
    try:
        keys = _validate_lama_checkpoint(handler, model)
        stream = handler.async_stream(
            keys,
            batch_size=1,
            prefetch_batches=1,
            pin_memory=False,
        )
        consumed = 0
        try:
            for batch in stream:
                for key, tensor in batch:
                    expected_key = keys[consumed]
                    if key != expected_key:
                        raise RuntimeError(
                            f"UEL stream order mismatch: expected {expected_key}, received {key}"
                        )
                    comfy.utils.copy_to_param(model, key, tensor)
                    handler.mark_processed(key)
                    consumed += 1
            if consumed != len(keys):
                raise RuntimeError(
                    f"UEL stream ended after {consumed} of {len(keys)} LaMa tensors"
                )
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
    finally:
        handler.close()

    model.train(False)
    load_device = resolve_lama_device(device_option)
    offload_device = (
        torch.device("cpu")
        if device_option == "cpu" or device_option not in (None, "default")
        else model_management.unet_offload_device()
    )
    model.to(offload_device)
    return comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
    )


def _broadcast_lama_mask(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("LaMa Inpaint requires IMAGE shaped [batch, height, width, 3].")
    if masks.ndim != 3:
        raise ValueError("LaMa Inpaint requires MASK shaped [batch, height, width].")
    batch = images.shape[0]
    if masks.shape[0] == 1 and batch != 1:
        masks = masks.expand(batch, -1, -1)
    elif masks.shape[0] != batch:
        raise ValueError("LaMa mask batch must be one or match the image batch.")
    if masks.shape[1:3] != images.shape[1:3]:
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=images.shape[1:3],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return masks.clamp(0.0, 1.0)


def prepare_lama_inputs(
    images: torch.Tensor,
    masks: torch.Tensor,
    mask_threshold: int,
    gaussblur_radius: int,
    invert_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    masks = _broadcast_lama_mask(images, masks)
    images_nchw = images.movedim(-1, 1).float()
    masks_nchw = masks.unsqueeze(1).float()
    height, width = images_nchw.shape[-2:]
    padded_height = (height + 7) // 8 * 8
    padded_width = (width + 7) // 8 * 8
    padding = (0, padded_width - width, 0, padded_height - height)
    images_nchw = F.pad(images_nchw, padding)
    masks_nchw = F.pad(masks_nchw, padding)

    working_mask = masks_nchw if invert_mask else 1.0 - masks_nchw
    working_mask = gaussian_blur_nchw(working_mask, sigma_px=gaussblur_radius)
    binary_mask = (working_mask <= mask_threshold / 255.0).to(images_nchw.dtype)
    return images_nchw, binary_mask, (height, width)


def run_lama_inpaint(
    patcher,
    images: torch.Tensor,
    masks: torch.Tensor,
    mask_threshold: int,
    gaussblur_radius: int,
    invert_mask: bool,
) -> torch.Tensor:
    images_nchw, binary_mask, original_size = prepare_lama_inputs(
        images, masks, mask_threshold, gaussblur_radius, invert_mask
    )
    model_management.load_models_gpu([patcher])
    device = patcher.load_device
    dtype = patcher.model_dtype()
    images_nchw = images_nchw.to(device=device, dtype=dtype)
    binary_mask = binary_mask.to(device=device, dtype=dtype)
    masked_image = images_nchw * (1.0 - binary_mask)
    generated = patcher.model(torch.cat((masked_image, binary_mask), dim=1))
    result = generated * binary_mask + images_nchw * (1.0 - binary_mask)
    height, width = original_size
    result = result[:, :, :height, :width].movedim(1, -1).clamp_(0.0, 1.0)
    return result.to(model_management.intermediate_device())


def image_mask_to_luma(mask_images: torch.Tensor) -> torch.Tensor:
    if mask_images.ndim != 4 or mask_images.shape[-1] < 3:
        raise ValueError("Legacy LaMa image mask requires an RGB IMAGE batch.")
    weights = mask_images.new_tensor((0.299, 0.587, 0.114))
    return (mask_images[..., :3] * weights).sum(dim=-1)
