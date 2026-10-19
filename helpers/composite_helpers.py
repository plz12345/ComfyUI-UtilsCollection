import json
import logging
import math
import re

import torch
import torch.nn.functional as F

from comfy_api.latest import io, ui

from .helper_functions import resize_nchw
from ..model_assets import require_huggingface_model


_RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

_COMPOSITE_RESIZE_METHODS = ["auto", *_RESIZE_METHODS]

_STAGED_BACKGROUND_OPTION_DEFAULTS = {
    "mask_threshold": 0.5,
    "border_cleanup_width": 2,
    "artifact_cleanup_radius": 2,
    "gap_fill_radius": 2,
    "mask_processing_resolution": 0,
    "feather_radius": 2,
    "image_resize_method": "auto",
    "mask_resize_method": "auto",
    "foreground_blend": 1.0,
}

_PAINT_LAYER_KEY = "__uc_paint__"

_DEFAULT_LAYER_PLACEMENT = {
    "scale": 0.9,
    "long_axis_shift": 0.0,
    "short_axis_shift": 0.0,
}

_DEFAULT_LAYER_PLACEMENT_V2 = {"scale": 0.9, "center_x": 0.5, "center_y": 0.5}

_BACKGROUND_REMOVAL_MODEL_FILES = {
    "birefnet": "birefnet.safetensors",
    "lucida": "lucida.safetensors",
}

_LUCIDA_IMAGE_MEAN = [0.485, 0.456, 0.406]

_LUCIDA_IMAGE_STD = [0.229, 0.224, 0.225]

def _load_internal_background_removal_model(model_name):
    selected = str(model_name or "birefnet").lower()
    filename = _BACKGROUND_REMOVAL_MODEL_FILES.get(selected)
    if filename is None:
        choices = ", ".join(_BACKGROUND_REMOVAL_MODEL_FILES)
        raise ValueError(
            f"Unsupported internal background-removal model {model_name!r}; choose {choices}."
        )

    from comfy import bg_removal_model

    model_path = require_huggingface_model(
        "background_removal",
        filename,
        "Comfy-Org/BiRefNet",
        f"background_removal/{filename}",
    )

    try:
        model = bg_removal_model.load(model_path)
    except Exception as exc:
        raise ValueError(
            f"Comfy Core could not load {filename} as the {selected} background-removal model."
        ) from exc
    if model is None or not callable(getattr(model, "encode_image", None)):
        raise ValueError(
            f"Comfy Core did not recognize {filename} as a supported background-removal model."
        )

    if selected == "lucida":
        model.image_size = 1024
        model.image_mean = list(_LUCIDA_IMAGE_MEAN)
        model.image_std = list(_LUCIDA_IMAGE_STD)
        if isinstance(getattr(model, "config", None), dict):
            model.config.update(
                {
                    "image_size": model.image_size,
                    "image_mean": list(model.image_mean),
                    "image_std": list(model.image_std),
                }
            )

    return model


def resolve_background_removal_model(model=None):
    """Use a connected Core model or the internal BiRefNet default."""
    return model if model is not None else _load_internal_background_removal_model("birefnet")


def background_removal_with_alpha(image, model, background_options=None):
    """Return source-resolution straight RGBA and its exact alpha mask."""
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("Background Removal (Preserve Alpha) requires a batched IMAGE.")
    if image.shape[-1] < 3:
        raise ValueError("Input images must have at least three channels.")
    rgb = image[..., :3]
    if image.shape[-1] >= 4:
        alpha = image[..., 3].to(rgb).clamp(0.0, 1.0)
    else:
        if model is None:
            raise ValueError("RGB background removal requires a removal model.")
        if background_options is not None and not isinstance(background_options, dict):
            raise ValueError("Background Options must be a dictionary.")
        options = (
            None
            if background_options is None
            else _STAGED_BACKGROUND_OPTION_DEFAULTS | background_options
        )
        mask_input = rgb
        if options is not None:
            requested_size = max(0, int(options["mask_processing_resolution"]))
            model_size = int(getattr(model, "image_size", 0) or 0)
            processing_size = requested_size or model_size
            source_height, source_width = rgb.shape[1:3]
            if processing_size > 0 and max(source_height, source_width) > processing_size:
                scale = processing_size / max(source_height, source_width)
                mask_input = _resize_composite_image(
                    rgb,
                    max(1, round(source_width * scale)),
                    max(1, round(source_height * scale)),
                    "auto",
                )

        raw_mask = model.encode_image(mask_input)
        if not torch.is_tensor(raw_mask):
            raise ValueError("Background removal returned an invalid mask.")
        if raw_mask.ndim == 4 and raw_mask.shape[1] == 1:
            raw_mask = raw_mask[:, 0]
        elif raw_mask.ndim == 4 and raw_mask.shape[-1] == 1:
            raw_mask = raw_mask[..., 0]
        if raw_mask.ndim != 3 or raw_mask.shape[0] != rgb.shape[0]:
            raise ValueError(
                "Background removal must return one [batch, height, width] mask per image."
            )
        if raw_mask.shape[-2:] != mask_input.shape[1:3]:
            raw_mask = _resize_composite_mask(
                raw_mask,
                mask_input.shape[2],
                mask_input.shape[1],
                "bilinear" if options is None else options["mask_resize_method"],
            )
        raw_mask = raw_mask.to(rgb).clamp(0.0, 1.0)
        if options is None:
            alpha = raw_mask
        else:
            alpha = torch.stack(
                [
                    _refine_foreground_mask(
                        mask,
                        float(options["mask_threshold"]),
                        options["border_cleanup_width"],
                        options["artifact_cleanup_radius"],
                        options["gap_fill_radius"],
                    )
                    for mask in raw_mask
                ]
            )
            if alpha.shape[-2:] != rgb.shape[1:3]:
                alpha = _resize_composite_mask(
                    alpha,
                    rgb.shape[2],
                    rgb.shape[1],
                    options["mask_resize_method"],
                )
            feather_radius = max(0, int(options["feather_radius"]))
            if feather_radius:
                alpha = torch.stack(
                    [_feather_mask(mask, -feather_radius) for mask in alpha]
                )
    return torch.cat((rgb, alpha.unsqueeze(-1)), dim=-1), alpha


def _resize_image(image, width, height, method, crop="disabled"):
    return resize_nchw(image.movedim(-1, 1), width, height, method, crop).movedim(1, -1)


def _resize_mask(mask, width, height, method, crop="disabled"):
    return resize_nchw(mask.unsqueeze(1), width, height, method, crop).squeeze(1)


def _composite_resize_method(
    method, source_width, source_height, width, height, mask=False
):
    if method not in _COMPOSITE_RESIZE_METHODS:
        raise ValueError(f"Unsupported composite resize method: {method!r}.")
    if method != "auto":
        return method
    shrinking = width < source_width or height < source_height
    reducing_both_axes = width <= source_width and height <= source_height
    if mask:
        return "area" if reducing_both_axes and shrinking else "bilinear"
    return "area" if reducing_both_axes and shrinking else "bicubic"


def _resize_composite_image(image, width, height, method):
    source_height, source_width = image.shape[1:3]
    if (source_width, source_height) == (width, height):
        return image
    selected = _composite_resize_method(
        method, source_width, source_height, width, height
    )
    return _resize_image(image, width, height, selected)


def _resize_composite_mask(mask, width, height, method):
    source_height, source_width = mask.shape[-2:]
    if (source_width, source_height) == (width, height):
        resized = mask
        selected = method
    else:
        selected = _composite_resize_method(
            method, source_width, source_height, width, height, mask=True
        )
        resized = _resize_mask(mask, width, height, selected)
    if selected == "nearest-exact":
        return (resized >= 0.5).to(resized)
    return resized.clamp(0.0, 1.0)


def _broadcast_batch(value, batch_size, name):
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] == 1:
        return value.expand(batch_size, *value.shape[1:])
    raise ValueError(f"{name} batch size must be 1 or {batch_size}.")


def _blur_mask(mask, radius):
    if radius <= 0:
        return mask
    sigma = max(float(radius) / 3.0, 0.1)
    kernel_radius = max(1, int(math.ceil(sigma * 3.0)))
    coordinates = torch.arange(
        -kernel_radius, kernel_radius + 1, device=mask.device, dtype=mask.dtype
    )
    kernel = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    mask = F.conv2d(
        mask.unsqueeze(1), kernel.view(1, 1, 1, -1), padding=(0, kernel_radius)
    )
    mask = F.conv2d(mask, kernel.view(1, 1, -1, 1), padding=(kernel_radius, 0))
    return mask.squeeze(1)


def _expand_mask(mask, amount):
    if amount == 0:
        return mask
    radius = abs(int(amount))
    kernel = 2 * radius + 1
    mask = mask.unsqueeze(0).unsqueeze(0)
    if amount > 0:
        mask = F.max_pool2d(mask, kernel, stride=1, padding=radius)
    else:
        mask = 1.0 - F.max_pool2d(1.0 - mask, kernel, stride=1, padding=radius)
    return mask[0, 0]


def _feather_mask(mask, radius):
    if radius == 0:
        return mask
    radius = int(radius)
    if radius < 0:
        contracted = _expand_mask(mask, -max(1, math.ceil(abs(radius) / 2)))
        blurred = _blur_mask(contracted.unsqueeze(0), abs(radius))[0].clamp(0.0, 1.0)
        return torch.minimum(mask, blurred)
    blurred = _blur_mask(mask.unsqueeze(0), radius)[0].clamp(0.0, 1.0)
    return torch.maximum(mask, blurred)


def _binary_dilate(mask, radius):
    radius = max(0, int(radius))
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    padded = F.pad(mask[None, None], (radius, radius, radius, radius), value=0.0)
    return F.max_pool2d(padded, kernel, stride=1)[0, 0]


def _binary_erode(mask, radius):
    radius = max(0, int(radius))
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    padded = F.pad(mask[None, None], (radius, radius, radius, radius), value=0.0)
    return 1.0 - F.max_pool2d(1.0 - padded, kernel, stride=1)[0, 0]


def _refine_foreground_mask(
    raw_mask, threshold, border_cleanup_width, artifact_cleanup_radius, gap_fill_radius
):
    raw_mask = raw_mask.clamp(0.0, 1.0)
    mask = (raw_mask >= threshold).to(raw_mask)
    border_width = min(max(0, int(border_cleanup_width)), min(mask.shape) // 2)
    if border_width:
        height, width = mask.shape
        rows = torch.arange(height, device=mask.device)
        columns = torch.arange(width, device=mask.device)
        border = (
            (rows[:, None] < border_width)
            | (rows[:, None] >= height - border_width)
            | (columns[None, :] < border_width)
            | (columns[None, :] >= width - border_width)
        )
        strong_threshold = min(1.0, float(threshold) + 0.25)
        mask = mask * (~(border & (raw_mask < strong_threshold))).to(mask)
    if artifact_cleanup_radius:
        mask = _binary_dilate(
            _binary_erode(mask, artifact_cleanup_radius), artifact_cleanup_radius
        )
    if gap_fill_radius:
        mask = _binary_erode(_binary_dilate(mask, gap_fill_radius), gap_fill_radius)
    return (mask >= 0.5).to(raw_mask)


def _flatten_autogrow_images(image_inputs):
    images = []
    for key in sorted(
        image_inputs or {},
        key=lambda value: int("".join(filter(str.isdigit, value)) or 0),
    ):
        value = image_inputs[key]
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None:
                continue
            if not torch.is_tensor(item) or item.ndim != 4:
                raise ValueError(
                    f"Foreground input {key} must have shape [batch, height, width, channels]."
                )
            images.extend(item[index : index + 1] for index in range(item.shape[0]))
    return images


def _foreground_input_order(key):
    digits = "".join(filter(str.isdigit, key))
    return int(digits or 0), key


def _ordered_single_foregrounds(image_inputs):
    foregrounds = []
    for key in sorted(image_inputs or {}, key=_foreground_input_order):
        value = image_inputs[key]
        values = value if isinstance(value, (list, tuple)) else [value]
        values = [item for item in values if item is not None]
        if len(values) != 1 or not torch.is_tensor(values[0]) or values[0].ndim != 4:
            raise ValueError(
                f"Foreground input {key} must contain exactly one image tensor."
            )
        image = values[0]
        if image.shape[0] != 1:
            raise ValueError(
                f"Foreground input {key} must contain exactly one image, not a batch."
            )
        foregrounds.append((key, image))
    return foregrounds


def _parse_layer_payload(value):
    if value is None or value == "":
        value = {}
    if isinstance(value, str):
        if value.strip() in _COMPOSITE_RESIZE_METHODS:
            logging.warning(
                "Detected legacy layered-composite widget ordering; using default placement data. "
                "Reload the workflow in the frontend once to migrate its saved values."
            )
            value = {}
        else:
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Layer placement data is not valid JSON: {error.msg}."
                ) from error
    if not isinstance(value, dict):
        raise ValueError("Layer placement data must be a JSON object.")
    version = value.get("version", 1)
    if version not in (1, 2, 3):
        raise ValueError(f"Unsupported layer placement data version: {version}.")
    layers = value.get("layers", {})
    if not isinstance(layers, dict):
        raise ValueError("Layer placement data 'layers' must be a JSON object.")
    layer_order = value.get("layer_order", [])
    if not isinstance(layer_order, list) or any(
        not isinstance(key, str) for key in layer_order
    ):
        raise ValueError(
            "Layer placement data 'layer_order' must be an array of layer identifiers."
        )
    workspace_padding = value.get("workspace_padding", 0.5)
    if isinstance(workspace_padding, bool):
        raise ValueError("Layer placement workspace_padding must be numeric.")
    try:
        workspace_padding = float(workspace_padding)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Layer placement workspace_padding must be numeric."
        ) from error
    if not math.isfinite(workspace_padding) or not 0.0 <= workspace_padding <= 1.0:
        raise ValueError("Layer placement workspace_padding must be between 0 and 1.")
    return version, layers, layer_order, workspace_padding


def _parse_paint_layer(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        if value.strip() in _COMPOSITE_RESIZE_METHODS:
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Layer placement data is not valid JSON: {error.msg}."
            ) from error
    if not isinstance(value, dict):
        raise ValueError("Layer placement data must be a JSON object.")
    paint = value.get("paint_layer")
    if paint is None:
        return None
    if not isinstance(paint, dict):
        raise ValueError("Layer placement paint_layer must be an object.")
    included = paint.get("included", True)
    if not isinstance(included, bool):
        raise ValueError("Layer placement paint_layer included must be Boolean.")
    asset = paint.get("asset")
    if asset is None:
        return None
    if not isinstance(asset, dict):
        raise ValueError("Layer placement paint_layer asset must be an object.")
    filename = asset.get("filename")
    subfolder = asset.get("subfolder", "clipspace")
    folder_type = asset.get("type", "input")
    if not isinstance(filename, str) or not filename:
        raise ValueError("Layer placement paint_layer asset filename is missing.")
    if not isinstance(subfolder, str):
        raise ValueError("Layer placement paint_layer asset subfolder must be text.")
    if folder_type != "input":
        raise ValueError("Layer placement paint_layer asset must use input storage.")
    return {
        "included": included,
        "asset": {
            "filename": filename,
            "subfolder": subfolder,
            "type": "input",
        },
    }


def _parse_layer_placements(value):
    version, layers, _, _ = _parse_layer_payload(value)

    parsed = {}
    for key, placement in layers.items():
        if not isinstance(key, str) or not isinstance(placement, dict):
            raise ValueError(
                "Every layer placement must be an object keyed by its foreground socket name."
            )
        result = dict(
            _DEFAULT_LAYER_PLACEMENT_V2 if version >= 2 else _DEFAULT_LAYER_PLACEMENT
        )
        fields = [("scale", 0.05, 10.0)]
        if version >= 2:
            fields.extend((("center_x", -10.0, 10.0), ("center_y", -10.0, 10.0)))
        else:
            fields.extend(
                (("long_axis_shift", -1.0, 1.0), ("short_axis_shift", -1.0, 1.0))
            )
        for field, minimum, maximum in fields:
            raw = placement.get(field, result[field])
            if isinstance(raw, bool):
                raise ValueError(f"Layer {key} field {field} must be numeric.")
            try:
                number = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Layer {key} field {field} must be numeric."
                ) from error
            if not math.isfinite(number) or number < minimum or number > maximum:
                raise ValueError(
                    f"Layer {key} field {field} must be between {minimum} and {maximum}."
                )
            result[field] = number
        for flip_field in ("flip_horizontal", "flip_vertical"):
            flip_value = placement.get(flip_field, False)
            if not isinstance(flip_value, bool):
                raise ValueError(f"Layer {key} field {flip_field} must be Boolean.")
            result[flip_field] = flip_value
        if version == 3:
            rotation = placement.get("rotation", 0.0)
            if isinstance(rotation, bool):
                raise ValueError(f"Layer {key} field rotation must be numeric.")
            try:
                rotation = float(rotation)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Layer {key} field rotation must be numeric."
                ) from error
            if not math.isfinite(rotation):
                raise ValueError(f"Layer {key} field rotation must be finite.")
            result["rotation"] = ((rotation + 180.0) % 360.0) - 180.0
            corners = placement.get("corners", [[-1, -1], [1, -1], [1, 1], [-1, 1]])
            result["corners"] = _validate_quad(corners, key)
            included = placement.get("included", True)
            if not isinstance(included, bool):
                raise ValueError(f"Layer {key} field included must be Boolean.")
            result["included"] = included
        result["_version"] = version
        parsed[key] = result
    return parsed


def _validate_quad(corners, key="face"):
    if not isinstance(corners, list) or len(corners) != 4:
        raise ValueError(f"Layer {key} corners must contain four points.")
    points = []
    for point in corners:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Layer {key} corners must contain [x, y] points.")
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError(f"Layer {key} corner coordinates must be numeric.")
        x, y = float(x), float(y)
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or not (-1 <= x <= 1 and -1 <= y <= 1)
        ):
            raise ValueError(f"Layer {key} corner coordinates must be within [-1, 1].")
        points.append([x, y])
    cross = []
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    area = (
        abs(
            sum(
                points[i][0] * points[(i + 1) % 4][1]
                - points[(i + 1) % 4][0] * points[i][1]
                for i in range(4)
            )
        )
        * 0.5
    )
    if area < 1e-4 or not (
        all(value > 1e-6 for value in cross) or all(value < -1e-6 for value in cross)
    ):
        raise ValueError(
            f"Layer {key} corners must form a convex, non-zero-area quadrilateral."
        )
    return points


def _ordered_layer_keys(value, available_keys):
    _, _, requested, _ = _parse_layer_payload(value)
    available = list(available_keys)
    available_set = set(available)
    ordered = []
    for key in requested:
        if key in available_set and key not in ordered:
            ordered.append(key)
    ordered.extend(key for key in available if key not in ordered)
    return ordered


def _placement_offsets(
    background_width,
    background_height,
    placed_width,
    placed_height,
    placement,
    workspace_padding=0.0,
):
    if (
        placement.get("_version") == 2
        or "center_x" in placement
        or "center_y" in placement
    ):
        offset_x = round(
            float(placement.get("center_x", 0.5)) * background_width
            - placed_width / 2.0
        )
        offset_y = round(
            float(placement.get("center_y", 0.5)) * background_height
            - placed_height / 2.0
        )
        padding_x = background_width * 0.25 * workspace_padding
        padding_y = background_height * 0.25 * workspace_padding
        x_limits = (
            -padding_x,
            background_width + padding_x - placed_width,
            0.0,
            background_width - placed_width,
        )
        y_limits = (
            -padding_y,
            background_height + padding_y - placed_height,
            0.0,
            background_height - placed_height,
        )
        offset_x = round(min(max(offset_x, min(x_limits)), max(x_limits)))
        offset_y = round(min(max(offset_y, min(y_limits)), max(y_limits)))
        return offset_x, offset_y
    long_shift = (placement["long_axis_shift"] + 1.0) / 2.0
    short_shift = (placement["short_axis_shift"] + 1.0) / 2.0
    padding_x = background_width * 0.25 * workspace_padding
    padding_y = background_height * 0.25 * workspace_padding
    travel_x = background_width + 2.0 * padding_x - placed_width
    travel_y = background_height + 2.0 * padding_y - placed_height
    if background_width > background_height:
        offset_x = round(-padding_x + travel_x * long_shift)
        offset_y = round(-padding_y + travel_y * short_shift)
    elif background_height > background_width:
        offset_y = round(-padding_y + travel_y * long_shift)
        offset_x = round(-padding_x + travel_x * short_shift)
    else:
        offset_x = round(-padding_x + travel_x * long_shift)
        offset_y = round(-padding_y + travel_y * short_shift)
    return offset_x, offset_y


def _visible_placement_slices(
    background_width, background_height, placed_width, placed_height, offset_x, offset_y
):
    destination_top = max(0, offset_y)
    destination_bottom = min(background_height, offset_y + placed_height)
    destination_left = max(0, offset_x)
    destination_right = min(background_width, offset_x + placed_width)
    if destination_bottom <= destination_top or destination_right <= destination_left:
        return None
    source_top = destination_top - offset_y
    source_bottom = source_top + destination_bottom - destination_top
    source_left = destination_left - offset_x
    source_right = source_left + destination_right - destination_left
    return (
        destination_top,
        destination_bottom,
        destination_left,
        destination_right,
        source_top,
        source_bottom,
        source_left,
        source_right,
    )


def _save_editor_preview(image, prefix):
    saved = ui.ImageSaveHelper.save_images(
        image,
        filename_prefix=prefix,
        folder_type=io.FolderType.temp,
        cls=None,
        compress_level=1,
    )
    return dict(saved[0]) if saved else None


def _crop_bounds(mask, padding, multiple=8):
    points = torch.nonzero(mask > 0, as_tuple=False)
    if points.numel() == 0:
        raise ValueError("Mask is empty.")
    height, width = mask.shape[-2:]
    min_y = int(points[:, -2].min())
    max_y = int(points[:, -2].max()) + 1
    min_x = int(points[:, -1].min())
    max_x = int(points[:, -1].max()) + 1
    side = max(max_x - min_x, max_y - min_y) + 2 * padding
    side = min(max(height, width), ((side + multiple - 1) // multiple) * multiple)
    crop_width = min(side, width)
    crop_height = min(side, height)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    x = max(0, min(center_x - crop_width // 2, width - crop_width))
    y = max(0, min(center_y - crop_height // 2, height - crop_height))
    return x, y, crop_width, crop_height


def crop_staged_layers_by_indices(image, layer_masks, layer_indices):
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Staged layer cropping requires exactly one composed image.")
    if layer_masks.ndim == 2:
        layer_masks = layer_masks.unsqueeze(0)
    if layer_masks.ndim != 3:
        raise ValueError("Layer masks must have shape [layers, height, width].")

    tokens = [token for token in re.split(r"[,\s]+", str(layer_indices).strip()) if token]
    if not tokens:
        raise ValueError("Layer indices must contain at least one zero-based index.")
    try:
        indices = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError("Layer indices must be integers separated by commas or spaces.") from error

    layer_count = int(layer_masks.shape[0])
    invalid = [index for index in indices if index < 0 or index >= layer_count]
    if invalid:
        raise ValueError(
            f"Layer index {invalid[0]} is outside the available range 0-{layer_count - 1}."
        )

    if layer_masks.shape[-2:] != image.shape[1:3]:
        layer_masks = _resize_mask(
            layer_masks,
            int(image.shape[2]),
            int(image.shape[1]),
            "nearest-exact",
        )

    crops = []
    for index in indices:
        mask = layer_masks[index : index + 1]
        try:
            x, y, width, height = _crop_bounds(mask, padding=0, multiple=8)
        except ValueError as error:
            raise ValueError(f"Layer mask index {index} is empty.") from error
        crops.append(image[:, y : y + height, x : x + width])
    return crops


_BBOX_DEFAULT_COLORS = (
    "#ff6b6b",
    "#4ecdc4",
    "#ffe66d",
    "#a29bfe",
    "#ff9f43",
    "#54a0ff",
    "#5f27cd",
    "#1dd1a1",
)


def _clamp_unit(value, default=0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return max(0.0, min(1.0, numeric))


def _normalize_region_entry(entry: dict, width: int, height: int) -> dict | None:
    if not isinstance(entry, dict):
        return None
    w_px = max(1, int(width or 1024))
    h_px = max(1, int(height or 1024))

    if "bbox" in entry and isinstance(entry.get("bbox"), (list, tuple)) and len(entry["bbox"]) == 4:
        try:
            ymin, xmin, ymax, xmax = (float(v) / 1000.0 for v in entry["bbox"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Bounding box element 'bbox' must contain four numeric coordinates.") from exc
        x = _clamp_unit(min(xmin, xmax))
        y = _clamp_unit(min(ymin, ymax))
        w = _clamp_unit(abs(xmax - xmin))
        h = _clamp_unit(abs(ymax - ymin))
        meta = entry
    elif "w" in entry and "h" in entry and ("width" not in entry or entry.get("normalized", True)):
        raw_x = float(entry.get("x", 0.0) or 0.0)
        raw_y = float(entry.get("y", 0.0) or 0.0)
        raw_w = float(entry.get("w", 0.0) or 0.0)
        raw_h = float(entry.get("h", 0.0) or 0.0)
        if raw_w < 0.0:
            raw_x, raw_w = raw_x + raw_w, -raw_w
        if raw_h < 0.0:
            raw_y, raw_h = raw_y + raw_h, -raw_h
        x = _clamp_unit(raw_x)
        y = _clamp_unit(raw_y)
        w = max(0.0, min(1.0 - x, _clamp_unit(raw_w)))
        h = max(0.0, min(1.0 - y, _clamp_unit(raw_h)))
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else entry
    elif all(k in entry for k in ("x", "y", "width", "height")):
        try:
            px_x = float(entry.get("x", 0.0) or 0.0)
            px_y = float(entry.get("y", 0.0) or 0.0)
            px_w = float(entry.get("width", 0.0) or 0.0)
            px_h = float(entry.get("height", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Pixel bounding box coordinates must be numeric.") from exc
        if px_w < 0.0:
            px_x, px_w = px_x + px_w, -px_w
        if px_h < 0.0:
            px_y, px_h = px_y + px_h, -px_h
        x = _clamp_unit(px_x / w_px)
        y = _clamp_unit(px_y / h_px)
        w = max(0.0, min(1.0 - x, _clamp_unit(px_w / w_px)))
        h = max(0.0, min(1.0 - y, _clamp_unit(px_h / h_px)))
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else entry
    else:
        return None

    from comfy_extras.color_util import normalize_palette

    raw_palette = meta.get("palette")
    if raw_palette is None:
        raw_palette = meta.get("color_palette")
    if raw_palette is None and meta.get("color"):
        raw_palette = [meta.get("color")]
    if isinstance(raw_palette, str):
        raw_palette = [part.strip() for part in raw_palette.split(",") if part.strip()]
    palette = normalize_palette(raw_palette if isinstance(raw_palette, list) else [])[:5]

    raw_outline = meta.get("outline_color")
    outline_list = normalize_palette([raw_outline] if isinstance(raw_outline, str) and raw_outline.strip() else [])
    outline_color = outline_list[0] if outline_list else (palette[0] if palette else "")

    etype = "text" if str(meta.get("type", entry.get("type", "obj"))).lower() == "text" else "obj"
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "type": etype,
        "text": str(meta.get("text", "") or "") if etype == "text" else "",
        "desc": str(meta.get("desc", "") or ""),
        "palette": palette,
        "outline_color": outline_color,
        "included": bool(meta.get("included", entry.get("included", True))),
        "locked": bool(meta.get("locked", entry.get("locked", False))),
    }


def parse_bounding_box_editor_data(box_data, width: int, height: int) -> tuple[list[dict], list[dict] | None]:
    if not box_data:
        return [], None
    parsed = box_data
    if isinstance(box_data, str):
        text = box_data.strip()
        if not text:
            return [], None
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return [], None

    raw_boxes = []
    raw_last_incoming = None
    if isinstance(parsed, dict):
        raw_boxes = parsed.get("boxes") if isinstance(parsed.get("boxes"), list) else []
        if isinstance(parsed.get("last_incoming"), list):
            raw_last_incoming = parsed.get("last_incoming")
    elif isinstance(parsed, list):
        raw_boxes = parsed

    regions = []
    for item in raw_boxes:
        norm = _normalize_region_entry(item, width, height)
        if norm is not None:
            regions.append(norm)

    last_incoming = None
    if raw_last_incoming is not None:
        last_incoming = []
        for item in raw_last_incoming:
            norm = _normalize_region_entry(item, width, height)
            if norm is not None:
                last_incoming.append({
                    k: norm[k] for k in ("x", "y", "w", "h", "type", "text", "desc", "palette")
                })
    return regions, last_incoming


def _extract_incoming_summary_and_labels(data) -> tuple[str, list[str], dict]:
    if data is None:
        return "", [], {}
    parsed = data
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return "", [], {}
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return "", [], {}
    if not isinstance(parsed, dict):
        return "", [], {}
    summary = parsed.get("summary")
    caption = ""
    summary_extras: dict = {}
    if isinstance(summary, dict):
        caption = str(summary.get("caption", "") or "").strip()
        if isinstance(summary.get("meta"), list):
            meta = [str(m).strip() for m in summary["meta"] if str(m).strip()]
            if meta:
                summary_extras["meta"] = meta
        if isinstance(summary.get("canvas"), str) and summary["canvas"].strip() in (
            "single",
            "multi_panel",
            "collage",
        ):
            summary_extras["canvas"] = summary["canvas"].strip()
        if isinstance(summary.get("palette"), list):
            pal = [str(p).strip() for p in summary["palette"] if str(p).strip()]
            if pal:
                summary_extras["palette"] = pal
    elif isinstance(parsed.get("high_level_description"), str):
        caption = parsed.get("high_level_description", "").strip()
    items = parsed.get("items")
    labels = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                lbl = str(item.get("label", "object") or "object").lower()
                labels.append(lbl if lbl in ("character", "object", "text") else "object")
    return caption, labels, summary_extras


def _coerce_incoming_bboxes_input(data, width: int, height: int, input_coord_space: str = "gemini_0_to_1000"):
    if data is None:
        return None
    parsed = data
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return data
    if not isinstance(parsed, dict):
        return parsed
    comp = parsed.get("compositional_deconstruction")
    if isinstance(comp, dict) and isinstance(comp.get("elements"), list):
        return comp["elements"]
    if isinstance(parsed.get("elements"), list):
        return parsed["elements"]
    items = parsed.get("items")
    if not isinstance(items, list):
        return parsed
    mode = str(input_coord_space or "gemini_0_to_1000").lower()
    w = max(1, int(width or 1024))
    h = max(1, int(height or 1024))
    from comfy_extras.color_util import normalize_palette

    elements = []
    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            b0, b1, b2, b3 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        has_negative = any(v < 0 for v in (b0, b1, b2, b3))
        use_centralized = mode in ("moss_centered_pixels", "abs_centralized") or has_negative
        use_qwen_xyxy = mode in ("qwen_0_to_1000", "qwen_xyxy", "xyxy_1000") and not use_centralized
        if use_centralized:
            y0, x0, y1, x1 = b0, b1, b2, b3
            ymin = max(0.0, min(1000.0, (y0 + (h // 2)) * 1000.0 / h))
            xmin = max(0.0, min(1000.0, (x0 + (w // 2)) * 1000.0 / w))
            ymax = max(0.0, min(1000.0, (y1 + (h // 2)) * 1000.0 / h))
            xmax = max(0.0, min(1000.0, (x1 + (w // 2)) * 1000.0 / w))
            norm_bbox = [
                round(min(ymin, ymax)),
                round(min(xmin, xmax)),
                round(max(ymin, ymax)),
                round(max(xmin, xmax)),
            ]
        elif use_qwen_xyxy:
            x0, y0, x1, y1 = b0, b1, b2, b3
            ymin = max(0, min(1000, round(min(y0, y1))))
            xmin = max(0, min(1000, round(min(x0, x1))))
            ymax = max(0, min(1000, round(max(y0, y1))))
            xmax = max(0, min(1000, round(max(x0, x1))))
            norm_bbox = [ymin, xmin, ymax, xmax]
        else:
            y0, x0, y1, x1 = b0, b1, b2, b3
            ymin = max(0, min(1000, round(min(y0, y1))))
            xmin = max(0, min(1000, round(min(x0, x1))))
            ymax = max(0, min(1000, round(max(y0, y1))))
            xmax = max(0, min(1000, round(max(x0, x1))))
            norm_bbox = [ymin, xmin, ymax, xmax]
        label = str(item.get("label", item.get("type", "object")) or "object").lower()
        etype = "text" if label == "text" else "obj"
        element = {
            "type": etype,
            "bbox": norm_bbox,
            "desc": str(item.get("caption", item.get("desc", "")) or ""),
        }
        if etype == "text":
            element["text"] = str(item.get("text", "") or "")
        raw_palette = item.get("palette") or item.get("color_palette")
        if isinstance(raw_palette, list) and raw_palette:
            clean_palette = normalize_palette(raw_palette)[:5]
            if clean_palette:
                element["color_palette"] = clean_palette
        raw_tags = item.get("tags")
        if isinstance(raw_tags, list) and raw_tags:
            clean_tags = [str(t).strip() for t in raw_tags if str(t).strip()]
            if clean_tags:
                element["tags"] = clean_tags
        elements.append(element)
    return elements


def build_moss_image_layout_prompt(
    prompt: str = "",
    elements: list | None = None,
    width: int = 0,
    height: int = 0,
    image=None,
    input_coord_space: str = "gemini_0_to_1000",
    character_boxes: str = "",
    item_labels: list[str] | None = None,
) -> str:
    bg_w, bg_h = 1024, 1024
    if image is not None and torch.is_tensor(image) and image.ndim == 4 and image.shape[0] > 0:
        bg_h = int(image.shape[1])
        bg_w = int(image.shape[2])
    raw_w = int(width or 0)
    raw_h = int(height or 0)
    w_px = max(16, raw_w if raw_w > 0 else bg_w)
    h_px = max(16, raw_h if raw_h > 0 else bg_h)

    summary_caption, parsed_labels, summary_extras = _extract_incoming_summary_and_labels(prompt)
    resolved_elements = list(elements) if isinstance(elements, list) and elements else None
    if resolved_elements is None:
        coerced = _coerce_incoming_bboxes_input(prompt, w_px, h_px, input_coord_space=input_coord_space)
        if isinstance(coerced, list):
            resolved_elements = coerced
        else:
            resolved_elements = []

    if not summary_caption and isinstance(prompt, str):
        stripped = prompt.strip()
        if not stripped.startswith("{") and not stripped.startswith("["):
            summary_caption = stripped

    labels = item_labels if item_labels is not None else parsed_labels
    characters = {int(i) for i in re.findall(r"\d+", str(character_boxes or ""))}
    items = []
    for i, el in enumerate(resolved_elements):
        if not isinstance(el, dict) or not (isinstance(el.get("bbox"), (list, tuple)) and len(el["bbox"]) == 4):
            continue
        if el.get("type") == "text":
            label = "text"
        elif i + 1 in characters:
            label = "character"
        elif i < len(labels) and labels[i] in ("character", "object", "text"):
            label = labels[i]
        else:
            label = "object"
        item: dict[str, Any] = {"label": label, "caption": str(el.get("desc", "") or "")}
        if label == "text":
            item["text"] = str(el.get("text", "") or "")
        if el.get("tags"):
            item["tags"] = [str(t).strip() for t in el["tags"] if str(t).strip()]
        if el.get("color_palette"):
            item["palette"] = list(el["color_palette"])
        y0, x0, y1, x1 = (float(v) for v in el["bbox"])
        item["bbox"] = [
            math.floor(round(y0 * h_px / 1000, 9)) - h_px // 2,
            math.floor(round(x0 * w_px / 1000, 9)) - w_px // 2,
            math.ceil(round(y1 * h_px / 1000, 9)) - h_px // 2,
            math.ceil(round(x1 * w_px / 1000, 9)) - w_px // 2,
        ]
        items.append(item)
    if not items:
        return summary_caption or str(prompt or "").strip()
    summary_obj: dict = {}
    if "meta" in summary_extras:
        summary_obj["meta"] = summary_extras["meta"]
    if "canvas" in summary_extras:
        summary_obj["canvas"] = summary_extras["canvas"]
    summary_obj["caption"] = summary_caption
    if "palette" in summary_extras:
        summary_obj["palette"] = summary_extras["palette"]
    layout = {
        "summary": summary_obj,
        "items": items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def execute_create_bounding_boxes(
    width: int = 0,
    height: int = 0,
    box_data: str = '{"version":1,"boxes":[],"last_incoming":null}',
    background=None,
    bboxes=None,
    input_coord_space: str = "gemini_0_to_1000",
):
    from comfy_extras import nodes_bounding_boxes as core_bboxes

    bg_w, bg_h = 1024, 1024
    if background is not None and torch.is_tensor(background) and background.ndim == 4 and background.shape[0] > 0:
        bg_h = int(background.shape[1])
        bg_w = int(background.shape[2])
    raw_w = int(width or 0)
    raw_h = int(height or 0)
    w_px = max(64, raw_w if raw_w > 0 else bg_w)
    h_px = max(64, raw_h if raw_h > 0 else bg_h)

    incoming_summary, incoming_labels, _ = _extract_incoming_summary_and_labels(bboxes)
    incoming_boxes = core_bboxes.boxes_from_input(
        _coerce_incoming_bboxes_input(bboxes, w_px, h_px, input_coord_space=input_coord_space),
        w_px,
        h_px,
    )
    incoming_regions = core_bboxes.boxes_to_regions(incoming_boxes, w_px, h_px) if incoming_boxes else []
    canonical_incoming = [
        {
            "x": round(float(r.get("x", 0.0)), 6),
            "y": round(float(r.get("y", 0.0)), 6),
            "w": round(float(r.get("w", 0.0)), 6),
            "h": round(float(r.get("h", 0.0)), 6),
            "type": "text" if r.get("type") == "text" else "obj",
            "text": str(r.get("text", "") or ""),
            "desc": str(r.get("desc", "") or ""),
            "palette": list(r.get("palette") or []),
        }
        for r in incoming_regions
    ]

    editor_regions, last_incoming = parse_bounding_box_editor_data(box_data, w_px, h_px)
    canonical_last = [
        {
            "x": round(float(r.get("x", 0.0)), 6),
            "y": round(float(r.get("y", 0.0)), 6),
            "w": round(float(r.get("w", 0.0)), 6),
            "h": round(float(r.get("h", 0.0)), 6),
            "type": "text" if r.get("type") == "text" else "obj",
            "text": str(r.get("text", "") or ""),
            "desc": str(r.get("desc", "") or ""),
            "palette": list(r.get("palette") or []),
        }
        for r in (last_incoming or [])
    ]

    upstream_changed = bool(canonical_incoming) and canonical_incoming != canonical_last
    if upstream_changed or (canonical_incoming and not editor_regions):
        active_regions = [
            {**region, "included": True, "locked": False}
            for region in incoming_regions
        ]
    else:
        active_regions = [
            region for region in editor_regions if region.get("included", True)
        ]

    preview_regions = []
    for index, region in enumerate(active_regions):
        palette = list(region.get("palette") or [])
        if not palette:
            outline = str(region.get("outline_color", "") or "").strip()
            palette = [outline] if outline else [_BBOX_DEFAULT_COLORS[index % len(_BBOX_DEFAULT_COLORS)]]
        preview_regions.append({**region, "palette": palette})

    preview = core_bboxes.render_preview(
        preview_regions,
        w_px,
        h_px,
        core_bboxes._bg_from_image(background),
    )
    pixel_bboxes = core_bboxes.fractions_to_bbox_frame(active_regions, w_px, h_px)
    elements = core_bboxes.build_elements(active_regions)
    moss_prompt = build_moss_image_layout_prompt(
        prompt=incoming_summary,
        elements=elements,
        width=w_px,
        height=h_px,
        item_labels=incoming_labels if len(incoming_labels) == len(elements) else None,
    )

    bg_descriptor = None
    if background is not None and torch.is_tensor(background) and background.ndim == 4 and background.shape[0] > 0:
        bg_descriptor = _save_editor_preview(background[:1, :, :, :3], "uc_bboxes_bg")

    ui_payload = {
        "version": 1,
        "dims": [w_px, h_px],
        "background": bg_descriptor,
    }
    if canonical_incoming:
        ui_payload["input_regions"] = canonical_incoming
        ui_payload["upstream_changed"] = upstream_changed

    return io.NodeOutput(
        preview,
        pixel_bboxes,
        elements,
        moss_prompt,
        ui={"uc_bounding_boxes_editor": (ui_payload,)},
    )
