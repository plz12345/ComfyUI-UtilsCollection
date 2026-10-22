"""Eager Depth Anything 3 architectures adapted from ByteDance Depth Anything 3 (Apache 2.0)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.model_management import throw_exception_if_processing_interrupted


TORCHSCRIPT_STATE_DICT_ROOT_MAP = {}
TORCHSCRIPT_STATE_DICT_KEY_MAP = {}

THRESH_FOR_REF_SELECTION = 3
_INT_MAX_INTERPOLATE = 1610612736


def affine_inverse(matrix: torch.Tensor) -> torch.Tensor:
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3:]
    bottom = matrix[..., 3:, :]
    inv_rot = rotation.mT
    return torch.cat((torch.cat((inv_rot, -inv_rot @ translation), dim=-1), bottom), dim=-2)


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp_min(1e-12)
    mat = torch.stack(
        (
            1.0 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1.0 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1.0 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return mat.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, torch.sqrt(x.clamp_min(0.0)), torch.zeros_like(x))


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    return standardize_quaternion(out[..., [1, 2, 3, 0]])


def extri_intri_to_pose_encoding(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
) -> torch.Tensor:
    rotation = extrinsics[:, :, :3, :3]
    translation = extrinsics[:, :, :3, 3]
    quat = mat_to_quat(rotation)
    height, width = image_size_hw
    fov_h = 2.0 * torch.atan((height / 2.0) / intrinsics[..., 1, 1])
    fov_w = 2.0 * torch.atan((width / 2.0) / intrinsics[..., 0, 0])
    return torch.cat((translation, quat, fov_h[..., None], fov_w[..., None]), dim=-1).float()


def pose_encoding_to_extri_intri(
    pose_encoding: torch.Tensor,
    image_size_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    translation = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]
    rotation = quat_to_mat(quat)
    c2w = torch.cat((rotation, translation[..., None]), dim=-1)
    height, width = image_size_hw
    fy = (height / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), min=1e-6)
    fx = (width / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), min=1e-6)
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), dtype=pose_encoding.dtype, device=pose_encoding.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = width / 2.0
    intrinsics[..., 1, 2] = height / 2.0
    intrinsics[..., 2, 2] = 1.0
    return c2w, intrinsics


def select_reference_view(x: torch.Tensor, strategy: str = "saddle_balanced") -> torch.Tensor:
    batch_size, num_views = x.shape[0], x.shape[1]
    if num_views <= 1 or strategy == "first":
        return torch.zeros(batch_size, dtype=torch.long, device=x.device)
    if strategy == "middle":
        return torch.full((batch_size,), num_views // 2, dtype=torch.long, device=x.device)

    cls_feat = x[:, :, 0]
    cls_normed = cls_feat / cls_feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = torch.matmul(cls_normed, cls_normed.transpose(1, 2))
    sim_no_diag = sim - torch.eye(num_views, dtype=sim.dtype, device=sim.device).unsqueeze(0)

    if strategy == "saddle_balanced":
        sim_score = sim_no_diag.sum(dim=-1) / (num_views - 1)
        feat_norm = cls_feat.norm(dim=-1)
        feat_var = cls_normed.var(dim=-1)

        def _norm_metric(metric: torch.Tensor) -> torch.Tensor:
            min_val = metric.min(dim=1, keepdim=True).values
            max_val = metric.max(dim=1, keepdim=True).values
            return (metric - min_val) / (max_val - min_val + 1e-8)

        balance = (
            (_norm_metric(sim_score) - 0.5).abs()
            + (_norm_metric(feat_norm) - 0.5).abs()
            + (_norm_metric(feat_var) - 0.5).abs()
        )
        return balance.argmin(dim=1)

    if strategy == "saddle_sim_range":
        sim_range = sim_no_diag.max(dim=-1).values - sim_no_diag.min(dim=-1).values
        return sim_range.argmax(dim=1)

    raise ValueError(
        f"Unknown reference view selection strategy: {strategy!r}. "
        "Must be one of: 'first', 'middle', 'saddle_balanced', 'saddle_sim_range'."
    )


def reorder_by_reference(x: torch.Tensor, ref_indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_views = x.shape[0], x.shape[1]
    if num_views <= 1:
        return x
    positions = torch.arange(num_views, device=x.device).unsqueeze(0).expand(batch_size, -1)
    ref_expanded = ref_indices.unsqueeze(1)
    reorder_indices = torch.where((positions > 0) & (positions <= ref_expanded), positions - 1, positions)
    reorder_indices[:, 0] = ref_indices
    batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
    return x[batch_indices, reorder_indices]


def restore_original_order(x: torch.Tensor, ref_indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_views = x.shape[0], x.shape[1]
    if num_views <= 1:
        return x
    target_positions = torch.arange(num_views, device=x.device).unsqueeze(0).expand(batch_size, -1)
    ref_expanded = ref_indices.unsqueeze(1)
    restore_indices = torch.where(target_positions < ref_expanded, target_positions + 1, target_positions)
    restore_indices = torch.scatter(restore_indices, dim=1, index=ref_expanded, src=torch.zeros_like(ref_expanded))
    batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
    return x[batch_indices, restore_indices]


def _build_rope_positions(
    batch_size: int,
    num_views: int,
    patch_h: int,
    patch_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_coords = torch.arange(patch_h, device=device, dtype=torch.long)
    x_coords = torch.arange(patch_w, device=device, dtype=torch.long)
    grid = torch.cartesian_prod(y_coords, x_coords).view(1, 1, patch_h * patch_w, 2)
    pos = (grid + 1).expand(batch_size, num_views, -1, -1)
    zero_special = torch.zeros((batch_size, num_views, 1, 2), dtype=torch.long, device=device)
    local_pos = torch.cat((zero_special, pos), dim=2)
    ones_patches = torch.ones((batch_size, num_views, patch_h * patch_w, 2), dtype=torch.long, device=device)
    global_pos = torch.cat((zero_special, ones_patches), dim=2)
    return local_pos, global_pos


def _apply_2d_rope(tokens: torch.Tensor, positions: torch.Tensor, base_frequency: float = 100.0) -> torch.Tensor:
    feature_dim = tokens.shape[-1] // 2
    max_position = int(positions.max()) + 1
    exponents = torch.arange(0, feature_dim, 2, device=tokens.device, dtype=torch.float32) / feature_dim
    inv_freq = 1.0 / (base_frequency ** exponents)
    seq_positions = torch.arange(max_position, device=tokens.device, dtype=inv_freq.dtype)
    angles = torch.outer(seq_positions, inv_freq).to(tokens.dtype)
    angles = torch.cat((angles, angles), dim=-1)
    cos_comp = angles.cos()
    sin_comp = angles.sin()

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def _rope_1d(feat: torch.Tensor, pos_1d: torch.Tensor) -> torch.Tensor:
        cos = F.embedding(pos_1d, cos_comp)[:, None, :, :]
        sin = F.embedding(pos_1d, sin_comp)[:, None, :, :]
        return (feat * cos) + (_rotate_half(feat) * sin)

    vertical, horizontal = tokens.chunk(2, dim=-1)
    return torch.cat((_rope_1d(vertical, positions[..., 0]), _rope_1d(horizontal, positions[..., 1])), dim=-1)


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = operations.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x.flatten(2).transpose(1, 2))


class _LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * comfy.ops.cast_to_input(self.gamma, x)


class _Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = operations.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = operations.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = operations.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class _DinoAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        use_rope: bool = False,
        rope_freq: float = 100.0,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.rope_freq = rope_freq
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = operations.LayerNorm(self.head_dim, eps=1e-5) if qk_norm else nn.Identity()
        self.k_norm = operations.LayerNorm(self.head_dim, eps=1e-5) if qk_norm else nn.Identity()
        self.proj = operations.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope and pos is not None:
            q = _apply_2d_rope(q, pos, self.rope_freq)
            k = _apply_2d_rope(k, pos, self.rope_freq)
        attention = optimized_attention_for_device(q.device, mask=attn_mask is not None)
        out = attention(q, k, v, self.num_heads, mask=attn_mask, skip_reshape=True)
        return self.proj(out)


class _DinoBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        ffn_type: str = "mlp",
        qk_norm: bool = False,
        use_rope: bool = False,
        rope_freq: float = 100.0,
        ln_eps: float = 1e-6,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, eps=ln_eps)
        self.attn = _DinoAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_rope=use_rope,
            rope_freq=rope_freq,
            operations=operations,
        )
        self.ls1 = _LayerScale(dim)
        self.norm2 = operations.LayerNorm(dim, eps=ln_eps)
        hidden_dim = int(dim * mlp_ratio)
        ffn_cls = _SwiGLUFFN if ffn_type in ("swiglu", "swiglufused") else _Mlp
        self.mlp = ffn_cls(dim, hidden_dim, dim, bias=ffn_bias, operations=operations)
        self.ls2 = _LayerScale(dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        interpolate_offset: float = 0.1,
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        rope_freq: float = 100.0,
        cat_token: bool = True,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        self.patch_start_idx = 1
        self.embed_dim = embed_dim
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.rope_freq = rope_freq
        self.cat_token = cat_token
        self.patch_size = patch_size
        self.interpolate_offset = interpolate_offset

        self.patch_embed = _PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            operations=operations,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.empty(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                _DinoBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    ffn_type=ffn_layer,
                    qk_norm=(i >= qknorm_start) if qknorm_start != -1 else False,
                    use_rope=(i >= rope_start) if rope_start != -1 else False,
                    rope_freq=rope_freq,
                    ln_eps=1e-6,
                    operations=operations,
                )
                for i in range(depth)
            ]
        )
        self.norm = operations.LayerNorm(embed_dim, eps=1e-5)

    def interpolate_pos_encoding(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        num_base_patches = self.pos_embed.shape[1] - 1
        if npatch == num_base_patches and height == width:
            return comfy.ops.cast_to_input(self.pos_embed, x)
        pos_embed = comfy.model_management.cast_to(self.pos_embed, torch.float32, x.device)
        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        h0 = height // self.patch_size
        w0 = width // self.patch_size
        base_grid = int(math.isqrt(num_base_patches))
        if self.interpolate_offset:
            sh = float(h0 + self.interpolate_offset) / base_grid
            sw = float(w0 + self.interpolate_offset) / base_grid
            patch_pos_embed = F.interpolate(
                patch_pos_embed.reshape(1, base_grid, base_grid, dim).permute(0, 3, 1, 2),
                scale_factor=(sh, sw),
                mode="bicubic",
                antialias=False,
            )
        else:
            patch_pos_embed = F.interpolate(
                patch_pos_embed.reshape(1, base_grid, base_grid, dim).permute(0, 3, 1, 2),
                size=(h0, w0),
                mode="bicubic",
                antialias=False,
            )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, channels, height, width = x.shape
        flat = x.reshape(batch_size * num_views, channels, height, width)
        tokens = self.patch_embed(flat)
        cls_token = comfy.ops.cast_to_input(self.cls_token, tokens).expand(batch_size * num_views, 1, self.embed_dim)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, height, width)
        return tokens.reshape(batch_size, num_views, -1, self.embed_dim)

    def forward_features(
        self,
        x: torch.Tensor,
        out_layers: Sequence[int],
        cam_token: torch.Tensor | None = None,
        ref_view_strategy: str = "saddle_balanced",
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_views, _, height, width = x.shape
        x = self.prepare_tokens(x)
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        local_pos, global_pos = (
            _build_rope_positions(batch_size, num_views, patch_h, patch_w, x.device)
            if self.rope_start != -1
            else (None, None)
        )
        out_layer_set = set(out_layers)
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        local_x = x
        reordered_idx: torch.Tensor | None = None

        for i, block in enumerate(self.blocks):
            throw_exception_if_processing_interrupted()
            use_rope = self.rope_start != -1 and i >= self.rope_start
            l_pos = local_pos if use_rope else None
            g_pos = global_pos if use_rope else None

            if (
                self.alt_start != -1
                and i == self.alt_start - 1
                and num_views >= THRESH_FOR_REF_SELECTION
                and cam_token is None
            ):
                reordered_idx = select_reference_view(x, strategy=ref_view_strategy)
                x = reorder_by_reference(x, reordered_idx)
                local_x = reorder_by_reference(local_x, reordered_idx)

            if self.alt_start != -1 and i == self.alt_start:
                if cam_token is not None:
                    injected_cam = cam_token.to(device=x.device, dtype=x.dtype)
                else:
                    learned_cam = comfy.ops.cast_to_input(self.camera_token, x)
                    ref_tok = learned_cam[:, :1].expand(batch_size, 1, self.embed_dim)
                    src_tok = learned_cam[:, 1:].expand(batch_size, num_views - 1, self.embed_dim)
                    injected_cam = torch.cat((ref_tok, src_tok), dim=1)
                x = torch.cat((injected_cam.unsqueeze(2), x[:, :, 1:]), dim=2)

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                num_tokens = x.shape[2]
                flat_x = x.reshape(batch_size, num_views * num_tokens, self.embed_dim)
                flat_pos = g_pos.reshape(batch_size, num_views * num_tokens, 2) if g_pos is not None else None
                x = block(flat_x, pos=flat_pos).reshape(batch_size, num_views, num_tokens, self.embed_dim)
            else:
                num_tokens = x.shape[2]
                flat_x = x.reshape(batch_size * num_views, num_tokens, self.embed_dim)
                flat_pos = l_pos.reshape(batch_size * num_views, num_tokens, 2) if l_pos is not None else None
                x = block(flat_x, pos=flat_pos).reshape(batch_size, num_views, num_tokens, self.embed_dim)
                local_x = x

            if i in out_layer_set:
                out_x = torch.cat((local_x, x), dim=-1) if self.cat_token else x
                if reordered_idx is not None:
                    out_x = restore_original_order(out_x, reordered_idx)
                cam_tok_out = out_x[:, :, 0]
                if self.cat_token:
                    normed = torch.cat((out_x[..., : self.embed_dim], self.norm(out_x[..., self.embed_dim :])), dim=-1)
                else:
                    normed = self.norm(out_x)
                outputs.append((normed[..., 1:, :], cam_tok_out))

        return outputs


class _DinoV2(nn.Module):
    _ARCH_PARAMS = {
        "vits": (384, 12, 6, "mlp"),
        "vitb": (768, 12, 12, "mlp"),
        "vitl": (1024, 24, 16, "mlp"),
        "vitg": (1536, 40, 24, "swiglufused"),
    }

    def __init__(
        self,
        name: str,
        out_layers: Sequence[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        embed_dim, depth, num_heads, ffn_layer = self._ARCH_PARAMS[name]
        self.name = name
        self.out_layers = tuple(out_layers)
        self.alt_start = alt_start
        self.cat_token = cat_token
        self.pretrained = _DinoVisionTransformer(
            img_size=518,
            patch_size=14,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            operations=operations,
        )

    def forward(
        self,
        x: torch.Tensor,
        cam_token: torch.Tensor | None = None,
        ref_view_strategy: str = "saddle_balanced",
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.pretrained.forward_features(
            x,
            self.out_layers,
            cam_token=cam_token,
            ref_view_strategy=ref_view_strategy,
        )


class _Permute(nn.Module):
    def __init__(self, dims: tuple[int, ...]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


def _custom_interpolate(
    x: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    total = size[0] * size[1] * x.shape[0] * x.shape[1]
    if total > _INT_MAX_INTERPOLATE:
        chunks = torch.chunk(x, chunks=(total // _INT_MAX_INTERPOLATE) + 1, dim=0)
        return torch.cat(
            [F.interpolate(c, size=size, mode=mode, align_corners=align_corners) for c in chunks],
            dim=0,
        ).contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


def _create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    diag_factor = math.sqrt(aspect_ratio ** 2 + 1.0)
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def _position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100.0) -> torch.Tensor:
    height, width, _ = pos_grid.shape
    pos_flat = pos_grid.reshape(-1, 2)
    half_dim = embed_dim // 2
    omega = torch.arange(half_dim // 2, dtype=torch.float32, device=pos_grid.device) / (half_dim / 2.0)
    omega = 1.0 / (omega_0 ** omega)

    def _sincos_1d(pos: torch.Tensor) -> torch.Tensor:
        angles = torch.outer(pos.float().reshape(-1), omega)
        return torch.cat((angles.sin(), angles.cos()), dim=1)

    emb = torch.cat((_sincos_1d(pos_flat[:, 0]), _sincos_1d(pos_flat[:, 1])), dim=-1)
    return emb.reshape(height, width, embed_dim).to(pos_grid.dtype)


def _add_uv_pos_embed(x: torch.Tensor, width: int, height: int, ratio: float = 0.1) -> torch.Tensor:
    pw, ph = x.shape[-1], x.shape[-2]
    uv_grid = _create_uv_grid(pw, ph, aspect_ratio=width / height, dtype=x.dtype, device=x.device)
    pos_emb = _position_grid_to_embed(uv_grid, x.shape[1]) * ratio
    return x + pos_emb.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)


class _ResidualConvUnit(nn.Module):
    def __init__(self, features: int, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.conv1 = operations.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = operations.Conv2d(features, features, 3, 1, 1, bias=True)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return out + x


class _FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.has_residual = has_residual
        self.resConfUnit1 = _ResidualConvUnit(features, operations=operations) if has_residual else None
        self.resConfUnit2 = _ResidualConvUnit(features, operations=operations)
        self.out_conv = operations.Conv2d(features, features, 1, 1, 0, bias=True)

    def forward(self, *xs: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(xs[1])
        y = self.resConfUnit2(y)
        if size is None:
            y = _custom_interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=True)
        else:
            y = _custom_interpolate(y, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(y)


def _make_scratch(in_channels: Sequence[int], out_features: int, operations=comfy.ops.disable_weight_init) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = operations.Conv2d(in_channels[0], out_features, 3, 1, 1, bias=False)
    scratch.layer2_rn = operations.Conv2d(in_channels[1], out_features, 3, 1, 1, bias=False)
    scratch.layer3_rn = operations.Conv2d(in_channels[2], out_features, 3, 1, 1, bias=False)
    scratch.layer4_rn = operations.Conv2d(in_channels[3], out_features, 3, 1, 1, bias=False)
    return scratch


class _DPT(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 1,
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = False,
        down_ratio: int = 1,
        use_sky_head: bool = True,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.has_conf = output_dim > 1
        self.use_sky_head = use_sky_head
        self.norm = nn.Identity()
        self.projects = nn.ModuleList(
            [operations.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                operations.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                operations.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                operations.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = _make_scratch(out_channels, features, operations=operations)
        self.scratch.refinenet1 = _FeatureFusionBlock(features, has_residual=True, operations=operations)
        self.scratch.refinenet2 = _FeatureFusionBlock(features, has_residual=True, operations=operations)
        self.scratch.refinenet3 = _FeatureFusionBlock(features, has_residual=True, operations=operations)
        self.scratch.refinenet4 = _FeatureFusionBlock(features, has_residual=False, operations=operations)
        self.scratch.output_conv1 = operations.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            operations.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            operations.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
        )
        if self.use_sky_head:
            self.scratch.sky_output_conv2 = nn.Sequential(
                operations.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                operations.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        feats: list[tuple[torch.Tensor, torch.Tensor]],
        height: int,
        width: int,
        chunk_size: int = 8,
        compute_ray: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_views, num_tokens, channels = feats[0][0].shape
        flat_feats = [feat[0].reshape(batch_size * num_views, num_tokens, channels) for feat in feats]
        total = batch_size * num_views
        if chunk_size is None or chunk_size >= total:
            out_dict = self._forward_impl(flat_feats, height, width)
        else:
            chunks = [
                self._forward_impl([f[s0 : min(s0 + chunk_size, total)] for f in flat_feats], height, width)
                for s0 in range(0, total, chunk_size)
            ]
            out_dict = {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0]}
        return {k: v.reshape(batch_size, num_views, *v.shape[1:]) for k, v in out_dict.items()}

    def _forward_impl(self, feats: list[torch.Tensor], height: int, width: int) -> dict[str, torch.Tensor]:
        batch_total, _, channels = feats[0].shape
        ph, pw = height // self.patch_size, width // self.patch_size
        resized = []
        for stage_idx in range(4):
            x = self.norm(feats[stage_idx])
            x = x.permute(0, 2, 1).contiguous().reshape(batch_total, channels, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = _add_uv_pos_embed(x, width, height)
            x = self.resize_layers[stage_idx](x)
            resized.append(x)

        l1_rn = self.scratch.layer1_rn(resized[0])
        l2_rn = self.scratch.layer2_rn(resized[1])
        l3_rn = self.scratch.layer3_rn(resized[2])
        l4_rn = self.scratch.layer4_rn(resized[3])

        fused = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        fused = self.scratch.refinenet3(fused, l3_rn, size=l2_rn.shape[2:])
        fused = self.scratch.refinenet2(fused, l2_rn, size=l1_rn.shape[2:])
        fused = self.scratch.refinenet1(fused, l1_rn)

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused = self.scratch.output_conv1(fused)
        fused = _custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused = _add_uv_pos_embed(fused, width, height)

        main_logits = self.scratch.output_conv2(fused)
        outs: dict[str, torch.Tensor] = {}
        if self.has_conf:
            fmap = main_logits.permute(0, 2, 3, 1)
            outs["depth"] = torch.exp(fmap[..., :-1]).squeeze(-1)
            outs["depth_conf"] = torch.exp(fmap[..., -1]) + 1.0
        else:
            outs["depth"] = torch.exp(main_logits).squeeze(1)

        if self.use_sky_head:
            outs["sky"] = torch.relu(self.scratch.sky_output_conv2(fused)).squeeze(1)
        return outs


class _DualDPT(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 2,
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        aux_pyramid_levels: int = 4,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.aux_levels = aux_pyramid_levels

        self.norm = operations.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [operations.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                operations.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                operations.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                operations.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = _make_scratch(out_channels, features, operations=operations)
        self.scratch.refinenet1 = _FeatureFusionBlock(features, has_residual=True, operations=operations)
        self.scratch.refinenet2 = _FeatureFusionBlock(features, has_residual=True, operations=operations)
        self.scratch.refinenet3 = _FeatureFusionBlock(features, has_residual=True, operations=operations)
        self.scratch.refinenet4 = _FeatureFusionBlock(features, has_residual=False, operations=operations)
        self.scratch.output_conv1 = operations.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            operations.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            operations.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
        )

        self.scratch.refinenet1_aux = _FeatureFusionBlock(features, has_residual=True, operations=operations)
        self.scratch.refinenet2_aux = _FeatureFusionBlock(features, has_residual=True, operations=operations)
        self.scratch.refinenet3_aux = _FeatureFusionBlock(features, has_residual=True, operations=operations)
        self.scratch.refinenet4_aux = _FeatureFusionBlock(features, has_residual=False, operations=operations)
        self.scratch.output_conv1_aux = nn.ModuleList(
            [
                nn.Sequential(
                    operations.Conv2d(features, features // 2, 3, 1, 1),
                    operations.Conv2d(features // 2, features, 3, 1, 1),
                    operations.Conv2d(features, features // 2, 3, 1, 1),
                    operations.Conv2d(features // 2, features, 3, 1, 1),
                    operations.Conv2d(features, features // 2, 3, 1, 1),
                )
                for _ in range(self.aux_levels)
            ]
        )
        self.scratch.output_conv2_aux = nn.ModuleList(
            [
                nn.Sequential(
                    operations.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                    _Permute((0, 2, 3, 1)),
                    operations.LayerNorm(32),
                    _Permute((0, 3, 1, 2)),
                    nn.ReLU(inplace=True),
                    operations.Conv2d(32, 7, kernel_size=1, stride=1, padding=0),
                )
                for _ in range(self.aux_levels)
            ]
        )

    def forward(
        self,
        feats: list[tuple[torch.Tensor, torch.Tensor]],
        height: int,
        width: int,
        chunk_size: int = 8,
        compute_ray: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_views, num_tokens, channels = feats[0][0].shape
        flat_feats = [feat[0].reshape(batch_size * num_views, num_tokens, channels) for feat in feats]
        total = batch_size * num_views
        if chunk_size is None or chunk_size >= total:
            out_dict = self._forward_impl(flat_feats, height, width, compute_ray=compute_ray)
        else:
            chunks = [
                self._forward_impl(
                    [f[s0 : min(s0 + chunk_size, total)] for f in flat_feats],
                    height,
                    width,
                    compute_ray=compute_ray,
                )
                for s0 in range(0, total, chunk_size)
            ]
            out_dict = {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0]}
        return {k: v.reshape(batch_size, num_views, *v.shape[1:]) for k, v in out_dict.items()}

    def _forward_impl(
        self,
        feats: list[torch.Tensor],
        height: int,
        width: int,
        compute_ray: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_total, _, channels = feats[0].shape
        ph, pw = height // self.patch_size, width // self.patch_size
        resized = []
        for stage_idx in range(4):
            x = self.norm(feats[stage_idx])
            x = x.permute(0, 2, 1).contiguous().reshape(batch_total, channels, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = _add_uv_pos_embed(x, width, height)
            x = self.resize_layers[stage_idx](x)
            resized.append(x)

        l1_rn = self.scratch.layer1_rn(resized[0])
        l2_rn = self.scratch.layer2_rn(resized[1])
        l3_rn = self.scratch.layer3_rn(resized[2])
        l4_rn = self.scratch.layer4_rn(resized[3])

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        fused_main = self.scratch.output_conv1(out)

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused_main = _custom_interpolate(fused_main, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused_main = _add_uv_pos_embed(fused_main, width, height)

        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        result: dict[str, torch.Tensor] = {
            "depth": torch.exp(fmap[..., :-1]).squeeze(-1),
            "depth_conf": torch.exp(fmap[..., -1]) + 1.0,
        }

        if compute_ray:
            aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
            aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])
            aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])
            aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
            last_aux = self.scratch.output_conv1_aux[-1](aux_out)
            if self.pos_embed:
                last_aux = _add_uv_pos_embed(last_aux, width, height)
            last_aux_logits = self.scratch.output_conv2_aux[-1](last_aux).permute(0, 2, 3, 1)
            result["ray"] = last_aux_logits[..., :-1]
            result["ray_conf"] = torch.exp(last_aux_logits[..., -1]) + 1.0

        return result


class _CameraEnc(nn.Module):
    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            *[
                _DinoBlock(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=False,
                    use_rope=False,
                    ln_eps=1e-5,
                    operations=operations,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = operations.LayerNorm(dim_out)
        self.trunk_norm = operations.LayerNorm(dim_out)
        self.pose_branch = _Mlp(dim_in, dim_out // 2, dim_out, operations=operations)

    def forward(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size_hw: tuple[int, int],
    ) -> torch.Tensor:
        c2ws = affine_inverse(extrinsics)
        pose_encoding = extri_intri_to_pose_encoding(c2ws, intrinsics, image_size_hw)
        pose_tokens = self.token_norm(self.pose_branch(pose_encoding))
        return self.trunk_norm(self.trunk(pose_tokens))


class _CameraDec(nn.Module):
    def __init__(self, dim_in: int = 1536, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.backbone = nn.Sequential(
            operations.Linear(dim_in, dim_in),
            nn.ReLU(),
            operations.Linear(dim_in, dim_in),
            nn.ReLU(),
        )
        self.fc_t = operations.Linear(dim_in, 3)
        self.fc_qvec = operations.Linear(dim_in, 4)
        self.fc_fov = nn.Sequential(operations.Linear(dim_in, 2), nn.ReLU())

    def forward(self, feat: torch.Tensor, camera_encoding: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_views = feat.shape[:2]
        hidden = self.backbone(feat.reshape(batch_size * num_views, -1)).float()
        out_t = self.fc_t(hidden).reshape(batch_size, num_views, 3)
        if camera_encoding is None:
            out_qvec = self.fc_qvec(hidden).reshape(batch_size, num_views, 4)
            out_fov = self.fc_fov(hidden).reshape(batch_size, num_views, 2)
        else:
            out_qvec = camera_encoding[..., 3:7]
            out_fov = camera_encoding[..., -2:]
        return torch.cat((out_t, out_qvec, out_fov), dim=-1)


class DepthAnything3Net(nn.Module):
    PATCH_SIZE = 14
    BACKBONE_NAME = "vitl"
    OUT_LAYERS: tuple[int, int, int, int] = (11, 15, 19, 23)
    ALT_START = 8
    QKNORM_START = 8
    ROPE_START = 8
    CAT_TOKEN = True
    HEAD_TYPE = "dualdpt"
    HEAD_DIM_IN = 2048
    HEAD_OUTPUT_DIM = 2
    HEAD_FEATURES = 256
    HEAD_OUT_CHANNELS: tuple[int, int, int, int] = (256, 512, 1024, 1024)
    CAM_ENC_DIM_OUT: int | None = 1024
    CAM_DEC_DIM_IN: int | None = 2048

    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.backbone = _DinoV2(
            name=self.BACKBONE_NAME,
            out_layers=self.OUT_LAYERS,
            alt_start=self.ALT_START,
            qknorm_start=self.QKNORM_START,
            rope_start=self.ROPE_START,
            cat_token=self.CAT_TOKEN,
            operations=operations,
        )
        if self.HEAD_TYPE == "dualdpt":
            self.head = _DualDPT(
                dim_in=self.HEAD_DIM_IN,
                output_dim=self.HEAD_OUTPUT_DIM,
                features=self.HEAD_FEATURES,
                out_channels=self.HEAD_OUT_CHANNELS,
                operations=operations,
            )
        else:
            self.head = _DPT(
                dim_in=self.HEAD_DIM_IN,
                output_dim=self.HEAD_OUTPUT_DIM,
                features=self.HEAD_FEATURES,
                out_channels=self.HEAD_OUT_CHANNELS,
                use_sky_head=True,
                operations=operations,
            )

        if self.CAM_DEC_DIM_IN is not None and self.CAM_ENC_DIM_OUT is not None:
            self.cam_dec = _CameraDec(dim_in=self.CAM_DEC_DIM_IN, operations=operations)
            self.cam_enc = _CameraEnc(dim_out=self.CAM_ENC_DIM_OUT, operations=operations)
        else:
            self.cam_dec = None
            self.cam_enc = None

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        ref_view_strategy: str = "saddle_balanced",
        compute_ray: bool = False,
    ) -> dict[str, torch.Tensor]:
        height, width = x.shape[-2], x.shape[-1]
        if extrinsics is not None and intrinsics is not None and self.cam_enc is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics.float(), intrinsics.float(), (height, width))
        else:
            cam_token = None

        use_autocast = x.device.type == "cuda"
        autocast_dtype = torch.bfloat16 if (use_autocast and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=use_autocast):
            feats = self.backbone(x, cam_token=cam_token, ref_view_strategy=ref_view_strategy)

        with torch.autocast(device_type=x.device.type, enabled=False):
            float_feats = [(patch_tok.float(), cam_tok.float()) for patch_tok, cam_tok in feats]
            output = self.head(float_feats, height, width, compute_ray=compute_ray)
            if self.cam_dec is not None:
                pose_enc = self.cam_dec(float_feats[-1][1])
                c2w, pred_intrinsics = pose_encoding_to_extri_intri(pose_enc, (height, width))
                output["extrinsics"] = affine_inverse(c2w)
                output["intrinsics"] = pred_intrinsics

        return output


class DA3Small(DepthAnything3Net):
    BACKBONE_NAME = "vits"
    OUT_LAYERS = (5, 7, 9, 11)
    ALT_START = 4
    QKNORM_START = 4
    ROPE_START = 4
    CAT_TOKEN = True
    HEAD_TYPE = "dualdpt"
    HEAD_DIM_IN = 768
    HEAD_OUTPUT_DIM = 2
    HEAD_FEATURES = 64
    HEAD_OUT_CHANNELS = (48, 96, 192, 384)
    CAM_ENC_DIM_OUT = 384
    CAM_DEC_DIM_IN = 768


class DA3Base(DepthAnything3Net):
    BACKBONE_NAME = "vitb"
    OUT_LAYERS = (5, 7, 9, 11)
    ALT_START = 4
    QKNORM_START = 4
    ROPE_START = 4
    CAT_TOKEN = True
    HEAD_TYPE = "dualdpt"
    HEAD_DIM_IN = 1536
    HEAD_OUTPUT_DIM = 2
    HEAD_FEATURES = 128
    HEAD_OUT_CHANNELS = (96, 192, 384, 768)
    CAM_ENC_DIM_OUT = 768
    CAM_DEC_DIM_IN = 1536


class DA3Large(DepthAnything3Net):
    BACKBONE_NAME = "vitl"
    OUT_LAYERS = (11, 15, 19, 23)
    ALT_START = 8
    QKNORM_START = 8
    ROPE_START = 8
    CAT_TOKEN = True
    HEAD_TYPE = "dualdpt"
    HEAD_DIM_IN = 2048
    HEAD_OUTPUT_DIM = 2
    HEAD_FEATURES = 256
    HEAD_OUT_CHANNELS = (256, 512, 1024, 1024)
    CAM_ENC_DIM_OUT = 1024
    CAM_DEC_DIM_IN = 2048


class DA3MonoLarge(DepthAnything3Net):
    BACKBONE_NAME = "vitl"
    OUT_LAYERS = (4, 11, 17, 23)
    ALT_START = -1
    QKNORM_START = -1
    ROPE_START = -1
    CAT_TOKEN = False
    HEAD_TYPE = "dpt"
    HEAD_DIM_IN = 1024
    HEAD_OUTPUT_DIM = 1
    HEAD_FEATURES = 256
    HEAD_OUT_CHANNELS = (256, 512, 1024, 1024)
    CAM_ENC_DIM_OUT = None
    CAM_DEC_DIM_IN = None


class DA3MetricLarge(DA3MonoLarge):
    pass
