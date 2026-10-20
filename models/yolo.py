"""Eager YOLOv8 and YOLO11 detection and segmentation architectures."""

import math

import torch
from torch import nn

import comfy.ops


TORCHSCRIPT_STATE_DICT_ROOT_MAP = {}
TORCHSCRIPT_STATE_DICT_KEY_MAP = {}


class _Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.conv = operations.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2 if padding is None else padding, groups=groups, bias=False)
        self.bn = operations.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, kernels=(3, 3), expansion=0.5, operations=comfy.ops.disable_weight_init):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = _Conv(in_channels, hidden, kernels[0], 1, operations=operations)
        self.cv2 = _Conv(hidden, out_channels, kernels[1], 1, groups=groups, operations=operations)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class _C2f(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut=False, groups=1, expansion=0.5, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = _Conv(in_channels, 2 * self.c, 1, 1, operations=operations)
        self.cv2 = _Conv((2 + depth) * self.c, out_channels, 1, 1, operations=operations)
        self.m = nn.ModuleList([_Bottleneck(self.c, self.c, shortcut, groups, kernels=(3, 3), expansion=1.0, operations=operations) for _ in range(depth)])

    def forward(self, x):
        chunks = list(self.cv1(x).chunk(2, 1))
        chunks.extend(block(chunks[-1]) for block in self.m)
        return self.cv2(torch.cat(chunks, 1))


class _C3k(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut=True, groups=1, expansion=0.5, kernel_size=3, operations=comfy.ops.disable_weight_init):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = _Conv(in_channels, hidden, 1, 1, operations=operations)
        self.cv2 = _Conv(in_channels, hidden, 1, 1, operations=operations)
        self.cv3 = _Conv(2 * hidden, out_channels, 1, 1, operations=operations)
        self.m = nn.Sequential(*[_Bottleneck(hidden, hidden, shortcut, groups, kernels=(kernel_size, kernel_size), expansion=1.0, operations=operations) for _ in range(depth)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class _C3k2(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, c3k=False, expansion=0.5, groups=1, shortcut=True, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = _Conv(in_channels, 2 * self.c, 1, 1, operations=operations)
        self.cv2 = _Conv((2 + depth) * self.c, out_channels, 1, 1, operations=operations)
        block = (lambda: _C3k(self.c, self.c, 2, shortcut, groups, operations=operations)) if c3k else (lambda: _Bottleneck(self.c, self.c, shortcut, groups, operations=operations))
        self.m = nn.ModuleList([block() for _ in range(depth)])

    def forward(self, x):
        chunks = list(self.cv1(x).chunk(2, 1))
        chunks.extend(layer(chunks[-1]) for layer in self.m)
        return self.cv2(torch.cat(chunks, 1))


class _SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, operations=comfy.ops.disable_weight_init):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = _Conv(in_channels, hidden, 1, 1, operations=operations)
        self.cv2 = _Conv(hidden * 4, out_channels, 1, 1, operations=operations)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        pooled = [self.cv1(x)]
        pooled.extend(self.m(pooled[-1]) for _ in range(3))
        return self.cv2(torch.cat(pooled, 1))


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        self.qkv = _Conv(dim, dim + self.key_dim * num_heads * 2, 1, act=False, operations=operations)
        self.proj = _Conv(dim, dim, 1, act=False, operations=operations)
        self.pe = _Conv(dim, dim, 3, 1, groups=dim, act=False, operations=operations)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(x).view(batch, self.num_heads, self.key_dim * 2 + self.head_dim, height * width).split((self.key_dim, self.key_dim, self.head_dim), dim=2)
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(batch, channels, height, width) + self.pe(v.reshape(batch, channels, height, width))
        return self.proj(x)


class _PSABlock(nn.Module):
    def __init__(self, channels, attn_ratio=0.5, num_heads=4, shortcut=True, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.attn = _Attention(channels, num_heads=num_heads, attn_ratio=attn_ratio, operations=operations)
        self.ffn = nn.Sequential(_Conv(channels, channels * 2, 1, operations=operations), _Conv(channels * 2, channels, 1, act=False, operations=operations))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class _C2PSA(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, expansion=0.5, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.c = int(in_channels * expansion)
        self.cv1 = _Conv(in_channels, 2 * self.c, 1, 1, operations=operations)
        self.cv2 = _Conv(2 * self.c, out_channels, 1, 1, operations=operations)
        self.m = nn.Sequential(*[_PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1), operations=operations) for _ in range(depth)])

    def forward(self, x):
        first, second = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((first, self.m(second)), 1))


class _DFL(nn.Module):
    def __init__(self, channels=16, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.conv = operations.Conv2d(channels, 1, 1, bias=False)
        self.c1 = channels

    def forward(self, x):
        batch, _, anchors = x.shape
        return self.conv(x.view(batch, 4, self.c1, anchors).transpose(2, 1).softmax(1)).view(batch, 4, anchors)


class _Proto(nn.Module):
    def __init__(self, in_channels, proto_channels=256, mask_channels=32, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.cv1 = _Conv(in_channels, proto_channels, kernel_size=3, operations=operations)
        self.upsample = operations.ConvTranspose2d(proto_channels, proto_channels, 2, 2, 0, bias=True)
        self.cv2 = _Conv(proto_channels, proto_channels, kernel_size=3, operations=operations)
        self.cv3 = _Conv(proto_channels, mask_channels, kernel_size=1, operations=operations)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


def _make_anchors(features, strides=(8.0, 16.0, 32.0), offset=0.5):
    anchor_points, stride_tensors = [], []
    dtype, device = features[0].dtype, features[0].device
    for feature, stride in zip(features, strides):
        height, width = feature.shape[-2:]
        sx = torch.arange(width, device=device, dtype=dtype) + offset
        sy = torch.arange(height, device=device, dtype=dtype) + offset
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((gx, gy), -1).reshape(-1, 2))
        stride_tensors.append(torch.full((height * width, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points, 0).transpose(0, 1), torch.cat(stride_tensors, 0).transpose(0, 1)


def _dist2bbox(distance, anchors):
    lt, rb = distance.chunk(2, 1)
    x1y1, x2y2 = anchors - lt, anchors + rb
    return torch.cat(((x1y1 + x2y2) * 0.5, x2y2 - x1y1), 1)


class _DetectHead(nn.Module):
    def __init__(self, nc=1, ch=(), reg_max=16, legacy_cv3=True, strides=(8.0, 16.0, 32.0), operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.nc, self.nl, self.reg_max, self.no, self.strides = nc, len(ch), reg_max, nc + reg_max * 4, strides
        c2, c3 = max(16, ch[0] // 4, reg_max * 4), max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList([nn.Sequential(_Conv(x, c2, 3, operations=operations), _Conv(c2, c2, 3, operations=operations), operations.Conv2d(c2, 4 * reg_max, 1)) for x in ch])
        if legacy_cv3:
            self.cv3 = nn.ModuleList([nn.Sequential(_Conv(x, c3, 3, operations=operations), _Conv(c3, c3, 3, operations=operations), operations.Conv2d(c3, nc, 1)) for x in ch])
        else:
            self.cv3 = nn.ModuleList([
                nn.Sequential(
                    nn.Sequential(_Conv(x, x, 3, groups=math.gcd(x, x), operations=operations), _Conv(x, c3, 1, operations=operations)),
                    nn.Sequential(_Conv(c3, c3, 3, groups=math.gcd(c3, c3), operations=operations), _Conv(c3, c3, 1, operations=operations)),
                    operations.Conv2d(c3, nc, 1),
                ) for x in ch
            ])
        self.dfl = _DFL(reg_max, operations=operations)

    def _decode(self, features):
        batch = features[0].shape[0]
        boxes = torch.cat([self.cv2[i](features[i]).view(batch, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.cv3[i](features[i]).view(batch, self.nc, -1) for i in range(self.nl)], dim=-1)
        anchors, strides = _make_anchors(features, self.strides)
        return torch.cat((_dist2bbox(self.dfl(boxes), anchors.unsqueeze(0)) * strides, scores.sigmoid()), dim=1)

    def forward(self, features):
        return self._decode(features)


class _SegmentHead(_DetectHead):
    def __init__(self, nc=1, nm=32, npr=256, ch=(), reg_max=16, legacy_cv3=True, strides=(8.0, 16.0, 32.0), operations=comfy.ops.disable_weight_init):
        super().__init__(nc=nc, ch=ch, reg_max=reg_max, legacy_cv3=legacy_cv3, strides=strides, operations=operations)
        self.nm, self.npr = nm, npr
        self.proto = _Proto(ch[0], npr, nm, operations=operations)
        c4 = max(ch[0] // 4, nm)
        self.cv4 = nn.ModuleList([nn.Sequential(_Conv(x, c4, 3, operations=operations), _Conv(c4, c4, 3, operations=operations), operations.Conv2d(c4, nm, 1)) for x in ch])

    def forward(self, features):
        batch = features[0].shape[0]
        mask_coeffs = torch.cat([self.cv4[i](features[i]).view(batch, self.nm, -1) for i in range(self.nl)], dim=-1)
        return torch.cat((self._decode(features), mask_coeffs), dim=1), self.proto(features[0])


class YOLOv8Model(nn.Module):
    """Eager YOLOv8 detection/segmentation network."""

    CHANNELS = (64, 128, 256, 512, 1024)
    DEPTHS = (3, 6, 6, 3)
    NC = 1
    SEGMENT = False
    NM = 32
    NPR = 256

    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        c1, c2, c3, c4, c5 = self.CHANNELS
        d1, d2, d3, d4 = self.DEPTHS
        head = _SegmentHead(nc=self.NC, nm=self.NM, npr=self.NPR, ch=(c3, c4, c5), legacy_cv3=True, operations=operations) if self.SEGMENT else _DetectHead(nc=self.NC, ch=(c3, c4, c5), legacy_cv3=True, operations=operations)
        self.model = nn.ModuleList([
            _Conv(3, c1, 3, 2, operations=operations), _Conv(c1, c2, 3, 2, operations=operations), _C2f(c2, c2, d1, True, operations=operations),
            _Conv(c2, c3, 3, 2, operations=operations), _C2f(c3, c3, d2, True, operations=operations),
            _Conv(c3, c4, 3, 2, operations=operations), _C2f(c4, c4, d3, True, operations=operations),
            _Conv(c4, c5, 3, 2, operations=operations), _C2f(c5, c5, d4, True, operations=operations), _SPPF(c5, c5, 5, operations=operations),
            nn.Upsample(scale_factor=2.0, mode="nearest"), nn.Identity(), _C2f(c5 + c4, c4, d1, False, operations=operations),
            nn.Upsample(scale_factor=2.0, mode="nearest"), nn.Identity(), _C2f(c4 + c3, c3, d1, False, operations=operations),
            _Conv(c3, c3, 3, 2, operations=operations), nn.Identity(), _C2f(c3 + c4, c4, d1, False, operations=operations),
            _Conv(c4, c4, 3, 2, operations=operations), nn.Identity(), _C2f(c4 + c5, c5, d1, False, operations=operations),
            head,
        ])

    def forward(self, x):
        layers = self.model
        x = layers[2](layers[1](layers[0](x)))
        p3 = layers[4](layers[3](x))
        p4 = layers[6](layers[5](p3))
        p5 = layers[9](layers[8](layers[7](p4)))
        head_p4 = layers[12](torch.cat((layers[10](p5), p4), dim=1))
        out_p3 = layers[15](torch.cat((layers[13](head_p4), p3), dim=1))
        out_p4 = layers[18](torch.cat((layers[16](out_p3), head_p4), dim=1))
        out_p5 = layers[21](torch.cat((layers[19](out_p4), p5), dim=1))
        return layers[22]([out_p3, out_p4, out_p5])


class YOLO11Model(nn.Module):
    """Eager YOLO11 detection/segmentation network."""

    CHANNELS = (64, 128, 256, 512, 512)
    DEPTHS = (2, 2, 2, 2)
    C3K_BACKBONE = False
    NC = 1
    SEGMENT = False
    NM = 32
    NPR = 256

    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        c1, c2, c3, c4, c5 = self.CHANNELS
        d1, d2, d3, d4 = self.DEPTHS
        c3k = self.C3K_BACKBONE
        head = _SegmentHead(nc=self.NC, nm=self.NM, npr=self.NPR, ch=(c3, c4, c5), legacy_cv3=False, operations=operations) if self.SEGMENT else _DetectHead(nc=self.NC, ch=(c3, c4, c5), legacy_cv3=False, operations=operations)
        self.model = nn.ModuleList([
            _Conv(3, c1, 3, 2, operations=operations), _Conv(c1, c2, 3, 2, operations=operations), _C3k2(c2, c3, d1, c3k=c3k, expansion=0.25, operations=operations),
            _Conv(c3, c3, 3, 2, operations=operations), _C3k2(c3, c4, d2, c3k=c3k, expansion=0.25, operations=operations),
            _Conv(c4, c4, 3, 2, operations=operations), _C3k2(c4, c4, d3, c3k=True, expansion=0.5, operations=operations),
            _Conv(c4, c5, 3, 2, operations=operations), _C3k2(c5, c5, d4, c3k=True, expansion=0.5, operations=operations),
            _SPPF(c5, c5, 5, operations=operations), _C2PSA(c5, c5, d1, expansion=0.5, operations=operations),
            nn.Upsample(scale_factor=2.0, mode="nearest"), nn.Identity(), _C3k2(c5 + c4, c4, d1, c3k=c3k, expansion=0.5, operations=operations),
            nn.Upsample(scale_factor=2.0, mode="nearest"), nn.Identity(), _C3k2(c4 + c4, c3, d1, c3k=c3k, expansion=0.5, operations=operations),
            _Conv(c3, c3, 3, 2, operations=operations), nn.Identity(), _C3k2(c3 + c4, c4, d1, c3k=c3k, expansion=0.5, operations=operations),
            _Conv(c4, c4, 3, 2, operations=operations), nn.Identity(), _C3k2(c4 + c5, c5, d1, c3k=True, expansion=0.5, operations=operations),
            head,
        ])

    def forward(self, x):
        layers = self.model
        x = layers[2](layers[1](layers[0](x)))
        p3 = layers[4](layers[3](x))
        p4 = layers[6](layers[5](p3))
        p5 = layers[10](layers[9](layers[8](layers[7](p4))))
        head_p4 = layers[13](torch.cat((layers[11](p5), p4), dim=1))
        out_p3 = layers[16](torch.cat((layers[14](head_p4), p3), dim=1))
        out_p4 = layers[19](torch.cat((layers[17](out_p3), head_p4), dim=1))
        out_p5 = layers[22](torch.cat((layers[20](out_p4), p5), dim=1))
        return layers[23]([out_p3, out_p4, out_p5])


class YOLOv8nDetect(YOLOv8Model):
    CHANNELS = (16, 32, 64, 128, 256)
    DEPTHS = (1, 2, 2, 1)


class YOLOv8sDetect(YOLOv8Model):
    CHANNELS = (32, 64, 128, 256, 512)
    DEPTHS = (1, 2, 2, 1)


class YOLOv8mDetect(YOLOv8Model):
    CHANNELS = (48, 96, 192, 384, 576)
    DEPTHS = (2, 4, 4, 2)


class YOLOv8nSeg(YOLOv8nDetect):
    SEGMENT = True
    NPR = 64


class YOLOv8sSeg(YOLOv8sDetect):
    SEGMENT = True
    NPR = 128


class YOLOv8mSeg(YOLOv8mDetect):
    SEGMENT = True
    NPR = 192


class YOLO11xDetect(YOLO11Model):
    CHANNELS = (96, 192, 384, 768, 768)
    DEPTHS = (2, 2, 2, 2)
    C3K_BACKBONE = True
