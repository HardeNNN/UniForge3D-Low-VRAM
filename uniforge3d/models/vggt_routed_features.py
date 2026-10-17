"""Frozen coords64 VGGT backbone and its full-precision point/DPT head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .vggt.heads.dpt_head import DPTHead, custom_interpolate
from .vggt.heads.head_act import activate_head
from .vggt_adapter import (
    FrozenVGGTFeatureExtractor,
    PackedCrossAttention,
    load_prefixed_state,
)


class DPTShapeAdapter32(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = PackedCrossAttention(256, 1024, 256, 8)

    def forward(self, feature, batch, views, shape_tokens, shape_offsets):
        batch_views, channels, height, width = feature.shape
        if batch_views != batch * views or channels != 256:
            raise ValueError(f"Invalid DPT feature shape: {feature.shape}")
        tokens = feature.view(batch, views, channels, height * width)
        tokens = tokens.permute(0, 1, 3, 2).reshape(batch, views * height * width, channels)
        tokens = self.attn(tokens, shape_tokens, shape_offsets)
        return tokens.view(batch, views, height * width, channels).permute(0, 1, 3, 2).reshape_as(feature)


class FrozenRoutedVGGTFeatureExtractor(FrozenVGGTFeatureExtractor):
    def _load_shared_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        self.point_head = DPTHead(dim_in=2048, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.dpt_shape_adapter_32 = DPTShapeAdapter32()
        load_prefixed_state(self.point_head, state, "point_head.", "point head")
        load_prefixed_state(
            self.dpt_shape_adapter_32, state, "dpt_shape_adapter_32.", "DPT shape adapter"
        )
        for module in (self.point_head, self.dpt_shape_adapter_32):
            module.requires_grad_(False)
            module.eval()
            module.to(device=self.device, dtype=torch.float32)

    def _point_forward(self, cached, images, patch_start, shape_tokens, shape_offsets):
        head = self.point_head
        batch, views, _, height, width = images.shape
        patch_height, patch_width = height // head.patch_size, width // head.patch_size
        features = []
        for dpt_index, layer_index in enumerate(head.intermediate_layer_idx):
            value = cached[layer_index][:, :, patch_start:].reshape(batch * views, -1, 2048).float()
            value = head.norm(value).permute(0, 2, 1).reshape(batch * views, 2048, patch_height, patch_width)
            value = head.projects[dpt_index](value)
            if head.pos_embed:
                value = head._apply_pos_embed(value, width, height)
            features.append(head.resize_layers[dpt_index](value))
        layer1, layer2, layer3, layer4 = features
        layer1, layer2 = head.scratch.layer1_rn(layer1), head.scratch.layer2_rn(layer2)
        layer3, layer4 = head.scratch.layer3_rn(layer3), head.scratch.layer4_rn(layer4)
        output = head.scratch.refinenet4(layer4, size=layer3.shape[2:])
        output = self.dpt_shape_adapter_32(output, batch, views, shape_tokens.float(), shape_offsets)
        output = head.scratch.refinenet3(output, layer3, size=layer2.shape[2:])
        output = head.scratch.refinenet2(output, layer2, size=layer1.shape[2:])
        output = head.scratch.refinenet1(output, layer1)
        output = head.scratch.output_conv1(output)
        output = custom_interpolate(output, (height, width), mode="bilinear", align_corners=True)
        if head.pos_embed:
            output = head._apply_pos_embed(output, width, height)
        points, confidence = activate_head(
            head.scratch.output_conv2(output), head.activation, head.conf_activation
        )
        return points.view(batch, views, *points.shape[1:]), confidence.view(
            batch, views, *confidence.shape[1:]
        )

    def __call__(self, images, shape_coords, shape_offsets, *, return_points: bool = True):
        cached, patch_start = super().__call__(images, shape_coords, shape_offsets)
        if not return_points:
            return cached, patch_start, None, None
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
            shape_tokens = self.shape_encoder(shape_coords.to(self.device))
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            points, confidence = self._point_forward(
                cached,
                images.to(self.device),
                patch_start,
                shape_tokens.float(),
                shape_offsets.to(self.device),
            )
        return cached, patch_start, points.float(), confidence.float()
