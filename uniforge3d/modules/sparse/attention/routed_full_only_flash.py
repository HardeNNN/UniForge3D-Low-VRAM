"""Packed Full-only routed cross-attention backed by FlashAttention varlen."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..basic import SparseTensor
from .modules import SparseMultiHeadRMSNorm

try:
    from flash_attn import flash_attn_varlen_func
except Exception as exc:  # pragma: no cover - validated at startup on production nodes
    flash_attn_varlen_func = None
    _FLASH_IMPORT_ERROR = exc
else:
    _FLASH_IMPORT_ERROR = None


class PackedRoutedFullOnlyLayoutBuilder:
    """Build one packed selected-view query layout shared by all routed layers."""

    def __init__(self, grid_resolution: int = 64) -> None:
        self.grid_resolution = int(grid_resolution)

    @staticmethod
    def _gather_grid(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 4:
            raise ValueError(f"Expected routed grid [B,R,R,R], got {tuple(grid.shape)}")
        return grid[
            coords[:, 0].long(),
            coords[:, 1].long(),
            coords[:, 2].long(),
            coords[:, 3].long(),
        ]

    def prepare(
        self,
        *,
        x: SparseTensor,
        selected_view: torch.Tensor,
        batch_size: int,
        views: int,
    ) -> Dict[str, torch.Tensor | int]:
        """Return the packed-query fields consumed directly by routed attention."""
        coords = x.coords
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"Expected sparse coords [N,4], got {tuple(coords.shape)}")
        xyz = coords[:, 1:].long()
        if xyz.numel() and (
            int(xyz.min().item()) < 0 or int(xyz.max().item()) >= self.grid_resolution
        ):
            raise ValueError(
                f"Sparse coords must be in [0,{self.grid_resolution - 1}]"
            )
        batch_id = coords[:, 0].long()
        if batch_id.numel() and (
            int(batch_id.min().item()) < 0 or int(batch_id.max().item()) >= batch_size
        ):
            raise ValueError("Sparse query batch id is out of range")

        selected = self._gather_grid(selected_view, coords).long()
        if selected.numel() and (
            int(selected.min().item()) < 0 or int(selected.max().item()) >= views
        ):
            raise ValueError("selected_view contains a non-existent view")

        group_id = batch_id * int(views) + selected
        group_count = int(batch_size) * int(views)
        counts = torch.bincount(group_id, minlength=group_count)
        active_groups = torch.nonzero(counts > 0, as_tuple=False).flatten()
        if active_groups.numel() == 0:
            raise ValueError("Routed attention received no sparse queries")

        try:
            order = torch.argsort(group_id, stable=True)
        except TypeError:
            order = torch.argsort(group_id)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)

        query_lengths = counts[active_groups].to(torch.int32)
        cu_q = torch.zeros(
            active_groups.numel() + 1, dtype=torch.int32, device=coords.device
        )
        cu_q[1:] = torch.cumsum(query_lengths, dim=0)
        full_tokens = self.grid_resolution ** 2 + 5
        cu_k = (
            torch.arange(
                active_groups.numel() + 1, dtype=torch.int32, device=coords.device
            )
            * full_tokens
        )
        return {
            "routed_query_order": order,
            "routed_query_inverse": inverse,
            "routed_active_groups": active_groups.long(),
            "routed_cu_seqlens_q": cu_q,
            "routed_cu_seqlens_k": cu_k,
            "routed_max_seqlen_q": int(query_lengths.max().item()),
        }


class RoutedFullOnlyFlashCrossAttention(nn.Module):
    """Each query attends to all tokens of exactly one selected real view."""

    full_tokens = 4101

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = int(channels)
        self.ctx_channels = int(ctx_channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.qk_rms_norm = bool(qk_rms_norm)
        if self.num_heads != 12 or self.head_dim != 128:
            raise ValueError("production Full-only Flash path requires 12 heads x 128")
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "flash_attn_varlen_func is required for routed Full-only attention"
            ) from _FLASH_IMPORT_ERROR

        self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
        self.to_kv = nn.Linear(ctx_channels, channels * 2, bias=qkv_bias)
        self.to_out = nn.Linear(channels, channels)
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)

    def forward(
        self,
        x: SparseTensor,
        context: Dict[str, torch.Tensor],
    ) -> SparseTensor:
        full = context["full_bank"]
        if full.ndim != 4 or int(full.shape[2]) != self.full_tokens:
            raise ValueError(
                f"Expected Full bank [B,V,{self.full_tokens},C], got {tuple(full.shape)}"
            )
        batch, views, _, width = full.shape
        if int(width) != self.ctx_channels:
            raise ValueError(
                f"Expected Full width {self.ctx_channels}, got {int(width)}"
            )

        q = self.to_q(x.feats).view(-1, self.num_heads, self.head_dim)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
        order = context["routed_query_order"]
        inverse = context["routed_query_inverse"]
        q = q[order].contiguous()

        active_groups = context["routed_active_groups"]
        flat_full = full.reshape(
            int(batch) * int(views), self.full_tokens, width
        )
        active_full = flat_full.index_select(0, active_groups).contiguous()
        kv = self.to_kv(active_full).view(
            active_groups.numel(),
            self.full_tokens,
            2,
            self.num_heads,
            self.head_dim,
        )
        k, v = kv.unbind(dim=2)
        if self.qk_rms_norm:
            k = self.k_rms_norm(k)
        k = k.contiguous()
        v = v.contiguous()

        k_packed = k.reshape(-1, self.num_heads, self.head_dim)
        v_packed = v.reshape(-1, self.num_heads, self.head_dim)
        out_grouped = flash_attn_varlen_func(
            q,
            k_packed,
            v_packed,
            context["routed_cu_seqlens_q"],
            context["routed_cu_seqlens_k"],
            int(context["routed_max_seqlen_q"]),
            self.full_tokens,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )
        out = out_grouped[inverse]
        return x.replace(self.to_out(out.reshape(-1, self.channels)))
