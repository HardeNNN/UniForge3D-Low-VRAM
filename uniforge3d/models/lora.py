"""Unmerged Stage1 LoRA, with the same operations and FP32 parameters as training."""

from fnmatch import fnmatchcase

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        result = self.base(x)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return result + update * self.scaling


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_channels, *base.kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, rank, 1, 1))

    def forward(self, x):
        result = self.base(x)
        update = F.conv2d(
            x, self.lora_A, stride=self.base.stride,
            padding=self.base.padding, dilation=self.base.dilation,
        )
        update = F.conv2d(update, self.lora_B)
        return result + update * self.scaling


class LoRAConvTranspose2d(nn.Module):
    def __init__(self, base: nn.ConvTranspose2d, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(base.in_channels, rank, *base.kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, rank, 1, 1))

    def forward(self, x):
        result = self.base(x)
        update = F.conv_transpose2d(
            x, self.lora_A, stride=self.base.stride, padding=self.base.padding,
            output_padding=self.base.output_padding, dilation=self.base.dilation,
        )
        update = F.conv2d(update, self.lora_B)
        return result + update * self.scaling


def attach_stage1_lora(model: nn.Module, *, rank: int, alpha: float, target_modules: list[str]):
    wrappers = {nn.Linear: LoRALinear, nn.Conv2d: LoRAConv2d, nn.ConvTranspose2d: LoRAConvTranspose2d}
    model.requires_grad_(False)
    selected = [
        (name, module) for name, module in model.named_modules()
        if type(module) in wrappers and any(fnmatchcase(name, pattern) for pattern in target_modules)
    ]
    if not selected:
        raise ValueError(f"No Stage1 LoRA modules matched {target_modules}")
    for name, module in selected:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and module.groups != 1:
            raise ValueError(f"Stage1 LoRA requires groups=1: {name}")
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, wrappers[type(module)](module, rank, alpha))
