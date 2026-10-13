"""MS-TCN: multi-stage temporal convolutional network (SW-7, FR-TMP-2).

A first stage of dilated residual 1-D convolutions maps a sequence of per-frame
features (T x D) to per-frame class scores; each later stage refines the previous
stage's softmax. Dilation doubles per layer, so a stage sees a receptive field of
2^layers frames, and the stack is small enough to train on a free-tier GPU from
precomputed features (NFR-PORT-1). Loss is cross-entropy per stage plus a truncated
MSE on the log-probability differences between adjacent frames, which damps the
over-segmentation a bare classifier produces.

This is the architecture of Farha & Gall (2019), written fresh for generic-verb labels;
MS-TCN++ and ASFormer (SW-7's alternatives) are not implemented. It is trained and
evaluated here on synthetic sequences only -- no public benchmark has been run (DR-PUB-1
is open) -- so nothing here is a benchmark result.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedResidual(nn.Module):
    def __init__(self, dilation: int, channels: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1x1(F.relu(self.conv_dilated(x)))
        return x + self.dropout(out)


class SingleStage(nn.Module):
    def __init__(self, layers: int, channels: int, dim_in: int, classes: int):
        super().__init__()
        self.conv_in = nn.Conv1d(dim_in, channels, 1)
        self.layers = nn.ModuleList(_DilatedResidual(2**i, channels) for i in range(layers))
        self.conv_out = nn.Conv1d(channels, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class MSTCN(nn.Module):
    """Input (batch, dim, T) -> list of per-stage logits, each (batch, classes, T)."""

    def __init__(self, dim_in: int, classes: int, stages: int = 3, layers: int = 8,
                 channels: int = 32):
        super().__init__()
        self.first = SingleStage(layers, channels, dim_in, classes)
        self.refine = nn.ModuleList(
            SingleStage(layers, channels, classes, classes) for _ in range(stages - 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.first(x)
        outs = [out]
        for stage in self.refine:
            out = stage(F.softmax(out, dim=1))
            outs.append(out)
        return outs


def loss_fn(outputs: list[torch.Tensor], target: torch.Tensor, smooth: float = 0.15,
            clamp: float = 16.0) -> torch.Tensor:
    """Sum over stages of cross-entropy + smoothing. `target` is (batch, T) of class ids."""
    total = outputs[0].new_zeros(())
    for logits in outputs:
        total = total + F.cross_entropy(logits, target)
        logp = F.log_softmax(logits, dim=1)
        diff = (logp[:, :, 1:] - logp[:, :, :-1].detach()) ** 2
        total = total + smooth * torch.clamp(diff, max=clamp).mean()
    return total


def train(
    model: MSTCN,
    data: list[tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Train on (features (T, D), labels (T,)) pairs, one sequence per step. Returns the
    mean loss per epoch."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for feats, labels in data:
            opt.zero_grad()
            loss = loss_fn(model(feats.t().unsqueeze(0)), labels.unsqueeze(0))
            loss.backward()
            opt.step()
            total += float(loss.detach())
        history.append(total / len(data))
    return history


@torch.no_grad()
def predict(model: MSTCN, feats: torch.Tensor) -> list[int]:
    """Per-frame class ids for one (T, D) sequence, from the last stage."""
    model.eval()
    return model(feats.t().unsqueeze(0))[-1].argmax(dim=1).squeeze(0).tolist()
