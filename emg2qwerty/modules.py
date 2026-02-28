# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
import torch
import torch.nn as nn
from typing import Sequence, Optional

class ChannelMask(nn.Module):
    """Masks entire electrode channels after normalization.

    Expects inputs of shape (T, N, bands, C, freq).

    Modes:
      - Stochastic: sample a new Bernoulli keep-mask every forward pass (fixed=False).
      - Deterministic: create one fixed keep-mask and reuse it (fixed=True).

    Deterministic mask can be specified by:
      - drop_indices: list of flattened (band, channel) indices to drop, OR
      - keep_indices: list of flattened (band, channel) indices to keep, OR
      - seed + drop_prob: randomly choose a fixed subset to drop once.
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        mode: str = "zero",
        gaussian_std: float = 1.0,
        per_example: bool = False,          # deterministic typically wants shared mask
        active_in_eval: bool = False,
        fixed: bool = False,
        seed: Optional[int] = None,
        drop_indices: Optional[Sequence[int]] = None,
        keep_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if not (0.0 <= drop_prob <= 1.0):
            raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
        if mode not in {"zero", "gaussian"}:
            raise ValueError(f"mode must be 'zero' or 'gaussian', got {mode}")
        if drop_indices is not None and keep_indices is not None:
            raise ValueError("Specify only one of drop_indices or keep_indices.")
        if fixed and drop_prob == 0.0 and drop_indices is None and keep_indices is None:
            # fixed True but nothing to do; allowed, but pointless
            pass

        self.drop_prob = float(drop_prob)
        self.mode = mode
        self.gaussian_std = float(gaussian_std)
        self.per_example = bool(per_example)
        self.active_in_eval = bool(active_in_eval)

        self.fixed = bool(fixed)
        self.seed = seed
        self.drop_indices = list(drop_indices) if drop_indices is not None else None
        self.keep_indices = list(keep_indices) if keep_indices is not None else None

        # Will be created lazily on first forward when we know (bands, C).
        self.register_buffer("fixed_keep_mask", None, persistent=False)

    def _build_fixed_keep_mask(self, bands: int, C: int, device, dtype) -> torch.Tensor:
        total = bands * C
        keep = torch.ones((bands, C), device=device, dtype=dtype)

        if self.drop_indices is not None:
            idx = torch.tensor(self.drop_indices, device=device)
            if (idx.min() < 0) or (idx.max() >= total):
                raise ValueError(f"drop_indices must be in [0, {total-1}]")
            keep.view(-1)[idx] = 0

        elif self.keep_indices is not None:
            idx = torch.tensor(self.keep_indices, device=device)
            if (idx.min() < 0) or (idx.max() >= total):
                raise ValueError(f"keep_indices must be in [0, {total-1}]")
            keep.zero_()
            keep.view(-1)[idx] = 1

        else:
            # Choose exactly round(drop_prob * total) channels to drop, deterministically.
            num_drop = int(round(self.drop_prob * total))
            perm = torch.randperm(total)  # on CPU
            drop = perm[:num_drop]
            keep.view(-1)[drop.to(device=device)] = 0

        return keep  # shape (bands, C)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 and self.drop_indices is None and self.keep_indices is None:
            return inputs
        if (not self.training) and (not self.active_in_eval):
            return inputs

        # inputs: (T, N, bands, C, freq)
        T, N, bands, C, freq = inputs.shape

        if self.fixed:
            if self.fixed_keep_mask is None or self.fixed_keep_mask.shape != (bands, C):
                self.fixed_keep_mask = self._build_fixed_keep_mask(
                    bands=bands, C=C, device=inputs.device, dtype=inputs.dtype
                )
            keep_bc = self.fixed_keep_mask  # (bands, C)

            if self.per_example:
                keep_mask = keep_bc.view(1, 1, bands, C, 1).expand(1, N, bands, C, 1)
            else:
                keep_mask = keep_bc.view(1, 1, bands, C, 1)
        else:
            # Stochastic per-forward mask
            keep_prob = 1.0 - self.drop_prob
            if self.per_example:
                mask_shape = (1, N, bands, C, 1)
            else:
                mask_shape = (1, 1, bands, C, 1)
            keep_mask = torch.empty(mask_shape, device=inputs.device, dtype=inputs.dtype).bernoulli_(keep_prob)

        if self.mode == "zero":
            return inputs * keep_mask

        noise = torch.randn_like(inputs) * self.gaussian_std
        return inputs * keep_mask + noise * (1.0 - keep_mask)


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)
