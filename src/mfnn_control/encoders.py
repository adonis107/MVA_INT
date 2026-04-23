from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import EncoderConfig


def make_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.Tanh())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class BinDensityEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.state_dim = config.state_dim
        self.bins = config.bins
        centers = torch.linspace(config.support_min, config.support_max, config.bins)
        edges = torch.linspace(config.support_min, config.support_max, config.bins + 1)
        self.register_buffer("centers", centers)
        self.register_buffer("edges", edges)

    @property
    def output_dim(self) -> int:
        return int(self.bins**self.state_dim)

    def forward(self, states: Tensor) -> Tensor:
        clamped = states.clamp(float(self.edges[0]), float(self.edges[-1]) - 1e-8)
        indices = torch.bucketize(clamped, self.edges[1:-1], right=False)
        linear = indices[..., 0]
        multiplier = 1
        for dim in range(1, self.state_dim):
            multiplier *= self.bins
            linear = linear + indices[..., dim] * multiplier
        one_hot = torch.nn.functional.one_hot(linear, num_classes=self.output_dim).to(states.dtype)
        return one_hot.mean(dim=1)


class CylindricalEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.state_dim = config.state_dim
        self.inner = make_mlp(self.state_dim, config.hidden_dims, config.latent_dim)
        self._output_dim = config.latent_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, states: Tensor) -> Tensor:
        embedded = self.inner(states)
        return embedded.mean(dim=1)


@dataclass(frozen=True)
class PolicyOutput:
    actions: Tensor
    measure_features: Tensor


class MeanFieldPolicy(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.encoder = encoder
        feature_dim = getattr(encoder, "output_dim")
        self.state_dim = getattr(encoder, "state_dim", 1)
        self.network = make_mlp(feature_dim + 1 + self.state_dim, hidden_dims, self.state_dim)

    def forward(self, time: Tensor | float, states: Tensor) -> PolicyOutput:
        batch_size, particles, _ = states.shape
        measure_features = self.encoder(states)
        expanded_features = measure_features.unsqueeze(1).expand(batch_size, particles, -1)
        time_tensor = self._time_tensor(time, batch_size, particles, states.device, states.dtype)
        inputs = torch.cat((time_tensor, states, expanded_features), dim=-1)
        actions = self.network(inputs)
        return PolicyOutput(actions=actions, measure_features=measure_features)

    @staticmethod
    def _time_tensor(
        time: Tensor | float,
        batch_size: int,
        particles: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if not torch.is_tensor(time):
            time = torch.tensor(time, device=device, dtype=dtype)
        if time.ndim == 0:
            time = time.repeat(batch_size)
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        return time.unsqueeze(1).expand(batch_size, particles, 1)


class MeanFieldInitialValue(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dims: tuple[int, ...], output_dim: int = 1) -> None:
        super().__init__()
        self.encoder = encoder
        feature_dim = getattr(encoder, "output_dim")
        self.state_dim = getattr(encoder, "state_dim", 1)
        self.network = make_mlp(feature_dim + self.state_dim, hidden_dims, output_dim)

    def forward(self, states: Tensor) -> Tensor:
        batch_size, particles, _ = states.shape
        measure_features = self.encoder(states)
        expanded_features = measure_features.unsqueeze(1).expand(batch_size, particles, -1)
        inputs = torch.cat((states, expanded_features), dim=-1)
        return self.network(inputs)


class MeanFieldProcess(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dims: tuple[int, ...], output_dim: int = 1) -> None:
        super().__init__()
        self.encoder = encoder
        feature_dim = getattr(encoder, "output_dim")
        self.state_dim = getattr(encoder, "state_dim", 1)
        self.network = make_mlp(feature_dim + 1 + self.state_dim, hidden_dims, output_dim)

    def forward(self, time: Tensor | float, states: Tensor) -> Tensor:
        batch_size, particles, _ = states.shape
        measure_features = self.encoder(states)
        expanded_features = measure_features.unsqueeze(1).expand(batch_size, particles, -1)
        time_tensor = MeanFieldPolicy._time_tensor(time, batch_size, particles, states.device, states.dtype)
        inputs = torch.cat((time_tensor, states, expanded_features), dim=-1)
        return self.network(inputs)


def build_encoder(config: EncoderConfig) -> nn.Module:
    if config.kind == "bins":
        return BinDensityEncoder(config)
    if config.kind == "cylindrical":
        return CylindricalEncoder(config)
    raise ValueError(f"Unsupported encoder kind: {config.kind}")