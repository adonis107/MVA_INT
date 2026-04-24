from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .encoders import BinDensityEncoder, CylindricalEncoder, MeanFieldPolicy


@dataclass(frozen=True)
class ABMConfig:
    horizon: float = 0.2
    steps: int = 20
    sigma: float = 1.0
    interaction_q: float = 0.8
    state_dim: int = 1
    default_threshold: float = -0.5

    @property
    def dt(self) -> float:
        return self.horizon / self.steps


@dataclass(frozen=True)
class ABMSimulationResult:
    states: Tensor
    actions: Tensor
    defaulted: Tensor
    new_defaults_per_step: Tensor


def _row_normalize(adjacency: Tensor) -> Tensor:
    row_sums = adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return adjacency / row_sums


def homogeneous_graph_weights(
    agents: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    return torch.full((agents, agents), 1.0 / agents, device=device, dtype=dtype)


def core_periphery_graph_weights(
    agents: int,
    hubs: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if hubs <= 0 or hubs >= agents:
        raise ValueError("hubs must satisfy 0 < hubs < agents")
    adjacency = torch.zeros((agents, agents), device=device, dtype=dtype)
    hub_idx = torch.arange(hubs, device=device)
    periphery_idx = torch.arange(hubs, agents, device=device)
    adjacency[hub_idx.unsqueeze(1), hub_idx.unsqueeze(0)] = 1.0
    adjacency[hub_idx.unsqueeze(1), periphery_idx.unsqueeze(0)] = 1.0
    adjacency[periphery_idx.unsqueeze(1), hub_idx.unsqueeze(0)] = 1.0
    adjacency.fill_diagonal_(1.0)
    return _row_normalize(adjacency)


def erdos_renyi_graph_weights(
    agents: int,
    edge_probability: float,
    *,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if edge_probability < 0.0 or edge_probability > 1.0:
        raise ValueError("edge_probability must be in [0, 1]")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    upper = torch.rand((agents, agents), generator=generator, device=device, dtype=dtype)
    upper = torch.triu((upper < edge_probability).to(dtype=dtype), diagonal=1)
    adjacency = upper + upper.transpose(0, 1)
    adjacency.fill_diagonal_(1.0)
    return _row_normalize(adjacency)


def expand_graph_weights(weights: Tensor, batch_size: int) -> Tensor:
    if weights.ndim == 2:
        return weights.unsqueeze(0).expand(batch_size, -1, -1)
    if weights.ndim == 3 and weights.shape[0] == batch_size:
        return weights
    raise ValueError("weights must have shape (N,N) or (B,N,N)")


def local_empirical_means(states: Tensor, weights: Tensor) -> Tensor:
    return torch.bmm(weights, states)


def apply_initial_shock(states: Tensor, node_indices: Tensor | list[int], shock_value: float) -> Tensor:
    shocked = states.clone()
    shocked[:, node_indices, :] = shock_value
    return shocked


def default_mask(states: Tensor, threshold: float) -> Tensor:
    return (states < threshold).any(dim=-1)


def weighted_encoder_features(policy: MeanFieldPolicy, states: Tensor, weights: Tensor) -> Tensor:
    encoder = policy.encoder
    if isinstance(encoder, BinDensityEncoder):
        edges = encoder.edges
        clamped = states.clamp(float(edges[0]), float(edges[-1]) - 1e-8)
        indices = torch.bucketize(clamped, edges[1:-1], right=False)
        linear = indices[..., 0]
        multiplier = 1
        for dim in range(1, encoder.state_dim):
            multiplier *= encoder.bins
            linear = linear + indices[..., dim] * multiplier
        classes = encoder.output_dim
        one_hot = torch.nn.functional.one_hot(linear, num_classes=classes).to(states.dtype)
        return torch.einsum("bij,bjc->bic", weights, one_hot)
    if isinstance(encoder, CylindricalEncoder):
        embedded = encoder.inner(states)
        return torch.bmm(weights, embedded)
    raise TypeError("Unsupported encoder type for weighted local measures")


def policy_actions_from_local_measures(
    policy: MeanFieldPolicy,
    time: float | Tensor,
    states: Tensor,
    weights: Tensor,
) -> Tensor:
    batch_size, agents, _ = states.shape
    features = weighted_encoder_features(policy, states, weights)
    if not torch.is_tensor(time):
        time = torch.tensor(time, device=states.device, dtype=states.dtype)
    if time.ndim == 0:
        time = time.repeat(batch_size)
    if time.ndim == 1:
        time = time.unsqueeze(-1)
    time_tensor = time.unsqueeze(1).expand(batch_size, agents, 1)
    inputs = torch.cat((time_tensor, states, features), dim=-1)
    return policy.network(inputs)


def euler_step(
    states: Tensor,
    config: ABMConfig,
    weights: Tensor,
    *,
    actions: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    dt = config.dt
    sqrt_dt = dt**0.5
    local_mean = local_empirical_means(states, weights)
    drift = config.interaction_q * (local_mean - states)
    if actions is not None:
        drift = drift + actions
    if noise is None:
        noise = torch.randn_like(states)
    return states + dt * drift + config.sigma * sqrt_dt * noise


def rollout_abm(
    initial_states: Tensor,
    config: ABMConfig,
    weights: Tensor,
    *,
    policy: MeanFieldPolicy | None = None,
    noise: Tensor | None = None,
) -> ABMSimulationResult:
    batch_size, agents, state_dim = initial_states.shape
    if state_dim != config.state_dim:
        raise ValueError("initial_states state_dim does not match config.state_dim")
    batched_weights = expand_graph_weights(weights, batch_size)
    if noise is None:
        noise = torch.randn(
            config.steps,
            batch_size,
            agents,
            state_dim,
            device=initial_states.device,
            dtype=initial_states.dtype,
        )
    states = [initial_states]
    actions_history = []
    current = initial_states
    defaulted = default_mask(current, config.default_threshold)
    new_defaults = []
    for step in range(config.steps):
        time = step * config.dt
        if policy is None:
            actions = torch.zeros_like(current)
        else:
            with torch.inference_mode():
                actions = policy_actions_from_local_measures(policy, time, current, batched_weights)
        next_states = euler_step(current, config, batched_weights, actions=actions, noise=noise[step])
        next_defaulted = default_mask(next_states, config.default_threshold)
        newly_defaulted = next_defaulted & (~defaulted)
        new_defaults.append(newly_defaulted.sum(dim=1))
        defaulted = defaulted | next_defaulted
        states.append(next_states)
        actions_history.append(actions)
        current = next_states
    return ABMSimulationResult(
        states=torch.stack(states, dim=0),
        actions=torch.stack(actions_history, dim=0),
        defaulted=defaulted,
        new_defaults_per_step=torch.stack(new_defaults, dim=0),
    )