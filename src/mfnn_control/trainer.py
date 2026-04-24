from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor

from .config import EncoderConfig, SystemicRiskConfig, TrainingConfig
from .encoders import MeanFieldInitialValue, MeanFieldPolicy, MeanFieldProcess, build_encoder
from .systemic_risk import (
    adjoint_induced_control,
    policy_loss,
    sample_initial_state_batch_with_dim,
    simulate_systemic_risk,
    systemic_risk_adjoint_drift,
    systemic_risk_adjoint_terminal,
    systemic_risk_bsde_driver,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_policy(encoder_config: EncoderConfig, training_config: TrainingConfig) -> MeanFieldPolicy:
    encoder = build_encoder(encoder_config)
    return MeanFieldPolicy(encoder=encoder, hidden_dims=training_config.hidden_dims)


def build_global_bsde_networks(
    encoder_config: EncoderConfig,
    training_config: TrainingConfig,
) -> tuple[MeanFieldInitialValue, MeanFieldProcess]:
    output_dim = encoder_config.state_dim
    return (
        MeanFieldInitialValue(build_encoder(encoder_config), training_config.hidden_dims, output_dim=output_dim),
        MeanFieldProcess(build_encoder(encoder_config), training_config.hidden_dims, output_dim=output_dim),
    )


def run_training_step(
    policy: MeanFieldPolicy,
    optimizer: torch.optim.Optimizer,
    config: SystemicRiskConfig,
    training_config: TrainingConfig,
) -> float:
    dtype = getattr(torch, config.dtype)
    initial_states, _ = sample_initial_state_batch_with_dim(
        training_config.training_cases,
        training_config.batch_size,
        config.particles,
        config.state_dim,
        device=config.device,
        dtype=dtype,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = policy_loss(policy, initial_states, config)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def train_global_dp(
    config: SystemicRiskConfig,
    encoder_config: EncoderConfig,
    training_config: TrainingConfig,
) -> tuple[MeanFieldPolicy, list[float]]:
    set_seed(training_config.seed)
    policy = build_policy(encoder_config, training_config).to(device=config.device, dtype=getattr(torch, config.dtype))
    optimizer = torch.optim.Adam(policy.parameters(), lr=training_config.learning_rate)
    losses: list[float] = []
    for _ in range(training_config.iterations):
        losses.append(run_training_step(policy, optimizer, config, training_config))
    return policy, losses


def global_bsde_loss(
    initial_value_network: MeanFieldInitialValue,
    process_network: MeanFieldProcess,
    initial_states: Tensor,
    config: SystemicRiskConfig,
) -> Tensor:
    dt = config.dt
    sqrt_dt = dt**0.5
    current_states = initial_states
    current_adjoint = initial_value_network(initial_states)
    noise = torch.randn(
        config.steps,
        *initial_states.shape,
        device=initial_states.device,
        dtype=initial_states.dtype,
    )
    for step in range(config.steps):
        time = torch.full(
            (initial_states.shape[0],),
            fill_value=step * dt,
            device=initial_states.device,
            dtype=initial_states.dtype,
        )
        z_value = process_network(time, current_states)
        next_states = current_states + dt * systemic_risk_adjoint_drift(current_states, current_adjoint, config)
        next_states = next_states + config.sigma * sqrt_dt * noise[step]
        next_adjoint = current_adjoint + dt * systemic_risk_bsde_driver(current_states, current_adjoint, config)
        next_adjoint = next_adjoint + z_value * noise[step]
        current_states = next_states
        current_adjoint = next_adjoint
    terminal_target = systemic_risk_adjoint_terminal(current_states, config)
    return (current_adjoint - terminal_target).square().mean()


def run_global_bsde_step(
    initial_value_network: MeanFieldInitialValue,
    process_network: MeanFieldProcess,
    optimizer: torch.optim.Optimizer,
    config: SystemicRiskConfig,
    training_config: TrainingConfig,
) -> float:
    dtype = getattr(torch, config.dtype)
    initial_states, _ = sample_initial_state_batch_with_dim(
        training_config.training_cases,
        training_config.batch_size,
        config.particles,
        config.state_dim,
        device=config.device,
        dtype=dtype,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = global_bsde_loss(initial_value_network, process_network, initial_states, config)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def train_global_bsde(
    config: SystemicRiskConfig,
    encoder_config: EncoderConfig,
    training_config: TrainingConfig,
) -> tuple[tuple[MeanFieldInitialValue, MeanFieldProcess], list[float]]:
    set_seed(training_config.seed)
    initial_value_network, process_network = build_global_bsde_networks(encoder_config, training_config)
    dtype = getattr(torch, config.dtype)
    initial_value_network = initial_value_network.to(device=config.device, dtype=dtype)
    process_network = process_network.to(device=config.device, dtype=dtype)
    optimizer = torch.optim.Adam(
        list(initial_value_network.parameters()) + list(process_network.parameters()),
        lr=training_config.learning_rate,
    )
    losses: list[float] = []
    for _ in range(training_config.iterations):
        losses.append(run_global_bsde_step(initial_value_network, process_network, optimizer, config, training_config))
    return (initial_value_network, process_network), losses


def evaluate_global_bsde_policy(
    networks: tuple[MeanFieldInitialValue, MeanFieldProcess],
    initial_states: Tensor,
    config: SystemicRiskConfig,
) -> Tensor:
    initial_value_network, _ = networks

    class InducedPolicy(torch.nn.Module):
        def forward(self, time: Tensor | float, states: Tensor):
            del time
            adjoint = initial_value_network(states)
            return type(
                "PolicyOutputLike",
                (),
                {"actions": adjoint_induced_control(states, adjoint, config)},
            )()

    policy = InducedPolicy()
    simulation = simulate_systemic_risk(policy, initial_states, config)
    return config.dt * simulation.running_costs.mean(dim=(1, 2, 3)).sum() + simulation.terminal_cost.mean()