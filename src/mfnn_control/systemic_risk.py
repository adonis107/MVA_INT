from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import SystemicRiskConfig


CASE_SPECS: dict[str, tuple[str, tuple[float, ...]]] = {
    "case_1": ("normal", (0.0, 0.2)),
    "case_2": ("normal", (0.3, 0.05)),
    "case_3": ("normal", (0.0, 0.05)),
    "case_4": ("two_point_mixture", (-3.0**0.5 / 10.0, 3.0**0.5 / 10.0, 0.1)),
    "case_5": ("two_point_mixture", (-0.25, -0.25, 0.1)),
    "case_6": ("three_point_mixture", (-0.3, 0.3, 0.07)),
}

CASE_SPECS_2D: dict[str, tuple[str, tuple[tuple[float, float], ...] | tuple[tuple[float, float], float] | tuple[tuple[float, float], tuple[float, float], float]]] = {
    "case_1": ("normal_2d", ((0.0, 0.0), 0.2)),
    "case_2": ("normal_2d", ((0.3, 0.3), 0.05)),
    "case_3": ("normal_2d", ((0.0, 0.0), 0.05)),
    "case_4": ("two_point_mixture_2d", ((-3.0**0.5 / 10.0, -3.0**0.5 / 10.0), (3.0**0.5 / 10.0, 3.0**0.5 / 10.0), 0.1)),
    "case_5": ("two_point_mixture_2d", ((-0.25, -0.25), (-0.25, -0.25), 0.1)),
    "case_6": ("three_point_mixture_2d", ((-0.3, -0.3), (0.3, 0.3), 0.07)),
}


@dataclass(frozen=True)
class SimulationResult:
    states: Tensor
    actions: Tensor
    running_costs: Tensor
    terminal_cost: Tensor


def sample_initial_states(
    case: str,
    batch_size: int,
    particles: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return sample_initial_states_with_dim(case, batch_size, particles, 1, device=device, dtype=dtype)


def sample_initial_states_with_dim(
    case: str,
    batch_size: int,
    particles: int,
    state_dim: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if state_dim == 2:
        if case not in CASE_SPECS_2D:
            raise ValueError(f"Unsupported 2D case: {case}")
        family_2d, params_2d = CASE_SPECS_2D[case]
        if family_2d == "normal_2d":
            mean_tuple, std = params_2d
            mean = torch.tensor(mean_tuple, device=device, dtype=dtype).view(1, 1, 2)
            base = torch.randn(batch_size, particles, 2, device=device, dtype=dtype)
            return mean + std * base
        if family_2d == "two_point_mixture_2d":
            left_mean, right_mean, std = params_2d
            selector = torch.rand(batch_size, particles, 1, device=device, dtype=dtype) > 0.5
            left_vec = torch.tensor(left_mean, device=device, dtype=dtype).view(1, 1, 2)
            right_vec = torch.tensor(right_mean, device=device, dtype=dtype).view(1, 1, 2)
            left = left_vec + std * torch.randn(batch_size, particles, 2, device=device, dtype=dtype)
            right = right_vec + std * torch.randn(batch_size, particles, 2, device=device, dtype=dtype)
            return torch.where(selector, right, left)
        left_mean, right_mean, std = params_2d
        categories = torch.randint(0, 3, (batch_size, particles, 1), device=device)
        left_vec = torch.tensor(left_mean, device=device, dtype=dtype).view(1, 1, 2)
        right_vec = torch.tensor(right_mean, device=device, dtype=dtype).view(1, 1, 2)
        means = torch.zeros(batch_size, particles, 2, device=device, dtype=dtype)
        means = torch.where(categories == 0, left_vec.expand_as(means), means)
        means = torch.where(categories == 1, right_vec.expand_as(means), means)
        return means + std * torch.randn(batch_size, particles, 2, device=device, dtype=dtype)

    family, params = CASE_SPECS[case]
    if family == "normal":
        mean, std = params
        base = torch.randn(batch_size, particles, state_dim, device=device, dtype=dtype)
        return mean + std * base
    if family == "two_point_mixture":
        left_mean, right_mean, std = params
        selector = torch.rand(batch_size, particles, 1, device=device, dtype=dtype) > 0.5
        left = left_mean + std * torch.randn(batch_size, particles, state_dim, device=device, dtype=dtype)
        right = right_mean + std * torch.randn(batch_size, particles, state_dim, device=device, dtype=dtype)
        return torch.where(selector, right, left)
    left_mean, right_mean, std = params
    categories = torch.randint(0, 3, (batch_size, particles, 1), device=device)
    means = torch.zeros(batch_size, particles, state_dim, device=device, dtype=dtype)
    means = torch.where(categories == 0, torch.full_like(means, left_mean), means)
    means = torch.where(categories == 1, torch.full_like(means, right_mean), means)
    return means + std * torch.randn(batch_size, particles, state_dim, device=device, dtype=dtype)


def sample_initial_state_batch(
    cases: tuple[str, ...],
    batch_size: int,
    particles: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, list[str]]:
    selected_cases = [cases[index % len(cases)] for index in range(batch_size)]
    states = [
        sample_initial_states_with_dim(case, 1, particles, 1, device=device, dtype=dtype)
        for case in selected_cases
    ]
    return torch.cat(states, dim=0), selected_cases


def sample_initial_state_batch_with_dim(
    cases: tuple[str, ...],
    batch_size: int,
    particles: int,
    state_dim: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, list[str]]:
    selected_cases = [cases[index % len(cases)] for index in range(batch_size)]
    states = [
        sample_initial_states_with_dim(case, 1, particles, state_dim, device=device, dtype=dtype)
        for case in selected_cases
    ]
    return torch.cat(states, dim=0), selected_cases


def case_variance(case: str) -> float:
    family, params = CASE_SPECS[case]
    if family == "normal":
        _, std = params
        return std**2
    if family == "two_point_mixture":
        left_mean, right_mean, std = params
        mean = 0.5 * (left_mean + right_mean)
        return std**2 + 0.5 * ((left_mean - mean) ** 2 + (right_mean - mean) ** 2)
    left_mean, right_mean, std = params
    means = (left_mean, 0.0, right_mean)
    mean = sum(means) / 3.0
    return std**2 + sum((value - mean) ** 2 for value in means) / 3.0


def case_variance_2d(case: str) -> float:
    """Per-component variance for the 2D case, summed over both components."""
    family_2d, params_2d = CASE_SPECS_2D[case]
    if family_2d == "normal_2d":
        _, std = params_2d
        return 2.0 * std**2
    if family_2d == "two_point_mixture_2d":
        left_mean, right_mean, std = params_2d
        total = 0.0
        for dim in range(2):
            lm, rm = left_mean[dim], right_mean[dim]
            mean = 0.5 * (lm + rm)
            total += std**2 + 0.5 * ((lm - mean) ** 2 + (rm - mean) ** 2)
        return total
    
    left_mean, right_mean, std = params_2d
    total = 0.0
    for dim in range(2):
        lm, rm = left_mean[dim], right_mean[dim]
        means_per_dim = (lm, 0.0, rm)
        mean = sum(means_per_dim) / 3.0
        total += std**2 + sum((v - mean) ** 2 for v in means_per_dim) / 3.0
    return total


def solve_riccati(config: SystemicRiskConfig, *, grid_size: int = 4096) -> tuple[Tensor, Tensor]:
    dt = config.horizon / grid_size
    times = torch.linspace(config.horizon, 0.0, grid_size + 1)
    q_values = torch.empty(grid_size + 1, dtype=torch.float64)
    q_values[0] = 0.5 * config.c
    linear_term = config.kappa + config.q
    source = 0.5 * (config.eta - config.q**2)
    for index in range(grid_size):
        current_q = q_values[index]
        derivative = -(2.0 * current_q.square() + 2.0 * linear_term * current_q - source)
        q_values[index + 1] = current_q - dt * derivative
    return times.flip(0), q_values.flip(0)


def optimal_feedback_coefficient(time: Tensor | float, config: SystemicRiskConfig) -> Tensor:
    times, q_values = solve_riccati(config)
    time_tensor = torch.as_tensor(time, dtype=torch.float64)
    clipped = time_tensor.clamp(float(times[0]), float(times[-1]))
    interpolated = torch.interp(clipped, times, q_values)
    return 2.0 * interpolated.to(dtype=torch.float32) + config.q


def systemic_risk_value(case: str, config: SystemicRiskConfig) -> float:
    times, q_values = solve_riccati(config)
    dt = times[1] - times[0]
    integral = torch.trapz(q_values, dx=dt)
    if config.state_dim == 2:
        variance = case_variance_2d(case)
    else:
        variance = case_variance(case)
    scalar_value = q_values[0] * variance + config.state_dim * config.sigma**2 * integral
    return float(scalar_value)


def systemic_risk_running_cost(states: Tensor, actions: Tensor, config: SystemicRiskConfig) -> Tensor:
    mean_state = states.mean(dim=1, keepdim=True)
    gap = mean_state - states
    return 0.5 * actions.square() - config.q * actions * gap + 0.5 * config.eta * gap.square()


def systemic_risk_terminal_cost(states: Tensor, config: SystemicRiskConfig) -> Tensor:
    terminal_mean = states.mean(dim=1, keepdim=True)
    return 0.5 * config.c * (states - terminal_mean).square()


def systemic_risk_adjoint_terminal(states: Tensor, config: SystemicRiskConfig) -> Tensor:
    mean_state = states.mean(dim=1, keepdim=True)
    return config.c * (states - mean_state)


def systemic_risk_adjoint_drift(states: Tensor, adjoint: Tensor, config: SystemicRiskConfig) -> Tensor:
    mean_state = states.mean(dim=1, keepdim=True)
    return (config.kappa + config.q) * (mean_state - states) - adjoint


def systemic_risk_bsde_driver(states: Tensor, adjoint: Tensor, config: SystemicRiskConfig) -> Tensor:
    mean_state = states.mean(dim=1, keepdim=True)
    mean_adjoint = adjoint.mean(dim=1, keepdim=True)
    return -(config.kappa + config.q) * (mean_adjoint - adjoint) + (config.eta - config.q**2) * (mean_state - states)


def simulate_systemic_risk(
    policy: nn.Module,
    initial_states: Tensor,
    config: SystemicRiskConfig,
    *,
    noise: Tensor | None = None,
) -> SimulationResult:
    dt = config.dt
    sqrt_dt = dt**0.5
    current_state = initial_states
    trajectories = [current_state]
    actions_list = []
    running_costs = []
    if noise is None:
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
        policy_output = policy(time, current_state)
        action = policy_output.actions
        mean_state = current_state.mean(dim=1, keepdim=True)
        gap = mean_state - current_state
        drift = config.kappa * gap + action
        next_state = current_state + dt * drift + config.sigma * sqrt_dt * noise[step]
        running = systemic_risk_running_cost(current_state, action, config)
        trajectories.append(next_state)
        actions_list.append(action)
        running_costs.append(running)
        current_state = next_state
    terminal_cost = systemic_risk_terminal_cost(current_state, config)
    return SimulationResult(
        states=torch.stack(trajectories, dim=0),
        actions=torch.stack(actions_list, dim=0),
        running_costs=torch.stack(running_costs, dim=0),
        terminal_cost=terminal_cost,
    )


def policy_loss(policy: nn.Module, initial_states: Tensor, config: SystemicRiskConfig) -> Tensor:
    simulation = simulate_systemic_risk(policy, initial_states, config)
    running_term = config.dt * simulation.running_costs.mean(dim=(1, 2, 3)).sum()
    terminal_term = simulation.terminal_cost.mean()
    return running_term + terminal_term


def adjoint_induced_control(states: Tensor, adjoint: Tensor, config: SystemicRiskConfig) -> Tensor:
    mean_state = states.mean(dim=1, keepdim=True)
    return config.q * (mean_state - states) - adjoint


def uncontrolled_default_rate(
    config: SystemicRiskConfig,
    *,
    case: str,
    interaction_q: float,
    default_threshold: float,
    mc_paths: int,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    initial_states = sample_initial_states_with_dim(
        case,
        mc_paths,
        config.particles,
        config.state_dim,
        device=config.device,
        dtype=getattr(torch, config.dtype),
    )
    dt = config.dt
    sqrt_dt = dt**0.5
    current_state = initial_states
    defaulted = (current_state < default_threshold).any(dim=2)
    for _ in range(config.steps):
        noise = torch.randn(
            *current_state.shape,
            device=current_state.device,
            dtype=current_state.dtype,
        )
        mean_state = current_state.mean(dim=1, keepdim=True)
        drift = interaction_q * (mean_state - current_state)
        current_state = current_state + dt * drift + config.sigma * sqrt_dt * noise
        defaulted = defaulted | (current_state < default_threshold).any(dim=2)
    return float(defaulted.float().mean().item())


def estimate_critical_q(
    config: SystemicRiskConfig,
    *,
    case: str,
    q_min: float,
    q_max: float,
    q_steps: int,
    default_threshold: float,
    target_default_rate: float,
    mc_paths: int,
    seed: int,
) -> dict[str, object]:
    q_values = torch.linspace(q_min, q_max, q_steps, dtype=torch.float64)
    sweep: list[dict[str, float]] = []
    critical_q: float | None = None
    for idx, q in enumerate(q_values.tolist()):
        rate = uncontrolled_default_rate(
            config,
            case=case,
            interaction_q=float(q),
            default_threshold=default_threshold,
            mc_paths=mc_paths,
            seed=seed + idx,
        )
        sweep.append({"q": float(q), "default_rate": float(rate)})
        if critical_q is None and rate >= target_default_rate:
            critical_q = float(q)
    return {
        "critical_q": critical_q,
        "target_default_rate": float(target_default_rate),
        "default_threshold": float(default_threshold),
        "sweep": sweep,
    }