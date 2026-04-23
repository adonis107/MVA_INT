from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from mfnn_control import (
    EncoderConfig,
    SystemicRiskConfig,
    Table28PrepProfile,
    TrainingConfig,
    estimate_critical_q,
    evaluate_algorithm_6_policy,
    sample_initial_states_with_dim,
    systemic_risk_value,
    table28_prep_output_schema,
    train_pham_warin_algorithm_1,
    train_pham_warin_algorithm_6,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("1", "6"), default="1")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--encoder", choices=("cylindrical", "bins"), default="cylindrical")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--state-dim", type=int, default=1)
    parser.add_argument("--case", default="case_1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--save-prefix", default="systemic_risk")
    parser.add_argument("--value-tolerance", type=float, default=0.05)
    parser.add_argument("--estimate-critical-q", action="store_true")
    parser.add_argument("--q-min", type=float, default=0.0)
    parser.add_argument("--q-max", type=float, default=2.0)
    parser.add_argument("--q-steps", type=int, default=21)
    parser.add_argument("--default-threshold", type=float, default=-0.5)
    parser.add_argument("--target-default-rate", type=float, default=0.5)
    parser.add_argument("--mc-paths", type=int, default=256)
    parser.add_argument("--table28-scaffold", action="store_true")
    parser.add_argument("--table28-profile", action="store_true")
    return parser.parse_args()


def parse_seeds(seed_arg: int, seeds_arg: str) -> list[int]:
    if not seeds_arg.strip():
        return [seed_arg]
    parsed = [int(item.strip()) for item in seeds_arg.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--seeds was provided but no valid integer seed was parsed")
    return parsed


def _prepare_memory_tracking(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
        return
    tracemalloc.start()


def _collect_peak_memory_mb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(peak / (1024.0 * 1024.0))


def save_checkpoint(
    algorithm: str,
    seed: int,
    save_dir: str,
    save_prefix: str,
    config: SystemicRiskConfig,
    encoder_config: EncoderConfig,
    training_config: TrainingConfig,
    trained,
) -> str:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{save_prefix}_alg{algorithm}_{encoder_config.kind}_{training_config.initial_case}_seed{seed}.pt"
    payload = {
        "algorithm": algorithm,
        "seed": seed,
        "config": asdict(config),
        "encoder_config": asdict(encoder_config),
        "training_config": asdict(training_config),
    }
    if algorithm == "1":
        payload["policy_state_dict"] = trained.state_dict()
    else:
        initial_value_network, process_network = trained
        payload["initial_value_state_dict"] = initial_value_network.state_dict()
        payload["process_state_dict"] = process_network.state_dict()
    torch.save(payload, path)
    return str(path)


def run_single_experiment(
    algorithm: str,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float | str | bool]:
    config = SystemicRiskConfig(
        device=args.device,
        particles=args.particles,
        steps=args.steps,
        state_dim=args.state_dim,
    )
    encoder_config = EncoderConfig(kind=args.encoder, state_dim=args.state_dim)
    training_config = TrainingConfig(
        iterations=args.iterations,
        batch_size=args.batch_size,
        initial_case=args.case,
        training_cases=(args.case,),
        seed=seed,
    )
    analytic_value = systemic_risk_value(args.case, config)
    _prepare_memory_tracking(args.device)
    start = time.perf_counter()
    checkpoint_path = ""
    if algorithm == "1":
        trained, losses = train_pham_warin_algorithm_1(config, encoder_config, training_config)
        objective_name = "global_control_cost"
        estimated_value = float(losses[-1])
        induced_cost = None
    else:
        trained, losses = train_pham_warin_algorithm_6(config, encoder_config, training_config)
        initial_states = sample_initial_states_with_dim(
            args.case,
            training_config.batch_size,
            config.particles,
            config.state_dim,
            device=config.device,
            dtype=getattr(torch, config.dtype),
        )
        induced = evaluate_algorithm_6_policy(trained, initial_states, config)
        induced_cost = float(induced.detach().cpu())
        estimated_value = induced_cost
        objective_name = "terminal_adjoint_mse"
    elapsed_s = time.perf_counter() - start
    peak_memory_mb = _collect_peak_memory_mb(args.device)
    if args.save_dir:
        checkpoint_path = save_checkpoint(
            algorithm,
            seed,
            args.save_dir,
            args.save_prefix,
            config,
            encoder_config,
            training_config,
            trained,
        )

    absolute_error = abs(estimated_value - analytic_value)
    result: dict[str, float | str | bool] = {
        "algorithm": algorithm,
        "encoder": args.encoder,
        "case": args.case,
        "seed": seed,
        "objective": objective_name,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "estimated_value": float(estimated_value),
        "analytic_value": float(analytic_value),
        "absolute_value_error": float(absolute_error),
        "within_tolerance": bool(absolute_error <= args.value_tolerance),
        "wall_time_s": float(elapsed_s),
        "peak_memory_mb": float(peak_memory_mb),
    }
    if induced_cost is not None:
        result["induced_policy_cost"] = float(induced_cost)
    if checkpoint_path:
        result["checkpoint_path"] = checkpoint_path
    return result


def summarize_runs(runs: list[dict[str, float | str | bool]]) -> dict[str, float | int]:
    final_losses = [float(run["final_loss"]) for run in runs]
    value_errors = [float(run["absolute_value_error"]) for run in runs]
    times = [float(run["wall_time_s"]) for run in runs]
    memories = [float(run["peak_memory_mb"]) for run in runs]
    successes = [bool(run["within_tolerance"]) for run in runs]
    return {
        "runs": len(runs),
        "mean_final_loss": float(sum(final_losses) / len(final_losses)),
        "mean_absolute_value_error": float(sum(value_errors) / len(value_errors)),
        "mean_wall_time_s": float(sum(times) / len(times)),
        "mean_peak_memory_mb": float(sum(memories) / len(memories)),
        "success_rate": float(sum(1 for flag in successes if flag) / len(successes)),
    }


def run_algorithm_over_seeds(algorithm: str, args: argparse.Namespace, seeds: list[int]) -> dict[str, object]:
    runs = [run_single_experiment(algorithm, args, seed) for seed in seeds]
    return {
        "algorithm": algorithm,
        "runs": runs,
        "aggregate": summarize_runs(runs),
    }


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seed, args.seeds)

    if args.table28_profile or args.table28_scaffold:
        profile = Table28PrepProfile(
            state_dim=2,
            particles=args.particles,
            steps=args.steps,
            iterations=args.iterations,
            batch_size=args.batch_size,
            seeds=tuple(seeds),
        )
        config = SystemicRiskConfig(
            device=args.device,
            particles=profile.particles,
            steps=profile.steps,
            state_dim=profile.state_dim,
        )
        training_template = TrainingConfig(
            iterations=profile.iterations,
            batch_size=profile.batch_size,
            initial_case=profile.cases[0],
            training_cases=profile.cases,
            seed=profile.seeds[0],
        )
        output = {
            "table28_prep": True,
            "profile": asdict(profile),
            "systemic_risk_config": asdict(config),
            "training_template": asdict(training_template),
            "run_grid": {
                "algorithms": list(profile.algorithms),
                "encoders": list(profile.encoders),
                "cases": list(profile.cases),
                "seeds": list(profile.seeds),
            },
            "output_schema": table28_prep_output_schema(),
        }
        print(json.dumps(output, indent=2))
        return

    if args.estimate_critical_q:
        config = SystemicRiskConfig(
            device=args.device,
            particles=args.particles,
            steps=args.steps,
            state_dim=args.state_dim,
        )
        output = {
            "critical_q_estimation": True,
            "case": args.case,
            "state_dim": args.state_dim,
            "particles": args.particles,
            "steps": args.steps,
            "mc_paths": args.mc_paths,
            "q_min": args.q_min,
            "q_max": args.q_max,
            "q_steps": args.q_steps,
            "seed": args.seed,
            "result": estimate_critical_q(
                config,
                case=args.case,
                q_min=args.q_min,
                q_max=args.q_max,
                q_steps=args.q_steps,
                default_threshold=args.default_threshold,
                target_default_rate=args.target_default_rate,
                mc_paths=args.mc_paths,
                seed=args.seed,
            ),
        }
        print(json.dumps(output, indent=2))
        return

    if args.benchmark:
        output = {
            "benchmark": True,
            "encoder": args.encoder,
            "case": args.case,
            "device": args.device,
            "state_dim": args.state_dim,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "particles": args.particles,
            "steps": args.steps,
            "seeds": seeds,
            "algorithm_1": run_algorithm_over_seeds("1", args, seeds),
            "algorithm_6": run_algorithm_over_seeds("6", args, seeds),
            "value_tolerance": args.value_tolerance,
        }
        print(json.dumps(output, indent=2))
        return

    output = {
        "benchmark": False,
        "encoder": args.encoder,
        "case": args.case,
        "device": args.device,
        "state_dim": args.state_dim,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "particles": args.particles,
        "steps": args.steps,
        "seeds": seeds,
        "value_tolerance": args.value_tolerance,
        "result": run_algorithm_over_seeds(args.algorithm, args, seeds),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()