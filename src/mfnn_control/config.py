from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemicRiskConfig:
    horizon: float = 0.2
    steps: int = 20
    particles: int = 128
    kappa: float = 0.6
    sigma: float = 1.0
    q: float = 0.8
    eta: float = 2.0
    c: float = 2.0
    state_dim: int = 1
    device: str = "cpu"
    dtype: str = "float32"

    @property
    def dt(self) -> float:
        return self.horizon / self.steps


@dataclass(frozen=True)
class EncoderConfig:
    kind: str = "cylindrical"
    state_dim: int = 1
    bins: int = 33
    support_min: float = -2.0
    support_max: float = 2.0
    bandwidth: float = 0.15
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (64, 64)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    iterations: int = 200
    learning_rate: float = 1e-3
    initial_case: str = "case_1"
    seed: int = 7
    hidden_dims: tuple[int, ...] = field(default_factory=lambda: (64, 64))
    training_cases: tuple[str, ...] = field(default_factory=lambda: ("case_1",))


@dataclass(frozen=True)
class PhamWarinBenchmarkProfile:
    profile_name: str = "pham_warin_2d_benchmark"
    state_dim: int = 2
    cases: tuple[str, ...] = ("case_1", "case_2", "case_3", "case_4", "case_5", "case_6")
    encoders: tuple[str, ...] = ("bins", "cylindrical")
    algorithms: tuple[str, ...] = ("global_dp", "global_bsde")
    seeds: tuple[int, ...] = (7, 13, 21)
    particles: int = 256
    steps: int = 20
    iterations: int = 400
    batch_size: int = 16


def pham_warin_benchmark_output_schema() -> dict[str, object]:
    return {
        "profile": {
            "profile_name": "string",
            "state_dim": "int",
            "cases": ["string"],
            "encoders": ["string"],
            "algorithms": ["string"],
            "seeds": ["int"],
            "particles": "int",
            "steps": "int",
            "iterations": "int",
            "batch_size": "int",
        },
        "systemic_risk_config": {
            "horizon": "float",
            "steps": "int",
            "particles": "int",
            "kappa": "float",
            "sigma": "float",
            "q": "float",
            "eta": "float",
            "c": "float",
            "state_dim": "int",
            "device": "string",
            "dtype": "string",
        },
        "training_template": {
            "iterations": "int",
            "batch_size": "int",
            "learning_rate": "float",
            "initial_case": "string",
            "seed": "int",
            "training_cases": ["string"],
            "hidden_dims": ["int"],
        },
        "run_grid": {
            "algorithms": ["string"],
            "encoders": ["string"],
            "cases": ["string"],
            "seeds": ["int"],
        },
    }