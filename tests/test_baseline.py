from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mfnn_control import (
    EncoderConfig,
    SystemicRiskConfig,
    Table28PrepProfile,
    TrainingConfig,
    build_algorithm_6_networks,
    build_policy,
    case_variance,
    estimate_critical_q,
    run_algorithm_6_step,
    run_training_step,
    sample_initial_states_with_dim,
    table28_prep_output_schema,
)
from mfnn_control.encoders import BinDensityEncoder, CylindricalEncoder


class EncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.states = torch.randn(4, 32, 1)
        self.permuted = self.states[:, torch.randperm(self.states.shape[1]), :]

    def test_bin_density_encoder_is_permutation_invariant(self) -> None:
        encoder = BinDensityEncoder(EncoderConfig(kind="bins"))
        original = encoder(self.states)
        permuted = encoder(self.permuted)
        self.assertTrue(torch.allclose(original, permuted, atol=1e-6, rtol=1e-6))

    def test_cylindrical_encoder_is_permutation_invariant(self) -> None:
        encoder = CylindricalEncoder(EncoderConfig(kind="cylindrical"))
        original = encoder(self.states)
        permuted = encoder(self.permuted)
        self.assertTrue(torch.allclose(original, permuted, atol=1e-6, rtol=1e-6))

    def test_bin_density_encoder_2d_is_permutation_invariant(self) -> None:
        states = torch.randn(4, 32, 2)
        permuted = states[:, torch.randperm(states.shape[1]), :]
        encoder = BinDensityEncoder(EncoderConfig(kind="bins", state_dim=2, bins=8))
        original = encoder(states)
        permuted_out = encoder(permuted)
        self.assertTrue(torch.allclose(original, permuted_out, atol=1e-6, rtol=1e-6))


class TrainingTests(unittest.TestCase):
    def test_paper_case_variances_match_expected_equalities(self) -> None:
        self.assertAlmostEqual(case_variance("case_1"), case_variance("case_4"), places=6)
        self.assertAlmostEqual(case_variance("case_2"), case_variance("case_3"), places=6)

    def test_single_training_step_returns_finite_loss(self) -> None:
        config = SystemicRiskConfig(steps=6, particles=24)
        training_config = TrainingConfig(iterations=1, batch_size=4)
        policy = build_policy(EncoderConfig(kind="cylindrical"), training_config).to(dtype=torch.float32)
        optimizer = torch.optim.Adam(policy.parameters(), lr=training_config.learning_rate)
        loss = run_training_step(policy, optimizer, config, training_config)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreater(loss, 0.0)

    def test_algorithm_6_step_returns_finite_loss(self) -> None:
        config = SystemicRiskConfig(steps=6, particles=24)
        training_config = TrainingConfig(iterations=1, batch_size=4)
        initial_value_network, process_network = build_algorithm_6_networks(EncoderConfig(kind="cylindrical"), training_config)
        optimizer = torch.optim.Adam(
            list(initial_value_network.parameters()) + list(process_network.parameters()),
            lr=training_config.learning_rate,
        )
        loss = run_algorithm_6_step(initial_value_network, process_network, optimizer, config, training_config)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreaterEqual(loss, 0.0)

    def test_sample_initial_states_with_dim_returns_2d_shape(self) -> None:
        states = sample_initial_states_with_dim("case_1", 3, 7, 2, device="cpu", dtype=torch.float32)
        self.assertEqual(states.shape, (3, 7, 2))

    def test_estimate_critical_q_returns_valid_structure(self) -> None:
        config = SystemicRiskConfig(steps=4, particles=8, state_dim=1)
        result = estimate_critical_q(
            config,
            case="case_1",
            q_min=0.0,
            q_max=0.4,
            q_steps=3,
            default_threshold=-0.3,
            target_default_rate=0.2,
            mc_paths=16,
            seed=3,
        )
        self.assertIn("sweep", result)
        self.assertEqual(len(result["sweep"]), 3)

    def test_strict_2d_case_sampling_shape(self) -> None:
        states = sample_initial_states_with_dim("case_4", 2, 10, 2, device="cpu", dtype=torch.float32)
        self.assertEqual(states.shape, (2, 10, 2))

    def test_table28_profile_and_schema(self) -> None:
        profile = Table28PrepProfile()
        schema = table28_prep_output_schema()
        self.assertEqual(profile.state_dim, 2)
        self.assertIn("profile", schema)
        self.assertIn("run_grid", schema)


if __name__ == "__main__":
    unittest.main()