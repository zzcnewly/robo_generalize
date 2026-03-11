"""
DreamZero evaluation config for MolmoSpaces benchmarks.

This file defines the eval config that plugs DreamZero into the MolmoSpaces
evaluation harness. It uses the same WebSocket protocol as DreamZero's
socket_test_optimized_AR.py server.

Usage (programmatic):
    from experiments.molmospaces.eval_config import DreamZeroBenchmarkEvalConfig
    from molmo_spaces.evaluation import run_evaluation

    results = run_evaluation(
        eval_config_cls=DreamZeroBenchmarkEvalConfig,
        benchmark_dir="/path/to/benchmark",
        task_horizon_steps=500,
    )

Usage (CLI):
    python -m molmo_spaces.evaluation.eval_main \
        experiments.molmospaces.eval_config:DreamZeroBenchmarkEvalConfig \
        --benchmark_dir /path/to/benchmark \
        --task_horizon_steps 500 \
        --no_wandb
"""

from __future__ import annotations

import os

from molmo_spaces.configs.policy_configs_baselines import DreamZeroPolicyConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig


class DreamZeroBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    """DreamZero evaluation config for MolmoSpaces JSON benchmarks.

    The DreamZero server must be running separately (in the dreamzero conda env)
    before starting evaluation. Configure host/port via environment variables:
        DREAMZERO_HOST (default: localhost)
        DREAMZERO_PORT (default: 5000)
    """

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: DreamZeroPolicyConfig = DreamZeroPolicyConfig(
        remote_config=dict(
            host=os.environ.get("DREAMZERO_HOST", "localhost"),
            port=int(os.environ.get("DREAMZERO_PORT", "8000")),
        ),
        chunk_size=24,
        grasping_type="binary",
        grasping_threshold=0.5,
    )
    policy_dt_ms: float = 66.0

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
