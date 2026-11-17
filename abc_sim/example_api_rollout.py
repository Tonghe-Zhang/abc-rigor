#!/usr/bin/env python3
"""Minimal rollout example using the public abc_sim environment API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import abc_sim


def _task_info(env) -> dict:
    result = env.evaluate_task()
    if result is None:
        return {"reward": 0.0, "success": False}
    return result.to_info(squeeze=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="put_bottles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-dim", type=int, default=15)
    parser.add_argument("--single-actions", type=int, default=5)
    parser.add_argument("--camera-height", type=int, default=64)
    parser.add_argument("--camera-width", type=int, default=64)
    parser.add_argument("--render-cameras", action="store_true")
    args = parser.parse_args()

    spec = abc_sim.get_task_spec(args.task)
    env = abc_sim.make_env(
        task=spec.name,
        prompt=spec.prompt,
        chunk_dim=args.chunk_dim,
        render_cameras=args.render_cameras,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )
    try:
        obs, _ = env.reset(seed=args.seed, randomize=True)
        print(f"reset state_shape={obs['state'].shape} prompt={obs['prompt']!r}")

        # Eval/tracing style: step one 14D policy action at a time, then score.
        for step_idx in range(args.single_actions):
            action_14d = obs["state"]
            obs, _reward, _terminated, _truncated, _step_info = env.step(action_14d)
            info = _task_info(env)
            state = env.capture_state()
            print(
                f"single_action={step_idx + 1} "
                f"sim_time={float(state['time']):.3f} "
                f"reward={float(info['reward']):.3f} "
                f"success={bool(info['success'])}"
            )

        # Chunked policy style: pass a full (chunk_dim, 14) action array to step_chunk().
        chunk = np.repeat(obs["state"][None, :], args.chunk_dim, axis=0)
        obs, history, reward, terminated, truncated, info = env.step_chunk(chunk)
        print(
            f"chunk_step state_shape={obs['state'].shape} "
            f"history_shape={history['state'].shape} "
            f"reward={float(reward):.3f} "
            f"success={bool(info.get('task_success', False))} "
            f"terminated={terminated} truncated={truncated}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
