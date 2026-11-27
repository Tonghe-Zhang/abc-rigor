"""Native asset-backed reset/replay qualification for the approved RL variants."""

from __future__ import annotations

import json
from contextlib import nullcontext
from unittest.mock import patch

import abc_sim
import mujoco
import numpy as np
import pytest
from abc_sim.randomization.core import RandomizationState


@pytest.mark.parametrize(
    "task",
    [
        "rl_count_into_opaque_box",
        "rl_grab_specific_object_from_clutter",
        "rl_put_relative",
        "rl_put_markers_in_drawer",
        "rl_load_plates_into_dish_rack",
    ],
)
def test_native_variant_constructs_scores_steps_and_replays_recorded_scene(task):
    env = abc_sim.make_env(task, render_cameras=False, chunk_dim=1)
    try:
        observation, info = env.reset(seed=11)
        state = RandomizationState.from_dict(json.loads(json.dumps(info["randomization"].to_dict())))
        initial_prompt = env.prompt
        initial_qpos = env.data.qpos.copy()
        initial = env.evaluate_task()
        assert np.isfinite(initial.reward).all()
        assert 0 <= initial.scalar_reward() <= 1
        assert not initial.scalar_success()
        assert observation["prompt"] == initial_prompt
        if "interruption" in state.metadata:
            assert state.metadata["interruption_initial_settled"]
            assert env._task_runtime.observation_prompt() == initial_prompt
        # Displace one recorded object to ensure replay actually restores poses.
        joint = next(name for name in state.object_states if model_is_free(env.model, name))
        env.data.joint(joint).qpos[0] += 0.05
        mujoco.mj_forward(env.model, env.data)
        env._task_randomizer.apply(env.model, env.data, state)
        assert env.prompt == initial_prompt
        np.testing.assert_allclose(env.data.qpos, initial_qpos, atol=2e-6)
        assert env._last_randomization.to_dict() == state.to_dict()
        action = np.asarray(env.get_state(), dtype=np.float32)
        cached_observer = task in (
            "rl_grab_specific_object_from_clutter",
            "rl_put_relative",
            "rl_put_markers_in_drawer",
        )
        refresh_check = (
            patch(
                "abc_sim.env.mujoco.mj_fwdPosition",
                side_effect=AssertionError("Solved-step cached scoring must not refresh positions."),
            )
            if cached_observer
            else nullcontext()
        )
        with refresh_check:
            env.step(action, render_obs=False)
        assert np.isfinite(env.evaluate_task().reward).all()
        if cached_observer:
            with patch(
                "abc_sim.task_eval.interrupted.mujoco.mj_contactForce",
                side_effect=AssertionError("Cached score reads must not inspect contacts."),
            ):
                assert env.evaluate_task() is env._task_evaluator._result
    finally:
        env.close()


def model_is_free(model, name):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE


def test_native_tray_initial_free_fall_is_safe_and_precontact_loss_terminates():
    env = abc_sim.make_env("ball_tray_balancing", render_cameras=False, chunk_dim=1)
    try:
        env.reset(seed=11)
        initial = env.evaluate_task()
        assert initial.metrics["inside_tray"][0]
        assert not initial.metrics["ball_dropped"][0]
        assert not initial.metrics["ball_tray_contact"][0]
        assert not initial.metrics["task_terminated"][0]
        # The native reset spawns one millimetre above the tray. Staying within
        # its footprint must remain safe even before any supported observation.
        env.data.time = 1.1
        assert not env.evaluate_task().metrics["task_terminated"][0]
        env.data.joint("ball_tray_ball_joint").qpos[0] += 0.4
        mujoco.mj_forward(env.model, env.data)
        assert not env.evaluate_task().metrics["task_terminated"][0]
        env.data.time = 2.099
        assert not env.evaluate_task().metrics["task_terminated"][0]
        env.data.time = 2.101
        assert env.evaluate_task().metrics["task_terminated"][0]
    finally:
        env.close()


@pytest.mark.parametrize("level", ["top", "middle", "bottom"])
def test_actual_marker_drawers_score_five_markers_at_open_and_closed_poses(level):
    from abc_sim.task_eval.marker_drawer import drawer_interior
    from abc_sim.task_eval.put_relative import PutRelativeEvaluator

    env = abc_sim.make_env("rl_put_markers_in_drawer", render_cameras=False, chunk_dim=1)
    try:
        _, info = env.reset(seed=3)
        state = info["randomization"]
        state.metadata["marker_target_drawer"] = level
        state.metadata["prompt"] = f"Put all the markers in the {level} drawer, then close that drawer."
        evaluator = env._task_evaluator
        evaluator.configure_from_randomization(env.model, state)
        assert len(evaluator._markers) == 5
        drawer = env.model.body(f"link_{level}_drawer_low").id
        lower, upper = drawer_interior(env.model, drawer)
        assert np.all(upper - lower > 0.02)
        local_quat = np.asarray([2**-0.5, 0, 2**-0.5, 0])
        local_rotation = np.empty(9)
        mujoco.mju_quat2Mat(local_rotation, local_quat)
        for opening, expected in ((0.2, 0.90), (0, 1.0)):
            env.data.joint(f"joint_{level}_drawer_low").qpos[0] = opening
            mujoco.mj_forward(env.model, env.data)
            drawer_rotation = env.data.xmat[drawer].reshape(3, 3)
            for index, body in enumerate(evaluator._markers):
                points = PutRelativeEvaluator._body_extent_points(env.model, body) @ local_rotation.reshape(3, 3).T
                midpoint = (points.min(axis=0) + points.max(axis=0)) / 2
                lane = lower[1] + (index + 0.5) * (upper[1] - lower[1]) / 5
                position = np.asarray(
                    [
                        (lower[0] + upper[0]) / 2 - midpoint[0],
                        lane - midpoint[1],
                        lower[2] - points[:, 2].min() - 0.0002,
                    ]
                )
                assert np.ptp(points[:, 1]) < (upper[1] - lower[1]) / 5
                joint = int(env.model.body_jntadr[body])
                address = int(env.model.jnt_qposadr[joint])
                env.data.qpos[address : address + 3] = env.data.xpos[drawer] + drawer_rotation @ position
                mujoco.mju_mulQuat(
                    env.data.qpos[address + 3 : address + 7],
                    env.data.xquat[drawer],
                    local_quat,
                )
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            evaluator.after_physics_step(env.model, env.data)
            result = evaluator.evaluate_model_data(env.model, env.data)
            assert result.scalar_reward() == pytest.approx(expected, abs=1e-6)
            assert result.scalar_success() == (opening == 0)
    finally:
        env.close()
