"""Measure cold/warm native resets and replay bowl grasp actions on CPU."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import abc_sim
import mujoco
import numpy as np
from abc_sim.task_eval.contacts import subtree_geom_ids


def measured(owner, name, report, label=None):
    original = getattr(owner, name)
    key = name if label is None else label

    def call(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        finally:
            report[key]["seconds"] += time.perf_counter() - start
            report[key]["calls"] += 1
        if isinstance(result, (bool, np.bool_)) and not result:
            report[key]["rejections"] += 1
        return result

    return patch.object(owner, name, call)


def bowl_sample(env):
    model, data = env.model, env.data
    bowl = model.body("source_bowl").id
    geoms = subtree_geom_ids(model, bowl)
    force = np.zeros(6)
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first not in geoms and second not in geoms:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        contacts.append(
            {
                "geoms": [model.geom(first).name, model.geom(second).name],
                "normal_force_n": float(force[0]),
                "distance_m": float(contact.dist),
                "effective_friction": contact.friction.tolist(),
            }
        )
    grippers = {}
    for arm in ("left", "right"):
        finger_positions, joint_positions = [], []
        for finger in ("left", "right"):
            joint = model.joint(f"{arm}_{finger}_finger").id
            joint_positions.append(float(data.qpos[model.jnt_qposadr[joint]]))
            finger_positions.append(data.xpos[model.jnt_bodyid[joint]].copy())
        actuator = model.actuator(f"{arm}_gripper").id
        grippers[arm] = {
            "command": float(data.ctrl[actuator]),
            "joint_positions": joint_positions,
            "finger_body_separation_m": float(
                np.linalg.norm(finger_positions[0] - finger_positions[1])
            ),
        }
    result = env.evaluate_task()
    return {
        "time_s": float(data.time),
        "source_position": data.xpos[bowl].tolist(),
        "source_tilt_deg": float(
            np.rad2deg(np.arccos(np.clip(data.xmat[bowl, 8], -1, 1)))
        ),
        "grippers": grippers,
        "contacts": contacts,
        "finite": bool(np.isfinite(data.qpos).all()),
        "task_eval": result.to_info(squeeze=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="lift_bottles_upright")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--scene-xml", type=Path)
    parser.add_argument(
        "--actions",
        type=Path,
        help="NPZ with actions shaped (T,14), for the bowl task.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    env = abc_sim.make_env(
        args.task,
        render_cameras=args.render,
        scene_xml=args.scene_xml,
        chunk_dim=1,
        terminate_on_success=False,
        terminate_on_failure=False,
    )
    report = {
        "task": args.task,
        "construction_seconds": time.perf_counter() - start,
        "resets": [],
    }
    pinned = None
    try:
        for offset, mode in enumerate(
            ("cold", "warm_same_identity", "warm_changed_identity")
        ):
            timing = defaultdict(lambda: {"seconds": 0.0, "calls": 0, "rejections": 0})
            randomizer = env._task_randomizer
            with ExitStack() as stack:
                for name in (
                    "_compile_scene_xml",
                    "_bind_model",
                    "_close_camera_backend",
                    "_create_camera_provider",
                ):
                    stack.enter_context(measured(env, name, timing))
                if args.task == "lift_bottles_upright":
                    from abc_sim.randomization.tasks import bottles_upright

                    for name in ("_reload_scene", "_pairwise_ok", "_contacts_ok"):
                        stack.enter_context(measured(randomizer, name, timing))
                    for name in (
                        "_water_bottle_flat_compiled_metadata",
                        "distractor_half_extents",
                    ):
                        stack.enter_context(measured(bottles_upright, name, timing))
                start = time.perf_counter()
                _, info = env.reset(
                    seed=args.seed + offset,
                    options={
                        "randomization": pinned
                        if mode == "warm_same_identity"
                        else None
                    },
                )
                total = time.perf_counter() - start
                state = info["randomization"]
                if offset == 0 and args.task == "lift_bottles_upright":
                    pinned = {
                        key: state.metadata[key]
                        for key in ("bottle_variants", "distractors")
                    }
                frame_seconds = None
                if args.render:
                    start = time.perf_counter()
                    env._render_cameras()
                    frame_seconds = time.perf_counter() - start
                report["resets"].append(
                    {
                        "mode": mode,
                        "seed": args.seed + offset,
                        "total_seconds": total,
                        "first_extra_frame_seconds": frame_seconds,
                        "stages": dict(timing),
                        "metadata": state.metadata,
                    }
                )
        if args.actions is not None:
            if args.task != "pack_and_pour_fruit":
                raise ValueError(
                    "Bowl contact diagnostics require pack_and_pour_fruit."
                )
            with np.load(args.actions) as archive:
                actions = archive["actions"]
            if actions.ndim != 2 or actions.shape[1] != 14:
                raise ValueError("Saved native joint actions must have shape (T,14).")
            report["bowl_trace"] = [bowl_sample(env)]
            for action in actions:
                env.step(action, render_obs=False)
                report["bowl_trace"].append(bowl_sample(env))
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
