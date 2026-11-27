"""Prompt delivery for native one-shot interrupted instructions."""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from abc_sim.task_eval.interrupted import InterruptedPickupEvaluator


class InterruptedInstructionRuntime:
    def bind(self, model, data):
        pass

    def make_progress_source(self, model, spec, evaluator):
        if evaluator is None:
            from abc_sim.task_eval.registry import make_task_evaluator

            evaluator = make_task_evaluator(model, spec)
        if not isinstance(evaluator, InterruptedPickupEvaluator):
            raise ValueError("Interrupted instructions require their native evaluator.")
        return evaluator

    def retained_randomization(self):
        return copy.deepcopy(self._randomization)

    def after_reset(self, model, data, *, randomization=None, episode):
        self._prompt = episode["prompt"]
        self._revised_prompt = episode["interruption"]["prompt"]
        self._switched = False
        # Store the actual settled initial scene, so return-to-origin does not
        # use a mesh's unsettled spawn center. Replay restores these same poses.
        if not randomization.metadata.get("interruption_initial_settled", False):
            start_time = float(data.time)
            reset_qpos = data.qpos.copy()
            first = model.body(episode["interruption"]["first_body"]).id
            joint = int(model.body_jntadr[first])
            dof = int(model.jnt_dofadr[joint])
            stable_since = None
            for _ in range(int(np.ceil(2.0 / model.opt.timestep))):
                mujoco.mj_step(model, data)
                velocity = data.qvel[dof : dof + 6]
                table = model.geom("table_plane").id
                table_supported = any(
                    contact.dist <= 0
                    and (
                        (
                            int(contact.geom1) == table
                            and int(model.geom_bodyid[contact.geom2]) == first
                        )
                        or (
                            int(contact.geom2) == table
                            and int(model.geom_bodyid[contact.geom1]) == first
                        )
                    )
                    for contact in data.contact[: data.ncon]
                )
                stable = (
                    table_supported
                    and np.linalg.norm(velocity[:3]) <= 0.01
                    and np.linalg.norm(velocity[3:]) <= 0.10
                )
                if stable:
                    if stable_since is None:
                        stable_since = float(data.time)
                    if data.time >= stable_since + 0.25:
                        break
                else:
                    stable_since = None
            else:
                raise RuntimeError(
                    "Interrupted task target did not settle before reset."
                )
            data.time = start_time
            settled_poses = {}
            for joint_name, pose in randomization.object_states.items():
                address = int(model.jnt_qposadr[model.joint(joint_name).id])
                pose["pos"] = data.qpos[address : address + 3].tolist()
                pose["quat"] = data.qpos[address + 3 : address + 7].tolist()
                settled_poses[address] = data.qpos[address : address + 7].copy()
            data.qpos[:] = reset_qpos
            for address, values in settled_poses.items():
                data.qpos[address : address + 7] = values
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            randomization.metadata["interruption_initial_settled"] = True
        episode["object_states"] = copy.deepcopy(randomization.object_states)
        self._randomization = copy.deepcopy(randomization)

    def before_step(self, model, data):
        pass

    def after_step(self, model, data):
        pass

    def on_task_evaluation(self, result):
        self._switched = bool(result.metrics["interruption_switched"])

    def observation_prompt(self):
        return self._revised_prompt if self._switched else self._prompt

    def debug_state(self):
        return {
            "interruption_switched": self._switched,
            "display_prompt": self.observation_prompt(),
        }
