"""Conveyor mechanics with task-owned, exactly-once removal decisions."""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from abc_sim.task_runtime.conveyor_pick import ConveyorPickRuntime


class RLConveyorRuntime(ConveyorPickRuntime):
    """Reuse belt motion while consuming the RL evaluator's removal outcomes."""

    def __init__(self):
        super().__init__()
        self._remove_mask = np.zeros(0, dtype=bool)
        self._episode_randomization = None

    def retained_randomization(self):
        return copy.deepcopy(self._episode_randomization)

    def after_reset(self, model, data, *, randomization=None, episode):
        super().after_reset(model, data, randomization=randomization)
        self._object_controls = None
        self._remove_mask = np.zeros(len(episode["objects"]), dtype=bool)
        self._episode_randomization = copy.deepcopy(randomization)

    def _object_controls_for(self, model):
        if self._object_controls is None:
            controls = []
            for index in range(1, len(self._remove_mask) + 1):
                joint = model.joint(f"conveyor_object_{index}_joint").id
                body = model.body(f"conveyor_object_{index}").id
                geoms = tuple(
                    g
                    for g in range(model.ngeom)
                    if model.geom_bodyid[g] == body
                    and (model.geom_contype[g] or model.geom_conaffinity[g])
                )
                if not geoms:
                    raise ValueError(
                        "Every RL conveyor object needs collision geometry."
                    )
                controls.append(
                    (int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]), geoms)
                )
            self._object_controls = tuple(controls)
        return self._object_controls

    def on_task_evaluation(self, result):
        mask = np.asarray(result.metrics["remove_mask"], dtype=bool)
        if mask.shape != (1, len(self._remove_mask)):
            raise ValueError(
                "RL conveyor runtime and evaluator object counts disagree."
            )
        self._remove_mask = mask[0].copy()

    def after_step(self, model, data):
        removed = set(self._removed_objects)
        for index, (qpos, qvel, _) in enumerate(self._object_controls_for(model), 1):
            if self._remove_mask[index - 1] and index not in removed:
                self._remove_object(data, index, qpos, qvel)
                removed.add(index)
        if len(removed) != len(self._removed_objects):
            self._removed_objects = tuple(sorted(removed))
            mujoco.mj_forward(model, data)
        self._last_active_objects = tuple(
            index for index in self._last_active_objects if index not in removed
        )
