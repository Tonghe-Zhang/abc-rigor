from __future__ import annotations

from pathlib import Path as _Path
from typing import Any

import mujoco
import numpy as np

from ..assets.inhand import (
    _INHAND_CATEGORIES,
    _INHAND_SCALE_FACTOR_RANGE,
    _OBJ_Z,
    _X_MAX,
    _X_MIN,
    _Y_LEFT_MAX,
    _Y_LEFT_MIN,
    _Y_RIGHT_MAX,
    _Y_RIGHT_MIN,
    _inhand_apply_scene_transforms,
    _inhand_asset_base,
    _inhand_build_xml,
    _inhand_get_variants,
)
from ..core import RandomizationState, SceneRandomizer, _yaw_from_quat

class InHandTransferRandomizer(SceneRandomizer):
    """Randomizer for inhand_transfer: swaps the entire MuJoCo model on each reset.

    Picks a new kitchen tool category/variant and reloads the model from XML so
    the mesh assets change. Call ``bind_env(env)`` after construction.
    """

    perturbations: list = []  # no PerturbRange — we handle placement ourselves

    def __init__(self, *, scene_xml_transform_options: Any = None) -> None:
        super().__init__()
        self._env_ref = None
        self._rng = np.random.default_rng()
        # Pre-cache variant lists so we don't re-scan on every reset.
        self._variants: dict[str, list[_Path]] = {}
        self._scene_xml_transform_options = scene_xml_transform_options

    def bind_env(self, env: Any) -> None:
        self._env_ref = env

    def clone(self) -> "SceneRandomizer":
        cloned = type(self)(
            scene_xml_transform_options=self._scene_xml_transform_options,
        )
        return cloned

    def _get_variants(self, category: str) -> list[_Path]:
        if category not in self._variants:
            self._variants[category] = _inhand_get_variants(category)
        return self._variants[category]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        env = self._env_ref
        categories = _INHAND_CATEGORIES
        category = categories[int(self._rng.integers(0, len(categories)))]
        variants = self._get_variants(category)
        variant_dir = variants[int(self._rng.integers(0, len(variants)))]

        side = "left" if self._rng.random() < 0.5 else "right"
        x = float(self._rng.uniform(_X_MIN, _X_MAX))
        y = float(self._rng.uniform(_Y_LEFT_MIN, _Y_LEFT_MAX) if side == "left"
                  else self._rng.uniform(_Y_RIGHT_MIN, _Y_RIGHT_MAX))
        yaw = float(self._rng.uniform(-np.pi, np.pi))
        scale_factor = float(self._rng.uniform(*_INHAND_SCALE_FACTOR_RANGE))

        xml = _inhand_build_xml(
            category,
            variant_dir,
            x,
            y,
            _OBJ_Z,
            yaw,
            scale_factor=scale_factor,
        )
        xml = _inhand_apply_scene_transforms(xml, self._scene_xml_transform_options)

        if env is not None:
            preserved_robot_qpos = env._get_reset_robot_qpos()
            env.reload_from_xml(xml)
            mujoco.mj_resetData(env.model, env.data)
            env._set_robot_qpos(preserved_robot_qpos)
            jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
            qadr = env.model.jnt_qposadr[jnt_id]
            w, s = np.cos(yaw / 2), np.sin(yaw / 2)
            env.data.qpos[qadr:qadr + 3] = [x, y, _OBJ_Z]
            env.data.qpos[qadr + 3:qadr + 7] = [w, 0, 0, s]
            mujoco.mj_forward(env.model, env.data)
            env._inhand_category = category
            env._inhand_variant = variant_dir.name
            env._inhand_side = side

        return RandomizationState(
            seed=seed or 0,
            object_states={"task_object_joint": {"pos": [x, y, _OBJ_Z], "quat": [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]}},
            scale_states={"task_object_joint": scale_factor},
            metadata={"category": category, "variant": variant_dir.name, "side": side},
        )

    def _restore_scene(self, model: Any, data: Any, state: RandomizationState) -> RandomizationState:
        env = self._env_ref
        category = str(state.metadata.get("category", ""))
        variant_name = str(state.metadata.get("variant", ""))
        if env is None or not category or not variant_name:
            return super()._restore_scene(model, data, state)

        variant_dir = _inhand_asset_base(category) / category / variant_name
        pose = state.object_states.get("task_object_joint", {})
        pos = pose.get("pos", [_X_MIN, _Y_LEFT_MIN, _OBJ_Z])
        quat = np.asarray(pose.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        yaw = _yaw_from_quat(quat)
        scale_factor = float(state.scale_states.get("task_object_joint", 1.0))

        xml = _inhand_build_xml(
            category,
            variant_dir,
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            yaw,
            scale_factor=scale_factor,
        )
        xml = _inhand_apply_scene_transforms(xml, self._scene_xml_transform_options)

        preserved_robot_qpos = env._get_reset_robot_qpos()
        env.reload_from_xml(xml)
        mujoco.mj_resetData(env.model, env.data)
        env._set_robot_qpos(preserved_robot_qpos)
        jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
        qadr = env.model.jnt_qposadr[jnt_id]
        env.data.qpos[qadr:qadr + 3] = pos
        env.data.qpos[qadr + 3:qadr + 7] = quat
        mujoco.mj_forward(env.model, env.data)
        env._inhand_category = category
        env._inhand_variant = variant_name
        env._inhand_side = state.metadata.get("side")
        return state


__all__ = ["InHandTransferRandomizer"]
