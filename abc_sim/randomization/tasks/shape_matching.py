"""Seeded shape/color reference permutations and sixteen-piece tabletop layouts."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from abc_sim.task_specs import SHAPE_MATCHING_PROMPTS

from ..assets.shape_matching import (
    COLORS,
    PIECE_COUNT,
    PIECE_HALF_HEIGHT,
    PIECE_JOINTS,
    TARGET_COUNT,
    WOOD_COLORS,
    build_board,
)
from ..core import PerturbRange, RandomizationState, SceneRandomizer, _resolve_scene_xml_paths

JOINTS = ("shape_board_joint", *PIECE_JOINTS)


def validate_options(request):
    options = {} if request is None else dict(request)
    if set(options) - {"palette", "order", "wood", "selected"}:
        raise ValueError("Shape matching accepts only palette, order, wood, and selected options.")
    if "palette" in options:
        values = options["palette"]
        if len(values) != 4 or len(set(values)) != 4 or any(value not in COLORS for value in values):
            raise ValueError("Shape matching needs four distinct named colors.")
    if "order" in options:
        values = options["order"]
        if len(values) != 16 or any(type(value) is not int for value in values) or set(values) != set(range(16)):
            raise ValueError("Shape matching order must be a permutation of the sixteen piece IDs.")
    if "wood" in options and (type(options["wood"]) is not int or not 0 <= options["wood"] < len(WOOD_COLORS)):
        raise ValueError("Unknown shape board wood color.")
    if "selected" in options:
        values = options["selected"]
        if (
            len(values) != TARGET_COUNT
            or any(type(value) is not int or not 0 <= value < PIECE_COUNT for value in values)
            or len(set(values)) != TARGET_COUNT
        ):
            raise ValueError("Shape matching needs eight distinct selected piece IDs from zero through fifteen.")
    return options


class ShapeMatchingRandomizer(SceneRandomizer):
    """Keep every piece outside the initially empty board with a reproducible goal."""

    perturbations = [PerturbRange(name, delta_x=(0, 0), delta_y=(0, 0)) for name in JOINTS]
    table_bounds = (0.35, 0.81, -0.52, 0.52)

    def prepare_reset(self, *, request=None, randomize=True):
        if not randomize:
            raise ValueError("Shape matching requires a seeded randomized reset or saved-state replay.")
        validate_options(request)

    def _reload_scene(self, palette, order, wood, selected):
        env = self._env_ref
        if env is None:
            raise RuntimeError("Shape matching randomization requires a bound environment.")
        root = ET.fromstring(self._base_scene_xml_string)
        asset, world = root.find("asset"), root.find("worldbody")
        for parent in (asset, world):
            for element in list(parent):
                if element.get("name", "").startswith("shape_"):
                    parent.remove(element)
        assets, bodies = build_board(tuple(palette), tuple(order), wood, tuple(selected))
        asset.extend(assets)
        world.extend(bodies)
        root.find("./keyframe/key[@name='home']").attrib.pop("qpos", None)
        xml = _resolve_scene_xml_paths(ET.tostring(root, encoding="unicode"), self._base_scene_xml_dir)
        robot = env._get_reset_robot_qpos()
        if xml != env._scene_xml_string:
            env.reload_from_xml(xml)
        env._set_robot_qpos(robot)

    @staticmethod
    def _pose(x, y, yaw, z):
        return {"pos": [float(x), float(y), float(z)], "quat": [float(np.cos(yaw / 2)), 0, 0, float(np.sin(yaw / 2))]}

    def randomize(self, model, data, seed=None, request=None):
        options = validate_options(request)
        rng = np.random.default_rng(seed)
        palette = (
            options["palette"] if "palette" in options else rng.choice(tuple(COLORS), size=4, replace=False).tolist()
        )
        order = options["order"] if "order" in options else rng.permutation(16).tolist()
        wood = options["wood"] if "wood" in options else int(rng.integers(len(WOOD_COLORS)))
        selected = (
            options["selected"]
            if "selected" in options
            else sorted(rng.choice(PIECE_COUNT, size=TARGET_COUNT, replace=False).tolist())
        )
        self._reload_scene(palette, order, wood, selected)
        model, data = self._env_ref.model, self._env_ref.data
        for _ in range(self.max_tries):
            states = {
                "shape_board_joint": self._pose(
                    rng.uniform(0.58, 0.60), rng.uniform(-0.015, 0.015), rng.uniform(-0.06, 0.06), 0.75
                )
            }
            positions = []
            for joint in PIECE_JOINTS:
                for _ in range(self.max_tries):
                    position = np.array((rng.uniform(0.38, 0.78), rng.choice((-1, 1)) * rng.uniform(0.225, 0.48)))
                    if all(np.linalg.norm(position - previous) >= 0.080 for previous in positions):
                        positions.append(position)
                        states[joint] = self._pose(*position, rng.uniform(-np.pi, np.pi), 0.752 + PIECE_HALF_HEIGHT)
                        break
                else:
                    break
            if len(states) != len(JOINTS):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if self._contacts_ok(model, data):
                return RandomizationState(
                    seed=seed or 0,
                    object_states=states,
                    metadata={
                        "palette": palette,
                        "order": order,
                        "wood": wood,
                        "selected": list(selected),
                        "prompt": SHAPE_MATCHING_PROMPTS[(seed or 0) % len(SHAPE_MATCHING_PROMPTS)],
                    },
                )
        self._raise_sampling_failure(seed)

    def _validate_scene(self, model, state):
        validate_options({name: state.metadata[name] for name in ("palette", "order", "wood", "selected")})
        if state.scale_states or set(state.object_states) != set(JOINTS):
            raise ValueError("Shape board replay requires all seventeen free-body poses and no scales.")
        if state.metadata["prompt"] not in SHAPE_MATCHING_PROMPTS:
            raise ValueError("Unknown shape matching prompt.")
        for pose in state.object_states.values():
            position, quaternion = np.asarray(pose["pos"]), np.asarray(pose["quat"])
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("Shape matching positions must be finite 3-vectors.")
            if (
                quaternion.shape != (4,)
                or not np.isfinite(quaternion).all()
                or not np.isclose(np.linalg.norm(quaternion), 1)
            ):
                raise ValueError("Shape matching orientations must be unit wxyz quaternions.")

    def _restore_scene(self, model, data, state):
        self._reload_scene(*(state.metadata[name] for name in ("palette", "order", "wood", "selected")))
        model, data = self._env_ref.model, self._env_ref.data
        self._apply_states(model, data, state.object_states)
        mujoco.mj_forward(model, data)
        return state
