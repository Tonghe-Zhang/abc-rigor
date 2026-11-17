from __future__ import annotations

from dataclasses import asdict
from pathlib import Path as _Path
from typing import Any

import mujoco
import numpy as np

from ..assets.paths import _MODELS_DIR
from ..assets.xml_common import (
    _LEGO_COLORS,
    _filter_sorting_scene_xml,
    _sample_bin_visual_styles,
)
from ..core import (
    PerturbRange,
    RandomizationState,
    ScalePerturbRange,
    SceneRandomizer,
)
from ..requests import (
    LegoBlocksResetRequest,
    NutsBoltsResetRequest,
    SortingResetRequest,
    _LEGO_MAX_BLOCKS_PER_COLOR,
    _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE,
)
from .count_box import _count_box_apply_scene_transforms

_NUTS_BOLTS_OBJECT_COUNT_RANGE = (8, _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE)
_LEGO_BLOCKS_PER_COLOR_COUNT_RANGE = (3, _LEGO_MAX_BLOCKS_PER_COLOR)
_NUTS_BOLTS_BIN_BODY_NAMES = ("nuts_sorting_bin", "bolts_sorting_bin")
_LEGO_BIN_BODY_NAMES = (
    "red_lego_sorting_bin",
    "yellow_lego_sorting_bin",
    "blue_lego_sorting_bin",
)
class _ScaleAwareTabletopRandomizer(SceneRandomizer):
    """Keep scaled tabletop objects resting at their original table clearance."""

    table_z_m: float = 0.75

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        states = super()._sample_once(nominals, rng)
        for joint_name, state in states.items():
            nominal = nominals.get(joint_name)
            if nominal is None:
                continue
            nominal_pos, _ = nominal
            table_clearance = max(float(nominal_pos[2]) - self.table_z_m, 0.0)
            scale_factor = float(self._current_scale_states.get(joint_name, 1.0))
            state["pos"][2] = self.table_z_m + table_clearance * scale_factor
        return states


class _VariableCountSortingRandomizer(_ScaleAwareTabletopRandomizer):
    """Reload a sorting scene with a sampled subset of object bodies."""

    base_scene_xml_path: _Path
    bin_body_names: tuple[str, ...] = ()
    candidate_object_body_names: tuple[str, ...] = ()
    _parked_joints: tuple[str, ...] = ()
    reset_request_cls: type[SortingResetRequest]

    def __init__(self) -> None:
        super().__init__()
        self._full_base_scene_xml_string: str | None = None
        self._full_base_scene_xml_dir: _Path | None = None
        self._full_base_scene_xml_transformed = False

    def bind_env(self, env: Any) -> None:
        super().bind_env(env)
        self._full_base_scene_xml_string = self._base_scene_xml_string
        self._full_base_scene_xml_dir = self._base_scene_xml_dir
        self._full_base_scene_xml_transformed = self._base_scene_xml_transformed

    def _sample_active_object_body_names(
        self,
        rng: np.random.Generator,
        request: SortingResetRequest | None = None,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        raise NotImplementedError

    def _sample_body_pose_source_names(
        self,
        active_object_body_names: tuple[str, ...],
        rng: np.random.Generator,
    ) -> dict[str, str]:
        return {}

    @staticmethod
    def _body_to_joint_name(body_name: str) -> str:
        return f"{body_name}_joint"

    def _configure_active_scene(
        self,
        active_object_body_names: tuple[str, ...],
        *,
        body_pose_source_names: dict[str, str] | None = None,
        bin_visual_styles: dict[str, str] | None = None,
        park_inactive: bool = False,
    ) -> None:
        active_body_names = (*self.bin_body_names, *active_object_body_names)
        active_joint_names = {self._body_to_joint_name(body_name) for body_name in active_body_names}
        self.perturbations = [
            perturbation
            for perturbation in type(self).perturbations
            if perturbation.joint_name in active_joint_names
        ]
        self.size_perturbations = [
            perturbation
            for perturbation in type(self).size_perturbations
            if perturbation.target_name in active_joint_names
        ]

        base_xml = self._full_base_scene_xml_string
        base_dir = self._full_base_scene_xml_dir
        base_transformed = self._full_base_scene_xml_transformed
        if base_xml is None:
            base_xml = self.base_scene_xml_path.read_text()
            base_dir = self.base_scene_xml_path.parent
            base_transformed = False

        scene_object_names = self.candidate_object_body_names if park_inactive else active_object_body_names
        xml = _filter_sorting_scene_xml(
            base_xml,
            candidate_object_body_names=self.candidate_object_body_names,
            active_object_body_names=scene_object_names,
            home_body_names=(*self.bin_body_names, *scene_object_names),
            body_pose_source_names=body_pose_source_names,
            bin_visual_styles=bin_visual_styles,
        )
        self._parked_joints = tuple(
            self._body_to_joint_name(name)
            for name in scene_object_names
            if name not in active_object_body_names
        )
        if self._scene_xml_transform_options is not None and not base_transformed:
            xml = _count_box_apply_scene_transforms(xml, self._scene_xml_transform_options)
            base_transformed = True

        self._base_scene_xml_string = xml
        self._base_scene_xml_dir = base_dir
        self._base_scene_xml_transformed = base_transformed
        self._fixed_body_nominals = None

    def _parked_states(self) -> dict[str, dict[str, list[float]]]:
        """Parking on the table plane behind the robot for the candidates an episode leaves out."""
        return {
            joint_name: {"pos": [-1.5 - 0.1 * index, 0.4, 0.76], "quat": [1.0, 0.0, 0.0, 0.0]}
            for index, joint_name in enumerate(self._parked_joints)
        }

    def _before_sampling(self, model: Any, data: Any) -> None:
        for joint_name, pose in self._parked_states().items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            adr = model.jnt_qposadr[joint_id]
            data.qpos[adr : adr + 7] = [*pose["pos"], *pose["quat"]]
        if self._parked_joints:
            mujoco.mj_forward(model, data)

    def _reload_configured_scene(self) -> None:
        env = self._env_ref
        preserved_robot_qpos = env._get_reset_robot_qpos()
        env.reload_from_xml(self._scene_xml_for_scale_states({}))
        mujoco.mj_resetData(env.model, env.data)
        env._set_robot_qpos(preserved_robot_qpos)
        mujoco.mj_forward(env.model, env.data)

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        options = self.reset_request_cls.from_value(request)
        active_object_body_names, metadata = self._sample_active_object_body_names(rng, options)
        body_pose_source_names = self._sample_body_pose_source_names(active_object_body_names, rng)
        style = options.bin_visual_style
        bin_visual_styles = (
            {name: str(style) for name in self.bin_body_names}
            if style
            else _sample_bin_visual_styles(self.bin_body_names, rng)
        )
        self._configure_active_scene(
            active_object_body_names,
            body_pose_source_names=body_pose_source_names,
            bin_visual_styles=bin_visual_styles,
            park_inactive=bool(options.park_inactive),
        )
        if not options.randomize_scales:
            self._reload_configured_scene()
            model, data = self._env_ref.model, self._env_ref.data
        state = super().randomize(model, data, seed=seed, request=asdict(options))
        state.object_states.update(self._parked_states())
        state.metadata.update(metadata)
        state.metadata["active_object_joints"] = [
            self._body_to_joint_name(body_name) for body_name in active_object_body_names
        ]
        state.metadata["bin_joints"] = [
            self._body_to_joint_name(body_name) for body_name in self.bin_body_names
        ]
        state.metadata["bin_visual_styles"] = dict(bin_visual_styles)
        state.metadata["active_object_count"] = len(active_object_body_names)
        if body_pose_source_names:
            object_pose_source_names = {
                body_name: source_name
                for body_name, source_name in body_pose_source_names.items()
                if body_name in active_object_body_names
            }
            bin_pose_source_names = {
                body_name: source_name
                for body_name, source_name in body_pose_source_names.items()
                if body_name in self.bin_body_names
            }
            if object_pose_source_names:
                state.metadata["spawn_slot_sources"] = object_pose_source_names
            if bin_pose_source_names:
                state.metadata["bin_slot_sources"] = bin_pose_source_names
        return state


class NutsBoltsSortingRandomizer(_VariableCountSortingRandomizer):
    """Shuffle loose nuts and bolts in the tabletop sorting workspace."""

    reset_request_cls = NutsBoltsResetRequest
    base_scene_xml_path = _MODELS_DIR / "yam_nuts_bolts_sorting_scene.xml"
    bin_body_names = _NUTS_BOLTS_BIN_BODY_NAMES
    candidate_object_body_names = (
        *[f"nut_{index}" for index in range(1, _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE + 1)],
        *[f"bolt_{index}" for index in range(1, _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE + 1)],
    )
    max_tries = 500
    min_clearance_m = 0.029
    table_bounds = (0.35, 0.80, -0.42, 0.42)
    size_perturbations = [
        ScalePerturbRange("nuts_sorting_bin_joint", scale_factor=(0.85, 1.15)),
        ScalePerturbRange("bolts_sorting_bin_joint", scale_factor=(0.85, 1.15)),
        *[
            ScalePerturbRange(f"nut_{index}_joint", scale_factor=(0.88, 1.12))
            for index in range(1, _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE + 1)
        ],
        *[
            ScalePerturbRange(f"bolt_{index}_joint", scale_factor=(0.88, 1.12))
            for index in range(1, _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE + 1)
        ],
    ]
    perturbations = [
        PerturbRange(
            "nuts_sorting_bin_joint",
            delta_x=(-0.035, 0.035),
            delta_y=(-0.040, 0.040),
            delta_yaw=(-np.pi, np.pi),
        ),
        PerturbRange(
            "bolts_sorting_bin_joint",
            delta_x=(-0.035, 0.035),
            delta_y=(-0.040, 0.040),
            delta_yaw=(-np.pi, np.pi),
        ),
        *[
            PerturbRange(
                f"nut_{index}_joint",
                delta_x=(-0.018, 0.018),
                delta_y=(-0.038, 0.038),
                delta_yaw=(-np.pi, np.pi),
            )
            for index in range(1, _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE + 1)
        ],
        *[
            PerturbRange(
                f"bolt_{index}_joint",
                delta_x=(-0.018, 0.018),
                delta_y=(-0.038, 0.038),
                delta_yaw=(-np.pi, np.pi),
            )
            for index in range(1, _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE + 1)
        ],
    ]

    def _sample_active_object_body_names(
        self,
        rng: np.random.Generator,
        request: NutsBoltsResetRequest | None = None,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        request = NutsBoltsResetRequest.from_value(request)
        count_min, count_max = _NUTS_BOLTS_OBJECT_COUNT_RANGE
        nut_count = (
            request.nut_count
            if request.nut_count is not None
            else int(rng.integers(count_min, count_max + 1))
        )
        bolt_count = (
            request.bolt_count
            if request.bolt_count is not None
            else int(rng.integers(count_min, count_max + 1))
        )
        nut_indices = sorted(
            int(index)
            for index in rng.choice(
                np.arange(1, _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE + 1),
                size=nut_count,
                replace=False,
            )
        )
        bolt_indices = sorted(
            int(index)
            for index in rng.choice(
                np.arange(1, _NUTS_BOLTS_MAX_OBJECTS_PER_TYPE + 1),
                size=bolt_count,
                replace=False,
            )
        )
        active_body_names = (
            *[f"nut_{index}" for index in nut_indices],
            *[f"bolt_{index}" for index in bolt_indices],
        )
        return active_body_names, {
            "nut_count": nut_count,
            "bolt_count": bolt_count,
            "active_counts": {
                "nuts": nut_count,
                "bolts": bolt_count,
            },
        }


class LegoBlocksSortingRandomizer(_VariableCountSortingRandomizer):
    """Shuffle loose LEGO-style blocks in the tabletop sorting workspace."""

    reset_request_cls = LegoBlocksResetRequest
    base_scene_xml_path = _MODELS_DIR / "yam_lego_blocks_sorting_scene.xml"
    bin_body_names = _LEGO_BIN_BODY_NAMES
    candidate_object_body_names = tuple(
        f"{color}_lego_{index}"
        for color in _LEGO_COLORS
        for index in range(1, _LEGO_MAX_BLOCKS_PER_COLOR + 1)
    )
    max_tries = 500
    min_clearance_m = 0.035
    table_bounds = (0.35, 0.80, -0.45, 0.45)
    size_perturbations = [
        ScalePerturbRange(f"{color}_lego_sorting_bin_joint", scale_factor=(0.85, 1.15))
        for color in _LEGO_COLORS
    ] + [
        ScalePerturbRange(f"{color}_lego_{index}_joint", scale_factor=(0.85, 1.15))
        for color in _LEGO_COLORS
        for index in range(1, _LEGO_MAX_BLOCKS_PER_COLOR + 1)
    ]
    perturbations = [
        *[
            PerturbRange(
                f"{color}_lego_sorting_bin_joint",
                delta_x=(0.0, 0.0),
                delta_y=(-0.035, 0.035),
                delta_yaw=(-np.pi, np.pi),
            )
            for color in _LEGO_COLORS
        ],
        *[
            PerturbRange(
                f"{color}_lego_{index}_joint",
                delta_x=(-0.008, 0.008),
                delta_y=(-0.008, 0.008),
                delta_yaw=(-np.pi, np.pi),
            )
            for color in _LEGO_COLORS
            for index in range(1, _LEGO_MAX_BLOCKS_PER_COLOR + 1)
        ],
    ]

    def _sample_active_object_body_names(
        self,
        rng: np.random.Generator,
        request: LegoBlocksResetRequest | None = None,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        request = LegoBlocksResetRequest.from_value(request)
        count_min, count_max = _LEGO_BLOCKS_PER_COLOR_COUNT_RANGE
        active_body_names: list[str] = []
        active_counts: dict[str, int] = {}
        for color in _LEGO_COLORS:
            count = (
                request.blocks_per_color
                if request.blocks_per_color is not None
                else int(rng.integers(count_min, count_max + 1))
            )
            active_counts[color] = count
            indices = sorted(
                int(index)
                for index in rng.choice(
                    np.arange(1, _LEGO_MAX_BLOCKS_PER_COLOR + 1),
                    size=count,
                    replace=False,
                )
            )
            active_body_names.extend(f"{color}_lego_{index}" for index in indices)
        return tuple(active_body_names), {
            "active_counts": active_counts,
            **{f"{color}_count": count for color, count in active_counts.items()},
        }

    def _sample_body_pose_source_names(
        self,
        active_object_body_names: tuple[str, ...],
        rng: np.random.Generator,
    ) -> dict[str, str]:
        slot_body_names = list(self.candidate_object_body_names)
        rng.shuffle(slot_body_names)
        pose_source_names = dict(
            zip(active_object_body_names, slot_body_names[: len(active_object_body_names)])
        )
        bin_slot_body_names = list(self.bin_body_names)
        rng.shuffle(bin_slot_body_names)
        pose_source_names.update(dict(zip(self.bin_body_names, bin_slot_body_names)))
        return pose_source_names
