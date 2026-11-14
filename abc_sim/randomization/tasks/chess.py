from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, ClassVar

import mujoco
import numpy as np

from ..assets.chess_tin import (
    _CHESS_BASE_SCENE_XML,
    _CHESS_TABLE_Z,
    _CHESS_TIN_DEFAULT_VARIANT,
    _CHESS_TIN_FLOOR_HALF_Z,
    _CHESS_TIN_JOINT,
    _build_chess_tin_scene_xml,
    _chess_tin_canonical_variant_name,
    _chess_tin_scaled_inner_half_xy,
    _chess_tin_scaled_outer_half_xy,
    _chess_tin_variant_names,
    _chess_tin_variant_scale,
)
from ..assets.water_bottles import _quat_rotate_vector
from ..core import (
    PerturbRange,
    RandomizationState,
    ScalePerturbRange,
    SceneRandomizer,
    _quat_from_axis_angle,
    _quat_from_yaw,
    _quat_mul,
    _yaw_from_quat,
)
from ..requests import ChessResetRequest, _ChessPlacementFailure

_CHESS_PIECE_JOINTS = [
    "black_bishop_1_jnt", "black_bishop_2_jnt", "black_king_1_jnt",
    "black_knight_1_jnt", "black_knight_2_jnt",
    "black_pawn_1_jnt", "black_pawn_2_jnt", "black_pawn_3_jnt",
    "black_pawn_4_jnt", "black_pawn_5_jnt", "black_pawn_6_jnt",
    "black_pawn_7_jnt", "black_pawn_8_jnt",
    "black_queen_1_jnt", "black_rook_1_jnt", "black_rook_2_jnt",
    "white_bishop_1_jnt", "white_bishop_2_jnt", "white_king_1_jnt",
    "white_knight_1_jnt", "white_knight_2_jnt",
    "white_pawn_1_jnt", "white_pawn_2_jnt", "white_pawn_3_jnt",
    "white_pawn_4_jnt", "white_pawn_5_jnt", "white_pawn_6_jnt",
    "white_pawn_7_jnt", "white_pawn_8_jnt",
    "white_queen_1_jnt", "white_rook_1_jnt", "white_rook_2_jnt",
]


class ChessRandomizer(SceneRandomizer):
    """Generate partial chess-board setup tasks.

    Non-target pieces stay on their correct board squares. Target pieces are
    either staged on the table, knocked near their target squares, or placed in
    a simple free-jointed tin box.
    """

    scenarios = ("table_setup", "knocked_setup", "tin_setup")
    color_modes = ("white_only", "black_only", "mixed_partial", "both_broad")
    scenario_weights = (0.40, 0.30, 0.30)
    target_count_range = (5, 10)
    min_clearance_m = 0.035
    max_tries = 200
    _scene_model_cache_size = 8
    table_bounds = (0.34, 0.84, -0.65, 0.65)
    _piece_sample_tries = 200
    _board_half = 0.224
    _board_margin = 0.025
    _table_stage_x_range = (0.40, 0.80)
    _table_stage_y_ranges = ((0.34, 0.58), (-0.58, -0.34))
    _table_stage_grid = (4, 4)
    _table_stage_jitter = 0.018
    _table_stage_board_margin = 0.08
    _knocked_center_grid = (4, 5)
    _knocked_center_half_xy = (0.075, 0.145)
    _knocked_center_jitter = 0.010
    _knocked_center_clearance_m = 0.045
    _tin_inner_half_xy = (0.220, 0.095)
    _tin_outer_half_xy = (0.245, 0.120)
    _tin_floor_half_z = _CHESS_TIN_FLOOR_HALF_Z
    _tin_piece_margin = 0.017
    _tin_piece_grid = (6, 3)
    _tin_board_clearance_m = 0.02
    _table_z = _CHESS_TABLE_Z
    _tin_joint = _CHESS_TIN_JOINT
    _hidden_tin_pos = (0.6, 0.0, -10.0)
    perturbations: ClassVar[list[PerturbRange]] = [
        PerturbRange(
            "chessboard",
            delta_x=(-0.05, 0.05),
            delta_y=(-0.10, 0.10),
            delta_yaw=(-0.15, 0.15),
        ),
        PerturbRange(
            _tin_joint,
            delta_x=(-0.04, 0.04),
            delta_y=(-0.005, 0.005),
            delta_yaw=(-0.25, 0.25),
        ),
        *[
            PerturbRange(jnt, delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0))
            for jnt in _CHESS_PIECE_JOINTS
        ],
    ]

    def prepare_env(self) -> None:
        self._compiled_scene_model_cache: OrderedDict[tuple[Any, ...], mujoco.MjModel] = OrderedDict()
        self._current_scene_model_cache_key: tuple[Any, ...] | None = None
        self._set_current_tin_variant(_CHESS_TIN_DEFAULT_VARIANT)
        self._reload_chess_tin_scene(_CHESS_TIN_DEFAULT_VARIANT, {})

    def _before_sampling(self, model: Any, data: Any) -> None:
        geom = model.geom("board_collision")
        if int(geom.type[0]) != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Chess board_collision must be a box")
        self._board_collision_pos = geom.pos.copy()
        self._board_collision_quat = geom.quat.copy()
        self._board_collision_half_xy = geom.size[:2].copy()

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float64)
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm <= 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / norm

    @staticmethod
    def _piece_color(joint_name: str) -> str:
        return "white" if joint_name.startswith("white_") else "black"

    @staticmethod
    def _rotate_xy(vec: np.ndarray, yaw: float) -> np.ndarray:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=np.float64)

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        return [ScalePerturbRange(joint_name) for joint_name in _CHESS_PIECE_JOINTS]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = ChessResetRequest.from_value(request)
        randomize_variants = (
            True if reset_request.randomize_variants is None else bool(reset_request.randomize_variants)
        )
        scenario = self._resolve_scenario(
            rng,
            reset_request.scenario,
            cycle_step=reset_request.cycle_scenario,
            randomize_variants=randomize_variants,
        )
        color_mode = self._resolve_color_mode(
            rng,
            reset_request.color_mode,
            cycle_step=reset_request.cycle_color_mode,
            randomize_variants=randomize_variants,
        )
        target_count = self._resolve_target_count(
            rng,
            reset_request.target_count,
            randomize_variants=randomize_variants,
        )
        tin_variant = self._resolve_tin_variant(
            rng,
            reset_request,
            randomize_variants=randomize_variants,
        )
        randomize_scales = (
            False if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        self._reload_chess_tin_scene(tin_variant, scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data

        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        for _attempt in range(self.max_tries):
            try:
                states, metadata = self._sample_setup_once(
                    nominals,
                    rng,
                    scenario=scenario,
                    color_mode=color_mode,
                    target_count=target_count,
                    tin_variant=tin_variant,
                )
            except _ChessPlacementFailure:
                continue
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            self._set_tin_collision_enabled(model, bool(metadata.get("tin_active", False)))
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(
                model, data, allow_piece_contacts=scenario == "knocked_setup"
            ):
                continue
            self._remember_selection(metadata)
            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                metadata=metadata,
            )

        self._raise_sampling_failure(seed)

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        self._current_scale_states = dict(state.scale_states)
        raw_tin_variant = state.metadata.get(
            "tin_variant",
            getattr(self, "_current_chess_tin_variant", _CHESS_TIN_DEFAULT_VARIANT),
        )
        tin_variant = _chess_tin_canonical_variant_name(str(raw_tin_variant))
        self._reload_chess_tin_scene(tin_variant, state.scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        self._set_tin_collision_enabled(model, bool(state.metadata.get("tin_active", False)))
        mujoco.mj_forward(model, data)
        self._remember_selection(state.metadata)

    def _remember_selection(self, metadata: dict[str, Any]) -> None:
        scenario = metadata.get("scenario")
        color_mode = metadata.get("color_mode")
        target_count = metadata.get("target_count")
        tin_variant = metadata.get("tin_variant")
        if scenario in self.scenarios:
            self._current_chess_scenario = str(scenario)
        if color_mode in self.color_modes:
            self._current_chess_color_mode = str(color_mode)
        if target_count is not None:
            self._current_chess_target_count = int(target_count)
        if tin_variant is not None:
            self._set_current_tin_variant(str(tin_variant))

    def _set_current_tin_variant(self, tin_variant: str) -> str:
        tin_variant = _chess_tin_canonical_variant_name(tin_variant)
        self._current_chess_tin_variant = tin_variant
        self._active_tin_scale = _chess_tin_variant_scale(tin_variant)
        self._active_tin_outer_half_xy = _chess_tin_scaled_outer_half_xy(tin_variant)
        self._active_tin_inner_half_xy = _chess_tin_scaled_inner_half_xy(tin_variant)
        return tin_variant

    def _current_tin_outer_half_xy(self) -> tuple[float, float]:
        return tuple(getattr(self, "_active_tin_outer_half_xy", self._tin_outer_half_xy))

    def _current_tin_inner_half_xy(self) -> tuple[float, float]:
        return tuple(getattr(self, "_active_tin_inner_half_xy", self._tin_inner_half_xy))

    def _set_tin_collision_enabled(self, model: Any, enabled: bool) -> None:
        contype = 1 if enabled else 0
        conaffinity = 1 if enabled else 0
        for geom_name in (
            "tin_box_floor",
            "tin_box_wall_y_pos",
            "tin_box_wall_y_neg",
            "tin_box_wall_x_pos",
            "tin_box_wall_x_neg",
        ):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                model.geom_contype[geom_id] = contype
                model.geom_conaffinity[geom_id] = conaffinity

    def _sample_setup_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
        *,
        scenario: str,
        color_mode: str,
        target_count: int,
        tin_variant: str,
    ) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
        target_joints = self._sample_target_joints(rng, color_mode, target_count)

        board_state = self._sample_board_state(nominals, rng)
        target_poses = self._target_board_poses(nominals, board_state)
        tin_active = scenario == "tin_setup"
        tin_state = (
            self._sample_tin_state(nominals, rng, board_state)
            if tin_active
            else self._hidden_tin_state(nominals)
        )
        states: dict[str, dict[str, list[float]]] = {
            "chessboard": board_state,
            self._tin_joint: tin_state,
        }

        target_set = set(target_joints)
        for joint_name in _CHESS_PIECE_JOINTS:
            states[joint_name] = copy.deepcopy(target_poses[joint_name])

        if scenario == "table_setup":
            self._place_table_targets(states, target_poses, target_joints, board_state, rng)
        elif scenario == "knocked_setup":
            self._place_knocked_targets(states, target_poses, target_joints, board_state, rng)
        elif scenario == "tin_setup":
            self._place_tin_targets(states, target_poses, target_joints, tin_state, rng)
        else:
            raise ValueError(f"Unsupported chess scenario {scenario!r}")

        metadata = {
            "scenario": scenario,
            "target_count": len(target_joints),
            "target_pieces": list(target_joints),
            "target_joints": list(target_joints),
            "target_colors": sorted({self._piece_color(joint) for joint in target_joints}),
            "color_mode": color_mode,
            "target_poses": copy.deepcopy({joint: target_poses[joint] for joint in target_joints}),
            "board_target_poses": copy.deepcopy(target_poses),
            "tin_active": tin_active,
            "tin_variant": tin_variant,
        }
        if tin_active:
            metadata["tin_pose"] = copy.deepcopy(tin_state)
            metadata["tin_inner_half_xy"] = list(self._current_tin_inner_half_xy())
            metadata["tin_outer_half_xy"] = list(self._current_tin_outer_half_xy())
            metadata["tin_scale"] = float(getattr(self, "_active_tin_scale", 1.0))
            metadata["tin_floor_z"] = float(tin_state["pos"][2] + self._tin_floor_half_z)
        metadata["non_target_pieces"] = [
            joint_name for joint_name in _CHESS_PIECE_JOINTS if joint_name not in target_set
        ]
        return states, metadata

    def _resolve_scenario(
        self,
        rng: np.random.Generator,
        requested: str | None,
        *,
        cycle_step: int,
        randomize_variants: bool,
    ) -> str:
        current = str(getattr(self, "_current_chess_scenario", self.scenarios[0]))
        if current not in self.scenarios:
            current = self.scenarios[0]
        if requested is None:
            if cycle_step:
                current_index = self.scenarios.index(current)
                return self.scenarios[(current_index + cycle_step) % len(self.scenarios)]
            if not randomize_variants:
                return current
            return str(rng.choice(self.scenarios, p=np.asarray(self.scenario_weights, dtype=np.float64)))
        requested = requested.strip().lower()
        aliases = {
            "table": "table_setup",
            "table_setup": "table_setup",
            "knocked": "knocked_setup",
            "knocked_setup": "knocked_setup",
            "tin": "tin_setup",
            "tin_setup": "tin_setup",
        }
        try:
            return aliases[requested]
        except KeyError as exc:
            raise ValueError(f"Unsupported chess scenario {requested!r}") from exc

    def _resolve_color_mode(
        self,
        rng: np.random.Generator,
        requested: str | None,
        *,
        cycle_step: int,
        randomize_variants: bool,
    ) -> str:
        current = str(getattr(self, "_current_chess_color_mode", self.color_modes[0]))
        if current not in self.color_modes:
            current = self.color_modes[0]
        if requested is None:
            if cycle_step:
                current_index = self.color_modes.index(current)
                return self.color_modes[(current_index + cycle_step) % len(self.color_modes)]
            if not randomize_variants:
                return current
            return str(rng.choice(self.color_modes))
        requested = requested.strip().lower()
        aliases = {
            "white": "white_only",
            "white_only": "white_only",
            "black": "black_only",
            "black_only": "black_only",
            "mixed": "mixed_partial",
            "mixed_partial": "mixed_partial",
            "both": "both_broad",
            "both_broad": "both_broad",
        }
        try:
            return aliases[requested]
        except KeyError as exc:
            raise ValueError(f"Unsupported chess color_mode {requested!r}") from exc

    def _resolve_target_count(
        self,
        rng: np.random.Generator,
        requested: int | None,
        *,
        randomize_variants: bool,
    ) -> int:
        lo, hi = self.target_count_range
        if requested is not None:
            count = int(requested)
        elif not randomize_variants:
            count = int(getattr(self, "_current_chess_target_count", lo))
        else:
            count = int(rng.integers(lo, hi + 1))
        if not lo <= count <= hi:
            raise ValueError(f"Chess target_count must be in [{lo}, {hi}], got {count}")
        return count

    def _resolve_tin_variant(
        self,
        rng: np.random.Generator,
        request: ChessResetRequest,
        *,
        randomize_variants: bool,
    ) -> str:
        variants = _chess_tin_variant_names()
        current = _chess_tin_canonical_variant_name(
            str(getattr(self, "_current_chess_tin_variant", _CHESS_TIN_DEFAULT_VARIANT))
        )
        if current not in variants:
            current = variants[0]

        if request.tin_variant is not None:
            requested = _chess_tin_canonical_variant_name(request.tin_variant)
            if requested not in variants:
                raise ValueError(
                    f"Unknown chess tin variant {requested!r}. Available: {', '.join(variants)}"
                )
            return requested

        if request.cycle_tin:
            return variants[(variants.index(current) + request.cycle_tin) % len(variants)]

        if not randomize_variants:
            return current

        return variants[int(rng.integers(0, len(variants)))]

    def _scene_model_cache_key(
        self,
        tin_variant: str,
        scale_states: dict[str, float],
    ) -> tuple[Any, ...]:
        scale_key = tuple(
            sorted((str(name), round(float(value), 8)) for name, value in scale_states.items())
        )
        return (tin_variant, scale_key)

    def _try_reload_cached_scene_model(self, cache_key: tuple[Any, ...]) -> bool:
        if self._env_ref is None:
            return False
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if not cache:
            return False
        cached_model = cache.get(cache_key)
        if cached_model is None:
            return False
        cache.move_to_end(cache_key)
        self._env_ref.reload_from_model(cached_model)
        self._current_scene_model_cache_key = cache_key
        return True

    def _store_compiled_scene_model(self, cache_key: tuple[Any, ...]) -> None:
        if self._env_ref is None:
            return
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if cache is None:
            self._compiled_scene_model_cache = OrderedDict()
            cache = self._compiled_scene_model_cache
        cache[cache_key] = copy.deepcopy(self._env_ref.model)
        cache.move_to_end(cache_key)
        while len(cache) > self._scene_model_cache_size:
            cache.popitem(last=False)

    def _reload_chess_tin_scene(
        self,
        tin_variant: str,
        scale_states: dict[str, float],
    ) -> None:
        tin_variant = self._set_current_tin_variant(tin_variant)
        if self._env_ref is None:
            return

        preserved_arm_state = self._env_ref._get_reset_arm_state()
        scene_cache_key = self._scene_model_cache_key(tin_variant, scale_states)
        scene_reloaded = False
        if getattr(self, "_current_scene_model_cache_key", None) == scene_cache_key:
            pass
        elif self._try_reload_cached_scene_model(scene_cache_key):
            scene_reloaded = True
        else:
            base_scene_xml = self._base_scene_xml_string
            base_scene_dir = self._base_scene_xml_dir
            base_scene_transformed = self._base_scene_xml_transformed
            if base_scene_xml is None:
                base_scene_xml = _CHESS_BASE_SCENE_XML.read_text()
                base_scene_dir = _CHESS_BASE_SCENE_XML.parent
                base_scene_transformed = False

            xml = _build_chess_tin_scene_xml(
                tin_variant=tin_variant,
                scale_states=scale_states,
                base_scene_xml=base_scene_xml,
                base_scene_dir=base_scene_dir,
            )
            if self._scene_xml_transform_options is not None and not base_scene_transformed:
                from abc_sim.scene_xml import transform_scene_xml

                xml, _ = transform_scene_xml(xml, options=self._scene_xml_transform_options)

            self._env_ref.reload_from_xml(xml)
            self._store_compiled_scene_model(scene_cache_key)
            self._current_scene_model_cache_key = scene_cache_key
            scene_reloaded = True

        if scene_reloaded:
            mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
            self._env_ref._set_qpos_from_state(preserved_arm_state)
            mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
            self._fixed_body_nominals = None

    def _hidden_tin_state(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, list[float]]:
        _nom_pos, nom_quat = nominals[self._tin_joint]
        return {
            "pos": [float(value) for value in self._hidden_tin_pos],
            "quat": self._normalize_quat(nom_quat).tolist(),
        }

    def _sample_target_joints(
        self,
        rng: np.random.Generator,
        color_mode: str,
        target_count: int,
    ) -> list[str]:
        white = [joint for joint in _CHESS_PIECE_JOINTS if joint.startswith("white_")]
        black = [joint for joint in _CHESS_PIECE_JOINTS if joint.startswith("black_")]
        if color_mode == "white_only":
            return list(rng.choice(white, size=min(target_count, len(white)), replace=False))
        if color_mode == "black_only":
            return list(rng.choice(black, size=min(target_count, len(black)), replace=False))

        if color_mode == "both_broad":
            white_count = target_count // 2
            black_count = target_count - white_count
            if bool(rng.integers(0, 2)):
                white_count, black_count = black_count, white_count
        else:
            white_count = int(rng.integers(1, target_count))
            black_count = target_count - white_count

        selected = [
            *rng.choice(white, size=min(white_count, len(white)), replace=False).tolist(),
            *rng.choice(black, size=min(black_count, len(black)), replace=False).tolist(),
        ]
        rng.shuffle(selected)
        return selected

    def _sample_board_state(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, list[float]]:
        board_p = next(p for p in self.perturbations if p.joint_name == "chessboard")
        nom_pos, nom_quat = nominals["chessboard"]
        x_min, x_max, y_min, y_max = self.table_bounds
        eff_dx = (
            max(board_p.delta_x[0], x_min - nom_pos[0]),
            min(board_p.delta_x[1], x_max - nom_pos[0]),
        )
        eff_dy = (
            max(board_p.delta_y[0], y_min - nom_pos[1]),
            min(board_p.delta_y[1], y_max - nom_pos[1]),
        )
        board_pos = nom_pos + np.array(
            [
                rng.uniform(*eff_dx),
                rng.uniform(*eff_dy),
                rng.uniform(*board_p.delta_z),
            ],
            dtype=np.float64,
        )
        board_quat = self._normalize_quat(
            _quat_mul(_quat_from_yaw(rng.uniform(*board_p.delta_yaw)), nom_quat)
        )
        return {"pos": board_pos.tolist(), "quat": board_quat.tolist()}

    def _sample_tin_state(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
        board_state: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        tin_p = next(p for p in self.perturbations if p.joint_name == self._tin_joint)
        nom_pos, nom_quat = nominals[self._tin_joint]
        x_min, x_max, y_min, y_max = self.table_bounds
        half_x, half_y = self._current_tin_outer_half_xy()
        board_y_max = float(
            self._board_footprint(board_state, margin=self._tin_board_clearance_m)[:, 1].max()
        )
        for _ in range(100):
            yaw = float(rng.uniform(*tin_p.delta_yaw))
            quat = self._normalize_quat(_quat_mul(_quat_from_yaw(yaw), nom_quat))
            tin_yaw = _yaw_from_quat(quat)
            c, s = abs(float(np.cos(tin_yaw))), abs(float(np.sin(tin_yaw)))
            extent_x = c * half_x + s * half_y
            extent_y = s * half_x + c * half_y
            x_lo = max(nom_pos[0] + tin_p.delta_x[0], x_min + extent_x)
            x_hi = min(nom_pos[0] + tin_p.delta_x[1], x_max - extent_x)
            # Move the nominal staging strip outward when the board needs room.
            y_lo = max(
                nom_pos[1] + tin_p.delta_y[0], y_min + extent_y, board_y_max + extent_y
            )
            y_hi = min(
                max(
                    nom_pos[1] + tin_p.delta_y[1],
                    y_lo + tin_p.delta_y[1] - tin_p.delta_y[0],
                ),
                y_max - extent_y,
            )
            if x_lo > x_hi or y_lo > y_hi:
                continue
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(y_lo, y_hi))
            candidate = {
                "pos": [x, y, float(self._table_z + self._tin_floor_half_z + 0.001)],
                "quat": quat.tolist(),
            }
            if not self._tin_overlaps_board(candidate, board_state):
                return candidate
        raise _ChessPlacementFailure("No separated chess tin placement fits on the table")

    def _target_board_poses(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        board_state: dict[str, list[float]],
    ) -> dict[str, dict[str, list[float]]]:
        board_nom_pos, board_nom_quat = nominals["chessboard"]
        board_new_pos = np.asarray(board_state["pos"], dtype=np.float64)
        board_new_quat = np.asarray(board_state["quat"], dtype=np.float64)
        board_nom_inv = self._quat_conjugate(board_nom_quat)
        poses: dict[str, dict[str, list[float]]] = {}
        for joint_name in _CHESS_PIECE_JOINTS:
            piece_nom_pos, piece_nom_quat = nominals[joint_name]
            local_pos = _quat_rotate_vector(board_nom_inv, piece_nom_pos - board_nom_pos)
            local_quat = _quat_mul(board_nom_inv, piece_nom_quat)
            pos = board_new_pos + _quat_rotate_vector(board_new_quat, local_pos)
            quat = self._normalize_quat(_quat_mul(board_new_quat, local_quat))
            poses[joint_name] = {"pos": pos.tolist(), "quat": quat.tolist()}
        return poses

    def _place_table_targets(
        self,
        states: dict[str, dict[str, list[float]]],
        target_poses: dict[str, dict[str, list[float]]],
        target_joints: list[str],
        board_state: dict[str, list[float]],
        rng: np.random.Generator,
    ) -> None:
        stage_slots: list[np.ndarray] = []
        for side_index in rng.permutation(len(self._table_stage_y_ranges)):
            stage_slots.extend(self._table_stage_slots(side_index, rng))
        rng.shuffle(stage_slots)

        placed_xy: list[np.ndarray] = []
        for joint_name in target_joints:
            target_pose = target_poses[joint_name]
            placed = False
            for slot_index, slot in enumerate(list(stage_slots)):
                pos = np.array(
                    [
                        slot[0] + rng.uniform(-self._table_stage_jitter, self._table_stage_jitter),
                        slot[1] + rng.uniform(-self._table_stage_jitter, self._table_stage_jitter),
                        target_pose["pos"][2],
                    ],
                    dtype=np.float64,
                )
                if not self._xy_inside_table(pos[:2], margin=0.02):
                    continue
                if self._inside_board_footprint(
                    pos[:2],
                    board_state,
                    margin=self._table_stage_board_margin,
                ):
                    continue
                if any(np.linalg.norm(pos[:2] - prev) < self.min_clearance_m for prev in placed_xy):
                    continue
                yaw = float(rng.uniform(-np.pi, np.pi))
                states[joint_name] = {
                    "pos": pos.tolist(),
                    "quat": self._normalize_quat(
                        _quat_mul(_quat_from_yaw(yaw), np.asarray(target_pose["quat"], dtype=np.float64))
                    ).tolist(),
                }
                placed_xy.append(pos[:2])
                placed = True
                del stage_slots[slot_index]
                break
            if not placed:
                raise _ChessPlacementFailure()

    def _table_stage_slots(self, side_index: int, rng: np.random.Generator) -> list[np.ndarray]:
        x_count, y_count = self._table_stage_grid
        x_min, x_max = self._table_stage_x_range
        y_min, y_max = self._table_stage_y_ranges[side_index]
        slots = [
            np.array([x, y], dtype=np.float64)
            for x in np.linspace(x_min, x_max, x_count)
            for y in np.linspace(y_min, y_max, y_count)
        ]
        rng.shuffle(slots)
        return slots

    def _xy_inside_table(self, xy: np.ndarray, *, margin: float = 0.0) -> bool:
        x_min, x_max, y_min, y_max = self.table_bounds
        return (
            x_min + margin <= float(xy[0]) <= x_max - margin
            and y_min + margin <= float(xy[1]) <= y_max - margin
        )

    def _place_knocked_targets(
        self,
        states: dict[str, dict[str, list[float]]],
        target_poses: dict[str, dict[str, list[float]]],
        target_joints: list[str],
        board_state: dict[str, list[float]],
        rng: np.random.Generator,
    ) -> None:
        cells = self._knocked_center_cells()
        rng.shuffle(cells)
        board_center_xy = np.mean(
            [np.asarray(pose["pos"][:2], dtype=np.float64) for pose in target_poses.values()],
            axis=0,
        )
        board_yaw = _yaw_from_quat(np.asarray(board_state["quat"], dtype=np.float64))
        target_set = set(target_joints)
        occupied_xy = [
            np.asarray(states[joint_name]["pos"][:2], dtype=np.float64)
            for joint_name in _CHESS_PIECE_JOINTS
            if joint_name not in target_set
        ]
        placed_xy: list[np.ndarray] = []
        for joint_name in target_joints:
            target_pose = target_poses[joint_name]
            placed = False
            for cell_index, cell in enumerate(list(cells)):
                local_xy = np.asarray(cell, dtype=np.float64) + rng.uniform(
                    -self._knocked_center_jitter,
                    self._knocked_center_jitter,
                    size=2,
                )
                world_xy = board_center_xy + self._rotate_xy(local_xy, board_yaw)
                if any(
                    np.linalg.norm(world_xy - prev) < self._knocked_center_clearance_m
                    for prev in (*occupied_xy, *placed_xy)
                ):
                    continue
                pos = np.array(
                    [
                        world_xy[0],
                        world_xy[1],
                        float(target_pose["pos"][2]) + rng.uniform(0.026, 0.040),
                    ],
                    dtype=np.float64,
                )
                tilt_axis = np.array(
                    [np.cos(rng.uniform(-np.pi, np.pi)), np.sin(rng.uniform(-np.pi, np.pi)), 0.0],
                    dtype=np.float64,
                )
                q_tilt = _quat_from_axis_angle(tilt_axis, float(rng.uniform(0.85, 1.25)))
                q_yaw = _quat_from_yaw(float(rng.uniform(-np.pi, np.pi)))
                states[joint_name] = {
                    "pos": pos.tolist(),
                    "quat": self._normalize_quat(
                        _quat_mul(_quat_mul(q_yaw, q_tilt), np.asarray(target_pose["quat"], dtype=np.float64))
                    ).tolist(),
                }
                placed_xy.append(world_xy)
                del cells[cell_index]
                placed = True
                break
            if not placed:
                raise _ChessPlacementFailure()

    def _knocked_center_cells(self) -> list[np.ndarray]:
        x_count, y_count = self._knocked_center_grid
        x_half, y_half = self._knocked_center_half_xy
        return [
            np.array([x, y], dtype=np.float64)
            for x in np.linspace(-x_half, x_half, x_count)
            for y in np.linspace(-y_half, y_half, y_count)
        ]

    def _place_tin_targets(
        self,
        states: dict[str, dict[str, list[float]]],
        target_poses: dict[str, dict[str, list[float]]],
        target_joints: list[str],
        tin_state: dict[str, list[float]],
        rng: np.random.Generator,
    ) -> None:
        cells = self._tin_cells()
        rng.shuffle(cells)
        tin_pos = np.asarray(tin_state["pos"], dtype=np.float64)
        tin_quat = np.asarray(tin_state["quat"], dtype=np.float64)
        floor_top_z = tin_pos[2] + self._tin_floor_half_z
        for joint_name, cell in zip(target_joints, cells):
            target_pose = target_poses[joint_name]
            piece_nom_z_offset = max(float(target_pose["pos"][2]) - self._table_z, 0.040)
            jitter = rng.uniform(-0.006, 0.006, size=2)
            local_xy = np.asarray(cell, dtype=np.float64) + jitter
            world_xy = tin_pos[:2] + self._rotate_xy(local_xy, _yaw_from_quat(tin_quat))
            pos = np.array([world_xy[0], world_xy[1], floor_top_z + piece_nom_z_offset], dtype=np.float64)
            q_yaw = _quat_from_yaw(float(rng.uniform(-np.pi, np.pi)))
            q_tilt = _quat_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                float(rng.uniform(-0.35, 0.35)),
            )
            states[joint_name] = {
                "pos": pos.tolist(),
                "quat": self._normalize_quat(
                    _quat_mul(_quat_mul(q_yaw, q_tilt), np.asarray(target_pose["quat"], dtype=np.float64))
                ).tolist(),
            }

    def _tin_cells(self) -> list[np.ndarray]:
        tin_inner_half_xy = self._current_tin_inner_half_xy()
        x_half = tin_inner_half_xy[0] - self._tin_piece_margin
        y_half = tin_inner_half_xy[1] - self._tin_piece_margin
        x_count, y_count = self._tin_piece_grid
        return [
            np.array([x, y], dtype=np.float64)
            for x in np.linspace(-x_half, x_half, x_count)
            for y in np.linspace(-y_half, y_half, y_count)
        ]

    def _inside_board_footprint(
        self,
        xy: np.ndarray,
        board_state: dict[str, list[float]],
        *,
        margin: float,
    ) -> bool:
        center = np.asarray(board_state["pos"][:2], dtype=np.float64)
        yaw = _yaw_from_quat(np.asarray(board_state["quat"], dtype=np.float64))
        local = self._rotate_xy(np.asarray(xy, dtype=np.float64) - center, -yaw)
        half = self._board_half + margin
        return abs(float(local[0])) < half and abs(float(local[1])) < half

    def _inside_tin_footprint(
        self,
        xy: np.ndarray,
        tin_state: dict[str, list[float]],
        *,
        margin: float,
    ) -> bool:
        center = np.asarray(tin_state["pos"][:2], dtype=np.float64)
        yaw = _yaw_from_quat(np.asarray(tin_state["quat"], dtype=np.float64))
        local = self._rotate_xy(np.asarray(xy, dtype=np.float64) - center, -yaw)
        tin_outer_half_xy = self._current_tin_outer_half_xy()
        return (
            abs(float(local[0])) < tin_outer_half_xy[0] + margin
            and abs(float(local[1])) < tin_outer_half_xy[1] + margin
        )

    def _tin_overlaps_board(
        self,
        tin_state: dict[str, list[float]],
        board_state: dict[str, list[float]],
    ) -> bool:
        tin = self._rectangle_footprint(
            np.asarray(tin_state["pos"][:2], dtype=np.float64),
            np.asarray(self._current_tin_outer_half_xy(), dtype=np.float64),
            _yaw_from_quat(np.asarray(tin_state["quat"], dtype=np.float64)),
        )
        board = self._board_footprint(board_state, margin=self._tin_board_clearance_m)
        # A separating axis from either rectangle proves their footprints are disjoint.
        for footprint in (board, tin):
            for edge in (footprint[1] - footprint[0], footprint[3] - footprint[0]):
                axis = np.array([-edge[1], edge[0]], dtype=np.float64)
                board_projection = board @ axis
                tin_projection = tin @ axis
                if (
                    board_projection.max() <= tin_projection.min()
                    or tin_projection.max() <= board_projection.min()
                ):
                    return False
        return True

    def _board_footprint(
        self, board_state: dict[str, list[float]], *, margin: float = 0.0
    ) -> np.ndarray:
        quat = np.asarray(board_state["quat"], dtype=np.float64)
        center = np.asarray(board_state["pos"], dtype=np.float64) + _quat_rotate_vector(
            quat, self._board_collision_pos
        )
        collision_quat = _quat_mul(quat, self._board_collision_quat)
        return self._rectangle_footprint(
            center[:2],
            self._board_collision_half_xy + margin,
            _yaw_from_quat(collision_quat),
        )

    @staticmethod
    def _rectangle_footprint(
        center: np.ndarray, half_xy: np.ndarray, yaw: float
    ) -> np.ndarray:
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        return center + (corners * half_xy) @ rotation.T

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        # Nominal chess squares are intentionally close, so the generic
        # centre-distance check would reject valid board setups.
        return True

    def _contacts_ok(
        self, model: Any, data: Any, *, allow_piece_contacts: bool = False
    ) -> bool:
        if not self._object_arm_contacts_ok(model, data):
            return False

        board_root = int(model.body_rootid[model.body("chessboard").id])
        tin_root = int(model.body_rootid[model.body("tin_box").id])
        piece_roots: set[int] = set()
        for joint_name in _CHESS_PIECE_JOINTS:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jnt_id >= 0:
                piece_roots.add(int(model.body_rootid[int(model.jnt_bodyid[jnt_id])]))

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body_1 = int(model.geom_bodyid[contact.geom1])
            body_2 = int(model.geom_bodyid[contact.geom2])
            root_1 = int(model.body_rootid[body_1])
            root_2 = int(model.body_rootid[body_2])
            if (root_1 == board_root and root_2 == tin_root) or (
                root_1 == tin_root and root_2 == board_root
            ):
                return False
            if (
                not allow_piece_contacts
                and root_1 != root_2
                and root_1 in piece_roots
                and root_2 in piece_roots
            ):
                return False
        return True


__all__ = [
    "ChessRandomizer",
]
