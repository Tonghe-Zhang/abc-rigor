from __future__ import annotations

from collections import OrderedDict
from typing import Any

import mujoco
import numpy as np

from ..assets.water_bottles import (
    _WATER_BOTTLE_BASE_SCENE_XML,
    _WATER_BOTTLE_DEFAULT_VARIANT,
    _WATER_BOTTLE_FLAT_SPAWN_CLEARANCE_M,
    _WATER_BOTTLE_MAX_COUNT,
    _WATER_BOTTLE_MIN_COUNT,
    _WATER_BOTTLE_TABLE_Z,
    _build_water_bottle_scene_xml,
    _water_bottle_canonical_variant_name,
    _water_bottle_flat_center_from_pose,
    _water_bottle_flat_compiled_metadata,
    _water_bottle_flat_quat,
    _water_bottle_flat_yaw_from_quat,
    _water_bottle_variant_names,
)
from ..core import (
    _WATER_BOTTLE_BIN_COLOR_PALETTE,
    PerturbRange,
    RandomizationState,
    ScalePerturbRange,
    SceneRandomizer,
    _apply_mat_color,
    _quat_from_yaw,
    _quat_mul,
    _yaw_from_quat,
    logger,
)
from ..requests import WaterBottleResetRequest, _WaterBottlePlacementFailure

class BottlesRandomizer(SceneRandomizer):
    min_clearance_m = 0.08
    perturbations = [
        # Bottles placed near edges so range covers a large area of the table.
        # table_bounds check ensures nothing falls off regardless of delta size.
        PerturbRange("bottle_1_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        PerturbRange("bottle_2_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        PerturbRange("bottle_3_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        PerturbRange("bottle_4_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        # Bin: moderate range to keep it reachable; small yaw (asymmetric shape).
        PerturbRange("bin_joint", delta_x=(-0.05, 0.03), delta_y=(-0.2, 0.2),
                     delta_yaw=(-0.5, 0.5)),
    ]


class WaterBottleRandomizer(SceneRandomizer):
    """Randomizer for the RoboCasa mesh water-bottle scene."""

    min_bottle_count = _WATER_BOTTLE_MIN_COUNT
    max_bottle_count = _WATER_BOTTLE_MAX_COUNT
    bottle_scale_factor = (0.90, 1.10)
    bin_scale_factor = (1.00, 1.40)
    table_edge_bounds: tuple[float, float, float, float] = (
        0.3025,
        0.8975,
        -0.65,
        0.65,
    )
    table_margin_m = 0.015
    bottle_margin_m = 0.010
    bin_margin_m = 0.020
    max_tries = 300
    _scene_model_cache_size = 12
    _bottle_sample_tries = 128
    _bin_footprint_half_xy = (0.105, 0.105)
    _bin_body_to_bottom_z = 0.084
    _bin_perturbation = PerturbRange(
        "bin_joint",
        delta_x=(-0.08, 0.08),
        delta_y=(-0.25, 0.25),
        delta_yaw=(-0.75, 0.75),
    )
    _bottle_perturbations = [
        PerturbRange(
            f"bottle_{index}_joint",
            delta_x=(-1.0, 1.0),
            delta_y=(-1.0, 1.0),
            delta_yaw=(-np.pi, np.pi),
        )
        for index in range(1, 7)
    ]
    perturbations = [*_bottle_perturbations[:min_bottle_count], _bin_perturbation]

    def prepare_env(self) -> None:
        self._compiled_scene_model_cache: OrderedDict[tuple[Any, ...], mujoco.MjModel] = OrderedDict()
        self._current_bottle_variants = [_WATER_BOTTLE_DEFAULT_VARIANT] * self.min_bottle_count
        self._current_active_bottle_count = self.min_bottle_count
        self._set_active_bottle_count(self.min_bottle_count)
        self._reload_bottle_variant_scene(self._current_bottle_variants, {})

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        active_count = int(getattr(self, "_current_active_bottle_count", self.min_bottle_count))
        return [
            *[
                ScalePerturbRange(perturbation.joint_name, scale_factor=self.bottle_scale_factor)
                for perturbation in self._bottle_perturbations[:active_count]
            ],
            ScalePerturbRange(self._bin_perturbation.joint_name, scale_factor=self.bin_scale_factor),
        ]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = WaterBottleResetRequest.from_value(request)
        bottle_variants = self._resolve_bottle_variants(rng, reset_request)
        self._set_active_bottle_count(len(bottle_variants))
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        self._reload_bottle_variant_scene(bottle_variants, scale_states)

        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data

        metadata = {
            "bottle_variant": bottle_variants[0],
            "bottle_variants": list(bottle_variants),
            "bottle_count": len(bottle_variants),
        }
        state = self._randomize_pose_with_rng(
            model=model,
            data=data,
            seed=seed,
            rng=rng,
            scale_states=scale_states,
            metadata=metadata,
        )
        self._sample_and_apply_bin_color(model, state, rng)
        return state

    def _restore_scene(self, model: Any, data: Any, state: RandomizationState) -> RandomizationState:
        bottle_variants = self._bottle_variants_from_state(state)
        self._set_active_bottle_count(len(bottle_variants))
        self._current_scale_states = dict(state.scale_states)
        self._reload_bottle_variant_scene(bottle_variants, state.scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        mujoco.mj_forward(model, data)
        self._apply_bin_color(model, state.metadata.get("bin_color"))
        return state

    def _resolve_bottle_variants(
        self,
        rng: np.random.Generator,
        request: WaterBottleResetRequest,
    ) -> list[str]:
        variants = _water_bottle_variant_names()
        active_count = self._resolve_bottle_count(rng, request)
        randomize_variants = request.randomize_variants
        if randomize_variants is None:
            randomize_variants = not any(
                (
                    request.bottle_variant is not None,
                    request.bottle_variants is not None,
                    request.cycle_bottle != 0,
                )
            )

        if request.bottle_variants is not None:
            requested = [_water_bottle_canonical_variant_name(variant) for variant in request.bottle_variants]
            invalid = [variant for variant in requested if variant not in variants]
            if invalid:
                raise ValueError(
                    f"Unknown water bottle variants {invalid}. Available: {', '.join(variants)}"
                )
            if request.bottle_count is not None and int(request.bottle_count) != len(requested):
                raise ValueError(
                    f"bottle_count={request.bottle_count} does not match "
                    f"{len(requested)} bottle_variants"
            )
            self._validate_bottle_count(len(requested))
            return requested

        if request.bottle_variant is not None or request.cycle_bottle != 0:
            if request.bottle_variant is not None:
                explicit_variant = _water_bottle_canonical_variant_name(request.bottle_variant)
                if explicit_variant not in variants:
                    raise ValueError(
                        f"Unknown water bottle variant {explicit_variant!r}. Available: {', '.join(variants)}"
                    )
            else:
                current_variants = list(
                    getattr(
                        self,
                        "_current_bottle_variants",
                        [_WATER_BOTTLE_DEFAULT_VARIANT],
                    )
                )
                current_variant = current_variants[0] if current_variants else _WATER_BOTTLE_DEFAULT_VARIANT
                if current_variant not in variants:
                    current_variant = variants[0]
                explicit_variant = variants[(variants.index(current_variant) + request.cycle_bottle) % len(variants)]
            return [explicit_variant] * active_count

        if not randomize_variants:
            current_variants = list(
                getattr(
                    self,
                    "_current_bottle_variants",
                    [_WATER_BOTTLE_DEFAULT_VARIANT] * active_count,
                )
            )
            if not current_variants:
                current_variants = [_WATER_BOTTLE_DEFAULT_VARIANT]
            while len(current_variants) < active_count:
                current_variants.append(current_variants[-1])
            return current_variants[:active_count]

        return [
            variants[int(rng.integers(0, len(variants)))]
            for _ in range(active_count)
        ]

    def _resolve_bottle_count(
        self,
        rng: np.random.Generator,
        request: WaterBottleResetRequest,
    ) -> int:
        if request.bottle_variants is not None:
            count = len(request.bottle_variants)
        elif request.bottle_count is not None:
            count = int(request.bottle_count)
        elif (
            request.bottle_variant is not None
            or request.cycle_bottle != 0
            or request.randomize_variants is False
        ):
            count = int(getattr(self, "_current_active_bottle_count", self.min_bottle_count))
        else:
            count = int(rng.integers(self.min_bottle_count, self.max_bottle_count + 1))
        self._validate_bottle_count(count)
        return count

    def _validate_bottle_count(self, count: int) -> None:
        if not self.min_bottle_count <= count <= self.max_bottle_count:
            raise ValueError(
                f"{type(self).__name__} bottle_count must be in "
                f"[{self.min_bottle_count}, {self.max_bottle_count}], got {count}"
            )

    def _set_active_bottle_count(self, bottle_count: int) -> None:
        self._validate_bottle_count(bottle_count)
        self._current_active_bottle_count = bottle_count
        self.perturbations = [
            *self._bottle_perturbations[:bottle_count],
            self._bin_perturbation,
        ]

    def _bottle_variants_from_state(self, state: RandomizationState) -> list[str]:
        raw_variants = state.metadata.get("bottle_variants")
        if isinstance(raw_variants, (list, tuple)) and raw_variants:
            variants = [_water_bottle_canonical_variant_name(str(variant)) for variant in raw_variants]
        else:
            count = int(
                state.metadata.get(
                    "bottle_count",
                    sum(1 for name in state.object_states if name.startswith("bottle_")),
                )
            )
            variant = _water_bottle_canonical_variant_name(
                str(state.metadata.get("bottle_variant", _WATER_BOTTLE_DEFAULT_VARIANT))
            )
            variants = [variant] * count
        self._validate_bottle_count(len(variants))
        return variants

    def _sample_and_apply_bin_color(
        self,
        model: Any,
        state: RandomizationState,
        rng: np.random.Generator,
    ) -> None:
        bin_color = list(
            _WATER_BOTTLE_BIN_COLOR_PALETTE[
                int(rng.integers(len(_WATER_BOTTLE_BIN_COLOR_PALETTE)))
            ]
        )
        state.metadata["bin_color"] = bin_color
        self._apply_bin_color(model, bin_color)

    def _apply_bin_color(self, model: Any, raw_rgba: Any) -> None:
        if raw_rgba is None:
            return
        rgba = tuple(float(value) for value in raw_rgba)
        if len(rgba) != 4:
            logger.warning("Invalid put_bottles bin color: %r", raw_rgba)
            return
        _apply_mat_color(model, "water_bottle_garbage_can_mat", rgba)

    def _scene_model_cache_key(
        self,
        bottle_variants: list[str],
        scale_states: dict[str, float],
    ) -> tuple[Any, ...]:
        scale_key = tuple(
            sorted((str(name), round(float(value), 8)) for name, value in scale_states.items())
        )
        return (tuple(bottle_variants), scale_key)

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
        return True

    def _store_compiled_scene_model(self, cache_key: tuple[Any, ...], *, scaled: bool = False) -> None:
        if self._env_ref is None:
            return
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if cache is None:
            self._compiled_scene_model_cache = OrderedDict()
            cache = self._compiled_scene_model_cache
        cache[cache_key] = self._env_ref.copy_model(self._env_ref.model)
        cache.move_to_end(cache_key)
        # Continuous random scales never repeat across episodes: keep only the latest scaled model
        # (replays of the current episode) instead of filling the cache with models no reset reuses.
        if scaled:
            previous = getattr(self, "_scaled_scene_model_key", None)
            if previous is not None and previous != cache_key:
                cache.pop(previous, None)
            self._scaled_scene_model_key = cache_key
        while len(cache) > self._scene_model_cache_size:
            cache.popitem(last=False)

    def _reload_bottle_variant_scene(
        self,
        bottle_variants: list[str],
        scale_states: dict[str, float],
    ) -> None:
        if self._env_ref is None:
            return

        bottle_variants = [_water_bottle_canonical_variant_name(variant) for variant in bottle_variants]
        preserved_robot_qpos = self._env_ref._get_reset_robot_qpos()
        scene_cache_key = self._scene_model_cache_key(bottle_variants, scale_states)
        if not self._try_reload_cached_scene_model(scene_cache_key):
            base_scene_xml = self._base_scene_xml_string
            base_scene_dir = self._base_scene_xml_dir
            base_scene_transformed = self._base_scene_xml_transformed
            if (
                base_scene_xml is None
                or "<!-- TASK_ASSETS_PLACEHOLDER -->" not in base_scene_xml
                or "<!-- TASK_BODY_PLACEHOLDER -->" not in base_scene_xml
            ):
                base_scene_xml = _WATER_BOTTLE_BASE_SCENE_XML.read_text()
                base_scene_dir = _WATER_BOTTLE_BASE_SCENE_XML.parent
                base_scene_transformed = False

            xml = _build_water_bottle_scene_xml(
                bottle_variants=bottle_variants,
                scale_states=scale_states,
                base_scene_xml=base_scene_xml,
                base_scene_dir=base_scene_dir,
            )
            if self._scene_xml_transform_options is not None and not base_scene_transformed:
                from abc_sim.scene_xml import transform_scene_xml

                xml, _ = transform_scene_xml(xml, options=self._scene_xml_transform_options)

            self._env_ref.reload_from_xml(xml)
            self._store_compiled_scene_model(scene_cache_key, scaled=bool(scale_states))

        self._current_bottle_variants = list(bottle_variants)
        self._current_active_bottle_count = len(bottle_variants)
        mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
        self._env_ref._set_robot_qpos(preserved_robot_qpos)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
        self._fixed_body_nominals = None

    def _randomize_pose_with_rng(
        self,
        *,
        model: Any,
        data: Any,
        seed: int | None,
        rng: np.random.Generator,
        scale_states: dict[str, float],
        metadata: dict[str, Any],
    ) -> RandomizationState:
        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        for _attempt in range(self.max_tries):
            try:
                states = self._sample_once(nominals, rng)
            except _WaterBottlePlacementFailure:
                continue
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue
            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                metadata=dict(metadata),
            )

        self._raise_sampling_failure(seed)

    @staticmethod
    def _rotated_half_extents(half_x: float, half_y: float, yaw: float) -> tuple[float, float]:
        c = abs(float(np.cos(yaw)))
        s = abs(float(np.sin(yaw)))
        return c * half_x + s * half_y, s * half_x + c * half_y

    @staticmethod
    def _obb_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return (
            np.array([c, s], dtype=np.float64),
            np.array([-s, c], dtype=np.float64),
        )

    @classmethod
    def _obb_overlap(
        cls,
        center_a: np.ndarray,
        half_a: tuple[float, float],
        yaw_a: float,
        center_b: np.ndarray,
        half_b: tuple[float, float],
        yaw_b: float,
        margin: float,
    ) -> bool:
        axis_ax, axis_ay = cls._obb_axes(yaw_a)
        axis_bx, axis_by = cls._obb_axes(yaw_b)
        axes = (axis_ax, axis_ay, axis_bx, axis_by)
        delta = center_b - center_a
        half_a = (half_a[0] + margin, half_a[1] + margin)
        half_b = (half_b[0] + margin, half_b[1] + margin)

        for axis in axes:
            distance = abs(float(np.dot(delta, axis)))
            radius_a = (
                half_a[0] * abs(float(np.dot(axis_ax, axis)))
                + half_a[1] * abs(float(np.dot(axis_ay, axis)))
            )
            radius_b = (
                half_b[0] * abs(float(np.dot(axis_bx, axis)))
                + half_b[1] * abs(float(np.dot(axis_by, axis)))
            )
            if distance > radius_a + radius_b:
                return False
        return True

    def _bottle_footprint(
        self,
        index: int,
        state: dict[str, list[float]],
    ) -> tuple[np.ndarray, tuple[float, float], float]:
        bottle_variants = list(getattr(self, "_current_bottle_variants", []))
        variant_name = (
            bottle_variants[index]
            if index < len(bottle_variants)
            else _WATER_BOTTLE_DEFAULT_VARIANT
        )
        scale_factor = float(
            getattr(self, "_current_scale_states", {}).get(
                self._bottle_perturbations[index].joint_name,
                1.0,
            )
        )
        half_length, half_radius, _vertical_radius = _water_bottle_flat_compiled_metadata(variant_name)
        yaw = _water_bottle_flat_yaw_from_quat(np.asarray(state["quat"], dtype=np.float64))
        return (
            _water_bottle_flat_center_from_pose(
                pos=state["pos"],
                quat=state["quat"],
                variant_name=variant_name,
                scale_factor=scale_factor,
            ),
            (half_length * scale_factor, half_radius * scale_factor),
            yaw,
        )

    def _bin_footprint(
        self,
        state: dict[str, list[float]],
    ) -> tuple[np.ndarray, tuple[float, float], float]:
        scale_factor = float(
            getattr(self, "_current_scale_states", {}).get(
                self._bin_perturbation.joint_name,
                1.0,
            )
        )
        half_x, half_y = self._bin_footprint_half_xy
        return (
            np.asarray(state["pos"][:2], dtype=np.float64),
            (half_x * scale_factor, half_y * scale_factor),
            _yaw_from_quat(np.asarray(state["quat"], dtype=np.float64)),
        )

    def _bounds_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        x_min, x_max, y_min, y_max = self.table_edge_bounds
        for index, perturbation in enumerate(self._bottle_perturbations[:self._current_active_bottle_count]):
            state = states.get(perturbation.joint_name)
            if state is None:
                return False
            center, half, yaw = self._bottle_footprint(index, state)
            extent_x, extent_y = self._rotated_half_extents(half[0], half[1], yaw)
            if center[0] - extent_x < x_min + self.table_margin_m:
                return False
            if center[0] + extent_x > x_max - self.table_margin_m:
                return False
            if center[1] - extent_y < y_min + self.table_margin_m:
                return False
            if center[1] + extent_y > y_max - self.table_margin_m:
                return False

        bin_state = states.get(self._bin_perturbation.joint_name)
        if bin_state is None:
            return False
        center, half, yaw = self._bin_footprint(bin_state)
        extent_x, extent_y = self._rotated_half_extents(half[0], half[1], yaw)
        if center[0] - extent_x < x_min + self.table_margin_m:
            return False
        if center[0] + extent_x > x_max - self.table_margin_m:
            return False
        if center[1] - extent_y < y_min + self.table_margin_m:
            return False
        if center[1] + extent_y > y_max - self.table_margin_m:
            return False
        return True

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        entries: list[tuple[np.ndarray, tuple[float, float], float]] = []
        for index, perturbation in enumerate(self._bottle_perturbations[:self._current_active_bottle_count]):
            state = states.get(perturbation.joint_name)
            if state is None:
                return False
            entries.append(self._bottle_footprint(index, state))

        for i in range(len(entries)):
            center_i, half_i, yaw_i = entries[i]
            for j in range(i + 1, len(entries)):
                center_j, half_j, yaw_j = entries[j]
                if self._obb_overlap(
                    center_i,
                    half_i,
                    yaw_i,
                    center_j,
                    half_j,
                    yaw_j,
                    self.bottle_margin_m,
                ):
                    return False

        bin_state = states.get(self._bin_perturbation.joint_name)
        if bin_state is None:
            return False
        bin_center, bin_half, bin_yaw = self._bin_footprint(bin_state)
        for bottle_center, bottle_half, bottle_yaw in entries:
            if self._obb_overlap(
                bottle_center,
                bottle_half,
                bottle_yaw,
                bin_center,
                bin_half,
                bin_yaw,
                self.bin_margin_m,
            ):
                return False
        return True

    def _contacts_ok(self, model: Any, data: Any) -> bool:
        return super()._contacts_ok(model, data)

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        states: dict[str, dict[str, list[float]]] = {}
        entries: list[tuple[np.ndarray, tuple[float, float], float]] = []
        x_min, x_max, y_min, y_max = self.table_edge_bounds

        bin_state = self._sample_bin_state(nominals, rng)
        states[self._bin_perturbation.joint_name] = bin_state
        bin_entry = self._bin_footprint(bin_state)

        for index, perturbation in enumerate(self._bottle_perturbations[:self._current_active_bottle_count]):
            variant_name = (
                self._current_bottle_variants[index]
                if index < len(self._current_bottle_variants)
                else _WATER_BOTTLE_DEFAULT_VARIANT
            )
            scale_factor = float(self._current_scale_states.get(perturbation.joint_name, 1.0))
            half_length, half_radius, vertical_radius = _water_bottle_flat_compiled_metadata(variant_name)
            half = (half_length * scale_factor, half_radius * scale_factor)
            z = (
                _WATER_BOTTLE_TABLE_Z
                + vertical_radius * scale_factor
                + _WATER_BOTTLE_FLAT_SPAWN_CLEARANCE_M
            )

            for _candidate_attempt in range(self._bottle_sample_tries):
                yaw = float(rng.uniform(*perturbation.delta_yaw))
                extent_x, extent_y = self._rotated_half_extents(half[0], half[1], yaw)
                x_lo = x_min + self.table_margin_m + extent_x
                x_hi = x_max - self.table_margin_m - extent_x
                y_lo = y_min + self.table_margin_m + extent_y
                y_hi = y_max - self.table_margin_m - extent_y
                if x_lo > x_hi or y_lo > y_hi:
                    break

                center = np.array(
                    [
                        float(rng.uniform(x_lo, x_hi)),
                        float(rng.uniform(y_lo, y_hi)),
                    ],
                    dtype=np.float64,
                )
                if any(
                    self._obb_overlap(
                        center,
                        half,
                        yaw,
                        prev_center,
                        prev_half,
                        prev_yaw,
                        self.bottle_margin_m,
                    )
                    for prev_center, prev_half, prev_yaw in entries
                ):
                    continue
                if self._obb_overlap(
                    center,
                    half,
                    yaw,
                    bin_entry[0],
                    bin_entry[1],
                    bin_entry[2],
                    self.bin_margin_m,
                ):
                    continue

                anchor_offset = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64) * half[0]
                pos_xy = center - anchor_offset
                quat = _water_bottle_flat_quat(yaw)
                states[perturbation.joint_name] = {
                    "pos": [float(pos_xy[0]), float(pos_xy[1]), float(z)],
                    "quat": quat.tolist(),
                }
                entries.append((center, half, yaw))
                break
            else:
                raise _WaterBottlePlacementFailure()

            if perturbation.joint_name not in states:
                raise _WaterBottlePlacementFailure()

        return states

    def _sample_bin_state(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, list[float]]:
        joint_name = self._bin_perturbation.joint_name
        if joint_name not in nominals:
            raise _WaterBottlePlacementFailure()

        nominal_pos, nominal_quat = nominals[joint_name]
        scale_factor = float(self._current_scale_states.get(joint_name, 1.0))
        half_x = self._bin_footprint_half_xy[0] * scale_factor
        half_y = self._bin_footprint_half_xy[1] * scale_factor
        x_min, x_max, y_min, y_max = self.table_edge_bounds
        x_lo = max(
            nominal_pos[0] + self._bin_perturbation.delta_x[0],
            x_min + self.table_margin_m + half_x,
        )
        x_hi = min(
            nominal_pos[0] + self._bin_perturbation.delta_x[1],
            x_max - self.table_margin_m - half_x,
        )
        y_lo = max(
            nominal_pos[1] + self._bin_perturbation.delta_y[0],
            y_min + self.table_margin_m + half_y,
        )
        y_hi = min(
            nominal_pos[1] + self._bin_perturbation.delta_y[1],
            y_max - self.table_margin_m - half_y,
        )
        if x_lo > x_hi or y_lo > y_hi:
            raise _WaterBottlePlacementFailure()

        yaw = float(rng.uniform(*self._bin_perturbation.delta_yaw))
        quat = _quat_mul(_quat_from_yaw(yaw), nominal_quat)
        z = _WATER_BOTTLE_TABLE_Z + self._bin_body_to_bottom_z * scale_factor + 0.001
        return {
            "pos": [
                float(rng.uniform(x_lo, x_hi)),
                float(rng.uniform(y_lo, y_hi)),
                float(z),
            ],
            "quat": quat.tolist(),
        }




__all__ = ["BottlesRandomizer", "WaterBottleRandomizer"]
