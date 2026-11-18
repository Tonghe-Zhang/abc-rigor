from __future__ import annotations

from collections import OrderedDict
from typing import Any

import mujoco
import numpy as np

from ..assets.mugs import (
    _MUG_BASE_SCENE_XMLS,
    _MUG_DEFAULT_VARIANT,
    _MUG_FLIP_MUG_MARGIN_M,
    _MUG_FLIP_SPAWN_CLEARANCE_M,
    _MUG_FLIP_TRAY_FLOOR_Z_OFFSET,
    _MUG_FLIP_TRAY_INNER_HALF_XY,
    _MUG_FLIP_TRAY_MARGIN_M,
    _build_mug_scene_xml,
    _mug_canonical_variant_name,
    _mug_flip_compiled_metadata,
    _mug_plain_color_material_names,
    _mug_variant_names,
    _mujoco_body_in_subtree,
)
from ..core import (
    _COLOR_RANDOMIZE_PROB,
    _MUG_COLOR_PALETTE,
    _TRAY_COLOR_PALETTE,
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
from ..requests import MugResetRequest

class MugVariantRandomizer(SceneRandomizer):
    """Base randomizer for mug tasks that swap mug mesh variants on reset."""

    mug_task_name: str = ""
    mug_body_names: tuple[str, ...] = ("mug_1", "mug_2")
    min_mug_count = 2
    max_mug_count = 2
    randomize_mug_count = False
    independent_mug_variants = False
    mug_scale_factor = (0.90, 1.10)
    _scene_model_cache_size = 16

    def prepare_env(self) -> None:
        self._current_mug_variants = [_MUG_DEFAULT_VARIANT] * self.max_mug_count
        self._current_active_mug_count = self.max_mug_count
        self._compiled_scene_model_cache: OrderedDict[tuple[Any, ...], mujoco.MjModel] = OrderedDict()
        self._set_active_mug_count(self._current_active_mug_count)

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        active_count = int(getattr(self, "_current_active_mug_count", self.max_mug_count))
        return [
            ScalePerturbRange(f"{body_name}_jnt", scale_factor=self.mug_scale_factor)
            for body_name in self.mug_body_names[:active_count]
        ]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = MugResetRequest.from_value(request)
        mug_variants = self._resolve_mug_variants(rng, reset_request)
        self._set_active_mug_count(len(mug_variants))
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        self._reload_mug_variant_scene(mug_variants, scale_states)

        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data

        metadata = {
            "mug_variant": mug_variants[0],
            "mug_variants": list(mug_variants),
            "mug_count": len(mug_variants),
        }

        state = self._randomize_pose_with_rng(
            model=model,
            data=data,
            seed=seed,
            rng=rng,
            scale_states=scale_states,
            metadata=metadata,
        )
        mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
        if mug_colors:
            state.metadata["mug_colors"] = mug_colors
            self._apply_plain_mug_colors(model, mug_colors, mug_variants)
        return state

    def _restore_scene(self, model: Any, data: Any, state: RandomizationState) -> RandomizationState:
        mug_variants = self._mug_variants_from_state(state)
        self._set_active_mug_count(len(mug_variants))
        self._current_scale_states = dict(state.scale_states)
        self._reload_mug_variant_scene(mug_variants, state.scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        raw_colors = state.metadata.get("mug_colors")
        if isinstance(raw_colors, dict):
            self._apply_plain_mug_colors(model, raw_colors, mug_variants)
        mujoco.mj_forward(model, data)
        return state

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
        if not self.perturbations:
            return RandomizationState(
                seed=seed or 0,
                object_states={},
                scale_states=scale_states,
                metadata=dict(metadata),
            )

        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        for _attempt in range(self.max_tries):
            states = self._sample_once(nominals, rng)
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

    def _resolve_mug_variants(
        self,
        rng: np.random.Generator,
        request: MugResetRequest,
    ) -> list[str]:
        if self._env_ref is None:
            return [_MUG_DEFAULT_VARIANT] * self.max_mug_count

        variants = _mug_variant_names(self.mug_task_name)
        variant_pool = (
            [_mug_canonical_variant_name(variant) for variant in request.mug_variant_pool]
            if request.mug_variant_pool is not None
            else variants
        )
        invalid_pool = [variant for variant in variant_pool if variant not in variants]
        if invalid_pool:
            raise ValueError(
                f"Unknown mug variant pool entries {invalid_pool}. Available: {', '.join(variants)}"
            )
        if request.mug_variant_pool is not None and not variant_pool:
            raise ValueError("mug_variant_pool cannot be empty")
        randomize_variants = request.randomize_variants
        if randomize_variants is None:
            randomize_variants = not any(
                (
                    request.mug_variant is not None,
                    request.mug_variants is not None,
                    request.cycle_mug != 0,
                )
            )

        active_count = self._resolve_mug_count(
            rng=rng,
            request=request,
            randomize_variants=randomize_variants,
        )

        if request.mug_variants is not None:
            requested = [_mug_canonical_variant_name(variant) for variant in request.mug_variants]
            invalid = [variant for variant in requested if variant not in variants]
            if invalid:
                raise ValueError(
                    f"Unknown mug variants {invalid}. Available: {', '.join(variants)}"
                )
            if request.mug_count is not None and int(request.mug_count) != len(requested):
                raise ValueError(
                    f"mug_count={request.mug_count} does not match {len(requested)} mug_variants"
                )
            self._validate_mug_count(len(requested))
            return requested

        if request.mug_variant_combos is not None:
            combos = tuple(
                tuple(_mug_canonical_variant_name(variant) for variant in combo)
                for combo in request.mug_variant_combos
            )
            if not combos:
                raise ValueError("mug_variant_combos cannot be empty")
            invalid = sorted({variant for combo in combos for variant in combo if variant not in variants})
            if invalid:
                raise ValueError(
                    f"Unknown mug variant combo entries {invalid}. Available: {', '.join(variants)}"
                )
            counts = {len(combo) for combo in combos}
            if len(counts) != 1:
                raise ValueError("All mug_variant_combos must have the same length")
            combo_count = counts.pop()
            self._validate_mug_count(combo_count)
            if request.mug_count is not None and int(request.mug_count) != combo_count:
                raise ValueError(
                    f"mug_count={request.mug_count} does not match {combo_count} mug_variant_combos"
                )
            return list(combos[int(rng.integers(0, len(combos)))])

        current_variants = list(
            getattr(self, "_current_mug_variants", [_MUG_DEFAULT_VARIANT] * self.max_mug_count)
        )
        if not current_variants:
            current_variants = [_MUG_DEFAULT_VARIANT] * self.max_mug_count
        current_variant = current_variants[0]
        if current_variant not in variants:
            current_variant = variants[0]

        if request.mug_variant is not None:
            explicit_variant = _mug_canonical_variant_name(request.mug_variant)
            if explicit_variant not in variants:
                raise ValueError(
                    f"Unknown mug variant {explicit_variant!r}. Available: {', '.join(variants)}"
                )
            return [explicit_variant] * active_count

        if request.cycle_mug:
            current_index = variants.index(current_variant)
            return [variants[(current_index + request.cycle_mug) % len(variants)]] * active_count

        if not randomize_variants:
            if len(current_variants) >= active_count:
                return current_variants[:active_count]
            return current_variants + [current_variants[-1]] * (active_count - len(current_variants))

        if self.independent_mug_variants:
            return [variant_pool[int(rng.integers(0, len(variant_pool)))] for _ in range(active_count)]
        return [variant_pool[int(rng.integers(0, len(variant_pool)))]] * active_count

    def _resolve_mug_count(
        self,
        *,
        rng: np.random.Generator,
        request: MugResetRequest,
        randomize_variants: bool,
    ) -> int:
        if request.mug_variants is not None:
            return len(request.mug_variants)
        if request.mug_variant_combos is not None:
            counts = {len(combo) for combo in request.mug_variant_combos}
            if len(counts) != 1:
                raise ValueError("All mug_variant_combos must have the same length")
            return counts.pop()
        if request.mug_count is not None:
            count = int(request.mug_count)
        elif not self.randomize_mug_count:
            count = int(getattr(self, "_current_active_mug_count", self.max_mug_count))
        elif request.mug_variant is not None or request.cycle_mug != 0 or not randomize_variants:
            count = int(getattr(self, "_current_active_mug_count", self.max_mug_count))
        else:
            count = int(rng.integers(self.min_mug_count, self.max_mug_count + 1))
        self._validate_mug_count(count)
        return count

    def _validate_mug_count(self, count: int) -> None:
        if not self.min_mug_count <= count <= self.max_mug_count:
            raise ValueError(
                f"{type(self).__name__} mug_count must be in "
                f"[{self.min_mug_count}, {self.max_mug_count}], got {count}"
            )

    def _set_active_mug_count(self, mug_count: int) -> None:
        self._validate_mug_count(mug_count)
        self._current_active_mug_count = mug_count

    def _mug_variants_from_state(self, state: RandomizationState) -> list[str]:
        raw_variants = state.metadata.get("mug_variants")
        if isinstance(raw_variants, (list, tuple)) and raw_variants:
            variants = [_mug_canonical_variant_name(str(variant)) for variant in raw_variants]
        else:
            inferred_count = int(
                state.metadata.get(
                    "mug_count",
                    self._infer_mug_count_from_object_states(state.object_states),
                )
            )
            variant = _mug_canonical_variant_name(
                str(state.metadata.get("mug_variant", _MUG_DEFAULT_VARIANT))
            )
            variants = [variant] * inferred_count
        self._validate_mug_count(len(variants))
        return variants

    def _infer_mug_count_from_object_states(
        self,
        object_states: dict[str, dict[str, list[float]]],
    ) -> int:
        count = 0
        for body_name in self.mug_body_names:
            joint_name = f"{body_name}_jnt"
            if joint_name in object_states:
                count += 1
        return count or int(getattr(self, "_current_active_mug_count", self.max_mug_count))

    def _scene_model_cache_key(
        self,
        mug_variants: list[str],
        scale_states: dict[str, float],
    ) -> tuple[Any, ...]:
        scale_key = tuple(
            sorted((str(name), round(float(value), 8)) for name, value in scale_states.items())
        )
        return (tuple(mug_variants), scale_key)

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

    def _reload_mug_variant_scene(
        self,
        mug_variants: list[str],
        scale_states: dict[str, float],
    ) -> None:
        if self._env_ref is None:
            return

        mug_variants = [_mug_canonical_variant_name(variant) for variant in mug_variants]
        preserved_robot_qpos = self._env_ref._get_reset_robot_qpos()
        scene_cache_key = self._scene_model_cache_key(mug_variants, scale_states)
        if not self._try_reload_cached_scene_model(scene_cache_key):
            base_scene_xml = self._base_scene_xml_string
            base_scene_dir = self._base_scene_xml_dir
            if base_scene_xml is None:
                base_scene_path = _MUG_BASE_SCENE_XMLS[self.mug_task_name]
                base_scene_xml = base_scene_path.read_text()
                base_scene_dir = base_scene_path.parent

            xml = _build_mug_scene_xml(
                task_name=self.mug_task_name,
                mug_variant=mug_variants[0],
                mug_variants=mug_variants,
                scale_states=scale_states,
                base_scene_xml=base_scene_xml,
                base_scene_dir=base_scene_dir,
            )
            if self._scene_xml_transform_options is not None and not self._base_scene_xml_transformed:
                from abc_sim.scene_xml import transform_scene_xml

                xml, _ = transform_scene_xml(xml, options=self._scene_xml_transform_options)

            self._env_ref.reload_from_xml(xml)
            self._store_compiled_scene_model(scene_cache_key, scaled=bool(scale_states))

        self._current_mug_variants = list(mug_variants)
        self._current_active_mug_count = len(mug_variants)
        mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
        self._env_ref._set_robot_qpos(preserved_robot_qpos)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
        self._fixed_body_nominals = None

    def _sample_plain_mug_colors(
        self,
        rng: np.random.Generator,
        mug_variants: list[str],
    ) -> dict[str, list[float]]:
        if rng.random() >= _COLOR_RANDOMIZE_PROB:
            return {}
        return {
            body_name: list(_MUG_COLOR_PALETTE[int(rng.integers(len(_MUG_COLOR_PALETTE)))])
            for body_name, mug_variant in zip(self.mug_body_names, mug_variants)
            if mug_variant == _MUG_DEFAULT_VARIANT
        }

    def _apply_plain_mug_colors(
        self,
        model: Any,
        mug_colors: dict[str, Any],
        mug_variants: list[str],
    ) -> None:
        for instance_index, (body_name, mug_variant) in enumerate(zip(self.mug_body_names, mug_variants)):
            if mug_variant != _MUG_DEFAULT_VARIANT:
                continue
            raw_rgba = mug_colors.get(body_name)
            if raw_rgba is None:
                continue
            rgba = tuple(float(value) for value in raw_rgba)
            if len(rgba) != 4:
                logger.warning("Invalid mug color for %s: %r", body_name, raw_rgba)
                continue
            candidate_names = (
                *_mug_plain_color_material_names(self.mug_task_name, instance_index),
                f"{body_name}_color",
            )
            applied = False
            for mat_name in candidate_names:
                mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
                if mat_id >= 0:
                    model.mat_rgba[mat_id] = rgba
                    applied = True
            if not applied:
                logger.warning("No colorable material found for %s", body_name)


class MugTreeRandomizer(MugVariantRandomizer):
    mug_task_name = "mug_tree"
    mug_body_names = ("mug_1", "mug_2", "mug_3")
    min_mug_count = 1
    max_mug_count = 3
    randomize_mug_count = True
    independent_mug_variants = True
    min_clearance_m = 0.12
    _tree_perturbation = PerturbRange(
        "mug_tree",
        delta_x=(-0.10, 0.10),
        delta_y=(-0.20, 0.20),
        delta_yaw=(-0.25, 0.25),
        fixed_body=True,
    )
    _mug_perturbations = [
        PerturbRange("mug_1_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.3, 0.3)),
        PerturbRange("mug_2_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.3, 0.3)),
        PerturbRange("mug_3_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.3, 0.3)),
    ]
    perturbations = [
        _tree_perturbation,
        *_mug_perturbations,
    ]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = MugResetRequest.from_value(request)
        requested_mug_variants = self._resolve_mug_variants(rng, reset_request)
        requested_count = len(requested_mug_variants)
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )

        last_scale_states: dict[str, float] = {}
        for candidate_count in range(requested_count, self.min_mug_count - 1, -1):
            mug_variants = requested_mug_variants[:candidate_count]
            self._set_active_mug_count(candidate_count)
            scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
            last_scale_states = scale_states
            self._current_scale_states = dict(scale_states)
            self._reload_mug_variant_scene(mug_variants, scale_states)

            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data

            metadata: dict[str, Any] = {
                "mug_variant": mug_variants[0],
                "mug_variants": list(mug_variants),
                "mug_count": len(mug_variants),
            }
            if candidate_count != requested_count:
                metadata["requested_mug_count"] = requested_count
                metadata["mug_count_reduced"] = True

            state = self._try_randomize_pose_with_rng(
                model=model,
                data=data,
                seed=seed,
                rng=rng,
                scale_states=scale_states,
                metadata=metadata,
            )
            if state is None:
                continue

            mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
            if mug_colors:
                state.metadata["mug_colors"] = mug_colors
                self._apply_plain_mug_colors(model, mug_colors, mug_variants)
            return state

        mug_variants = requested_mug_variants[: self.min_mug_count]
        self._set_active_mug_count(len(mug_variants))
        self._current_scale_states = dict(last_scale_states)
        self._reload_mug_variant_scene(mug_variants, last_scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        logger.warning(
            "%s: no collision-free mug placement found after reducing to %d mug; using a single-mug sample",
            type(self).__name__,
            self.min_mug_count,
        )
        state = self._randomize_pose_with_rng(
            model=model,
            data=data,
            seed=seed,
            rng=rng,
            scale_states=last_scale_states,
            metadata={
                "mug_variant": mug_variants[0],
                "mug_variants": list(mug_variants),
                "mug_count": len(mug_variants),
                "requested_mug_count": requested_count,
                "mug_count_reduced": True,
            },
        )
        mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
        if mug_colors:
            state.metadata["mug_colors"] = mug_colors
            self._apply_plain_mug_colors(model, mug_colors, mug_variants)
        return state

    def _set_active_mug_count(self, mug_count: int) -> None:
        self._validate_mug_count(mug_count)
        self._current_active_mug_count = mug_count
        self.perturbations = [
            self._tree_perturbation,
            *self._mug_perturbations[:mug_count],
        ]

    def _try_randomize_pose_with_rng(
        self,
        *,
        model: Any,
        data: Any,
        seed: int | None,
        rng: np.random.Generator,
        scale_states: dict[str, float],
        metadata: dict[str, Any],
    ) -> RandomizationState | None:
        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        for _attempt in range(self.max_tries):
            states = self._sample_once(nominals, rng)
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
        return None


class MugFlipRandomizer(MugVariantRandomizer):
    mug_task_name = "mug_flip"
    mug_body_names = ("mug_1", "mug_2", "mug_3", "mug_4")
    min_mug_count = 1
    max_mug_count = 4
    randomize_mug_count = True
    independent_mug_variants = True
    # Mugs start upside-down (quat=[0,1,0,0]); yaw rotation is still world-Z.
    # Tray is a fixed body; mugs are sampled relative to the tray's new position
    # so they stay on the tray after randomization.
    min_clearance_m = 0.03
    _tray_perturbation = PerturbRange("tray", delta_x=(-0.10, 0.05), delta_y=(-0.3, 0.3),
                                      delta_yaw=(-0.25, 0.25), fixed_body=True)
    _mug_slot_perturbations = [
        # Mug deltas are tray-local jitter around count-specific slots.
        PerturbRange("mug_1_jnt", delta_x=(-0.012, 0.012), delta_y=(-0.012, 0.012)),
        PerturbRange("mug_2_jnt", delta_x=(-0.012, 0.012), delta_y=(-0.012, 0.012)),
        PerturbRange("mug_3_jnt", delta_x=(-0.012, 0.012), delta_y=(-0.012, 0.012)),
        PerturbRange("mug_4_jnt", delta_x=(-0.012, 0.012), delta_y=(-0.012, 0.012)),
    ]
    _mug_slot_centers_by_count: dict[int, tuple[tuple[float, float], ...]] = {
        1: ((0.0, 0.0),),
        2: ((-0.064, 0.044), (0.064, -0.044)),
        3: ((-0.068, 0.044), (0.068, 0.044), (0.0, -0.050)),
        4: ((-0.068, 0.044), (-0.068, -0.044), (0.068, 0.044), (0.068, -0.044)),
    }
    perturbations = [
        _tray_perturbation,
        *_mug_slot_perturbations,
    ]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = MugResetRequest.from_value(request)
        requested_mug_variants = self._resolve_mug_variants(rng, reset_request)
        requested_count = len(requested_mug_variants)
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )

        last_scale_states: dict[str, float] = {}
        for candidate_count in range(requested_count, self.min_mug_count - 1, -1):
            mug_variants = requested_mug_variants[:candidate_count]
            self._set_active_mug_count(candidate_count)
            scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
            last_scale_states = scale_states
            self._current_scale_states = dict(scale_states)
            self._reload_mug_variant_scene(mug_variants, scale_states)

            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data

            metadata: dict[str, Any] = {
                "mug_variant": mug_variants[0],
                "mug_variants": list(mug_variants),
                "mug_count": len(mug_variants),
            }
            if candidate_count != requested_count:
                metadata["requested_mug_count"] = requested_count
                metadata["mug_count_reduced"] = True

            state = self._try_randomize_pose_with_rng(
                model=model,
                data=data,
                seed=seed,
                rng=rng,
                scale_states=scale_states,
                metadata=metadata,
            )
            if state is None:
                continue

            if candidate_count != requested_count:
                logger.info(
                    "%s: reduced mug_count from %d to %d because selected mug assets did not fit",
                    type(self).__name__,
                    requested_count,
                    candidate_count,
                )
            mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
            if mug_colors:
                state.metadata["mug_colors"] = mug_colors
                self._apply_plain_mug_colors(model, mug_colors, mug_variants)
            self._sample_and_apply_tray_color(model, state, rng)
            return state

        mug_variants = requested_mug_variants[: self.min_mug_count]
        self._set_active_mug_count(len(mug_variants))
        self._current_scale_states = dict(last_scale_states)
        self._reload_mug_variant_scene(mug_variants, last_scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        logger.warning(
            "%s: no collision-free mug placement found after reducing to %d mug; using a single-mug sample",
            type(self).__name__,
            self.min_mug_count,
        )
        state = self._randomize_pose_with_rng(
            model=model,
            data=data,
            seed=seed,
            rng=rng,
            scale_states=last_scale_states,
            metadata={
                "mug_variant": mug_variants[0],
                "mug_variants": list(mug_variants),
                "mug_count": len(mug_variants),
                "requested_mug_count": requested_count,
                "mug_count_reduced": True,
            },
        )
        mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
        if mug_colors:
            state.metadata["mug_colors"] = mug_colors
            self._apply_plain_mug_colors(model, mug_colors, mug_variants)
        self._sample_and_apply_tray_color(model, state, rng)
        return state

    def _restore_scene(self, model: Any, data: Any, state: RandomizationState) -> RandomizationState:
        super()._restore_scene(model, data, state)
        target_model = self._env_ref.model if self._env_ref is not None else model
        tray_color = state.metadata.get("tray_color")
        if tray_color is not None:
            self._apply_tray_color(target_model, tray_color)
        return state

    def _sample_and_apply_tray_color(
        self,
        model: Any,
        state: RandomizationState,
        rng: np.random.Generator,
    ) -> None:
        tray_color = list(_TRAY_COLOR_PALETTE[int(rng.integers(len(_TRAY_COLOR_PALETTE)))])
        state.metadata["tray_color"] = tray_color
        self._apply_tray_color(model, tray_color)

    def _apply_tray_color(self, model: Any, raw_rgba: Any) -> None:
        rgba = tuple(float(value) for value in raw_rgba)
        if len(rgba) != 4:
            logger.warning("Invalid mug_flip tray color: %r", raw_rgba)
            return
        _apply_mat_color(model, "tray_blue", rgba)

    def _set_active_mug_count(self, mug_count: int) -> None:
        self._validate_mug_count(mug_count)
        self._current_active_mug_count = mug_count
        self.perturbations = [
            self._tray_perturbation,
            *self._mug_slot_perturbations[:mug_count],
        ]

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        return [
            ScalePerturbRange(perturbation.joint_name, scale_factor=self.mug_scale_factor)
            for perturbation in self._mug_slot_perturbations[: int(getattr(self, "_current_active_mug_count", 2))]
        ]

    def _try_randomize_pose_with_rng(
        self,
        *,
        model: Any,
        data: Any,
        seed: int | None,
        rng: np.random.Generator,
        scale_states: dict[str, float],
        metadata: dict[str, Any],
    ) -> RandomizationState | None:
        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        for _attempt in range(self.max_tries):
            states = self._sample_once(nominals, rng)
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
        return None

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
    ) -> bool:
        axis_ax, axis_ay = cls._obb_axes(yaw_a)
        axis_bx, axis_by = cls._obb_axes(yaw_b)
        axes = (axis_ax, axis_ay, axis_bx, axis_by)
        delta = center_b - center_a
        half_a = (half_a[0] + _MUG_FLIP_MUG_MARGIN_M, half_a[1] + _MUG_FLIP_MUG_MARGIN_M)
        half_b = (half_b[0] + _MUG_FLIP_MUG_MARGIN_M, half_b[1] + _MUG_FLIP_MUG_MARGIN_M)

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

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        tray_state = states.get("tray")
        if tray_state is None:
            return False

        tray_pos = np.asarray(tray_state["pos"], dtype=np.float64)
        tray_yaw = _yaw_from_quat(np.asarray(tray_state["quat"], dtype=np.float64))
        c = float(np.cos(-tray_yaw))
        s = float(np.sin(-tray_yaw))

        entries: list[tuple[np.ndarray, tuple[float, float], float]] = []
        mug_variants = list(getattr(self, "_current_mug_variants", []))
        scale_states = getattr(self, "_current_scale_states", {})
        for instance_index, p in enumerate(self.perturbations):
            if p.joint_name == "tray":
                continue
            state = states.get(p.joint_name)
            if state is None:
                continue
            rel_xy = np.asarray(state["pos"][:2], dtype=np.float64) - tray_pos[:2]
            local_center = np.array(
                [c * rel_xy[0] - s * rel_xy[1], s * rel_xy[0] + c * rel_xy[1]],
                dtype=np.float64,
            )
            variant_index = instance_index - 1
            variant_name = (
                mug_variants[variant_index]
                if variant_index < len(mug_variants)
                else _MUG_DEFAULT_VARIANT
            )
            scale_factor = float(scale_states.get(p.joint_name, 1.0))
            half_x, half_y, _min_z, _max_z = _mug_flip_compiled_metadata(variant_name)
            local_yaw = _yaw_from_quat(np.asarray(state["quat"], dtype=np.float64)) - tray_yaw
            entries.append(
                (
                    local_center,
                    (half_x * scale_factor, half_y * scale_factor),
                    local_yaw,
                )
            )

        for i in range(len(entries)):
            center_i, half_i, yaw_i = entries[i]
            for j in range(i + 1, len(entries)):
                center_j, half_j, yaw_j = entries[j]
                if self._obb_overlap(center_i, half_i, yaw_i, center_j, half_j, yaw_j):
                    return False
        return True

    def _contacts_ok(self, model: Any, data: Any) -> bool:
        if not self._object_arm_contacts_ok(model, data):
            return False

        mug_root_ids: set[int] = set()
        for p in self.perturbations:
            if p.joint_name == "tray":
                continue
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p.joint_name)
            if jnt_id >= 0:
                body_id = int(model.jnt_bodyid[jnt_id])
                mug_root_ids.add(int(model.body_rootid[body_id]))

        tray_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
        for c in range(data.ncon):
            contact = data.contact[c]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            r1 = int(model.body_rootid[b1])
            r2 = int(model.body_rootid[b2])
            if r1 != r2 and r1 in mug_root_ids and r2 in mug_root_ids:
                return False
            if tray_body_id >= 0:
                mug_tray_contact = (
                    (r1 in mug_root_ids and _mujoco_body_in_subtree(model, b2, tray_body_id))
                    or (r2 in mug_root_ids and _mujoco_body_in_subtree(model, b1, tray_body_id))
                )
                if mug_tray_contact:
                    normal_z = abs(float(np.asarray(contact.frame, dtype=np.float64).reshape(-1)[2]))
                    if normal_z < 0.5:
                        return False
        return True

    @staticmethod
    def _rotated_mug_half_extents(half_x: float, half_y: float, yaw: float) -> tuple[float, float]:
        c = abs(float(np.cos(yaw)))
        s = abs(float(np.sin(yaw)))
        return c * half_x + s * half_y, s * half_x + c * half_y

    @staticmethod
    def _clamp_tray_local_xy(
        local_xy: np.ndarray,
        extent_x: float,
        extent_y: float,
    ) -> np.ndarray:
        tray_half_x, tray_half_y = _MUG_FLIP_TRAY_INNER_HALF_XY
        x_limit = max(0.0, tray_half_x - extent_x - _MUG_FLIP_TRAY_MARGIN_M)
        y_limit = max(0.0, tray_half_y - extent_y - _MUG_FLIP_TRAY_MARGIN_M)
        return np.array(
            [
                float(np.clip(local_xy[0], -x_limit, x_limit)),
                float(np.clip(local_xy[1], -y_limit, y_limit)),
            ],
            dtype=np.float64,
        )

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}

        tray_p = next(p for p in self.perturbations if p.joint_name == "tray")
        tray_nom_pos, tray_nom_quat = nominals["tray"]
        eff_dx = (max(tray_p.delta_x[0], x_min - tray_nom_pos[0]),
                  min(tray_p.delta_x[1], x_max - tray_nom_pos[0]))
        eff_dy = (max(tray_p.delta_y[0], y_min - tray_nom_pos[1]),
                  min(tray_p.delta_y[1], y_max - tray_nom_pos[1]))
        tray_new_pos = tray_nom_pos + np.array([
            rng.uniform(*eff_dx),
            rng.uniform(*eff_dy),
            rng.uniform(*tray_p.delta_z),
        ])
        q_yaw = _quat_from_yaw(rng.uniform(*tray_p.delta_yaw))
        tray_new_quat = _quat_mul(q_yaw, tray_nom_quat)
        states["tray"] = {"pos": tray_new_pos.tolist(), "quat": tray_new_quat.tolist()}

        # Offsets and per-mug deltas are tray-local, then rotated by the
        # sampled tray yaw so every active mug stays in the tray footprint.
        tray_nom_yaw = _yaw_from_quat(tray_nom_quat)
        tray_new_yaw = _yaw_from_quat(tray_new_quat)
        c_new, s_new = float(np.cos(tray_new_yaw)), float(np.sin(tray_new_yaw))

        def to_world_xy(local_xy: np.ndarray) -> np.ndarray:
            return np.array(
                [c_new * local_xy[0] - s_new * local_xy[1], s_new * local_xy[0] + c_new * local_xy[1]],
                dtype=np.float64,
            )

        mug_perturbations = [p for p in self.perturbations if p.joint_name != "tray"]
        mug_variants = list(
            getattr(self, "_current_mug_variants", [_MUG_DEFAULT_VARIANT] * len(mug_perturbations))
        )
        scale_states = getattr(self, "_current_scale_states", {})
        tray_floor_z = tray_new_pos[2] + _MUG_FLIP_TRAY_FLOOR_Z_OFFSET

        for instance_index, p in enumerate(mug_perturbations):
            if p.joint_name not in nominals:
                continue
            _mug_nom_pos, mug_nom_quat = nominals[p.joint_name]
            local_yaw = rng.uniform(*p.delta_yaw)
            variant_name = (
                mug_variants[instance_index]
                if instance_index < len(mug_variants)
                else _MUG_DEFAULT_VARIANT
            )
            scale_factor = float(scale_states.get(p.joint_name, 1.0))
            half_x, half_y, min_z, _max_z = _mug_flip_compiled_metadata(variant_name)
            half_x *= scale_factor
            half_y *= scale_factor
            min_z *= scale_factor
            extent_x, extent_y = self._rotated_mug_half_extents(half_x, half_y, local_yaw)

            slot_centers = self._mug_slot_centers_by_count[len(mug_perturbations)]
            local_xy = np.asarray(slot_centers[instance_index], dtype=np.float64) + np.array(
                [rng.uniform(*p.delta_x), rng.uniform(*p.delta_y)],
                dtype=np.float64,
            )
            local_xy = self._clamp_tray_local_xy(local_xy, extent_x, extent_y)
            world_xy = to_world_xy(local_xy)
            mug_new_z = tray_floor_z + _MUG_FLIP_SPAWN_CLEARANCE_M - min_z + rng.uniform(*p.delta_z)
            mug_new_pos = np.array(
                [tray_new_pos[0] + world_xy[0], tray_new_pos[1] + world_xy[1], mug_new_z],
                dtype=np.float64,
            )
            q_yaw_mug = _quat_from_yaw((tray_new_yaw - tray_nom_yaw) + local_yaw)
            states[p.joint_name] = {
                "pos": mug_new_pos.tolist(),
                "quat": _quat_mul(q_yaw_mug, mug_nom_quat).tolist(),
            }

        return states




__all__ = ["MugFlipRandomizer", "MugTreeRandomizer", "MugVariantRandomizer"]
