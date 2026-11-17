"""Shared randomization state, math helpers, and base sampler."""

from __future__ import annotations

import copy
import logging
import xml.etree.ElementTree as _ET
from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerturbRange:
    """Translation and orientation perturbation range for one object.

    For free-jointed objects set ``joint_name`` to the joint name and leave
    ``fixed_body=False`` (default).  For fixed bodies that should be moved by
    writing directly to ``model.body_pos`` / ``model.body_quat``, set
    ``fixed_body=True`` and use the body name in ``joint_name``.

    Deltas are relative to the object's nominal position at the time
    ``randomize()`` is called (i.e. after ``mj_resetData``), so they remain
    valid even if the XML default positions change.
    """

    joint_name: str
    delta_x: tuple[float, float]       # (min, max) metres
    delta_y: tuple[float, float]       # (min, max) metres
    delta_z: tuple[float, float] = field(default=(0.0, 0.0))   # stay on table
    delta_roll: tuple[float, float] = field(default=(0.0, 0.0))
    delta_pitch: tuple[float, float] = field(default=(0.0, 0.0))
    delta_yaw: tuple[float, float] = field(default=(-np.pi, np.pi))
    fixed_body: bool = False  # if True, treat joint_name as a body name


@dataclass
class ScalePerturbRange:
    """Uniform multiplicative object-scale perturbation for one movable object."""

    target_name: str
    scale_factor: tuple[float, float] = field(default=(0.95, 1.05))


@dataclass
class RandomizationState:
    """Serialisable randomization outcome.

    Stores absolute positions and quaternions — fully self-contained for
    replay without re-running the sampler.  The ``seed`` field is retained
    for audit / debugging only; replay always uses ``object_states`` directly.
    """

    seed: int
    object_states: dict[str, dict[str, list[float]]]
    scale_states: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # e.g. {"bottle_1_joint": {"pos": [x,y,z], "quat": [w,x,y,z]}, ...}

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "object_states": self.object_states,
            "scale_states": self.scale_states,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RandomizationState":
        return cls(
            seed=d["seed"],
            object_states=d["object_states"],
            scale_states=d.get("scale_states", {}),
            metadata=d.get("metadata", {}),
        )


class RandomizationSamplingError(RuntimeError):
    """Raised when a randomizer cannot find a valid startup state."""



def _quat_from_yaw(yaw: float) -> np.ndarray:
    """wxyz quaternion for a rotation of ``yaw`` radians about world Z."""
    return _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """wxyz quaternion for a rotation of ``angle`` radians about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / norm
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    return np.array(
        [np.cos(half_angle), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=np.float64,
    )


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions: result = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _yaw_from_quat(q: np.ndarray) -> float:
    """Return the world-Z yaw for a wxyz quaternion."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _sample_orientation_delta(
    perturbation: PerturbRange,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a quaternion delta from roll/pitch/yaw ranges."""
    q_roll = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), rng.uniform(*perturbation.delta_roll))
    q_pitch = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), rng.uniform(*perturbation.delta_pitch))
    q_yaw = _quat_from_yaw(rng.uniform(*perturbation.delta_yaw))
    return _quat_mul(_quat_mul(q_yaw, q_pitch), q_roll)


def _parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split()]


def _format_float_list(values: list[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _scale_numeric_attr(elem: _ET.Element, attr: str, factor: float) -> None:
    value = elem.get(attr)
    if not value:
        return
    elem.set(attr, _format_float_list([part * factor for part in _parse_float_list(value)]))


def _scaled_mesh_attr(scale_value: str | None, factor: float) -> str:
    if not scale_value:
        parts = [1.0, 1.0, 1.0]
    else:
        parts = _parse_float_list(scale_value)
        if len(parts) == 1:
            parts = parts * 3
    return _format_float_list([part * factor for part in parts])


def _safe_scale_suffix(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    return safe or "scaled"


def _find_body_for_scale_target(root: _ET.Element, target_name: str) -> _ET.Element | None:
    for body in root.iter("body"):
        for child in body:
            if child.tag in {"joint", "freejoint"} and child.get("name") == target_name:
                return body
    for body in root.iter("body"):
        if body.get("name") == target_name:
            return body
    return None


def _clone_scaled_mesh(
    *,
    asset_elem: _ET.Element,
    mesh_assets: dict[str, _ET.Element],
    mesh_name: str,
    factor: float,
    target_name: str,
    cache: dict[str, str],
) -> str:
    if mesh_name in cache:
        return cache[mesh_name]

    mesh_elem = mesh_assets.get(mesh_name)
    if mesh_elem is None:
        return mesh_name

    new_name = f"{mesh_name}__scaled__{_safe_scale_suffix(target_name)}"
    suffix = 1
    while new_name in mesh_assets:
        suffix += 1
        new_name = f"{mesh_name}__scaled__{_safe_scale_suffix(target_name)}_{suffix}"

    cloned = copy.deepcopy(mesh_elem)
    cloned.set("name", new_name)
    cloned.set("scale", _scaled_mesh_attr(mesh_elem.get("scale"), factor))
    asset_elem.append(cloned)
    mesh_assets[new_name] = cloned
    cache[mesh_name] = new_name
    return new_name


def _scale_body_subtree(
    *,
    body: _ET.Element,
    factor: float,
    target_name: str,
    asset_elem: _ET.Element,
    mesh_assets: dict[str, _ET.Element],
) -> None:
    mesh_cache: dict[str, str] = {}
    for elem in body.iter():
        if elem is not body and elem.tag == "body":
            _scale_numeric_attr(elem, "pos", factor)
            continue

        if elem.tag == "geom":
            _scale_numeric_attr(elem, "pos", factor)
            _scale_numeric_attr(elem, "size", factor)
            _scale_numeric_attr(elem, "fromto", factor)
            mesh_name = elem.get("mesh")
            if mesh_name:
                elem.set(
                    "mesh",
                    _clone_scaled_mesh(
                        asset_elem=asset_elem,
                        mesh_assets=mesh_assets,
                        mesh_name=mesh_name,
                        factor=factor,
                        target_name=target_name,
                        cache=mesh_cache,
                    ),
                )
            continue

        if elem.tag == "site":
            _scale_numeric_attr(elem, "pos", factor)
            _scale_numeric_attr(elem, "size", factor)
            continue

        if elem.tag == "inertial":
            _scale_numeric_attr(elem, "pos", factor)


def _apply_object_scales_to_scene_xml(xml: str, scale_states: dict[str, float]) -> str:
    if not scale_states:
        return xml

    root = _ET.fromstring(xml)
    asset_elem = root.find("asset")
    if asset_elem is None:
        return xml
    mesh_assets = {
        mesh.get("name", ""): mesh
        for mesh in asset_elem.findall("mesh")
        if mesh.get("name")
    }

    for target_name, factor in scale_states.items():
        if abs(factor - 1.0) < 1e-9:
            continue
        body = _find_body_for_scale_target(root, target_name)
        if body is None:
            logger.warning("Scale target '%s' not found in scene XML — skipping", target_name)
            continue
        _scale_body_subtree(
            body=body,
            factor=factor,
            target_name=target_name,
            asset_elem=asset_elem,
            mesh_assets=mesh_assets,
        )

    return _ET.tostring(root, encoding="unicode")


def _resolve_scene_xml_paths(xml: str, base_dir: _Path | None) -> str:
    if base_dir is None:
        return xml

    root = _ET.fromstring(xml)
    compiler = root.find("compiler")
    if compiler is not None:
        for attr in ("meshdir", "texturedir"):
            value = compiler.get(attr)
            if value and not _Path(value).is_absolute():
                compiler.set(attr, str((base_dir / value).resolve()))
    return _ET.tostring(root, encoding="unicode")


def _body_subtree_xy_keepout_discs(
    model: Any,
    data: Any,
    root_body_id: int,
) -> list[tuple[np.ndarray, float]]:
    """Return XY geom discs relative to the root body origin.

    Each disc is represented as ``(offset_xy, radius)`` using the geom centre
    relative to the root body plus MuJoCo's bounding-sphere radius. For sweep
    randomization this provides a cheap footprint proxy that is far more
    accurate than a single body-origin clearance.
    """
    if root_body_id < 0:
        return []

    root_xy = np.asarray(data.xpos[root_body_id][:2], dtype=np.float64)
    discs: list[tuple[np.ndarray, float]] = []
    for geom_id in range(model.ngeom):
        geom_body_id = int(model.geom_bodyid[geom_id])
        if int(model.body_rootid[geom_body_id]) != root_body_id:
            continue
        geom_xy = np.asarray(data.geom_xpos[geom_id][:2], dtype=np.float64)
        discs.append((geom_xy - root_xy, float(model.geom_rbound[geom_id])))
    return discs


class SceneRandomizer:
    """Base class for scene object randomization with two-stage rejection sampling.

    Subclasses only need to define ``perturbations`` (and optionally override
    ``min_clearance_m``, ``max_tries``, or ``table_bounds``).  The base class
    handles reading nominal positions, sampling, collision checking, and state
    serialisation.

    ``table_bounds`` is checked on every sample as Stage 0 (before the cheap
    pairwise distance check).  It is expressed as absolute world XY coordinates:
    ``(x_min, x_max, y_min, y_max)``.  Objects whose absolute position falls
    outside these bounds are immediately rejected, so delta ranges can be set
    generously without risk of objects falling off the edge.

    The default matches the standard sim table:
      centre (0.6, 0), half-extents (0.2975, 0.65) minus a 0.06 m edge margin.
    """

    perturbations: list[PerturbRange] = []
    max_tries: int = 200
    min_clearance_m: float = 0.03  # minimum XY centre-to-centre distance
    size_perturbations: list[ScalePerturbRange] = []
    # Absolute XY workspace limits — overrideable per task if the table differs.
    table_bounds: tuple[float, float, float, float] = (0.36, 0.82, -0.55, 0.55)
    reject_arm_contacts: bool = True
    arm_root_body_names: tuple[str, ...] = ("left_arm", "right_arm")

    def __init__(self) -> None:
        # Cache fixed-body nominal positions on first read so that subsequent
        # randomizations don't drift (mj_resetData restores data.qpos but not
        # model.body_pos, so we must remember the original XML values ourselves).
        self._fixed_body_nominals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._env_ref = None
        self._base_scene_xml_string: str | None = None
        self._base_scene_xml_dir: _Path | None = None
        self._base_scene_xml_transformed = False
        self._scene_xml_transform_options = None
        self._current_scale_states: dict[str, float] = {}

    def clone(self) -> "SceneRandomizer":
        return type(self)()

    def bind_env(self, env: Any) -> None:
        self._env_ref = env
        self._scene_xml_transform_options = getattr(env, "_scene_xml_transform_options", None)
        if getattr(env, "_scene_xml_string", None):
            self._base_scene_xml_string = env._scene_xml_string
            self._base_scene_xml_dir = getattr(env, "_scene_xml", None)
            if self._base_scene_xml_dir is not None:
                self._base_scene_xml_dir = _Path(self._base_scene_xml_dir).parent
            self._base_scene_xml_transformed = self._scene_xml_transform_options is not None
        else:
            self._base_scene_xml_string = env._scene_xml.read_text()
            self._base_scene_xml_dir = _Path(env._scene_xml).parent
            self._base_scene_xml_transformed = False

    def prepare_reset(
        self, *, request: Any | None = None, randomize: bool = True
    ) -> None:
        """Optional hook for task-owned model state before placement sampling."""
        return None

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        """Sample a collision-free placement, apply it to data, and return state."""
        rng = np.random.default_rng(seed)
        randomize_scales = request.get("randomize_scales", True) if isinstance(request, dict) else True
        scale_states = self._sample_scale_states(rng) if randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        if scale_states:
            self._reload_scene_for_scale_states(scale_states)
            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data

        if not self.perturbations:
            return RandomizationState(seed=seed or 0, object_states={}, scale_states=scale_states)

        self._before_sampling(model, data)

        # Nominal positions come from the current qpos (caller should have
        # called mj_resetData + mj_forward before invoking randomize).
        nominals = self._read_nominals(model, data)

        for attempt in range(self.max_tries):
            states = self._sample_once(nominals, rng)

            # Stage 0: table bounds — free, no geometry queries needed.
            if not self._bounds_ok(states):
                continue

            # Stage 1: fast pairwise XY distance (no mj_forward cost).
            if not self._pairwise_ok(states):
                continue

            # Stage 2: full MuJoCo contact check.
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue

            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
            )

        self._raise_sampling_failure(seed)

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        """Restore a previously saved state without sampling (for replay)."""
        self._current_scale_states = dict(state.scale_states)
        if state.scale_states:
            self._reload_scene_for_scale_states(state.scale_states)
            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        mujoco.mj_forward(model, data)

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        if self.size_perturbations:
            return self.size_perturbations
        return [
            ScalePerturbRange(p.joint_name)
            for p in self.perturbations
            if not p.fixed_body
        ]

    def _sample_scale_states(self, rng: np.random.Generator) -> dict[str, float]:
        if self._env_ref is None:
            return {}
        return {
            target.target_name: float(rng.uniform(*target.scale_factor))
            for target in self._get_size_perturbations()
        }

    def _before_sampling(self, model: Any, data: Any) -> None:
        """Hook for subclasses that need model/data-derived metadata."""
        return None

    def _scene_xml_for_scale_states(self, scale_states: dict[str, float]) -> str | None:
        if self._base_scene_xml_string is None:
            return None
        xml = self._base_scene_xml_string
        if scale_states:
            xml = _apply_object_scales_to_scene_xml(xml, scale_states)
        return _resolve_scene_xml_paths(xml, self._base_scene_xml_dir)

    def _reload_scene_for_scale_states(self, scale_states: dict[str, float]) -> None:
        if not scale_states:
            return
        if self._env_ref is None:
            logger.warning(
                "%s: scale replay requested without a bound env — skipping scale restoration",
                type(self).__name__,
            )
            return

        xml = self._scene_xml_for_scale_states(scale_states)
        if xml is None:
            return

        preserved_arm_state = self._env_ref._get_reset_arm_state()
        self._env_ref.reload_from_xml(xml)
        mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
        self._env_ref._set_qpos_from_state(preserved_arm_state)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)

    def _read_nominals(
        self, model: Any, data: Any
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Read current pos and quat for each perturbation.

        Free-jointed objects read from ``data.qpos``; fixed bodies read from
        a cache populated on the first call (the XML default values).  We must
        cache because ``_apply_states`` writes to ``model.body_pos`` and
        ``mj_resetData`` does not restore it, which would cause the nominal to
        drift on every reset if we re-read from the model each time.
        """
        # Populate fixed-body cache on first call (before any writes).
        if self._fixed_body_nominals is None:
            self._fixed_body_nominals = {}
            for p in self.perturbations:
                if p.fixed_body:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p.joint_name)
                    if body_id >= 0:
                        self._fixed_body_nominals[p.joint_name] = (
                            model.body_pos[body_id].copy(),
                            model.body_quat[body_id].copy(),
                        )

        nominals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for p in self.perturbations:
            if p.fixed_body:
                if p.joint_name not in self._fixed_body_nominals:
                    logger.warning("Body '%s' not found in model — skipping", p.joint_name)
                    continue
                pos, quat = self._fixed_body_nominals[p.joint_name]
                nominals[p.joint_name] = (pos.copy(), quat.copy())
            else:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p.joint_name)
                if jnt_id < 0:
                    logger.warning("Joint '%s' not found in model — skipping", p.joint_name)
                    continue
                adr = int(model.jnt_qposadr[jnt_id])
                nominals[p.joint_name] = (
                    data.qpos[adr: adr + 3].copy(),
                    data.qpos[adr + 3: adr + 7].copy(),
                )
        return nominals

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        """Draw one set of absolute object placements from nominal + delta.

        The delta ranges are intersected with ``table_bounds`` before sampling
        so every draw is guaranteed to land on the table.  This avoids the
        exponential failure rate of rejection sampling when large delta ranges
        are combined with a bounded workspace.
        """
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}
        for p in self.perturbations:
            if p.joint_name not in nominals:
                continue
            nom_pos, nom_quat = nominals[p.joint_name]

            # Clamp the effective delta range so the absolute position stays
            # within table_bounds regardless of how large the delta is set.
            eff_dx = (
                max(p.delta_x[0], x_min - nom_pos[0]),
                min(p.delta_x[1], x_max - nom_pos[0]),
            )
            eff_dy = (
                max(p.delta_y[0], y_min - nom_pos[1]),
                min(p.delta_y[1], y_max - nom_pos[1]),
            )
            # If nominal is outside bounds (shouldn't happen), sample at 0.
            if eff_dx[0] > eff_dx[1]:
                eff_dx = (0.0, 0.0)
            if eff_dy[0] > eff_dy[1]:
                eff_dy = (0.0, 0.0)

            new_pos = nom_pos + np.array([
                rng.uniform(*eff_dx),
                rng.uniform(*eff_dy),
                rng.uniform(*p.delta_z),
            ])
            new_quat = _quat_mul(_sample_orientation_delta(p, rng), nom_quat)

            states[p.joint_name] = {
                "pos": new_pos.tolist(),
                "quat": new_quat.tolist(),
            }
        return states

    def _apply_states(
        self,
        model: Any,
        data: Any,
        states: dict[str, dict[str, list[float]]],
    ) -> None:
        """Write absolute pos/quat for each object.

        Tries joint lookup first (free-jointed objects → ``data.qpos``).
        Falls back to body lookup (fixed bodies → ``model.body_pos/quat``).
        """
        for name, s in states.items():
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                adr = int(model.jnt_qposadr[jnt_id])
                data.qpos[adr: adr + 3] = s["pos"]
                data.qpos[adr + 3: adr + 7] = s["quat"]
            else:
                # Fixed body — write directly into model geometry.
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    model.body_pos[body_id] = s["pos"]
                    model.body_quat[body_id] = s["quat"]

    def _bounds_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        """Return False if any object's absolute XY position is outside table_bounds."""
        x_min, x_max, y_min, y_max = self.table_bounds
        for s in states.values():
            x, y = s["pos"][0], s["pos"][1]
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return False
        return True

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        """Return False if any two objects are closer than min_clearance_m in XY."""
        if self.min_clearance_m <= 0.0 or len(states) < 2:
            return True
        positions = [np.array(s["pos"][:2]) for s in states.values()]
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < self.min_clearance_m:
                    return False
        return True

    def _contacts_ok(self, model: Any, data: Any) -> bool:
        """Return False on randomized-object contacts or object-arm contacts."""
        if not self._object_arm_contacts_ok(model, data):
            return False

        object_body_ids = self._randomized_object_body_ids(model)
        obj_root_ids: set[int] = set()
        for body_id in object_body_ids:
            obj_root_ids.add(int(model.body_rootid[body_id]))

        for c in range(data.ncon):
            contact = data.contact[c]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            r1 = int(model.body_rootid[b1])
            r2 = int(model.body_rootid[b2])
            if r1 != r2 and r1 in obj_root_ids and r2 in obj_root_ids:
                return False
        return True

    def _raise_sampling_failure(self, seed: int | None) -> None:
        seed_text = "None" if seed is None else str(seed)
        raise RandomizationSamplingError(
            f"{type(self).__name__}: no valid startup placement found after "
            f"{self.max_tries} tries for seed={seed_text}"
        )

    def _randomized_object_body_ids(self, model: Any) -> set[int]:
        body_ids: set[int] = set()
        for p in self.perturbations:
            if p.fixed_body:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p.joint_name)
                if body_id >= 0:
                    body_ids.add(int(body_id))
            else:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p.joint_name)
                if jnt_id >= 0:
                    body_ids.add(int(model.jnt_bodyid[jnt_id]))
        return body_ids

    def _arm_root_body_ids(self, model: Any) -> set[int]:
        body_ids: set[int] = set()
        for body_name in self.arm_root_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                body_ids.add(int(body_id))
        return body_ids

    def _object_arm_contacts_ok(self, model: Any, data: Any) -> bool:
        if not self.reject_arm_contacts:
            return True
        object_body_ids = self._randomized_object_body_ids(model)
        arm_root_ids = self._arm_root_body_ids(model)
        if not object_body_ids or not arm_root_ids:
            return True
        for c in range(data.ncon):
            contact = data.contact[c]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            if self._contact_is_object_arm_pair(
                model,
                b1,
                b2,
                object_body_ids=object_body_ids,
                arm_root_ids=arm_root_ids,
            ):
                return False
        return True

    @staticmethod
    def _is_body_descendant(model: Any, body_id: int, ancestor_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return body_id == ancestor_id

    @classmethod
    def _body_in_any_subtree(cls, model: Any, body_id: int, root_ids: set[int]) -> bool:
        return any(cls._is_body_descendant(model, body_id, root_id) for root_id in root_ids)

    @classmethod
    def _contact_is_object_arm_pair(
        cls,
        model: Any,
        body_1: int,
        body_2: int,
        *,
        object_body_ids: set[int],
        arm_root_ids: set[int],
    ) -> bool:
        body_1_is_object = cls._body_in_any_subtree(model, body_1, object_body_ids)
        body_2_is_object = cls._body_in_any_subtree(model, body_2, object_body_ids)
        if body_1_is_object == body_2_is_object:
            return False
        body_1_is_arm = cls._body_in_any_subtree(model, body_1, arm_root_ids)
        body_2_is_arm = cls._body_in_any_subtree(model, body_2, arm_root_ids)
        return (body_1_is_object and body_2_is_arm) or (body_2_is_object and body_1_is_arm)


# Preset mug color palette (RGBA, alpha=1.0).  Includes the original green so
# the default appearance is part of the distribution.
_MUG_COLOR_PALETTE: list[tuple[float, float, float, float]] = [
    (0.172, 0.780, 0.435, 1.0),  # original green
    (0.850, 0.325, 0.098, 1.0),  # red-orange
    (0.929, 0.694, 0.125, 1.0),  # amber
    (0.494, 0.184, 0.557, 1.0),  # purple
    (0.301, 0.745, 0.933, 1.0),  # sky blue
    (0.635, 0.078, 0.184, 1.0),  # dark red
    (0.047, 0.482, 0.863, 1.0),  # blue
    (0.960, 0.960, 0.960, 1.0),  # near-white
    (0.173, 0.173, 0.173, 1.0),  # near-black
]

_TRAY_COLOR_PALETTE: list[tuple[float, float, float, float]] = [
    (0.000, 0.188, 1.000, 1.0),  # original blue
    (0.050, 0.580, 0.420, 1.0),  # teal
    (0.950, 0.420, 0.120, 1.0),  # orange
    (0.620, 0.220, 0.780, 1.0),  # purple
    (0.930, 0.820, 0.160, 1.0),  # yellow
    (0.780, 0.120, 0.180, 1.0),  # red
    (0.120, 0.140, 0.160, 1.0),  # charcoal
    (0.880, 0.900, 0.880, 1.0),  # light gray
]

_WATER_BOTTLE_BIN_COLOR_PALETTE: list[tuple[float, float, float, float]] = [
    (0.100, 0.100, 0.250, 1.0),  # original navy
    (0.950, 0.420, 0.120, 1.0),  # orange
    (0.080, 0.500, 0.360, 1.0),  # green
    (0.610, 0.200, 0.760, 1.0),  # purple
    (0.820, 0.130, 0.180, 1.0),  # red
    (0.160, 0.170, 0.180, 1.0),  # charcoal
    (0.870, 0.890, 0.860, 1.0),  # light gray
]

# Probability of applying a random color (vs. keeping the XML default) each episode.
_COLOR_RANDOMIZE_PROB = 1.0


def _apply_mat_color(
    model: Any, mat_name: str, rgba: tuple[float, float, float, float]
) -> None:
    """Set a MuJoCo material's RGBA at runtime."""
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
    if mat_id >= 0:
        model.mat_rgba[mat_id] = rgba
    else:
        logger.warning("Material '%s' not found in model — skipping color randomization", mat_name)


__all__ = [name for name in globals() if not name.startswith('__')]
