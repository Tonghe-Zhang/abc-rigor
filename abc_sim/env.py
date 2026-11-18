"""MuJoCo simulation environment for bimanual YAM robot."""

from __future__ import annotations

import copy
import math
import os
import string
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from abc_sim.background import BackgroundConfig, BackgroundController, background_texture_ids
from abc_sim.config import RobotSystemConfig
from abc_sim.physics import PhysicsProfile, apply_physics_profile, resolve_physics_profile
from abc_sim.rendering.live import LiveRenderBackend, create_live_camera_provider
from abc_sim.task_eval import (
    PhysicsObservedTaskEvaluator,
    TaskEvalResult,
    make_task_evaluator,
)
from abc_sim.task_eval.base import EpisodeTaskEvaluator, TaskEvaluator
from abc_sim.task_registry import (
    DEFAULT_SCENE_XML,
    SCENE_XMLS,
    get_task_randomizer,
    resolve_task,
)
from abc_sim.task_runtime import make_task_runtime
from abc_sim.task_runtime.base import TaskProgressRuntime, TaskRuntime
from abc_sim.task_specs import SimTaskSpec, maybe_get_task_spec

if TYPE_CHECKING:
    from abc_sim.randomization.core import RandomizationState, SceneRandomizer

# Backwards-compatible aliases for older imports.
_SCENE_XMLS = dict(SCENE_XMLS)
_DEFAULT_SCENE_XML = DEFAULT_SCENE_XML

# Gripper actuator ctrl range max (from menagerie model)
_GRIPPER_CTRL_MAX = 0.0475
_OBS_PROMPT_CHARSET = string.ascii_letters + string.digits + string.punctuation + " "


def project_policy_state(
    qpos: np.ndarray,
    qpos_indices: list[int],
    gripper_indices: list[int],
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Project MuJoCo qpos into the 14D policy-space state vector."""
    state = np.zeros(len(qpos_indices), dtype=dtype)
    gripper_set = set(gripper_indices)
    for i, qpos_idx in enumerate(qpos_indices):
        val = qpos[qpos_idx]
        if i in gripper_set:
            val = np.clip(val / _GRIPPER_CTRL_MAX, 0.0, 1.0)
        state[i] = val
    return state


def project_policy_state_batch(
    qpos_batch: np.ndarray,
    qpos_indices: list[int],
    gripper_indices: list[int],
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Project a batch of MuJoCo qpos vectors into policy-space states."""
    qpos_batch = np.asarray(qpos_batch)
    if qpos_batch.ndim != 2:
        raise ValueError(f"Expected batched qpos with shape (B, nq), got {qpos_batch.shape}")

    state = np.asarray(qpos_batch[:, qpos_indices], dtype=dtype)
    if gripper_indices:
        state[:, gripper_indices] = np.clip(
            state[:, gripper_indices] / _GRIPPER_CTRL_MAX,
            0.0,
            1.0,
        )
    return state


class MuJoCoYAMEnv(gym.Env):
    """MuJoCo-based simulation of the bimanual YAM robot.

    Actions and observations use the same 14D format as the real robot:
    [left_j1..6, left_grip, right_j1..6, right_grip].
    """

    _background: BackgroundController | None = None  # class default: disabled
    _background_pending = False  # a draw not yet uploaded to the live renderer

    def __init__(
        self,
        config: RobotSystemConfig,
        chunk_dim: int = 30,
        prompt: str | None = None,
        render_cameras: bool = True,
        camera_backend: LiveRenderBackend = "mujoco",
        camera_gpu_id: int | None = None,
        camera_height: int = 480,
        camera_width: int = 640,
        physics_dt: float = 0.002,
        control_decimation: int = 17,  # 0.002 * 17 ≈ 0.034s ≈ 30Hz
        scene_xml: str | Path | None = None,
        max_episode_steps: int | None = None,
        terminate_on_success: bool = False,
        wrist_fov: float = 58.0,
        physics_profile: str | PhysicsProfile | None = None,
        background: BackgroundConfig | None = None,
    ):
        super().__init__()
        # Opt-in visual background variations (abc_sim.background); None keeps the scene unchanged.
        self._background = BackgroundController(background) if background is not None else None
        self.config = config
        self.chunk_dim = chunk_dim
        self._prompt_override = prompt
        self.prompt = "fold the towel" if prompt is None else prompt
        self._render_cameras_flag = render_cameras
        self._camera_backend: LiveRenderBackend = camera_backend
        self._camera_gpu_id = camera_gpu_id
        self._camera_height = camera_height
        self._camera_width = camera_width
        if not 0 < wrist_fov < 180:
            raise ValueError("wrist_fov must be between 0 and 180 degrees.")
        self._wrist_fov = wrist_fov
        self._wrist_camera_positions: weakref.WeakKeyDictionary[
            mujoco.MjModel, np.ndarray
        ] = weakref.WeakKeyDictionary()
        self._physics_dt = physics_dt
        self.physics_profile = resolve_physics_profile(physics_profile)
        self._control_decimation = control_decimation
        self._scene_xml = Path(scene_xml) if scene_xml else _DEFAULT_SCENE_XML
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be >= 1 or None, got {max_episode_steps}"
            )
        self.max_episode_steps = max_episode_steps
        self.terminate_on_success = bool(terminate_on_success)
        self._scene_xml_string: str | None = None  # set directly for in-memory XML
        self._task_request: str | SimTaskSpec = ""
        self._task_reset_request: dict[str, Any] | None = None
        self._task: str = ""  # set by make_env after construction
        self._task_spec: SimTaskSpec | None = None
        self._task_evaluator = None
        self._step_evaluator: TaskEvaluator | None = None
        self._uses_runtime_progress = False
        self._step_updates = False
        self._step_requires_position_refresh = False
        self._step_update_interval = 1
        self._step_task_eval: TaskEvalResult | None = None
        self._physics_evaluator: PhysicsObservedTaskEvaluator | None = None
        self._physics_observer = None
        self._task_runtime = None
        self._arm_state_initialized = False

        # Set by sampling or bound replay of an initial episode.
        self._last_randomization: Any = None
        self._episode_restorer: SceneRandomizer | None = None

        self.camera_names = list(config.cameras.keys())
        self.robot_names = list(config.robots.keys())
        self.single_timestep_action_dim = 7 * len(config.robots)  # 14
        self.action_dim = self.single_timestep_action_dim
        self.chunk_action_dim = self.single_timestep_action_dim * self.chunk_dim

        self.observation_space = spaces.Dict(
            {
                "images": spaces.Dict(
                    {
                        name: spaces.Box(
                            low=0,
                            high=255,
                            shape=(
                                3,
                                self._camera_height,
                                self._camera_width,
                            ),
                            dtype=np.uint8,
                        )
                        for name in self.camera_names
                    }
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.single_timestep_action_dim,),
                    dtype=np.float32,
                ),
                "masks": spaces.Dict(
                    {name: spaces.Discrete(2) for name in self.camera_names}
                ),
                "prompt": spaces.Text(
                    max_length=4096,
                    min_length=0,
                    charset=_OBS_PROMPT_CHARSET,
                ),
                "camera_timestamps": spaces.Dict(
                    {
                        name: spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(),
                            dtype=np.float64,
                        )
                        for name in self.camera_names
                    }
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.single_timestep_action_dim,),
            dtype=np.float32,
        )
        self.chunk_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.chunk_dim, self.single_timestep_action_dim),
            dtype=np.float32,
        )

        self.setup_model()
        self.cur_step = 0

    def setup_model(self):
        if self._scene_xml_string is not None:
            model = self._compile_scene_xml(self._scene_xml_string)
        elif self._background is not None:
            model = self._compile_scene_xml(
                self._scene_xml.read_text(), base_dir=self._scene_xml.parent
            )
        else:
            model = mujoco.MjModel.from_xml_path(str(self._scene_xml))
        self._bind_model(model)

    def background_xml(self, xml_string: str) -> str:
        """The scene XML with background texture slots when backgrounds are enabled, else unchanged."""
        if self._background is None:
            return xml_string
        return self._background.xml(xml_string)

    def _compile_scene_xml(
        self, xml_string: str, *, base_dir: str | Path | None = None
    ) -> mujoco.MjModel:
        """Compile a candidate with asset paths relative to the bundled models."""
        if self._background is not None:
            xml_string = self._background.xml(xml_string)
        prev_cwd = os.getcwd()
        os.chdir(base_dir or Path(__file__).parent / "models")
        try:
            return mujoco.MjModel.from_xml_string(xml_string)
        finally:
            os.chdir(prev_cwd)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        apply_physics_profile(model, self.physics_profile)
        if self._background is not None:
            self._background.apply(model)  # pixels only: before any renderer uploads textures
            self._background_pending = False  # the renderers created below upload the current draw
        self.model = model
        self._step_task_eval = None
        self.model.opt.timestep = self._physics_dt
        self._configure_model_cameras()
        self.data = mujoco.MjData(self.model)
        self._arm_state_initialized = False
        if self._task_runtime is not None:
            self._task_runtime.bind(self.model, self.data)
        self._camera_provider = None

        if self._render_cameras_flag:
            if self._camera_backend == "mujoco":
                self.renderer = mujoco.Renderer(
                    self.model,
                    height=self._camera_height,
                    width=self._camera_width,
                )
            else:
                self._create_camera_provider()

        self._build_index_maps()

    def _create_camera_provider(self) -> None:
        self._camera_provider = create_live_camera_provider(
            model=self.model,
            data=self.data,
            backend=self._camera_backend,
            width=self._camera_width,
            height=self._camera_height,
            gpu_id=self._camera_gpu_id,
            camera_names=tuple(self.camera_names),
        )
        if self._background is not None and self._camera_backend == "mjwarp":
            self._upload_mjwarp_background(list(background_texture_ids(self.model).values()))

    def _upload_mjwarp_background(self, texture_ids: list[int]) -> None:
        """MJWarp copies textures at context creation and ignores emission: upload with the wall gain."""
        slot_of = {tid: k for k, tid in background_texture_ids(self.model).items()}
        gains = {tid: self._background.mjwarp_gain(slot_of[tid]) for tid in texture_ids}
        self._camera_provider.update_textures(self.model, texture_ids, gains)

    def _configure_model_cameras(self) -> None:
        """Apply constructor camera calibration before allocating renderers."""
        if self._wrist_fov == 58.0:
            return
        tan_baseline = math.tan(math.radians(58.0 / 2))
        tan_new = math.tan(math.radians(self._wrist_fov / 2))
        clearance_offset = max(0.0, (tan_new - tan_baseline) / tan_baseline) * 0.025
        positions = self._wrist_camera_positions.get(self.model)
        if positions is None:
            positions = self.model.cam_pos.copy()
            self._wrist_camera_positions[self.model] = positions
        for name in ("left", "right"):
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if camera_id < 0:
                continue
            self.model.cam_fovy[camera_id] = self._wrist_fov
            self.model.cam_pos[camera_id] = positions[camera_id]
            self.model.cam_pos[camera_id, 1] -= clearance_offset

    def reload_from_xml(
        self,
        xml_string: str,
        *,
        validate_model: Callable[[mujoco.MjModel], None] | None = None,
    ) -> None:
        """Compile and validate a replacement before changing the live scene.

        ``validate_model`` is a read-only task preflight on the compiled candidate.
        Compilation and validation failure leave live model, data, renderers and
        XML unchanged. A successful candidate is bound directly, without another
        compilation or model copy. Callers initialize its fresh data afterward.
        """
        model = self._compile_scene_xml(xml_string)
        if validate_model is not None:
            validate_model(model)
        if isinstance(self._step_evaluator, EpisodeTaskEvaluator):
            self._step_evaluator.validate_model(model)
        self._scene_xml_string = xml_string
        self._close_camera_backend()
        self._bind_model(model)
        self._bind_task_evaluator(self._task_spec, self._task_runtime)

    def copy_model(self, model: mujoco.MjModel) -> mujoco.MjModel:
        """Copy a compiled model, preserving its original camera mounts.

        Environment-owned model caches use this method so later reloads apply
        constructor calibration relative to the original mounts. Calibration
        records remain weakly held and expire with their corresponding models.
        """
        replacement = copy.deepcopy(model)
        positions = self._wrist_camera_positions.get(model)
        if positions is not None:
            self._wrist_camera_positions[replacement] = positions.copy()
        return replacement

    def reload_from_model(self, model: mujoco.MjModel) -> None:
        """Swap in a copy of an already-compiled MuJoCo model."""
        if isinstance(self._step_evaluator, EpisodeTaskEvaluator):
            self._step_evaluator.validate_model(model)
        replacement = self.copy_model(model)
        self._scene_xml_string = None
        self._close_camera_backend()
        self._bind_model(replacement)
        self._bind_task_evaluator(self._task_spec, self._task_runtime)

    def _bind_task_evaluator(
        self, task_spec: SimTaskSpec | None, runtime: TaskRuntime | None
    ) -> None:
        """Bind optional scoring and the runtime's shared progress requirements."""
        evaluator = make_task_evaluator(self.model, task_spec)
        progress_runtime = isinstance(runtime, TaskProgressRuntime)
        source = evaluator
        if progress_runtime:
            source = runtime.make_progress_source(
                self.model, task_spec, evaluator
            )
            if not isinstance(source, EpisodeTaskEvaluator):
                raise ValueError("Runtime progress requires an episode-aware source.")
            if evaluator is not None and source is not evaluator:
                raise ValueError("Runtime progress and task scoring must share one source.")
        self._task_runtime = runtime
        self._task_evaluator = evaluator
        self._step_evaluator = source
        self._uses_runtime_progress = progress_runtime
        self._physics_evaluator = (
            evaluator
            if isinstance(evaluator, PhysicsObservedTaskEvaluator)
            else None
        )
        self._physics_observer = None
        self._step_updates = progress_runtime or bool(
            getattr(source, "requires_step_updates", False)
        )
        self._step_requires_position_refresh = self._step_updates and hasattr(
            source, "evaluate_model_data"
        )
        self._step_update_interval = (
            self._control_decimation
            if not progress_runtime and getattr(source, "evaluate_at_control_rate", False)
            else 1
        )
        self._step_task_eval = None

    def _refresh_background_textures(self, texture_ids: list[int]) -> None:
        """Upload rewritten background textures to the live renderer."""
        if not texture_ids or not self._render_cameras_flag:
            return
        if self._camera_provider is None:  # CPU renderer: re-upload just these textures
            self.renderer._gl_context.make_current()
            for tid in texture_ids:
                mujoco.mjr_uploadTexture(self.model, self.renderer._mjr_context, tid)
            return
        if self._camera_backend == "mjwarp":  # swap the textures in the live render context
            self._upload_mjwarp_background(texture_ids)
            return
        # Other backends copy textures when they are created: rebuild the camera backend.
        self._close_camera_backend()
        self._create_camera_provider()

    def set_background(self, background_id: int | None = None) -> dict[str, Any] | None:
        """Draw a background from the background stream (or pin ``background_id``) and show it."""
        if self._background is None:
            raise RuntimeError("Backgrounds are disabled: create the env with randomize_background=True.")
        self._background.draw(background_id)
        self._background_pending = True
        self._show_pending_background()
        return self._background.info()

    def _show_pending_background(self) -> None:
        if self._background_pending:
            self._refresh_background_textures(self._background.apply(self.model))
            self._background_pending = False

    @property
    def background_info(self) -> dict[str, Any] | None:
        """The current background draw (id, wallpaper and table instance), or None when disabled."""
        return None if self._background is None else self._background.info()

    def _reset_camera_backend(self) -> None:
        if self._camera_provider is not None:
            self._camera_provider.reset()

    def _close_camera_backend(self) -> None:
        if self._camera_provider is not None:
            self._camera_provider.close()
            self._camera_provider = None
        if hasattr(self, "renderer"):
            # mujoco.Renderer.close() deletes its GL objects in whichever context is current.
            self.renderer._gl_context.make_current()
            self.renderer.close()
            del self.renderer

    def _build_index_maps(self):
        """Build mappings from 14D state/action to MuJoCo qpos/ctrl indices."""
        self._qpos_indices: list[int] = []
        self._ctrl_indices: list[int] = []
        self._gripper_indices: list[int] = []  # which of the 14 are grippers
        robot_roots: set[int] = set()

        idx = 0
        for robot_name in self.robot_names:
            for j in range(1, 7):
                joint_name = f"{robot_name}_joint{j}"
                jnt_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                self._qpos_indices.append(self.model.jnt_qposadr[jnt_id])
                robot_roots.add(
                    int(self.model.body_rootid[self.model.jnt_bodyid[jnt_id]])
                )

                act_name = f"{robot_name}_joint{j}"
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
                )
                self._ctrl_indices.append(act_id)
                idx += 1

            finger_name = f"{robot_name}_left_finger"
            finger_jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, finger_name
            )
            self._qpos_indices.append(self.model.jnt_qposadr[finger_jnt_id])
            robot_roots.add(
                int(self.model.body_rootid[self.model.jnt_bodyid[finger_jnt_id]])
            )

            grip_act_name = f"{robot_name}_gripper"
            grip_act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, grip_act_name
            )
            self._ctrl_indices.append(grip_act_id)
            self._gripper_indices.append(idx)
            idx += 1

        self._gripper_set = set(self._gripper_indices)
        self._robot_qpos_slices: dict[str, slice] = {}
        for joint_id in range(self.model.njnt):
            if self.model.body_rootid[self.model.jnt_bodyid[joint_id]] not in robot_roots:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None:
                raise ValueError("Robot joints must be named for scene reloads.")
            start = int(self.model.jnt_qposadr[joint_id])
            stop = (
                int(self.model.jnt_qposadr[joint_id + 1])
                if joint_id + 1 < self.model.njnt
                else self.model.nq
            )
            self._robot_qpos_slices[name] = slice(start, stop)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        randomize: bool = True,
    ):
        randomization_request = (options or {}).get("randomization")
        if self._task_spec is not None:
            randomization_request = self._task_spec.resolve_reset_request(
                randomization_request
            )
        retained = None
        if not randomize and self._task_runtime is not None:
            retained_state = getattr(self._task_runtime, "retained_randomization", None)
            if retained_state is not None:
                retained = retained_state()
        if retained is not None:
            restorer = self._episode_restorer
            if restorer is None or restorer._env_ref is not self:
                raise RuntimeError("Retained episode requires its bound scene restorer.")
            restorer.apply(
                self.model, self.data, retained,
                keep_arm_state=(options or {}).get("keep_arm_state", False),
            )
            super().reset(seed=seed)
            info = {"physics_profile": self.physics_profile_metadata()}
            if self._background is not None:
                if (options or {}).get("background_id") is not None:
                    self._background.draw(options["background_id"])
                    self._background_pending = True
                self._show_pending_background()
                info["background"] = self._background.info()
            return self.get_obs(), info
        self._validate_physics_observation_cadence()
        if self._background is not None:
            pinned = (options or {}).get("background_id")
            if (
                pinned is not None or (randomize and self._background.config.resample == "episode")
            ) and self._background.next_episode(pinned):
                self._background_pending = True  # shown once the scene is final (a reload binds it itself)
        randomizer = getattr(self, "_task_randomizer", None)
        if randomizer is None and self._task:
            randomizer = get_task_randomizer(self._task)
        sampling = randomize and randomizer is not None
        episode = None if sampling else self.prepare_randomization(None)
        if sampling and getattr(self, "_task_randomizer", None) is None:
            randomizer = randomizer.clone()
            randomizer.bind_env(self)
        if randomizer is not None:
            prepare_reset = getattr(randomizer, "prepare_reset", None)
            if prepare_reset is not None:
                prepare_reset(request=randomization_request, randomize=randomize)
        if randomize and randomizer is not None:
            self._task_randomizer = randomizer
        self._physics_observer = None
        super().reset(seed=seed)
        self.initialize_randomization(
            keep_arm_state=(options or {}).get("keep_arm_state", False)
        )

        # _task_randomizer (set by make_env) takes priority over TASK_RANDOMIZERS
        # so that model-swapping tasks (e.g. inhand_transfer) work correctly.
        self._last_randomization = None
        if randomize and randomizer is not None:
            self._last_randomization = randomizer.randomize(
                self.model,
                self.data,
                seed=seed,
                request=randomization_request,
            )
            self._last_randomization.metadata["physics_profile"] = (
                self.physics_profile_metadata()
            )
            episode = self.prepare_randomization(self._last_randomization)

        self.finalize_randomization(
            self._last_randomization,
            restorer=randomizer if self._last_randomization is not None else None,
            episode=episode,
            keep_arm_state=(options or {}).get("keep_arm_state", False),
        )
        self._show_pending_background()
        info: dict[str, Any] = {"physics_profile": self.physics_profile_metadata()}
        if self._last_randomization is not None:
            info["randomization"] = self._last_randomization
        if self._background is not None:
            info["background"] = self._background.info()
        return self.get_obs(), info

    def initialize_randomization(self, *, keep_arm_state: bool = False) -> None:
        """Reset physics before sampling or restoring an initial object scene.

        Arms start at configured ``init_q`` unless ``keep_arm_state`` explicitly
        carries their current pose. Object randomization states do not record
        robot poses. Velocities, controls, time, and other transient MuJoCo state
        are reset in either case. Validate replay compatibility before calling.
        """
        self._validate_physics_observation_cadence()
        self._physics_observer = None
        if not keep_arm_state:
            self.forget_arm_state()
        robot_qpos = self._get_reset_robot_qpos()
        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_qpos(robot_qpos)
        mujoco.mj_forward(self.model, self.data)

    def prepare_randomization(self, state: RandomizationState | None) -> Any:
        """Validate a recorded or compiled initial episode before mutation."""
        if isinstance(self._step_evaluator, EpisodeTaskEvaluator):
            return self._step_evaluator.prepare_randomization(self.model, state)
        return None

    def finalize_randomization(
        self,
        state: RandomizationState | None,
        *,
        restorer: SceneRandomizer | None,
        episode: Any = None,
        keep_arm_state: bool = False,
    ) -> None:
        """Bind a restored initial scene to its episode metadata and evaluator.

        The model and object poses must already be restored. This starts fresh
        runtime/evaluator history and resets the episode step count; it does not
        restore the history of a partially completed trajectory. A recorded
        scene keeps its bound restorer for subsequent retained resets,
        independently of the optional sampling randomizer.
        Task-specific default robot poses apply only when ``keep_arm_state`` is
        false, after any scene replacement and before evaluation.
        """
        if state is not None and (restorer is None or restorer._env_ref is not self):
            raise ValueError(
                "Episode state requires a restorer bound to this environment."
            )
        self._validate_physics_observation_cadence()
        if episode is None:
            episode = self.prepare_randomization(state)
        self._physics_observer = None
        self._last_randomization = state
        if state is not None:
            prompt = state.metadata.get("prompt")
            if prompt is not None:
                self.prompt = str(prompt)
        if self._task_runtime is not None:
            if not keep_arm_state:
                initialize_pose = getattr(
                    self._task_runtime, "initialize_robot_pose", None
                )
                if initialize_pose is not None:
                    initialize_pose(self.model, self.data)
            episode_kwargs = {} if episode is None else {"episode": episode}
            self._task_runtime.after_reset(
                self.model,
                self.data,
                randomization=state,
                **episode_kwargs,
            )

        self._reset_camera_backend()
        self._step_task_eval = None
        source = self._step_evaluator
        if source is not None:
            source.reset(nworld=1)
            if episode is not None:
                source.configure_episode(self.model, episode)
            elif state is not None and hasattr(
                source,
                "configure_from_randomization",
            ):
                source.configure_from_randomization(
                    self.model,
                    state,
                )
            if self._last_randomization is not None and hasattr(
                source, "set_active_trash_joints"
            ):
                trash_joints = getattr(self._last_randomization, "metadata", {}).get(
                    "trash_joints"
                )
                if trash_joints is not None:
                    source.set_active_trash_joints(trash_joints)
            if self._last_randomization is not None and hasattr(
                source, "set_active_object_joints"
            ):
                active_joints = getattr(self._last_randomization, "metadata", {}).get(
                    "active_object_joints"
                )
                if active_joints is not None:
                    source.set_active_object_joints(active_joints)
            if self._step_updates:
                self._update_step_task_eval()
            if self._physics_evaluator is not None:
                self._physics_evaluator.after_reset(self.model, self.data)
                self._physics_observer = self._physics_evaluator.after_physics_step
        self._sync_prompt_from_task_runtime()

        self.cur_step = 0
        self._episode_restorer = restorer if state is not None else None

    def step(self, action: np.ndarray, *, render_obs: bool = True):
        """Apply one 14D policy-space action.

        Returns the Gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``.
        If ``render_obs`` is false, ``obs`` is ``None`` and only physics/reward
        state is advanced.
        """

        self._validate_physics_observation_cadence()
        obs = self._apply_single_action(action, render_obs=render_obs)
        reward, info = self._task_step_info()
        terminated, truncated = self._episode_status(info)
        return obs, reward, terminated, truncated, info

    def step_chunk(self, action_chunk: np.ndarray):
        """Apply a ``(chunk_dim, 14)`` action chunk and return chunk history."""

        action_chunk = self._as_action_chunk(action_chunk)
        all_obs = []
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        for i in range(self.chunk_dim):
            obs, reward, terminated, truncated, info = self.step(
                action_chunk[i],
                render_obs=True,
            )
            if obs is None:
                raise RuntimeError(
                    "step_chunk(render_obs=True) returned no observation"
                )
            all_obs.append(obs)
            if terminated or truncated:
                break

        final_obs = all_obs[-1]
        chunk_history = self._stack_obs(all_obs)
        return final_obs, chunk_history, reward, terminated, truncated, info

    def _apply_single_action(
        self,
        action_14d: np.ndarray,
        *,
        render_obs: bool,
    ) -> dict[str, Any] | None:
        action = self._as_single_action(action_14d)
        self._step_single(action)
        self.cur_step += 1
        if not render_obs:
            return None
        return self.get_obs()

    def apply_action(self, action: np.ndarray, *, physics_steps: int) -> None:
        """Apply one policy action for an explicit positive physics-step count.

        Native actuator mapping, task runtime callbacks and physics observers
        run normally. This increments ``cur_step`` once without rendering or
        returning reward/termination; the caller owns when to read those.
        """
        if type(physics_steps) is not int or physics_steps <= 0:
            raise ValueError("physics_steps must be a positive integer")
        self._step_single(self._as_single_action(action), physics_steps=physics_steps)
        self.cur_step += 1

    def _validate_physics_observation_cadence(self) -> None:
        if self._physics_evaluator is not None:
            minimum = self._physics_evaluator.evaluation_interval_steps
            if type(self._control_decimation) is not int or self._control_decimation < minimum:
                raise ValueError(
                    f"control_decimation must be an integer >= {minimum} for "
                    f"{self._task_spec.name!r}; use apply_action or advance_physics "
                    "for shorter low-level steps."
                )

    def _as_single_action(self, action_14d: np.ndarray) -> np.ndarray:
        action = np.asarray(action_14d, dtype=np.float32)
        if action.size != self.single_timestep_action_dim:
            raise ValueError(
                "step() expects one 14D policy action; "
                "use step_chunk() for a (chunk_dim, 14) action chunk"
            )
        return action.reshape(self.single_timestep_action_dim)

    def _as_action_chunk(self, action_chunk: np.ndarray) -> np.ndarray:
        action = np.asarray(action_chunk, dtype=np.float32)
        expected_size = self.chunk_dim * self.single_timestep_action_dim
        if action.size != expected_size:
            raise ValueError(
                f"step_chunk() expects {self.chunk_dim} actions of size "
                f"{self.single_timestep_action_dim}; got shape {action.shape}"
            )
        return action.reshape(self.chunk_dim, self.single_timestep_action_dim)

    def _task_step_info(self) -> tuple[float, dict[str, Any]]:
        task_eval = self._step_task_eval
        if task_eval is None:
            task_eval = self.evaluate_task()
        reward = task_eval.scalar_reward() if task_eval is not None else 0.0
        info: dict[str, Any] = {}
        if task_eval is not None:
            info["task_reward"] = reward
            info["task_success"] = task_eval.scalar_success()
            info["task_eval"] = task_eval.to_info(squeeze=True)
        return float(reward), info

    def _episode_status(self, info: dict[str, Any]) -> tuple[bool, bool]:
        task_success = bool(info.get("task_success", False))
        terminated = bool(self.terminate_on_success and task_success)
        time_limit_reached = (
            self.max_episode_steps is not None
            and self.cur_step >= self.max_episode_steps
        )
        truncated = bool(time_limit_reached and not terminated)
        info["episode_step"] = self.cur_step
        info["terminate_on_success"] = self.terminate_on_success
        info["time_limit_reached"] = bool(time_limit_reached)
        if self.max_episode_steps is not None:
            info["max_episode_steps"] = self.max_episode_steps
        return terminated, truncated

    def capture_state(self) -> dict[str, Any]:
        """Return a copy of the current MuJoCo state for tracing/replay."""

        return {
            "qpos": np.asarray(self.data.qpos, dtype=np.float32).copy(),
            "qvel": np.asarray(self.data.qvel, dtype=np.float32).copy(),
            "time": np.float32(self.data.time),
            "physics_profile": self.physics_profile_metadata(),
        }

    def physics_profile_metadata(self) -> dict[str, Any] | None:
        """Return the physics condition to store alongside scene XML and state."""
        return None if self.physics_profile is None else self.physics_profile.to_dict()

    def close(self):
        self._close_camera_backend()

    def _set_qpos_from_state(self, state: np.ndarray):
        """Write a 14D state vector into MuJoCo qpos."""
        self._write_policy_qpos(self.data.qpos, state)
        self._arm_state_initialized = True

    def _write_policy_qpos(self, qpos: np.ndarray, state: np.ndarray) -> None:
        """Write policy coordinates into the supplied physical qpos vector."""
        for i, qpos_idx in enumerate(self._qpos_indices):
            val = state[i]
            if i in self._gripper_set:
                # Scale [0, 1] policy space → MuJoCo joint space
                val = val * _GRIPPER_CTRL_MAX
            qpos[qpos_idx] = val

    def forget_arm_state(self) -> None:
        """Drop the carried arm pose so the next reset starts from init_q."""
        self._arm_state_initialized = False

    def _get_reset_arm_state(self) -> np.ndarray:
        """Return the reset arm pose projected into policy coordinates."""
        if self._arm_state_initialized:
            return project_policy_state(
                np.asarray(self.data.qpos, dtype=np.float64),
                self._qpos_indices,
                self._gripper_indices,
            )
        return self.get_init_q().astype(np.float32, copy=True)

    def _get_reset_robot_qpos(self) -> dict[str, np.ndarray]:
        """Capture every physical joint in the configured robots' kinematic trees.

        Joint names preserve identity when task scene reloads move qpos addresses.
        Passive joints retain their exact physical values. Before initialization,
        robot joints use model defaults with the configured policy-space init_q.
        Independent scene objects are excluded.
        """
        qpos = self.data.qpos
        if not self._arm_state_initialized:
            qpos = self.model.qpos0.copy()
            self._write_policy_qpos(
                qpos, self.get_init_q().astype(np.float32, copy=True)
            )
        return {
            name: qpos[span].copy() for name, span in self._robot_qpos_slices.items()
        }

    def _set_robot_qpos(self, pose: dict[str, np.ndarray]) -> None:
        """Restore a complete physical robot pose after a compatible scene reload."""
        if pose.keys() != self._robot_qpos_slices.keys():
            raise ValueError("Scene reload changed the physical robot joint inventory.")
        for name, span in self._robot_qpos_slices.items():
            if np.shape(pose[name]) != (span.stop - span.start,):
                raise ValueError(f"Scene reload changed robot joint {name!r} qpos size.")
        for name, span in self._robot_qpos_slices.items():
            self.data.qpos[span] = pose[name]
        self._arm_state_initialized = True

    def _step_single(self, action_14d: np.ndarray, *, physics_steps: int | None = None):
        """Apply a single 14D action and step physics."""
        ctrl = np.zeros(self.model.nu)
        for i in range(self.single_timestep_action_dim):
            val = action_14d[i]
            if i in self._gripper_set:
                # Scale [0, 1] policy space → actuator ctrl range
                val = val * _GRIPPER_CTRL_MAX
            ctrl[self._ctrl_indices[i]] = val
        self.data.ctrl[:] = ctrl
        self.advance_physics(physics_steps)

    def advance_physics(self, steps: int | None = None) -> None:
        """Advance MuJoCo physics while applying any task runtime controls."""

        num_steps = self._control_decimation if steps is None else steps
        if num_steps < 0:
            raise ValueError(f"steps must be non-negative, got {num_steps}")
        observer = self._physics_observer
        for step in range(num_steps):
            if self._task_runtime is not None:
                self._task_runtime.before_step(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            if observer is not None:
                observer(self.model, self.data)
            if self._step_updates and (
                (step + 1) % self._step_update_interval == 0
                or step == num_steps - 1
            ):
                # Refresh only when scoring needs the integrated positions/contacts.
                # Always score the final state, including a partial control step.
                if self._step_requires_position_refresh:
                    mujoco.mj_fwdPosition(self.model, self.data)
                self._update_step_task_eval()
            if self._task_runtime is not None:
                after_step = getattr(self._task_runtime, "after_step", None)
                if after_step is not None:
                    after_step(self.model, self.data)
                self._sync_prompt_from_task_runtime()

    def _update_step_task_eval(self) -> None:
        """Observe one source for runtime progress and optional reward exposure."""
        result = self._evaluate_task_source(self._step_evaluator)
        self._step_task_eval = result if self._task_evaluator is not None else None
        if self._task_runtime is not None:
            on_task_evaluation = getattr(self._task_runtime, "on_task_evaluation", None)
            if on_task_evaluation is not None:
                on_task_evaluation(result)

    def _sync_prompt_from_task_runtime(self) -> None:
        """Adopt a runtime-owned prompt when the task exposes one."""

        if self._task_runtime is None:
            return
        observation_prompt = getattr(self._task_runtime, "observation_prompt", None)
        if observation_prompt is not None:
            self.prompt = str(observation_prompt())

    def project_state_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Project an arbitrary qpos vector into the 14D policy-space state."""
        return project_policy_state(
            qpos,
            self._qpos_indices,
            self._gripper_indices,
            dtype=np.float32,
        )

    def get_state(self) -> np.ndarray:
        """Return the current 14D policy-space state from MuJoCo qpos."""
        return self.project_state_from_qpos(self.data.qpos)

    def set_task(self, task: str | SimTaskSpec) -> None:
        """Bind task reset options, prompt, and evaluator.

        An explicit constructor prompt remains the caller's override. Episode
        randomization and task runtimes can still supply a live prompt on reset.
        """

        resolved = resolve_task(task)
        env_task = resolved.env_task or task
        task_spec = resolved.task_spec or maybe_get_task_spec(task)
        reset_request = (
            task_spec.reset_request_options() if task_spec is not None else None
        )
        if self._prompt_override is not None:
            prompt = self._prompt_override
        elif task_spec is not None:
            prompt = task_spec.prompt
        elif isinstance(task, str) and task:
            prompt = task.replace("_", " ")
        else:
            prompt = "fold the towel"
        runtime = make_task_runtime(
            resolved.task_spec or env_task,
            self.model,
            self.data,
        )
        self._bind_task_evaluator(task_spec, runtime)
        self._task_request = task
        self._task = env_task
        self._task_spec = task_spec
        self._task_reset_request = reset_request
        self.prompt = prompt

    def task_runtime_debug_state(self) -> dict[str, Any] | None:
        """Return task runtime diagnostics, when a runtime is attached."""

        if self._task_runtime is None:
            return None
        return self._task_runtime.debug_state()

    def evaluate_task(self) -> TaskEvalResult | None:
        """Return optional reward/success for the current simulation state.

        Chronological runtime progress reads the last reset/physics observation;
        querying a score does not advance its history again. Before reset, the
        source may evaluate its compiled initial state if it supports that state.
        """

        if self._task_evaluator is None:
            return None
        if self._uses_runtime_progress and self._step_task_eval is not None:
            return self._step_task_eval
        result = self._evaluate_task_source(self._task_evaluator)
        if self._uses_runtime_progress:
            self._step_task_eval = result
        return result

    def _evaluate_task_source(self, source: TaskEvaluator) -> TaskEvalResult:
        """Observe a bound source without deciding whether its score is exposed."""
        if source is self._physics_evaluator:
            return self._physics_evaluator.evaluate_model_data(self.model, self.data)
        evaluate_model_data = getattr(source, "evaluate_model_data", None)
        if evaluate_model_data is not None:
            if not self._step_updates:
                # Contact-based tasks still need current state when explicitly
                # scored, without refreshing after every physics integration.
                mujoco.mj_fwdPosition(self.model, self.data)
            return evaluate_model_data(self.model, self.data)
        qpos_batch = np.asarray(self.data.qpos, dtype=np.float32)[None, :]
        return source.evaluate_qpos_batch(qpos_batch)

    def get_obs(self) -> dict[str, Any]:
        """Return observation dict with state, images, and metadata."""
        state = self.get_state()

        if self._render_cameras_flag:
            images = self._render_cameras()
        else:
            images = {
                name: np.zeros(
                    (3, self._camera_height, self._camera_width), dtype=np.uint8
                )
                for name in self.camera_names
            }

        masks = {name: True for name in self.camera_names}
        sim_time = np.asarray(self.data.time, dtype=np.float64)
        timestamps = {name: sim_time.copy() for name in self.camera_names}

        return {
            "images": images,
            "masks": masks,
            "state": state,
            "prompt": self.prompt,
            "camera_timestamps": timestamps,
        }

    def _render_cameras(self) -> dict[str, np.ndarray]:
        """Render all cameras and return (3, H, W) uint8 images."""
        if self._camera_provider is not None:
            frames = self._camera_provider.frames_for_step(self.cur_step, self.data.time)
            return {
                name: frame.transpose(2, 0, 1).copy()
                for name, frame in frames.items()
            }

        images = {}
        for name in self.camera_names:
            self.renderer.update_scene(self.data, camera=name)
            img = self.renderer.render()  # (H, W, 3) uint8
            images[name] = img.transpose(2, 0, 1).copy()  # → (3, H, W)
        return images

    def get_init_q(self) -> np.ndarray:
        """Return concatenated init_q across all robots as a flat 14D array."""
        return np.concatenate(
            [np.array(self.config.robots[name].init_q) for name in self.robot_names]
        )

    def _stack_obs(self, obses: list[dict]) -> dict:
        """Stack a list of observations into chunked arrays."""
        stacked: dict[str, Any] = {}
        for key in obses[0]:
            if isinstance(obses[0][key], str):
                stacked[key] = obses[0][key]
            elif isinstance(obses[0][key], dict):
                stacked[key] = {
                    k: np.stack([obs[key][k] for obs in obses])
                    for k in obses[0][key]
                }
            else:
                stacked[key] = np.stack([obs[key] for obs in obses])
        return stacked
