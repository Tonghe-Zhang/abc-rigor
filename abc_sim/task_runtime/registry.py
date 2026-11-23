"""Registry for Python-side task runtimes."""

from __future__ import annotations

from typing import TypeAlias

import mujoco

from abc_sim.task_runtime.base import TaskRuntime
from abc_sim.task_runtime.ball_tray import BallTrayBalancingRuntime
from abc_sim.task_runtime.conveyor_pick import ConveyorPickRuntime
from abc_sim.task_runtime.rl_conveyor import RLConveyorRuntime
from abc_sim.task_runtime.multi_drawer import MultiDrawerSearchRuntime
from abc_sim.task_registry import resolve_env_task_name
from abc_sim.task_specs import SimTaskSpec


TaskRuntimeFactory: TypeAlias = type[TaskRuntime]

_TASK_RUNTIMES: dict[str, TaskRuntimeFactory] = {
    "ball_tray_balancing": BallTrayBalancingRuntime,
    "conveyor_pick": ConveyorPickRuntime,
    "rl_conveyor_pick": RLConveyorRuntime,
    "multi_drawer_search": MultiDrawerSearchRuntime,
}


def make_task_runtime(
    task: str | SimTaskSpec | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> TaskRuntime | None:
    """Instantiate and bind the task runtime for a task, if one is registered."""

    env_task = resolve_env_task_name(task)
    if env_task is None:
        return None

    runtime_cls = _TASK_RUNTIMES.get(env_task)
    if runtime_cls is None:
        return None

    runtime = runtime_cls()
    runtime.bind(model, data)
    return runtime
