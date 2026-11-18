"""Shared closure and gravity-loaded lift for authored-scene grasp tests."""

from __future__ import annotations

import unittest
from collections.abc import Callable

import mujoco
import numpy as np


def close_and_lift(
    test: unittest.TestCase,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: str,
    gravity: np.ndarray,
    step: Callable[[], None],
    *,
    close_seconds: float = 1.5,
) -> np.ndarray:
    """Close a positioned grasp, lift 8 cm under gravity, and return the starting TCP position."""
    dt = model.opt.timestep
    gripper = model.actuator(arm + "_gripper").id
    for n in range(round(close_seconds / dt)):
        data.ctrl[gripper] = 0.0475 * max(0, 1 - n * dt / 0.5)
        step()
    mujoco.mj_forward(model, data)
    site_id = model.site(arm + "_grasp_site").id
    start_site = data.site(site_id).xpos.copy()
    target_position = start_site + [0, 0, 0.080]
    target_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(target_quaternion, data.site(site_id).xmat)
    joints = [model.joint(f"{arm}_joint{i}").id for i in range(1, 7)]
    qpos_indices = model.jnt_qposadr[joints]
    dof_indices = model.jnt_dofadr[joints]
    actuators = [model.actuator(f"{arm}_joint{i}").id for i in range(1, 7)]
    kinematics = mujoco.MjData(model)
    mujoco.mj_copyData(kinematics, model, data)
    for _ in range(200):
        mujoco.mj_forward(model, kinematics)
        site = kinematics.site(site_id)
        quaternion, angular_error = np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(quaternion, site.xmat)
        mujoco.mju_subQuat(angular_error, target_quaternion, quaternion)
        angular_error = site.xmat.reshape(3, 3) @ angular_error
        error = np.r_[target_position - site.xpos, angular_error]
        if np.linalg.norm(error) < 1e-5:
            break
        jac_position = np.zeros((3, model.nv))
        jac_rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, kinematics, jac_position, jac_rotation, site_id)
        jacobian = np.vstack([jac_position[:, dof_indices], jac_rotation[:, dof_indices]])
        kinematics.qpos[qpos_indices] += jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 1e-5, error)
    test.assertLess(np.linalg.norm(error), 1e-5, "Lift target is unreachable")
    goal = kinematics.qpos[qpos_indices].copy()
    test.assertTrue(np.all(goal >= model.jnt_range[joints, 0]))
    test.assertTrue(np.all(goal <= model.jnt_range[joints, 1]))
    before = data.qpos[qpos_indices].copy()
    model.opt.gravity[:] = gravity
    for n in range(round(2.5 / dt)):
        data.ctrl[actuators] = before + (goal - before) * min(1, (n + 1) * dt / 1.5)
        step()
    mujoco.mj_forward(model, data)
    return start_site
