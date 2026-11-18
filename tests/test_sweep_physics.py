"""CPU sweep regressions using the task's authored collision meshes.

Prepare assets with ``python prepare.py --sim-task sweep`` before running.
"""

from __future__ import annotations

import copy
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import abc_sim
from tests.grasp_fixture import close_and_lift


class SweepPhysicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assets = Path(abc_sim.__file__).parent / "models" / "assets"
        packages = ("i2rt_yam", "brush_flat", "dustpan", "garbage_can", "paper_ball")
        if any(not (assets / package).is_dir() for package in packages):
            raise unittest.SkipTest("Prepare sweep assets: python prepare.py --sim-task sweep")

    def setUp(self) -> None:
        self.env = abc_sim.make_env(
            task="sweep",
            render_cameras=False,
            chunk_dim=1,
            control_decimation=1,
        )
        self.addCleanup(self.env.close)

    def _assert_inactive_balls_stay_outside(self, count: int, seconds: float) -> None:
        model, data = self.env.model, self.env.data
        inactive = [model.body(f"trash_{i}").id for i in range(count + 1, 8)]
        for actuator in range(model.nu):
            joint = model.actuator_trnid[actuator, 0]
            data.ctrl[actuator] = data.qpos[model.jnt_qposadr[joint]]
        max_z = -np.inf
        max_x = -np.inf
        for _ in range(round(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
            max_z = max(max_z, float(data.xpos[inactive, 2].max()))
            max_x = max(max_x, float(data.xpos[inactive, 0].max()))
        self.assertLess(max_z, 0.85, "Inactive balls launched by the infinite table plane")
        self.assertLess(max_x, -1.0, "Inactive balls entered the robot workspace")
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_randomized_inactive_balls_do_not_return_later(self) -> None:
        # The old parking positions launch balls >100 m up; they return at 12–14 s.
        for count in (2, 3, 4):
            for scales in (False, True):
                with self.subTest(count=count, scales=scales):
                    self.env.reset(
                        seed=7,
                        options={"randomization": {"trash_count": count, "randomize_scales": scales}},
                    )
                    self._assert_inactive_balls_stay_outside(count, seconds=20)

    def test_saved_resets_repark_inactive_balls_and_preserve_active_poses(self) -> None:
        for count in (2, 4, 3):
            self.env.reset(
                seed=7,
                options={"randomization": {"trash_count": count, "randomize_scales": False}},
            )
            state = self.env._last_randomization
            for legacy in (False, True):
                for metadata in (False, True):
                    with self.subTest(count=count, legacy=legacy, metadata=metadata):
                        saved = copy.deepcopy(state)
                        if legacy:
                            for i in range(count + 1, 8):
                                saved.object_states[f"trash_{i}_jnt"]["pos"][2] = -1.0
                        if not metadata:
                            saved.metadata.clear()
                        before = copy.deepcopy(saved.to_dict())
                        mujoco.mj_resetData(self.env.model, self.env.data)
                        randomizer = self.env._task_randomizer
                        randomizer.apply(self.env.model, self.env.data, saved)
                        self.assertEqual(len(randomizer._trash_joints), count)
                        for i in range(1, count + 1):
                            joint = f"trash_{i}_jnt"
                            np.testing.assert_allclose(
                                self.env.data.joint(joint).qpos[:3],
                                state.object_states[joint]["pos"],
                            )
                        self.assertEqual(saved.to_dict(), before)
                        self._assert_inactive_balls_stay_outside(count, seconds=0.5)

    def test_visible_finger_surfaces_block_thin_objects(self) -> None:
        source = Path(abc_sim.__file__).parent / "models" / "yam_sweep_scene.xml"
        root = ET.parse(source).getroot()
        for attribute in ("meshdir", "texturedir"):
            root.find("compiler").set(attribute, str(source.parent / "assets"))
        probe = ET.SubElement(root.find("worldbody"), "body", name="probe", mocap="true")
        ET.SubElement(probe, "geom", name="probe", type="sphere", size="0.002", mass="0.001")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        groups = model.geom_group.copy()
        ray_groups = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
        for arm in ("left", "right"):
            for finger, suffix, mesh, sign in (
                ("left", "lf", "model2__14", 1),
                ("right", "rf", "model2__17", -1),
            ):
                base = model.body(f"{arm}_link_{finger}_finger").id
                body_ids = {base, model.body(f"{arm}_{suffix}_rot").id, model.body(f"{arm}_{suffix}_down").id}
                collision = [
                    g for g in range(model.ngeom) if model.geom_bodyid[g] in body_ids and model.geom_contype[g]
                ]
                visual = next(
                    g
                    for g in range(model.ngeom)
                    if model.geom_bodyid[g] == base
                    and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                    and model.geom_dataid[g] == model.mesh(mesh).id
                    and model.geom_contype[g] == 0
                )
                model.geom_group[:] = groups
                model.geom_group[visual] = 5
                frame = data.body(f"{arm}_{suffix}_down")
                rotation = frame.xmat.reshape(3, 3).copy()
                for x, z in ((0.01, 0.025), (0.015, 0.02), (0.06, 0.02)):
                    with self.subTest(arm=arm, finger=finger, x=x, z=z):
                        origin = frame.xpos + rotation @ [sign * x, 0.04, z]
                        direction = rotation @ np.array([0, -1, 0])
                        hit = np.array([-1], dtype=np.int32)
                        distance = mujoco.mj_ray(model, data, origin, direction, ray_groups, 1, -1, hit)
                        self.assertEqual(hit[0], visual)
                        # Put a thin object just inside the actual rendered surface.
                        data.mocap_pos[0] = origin + direction * (distance + 0.0001)
                        mujoco.mj_forward(model, data)
                        separation = min(
                            mujoco.mj_geomDistance(model, data, model.geom("probe").id, g, 1, None) for g in collision
                        )
                        self.assertLess(separation, 0, "Object passed through the visible finger")

    def test_brush_grasp_survives_lift_under_gravity(self) -> None:
        self.env.reset(randomize=False)
        model = self.env.model
        gravity = model.opt.gravity.copy()
        brush = model.body("brush").id
        brush_joint = int(model.body_jntadr[brush])
        brush_adr = int(model.jnt_qposadr[brush_joint])
        brush_geoms = set(np.flatnonzero(model.geom_bodyid == brush))
        for side in ("left", "right"):
            site_id = model.site(side + "_grasp_site").id
            finger_bodies = {
                model.body(name).id
                for name in (
                    side + "_link_left_finger",
                    side + "_link_right_finger",
                    side + "_lf_rot",
                    side + "_rf_rot",
                    side + "_lf_down",
                    side + "_rf_down",
                )
            }
            finger_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in finger_bodies}
            for offset in (0.017, 0.03, 0.044):
                for grip_point, depth, orientation in (
                    (np.array([0, -0.095, 0]), -0.02, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),
                    (np.array([0, -0.095, 0]), -0.02, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])),
                    (np.array([0, -0.015, 0]), 0.02, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])),
                ):
                    with self.subTest(side=side, offset=offset, grip_point=grip_point, orientation=orientation):
                        model.opt.gravity[:] = 0
                        data = mujoco.MjData(model)
                        mujoco.mj_copyData(data, model, self.env.data)
                        for joint in range(model.njnt):
                            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE and joint != brush_joint:
                                adr = model.jnt_qposadr[joint]
                                data.qpos[adr : adr + 3] = [-10 - joint, 0, 0.8]
                        for hand in ("left", "right"):
                            data.joint(hand + "_left_finger").qpos[0] = 0.0475
                            data.joint(hand + "_right_finger").qpos[0] = -0.0475
                        for actuator in range(model.nu):
                            data.ctrl[actuator] = data.qpos[model.jnt_qposadr[model.actuator_trnid[actuator, 0]]]
                        mujoco.mj_forward(model, data)
                        site = data.site(site_id)
                        rotation = site.xmat.reshape(3, 3).copy()
                        grasp_local = np.array([offset, 0, depth])
                        point = site.xpos + rotation @ grasp_local
                        brush_rotation = rotation @ orientation
                        data.qpos[brush_adr : brush_adr + 3] = point - brush_rotation @ grip_point
                        mujoco.mju_mat2Quat(data.qpos[brush_adr + 3 : brush_adr + 7], brush_rotation.ravel())
                        mujoco.mj_forward(model, data)
                        for contact in data.contact:
                            if set(contact.geom) & brush_geoms:
                                self.assertGreaterEqual(contact.dist, -1e-5, "Grasp starts in penetration")
                        contacts, penetration = 0, 0.0

                        def step() -> None:
                            nonlocal contacts, penetration
                            mujoco.mj_step(model, data)
                            for contact in data.contact:
                                pair = set(contact.geom)
                                if pair & brush_geoms and pair & finger_geoms:
                                    contacts += 1
                                    penetration = max(penetration, -float(contact.dist))

                        close_and_lift(self, model, data, side, gravity, step)
                        grip_world = data.xpos[brush] + data.xmat[brush].reshape(3, 3) @ grip_point
                        site = data.site(site_id)
                        grasp_error = np.linalg.norm(site.xmat.reshape(3, 3).T @ (grip_world - site.xpos) - grasp_local)
                        self.assertGreater(contacts, 0)
                        self.assertLess(penetration, 0.0015)
                        self.assertLess(grasp_error, 0.005)
                        self.assertGreater(grip_world[2] - point[2], 0.07)


if __name__ == "__main__":
    unittest.main()
