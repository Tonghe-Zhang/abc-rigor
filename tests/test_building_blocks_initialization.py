"""Wooden blocks start above the table and settle without an initialization impulse."""

import unittest

import mujoco
import numpy as np
from abc_sim import make_env
from abc_sim.scene_xml import SceneXmlTransformOptions


class BuildingBlocksInitializationTests(unittest.TestCase):
    def make_env(self, flexible_gripper):
        env = make_env(
            "build_wood_block_tower",
            render_cameras=False,
            scene_xml_transform_options=SceneXmlTransformOptions(
                flexible_gripper=flexible_gripper,
            ),
        )
        self.addCleanup(env.close)
        return env

    def block_bodies(self, model):
        joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(len(joints), 9)
        return model.jnt_bodyid[joints]

    def reset(self, env, home):
        env.reset(seed=11)
        if home:
            mujoco.mj_resetDataKeyframe(env.model, env.data, env.model.key("home").id)
            mujoco.mj_forward(env.model, env.data)

    def test_scene_and_home_start_without_block_penetration(self):
        for flexible_gripper in (False, True):
            env = self.make_env(flexible_gripper)
            model, data = env.model, env.data
            blocks = set(self.block_bodies(model))
            for home in (False, True):
                with self.subTest(flexible_gripper=flexible_gripper, home=home):
                    self.reset(env, home)
                    for contact in data.contact:
                        bodies = model.geom_bodyid[contact.geom]
                        if any(body in blocks for body in bodies):
                            self.assertGreaterEqual(
                                contact.dist,
                                -1e-6,
                                [model.body(body).name for body in bodies],
                            )

    def test_all_blocks_settle_near_their_spawn_positions(self):
        for flexible_gripper in (False, True):
            env = self.make_env(flexible_gripper)
            model, data = env.model, env.data
            blocks = self.block_bodies(model)
            dofs = model.jnt_dofadr[model.body_jntadr[blocks]]
            linear_dofs = dofs[:, None] + np.arange(3)
            for home in (False, True):
                with self.subTest(flexible_gripper=flexible_gripper, home=home):
                    self.reset(env, home)
                    initial = data.xpos[blocks].copy()
                    max_rise = 0.0
                    max_speed = 0.0
                    max_drift = 0.0
                    for _ in range(round(2.0 / model.opt.timestep)):
                        env.advance_physics(1)
                        max_rise = max(max_rise, np.max(data.xpos[blocks, 2] - initial[:, 2]))
                        max_speed = max(
                            max_speed,
                            np.max(np.linalg.norm(data.qvel[linear_dofs], axis=1)),
                        )
                        max_drift = max(
                            max_drift,
                            np.max(np.linalg.norm(data.xpos[blocks, :2] - initial[:, :2], axis=1)),
                        )
                    self.assertTrue(np.isfinite(data.qpos).all())
                    self.assertLess(max_rise, 0.005)
                    self.assertLess(max_speed, 0.3)
                    self.assertLess(max_drift, 0.01)
                    self.assertLess(np.max(np.linalg.norm(data.qvel[linear_dofs], axis=1)), 0.01)
                    supported = set()
                    table = model.geom("table_plane").id
                    for contact in data.contact:
                        if table in contact.geom:
                            supported.update(model.geom_bodyid[contact.geom])
                    self.assertTrue(set(blocks).issubset(supported))

    def test_fixed_layout_is_preserved_across_reset_seeds(self):
        env = self.make_env(False)
        env.reset(seed=0)
        initial = env.data.qpos.copy()
        for seed in (1, 11, 42, 1234):
            env.advance_physics(10)
            env.reset(seed=seed)
            np.testing.assert_array_equal(env.data.qpos, initial)


if __name__ == "__main__":
    unittest.main()
