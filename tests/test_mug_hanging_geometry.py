"""Real asset geometry, scale, finite insertion and public reset contracts."""

import unittest

import mujoco
import numpy as np
from abc_sim import make_env
from abc_sim.batched_env import BatchedWarpYAMEnv
from abc_sim.randomization.assets.mugs import _build_mug_scene_xml
from abc_sim.task_eval.contacts import subtree_geom_ids
from abc_sim.task_eval.mug_hanging_geometry import measure_handle_aperture, measure_rack_pegs
from abc_sim.task_specs import get_task_spec

# Interior handle points measured in the nominal assets, in local YZ meters.
HANDLE_POINTS = {
    0: (-0.03254, 0.00699),
    2: (-0.03675, 0.00006),
    3: (-0.03225, -0.00176),
    4: (-0.04346, 0.00511),
    5: (-0.03650, 0.00356),
    6: (-0.03660, 0.01438),
    8: (-0.03146, -0.00092),
    9: (-0.03675, 0.00131),
    11: (-0.03375, 0.00781),
    12: (-0.04375, 0.01072),
    13: (-0.03375, -0.00289),
    14: (-0.03525, -0.01258),
    15: (-0.03945, 0.00437),
    16: (-0.03300, 0.00769),
    17: (-0.03425, 0.00345),
    18: (-0.03550, 0.00777),
}


def model_for(variant, scale=1.0):
    model = mujoco.MjModel.from_xml_string(
        _build_mug_scene_xml(
            task_name="mug_tree",
            mug_variant=f"mug_{variant}",
            mug_variants=[f"mug_{variant}"],
            scale_states={"mug_1_jnt": scale},
            base_scene_xml=None,
            base_scene_dir=None,
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    mujoco.mj_forward(model, data)
    return model, data


def peg_in_mug(model, data, peg):
    tree, mug = model.body("mug_tree").id, model.body("mug_1").id
    tree_rotation, mug_rotation = data.xmat[[tree, mug]].reshape(2, 3, 3)

    def transform(points):
        return (points @ tree_rotation.T + data.xpos[tree] - data.xpos[mug]) @ mug_rotation

    return transform(peg.endpoints), transform(peg.vertices)


def threaded(model, data, peg, aperture):
    endpoints, vertices = peg_in_mug(model, data, peg)
    return aperture.threaded_by(endpoints, peg.radius, vertices=vertices, edges=peg.edges)


class MugHangingGeometryTests(unittest.TestCase):
    def test_all_sampled_styles_fit_real_branches_at_both_scale_limits(self):
        for variant, point in HANDLE_POINTS.items():
            for scale in (0.9, 1.1):
                with self.subTest(variant=variant, scale=scale):
                    model, data = model_for(variant, scale)
                    tree, mug = model.body("mug_tree").id, model.body("mug_1").id
                    aperture = measure_handle_aperture(model, data, mug)
                    pegs = measure_rack_pegs(model, data, tree)
                    self.assertEqual(len(pegs), 6)
                    peg = pegs[0]
                    rotation = data.xmat[tree].reshape(3, 3)
                    axis = rotation @ (peg.endpoints[1] - peg.endpoints[0])
                    axis /= np.linalg.norm(axis)
                    down = np.array([0.0, 0.0, -1.0])
                    down -= axis * (down @ axis)
                    down /= np.linalg.norm(down)
                    mug_rotation = np.stack([axis, down, np.cross(axis, down)], axis=1)
                    quaternion = np.empty(4)
                    mujoco.mju_mat2Quat(quaternion, mug_rotation.ravel())
                    center = peg.endpoints.mean(axis=0) @ rotation.T + data.xpos[tree]
                    address = int(model.joint("mug_1_jnt").qposadr[0])
                    mug_geoms = subtree_geom_ids(model, mug)
                    tree_geoms = subtree_geom_ids(model, tree)
                    fits = False
                    for dy in (0, -0.001, 0.001):
                        for dz in (0, -0.001, 0.001):
                            local_center = np.array(
                                [aperture.x_faces.mean(), scale * point[0] + dy, scale * point[1] + dz]
                            )
                            data.qpos[address : address + 3] = center - mug_rotation @ local_center
                            data.qpos[address + 3 : address + 7] = quaternion
                            mujoco.mj_forward(model, data)
                            penetrating = any(
                                contact.dist < -1e-5
                                and (
                                    (int(contact.geom1) in mug_geoms and int(contact.geom2) in tree_geoms)
                                    or (int(contact.geom2) in mug_geoms and int(contact.geom1) in tree_geoms)
                                )
                                for contact in data.contact[: data.ncon]
                            )
                            fits |= not penetrating and threaded(model, data, peg, aperture)
                    self.assertTrue(fits, "The sampled mug needs a collision-free, fully threaded placement.")

    def test_finite_threading_rejects_partial_tip_and_cup_body(self):
        model, data = model_for(0)
        mug = model.body("mug_1").id
        aperture = measure_handle_aperture(model, data, mug)
        peg = measure_rack_pegs(model, data, model.body("mug_tree").id)[0]
        address = int(model.joint("mug_1_jnt").qposadr[0])
        data.qpos[address : address + 7] = [
            0.6834258629,
            -0.0033203303,
            0.9951174564,
            0.7422521996,
            -0.6693819098,
            -0.0289006264,
            0.0124211476,
        ]
        mujoco.mj_forward(model, data)
        self.assertTrue(threaded(model, data, peg, aperture))
        endpoints, vertices = peg_in_mug(model, data, peg)
        # Pull the peg back until its tip has entered but not cleared the handle.
        offset = np.array([aperture.x_faces.mean() - endpoints[:, 0].max(), 0, 0])
        self.assertFalse(
            aperture.threaded_by(endpoints + offset, peg.radius, vertices=vertices + offset, edges=peg.edges)
        )
        # A branch through the cup's solid side has no handle threading credit.
        offset = np.array([0, 0.05, 0])
        self.assertFalse(
            aperture.threaded_by(endpoints + offset, peg.radius, vertices=vertices + offset, edges=peg.edges)
        )
        # A thick object through the opening's center still cannot fit its rim.
        enlarged = vertices.copy()
        enlarged[:, 1:] = (enlarged[:, 1:] - endpoints[:, 1:].mean(axis=0)) * 4 + endpoints[:, 1:].mean(axis=0)
        self.assertFalse(aperture.threaded_by(endpoints, peg.radius * 4, vertices=enlarged, edges=peg.edges))

    def test_public_reset_uses_threadable_styles_and_zero_idle_reward(self):
        spec = get_task_spec("hang_mug_on_mug_rack")
        self.assertEqual(spec.evaluator_name, "mug_on_rack_v2")
        self.assertEqual(set(dict(spec.reset_request)["mug_variant_pool"]), {f"mug_{i}" for i in HANDLE_POINTS})
        env = make_env(spec.name, render_cameras=False, chunk_dim=1, control_decimation=1)
        try:
            for seed in (0, 11, 31):
                with self.subTest(seed=seed):
                    env.reset(seed=seed)
                    self.assertEqual(env.evaluate_task().scalar_reward(), 0)
                    env.advance_physics(17)
                    self.assertEqual(env.evaluate_task().scalar_reward(), 0)
                    self.assertEqual(env.evaluate_task().metrics["evaluator_revision"], "abc-mug-hanging-v2")
            with self.assertRaisesRegex(NotImplementedError, "solved contacts"):
                BatchedWarpYAMEnv(env, num_worlds=2, camera_height=8, camera_width=8)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
