"""Opt-in background variations: id mapping, independent seeding, visual-only scene edits."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import mujoco
import numpy as np

from abc_sim.background import (
    BackgroundConfig,
    BackgroundController,
    BackgroundLibrary,
    apply_background_xml,
    background_material_ids,
    background_rng,
    background_texture_ids,
    has_background_slots,
    resolve_asset_dir,
    slot_sizes,
)

_ASSETS = Path(__file__).resolve().parents[1] / "abc_sim" / "models" / "assets"
_HAVE_SCENES = (_ASSETS / "i2rt_yam").exists()


def _write_image(path: Path, w: int, h: int, color) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), np.uint8)
    img[...] = color
    img[::7] = 255 - np.asarray(color, np.uint8)  # some structure
    Image.fromarray(img).save(path)


def make_package(root: Path, walls=("alpha", "beta"), tables=("oak", "slate", "white")) -> Path:
    """A tiny package in the published layout: wallpapers/<type>/<inst>/wall_*.jpg, tables/<type>/<inst>/tile.jpg."""
    ix = {"version": "test", "texel_m": 0.01, "wallpapers": {}, "tables": {}}
    for i, w in enumerate(walls):
        insts = []
        for j in range(2):
            d = f"wallpapers/{w}/s{j}"
            for k in ("wall_far", "wall_left", "wall_right"):
                _write_image(root / d / f"{k}.jpg", 80, 50, (40 * i + 20, 100, 30 * j + 10))
            insts.append({"id": f"{w}/s{j}", "dir": d, "fleet_weight": 0.5})
        ix["wallpapers"][w] = {"n_instances": 2, "fleet_weight": 1.0, "instances": insts}
    for i, t in enumerate(tables):
        d = f"tables/{t}/s0"
        _write_image(root / d / "tile.jpg", 20, 20, (200, 60 * i, 50))
        ix["tables"][t] = {"n_instances": 1, "fleet_weight": 1.0, "instances": [{"id": f"{t}/s0", "dir": d, "fleet_weight": 1.0}]}
    (root / "index.json").write_text(json.dumps(ix))
    return root


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_package(Path(self.tmp.name) / "v1")
        self.lib = BackgroundLibrary(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_combination_ids_cover_every_pair_once(self):
        self.assertEqual(self.lib.n_combinations, 6)
        pairs = [self.lib.combination(i) for i in range(6)]
        self.assertEqual(len(set(pairs)), 6)
        self.assertEqual(pairs[0], ("alpha", "oak"))
        self.assertEqual(pairs[4], ("beta", "slate"))
        for i, (w, t) in enumerate(pairs):
            self.assertEqual(self.lib.combination_id(w, t), i)
        with self.assertRaises(ValueError):
            self.lib.combination(6)
        with self.assertRaises(ValueError):
            self.lib.combination(-1)

    def test_asset_dir_must_hold_the_package(self):
        self.assertEqual(resolve_asset_dir(self.root), self.root)
        with self.assertRaises(FileNotFoundError):
            resolve_asset_dir(self.tmp.name)  # the parent of a package is not a package

    def test_background_stream_is_seeded_and_separate_from_plain_seeds(self):
        a = [int(background_rng(5).integers(1 << 30)) for _ in range(2)]
        self.assertEqual(a[0], a[1])
        self.assertNotEqual(int(background_rng(5).integers(1 << 30)), int(np.random.default_rng(5).integers(1 << 30)))
        s1 = [self.lib.draw(background_rng(1))["background_id"] for _ in range(1)]
        r = background_rng(1)
        self.assertEqual(self.lib.draw(r)["background_id"], s1[0])

    def test_pinned_id_keeps_the_pair_and_varies_instances(self):
        r = background_rng(0)
        draws = [self.lib.draw(r, 3) for _ in range(20)]
        self.assertTrue(all(d["background_id"] == 3 for d in draws))
        self.assertEqual({d["wallpaper"].split("/")[0] for d in draws}, {"beta"})
        self.assertEqual(len({d["wallpaper"] for d in draws}), 2)

    def test_slot_images_have_slot_sizes(self):
        for scale in (1.0, 0.5):
            imgs = self.lib.slot_images(self.lib.draw(background_rng(0)), scale)
            for k, (w, h) in slot_sizes(self.lib.texel_m, scale).items():
                self.assertEqual(imgs[k].shape, (h, w, 3))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            BackgroundConfig(asset_dir=str(self.root), resample="never")
        with self.assertRaises(ValueError):
            BackgroundController(BackgroundConfig(asset_dir=str(self.root), background_id=99))
        import abc_sim

        self.assertIsNone(abc_sim.background_config())
        with self.assertRaises(ValueError):
            abc_sim.background_config(randomize_background=True)
        with self.assertRaises(ValueError):
            abc_sim.background_config(background_id=1)


class SceneXmlTest(unittest.TestCase):
    XML = """<mujoco><asset/><worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body name="ball" pos="0.6 0 0.9"><freejoint/><geom type="sphere" size="0.03" mass="0.1"/></body>
    </worldbody></mujoco>"""

    def test_slots_are_visual_only_and_physics_is_unchanged(self):
        plain = mujoco.MjModel.from_xml_string(self.XML)
        bg = mujoco.MjModel.from_xml_string(apply_background_xml(self.XML))
        self.assertFalse(has_background_slots(plain))
        self.assertTrue(has_background_slots(bg))
        self.assertEqual(bg.ngeom, plain.ngeom + 4)
        self.assertEqual(bg.nbody, plain.nbody)
        self.assertEqual(bg.nq, plain.nq)
        added = [g for g in range(bg.ngeom) if (mujoco.mj_id2name(bg, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("abc_bg_")]
        self.assertEqual(len(added), 4)
        for g in added:
            self.assertEqual(bg.geom_contype[g], 0)
            self.assertEqual(bg.geom_conaffinity[g], 0)
            self.assertEqual(bg.geom_bodyid[g], 0)
        np.testing.assert_array_equal(bg.body_mass, plain.body_mass)
        d0, d1 = mujoco.MjData(plain), mujoco.MjData(bg)
        for _ in range(300):
            mujoco.mj_step(plain, d0)
            mujoco.mj_step(bg, d1)
        np.testing.assert_array_equal(d0.qpos, d1.qpos)
        self.assertEqual(apply_background_xml(apply_background_xml(self.XML)).count("abc_bg_wall_far"),
                         apply_background_xml(self.XML).count("abc_bg_wall_far"))


@unittest.skipUnless(_HAVE_SCENES, "sim assets are not provisioned (prepare.py --sim-task put_plastic_bottles_in_bin)")
class EnvTest(unittest.TestCase):
    TASK = "put_plastic_bottles_in_bin"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = make_package(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make(self, **bg):
        import abc_sim

        if bg:
            bg = {"randomize_background": True, "background_variation_asset_dir": str(self.root), **bg}
        return abc_sim.make_env(task=self.TASK, render_cameras=False, **bg)

    def rollout(self, env, seed=4, steps=60):
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(0)
        a0 = np.asarray(obs["state"], np.float32)
        traj = [env.data.qpos.copy()]
        for _ in range(steps):
            env.step(a0 + rng.normal(0, 0.05, a0.shape).astype(np.float32), render_obs=False)
            traj.append(env.data.qpos.copy())
        return np.array(traj), info

    def test_default_off_leaves_the_scene_unchanged(self):
        env, bg = self.make(), self.make(background_seed=0)
        self.assertFalse(has_background_slots(env.model))
        self.assertEqual(env.model.ngeom + 4, bg.model.ngeom)
        self.assertEqual(env.model.ntex + 4, bg.model.ntex)
        bg.close()
        self.assertIsNone(env.background_info)
        self.assertNotIn("background", env.reset(seed=0)[1])
        env.close()

    def test_backgrounds_change_no_physics(self):
        plain, bg = self.make(), self.make(background_seed=1)
        t0, _ = self.rollout(plain)
        t1, info = self.rollout(bg)
        np.testing.assert_array_equal(t0, t1)
        self.assertIn("background_id", info["background"])
        plain.close(), bg.close()

    def test_background_stream_is_independent_of_task_seeds(self):
        a, b = self.make(background_seed=9), self.make(background_seed=9)
        ids_a = [a.reset(seed=s)[1]["background"]["background_id"] for s in (0, 1, 2)]
        ids_b = [b.reset(seed=s)[1]["background"]["background_id"] for s in (7, 7, 7)]
        self.assertEqual(ids_a, ids_b)
        # the same task seed gives the same objects under different background seeds
        c = self.make(background_seed=10)
        a.reset(seed=3), c.reset(seed=3)
        np.testing.assert_array_equal(a.data.qpos, c.data.qpos)
        for e in (a, b, c):
            e.close()

    def test_pinning_and_episode_policy(self):
        env = self.make(background_seed=0, background_id=2)
        self.assertTrue(all(env.reset(seed=s)[1]["background"]["background_id"] == 2 for s in range(3)))
        self.assertEqual(env.reset(seed=0, options={"background_id": 5})[1]["background"]["background_id"], 5)
        fixed = self.make(background_seed=0, background_resample="env")
        ids = {fixed.reset(seed=s)[1]["background"]["background_id"] for s in range(4)}
        self.assertEqual(len(ids), 1)
        env.close(), fixed.close()



def _have_warp() -> bool:
    try:
        import mujoco_warp  # noqa: F401
        import torch  # noqa: F401
        import warp  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_HAVE_SCENES and _have_warp(), "needs sim assets and mujoco_warp / warp / torch")
class BatchedTest(unittest.TestCase):
    TASK = "put_plastic_bottles_in_bin"
    OPTS: ClassVar[dict] = {"randomization": {"randomize_variants": False, "randomize_scales": False}}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = make_package(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make(self, **bg):
        import abc_sim

        if bg:
            bg = {"randomize_background": True, "background_variation_asset_dir": str(self.root), **bg}
        return abc_sim.make_batched_env(self.TASK, num_worlds=3, camera_height=48, camera_width=64, **bg)

    def test_per_world_backgrounds_redrawn_per_reset_and_physics_unchanged(self):
        plain, bg = self.make(), self.make(background_seed=2, background_pool_size=2)
        trajs = []
        for env in (plain, bg):
            env.reset(seeds=[0, 1, 2], options=self.OPTS)
            action = np.asarray(env.state(), np.float32)
            qs = [env.qpos.copy()]
            for _ in range(3):
                env.step(action)
                qs.append(env.qpos.copy())
            trajs.append(np.array(qs))
        np.testing.assert_array_equal(trajs[0], trajs[1])
        self.assertIsNone(plain.background_info)
        first = bg.background_info
        self.assertEqual(len(first), 3)
        self.assertEqual(first[0], first[2])  # pool of 2: world 2 shares set 0
        texid = bg._runtime.m_warp.mat_texid.numpy()
        mats = list(background_material_ids(bg.model).values())
        rgb = mujoco.mjtTextureRole.mjTEXROLE_RGB
        self.assertTrue((texid[0, mats, rgb] != texid[1, mats, rgb]).all())
        np.testing.assert_array_equal(texid[0, mats, rgb], texid[2, mats, rgb])
        bg.reset(seeds=[3, 4, 5], options=self.OPTS)
        self.assertEqual(len(bg.background_info), 3)
        bg.reset(seeds=[3, 4, 5], options={**self.OPTS, "background_id": 4})
        self.assertTrue(all(b["background_id"] == 4 for b in bg.background_info))
        plain.close(), bg.close()


def _can_render() -> bool:
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8).close()
    except Exception:  # noqa: BLE001 - no GL backend on this machine
        return False
    return True


@unittest.skipUnless(_HAVE_SCENES and _can_render(), "needs sim assets and a MuJoCo GL backend")
class RenderSwapTest(unittest.TestCase):
    def test_in_place_swap_matches_a_fresh_renderer(self):
        import abc_sim

        with tempfile.TemporaryDirectory() as tmp:
            root = make_package(Path(tmp))
            env = abc_sim.make_env(task="put_plastic_bottles_in_bin", camera_height=60, camera_width=80,
                                   randomize_background=True, background_seed=0, background_variation_asset_dir=str(root))
            before = env.reset(seed=0)[0]["images"]["top"]
            env.set_background((env.background_info["background_id"] + 1) % 6)
            live = env.get_obs()["images"]["top"]
            fresh = mujoco.Renderer(env.model, 60, 80)
            fresh.update_scene(env.data, camera="top")
            np.testing.assert_array_equal(live.transpose(1, 2, 0), fresh.render())
            self.assertGreater(np.abs(live.astype(int) - before.astype(int)).mean(), 1.0)
            fresh.close()
            env.close()

    def test_failed_reset_does_not_leave_a_stale_render(self):
        from unittest import mock

        import abc_sim

        with tempfile.TemporaryDirectory() as tmp:
            root = make_package(Path(tmp))
            env = abc_sim.make_env(task="spell_cat", camera_height=60, camera_width=80,
                                   randomize_background=True, background_seed=0, background_variation_asset_dir=str(root))
            env.reset(seed=0)
            with mock.patch.object(env, "finalize_randomization", side_effect=RuntimeError("forced")), \
                    self.assertRaises(RuntimeError):
                env.reset(seed=1)  # draws a new background, then fails before showing it
            obs, info = env.reset(seed=2, randomize=False)
            fresh = mujoco.Renderer(env.model, 60, 80)
            fresh.update_scene(env.data, camera="top")
            np.testing.assert_array_equal(obs["images"]["top"].transpose(1, 2, 0), fresh.render())
            self.assertEqual(info["background"], env.background_info)
            # the model shows exactly the reported draw
            shown = env._background.library.slot_images(env._background.state)["wall_far"]
            tid = background_texture_ids(env.model)["wall_far"]
            adr, n = int(env.model.tex_adr[tid]), shown.size
            np.testing.assert_array_equal(env.model.tex_data[adr:adr + n], shown.reshape(-1))
            fresh.close()
            env.close()


if __name__ == "__main__":
    unittest.main()
