"""Small native fixtures for both mug loaders; no assets or rendering required."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from abc_sim.mug_inertia import MUG_INERTIA_REVISION, MUG_MASS_KG, surface_inertia
from abc_sim.randomization.assets import mugs
from abc_sim.randomization.core import RandomizationState
from abc_sim.randomization.tasks.mugs import (
    MugFlipRandomizer,
    MugTreeRandomizer,
    MugVariantRandomizer,
)


ASSET = """
<mujoco><asset>
  <mesh name="fragment" vertex="0 0 0 .01 0 0 0 .01 0 0 0 .01"
        face="0 2 1 0 1 3 0 3 2 1 2 3" scale=".4 .6 .8"
        refquat=".923879532511 0 0 .382683432365"/>
  <mesh name="cup" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1"
        face="0 2 1 0 1 3 0 3 2 1 2 3" scale="1.1 .9 .8"/>
</asset><worldbody><body name="object" pos=".01 -.02 .03" euler=".1 .2 .3">
  <geom type="mesh" mesh="fragment" pos="-.05 .02 .01"
        contype="0" conaffinity="0"/>
  <body name="cup_part" pos=".02 .01 -.01" quat=".923879532511 .382683432365 0 0">
    <geom type="mesh" mesh="cup" pos=".01 .02 0" euler="-.1 .2 0"
          contype="0" conaffinity="0"/>
    <geom type="mesh" mesh="cup" name="collision"/>
  </body>
</body></worldbody></mujoco>
"""


def scene(count):
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", integrator="implicitfast")
    ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    for index in range(count):
        body = ET.SubElement(
            world, "body", name=f"mug_{index + 1}", pos=f"{index} 0 .5"
        )
        ET.SubElement(body, "freejoint", name=f"mug_{index + 1}_jnt")
        ET.SubElement(body, "geom", type="sphere", size=".01", mass=".05")
    return ET.tostring(root, encoding="unicode")


def visual_triangles(model, data, root):
    triangles = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if model.geom_group[geom] != 2 or not mugs._mujoco_body_in_subtree(
            model, body, root
        ):
            continue
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        vertices = model.mesh_vert[
            start : start + int(model.mesh_vertnum[mesh])
        ].astype(float)
        vertices = (
            vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        )
        vertices = (vertices - data.xpos[root]) @ data.xmat[root].reshape(3, 3)
        start = int(model.mesh_faceadr[mesh])
        faces = model.mesh_face[start : start + int(model.mesh_facenum[mesh])]
        triangles.append(vertices[faces])
    return np.concatenate(triangles)


class MugAssemblyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.roots = {}
        for task in ("mug_tree", "mug_flip"):
            self.roots[task] = self.root / task
            variant = self.roots[task] / "mug_0"
            variant.mkdir(parents=True)
            (variant / "model.xml").write_text(ASSET)
        self.patch = patch.object(mugs, "_MUG_TASK_ASSET_ROOTS", self.roots)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        mugs._mug_unit_surface_inertia.cache_clear()
        self.addCleanup(mugs._mug_unit_surface_inertia.cache_clear)

    def compile(self, task, scales):
        count = len(mugs._MUG_TASK_BODY_NAMES[task])
        xml = mugs._build_mug_scene_xml(
            task_name=task,
            mug_variant="mug_0",
            mug_variants=["mug_0"] * count,
            scale_states={f"mug_{i + 1}_jnt": s for i, s in enumerate(scales)},
            base_scene_xml=scene(count),
            base_scene_dir=self.root,
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, xml

    def test_every_mug_in_both_families_has_whole_surface_inertia(self):
        for task, scales in (
            ("mug_tree", [0.9, 1, 1.1]),
            ("mug_flip", [1.1, 1, 0.9, 1.03]),
        ):
            with self.subTest(task=task):
                model, data, _ = self.compile(task, scales)
                for index in range(len(scales)):
                    body = model.body(f"mug_{index + 1}").id
                    center, tensor = surface_inertia(
                        visual_triangles(model, data, body)
                    )
                    rotation = np.zeros(9)
                    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
                    rotation = rotation.reshape(3, 3)
                    actual = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
                    self.assertEqual(model.body_mass[body], MUG_MASS_KG)
                    self.assertEqual(model.body_subtreemass[body], MUG_MASS_KG)
                    np.testing.assert_allclose(model.body_ipos[body], center, atol=1e-8)
                    np.testing.assert_allclose(actual, tensor, rtol=2e-6, atol=1e-12)
                    for child in range(1, model.nbody):
                        if child != body and mugs._mujoco_body_in_subtree(
                            model, child, body
                        ):
                            self.assertEqual(model.body_mass[child], 0)

    def test_unit_cache_and_scale_laws(self):
        for task in ("mug_tree", "mug_flip"):
            center, tensor, anchor = mugs._mug_unit_surface_inertia(task, "mug_0")
            for factor in (0.9, 1.1):
                item = mugs._mug_wrapper_inertial(task, "mug_0", factor)
                np.testing.assert_allclose(
                    np.fromstring(item.get("pos"), sep=" "),
                    anchor + factor * (center - anchor),
                    atol=1e-15,
                )
                values = tensor[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
                np.testing.assert_allclose(
                    np.fromstring(item.get("fullinertia"), sep=" "), factor**2 * values
                )
                self.assertEqual(float(item.get("mass")), MUG_MASS_KG)
            self.assertFalse(center.flags.writeable)
            self.assertFalse(tensor.flags.writeable)
        self.assertEqual(mugs._mug_unit_surface_inertia.cache_info().currsize, 2)

    def test_inertial_emission_preserves_geometry_and_contact_parameters(self):
        model, data, xml = self.compile("mug_tree", [0.9, 1, 1.1])
        root = ET.fromstring(xml)
        for inertial in root.iter("inertial"):
            inertial.set("pos", "0 0 0")
            inertial.attrib.pop("fullinertia")
            inertial.set("diaginertia", ".001 .001 .001")
        reference = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        other = mujoco.MjData(reference)
        mujoco.mj_forward(reference, other)
        self.assertEqual(reference.names, model.names)
        for field in (
            "geom_friction",
            "geom_solref",
            "geom_solimp",
            "geom_contype",
            "geom_conaffinity",
        ):
            np.testing.assert_array_equal(
                getattr(model, field), getattr(reference, field)
            )
        for field in ("geom_xpos", "geom_xmat", "xpos", "xmat"):
            np.testing.assert_array_equal(getattr(data, field), getattr(other, field))

    def test_missing_visuals_and_invalid_scales_fail(self):
        (self.roots["mug_tree"] / "mug_0/model.xml").write_text(
            ASSET.replace('contype="0" conaffinity="0"', 'contype="1" conaffinity="1"')
        )
        with self.assertRaisesRegex(ValueError, "no visual"):
            mugs._mug_unit_surface_inertia("mug_tree", "mug_0")
        for scale in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                mugs._mug_wrapper_inertial("mug_flip", "mug_0", scale)


class MugRevisionTest(unittest.TestCase):
    def test_old_or_different_revision_cannot_restore(self):
        for cls in (MugVariantRandomizer, MugTreeRandomizer, MugFlipRandomizer):
            for metadata in ({}, {"mug_inertia_revision": "other"}):
                with self.subTest(cls=cls, metadata=metadata):
                    randomizer = cls()
                    state = RandomizationState(
                        seed=1, object_states={}, metadata=metadata
                    )
                    with patch.object(
                        randomizer, "_reload_mug_variant_scene"
                    ) as reload:
                        with self.assertRaisesRegex(ValueError, "inertia revision"):
                            randomizer._restore_scene(None, None, state)
                        reload.assert_not_called()

    def test_generated_metadata_declares_revision_for_each_family(self):
        for cls in (MugVariantRandomizer, MugTreeRandomizer, MugFlipRandomizer):
            randomizer = cls()

            def sampled(**kwargs):
                return RandomizationState(
                    seed=1, object_states={}, metadata=dict(kwargs["metadata"])
                )

            with (
                patch.object(MugFlipRandomizer, "_sample_and_apply_tray_color"),
                patch.object(
                    randomizer, "_resolve_mug_variants", return_value=["mug_0"]
                ),
                patch.object(randomizer, "_set_active_mug_count"),
                patch.object(randomizer, "_sample_scale_states", return_value={}),
                patch.object(randomizer, "_reload_mug_variant_scene"),
                patch.object(
                    randomizer, "_randomize_pose_with_rng", side_effect=sampled
                ),
                patch.object(randomizer, "_sample_plain_mug_colors", return_value={}),
            ):
                if cls is MugVariantRandomizer:
                    state = randomizer.randomize(None, None, seed=1)
                else:
                    with patch.object(
                        randomizer, "_try_randomize_pose_with_rng", side_effect=sampled
                    ):
                        state = randomizer.randomize(None, None, seed=1)
                self.assertEqual(
                    state.metadata["mug_inertia_revision"], MUG_INERTIA_REVISION
                )


if __name__ == "__main__":
    unittest.main()
