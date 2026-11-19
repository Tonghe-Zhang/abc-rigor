"""Named scene installation includes the meshes needed by the tower task."""

import unittest

import prepare


class PrepareAssetsTests(unittest.TestCase):
    def test_tower_task_resolves_the_building_block_package(self):
        self.assertEqual(
            prepare.sim_task_packages(["build_wood_block_tower"]),
            {"i2rt_yam", "building_blocks"},
        )
        self.assertIn("building_blocks", prepare.load_sim_manifest())

    def test_marker_drawer_tasks_resolve_both_mesh_packages(self):
        for level in ("top", "middle", "bottom"):
            self.assertEqual(
                prepare.sim_task_packages([f"put_markers_in_{level}_drawer"]),
                {"i2rt_yam", "drawer", "marker"},
            )


if __name__ == "__main__":
    unittest.main()
