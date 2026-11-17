"""Conveyor belt speed: 0.10-0.20 m/s by default; a reset request can move either bound."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

import numpy as np

from abc_sim import make_env
from abc_sim.randomization.core import RandomizationState
from abc_sim.randomization.tasks.conveyor_pick import (
    ConveyorPickObjectRandomizer,
    _conveyor_belt_speed_range,
)
from tests.test_spelling_goals import blocks_xml


def conveyor_xml():
    root = ET.fromstring(blocks_xml())
    ET.SubElement(root, "option", gravity="0 0 0")
    world = root.find("worldbody")
    for body in list(world):
        if body.tag == "geom" or body.get("name", "").startswith("block_"):
            world.remove(body)
    for arm in ("left", "right"):
        owner = next(
            body
            for body in root.iter("body")
            if body.find(f"joint[@name='{arm}_left_finger']") is not None
        )
        owner.find("joint").set("type", "slide")
        owner.find("joint").set("range", "0 .0475")
        passive = ET.SubElement(owner, "body", name=f"{arm}_passive_finger_body")
        ET.SubElement(
            passive, "joint", name=f"{arm}_right_finger", type="slide", range="-.0475 0"
        )
        ET.SubElement(
            passive,
            "geom",
            type="sphere",
            size=".01",
            mass=".1",
            contype="0",
            conaffinity="0",
        )
        root.find(f"actuator/position[@name='{arm}_gripper']").set(
            "ctrlrange", "0 .0475"
        )
    for index in range(1, 4):
        body = ET.SubElement(
            world, "body", name=f"conveyor_object_{index}", pos=f"0 {index * 0.1} .019"
        )
        ET.SubElement(body, "freejoint", name=f"conveyor_object_{index}_joint")
        ET.SubElement(body, "geom", type="box", size=".01 .01 .01", mass=".01")
    bin_body = ET.SubElement(world, "body", name="conveyor_bin", pos="1 0 0")
    for pos, size in (
        ("0 0 0", ".2 .2 .01"),
        (".2 0 .1", ".01 .2 .1"),
        ("-.2 0 .1", ".01 .2 .1"),
        ("0 .2 .1", ".2 .01 .1"),
        ("0 -.2 .1", ".2 .01 .1"),
    ):
        ET.SubElement(bin_body, "geom", type="box", pos=pos, size=size)
    ET.SubElement(
        world, "geom", name="conveyor_belt_surface", type="box", size=".4 .4 .01"
    )
    for name in (
        "table_collision",
        "floor_collision",
        "back_wall_collision",
        "left_wall_collision",
        "right_wall_collision",
    ):
        ET.SubElement(
            world, "geom", name=name, type="box", pos="10 10 10", size=".1 .1 .1"
        )
    for index, side in enumerate(("infeed", "outfeed")):
        name = f"conveyor_{side}_roller"
        body = ET.SubElement(world, "body", pos=f"-3 {index} 0")
        ET.SubElement(body, "joint", name=name)
        ET.SubElement(body, "geom", size=".01", mass=".1", contype="0", conaffinity="0")
        ET.SubElement(root.find("actuator"), "velocity", name=f"{name}_vel", joint=name)
    return ET.tostring(root, encoding="unicode")


class ConveyorBeltSpeedTest(unittest.TestCase):
    def speeds(self, request=None, seeds=range(12)):
        env = make_env("conveyor_pick", render_cameras=False)
        self.addCleanup(env.close)
        options = None if request is None else {"randomization": request}
        result = []
        for seed in seeds:
            _, info = env.reset(seed=seed, options=options)
            metadata = info["randomization"].metadata
            self.assertAlmostEqual(
                abs(env._task_runtime._belt_speed_y), metadata["belt_speed_m_s"]
            )
            result.append(
                (metadata["belt_speed_m_s"], tuple(metadata["belt_speed_range_m_s"]))
            )
        return result

    def test_default_range_is_unchanged(self):
        for speed, speed_range in self.speeds():
            self.assertEqual(speed_range, (0.10, 0.20))
            self.assertTrue(0.10 <= speed <= 0.20)

    def test_requested_bounds_rescale_the_same_draw(self):
        default = [speed for speed, _ in self.speeds()]
        wide = self.speeds({"belt_speed_min_m_s": 0.05, "belt_speed_max_m_s": 0.30})
        for (speed, speed_range), base in zip(wide, default):
            self.assertEqual(speed_range, (0.05, 0.30))
            self.assertAlmostEqual(
                (speed - 0.05) / 0.25, (base - 0.10) / 0.10, places=6
            )
        self.assertTrue(
            any(speed < 0.10 for speed, _ in wide)
            and any(speed > 0.20 for speed, _ in wide)
        )

    def test_either_bound_alone_and_validation(self):
        self.assertEqual(_conveyor_belt_speed_range(None), (0.10, 0.20))
        self.assertEqual(
            _conveyor_belt_speed_range({"belt_speed_min_m_s": 0.05}), (0.05, 0.20)
        )
        self.assertEqual(
            _conveyor_belt_speed_range({"belt_speed_max_m_s": 0.3}), (0.10, 0.3)
        )
        for bad in (
            {"belt_speed_min_m_s": 0.3},
            {"belt_speed_min_m_s": 0.0},
            {"belt_speed_max_m_s": 0.6},
            {"belt_speed": 0.1},
        ):
            with self.subTest(request=bad), self.assertRaises(ValueError):
                _conveyor_belt_speed_range(bad)

    def test_public_replay_and_reset_preserve_requested_robot_pose(self):
        env = make_env(
            "conveyor_pick",
            scene_xml_string=conveyor_xml(),
            enable_task_randomizer=False,
            render_cameras=False,
        )
        self.addCleanup(env.close)
        env.reset(randomize=False)
        for arm in ("left", "right"):
            for side, endpoint in (("left", 1), ("right", 0)):
                joint = env.model.joint(f"{arm}_{side}_finger")
                self.assertEqual(env.data.qpos[joint.qposadr[0]], joint.range[endpoint])
            actuator = env.model.actuator(f"{arm}_gripper")
            self.assertEqual(env.data.ctrl[actuator.id], actuator.ctrlrange[1])
        initial_qpos = env.data.qpos.copy()
        initial_ctrl = env.data.ctrl.copy()
        restorer = ConveyorPickObjectRandomizer()
        restorer.bind_env(env)
        saved = RandomizationState(
            seed=31,
            object_states={
                f"conveyor_object_{index}_joint": {
                    "pos": [0, index * 0.1, 0.019],
                    "quat": [1, 0, 0, 0],
                }
                for index in range(1, 4)
            },
            metadata={"belt_speed_y": -0.17},
        )
        robot_names = [
            *(
                f"{arm}_joint{index}"
                for arm in ("left", "right")
                for index in range(1, 7)
            ),
            *(
                f"{arm}_{side}_finger"
                for arm in ("left", "right")
                for side in ("left", "right")
            ),
        ]
        for operation in ("replay", "ordinary_reset"):
            for keep in (True, False):
                with self.subTest(operation=operation, keep_arm_state=keep):
                    env.advance_physics(1)
                    self.assertEqual(
                        env.task_runtime_debug_state()["active_objects"], (1, 2, 3)
                    )
                    expected = initial_qpos.copy()
                    for index, name in enumerate(robot_names):
                        address = env.model.joint(name).qposadr[0]
                        moved = (
                            -0.011 - index * 0.0007
                            if name.endswith("_right_finger")
                            else 0.011 + index * 0.0007
                            if name.endswith("_left_finger")
                            else 0.1 + index * 0.017
                        )
                        env.data.qpos[address] = moved
                        if keep:
                            expected[address] = moved
                    object_address = env.model.joint("conveyor_object_1_joint").qposadr[
                        0
                    ]
                    env.data.qpos[object_address : object_address + 3] = [4, 5, 6]
                    env.data.qvel[:] = 0.3
                    env.data.ctrl[:] = 0.2
                    env.data.time = 12
                    env.cur_step = 7
                    if operation == "replay":
                        restorer.apply(env.model, env.data, saved, keep_arm_state=keep)
                    else:
                        env.reset(randomize=False, options={"keep_arm_state": keep})
                    np.testing.assert_array_equal(env.data.qpos, expected)
                    np.testing.assert_array_equal(
                        env.data.ctrl,
                        np.zeros_like(initial_ctrl) if keep else initial_ctrl,
                    )
                    np.testing.assert_array_equal(env.data.qvel, 0)
                    self.assertEqual(env.data.time, 0)
                    self.assertEqual(env.cur_step, 0)
                    self.assertEqual(env.get_state().shape, (14,))
                    runtime = env.task_runtime_debug_state()
                    self.assertEqual(runtime["active_objects"], ())
                    self.assertEqual(runtime["removed_objects"], ())
                    self.assertEqual(
                        runtime["belt_speed_y"],
                        -0.17 if operation == "replay" else -0.20,
                    )

        restorer.apply(env.model, env.data, saved)
        addresses = [
            env.model.joint(f"conveyor_object_{index}_joint").qposadr[0] + 1
            for index in range(1, 4)
        ]
        before = env.data.qpos[addresses].copy()
        env.advance_physics(1)
        np.testing.assert_allclose(
            env.data.qpos[addresses] - before,
            -0.17 * env.model.opt.timestep,
            rtol=0,
            atol=1e-12,
        )
        for side in ("infeed", "outfeed"):
            self.assertEqual(
                env.data.ctrl[env.model.actuator(f"conveyor_{side}_roller_vel").id], 6
            )


if __name__ == "__main__":
    unittest.main()
