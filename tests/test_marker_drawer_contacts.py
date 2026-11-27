"""Solved-step marker support paths and side-effect-free score reads."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
from abc_sim.config import RobotSystemConfig
from abc_sim.env import MuJoCoYAMEnv
from abc_sim.randomization.core import RandomizationState, SceneRandomizer


def marker_model():
    root = ET.fromstring(
        '<mujoco><option timestep=".002" integrator="implicitfast"/>'
        '<default><geom friction=".8 .005 .0001" solref=".004 1"/></default>'
        "<worldbody/></mujoco>"
    )
    world = root.find("worldbody")
    for level, height in (("bottom", 0), ("middle", 0.3), ("top", 0.6)):
        drawer = ET.SubElement(world, "body", name=f"link_{level}_drawer_low", pos=f"0 0 {height}")
        ET.SubElement(
            drawer,
            "joint",
            name=f"joint_{level}_drawer_low",
            type="slide",
            axis="0 -1 0",
            range="0 .2",
            damping="20",
        )
        for pos, size in (
            ("0 0 0", ".12 .15 .005"),
            ("-.115 0 .04", ".005 .15 .04"),
            (".115 0 .04", ".005 .15 .04"),
            ("0 -.145 .04", ".11 .005 .04"),
            ("0 .145 .04", ".11 .005 .04"),
        ):
            ET.SubElement(drawer, "geom", type="box", pos=pos, size=size)
    for index in range(1, 4):
        marker = ET.SubElement(world, "body", name=f"marker_{index}", pos=f"{1 + index * 0.1} 0 .1")
        ET.SubElement(marker, "freejoint", name=f"marker_{index}_joint")
        ET.SubElement(
            marker,
            "geom",
            name=f"marker_{index}_collision",
            type="box",
            size=".035 .008 .008",
            mass=".01",
        )
    for arm in ("left", "right"):
        body = ET.SubElement(world, "body", name=f"{arm}_arm", pos="3 3 3", mocap="true")
        ET.SubElement(body, "geom", type="sphere", size=".01")
    support = ET.SubElement(world, "body", name="foreign_support", pos="3 0 3", mocap="true")
    ET.SubElement(support, "geom", type="box", size=".06 .06 .005")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


@pytest.fixture
def env():
    model = marker_model()
    with patch.object(MuJoCoYAMEnv, "setup_model", lambda instance: instance._bind_model(model)):
        instance = MuJoCoYAMEnv(RobotSystemConfig(), render_cameras=False, control_decimation=1)
    instance.get_init_q = lambda: np.zeros(0)
    instance.set_task("rl_put_markers_in_drawer")

    class FixtureRandomizer(SceneRandomizer):
        def randomize(self, model, data, seed=None, request=None):
            return RandomizationState(
                seed=seed or 0,
                object_states={},
                metadata={"marker_target_drawer": "bottom"},
            )

    instance._task_randomizer = FixtureRandomizer()
    instance._task_randomizer.bind_env(instance)
    try:
        instance.reset(seed=1)
        yield instance
    finally:
        instance.close()


def place(env, index, position, *, yaw=0):
    joint = env.data.joint(f"marker_{index}_joint")
    joint.qpos[:] = [*position, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    joint.qvel[:] = 0


def score_step(env):
    env.step(np.zeros(0), render_obs=False)
    return env.evaluate_task()


def vertical_stack(env, *, bottom=0.0129):
    for index in range(1, 4):
        place(env, index, [0, 0, bottom + (index - 1) * 0.0159])


@pytest.mark.parametrize("layout", ["stacked", "crossed"])
def test_released_marker_support_reaches_drawer_through_other_markers(env, layout):
    if layout == "stacked":
        vertical_stack(env)
        expected_direct = [True, False, False]
    else:
        place(env, 1, [0, -0.025, 0.0129])
        place(env, 2, [0, 0.025, 0.0129])
        place(env, 3, [0, 0, 0.0288], yaw=np.pi / 2)
        expected_direct = [True, True, False]
    result = score_step(env)
    np.testing.assert_array_equal(result.metrics["direct_drawer_support_mask"][0], expected_direct)
    assert result.metrics["placed_marker_mask"].all()
    assert result.metrics["placement_credit"] == pytest.approx(0.7)
    # Keep all marker free joints untouched while the support chain settles.
    for _ in range(100):
        result = score_step(env)
    assert result.metrics["placed_marker_mask"].all()


@pytest.mark.parametrize("invalid", ["held", "penetrating", "outside"])
def test_invalid_marker_cannot_carry_support_to_markers_above_it(env, invalid):
    vertical_stack(env, bottom=0.010 if invalid == "penetrating" else 0.0129)
    if invalid == "held":
        hand = env.model.body("left_arm").mocapid[0]
        env.data.mocap_pos[hand] = [-0.0445, 0, 0.0129]
    elif invalid == "outside":
        # The lower marker overlaps the inner wall; the two upper markers fit.
        place(env, 1, [0.09, 0, 0.0129])
        place(env, 2, [0.06, 0, 0.0288])
        place(env, 3, [0.06, 0, 0.0447])
    result = score_step(env)
    metric = {
        "held": "released_marker_mask",
        "penetrating": "nonpenetrating_marker_mask",
        "outside": "contained_marker_mask",
    }[invalid]
    assert not result.metrics[metric][0, 0]
    assert result.metrics["contained_marker_mask"][0, 1:].all()
    assert not result.metrics["placed_marker_mask"].any()
    assert result.metrics["placement_credit"] == 0


@pytest.mark.parametrize("foreign_support", [False, True])
def test_contained_floating_marker_chain_has_no_selected_drawer_root(env, foreign_support):
    vertical_stack(env, bottom=0.0289)
    if foreign_support:
        support = env.model.body("foreign_support").mocapid[0]
        env.data.mocap_pos[support] = [0, 0, 0.016]
    result = score_step(env)
    assert result.metrics["contained_marker_mask"].all()
    assert result.metrics["released_marker_mask"].all()
    assert result.metrics["nonpenetrating_marker_mask"].all()
    assert not result.metrics["direct_drawer_support_mask"].any()
    assert not result.metrics["placed_marker_mask"].any()


def test_env_observes_changed_solved_contacts_before_cached_score_reads(env):
    evaluator = env._task_evaluator
    assert env._physics_evaluator is evaluator
    assert not env._step_requires_position_refresh
    place(env, 1, [0, 0, 0.05])
    signatures = []
    native_step = mujoco.mj_step
    native_force = mujoco.mj_contactForce

    def solve(model, data):
        native_step(model, data)
        signatures.append(tuple((int(c.geom1), int(c.geom2)) for c in data.contact[: data.ncon]))

    def solved_force(model, data, index, force):
        contact = data.contact[index]
        assert (int(contact.geom1), int(contact.geom2)) == signatures[-1][index]
        native_force(model, data, index, force)

    with (
        patch("abc_sim.env.mujoco.mj_step", side_effect=solve),
        patch(
            "abc_sim.env.mujoco.mj_fwdPosition",
            side_effect=AssertionError("A cached solved-step score must not refresh contacts."),
        ),
        patch("abc_sim.task_eval.marker_drawer.mujoco.mj_contactForce", side_effect=solved_force),
    ):
        for _ in range(170):
            first = score_step(env)
        np.testing.assert_array_equal(first.metrics["placed_marker_mask"], [[True, False, False]])
        env.data.joint("marker_1_joint").qvel[2] = 2
        # Separate the drop lane along the narrow axis so the rising marker
        # cannot strike the falling marker before it reaches the drawer floor.
        place(env, 2, [0, 0.06, 0.05])
        for _ in range(170):
            second = score_step(env)
        np.testing.assert_array_equal(
            second.metrics["placed_marker_mask"],
            [[False, True, False]],
            err_msg=str(second.metrics),
        )
    marker_pair = {env.model.geom(f"marker_{index}_collision").id for index in (1, 2)}
    assert all(set(pair) != marker_pair for signature in signatures for pair in signature)
    assert len(set(signatures)) >= 3
    with patch(
        "abc_sim.task_eval.marker_drawer.mujoco.mj_contactForce",
        side_effect=AssertionError("Reading a cached score must not observe contacts."),
    ):
        assert env.evaluate_task() is second
        assert env.evaluate_task() is second


def test_marker_reset_clears_banked_opening_and_cached_contents(env):
    env.data.joint("joint_bottom_drawer_low").qpos[0] = 0.2
    vertical_stack(env)
    for index in range(1, 4):
        env.data.joint(f"marker_{index}_joint").qpos[1] -= 0.2
    result = score_step(env)
    assert result.metrics["opening_credit"] == pytest.approx(0.2)
    assert result.metrics["placement_credit"] == pytest.approx(0.7)
    env.reset(seed=2)
    reset = env.evaluate_task()
    assert reset.scalar_reward() == 0
    assert not reset.metrics["placed_marker_mask"].any()
