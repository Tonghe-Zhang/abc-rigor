"""Small native fixtures for conveyor reward history, exits and seeded metadata."""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np
import pytest
from abc_sim.randomization.core import RandomizationState
from abc_sim.randomization.tasks.rl_conveyor import (
    CATEGORY_POOL,
    TARGET_CATEGORIES,
    RLConveyorRandomizer,
    build_xml,
    episode_prompt,
    validate_episode,
)
from abc_sim.task_eval.rl_conveyor import RLConveyorEvaluator
from abc_sim.task_runtime.rl_conveyor import RLConveyorRuntime
from abc_sim.task_specs import get_task_spec


def metadata(count=3, targets=None, direction=-1):
    variant = "all" if targets is None else "category"
    target_mask = [True] * count if targets is None else targets
    category = "all" if targets is None else "fruit"
    return {
        "task_category": "rl",
        "variant": variant,
        "target_category": category,
        "objects": [
            {
                "index": i + 1,
                "joint": f"conveyor_object_{i + 1}_joint",
                "category": "apple" if target else "sponge",
                "variant": "fixture",
                "target": target,
            }
            for i, target in enumerate(target_mask)
        ],
        "belt_speed_y": direction * 0.1,
        "prompt": episode_prompt(variant, category, count),
    }


def fixture_xml(count=3):
    root = ET.Element("mujoco")
    ET.SubElement(
        root, "option", timestep=".002", gravity="0 0 -9.81", integrator="implicitfast"
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world, "geom", name="conveyor_belt_surface", type="box", size=".095 .5 .015"
    )
    ET.SubElement(
        world,
        "geom",
        name="table_collision",
        type="box",
        size="1 1 .01",
        pos="0 0 -.10",
    )
    ET.SubElement(
        world,
        "geom",
        name="floor_collision",
        type="plane",
        size="5 5 .1",
        pos="0 0 -.5",
    )
    container = ET.SubElement(world, "body", name="conveyor_bin", pos=".4 0 0")
    for name, pos, size in (
        ("bottom", "0 0 0", ".12 .175 .01"),
        ("left", "-.12 0 .085", ".01 .175 .085"),
        ("right", ".12 0 .085", ".01 .175 .085"),
        ("front", "0 -.175 .085", ".12 .01 .085"),
        ("back", "0 .175 .085", ".12 .01 .085"),
    ):
        ET.SubElement(
            container,
            "geom",
            name=f"conveyor_bin_{name}",
            type="box",
            pos=pos,
            size=size,
        )
    for i in range(1, count + 1):
        obj = ET.SubElement(
            world,
            "body",
            name=f"conveyor_object_{i}",
            pos=f"0 {0.42 - 0.12 * (i - 1)} .0345",
        )
        ET.SubElement(obj, "freejoint", name=f"conveyor_object_{i}_joint")
        ET.SubElement(
            obj, "geom", name=f"object_{i}", type="sphere", size=".02", mass=".02"
        )
    actuators = ET.SubElement(root, "actuator")
    for side in ("left", "right"):
        arm = ET.SubElement(world, "body", name=f"{side}_arm", pos="2 0 2")
        ET.SubElement(arm, "freejoint", name=f"{side}_arm_pose")
        ET.SubElement(
            arm,
            "geom",
            type="sphere",
            size=".01",
            mass="1",
            contype="0",
            conaffinity="0",
        )
        for finger, x in (("left", -0.031), ("right", 0.031)):
            body = ET.SubElement(
                arm, "body", name=f"{side}_link_{finger}_finger", pos=f"{x} 0 0"
            )
            ET.SubElement(
                body,
                "joint",
                name=f"{side}_{finger}_finger",
                type="slide",
                axis="1 0 0",
                range="-.0475 .0475",
            )
            ET.SubElement(
                body,
                "geom",
                name=f"{side}_{finger}_pad",
                type="sphere",
                size=".012",
                mass=".1",
            )
        ET.SubElement(
            actuators,
            "position",
            name=f"{side}_gripper",
            joint=f"{side}_left_finger",
            ctrlrange="0 .0475",
        )
    for side, y in (("infeed", 0.52), ("outfeed", -0.52)):
        body = ET.SubElement(world, "body", pos=f"0 {y} 0")
        ET.SubElement(body, "joint", name=f"{side}_roller")
        ET.SubElement(body, "geom", size=".01", mass=".1", contype="0", conaffinity="0")
        ET.SubElement(
            actuators,
            "velocity",
            name=f"conveyor_{side}_roller_vel",
            joint=f"{side}_roller",
        )
    return ET.tostring(root, encoding="unicode")


class Fixture:
    def __init__(self, count=3, targets=None, direction=-1, xml=None):
        self.model = mujoco.MjModel.from_xml_string(
            fixture_xml(count) if xml is None else xml
        )
        self.data = mujoco.MjData(self.model)
        self.metadata = metadata(count, targets, direction)
        self.evaluator = RLConveyorEvaluator(
            model=self.model, spec=get_task_spec("rl_conveyor_pick")
        )
        self.evaluator.configure_episode(self.model, self.metadata)
        mujoco.mj_forward(self.model, self.data)
        self.evaluator.after_reset(self.model, self.data)

    def move(self, index, pos):
        self.data.joint(f"conveyor_object_{index}_joint").qpos[:3] = pos

    def arm(self, side, pos):
        self.data.joint(f"{side}_arm_pose").qpos[:3] = pos

    def sample(self, seconds=0.002):
        for _ in range(round(seconds / self.model.opt.timestep)):
            self.data.time += self.model.opt.timestep
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            self.evaluator.after_physics_step(self.model, self.data)
        result = self.evaluator.evaluate_model_data(self.model, self.data)
        assert 0 <= result.scalar_reward() <= 1
        return result

    def grasp(self, index=1, side="left"):
        pos = [0, 0.42 - 0.12 * (index - 1), 0.10]
        self.move(index, pos)
        self.arm(side, pos)
        self.sample(0.06)
        assert self.evaluator._grasped[index - 1]

    def place(self, index=1):
        self.arm("left", [2, 0, 2])
        self.arm("right", [2, 0, 2])
        self.move(index, [0.4, -0.09 + 0.06 * (index - 1), 0.0295])
        return self.sample(0.51)


def test_original_task_registry_is_unchanged_and_new_task_is_distinct():
    from abc_sim.randomization.registry import TASK_RANDOMIZERS
    from abc_sim.task_eval.registry import _TASK_EVALUATORS
    from abc_sim.task_runtime.registry import _TASK_RUNTIMES

    assert (
        get_task_spec("conveyor_pick").evaluator_name == "conveyor_pick_objects_in_bin"
    )
    assert _TASK_RUNTIMES["conveyor_pick"].__name__ == "ConveyorPickRuntime"
    assert (
        type(TASK_RANDOMIZERS["conveyor_pick"]).__name__
        == "ConveyorPickObjectRandomizer"
    )
    assert _TASK_EVALUATORS["rl_conveyor_pick"] is RLConveyorEvaluator


@pytest.mark.parametrize(
    "targets,initial",
    [
        ([True, False], -0.5),
        ([True, True, False, False], -0.5),
        ([True, False, False, False], -1.5),
    ],
)
def test_pending_liability_is_present_on_initial_and_early_done_reads(targets, initial):
    f = Fixture(len(targets), targets)
    result = f.evaluator.evaluate_model_data(f.model, f.data)
    assert result.scalar_reward() == 0
    assert result.metrics["unclipped_reward"][0] == pytest.approx(initial)
    assert result.metrics["pending_distractor_count"][0] == targets.count(False)
    assert result.metrics["failed_distractor_count"][0] == 0
    for _ in range(5):
        assert f.evaluator.evaluate_model_data(f.model, f.data).scalar_reward() == 0
    assert not result.scalar_success()


def test_tall_supported_deposit_can_protrude_above_rim():
    root = ET.fromstring(fixture_xml())
    root.find(".//geom[@name='object_1']").set("type", "box")
    root.find(".//geom[@name='object_1']").set("size", ".02 .02 .10")
    f = Fixture(xml=ET.tostring(root, encoding="unicode"))
    f.move(1, [0, 0.42, 0.20])
    f.arm("left", [0, 0.42, 0.20])
    f.sample(0.06)
    assert f.evaluator._grasped[0]
    f.arm("left", [2, 0, 2])
    f.move(1, [0.4, -0.09, 0.1095])
    result = f.sample(0.51)
    assert f.evaluator._bounds(f.data)[1][0, 2] > f.evaluator._bin_upper[2]
    assert result.metrics["placement_credit"][0, 0] == pytest.approx(0.20)

    # Floating, wall contact and an outside footprint cannot retain a deposit.
    for pos in ([0.4, -0.09, 0.13], [0.51, -0.09, 0.1095], [0.7, -0.09, 0.1095]):
        f.move(1, pos)
        result = f.sample(0.51)
        assert result.metrics["placement_credit"][0, 0] == 0


def test_three_targets_get_exact_units_only_after_grasp_and_supported_settle():
    f = Fixture()
    f.move(1, [0.4, -0.09, 0.0295])
    assert f.sample(0.6).scalar_reward() == 0
    for i in range(1, 4):
        f.grasp(i)
        f.place(i)
    result = f.evaluator.evaluate_model_data(f.model, f.data)
    np.testing.assert_allclose(result.metrics["grasp_credit"], [[0.05] * 3])
    np.testing.assert_allclose(result.metrics["placement_credit"], [[0.20] * 3])
    np.testing.assert_allclose(result.metrics["gentle_credit"], [[1 / 12] * 3])
    assert result.scalar_reward() == pytest.approx(1)
    assert result.scalar_success()


@pytest.mark.parametrize("target_count", [1, 2])
def test_category_units_and_success_wait_for_every_normal_exit(target_count):
    targets = [True] * target_count + [False] * (4 - target_count)
    f = Fixture(4, targets)
    for i in range(1, target_count + 1):
        f.grasp(i)
        f.place(i)
    result = f.evaluator.evaluate_model_data(f.model, f.data)
    np.testing.assert_allclose(
        result.metrics["grasp_credit"][0, :target_count], 0.15 / target_count
    )
    np.testing.assert_allclose(
        result.metrics["placement_credit"][0, :target_count], 0.6 / target_count
    )
    np.testing.assert_allclose(
        result.metrics["gentle_credit"][0, :target_count], 0.25 / target_count
    )
    assert not result.scalar_success()
    for i in range(target_count + 1, 5):
        f.move(i, [0, -0.499, 0.0345])
        f.sample()
        f.move(i, [0, -0.501, 0.0345])
        result = f.sample()
    assert result.scalar_reward() == pytest.approx(1)
    assert result.scalar_success()
    assert not result.metrics["distractor_penalty"].any()


def test_floor_preserves_penalties_and_refunds_on_positive_partial_credit():
    f = Fixture(4, [True, False, False, False])
    f.grasp()
    result = f.place()
    assert result.scalar_reward() == 0
    assert result.metrics["unclipped_reward"][0] == pytest.approx(-0.5)
    for index, expected in ((2, 0), (3, 0.5), (4, 1)):
        f.move(index, [0, -0.499, 0.0345])
        f.sample()
        f.move(index, [0, -0.501, 0.0345])
        assert f.sample().scalar_reward() == pytest.approx(expected)
    f = Fixture(4, [True, True, False, False])
    f.grasp()
    f.place()
    f.move(3, [0, -0.499, 0.0345])
    f.sample()
    f.move(3, [0, -0.501, 0.0345])
    assert f.sample().scalar_reward() == pytest.approx(0.25)


def test_one_finger_two_hands_low_lift_and_airborne_bin_do_not_qualify():
    f = Fixture()
    f.move(1, [0, 0.42, 0.10])
    f.arm("left", [-0.031, 0.42, 0.10])
    f.arm("right", [0.031, 0.42, 0.10])
    assert f.sample(0.1).scalar_reward() == 0
    f.arm("right", [2, 0, 2])
    f.move(1, [0, 0.42, 0.06])
    f.arm("left", [0, 0.42, 0.06])
    assert f.sample(0.1).scalar_reward() == 0
    f.grasp()
    f.arm("left", [2, 0, 2])
    f.move(1, [0.4, 0, 0.10])
    result = f.sample(0.6)
    assert result.scalar_reward() == pytest.approx(0.05)
    assert not result.metrics["placed_mask"][0, 0]


def test_interrupted_grasps_and_settling_never_sum():
    f = Fixture()
    for _ in range(3):
        f.move(1, [0, 0.42, 0.10])
        f.arm("left", [0, 0.42, 0.10])
        f.sample(0.03)
        f.arm("left", [2, 0, 2])
        f.sample()
    assert not f.evaluator._grasped[0]
    f.grasp()
    f.arm("left", [2, 0, 2])
    for _ in range(3):
        f.move(1, [0.4, 0, 0.0295])
        f.sample(0.3)
        f.move(1, [0.4, 0, 0.10])
        f.sample()
    assert not f.evaluator._placed[0]
    f.place()
    assert f.evaluator._placed[0]


def test_grasp_and_replacement_cannot_accumulate_or_restore_force_bonus():
    f = Fixture()
    f.grasp()
    first = f.place().scalar_reward()
    assert first == pytest.approx(1 / 3)
    for _ in range(3):
        f.grasp()
        assert f.place().scalar_reward() == pytest.approx(first)
    # A later excessive contact removes the bonus permanently, even after replacement.
    f.grasp()
    real = mujoco.mj_contactForce

    def high_force(model, data, index, force):
        real(model, data, index, force)
        force[0] += 20

    with mock.patch(
        "abc_sim.task_eval.rl_conveyor.mujoco.mj_contactForce", side_effect=high_force
    ):
        f.sample()
    assert f.place().scalar_reward() == pytest.approx(0.25)
    f.grasp()
    assert f.place().scalar_reward() == pytest.approx(0.25)


def test_reentry_requires_a_fresh_released_supported_settle():
    f = Fixture()
    f.grasp()
    assert f.place().scalar_reward() == pytest.approx(1 / 3)
    f.move(1, [0.4, 0, 0.10])
    assert f.sample().scalar_reward() == pytest.approx(0.05)
    f.move(1, [0.4, 0, 0.0295])
    assert f.sample(0.3).scalar_reward() == pytest.approx(0.05)
    assert f.sample(0.21).scalar_reward() == pytest.approx(1 / 3)
    f.arm("left", [0.4, 0, 0.0295])
    assert f.sample().scalar_reward() == pytest.approx(0.05)


def test_current_tool_push_blocks_exit_but_clean_belt_recovery_is_allowed():
    f = Fixture(2, [True, False])
    f.grasp(1)
    f.arm("left", [0, -0.461, 0.035])
    f.move(1, [0, -0.461, 0.035])
    f.move(2, [0, -0.499, 0.035])
    result = f.sample()
    assert f.evaluator._robot_touched[1]
    assert result.metrics["failed_distractor_count"][0] == 0
    assert result.metrics["pending_distractor_count"][0] == 1
    f.arm("left", [0, -0.463, 0.035])
    f.move(1, [0, -0.463, 0.035])
    f.move(2, [0, -0.501, 0.035])
    result = f.sample()
    assert not result.metrics["normal_exit_mask"][0, 1]
    assert result.metrics["distractor_penalty"][0, 1] == 0.5
    f.arm("left", [2, 0, 2])
    f.move(1, [0, 0.2, 0.0345])
    f.move(2, [0, -0.499, 0.0345])
    f.sample()
    f.move(2, [0, -0.501, 0.0345])
    result = f.sample()
    assert result.metrics["normal_exit_mask"][0, 1]
    assert result.metrics["distractor_penalty"][0, 1] == 0


def test_passive_spawn_force_does_not_forfeit_gentle_handling():
    f = Fixture()
    real = mujoco.mj_contactForce

    def high_force(model, data, index, force):
        real(model, data, index, force)
        force[0] += 20

    with mock.patch(
        "abc_sim.task_eval.rl_conveyor.mujoco.mj_contactForce", side_effect=high_force
    ):
        f.sample()
    assert not f.evaluator._peak_force.any()
    f.grasp()
    assert f.place().scalar_reward() == pytest.approx(1 / 3)


@pytest.mark.parametrize("direction", [-1, 1])
def test_unassisted_dynamic_belt_exits_refund_only_distractors(direction):
    f = Fixture(4, [True, False, False, False], direction)
    runtime = RLConveyorRuntime()
    runtime.bind(f.model, f.data)
    runtime.after_reset(
        f.model,
        f.data,
        randomization=SimpleNamespace(metadata=f.metadata),
        episode=f.metadata,
    )
    for _ in range(5200):
        runtime.before_step(f.model, f.data)
        mujoco.mj_step(f.model, f.data)
        f.evaluator.after_physics_step(f.model, f.data)
        result = f.evaluator.evaluate_model_data(f.model, f.data)
        runtime.on_task_evaluation(result)
        runtime.after_step(f.model, f.data)
        if result.metrics["normal_exit_mask"].all():
            break
    assert result.metrics["normal_exit_mask"].all()
    assert result.scalar_reward() == 0
    assert not result.scalar_success()
    assert not result.metrics["ever_failed"][0]
    result = f.sample(3.01)
    assert result.metrics["ever_failed"][0]
    assert len(runtime._removed_objects) == 4
    assert all(
        f.data.joint(f"conveyor_object_{i}_joint").qpos[0] == 3 for i in range(1, 5)
    )


@pytest.mark.parametrize("case", ["upstream", "side", "held", "teleported"])
def test_wrong_exit_never_refunds_distractor(case):
    f = Fixture(2, [True, False])
    if case == "upstream":
        f.move(2, [0, 0.51, 0.035])
    elif case == "side":
        f.move(2, [0.3, -0.51, -0.0695])
    elif case == "held":
        f.arm("left", [0, 0.30, 0.10])
        f.move(2, [0, 0.30, 0.10])
        f.sample(0.06)
        f.arm("left", [0, -0.51, 0.10])
        f.move(2, [0, -0.51, 0.10])
    else:
        f.move(2, [3, 0, 0.2])
    result = f.sample()
    assert result.scalar_reward() == 0
    assert result.metrics["distractor_penalty"][0, 1] == 0.5
    assert not result.metrics["normal_exit_mask"][0, 1]
    assert not result.scalar_success()


def test_removed_object_reappearance_is_not_scoreable_and_worlds_reset_independently():
    f, other = Fixture(), Fixture()
    f.grasp()
    f.move(1, [3, 0, 0.2])
    f.sample()
    assert f.place().scalar_reward() == pytest.approx(0.05)
    assert not f.evaluator._placed[0]
    assert (
        other.evaluator.evaluate_model_data(other.model, other.data).scalar_reward()
        == 0
    )
    f.evaluator.reset()
    f.evaluator.configure_episode(f.model, f.metadata)
    mujoco.mj_resetData(f.model, f.data)
    mujoco.mj_forward(f.model, f.data)
    f.evaluator.after_reset(f.model, f.data)
    assert f.evaluator.evaluate_model_data(f.model, f.data).scalar_reward() == 0
    with pytest.raises(NotImplementedError):
        f.evaluator.reset(nworld=2)
    with pytest.raises(ValueError, match="exactly once"):
        f.evaluator.after_physics_step(f.model, f.data)
    with pytest.raises(NotImplementedError):
        f.evaluator.evaluate_qpos_batch(f.data.qpos[None])


def test_category_mask_prompt_and_count_validation():
    original = metadata(4, [True, True, False, False])
    validate_episode(original)
    for change in (
        lambda m: m.update(prompt="collect everything"),
        lambda m: m["objects"][0].update(target=False),
        lambda m: m["objects"][1].update(index=1),
        lambda m: m.update(belt_speed_y=float("nan")),
        lambda m: m.update(task_category="ordinary"),
    ):
        changed = copy.deepcopy(original)
        change(changed)
        with pytest.raises(ValueError):
            validate_episode(changed)


def test_seeded_sampler_covers_both_variants_categories_counts_and_orders():
    randomizer = RLConveyorRandomizer()
    results = []
    with (
        mock.patch.object(
            randomizer,
            "_get_variants",
            side_effect=lambda c: [Path("/fixture") / c / "v1"],
        ),
        mock.patch(
            "abc_sim.randomization.tasks.rl_conveyor._conveyor_pick_spawn_z",
            return_value=0.84,
        ),
        mock.patch.object(randomizer, "_reload"),
    ):
        for seed in range(128):
            state = randomizer.randomize(None, None, seed=seed)
            validate_episode(state.metadata)
            results.append(state.to_dict())
        assert randomizer.randomize(None, None, seed=7).to_dict() == results[7]
    assert {m["metadata"]["variant"] for m in results} == {"all", "category"}
    category = [
        m["metadata"] for m in results if m["metadata"]["variant"] == "category"
    ]
    assert {m["target_category"] for m in category} == set(TARGET_CATEGORIES)
    assert {len(m["objects"]) for m in category} == {2, 3, 4}
    assert {sum(o["target"] for o in m["objects"]) for m in category} == {1, 2}
    assert all(o["category"] in CATEGORY_POOL for m in category for o in m["objects"])
    assert any(not m["objects"][0]["target"] for m in category)
    assert any(m["objects"][0]["target"] for m in category)
    assert json.loads(json.dumps(results[7])) == results[7]


@pytest.mark.parametrize("count", [2, 3, 4])
def test_builder_rewrites_home_keyframe_for_variable_object_count(count):
    # Keep the existing builder's keyframe contract while removing external meshes.
    source = f'<mujoco><keyframe><key name="home" qpos="{" ".join(["0"] * 39)}"/></keyframe></mujoco>'
    selections = [{"x": 0, "y": 0.1 * i, "z": 0.8, "yaw": 0} for i in range(count)]
    with mock.patch(
        "abc_sim.randomization.tasks.rl_conveyor._conveyor_pick_build_xml",
        return_value=source,
    ):
        root = ET.fromstring(build_xml(selections))
    assert len(root.find("./keyframe/key").get("qpos").split()) == 2 + 7 * count + 16


def test_replay_checks_category_and_pose_before_reload():
    randomizer = RLConveyorRandomizer()
    state = RandomizationState(
        seed=1,
        metadata=metadata(2, [True, False]),
        object_states={
            f"conveyor_object_{i}_joint": {
                "pos": [0, 0.4 - 0.1 * i, 0.035],
                "quat": [1, 0, 0, 0],
            }
            for i in (1, 2)
        },
    )
    with mock.patch.object(
        randomizer, "_get_variants", return_value=[Path("/fixture") / "fixture"]
    ):
        assert len(randomizer._replay_selections(state)) == 2
        state.object_states["conveyor_object_1_joint"]["quat"] = [0, 0, 0, 0]
        with pytest.raises(ValueError, match="unit quaternions"):
            randomizer._replay_selections(state)


def test_sampled_state_is_applied_at_recorded_precision():
    f = Fixture(2, [True, False])
    randomizer = RLConveyorRandomizer()
    randomizer._env_ref = SimpleNamespace(model=f.model, data=f.data)
    with (
        mock.patch.object(
            randomizer,
            "_get_variants",
            side_effect=lambda c: [Path("/fixture") / c / "v1"],
        ),
        mock.patch(
            "abc_sim.randomization.tasks.rl_conveyor._conveyor_pick_spawn_z",
            return_value=0.84123456,
        ),
        mock.patch.object(randomizer, "_reload"),
    ):
        state = randomizer.randomize(f.model, f.data, seed=0)
    assert len(state.object_states) == 2
    for joint, pose in state.object_states.items():
        np.testing.assert_array_equal(
            f.data.joint(joint).qpos, [*pose["pos"], *pose["quat"]]
        )
        assert pose["pos"][2] != round(pose["pos"][2], 4)
