"""Approved native RL variants: chronology, normalized credit and replay."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
from abc_sim.block_appearance import sample_native_wood_appearance, validate_appearance
from abc_sim.randomization.assets.block_appearance import (
    native_wood_pixels,
    original_atlas,
)
from abc_sim.randomization.assets.fruit_bowls import source_bowl_spec
from abc_sim.randomization.tasks.count_box import (
    COUNT_BOX_FAMILIES,
    CountIntoOpaqueBoxRandomizer,
)
from abc_sim.randomization.tasks.mugs import MugTreeRandomizer
from abc_sim.task_eval.count_box import CountIntoOpaqueBoxEvaluator
from abc_sim.task_eval.interrupted import (
    InterruptedClutterEvaluator,
    InterruptedRelativeEvaluator,
)
from abc_sim.task_eval.marker_drawer import MarkerDrawerEvaluator, drawer_interior
from abc_sim.task_runtime.interrupted import InterruptedInstructionRuntime
from abc_sim.task_runtime.registry import make_task_runtime
from abc_sim.task_runtime.rl_conveyor import RLConveyorRuntime
from abc_sim.task_specs import get_task_spec

from tests import test_ball_tray_window as tray_tests
from tests import test_multi_drawer_rewards as drawer_tests
from tests.test_rl_conveyor import Fixture, fixture_xml


def test_source_bowl_keeps_thickness_and_height_with_approved_cavity():
    bowl = source_bowl_spec()
    assert (bowl.bottom_radius, bowl.rim_radius, bowl.height) == (0.060, 0.070, 0.060)
    assert (bowl.wall, bowl.floor) == (0.0025, 0.003)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_all_pick_counts_have_exact_normalized_grasp_and_placement_shares(count):
    fixture = Fixture(count=count)
    for index in range(1, count + 1):
        fixture.grasp(index)
        fixture.move(index, [0.4, -0.10 + 0.08 * index, 0.0295])
        fixture.arm("left", [2, 0, 2])
        result = fixture.sample(0.51)
        assert result.scalar_reward() == pytest.approx(index / count)
    assert result.scalar_success()


def test_resolved_conveyor_grace_is_recoverable_and_ignores_wrong_bin():
    fixture = Fixture(count=2, targets=[True, False])
    fixture.evaluator._placed[:] = True
    fixture.evaluator._inside[:] = True
    fixture.evaluator._normal_exit[:] = False
    fixture.sample()
    assert fixture.evaluator._resolved_since is None
    fixture.evaluator._normal_exit[:] = True
    assert not fixture.sample(2.99).metrics["task_terminated"][0]
    assert fixture.sample(0.02).metrics["task_terminated"][0]


def test_conveyor_last_held_target_does_not_start_terminal_grace():
    fixture = Fixture(count=2)
    fixture.grasp(1)
    fixture.evaluator._normal_exit[1] = True
    assert not fixture.sample(4).metrics["task_terminated"][0]
    assert fixture.evaluator._resolved_since is None


def test_slowest_conveyor_last_spawn_exits_and_finishes_grace_before_thirty_seconds():
    fixture = Fixture(count=1)
    fixture.metadata["belt_speed_y"] = -0.05
    fixture.move(1, [0, 0.425, 0.0345])
    mujoco.mj_forward(fixture.model, fixture.data)
    fixture.evaluator.configure_episode(fixture.model, fixture.metadata)
    fixture.evaluator.after_reset(fixture.model, fixture.data)
    runtime = RLConveyorRuntime()
    runtime.bind(fixture.model, fixture.data)
    runtime.after_reset(
        fixture.model,
        fixture.data,
        randomization=SimpleNamespace(metadata=fixture.metadata),
        episode=fixture.metadata,
    )
    while fixture.data.time < 22:
        runtime.before_step(fixture.model, fixture.data)
        result = fixture.sample()
        runtime.on_task_evaluation(result)
        runtime.after_step(fixture.model, fixture.data)
        if result.metrics["task_terminated"][0]:
            break
    assert result.metrics["normal_exit_mask"][0, 0]
    assert result.metrics["task_terminated"][0]
    assert fixture.data.time == pytest.approx(21.5, abs=0.01)


def test_ball_loss_before_lift_terminates_only_after_grace_and_can_recover():
    fixture = tray_tests.BallTrayWindowTest()
    fixture.setUp()
    assert not fixture.sample(0.1, tray_tests.REST_Z, ball_inside=False).metrics["task_terminated"][0]
    assert not fixture.sample(0.9, tray_tests.REST_Z).metrics["task_terminated"][0]
    assert not fixture.sample(1.0, tray_tests.REST_Z, ball_inside=False).metrics["task_terminated"][0]
    assert not fixture.sample(1.999, tray_tests.REST_Z, ball_inside=False).metrics["task_terminated"][0]
    assert fixture.sample(2.001, tray_tests.REST_Z, ball_inside=False).metrics["task_terminated"][0]


def test_ball_bounce_and_safe_lowering_do_not_trigger_loss():
    fixture = tray_tests.BallTrayWindowTest()
    fixture.setUp()
    fixture.sample(0.1, tray_tests.UP_Z)
    fixture.sample(0.3, tray_tests.UP_Z, ball_height=0.08)
    result = fixture.sample(1.4, tray_tests.UP_Z, ball_height=0.08)
    assert not result.metrics["task_terminated"][0]
    fixture.sample(1.5, tray_tests.UP_Z)
    assert not fixture.sample(1.6, tray_tests.REST_Z).metrics["task_terminated"][0]


def test_ball_loss_before_first_support_observation_starts_one_second_grace():
    fixture = tray_tests.BallTrayWindowTest()
    fixture.setUp()
    fixture.evaluator.reset()
    initial = fixture.sample(0, tray_tests.REST_Z, ball_height=0.04)
    assert not initial.metrics["ball_tray_contact"][0]
    assert not initial.metrics["task_terminated"][0]
    assert not fixture.sample(0.01, tray_tests.REST_Z, ball_inside=False, ball_height=0.04).metrics["task_terminated"][
        0
    ]
    assert not fixture.sample(1.009, tray_tests.REST_Z, ball_inside=False, ball_height=0.04).metrics["task_terminated"][
        0
    ]
    assert fixture.sample(1.011, tray_tests.REST_Z, ball_inside=False, ball_height=0.04).metrics["task_terminated"][0]


@pytest.mark.parametrize("worlds", [1, 3])
def test_ball_loss_timestamp_is_json_null_until_loss_and_after_recovery(worlds):
    fixture = tray_tests.BallTrayWindowTest()
    fixture.setUp()
    fixture.evaluator.reset(nworld=worlds)
    supported = np.stack([fixture.data.qpos.copy()] * worlds)
    lost = supported.copy()
    address = fixture.model.joint("ball_tray_ball_joint").qposadr[0]
    lost[:, address] = 1.0
    for qpos, time_s, expected in (
        (supported, 0.0, None),
        (lost, 0.2, 0.2),
        (supported, 0.5, None),
    ):
        result = fixture.evaluator.evaluate_qpos_batch(qpos, time_s=time_s)
        assert result.metrics["ball_loss_started_at_s"].tolist() == [expected] * worlds
        for squeeze in (False, True):
            info = result.to_info(squeeze=squeeze)
            assert json.loads(json.dumps(info, allow_nan=False)) == info


def test_drawer_timing_uses_native_qpos_clock_without_position_refresh():
    fixture = drawer_tests.DrawerRewardsTest()
    fixture.setUp()
    env = fixture.make_env()
    try:
        env.reset(seed=4)
        initial = env.evaluate_task().to_info(squeeze=True)
        assert initial["second_prompt_at_s"] is None
        assert initial["second_retrieval_latency_s"] is None
        assert json.loads(json.dumps(initial, allow_nan=False)) == initial
        target = env.model.joint("drawer_search_target_joint")
        env.data.qpos[target.qposadr[0]] = -0.08
        env.data.qvel[target.dofadr[0]] = 10
        with patch(
            "abc_sim.env.mujoco.mj_fwdPosition",
            side_effect=AssertionError("Qpos/time drawer scoring must not refresh positions."),
        ):
            env.step(np.zeros(0), render_obs=False)
            result = env.evaluate_task().to_info(squeeze=True)
        assert 0 < result["second_prompt_at_s"] <= env.data.time
        assert result["second_retrieval_latency_s"] is None
        assert json.loads(json.dumps(result, allow_nan=False)) == result
    finally:
        env.close()


@pytest.mark.parametrize("observed, expected", [(False, 1.0), (True, 0.7)])
def test_recall_deduction_requires_delivered_target_exposure(observed, expected):
    fixture = drawer_tests.DrawerRewardsTest()
    fixture.setUp()
    fixture.open(5)  # A physics opening alone is not exposure.
    if observed:
        fixture.evaluator.record_observed_targets(["second"])
    fixture.data.joint("drawer_search_target_joint").qpos[:3] = 0
    fixture.score()
    for drawer in (0, 1, 2, 0):
        fixture.open(drawer)
    fixture.data.joint("second_joint").qpos[:3] = 0
    result = fixture.score()
    assert result.scalar_success()
    assert result.scalar_reward() == pytest.approx(expected)
    assert result.metrics["recall_wrong_drawer_count"][0] == (3 if observed else 0)


def test_recall_reward_has_distinct_declared_version_and_rejects_old_selection():
    from dataclasses import replace

    from abc_sim.task_eval.registry import make_task_evaluator

    fixture = drawer_tests.DrawerRewardsTest()
    fixture.setUp()
    assert fixture.spec.evaluator_name == "multi_drawer_search_shaped_v2"
    evaluator = make_task_evaluator(fixture.model, fixture.spec)
    assert type(evaluator) is type(fixture.evaluator)
    with pytest.raises(KeyError, match="multi_drawer_search_shaped_v1"):
        make_task_evaluator(
            fixture.model,
            replace(fixture.spec, evaluator_name="multi_drawer_search_shaped_v1"),
        )


def test_recall_deduction_preserves_first_retrieval_and_ignores_preopened_drawers():
    fixture = drawer_tests.DrawerRewardsTest()
    fixture.setUp()
    fixture.open(0)
    fixture.evaluator.record_observed_targets(["second"])
    fixture.data.joint("drawer_search_target_joint").qpos[:3] = 0
    first = fixture.score().scalar_reward()
    fixture.open(0)
    assert fixture.score().metrics["recall_wrong_drawer_count"][0] == 0
    for drawer in (1, 2, 3):
        assert fixture.open(drawer).scalar_reward() >= first


def test_recall_native_metrics_record_first_search_and_second_prompt_latency():
    fixture = drawer_tests.DrawerRewardsTest()
    fixture.setUp()
    fixture.evaluator.record_observed_targets(["second"])
    fixture.data.time = 4
    fixture.data.joint("drawer_search_target_joint").qpos[:3] = 0
    fixture.evaluator.evaluate_model_data(fixture.model, fixture.data)
    fixture.data.time = 5
    fixture.data.joint("drawer5").qpos[0] = 0.15
    first_search = fixture.evaluator.evaluate_model_data(fixture.model, fixture.data)
    assert first_search.metrics["recall_correct_drawer_first"][0]
    fixture.data.time = 8
    fixture.data.joint("second_joint").qpos[:3] = 0
    retrieved = fixture.evaluator.evaluate_model_data(fixture.model, fixture.data)
    assert retrieved.metrics["second_retrieval_latency_s"][0] == 4
    fixture.evaluator.reset()
    assert np.isnan(fixture.evaluator._second_requested_at).all()


def test_counting_family_first_covers_all_six_and_exact_one_with_replay():
    randomizer = CountIntoOpaqueBoxRandomizer()
    families, exact_counts = set(), set()
    for seed in range(120):
        state = randomizer.randomize(None, None, seed=seed, request={"family_first": True})
        repeat = randomizer.randomize(None, None, seed=seed, request={"family_first": True})
        assert state.metadata == repeat.metadata
        metadata = state.metadata
        families.add(metadata["prompt_type"])
        if metadata["prompt_type"] == "exact_count":
            exact_counts.add(metadata["target_count"])
        assert 1 <= metadata["target_count"] <= len(metadata["eligible_objects"])
    assert families == set(COUNT_BOX_FAMILIES)
    assert exact_counts == {1, 2, 3, 4}


@pytest.mark.parametrize(
    "correct, wrong, expected",
    [
        (2, 0, 2 / 3),
        (3, 0, 1.0),
        (4, 0, 0.4),
        (3, 1, 0.4),
        (4, 1, 0.2),
        (3, 3, 0.1),
        (3, 4, 0.0),
    ],
)
def test_counting_current_contents_apply_each_error_once(correct, wrong, expected):
    evaluator = object.__new__(CountIntoOpaqueBoxEvaluator)
    evaluator.model = SimpleNamespace(nq=8 * 7)
    evaluator._nworld = 1
    evaluator._ever_correct = np.zeros(1, dtype=bool)
    evaluator._box_x_bounds = evaluator._box_y_bounds = (-1.0, 1.0)
    evaluator._box_floor_z, evaluator._box_ceiling_z = 0.0, 1.0
    evaluator.floor_tolerance_m = 0.0
    evaluator.partial_overfill_penalty = True
    evaluator._directive = {
        "names": [str(index) for index in range(8)],
        "addrs": np.arange(8) * 7,
        "eligible_mask": np.array([True] * 4 + [False] * 4),
        "target_count": 3,
        "prompt_type": "exact_color_count",
    }
    poses = np.zeros((1, 56))
    poses[0, np.arange(8) * 7] = 2.0
    for index in [*range(correct), *range(4, 4 + wrong)]:
        poses[0, index * 7 : index * 7 + 3] = [0.0, 0.0, 0.1]
    result = evaluator.evaluate_qpos_batch(poses)
    assert result.scalar_reward() == pytest.approx(expected)
    assert result.scalar_success() == (correct == 3 and wrong == 0)


class InterruptedFixture(Fixture):
    def __init__(self):
        root = ET.fromstring(fixture_xml(2))
        table = root.find(".//geom[@name='conveyor_belt_surface']")
        table.set("name", "table_plane")
        table.set("size", "1 1 .015")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.evaluator = InterruptedClutterEvaluator(
            model=self.model, spec=get_task_spec("rl_grab_specific_object_from_clutter")
        )
        metadata = {
            "prompt": "Pick up the first object.",
            "objects": [{"name": f"conveyor_object_{i}"} for i in (1, 2)],
            "interruption": {
                "kind": "clutter",
                "delay_s": 1.0,
                "first_body": "conveyor_object_1",
                "second_body": "conveyor_object_2",
                "prompt": "Return the first object and pick up the second.",
            },
        }
        state = SimpleNamespace(metadata=metadata, object_states={})
        episode = self.evaluator.prepare_randomization(self.model, state)
        self.evaluator.configure_episode(self.model, episode)
        mujoco.mj_forward(self.model, self.data)
        self.evaluator.after_reset(self.model, self.data)

    def grasp(self, index=1, side="left"):
        pos = [0, 0.42 - 0.12 * (index - 1), 0.10]
        self.move(index, pos)
        self.arm(side, pos)
        result = self.sample(0.06)
        assert np.isfinite(self.evaluator._grasp_since[index - 1])
        return result


@pytest.mark.parametrize("distance, expected", [(0.049, 0.7), (0.051, 0.5), (0.099, 0.5), (0.101, 0.3)])
def test_interrupted_clutter_current_return_bands_and_one_shot_clock(distance, expected):
    fixture = InterruptedFixture()
    assert fixture.sample().scalar_reward() == 0.0
    fixture.grasp(1)
    first_time = fixture.evaluator._first_lift_at
    assert fixture.sample(0.9).scalar_reward() == pytest.approx(0.1)
    fixture.sample(0.2)
    fixture.move(1, [distance, 0.42, 0.0345])
    fixture.arm("left", [2, 0, 2])
    result = fixture.sample(0.1)
    assert result.metrics["return_credit"] == expected
    assert fixture.evaluator._first_lift_at == first_time
    fixture.grasp(2)
    assert fixture.sample().scalar_reward() == pytest.approx(0.1 + expected + 0.2)


def test_native_wood_preserves_original_atlas_and_seeded_records():
    baseline = next(
        sample_native_wood_appearance(seed) for seed in range(100) if sample_native_wood_appearance(seed)["original"]
    )
    np.testing.assert_array_equal(native_wood_pixels(baseline), original_atlas())
    varied = next(
        sample_native_wood_appearance(seed)
        for seed in range(100)
        if not sample_native_wood_appearance(seed)["original"]
    )
    validate_appearance(json.loads(json.dumps(varied)))
    assert varied == sample_native_wood_appearance(varied["seed"])
    assert not np.array_equal(native_wood_pixels(varied), original_atlas())
    assert get_task_spec("spell_random_3_appearance").reset_request_options()["appearance_profile"] == "native_wood"


def test_mug_rack_finishes_are_distinct_replayable_and_physics_preserving():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><asset>
        <texture name="mug_tree_tex" type="2d" builtin="checker" width="8" height="8" rgb1=".8 .6 .4" rgb2=".6 .4 .2"/>
        <material name="mug_tree_mat" texture="mug_tree_tex"/>
      </asset><worldbody><geom size=".1" material="mug_tree_mat"/></worldbody></mujoco>
    """)
    randomizer = MugTreeRandomizer()
    physics = (
        model.geom_size.copy(),
        model.geom_friction.copy(),
        model.body_mass.copy(),
    )
    original = model.tex_data.copy()
    snapshots = []
    for name, _ in randomizer._rack_finishes:
        randomizer._apply_rack_finish(model, name)
        snapshots.append(model.tex_data.tobytes())
    assert len(set(snapshots)) == len(randomizer._rack_finishes) == 10
    randomizer._apply_rack_finish(model, "original")
    np.testing.assert_array_equal(model.tex_data, original)
    for expected, actual in zip(physics, (model.geom_size, model.geom_friction, model.body_mass)):
        np.testing.assert_array_equal(actual, expected)


def test_marker_drawer_interior_uses_inner_wall_faces():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body name="drawer">
        <geom type="box" size=".12 .15 .005" pos="0 0 0"/>
        <geom type="box" size=".005 .15 .04" pos="-.115 0 .04"/>
        <geom type="box" size=".005 .15 .04" pos=".115 0 .04"/>
        <geom type="box" size=".11 .005 .04" pos="0 -.145 .04"/>
        <geom type="box" size=".11 .005 .04" pos="0 .145 .04"/>
      </body></worldbody></mujoco>
    """)
    lower, upper = drawer_interior(model, model.body("drawer").id)
    np.testing.assert_allclose(lower, [-0.11, -0.14, 0.005])
    np.testing.assert_allclose(upper, [0.11, 0.14, 0.08])


def test_four_plate_variant_is_a_fixed_native_condition():
    spec = get_task_spec("rl_load_plates_into_dish_rack")
    assert spec.resolve_reset_request(None)["plate_count"] == 4
    with pytest.raises(ValueError, match="requires"):
        spec.resolve_reset_request({"plate_count": 3})


@pytest.mark.parametrize(
    "task, flag",
    [
        ("rl_count_into_opaque_box", "family_first"),
        ("rl_grab_specific_object_from_clutter", "interrupted"),
        ("rl_put_relative", "interrupted"),
        ("rl_put_markers_in_drawer", "random_target_drawer"),
    ],
)
def test_rl_identity_cannot_be_disabled_by_reset_override(task, flag):
    spec = get_task_spec(task)
    assert spec.resolve_reset_request({})[flag] is True
    with pytest.raises(ValueError, match="requires"):
        spec.resolve_reset_request({flag: False})


def test_interrupted_clutter_regrasp_removes_return_and_early_second_pick_needs_a_fresh_lift():
    fixture = InterruptedFixture()
    fixture.grasp(2, "right")
    fixture.grasp(1, "left")
    fixture.sample(1.1)
    assert fixture.evaluator._switch_at is not None
    assert fixture.sample().metrics["second_pick_credit"] == 0.0
    fixture.move(1, fixture.evaluator._origin[0])
    fixture.arm("left", [2, 0, 2])
    assert fixture.sample(0.1).metrics["return_credit"] == 0.70
    fixture.grasp(1)
    assert fixture.sample().metrics["return_credit"] == 0.0
    fixture.move(2, fixture.evaluator._origin[1])
    fixture.arm("right", [2, 0, 2])
    fixture.sample(0.1)
    fixture.grasp(2, "right")
    assert fixture.sample().metrics["second_pick_credit"] == 0.20


def test_interruption_updates_the_runtime_prompt_once_without_replacing_native_tasks():
    fixture = InterruptedFixture()
    runtime = make_task_runtime("rl_grab_specific_object_from_clutter", fixture.model, fixture.data)
    assert isinstance(runtime, InterruptedInstructionRuntime)
    assert make_task_runtime("grab_specific_object_from_clutter", fixture.model, fixture.data) is None
    assert make_task_runtime("put_relative", fixture.model, fixture.data) is None
    episode = fixture.evaluator._episode
    state = SimpleNamespace(metadata={**episode, "interruption_initial_settled": True}, object_states={})
    runtime.after_reset(fixture.model, fixture.data, randomization=state, episode=episode)
    assert runtime.observation_prompt() == episode["prompt"]
    fixture.grasp()
    runtime.on_task_evaluation(fixture.sample(0.9))
    assert runtime.observation_prompt() == episode["prompt"]
    runtime.on_task_evaluation(fixture.sample(0.2))
    assert runtime.observation_prompt() == episode["interruption"]["prompt"]
    first_switch = fixture.evaluator._switch_at
    fixture.move(1, fixture.evaluator._origin[0])
    fixture.arm("left", [2, 0, 2])
    fixture.sample(0.1)
    fixture.grasp()
    runtime.on_task_evaluation(fixture.sample(2))
    assert fixture.evaluator._switch_at == first_switch
    assert runtime.retained_randomization().metadata == state.metadata


class RelativeFixture(InterruptedFixture):
    def __init__(self):
        root = ET.fromstring(fixture_xml(3))
        table = root.find(".//geom[@name='conveyor_belt_surface']")
        table.set("name", "table_plane")
        table.set("size", "1 1 .015")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.move(2, [-0.4, 0.42, 0.0345])
        self.move(3, [0.4, 0.42, 0.0345])
        mujoco.mj_forward(self.model, self.data)
        self.evaluator = InterruptedRelativeEvaluator(model=self.model, spec=get_task_spec("rl_put_relative"))
        states = {
            f"conveyor_object_{i}_joint": {
                "pos": self.data.body(f"conveyor_object_{i}").xpos.tolist(),
                "quat": [1, 0, 0, 0],
            }
            for i in range(1, 4)
        }
        initial = {
            "prompt": "Put the first object to the left of the second.",
            "objects": [{"name": f"conveyor_object_{i}"} for i in (1, 2, 3)],
            "mover_object": "conveyor_object_1",
            "reference_object": "conveyor_object_2",
            "direction_offset_xy": [-0.1, 0],
            "direction": "left",
        }
        revised = {
            **initial,
            "prompt": "Actually, put the first object to the right of the third instead.",
            "reference_object": "conveyor_object_3",
            "direction_offset_xy": [0.1, 0],
            "direction": "right",
        }
        initial["interruption"] = {
            "kind": "relative",
            "delay_s": 1.0,
            "first_body": "conveyor_object_1",
            "prompt": revised["prompt"],
            "directive": revised,
        }
        self.evaluator.configure_episode(
            self.model,
            self.evaluator.prepare_randomization(self.model, SimpleNamespace(metadata=initial, object_states=states)),
        )
        self.evaluator.after_reset(self.model, self.data)


def test_relative_variant_scores_only_the_revised_released_relation():
    fixture = RelativeFixture()
    fixture.move(1, [-0.50, 0.42, 0.0345])
    assert fixture.sample().scalar_reward() == 0.0
    assert not fixture.sample().scalar_success()
    fixture.grasp()
    fixture.sample(1.1)
    fixture.move(1, [0.30, 0.42, 0.0345])
    fixture.arm("left", [2, 0, 2])
    wrong_side = fixture.sample()
    assert wrong_side.metrics["approach_credit"] == pytest.approx(0.30)
    assert wrong_side.metrics["relative_placement_credit"] == 0.0
    fixture.move(1, [0.50, 0.42, 0.10])
    fixture.arm("left", [0.50, 0.42, 0.10])
    held = fixture.sample()
    assert held.scalar_reward() == pytest.approx(0.30)
    assert not held.scalar_success()
    fixture.move(1, [0.50, 0.42, 0.0345])
    fixture.arm("left", [2, 0, 2])
    placed = fixture.sample()
    assert placed.scalar_reward() == pytest.approx(1.0)
    assert placed.scalar_success()
    fixture.move(3, [0.45, 0.42, 0.0345])
    moved_reference = fixture.sample()
    assert moved_reference.metrics["relative_placement_credit"] == 0.0
    assert not moved_reference.scalar_success()


class MarkerFixture:
    def __init__(self, target):
        root = ET.fromstring("<mujoco><worldbody/></mujoco>")
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
            )
            for pos, size in (
                ("0 0 0", ".12 .15 .005"),
                ("-.115 0 .04", ".005 .15 .04"),
                (".115 0 .04", ".005 .15 .04"),
                ("0 -.145 .04", ".11 .005 .04"),
                ("0 .145 .04", ".11 .005 .04"),
                ("0 -.165 .04", ".02 .01 .005"),
                ("0 -.18 .04", ".025 .006 .01"),
            ):
                ET.SubElement(drawer, "geom", type="box", pos=pos, size=size)
        for i in range(1, 6):
            body = ET.SubElement(world, "body", name=f"marker_{i}", pos=f"{1 + i * 0.05} 0 .1")
            ET.SubElement(body, "freejoint", name=f"marker_{i}_joint")
            ET.SubElement(body, "geom", type="sphere", size=".01", mass=".01")
        for arm in ("left", "right"):
            body = ET.SubElement(world, "body", name=f"{arm}_arm", pos="3 3 3")
            ET.SubElement(body, "freejoint", name=f"{arm}_arm_pose")
            ET.SubElement(body, "geom", type="sphere", size=".01", mass=".01")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.target = target
        self.evaluator = MarkerDrawerEvaluator(model=self.model, spec=get_task_spec("rl_put_markers_in_drawer"))
        self.evaluator.configure_from_randomization(
            self.model, SimpleNamespace(metadata={"marker_target_drawer": target})
        )
        mujoco.mj_forward(self.model, self.data)

    def score(self, opening, count):
        self.data.joint(f"joint_{self.target}_drawer_low").qpos[0] = opening
        mujoco.mj_forward(self.model, self.data)
        origin = self.data.body(f"link_{self.target}_drawer_low").xpos.copy()
        for i in range(1, 6):
            self.data.joint(f"marker_{i}_joint").qpos[:3] = (
                origin + [(i - 3) * 0.03, 0, 0.0145] if i <= count else [1 + i * 0.05, 0, 0.1]
            )
        mujoco.mj_forward(self.model, self.data)
        self.evaluator.after_physics_step(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)


@pytest.mark.parametrize("target", ["top", "middle", "bottom"])
def test_marker_current_contents_opening_retention_closure_and_reset(target):
    fixture = MarkerFixture(target)
    assert fixture.score(0, 0).scalar_reward() == 0.0
    assert fixture.score(0.2, 0).scalar_reward() == pytest.approx(0.2)
    assert fixture.score(0.2, 3).scalar_reward() == pytest.approx(0.62)
    assert fixture.score(0.2, 5).scalar_reward() == pytest.approx(0.90)
    closed = fixture.score(0, 5)
    assert closed.scalar_reward() == pytest.approx(1.0)
    assert closed.scalar_success()
    assert fixture.score(0.2, 5).scalar_reward() == pytest.approx(0.90)
    removed = fixture.score(0, 4)
    assert removed.scalar_reward() == pytest.approx(0.76)
    assert not removed.scalar_success()
    fixture.evaluator.reset()
    assert fixture.score(0, 0).scalar_reward() == 0.0


def test_tray_pickup_hold_lowering_and_settle_fit_thirty_seconds():
    fixture = tray_tests.BallTrayWindowTest()
    fixture.setUp()
    fixture.sample(12, tray_tests.REST_Z)
    fixture.hold(13, 23)
    fixture.place(23.6)
    completed = fixture.sample(24.6, tray_tests.REST_Z, held=False)
    assert completed.scalar_success()
    assert completed.scalar_reward() == pytest.approx(1.0)
    assert fixture.data.time < 30
