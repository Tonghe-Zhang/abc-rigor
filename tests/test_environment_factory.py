"""A supplied environment constructor participates in native task preparation."""

import numpy as np
import pytest
from abc_sim import make_env
from abc_sim.env import MuJoCoYAMEnv


@pytest.mark.parametrize(
    "task",
    ["put_plastic_bottles_in_bin", "spell_random_4_appearance", "inhand_transfer_item_to_other_side"],
)
def test_environment_factory_keeps_task_binding_and_seeded_reset(task):
    created = []

    class CustomEnvironment(MuJoCoYAMEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    env = make_env(task, render_cameras=False, environment_factory=CustomEnvironment)
    try:
        assert created == [env]
        assert env._task_spec.name == task
        snapshots = []
        for seed in (23, 24, 23):
            env.forget_arm_state()
            env.reset(seed=seed, randomize=True)
            snapshots.append(env.data.qpos.copy())
        np.testing.assert_array_equal(snapshots[0], snapshots[2])
        assert env.prompt
    finally:
        env.close()
