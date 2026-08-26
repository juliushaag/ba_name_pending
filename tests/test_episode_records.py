"""What an evaluated episode records, and specifically what `success` means.

Two env families answer that question in opposite directions. ManiSkill reports
`info["success"]` and sets `terminated` *from* it, so terminating is winning;
MuJoCo and DMC report no such key and terminate by failing out (a humanoid falls
over). `episode_success` is the one place that reconciles them, and getting it
backwards silently reports a policy that never succeeds as a perfect one.
"""

import numpy as np
import pytest

from barl_jax.policy import episode_success


def vector_info(success, present=None):
    """Gymnasium batches each sub-env info key into an array plus an `_<key>`
    presence mask -- the shape `run_episodes` actually receives."""
    info = {"success": np.asarray(success)}
    if present is not None:
        info["_success"] = np.asarray(present)
    return info


@pytest.mark.parametrize("terminated", [True, False])
def test_the_envs_own_flag_wins_over_termination(terminated):
    """ManiSkill's answer is used as-is, whichever way `terminated` points --
    the whole reason this function exists."""
    assert episode_success(vector_info([True], [True]), 0, True, terminated) is True
    assert episode_success(vector_info([False], [True]), 0, True, terminated) is False


def test_survival_is_the_fallback_where_no_flag_is_reported():
    """MuJoCo/DMC: an episode that never terminated is one the agent did not
    fail out of. This is the pre-existing convention and must not shift."""
    dmc = {"timestep.discount": np.array([1.0])}
    assert episode_success(dmc, 0, True, terminated=False) is True
    assert episode_success(dmc, 0, True, terminated=True) is False


def test_a_slot_the_key_is_absent_for_falls_back():
    """The `_success` mask marks which sub-envs reported the key at all; an
    unreported slot must not read another slot's value."""
    info = vector_info([True], [False])
    assert episode_success(info, 0, True, terminated=False) is True
    assert episode_success(info, 0, True, terminated=True) is False


def test_per_slot_indexing():
    info = vector_info([True, False, True], [True, True, True])
    assert [episode_success(info, i, True, False) for i in range(3)] == [True, False, True]


def test_single_env_info_is_not_indexed():
    """A plain gym.Env carries a scalar, not a per-slot array."""
    assert episode_success({"success": np.asarray(True)}, 0, False, False) is True
    assert episode_success({"success": np.asarray(False)}, 0, False, True) is False
