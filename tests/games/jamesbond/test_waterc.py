"""Tests for the third water scene (daylight, stage 3) of James Bond.

Everything asserted here mirrors what the longplay recording showed;
see the "water C" block of JamesBondConstants for the sources.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxatari.games.jax_jamesbond import JamesBondConstants, JaxJamesBond


NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5


@pytest.fixture(scope="module")
def env():
    return JaxJamesBond(JamesBondConstants(START_STAGE=3))


def _quiet_state(env, stage=3):
    """A calm scene with every automatic spawner held back."""

    _, state = env.reset(jax.random.PRNGKey(0))
    slots = env.consts.WC_ROCKET_SLOTS
    return state.replace(
        stage=jnp.array(stage, dtype=jnp.int32),
        step_count=jnp.array(100, dtype=jnp.int32),
        stage_start_step=jnp.array(0, dtype=jnp.int32),
        helicopter_active=jnp.array(False),
        helicopter_bomb_active=jnp.array(False),
        rocket_active=jnp.zeros((slots,), dtype=jnp.bool_),
        rocket_timer=jnp.full((slots,), 500, dtype=jnp.int32),
        submarine_active=jnp.array(False),
        submarine_timer=jnp.array(500, dtype=jnp.int32),
        ship_active=jnp.array(False),
        ship_timer=jnp.array(500, dtype=jnp.int32),
        wb_flyer_active=jnp.zeros((slots,), dtype=jnp.bool_),
        base_active=jnp.array(False),
        goal_active=jnp.array(False),
    )


def _run(env, state, actions):
    for action in actions:
        _, state, _, done, _ = env.step(state, action)
    return state


def _slot(array, index, value):
    return array.at[index].set(jnp.asarray(value, dtype=array.dtype))


def test_submarine_enters_left_and_cruises_right(env):
    state = _quiet_state(env).replace(
        submarine_timer=jnp.array(1, dtype=jnp.int32),
    )
    state = _run(env, state, [NOOP] * 2)
    assert bool(state.submarine_active)
    assert int(state.submarine_x) == env.consts.SUB_SPAWN_X
    start = int(state.submarine_x)
    state = _run(env, state, [NOOP] * 30)
    ## 2px every 3 frames, 30 frames -> 20px
    assert int(state.submarine_x) - start == 20


def test_depth_charge_sinks_submarine_for_200(env):
    state = _quiet_state(env).replace(
        submarine_active=jnp.array(True),
        submarine_x=jnp.array(40, dtype=jnp.int32),
        player_wbullet_active=jnp.array(True),
        player_wbullet_step=jnp.array(20, dtype=jnp.int32),
        player_wbullet_x=jnp.array(44, dtype=jnp.int32),
        player_wbullet_y=jnp.array(env.consts.SUBMARINE_Y - 2, dtype=jnp.int32),
    )
    state = _run(env, state, [NOOP] * 3)
    assert not bool(state.submarine_active)
    assert int(state.score) == env.consts.SCORE_SUBMARINE_SHOT
    assert not bool(state.player_wbullet_active)


def test_anti_air_shot_destroys_climbing_rocket(env):
    consts = env.consts
    state = _quiet_state(env)
    state = state.replace(
        rocket_active=_slot(state.rocket_active, 0, True),
        rocket_x=_slot(state.rocket_x, 0, 60),
        rocket_y=_slot(state.rocket_y, 0, 90),
        rocket_age=_slot(state.rocket_age, 0, 200),
        rocket_launch_age=_slot(state.rocket_launch_age, 0, 10),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(5, dtype=jnp.int32),
        player_bullet_x=jnp.array(56, dtype=jnp.int32),
        player_bullet_y=jnp.array(96, dtype=jnp.int32),
    )
    state = _run(env, state, [NOOP] * 4)
    assert not bool(state.rocket_active[0])
    assert int(state.score) == consts.SCORE_ROCKET_SHOT
    assert not bool(state.player_bullet_active)
    ## Shot rockets leave no debris behind
    assert not bool(jnp.any(state.wb_flyer_active))


def test_depth_charge_destroys_submerged_rocket(env):
    consts = env.consts
    state = _quiet_state(env)
    state = state.replace(
        rocket_active=_slot(state.rocket_active, 1, True),
        rocket_x=_slot(state.rocket_x, 1, 50),
        rocket_y=_slot(state.rocket_y, 1, consts.ROCKET_Y),
        rocket_launch_age=_slot(state.rocket_launch_age, 1, 500),
        player_wbullet_active=jnp.array(True),
        player_wbullet_step=jnp.array(20, dtype=jnp.int32),
        player_wbullet_x=jnp.array(52, dtype=jnp.int32),
        player_wbullet_y=jnp.array(consts.ROCKET_Y, dtype=jnp.int32),
    )
    state = _run(env, state, [NOOP] * 2)
    assert not bool(state.rocket_active[1])
    assert int(state.score) == consts.SCORE_ROCKET_SHOT


def test_two_rocket_pads_can_share_the_water(env):
    state = _quiet_state(env).replace(
        rocket_timer=jnp.array([1, 1], dtype=jnp.int32),
    )
    state = _run(env, state, [NOOP] * 2)
    assert bool(jnp.all(state.rocket_active))
    assert bool(jnp.all(state.rocket_y == env.consts.ROCKET_Y))
    assert bool(jnp.all(state.rocket_x >= env.consts.WC_ROCKET_SPAWN_MIN_X))
    assert bool(jnp.all(state.rocket_x <= env.consts.WC_ROCKET_SPAWN_MAX_X))


def test_rockets_first_then_ships(env):
    """Phase one has rockets and no ship; phase two has ships and no new rockets."""

    consts = env.consts
    early = _quiet_state(env).replace(
        rocket_timer=jnp.array([1, 1], dtype=jnp.int32),
        ship_timer=jnp.array(1, dtype=jnp.int32),
    )
    early = _run(env, early, [NOOP] * 3)
    assert bool(jnp.all(early.rocket_active))
    assert not bool(early.ship_active)
    late = _quiet_state(env).replace(
        step_count=jnp.array(consts.WC_SHIP_PHASE_START + 1, dtype=jnp.int32),
        rocket_timer=jnp.array([1, 1], dtype=jnp.int32),
        ship_timer=jnp.array(1, dtype=jnp.int32),
    )
    late = _run(env, late, [NOOP] * 3)
    assert not bool(jnp.any(late.rocket_active))
    assert bool(late.ship_active)


def test_water_b_still_uses_a_single_pad(env):
    state = _quiet_state(env, stage=2).replace(
        rocket_timer=jnp.array([1, 1], dtype=jnp.int32),
    )
    state = _run(env, state, [NOOP] * 2)
    assert bool(state.rocket_active[0])
    assert not bool(state.rocket_active[1])
    assert int(state.rocket_x[0]) == env.consts.ROCKET_SPAWN_X


def test_unshot_rocket_bursts_into_falling_lethal_debris(env):
    consts = env.consts
    state = _quiet_state(env)
    state = state.replace(
        rocket_active=_slot(state.rocket_active, 0, True),
        rocket_x=_slot(state.rocket_x, 0, 100),
        rocket_y=_slot(state.rocket_y, 0, consts.ROCKET_EXPLODE_Y + 1),
        rocket_age=_slot(state.rocket_age, 0, 50),
        rocket_launch_age=_slot(state.rocket_launch_age, 0, 10),
    )
    state = _run(env, state, [NOOP])
    assert not bool(state.rocket_active[0])
    assert bool(state.wb_flyer_active[0])
    assert int(state.wb_flyer_y[0]) == consts.WB_FLYER_Y
    ## The bars fall 1px/frame back to the waterline...
    state = _run(env, state, [NOOP] * 20)
    assert int(state.wb_flyer_y[0]) == consts.WB_FLYER_Y + 20
    state = _run(env, state, [NOOP] * (consts.WC_DEBRIS_REST_Y - consts.WB_FLYER_Y - 20))
    assert int(state.wb_flyer_y[0]) == consts.WC_DEBRIS_REST_Y
    assert bool(state.wb_flyer_active[0])
    ## ...float there for the rest window, then vanish
    state = _run(env, state, [NOOP] * (consts.WC_DEBRIS_REST_FRAMES + 2))
    assert not bool(state.wb_flyer_active[0])


def test_falling_debris_costs_a_life(env):
    consts = env.consts
    state = _quiet_state(env)
    state = state.replace(
        wb_flyer_active=_slot(state.wb_flyer_active, 0, True),
        wb_flyer_x=_slot(state.wb_flyer_x, 0, int(state.player_x) + 2),
        wb_flyer_y=_slot(state.wb_flyer_y, 0, int(state.player_y) - 3),
        wb_flyer_timer=_slot(state.wb_flyer_timer, 0, 0),
    )
    lives = int(state.lives)
    state = _run(env, state, [NOOP] * 2)
    assert int(state.lives) == lives - 1
    assert int(state.death_timer) > 0


def test_ship_hull_is_lethal_on_the_surface(env):
    state = _quiet_state(env).replace(
        ship_active=jnp.array(True),
        ship_x=jnp.array(int(_quiet_state(env).player_x) + 2, dtype=jnp.int32),
    )
    lives = int(state.lives)
    state = _run(env, state, [NOOP])
    assert int(state.lives) == lives - 1


def test_ship_crosses_right_to_left(env):
    consts = env.consts
    state = _quiet_state(env).replace(
        step_count=jnp.array(consts.WC_SHIP_PHASE_START + 100, dtype=jnp.int32),
        ship_timer=jnp.array(1, dtype=jnp.int32),
    )
    state = _run(env, state, [NOOP] * 2)
    assert bool(state.ship_active)
    start = int(state.ship_x)
    state = _run(env, state, [NOOP] * 50)
    ## 3px every 5 frames
    assert start - int(state.ship_x) == 30


def test_base_then_objective_then_bonus_ends_the_game(env):
    consts = env.consts
    state = _quiet_state(env).replace(
        step_count=jnp.array(consts.WC_LENGTH + 5, dtype=jnp.int32),
    )
    state = _run(env, state, [NOOP])
    assert bool(state.base_active)
    assert int(state.base_x) == consts.OBJECT_SPAWN_X_FAR
    assert not bool(state.goal_active)
    ## The base rides the world scroll until it reaches the trigger column
    frames = (consts.OBJECT_SPAWN_X_FAR - consts.WC_GOAL_TRIGGER_X) * 4
    state = _run(env, state, [NOOP] * frames)
    assert bool(state.goal_active)
    assert int(state.goal_x) == consts.WC_GOAL_TRIGGER_X + consts.WC_GOAL_OFFSET_X
    ## Put the objective within reach and dive into it
    state = state.replace(goal_x=jnp.array(int(state.player_x) + 2, dtype=jnp.int32))
    score = int(state.score)
    lives = int(state.lives)
    done = False
    for _ in range(200):
        _, state, _, done, _ = env.step(state, DOWN)
        if bool(done):
            break
    assert bool(done)
    assert int(state.stage) == 4
    assert int(state.score) == score + consts.SCORE_STAGE_BONUS
    assert int(state.lives) == lives


def test_helicopter_keeps_bombing_across_chained_passes(env):
    """Back-to-back passes must each get a fresh bomb allowance."""

    _, state = env.reset(jax.random.PRNGKey(0))
    drops = 0
    prev = False
    for _ in range(2500):
        _, state, _, _, _ = env.step(state, NOOP)
        active = bool(state.helicopter_bomb_active)
        drops += int(active and not prev)
        prev = active
    assert drops >= 3


def test_submarine_shot_runs_back_up_then_along_the_surface(env):
    """Video: two dots leave the bow as the submarine crosses the fire
    column, run (-2,-1) until just under the surface, then straight left."""

    consts = env.consts
    state = _quiet_state(env).replace(
        submarine_active=jnp.array(True),
        submarine_x=jnp.array(consts.SUB_FIRE_X - 1, dtype=jnp.int32),
        player_x=jnp.array(10, dtype=jnp.int32),
    )
    for _ in range(3):
        _, state, _, _, _ = env.step(state, NOOP)
        if bool(state.sub_torp_active):
            break
    assert bool(state.sub_torp_active)
    x0, y0 = int(state.sub_torp_x), int(state.sub_torp_y)
    assert y0 == consts.SUBMARINE_Y - 3
    state = _run(env, state, [NOOP] * 3)
    assert int(state.sub_torp_x) == x0 - 6
    assert int(state.sub_torp_y) == y0 - 3
    ## reach the surface lane and run along it: 4px every 3 frames
    for _ in range(30):
        _, state, _, _, _ = env.step(state, NOOP)
        if int(state.sub_torp_y) <= consts.SUB_SHOT_LEVEL_Y:
            break
    assert int(state.sub_torp_y) == consts.SUB_SHOT_LEVEL_Y
    x1 = int(state.sub_torp_x)
    state = _run(env, state, [NOOP] * 12)
    assert int(state.sub_torp_y) == consts.SUB_SHOT_LEVEL_Y
    assert x1 - int(state.sub_torp_x) == 16
    ## one shot per pass
    assert bool(state.sub_fired)
    ## and it clears the screen on the left eventually
    state = _run(env, state, [NOOP] * 120)
    assert not bool(state.sub_torp_active)


def test_submarine_shot_kills_a_diving_boat(env):
    consts = env.consts
    state = _quiet_state(env)
    state = state.replace(
        player_y=jnp.array(consts.SUB_SHOT_LEVEL_Y + 1, dtype=jnp.int32),
        sub_torp_active=jnp.array(True),
        sub_torp_x=jnp.array(int(state.player_x) + 8, dtype=jnp.int32),
        sub_torp_y=jnp.array(consts.SUB_SHOT_LEVEL_Y, dtype=jnp.int32),
    )
    lives = int(state.lives)
    state = _run(env, state, [NOOP] * 2)
    assert int(state.lives) == lives - 1
    assert not bool(state.sub_torp_active)


def test_anti_air_shot_pops_falling_debris(env):
    consts = env.consts
    state = _quiet_state(env)
    state = state.replace(
        wb_flyer_active=_slot(state.wb_flyer_active, 0, True),
        wb_flyer_x=_slot(state.wb_flyer_x, 0, 60),
        wb_flyer_y=_slot(state.wb_flyer_y, 0, 90),
        wb_flyer_timer=_slot(state.wb_flyer_timer, 0, 0),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(5, dtype=jnp.int32),
        player_bullet_x=jnp.array(58, dtype=jnp.int32),
        player_bullet_y=jnp.array(96, dtype=jnp.int32),
    )
    state = _run(env, state, [NOOP] * 3)
    assert not bool(state.wb_flyer_active[0])
    assert int(state.score) == consts.SCORE_DEBRIS_SHOT
    assert not bool(state.player_bullet_active)
