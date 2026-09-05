"""Water B (stage index 2): the pink ball and the scene's exit.

Read off a longplay recording at 60 fps: the ball enters at the left,
crosses at 1.75 px/frame, pops for 500, and the hit that reaches the
exit count freezes the scene for the death-style colour cycle before
the 5000 bonus hands over to the daylight scene.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxatari.games.jax_jamesbond import JamesBondConstants, JaxJamesBond


NOOP = 0


@pytest.fixture(scope="module")
def env():
    return JaxJamesBond(JamesBondConstants(START_STAGE=2))


def _quiet(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    slots = env.consts.WC_ROCKET_SLOTS
    return state.replace(
        stage=jnp.array(2, dtype=jnp.int32),
        step_count=jnp.array(100, dtype=jnp.int32),
        stage_start_step=jnp.array(0, dtype=jnp.int32),
        diamond_active=jnp.array(False),
        rocket_active=jnp.zeros((slots,), dtype=jnp.bool_),
        rocket_timer=jnp.full((slots,), 900, dtype=jnp.int32),
        submarine_active=jnp.array(False),
        submarine_timer=jnp.array(900, dtype=jnp.int32),
        wb_heli_active=jnp.array(False),
        wb_heli_timer=jnp.array(900, dtype=jnp.int32),
        wb_flyer_active=jnp.zeros((slots,), dtype=jnp.bool_),
    )


def _run(env, state, n, action=NOOP):
    for _ in range(n):
        _, state, _, _, _ = env.step(state, action)
    return state


def _ball_over_boat(env, state):
    """Put the ball right above the boat and a shot on its way up."""

    consts = env.consts
    return state.replace(
        wb_heli_active=jnp.array(True),
        wb_heli_x=jnp.array(60, dtype=jnp.int32),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(5, dtype=jnp.int32),
        ## the ball moves 2px a frame too, so the shot starts just ahead of it
        player_bullet_x=jnp.array(64, dtype=jnp.int32),
        player_bullet_y=jnp.array(consts.WB_HELI_Y + 12, dtype=jnp.int32),
    )


def test_ball_enters_left_and_crosses_right(env):
    consts = env.consts
    state = _quiet(env).replace(wb_heli_timer=jnp.array(1, dtype=jnp.int32))
    state = _run(env, state, 1)
    assert bool(state.wb_heli_active)
    assert int(state.wb_heli_x) == consts.GAME_AREA_MIN_X - consts.WB_HELI_WIDTH
    start = int(state.wb_heli_x)
    state = _run(env, state, 40)
    ## 7px every 4 frames
    assert int(state.wb_heli_x) - start == 70
    ## and it leaves on the right if nobody shoots it
    state = _run(env, state, 60)
    assert not bool(state.wb_heli_active)


def test_shot_ball_scores_500_and_disappears(env):
    consts = env.consts
    state = _ball_over_boat(env, _quiet(env))
    state = _run(env, state, 6)
    assert int(state.score) == consts.SCORE_BALL
    assert int(state.wb_ball_hits) == 1
    assert not bool(state.wb_heli_active)
    assert not bool(state.player_bullet_active)
    assert int(state.win_timer) == 0


def test_final_ball_freezes_pays_bonus_and_starts_daylight(env):
    consts = env.consts
    state = _quiet(env).replace(
        wb_ball_hits=jnp.array(consts.WB_BALL_HITS_TO_EXIT - 1, dtype=jnp.int32),
    )
    state = _ball_over_boat(env, state)
    for _ in range(8):
        _, state, _, _, _ = env.step(state, NOOP)
        if int(state.win_timer) > 0:
            break
    assert int(state.win_timer) == consts.WIN_ANIMATION_FRAMES
    score = int(state.score)
    lives = int(state.lives)
    ## The ball hangs on screen and the world stands still during the freeze
    assert bool(state.wb_heli_active)
    frozen_step = int(state.step_count)
    state = _run(env, state, 10)
    assert int(state.step_count) == frozen_step
    assert int(state.stage) == 2
    ## Then the bonus pays, the roster is swept and the daylight scene begins
    state = _run(env, state, consts.WIN_ANIMATION_FRAMES)
    assert int(state.stage) == 3
    assert int(state.score) == score + consts.SCORE_STAGE_BONUS
    assert int(state.lives) == lives
    assert not bool(state.wb_heli_active)
    assert int(state.wb_ball_hits) == 0
    assert int(state.stage_start_step) == frozen_step
    assert int(state.win_timer) == 0
