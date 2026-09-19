"""Water B (stage index 2): the pink ball, the submarine, the rocket and
the scene's exit.

Read off longplay recordings at 60 fps: the ball enters at the left,
crosses at 1.75 px/frame, pops for 500, and the pop that reaches the exit
count pays the 5000 bonus and ends the scene; the submarine enters at the
left, cruises right and fires its double-dot shot; both player rounds
destroy the rocket for 200 and the depth charge sinks the submarine.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxatari.games.jax_jamesbond import JamesBondConstants, JaxJamesBond


NOOP, DOWN = 0, 5


@pytest.fixture(scope="module")
def env():
    return JaxJamesBond(JamesBondConstants(START_STAGE=2))


def _quiet(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    return state.replace(
        stage=jnp.array(2, dtype=jnp.int32),
        step_count=jnp.array(100, dtype=jnp.int32),
        diamond_active=jnp.array(False),
        rocket_active=jnp.array(False),
        rocket_timer=jnp.array(900, dtype=jnp.int32),
        submarine_active=jnp.array(False),
        submarine_timer=jnp.array(900, dtype=jnp.int32),
        wb_heli_active=jnp.array(False),
        wb_heli_timer=jnp.array(900, dtype=jnp.int32),
        wb_flyer_active=jnp.array(False),
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
    assert int(state.stage) == 2


def test_final_ball_pays_the_bonus_and_ends_the_scene(env):
    consts = env.consts
    state = _quiet(env).replace(
        wb_ball_hits=jnp.array(consts.WB_BALL_HITS_TO_EXIT - 1, dtype=jnp.int32),
    )
    state = _ball_over_boat(env, state)
    lives = int(state.lives)
    done = False
    for _ in range(8):
        _, state, _, done, _ = env.step(state, NOOP)
        if bool(done):
            break
    assert bool(done)
    assert int(state.stage) == 3
    assert int(state.score) == consts.SCORE_BALL + consts.SCORE_STAGE_BONUS
    assert int(state.lives) == lives
    assert not bool(state.wb_heli_active)


def test_submarine_enters_left_and_cruises_right(env):
    state = _quiet(env).replace(submarine_timer=jnp.array(1, dtype=jnp.int32))
    state = _run(env, state, 1)
    assert bool(state.submarine_active)
    assert int(state.submarine_x) == env.consts.SUB_SPAWN_X
    start = int(state.submarine_x)
    state = _run(env, state, 30)
    ## 2px every 3 frames, 30 frames -> 20px
    assert int(state.submarine_x) - start == 20


def test_submarine_shot_runs_back_up_then_along_the_surface(env):
    consts = env.consts
    ## Park the boat out of the way so nothing dies during the shot
    state = _quiet(env).replace(
        submarine_active=jnp.array(True),
        submarine_x=jnp.array(consts.SUB_FIRE_X - 1, dtype=jnp.int32),
        player_x=jnp.array(10, dtype=jnp.int32),
    )
    for _ in range(4):
        _, state, _, _, _ = env.step(state, NOOP)
        if bool(state.sub_torp_active):
            break
    assert bool(state.sub_torp_active)
    x0, y0 = int(state.sub_torp_x), int(state.sub_torp_y)
    state = _run(env, state, 3)
    ## diagonal leg: (-2,-1) per frame
    assert int(state.sub_torp_x) == x0 - 6
    assert int(state.sub_torp_y) == y0 - 3
    ## then it settles just under the surface and runs straight left
    state = _run(env, state, 20)
    assert int(state.sub_torp_y) == consts.SUB_SHOT_LEVEL_Y
    x1 = int(state.sub_torp_x)
    state = _run(env, state, 12)
    assert int(state.sub_torp_y) == consts.SUB_SHOT_LEVEL_Y
    ## 4px every 3 frames
    assert x1 - int(state.sub_torp_x) == 16


def test_submarine_shot_costs_a_diving_boat_a_life(env):
    """The shot runs just under the surface: a surfaced hull is above its
    lane, a boat on its way down is not."""

    consts = env.consts
    state = _quiet(env)
    state = state.replace(
        player_y=jnp.array(consts.SUB_SHOT_LEVEL_Y + 2, dtype=jnp.int32),
        player_diving=jnp.array(True),
        player_in_water_step=jnp.array(20, dtype=jnp.int32),
        sub_torp_active=jnp.array(True),
        sub_torp_x=jnp.array(int(state.player_x) + 3, dtype=jnp.int32),
        sub_torp_y=jnp.array(consts.SUB_SHOT_LEVEL_Y, dtype=jnp.int32),
    )
    lives = int(state.lives)
    state = _run(env, state, 1)
    assert int(state.lives) == lives - 1

    surfaced = _quiet(env)
    surfaced = surfaced.replace(
        sub_torp_active=jnp.array(True),
        sub_torp_x=jnp.array(int(surfaced.player_x) + 3, dtype=jnp.int32),
        sub_torp_y=jnp.array(consts.SUB_SHOT_LEVEL_Y, dtype=jnp.int32),
    )
    surfaced = _run(env, surfaced, 1)
    assert int(surfaced.lives) == lives


def test_anti_air_shot_destroys_water_b_rocket_for_200(env):
    """Third recording, 60 fps: the shot touches the climbing rocket at
    row 97 and the counter goes 6500 -> 6700 as it vanishes."""

    consts = env.consts
    state = _quiet(env).replace(
        rocket_active=jnp.array(True),
        rocket_x=jnp.array(60, dtype=jnp.int32),
        rocket_y=jnp.array(90, dtype=jnp.int32),
        rocket_age=jnp.array(200, dtype=jnp.int32),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(5, dtype=jnp.int32),
        player_bullet_x=jnp.array(56, dtype=jnp.int32),
        player_bullet_y=jnp.array(98, dtype=jnp.int32),
    )
    state = _run(env, state, 4)
    assert not bool(state.rocket_active)
    assert int(state.score) == consts.SCORE_ROCKET
    assert not bool(state.player_bullet_active)
    assert not bool(state.wb_flyer_active)


def test_depth_charge_sinks_water_b_submarine(env):
    consts = env.consts
    state = _quiet(env).replace(
        submarine_active=jnp.array(True),
        submarine_x=jnp.array(40, dtype=jnp.int32),
        player_wbullet_active=jnp.array(True),
        player_wbullet_step=jnp.array(20, dtype=jnp.int32),
        player_wbullet_x=jnp.array(44, dtype=jnp.int32),
        player_wbullet_y=jnp.array(consts.SUBMARINE_Y - 2, dtype=jnp.int32),
    )
    state = _run(env, state, 3)
    assert not bool(state.submarine_active)
    assert int(state.score) == consts.SCORE_SUBMARINE_SHOT


def test_unshot_rocket_bursts_and_its_debris_falls_to_the_waterline(env):
    """Team decision on top of the ALE-measured burst: the bars fall to
    the waterline, float there as the sparkle, then vanish."""

    consts = env.consts
    state = _quiet(env).replace(
        rocket_active=jnp.array(True),
        rocket_x=jnp.array(100, dtype=jnp.int32),
        rocket_y=jnp.array(consts.ROCKET_EXPLODE_Y + 1, dtype=jnp.int32),
        rocket_age=jnp.array(50, dtype=jnp.int32),
    )
    state = _run(env, state, 1)
    assert not bool(state.rocket_active)
    assert bool(state.wb_flyer_active)
    assert int(state.wb_flyer_y) == consts.WB_FLYER_Y
    state = _run(env, state, 20)
    assert int(state.wb_flyer_y) == consts.WB_FLYER_Y + 20
    state = _run(env, state, consts.DEBRIS_REST_Y - consts.WB_FLYER_Y - 20)
    assert int(state.wb_flyer_y) == consts.DEBRIS_REST_Y
    assert bool(state.wb_flyer_active)
    state = _run(env, state, consts.DEBRIS_REST_FRAMES + 2)
    assert not bool(state.wb_flyer_active)


def test_falling_debris_costs_a_life(env):
    state = _quiet(env)
    state = state.replace(
        wb_flyer_active=jnp.array(True),
        wb_flyer_x=jnp.array(int(state.player_x) + 2, dtype=jnp.int32),
        wb_flyer_y=jnp.array(int(state.player_y) - 3, dtype=jnp.int32),
        wb_flyer_timer=jnp.array(0, dtype=jnp.int32),
    )
    lives = int(state.lives)
    state = _run(env, state, 2)
    assert int(state.lives) == lives - 1


def test_anti_air_shot_pops_falling_debris_for_100(env):
    consts = env.consts
    state = _quiet(env).replace(
        wb_flyer_active=jnp.array(True),
        wb_flyer_x=jnp.array(60, dtype=jnp.int32),
        wb_flyer_y=jnp.array(90, dtype=jnp.int32),
        wb_flyer_timer=jnp.array(0, dtype=jnp.int32),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(5, dtype=jnp.int32),
        player_bullet_x=jnp.array(58, dtype=jnp.int32),
        player_bullet_y=jnp.array(94, dtype=jnp.int32),
    )
    state = _run(env, state, 2)
    assert not bool(state.wb_flyer_active)
    assert int(state.score) == consts.SCORE_DEBRIS_SHOT
