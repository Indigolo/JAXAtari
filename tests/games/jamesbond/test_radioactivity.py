"""Focused tests for radioactive objects in James Bond's first water scene."""

import jax
import jax.numpy as jnp
import pytest

from jaxatari.games.jax_jamesbond import JamesBondConstants, JaxJamesBond


NOOP = 0


@pytest.fixture(scope="module")
def env():
    return JaxJamesBond(JamesBondConstants(START_STAGE=1))


def _water_state(env):
    """Return a quiet water-scene state with automatic spawns held back."""

    _, state = env.reset(jax.random.PRNGKey(0))
    return state.replace(
        stage=jnp.array(1, dtype=jnp.int32),
        step_count=jnp.array(100, dtype=jnp.int32),
        diamond_active=jnp.array(False),
        pit_active=jnp.array(False),
        helicopter_active=jnp.array(False),
        helicopter_bomb_active=jnp.array(False),
        satellite_active=jnp.array(False),
        satellite_respawn_timer=jnp.array(100, dtype=jnp.int32),
        satellite_laser_active=jnp.array(False),
        scuba_active=jnp.array(False),
        scuba_respawn_timer=jnp.array(100, dtype=jnp.int32),
        scuba_radioactive=jnp.array(False),
        scuba_radioactive_age=jnp.array(0, dtype=jnp.int32),
        splash_active=jnp.array(False),
        splash_age=jnp.array(0, dtype=jnp.int32),
    )


def _step(env, state):
    _, state, _, _, _ = env.step(state, jnp.array(NOOP, dtype=jnp.int32))
    return state


def _scuba_x_at_gap(env, state, gap):
    return (
        state.player_x + env.consts.PLAYER_WIDTH + gap
    ).astype(jnp.int32)


def _scuba_gap(env, state):
    player_right = state.player_x + env.consts.PLAYER_WIDTH
    scuba_right = state.scuba_x + env.consts.SCUBA_WIDTH
    return int(
        jnp.maximum(
            jnp.maximum(
                state.scuba_x - player_right,
                state.player_x - scuba_right,
            ),
            0,
        )
    )


def test_nearby_scuba_becomes_radioactive_without_satellite_bolt(env):
    state = _water_state(env)
    state = state.replace(
        scuba_active=jnp.array(True),
        scuba_x=_scuba_x_at_gap(
            env, state, env.consts.SCUBA_RADIOACTIVE_RANGE
        ),
        scuba_y=jnp.array(env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
        scuba_age=jnp.array(10, dtype=jnp.int32),
    )

    state = _step(env, state)

    assert bool(state.scuba_radioactive)
    assert int(state.scuba_radioactive_age) == 0
    assert not bool(state.satellite_laser_active)
    assert not bool(state.splash_active)


def test_scuba_uses_its_post_movement_position_for_proximity(env):
    state = _water_state(env)
    state = state.replace(
        # The step advances 3 -> 4, moving the diver from a 41px to 40px gap.
        step_count=jnp.array(3, dtype=jnp.int32),
        scuba_active=jnp.array(True),
        scuba_x=_scuba_x_at_gap(
            env, state, env.consts.SCUBA_RADIOACTIVE_RANGE + 1
        ),
        scuba_y=jnp.array(env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
        scuba_age=jnp.array(10, dtype=jnp.int32),
    )

    state = _step(env, state)

    assert _scuba_gap(env, state) == env.consts.SCUBA_RADIOACTIVE_RANGE
    assert bool(state.scuba_radioactive)


def test_far_scuba_keeps_bolt_normal_and_no_object_becomes_radioactive(env):
    state = _water_state(env)
    state = state.replace(
        scuba_active=jnp.array(True),
        scuba_x=_scuba_x_at_gap(
            env, state, env.consts.SCUBA_RADIOACTIVE_RANGE + 1
        ),
        scuba_y=jnp.array(env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
        scuba_age=jnp.array(10, dtype=jnp.int32),
        satellite_laser_active=jnp.array(True),
        satellite_laser_x=jnp.array(120, dtype=jnp.int32),
        satellite_laser_y=jnp.array(
            env.consts.WATER_LASER_FLOOR - 1, dtype=jnp.int32
        ),
    )

    state = _step(env, state)

    assert not bool(state.satellite_laser_active)
    assert not bool(state.splash_active)
    assert not bool(state.scuba_radioactive)


def test_nearby_scuba_suppresses_radioactive_bolt_splash(env):
    state = _water_state(env)
    state = state.replace(
        scuba_active=jnp.array(True),
        scuba_x=(state.player_x + 1).astype(jnp.int32),
        scuba_y=jnp.array(env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
        scuba_age=jnp.array(10, dtype=jnp.int32),
        satellite_laser_active=jnp.array(True),
        satellite_laser_x=jnp.array(120, dtype=jnp.int32),
        satellite_laser_y=jnp.array(
            env.consts.WATER_LASER_FLOOR - 1, dtype=jnp.int32
        ),
    )

    state = _step(env, state)

    assert bool(state.scuba_radioactive)
    assert not bool(state.satellite_laser_active)
    assert not bool(state.splash_active)


def test_satellite_bolt_does_not_refresh_radioactive_scuba(env):
    state = _water_state(env)
    state = state.replace(
        scuba_active=jnp.array(True),
        scuba_x=_scuba_x_at_gap(
            env, state, env.consts.SCUBA_RADIOACTIVE_RANGE + 1
        ),
        scuba_y=jnp.array(env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
        scuba_age=jnp.array(10, dtype=jnp.int32),
        scuba_radioactive=jnp.array(True),
        scuba_radioactive_age=jnp.array(10, dtype=jnp.int32),
        satellite_laser_active=jnp.array(True),
        satellite_laser_x=jnp.array(120, dtype=jnp.int32),
        satellite_laser_y=jnp.array(
            env.consts.WATER_LASER_FLOOR - 1, dtype=jnp.int32
        ),
    )

    state = _step(env, state)

    assert not bool(state.satellite_laser_active)
    assert not bool(state.splash_active)
    assert bool(state.scuba_radioactive)
    assert int(state.scuba_radioactive_age) == 11


def test_bolt_still_creates_radioactive_splash_without_scuba(env):
    state = _water_state(env)
    state = state.replace(
        satellite_laser_active=jnp.array(True),
        satellite_laser_x=jnp.array(120, dtype=jnp.int32),
        satellite_laser_y=jnp.array(
            env.consts.WATER_LASER_FLOOR - 1, dtype=jnp.int32
        ),
    )

    state = _step(env, state)

    assert not bool(state.satellite_laser_active)
    assert bool(state.splash_active)
    assert not bool(state.scuba_radioactive)


def test_scuba_presence_removes_an_older_radioactive_splash(env):
    state = _water_state(env)
    state = state.replace(
        scuba_active=jnp.array(True),
        scuba_x=_scuba_x_at_gap(
            env, state, env.consts.SCUBA_RADIOACTIVE_RANGE + 1
        ),
        scuba_y=jnp.array(env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
        scuba_age=jnp.array(10, dtype=jnp.int32),
        splash_active=jnp.array(True),
        splash_x=jnp.array(120, dtype=jnp.int32),
        splash_age=jnp.array(10, dtype=jnp.int32),
    )

    state = _step(env, state)

    assert not bool(state.splash_active)
    assert int(state.splash_age) == 0
    assert not bool(state.scuba_radioactive)


def test_same_frame_scuba_spawn_removes_an_older_radioactive_splash(env):
    state = _water_state(env)
    state = state.replace(
        scuba_respawn_timer=jnp.array(1, dtype=jnp.int32),
        splash_active=jnp.array(True),
        splash_x=jnp.array(120, dtype=jnp.int32),
        splash_age=jnp.array(10, dtype=jnp.int32),
    )

    state = _step(env, state)

    assert bool(state.scuba_active)
    assert not bool(state.splash_active)
    assert int(state.splash_age) == 0


def test_depth_charge_removes_scuba_and_scores(env):
    state = _water_state(env)
    state = state.replace(
        scuba_active=jnp.array(True),
        scuba_x=jnp.array(50, dtype=jnp.int32),
        scuba_y=jnp.array(env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
        player_wbullet_active=jnp.array(True),
        player_wbullet_step=jnp.array(10, dtype=jnp.int32),
        player_wbullet_x=jnp.array(50, dtype=jnp.int32),
        player_wbullet_y=jnp.array(
            env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32
        ),
    )

    result = env._resolve_player_wbullet_collisions(state)

    assert not bool(result.scuba_active)
    assert not bool(result.player_wbullet_active)
    assert int(result.score) == int(state.score) + env.consts.SCORE_SCUBA


def test_depth_charge_hits_radioactive_scuba_at_the_surface(env):
    """The radioactive figure sits at the waterline, so the hit box moves
    there with it; a shot at the old depth misses."""

    state = _water_state(env)
    state = state.replace(
        scuba_active=jnp.array(True),
        scuba_radioactive=jnp.array(True),
        scuba_radioactive_age=jnp.array(5, dtype=jnp.int32),
        scuba_x=jnp.array(50, dtype=jnp.int32),
        scuba_y=jnp.array(env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
        player_wbullet_active=jnp.array(True),
        player_wbullet_step=jnp.array(10, dtype=jnp.int32),
        player_wbullet_x=jnp.array(52, dtype=jnp.int32),
        player_wbullet_y=jnp.array(env.consts.SPLASH_Y, dtype=jnp.int32),
    )
    result = env._resolve_player_wbullet_collisions(state)
    assert not bool(result.scuba_active)
    assert not bool(result.scuba_radioactive)
    assert int(result.score) == int(state.score) + env.consts.SCORE_SCUBA

    deep = state.replace(
        player_wbullet_y=jnp.array(env.consts.SCUBA_SPAWN_Y + 10, dtype=jnp.int32),
    )
    result = env._resolve_player_wbullet_collisions(deep)
    assert bool(result.scuba_active)
    assert int(result.score) == int(deep.score)


def test_satellite_never_splashes_after_the_first_diver(env):
    """A diver that has come and gone still keeps the bolt from splashing."""

    state = _water_state(env).replace(
        scuba_seen=jnp.array(True),
        satellite_laser_active=jnp.array(True),
        satellite_laser_x=jnp.array(120, dtype=jnp.int32),
        satellite_laser_y=jnp.array(
            env.consts.WATER_LASER_FLOOR - 1, dtype=jnp.int32
        ),
    )

    state = _step(env, state)

    assert not bool(state.satellite_laser_active)
    assert not bool(state.splash_active)


def test_first_diver_spawn_sets_the_latch(env):
    state = _water_state(env).replace(
        scuba_respawn_timer=jnp.array(1, dtype=jnp.int32),
    )
    assert not bool(state.scuba_seen)
    state = _step(env, state)
    assert bool(state.scuba_active)
    assert bool(state.scuba_seen)


def test_radioactive_scuba_uses_satellite_drop_figure(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        scuba_active=jnp.array(True),
        scuba_radioactive=jnp.array(True),
        scuba_radioactive_age=jnp.array(0, dtype=jnp.int32),
        scuba_x=jnp.array(60, dtype=jnp.int32),
        scuba_y=jnp.array(env.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
    )
    raster = env.renderer.jr.create_object_raster(env.renderer.BACKGROUND)

    actual = env.renderer._render_scuba(raster, state)
    expected = env.renderer.jr.render_at_clipped(
        raster,
        state.scuba_x,
        env.consts.SPLASH_Y,
        env.renderer.SHAPE_MASKS['splash'],
    )

    assert bool(jnp.array_equal(actual, expected))
