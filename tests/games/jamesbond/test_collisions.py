"""Unit tests for the JamesBond collision systems.

Each test builds a deterministic state via state.replace and steps once or a
few times, so the checks stay independent of the spawn schedule. All objects
are single scalar entities since the single object migration.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxatari.games.jax_jamesbond import JaxJamesBond

NOOP = 0
FIRE = 1


@pytest.fixture(scope="module")
def env():
    return JaxJamesBond()


def _clean_state(env, state):
    """Deactivate every spawned object so tests control the scene."""

    return state.replace(
        diamond_active=jnp.array(False),
        helicopter_active=jnp.array(False),
        helicopter_melee_step=jnp.array(0),
        satellite_active=jnp.array(False),
        helicopter_bomb_active=jnp.array(False),
        satellite_laser_active=jnp.array(False),
        satellite_laser_timer=jnp.array(env.consts.SATELLITE_LASER_DROP_PERIOD),
        pit_active=jnp.array(False),
        pit_x=jnp.array(-100),
    )


def test_player_bullet_position_persists(env):
    """Regression: bullet x/y must survive across frames (clobber bug)."""

    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    _, state, _, _, _ = env.step(state, jnp.array(FIRE))
    x0, y0 = int(state.player_bullet_x), int(state.player_bullet_y)
    _, state, _, _, _ = env.step(state, jnp.array(NOOP))
    assert bool(state.player_bullet_active)
    assert int(state.player_bullet_x) == x0 + 2
    assert int(state.player_bullet_y) == y0 - 2


def test_bullet_shoots_diamond_scores_once(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        diamond_active=jnp.array(True),
        diamond_x=jnp.array(49),
        diamond_y=jnp.array(105),
    )
    _, state, _, _, _ = env.step(state, jnp.array(FIRE))
    for _ in range(12):
        _, state, _, _, _ = env.step(state, jnp.array(NOOP))
        # Freeze respawning so only our diamond exists.
        state = _clean_state(env, state).replace(
            diamond_active=state.diamond_active,
            diamond_x=state.diamond_x,
            diamond_y=state.diamond_y,
            player_bullet_active=state.player_bullet_active,
            player_bullet_x=state.player_bullet_x,
            player_bullet_y=state.player_bullet_y,
            player_bullet_step=state.player_bullet_step,
        )
    assert int(state.score) == env.consts.SCORE_DIAMOND
    assert not bool(state.player_bullet_active)


def test_no_score_while_bullet_misses(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    _, state, _, _, _ = env.step(state, jnp.array(FIRE))
    for _ in range(5):
        _, state, _, _, _ = env.step(state, jnp.array(NOOP))
        state = _clean_state(env, state).replace(
            player_bullet_active=state.player_bullet_active,
            player_bullet_x=state.player_bullet_x,
            player_bullet_y=state.player_bullet_y,
            player_bullet_step=state.player_bullet_step,
        )
    assert int(state.score) == 0


def test_bullet_destroys_satellite(env):
    """The manual's scoring: shooting the satellite pays the enemy score.

    The round is spent on the hit and the satellite leaves the screen
    until its respawn breather runs out.
    """

    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        satellite_active=jnp.array(True),
        satellite_x=jnp.array(49),
        satellite_y=jnp.array(100),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(2),
        player_bullet_x=jnp.array(48),
        player_bullet_y=jnp.array(106),
    )
    _, state, _, _, info = env.step(state, jnp.array(NOOP))
    assert not bool(state.satellite_active)
    assert int(state.score) == env.consts.SCORE_ENEMY
    assert not bool(state.player_bullet_active)


def test_bullet_passes_through_helicopter(env):
    """Original behavior: the bullet ignores helicopters entirely."""

    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        helicopter_active=jnp.array(True),
        helicopter_x=jnp.array(49),
        helicopter_y=jnp.array(103),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(2),
        player_bullet_x=jnp.array(48),
        player_bullet_y=jnp.array(104),
    )
    _, state, _, _, _ = env.step(state, jnp.array(NOOP))
    assert bool(state.helicopter_active)
    assert int(state.score) == 0
    assert bool(state.player_bullet_active)


def test_helicopter_drops_bomb(env):
    """The first bomb releases when the heli closes to RANGE_FAR of the player.

    Uses a drop chance of 1.0 so the coin flip can't make the test flaky.
    """

    from jaxatari.games.jax_jamesbond import JamesBondConstants

    sure_env = JaxJamesBond(JamesBondConstants(HELICOPTER_BOMB_DROP_CHANCE=1.0))
    _, state = sure_env.reset(jax.random.PRNGKey(0))
    state = _clean_state(sure_env, state)
    far = sure_env.consts.HELICOPTER_BOMB_RANGE
    state = state.replace(
        helicopter_active=jnp.array(True),
        helicopter_x=(state.player_x + far - 1).astype(jnp.int32),  # just in range
        helicopter_y=jnp.array(57),
    )
    _, state, _, _, _ = sure_env.step(state, jnp.array(NOOP))
    assert bool(state.helicopter_bomb_active)
    assert int(state.helicopter_bombs_dropped) == 1


def test_bomb_hits_player(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        helicopter_bomb_active=jnp.array(True),
        helicopter_bomb_x=(state.player_x + 2).astype(jnp.int32),
        helicopter_bomb_y=(state.player_y - 4).astype(jnp.int32),
        helicopter_bomb_vx=jnp.array(-1),
    )
    lives_before = int(state.lives)
    _, state, reward, _, _ = env.step(state, jnp.array(NOOP))
    assert int(state.lives) == lives_before - 1
    assert not bool(state.helicopter_bomb_active)


def test_player_dies_in_pit(env):
    """Driving into the fire pit on the ground costs one life (cooldown)."""

    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        pit_active=jnp.array(True),
        pit_x=(state.player_x - 4).astype(state.pit_x.dtype),
        pit_y=jnp.array(122).astype(state.pit_y.dtype),
    )
    lives_before = int(state.lives)
    for _ in range(5):
        _, state, _, _, _ = env.step(state, jnp.array(NOOP))
        # Keep the pit under the player despite scrolling and switch off the
        # enemy fire that spawns during the step.
        state = state.replace(
            pit_active=jnp.array(True),
            pit_x=(state.player_x - 4).astype(state.pit_x.dtype),
            helicopter_bomb_active=jnp.array(False),
            satellite_laser_active=jnp.array(False),
        )
    assert int(state.lives) == lives_before - 1
    assert int(state.hit_cooldown) > 0


def test_jumping_player_clears_pit(env):
    """A player in the air passes over the pit unharmed."""

    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        pit_active=jnp.array(True),
        pit_x=(state.player_x - 4).astype(state.pit_x.dtype),
        pit_y=jnp.array(122).astype(state.pit_y.dtype),
        player_y=jnp.array(110).astype(state.player_y.dtype),
        player_jumping=jnp.array(True),
        player_in_air_step=jnp.array(30),
    )
    lives_before = int(state.lives)
    _, state, _, _, _ = env.step(state, jnp.array(NOOP))
    assert int(state.lives) == lives_before


def test_bullet_ignores_pit(env):
    """The player shot passes over the pit without any collision response."""

    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        pit_active=jnp.array(True),
        pit_x=jnp.array(40).astype(state.pit_x.dtype),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(1),
        player_bullet_x=jnp.array(40),
        player_bullet_y=jnp.array(112),
    )
    _, state, _, _, _ = env.step(state, jnp.array(NOOP))
    assert bool(state.player_bullet_active)
    assert int(state.score) == 0
