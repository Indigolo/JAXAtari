"""Unit tests for the JamesBond collision systems.

Each test builds a deterministic state via state.replace and steps once or a
few times, so the checks stay independent of the spawn schedule.
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
        diamond_active=jnp.zeros_like(state.diamond_active),
        helicopter_active=jnp.zeros_like(state.helicopter_active),
        satellite_active=jnp.zeros_like(state.satellite_active),
        pit_active=jnp.array(False),
        pit_x=jnp.array(-100),
        bullet_active=jnp.zeros_like(state.bullet_active),
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
        diamond_active=state.diamond_active.at[0].set(True),
        diamond_x=state.diamond_x.at[0].set(49.0),
        diamond_y=state.diamond_y.at[0].set(105.0),
    )
    _, state, _, _, _ = env.step(state, jnp.array(FIRE))
    rewards = 0.0
    for _ in range(12):
        _, state, reward, _, info = env.step(state, jnp.array(NOOP))
        rewards += float(reward)
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
    assert rewards == pytest.approx(env.consts.REWARD_DIAMOND)
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


def test_bullet_passes_through_satellite(env):
    """Original behavior: the bullet ignores satellites entirely."""

    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        satellite_active=state.satellite_active.at[0].set(True),
        satellite_x=state.satellite_x.at[0].set(49.0),
        satellite_y=state.satellite_y.at[0].set(100.0),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(2),
        player_bullet_x=jnp.array(48),
        player_bullet_y=jnp.array(106),
    )
    _, state, _, _, info = env.step(state, jnp.array(NOOP))
    assert bool(state.satellite_active[0])
    assert int(state.score) == 0
    assert bool(state.player_bullet_active)
    assert not bool(info.hit_enemy)


def test_bullet_passes_through_helicopter(env):
    """Original behavior: the bullet ignores helicopters entirely."""

    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        helicopter_active=state.helicopter_active.at[0].set(True),
        helicopter_x=state.helicopter_x.at[0].set(49.0),
        helicopter_y=state.helicopter_y.at[0].set(103.0),
        player_bullet_active=jnp.array(True),
        player_bullet_step=jnp.array(2),
        player_bullet_x=jnp.array(48),
        player_bullet_y=jnp.array(104),
    )
    _, state, _, _, _ = env.step(state, jnp.array(NOOP))
    assert bool(state.helicopter_active[0])
    assert int(state.score) == 0
    assert bool(state.player_bullet_active)


def test_helicopter_contact_costs_one_life(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        helicopter_active=state.helicopter_active.at[0].set(True),
        helicopter_x=state.helicopter_x.at[0].set(state.player_x),
        helicopter_y=state.helicopter_y.at[0].set(state.player_y),
    )
    lives_before = int(state.lives)
    for _ in range(5):
        _, state, _, _, _ = env.step(state, jnp.array(NOOP))
        state = state.replace(
            helicopter_x=state.helicopter_x.at[0].set(state.player_x),
            helicopter_y=state.helicopter_y.at[0].set(state.player_y),
            pit_active=jnp.array(False),
            pit_x=jnp.array(-100),
        )
    # Cooldown ensures a sustained overlap only costs a single life.
    assert int(state.lives) == lives_before - 1
    assert int(state.hit_cooldown) > 0


def test_helicopter_drops_bomb(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        helicopter_active=state.helicopter_active.at[0].set(True),
        helicopter_x=state.helicopter_x.at[0].set(60.0),
        helicopter_y=state.helicopter_y.at[0].set(57.0),
    )
    dropped = False
    for _ in range(env.consts.ENEMY_BOMB_DROP_PERIOD + 1):
        _, state, _, _, _ = env.step(state, jnp.array(NOOP))
        if bool(jnp.any(state.bullet_active)):
            dropped = True
            break
    assert dropped


def test_bomb_hits_player(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _clean_state(env, state)
    state = state.replace(
        bullet_active=state.bullet_active.at[0].set(True),
        bullet_x=state.bullet_x.at[0].set(state.player_x + 2.0),
        bullet_y=state.bullet_y.at[0].set(state.player_y - 4.0),
    )
    lives_before = int(state.lives)
    _, state, reward, _, _ = env.step(state, jnp.array(NOOP))
    assert int(state.lives) == lives_before - 1
    assert float(reward) == env.consts.REWARD_LOST_LIFE
    assert not bool(state.bullet_active[0])


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
        # Keep the pit under the player despite scrolling.
        state = state.replace(
            pit_active=jnp.array(True),
            pit_x=(state.player_x - 4).astype(state.pit_x.dtype),
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
        player_y=jnp.array(110.0),
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
