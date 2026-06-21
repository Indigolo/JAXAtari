import jax
import jax.numpy as jnp

from jaxatari.games.jax_jamesbond import JaxJamesBond, _aabb_overlap


def _reset_env():
    env = JaxJamesBond()
    _, state = env.reset(jax.random.PRNGKey(0))
    return env, state


def test_aabb_overlap_scalar_and_vector_cases():
    assert bool(_aabb_overlap(0, 0, 5, 5, 4, 4, 5, 5))
    assert not bool(_aabb_overlap(0, 0, 5, 5, 5, 5, 5, 5))

    overlaps = _aabb_overlap(
        0,
        0,
        5,
        5,
        jnp.array([4, 5], dtype=jnp.float32),
        jnp.array([4, 5], dtype=jnp.float32),
        5,
        5,
    )
    assert jnp.array_equal(overlaps, jnp.array([True, False]))


def test_inactive_collectible_is_ignored():
    env, state = _reset_env()
    state = state.replace(
        diamond_x=state.diamond_x.at[0].set(state.player_x),
        diamond_y=state.diamond_y.at[0].set(state.player_y),
        diamond_active=state.diamond_active.at[0].set(False),
    )

    _, next_state, reward, _, info = env.step(state, jnp.array(0, dtype=jnp.int32))

    assert int(next_state.score) == 0
    assert float(reward) == 0.0
    assert not bool(info.collected_diamond)
    assert not bool(next_state.diamond_active[0])


def test_collectible_collision_deactivates_item_and_scores():
    env, state = _reset_env()
    state = state.replace(
        diamond_x=state.diamond_x.at[0].set(state.player_x),
        diamond_y=state.diamond_y.at[0].set(state.player_y),
        diamond_active=state.diamond_active.at[0].set(True),
    )

    _, next_state, reward, _, info = env.step(state, jnp.array(0, dtype=jnp.int32))

    assert not bool(next_state.diamond_active[0])
    assert int(next_state.score) == env.consts.SCORE_DIAMOND
    assert float(reward) == env.consts.REWARD_DIAMOND
    assert bool(info.collected_diamond)
    assert bool(info.collision_happened)


def test_player_enemy_collision_damages_once_during_cooldown():
    env, state = _reset_env()
    state = state.replace(
        enemy_x=state.enemy_x.at[0].set(state.player_x),
        enemy_y=state.enemy_y.at[0].set(state.player_y),
        enemy_active=state.enemy_active.at[0].set(True),
    )

    _, damaged_state, reward, _, info = env.step(state, jnp.array(0, dtype=jnp.int32))
    assert int(damaged_state.lives) == env.consts.MAX_LIVES - 1
    assert int(damaged_state.hit_cooldown) == env.consts.HIT_COOLDOWN_STEPS
    assert float(reward) == env.consts.REWARD_LOST_LIFE
    assert bool(info.hit_enemy)

    _, cooldown_state, reward, _, info = env.step(
        damaged_state, jnp.array(0, dtype=jnp.int32)
    )
    assert int(cooldown_state.lives) == env.consts.MAX_LIVES - 1
    assert int(cooldown_state.hit_cooldown) == env.consts.HIT_COOLDOWN_STEPS - 1
    assert float(reward) == 0.0
    assert bool(info.hit_enemy)


def test_bullet_enemy_collision_deactivates_both_and_scores():
    env, state = _reset_env()
    state = state.replace(
        bullet_x=state.bullet_x.at[0].set(80.0),
        bullet_y=state.bullet_y.at[0].set(80.0),
        bullet_active=state.bullet_active.at[0].set(True),
        enemy_x=state.enemy_x.at[0].set(80.0),
        enemy_y=state.enemy_y.at[0].set(80.0),
        enemy_active=state.enemy_active.at[0].set(True),
    )

    _, next_state, reward, _, info = env.step(state, jnp.array(0, dtype=jnp.int32))

    assert not bool(next_state.bullet_active[0])
    assert not bool(next_state.enemy_active[0])
    assert int(next_state.score) == env.consts.SCORE_ENEMY
    assert float(reward) == env.consts.REWARD_ENEMY
    assert bool(info.hit_enemy)
    assert bool(info.collision_happened)
