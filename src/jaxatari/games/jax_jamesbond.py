"""Runnable JamesBond skeleton environment.

This file intentionally defines only the shared environment contract and minimal
placeholder behavior. Gameplay systems such as object spawning, collisions,
scoring, lives, and sprite-accurate rendering are left for follow-up work.
"""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax import lax
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


def _aabb_overlap(
    ax: chex.Array,
    ay: chex.Array,
    aw: chex.Array,
    ah: chex.Array,
    bx: chex.Array,
    by: chex.Array,
    bw: chex.Array,
    bh: chex.Array,
) -> chex.Array:
    """Return whether two top-left anchored AABB rectangles overlap."""

    x_overlap = jnp.logical_and(ax < bx + bw, ax + aw > bx)
    y_overlap = jnp.logical_and(ay < by + bh, ay + ah > by)
    return jnp.logical_and(x_overlap, y_overlap)


class JamesBondConstants(struct.PyTreeNode):
    """Static JamesBond placeholder constants shared by state, spaces, and render."""

    # Atari-style frame dimensions and initial play-area bounds.
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=210)
    ## Already has player sprite width/height respected
    GAME_AREA_MIN_X: int = struct.field(pytree_node=False, default=4) ## Playable Area: 5 (Coordinate system starting with 1) -- Shown in /jb_sprites/game_area_min_x.npy
    GAME_AREA_MAX_X: int = struct.field(pytree_node=False, default=73) ## Playable Area: 81 (Coordinate system starting with 1)
    GAME_AREA_MIN_Y: int = struct.field(pytree_node=False, default=0) ## Playable Area: 123 (Top-left coordinate system); 87 (Bottom-right co-sys)
    GAME_AREA_MAX_Y: int = struct.field(pytree_node=False, default=119)

    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=8)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=4)
    PLAYER_INIT_X: int = struct.field(pytree_node=False, default=29) ## 30 if starting with 1
    PLAYER_INIT_Y: int = struct.field(pytree_node=False, default=119) ## 120 if starting with 1
    PLAYER_IN_AIR_STEPS = jnp.array([ ## For the gravity feel of jumps. Each jump is 71 frames, 72nd frame is the start of the fall
        0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ## TODO: Remove first zero?
        0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 
        0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1
    ])

    MAX_LIVES: int = struct.field(pytree_node=False, default=3)
    MAX_DIAMONDS: int = struct.field(pytree_node=False, default=8)
    MAX_ENEMIES: int = struct.field(pytree_node=False, default=8)
    MAX_HELICOPTERS: int = struct.field(pytree_node=False, default=4)
    MAX_SATELLITES: int = struct.field(pytree_node=False, default=4)
    MAX_BULLETS: int = struct.field(pytree_node=False, default=4)
    MAX_EPISODE_STEPS: int = struct.field(pytree_node=False, default=5000)

    DIAMOND_WIDTH: int = struct.field(pytree_node=False, default=7) ##TODO: There is 7 pixels in the diamond sprite, including the shining thing of diamond
    DIAMOND_HEIGHT: int = struct.field(pytree_node=False, default=13) ##TODO: There is 13 pixels in the diamond sprite, including the shining thing of diamond 
    ## TODO: Change / Remove after observation and collision enemey variables have been changed; or else will cause fail tests
    ENEMY_WIDTH: int = struct.field(pytree_node=False, default=10)
    ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=8)
    ## TODO: Enemies (now i only have the helicopter and satellite enemies)
    HELICOPTER_ENEMY_WIDTH: int = struct.field(pytree_node=False, default=8) ## TODO: Helicopter width is 8 pixels
    HELICOPTER_ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=6) ## TODO: Helicopter height is 6 pixels
    SATELLITE_ENEMY_WIDTH: int = struct.field(pytree_node=False, default=8) ## TODO: Satellite width is 8 pixels
    SATELLITE_ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=14) ## TODO: Satellite height is 14 pixels
    BULLET_WIDTH: int = struct.field(pytree_node=False, default=1) ## TODO: which bullet?
    BULLET_HEIGHT: int = struct.field(pytree_node=False, default=4)
    ## The player shot vanishes just above the row of the highest on-screen
    ## object (diamond/helicopter/satellite), like in the original game.
    BULLET_DESPAWN_MARGIN: int = struct.field(pytree_node=False, default=4)

    # Collision boxes are kept a little smaller than the real sprite sizes so
    # near-misses do not register, matching how the original game feels.
    PLAYER_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=6)   ## < PLAYER_WIDTH 8
    PLAYER_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=3)  ## < PLAYER_HEIGHT 4
    DIAMOND_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=4)  ## < DIAMOND_WIDTH 7
    DIAMOND_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=4) ## < DIAMOND_HEIGHT 13
    ENEMY_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=8)    ## < ENEMY_WIDTH 10
    ENEMY_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=6)   ## < ENEMY_HEIGHT 8
    HELICOPTER_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=6)  ## < HELICOPTER_ENEMY_WIDTH 8
    HELICOPTER_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=4) ## < HELICOPTER_ENEMY_HEIGHT 6
    SATELLITE_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=6)   ## < SATELLITE_ENEMY_WIDTH 8
    SATELLITE_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=12) ## < SATELLITE_ENEMY_HEIGHT 14
    BULLET_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=1) ## TODO: Only player?
    BULLET_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=4)

    SCORE_DIAMOND: int = struct.field(pytree_node=False, default=100)
    SCORE_ENEMY: int = struct.field(pytree_node=False, default=250)

    ## Helicopter bombs (enemy fire), stored in the generic bullet_* arrays.
    ENEMY_BOMB_DROP_PERIOD: int = struct.field(pytree_node=False, default=60) ## Frames between drops
    ENEMY_BOMB_FALL_SPEED: float = struct.field(pytree_node=False, default=1.0)
    ENEMY_BOMB_DRIFT_SPEED: float = struct.field(pytree_node=False, default=0.75) ## Keeps world-scroll momentum (SPEED_R2L)

    ## Fire pits: ground holes the player must jump over. Deadly on ground
    ## contact only; bullets and bombs pass over them like in the original.
    MAX_FIREPITS: int = struct.field(pytree_node=False, default=2)
    FIREPIT_WIDTH: int = struct.field(pytree_node=False, default=20) ## TODO: Tune against the original sprite (48px on the 160px screen)
    FIREPIT_HEIGHT: int = struct.field(pytree_node=False, default=5)
    FIREPIT_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=14) ## < FIREPIT_WIDTH 20, so an edge tap is survivable
    FIREPIT_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=4) ## < FIREPIT_HEIGHT 5
    FIREPIT_Y: int = struct.field(pytree_node=False, default=120) ## Ground row, just below the player top
    FIREPIT_SPEED: float = struct.field(pytree_node=False, default=0.75) ## Same as SPEED_R2L world scroll

    HIT_COOLDOWN_STEPS: int = struct.field(pytree_node=False, default=60)

    REWARD_STEP: float = struct.field(pytree_node=False, default=0.0)
    REWARD_DIAMOND: float = struct.field(pytree_node=False, default=1.0)
    REWARD_ENEMY: float = struct.field(pytree_node=False, default=2.0)
    REWARD_HIT_ENEMY: float = struct.field(pytree_node=False, default=-1.0)
    REWARD_LOST_LIFE: float = struct.field(pytree_node=False, default=-1.0)

    ACTION_MEANINGS: Tuple[str, ...] = struct.field( ## TODO: What is this for?
        pytree_node=False,
        default=(
            "NOOP", 
            "FIRE", 
            "UP", 
            "RIGHT", 
            "LEFT", 
            "DOWN",
            "UPRIGHT",
            "UPLEFT",
            "DOWNRIGHT",
            "DOWNLEFT",
            "UPFIRE",
            "RIGHTFIRE",
            "LEFTFIRE",
            "DOWNFIRE",
            "UPRIGHTFIRE",
            "UPLEFTFIRE",
            "DOWNRIGHTFIRE",
            "DOWNLEFTFIRE"
            ),
    )

    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(8, 14, 32)
    )
    PLAY_AREA_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(20, 42, 66)
    )
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(236, 236, 236)
    )
    DIAMOND_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(0, 216, 255)
    )
    ENEMY_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(220, 64, 64)
    )
    BULLET_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(250, 220, 72)
    )
    FIREPIT_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(200, 72, 24)
    )


@struct.dataclass
class JamesBondState:
    """Full internal state with fixed-size object arrays and active masks."""

    player_x: chex.Array
    player_y: chex.Array
    player_vy: chex.Array
    player_vx: chex.Array
    player_jumping: chex.Array
    player_falling: chex.Array
    player_fast_falling: chex.Array
    player_in_air_step: chex.Array
    player_bullet_active: chex.Array
    player_bullet_step: chex.Array
    player_bullet_x: chex.Array
    player_bullet_y: chex.Array
    bullet_vx: chex.Array
    lives: chex.Array
    score: chex.Array
    step_count: chex.Array
    level_progress: chex.Array
    hit_cooldown: chex.Array ## TODO: What for?
    diamond_x: chex.Array
    diamond_y: chex.Array
    diamond_active: chex.Array
    spawn_diamond_next: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_active: chex.Array
    ## TODO: Here using helicopter and satellite instead of enemy
    helicopter_x: chex.Array
    helicopter_y: chex.Array
    helicopter_active: chex.Array
    satellite_x: chex.Array
    satellite_y: chex.Array
    satellite_active: chex.Array
    firepit_x: chex.Array
    firepit_active: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_active: chex.Array
    reward_delta: chex.Array
    collision_happened: chex.Array
    collected_diamond: chex.Array
    hit_enemy: chex.Array
    fired_bullet: chex.Array
    key: chex.PRNGKey


@struct.dataclass
class JamesBondObservation:
    """Object-centric observation matching observation_space()."""

    player: ObjectObservation
    diamonds: ObjectObservation
    enemies: ObjectObservation
    player_velocity: jnp.ndarray
    ## helicopters: ObjectObservation
    ## satellites: ObjectObservation
    bullets: ObjectObservation
    lives: jnp.ndarray
    score: jnp.ndarray
    level_progress: jnp.ndarray


@struct.dataclass
class JamesBondInfo:
    """Debug/event info for smoke tests and future gameplay systems."""

    collision_happened: jnp.ndarray
    collected_diamond: jnp.ndarray
    hit_enemy: jnp.ndarray
    fired_bullet: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    level_progress: jnp.ndarray
    step_count: jnp.ndarray


class JaxJamesBond(
    JaxEnvironment[JamesBondState, JamesBondObservation, JamesBondInfo, JamesBondConstants]
):
    """Minimal runnable JamesBond environment following the JAXAtari API."""

    # Compact agent action indices map to these ALE-style actions.
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP, 
            Action.FIRE, 
            Action.UP, 
            Action.RIGHT, 
            Action.LEFT, 
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: JamesBondConstants = None):
        consts = consts or JamesBondConstants()
        super().__init__(consts)
        self.renderer = JamesBondRenderer(self.consts)

    def reset(
        self, key: chex.PRNGKey = jax.random.PRNGKey(0)
    ) -> Tuple[JamesBondObservation, JamesBondState]:
        """Create an empty level state with inactive object slots."""

        if key is None:
            key = jax.random.PRNGKey(0)
        state_key, _ = jax.random.split(key)

        state = JamesBondState(
            player_x=jnp.array(self.consts.PLAYER_INIT_X, dtype=jnp.float32),
            player_y=jnp.array(self.consts.PLAYER_INIT_Y, dtype=jnp.float32),
            player_vx=jnp.array(0, dtype=jnp.int32),
            player_vy=jnp.array(0, dtype=jnp.int32),
            player_jumping=jnp.array(False, dtype=jnp.bool_),
            player_falling=jnp.array(False, dtype=jnp.bool_),
            player_fast_falling=jnp.array(False, dtype=jnp.bool_),
            player_in_air_step=jnp.array(0, dtype=jnp.int32),
            player_bullet_active=jnp.array(False, dtype=jnp.bool_),
            player_bullet_step=jnp.array(-1, dtype=jnp.int32),
            player_bullet_x=jnp.array(-1, dtype=jnp.int32),
            player_bullet_y=jnp.array(-1, dtype=jnp.int32),
            bullet_vx=jnp.array(0, dtype=jnp.float32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            level_progress=jnp.array(0, dtype=jnp.int32),
            hit_cooldown=jnp.array(0, dtype=jnp.int32),
            diamond_x=jnp.zeros((self.consts.MAX_DIAMONDS,), dtype=jnp.float32),
            diamond_y=jnp.zeros((self.consts.MAX_DIAMONDS,), dtype=jnp.float32),
            diamond_active=jnp.zeros((self.consts.MAX_DIAMONDS,), dtype=jnp.bool_),
            spawn_diamond_next=jnp.array(True, dtype=jnp.bool_), ## TODO: In state requires this, but is this array or zero-dimensional?
            ## TODO: Change / Remove after observation and collision enemy variables have been changed; or else will cause fail tests
            enemy_x=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.float32),
            enemy_y=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.float32),
            enemy_active=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.bool_),
            ## TODO: Here using helicopter and satellite instead of enemy
            helicopter_x=jnp.zeros((self.consts.MAX_HELICOPTERS,), dtype=jnp.float32),
            helicopter_y=jnp.zeros((self.consts.MAX_HELICOPTERS,), dtype=jnp.float32),
            helicopter_active=jnp.zeros((self.consts.MAX_HELICOPTERS,), dtype=jnp.bool_),
            satellite_x=jnp.zeros((self.consts.MAX_SATELLITES,), dtype=jnp.float32),
            satellite_y=jnp.zeros((self.consts.MAX_SATELLITES,), dtype=jnp.float32),
            satellite_active=jnp.zeros((self.consts.MAX_SATELLITES,), dtype=jnp.bool_),
            firepit_x=jnp.zeros((self.consts.MAX_FIREPITS,), dtype=jnp.float32),
            firepit_active=jnp.zeros((self.consts.MAX_FIREPITS,), dtype=jnp.bool_),
            bullet_x=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.float32),
            bullet_y=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.float32),
            bullet_active=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.bool_),
            reward_delta=jnp.array(0.0, dtype=jnp.float32), ## TODO: What for?
            collision_happened=jnp.array(False, dtype=jnp.bool_), ## TODO: What for?
            collected_diamond=jnp.array(False, dtype=jnp.bool_), ## TODO: Does this reset?
            hit_enemy=jnp.array(False, dtype=jnp.bool_), ## TODO: Does this reset?
            fired_bullet=jnp.array(False, dtype=jnp.bool_), ## TODO: Already implemented for player through 'player_bullet_active'
            key=state_key,
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: JamesBondState, action: chex.Array
    ) -> Tuple[JamesBondObservation, JamesBondState, chex.Array, chex.Array, JamesBondInfo]:
        """Advance one placeholder frame and return the repo-standard tuple."""

        atari_action = self._decode_action(action)
        previous_state = state

        state = state.replace(
            step_count=state.step_count + 1,
            collision_happened=jnp.array(False, dtype=jnp.bool_),
            collected_diamond=jnp.array(False, dtype=jnp.bool_),
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
            reward_delta=jnp.array(0.0, dtype=jnp.float32),
            hit_cooldown=jnp.maximum(state.hit_cooldown - 1, 0),
            fired_bullet=atari_action == Action.FIRE,
        )
        state = self._step_player(state, atari_action)
        state = self._update_objects_placeholder(state)
        state = self._update_enemy_bombs(state)
        state = self._update_firepits(state)
        state = self._check_collisions_placeholder(state)

        _, next_key = jax.random.split(state.key)
        state = state.replace(key=next_key)

        observation = self._get_observation(state)
        reward = self._calculate_reward_placeholder(previous_state, state)
        done = self._is_done(state)
        info = self._get_info(state)

        return observation, state, reward, done, info

    def render(self, state: JamesBondState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict: # TODO: Test error here?
        screen_size = (self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH)
        return spaces.Dict(
            {
                "player": spaces.get_object_space(n=None, screen_size=screen_size),
                "diamonds": spaces.get_object_space(
                    n=self.consts.MAX_DIAMONDS, screen_size=screen_size
                ),
                "player_velocity": spaces.Box(
                    low=jnp.array([-10.0, -20.0], dtype=jnp.float32),
                    high=jnp.array([10.0, 20.0], dtype=jnp.float32),
                    shape=(2,),
                    dtype=jnp.float32,
                ),
                "enemies": spaces.get_object_space(
                    n=self.consts.MAX_ENEMIES, screen_size=screen_size
                ),
                ## "helicopters": spaces.get_object_space(
                ##     n=self.consts.MAX_HELICOPTERS, screen_size=screen_size
                ## ),
                ## "satellites": spaces.get_object_space(
                ##     n=self.consts.MAX_SATELLITES, screen_size=screen_size
                ## ),
                "bullets": spaces.get_object_space(
                    n=self.consts.MAX_BULLETS, screen_size=screen_size
                ),
                "lives": spaces.Box(
                    low=0,
                    high=self.consts.MAX_LIVES,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "score": spaces.Box(
                    low=0,
                    high=1_000_000,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "level_progress": spaces.Box(
                    low=0,
                    high=self.consts.MAX_EPISODE_STEPS,
                    shape=(),
                    dtype=jnp.int32,
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: JamesBondState) -> JamesBondObservation:
        """Build the structured object observation from internal state."""

        player = ObjectObservation.create(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_WIDTH, dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_HEIGHT, dtype=jnp.int32),
            active=jnp.array(True, dtype=jnp.bool_),
        )
        diamonds = self._object_group_observation(
            state.diamond_x,
            state.diamond_y,
            state.diamond_active,
            self.consts.DIAMOND_WIDTH,
            self.consts.DIAMOND_HEIGHT,
        )
        enemies = self._object_group_observation(
            state.enemy_x,
            state.enemy_y,
            state.enemy_active,
            self.consts.ENEMY_WIDTH,
            self.consts.ENEMY_HEIGHT,
        )
        ## helicopters = self._object_group_observation(
        ##     state.helicopter_x,
        ##     state.helicopter_y,
        ##     state.helicopter_active,
        ##     self.consts.HELICOPTER_ENEMY_WIDTH,
        ##     self.consts.HELICOPTER_ENEMY_HEIGHT,
        ## )
        ## satellites = self._object_group_observation(
        ##     state.satellite_x,
        ##     state.satellite_y,
        ##     state.satellite_active,
        ##     self.consts.SATELLITE_ENEMY_WIDTH,
        ##     self.consts.SATELLITE_ENEMY_HEIGHT,
        ## )
        bullets = self._object_group_observation(
            state.bullet_x,
            state.bullet_y,
            state.bullet_active,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            ## bullet_vx is a scalar, so broadcast the orientation to one value
            ## per bullet slot or the observation loses 3 features vs the space.
            orientation=jnp.broadcast_to(
                jnp.where(state.bullet_vx < 0, 270.0, 90.0), state.bullet_x.shape
            ),
        )
        return JamesBondObservation(
            player=player,
            player_velocity=jnp.stack([state.player_vx, state.player_vy]).astype( ## TODO: Does observation need this or can we remove it?
                jnp.float32
            ),
            diamonds=diamonds,
            enemies=enemies,
            ## helicopters=helicopters,
            ## satellites=satellites,
            bullets=bullets,
            lives=state.lives,
            score=state.score,
            level_progress=state.level_progress,
        )

    def _object_group_observation(
        self,
        x: chex.Array,
        y: chex.Array,
        active: chex.Array,
        width: int,
        height: int,
        orientation: chex.Array = None,
    ) -> ObjectObservation:
        """Convert fixed-size object arrays plus masks into ObjectObservation."""

        ## Inactive objects keep drifting in _update_objects_placeholder, so
        ## their stale coordinates can leave the screen bounds. Zero them out
        ## and clamp active ones so the observation stays inside its space.
        safe_x = jnp.where(active, jnp.clip(x, 0, self.consts.SCREEN_WIDTH), 0.0)
        safe_y = jnp.where(active, jnp.clip(y, 0, self.consts.SCREEN_HEIGHT), 0.0)

        return ObjectObservation.create(
            x=safe_x,
            y=safe_y,
            width=jnp.full(x.shape, width, dtype=jnp.int32),
            height=jnp.full(y.shape, height, dtype=jnp.int32),
            active=active,
            orientation=orientation,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: JamesBondState) -> JamesBondInfo:
        return JamesBondInfo(
            collision_happened=state.collision_happened,
            collected_diamond=state.collected_diamond,
            hit_enemy=state.hit_enemy,
            fired_bullet=state.fired_bullet,
            score=state.score,
            lives=state.lives,
            level_progress=state.level_progress,
            step_count=state.step_count,
        )

    def _decode_action(self, action: chex.Array) -> chex.Array:
        """Translate compact action-space indices to JAXAtariAction values."""

        return jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

    def _step_player(
        self, state: JamesBondState, atari_action: chex.Array
    ) -> JamesBondState:
        player_x = state.player_x
        player_y = state.player_y
        player_jumping = state.player_jumping
        player_falling = state.player_falling
        player_fast_falling = state.player_fast_falling
        player_in_air_step = state.player_in_air_step

        player_bullet_active = state.player_bullet_active
        player_bullet_step = state.player_bullet_step
        player_bullet_x = state.player_bullet_x
        player_bullet_y = state.player_bullet_y

        up_pressed = jnp.any(
            jnp.array([
                atari_action == Action.UP,
                atari_action == Action.UPRIGHT,
                atari_action == Action.UPLEFT,
                atari_action == Action.UPFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.UPLEFTFIRE,
            ])
        )

        right_pressed = jnp.any(
            jnp.array([
                atari_action == Action.RIGHT,
                atari_action == Action.UPRIGHT,
                atari_action == Action.DOWNRIGHT,
                atari_action == Action.RIGHTFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.DOWNRIGHTFIRE
            ])
        )

        left_pressed = jnp.any(
            jnp.array([
                atari_action == Action.LEFT,
                atari_action == Action.UPLEFT,
                atari_action == Action.DOWNLEFT,
                atari_action == Action.LEFTFIRE,
                atari_action == Action.UPLEFTFIRE,
                atari_action == Action.DOWNLEFTFIRE
            ])
        )

        down_pressed = jnp.any(
            jnp.array([
                atari_action == Action.DOWN,
                atari_action == Action.DOWNLEFT,
                atari_action == Action.DOWNRIGHT,
                atari_action == Action.DOWNFIRE,
                atari_action == Action.DOWNLEFTFIRE,
                atari_action == Action.DOWNRIGHTFIRE,
            ])
        )

        fire_pressed = jnp.any(
            jnp.array([
                atari_action == Action.FIRE,
                atari_action == Action.RIGHTFIRE,
                atari_action == Action.LEFTFIRE,
                atari_action == Action.UPFIRE,
                atari_action == Action.DOWNFIRE,
                atari_action == Action.UPLEFTFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.DOWNLEFTFIRE,
                atari_action == Action.DOWNRIGHTFIRE,
            ])
        )


        ###
        ### Player Movement Controller
        ###

        player_x = jnp.where(
            right_pressed, 
            jnp.where(
                state.step_count % 2 == 0,
                jnp.clip(player_x + 1, self.consts.GAME_AREA_MIN_X, self.consts.GAME_AREA_MAX_X), 
                player_x
            ),
            jnp.where(
                left_pressed, 
                jnp.where(
                    state.step_count % 4 == 0, 
                    jnp.clip(player_x - 1, self.consts.GAME_AREA_MIN_X, self.consts.GAME_AREA_MAX_X), 
                    player_x
                ), 
                player_x
            )
        )

        up_pressed = jnp.where(player_jumping, False, up_pressed)
        down_pressed = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, down_pressed) ## TODO: Maybe change for 2nd stage?
        
        player_jumping = jnp.where(
            player_jumping,
            player_jumping, 
            jnp.where(
                jnp.logical_and(up_pressed, player_in_air_step < 71), 
                True, 
                False
            )
        )
        
        player_falling = jnp.where(
            jnp.logical_and(
                jnp.logical_or(player_falling, player_in_air_step >= 71), 
                player_y != self.consts.PLAYER_INIT_Y
            ), 
            True, 
            player_falling
        )
        
        player_fast_falling = jnp.where(
            player_fast_falling,
            player_fast_falling,
            jnp.where(
                jnp.logical_and(
                    down_pressed,
                    jnp.logical_or(player_jumping, player_falling)
                ),
                True,
                False
            )
        )

        player_falling = jnp.where(
            player_fast_falling,
            False,
            player_falling,
        )

        player_jumping = jnp.where(
            jnp.logical_or(player_falling, player_fast_falling),
            False,
            player_jumping
        )
        
        player_in_air_step = jnp.where( ## Start immediately falling when reaching the peak of the jump
            player_in_air_step >= 71, 
            63, 
            player_in_air_step
        )
        
        player_y = jnp.where(
            player_fast_falling,
            jnp.clip(player_y + self.consts.PLAYER_IN_AIR_STEPS[player_in_air_step] + 1, self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), 
            jnp.where(
                player_jumping, 
                player_y - self.consts.PLAYER_IN_AIR_STEPS[player_in_air_step], 
                jnp.where(
                    player_falling, 
                    jnp.clip(player_y + self.consts.PLAYER_IN_AIR_STEPS[player_in_air_step], self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), 
                    player_y
                )
            )
        )

        player_falling = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_falling)
        player_fast_falling = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_fast_falling)

        player_in_air_step = jnp.where(
            player_jumping, 
            player_in_air_step + 1, 
            jnp.where(
                player_y == self.consts.PLAYER_INIT_Y,
                0,
                jnp.where(
                    jnp.logical_or(player_falling, player_fast_falling), 
                    player_in_air_step - 1, 
                    0,
                )
            )
        )


        ###
        ### Player Bullet controller
        ###

        fire_pressed = jnp.where(
            jnp.logical_or(player_bullet_active, player_bullet_step >= 30),
            False,
            fire_pressed
        )

        player_bullet_active = jnp.where( ## 1st frame is creation, 31st is deactivation, 30th is the last active
            player_bullet_step < 30, 
            jnp.where(
                player_bullet_active,
                player_bullet_active,
                jnp.where(
                    fire_pressed,
                    True,
                    False
                )
            ),
            False
        )

        player_bullet_x = jnp.where(
            jnp.logical_and(player_bullet_active, player_bullet_x == -1), 
            player_x + self.consts.PLAYER_WIDTH + 2, ## TODO: +2 or +3?
            jnp.where(
                player_bullet_active,
                player_bullet_x + 2,
                -1
            )
        )

        player_bullet_y = jnp.where(
            jnp.logical_and(player_bullet_active, player_bullet_y == -1), 
            player_y - 4, ## If top-left drawing; TODO: Sometimes spawns at +5?
            jnp.where(
                player_bullet_active,
                player_bullet_y - 2,
                -1
            )
        )

        player_bullet_step = jnp.where(
            player_bullet_active,
            player_bullet_step + 1,
            -1
        )

        player_bullet_active = jnp.where(
            player_bullet_step >= 30,
            False,
            player_bullet_active
        )

        ## Original: the shot disappears once it crosses just above the row of
        ## the highest active object (diamond, helicopter or satellite). With
        ## no objects on screen the step-30 lifetime above is the only limit.
        object_rows = jnp.concatenate([
            jnp.where(state.diamond_active, state.diamond_y, jnp.inf),
            jnp.where(state.helicopter_active, state.helicopter_y, jnp.inf),
            jnp.where(state.satellite_active, state.satellite_y, jnp.inf),
        ])
        has_objects = jnp.any(jnp.isfinite(object_rows))
        despawn_line = jnp.where(
            has_objects,
            jnp.min(object_rows) - self.consts.BULLET_DESPAWN_MARGIN,
            -jnp.inf,
        )
        player_bullet_active = jnp.where(
            player_bullet_y < despawn_line,
            False,
            player_bullet_active
        )

        return state.replace( ## TODO: Use state.replace or output just the values?
            player_x = player_x.astype(jnp.float32),
            player_y = player_y.astype(jnp.float32),
            player_jumping = player_jumping.astype(jnp.bool_),
            player_falling = player_falling.astype(jnp.bool_),
            player_fast_falling = player_fast_falling.astype(jnp.bool_),
            player_in_air_step = player_in_air_step.astype(jnp.int32),
            player_bullet_active = player_bullet_active.astype(jnp.bool_),
            player_bullet_step = player_bullet_step.astype(jnp.int32),
            player_bullet_x = player_bullet_x.astype(jnp.int32),
            player_bullet_y = player_bullet_y.astype(jnp.int32),
        )

    def _update_objects_placeholder(self, state: JamesBondState) -> JamesBondState:
        # Future object lifecycle logic belongs here.

        # === 1. Movement and off-screen cleanup ===
        ## TODO: Checking the speed of objects per frame, will change if it is wrong, if the speed is constant, then move to consts
        ## TODO: If the speed is not float, then use jnp.where to move it
        SPEED_R2L = 0.75 ## Speed right to left, apply for diamond and helicopter
        SPEED_L2R = 1.5 ## Speed left to right, apply for satelitte

        # Diamonds (Scroll left)
        next_diamond_x = state.diamond_x - SPEED_R2L ## TODO: Diamond speed, will change if old speed is wrong
        next_diamond_y = state.diamond_y
        diamond_on_screen = next_diamond_x >= (self.consts.GAME_AREA_MIN_X - self.consts.DIAMOND_WIDTH)
        next_diamond_active = state.diamond_active & diamond_on_screen

        # Enemies
        ## Helicopter enemy (Scroll left)
        next_helicopter_x = state.helicopter_x - SPEED_R2L ## TODO: Helicopter enemy speed, will change if old speed is wrong
        next_helicopter_y = state.helicopter_y
        helicopter_on_screen = next_helicopter_x >= (self.consts.GAME_AREA_MIN_X - self.consts.HELICOPTER_ENEMY_WIDTH)
        next_helicopter_active = state.helicopter_active & helicopter_on_screen
        ## Satellite enemy (Scroll right)
        next_satellite_x = state.satellite_x + SPEED_L2R ## TODO: Satellite enemy speed, will change if old speed is wrong
        next_satellite_y = state.satellite_y
        satellite_on_screen = next_satellite_x <= (self.consts.GAME_AREA_MAX_X)
        next_satellite_active = state.satellite_active & satellite_on_screen

        # === 2. Spawning logic ===
        ## TODO: Before spawining logic, will add the logic of cooldown, so we can't have two same objects spawning at the same time on screen, also helicopter and diamond spawn alternatively
        ## Rule: Alternative spawning only when the entire row is empty
        row_57_empty = (~jnp.any(next_helicopter_active)) & (~jnp.any(next_diamond_active))
        # Check whose turn it is to spawn
        spawn_diamond = row_57_empty & state.spawn_diamond_next
        spawn_helicopter = row_57_empty & (~state.spawn_diamond_next)
        # Flip the turn flag ONLY if a spawn is happening on this frame
        next_spawn_diamond_next = jnp.where(
            row_57_empty,
            ~state.spawn_diamond_next, ## Swap to the other object for next time
            state.spawn_diamond_next ## Keep it the same while they are flying
        )
        # Diamonds
        available_diamond_idx = jnp.argmin(next_diamond_active) ## Get the first inactive diamond index

        # Apply new active status, position coordinates for spawned diamonds
        next_diamond_active = next_diamond_active.at[available_diamond_idx].set(
            jnp.where(spawn_diamond,
                      True, 
                      next_diamond_active[available_diamond_idx])
        )
        next_diamond_x = next_diamond_x.at[available_diamond_idx].set(
            jnp.where(spawn_diamond,
                      self.consts.GAME_AREA_MAX_X,
                      next_diamond_x[available_diamond_idx])
        )
        next_diamond_y = next_diamond_y.at[available_diamond_idx].set(
            jnp.where(spawn_diamond,
                      57.0, ## TODO: Diamond spawn height, will change if the number is wrong
                      next_diamond_y[available_diamond_idx])
        )

        # Enemies
        ## Helicopter enemy
        available_helicopter_idx = jnp.argmin(next_helicopter_active) ## Get the first inactive helicopter index

        # Apply new active status, position coordinates for spawned helicopter enemies
        next_helicopter_active = next_helicopter_active.at[available_helicopter_idx].set(
            jnp.where(spawn_helicopter,
                      True,
                      next_helicopter_active[available_helicopter_idx])
        )
        next_helicopter_x = next_helicopter_x.at[available_helicopter_idx].set(
            jnp.where(spawn_helicopter,
                      self.consts.GAME_AREA_MAX_X,
                      next_helicopter_x[available_helicopter_idx])
        )
        next_helicopter_y = next_helicopter_y.at[available_helicopter_idx].set(
            jnp.where(spawn_helicopter,
                      57.0, ## TODO: Helicopter spawn at the same height as diamond, will change if the number is wrong
                      next_helicopter_y[available_helicopter_idx])
        )
        ## Satellite enemy
        available_satellite_idx = jnp.argmin(next_satellite_active) ## Get the first inactive satellite index
        can_spawn_satellite = ~jnp.any(next_satellite_active) ## Only spawn if the chosen index is inactive
        # Apply new active status, position coordinates for spawned satellite enemies
        next_satellite_active = next_satellite_active.at[available_satellite_idx].set(
            jnp.where(can_spawn_satellite,
                      True,
                      next_satellite_active[available_satellite_idx])
        )
        next_satellite_x = next_satellite_x.at[available_satellite_idx].set(
            jnp.where(can_spawn_satellite,
                      self.consts.GAME_AREA_MIN_X - self.consts.SATELLITE_ENEMY_WIDTH, ## TODO: In game, 
                      next_satellite_x[available_satellite_idx])
        )
        next_satellite_y = next_satellite_y.at[available_satellite_idx].set(
            jnp.where(can_spawn_satellite,
                      75.0, ## TODO: Satellite spawn height, will change if the number is wrong
                      next_satellite_y[available_satellite_idx])
        )

        return state.replace(
            diamond_x=next_diamond_x,
            diamond_y=next_diamond_y,
            diamond_active=next_diamond_active,
            helicopter_x=next_helicopter_x,
            helicopter_y=next_helicopter_y,
            helicopter_active=next_helicopter_active,
            satellite_x=next_satellite_x,
            satellite_y=next_satellite_y,
            satellite_active=next_satellite_active,
            spawn_diamond_next=next_spawn_diamond_next
        )

    def _update_enemy_bombs(self, state: JamesBondState) -> JamesBondState:
        """Move falling helicopter bombs and periodically drop new ones.

        Bombs live in the generic bullet_* arrays. They fall straight down
        while keeping the world-scroll drift, and despawn once they are
        fully below the player's row.
        """

        # === 1. Move active bombs, despawn below the play area ===
        next_bomb_x = state.bullet_x - self.consts.ENEMY_BOMB_DRIFT_SPEED
        next_bomb_y = state.bullet_y + self.consts.ENEMY_BOMB_FALL_SPEED
        bomb_on_screen = next_bomb_y <= (
            self.consts.GAME_AREA_MAX_Y + self.consts.PLAYER_COLLISION_HEIGHT
        )
        next_bomb_active = jnp.logical_and(state.bullet_active, bomb_on_screen)

        # === 2. Periodic drop from the first active helicopter ===
        drop_frame = (state.step_count % self.consts.ENEMY_BOMB_DROP_PERIOD) == 0
        shooter_idx = jnp.argmax(state.helicopter_active)
        has_shooter = jnp.any(state.helicopter_active)
        free_slot = jnp.argmax(jnp.logical_not(next_bomb_active))
        has_free_slot = jnp.any(jnp.logical_not(next_bomb_active))
        do_drop = jnp.logical_and(
            jnp.logical_and(drop_frame, has_shooter), has_free_slot
        )

        drop_x = (
            state.helicopter_x[shooter_idx]
            + self.consts.HELICOPTER_ENEMY_WIDTH / 2
        )
        drop_y = state.helicopter_y[shooter_idx] + self.consts.HELICOPTER_ENEMY_HEIGHT

        next_bomb_x = next_bomb_x.at[free_slot].set(
            jnp.where(do_drop, drop_x, next_bomb_x[free_slot])
        )
        next_bomb_y = next_bomb_y.at[free_slot].set(
            jnp.where(do_drop, drop_y, next_bomb_y[free_slot])
        )
        next_bomb_active = next_bomb_active.at[free_slot].set(
            jnp.where(do_drop, True, next_bomb_active[free_slot])
        )

        return state.replace(
            bullet_x=next_bomb_x,
            bullet_y=next_bomb_y,
            bullet_active=next_bomb_active,
        )

    def _update_firepits(self, state: JamesBondState) -> JamesBondState:
        """Scroll fire pits with the ground and keep one on screen.

        Pits sit on the ground row and scroll right to left with the world
        like the other stage objects. A new pit spawns at the right edge as
        soon as no pit is active.
        """

        next_firepit_x = state.firepit_x - self.consts.FIREPIT_SPEED
        firepit_on_screen = next_firepit_x >= (
            self.consts.GAME_AREA_MIN_X - self.consts.FIREPIT_WIDTH
        )
        next_firepit_active = jnp.logical_and(state.firepit_active, firepit_on_screen)

        can_spawn = jnp.logical_not(jnp.any(next_firepit_active))
        available_idx = jnp.argmin(next_firepit_active)
        next_firepit_active = next_firepit_active.at[available_idx].set(
            jnp.where(can_spawn, True, next_firepit_active[available_idx])
        )
        next_firepit_x = next_firepit_x.at[available_idx].set(
            jnp.where(
                can_spawn,
                float(self.consts.GAME_AREA_MAX_X),
                next_firepit_x[available_idx],
            )
        )

        return state.replace(
            firepit_x=next_firepit_x,
            firepit_active=next_firepit_active,
        )

    def _check_collisions_placeholder(self, state: JamesBondState) -> JamesBondState:
        # Future diamond, enemy, bullet, and life collision logic belongs here.
        state = state.replace(
            collision_happened=jnp.array(False, dtype=jnp.bool_),
            collected_diamond=jnp.array(False, dtype=jnp.bool_),
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
        )
        return self._resolve_collisions(state)

    def _calculate_reward_placeholder(
        self, previous_state: JamesBondState, state: JamesBondState
    ) -> chex.Array:
        if False:
            del previous_state, state
            return jnp.array(self.consts.REWARD_STEP, dtype=jnp.float32)
        return self._get_reward(previous_state, state)

    def _is_done(self, state: JamesBondState) -> chex.Array:
        return self._get_done(state)

    def _resolve_collisions(self, state: JamesBondState) -> JamesBondState:
        """Run all collision systems after movement and object updates."""

        state = self._resolve_player_bullet_collisions(state)
        state = self._resolve_bomb_player_collisions(state)
        state = self._resolve_firepit_player_collisions(state)
        return self._resolve_player_hazard_collisions(state)

    def _resolve_firepit_player_collisions(self, state: JamesBondState) -> JamesBondState:
        """Apply one life of damage when the player drives into a fire pit.

        Only ground contact is deadly: a jumping player clears the pit. The
        player bullet and helicopter bombs pass over pits without responding,
        matching the original game, so no projectile checks happen here.
        """

        ## Center the smaller collision box inside the wider pit sprite so an
        ## edge tap is survivable but driving into the middle is fatal.
        firepit_inset = (self.consts.FIREPIT_WIDTH - self.consts.FIREPIT_COLLISION_WIDTH) / 2
        firepit_cx = state.firepit_x + firepit_inset
        firepit_y = jnp.full_like(state.firepit_x, self.consts.FIREPIT_Y)
        overlaps = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            firepit_cx,
            firepit_y,
            self.consts.FIREPIT_COLLISION_WIDTH,
            self.consts.FIREPIT_COLLISION_HEIGHT,
        )
        on_ground = state.player_y >= self.consts.PLAYER_INIT_Y
        firepit_collision = jnp.logical_and(
            on_ground,
            jnp.any(jnp.logical_and(state.firepit_active, overlaps)),
        )
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(firepit_collision, can_take_damage)

        return state.replace(
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            reward_delta=state.reward_delta
            + took_damage.astype(jnp.float32) * self.consts.REWARD_LOST_LIFE,
            collision_happened=jnp.logical_or(
                state.collision_happened, firepit_collision
            ),
        )

    def _resolve_bomb_player_collisions(self, state: JamesBondState) -> JamesBondState:
        """Apply one life of damage when a helicopter bomb hits the player.

        The bomb always detonates (deactivates) on contact; the life is only
        lost when the hit cooldown has expired, mirroring the hazard rule.
        """

        overlaps = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            state.bullet_x,
            state.bullet_y,
            self.consts.BULLET_COLLISION_WIDTH,
            self.consts.BULLET_COLLISION_HEIGHT,
        )
        bomb_hits = jnp.logical_and(state.bullet_active, overlaps)
        hit_any = jnp.any(bomb_hits)
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(hit_any, can_take_damage)

        return state.replace(
            bullet_active=jnp.logical_and(
                state.bullet_active, jnp.logical_not(bomb_hits)
            ),
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            reward_delta=state.reward_delta
            + took_damage.astype(jnp.float32) * self.consts.REWARD_LOST_LIFE,
            collision_happened=jnp.logical_or(state.collision_happened, hit_any),
        )

    def _resolve_player_hazard_collisions(self, state: JamesBondState) -> JamesBondState:
        """Apply one life of damage when the player touches an active enemy."""

        helicopter_overlaps = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            state.helicopter_x,
            state.helicopter_y,
            self.consts.HELICOPTER_COLLISION_WIDTH,
            self.consts.HELICOPTER_COLLISION_HEIGHT,
        )
        satellite_overlaps = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            state.satellite_x,
            state.satellite_y,
            self.consts.SATELLITE_COLLISION_WIDTH,
            self.consts.SATELLITE_COLLISION_HEIGHT,
        )

        hazard_collision = jnp.logical_or(
            jnp.any(jnp.logical_and(state.helicopter_active, helicopter_overlaps)),
            jnp.any(jnp.logical_and(state.satellite_active, satellite_overlaps)),
        )
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(hazard_collision, can_take_damage)

        return state.replace(
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            reward_delta=state.reward_delta
            + took_damage.astype(jnp.float32) * self.consts.REWARD_LOST_LIFE,
            collision_happened=jnp.logical_or(
                state.collision_happened, hazard_collision
            ),
            hit_enemy=jnp.logical_or(state.hit_enemy, hazard_collision),
        )
    
    def collectible_collisions_logic(self, state: JamesBondState) -> JamesBondState:
        """Collect active diamonds that overlap the player's bullet collision box."""

        overlaps = _aabb_overlap(
            state.player_bullet_x,
            state.player_bullet_y,
            self.consts.BULLET_COLLISION_WIDTH,
            self.consts.BULLET_COLLISION_HEIGHT,
            state.diamond_x,
            state.diamond_y,
            self.consts.DIAMOND_COLLISION_WIDTH,
            self.consts.DIAMOND_COLLISION_HEIGHT,
        )

        ## Gate per diamond slot: only active diamonds can be hit, and only
        ## while the bullet itself is active.
        collected = jnp.logical_and(
            jnp.logical_and(state.diamond_active, state.player_bullet_active),
            overlaps,
        )
        collected_any = jnp.any(collected)
        collected_count = jnp.sum(collected.astype(jnp.int32))

        player_bullet_active = jnp.where(
            jnp.logical_and(
                state.player_bullet_active, 
                jnp.logical_not(jnp.any(collected))
            ),
            state.player_bullet_active,
            False
        )

        player_bullet_x = jnp.where(
            player_bullet_active,
            state.player_bullet_x,
            -1
        )

        player_bullet_y = jnp.where(
            player_bullet_active,
            state.player_bullet_y,
            -1
        )

        player_bullet_step = jnp.where(
            player_bullet_active,
            state.player_bullet_step,
            -1
        )

        return state.replace(
            diamond_active = jnp.logical_and(
                state.diamond_active, ~collected
            ),
            player_bullet_active=player_bullet_active,
            player_bullet_step=player_bullet_step,
            player_bullet_x=player_bullet_x,
            player_bullet_y=player_bullet_y,
            score=state.score + collected_count * self.consts.SCORE_DIAMOND,
            reward_delta=state.reward_delta
            + collected_count.astype(jnp.float32) * self.consts.REWARD_DIAMOND,
            collision_happened=jnp.logical_or(
                state.collision_happened, collected_any
            ),
            collected_diamond=jnp.logical_or(state.collected_diamond, collected_any),
        )

    def _resolve_player_bullet_collisions(self, state: JamesBondState) -> JamesBondState:
        ## In the original game the bullet passes straight through helicopters
        ## and satellites without any visible response: the diamond is the
        ## only object the player bullet collides with.
        check_collisions = jnp.where(
            state.player_bullet_active,
            True,
            False
        )

        return lax.cond(
            check_collisions,
            self.collectible_collisions_logic,
            lambda s: s,
            state
        )

    def _get_reward(
        self, previous_state: JamesBondState, state: JamesBondState
    ) -> chex.Array:
        """Return the step reward until scoring events are implemented."""

        del previous_state
        return jnp.array(self.consts.REWARD_STEP, dtype=jnp.float32) + state.reward_delta

    def _get_done(self, state: JamesBondState) -> chex.Array:
        return jnp.logical_or(
            state.lives <= 0,
            state.step_count >= self.consts.MAX_EPISODE_STEPS,
        )


class JamesBondRenderer(JAXGameRenderer):
    """Procedural rectangle renderer for the skeleton environment."""

    def __init__(
        self,
        consts: JamesBondConstants = None,
        config: render_utils.RendererConfig = None,
    ):
        self.consts = consts or JamesBondConstants()
        if config is None:
            config = render_utils.RendererConfig(
                game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
                channels=3,
                downscale=None,
            )
        super().__init__(self.consts, config)
        self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        self.PALETTE = jnp.array(
            [
                self.consts.BACKGROUND_COLOR,
                self.consts.PLAY_AREA_COLOR,
                self.consts.PLAYER_COLOR,
                self.consts.DIAMOND_COLOR,
                self.consts.ENEMY_COLOR,
                self.consts.BULLET_COLOR,
                self.consts.FIREPIT_COLOR,
            ],
            dtype=jnp.uint8,
        )
        self.BACKGROUND_ID = 0
        self.PLAY_AREA_ID = 1
        self.PLAYER_ID = 2
        self.DIAMOND_ID = 3
        self.ENEMY_ID = 4
        self.BULLET_ID = 5
        self.FIREPIT_ID = 6
        self.BACKGROUND = jnp.full(
            (self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
            self.BACKGROUND_ID,
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: JamesBondState) -> jnp.ndarray:
        """Render a simple background, inactive object slots, and player box."""

        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self._render_background(raster)
        raster = self._render_objects(raster, state)
        raster = self._render_player(raster, state)
        return self.jr.render_from_palette(raster, self.PALETTE)

    def _render_background(self, raster: jnp.ndarray) -> jnp.ndarray:
        """Draw the placeholder play area."""

        position = jnp.array(
            [[self.consts.GAME_AREA_MIN_X, self.consts.GAME_AREA_MIN_Y]],
            dtype=jnp.int32,
        )
        size = jnp.array(
            [
                [
                    self.consts.GAME_AREA_MAX_X - self.consts.GAME_AREA_MIN_X,
                    self.consts.GAME_AREA_MAX_Y - self.consts.GAME_AREA_MIN_Y,
                ]
            ],
            dtype=jnp.int32,
        )
        return self.jr.draw_rects(raster, position, size, self.PLAY_AREA_ID)

    def _render_player(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the player placeholder rectangle."""

        position = jnp.stack(
            [
                jnp.round(state.player_x).astype(jnp.int32),
                jnp.round(state.player_y).astype(jnp.int32),
            ]
        )[None, :] ## TODO: Why this definiton and not just 2 arrays?
        size = jnp.array(
            [[self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT]], dtype=jnp.int32
        )
        return self.jr.draw_rects(raster, position, size, self.PLAYER_ID)

    def _render_objects(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw any active placeholder object rectangles."""

        raster = self._render_object_group(
            raster,
            state.diamond_x,
            state.diamond_y,
            state.diamond_active,
            self.consts.DIAMOND_WIDTH,
            self.consts.DIAMOND_HEIGHT,
            self.DIAMOND_ID,
        )
        raster = self._render_object_group(
            raster,
            state.enemy_x,
            state.enemy_y,
            state.enemy_active,
            self.consts.ENEMY_WIDTH,
            self.consts.ENEMY_HEIGHT,
            self.ENEMY_ID,
        )
        ## raster = self._render_object_group(
        ##     raster,
        ##     state.helicopter_x,
        ##     state.helicopter_y,
        ##     state.helicopter_active,
        ##     self.consts.HELICOPTER_ENEMY_WIDTH,
        ##     self.consts.HELICOPTER_ENEMY_HEIGHT,
        ##     self.ENEMY_ID,
        ## )
        ## raster = self._render_object_group(
        ##     raster,
        ##     state.satellite_x,
        ##     state.satellite_y,
        ##     state.satellite_active,
        ##     self.consts.SATELLITE_ENEMY_WIDTH,
        ##     self.consts.SATELLITE_ENEMY_HEIGHT,
        ##     self.ENEMY_ID,
        ## )
        raster = self._render_object_group(
            raster,
            state.firepit_x,
            jnp.full_like(state.firepit_x, self.consts.FIREPIT_Y),
            state.firepit_active,
            self.consts.FIREPIT_WIDTH,
            self.consts.FIREPIT_HEIGHT,
            self.FIREPIT_ID,
        )
        return self._render_object_group(
            raster,
            state.bullet_x,
            state.bullet_y,
            state.bullet_active,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            self.BULLET_ID,
        )

    def _render_object_group(
        self,
        raster: jnp.ndarray,
        x: chex.Array,
        y: chex.Array,
        active: chex.Array,
        width: int,
        height: int,
        color_id: int,
    ) -> jnp.ndarray:
        """Draw a fixed-size object group, hiding inactive slots at x=-1."""

        draw_x = jnp.where(active, jnp.round(x).astype(jnp.int32), -1)
        draw_y = jnp.round(y).astype(jnp.int32)
        positions = jnp.stack([draw_x, draw_y], axis=1)
        sizes = jnp.stack(
            [
                jnp.full(x.shape, width, dtype=jnp.int32),
                jnp.full(y.shape, height, dtype=jnp.int32),
            ],
            axis=1,
        )
        return self.jr.draw_rects(raster, positions, sizes, color_id)
