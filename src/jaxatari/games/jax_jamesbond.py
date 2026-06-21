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
    GAME_AREA_MIN_X: int = struct.field(pytree_node=False, default=8) ## Playable Area: 5 (Coordinate system starting with 1) -- Shown in /jb_sprites/game_area_min_x.npy
    GAME_AREA_MAX_X: int = struct.field(pytree_node=False, default=152) ## Playable Area: 81 (Coordinate system starting with 1)
    GAME_AREA_MIN_Y: int = struct.field(pytree_node=False, default=28) ## Playable Area: 123 (Top-left coordinate system); 87 (Bottom-right co-sys)
    GAME_AREA_MAX_Y: int = struct.field(pytree_node=False, default=196)

    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=10)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=8)
    PLAYER_INIT_X: int = struct.field(pytree_node=False, default=32)
    PLAYER_INIT_Y: int = struct.field(pytree_node=False, default=160)
    PLAYER_SPEED: float = struct.field(pytree_node=False, default=2.0)
    GRAVITY: float = struct.field(pytree_node=False, default=0.0)
    JUMP_VELOCITY: float = struct.field(pytree_node=False, default=0.0)

    # Fixed capacities keep object state JAX-friendly for future lifecycle logic.
    MAX_LIVES: int = struct.field(pytree_node=False, default=3)
    MAX_DIAMONDS: int = struct.field(pytree_node=False, default=8)
    MAX_ENEMIES: int = struct.field(pytree_node=False, default=8)
    MAX_BULLETS: int = struct.field(pytree_node=False, default=4)
    MAX_EPISODE_STEPS: int = struct.field(pytree_node=False, default=5000)

    # Placeholder render/collision sizes for object-centric observations.
    DIAMOND_WIDTH: int = struct.field(pytree_node=False, default=4)
    DIAMOND_HEIGHT: int = struct.field(pytree_node=False, default=4)
    ENEMY_WIDTH: int = struct.field(pytree_node=False, default=10)
    ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=8)
    BULLET_WIDTH: int = struct.field(pytree_node=False, default=3)
    BULLET_HEIGHT: int = struct.field(pytree_node=False, default=2)

    # Collision boxes are separate from render sizes for future tuning.
    PLAYER_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=10)
    PLAYER_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=8)
    DIAMOND_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=4)
    DIAMOND_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=4)
    ENEMY_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=10)
    ENEMY_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=8)
    BULLET_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=3)
    BULLET_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=2)

    SCORE_DIAMOND: int = struct.field(pytree_node=False, default=100)
    SCORE_ENEMY: int = struct.field(pytree_node=False, default=250)
    HIT_COOLDOWN_STEPS: int = struct.field(pytree_node=False, default=30)

    # Reward constants are named now so scoring work can reuse the contract.
    REWARD_STEP: float = struct.field(pytree_node=False, default=0.0)
    REWARD_DIAMOND: float = struct.field(pytree_node=False, default=1.0)
    REWARD_ENEMY: float = struct.field(pytree_node=False, default=2.0)
    REWARD_HIT_ENEMY: float = struct.field(pytree_node=False, default=-1.0)
    REWARD_LOST_LIFE: float = struct.field(pytree_node=False, default=-1.0)

    ACTION_MEANINGS: Tuple[str, ...] = struct.field(
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

    # Procedural colors keep the skeleton renderable before final sprites land.
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


@struct.dataclass
class JamesBondState:
    """Full internal state with fixed-size object arrays and active masks."""

    player_x: chex.Array
    player_y: chex.Array
    player_vx: chex.Array
    player_vy: chex.Array
    player_jumping: chex.Array
    player_falling: chex.Array
    player_fast_falling: chex.Array
    player_in_air_step: chex.Array
    ## player_direction: chex.Array -- Maybe not needed
    player_bullet_active: chex.Array
    player_bullet_step: chex.Array
    player_bullet_x: chex.Array
    player_bullet_y: chex.Array
    ## player_bullet_vx: chex.Array
    ## player_bullet_vx: chex.Array
    lives: chex.Array
    score: chex.Array
    step_count: chex.Array
    level_progress: chex.Array
    diamond_x: chex.Array
    diamond_y: chex.Array
    diamond_active: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_active: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_vx: chex.Array
    bullet_active: chex.Array
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
    bullets: ObjectObservation
    player_velocity: jnp.ndarray
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
            player_x=jnp.array(self.consts.PLAYER_INIT_X, dtype=jnp.float32), ## TODO: float32 or int?
            player_y=jnp.array(self.consts.PLAYER_INIT_Y, dtype=jnp.float32),
            player_vx=jnp.array(0.0, dtype=jnp.float32),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            player_jumping=jnp.array(False, dtype=jnp.bool_),
            player_falling=jnp.array(False, dtype=jnp.bool_),
            player_fast_falling=jnp.array(False, dtype=jnp.bool_),
            player_in_air_step=jnp.array(0, dtype=jnp.int32),
            ## player_direction=jnp.array(1, dtype=jnp.int32), ## TODO: Is there a need for this?
            player_bullet_active=jnp.array(False, dtype=jnp.bool_),
            player_bullet_step=jnp.array(-1, dtype=jnp.int32),
            player_bullet_x=jnp.array(-1, dtype=jnp.int32),
            player_bullet_y=jnp.array(-1, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            level_progress=jnp.array(0, dtype=jnp.int32),
            diamond_x=jnp.zeros((self.consts.MAX_DIAMONDS,), dtype=jnp.float32),
            diamond_y=jnp.zeros((self.consts.MAX_DIAMONDS,), dtype=jnp.float32),
            diamond_active=jnp.zeros((self.consts.MAX_DIAMONDS,), dtype=jnp.bool_),
            enemy_x=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.float32),
            enemy_y=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.float32),
            enemy_active=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.bool_),
            bullet_x=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.float32),
            bullet_y=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.float32),
            bullet_vx=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.float32),
            bullet_active=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.bool_),
            collision_happened=jnp.array(False, dtype=jnp.bool_),
            collected_diamond=jnp.array(False, dtype=jnp.bool_),
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
            fired_bullet=jnp.array(False, dtype=jnp.bool_),
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

        # Clear one-frame event flags before placeholder systems update them.
        state = state.replace(
            step_count=state.step_count + 1,
            collision_happened=jnp.array(False, dtype=jnp.bool_),
            collected_diamond=jnp.array(False, dtype=jnp.bool_),
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
            fired_bullet=atari_action == Action.FIRE,
        )
        state = self._step_player(state, atari_action)
        state = self._update_objects_placeholder(state)
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

    def observation_space(self) -> spaces.Dict:
        screen_size = (self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH)
        return spaces.Dict(
            {
                "player": spaces.get_object_space(n=None, screen_size=screen_size),
                "diamonds": spaces.get_object_space(
                    n=self.consts.MAX_DIAMONDS, screen_size=screen_size
                ),
                "enemies": spaces.get_object_space(
                    n=self.consts.MAX_ENEMIES, screen_size=screen_size
                ),
                "bullets": spaces.get_object_space(
                    n=self.consts.MAX_BULLETS, screen_size=screen_size
                ),
                "player_velocity": spaces.Box(
                    low=jnp.array([-10.0, -20.0], dtype=jnp.float32),
                    high=jnp.array([10.0, 20.0], dtype=jnp.float32),
                    shape=(2,),
                    dtype=jnp.float32,
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
            ## orientation=jnp.where(state.player_direction < 0, 270.0, 90.0),
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
        bullets = self._object_group_observation(
            state.bullet_x,
            state.bullet_y,
            state.bullet_active,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            orientation=jnp.where(state.bullet_vx < 0, 270.0, 90.0), ## TODO: Isn't 90/270 Top/Bottom, which coordinate system are we using?
        )
        return JamesBondObservation(
            player=player,
            diamonds=diamonds,
            enemies=enemies,
            bullets=bullets,
            player_velocity=jnp.stack([state.player_vx, state.player_vy]).astype( ## TODO: Does observation need this or can we remove it?
                jnp.float32
            ),
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

        return ObjectObservation.create(
            x=x,
            y=y,
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
        left = atari_action == Action.LEFT
        right = atari_action == Action.RIGHT
        up = atari_action == Action.UP
        down = atari_action == Action.DOWN

        player_vx = (
            right.astype(jnp.float32) - left.astype(jnp.float32)
        ) * self.consts.PLAYER_SPEED
        player_vy = (
            down.astype(jnp.float32) - up.astype(jnp.float32)
        ) * self.consts.PLAYER_SPEED

        player_x = jnp.clip(
            state.player_x + player_vx,
            self.consts.GAME_AREA_MIN_X,
            self.consts.GAME_AREA_MAX_X - self.consts.PLAYER_WIDTH,
        )
        player_y = jnp.clip(
            state.player_y + player_vy,
            self.consts.GAME_AREA_MIN_Y,
            self.consts.GAME_AREA_MAX_Y - self.consts.PLAYER_HEIGHT,
        )
        player_direction = jnp.where(
            left, -1, jnp.where(right, 1, state.player_direction)
        ).astype(jnp.int32)

        return state.replace( ## TODO: Use state.replace or output just the values?
            player_x = player_x.astype(jnp.float32),
            player_y = player_y.astype(jnp.float32),
            player_jumping = player_jumping.astype(jnp.bool_),
            player_falling = player_falling.astype(jnp.bool_),
            player_fast_falling = player_fast_falling.astype(jnp.bool_),
            player_in_air_step = player_in_air_step.astype(jnp.int32),
            player_bullet_active = player_bullet_active.astype(jnp.bool_),
            player_bullet_step = player_bullet_active.astype(jnp.int32),
            player_bullet_x = player_bullet_active.astype(jnp.int32),
            player_bullet_y = player_bullet_active.astype(jnp.int32),
            ## player_vx = player_vx.astype(jnp.float32), ## TODO: Should we give the speed as well, or just use it for calc?
            ## player_vy = player_vy.astype(jnp.float32),
        )

    def _update_objects_placeholder(self, state: JamesBondState) -> JamesBondState:
        # Future object lifecycle logic belongs here.
        return state

    def _check_collisions_placeholder(self, state: JamesBondState) -> JamesBondState:
        # Future diamond, enemy, bullet, and life collision logic belongs here.
        return state.replace(
            collision_happened=jnp.array(False, dtype=jnp.bool_),
            collected_diamond=jnp.array(False, dtype=jnp.bool_),
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
        )

    def _calculate_reward_placeholder(
        self, previous_state: JamesBondState, state: JamesBondState
    ) -> chex.Array:
        """Return the step reward until scoring events are implemented."""

        del previous_state, state
        return jnp.array(self.consts.REWARD_STEP, dtype=jnp.float32)

    def _is_done(self, state: JamesBondState) -> chex.Array:
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
            ],
            dtype=jnp.uint8,
        )
        self.BACKGROUND_ID = 0
        self.PLAY_AREA_ID = 1
        self.PLAYER_ID = 2
        self.DIAMOND_ID = 3
        self.ENEMY_ID = 4
        self.BULLET_ID = 5
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
