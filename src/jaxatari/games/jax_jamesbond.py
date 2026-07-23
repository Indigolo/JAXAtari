"""Runnable JamesBond skeleton environment.

This file intentionally defines only the shared environment contract and minimal
placeholder behavior. Gameplay systems such as object spawning, collisions,
scoring, lives, and sprite-accurate rendering are left for follow-up work.
"""
import os

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

## Sprites live in the repo (src/jaxatari/jb_sprites), not in the downloaded
## sprite pack, so the renderer must load them from here.
## NOTE: background.npy, bullet.npy and score_6..9.npy are placeholder
## sprites so the environment can run; the sprite task owner should replace
## them with real extractions.
JB_SPRITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jb_sprites")


def get_default_asset_config() -> tuple:
        asset_config = [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'}, ## TODO: Placeholder, extract the real background sprite
            {'name': 'ground', 'type': 'single', 'file': 'ground_unkempt.npy'}, ## TODO: Ground and Background the same sprite?
            {
                'name': 'car', 'type': 'group', 
                'files': ['car.npy', 'car_dead_1.npy', 'car_dead_2.npy'] ## TODO: maybe delete car_dead_3 sprite
            },
            {'name': 'satellite', 'type': 'single', 'file': 'satellite.npy'},
            {
                'name': 'helicopter', 'type': 'group',
                'files': ['helicopter_1.npy', 'helicopter_2.npy']
            },
            {
                'name': 'helicopter_melee', 'type': 'group',
                'files': ['helicopter_shot_1.npy', 'helicopter_shot_2.npy'] ## TODO: Add others
            },
            {
                'name': 'pit', 'type': 'group',
                'files': ['fire_pit_1.npy', 'fire_pit_2.npy']
            },
            {
                'name': 'diamond', 'type': 'group',
                'files': ['diamond_1.npy', 'diamond_2.npy']
            },
            {
                'name': 'stars', 'type': 'group',
                'files': ['stars_1.npy', 'stars_2.npy']
             },

            {'name': 'life', 'type': 'single', 'file': 'car_life.npy'},

            {
                'name': 'score_digits', 'type': 'digits',
                'pattern': 'score_{}.npy' ## TODO: 6-9 are placeholders, extract the real digits
            },
            {
                'name': 'bullet', 'type': 'single', ## TODO: Placeholder, extract the real bullet sprite. Comment: What? no it's not, it's the real bullet sprite
                'file': 'bullet.npy'
            }
        ]
        return asset_config

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
    PLAYER_INIT_Y: int = struct.field(pytree_node=False, default=119) ## 120 if starting with 1 (Not 122?)
    PLAYER_IN_AIR_STEPS = jnp.array([ ## For the gravity feel of jumps. Each jump is 71 frames, 72nd frame is the start of the fall
        0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ## TODO: Remove first zero?
        0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 
        0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1 ## Water matrix: 111010101110011010101010010010100100010000100000001000000000-1; Nearly the same as air. Also, can jump higher and faster from water to air
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
    ## Fire pit sprite size as loaded from fire_pit_*.npy.
    ## TODO: The npy looks stored transposed (48x16); revisit with the sprite team.
    PIT_WIDTH: int = struct.field(pytree_node=False, default=16)
    PIT_HEIGHT: int = struct.field(pytree_node=False, default=48)

    # Collision boxes are kept a little smaller than the real sprite sizes so
    # near-misses do not register, matching how the original game feels.
    PLAYER_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=6)   ## < PLAYER_WIDTH 8
    PLAYER_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=3)  ## < PLAYER_HEIGHT 4
    DIAMOND_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=4)  ## < DIAMOND_WIDTH 7
    DIAMOND_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=4) ## < DIAMOND_HEIGHT 13
    HELICOPTER_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=6)  ## < HELICOPTER_ENEMY_WIDTH 8
    HELICOPTER_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=4) ## < HELICOPTER_ENEMY_HEIGHT 6
    SATELLITE_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=6)   ## < SATELLITE_ENEMY_WIDTH 8
    SATELLITE_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=12) ## < SATELLITE_ENEMY_HEIGHT 14
    PIT_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=12)  ## < PIT_WIDTH 16, edge taps survivable

    SCORE_DIAMOND: int = struct.field(pytree_node=False, default=100)
    SCORE_ENEMY: int = struct.field(pytree_node=False, default=250)
    HIT_COOLDOWN_STEPS: int = struct.field(pytree_node=False, default=60)

    ## Helicopter bombs (enemy fire), stored in the generic bullet_* arrays.
    ENEMY_BOMB_DROP_PERIOD: int = struct.field(pytree_node=False, default=60) ## Frames between drops
    ENEMY_BOMB_FALL_SPEED: float = struct.field(pytree_node=False, default=1.0)
    ENEMY_BOMB_DRIFT_SPEED: float = struct.field(pytree_node=False, default=0.25) ## Match the ground scroll speed

    REWARD_STEP: float = struct.field(pytree_node=False, default=0.0)
    REWARD_DIAMOND: float = struct.field(pytree_node=False, default=1.0)
    REWARD_ENEMY: float = struct.field(pytree_node=False, default=2.0)
    REWARD_HIT_ENEMY: float = struct.field(pytree_node=False, default=-1.0)
    REWARD_LOST_LIFE: float = struct.field(pytree_node=False, default=-1.0)

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=get_default_asset_config)

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
    player_bullet_active: chex.Array ## TODO: Maybe think about every bullet as bullet with different and direction (and speed?)?
    player_bullet_step: chex.Array
    player_bullet_x: chex.Array
    player_bullet_y: chex.Array
    bullet_vx: chex.Array
    lives: chex.Array
    score: chex.Array
    step_count: chex.Array
    stage: chex.Array
    hit_cooldown: chex.Array ## TODO: What for?
    diamond_x: chex.Array
    diamond_y: chex.Array
    diamond_active: chex.Array
    spawn_diamond_next: chex.Array
    pit_x: chex.Array
    pit_y: chex.Array
    pit_active: chex.Array
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
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_active: chex.Array
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
    stage: jnp.ndarray


@struct.dataclass
class JamesBondInfo:
    """Debug/event info for smoke tests and future gameplay systems."""

    collected_diamond: jnp.ndarray
    hit_enemy: jnp.ndarray
    fired_bullet: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    stage: jnp.ndarray
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
            player_x=jnp.array(self.consts.PLAYER_INIT_X, dtype=jnp.int32), ## TODO: Change all float32 pos-s to int32
            player_y=jnp.array(self.consts.PLAYER_INIT_Y, dtype=jnp.int32),
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
            stage=jnp.array(0, dtype=jnp.int32),
            hit_cooldown=jnp.array(0, dtype=jnp.int32),
            diamond_x=jnp.array(0, dtype=jnp.int32),
            diamond_y=jnp.array(0, dtype=jnp.int32),
            diamond_active=jnp.array(0, dtype=jnp.bool_),
            pit_x=jnp.array(0, dtype=jnp.int32),
            pit_y=jnp.array(0, dtype=jnp.int32),
            pit_active=jnp.array(False, dtype=jnp.bool_),
            spawn_diamond_next=jnp.array(True, dtype=jnp.bool_), ## TODO: In state requires this, but is this array or zero-dimensional?
            ## TODO: Change / Remove after observation and collision enemy variables have been changed; or else will cause fail tests
            enemy_x=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.float32),
            enemy_y=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.float32),
            enemy_active=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.bool_),
            ## TODO: Here using helicopter and satellite instead of enemy
            helicopter_x=jnp.array(0, dtype=jnp.int32),
            helicopter_y=jnp.array(0, dtype=jnp.int32),
            helicopter_active=jnp.array(0, dtype=jnp.bool_),
            satellite_x=jnp.array(0, dtype=jnp.int32),
            satellite_y=jnp.array(0, dtype=jnp.int32),
            satellite_active=jnp.array(0, dtype=jnp.bool_),
            bullet_x=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.int32),
            bullet_y=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.int32),
            bullet_active=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.bool_),
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
            collected_diamond=jnp.array(False, dtype=jnp.bool_),
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
            hit_cooldown=jnp.maximum(state.hit_cooldown - 1, 0),
            fired_bullet=atari_action == Action.FIRE,
        )
        state = self._step_player(state, atari_action)
        state = self._update_objects_placeholder(state)
        state = self._update_enemy_bombs(state)
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
                "stage": spaces.Box(
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
            orientation=jnp.array(0.0, dtype=jnp.float32),
            state=jnp.array(0, dtype=jnp.int32),
            visual_id=jnp.array(0, dtype=jnp.int32),
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
            ## per bullet slot or the observation loses features vs the space.
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
            stage=state.stage,
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

        if orientation is None: ## TODO: Maybe remove if not needed
            orientation = jnp.zeros_like(x, dtype=jnp.float32)

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
            collected_diamond=state.collected_diamond,
            hit_enemy=state.hit_enemy,
            fired_bullet=state.fired_bullet,
            score=state.score,
            lives=state.lives,
            stage=state.stage,
            step_count=state.step_count,
        )

    def _decode_action(self, action: chex.Array) -> chex.Array:
        """Translate compact action-space indices to JAXAtariAction values."""

        return jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

    def step_player_stage_one(
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
    
    def step_player_stage_two(
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
    
    def step_player_stage_three_placeholder(
        self, state: JamesBondState, atari_action: chex.Array
    ):
        return state
        
    def _step_player(
        self, state: JamesBondState, atari_action: chex.Array
    ) -> JamesBondState:

        return jax.lax.switch( ## First stage's index is 0
            state.stage,
            [
                self.step_player_stage_one,
                self.step_player_stage_two,
                self.step_player_stage_three_placeholder,
            ],
            (state, atari_action)
        )

    def _update_objects_placeholder(self, state: JamesBondState) -> JamesBondState: ## TODO: Implement fire pit
        # Future object lifecycle logic belongs here.

        # === 1. Movement and off-screen cleanup ===

        # Diamonds (Scroll left)
        ## Diamond speed, here is 0.5 pixels per frame
        next_diamond_x = jnp.where(
            state.step_count % 2 == 0,
            state.diamond_x - 1,
            state.diamond_x
        )
        next_diamond_y = state.diamond_y
        diamond_on_screen = next_diamond_x >= (self.consts.GAME_AREA_MIN_X - self.consts.DIAMOND_WIDTH)
        next_diamond_active = state.diamond_active & diamond_on_screen

        # Enemies
        ## Helicopter enemy (Scroll left)
        ## 1. Determine which speed zone the helicopter is currently in
        in_slow_mode = (state.helicopter_x <= 96) & (state.helicopter_x > 63)
        ## 2. The speed of helicopter is 0.6 pixels per frame in normal mode, and 0.375 pixels per frame in slow mode
        move_normal = (state.step_count % 5 == 0) | (state.step_count % 5 == 2) | (state.step_count % 5 == 4)
        move_slow = (state.step_count % 8 == 1) | (state.step_count % 8 == 4) | (state.step_count % 8 == 7)
        move_helicopter = jnp.where(
            in_slow_mode,
            move_slow,
            move_normal
        )
        next_helicopter_x = jnp.where(
            move_helicopter,
            state.helicopter_x - 1,
            state.helicopter_x
        )
        next_helicopter_y = state.helicopter_y
        helicopter_on_screen = next_helicopter_x >= (self.consts.GAME_AREA_MIN_X - self.consts.HELICOPTER_ENEMY_WIDTH)
        next_helicopter_active = state.helicopter_active & helicopter_on_screen
        
        ## Satellite enemy (Scroll right)
        ## Satellite enemy speed, here is 0.8 pixels per frame
        next_satellite_x = jnp.where(
            state.step_count % 5 != 4,
            state.satellite_x + 1,
            state.satellite_x
        )
        next_satellite_y = state.satellite_y
        satellite_on_screen = next_satellite_x <= (self.consts.GAME_AREA_MAX_X)
        next_satellite_active = state.satellite_active & satellite_on_screen

        # Fire pit (Scroll left)
        ## Fire pit speed, here is 0.25 pixels per frame, as far as i checked, fire pit will move if step count % 4 == 3
        next_pit_x = jnp.where(
            state.step_count % 4 == 3,
            state.pit_x -1,
            state.pit_x
        )
        next_pit_y = state.pit_y
        pit_on_screen = next_pit_x >= (self.consts.GAME_AREA_MIN_X - self.consts.PIT_WIDTH)
        next_pit_active = state.pit_active & pit_on_screen

        # === 2. Spawning logic ===
        ## TODO: Before spawining logic, will add the logic of cooldown, so we can't have two same objects spawning at the same time on screen, also helicopter and diamond spawn alternatively
        ## Rule: Alternative spawning only when the entire row is empty
        row_57_empty = (~jnp.any(next_helicopter_active)) & (~jnp.any(next_diamond_active))
        # Check whose turn it is to spawn
        spawn_diamond = row_57_empty & state.spawn_diamond_next
        spawn_helicopter = row_57_empty & (~state.spawn_diamond_next)
        can_spawn_satellite = ~next_satellite_active ## Only spawn when the previous satellite left the screen
        can_spawn_pit = ~next_pit_active ## Only spawn when the previous pit left the screen
        # Flip the turn flag ONLY if a spawn is happening on this frame
        next_spawn_diamond_next = jnp.where(
            row_57_empty,
            ~state.spawn_diamond_next, ## Swap to the other object for next time
            state.spawn_diamond_next ## Keep it the same while they are flying
        )
        # Diamonds
        # Apply new active status, position coordinates for spawned diamonds
        next_diamond_active = next_diamond_active | spawn_diamond
        next_diamond_x = jnp.where(
            spawn_diamond,
            self.consts.GAME_AREA_MAX_X,
            next_diamond_x
        )
        next_diamond_y = jnp.where(
            spawn_diamond,
            57.0, ## TODO: Diamond spawn height, will change if the number is wrong
            next_diamond_y
        )

        # Enemies
        ## Helicopter
        # Apply new active status, position coordinates for spawned helicopter enemies
        next_helicopter_active = next_helicopter_active | spawn_helicopter
        next_helicopter_x = jnp.where(
            spawn_helicopter,
            self.consts.GAME_AREA_MAX_X,
            next_helicopter_x
        )
        next_helicopter_y = jnp.where(
            spawn_helicopter,
            57.0, ## TODO: Helicopter spawn height, will change if the number is wrong
            next_helicopter_y
        )
        ## Satellite
        # Apply new active status, position coordinates for spawned satellite enemies
        next_satellite_active = next_satellite_active | can_spawn_satellite
        next_satellite_x = jnp.where(
            can_spawn_satellite,
            self.consts.GAME_AREA_MIN_X - self.consts.SATELLITE_ENEMY_WIDTH,
            next_satellite_x
        )
        next_satellite_y = jnp.where(
            can_spawn_satellite,
            75.0, ## TODO: Satellite spawn height, will change if the number is wrong
            next_satellite_y
        )
        ## Fire pit
        ## TODO: The spawn of fire pit is a little bit complicated, first one spawn at x=124, but from the next one it will spawn at GAME_AREA_MAX_X, and the next one always spawn even the previous one is still on screen(as far as i checked, after the yellow part of fire pit disappears on GAME_AREA_MIN_X)
        ## TODO: Now i apply the same logic as enemy and diamond, which is only spawn when the entire row is empty, will change it after we discuss about it
        ## The pit is a single scalar object (see reset and _render_pit), so
        ## spawn with plain jnp.where instead of array indexing.
        next_pit_active = next_pit_active | can_spawn_pit
        next_pit_x = jnp.where(
            can_spawn_pit,
            self.consts.GAME_AREA_MAX_X,
            next_pit_x
        )
        next_pit_y = jnp.where(
            can_spawn_pit,
            122.0, ## TODO: Pit spawn height, will change if the number is wrong
            next_pit_y
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
            spawn_diamond_next=next_spawn_diamond_next,
            pit_x=next_pit_x,
            pit_y=next_pit_y,
            pit_active=next_pit_active
        )

    def _update_enemy_bombs(self, state: JamesBondState) -> JamesBondState: ## TODO: Way too complex, can easily optimize
        """Move falling helicopter bombs and periodically drop new ones.

        Bombs live in the generic bullet_* arrays. They fall straight down
        while keeping the world-scroll drift, and despawn once they are
        fully below the player's row.
        """

        # === 1. Move active bombs, despawn below the play area ===
        next_bomb_x = state.bullet_x - self.consts.ENEMY_BOMB_DRIFT_SPEED
        next_bomb_y = state.bullet_y + self.consts.ENEMY_BOMB_FALL_SPEED
        bomb_on_screen = next_bomb_y <= (
            self.consts.GAME_AREA_MAX_Y + self.consts.PLAYER_HEIGHT
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

    def _check_collisions_placeholder(self, state: JamesBondState) -> JamesBondState: ## TODO: what is this for?
        # Future diamond, enemy, bullet, and life collision logic belongs here.
        state = state.replace(
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
        state = self._resolve_bullet_player_collisions(state)
        state = self._resolve_pit_player_collisions(state)
        return state

    def _resolve_bullet_player_collisions(self, state: JamesBondState) -> JamesBondState: ## TODO: Make into multi-bullet
        """Apply one life of damage when a helicopter bullet hits the player.

        The bullet always deactivates on contact; the life is only
        lost when the hit cooldown has expired, mirroring the hazard rule.
        """

        overlaps = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            state.bullet_x,
            state.bullet_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
        )
        hits = jnp.logical_and(state.bullet_active, overlaps)
        hit_any = jnp.any(hits)
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(hit_any, can_take_damage)

        return state.replace(
            bullet_active=jnp.logical_and(
                state.bullet_active, jnp.logical_not(hits)
            ),
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
        )

    def _resolve_pit_player_collisions(self, state: JamesBondState) -> JamesBondState:
        """Apply one life of damage when the player drives into the fire pit.

        Only ground contact is deadly: a jumping player clears the pit. The
        player bullet and helicopter bombs pass over pits without responding,
        matching the original game, so no projectile checks happen here. The
        deadly zone is centered inside the wider pit sprite so an edge tap is
        survivable.
        """

        pit_inset = (self.consts.PIT_WIDTH - self.consts.PIT_COLLISION_WIDTH) / 2
        pit_left = state.pit_x + pit_inset
        pit_right = pit_left + self.consts.PIT_COLLISION_WIDTH
        x_overlap = jnp.logical_and(
            state.player_x < pit_right,
            state.player_x + self.consts.PLAYER_COLLISION_WIDTH > pit_left,
        )
        on_ground = (state.player_y == self.consts.PLAYER_INIT_Y)
        pit_collision = jnp.logical_and(
            state.pit_active, jnp.logical_and(on_ground, x_overlap)
        )
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(pit_collision, can_take_damage)

        return state.replace(
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
        )

    def collectible_collisions_logic(self, state: JamesBondState) -> JamesBondState:
        """Collect active diamonds that overlap the player's bullet collision box."""

        overlaps = _aabb_overlap(
            state.player_bullet_x,
            state.player_bullet_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
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
            diamond_active = jnp.logical_and( ## TODO: change diamond x and y?
                state.diamond_active, ~collected
            ),
            player_bullet_active=player_bullet_active,
            player_bullet_step=player_bullet_step,
            player_bullet_x=player_bullet_x,
            player_bullet_y=player_bullet_y,
            score=state.score + self.consts.SCORE_DIAMOND,
            collected_diamond=jnp.logical_or(state.collected_diamond, collected_any),
        )

    def _resolve_player_bullet_collisions(self, state: JamesBondState) -> JamesBondState:
        ## In the original game the bullet passes straight through helicopters
        ## and satellites without any visible response: the diamond is the
        ## only object the player bullet collides with, so only that check
        ## runs, and only while a bullet is in flight.
        return lax.cond(
            state.player_bullet_active,
            self.collectible_collisions_logic,
            lambda s: s,
            state
        )

    def _get_reward( ## TODO: Wrong logic
        self, previous_state: JamesBondState, state: JamesBondState
    ) -> chex.Array:
        """Calculate reward from collision-driven state transitions."""

        score_gained = jnp.maximum(state.score - previous_state.score, 0)
        diamonds_collected = jnp.where(
            state.collected_diamond,
            jnp.floor_divide(score_gained, self.consts.SCORE_DIAMOND),
            0,
        )
        lives_lost = jnp.maximum(previous_state.lives - state.lives, 0)

        return (
            jnp.array(self.consts.REWARD_STEP, dtype=jnp.float32)
            + diamonds_collected.astype(jnp.float32) * self.consts.REWARD_DIAMOND
            + lives_lost.astype(jnp.float32) * self.consts.REWARD_LOST_LIFE
        )

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
        super().__init__(self.consts)

        if config is None:
            config = render_utils.RendererConfig(
                game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
                channels=3,
                downscale=None,
            )
        self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        ## The jamesbond sprites are committed in the repo, not part of the
        ## downloadable sprite pack, so load them from jb_sprites directly.
        sprite_path = JB_SPRITE_DIR

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(self.consts.ASSET_CONFIG, sprite_path)


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: JamesBondState) -> jnp.ndarray:
        """Render a simple background, inactive object slots, and player box."""

        raster = self.jr.create_object_raster(self.BACKGROUND) ## TODO: Render ground, maybe as part of background?

        raster = self._render_car(raster, state)
        raster = self._render_diamond(raster, state)
        raster = self._render_pit(raster, state)
        raster = self._render_helicopter(raster, state)
        raster = self._render_satellite(raster, state)

        raster = self._render_bullets(raster, state)

        ## Render life counter
        raster = self.jr.render_indicator(raster, 9, 184, state.lives, self.SHAPE_MASKS['life'], 16, 3) ## TODO: Maybe 5 like in ALE?
        
        ## Render Score counter
        score_digits = self.jr.int_to_digits(state.score, 4) ## TODO: Max score 4 digits?
        raster = self.jr.render_label(raster, 95, 15, score_digits, self.SHAPE_MASKS['score_digits'], 8, 4) ## TODO: Position offset per digit?

        return self.jr.render_from_palette(raster, self.PALETTE)

    def _render_background(self, raster: jnp.ndarray) -> jnp.ndarray: ## TODO: Turn to ground renderer
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

    def _render_car(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray: ## TODO: Implement death animation
        """Draw the player car."""

        sprite_idx = jnp.where(
            state.hit_cooldown > 0,
            jnp.where(
                state.step_count % 2 == 0,
                1,
                2
            ),
            0
        )

        return self.jr.render_at_clipped(
            raster, 
            state.player_x, 
            state.player_y, 
            self.SHAPE_MASKS['car'][sprite_idx]
        )
    
    def _render_diamond(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw every active diamond (diamond state is a fixed-size array)."""

        sprite_idx = jnp.where(
            state.step_count % 2 == 0,
            0,
            1
        )

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.diamond_x,
            state.diamond_y,
            self.SHAPE_MASKS['diamond'][sprite_idx],
        )

        return jax.lax.cond(state.diamond_active, draw_fn, lambda r: r, raster)
    
    def _render_pit(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the fire pit."""

        sprite_idx = jnp.where(
            state.step_count % 2 == 0,
            0,
            1
        )

        draw_fn = lambda r: self.jr.render_at_clipped(
            r, 
            state.pit_x, 
            state.pit_y, 
            self.SHAPE_MASKS['pit'][sprite_idx],
        )

        return jax.lax.cond(state.pit_active, draw_fn, lambda r: r, raster)
    
    def _render_helicopter(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray: ## TODO: Implement melee animation
        """Draw every active helicopter (helicopter state is a fixed-size array)."""

        sprite_idx = jnp.where(
            state.step_count % 2 == 0,
            0,
            1
        )

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.helicopter_x,
            state.helicopter_y,
            self.SHAPE_MASKS['helicopter'][sprite_idx],
        )

        return jax.lax.cond(state.helicopter_active, draw_fn, lambda r: r, raster)

    def _render_satellite(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray: ## TODO: Implement spawning blinking animation? No need, doesn't do anything and is random?
        """Draw every active satellite (satellite state is a fixed-size array)."""

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.satellite_x,
            state.satellite_y,
            self.SHAPE_MASKS['satellite'],
        )

        return jax.lax.cond(state.satellite_active, draw_fn, lambda r: r, raster)
    
    def _render_bullets(self, raster: jnp.ndarray, state: JamesBondState,) -> jnp.ndarray: ## TODO: Make into multi-bullet
        """Draw all bullets."""
        
        active_bullets = jnp.concatenate([
            jnp.array([state.player_bullet_active]),
            state.bullet_active,
        ])

        bullet_positions = jnp.vstack([
            jnp.stack([state.player_bullet_x, state.player_bullet_y]),
            jnp.stack(
                [state.bullet_x, state.bullet_y],
            ),
        ])

        def render_single_bullet(i, current_raster):
            should_draw = (active_bullets[i] == 1)

            draw_fn = lambda r: self.jr.render_at_clipped(
                r, 
                bullet_positions[i][0], 
                bullet_positions[i][1], 
                self.SHAPE_MASKS['bullet'],
            )

            return jax.lax.cond(should_draw, draw_fn, lambda r: r, current_raster)

        return jax.lax.fori_loop(0, jnp.size(active_bullets), render_single_bullet, raster)
