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
                ## The searchlight sweep table indexes sprites 0..14, so the
                ## whole extracted sequence has to be loaded (the group loader
                ## pads the differing widths)
                'files': [f'helicopter_shot_{i}.npy' for i in range(1, 17)]
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
            ## Water scene actors
            {
                'name': 'scuba', 'type': 'group',
                'files': ['scuba_1.npy', 'scuba_2.npy']
            },
            ## The green figure where a laser hits the water, two poses
            {'name': 'splash', 'type': 'single', 'file': 'explosion_1_(small).npy'},
            {'name': 'splash_wide', 'type': 'single', 'file': 'explosion_2.npy'},
            ## Recolored bolt for sinking past a living splash figure
            {'name': 'laser_green', 'type': 'single', 'file': 'laser_green.npy'},

            ## Water scene terrain. water_unkempt is already the real ROM blue
            ## (45,50,184); the sky is a solid (74,74,74) slab; seabed_water is
            ## the wide seabed strip recoloured to the real (50,132,50) green
            ## (the shared seabed.npy keeps its land colours, untouched)
            ## water_full is the solid band: water_unkempt tapers into a valley
            ## notch at the bottom, which let the black background show through
            ## where the real game has plain blue behind the seabed
            {'name': 'water', 'type': 'single', 'file': 'water_full.npy'},
            {'name': 'water_sky', 'type': 'single', 'file': 'water_sky.npy'},
            {'name': 'seabed_water', 'type': 'single', 'file': 'seabed_water.npy'},
            ## Second water scene roster (all cropped from real ALE frames on
            ## the test branch): floating rocket, submarine, pink flyer, and
            ## the red debris bars the rocket burst leaves behind
            {'name': 'rocket', 'type': 'single', 'file': 'rocket.npy'},
            {'name': 'submarine', 'type': 'single', 'file': 'submarine.npy'},
            {'name': 'heli_pink', 'type': 'single', 'file': 'heli_pink.npy'},
            {'name': 'flyer_red', 'type': 'single', 'file': 'flyer_red.npy'},
            {'name': 'sky_flash', 'type': 'single', 'file': 'sky_flash.npy'}, ## whole sky flashes gray on the burst

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
    PLAYER_IN_Y_STEPS = jnp.array([ ## For the gravity feel of jumps. Each jump is 71 frames, 72nd frame is the start of the fall
        0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ## TODO: Remove first zero?
        0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 
        0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1 ## Water matrix: 111010101110011010101010010010100100010000100000001000000000-1; Nearly the same as air. Also, can jump higher and faster from water to air
    ], dtype=jnp.int32)
    PLAYER_WATER_BULLET_STEPS = jnp.array([
        (2, -1), (2, -1), (2, -1), (2, -1),
        (1, 1),  (1, 2),  (1, 1),  (1, 2)  ## And then (1, 0), (0, 2) until 60 frames
    ], dtype=jnp.int32)

    MAX_LIVES: int = struct.field(pytree_node=False, default=3)
    MAX_DIAMONDS: int = struct.field(pytree_node=False, default=8)
    MAX_ENEMIES: int = struct.field(pytree_node=False, default=8)
    MAX_HELICOPTERS: int = struct.field(pytree_node=False, default=4)
    MAX_SATELLITES: int = struct.field(pytree_node=False, default=4)
    MAX_EPISODE_STEPS: int = struct.field(pytree_node=False, default=5000)

    DIAMOND_WIDTH: int = struct.field(pytree_node=False, default=7) ##TODO: There is 7 pixels in the diamond sprite, including the shining thing of diamond
    DIAMOND_HEIGHT: int = struct.field(pytree_node=False, default=13) ##TODO: There is 13 pixels in the diamond sprite, including the shining thing of diamond 
    ## TODO: Change / Remove after observation and collision enemey variables have been changed; or else will cause fail tests
    ENEMY_WIDTH: int = struct.field(pytree_node=False, default=10)
    ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=8)
    ## TODO: Enemies (now i only have the helicopter and satellite enemies)
    HELICOPTER_ENEMY_WIDTH: int = struct.field(pytree_node=False, default=8) ## TODO: Helicopter width is 8 pixels
    HELICOPTER_ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=6) ## TODO: Helicopter height is 6 pixels
    HELICOPTER_MELEE_SPRITE_STEPS = jnp.array([ ## 2nd elements are x positions of sprites. Follows the sequence: Sprite 1 -> Nothing -> Sprite 1 -> Nothing -> Sprite 2 -> ...
        (-1,-1), (0,5), (-1,-1), (1,4), (-1,-1), (1,4), (-1,-1), (2,3), (-1,-1), (2,3), 
        (-1,-1), (3,0), (-1,-1), (3,0), (-1,-1), (4,-8), (-1,-1), (4,-8), (-1,-1), (5,-15), 
        (-1,-1), (5,-15), (-1,-1), (6,-22), (-1,-1), (6,-22), (-1,-1), (7,-30), (-1,-1), (7,-38),
        (-1,-1), (8,-46), (-1,-1), (8,-46), (-1,-1), (9,-54), (-1,-1), (9,-54), (-1,-1), (10,9),
        (-1,-1), (10,9), (-1,-1), (11,6), (-1,-1), (11,6), (-1,-1), (12,5), (-1,-1), (12,5),
        (-1,-1), (13,4), (-1,-1), (13,4), (-1,-1), (14,3), (-1,-1), (14,3), (-1,-1), (1,4),
        (-1,-1), (1,4), (-1,-1), (2,3), (-1,-1), (2,3), (-1,-1), (3,0), (-1,-1), (3,0),
        (-1,-1), (4,-8), (-1,-1), (4,-8), (-1,-1), (5,-15), (-1,-1), (5,-15), (-1,-1), (6,-22), 
        (-1,-1), (6,-22), (-1,-1), (7,-30), (-1,-1), (7,-30), (-1,-1), (8,-38), (-1,-1), (8,-38)
    ], dtype=jnp.int32)
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

    SCORE_DIAMOND: int = struct.field(pytree_node=False, default=50) ## Manual scoring table says diamond = 50
    SCORE_ENEMY: int = struct.field(pytree_node=False, default=250)
    HIT_COOLDOWN_STEPS: int = struct.field(pytree_node=False, default=60)

    ## Enemy fire: one helicopter bomb and one satellite laser, single objects
    ## like everything else. Numbers checked against the real ROM in ALE frame
    ## by frame, not guessed.
    ## The bomb picks its direction once when it drops: 1px per frame towards
    ## the side the player is on, and keeps it for the whole fall. Found while
    ## playing (bombs fall left AND right diagonal) and confirmed in ALE by
    ## teleporting the player around via RAM.
    HELICOPTER_BOMB_SPEED_X: int = struct.field(pytree_node=False, default=1)
    HELICOPTER_BOMB_VY: int = struct.field(pytree_node=False, default=2) ## falls 2px per frame
    ## Drop trigger is player relative, not screen relative. Measured in ALE:
    ## first bomb releases when the heli closes to ~66-70 real px of the player
    ## (that is ~31 in our half width coordinates), later bombs at ~30 real px
    ## (~14 here). The old fixed searchlight zone only matched because the test
    ## player never moved. TODO: the melee zone probably wants the same
    ## treatment, talk to Indi before touching it.
    HELICOPTER_BOMB_RANGE: int = struct.field(pytree_node=False, default=31) ## first chance when the heli closes to this (~66-70 real px, measured)
    HELICOPTER_BOMB_RETRY_FRAMES: int = struct.field(pytree_node=False, default=45) ## next chance this many frames later (measured gaps 38-53)
    HELICOPTER_BOMB_MAX_PER_PASS: int = struct.field(pytree_node=False, default=4)
    ## The real picker looks like the ROM's internal random generator (it's
    ## not position, speed or the missile slot, we tested all three). So:
    ## while the heli is in range it gets a chance every RETRY_FRAMES, each
    ## one rarer than the last (chance / (1 + drops so far)), which lands at
    ## roughly: one bomb common, two rarer, three much rarer, four rare.
    HELICOPTER_BOMB_DROP_CHANCE: float = struct.field(pytree_node=False, default=0.5)
    ## 75 lands on ~2 drops per full-screen pass like in ALE (52 gave 3-4)
    SATELLITE_LASER_DROP_PERIOD: int = struct.field(pytree_node=False, default=75)
    SATELLITE_LASER_FALL_SPEED: int = struct.field(pytree_node=False, default=1) ## laser falls straight down, no sideways drift
    SATELLITE_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=46) ## measured 44-48 frame gap between passes

    ## GAME_AREA_MAX_X is only the player's hard stop; world objects use
    ## the real screen edges like in ALE
    OBJECT_EXIT_X: int = struct.field(pytree_node=False, default=159) ## rightward movers leave here

    ## Water scene scuba diver, measured in ALE on the test branch: a 7x20
    ## vertical swimmer at a fixed depth, cannot be shot, deadly on touch
    SCUBA_WIDTH: int = struct.field(pytree_node=False, default=7)
    SCUBA_HEIGHT: int = struct.field(pytree_node=False, default=20)
    SCUBA_SPAWN_X: int = struct.field(pytree_node=False, default=155) ## enters at the right screen edge
    SCUBA_SPAWN_Y: int = struct.field(pytree_node=False, default=129) ## body below the surface row
    SCUBA_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=333) ## vanishes on a clock, not at the edge
    SCUBA_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=150) ## breather between divers
    ## When a bomb comes down and the diver is in the water, HE is the one that
    ## goes radioactive, and not straight away -- it takes a moment.
    SCUBA_RADIOACTIVE_DELAY: int = struct.field(pytree_node=False, default=30) ## bomb reaches him -> he turns
    SCUBA_RADIOACTIVE_FRAMES: int = struct.field(pytree_node=False, default=120) ## how long he stays radioactive

    ## The green splash figure a spent bolt detonates into. It is its own
    ## object, not the scuba diver recolored
    SPLASH_WIDTH: int = struct.field(pytree_node=False, default=20)  ## explosion_1_(small).npy is 20x7
    SPLASH_HEIGHT: int = struct.field(pytree_node=False, default=7)
    SPLASH_Y: int = struct.field(pytree_node=False, default=123) ## straddles the surface row
    SPLASH_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=120)
    SPLASH_SAFE_PLAYER_Y: int = struct.field(pytree_node=False, default=112) ## airborne above this is safe

    ## In the water the bolt sinks past the surface. It does NOT have to reach
    ## the sea floor: playtesting the real game shows it goes radioactive part
    ## way down, a moment after it enters the water.
    WATER_LASER_FLOOR: int = struct.field(pytree_node=False, default=132)
    WATER_RADIOACTIVE_Y: int = struct.field(pytree_node=False, default=127) ## depth where the bolt goes radioactive

    ## Second water scene (stage 2): the floating rocket is the only scoring
    ## object (+200 for ramming it, though the ram usually costs a life too).
    ## Its cycle, measured frame by frame on the test branch: appears
    ## mid-screen submerged, floats briefly, climbs 1px/frame, and bursts
    ## with its tip at row 61 into two red debris bars plus a 1-2 frame
    ## full-sky flash. The submarine cruises deep and only threatens a
    ## diving boat; the pink flyer just crosses the sky.
    SCORE_ROCKET: int = struct.field(pytree_node=False, default=200)
    ROCKET_WIDTH: int = struct.field(pytree_node=False, default=8)
    ROCKET_HEIGHT: int = struct.field(pytree_node=False, default=11)
    ROCKET_Y: int = struct.field(pytree_node=False, default=138) ## rests low, a diving boat can ram it
    ROCKET_SPAWN_X: int = struct.field(pytree_node=False, default=85) ## appears mid-screen, not at the edge
    ROCKET_IGNITE_AGE: int = struct.field(pytree_node=False, default=6)
    ROCKET_EXPLODE_Y: int = struct.field(pytree_node=False, default=61) ## tip row where it bursts
    ROCKET_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=171) ## 256 frame cycle minus its life
    DEBRIS_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=120)
    SKY_FLASH_FRAMES: int = struct.field(pytree_node=False, default=2)
    SUBMARINE_WIDTH: int = struct.field(pytree_node=False, default=16)
    SUBMARINE_HEIGHT: int = struct.field(pytree_node=False, default=11)
    SUBMARINE_Y: int = struct.field(pytree_node=False, default=135) ## deep under the surface
    SUBMARINE_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=260)
    WB_HELI_Y: int = struct.field(pytree_node=False, default=57)
    WB_HELI_WIDTH: int = struct.field(pytree_node=False, default=4)
    WB_HELI_HEIGHT: int = struct.field(pytree_node=False, default=11)
    WB_HELI_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=320)
    WB_FLYER_Y: int = struct.field(pytree_node=False, default=61) ## debris bars linger where the rocket burst
    WB_FLYER_WIDTH: int = struct.field(pytree_node=False, default=4)
    WB_FLYER_HEIGHT: int = struct.field(pytree_node=False, default=5)
    ## The submarine's short underwater torpedo: up-left (-2,-1) px/frame,
    ## dies before the surface. Measured harmless, pure theatre
    SUB_TORPEDO_PERIOD: int = struct.field(pytree_node=False, default=400)

    ## Stage progression, forward only, cycling 0->1->2->3->0. The land and
    ## first-water lengths were measured in ALE on the test branch; the later
    ## two were timed off a longplay video (water B ~14s, water C ~96s).
    ## Completing any water scene pays the 5000 point bonus (seen in the
    ## video: 600->5600, 8100->13200, 16600->21600).
    STAGE_LAND_LENGTH: int = struct.field(pytree_node=False, default=4454)
    STAGE_WATER_A_LENGTH: int = struct.field(pytree_node=False, default=4435)
    STAGE_WATER_B_LENGTH: int = struct.field(pytree_node=False, default=840)
    STAGE_WATER_C_LENGTH: int = struct.field(pytree_node=False, default=5760)
    SCORE_STAGE_BONUS: int = struct.field(pytree_node=False, default=5000)
    ## Playtest knob: JB_START_STAGE=2 python scripts/play.py -g jamesbond
    ## drops a fresh game straight into that scene (read once at import)
    START_STAGE: int = struct.field(
        pytree_node=False, default=int(os.environ.get("JB_START_STAGE", "0"))
    )

    ## Water scene satellite: fires when its belly is this far to the RIGHT
    ## of the player's hull, checked at discrete moments, never from behind
    SATELLITE_DROP_AHEAD_MIN: int = struct.field(pytree_node=False, default=1)
    SATELLITE_DROP_AHEAD_MAX: int = struct.field(pytree_node=False, default=95)
    SATELLITE_CHECK_PERIOD: int = struct.field(pytree_node=False, default=30) ## check moments ~10-60f apart in ALE
    SATELLITE_WATER_MAX_DROPS: int = struct.field(pytree_node=False, default=2) ## 1-2 per pass observed

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
    player_diving: chex.Array
    player_floating: chex.Array
    player_fast_floating: chex.Array
    player_in_water_step: chex.Array
    player_bullet_active: chex.Array ## TODO: Maybe think about every bullet as bullet with different and direction (and speed?)?
    player_bullet_step: chex.Array
    player_bullet_x: chex.Array
    player_bullet_y: chex.Array
    player_wbullet_active: chex.Array ## Water bullet
    player_wbullet_step: chex.Array
    player_wbullet_x: chex.Array
    player_wbullet_y: chex.Array
    lives: chex.Array
    score: chex.Array
    step_count: chex.Array
    stage: chex.Array
    stage_step: chex.Array ## frames into the current scene
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
    helicopter_melee_step: chex.Array
    satellite_x: chex.Array
    satellite_y: chex.Array
    satellite_active: chex.Array
    ## Enemy fire, single objects (the 2600 also only had one missile per object)
    helicopter_bomb_x: chex.Array
    helicopter_bomb_y: chex.Array
    helicopter_bomb_vx: chex.Array ## +-1, aimed at the player once on release
    helicopter_bomb_active: chex.Array
    helicopter_bombs_dropped: chex.Array ## chances used this pass, resets with the heli
    helicopter_bomb_timer: chex.Array ## frames until the next chance
    satellite_laser_x: chex.Array
    satellite_laser_y: chex.Array
    satellite_laser_active: chex.Array
    satellite_laser_timer: chex.Array ## counts down to the next laser drop
    satellite_lasers_dropped: chex.Array ## water scene: lasers used this pass
    satellite_respawn_timer: chex.Array ## breather between satellite passes
    ## Water scene scuba diver, single scalar object like the enemies
    scuba_x: chex.Array
    scuba_y: chex.Array
    scuba_active: chex.Array
    scuba_age: chex.Array ## frames since he entered; he vanishes on a clock
    scuba_respawn_timer: chex.Array ## breather before the next diver enters
    scuba_radioactive: chex.Array ## he is currently the radioactive one
    scuba_radioactive_age: chex.Array ## how long he has been glowing
    scuba_radioactive_timer: chex.Array ## bomb reached him, counting down to the change
    ## Green splash figure: static in world space, rides the scroll left
    splash_x: chex.Array
    splash_active: chex.Array
    splash_age: chex.Array
    ## Second water scene roster (stage 2)
    rocket_x: chex.Array
    rocket_y: chex.Array
    rocket_active: chex.Array
    rocket_age: chex.Array ## ignites and launches after idling
    rocket_timer: chex.Array
    submarine_x: chex.Array
    submarine_active: chex.Array
    submarine_timer: chex.Array
    wb_heli_x: chex.Array
    wb_heli_active: chex.Array
    wb_heli_timer: chex.Array
    wb_flyer_x: chex.Array ## the rocket's debris bars
    wb_flyer_active: chex.Array
    wb_flyer_timer: chex.Array
    sub_torp_x: chex.Array
    sub_torp_y: chex.Array
    sub_torp_active: chex.Array
    collected_diamond: chex.Array
    hit_enemy: chex.Array
    fired_bullet: chex.Array
    key: chex.PRNGKey


@struct.dataclass
class JamesBondObservation:
    """Object-centric observation matching observation_space()."""

    player: ObjectObservation
    diamonds: ObjectObservation
    player_velocity: jnp.ndarray
    helicopters: ObjectObservation
    satellites: ObjectObservation
    ## Field order must match the space Dict order
    scubas: ObjectObservation
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
            player_diving=jnp.array(False, dtype=jnp.bool_),
            player_floating=jnp.array(False, dtype=jnp.bool_),
            player_fast_floating=jnp.array(False, dtype=jnp.bool_),
            player_in_water_step=jnp.array(0, dtype=jnp.int32),
            player_bullet_active=jnp.array(False, dtype=jnp.bool_),
            player_bullet_step=jnp.array(-1, dtype=jnp.int32),
            player_bullet_x=jnp.array(-1, dtype=jnp.int32),
            player_bullet_y=jnp.array(-1, dtype=jnp.int32),
            player_wbullet_active=jnp.array(False, dtype=jnp.bool_),
            player_wbullet_step=jnp.array(-1, dtype=jnp.int32),
            player_wbullet_x=jnp.array(-1, dtype=jnp.int32),
            player_wbullet_y=jnp.array(-1, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            stage=jnp.array(self.consts.START_STAGE, dtype=jnp.int32),
            stage_step=jnp.array(0, dtype=jnp.int32),
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
            helicopter_melee_step=jnp.array(0, dtype=jnp.int32),
            satellite_x=jnp.array(0, dtype=jnp.int32),
            satellite_y=jnp.array(0, dtype=jnp.int32),
            satellite_active=jnp.array(0, dtype=jnp.bool_),
            helicopter_bomb_x=jnp.array(-1, dtype=jnp.int32),
            helicopter_bomb_y=jnp.array(-1, dtype=jnp.int32),
            helicopter_bomb_vx=jnp.array(0, dtype=jnp.int32),
            helicopter_bomb_active=jnp.array(False, dtype=jnp.bool_),
            helicopter_bombs_dropped=jnp.array(0, dtype=jnp.int32),
            helicopter_bomb_timer=jnp.array(0, dtype=jnp.int32),
            satellite_laser_x=jnp.array(-1, dtype=jnp.int32),
            satellite_laser_y=jnp.array(-1, dtype=jnp.int32),
            satellite_laser_active=jnp.array(False, dtype=jnp.bool_),
            ## Start full so the first laser comes one full period after the satellite shows up
            satellite_laser_timer=jnp.array(
                self.consts.SATELLITE_LASER_DROP_PERIOD, dtype=jnp.int32
            ),
            satellite_lasers_dropped=jnp.array(0, dtype=jnp.int32),
            ## Timer at 0 so the very first satellite appears right away
            satellite_respawn_timer=jnp.array(0, dtype=jnp.int32),
            ## The scuba diver starts parked off screen, water scene only
            scuba_x=jnp.array(-1, dtype=jnp.int32),
            scuba_y=jnp.array(-1, dtype=jnp.int32),
            scuba_active=jnp.array(False, dtype=jnp.bool_),
            scuba_age=jnp.array(0, dtype=jnp.int32),
            scuba_respawn_timer=jnp.array(0, dtype=jnp.int32),
            scuba_radioactive=jnp.array(False, dtype=jnp.bool_),
            scuba_radioactive_age=jnp.array(0, dtype=jnp.int32),
            scuba_radioactive_timer=jnp.array(0, dtype=jnp.int32),
            ## Same for the splash figure, it is only born from a bolt impact
            splash_x=jnp.array(-1, dtype=jnp.int32),
            splash_active=jnp.array(False, dtype=jnp.bool_),
            splash_age=jnp.array(0, dtype=jnp.int32),
            ## Second water scene roster, all parked until stage 2
            rocket_x=jnp.array(-1, dtype=jnp.int32),
            rocket_y=jnp.array(-1, dtype=jnp.int32),
            rocket_active=jnp.array(False, dtype=jnp.bool_),
            rocket_age=jnp.array(0, dtype=jnp.int32),
            rocket_timer=jnp.array(0, dtype=jnp.int32),
            submarine_x=jnp.array(-1, dtype=jnp.int32),
            submarine_active=jnp.array(False, dtype=jnp.bool_),
            submarine_timer=jnp.array(0, dtype=jnp.int32),
            wb_heli_x=jnp.array(-1, dtype=jnp.int32),
            wb_heli_active=jnp.array(False, dtype=jnp.bool_),
            wb_heli_timer=jnp.array(0, dtype=jnp.int32),
            wb_flyer_x=jnp.array(-1, dtype=jnp.int32),
            wb_flyer_active=jnp.array(False, dtype=jnp.bool_),
            wb_flyer_timer=jnp.array(0, dtype=jnp.int32),
            sub_torp_x=jnp.array(-1, dtype=jnp.int32),
            sub_torp_y=jnp.array(-1, dtype=jnp.int32),
            sub_torp_active=jnp.array(False, dtype=jnp.bool_),
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
        """Advance one frame and return the repo-standard tuple."""

        atari_action = self._decode_action(action)
        previous_state = state

        state = state.replace(
            step_count=state.step_count + 1,
            collected_diamond=jnp.array(False, dtype=jnp.bool_), ## TODO: Needed?
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
            hit_cooldown=jnp.maximum(state.hit_cooldown - 1, 0),
            fired_bullet=atari_action == Action.FIRE,
        )
        state = self._step_player(state, atari_action)
        state = self._update_objects(state)
        state = self._update_enemy_bombs(state)
        state = self._check_collisions_placeholder(state)
        state = self._stage_step(state) ## TODO: place above?

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
                ## Diamond is a single object now like the enemies
                "diamonds": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                "player_velocity": spaces.Box(
                    low=jnp.array([-10.0, -20.0], dtype=jnp.float32),
                    high=jnp.array([10.0, 20.0], dtype=jnp.float32),
                    shape=(2,),
                    dtype=jnp.float32,
                ),
                "helicopters": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                "satellites": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                "scubas": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                ## player air bullet, player water bullet, helicopter bomb, satellite laser
                "bullets": spaces.get_object_space(
                    n=4, screen_size=screen_size
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
        helicopters = self._object_group_observation(
            state.helicopter_x,
            state.helicopter_y,
            state.helicopter_active,
            self.consts.HELICOPTER_ENEMY_WIDTH,
            self.consts.HELICOPTER_ENEMY_HEIGHT,
        )
        satellites = self._object_group_observation(
            state.satellite_x,
            state.satellite_y,
            state.satellite_active,
            self.consts.SATELLITE_ENEMY_WIDTH,
            self.consts.SATELLITE_ENEMY_HEIGHT,
        )
        scubas = self._object_group_observation(
            state.scuba_x,
            state.scuba_y,
            state.scuba_active,
            self.consts.SCUBA_WIDTH,
            self.consts.SCUBA_HEIGHT,
        )
        ## All four projectiles in one group, they share the same 1x4 sprite:
        ## player air bullet, player water bullet, helicopter bomb, satellite laser
        bullets = self._object_group_observation(
            jnp.stack([
                state.player_bullet_x,
                state.player_wbullet_x,
                state.helicopter_bomb_x,
                state.satellite_laser_x,
            ]),
            jnp.stack([
                state.player_bullet_y,
                state.player_wbullet_y,
                state.helicopter_bomb_y,
                state.satellite_laser_y,
            ]),
            jnp.stack([
                state.player_bullet_active,
                state.player_wbullet_active,
                state.helicopter_bomb_active,
                state.satellite_laser_active,
            ]),
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
        )
        return JamesBondObservation(
            player=player,
            player_velocity=jnp.stack([state.player_vx, state.player_vy]).astype( ## TODO: Does observation need this or can we remove it?
                jnp.float32
            ),
            diamonds=diamonds,
            helicopters=helicopters,
            satellites=satellites,
            scubas=scubas,
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

        ## Inactive objects keep drifting in _update_objects, so
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

    def _stage_step(
        self, state: JamesBondState
    ) -> JamesBondState:
        """Advance the scene clock, roll to the next scene when it is due.

        Forward only, cycling land -> water A -> water B -> water C -> land.
        The old version recomputed the stage from step_count with a lives
        threshold, which sent the game BACK to land whenever a life was
        lost mid-water; the longplay video shows scenes only ever advance,
        each completed water scene paying the 5000 point bonus.
        """

        stage_step = state.stage_step + 1
        stage_length = jnp.select(
            [state.stage == 0, state.stage == 1, state.stage == 2],
            [
                jnp.array(self.consts.STAGE_LAND_LENGTH, dtype=jnp.int32),
                jnp.array(self.consts.STAGE_WATER_A_LENGTH, dtype=jnp.int32),
                jnp.array(self.consts.STAGE_WATER_B_LENGTH, dtype=jnp.int32),
            ],
            jnp.array(self.consts.STAGE_WATER_C_LENGTH, dtype=jnp.int32),
        )
        switched = stage_step >= stage_length
        new_stage = jnp.where(switched, (state.stage + 1) % 4, state.stage)
        stage_step = jnp.where(switched, 0, stage_step)
        ## Completed WATER scenes pay the bonus; leaving land does not
        bonus = jnp.logical_and(switched, state.stage >= 1)

        ## Scene handover: clear the objects that belong to the old terrain
        ## so pits don't leak into the water and a stale splash can't kill
        ## the car back on land. The satellite keeps flying, only its laser
        ## and per-pass counter start fresh.

        def park(v, park_value):
            return jnp.where(switched, jnp.array(park_value, dtype=v.dtype), v)

        return state.replace(
            stage = new_stage,
            stage_step=stage_step,
            score=state.score + bonus.astype(jnp.int32) * self.consts.SCORE_STAGE_BONUS,
            pit_active=park(state.pit_active, False),
            scuba_active=park(state.scuba_active, False),
            scuba_age=park(state.scuba_age, 0),
            scuba_radioactive=park(state.scuba_radioactive, False),
            scuba_radioactive_age=park(state.scuba_radioactive_age, 0),
            scuba_radioactive_timer=park(state.scuba_radioactive_timer, 0),
            splash_active=park(state.splash_active, False),
            splash_age=park(state.splash_age, 0),
            satellite_laser_active=park(state.satellite_laser_active, False),
            satellite_laser_timer=park(
                state.satellite_laser_timer, self.consts.SATELLITE_LASER_DROP_PERIOD
            ),
            satellite_lasers_dropped=park(state.satellite_lasers_dropped, 0),
            ## Stage-2 roster vanishes when its scene ends
            rocket_active=park(state.rocket_active, False),
            rocket_age=park(state.rocket_age, 0),
            submarine_active=park(state.submarine_active, False),
            wb_heli_active=park(state.wb_heli_active, False),
            wb_flyer_active=park(state.wb_flyer_active, False),
            wb_flyer_timer=park(state.wb_flyer_timer, 0),
            sub_torp_active=park(state.sub_torp_active, False),
        )

    def step_player_stage_one( ## TODO: Switch to air logic?
        self, args
    ) -> JamesBondState:
        state, atari_action = args
        
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
        down_pressed = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, down_pressed)
        
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
            jnp.clip(player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_air_step] + 1, self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), ## TODO: Copy player_int_y_steps for performance?
            jnp.where(
                player_jumping, 
                player_y - self.consts.PLAYER_IN_Y_STEPS[player_in_air_step], 
                jnp.where(
                    player_falling, 
                    jnp.clip(player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_air_step], self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), 
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
            player_x = player_x,
            player_y = player_y,

            player_jumping = player_jumping,
            player_falling = player_falling,
            player_fast_falling = player_fast_falling,
            player_in_air_step = player_in_air_step,

            player_bullet_active = player_bullet_active,
            player_bullet_step = player_bullet_step,
            player_bullet_x = player_bullet_x,
            player_bullet_y = player_bullet_y,
        )

    def air_movement_logic(
        self, args
    ) -> JamesBondState:
        state, up_pressed, down_pressed = args

        player_y = state.player_y

        player_jumping = state.player_jumping
        player_falling = state.player_falling
        player_fast_falling = state.player_fast_falling
        player_in_air_step = state.player_in_air_step

        up_pressed = jnp.where(player_jumping, False, up_pressed)
        
        player_jumping = jnp.where(
            player_jumping,
            player_jumping, 
            jnp.where(
                jnp.logical_and(
                    jnp.logical_and(up_pressed, player_in_air_step < 71),
                    player_y == self.consts.PLAYER_INIT_Y
                ), 
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
            player_falling ## TODO: or False?
        )
        
        player_fast_falling = jnp.where(
            player_fast_falling,
            True,
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
            jnp.clip(player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_air_step] + 1, self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), 
            jnp.where(
                player_jumping, 
                player_y - self.consts.PLAYER_IN_Y_STEPS[player_in_air_step], 
                jnp.where(
                    player_falling, 
                    jnp.clip(player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_air_step], self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), 
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

        return state.replace( ## TODO: Use state.replace or output just the values? Use astypes?
            player_y = player_y,

            player_jumping = player_jumping,
            player_falling = player_falling,
            player_fast_falling = player_fast_falling,
            player_in_air_step = player_in_air_step,
        )

    def water_movement_logic(
        self, args
    ) -> JamesBondState:
        state, up_pressed, down_pressed = args

        player_y = state.player_y

        player_diving = state.player_diving
        player_floating = state.player_floating
        player_fast_floating = state.player_fast_floating
        player_in_water_step = state.player_in_water_step
        
        down_pressed = jnp.where(player_diving, False, down_pressed)
        
        player_diving = jnp.where(
            player_diving,
            player_diving, 
            jnp.where(
                jnp.logical_and(
                    jnp.logical_and(up_pressed, player_in_water_step < 71),
                    player_y == self.consts.PLAYER_INIT_Y
                ), 
                True, 
                False
            )
        )
        
        player_floating = jnp.where(
            jnp.logical_and(
                jnp.logical_or(player_floating, player_in_water_step >= 71), 
                player_y != self.consts.PLAYER_INIT_Y
            ), 
            True, 
            player_floating ## TODO: or False?
        )
        
        player_fast_floating = jnp.where(
            player_fast_floating,
            True,
            jnp.where(
                jnp.logical_and(
                    down_pressed,
                    jnp.logical_or(player_diving, player_floating)
                ),
                True,
                False
            )
        )

        player_floating = jnp.where(
            player_fast_floating,
            False,
            player_floating,
        )

        player_diving = jnp.where(
            jnp.logical_or(player_floating, player_fast_floating),
            False,
            player_diving
        )
        
        player_in_water_step = jnp.where( ## Start immediately floating up when reaching the bottom of the dive
            player_in_water_step >= 71, 
            63, 
            player_in_water_step
        )

        player_y = jnp.where(
            player_fast_floating,
            jnp.clip(player_y - (self.consts.PLAYER_IN_Y_STEPS[player_in_water_step] + 1), self.consts.GAME_AREA_MAX_Y, 210), ## TODO: 210 is arbitrary
            jnp.where(
                player_diving, 
                player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_water_step], 
                jnp.where(
                    player_floating, 
                    jnp.clip(player_y - self.consts.PLAYER_IN_Y_STEPS[player_in_water_step], self.consts.GAME_AREA_MAX_Y, 210), ## TODO: 210 is arbitrary
                    player_y
                )
            )
        )

        player_floating = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_floating)
        player_fast_floating = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_fast_floating)

        player_in_water_step = jnp.where(
            player_diving, 
            player_in_water_step + 1, 
            jnp.where(
                player_y == self.consts.PLAYER_INIT_Y,
                0,
                jnp.where(
                    jnp.logical_or(player_floating, player_fast_floating), 
                    player_in_water_step - 1, 
                    0,
                )
            )
        )

        return state.replace( ## TODO: Use state.replace or output just the values? Use astypes?
            player_y = player_y,

            player_diving = player_diving,
            player_floating = player_floating,
            player_fast_floating = player_fast_floating,
            player_in_water_step = player_in_water_step,
        )
    
    def step_player_stage_two(
        self, args
    ) -> JamesBondState:
        state, atari_action = args

        player_x = state.player_x
        player_y = state.player_y

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

        y_function = jnp.where(
            jnp.logical_or(
                state.player_in_air_step > 0,
                jnp.logical_and(up_pressed, player_y == self.consts.PLAYER_INIT_Y),
            ),
            0, ## Air
            jnp.where(
                jnp.logical_or(
                    state.player_in_water_step > 0,
                    jnp.logical_and(down_pressed, player_y == self.consts.PLAYER_INIT_Y),
                ),
                1, ## Water
                2  ## No change
            )
        )

        y_state = jax.lax.switch(
            y_function,
            [
                self.air_movement_logic,
                self.water_movement_logic,
                lambda args: args[0],
            ],
            (state, up_pressed, down_pressed)
        )

        ###
        ### Player Bullet controller
        ###

        def air_bullet_logic(
            state
        ) -> JamesBondState:

            player_bullet_active = state.player_bullet_active
            player_bullet_step = state.player_bullet_step
            player_bullet_x = state.player_bullet_x
            player_bullet_y = state.player_bullet_y
            
            fire_pressed = True
            
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

            return state.replace( ## TODO: Use state.replace or output just the values? Use astypes?
                player_bullet_active = player_bullet_active,
                player_bullet_step = player_bullet_step,
                player_bullet_x = player_bullet_x,
                player_bullet_y = player_bullet_y,
            )

        def water_bullet_logic(
            state
        ) -> JamesBondState:
    
            player_wbullet_active = state.player_wbullet_active
            player_wbullet_step = state.player_wbullet_step
            player_wbullet_x = state.player_wbullet_x
            player_wbullet_y = state.player_wbullet_y
            
            fire_pressed = True
            
            fire_pressed = jnp.where(
                jnp.logical_or(player_wbullet_active, player_wbullet_step >= 60),
                False,
                fire_pressed
            )

            player_wbullet_active = jnp.where( ## 1st frame is creation, 61st is deactivation, 60th is the last active
                player_wbullet_step < 60, 
                jnp.where(
                    player_wbullet_active,
                    player_wbullet_active,
                    jnp.where(
                        fire_pressed,
                        True,
                        False
                    )
                ),
                False
            )

            player_wbullet_x = jnp.where(
                jnp.logical_and(player_wbullet_active, player_wbullet_x == -1), 
                player_x + 7,
                jnp.where(
                    player_wbullet_active,
                    jnp.where(player_wbullet_step < 8,
                        player_wbullet_x + self.consts.PLAYER_WATER_BULLET_STEPS[player_wbullet_step][0],
                        jnp.where(
                            player_wbullet_step % 2 == 1,
                            player_wbullet_x + 1,
                            player_wbullet_x
                        )
                    ),
                    -1
                )
            )

            player_wbullet_y = jnp.where(
                jnp.logical_and(player_wbullet_active, player_wbullet_y == -1), 
                player_y - 1,
                jnp.where(
                    player_wbullet_active,
                    jnp.where(
                        player_wbullet_step < 8,
                        player_wbullet_y + self.consts.PLAYER_WATER_BULLET_STEPS[player_wbullet_step][1],
                        jnp.where(
                            player_wbullet_step % 2 == 1,
                            player_wbullet_y + 1,
                            player_wbullet_y
                        )
                    ),
                    -1
                )
            )

            player_wbullet_step = jnp.where(
                player_wbullet_active,
                player_wbullet_step + 1,
                -1
            )

            player_wbullet_active = jnp.where(
                player_wbullet_step >= 60,
                False,
                player_wbullet_active
            )

            return state.replace(
                player_wbullet_active = player_wbullet_active,
                player_wbullet_step = player_wbullet_step,
                player_wbullet_x = player_wbullet_x,
                player_wbullet_y = player_wbullet_y,
            )

        """
        bullet_function = jnp.where( ## Water bullet is always the first one shot
            fire_pressed,
            jnp.where(
                ~state.player_wbullet_active,
                1, ## Only run water_bullet_logic
                2  ## Run both bullet logics
            ),
            jnp.where(
                state.player_wbullet_active,
                jnp.where(
                    state.player_bullet_active,
                    2,
                    1
                ),
                jnp.where(
                    state.player_bullet_active,
                    3, ## Only run air_bullet_logic
                    0
                )
            )
            0 ## Don't run anything
        )
        """

        bullet_function = (state.player_bullet_active.astype(jnp.int32) << 1) | state.player_wbullet_active.astype(jnp.int32)

        bullet_function = jnp.where(
            bullet_function == 0,
            jnp.where(
                fire_pressed,
                1, ## Water bullet is always the first one shot
                0
            ),
            bullet_function
        )

        bullet_function = jnp.where(
            jnp.logical_and(
                fire_pressed,
                jnp.logical_and(
                    state.player_wbullet_step >= 1, ## TODO: Maybe more?
                    ~state.player_bullet_active,
                )
            ),
            3,
            bullet_function
        )

        bullet_state = jax.lax.switch(
            bullet_function,
            [
                lambda r: r,
                water_bullet_logic,
                air_bullet_logic,
                lambda r: air_bullet_logic(water_bullet_logic(r))
            ],
            state
        )

        return state.replace(
            player_x = player_x,
            player_y = y_state.player_y,

            player_jumping = y_state.player_jumping,
            player_falling = y_state.player_falling,
            player_fast_falling = y_state.player_fast_falling,
            player_in_air_step = y_state.player_in_air_step,

            player_diving = y_state.player_diving,
            player_floating = y_state.player_floating,
            player_fast_floating = y_state.player_fast_floating,
            player_in_water_step = y_state.player_in_water_step,

            player_wbullet_active = bullet_state.player_wbullet_active,
            player_wbullet_step = bullet_state.player_wbullet_step,
            player_wbullet_x = bullet_state.player_wbullet_x,
            player_wbullet_y = bullet_state.player_wbullet_y,

            player_bullet_active = bullet_state.player_bullet_active,
            player_bullet_step = bullet_state.player_bullet_step,
            player_bullet_x = bullet_state.player_bullet_x,
            player_bullet_y = bullet_state.player_bullet_y,
        )

    def step_player_stage_three_placeholder(
        self, args
    ):
        state, atari_action = args
        
        return state

    def _step_player(
        self, state: JamesBondState, atari_action: chex.Array
    ) -> JamesBondState:

        return jax.lax.switch( ## Stage indexing starts with 0
            state.stage,
            [
                self.step_player_stage_two,
                self.step_player_stage_two,
                ## The later water scenes drive the same boat with the same
                ## water physics, so they share the controller (the old
                ## placeholder froze the player solid in stage 2)
                self.step_player_stage_two,
                self.step_player_stage_two,
            ],
            (state, atari_action)
        )

    def _update_objects(self, state: JamesBondState) -> JamesBondState: ## TODO: Implement fire pit
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

        # Scuba (Scroll left)
        ## Swims left 1px every 4th frame at a fixed depth, never chases.
        ## No off-screen check: he vanishes mid-screen on his age clock
        next_scuba_x = jnp.where(
            state.step_count % 4 == 0,
            state.scuba_x - 1,
            state.scuba_x
        )
        next_scuba_y = state.scuba_y
        scuba_age = jnp.where(state.scuba_active, state.scuba_age + 1, 0)
        ## The glow runs on its own clock and ends by itself
        scuba_rad_age = jnp.where(
            state.scuba_radioactive, state.scuba_radioactive_age + 1, 0
        )
        still_glowing = state.scuba_radioactive & (
            scuba_rad_age < self.consts.SCUBA_RADIOACTIVE_FRAMES
        )
        ## He normally vanishes on his age clock, but a radioactive diver stays
        ## until he has finished glowing -- otherwise the radioactive slot could
        ## disappear mid-glow and the rule "only one at a time" would be moot
        next_scuba_active = state.scuba_active & (
            (scuba_age < self.consts.SCUBA_LIFETIME_FRAMES) | still_glowing
        )
        next_scuba_radioactive = still_glowing & next_scuba_active

        # Green splash figure (Scroll left)
        ## Static in world space, so it drifts with the terrain scroll and
        ## burns out on its own clock. Born in _update_enemy_bombs
        next_splash_x = jnp.where(
            state.step_count % 4 == 0,
            state.splash_x - 1,
            state.splash_x
        )
        splash_age = jnp.where(state.splash_active, state.splash_age + 1, 0)
        next_splash_active = state.splash_active & (
            splash_age < self.consts.SPLASH_LIFETIME_FRAMES
        )

        # Second water scene roster (stage 2)
        in_water_b = state.stage == 2
        ## Rocket: floats submerged a few frames, then climbs straight up
        ## 1px/frame and bursts with its tip at the measured explosion row,
        ## leaving the red debris bars and a short full-sky flash
        rocket_age = jnp.where(state.rocket_active, state.rocket_age + 1, 0)
        rocket_flying = rocket_age >= self.consts.ROCKET_IGNITE_AGE
        next_rocket_x = jnp.where(
            state.rocket_active & (~rocket_flying) & (state.step_count % 4 == 0),
            state.rocket_x - 1,
            state.rocket_x
        )
        next_rocket_y = jnp.where(
            state.rocket_active & rocket_flying,
            state.rocket_y - 1,
            state.rocket_y
        )
        rocket_explodes = state.rocket_active & (
            next_rocket_y <= self.consts.ROCKET_EXPLODE_Y
        )
        next_rocket_active = state.rocket_active & (~rocket_explodes) & (
            next_rocket_x > self.consts.GAME_AREA_MIN_X - self.consts.ROCKET_WIDTH
        )
        ## Submarine: cruises left under water, a touch faster than the scroll
        next_submarine_x = jnp.where(
            state.submarine_active & (state.step_count % 3 == 0),
            state.submarine_x - 1,
            state.submarine_x
        )
        next_submarine_active = state.submarine_active & (
            next_submarine_x > self.consts.GAME_AREA_MIN_X - self.consts.SUBMARINE_WIDTH
        )
        ## Torpedo: a short underwater dart up-left, gone before the surface
        next_torp_x = jnp.where(state.sub_torp_active, state.sub_torp_x - 2, -1)
        next_torp_y = jnp.where(state.sub_torp_active, state.sub_torp_y - 1, -1)
        next_torp_active = state.sub_torp_active & (next_torp_y > 124)
        fire_torp = (
            in_water_b & next_submarine_active & (~next_torp_active)
            & (state.step_count % self.consts.SUB_TORPEDO_PERIOD == 0)
        )
        next_torp_x = jnp.where(fire_torp, next_submarine_x + 3, next_torp_x)
        next_torp_y = jnp.where(fire_torp, jnp.array(self.consts.SUBMARINE_Y - 2, dtype=jnp.int32), next_torp_y)
        next_torp_active = next_torp_active | fire_torp
        ## Pink flyer: drifts left across the sky, drops nothing
        next_wb_heli_x = jnp.where(
            state.wb_heli_active & (state.step_count % 2 == 0),
            state.wb_heli_x - 1,
            state.wb_heli_x
        )
        next_wb_heli_active = state.wb_heli_active & (
            next_wb_heli_x > self.consts.GAME_AREA_MIN_X - self.consts.WB_HELI_WIDTH
        )
        ## Debris: the two red bars linger where the rocket burst
        debris_age = jnp.where(state.wb_flyer_active, state.wb_flyer_timer + 1, 0)
        next_wb_flyer_x = jnp.where(
            state.wb_flyer_active & (state.step_count % 4 == 0),
            state.wb_flyer_x - 1,
            state.wb_flyer_x
        )
        next_wb_flyer_active = state.wb_flyer_active & (
            debris_age < self.consts.DEBRIS_LIFETIME_FRAMES
        )
        ## The burst hands over: rocket out, debris in at the burst column
        next_wb_flyer_active = next_wb_flyer_active | rocket_explodes
        next_wb_flyer_x = jnp.where(rocket_explodes, next_rocket_x + 2, next_wb_flyer_x)
        debris_age = jnp.where(rocket_explodes, 0, debris_age)

        # Enemies
        ## Helicopter enemy (Scroll left)
        ## 1. Determine which speed zone the helicopter is currently in, also affecting melee behavior of helicopter
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
        ## 3. The melee step of helicopter is only incremented when the helicopter is in slow mode
        ## Update the step counter for the next frame, reset to 0 if the helicopter is not in slow mode or not active
        next_helicopter_melee_step = jnp.where(
            next_helicopter_active & in_slow_mode,
            state.helicopter_melee_step + 1,
            0
        )
        
        ## Satellite enemy (Scroll right)
        ## Measured in ALE: +1,+1,+1,+0 repeating = 3px every 4 frames
        next_satellite_x = jnp.where(
            state.step_count % 4 != 3,
            state.satellite_x + 1,
            state.satellite_x
        )
        next_satellite_y = state.satellite_y
        ## It crosses the whole screen (~209 frame pass), not just the play area
        satellite_on_screen = next_satellite_x <= (self.consts.OBJECT_EXIT_X)
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
        on_land = state.stage == 0
        in_water = state.stage == 1
        row_57_empty = (~jnp.any(next_helicopter_active)) & (~jnp.any(next_diamond_active))
        # Check whose turn it is to spawn
        ## The red helicopter patrols the land and the first water scene but
        ## the later scenes retire it (the video shows the pink flyer and the
        ## small red heli there instead); the diamond keeps floating through
        spawn_diamond = row_57_empty & (state.spawn_diamond_next | (state.stage >= 2))
        spawn_helicopter = row_57_empty & (~state.spawn_diamond_next) & (state.stage <= 1)
        ## Measured ~46 frame gap between passes: park the timer at full
        ## while one is flying, count down while the sky is empty
        satellite_respawn_timer = jnp.where(
            next_satellite_active,
            jnp.array(self.consts.SATELLITE_RESPAWN_FRAMES, dtype=jnp.int32),
            jnp.maximum(state.satellite_respawn_timer - 1, 0),
        )
        can_spawn_satellite = (
            (~next_satellite_active) & (satellite_respawn_timer == 0) & (state.stage <= 1)
        )
        ## Land gate added with the water work: pits were spawning into the
        ## water and killing the boat (it rides at the same y as the car)
        can_spawn_pit = (~next_pit_active) & on_land

        ## Scuba diver: water only, one at a time, same parked-timer trick
        ## as the satellite so the countdown starts once he is gone
        scuba_respawn_timer = jnp.where(
            next_scuba_active | on_land,
            jnp.array(self.consts.SCUBA_RESPAWN_FRAMES, dtype=jnp.int32),
            jnp.maximum(state.scuba_respawn_timer - 1, 0),
        )
        ## Stage-2 spawners: each object enters on its own staggered breather
        ## so the roster stays mixed (same parked-timer pattern as above)
        def waterb_spawner(active, timer, gap):
            next_timer = jnp.where(
                active | (~in_water_b),
                jnp.array(gap, dtype=jnp.int32),
                jnp.maximum(timer - 1, 0),
            )
            spawn = in_water_b & (~active) & (next_timer == 0)
            return spawn, next_timer

        spawn_rocket, rocket_timer = waterb_spawner(
            next_rocket_active, state.rocket_timer, self.consts.ROCKET_RESPAWN_FRAMES)
        next_rocket_active = next_rocket_active | spawn_rocket
        next_rocket_x = jnp.where(spawn_rocket, self.consts.ROCKET_SPAWN_X, next_rocket_x)
        next_rocket_y = jnp.where(spawn_rocket, self.consts.ROCKET_Y, next_rocket_y)
        rocket_age = jnp.where(spawn_rocket, 0, rocket_age)

        spawn_submarine, submarine_timer = waterb_spawner(
            next_submarine_active, state.submarine_timer, self.consts.SUBMARINE_RESPAWN_FRAMES)
        next_submarine_active = next_submarine_active | spawn_submarine
        next_submarine_x = jnp.where(spawn_submarine, self.consts.SCUBA_SPAWN_X, next_submarine_x)

        spawn_wb_heli, wb_heli_timer = waterb_spawner(
            next_wb_heli_active, state.wb_heli_timer, self.consts.WB_HELI_RESPAWN_FRAMES)
        next_wb_heli_active = next_wb_heli_active | spawn_wb_heli
        next_wb_heli_x = jnp.where(spawn_wb_heli, self.consts.SCUBA_SPAWN_X, next_wb_heli_x)

        ## The debris has no spawner: it is born where the rocket bursts
        wb_flyer_timer = debris_age

        spawn_scuba = in_water & (~next_scuba_active) & (scuba_respawn_timer == 0)
        next_scuba_active = next_scuba_active | spawn_scuba
        next_scuba_x = jnp.where(
            spawn_scuba,
            jnp.array(self.consts.SCUBA_SPAWN_X, dtype=jnp.int32),
            next_scuba_x
        )
        next_scuba_y = jnp.where(
            spawn_scuba,
            jnp.array(self.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
            next_scuba_y
        )
        scuba_age = jnp.where(spawn_scuba, 0, scuba_age)
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
            57, ## TODO: Diamond spawn height, will change if the number is wrong
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
            57, ## TODO: Helicopter spawn height, will change if the number is wrong
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
            75, ## TODO: Satellite spawn height, will change if the number is wrong
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
            122, ## TODO: Pit spawn height, will change if the number is wrong
            next_pit_y
        )

        return state.replace(
            diamond_x=next_diamond_x,
            diamond_y=next_diamond_y,
            diamond_active=next_diamond_active,
            helicopter_x=next_helicopter_x,
            helicopter_y=next_helicopter_y,
            helicopter_active=next_helicopter_active,
            helicopter_melee_step=next_helicopter_melee_step,
            satellite_x=next_satellite_x,
            satellite_y=next_satellite_y,
            satellite_active=next_satellite_active,
            satellite_respawn_timer=satellite_respawn_timer,
            spawn_diamond_next=next_spawn_diamond_next,
            pit_x=next_pit_x,
            pit_y=next_pit_y,
            pit_active=next_pit_active,
            scuba_x=next_scuba_x,
            scuba_y=next_scuba_y,
            scuba_active=next_scuba_active,
            scuba_age=scuba_age,
            scuba_respawn_timer=scuba_respawn_timer,
            scuba_radioactive=next_scuba_radioactive,
            scuba_radioactive_age=scuba_rad_age,
            splash_x=next_splash_x,
            splash_active=next_splash_active,
            splash_age=splash_age,
            rocket_x=next_rocket_x,
            rocket_y=next_rocket_y,
            rocket_active=next_rocket_active,
            rocket_age=rocket_age,
            rocket_timer=rocket_timer,
            submarine_x=next_submarine_x,
            submarine_active=next_submarine_active,
            submarine_timer=submarine_timer,
            wb_heli_x=next_wb_heli_x,
            wb_heli_active=next_wb_heli_active,
            wb_heli_timer=wb_heli_timer,
            wb_flyer_x=next_wb_flyer_x,
            wb_flyer_active=next_wb_flyer_active,
            wb_flyer_timer=wb_flyer_timer,
            sub_torp_x=next_torp_x,
            sub_torp_y=next_torp_y,
            sub_torp_active=next_torp_active,
        )

    def _update_enemy_bombs(self, state: JamesBondState) -> JamesBondState:
        """Move and spawn the helicopter bomb and the satellite laser."""

        ## The old version still indexed helicopter_x like an array and crashed
        ## every step after the single object change. Rewritten for scalars.
        ground = self.consts.GAME_AREA_MAX_Y

        ## 1. Move whatever is flying, remove it once it reaches the ground
        ## (checked in the real game: they just disappear there, no explosion)
        heli_bomb_x = jnp.where(
            state.helicopter_bomb_active,
            state.helicopter_bomb_x + state.helicopter_bomb_vx,
            -1,
        )
        heli_bomb_y = jnp.where(
            state.helicopter_bomb_active,
            state.helicopter_bomb_y + self.consts.HELICOPTER_BOMB_VY,
            -1,
        )
        heli_bomb_active = jnp.logical_and(
            state.helicopter_bomb_active, heli_bomb_y < ground
        )

        laser_x = jnp.where(state.satellite_laser_active, state.satellite_laser_x, -1)
        laser_y = jnp.where(
            state.satellite_laser_active,
            state.satellite_laser_y + self.consts.SATELLITE_LASER_FALL_SPEED,
            -1,
        )
        ## On land the bolt vanishes at the ground line like before; in the
        ## water it keeps sinking under the surface and dies deeper down
        laser_floor = jnp.where(
            state.stage == 1,
            jnp.array(self.consts.WATER_LASER_FLOOR, dtype=jnp.int32),
            jnp.array(ground, dtype=jnp.int32),
        )
        laser_active = jnp.logical_and(state.satellite_laser_active, laser_y < laser_floor)

        ## The bolt goes radioactive a moment AFTER it enters the water, part
        ## way down -- not at the sea floor (checked by playing the real game).
        reached_depth = jnp.logical_and(
            jnp.logical_and(
                state.satellite_laser_active,
                laser_y >= self.consts.WATER_RADIOACTIVE_Y,
            ),
            state.stage == 1,
        )
        ## Whichever object goes radioactive, the bolt is spent at that point
        laser_active = jnp.logical_and(laser_active, jnp.logical_not(reached_depth))

        ## Only ONE radioactive thing may exist at a time, and the diver has
        ## priority: if he is in the water HE becomes the radioactive one and
        ## the bomb does not. The bomb only goes radioactive itself when there
        ## is no diver around.
        already_radioactive = jnp.logical_or(state.splash_active, state.scuba_radioactive)
        free_slot = jnp.logical_not(already_radioactive)

        spawn_splash = jnp.logical_and(
            jnp.logical_and(reached_depth, free_slot),
            jnp.logical_not(state.scuba_active), ## no diver -> the bomb glows
        )
        refresh_splash = jnp.logical_and(reached_depth, state.splash_active)
        splash_x = jnp.where(
            spawn_splash,
            laser_x, ## the figure surfaces at [laser_x, laser_x+19], not centered
            state.splash_x,
        )
        splash_active = jnp.logical_or(state.splash_active, spawn_splash)
        splash_age = jnp.where(
            jnp.logical_or(spawn_splash, refresh_splash), 0, state.splash_age
        )

        ## Diver present and the slot is free -> arm HIM instead. He does not
        ## light up instantly, the change takes SCUBA_RADIOACTIVE_DELAY frames.
        arm_scuba = jnp.logical_and(
            jnp.logical_and(reached_depth, free_slot),
            state.scuba_active,
        )
        ## A second bomb while he is already glowing just tops his clock back up
        refresh_scuba = jnp.logical_and(reached_depth, state.scuba_radioactive)

        scuba_rad_timer = jnp.where(
            arm_scuba,
            jnp.array(self.consts.SCUBA_RADIOACTIVE_DELAY, dtype=jnp.int32),
            jnp.maximum(state.scuba_radioactive_timer - 1, 0),
        )
        ## He turns on the frame the delay runs out (and only if he is still there)
        turns_now = jnp.logical_and(
            jnp.logical_and(state.scuba_radioactive_timer == 1, scuba_rad_timer == 0),
            state.scuba_active,
        )
        scuba_radioactive = jnp.logical_and(
            jnp.logical_or(state.scuba_radioactive, turns_now),
            state.scuba_active, ## if he despawns the glow goes with him
        )
        scuba_radioactive_age = jnp.where(
            jnp.logical_or(turns_now, refresh_scuba), 0, state.scuba_radioactive_age
        )

        ## 2. Helicopter drop. Measured against the ROM: the trigger is the
        ## distance to the player, not the searchlight. First bomb when the
        ## heli closes to RANGE_FAR, one more at RANGE_NEAR. The near one can
        ## happen while the heli is basically on top of the player or already
        ## past, which is why bombs also fall right diagonal in the real game.
        distance = state.helicopter_x - state.player_x
        in_range = jnp.logical_and(
            state.helicopter_active,
            distance <= self.consts.HELICOPTER_BOMB_RANGE,
        )
        ## The timer sits at 0 outside the range, so entering it gives the
        ## first chance right away, then one more every RETRY_FRAMES. Each
        ## chance is consumed whether or not the coin flip succeeds, and
        ## every next chance is rarer than the one before.
        bomb_timer = jnp.where(
            in_range, jnp.maximum(state.helicopter_bomb_timer - 1, 0), 0
        )
        chance = jnp.logical_and(
            jnp.logical_and(in_range, bomb_timer == 0),
            state.helicopter_bombs_dropped < self.consts.HELICOPTER_BOMB_MAX_PER_PASS,
        )
        roll = jax.random.uniform(
            jax.random.fold_in(state.key, state.helicopter_bombs_dropped)
        )
        lucky = roll < self.consts.HELICOPTER_BOMB_DROP_CHANCE / (
            1 + state.helicopter_bombs_dropped
        )
        ## The bomb and the laser share one missile slot in the real game,
        ## they were never airborne together in 3k+ measured frames
        drop_bomb = jnp.logical_and(
            jnp.logical_and(chance, lucky),
            jnp.logical_not(
                jnp.logical_or(heli_bomb_active, state.satellite_laser_active)
            ),
        )
        bomb_timer = jnp.where(
            chance,
            jnp.array(self.consts.HELICOPTER_BOMB_RETRY_FRAMES, dtype=jnp.int32),
            bomb_timer,
        )
        ## Count used chances (not drops), forget once the heli is gone
        bombs_dropped = jnp.where(
            state.helicopter_active,
            state.helicopter_bombs_dropped + chance.astype(jnp.int32),
            0,
        )
        heli_bomb_x = jnp.where(
            drop_bomb,
            state.helicopter_x + self.consts.HELICOPTER_ENEMY_WIDTH // 2,
            heli_bomb_x,
        )
        heli_bomb_y = jnp.where(
            drop_bomb,
            state.helicopter_y + self.consts.HELICOPTER_ENEMY_HEIGHT,
            heli_bomb_y,
        )
        ## Aim once on release: fall towards whichever side the player is on
        ## right now. No homing afterwards, teleporting the player mid fall
        ## changes nothing in the real game.
        player_center = state.player_x + self.consts.PLAYER_WIDTH // 2
        aimed_vx = jnp.where(
            player_center < heli_bomb_x,
            -self.consts.HELICOPTER_BOMB_SPEED_X,
            self.consts.HELICOPTER_BOMB_SPEED_X,
        ).astype(jnp.int32)
        heli_bomb_vx = jnp.where(drop_bomb, aimed_vx, state.helicopter_bomb_vx)
        heli_bomb_active = jnp.logical_or(heli_bomb_active, drop_bomb)

        ## 3. Satellite laser. Two triggers, one per scene:
        ## - Land: the kitchen timer, parked at full while no satellite is
        ##   around so every new pass starts a fresh countdown.
        ## - Water: the timer is ignored, it fires by position instead.
        laser_timer = jnp.where(
            state.satellite_active,
            jnp.maximum(state.satellite_laser_timer - 1, 0),
            jnp.array(self.consts.SATELLITE_LASER_DROP_PERIOD, dtype=jnp.int32),
        )
        timer_drop = jnp.logical_and(state.satellite_active, laser_timer == 0)

        ## Water rule from RAM-injection scans in ALE: at check moments the
        ## satellite fires iff its belly is 1..95px ahead (right) of the
        ## player, never from behind, at most 2 drops per pass
        sat_belly = state.satellite_x + self.consts.SATELLITE_ENEMY_WIDTH // 2
        ahead = sat_belly - state.player_x
        in_drop_window = jnp.logical_and(
            ahead >= self.consts.SATELLITE_DROP_AHEAD_MIN,
            ahead <= self.consts.SATELLITE_DROP_AHEAD_MAX,
        )
        check_moment = (state.step_count % self.consts.SATELLITE_CHECK_PERIOD) == 0
        window_drop = jnp.logical_and(
            jnp.logical_and(state.satellite_active, check_moment),
            jnp.logical_and(
                in_drop_window,
                state.satellite_lasers_dropped < self.consts.SATELLITE_WATER_MAX_DROPS,
            ),
        )

        in_water = state.stage == 1
        drop_laser = jnp.logical_and(
            jnp.where(in_water, window_drop, timer_drop),
            ## One laser at a time, and never while the bomb is airborne
            jnp.logical_not(jnp.logical_or(laser_active, heli_bomb_active)),
        )
        laser_x = jnp.where(drop_laser, sat_belly, laser_x)
        laser_y = jnp.where(
            drop_laser,
            state.satellite_y + self.consts.SATELLITE_ENEMY_HEIGHT,
            laser_y,
        )
        laser_active = jnp.logical_or(laser_active, drop_laser)
        laser_timer = jnp.where(
            drop_laser,
            jnp.array(self.consts.SATELLITE_LASER_DROP_PERIOD, dtype=jnp.int32),
            laser_timer,
        )
        ## Count the drops used this pass, forget once the satellite is gone
        lasers_dropped = jnp.where(
            state.satellite_active,
            state.satellite_lasers_dropped + drop_laser.astype(jnp.int32),
            0,
        )

        return state.replace(
            helicopter_bomb_x=heli_bomb_x.astype(jnp.int32),
            helicopter_bomb_y=heli_bomb_y.astype(jnp.int32),
            helicopter_bomb_vx=heli_bomb_vx,
            helicopter_bomb_active=heli_bomb_active,
            helicopter_bombs_dropped=bombs_dropped,
            helicopter_bomb_timer=bomb_timer,
            satellite_laser_x=laser_x.astype(jnp.int32),
            satellite_laser_y=laser_y.astype(jnp.int32),
            satellite_laser_active=laser_active,
            satellite_laser_timer=laser_timer,
            satellite_lasers_dropped=lasers_dropped,
            splash_x=splash_x.astype(jnp.int32),
            splash_active=splash_active,
            splash_age=splash_age,
            scuba_radioactive=scuba_radioactive,
            scuba_radioactive_age=scuba_radioactive_age,
            scuba_radioactive_timer=scuba_rad_timer,
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
        state = self._resolve_scuba_player_collisions(state)
        state = self._resolve_waterb_collisions(state)
        return state

    def _resolve_waterb_collisions(self, state: JamesBondState) -> JamesBondState:
        """Second water scene contacts.

        Ramming the floating rocket is the scene's only score (+200), and
        in the real game the ram usually costs a life as well, so both
        effects fire. The submarine can only reach a diving boat, the pink
        flyer only a jumping one. The debris bars are harmless wreckage.
        """

        def touch(ox, oy, ow, oh):
            return _aabb_overlap(
                state.player_x,
                state.player_y,
                self.consts.PLAYER_COLLISION_WIDTH,
                self.consts.PLAYER_COLLISION_HEIGHT,
                ox, oy, ow, oh,
            )

        rocket_hit = jnp.logical_and(
            state.rocket_active,
            touch(state.rocket_x, state.rocket_y,
                  self.consts.ROCKET_WIDTH, self.consts.ROCKET_HEIGHT),
        )
        submarine_hit = jnp.logical_and(
            state.submarine_active,
            touch(state.submarine_x, jnp.array(self.consts.SUBMARINE_Y, dtype=jnp.int32),
                  self.consts.SUBMARINE_WIDTH, self.consts.SUBMARINE_HEIGHT),
        )
        heli_hit = jnp.logical_and(
            state.wb_heli_active,
            touch(state.wb_heli_x, jnp.array(self.consts.WB_HELI_Y, dtype=jnp.int32),
                  self.consts.WB_HELI_WIDTH, self.consts.WB_HELI_HEIGHT),
        )

        any_hit = rocket_hit | submarine_hit | heli_hit
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(any_hit, can_take_damage)

        return state.replace(
            score=state.score + rocket_hit.astype(jnp.int32) * self.consts.SCORE_ROCKET,
            hit_enemy=jnp.logical_or(state.hit_enemy, rocket_hit),
            rocket_active=jnp.logical_and(state.rocket_active, ~rocket_hit),
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
        )

    def _resolve_scuba_player_collisions(self, state: JamesBondState) -> JamesBondState:
        """One life of damage from the two water hazards, neither can be shot.

        The diver's body sits below the surface, so a boat riding on top
        floats past him. The splash figure straddles the surface and kills
        on near contact; only a clearly airborne boat clears it.
        """

        diver_overlap = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            state.scuba_x,
            state.scuba_y,
            self.consts.SCUBA_WIDTH,
            self.consts.SCUBA_HEIGHT,
        )
        diver_hit = jnp.logical_and(state.scuba_active, diver_overlap)

        ## Measured kill window: boat_x in [splash_x - 9, splash_x + 19]
        x_touch = jnp.logical_and(
            state.player_x >= state.splash_x - 9,
            state.player_x <= state.splash_x + 19,
        )
        low_enough = state.player_y >= self.consts.SPLASH_SAFE_PLAYER_Y
        splash_hit = jnp.logical_and(
            state.splash_active, jnp.logical_and(x_touch, low_enough)
        )

        scuba_collision = jnp.logical_or(diver_hit, splash_hit)
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(scuba_collision, can_take_damage)

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

    def _resolve_bullet_player_collisions(self, state: JamesBondState) -> JamesBondState:
        """One life of damage when the bomb or the laser hits the player.

        The projectile always disappears on contact, the life is only lost
        when the hit cooldown ran out, same rule as the pit. Manual says the
        laser destroys on impact and can't be shot down in stage 1.
        """

        bomb_overlap = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            state.helicopter_bomb_x,
            state.helicopter_bomb_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
        )
        laser_overlap = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            state.satellite_laser_x,
            state.satellite_laser_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
        )
        bomb_hit = jnp.logical_and(state.helicopter_bomb_active, bomb_overlap)
        laser_hit = jnp.logical_and(state.satellite_laser_active, laser_overlap)
        hit_any = jnp.logical_or(bomb_hit, laser_hit)
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(hit_any, can_take_damage)

        return state.replace(
            helicopter_bomb_active=jnp.logical_and(
                state.helicopter_bomb_active, jnp.logical_not(bomb_hit)
            ),
            satellite_laser_active=jnp.logical_and(
                state.satellite_laser_active, jnp.logical_not(laser_hit)
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
            ## Only add points for actual hits, not every frame the bullet flies
            score=state.score + collected_count * self.consts.SCORE_DIAMOND,
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

        raster = self.jr.create_object_raster(self.BACKGROUND)

        ## The water scene has a solid gray sky where the land scene has black.
        ## It goes UNDER the stars, because in the real game the stars still
        ## twinkle on top of the gray.
        raster = jax.lax.cond(
            state.stage >= 1,
            lambda r: self.jr.render_at_clipped(r, 8, 29, self.SHAPE_MASKS['water_sky']),
            lambda r: r,
            raster,
        )
        raster = self._render_stars(raster, state)
        ## Terrain follows the scene: dry land in stage 0, water in stage 1.
        ## Before this the renderer always drew the ground, so the whole water
        ## scene was played on a picture of the road.
        raster = jax.lax.cond(
            state.stage == 0,
            lambda r: self._render_ground(r, state),
            lambda r: self._render_water(r, state),
            raster,
        )

        raster = self._render_car(raster, state)
        raster = self._render_diamond(raster, state)
        raster = self._render_pit(raster, state)
        raster = self._render_helicopter(raster, state)
        ## The searchlight sweep was implemented but never drawn -- in the
        ## real game (checked against a longplay video) the yellow beam is
        ## clearly visible whenever the helicopter slows mid-screen
        raster = self._render_helicopter_melee(raster, state)
        raster = self._render_satellite(raster, state)
        raster = self._render_scuba(raster, state)
        raster = self._render_splash(raster, state)
        raster = self._render_waterb(raster, state)

        raster = self._render_bullets(raster, state)
        ## After the bullets so the green recolor overdraws the normal bolt
        raster = self._render_sinking_bolt(raster, state)

        ## Render life counter
        raster = self.jr.render_indicator(raster, 9, 184, state.lives, self.SHAPE_MASKS['life'], 16, 3) ## TODO: Maybe 5 like in ALE?
        
        ## Render Score counter
        score_digits = self.jr.int_to_digits(state.score, 4) ## TODO: Max score 4 digits?
        raster = self.jr.render_label(raster, 95, 15, score_digits, self.SHAPE_MASKS['score_digits'], 8, 4) ## TODO: Position offset per digit?

        return self.jr.render_from_palette(raster, self.PALETTE)

    def _render_ground(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the static ground strip using the existing ground_unkempt sprite."""

        return self.jr.render_at_clipped(
            raster,
            4,    # x - matches GAME_AREA_MIN_X
            119,  # y - matches GAME_AREA_MAX_Y / PLAYER_INIT_Y
            self.SHAPE_MASKS['ground'],
        )

    def _render_water(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the water scene terrain: blue water with the seabed under it.

        Rows measured off the real ROM (jump straight there in ALE with
        RAM[13]=1): gray sky 29-120, blue water 121-179 starting at x=8,
        and the seabed mounds rising out of the water near the bottom.
        """

        raster = self.jr.render_at_clipped(
            raster,
            8,    # x - the play area starts here in the real scene
            121,  # y - the waterline, right under the gray sky
            self.SHAPE_MASKS['water'],
        )
        ## The seabed sits on the bottom of the water band (base at row 180)
        return self.jr.render_at_clipped(
            raster,
            0,
            158,
            self.SHAPE_MASKS['seabed_water'],
        )

    def _render_stars(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the twinkling star field, alternating between the two frames."""

        sprite_idx = jnp.where(state.step_count % 2 == 0, 0, 1)
        return self.jr.render_at_clipped(
            raster,
            0,  # x
            0,  # y
            self.SHAPE_MASKS['stars'][sprite_idx],
        )

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

    def _render_car(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:

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
    
    def _render_helicopter(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:

        sprite_idx = jnp.where(
            state.step_count % 2 == 0,
            0,
            1
        )

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.helicopter_x,
            state.helicopter_y,
            self.SHAPE_MASKS['helicopter'][sprite_idx]
        )

        return jax.lax.cond(state.helicopter_active, draw_fn, lambda r: r, raster)

    def _render_helicopter_melee(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the helicopter melee animation."""

        """
        ## UNFINISHED
        helicopter_melee_step = jnp.where( ## Only first 14 needed; ## 39 -> last visible left, 40 -> invisible, 41 -> second right
                    state.helicopter_melee_step > 40,
                    jnp.ceil(state.helicopter_melee_step / 2),
                    state.helicopter_melee_step
                )
                
                melee_idx = jnp.where( ## Create -> gone -> same -> gone -> new
                    helicopter_melee_step % 2 == 0,
                    0,
                    jnp.where(
                        state.step_count % 2 == 0,
                        (helicopter_melee_step - 1) / 2,
                        jnp.maximum(
                            (helicopter_melee_step - 3) / 4,
                            (helicopter_melee_step - 1) / 2,
                        )
                    )
                )
        """

        melee_idx = self.consts.HELICOPTER_MELEE_SPRITE_STEPS[state.helicopter_melee_step][0]

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.helicopter_x + self.consts.HELICOPTER_MELEE_SPRITE_STEPS[state.helicopter_melee_step][1],
            state.helicopter_y + 7,
            self.SHAPE_MASKS['helicopter_melee'][melee_idx]
        )

        return jax.lax.cond(
            (melee_idx != -1) & state.helicopter_active, draw_fn, lambda r: r, raster
        )

    def _render_satellite(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.satellite_x,
            state.satellite_y,
            self.SHAPE_MASKS['satellite'],
        )

        return jax.lax.cond(state.satellite_active, draw_fn, lambda r: r, raster)

    def _render_scuba(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the scuba diver, alternating his two swim frames.

        Normally he flips sprites every 15 frames. Once a bomb has made him
        radioactive he flickers fast instead, so you can see at a glance
        which object is currently the radioactive one.
        """

        sprite_idx = jnp.where(
            state.scuba_radioactive,
            (state.step_count // 2) % 2,   ## fast flicker while radioactive
            (state.step_count // 15) % 2,  ## normal lazy swim
        )

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.scuba_x,
            state.scuba_y,
            self.SHAPE_MASKS['scuba'][sprite_idx],
        )

        return jax.lax.cond(state.scuba_active, draw_fn, lambda r: r, raster)

    def _render_splash(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the green splash figure with its measured two-pose animation.

        Frame-exact from ALE: narrow pose for 1 frame at spawn, then strict
        7-frame phases alternate starting with the wide pose (4px further left).
        """

        ## The +6 shifts the phase so age 0 lands on narrow for one frame
        narrow = ((state.splash_age + 6) // 7) % 2 == 0

        def draw_fn(r):
            return jax.lax.cond(
                narrow,
                lambda rr: self.jr.render_at_clipped(
                    rr, state.splash_x, self.consts.SPLASH_Y,
                    self.SHAPE_MASKS['splash'],
                ),
                lambda rr: self.jr.render_at_clipped(
                    rr, state.splash_x - 4, self.consts.SPLASH_Y,
                    self.SHAPE_MASKS['splash_wide'],
                ),
                r,
            )

        return jax.lax.cond(state.splash_active, draw_fn, lambda r: r, raster)

    def _render_waterb(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the second water scene roster."""

        def one(raster, active, x, y, mask_name):
            return jax.lax.cond(
                active,
                lambda r: self.jr.render_at_clipped(r, x, y, self.SHAPE_MASKS[mask_name]),
                lambda r: r,
                raster,
            )

        ## The rocket burst flashes the whole sky gray for a frame or two
        raster = one(
            raster,
            jnp.logical_and(
                state.wb_flyer_active,
                state.wb_flyer_timer < self.consts.SKY_FLASH_FRAMES,
            ),
            jnp.array(8, dtype=jnp.int32), jnp.array(29, dtype=jnp.int32), 'sky_flash',
        )
        raster = one(raster, state.rocket_active, state.rocket_x, state.rocket_y, 'rocket')
        raster = one(raster, state.submarine_active, state.submarine_x,
                     jnp.array(self.consts.SUBMARINE_Y, dtype=jnp.int32), 'submarine')
        raster = one(raster, state.wb_heli_active, state.wb_heli_x,
                     jnp.array(self.consts.WB_HELI_Y, dtype=jnp.int32), 'heli_pink')
        raster = one(raster, state.wb_flyer_active, state.wb_flyer_x,
                     jnp.array(self.consts.WB_FLYER_Y, dtype=jnp.int32), 'flyer_red')
        raster = one(raster, state.sub_torp_active, state.sub_torp_x,
                     state.sub_torp_y, 'bullet')
        return raster

    def _render_sinking_bolt(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """A bolt landing while the figure lives sinks radioactive-green."""

        submerged = jnp.logical_and(
            jnp.logical_and(state.stage == 1, state.splash_active),
            jnp.logical_and(
                state.satellite_laser_active,
                state.satellite_laser_y > 119, ## below the waterline row
            ),
        )

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.satellite_laser_x,
            state.satellite_laser_y,
            self.SHAPE_MASKS['laser_green'],
        )

        return jax.lax.cond(submerged, draw_fn, lambda r: r, raster)

    def _render_bullets(self, raster: jnp.ndarray, state: JamesBondState,) -> jnp.ndarray:
        """Draw all projectiles, they share the same 1x4 bullet sprite."""

        ## player air bullet, player water bullet, helicopter bomb, satellite laser
        active_bullets = jnp.stack([
            state.player_bullet_active,
            state.player_wbullet_active,
            state.helicopter_bomb_active,
            state.satellite_laser_active,
        ])

        bullet_positions = jnp.stack([
            jnp.stack([state.player_bullet_x, state.player_bullet_y]),
            jnp.stack([state.player_wbullet_x, state.player_wbullet_y]),
            jnp.stack([state.helicopter_bomb_x, state.helicopter_bomb_y]),
            jnp.stack([state.satellite_laser_x, state.satellite_laser_y]),
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
